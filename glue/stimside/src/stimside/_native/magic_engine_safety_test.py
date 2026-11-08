"""Memory-safety contract of the `MagicBatchEngine` wrapper (WP-25 S4a fix pass).

The C++ engine trusts its caller (live handle, buffers sized from the header, aligned words) and
ctypes drops the GIL during every native call, so the wrapper must: copy misaligned buffers
(the kernel loads uint64 directly), serialise the native calls of one engine while leaving
independent engines alone, validate every number it forwards, and refuse use after close.

The threading tests use a gated stand-in for the engine's CDLL. It holds one native call back at
a barrier and never calls into a freed handle, so an unlocked wrapper is *detected* (a command
finished while another was inside the engine, or a call reached a destroyed handle) instead of
corrupting memory.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import expansion_to_arrays
from stimside._native import magic_engine as me

_PASSIVE = ("mbe_abi_v1", "mbe_live_engines")
# Everything the wrapper may raise for misuse; any other exception type is a test error and
# propagates out of the thread (pytest reports it as an unhandled thread exception).
_ENGINE_ERRORS = (ValueError, TypeError, IndexError, MemoryError, FloatingPointError)
_WAIT_S = 0.5  # how long a command must stay blocked behind a running native call


class _GatedLib:
    """Wraps the engine's CDLL: counts concurrent native calls, optionally holds one back."""

    def __init__(
        self, lib: Any, gates: dict[str, Callable[[tuple[Any, ...]], None]] | None = None,
        nap: float = 0.0,
    ) -> None:
        self._lib, self._gates, self._nap = lib, gates or {}, nap
        self._guard = threading.Lock()
        self.inflight = self.max_inflight = 0
        self.log: list[tuple[str, int]] = []  # (call name, thread id) in start order
        self.args: list[tuple[str, tuple[Any, ...]]] = []
        self.destroyed: set[int] = set()
        self.use_after_destroy: list[str] = []

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if not name.startswith("mbe_") or name in _PASSIVE:
            return fn

        def call(*args: Any) -> Any:
            with self._guard:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
                self.log.append((name, threading.get_ident()))
                self.args.append((name, args))
            try:
                if name in self._gates:
                    self._gates[name](args)
                elif self._nap:
                    time.sleep(self._nap)
                with self._guard:
                    dead = bool(args) and args[0] in self.destroyed
                    if name == "mbe_destroy":
                        self.destroyed.add(args[0])
                    elif dead:
                        self.use_after_destroy.append(name)
                if dead and name != "mbe_destroy":
                    return -2  # what an unlocked wrapper would do: never really touch freed memory
                return fn(*args)
            finally:
                with self._guard:
                    self.inflight -= 1

        return call


def _state(n: int = 70, d: bool = True) -> pq.ShotExpansion:
    rng = np.random.default_rng(5)
    e = pq.ShotExpansion.zero(n)
    for step in range(4):
        a = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
        b = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
        if not (a.any() or b.any()):
            a = pq.pack_bit_vector(np.eye(n, dtype=np.uint8)[step], n)
        kap = int(pq.word_dot_parity(a, b)) & 3
        e = pq.apply_pauli_t(e, a, b, kap, dag=False)
        if d and step == 1:
            e = pq.apply_quarter_turn(e, a, b, kap)
    return e


def _misaligned(values: Any) -> Any:
    """A writable C-contiguous copy of `values` whose data pointer is 1 byte off alignment."""
    values = np.ascontiguousarray(values)
    raw = np.zeros(values.nbytes + 8, np.uint8)
    out = raw[1 : 1 + values.nbytes].view(values.dtype).reshape(values.shape)
    out[...] = values
    assert values.size > 0 and not out.flags.aligned and out.flags.c_contiguous
    return out


class _TracedLock:
    """The engine's lock, logging every release into the gated library's call log."""

    def __init__(self, inner: Any, probe: _GatedLib) -> None:
        self._inner, self._probe = inner, probe

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return bool(self._inner.acquire(*args, **kwargs))

    def release(self) -> None:
        with self._probe._guard:  # logged before the release: nobody can have acted yet
            self._probe.log.append(("lock_release", threading.get_ident()))
        self._inner.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def _gated(eng: me.MagicBatchEngine, **kw: Any) -> _GatedLib:
    probe = _GatedLib(eng._lib, **kw)
    eng._lib = probe  # type: ignore[assignment]
    eng._lock = _TracedLock(eng._lock, probe)  # type: ignore[assignment]
    return probe


# ---------------------------------------------------------------------------------------------
# F1: misaligned buffers
# ---------------------------------------------------------------------------------------------


def test_misaligned_pauli_words_reach_the_native_code_as_aligned_copies() -> None:
    n = 70
    eng = me.MagicBatchEngine(2, n, 1e-10)
    eng.import_shot(0, expansion_to_arrays(_state(n)))
    probe = _gated(eng)
    rng = np.random.default_rng(0)
    a = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
    b = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
    kap = int(pq.word_dot_parity(a, b)) & 3
    eng.apply_pauli_mask(_misaligned(a), _misaligned(b), kap, np.array([True, False]))
    _, args = next(x for x in probe.args if x[0] == "mbe_apply_pauli_mask")
    assert args[1] % 8 == 0 and args[2] % 8 == 0  # uint64 loads need 8-byte alignment
    want = pq.apply_pauli(_state(n), a, b, kap)
    got = eng.export_shot(0)
    assert got.h_words.tobytes() == want.h_words.tobytes()
    assert got.weights.view(np.uint64).tobytes() == want.weights.view(np.uint64).tobytes()
    eng.close()


def test_misaligned_import_arrays_are_copied_and_round_trip_exactly() -> None:
    n = 70
    eng = me.MagicBatchEngine(1, n, 1e-10)
    s = expansion_to_arrays(_state(n))
    assert s.d > 0 and s.chi > 1
    bad = s._replace(
        weights=_misaligned(s.weights), h_words=_misaligned(s.h_words),
        A_words=_misaligned(s.A_words), a_union=_misaligned(s.a_union),
        pivots=_misaligned(s.pivots),
    )
    probe = _gated(eng)
    eng.import_shot(0, bad)
    (_, args), = [x for x in probe.args if x[0] == "mbe_import_shot"]
    assert all(args[i] % 8 == 0 for i in (7, 8, 10, 11, 13))  # w, h, A, a_union, pivots
    back = eng.export_shot(0)
    for field in ("weights", "h_words", "ell", "A_words", "a_union", "Gamma", "pivots", "ell_odd"):
        assert getattr(back, field).tobytes() == getattr(s, field).tobytes(), field
    eng.close()


def test_addr_refuses_misaligned_and_strided_buffers() -> None:
    ok = np.zeros(4, np.uint64)
    assert me.addr(ok) == int(ok.ctypes.data)
    with pytest.raises(TypeError, match="aligned"):
        me.addr(_misaligned(np.arange(4, dtype=np.uint64)))
    with pytest.raises(TypeError, match="C-contiguous"):
        me.addr(np.zeros(8, np.uint64)[::2])


# ---------------------------------------------------------------------------------------------
# F2: one engine shared by threads; independent engines stay independent
# ---------------------------------------------------------------------------------------------


def _engine_with_shots(batch: int = 4, n: int = 70) -> me.MagicBatchEngine:
    eng = me.MagicBatchEngine(batch, n, 1e-10)
    for b in range(batch):
        eng.import_shot(b, expansion_to_arrays(_state(n)))
    return eng


def _commands(eng: me.MagicBatchEngine) -> dict[str, Callable[[], Any]]:
    s = expansion_to_arrays(_state(eng.n_qubits))
    return {
        "close": eng.close,
        "reset": eng.reset,
        "import_shot": lambda: eng.import_shot(1, s),
        "export_shot": lambda: eng.export_shot(1),
    }


@pytest.mark.parametrize("other", ["close", "reset", "import_shot", "export_shot"])
def test_a_command_waits_while_another_thread_is_inside_apply_pauli_mask(other: str) -> None:
    eng = _engine_with_shots()
    inside, release = threading.Event(), threading.Event()

    def hold(_: tuple[Any, ...]) -> None:
        inside.set()
        release.wait(30)

    probe = _gated(eng, gates={"mbe_apply_pauli_mask": hold})
    words = np.zeros(eng.n_words, np.uint64)
    runner = threading.Thread(
        target=lambda: eng.apply_pauli_mask(words, words, 0, np.ones(4, bool))
    )
    command = threading.Thread(target=_commands(eng)[other])
    runner.start()
    try:
        assert inside.wait(30)
        command.start()
        command.join(_WAIT_S)
        finished_while_inside = not command.is_alive()
    finally:
        release.set()  # never leave a thread parked, whatever happened above
        runner.join(30)
        command.join(30)
    assert not finished_while_inside, f"{other} ran while apply_pauli_mask was in the engine"
    assert not probe.use_after_destroy, probe.use_after_destroy
    eng.close()


def test_independent_engines_never_wait_for_each_other() -> None:
    slow, fast = _engine_with_shots(), _engine_with_shots()
    inside, release = threading.Event(), threading.Event()

    def hold(_: tuple[Any, ...]) -> None:
        inside.set()
        release.wait(30)

    _gated(slow, gates={"mbe_apply_pauli_mask": hold})
    words = np.zeros(slow.n_words, np.uint64)
    parked = threading.Thread(
        target=lambda: slow.apply_pauli_mask(words, words, 0, np.ones(4, bool))
    )
    done = threading.Event()

    def work_on_the_other_engine() -> None:
        fast.apply_pauli_mask(words, words, 0, np.ones(4, bool))
        fast.import_shot(0, expansion_to_arrays(_state()))
        fast.export_shot(0)
        done.set()

    other = threading.Thread(target=work_on_the_other_engine)
    parked.start()
    try:
        assert inside.wait(30)
        other.start()
        finished = done.wait(30)  # while `slow` is still held inside its native call
    finally:
        release.set()
        parked.join(30)
        other.join(30)
    assert finished, "a command on engine B was blocked by a running call on engine A"
    slow.close()
    fast.close()


def test_threads_sharing_one_engine_never_overlap_and_never_split_header_and_export() -> None:
    eng = _engine_with_shots(batch=3)
    probe = _gated(eng, nap=0.002)  # sleeping inside every call lets any overlap show up
    big, small = expansion_to_arrays(_state()), expansion_to_arrays(pq.ShotExpansion.zero(70))
    words = np.zeros(eng.n_words, np.uint64)
    start = threading.Barrier(6)
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            start.wait(30)
            for i in range(12):
                if k % 3 == 0:
                    eng.export_shot(i % 3)
                elif k % 3 == 1:
                    eng.import_shot(i % 3, big if i % 2 else small)
                else:
                    eng.apply_pauli_mask(words, words, 0, np.ones(3, bool))
        except _ENGINE_ERRORS as exc:  # reported through the list below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    assert probe.max_inflight == 1, f"{probe.max_inflight} native calls ran at once"
    headers = 0
    for i, (name, tid) in enumerate(probe.log):
        if name == "mbe_shot_header":  # the export is sized from this header
            headers += 1
            nxt = probe.log[i + 1 :]
            export = next(j for j, (n, t) in enumerate(nxt) if n == "mbe_export_shot" and t == tid)
            # the lock stayed held from the header to the export, and nobody ran in between
            assert not any(n == "lock_release" and t == tid for n, t in nxt[:export]), "split"
            assert all(t == tid for _, t in nxt[:export]), "another thread ran in between"
    assert headers >= 8
    eng.close()


def test_real_threads_hammering_one_engine_end_cleanly_without_a_crash() -> None:
    n = 70
    eng = _engine_with_shots(batch=48, n=n)
    live = me.get_magic_engine_lib().mbe_live_engines
    before = live()
    state = expansion_to_arrays(_state(n))
    words = np.zeros(eng.n_words, np.uint64)
    words[0] = 1
    start = threading.Barrier(5)
    errors: list[BaseException] = []

    def command(which: int, i: int) -> None:
        if which == 0:
            eng.apply_pauli_mask(words, words, 1, np.ones(48, bool))
        elif which == 1:
            eng.import_shot(i % 48, state)
        elif which == 2:
            eng.export_shot((i * 7) % 48)
        else:
            eng.reset()

    def worker(k: int) -> None:
        try:
            start.wait(30)
            for i in range(60):
                if k == 0 and i == 40:
                    eng.close()
                command((i + k) % 4, i)
        except RuntimeError as exc:
            if "closed" not in str(exc):
                errors.append(exc)
        except _ENGINE_ERRORS as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert not errors, errors
    eng.close()
    assert live() == before - 1


# ---------------------------------------------------------------------------------------------
# F4: every number the wrapper forwards is validated
# ---------------------------------------------------------------------------------------------

_BAD_ATOLS = [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, -1e-300, True, None, "x",
              np.array([1e-10, 1e-10])]


@pytest.mark.parametrize("atol", _BAD_ATOLS, ids=repr)
def test_a_bad_atol_is_rejected_by_the_engine_and_by_import(atol: Any) -> None:
    with pytest.raises(ValueError, match="atol"):
        me.MagicBatchEngine(2, 5, atol)
    eng = me.MagicBatchEngine(2, 5, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(5))
    with pytest.raises(ValueError, match="atol"):
        eng.import_shot(0, s._replace(atol=atol))
    assert eng.export_shot(0).atol == 1e-10  # the rejected import changed nothing
    eng.close()


def test_numpy_scalars_are_accepted_where_python_numbers_are() -> None:
    eng = me.MagicBatchEngine(np.int64(2), np.int32(70), np.float64(1e-9))  # type: ignore[arg-type]
    assert (eng.batch_size, eng.n_qubits, eng.atol) == (2, 70, 1e-9)
    assert type(eng.batch_size) is int
    eng.export_shot(np.int64(1))  # type: ignore[arg-type]
    eng.close()


@pytest.mark.parametrize("bad", [1.5, True, None, "2", 2.0], ids=repr)
def test_non_integer_sizes_and_indices_are_a_type_error_not_a_truncation(bad: Any) -> None:
    with pytest.raises(TypeError, match="batch_size"):
        me.MagicBatchEngine(bad, 5, 1e-10)
    with pytest.raises(TypeError, match="n_qubits"):
        me.MagicBatchEngine(2, bad, 1e-10)
    eng = me.MagicBatchEngine(3, 5, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(5))
    with pytest.raises(TypeError, match="shot index"):
        eng.export_shot(bad)
    with pytest.raises(TypeError, match="shot index"):
        eng.import_shot(bad, s)
    with pytest.raises(TypeError, match="chi"):
        eng.import_shot(0, s._replace(chi=bad))
    with pytest.raises(TypeError, match="d must"):
        eng.import_shot(0, s._replace(d=bad))
    assert eng.export_shot(1).chi == 1  # the engine is untouched
    eng.close()


@pytest.mark.parametrize("big", [2**63, 2**64, 2**64 + 3, 10**30])
def test_sizes_that_do_not_fit_a_size_t_are_rejected_not_wrapped(big: int) -> None:
    with pytest.raises(ValueError):
        me.MagicBatchEngine(big, 5, 1e-10)  # 2**64 used to wrap to 0 shots
    with pytest.raises(ValueError):
        me.MagicBatchEngine(2, big, 1e-10)
    eng = me.MagicBatchEngine(2, 5, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(5))
    with pytest.raises(ValueError, match="chi"):
        eng.import_shot(0, s._replace(chi=big))
    with pytest.raises(ValueError, match="d="):
        eng.import_shot(0, s._replace(d=big))
    with pytest.raises(IndexError):
        eng.export_shot(big)
    eng.close()


@pytest.mark.parametrize(
    ("n", "bad"),
    [(5, [1 << 5]), (5, [1 << 63]), (70, [0, 1 << 6]), (70, [0, 1 << 63]), (130, [0, 0, 4])],
)
def test_pauli_words_with_padding_bits_are_rejected_like_the_reference(
    n: int, bad: list[int]
) -> None:
    nw = len(bad)
    eng = me.MagicBatchEngine(2, n, 1e-10)
    eng.import_shot(0, expansion_to_arrays(pq.ShotExpansion.zero(n)))
    words, zero = np.array(bad, np.uint64), np.zeros(nw, np.uint64)
    mask = np.ones(2, bool)
    for a, b in ((words, zero), (zero, words)):
        with pytest.raises(ValueError, match="padding"):
            eng.apply_pauli_mask(a, b, 0, mask)
        with pytest.raises(ValueError, match="padding"):  # the per-shot reference agrees
            pq.apply_pauli(pq.ShotExpansion.zero(n), a, b, 0)
    assert eng.export_shot(0).h_words.tobytes() == np.zeros((1, nw), np.uint64).tobytes()
    eng.close()


@pytest.mark.parametrize("n", [1, 63, 64, 65, 128])
def test_full_words_and_top_valid_bits_are_not_mistaken_for_padding(n: int) -> None:
    nw = max(1, (n + 63) // 64)
    top = np.zeros(nw, np.uint64)
    top[(n - 1) >> 6] = np.uint64(1) << np.uint64((n - 1) & 63)  # the highest legal qubit
    eng = me.MagicBatchEngine(1, n, 1e-10)
    eng.apply_pauli_mask(top, np.zeros(nw, np.uint64), 0, np.ones(1, bool))
    eng.close()


def test_import_requires_a_shot_arrays_value() -> None:
    eng = me.MagicBatchEngine(1, 5, 1e-10)
    with pytest.raises(TypeError, match="ShotArrays"):
        eng.import_shot(0, pq.ShotExpansion.zero(5))  # type: ignore[arg-type]
    eng.close()


# ---------------------------------------------------------------------------------------------
# F3: misuse of a handle is a clear Python error (the raw `mbe_*` ABI is private and trusting)
# ---------------------------------------------------------------------------------------------


def test_every_method_of_a_closed_engine_raises_and_close_stays_idempotent() -> None:
    eng = me.MagicBatchEngine(2, 5, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(5))
    words = np.zeros(1, np.uint64)
    calls: list[Callable[[], Any]] = [
        eng.reset, lambda: eng.import_shot(0, s), lambda: eng.export_shot(0),
        lambda: eng.apply_pauli_mask(words, words, 0, np.ones(2, bool)),
    ]
    eng.close()
    for call in calls:
        with pytest.raises(RuntimeError, match="closed"):
            call()
    eng.close()
    eng.close()
    assert eng._handle is None


def test_an_engine_cannot_be_copied_pickled_or_deep_copied() -> None:
    eng = me.MagicBatchEngine(1, 5, 1e-10)
    for clone in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="cannot be pickled or copied"):
            clone(eng)
    eng.close()


def test_the_lock_is_per_engine_and_is_released_after_an_error() -> None:
    a, b = me.MagicBatchEngine(1, 5, 1e-10), me.MagicBatchEngine(1, 5, 1e-10)
    assert a._lock is not b._lock
    with pytest.raises(IndexError):
        a.export_shot(5)
    with pytest.raises(ValueError, match="mask"):
        a.apply_pauli_mask(np.zeros(1, np.uint64), np.zeros(1, np.uint64), 0, np.zeros(3, bool))
    assert a._lock.acquire(timeout=1)  # nothing left it locked
    a._lock.release()
    a.close()
    b.close()


def test_the_c_import_itself_rejects_a_geometry_wider_than_the_register() -> None:
    """The wrapper checks d <= n_qubits first; the C++ check is a second line of defence."""
    n, d, chi = 5, 6, 1  # d > n_qubits, with fully valid buffers: only the bound can refuse
    eng = me.MagicBatchEngine(1, n, 1e-10)
    nw = eng.n_words
    arrays = (
        np.ones(chi, np.complex128), np.zeros((chi, nw), np.uint64), np.zeros((chi, d), np.uint8),
        np.zeros((d, nw), np.uint64), np.zeros(nw, np.uint64), np.zeros((d, d), np.uint8),
        np.zeros(d, np.int32), np.zeros(d, np.uint8),
    )
    w, *rest = arrays

    def raw_import(geometry: int) -> int:
        return int(eng._lib.mbe_import_shot(
            eng._h(), 0, chi, geometry, 1, -3.0, 1e-10, me.w_addr(w), *(me.addr(a) for a in rest),
        ))

    assert raw_import(d) == 1  # d=6 > n_qubits=5
    assert eng.export_shot(0).log_mass == 0.0  # the rejected call changed nothing
    assert raw_import(n) == 0  # control: d == n_qubits is accepted (the buffers are big enough)
    assert eng.export_shot(0).log_mass == -3.0
    eng.close()
