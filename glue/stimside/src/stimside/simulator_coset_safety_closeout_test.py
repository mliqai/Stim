"""Safety closeout tests for F12, H2 atomicity/re-entrancy, R3-F2, and D-03."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import signal
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import stim

from stimside import simulator_coset as _sim_mod
from stimside._native import coset_kernels, marginal_dem_kernels
from stimside.dem_generators.dem_generator_marginal import MarginalLeakageDemGenerator
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_DCC_KERNEL = "engine_batch_inject_unitary_v4"
_GROUP_KERNEL = "engine_inject_group_v3"
_RESET_KERNEL = "engine_batch_unmatched_noisy_reset_z"
_PROJECT_KERNEL = "engine_project_pauli"
_POSTSELECT_KERNEL = "engine_postselect_batch"
_PEEK_KERNEL = "engine_peek_pauli_batch"
_B = 16


def _coset_cpp_missing() -> bool:
    native_dir = Path(coset_kernels.__file__).resolve().parent
    has_so = (native_dir / "libcoset_kernels.so").exists()
    can_build = (native_dir / "coset_kernels.cpp").exists() and os.access(native_dir, os.W_OK)
    return not (has_so or can_build)


def _marginal_cpp_missing() -> bool:
    native_dir = Path(marginal_dem_kernels.__file__).resolve().parent
    has_so = (native_dir / "libmarginal_dem_kernels.so").exists()
    can_build = (
        (native_dir / "marginal_dem_kernels.cpp").exists() and os.access(native_dir, os.W_OK)
    )
    return not (has_so or can_build)


_SKIP_NO_CPP = pytest.mark.skipif(
    _coset_cpp_missing(), reason="coset C++ kernel library is unavailable"
)
_SKIP_NO_MARGINAL_CPP = pytest.mark.skipif(
    _marginal_cpp_missing(), reason="marginal DEM C++ kernel library is unavailable"
)
_NO_SIGALRM = not (hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"))
_SKIP_NO_SIGALRM = pytest.mark.skipif(_NO_SIGALRM, reason="SIGALRM/setitimer requires POSIX")
_PATH_PARAMS = [pytest.param(True, marks=_SKIP_NO_CPP), False]
_PATH_IDS = ["kernel", "python"]


def _coset_cpp_available() -> bool:
    if _coset_cpp_missing():
        return False
    for mod in (coset_kernels, _sim_mod):
        getter = getattr(mod, "get_coset_kernels_lib", None)
        if getter is None:
            return False
        try:
            if getter() is None:
                return False
        except (OSError, RuntimeError):
            return False
    return True


def _marginal_cpp_available() -> bool:
    if _marginal_cpp_missing():
        return False
    getter = getattr(marginal_dem_kernels, "get_marginal_dem_kernels_lib", None)
    if getter is None:
        return False
    try:
        return getter() is not None
    except (OSError, RuntimeError):
        return False


def _mk(text: str, *, use_cpp: bool = True, batch: int = _B) -> CosetsideSimulator:
    cpp_ok = _coset_cpp_available()
    sync_rng: bool | None = None
    if not cpp_ok:
        if use_cpp:
            pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
        sync_rng = True
    circ = stim.Circuit(text)
    coh = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch)
    sim = CosetsideSimulator(
        circ, compiled_op_handler=coh, batch_size=batch, seed=7, sync_tableside_rng=sync_rng
    )
    if not use_cpp:
        sim._use_cpp = False
    elif sim._cpp_lib is None:
        pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
    return sim


def _fp(sim: CosetsideSimulator) -> str:
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    if sim._use_cpp and sim._cpp_dirty:
        sim._sync_cpp_to_py()
    shots = sorted(
        (b, st.d, st.h_words.tolist(), st.A_words.tolist(), st.ell.tolist())
        for b, st in sim._active_shots.items()
    )
    ref = sim._compiled_ref
    words = [w.tobytes() for w in (*ref.ref_chp.snapshot_packed(), *sim._current_packed_words())]
    rng = repr(sim.np_rng.bit_generator.state["state"]).encode()
    parts = (
        xs.tobytes(),
        zs.tobytes(),
        repr(shots).encode(),
        sim._shot_tiebreak_rng.tobytes(),
        rng,
        repr((sim._step_cursor, len(ref.steps), len(sim._measurement_columns))).encode(),
        *words,
    )
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


class _SafetyKernelSpy:
    """Validates kernel buffer sizes/aliases/ranges before forwarding to the real C++ library."""

    def __init__(
        self,
        lib: ctypes.CDLL,
        extents: dict[int, int],
        num_qubits: int,
        batch: int,
        caller_bufs: set[int] | None = None,
    ) -> None:
        self._lib = lib
        self._extents = extents
        self._num_qubits = num_qubits
        self._batch = batch
        self._caller_bufs = caller_bufs or set()
        self.calls: dict[str, int] = {
            _DCC_KERNEL: 0,
            _GROUP_KERNEL: 0,
            _RESET_KERNEL: 0,
            _PROJECT_KERNEL: 0,
            _POSTSELECT_KERNEL: 0,
            _PEEK_KERNEL: 0,
        }
        self.expected_targets_len: int | None = None

    def _addr(self, ptr: Any) -> int:
        return ctypes.cast(ptr, ctypes.c_void_p).value or 0

    def _check_buf(self, name: str, ptr: Any, need_bytes: int) -> None:
        addr = self._addr(ptr)
        if addr in self._caller_bufs:
            raise AssertionError(f"{name} received caller-owned buffer {addr:#x}")
        have = self._extents.get(addr)
        if have is not None and have < need_bytes:
            raise AssertionError(f"{name} needs {need_bytes} bytes, buffer has {have}")

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if name not in self.calls:
            return fn

        def spied(*args: Any) -> Any:
            if name == _DCC_KERNEL:
                self._check_buf(name, args[4], int(args[2]) * self._batch)
            elif name == _GROUP_KERNEL:
                self._check_buf(name, args[4], int(args[3]))
            elif name == _RESET_KERNEL:
                self._check_buf(name, args[1], self._num_qubits * self._batch)
                self._check_buf(name, args[2], self._num_qubits * self._batch)
            elif name == _PROJECT_KERNEL:
                self._check_buf(name, args[4], int(args[2]) * self._batch)
                self._check_buf(name, args[7], int(args[5]) * self._batch)
            elif name == _POSTSELECT_KERNEL:
                self._check_buf(name, args[4], int(args[3]))
            elif name == _PEEK_KERNEL:
                n_t = int(args[1])
                t_addr = self._addr(args[2])
                if t_addr in self._caller_bufs:
                    raise AssertionError(f"{name} received caller-owned targets buffer")
                if self.expected_targets_len is not None and n_t != self.expected_targets_len:
                    raise AssertionError(f"{name} n_targets={n_t} != {self.expected_targets_len}")
                vals = (ctypes.c_int64 * n_t).from_address(t_addr)
                if any(q < 0 or q >= self._num_qubits for q in vals):
                    raise AssertionError(f"{name} out-of-range targets: {list(vals)}")
                self._check_buf(name, args[2], n_t * 8)
            self.calls[name] += 1
            return fn(*args)

        return spied


def _attach_spy(
    sim: CosetsideSimulator,
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_qubits: int,
    batch: int,
    caller_bufs: set[int] | None = None,
) -> _SafetyKernelSpy | None:
    if not sim._use_cpp or sim._cpp_lib is None:
        return None
    extents: dict[int, int] = {}
    kept: list[np.ndarray] = []
    real_contig = np.ascontiguousarray

    def rec_contig(*args: Any, **kwargs: Any) -> Any:
        out = real_contig(*args, **kwargs)
        kept.append(out)
        extents[out.ctypes.data] = out.nbytes
        return out

    monkeypatch.setattr(np, "ascontiguousarray", rec_contig)
    spy = _SafetyKernelSpy(sim._cpp_lib, extents, num_qubits, batch, caller_bufs)
    sim._cpp_lib = cast(ctypes.CDLL, spy)
    return spy


# ---- F12: peek_pauli_batch & peek wrappers ----


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_f12_setprofile_dtype_flip_on_targets_is_ignored(
    use_cpp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    c_txt = "H 0 1 2 3\nTICK\nM 0 1 2 3"
    m4 = np.zeros((4, _B), bool)
    m4[:, ::2] = True
    ref = _mk(c_txt, use_cpp=use_cpp)
    ref.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m4)
    base = np.zeros(64, dtype=np.int64)
    t = base[:4]
    t[:] = [0, 1, 2, 3]
    expected = ref.peek_pauli_batch(t, "Z")

    sim = _mk(c_txt, use_cpp=use_cpp)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m4)
    spy = _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B, caller_bufs={t.ctypes.data})
    if spy is not None:
        spy.expected_targets_len = 4
    done = [False]

    def prof(frame: Any, event: str, arg: Any) -> None:
        if done[0]:
            return
        hit_cpp = (
            use_cpp
            and event == "c_return"
            and frame.f_code.co_name == "peek_pauli_batch"
            and arg is np.empty
        )
        hit_py = (
            not use_cpp
            and event == "return"
            and frame.f_code.co_name == "_current_packed_words"
        )
        if hit_cpp or hit_py:
            done[0] = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t.dtype = np.int8

    sys.setprofile(prof)
    try:
        got = sim.peek_pauli_batch(t, "Z")
    finally:
        sys.setprofile(None)
    assert done[0] and got.shape == (4, _B)
    assert np.array_equal(got, expected)
    if use_cpp and spy is not None:
        assert spy.calls[_PEEK_KERNEL] == 1


@_SKIP_NO_SIGALRM
@_SKIP_NO_CPP
def test_f12_sigalrm_dtype_race_on_targets_never_reaches_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c_txt = "H 0 1 2 3\nTICK\nM 0 1 2 3"
    sim = _mk(c_txt, use_cpp=True)
    base = np.zeros(64, dtype=np.int64)
    t = base[:4]
    t[:] = [0, 0, 0, 0]
    spy = _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B, caller_bufs={t.ctypes.data})
    assert spy is not None

    fired = [0]

    def handler(_sig: int, _frm: Any) -> None:
        fired[0] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t.dtype = np.int8 if t.dtype == np.int64 else np.int64

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 5e-5, 5e-5)
    ok = bad = 0
    try:
        for _ in range(1500):
            try:
                out = sim.peek_pauli_batch(t, "Z")
                assert out.shape[0] in (4, 32) and out.shape[1] == _B
                ok += 1
            except ValueError:
                bad += 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert fired[0] > 0 and ok + bad == 1500 and ok > 0 and spy.calls[_PEEK_KERNEL] == ok


@pytest.mark.parametrize("kind", ["values", "resize"])
@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_f12_post_check_target_mutation_is_isolated(
    kind: str, use_cpp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    c_txt = "H 0 1 2 3\nTICK\nM 0 1 2 3"
    t = np.array([0, 1, 2, 3], dtype=np.int64)
    expected = _mk(c_txt, use_cpp=use_cpp).peek_pauli_batch(t, "Z")

    sim = _mk(c_txt, use_cpp=use_cpp)
    spy = _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B, caller_bufs={t.ctypes.data})
    if spy is not None:
        spy.expected_targets_len = 4
    real = sim._current_packed_words
    done = [False]

    def hook() -> Any:
        if not done[0]:
            done[0] = True
            if kind == "values":
                t[:] = 1_000_000
            else:
                t.resize((1,), refcheck=False)
        return real()

    monkeypatch.setattr(sim, "_current_packed_words", hook)
    got = sim.peek_pauli_batch(t, "Z")
    assert got.shape == (4, _B)
    assert np.array_equal(got, expected)
    assert done[0]


@_SKIP_NO_CPP
def test_f12_peek_wrappers_propagate_target_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    c_txt = "H 0 1\nTICK\nM 0 1 2 3"
    sim = _mk(c_txt, use_cpp=True)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), np.zeros((2, _B), bool))
    t = np.array([0, 1], dtype=np.int64)
    ref_born = sim.peek_pauli_born_batch(t, "Y")
    ref_state = sim.get_current_noisy_tableau_pauli_state(t, "X")

    spy = _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B, caller_bufs={t.ctypes.data})
    real = sim._current_packed_words
    if spy is not None:
        spy.expected_targets_len = 2

    def hook() -> Any:
        t[:] = 999
        return real()

    monkeypatch.setattr(sim, "_current_packed_words", hook)
    assert np.array_equal(sim.peek_pauli_born_batch(t, "Y"), ref_born)
    t = np.array([0, 1], dtype=np.int64)
    if spy is not None:
        spy._caller_bufs = {t.ctypes.data}
    assert np.array_equal(sim.get_current_noisy_tableau_pauli_state(t, "X"), ref_state)

    if spy is not None:
        spy.expected_targets_len = 1
    for fn in (sim.peek_x, sim.peek_y, sim.peek_z):
        with pytest.raises(ValueError, match="peek target out of range"):
            fn(99)
        assert isinstance(fn(0), (int, np.ndarray))


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_f12_edge_containers_dtypes_and_uint64_bounds(use_cpp: bool) -> None:
    sim = _mk("H 0 1 2 3\nTICK\nM 0 1 2 3 4 5", use_cpp=use_cpp)
    ref = sim.peek_pauli_batch(np.array([0, 3], dtype=np.int64), "Z")

    ro = np.array([0, 3], dtype=np.int64)
    ro.flags.writeable = False
    assert np.array_equal(sim.peek_pauli_batch(ro, "Z"), ref)
    assert np.array_equal(sim.peek_pauli_batch(np.array([0, 3], dtype=np.uint64), "Z"), ref)

    seen_hooks: list[str] = []

    class LyingSub(np.ndarray):
        @property
        def shape(self) -> tuple[int, ...]:
            seen_hooks.append("shape")
            return (1,)

        def __getitem__(self, item: Any) -> Any:
            seen_hooks.append("getitem")
            return super().__getitem__(item)

    sub = np.array([0, 3], dtype=np.int64).view(LyingSub)
    assert np.array_equal(sim.peek_pauli_batch(sub, "Z"), ref)
    assert seen_hooks == []

    calls: list[int] = []

    class LegacyArr:
        def __array__(self, dtype: Any = None) -> np.ndarray:
            calls.append(1)
            return np.array([0, 3], dtype=np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sim.peek_pauli_batch(cast(Any, LegacyArr()), "Z")
    assert len(calls) == 1 and np.array_equal(got, ref)

    assert sim.peek_pauli_batch([], "Z").shape == (0, _B)
    assert sim.peek_pauli_batch(np.array([], dtype=np.float64), "Z").shape == (0, _B)
    for bad in (
        np.array([2**63], dtype=np.uint64),
        np.array([2**64 - 1], dtype=np.uint64),
        [1.5],
        [True],
        np.array([[0, 1]], dtype=np.int64),
        np.array(0, dtype=np.int64),
        np.zeros((0, 2), dtype=np.int64),
    ):
        with pytest.raises(ValueError):
            sim.peek_pauli_batch(cast(Any, bad), "Z")


def test_f12_huge_2d_broadcast_rejects_before_copy_without_memory_error() -> None:
    sim = _mk("H 0 1\nTICK\nM 0 1", use_cpp=False)
    for shape in ((2**25, 2**25), (2**17, 2**17)):
        huge = np.broadcast_to(0, shape)
        t0 = time.monotonic()
        with pytest.raises(ValueError, match="1-D"):
            sim.peek_pauli_batch(cast(Any, huge), "Z")
        assert time.monotonic() - t0 < 1.0


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_f12_setprofile_recheck_after_copy_rejects_float64_and_2d(use_cpp: bool) -> None:
    sim = _mk("H 0 1 2 3\nTICK\nM 0 1 2 3", use_cpp=use_cpp)
    t_f = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64).view(np.int64).copy()
    t_2d = np.array([0, 1, 2, 3], dtype=np.int64)

    def run_mut(arr: np.ndarray, mut: Callable[[], None]) -> None:
        fired = [False]

        def prof(frame: Any, event: str, arg: Any) -> None:
            if (
                not fired[0]
                and event == "c_call"
                and frame.f_code.co_name == "_own_peek_targets"
                and arg is np.array
            ):
                fired[0] = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mut()

        sys.setprofile(prof)
        try:
            with pytest.raises(ValueError, match="1-D"):
                sim.peek_pauli_batch(arr, "Z")
        finally:
            sys.setprofile(None)
        assert fired[0]

    run_mut(t_f, lambda: setattr(t_f, "dtype", np.float64))
    run_mut(t_2d, lambda: t_2d.resize((2, 2), refcheck=False))


# ---- H2: re-entrancy & strict atomicity ----


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_h2_profile_replay_to_replay_is_atomic(
    use_cpp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    c_txt = "H 0 1\nS 0\nH 0 1 2 3\nTICK\nM 0 1 2 3 4 5"
    h2 = stim.CircuitInstruction("H", [0, 1])
    s0 = stim.CircuitInstruction("S", [0])
    m2 = np.zeros((2, _B), bool)
    m2[:, ::2] = True

    ref = _mk(c_txt, use_cpp=use_cpp)
    ref.do_conditional_clifford(h2, np.ones((2, _B), bool))
    ref.do_conditional_clifford(s0, np.ones((1, _B), bool))

    sim = _mk(c_txt, use_cpp=use_cpp)
    _attach_spy(sim, monkeypatch, num_qubits=6, batch=_B)
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_check_dcc_mask":
            fired[0] = True
            sim.do_conditional_clifford(h2, np.ones((2, _B), bool))
            sim.do_conditional_clifford(s0, np.ones((1, _B), bool))

    sys.setprofile(prof)
    try:
        with pytest.raises(ValueError, match="condition_met_mask has shape"):
            sim.do_conditional_clifford(h2, m2)
    finally:
        sys.setprofile(None)

    assert fired[0] and sim._step_cursor == ref._step_cursor == 2
    assert _fp(sim) == _fp(ref)
    with pytest.raises(RuntimeError, match=r"ValueError.*clear\(\)"):
        sim.do_conditional_clifford(
            stim.CircuitInstruction("H", [0, 1, 2, 3]), np.ones((4, _B), bool)
        )
    sim.clear()
    assert sim._step_cursor == 0 and sim.active_shots == {}


@_SKIP_NO_SIGALRM
@_SKIP_NO_CPP
def test_h2_sigalrm_reentrant_dcc_never_overflows(monkeypatch: pytest.MonkeyPatch) -> None:
    c_txt = "H 0 1\nS 0\nH 0 1 2 3\nTICK\nM 0 1 2 3 4 5"
    h2 = stim.CircuitInstruction("H", [0, 1])
    s0 = stim.CircuitInstruction("S", [0])
    m2 = np.zeros((2, _B), bool)
    m2[:, ::2] = True
    ones2, ones1 = np.ones((2, _B), bool), np.ones((1, _B), bool)
    sim = _mk(c_txt, use_cpp=True)
    _attach_spy(sim, monkeypatch, num_qubits=6, batch=_B)
    state = {"on": False, "fired": False, "fired_count": 0}

    def reenter(_sig: int, _frm: Any) -> None:
        if state["on"] and not state["fired"] and sim._step_cursor == 0:
            state["fired"] = True
            state["fired_count"] += 1
            sim.do_conditional_clifford(h2, ones2)
            sim.do_conditional_clifford(s0, ones1)

    old = signal.signal(signal.SIGALRM, reenter)
    try:
        for idx in range(600):
            if sim._step_cursor != 0 or idx % 20 == 0:
                sim = _mk(c_txt, use_cpp=True)
                _attach_spy(sim, monkeypatch, num_qubits=6, batch=_B)
            state.update(on=True, fired=False)
            signal.setitimer(signal.ITIMER_REAL, 2e-5 + (idx % 5) * 1e-5)
            try:
                sim.do_conditional_clifford(h2, m2)
            except ValueError:
                pass
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                state["on"] = False
            assert sim._step_cursor in (0, 1, 2)
            if state["fired"]:
                assert sim._step_cursor == 2
    finally:
        signal.signal(signal.SIGALRM, old)
    assert state["fired_count"] > 0


@_SKIP_NO_CPP
def test_h2_replay_to_append_and_clear_rollbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    c_txt = "H 0 1\nTICK\nM 0 1 2 3"
    h2 = stim.CircuitInstruction("H", [0, 1])
    h4 = stim.CircuitInstruction("H", [0, 1, 2, 3])
    m2 = np.zeros((2, _B), bool)
    m2[:, ::2] = True

    ref = _mk(c_txt)
    ref.do_conditional_clifford(h2, np.ones((2, _B), bool))
    ref._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    ref._do_bare_instruction(stim.CircuitInstruction("M", [0, 1, 2, 3]))

    sim = _mk(c_txt)
    _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B)
    real = sim._next_step_meta
    done = [False]

    def wrapped(op: stim.CircuitInstruction) -> Any:
        if not done[0]:
            done[0] = True
            sim.do_conditional_clifford(h2, np.ones((2, _B), bool))
            sim._do_bare_instruction(stim.CircuitInstruction("TICK", []))
            sim._do_bare_instruction(stim.CircuitInstruction("M", [0, 1, 2, 3]))
        return real(op)

    monkeypatch.setattr(sim, "_next_step_meta", wrapped)
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(h4, m2)
    assert len(sim._compiled_ref.steps) == len(ref._compiled_ref.steps) == 3
    assert sim._step_cursor == ref._step_cursor == 3
    assert _fp(sim) == _fp(ref)
    with pytest.raises(RuntimeError, match=r"ValueError.*clear\(\)"):
        sim.do_conditional_clifford(h4, np.ones((4, _B), bool))
    sim.clear()
    assert sim._step_cursor == 0 and sim.active_shots == {}

    c_clr = "H 0 1 2 3\nTICK\nM 0 1 2 3"
    ref_clr = _mk(c_clr)
    ref_clr.run()
    ref_clr.clear()
    sim_clr = _mk(c_clr)
    sim_clr.run()
    clr_done = [False]

    def prof_clr(frame: Any, event: str, _arg: Any) -> None:
        if not clr_done[0] and event == "return" and frame.f_code.co_name == "_check_dcc_mask":
            clr_done[0] = True
            sim_clr.clear()

    sys.setprofile(prof_clr)
    try:
        with pytest.raises(ValueError, match="condition_met_mask has shape"):
            sim_clr.do_conditional_clifford(h2, m2)
    finally:
        sys.setprofile(None)
    assert clr_done[0] and sim_clr._step_cursor == 0 and _fp(sim_clr) == _fp(ref_clr)


def _run_h2_hook(
    c_txt: str,
    prelude: int,
    outer_op: stim.CircuitInstruction,
    outer_mask: np.ndarray,
    fn_name: str,
    hook_fn: Callable[[CosetsideSimulator], None],
    expect_cursor: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    interactive: bool = True,
) -> tuple[CosetsideSimulator, str]:
    sim = _mk(c_txt)
    _attach_spy(sim, monkeypatch, num_qubits=6, batch=_B)
    for idx in range(prelude):
        if interactive:
            sim.interactive_do(sim.circuit[idx])
        else:
            sim._do_bare_instruction(sim.circuit[idx])
    done = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not done[0] and event == "return" and frame.f_code.co_name == fn_name:
            done[0] = True
            hook_fn(sim)

    sys.setprofile(prof)
    try:
        with pytest.raises(ValueError, match="condition_met_mask"):
            sim.do_conditional_clifford(outer_op, outer_mask)
    finally:
        sys.setprofile(None)
    assert done[0] and sim._step_cursor == expect_cursor
    return sim, _fp(sim)


@_SKIP_NO_CPP
def test_h2_replays_mixed_alltrue_pauli_and_overlap_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    h2 = stim.CircuitInstruction("H", [0, 1])
    s0 = stim.CircuitInstruction("S", [0])
    tick = stim.CircuitInstruction("TICK", [])
    m2 = np.zeros((2, _B), bool)

    ref1 = _mk("H 0 1\nS 0\nH 0 1 2 3")
    ref1.do_conditional_clifford(h2, np.ones((2, _B), bool))
    ref1.do_conditional_clifford(s0, np.ones((1, _B), bool))
    _, fp1 = _run_h2_hook(
        "H 0 1\nS 0\nH 0 1 2 3", 0, h2, m2, "_replays",
        lambda s: (s.do_conditional_clifford(h2, np.ones((2, _B), bool)),
                   s.do_conditional_clifford(s0, np.ones((1, _B), bool))),
        2, monkeypatch,
    )
    assert fp1 == _fp(ref1)

    m_txt = "X_ERROR(1) 0\nM 0\nCX rec[-1] 1 2 3\nH 0 1 2 3\nTICK\nM 0 1 2 3"
    m_circ = stim.Circuit(m_txt)
    ref_m = _mk(m_txt)
    ref_m._do_bare_instruction(m_circ[0])
    ref_m._do_bare_instruction(m_circ[1])
    ref_m.do_conditional_clifford(m_circ[2], np.ones((2, _B), bool))
    _, fp_m = _run_h2_hook(
        m_txt, 2, m_circ[2], m2, "_check_dcc_mask",
        lambda s: s.do_conditional_clifford(m_circ[2], np.ones((2, _B), bool)),
        3, monkeypatch, interactive=False,
    )
    assert fp_m == _fp(ref_m)

    _run_h2_hook(
        "H 0 1\nTICK\nH 0 1 2 3", 0, h2, np.ones((2, _B), bool), "_check_dcc_mask",
        lambda s: (s.do_conditional_clifford(h2, np.ones((2, _B), bool)),
                   s._do_bare_instruction(tick)),
        2, monkeypatch,
    )
    x2 = stim.CircuitInstruction("X", [0, 1])
    ref_x = _mk("X 0 1\nTICK\nX 0 1 2 3")
    ref_x.do_conditional_clifford(x2, np.ones((2, _B), bool))
    ref_x._do_bare_instruction(tick)
    _, fp_x = _run_h2_hook(
        "X 0 1\nTICK\nX 0 1 2 3", 0, x2, m2, "_check_dcc_mask",
        lambda s: (s.do_conditional_clifford(x2, np.ones((2, _B), bool)),
                   s._do_bare_instruction(tick)),
        2, monkeypatch,
    )
    assert fp_x == _fp(ref_x)

    cx2 = stim.CircuitInstruction("CX", [0, 1, 2, 3])
    ref_ov = _mk("CX 0 1 2 3\nTICK\nCX 0 1 1 2 2 3")
    ref_ov.do_conditional_clifford(cx2, np.ones((2, _B), bool))
    ref_ov._do_bare_instruction(tick)
    _, fp_ov = _run_h2_hook(
        "CX 0 1 2 3\nTICK\nCX 0 1 1 2 2 3", 0, cx2, m2, "_check_dcc_mask",
        lambda s: (s.do_conditional_clifford(cx2, np.ones((2, _B), bool)),
                   s._do_bare_instruction(tick)),
        2, monkeypatch,
    )
    assert fp_ov == _fp(ref_ov)


class _IntTrigger:
    def __init__(self, v: bool, cb: Callable[[], None], fired: list[bool]) -> None:
        self.v, self.cb, self.fired = v, cb, fired

    def __bool__(self) -> bool:
        if not self.fired[0]:
            self.fired[0] = True
            self.cb()
        return self.v

    def __int__(self) -> int:
        return int(self.__bool__())


@_SKIP_NO_CPP
def test_h2_array_and_int_reentrancy_in_dcc() -> None:
    h2 = stim.CircuitInstruction("H", [0, 1])
    cx_ov = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    m_ov = np.zeros((2, _B), bool)
    m_ov[:, ::2] = True
    ref1 = _mk("H 0 1\nCX 0 1 1 2")
    ref1.do_conditional_clifford(h2, np.ones((2, _B), bool))
    ref1.do_conditional_clifford(cx_ov, m_ov)
    sim1 = _mk("H 0 1\nCX 0 1 1 2")

    class ReenterArr:
        def __array__(self, dtype: Any = None) -> np.ndarray:
            sim1.do_conditional_clifford(h2, np.ones((2, _B), bool))
            return m_ov

    sim1.do_conditional_clifford(cx_ov, cast(Any, ReenterArr()))
    assert _fp(sim1) == _fp(ref1)

    h01, h23 = stim.CircuitInstruction("H", [0, 1]), stim.CircuitInstruction("H", [2, 3])
    tick = stim.CircuitInstruction("TICK", [])
    ref2 = _mk("H 0 1\nTICK\nH 2 3")
    ref2.do_conditional_clifford(h01, m_ov)
    ref2._do_bare_instruction(tick)
    ref2.postselect_z_batch(0, np.arange(_B) % 2 == 0, np.ones(_B, np.uint8), "Z")
    ref2.do_conditional_clifford(h23, m_ov)
    sim2 = _mk("H 0 1\nTICK\nH 2 3")
    sim2.do_conditional_clifford(h01, m_ov)
    sim2._do_bare_instruction(tick)
    fired2 = [False]
    obj_m = np.empty((2, _B), dtype=object)
    for idx in np.ndindex(2, _B):
        obj_m[idx] = _IntTrigger(
            bool(m_ov[idx]),
            lambda: sim2.postselect_z_batch(0, np.arange(_B) % 2 == 0, np.ones(_B, np.uint8), "Z"),
            fired2,
        )
    sim2.do_conditional_clifford(h23, cast(Any, obj_m))
    assert fired2[0] and _fp(sim2) == _fp(ref2)


@_SKIP_NO_CPP
def test_reset_int_reentrancy_in_ascontiguousarray() -> None:
    r0, r1 = np.zeros((4, _B), bool), np.zeros((4, _B), bool)
    r1[0, ::2] = True
    h4 = stim.CircuitInstruction("H", [0, 1, 2, 3])
    ref = _mk("H 0 1 2 3\nTICK\nM 0 1 2 3")
    ref.do_conditional_clifford(h4, np.zeros((4, _B), bool))
    ref.postselect_z_batch(1, np.arange(_B) % 2 == 0, np.ones(_B, np.uint8), "Z")
    ref.do_unmatched_noisy_reset_z(r0, r1)
    sim = _mk("H 0 1 2 3\nTICK\nM 0 1 2 3")
    sim.do_conditional_clifford(h4, np.zeros((4, _B), bool))
    fired = [False]
    obj_r1 = np.empty((4, _B), dtype=object)
    for idx in np.ndindex(4, _B):
        obj_r1[idx] = _IntTrigger(
            bool(r1[idx]),
            lambda: sim.postselect_z_batch(1, np.arange(_B) % 2 == 0, np.ones(_B, np.uint8), "Z"),
            fired,
        )
    sim.do_unmatched_noisy_reset_z(r0, cast(Any, obj_r1))
    assert fired[0] and _fp(sim) == _fp(ref)


@contextlib.contextmanager
def _profile_replays_in_nsm(action: Callable[[], None]) -> Iterator[list[bool]]:
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired[0]
            and event == "return"
            and frame.f_code.co_name == "_replays"
            and frame.f_back is not None
            and frame.f_back.f_code.co_name == "_next_step_meta"
        ):
            fired[0] = True
            action()

    sys.setprofile(prof)
    try:
        yield fired
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_h2_next_step_meta_replays_while_overlap_alltrue_rejects(use_cpp: bool) -> None:
    c_txt = "I 0\nCX 0 1 1 2\nS 0\nTICK\nM 0 1 2"
    i0 = stim.CircuitInstruction("I", [0])
    cx = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    ones2 = np.ones((2, _B), bool)

    ref = _mk(c_txt, use_cpp=use_cpp)
    ref.interactive_do(i0)
    ref.do_conditional_clifford(cx, ones2)

    sim = _mk(c_txt, use_cpp=use_cpp)
    sim.interactive_do(i0)
    steps_before = len(sim._compiled_ref.steps)
    with (
        _profile_replays_in_nsm(lambda: sim.do_conditional_clifford(cx, ones2)) as fired,
        pytest.raises(ValueError, match="interactive_do"),
    ):
        sim.do_conditional_clifford(cx, ones2)
    assert fired[0] and sim._step_cursor == ref._step_cursor == 2
    assert len(sim._compiled_ref.steps) == steps_before == 5
    assert _fp(sim) == _fp(ref)


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_h2_next_step_meta_replays_refreshes_packed_words_on_overlap_cx(
    use_cpp: bool,
) -> None:
    c_txt = "H 0 1\nCX 0 1 1 2\nTICK\nM 0 1 2"
    h2 = stim.CircuitInstruction("H", [0, 1])
    cx = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    ones2 = np.ones((2, _B), bool)
    m_mix = np.zeros(_B, bool)
    m_mix[::2] = True

    def make_gfn(s: CosetsideSimulator, seen: list[Any]) -> Callable[[int], np.ndarray]:
        def gfn(g_idx: int) -> np.ndarray:
            if g_idx == 0:
                z0 = s.peek_z(0)
                seen.append(z0)
                return np.ones(_B, bool) if z0 == 0 else np.zeros(_B, bool)
            return m_mix

        return gfn

    ref = _mk(c_txt, use_cpp=use_cpp)
    ref_seen: list[Any] = []
    ref.do_conditional_clifford(h2, ones2)
    ref.do_conditional_clifford(cx, condition_group_fn=make_gfn(ref, ref_seen))

    sim = _mk(c_txt, use_cpp=use_cpp)
    sim_seen: list[Any] = []
    with _profile_replays_in_nsm(lambda: sim.do_conditional_clifford(h2, ones2)) as fired:
        sim.do_conditional_clifford(cx, condition_group_fn=make_gfn(sim, sim_seen))
    assert fired[0] and sim_seen == ref_seen == [0]
    assert sim._step_cursor == ref._step_cursor == 2
    assert _fp(sim) == _fp(ref)


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_h2_next_step_meta_replays_mismatched_groups_post_step_rollback(use_cpp: bool) -> None:
    c_txt = "H 0 1\nS 0 1 2 3\nTICK\nM 0 1 2 3"
    h2 = stim.CircuitInstruction("H", [0, 1])
    ones2 = np.ones((2, _B), bool)

    ref = _mk(c_txt, use_cpp=use_cpp)
    ref.do_conditional_clifford(h2, ones2)

    sim = _mk(c_txt, use_cpp=use_cpp)
    steps_before = len(sim._compiled_ref.steps)
    with (
        _profile_replays_in_nsm(lambda: sim.do_conditional_clifford(h2, ones2)) as fired,
        pytest.raises(ValueError, match="condition_met_mask has shape"),
    ):
        sim.do_conditional_clifford(h2, ones2)
    assert fired[0] and sim._step_cursor == ref._step_cursor == 1
    assert len(sim._compiled_ref.steps) == steps_before
    assert _fp(sim) == _fp(ref)


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_h2_object_mask_bool_raise_in_np_all_rolls_back_atomically(use_cpp: bool) -> None:
    class BoomBool:
        def __bool__(self) -> bool:
            raise RuntimeError("boom in __bool__")

        def __int__(self) -> int:
            return 0

    c_txt = "H 0 1\nTICK\nM 0 1"
    h2 = stim.CircuitInstruction("H", [0, 1])
    m_boom = np.empty((2, _B), dtype=object)
    m_boom[:, :] = BoomBool()

    sim_rep = _mk(c_txt, use_cpp=use_cpp)
    before_rep = _fp(sim_rep)
    steps_rep = len(sim_rep._compiled_ref.steps)
    with pytest.raises(RuntimeError, match="boom in __bool__"):
        sim_rep.do_conditional_clifford(h2, cast(Any, m_boom))
    assert sim_rep._step_cursor == 0 and len(sim_rep._compiled_ref.steps) == steps_rep
    assert _fp(sim_rep) == before_rep

    sim_app = _mk(c_txt, use_cpp=use_cpp)
    sim_app.run()
    before_app = _fp(sim_app)
    cursor_app = sim_app._step_cursor
    steps_app = len(sim_app._compiled_ref.steps)
    with pytest.raises(RuntimeError, match="boom in __bool__"):
        sim_app.do_conditional_clifford(h2, cast(Any, m_boom))
    assert sim_app._step_cursor == cursor_app
    assert len(sim_app._compiled_ref.steps) == steps_app
    assert _fp(sim_app) == before_app


# ---- R3-F2: mutable num_qubits / batch_size ----


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_r3_f2_mutating_num_qubits_and_batch_size_after_init(
    use_cpp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = _mk("H 0 1 2 3\nS 0 1 2 3\nTICK\nM 0 1 2 3", use_cpp=use_cpp)
    _attach_spy(sim, monkeypatch, num_qubits=4, batch=_B)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), np.zeros((4, _B), bool))
    sim.num_qubits = 1000
    sim.batch_size = 1

    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(
            stim.CircuitInstruction("S", [0, 1, 2, 3]), np.ones((4, 1), bool)
        )
    with pytest.raises(ValueError, match="reset_zero_mask has shape"):
        sim.do_unmatched_noisy_reset_z(np.ones((2, _B), bool), np.zeros((2, _B), bool))
    with pytest.raises(ValueError, match="reset_zero_mask has shape"):
        sim.do_unmatched_noisy_reset_z(np.ones((4, 1), bool), np.zeros((4, 1), bool))
    for pauli in ("Z", "X"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        for n_bad, b_bad in ((1000, 1), (1000, _B), (2, _B), (4, 1)):
            sim.num_qubits, sim.batch_size = n_bad, b_bad
            with pytest.raises(ValueError, match="reset_zero_mask has shape"):
                sim.do_unmatched_noisy_reset_z(
                    np.ones((n_bad, b_bad), bool), np.zeros((n_bad, b_bad), bool), pauli=p_lit
                )
    sim.num_qubits, sim.batch_size = 1000, 1
    with pytest.raises(ValueError, match="peek target out of range"):
        sim.peek_pauli_batch([500], "Z")

    out = sim.peek_pauli_batch([0, 1], "Z")
    assert out.shape == (2, _B)
    assert isinstance(sim.get_current_noisy_tableau_pauli_state([0, 1], "Z"), np.ndarray)
    assert len(sim.active_shots) > 0
    assert all(st.num_qubits == 4 for st in sim.active_shots.values())
    sim.num_qubits, sim.batch_size = 2, 1
    for pauli in ("Z", "X"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        sim.do_unmatched_noisy_reset_z(
            np.ones((4, _B), bool), np.zeros((4, _B), bool), pauli=p_lit
        )

    sim2 = _mk("M 0 1 2 3", use_cpp=use_cpp)
    sim2.batch_size = 1
    sim2.run()
    assert sim2.get_final_measurement_records().shape == (_B, 4)

    sim3 = _mk("H 0 1\nM 0 1", use_cpp=use_cpp)
    sim3.batch_size = 1
    sim3.run()
    assert sim3.get_final_measurement_records().shape == (_B, 2)


# ---- D-03: process_shots_cpp & Analysis.generate_batch ----


def _d03_analysis() -> Any:
    circ = stim.Circuit(
        """
        R 0 1 2
        TICK
        I_ERROR[LEAKAGE_TRANSITION_1: (0.01, U-->2) (0.02, U-->0)] 0 2
        X_ERROR(0.001) 0 2
        CX 0 1
        CX 2 1
        MR 1
        DETECTOR rec[-1]
        MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) (0.01, 0) (0.03, 1): 0 2] 0 0
        I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 2-->U)] 0 2
        M 0 2
        DETECTOR rec[-1] rec[-2] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-2]
        """
    )
    gen = MarginalLeakageDemGenerator()
    recs = np.zeros((8, circ.num_measurements), dtype=bool)
    analysis = gen(circ, recs).analysis
    assert analysis is not None
    return analysis


@_SKIP_NO_MARGINAL_CPP
def test_d03_generate_batch_and_process_shots_copy_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _marginal_cpp_available():
        pytest.skip("marginal DEM C++ kernel library unavailable")
    analysis = _d03_analysis()
    n_flags = len(analysis.flag_env_offsets) - 1
    raised_u8 = np.zeros((8, n_flags), dtype=np.uint8)
    raised_u8[:, 0] = 1
    raised_bool = raised_u8.astype(bool)

    for dec, rew in ((False, False), (False, True), (True, True)):
        exp = analysis.generate_batch(raised_bool, decompose_errors=dec, reweight_only=rew)
        got = analysis.generate_batch(cast(Any, raised_u8), decompose_errors=dec, reweight_only=rew)
        assert len(got) == len(exp) == 8
        for a, b in zip(got, exp):
            assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b

    lib = marginal_dem_kernels.get_marginal_dem_kernels_lib()
    extents: dict[int, int] = {}
    real_c_ptr = marginal_dem_kernels.c_ptr
    done = [False]
    caller_addr = raised_u8.ctypes.data

    def spied_c_ptr(arr: Any, ptr_t: Any) -> Any:
        if not done[0] and arr is not None and arr.dtype == np.uint8:
            done[0] = True
            raised_u8.resize((1, 1), refcheck=False)
        if arr is not None:
            extents[arr.ctypes.data] = arr.nbytes
        return real_c_ptr(arr, ptr_t)

    class MarginalSpy:
        def __init__(self, inner: ctypes.CDLL) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            fn = getattr(self._inner, name)
            if name != "marginal_process_shots":
                return fn

            def wrapped(*args: Any) -> Any:
                n_s, n_f = int(args[0].value), int(args[1].value)
                ptr = ctypes.cast(args[2], ctypes.c_void_p).value or 0
                assert ptr != caller_addr, "marginal_process_shots received caller-owned raised"
                have = extents.get(ptr)
                assert have is not None and have >= n_s * n_f
                return fn(*args)

            return wrapped

    expected = analysis.generate_batch(raised_bool, decompose_errors=False, reweight_only=False)
    monkeypatch.setattr(marginal_dem_kernels, "c_ptr", spied_c_ptr)
    monkeypatch.setattr(marginal_dem_kernels, "_LIB_INSTANCE", MarginalSpy(lib))
    got = analysis.generate_batch(cast(Any, raised_u8), decompose_errors=False, reweight_only=False)
    assert done[0] and got == expected


@_SKIP_NO_MARGINAL_CPP
def test_d03_1d_rejection_and_bool_single_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _marginal_cpp_available():
        pytest.skip("marginal DEM C++ kernel library unavailable")
    analysis = _d03_analysis()
    n_flags = len(analysis.flag_env_offsets) - 1
    with pytest.raises(ValueError, match="Expected 2-D"):
        marginal_dem_kernels.process_shots_cpp(
            cast(Any, np.zeros(n_flags, dtype=np.uint8)),
            analysis.pred_offsets,
            analysis.pred_flags,
            analysis.flag_env_offsets,
            analysis.flag_env_sym_ids,
            analysis.flag_env_probs,
            len(analysis.symptoms),
            0,
        )

    raised_bool = np.zeros((4, n_flags), dtype=bool)
    raised_bool[:, 0] = True
    contig_calls = [0]
    real_contig = np.ascontiguousarray

    def count_contig(*args: Any, **kwargs: Any) -> Any:
        contig_calls[0] += 1
        return real_contig(*args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", count_contig)
    out = analysis.generate_batch(raised_bool, decompose_errors=False, reweight_only=False)
    assert len(out) == 4 and contig_calls[0] == 0


@_SKIP_NO_MARGINAL_CPP
def test_d03_low2_process_shots_cpp_rejects_huge_3d_and_int32_overflow() -> None:
    if not _marginal_cpp_available():
        pytest.skip("marginal DEM C++ kernel library unavailable")
    analysis = _d03_analysis()
    ns = len(analysis.symptoms)
    t0 = time.monotonic()
    with pytest.raises(ValueError, match="Expected 2-D"):
        marginal_dem_kernels.process_shots_cpp(
            cast(Any, np.broadcast_to(np.uint8(0), (2**17, 2**17, 2**17))),
            analysis.pred_offsets,
            analysis.pred_flags,
            analysis.flag_env_offsets,
            analysis.flag_env_sym_ids,
            analysis.flag_env_probs,
            ns,
            0,
        )
    assert time.monotonic() - t0 < 0.5

    po0, pf0 = np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32)
    feo0, fes0 = np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32)
    fep0 = np.empty(0, dtype=np.float64)
    for bad_shape in ((2**31, 0), (0, 2**31), (2**31, 2**31)):
        t1 = time.monotonic()
        with pytest.raises(ValueError, match="int32 bounds"):
            marginal_dem_kernels.process_shots_cpp(
                cast(Any, np.broadcast_to(np.uint8(0), bad_shape)),
                po0, pf0, feo0, fes0, fep0, 0, 0,
            )
        assert time.monotonic() - t1 < 0.5

    def run_post_copy_resize(mut_shape: tuple[int, ...], pat: str) -> None:
        arr = np.zeros((2, 0) if mut_shape[1] == 0 else (2, 2), dtype=np.uint8)
        fired = [False]

        def prof(frame: Any, event: str, arg: Any) -> None:
            if (
                not fired[0]
                and event == "c_call"
                and frame.f_code.co_name == "_own_raised_matrix"
                and arg is np.array
            ):
                fired[0] = True
                arr.resize(mut_shape, refcheck=False)

        sys.setprofile(prof)
        try:
            with pytest.raises(ValueError, match=pat):
                marginal_dem_kernels.process_shots_cpp(
                    arr, po0, pf0, feo0, fes0, fep0, 0, 0,
                )
        finally:
            sys.setprofile(None)
        assert fired[0]

    run_post_copy_resize((1, 2, 2), "Expected 2-D")
    run_post_copy_resize((2**31, 0), "int32 bounds")
