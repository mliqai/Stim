"""Same-thread re-entrancy, fork and store self-deadlock safety of the engine wrapper (S4a fix 2).

ctypes drops the GIL during a native call, but Python code can also run on the thread that is
inside a native section: a signal handler, a garbage-collection finalizer, an `__array__` hook.
If that code touches the same engine it must neither free it under a pending native call (G1),
nor resize a shot between the header read and the export (G2); and a finalizer that reads a
store while its engine is being created must not deadlock on the store lock (G4). The fork
behaviour (G3, and the lock-order inversion of fix 3) is in `magic_engine_fork_test.py`.

The tests inject the asynchronous event deterministically through a stand-in for the engine's
CDLL (`_Spy`) that runs a hook just before a chosen native call is forwarded - exactly the point
where a handler or finalizer can fire - and that never forwards a call that would touch a freed
handle or a stale buffer size, so a wrapper without the protection is *detected* instead of
corrupting memory. The sanitizer runs of the same scenarios live in the worker's handoff.
"""

from __future__ import annotations

import gc
import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import stimside._magic.live_store as ls
import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions, expansion_to_arrays
from stimside._native import magic_engine as me

_PASSIVE = ("mbe_abi_v1", "mbe_live_engines")
_MUTATORS = ("mbe_import_shot", "mbe_reset", "mbe_destroy")
_N = 70


class _Spy:
    """Stand-in for an engine's CDLL: runs one-shot hooks before a call, flags unsafe calls."""

    def __init__(self, lib: Any, hooks: dict[str, Callable[[], None]]) -> None:
        self._lib, self.hooks = lib, dict(hooks)
        self.log: list[tuple[str, int]] = []  # (call name, handle) of the calls that ran
        self.dead_calls: list[str] = []  # calls on a handle that had been destroyed
        self.stale_exports = 0  # exports whose header was followed by a mutation

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if not name.startswith("mbe_") or name in _PASSIVE:
            return fn

        def call(*args: Any) -> Any:
            hook = self.hooks.pop(name, None)
            if hook is not None:
                hook()  # what a handler / finalizer would do at this very point
            handle = int(args[0]) if args else 0
            if name != "mbe_destroy" and (("mbe_destroy", handle) in self.log):
                self.dead_calls.append(name)
                return -2  # never really touch freed memory
            if name == "mbe_export_shot" and self._stale(handle):
                self.stale_exports += 1
                return -2  # the buffers were sized from a header that no longer holds
            self.log.append((name, handle))
            return fn(*args)

        return call

    def _stale(self, handle: int) -> bool:
        for name, h in reversed(self.log):
            if h == handle and name == "mbe_shot_header":
                return False
            if h == handle and name in _MUTATORS:
                return True
        return False

    def names(self) -> list[str]:
        return [name for name, _ in self.log]


def _state(n: int = _N, steps: int = 4) -> pq.ShotExpansion:
    rng = np.random.default_rng(11)
    e = pq.ShotExpansion.zero(n)
    for _ in range(steps):
        a = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
        b = pq.pack_bit_vector((rng.random(n) < 0.4).astype(np.uint8), n)
        if not (a.any() or b.any()):
            a = pq.pack_bit_vector(np.eye(n, dtype=np.uint8)[0], n)
        e = pq.apply_pauli_t(e, a, b, int(pq.word_dot_parity(a, b)) & 3, dag=False, merge=True)
    return e


def _spied(
    hooks: dict[str, Callable[[], None]], shots: int = 2
) -> tuple[me.MagicBatchEngine, _Spy]:
    eng = me.MagicBatchEngine(shots, _N, 1e-10)
    for b in range(shots):
        eng.import_shot(b, expansion_to_arrays(_state(_N, 2)))
    spy = _Spy(eng._lib, hooks)
    eng._lib = spy  # type: ignore[assignment]
    return eng, spy


def _call(eng: me.MagicBatchEngine, method: str) -> Any:
    words = np.zeros(eng.n_words, np.uint64)
    return {
        "apply_pauli_mask": lambda: eng.apply_pauli_mask(words, words, 0, np.ones(2, bool)),
        "import_shot": lambda: eng.import_shot(1, expansion_to_arrays(_state())),
        "export_shot": lambda: eng.export_shot(1),
        "reset": eng.reset,
    }[method]()


# The first native call of each wrapper method: where an asynchronous event is injected.
_FIRST_CALL = {
    "apply_pauli_mask": "mbe_apply_pauli_mask",
    "import_shot": "mbe_import_shot",
    "export_shot": "mbe_shot_header",
    "reset": "mbe_reset",
}


def _signal_handler_runs(
    arm: Callable[[Callable[[], None]], None], callback: Callable[[], None]
) -> None:
    """Deliver a real SIGUSR1 to this thread whose handler runs `callback`."""
    arm(callback)
    signal.raise_signal(signal.SIGUSR1)


@pytest.fixture
def sigusr1() -> Iterator[Callable[[Callable[[], None]], None]]:
    """Install a SIGUSR1 handler that runs a callback; restores the old handler afterwards."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers run on the main thread")
    old = signal.getsignal(signal.SIGUSR1)
    box: list[Callable[[], None]] = []
    signal.signal(signal.SIGUSR1, lambda *_: box[0]())

    def arm(callback: Callable[[], None]) -> None:
        box[:] = [callback]

    try:
        yield arm
    finally:
        signal.signal(signal.SIGUSR1, old)


# ---------------------------------------------------------------------------------------------
# G1: close() re-entered on the owner thread inside a section (handler / finalizer)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", ["close", "finalizer", "signal"])
@pytest.mark.parametrize("method", list(_FIRST_CALL))
def test_a_close_from_inside_a_native_section_waits_for_the_section_to_end(
    method: str, trigger: str, sigusr1: Callable[[Callable[[], None]], None]
) -> None:
    box: list[me.MagicBatchEngine] = []
    fire = {
        "close": lambda: box[0].close(),
        "finalizer": lambda: box[0].__del__(),
        "signal": lambda: _signal_handler_runs(sigusr1, lambda: box[0].close()),
    }[trigger]
    eng, spy = _spied({_FIRST_CALL[method]: fire})
    box.append(eng)
    _call(eng, method)  # must neither raise nor reach a freed handle
    assert spy.dead_calls == [], spy.dead_calls
    names = spy.names()
    assert names.count("mbe_destroy") == 1
    assert names.index(_FIRST_CALL[method]) < names.index("mbe_destroy"), names
    assert eng._handle is None  # the deferred close ran when the section ended
    assert not eng._busy and not eng._close_pending  # and left no request behind
    with pytest.raises(RuntimeError, match="closed"):
        eng.reset()


def test_a_deferred_close_is_idempotent_and_frees_the_engine_exactly_once() -> None:
    box: list[me.MagicBatchEngine] = []

    def twice() -> None:
        box[0].close()
        box[0].close()
        box[0].__del__()

    eng, spy = _spied({"mbe_reset": twice})
    box.append(eng)
    live = me.get_magic_engine_lib().mbe_live_engines
    before = live()
    eng.reset()
    assert spy.names().count("mbe_destroy") == 1 and live() == before - 1
    eng.close()
    assert spy.names().count("mbe_destroy") == 1


@pytest.mark.parametrize("inner", list(_FIRST_CALL))
@pytest.mark.parametrize("outer", list(_FIRST_CALL))
def test_any_other_call_into_the_engine_from_inside_a_section_is_refused(
    outer: str, inner: str
) -> None:
    seen: list[BaseException] = []
    box: list[me.MagicBatchEngine] = []

    def reenter() -> None:
        try:
            _call(box[0], inner)
        except RuntimeError as exc:
            seen.append(exc)

    eng, spy = _spied({_FIRST_CALL[outer]: reenter})
    box.append(eng)
    _call(eng, outer)  # the outer call completes normally
    assert len(seen) == 1 and "re-entrant" in str(seen[0]), seen
    assert spy.dead_calls == [] and spy.stale_exports == 0
    alone, alone_spy = _spied({})
    _call(alone, outer)
    assert spy.names() == alone_spy.names()  # the refused call never reached the native code
    _call(eng, "export_shot")  # the engine is not left locked or half-open
    _call(eng, "reset")
    eng.close()


def test_a_refused_reentrant_call_does_not_end_the_section_it_interrupted() -> None:
    """After a handler's call was refused, the section is still running: a second handler's
    close() must still be deferred (not free the engine under the pending native call)."""
    box: list[me.MagicBatchEngine] = []
    refused: list[str] = []

    def two_handlers() -> None:
        try:
            box[0].reset()  # the first handler: refused
        except RuntimeError as exc:
            refused.append(str(exc))
        box[0].close()  # the second handler: must wait for the section to end
        try:
            box[0].export_shot(0)  # still inside the section: refused, not "closed"
        except RuntimeError as exc:
            refused.append(str(exc))

    eng, spy = _spied({"mbe_apply_pauli_mask": two_handlers})
    box.append(eng)
    _call(eng, "apply_pauli_mask")
    assert spy.dead_calls == [] and spy.names().count("mbe_destroy") == 1
    assert spy.names().index("mbe_apply_pauli_mask") < spy.names().index("mbe_destroy")
    assert len(refused) == 2 and all("re-entrant" in r for r in refused), refused


def test_the_lock_is_free_again_after_every_section_even_one_that_raised() -> None:
    eng, _ = _spied({})
    with pytest.raises(ValueError):
        eng.apply_pauli_mask(
            np.zeros(eng.n_words, np.uint64), np.zeros(eng.n_words, np.uint64), 0, np.ones(3, bool)
        )
    done = threading.Event()

    def other() -> None:
        eng.export_shot(0)
        done.set()

    t = threading.Thread(target=other)
    t.start()
    t.join(30)
    assert done.is_set()
    assert not eng._busy and not eng._close_pending
    eng.close()


class _AfterAcquireLock:
    """The engine's lock, running a callback once, right after the first acquire."""

    def __init__(self, inner: Any, callback: Callable[[], None]) -> None:
        self._inner, self._callback = inner, callback

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        got = bool(self._inner.acquire(*args, **kwargs))
        callback, self._callback = self._callback, lambda: None
        callback()
        return got

    def release(self) -> None:
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def test_a_handler_firing_right_after_the_lock_is_taken_does_not_deadlock() -> None:
    """Between acquiring the lock and entering the section the thread owns a lock it has not
    yet flagged: a close() / call from a handler there must run, not wait for itself forever."""
    eng, spy = _spied({})
    result: list[str] = []

    def handler() -> None:
        eng.close()  # a handler on the owner thread, before the section started
        result.append("handler ran")

    eng._lock = _AfterAcquireLock(eng._lock, handler)  # type: ignore[assignment]

    def outer() -> None:
        try:
            eng.reset()
        except RuntimeError as exc:
            result.append(str(exc))

    t = threading.Thread(target=outer, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "deadlock on the engine lock"
    assert result[0] == "handler ran" and "closed" in result[1], result
    assert spy.dead_calls == []


# ---------------------------------------------------------------------------------------------
# G2: header + export of one shot stay atomic against a re-entrant import
# ---------------------------------------------------------------------------------------------


def _big_and_small() -> tuple[Any, Any]:
    rng, big = np.random.default_rng(2), pq.ShotExpansion.zero(_N)
    while big.chi < 33:  # chi grows by at most a factor of two per non-Clifford step
        a = pq.pack_bit_vector((rng.random(_N) < 0.3).astype(np.uint8), _N)
        b = pq.pack_bit_vector((rng.random(_N) < 0.3).astype(np.uint8), _N)
        if a.any() or b.any():
            big = pq.apply_pauli_t(big, a, b, int(pq.word_dot_parity(a, b)), dag=False, merge=True)
    return expansion_to_arrays(big), expansion_to_arrays(pq.ShotExpansion.zero(_N))


class _Bomb:
    """A cyclic garbage object whose finalizer imports a bigger shot into the engine."""

    def __init__(self, eng: me.MagicBatchEngine, shot: Any, outcome: list[str]) -> None:
        self.me, self.eng, self.shot, self.outcome = self, eng, shot, outcome  # `me`: the cycle

    def __del__(self) -> None:
        try:
            self.eng.import_shot(0, self.shot)
            self.outcome.append("imported")
        except RuntimeError as exc:
            self.outcome.append(f"refused: {exc}")


@pytest.mark.parametrize("trigger", ["finalizer", "signal", "direct"])
def test_export_never_sees_a_shot_resized_between_its_header_and_its_export(
    trigger: str, sigusr1: Callable[[Callable[[], None]], None]
) -> None:
    big, small = _big_and_small()
    outcome: list[str] = []
    box: list[me.MagicBatchEngine] = []

    def reimport() -> None:
        try:
            box[0].import_shot(0, big)
            outcome.append("imported")
        except RuntimeError as exc:
            outcome.append(f"refused: {exc}")

    def collect() -> None:
        _Bomb(box[0], big, outcome)  # a cycle (it points to itself): only the collector frees it
        gc.collect()

    fire = {
        "finalizer": collect,
        "signal": lambda: _signal_handler_runs(sigusr1, reimport),
        "direct": reimport,
    }[trigger]
    eng, spy = _spied({"mbe_export_shot": fire})
    box.append(eng)
    eng.import_shot(0, small)
    out = eng.export_shot(0)  # the hook runs between the header and the export
    assert spy.stale_exports == 0, "the export ran on buffers sized from a stale header"
    assert outcome and outcome[0].startswith("refused") and "re-entrant" in outcome[0], outcome
    assert out.chi == small.chi and out.weights.shape == (small.chi,)
    assert eng.export_shot(0).chi == small.chi  # the refused import changed nothing
    eng.import_shot(0, big)  # and the engine is still fully usable
    assert eng.export_shot(0).chi == big.chi
    eng.close()


# ---------------------------------------------------------------------------------------------
# G4: the store lock around engine creation, entered again by a finalizer / handler
# ---------------------------------------------------------------------------------------------


def _first_engine_access_in_a_thread(store: LiveExpansions) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def grab() -> None:
        try:
            out.append(store.engine)
        except RuntimeError as exc:  # reported through the list
            out.append(exc)

    t = threading.Thread(target=grab, daemon=True)
    t.start()
    t.join(60)
    return t, out


def test_a_reentrant_first_access_during_engine_creation_fails_instead_of_deadlocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LiveExpansions(2, 5, 1e-10)
    seen: list[str] = []
    real = ls.MagicBatchEngine

    class Reentrant(real):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any) -> None:
            try:
                assert store.engine  # a finalizer / handler reading the store, mid-creation
            except RuntimeError as exc:
                seen.append(str(exc))
            super().__init__(*args)

    monkeypatch.setattr(ls, "MagicBatchEngine", Reentrant)
    t, out = _first_engine_access_in_a_thread(store)
    assert not t.is_alive(), "deadlock: the store lock was taken again by the same thread"
    assert len(seen) == 1 and "re-entrant" in seen[0], seen
    assert isinstance(out[0], Reentrant) and store.engine is out[0]  # exactly one engine
    store.close()


def test_a_garbage_collected_finalizer_reading_the_store_inside_creation_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LiveExpansions(4, _N, 1e-10)
    reads: list[str] = []

    class Bomb:
        def __del__(self) -> None:
            try:
                assert store.engine
                reads.append("read")
            except RuntimeError as exc:
                reads.append(str(exc))

    class Cycle(Bomb):
        pass

    gc.collect()
    gc.disable()
    try:
        cyc = Cycle()
        cyc.me = cyc  # type: ignore[attr-defined]
        del cyc
        real, fired = me._valid_atol, [False]

        def collecting(atol: Any) -> float:
            if not fired[0]:
                fired[0] = True
                gc.collect()  # an automatic collection pass inside the creation
            return real(atol)

        monkeypatch.setattr(me, "_valid_atol", collecting)
        t, out = _first_engine_access_in_a_thread(store)
    finally:
        gc.enable()
    assert not t.is_alive(), "deadlock: a finalizer waited for the store lock its thread holds"
    assert fired[0] and len(reads) == 1 and "re-entrant" in reads[0], reads
    assert out and store.engine is out[0]
    store.close()
