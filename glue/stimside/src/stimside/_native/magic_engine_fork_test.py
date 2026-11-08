"""`fork()` safety of the engine wrapper (S4a fix 3): no parent lock, fresh child locks.

The parent takes no engine/store lock around a `fork()`, so it never waits for a native section
and cannot be part of a lock-order inversion (a handler that nests engine B while another thread
is inside engine A, and a third thread forking). The child, which has only the forking thread,
renews every engine's and every store's lock and flags, so it never inherits a lock held by a
thread that does not exist there. An engine another thread was inside a native section of is
*poisoned* in the child (a copy-on-fork snapshot, possibly mid-update): any use raises a clear
`RuntimeError` and it is never freed there. An engine that was idle at the fork, or is created
afterwards, works normally.

Every fork scenario runs in a subprocess with a watchdog (the script prints `WATCHDOG` and exits
when it hangs), a hard `subprocess.run` timeout and, in each forked child, an `alarm`: a
regression is a failed assertion, never a hung test run.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

from stimside._magic.live_store import LiveExpansions
from stimside._native import magic_engine as me

_SRC = Path(me.__file__).resolve().parents[2]  # the directory that contains the `stimside` package

_SCRIPT = r"""
import os, signal, sys, threading, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
import stimside._magic.live_store as ls
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
scenario = sys.argv[2]
SHOT = ls.expansion_to_arrays(pq.ShotExpansion.zero(70))
inside, release, forked = threading.Event(), threading.Event(), threading.Event()


def watchdog():
    time.sleep(40)
    print("WATCHDOG", flush=True)
    os._exit(7)


threading.Thread(target=watchdog, daemon=True).start()


def make_engine(n=3):
    eng = me.MagicBatchEngine(n, 70, 1e-10)
    for b in range(n):
        eng.import_shot(b, SHOT)
    return eng


def works(eng):
    assert eng.export_shot(1).chi == 1
    eng.import_shot(2, SHOT)
    eng.reset()
    eng.close()


def refused_as_poisoned(call):
    try:
        call()
    except RuntimeError as exc:
        assert "forked" in str(exc), exc
    else:
        raise AssertionError("a poisoned engine was used")


def poisoned(eng):
    refused_as_poisoned(lambda: eng.export_shot(1))
    refused_as_poisoned(eng.reset)
    refused_as_poisoned(lambda: eng.import_shot(2, SHOT))
    eng.close()  # forgets, never hangs
    assert eng._busy is False and eng._close_pending is False and eng._handle is None


class Gate:
    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        if name != "mbe_apply_pauli_mask":
            return fn

        def held(*args):
            inside.set()
            release.wait(30)  # the engine is inside a native section across this wait
            return fn(*args)

        return held


def hold_section(eng):
    eng._lib = Gate(eng._lib)
    words = np.zeros(eng.n_words, np.uint64)
    mask = np.ones(eng.batch_size, bool)
    t = threading.Thread(target=lambda: eng.apply_pauli_mask(words, words, 0, mask))
    t.start()
    assert inside.wait(30)
    return t


def hold_lock(lock):
    def holder():
        with lock:
            inside.set()
            release.wait(30)

    t = threading.Thread(target=holder)
    t.start()
    assert inside.wait(30)
    return t


def fork_and_run(work):
    # The parent must get past fork() while the holder is still inside: a parent that waited for
    # it would sit here until the watchdog, because only this thread releases the holder.
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        signal.alarm(30)
        try:
            work()
            msg = "OK"
        except BaseException as exc:
            msg = "FAILED " + repr(exc)
        os.write(w, msg.encode())
        os._exit(0)
    os.close(w)
    print("FORK RETURNED")
    release.set()
    print("CHILD", os.read(r, 4096).decode() or "KILLED")
    os.waitpid(pid, 0)


def parent_check(holder, check):
    holder.join(30)
    check()
    print("PARENT OK")


def idle_engine_with_lock_held():
    eng, other = make_engine(), make_engine()
    t = hold_lock(eng._lock)  # not a native section: only the lock is held at the fork
    fork_and_run(lambda: (works(eng), works(other)))
    parent_check(t, lambda: (works(eng), works(other)))


def engine_inside_a_section():
    eng, idle = make_engine(), make_engine()
    t = hold_section(eng)

    def child():
        poisoned(eng)
        works(idle)
        works(make_engine())  # created after the fork

    fork_and_run(child)
    parent_check(t, lambda: (eng.export_shot(1), works(eng)))


def store_inside_engine_creation():
    real = ls.MagicBatchEngine
    store = ls.LiveExpansions(3, 70, 1e-10)

    class Slow(real):
        def __init__(self, *args):
            inside.set()
            release.wait(30)  # the store lock is held across this wait
            super().__init__(*args)

    ls.MagicBatchEngine = Slow
    t = threading.Thread(target=lambda: store.engine)
    t.start()
    assert inside.wait(30)

    def child():
        ls.MagicBatchEngine = real
        assert store[1].chi == 1
        store[2] = store[1]
        store.close()

    fork_and_run(child)

    def check():
        ls.MagicBatchEngine = real
        assert store[1].chi == 1

    parent_check(t, check)


def store_whose_engine_is_inside_a_section():
    store = ls.LiveExpansions(3, 70, 1e-10)
    t = hold_section(store.engine)

    def child():
        refused_as_poisoned(lambda: store[1])
        store.close()  # drops the poisoned engine
        assert store[1].chi == 1  # a fresh one is created on demand

    fork_and_run(child)
    parent_check(t, lambda: store[1])


def fork_from_inside_a_section_of_the_forking_thread():
    eng = make_engine()
    state = {"forked": False, "pid": None}
    r, w = os.pipe()

    class ForkAt:
        def __init__(self, lib):
            self._lib = lib

        def __getattr__(self, name):
            fn = getattr(self._lib, name)
            if name != "mbe_reset" or state["forked"]:
                return fn

            def call(*args):  # where a handler would run: inside the section, before the call
                state["forked"] = True
                state["pid"] = os.fork()
                return fn(*args)

            return call

    eng._lib = ForkAt(eng._lib)
    eng.reset()  # parent and child both return from here
    if state["pid"] == 0:
        signal.alarm(30)
        try:
            assert eng._busy is False  # the section ended normally in the child
            works(eng)
            msg = "OK"
        except BaseException as exc:
            msg = "FAILED " + repr(exc)
        os.write(w, msg.encode())
        os._exit(0)
    os.close(w)
    print("CHILD", os.read(r, 4096).decode() or "KILLED")
    os.waitpid(state["pid"], 0)
    works(eng)
    print("PARENT OK")


def engines_created_and_used_while_forking():
    made, stop = [], threading.Event()

    def churn():
        while not stop.is_set():
            made.append(make_engine(2))
            if len(made) > 40:
                made.pop(0).close()

    t = threading.Thread(target=churn)
    t.start()
    kids = []
    for _ in range(25):
        pid = os.fork()
        if pid == 0:
            signal.alarm(30)
            bad = 0
            for eng in list(made):
                try:
                    eng.export_shot(0)
                except RuntimeError as exc:
                    bad += not ("forked" in str(exc) or "closed" in str(exc))
            works(make_engine())
            os._exit(bad)
        kids.append(pid)
    codes = [os.waitpid(pid, 0)[1] for pid in kids]
    stop.set()
    t.join()
    print("CHILDREN", "OK" if codes == [0] * 25 else codes)
    print("PARENT OK")


def nested_access_while_another_thread_forks():
    op = sys.argv[3]
    old, new = make_engine(), make_engine()  # a fork locking in creation order takes `old` first

    def handler_like():  # touches the OLDER engine from inside the newer one's section
        with new._section():
            inside.set()
            forked.wait(3)  # a fork that took `old` and waits for `new` can never finish
            try:
                {
                    "close": old.close,
                    "export": lambda: old.export_shot(0),
                    "reset": old.reset,
                    "import": lambda: old.import_shot(0, SHOT),
                }[op]()
            except RuntimeError:
                pass

    def forker():
        inside.wait(30)
        time.sleep(0.2)
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        forked.set()

    ts = [threading.Thread(target=handler_like), threading.Thread(target=forker)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert forked.is_set()
    print("PARENT OK")


def signal_handler_nesting_engines_while_a_thread_forks():
    old, new = make_engine(2), make_engine(2)
    state = {"forks": 0, "signals": 0}

    def on_alarm(signum, frame):
        state["signals"] += 1
        try:
            old.export_shot(0)  # a handler touching the OLDER engine, at any bytecode of main
        except RuntimeError:
            pass

    signal.signal(signal.SIGALRM, on_alarm)
    stop = threading.Event()

    def forker():
        while not stop.is_set():
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            state["forks"] += 1

    t = threading.Thread(target=forker)
    t.start()
    signal.setitimer(signal.ITIMER_REAL, 1e-4, 1e-4)
    end = time.time() + 4
    while time.time() < end:
        new.import_shot(0, SHOT)
        new.export_shot(0)
        new.reset()
    signal.setitimer(signal.ITIMER_REAL, 0)
    stop.set()
    t.join()
    assert state["forks"] > 0 and state["signals"] > 0, state
    print("PARENT OK")


globals()[scenario]()
"""

_HOOK_SCRIPT = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
registered = []
real = os.register_at_fork
os.register_at_fork = lambda **kw: (registered.append(kw), real(**kw))[0]
from stimside._native import magic_engine
mine = [
    sorted(kw) for kw in registered
    if getattr(kw.get("after_in_child"), "__module__", "") == magic_engine.__name__
    or any(getattr(f, "__module__", "") == magic_engine.__name__ for f in kw.values())
]
print("HOOKS", mine)
"""

_AFTER = ["FORK RETURNED", "CHILD OK", "PARENT OK"]
_SCENARIOS = {
    "idle_engine_with_lock_held": _AFTER,
    "engine_inside_a_section": _AFTER,
    "store_inside_engine_creation": _AFTER,
    "store_whose_engine_is_inside_a_section": _AFTER,
    "fork_from_inside_a_section_of_the_forking_thread": ["CHILD OK", "PARENT OK"],
    "engines_created_and_used_while_forking": ["CHILDREN OK", "PARENT OK"],
}


def _run(script: str, tmp_path: Path, *args: str) -> list[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path)}
    run = subprocess.run(
        [sys.executable, "-c", script, str(_SRC), *args],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    lines = run.stdout.split("\n")
    assert "WATCHDOG" not in lines, f"{args} hung: {run.stdout}{run.stderr}"
    assert run.returncode == 0, run.stdout + run.stderr
    return [line for line in lines if line]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_a_forked_child_and_the_parent_both_keep_working(scenario: str, tmp_path: Path) -> None:
    assert _run(_SCRIPT, tmp_path, scenario) == _SCENARIOS[scenario]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("op", ["close", "export", "reset", "import"])
def test_a_fork_cannot_deadlock_with_a_handler_that_nests_engines_in_the_other_order(
    op: str, tmp_path: Path
) -> None:
    lines = _run(_SCRIPT, tmp_path, "nested_access_while_another_thread_forks", op)
    assert lines == ["PARENT OK"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_real_signal_handler_nesting_engines_never_deadlocks_a_forking_thread(
    tmp_path: Path,
) -> None:
    lines = _run(_SCRIPT, tmp_path, "signal_handler_nesting_engines_while_a_thread_forks")
    assert lines == ["PARENT OK"]


@pytest.mark.skipif(not hasattr(os, "register_at_fork"), reason="needs os.register_at_fork")
def test_only_a_child_hook_is_registered_so_the_parent_takes_no_lock_around_a_fork(
    tmp_path: Path,
) -> None:
    assert _run(_HOOK_SCRIPT, tmp_path) == ["HOOKS [['after_in_child']]"]


# ---------------------------------------------------------------------------------------------
# The child-side renewal and the registry, in this process
# ---------------------------------------------------------------------------------------------


def _held_by_another_thread(lock: Any) -> bool:
    got: list[bool] = []

    def probe() -> None:
        got.append(bool(lock.acquire(blocking=False)))
        if got[0]:
            lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join(30)
    return not got[0]


class _HeldLock:
    """A thread that holds `lock` until `release()` (what a fork would inherit)."""

    def __init__(self, lock: Any) -> None:
        self._go, self._held = threading.Event(), threading.Event()

        def run() -> None:
            with lock:
                self._held.set()
                self._go.wait(30)

        self._t = threading.Thread(target=run)
        self._t.start()
        assert self._held.wait(30)

    def release(self) -> None:
        self._go.set()
        self._t.join(30)


def test_the_engine_and_the_store_are_tracked_weakly_for_the_child_to_renew() -> None:
    eng, store = me.MagicBatchEngine(1, 5, 1e-10), LiveExpansions(1, 5, 1e-10)
    assert eng in me._FORK_OWNERS and store in me._FORK_OWNERS
    gc.collect()  # engines of earlier tests may be garbage not yet collected: not ours to count
    tracked = len(me._FORK_OWNERS)
    eng_ref, store_ref = weakref.ref(eng), weakref.ref(store)
    del eng, store
    gc.collect()
    assert eng_ref() is None and store_ref() is None
    assert len(me._FORK_OWNERS) == tracked - 2


def test_the_child_hook_renews_the_lock_of_every_tracked_engine_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(me, "_FORK_OWNERS", weakref.WeakSet())
    eng, store = me.MagicBatchEngine(1, 5, 1e-10), LiveExpansions(1, 5, 1e-10)
    holds = [_HeldLock(eng._lock), _HeldLock(store._lock)]
    assert _held_by_another_thread(eng._lock) and _held_by_another_thread(store._lock)
    me._reinit_owners_in_child()
    try:
        assert not _held_by_another_thread(eng._lock)
        assert not _held_by_another_thread(store._lock)
        assert eng.export_shot(0).chi == 1 and store[0].chi == 1  # both still work
    finally:
        for h in holds:
            h.release()
        eng.close()
        store.close()


def test_an_engine_another_thread_was_inside_is_poisoned_and_never_freed_in_the_child() -> None:
    eng = me.MagicBatchEngine(1, 5, 1e-10)
    handle = eng._handle
    eng._busy, eng._close_pending, eng._owner = True, True, -1  # `-1`: not this thread
    held = _HeldLock(eng._lock)
    eng._reinit_after_fork()
    try:
        assert not _held_by_another_thread(eng._lock)
        assert (eng._busy, eng._close_pending, eng._handle) == (False, False, None)
        with pytest.raises(RuntimeError, match="forked"):
            eng.export_shot(0)
        eng.close()  # no hang, and no `mbe_destroy` of a snapshot taken mid-update
    finally:
        held.release()
        eng._handle, eng._poisoned = handle, False  # this process did not fork: free it for real
        eng.close()


def test_a_forking_threads_own_section_is_not_poisoned_in_the_child() -> None:
    eng = me.MagicBatchEngine(1, 5, 1e-10)
    with eng._section():  # a fork from a handler on this thread: not inside a native call
        old_lock = eng._lock
        eng._reinit_after_fork()
        assert eng._lock is not old_lock and eng._busy and eng._handle is not None
    assert not eng._busy
    assert eng.export_shot(0).chi == 1
    eng.close()


def test_an_idle_engine_and_a_store_only_get_a_new_lock_in_the_child() -> None:
    eng, store = me.MagicBatchEngine(1, 5, 1e-10), LiveExpansions(1, 5, 1e-10)
    old = (eng._lock, store._lock)
    store._creating = True  # a creation another thread was in the middle of
    eng._reinit_after_fork()
    store._reinit_after_fork()
    assert eng._lock is not old[0] and store._lock is not old[1]
    assert (eng._poisoned, eng._handle is None) == (False, False)
    assert store._creating is False and store.engine is not None
    eng.close()
    store.close()
