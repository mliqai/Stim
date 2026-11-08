"""A *closed* engine must stay tracked for `fork()` (S4a closing item L1).

`MagicBatchEngine.close()` frees the native state but the Python object, and its lock, live on:
any later `close()`, `reset()` or `export_shot()` still takes the lock. If `close()` dropped the
object from the fork registry, a child forked while another thread held that lock would inherit
it held and hang on the first call, so the registry membership of a closed engine is behaviour,
not bookkeeping. Both views are tested: the registry directly, and a real forked child with a
watchdog (a regression is a failed assertion, never a hung run).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

from stimside._magic.live_store import LiveExpansions
from stimside._native import magic_engine as me

_SRC = Path(me.__file__).resolve().parents[2]  # the directory that contains the `stimside` package

_SCRIPT = r"""
import os, signal, sys, threading, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
from stimside._native import magic_engine as me
import stimside._magic.live_store as ls
kind = sys.argv[2]
held, go = threading.Event(), threading.Event()


def watchdog():
    time.sleep(40)
    print("WATCHDOG", flush=True)
    os._exit(7)


threading.Thread(target=watchdog, daemon=True).start()
if kind == "engine":
    owner = me.MagicBatchEngine(2, 70, 1e-10)
else:
    owner = ls.LiveExpansions(2, 70, 1e-10)
    owner.engine  # create the engine, then close the store
owner.close()
lock = owner._lock


def holder():
    with lock:
        held.set()
        go.wait(30)


t = threading.Thread(target=holder)
t.start()
assert held.wait(30)
pid = os.fork()
if pid == 0:
    signal.alarm(20)
    try:
        owner.close()  # idempotent: takes the lock a thread of the parent holds
        if kind == "engine":
            try:
                owner.reset()
            except RuntimeError as exc:
                assert "closed" in str(exc), exc
            else:
                raise AssertionError("a closed engine was used")
        msg = "OK"
    except BaseException as exc:
        msg = "FAILED " + repr(exc)
    os.write(1, ("CHILD " + msg + "\n").encode())
    os._exit(0)
go.set()
t.join(30)
_, status = os.waitpid(pid, 0)
print("PARENT OK" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f"PARENT {status}")
"""


def test_a_closed_engine_stays_tracked_for_fork() -> None:
    eng = me.MagicBatchEngine(2, 70, 1e-10)
    assert eng in me._FORK_OWNERS
    eng.close()
    assert eng._handle is None
    assert eng in me._FORK_OWNERS


def test_closing_twice_and_resetting_a_closed_engine_keep_it_tracked() -> None:
    eng = me.MagicBatchEngine(2, 70, 1e-10)
    eng.close()
    eng.close()
    with pytest.raises(RuntimeError, match="closed"):
        eng.reset()
    assert eng in me._FORK_OWNERS


def test_a_closed_store_and_its_closed_engine_stay_tracked_for_fork() -> None:
    store = LiveExpansions(2, 70, 1e-10)
    engine = store.engine
    store.close()
    assert engine._handle is None
    assert store in me._FORK_OWNERS and engine in me._FORK_OWNERS


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("kind", ["engine", "store"])
def test_a_child_forked_while_a_thread_holds_a_closed_objects_lock_does_not_hang(
    kind: str, tmp_path: Path
) -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path)}
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(_SRC), kind],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    lines = [line for line in run.stdout.split("\n") if line]
    assert "WATCHDOG" not in lines, f"the child hung on an inherited lock: {run.stdout}{run.stderr}"
    assert run.returncode == 0, run.stdout + run.stderr
    assert sorted(lines) == ["CHILD OK", "PARENT OK"], run.stdout


def test_the_registry_does_not_keep_a_closed_engine_alive() -> None:
    eng = me.MagicBatchEngine(2, 70, 1e-10)
    eng.close()
    ref = weakref.ref(eng)
    del eng
    gc.collect()
    assert ref() is None  # the registry is weak: tracking a closed engine does not leak it
