"""Hostile-caller, race, out-of-memory and fork tests of `mbe_peek`, `mbe_postselect` and
`mbe_unmatched_apply` (WP-25 stage S5b).

* The C++ runs with the GIL released: it reads the caller's observable, kick, shot list,
  desired / drawn bits and flags ONCE (private copies), whatever another thread does to those
  buffers meanwhile. Tested deterministically (the arrays are scribbled at the very moment the
  native call starts) and by a race against the raw ABI.
* The raw ABI trusts pointers but not numbers: a null handle / pointer, a shot index outside the
  batch, a negative one and a list length no array can have are refused with a code and change
  nothing.
* Every allocation failure inside one call (`LD_PRELOAD`ed `operator new`) becomes `MemoryError`
  with no torn state: `postselect` and `unmatched_apply` leave exactly the entries before the one
  they were at committed.
* A forked child works with an engine that was idle at the fork.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_measure_edges_test import (
    _before,
    _engine_with,
    _t_state,
    _x,
    _z,
)
from stimside._native import magic_engine as me
from stimside._native.magic_engine_hostile_test import _run_child
from stimside._native.magic_engine_oom_test import _FI_CPP, _under_sanitizer
from stimside._native.magic_engine_test import _SpyLib

_N = 5
_SRC = Path(me.__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Simulators and their spies form reference cycles that hold a native engine until the
    cyclic gc runs; free them here so the live-engine counts other tests assert on stay exact."""
    yield
    gc.collect()



def _states() -> list[pq.ShotExpansion]:
    return [_t_state(), pq.ShotExpansion.zero(_N), _t_state(_N, 3), _t_state(_N, 1)]


def _t(ax: CompiledPauliAxis, kappa: int | None = None) -> tuple[Any, Any, int]:
    return ax.a_words, ax.b_words, ax.kappa if kappa is None else kappa


def _hooked(store: LiveExpansions, hook: Any, name: str) -> None:
    eng = store.engine
    spy = _SpyLib(eng._lib, lambda n, _a: hook() if n == name else None)
    eng._lib = spy  # type: ignore[assignment]


ALL = np.arange(4, dtype=np.int32)


# --------------------------------------------------------------------------------------------
# One read of every caller buffer (deterministic: scribbled when the native call starts)
# --------------------------------------------------------------------------------------------


def test_peek_reads_private_copies_of_the_observable_and_the_shot_list() -> None:
    obs = _x(0)
    ref = _engine_with(_states())
    want, _ = ref.peek(_t(obs), ALL)
    store = _engine_with(_states())
    a, b, lst = obs.a_words.copy(), obs.b_words.copy(), ALL.copy()

    def scribble() -> None:
        a[:] = 0
        b[:] = 0
        lst[:] = 0

    _hooked(store, scribble, "mbe_peek")
    got, fail = store.peek((a, b, 0), lst)
    assert not a.any() and not lst.any()  # the scribble really ran
    assert fail is None and got.tobytes() == want.tobytes()


def test_postselect_reads_private_copies_of_everything_it_is_given() -> None:
    obs = _x(1)
    des = np.array([0, 1, 0, 1], np.uint8)
    ref = _engine_with(_states())
    rres = ref.postselect(_t(obs), ALL, des)
    want = _before(ref)
    store = _engine_with(_states())
    a, b, lst, d = obs.a_words.copy(), obs.b_words.copy(), ALL.copy(), des.copy()

    def scribble() -> None:
        a[:] = 0
        lst[:] = 3
        d[:] = 1 - d

    _hooked(store, scribble, "mbe_postselect")
    res = store.postselect((a, b, 0), lst, d)
    assert lst.tolist() == [3, 3, 3, 3]
    assert (res.failure, res.oom_at) == (rres.failure, rres.oom_at)
    assert res.done.tobytes() == rres.done.tobytes() and res.z01.tobytes() == rres.z01.tobytes()
    assert _before(store) == want


def test_unmatched_apply_reads_private_copies_of_everything_it_is_given() -> None:
    obs, kick = _z(0), _x(2)
    args = (ALL, np.array([1, 0, 1, 1], np.uint8), np.array([1, 1, 0, 1], np.uint8),
            np.array([1, 0, 1, 1], np.uint8))
    ref = _engine_with(_states())
    rres = ref.unmatched_apply(_t(obs), _t(kick), *args)
    want = _before(ref)
    store = _engine_with(_states())
    a, b, ka, kb = (x.copy() for x in (obs.a_words, obs.b_words, kick.a_words, kick.b_words))
    lst, bit, meas, kck = (x.copy() for x in args)

    def scribble() -> None:
        for arr in (a, b, ka, kb):
            arr[:] = 0
        lst[:] = 3
        bit[:] = 1 - bit
        meas[:] = 1 - meas
        kck[:] = 1 - kck

    _hooked(store, scribble, "mbe_unmatched_apply")
    res = store.unmatched_apply((a, b, 0), (ka, kb, 0), lst, bit, meas, kck)
    assert lst.tolist() == [3, 3, 3, 3] and not ka.any()
    assert (res.failure, res.oom_at, res.phase) == (rres.failure, rres.oom_at, rres.phase)
    assert _before(store) == want


# --------------------------------------------------------------------------------------------
# The raw ABI: numbers, indices and pointers
# --------------------------------------------------------------------------------------------

_RAW = r"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr
from stimside._magic.live_store_measure_edges_test import _engine_with, _t_state, _x, _z, _before

N, B = 5, 4
lib = me.get_magic_engine_lib()
store = _engine_with([_t_state(), pq.ShotExpansion.zero(N), _t_state(N, 3), _t_state(N, 1)])
h = store.engine._h()
ax, kick = _x(0), _z(2)
lst = np.arange(B, dtype=np.int32)
p1, done, z01 = np.zeros(B), np.zeros(B, np.uint8), np.zeros((B, 2))
u8 = [np.zeros(B, np.uint8) for _ in range(3)]
err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(2, np.int64)


def peek(handle=h, **o):
    a = dict(h=handle, a=addr(ax.a_words), b=addr(ax.b_words), k=0, s=addr(lst), n=B,
             p=addr(p1), e=addr(err), d=addr(errd), i=addr(info))
    a.update(o)
    return lib.mbe_peek(*a.values())


def post(handle=h, **o):
    a = dict(h=handle, a=addr(ax.a_words), b=addr(ax.b_words), k=0, s=addr(lst), n=B,
             w=addr(u8[0]), dn=addr(done), z=addr(z01), e=addr(err), d=addr(errd), i=addr(info))
    a.update(o)
    return lib.mbe_postselect(*a.values())


def um(handle=h, **o):
    a = dict(h=handle, a=addr(ax.a_words), b=addr(ax.b_words), k=0, ka=addr(kick.a_words),
             kb=addr(kick.b_words), kk=0, s=addr(lst), n=B, bit=addr(u8[0]), me=addr(u8[1]),
             ki=addr(u8[2]), e=addr(err), d=addr(errd), i=addr(info))
    a.update(o)
    return lib.mbe_unmatched_apply(*a.values())


before = _before(store)
bad_lists = {
    "negative": np.array([0, -1, 2, 3], np.int32), "too_big": np.array([0, 1, 4, 3], np.int32),
    "huge": np.array([0, 2**31 - 1, 2, 3], np.int32), "min": np.array([-2**31, 1, 2, 3], np.int32),
}
checks = []
for fn, name in ((peek, "peek"), (post, "post"), (um, "um")):
    checks.append((f"{name} null handle", fn(handle=0), -2))
    for key, val in bad_lists.items():
        checks.append((f"{name} shot list {key}", fn(s=addr(val)), -2))
    checks.append((f"{name} null shots with n > 0", fn(s=0), -2))
    checks.append((f"{name} n 2**63", fn(n=2**63), -2))
    checks.append((f"{name} n 2**64-1", fn(n=2**64 - 1), -2))
    checks.append((f"{name} null a", fn(a=0), -2))
    checks.append((f"{name} null b", fn(b=0), -2))
    checks.append((f"{name} null err", fn(e=0), -2))
    checks.append((f"{name} null errd", fn(d=0), -2))
    checks.append((f"{name} null info", fn(i=0), -2))
checks.append(("peek null out_p1", peek(p=0), -2))
checks.append(("post null desired", post(w=0), -2))
checks.append(("post null done", post(dn=0), -2))
checks.append(("post null z01", post(z=0), -2))
checks.append(("um null kick a", um(ka=0), -2))
checks.append(("um null kick b", um(kb=0), -2))
for nm in ("bit", "me", "ki"):
    checks.append((f"um null {nm}", um(**{nm: 0}), -2))
assert _before(store) == before, "a refused call changed a shot"
# n == 0 never reads the (null) arrays and does nothing
checks.append(("peek n=0", peek(n=0, p=0, s=0), 0))
checks.append(("post n=0", post(n=0, w=0, dn=0, z=0, s=0), 0))
checks.append(("um n=0", um(n=0, bit=0, me=0, ki=0, s=0), 0))
assert _before(store) == before
# a refused list leaves a pending split alone? (peek) / drops it (post, um): refused calls come
# before the pending split is touched only for peek; postselect and unmatched drop it first.
bad = [c for c in checks if c[1] != c[2]]
print("BAD" if bad else "OK", bad)
"""


def test_the_raw_born_entry_points_refuse_bad_numbers_and_pointers_without_changes() -> None:
    run = _run_child(_RAW, timeout=300)
    assert run.returncode == 0, (run.stdout[-500:], run.stderr[-1500:])
    assert run.stdout.strip().splitlines()[-1].startswith("OK"), run.stdout[-800:]


def test_wrapper_refuses_a_shot_outside_the_batch_before_any_native_call() -> None:
    store = _engine_with(_states())
    before = _before(store)
    for bad in ([0, 4], [-1], [0, 1, 2, 3, 99]):
        arr = np.array(bad, np.int32)
        with pytest.raises(RuntimeError, match="failed before running"):
            store.peek(_t(_x(0)), arr)
        with pytest.raises(RuntimeError, match="failed before running"):
            store.postselect(_t(_x(0)), arr, np.zeros(len(bad), np.uint8))
        with pytest.raises(RuntimeError, match="failed before running"):
            store.unmatched_apply(
                _t(_z(0)), _t(_x(1)), arr, *(np.zeros(len(bad), np.uint8),) * 3
            )
    assert _before(store) == before


def test_wrapper_refuses_a_shot_list_or_flag_array_of_the_wrong_kind() -> None:
    store = _engine_with(_states())
    with pytest.raises(TypeError, match="integer"):
        store.peek(_t(_x(0)), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        store.postselect(_t(_x(0)), ALL, np.zeros(3, np.uint8))
    with pytest.raises(ValueError, match="shape"):
        store.unmatched_apply(_t(_z(0)), _t(_x(1)), ALL, [0] * 4, [0] * 3, [0] * 4)
    bad = _x(0)
    bad.a_words[:] = np.uint64(1) << np.uint64(_N)  # a bit above qubit n - 1
    _, fail = store.peek(_t(bad), ALL)  # a failure of the first shot (the caller raises)
    assert fail is not None and fail.kind == me.FAIL_PADDING and fail.shot == 0
    # a Z-type observable (no X part: the unrotated branch of the peek) with a padding bit in b
    bad_z = _z(0)
    bad_z.b_words[:] = np.uint64(1) << np.uint64(_N)
    # list entry 0 is |0..0>, trivial: the shot never looks at the observable words
    _, fail = store.peek(_t(bad_z), ALL[1:])
    assert fail is not None and fail.kind == me.FAIL_PADDING and fail.shot == 1


def test_a_peek_keeps_a_pending_split_and_the_other_two_drop_it() -> None:
    obs = _x(0)
    for name, call in (
        ("peek", lambda s: s.peek(_t(obs), ALL)),
        ("postselect", lambda s: s.postselect(_t(_z(0)), ALL[:1], np.zeros(1, np.uint8))),
        ("unmatched", lambda s: s.unmatched_apply(
            _t(_z(0)), _t(_x(1)), ALL[:1], [0], [0], [0])),
    ):
        store = _engine_with(_states())
        res = store.measure_split(_t(obs), False, None)
        call(store)
        out = np.arange(res.n_active) % 2 == 1
        z01 = res.z01[res.cls == me.CLS_ACTIVE]
        chosen = np.where(out, z01[:, 1], z01[:, 0])
        if name == "peek":
            assert store.measure_finish(out, chosen, res).failure is None
        else:
            # unguarded (no `split`): the NATIVE call decides, not the wrapper's token check
            with pytest.raises(RuntimeError, match="no pending measure_split for this engine"):
                store.measure_finish(out, chosen)


# --------------------------------------------------------------------------------------------
# A racing thread rewriting the caller's raw buffers
# --------------------------------------------------------------------------------------------

_RACE = r"""
import sys, threading
import numpy as np
sys.path.insert(0, sys.argv[1])
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_test import random_state

N, B = 5, 8
mode = sys.argv[2]
lib = me.get_magic_engine_lib()
rs = np.random.default_rng(1)
states = [random_state(rs, N, 4, "plain") for _ in range(B)]


def fresh():
    s = LiveExpansions(B, N, 1e-10)
    for b, st in enumerate(states):
        s[b] = st
    return s


def words(bits):
    w = np.zeros(1, np.uint64)
    w[0] = bits
    return w


x0, x1 = words(14), words(4)  # two valid X-type observables the racing thread alternates between
a, zero = x0.copy(), np.zeros(1, np.uint64)
good, bad_ = int(x0[0]), int(x1[0])
lst = np.arange(B, dtype=np.int32)
want = np.ones(B, np.uint8)
stop = [False]


def flip():
    while not stop[0]:
        a[0] = bad_
        a[0] = good


def run(obs):
    s = fresh()
    err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(2, np.int64)
    if mode == "peek":
        out = np.zeros(B)
        rc = lib.mbe_peek(s.engine._h(), addr(obs), addr(zero), 0, addr(lst), B, addr(out),
                          addr(err), addr(errd), addr(info))
        return rc, out.tobytes()
    done, z01 = np.zeros(B, np.uint8), np.zeros((B, 2))
    rc = lib.mbe_postselect(s.engine._h(), addr(obs), addr(zero), 0, addr(lst), B, addr(want),
                            addr(done), addr(z01), addr(err), addr(errd), addr(info))
    return rc, z01.tobytes() + done.tobytes() + repr(info.tolist()).encode()


ref0, ref1 = run(x0), run(x1)
assert ref0 != ref1
th = threading.Thread(target=flip)
th.start()
seen = set()
for _ in range(300):
    got = run(a)
    seen.add(got)
    assert got in (ref0, ref1), "a mixed (torn) observable"
stop[0] = True
th.join()
print("OK", len(seen))
"""


@pytest.mark.skipif(
    hasattr(ctypes.CDLL(None), "__tsan_init"),
    reason="a data race by design (a thread rewrites the caller's buffer while the C++ makes its "
    "single read of it); TSan reports exactly that, so it runs plain and under ASan",
)
@pytest.mark.parametrize("mode", ["peek", "postselect"])
def test_a_racing_thread_cannot_tear_the_observable_of_one_command(mode: str) -> None:
    run = _run_child(_RACE, mode, timeout=600)
    assert run.returncode == 0, (run.stdout[-500:], run.stderr[-1500:])
    assert run.stdout.strip().splitlines()[-1].startswith("OK"), run.stdout[-500:]


# --------------------------------------------------------------------------------------------
# Out of memory at every allocation of one call
# --------------------------------------------------------------------------------------------

_OOM = r"""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
fi = ctypes.CDLL(sys.argv[2])
fi.fi_arm.argtypes = [ctypes.c_long]
fi.fi_count.restype = ctypes.c_long
case = sys.argv[3]
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._magic.live_store_measure_edges_test import _before, _t_state, _x, _z, _engine_with

N = 70 if case.endswith("70") else 5
ALL = np.arange(5, dtype=np.int32)
ones = np.ones(5, np.uint8)


def fresh():
    store = _engine_with([_t_state(N, 2), pq.ShotExpansion.zero(N), _t_state(N, 3),
                          _t_state(N, 1), pq.apply_pauli(pq.ShotExpansion.zero(N), _x(0, N))])
    return store


def words(ax):
    return ax.a_words, ax.b_words, ax.kappa


def command(store):
    if case.startswith("peek"):
        return lambda: store.peek(words(_x(1, N)), ALL)
    if case.startswith("postselect"):
        return lambda: store.postselect(words(_x(1, N)), ALL, np.zeros(5, np.uint8))
    return lambda: store.unmatched_apply(words(_x(1, N)), words(_z(2, N)), ALL, ones, ones, ones)


def hook(name, state):
    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, attr):
            fn = getattr(self._inner, attr)
            if attr != name or state["k"] is None:
                return fn

            def call(*args):
                fi.fi_arm(state["k"])
                try:
                    return fn(*args)
                finally:
                    state["n"] = fi.fi_count()
                    fi.fi_disarm()
            return call
    return Spy


NAME = {"peek": "mbe_peek", "postselect": "mbe_postselect", "unmatched": "mbe_unmatched_apply"}
name = NAME[case.rstrip("70")]


def attempt(k):
    store = fresh()
    eng = store.engine
    ref = fresh()
    state = {"k": None, "n": None}
    eng._lib = hook(name, state)(eng._lib)
    state["k"] = k
    try:
        res = command(store)()
        # `postselect` / `unmatched_apply` report an out-of-memory in the result (`oom_at`), the
        # simulator raises it; `peek` raises MemoryError itself.
        if getattr(res, "oom_at", None) is None:
            return "ok", state["n"], True
        raise MemoryError
    except MemoryError:
        n_alloc = state["n"]
        eng._lib = eng._lib._inner
        # whatever was committed is a prefix of the entries: every shot is either its old state
        # or the state a complete (OOM-free) run gives it, and the engine still works.
        done = command(ref)()
        full, mine, old = _before(ref), _before(store), _before(fresh())
        torn = [b for b in range(5) if mine[b] not in (full[b], old[b])]
        usable = True
        try:
            store.peek(words(_x(1, N)), ALL)
        except Exception:
            usable = False
        return "MemoryError", n_alloc, not torn and usable


first = attempt(10**9)
assert first[0] == "ok", first
n_allocs = first[1]
results = [attempt(k) for k in range(n_allocs)]
bad = [(k, r) for k, r in enumerate(results) if not r[2]]
print(json.dumps({"allocs": n_allocs, "kinds": sorted({r[0] for r in results}), "bad": bad}))
"""


def _oom_child(fi_so: Path, case: str) -> dict[str, Any]:
    env = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LD_PRELOAD": str(fi_so),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _OOM, str(_SRC), str(fi_so), case],
        capture_output=True, text=True, env=env, timeout=900, check=False,
    )
    msg = f"{case}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2500:]}"
    assert proc.returncode == 0, msg
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def fi_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("fi_born")
    cpp, so = d / "fi_new.cpp", d / "libfi_new.so"
    cpp.write_text(_FI_CPP)
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)
    return so


@pytest.mark.skipif(_under_sanitizer(), reason="the sanitizer allocator owns operator new")
@pytest.mark.parametrize(
    "case", ["peek", "peek70", "postselect", "postselect70", "unmatched", "unmatched70"]
)
def test_every_allocation_failure_inside_one_born_command_is_a_memory_error_without_torn_state(
    case: str, fi_so: Path
) -> None:
    out = _oom_child(fi_so, case)
    assert out["allocs"] >= 3, out  # the call really allocates (else this proves nothing)
    assert out["bad"] == [], out
    assert set(out["kinds"]) <= {"MemoryError", "ok"} and "MemoryError" in out["kinds"], out


# --------------------------------------------------------------------------------------------
# fork
# --------------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_runs_the_new_commands_on_an_engine_that_was_idle_at_the_fork() -> None:
    store = _engine_with(_states())
    store.peek(_t(_x(0)), ALL)  # the engine is warm, its lock released
    want, _ = _engine_with(_states()).peek(_t(_x(1)), ALL)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: only the forking thread exists
        code = 1
        try:
            got, fail = store.peek(_t(_x(1)), ALL)
            res = store.postselect(_t(_z(0)), ALL[:1], np.zeros(1, np.uint8))
            ok = fail is None and got.tobytes() == want.tobytes() and res.failure is None
            ok = ok and store.unmatched_apply(
                _t(_z(1)), _t(_x(2)), ALL[:2], [1, 0], [0, 0], [1, 1]).failure is None
            code = 0 if ok else 2
        finally:
            os.write(w, b"x")
            os._exit(code)
    os.close(w)
    _, status = os.waitpid(pid, 0)
    os.close(r)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
    got, _ = store.peek(_t(_x(1)), ALL)  # the parent is unaffected
    assert got.tobytes() == want.tobytes()
