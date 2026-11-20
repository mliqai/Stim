"""Hostile-caller, race and out-of-memory tests of `mbe_measure_split` / `mbe_measure_finish`
(WP-25 stage S5a).

* The C++ runs with the GIL released: it must read the caller's observable, kick, outcomes and
  masses ONCE (private copies), whatever another thread does to those buffers meanwhile. Tested
  deterministically (the arrays are scribbled at the very moment the native call starts) and by a
  race against the raw ABI, where every shot must still see ONE consistent observable.
* The raw ABI trusts pointers but not numbers or call order: a finish without a split, a second
  finish, a count that is not the number of active shots, null pointers and a bad handle are
  refused with a code, and refuse to change anything.
* Every allocation failure inside one call (`LD_PRELOAD`ed `operator new`) becomes `MemoryError`
  with no torn state: nothing changed in a split, and in a finish nothing (phase 1) or only the
  shots before the one whose kick could not run (phase 2). The pending split is used up.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
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
from stimside._native.magic_engine_oom_test import _FI_CPP
from stimside._native.magic_engine_test import _SpyLib

_N, _ATOL = 5, 1e-10


def _under_tsan() -> bool:
    try:
        return hasattr(ctypes.CDLL(None), "__tsan_init")
    except OSError:
        return False

_SRC = Path(me.__file__).resolve().parents[2]


def _hooked(store: LiveExpansions, hook: Any, name: str) -> None:
    eng = store.engine
    spy = _SpyLib(eng._lib, lambda n, _a: hook() if n == name else None)
    eng._lib = spy  # type: ignore[assignment]


def _states() -> list[pq.ShotExpansion]:
    return [_t_state(), pq.ShotExpansion.zero(_N), _t_state(_N, 3), _t_state(_N, 1)]


def _outcomes(n: int) -> Any:
    return (np.arange(n) % 2) == 1


def _split_args(obs: CompiledPauliAxis) -> tuple[Any, bool, Any]:
    kick = _z(2)
    return (obs.a_words, obs.b_words, 0), False, (kick.a_words, kick.b_words, 0, False)


def test_split_reads_a_private_copy_of_the_observable_and_the_kick() -> None:
    """The caller's observable and kick words are scribbled when the native call starts: the
    result (masses, classes, and the state a finish then commits, kicks included) is the one of
    the untouched words."""
    obs, kick = _x(0), _z(2)
    states = _states() + [pq.apply_pauli(pq.ShotExpansion.zero(_N), _x(1))]
    ref = _engine_with(states)
    obs_t, keep, reset = _split_args(obs)
    ref_res = ref.engine.measure_split(obs_t, keep, reset)
    out = _outcomes(ref_res.n_active)
    z01 = ref_res.z01[ref_res.cls == me.CLS_ACTIVE]
    ref.engine.measure_finish(out, np.where(out, z01[:, 1], z01[:, 0]))
    want = _before(ref)
    store = _engine_with(states)
    a, b = obs.a_words.copy(), obs.b_words.copy()
    ka, kb = kick.a_words.copy(), kick.b_words.copy()

    def scribble() -> None:
        for arr in (a, b, ka, kb):
            arr[:] = 0

    _hooked(store, scribble, "mbe_measure_split")
    res = store.engine.measure_split((a, b, 0), False, (ka, kb, 0, False))
    assert not a.any() and not ka.any()  # the scribble really ran
    assert res.z01.tobytes() == ref_res.z01.tobytes()
    assert res.cls.tobytes() == ref_res.cls.tobytes()
    store.engine.measure_finish(out, np.where(out, z01[:, 1], z01[:, 0]))
    assert _before(store) == want
    store.close()
    ref.close()


def test_finish_reads_private_copies_of_the_outcomes_and_the_masses() -> None:
    obs = _x(0)
    ref = _engine_with(_states())
    ref_res = ref.engine.measure_split((obs.a_words, obs.b_words, 0), False, None)
    out = _outcomes(ref_res.n_active)
    z01 = ref_res.z01[ref_res.cls == me.CLS_ACTIVE]
    ref.engine.measure_finish(out, np.where(out, z01[:, 1], z01[:, 0]))
    want = _before(ref)
    store = _engine_with(_states())
    store.engine.measure_split((obs.a_words, obs.b_words, 0), False, None)
    out2 = out.copy()
    chosen = np.where(out2, z01[:, 1], z01[:, 0])

    def scribble() -> None:
        out2[:] = ~out2
        chosen[:] = 0.0

    _hooked(store, scribble, "mbe_measure_finish")
    fin = store.engine.measure_finish(out2, chosen)
    assert fin.failure is None and not chosen.any()  # the scribble really ran
    assert _before(store) == want
    store.close()
    ref.close()


# --------------------------------------------------------------------------------------------
# The raw ABI: numbers and call order
# --------------------------------------------------------------------------------------------

_RAW = r"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr
from stimside._magic.live_store_measure_edges_test import _engine_with, _t_state, _x, _z, _before

N = 5
lib = me.get_magic_engine_lib()
store = _engine_with([_t_state(), pq.ShotExpansion.zero(N), _t_state(N, 3), _t_state(N, 1)])
eng = store.engine
h = eng._h()
ax, kick = _x(0), _z(2)
B = 4
cls, bit = np.full(B, 3, np.uint8), np.zeros(B, np.uint8)
z, z01, chi2 = np.zeros(B), np.zeros((B, 2)), np.zeros((B, 2), np.int64)
err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
outc, mass = np.zeros(B, np.uint8), np.ones(B)


def split(handle=h, **over):
    args = dict(handle=handle, a=addr(ax.a_words), b=addr(ax.b_words), kappa=0, keep=0, reset=0,
                flip=0, ka=addr(kick.a_words), kb=addr(kick.b_words), kk=0, cls=addr(cls),
                bit=addr(bit),
                z=addr(z), z01=addr(z01), chi=addr(chi2), err=addr(err), errd=addr(errd),
                info=addr(info))
    args.update(over)
    return lib.mbe_measure_split(*args.values())


def finish(n=B, handle=h, **over):
    args = dict(handle=handle, n=n, o=addr(outc), m=addr(mass), err=addr(err), errd=addr(errd),
                info=addr(info))
    args.update(over)
    return lib.mbe_measure_finish(*args.values())


before = _before(store)
checks = []
checks.append(("finish without a split", finish(), -2))
checks.append(("null handle split", split(handle=0), -2))
checks.append(("null handle finish", finish(handle=0), -2))
for name in ("a", "b", "cls", "bit", "z", "z01", "chi", "err", "errd", "info"):
    checks.append((f"null {name}", split(**{name: 0}), -2))
checks.append(("null kick with reset", split(reset=1, ka=0), -2))
assert _before(store) == before
# a good split, then a finish with a wrong count: nothing read, nothing changed, split used up
rc = split()
assert rc == 0 and info[0] == 3, (rc, info)  # three active shots
checks.append(("wrong n", finish(n=2), -4))
checks.append(("finish after a refused finish", finish(n=3), -2))
assert _before(store) == before
rc = split()
checks.append(("finish null outcomes", finish(n=3, o=0), -2))
rc = split()
checks.append(("finish null chosen", finish(n=3, m=0), -2))
rc = split()
checks.append(("finish n huge", finish(n=2**63), -4))
rc = split()
checks.append(("finish n 2**64-1", finish(n=2**64 - 1), -4))
checks.append(("finish after refused", finish(n=3), -2))
assert _before(store) == before
# a split on a destroyed-then-recreated engine does not leak a pending split to the new one
rc = split()
store2 = _engine_with([_t_state()] * 4)
checks.append(("pending split is per engine", lib.mbe_measure_finish(
    store2.engine._h(), 3, addr(outc), addr(mass), addr(err), addr(errd), addr(info)), -2))
rc = finish(n=3)
assert rc in (0, 1), rc  # the first engine's own split is still there
bad = [c for c in checks if c[1] != c[2]]
print("BAD" if bad else "OK", bad)
"""


def test_the_raw_measure_entry_points_refuse_bad_numbers_and_call_order_without_changes() -> None:
    run = _run_child(_RAW, timeout=300)
    assert run.returncode == 0, (run.stdout[-500:], run.stderr[-1500:])
    assert run.stdout.strip().splitlines()[-1].startswith("OK"), run.stdout[-800:]


_RACE = r"""
import sys, threading
import numpy as np
sys.path.insert(0, sys.argv[1])
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr
from stimside._magic.live_store_measure_edges_test import _engine_with, _t_state, _x, _z, _before

N, B = 5, 8
mode = sys.argv[2]
lib = me.get_magic_engine_lib()
from stimside._magic.live_store_test import random_state
_rs = np.random.default_rng(1)
states = [random_state(_rs, N, 4, "plain") for i in range(B)]
def obs(bits):
    w = np.zeros(1, np.uint64)
    w[0] = bits
    return w


x0, x1 = obs(14), obs(4)  # two valid X-type observables (b = 0) the thread alternates between
a = x0.copy()
b = np.zeros_like(a)
good, bad_ = int(x0[0]), int(x1[0])
stop = [False]


def flip():
    while not stop[0]:
        a[0] = bad_
        a[0] = good


def run_split(obs_a, obs_b):
    s = _engine_with(states)
    cls, bit = np.full(B, 3, np.uint8), np.zeros(B, np.uint8)
    z, z01, chi2 = np.zeros(B), np.zeros((B, 2)), np.zeros((B, 2), np.int64)
    err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
    kick = _z(2)
    rc = lib.mbe_measure_split(
        s.engine._h(), addr(obs_a), addr(obs_b), 0, 0, 0, 0, addr(kick.a_words),
        addr(kick.b_words), 0, addr(cls), addr(bit), addr(z), addr(z01), addr(chi2), addr(err),
        addr(errd), addr(info))
    return s, rc, z01.copy(), cls.copy()


ref0 = run_split(x0, b)[2].tobytes()
ref1 = run_split(x1, b)[2].tobytes()
assert ref0 != ref1
th = threading.Thread(target=flip)
th.start()
seen = set()
for _ in range(300):
    s, rc, z01, cls = run_split(a, b)
    assert rc == 0, rc
    seen.add(z01.tobytes())
    assert z01.tobytes() in (ref0, ref1), "a mixed (torn) observable"
stop[0] = True
th.join()
print("OK", len(seen))
"""


def test_a_racing_thread_cannot_tear_the_observable_of_one_split() -> None:
    """Raw ABI (no wrapper copies): the observable is copied once, so every shot of one split
    is measured against the SAME observable, never half of one and half of the other."""
    run = _run_child(_RACE, "obs", timeout=600)
    assert run.returncode == 0, (run.stdout[-500:], run.stderr[-1500:])
    assert run.stdout.strip().splitlines()[-1].startswith("OK"), run.stdout[-500:]


_RACE_FINISH = r"""
import sys, threading
import numpy as np
sys.path.insert(0, sys.argv[1])
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_test import random_state, _nonzero_axis

N, B, CHI = 70, 48, 32
lib = me.get_magic_engine_lib()
rng = np.random.default_rng(3)
states = [random_state(rng, N, CHI, "d") for _ in range(B)]
ax = _nonzero_axis(np.random.default_rng(5), N, density=0.3)
assert ax[0].any()


def fresh():
    s = LiveExpansions(B, N, 1e-10)
    for b, st in enumerate(states):
        s[b] = st
    return s


def split(s):
    return s.engine.measure_split(ax, True, None)  # keep the rotated frame: finish = renormalise


def shot_bytes(s, b):
    x = s.engine.export_shot(b)
    return b"".join(np.asarray(v).tobytes() for v in x[5:]) + repr(x[:5]).encode()


def committed(outcome_all):
    s = fresh()
    res = split(s)
    act = res.cls == me.CLS_ACTIVE
    z01 = res.z01[act]
    out = np.full(res.n_active, outcome_all, bool)
    s.engine.measure_finish(out, np.where(out, z01[:, 1], z01[:, 0]))
    return [shot_bytes(s, b) for b in range(B)], res.n_active


want0, n_active = committed(False)
want1, _ = committed(True)
assert n_active >= 20 and want0 != want1
outc = np.zeros(n_active, np.uint8)
stop = [False]


def flip():
    while not stop[0]:
        outc[:] = 1
        outc[:] = 0


th = threading.Thread(target=flip)
th.start()
torn = 0
mixed = set()
for _ in range(150):
    s = fresh()
    res = split(s)
    act = res.cls == me.CLS_ACTIVE
    z01 = res.z01[act]
    chosen = np.ascontiguousarray(np.minimum(z01[:, 0], z01[:, 1]))  # same mass either way:
    # the masses differ per outcome, so a torn read of the outcomes also tears the masses
    chosen = np.where(z01[:, 0] > 0, z01[:, 0], z01[:, 1])
    err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
    rc = lib.mbe_measure_finish(s.engine._h(), n_active, addr(outc), addr(chosen), addr(err),
                                addr(errd), addr(info))
    assert rc == 0, rc
    got = [shot_bytes(s, b) for b in range(B)]
    for b in range(B):
        mixed.add(got[b] == want0[b])
        if got[b] not in (want0[b], want1[b]) and b in set(np.flatnonzero(act).tolist()):
            torn += 1
stop[0] = True
th.join()
print("OK" if torn == 0 else "TORN", torn, sorted(mixed))
"""


@pytest.mark.skipif(
    _under_tsan(),
    reason="the test IS a data race by design (one thread writes the caller's buffer while the "
    "C++ makes its single read of it); TSan reports exactly that, so it runs plain and under ASan",
)
def test_a_racing_thread_cannot_tear_the_outcomes_of_one_finish() -> None:
    """Raw ABI: the outcomes flip between all-0 and all-1 while the C++ runs. It reads them once,
    so a committed shot is always one COMPLETE branch (un-rotated / renormalised and committed
    from the same branch), never the renormalised one of branch 0 swapped for the raw one of 1."""
    run = _run_child(_RACE_FINISH, timeout=900)
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
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_measure_edges_test import _before, _t_state, _x, _z, _engine_with

N = 70 if case.endswith("70") else 5
RNG = np.random.default_rng(2)


def states():
    out = [_t_state(N, 2), pq.ShotExpansion.zero(N), _t_state(N, 3), _t_state(N, 1)]
    if case.startswith("split_reset"):
        out.append(pq.apply_pauli(pq.ShotExpansion.zero(N), _x(0, N)))
    return out


def fresh():
    store = _engine_with(states())
    ax = _x(0, N)
    kick = _z(2, N)
    reset = (kick.a_words, kick.b_words, 0, False) if case.startswith("split_reset") else None
    return store, ax, reset


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


def attempt(k):
    store, ax, reset = fresh()
    eng = store.engine
    before = _before(store)
    state = {"k": None, "n": None}
    if case.startswith("split"):
        name = "mbe_measure_split"
        eng._lib = hook(name, state)(eng._lib)
        state["k"] = k
        try:
            res = eng.measure_split((ax.a_words, ax.b_words, 0), False, reset)
        except MemoryError:
            return "MemoryError", state["n"], True
        n_alloc = state["n"]
        ok = res.oom and _before(store) == before
        if res.oom:
            # no pending split is left behind
            try:
                eng.measure_finish(np.zeros(0, bool), np.zeros(0))
                return "pending split left behind", n_alloc, False
            except RuntimeError:
                pass
            return "oom", n_alloc, ok or case.startswith("split_reset")
        return "ok", n_alloc, True
    name = "mbe_measure_finish"
    res = eng.measure_split((ax.a_words, ax.b_words, 0), False, None)
    act = res.cls == me.CLS_ACTIVE
    z01 = res.z01[act]
    out = np.arange(res.n_active) % 2 == 1
    chosen = np.where(out, z01[:, 1], z01[:, 0])
    eng._lib = hook(name, state)(eng._lib)
    state["k"] = k
    try:
        eng.measure_finish(out, chosen)
    except MemoryError:
        return "MemoryError", state["n"], _before(store) == before
    return "ok", state["n"], True


first = attempt(10**9)
assert first[0] == "ok", first
n_allocs = first[1]
results = [attempt(k) for k in range(n_allocs)]
bad = [(k, r) for k, r in enumerate(results) if not r[2]]
kinds = sorted({r[0] for r in results})
print(json.dumps({"allocs": n_allocs, "kinds": kinds, "bad": bad}))
"""


def _oom_child(fi_so: Path, case: str) -> dict[str, Any]:
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    ld_preload = f"{fi_so} {existing_preload}" if existing_preload else str(fi_so)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_PRELOAD": ld_preload,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if existing_preload:
        asan_opts = env.get("ASAN_OPTIONS", "")
        if "verify_asan_link_order=0" not in asan_opts:
            env["ASAN_OPTIONS"] = (
                f"{asan_opts}:verify_asan_link_order=0"
                if asan_opts
                else "verify_asan_link_order=0"
            )
    proc = subprocess.run(
        [sys.executable, "-c", _OOM, str(_SRC), str(fi_so), case],
        capture_output=True, text=True, env=env, timeout=900, check=False,
    )
    msg = f"{case}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2500:]}"
    assert proc.returncode == 0, msg
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def fi_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fault-injecting `operator new` of the out-of-memory tests (same source)."""
    d = tmp_path_factory.mktemp("fi_measure")
    cpp, so = d / "fi_new.cpp", d / "libfi_new.so"
    cpp.write_text(_FI_CPP)
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)
    return so


_skip_sanitized = pytest.mark.skipif(
    _under_tsan(),
    reason="TSan owns operator new",
)


@_skip_sanitized
@pytest.mark.parametrize(
    "case", ["split_plain", "split_plain70", "split_reset", "finish", "finish70"]
)
def test_every_allocation_failure_inside_split_or_finish_is_a_memory_error_without_torn_state(
    case: str, fi_so: Path
) -> None:
    out = _oom_child(fi_so, case)
    assert out["allocs"] >= 3, out  # the call really allocates (else this proves nothing)
    assert out["bad"] == [], out
    assert set(out["kinds"]) <= {"MemoryError", "oom", "ok"}, out
    assert "oom" in out["kinds"] or "MemoryError" in out["kinds"], out
