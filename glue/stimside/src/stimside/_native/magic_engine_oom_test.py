"""Out-of-memory behaviour of EVERY allocating `mbe_*` entry point, by fault injection.

An `LD_PRELOAD`ed `operator new` fails the N-th C++ allocation made inside ONE native call (it is
armed right before that call, on the calling thread). For each entry point, for every N the call
makes, the process must survive and the wrapper must raise `MemoryError`: no `std::terminate`,
no other exception type, no hang. An entry point that makes no allocation (the numpy-recipe
helpers) is pinned at zero allocations. The probes run in a child process because a regression
is a dead process (rc 134), not a failed assertion.

Skipped under ASan/TSan (their allocator aborts on an impossible `new` / owns `operator new`)
and without a C++ compiler.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from stimside._native import magic_engine as me

_SRC = Path(me.__file__).resolve().parents[2]  # the directory that contains the `stimside` package

_FI_CPP = r"""
#include <cstdlib>
#include <new>
static thread_local long g_countdown = -1;
static thread_local long g_count = 0;
extern "C" void fi_arm(long n) { g_countdown = n; g_count = 0; }
extern "C" long fi_count() { return g_count; }
extern "C" void fi_disarm() { g_countdown = -1; }
static void* alloc(size_t sz) {
    g_count++;
    if (g_countdown == 0) { g_countdown = -1; throw std::bad_alloc(); }
    if (g_countdown > 0) --g_countdown;
    void* p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new(size_t sz) { return alloc(sz); }
void* operator new[](size_t sz) { return alloc(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
"""

# Child: argv = [src, fi_so, case]. Prints one JSON line.
_CHILD = r"""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
fi = ctypes.CDLL(sys.argv[2])
fi.fi_arm.argtypes = [ctypes.c_long]
fi.fi_count.restype = ctypes.c_long
case = sys.argv[3]
import stimside._magic.phase_qfe as pq
from stimside._native import magic_engine as me
from stimside._native.magic_engine import MagicBatchEngine, ShotArrays
from stimside._magic.live_store import expansion_to_arrays
lib = me.get_magic_engine_lib()
RNG = np.random.default_rng(5)


def words(n, density=0.4):
    while True:
        a = pq.pack_bit_vector((RNG.random(n) < density).astype(np.uint8), n)
        b = pq.pack_bit_vector((RNG.random(n) < density).astype(np.uint8), n)
        if a.any() or b.any():
            kap = (int(pq.word_dot_parity(a, b)) + 2 * int(RNG.integers(0, 2))) & 3
            return a, b, kap


def t_state(n, n_t, d_turns=0):
    e = pq.ShotExpansion.zero(n)
    for _ in range(d_turns):
        e = pq.apply_quarter_turn(e, *words(n))
    for _ in range(n_t):
        e = pq.apply_pauli_t(e, *words(n))
    return e


def hook(name):
    state = {"k": None, "n": None}

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, attr):
            fn = getattr(self._inner, attr)
            if attr != name or state["k"] is None:
                return fn

            def call(*args):
                fi.fi_arm(state["k"])  # armed on this thread, immediately before the call
                try:
                    return fn(*args)
                finally:
                    state["n"] = fi.fi_count()
                    fi.fi_disarm()

            return call

    return Spy, state


def block(n, sequential):
    from stimside._magic.channels import MagicPhaseTerm, MagicTermRecord
    from stimside._magic.gates import CompiledPauliAxis
    from stimside._magic.live_store import block_table
    from stimside._magic.plan import compile_open_block

    ax = CompiledPauliAxis(*words(n))
    recs = [MagicTermRecord(MagicPhaseTerm((ax,), "T", 1))]
    blk = compile_open_block(recs, n_qubits=n, budget=1 if sequential else 64)[0].block
    assert (blk.mode == "sequential") == sequential
    return block_table(blk, n)


def make(case):
    # returns (engine, native function name, thunk calling the wrapper method once)
    if case == "reset_after_chi0":
        e = MagicBatchEngine(2, 5, 1e-10)
        z = lambda *s, dt=np.uint64: np.zeros(s, dtype=dt)
        nw = e.n_words
        e.import_shot(0, ShotArrays(0, 0, True, 0.0, 1e-10, z(0, dt=np.complex128), z(0, nw),
                                    z(0, 0, dt=np.uint8), z(0, nw), z(nw), z(0, 0, dt=np.uint8),
                                    z(0, dt=np.int32), z(0, dt=np.uint8)))
        return e, "mbe_reset", lambda: e.reset()
    if case == "pauli_mask_d300":
        nn, d = 300, 270
        e = MagicBatchEngine(1, nn, 1e-10)
        nw = e.n_words
        A = np.zeros((d, nw), np.uint64)
        for r in range(d):
            A[r, r >> 6] = np.uint64(1) << np.uint64(r & 63)
        e.import_shot(0, ShotArrays(
            1, d, True, 0.0, 1e-10, np.ones(1, np.complex128), np.zeros((1, nw), np.uint64),
            np.zeros((1, d), np.uint8), A, np.bitwise_or.reduce(A, axis=0),
            np.zeros((d, d), np.uint8), np.arange(d, dtype=np.int32), np.zeros(d, np.uint8)))
        a = np.zeros(nw, np.uint64)
        b = np.zeros(nw, np.uint64)
        b[0] = 1
        return e, "mbe_apply_pauli_mask", lambda: e.apply_pauli_mask(a, b, 0, np.ones(1, bool))
    if case == "import_shot":
        e = MagicBatchEngine(2, 70, 1e-10)
        arrs = expansion_to_arrays(t_state(70, 3, 2))
        return e, "mbe_import_shot", lambda: e.import_shot(1, arrs)
    if case in ("block_pauli_sum", "block_sequential"):
        e = MagicBatchEngine(4, 70, 1e-10)
        for i in range(4):
            e.import_shot(i, expansion_to_arrays(t_state(70, 3, 2)))
        tab = block(70, case == "block_sequential")
        return e, "mbe_apply_block", lambda: e.apply_block(tab, True, None)
    if case == "partition":
        e = MagicBatchEngine(6, 70, 1e-10)
        for i in range(6):
            e.import_shot(i, expansion_to_arrays(t_state(70, 3, 1)))
        return e, "mbe_partition", lambda: e.partition(True)
    if case == "inject_cliffords":
        from stimside._magic.channels import CompiledCliffordFactorList as CL
        from stimside._magic.gates import CompiledPauliAxis as AX
        from stimside._magic.live_store import factor_table
        e = MagicBatchEngine(3, 70, 1e-10)
        for i in range(3):
            e.import_shot(i, expansion_to_arrays(t_state(70, 3, 1)))
        fac = CL(3, AX(*words(70)), (AX(*words(70)), AX(*words(70))))
        tab = factor_table([fac], e.n_words)
        which = np.zeros(3, np.int32)
        return e, "mbe_inject_cliffords", lambda: e.inject_cliffords(tab, which, True)
    raise SystemExit("unknown case " + case)


def trip(case, k):
    e, name, thunk = make(case)
    Spy, state = hook(name)
    state["k"] = k
    e._lib = Spy(e._lib)
    try:
        thunk()
    except MemoryError:
        e._lib = e._lib._inner  # drop the spy
        return "MemoryError" + ("" if consistent(case, e) else " but the state is torn")
    except Exception as exc:  # any other type is a failure of the contract
        return type(exc).__name__ + ": " + str(exc)
    return "ok"


def consistent(case, e):
    # After a failed reset every shot is exactly its old state or exactly ShotExpansion.zero.
    if case != "reset_after_chi0":
        return True
    for b in range(e.batch_size):
        x = e.export_shot(b)
        zero = x.chi == 1 and x.d == 0 and x.canonical and x.log_mass == 0.0
        zero = zero and x.weights.tolist() == [1 + 0j] and not x.h_words.any()
        old = x.chi == 0 and x.d == 0 and x.weights.shape == (0,)
        if not (zero or old):
            return False
    return True


if case == "norm_sq_allocates_nothing":
    w = (RNG.standard_normal(300) + 1j * RNG.standard_normal(300)).astype(np.complex128)
    fi.fi_arm(10**9)
    lib.mbe_norm_sq(300, me.w_addr(w))
    lib.mbe_cmul_scalar(300, me.w_addr(w), 0.5, 0.5)
    lib.mbe_cmul_form(300, me.w_addr(w), 0.5, 0.25, 1)
    print(json.dumps({"allocs": fi.fi_count()}))
elif case == "create":
    fi.fi_arm(0)
    h = lib.mbe_create(4, 70, 1e-10)
    fi.fi_disarm()
    print(json.dumps({"handle_is_null": not h}))
elif case == "import_huge":
    e = MagicBatchEngine(1, 70, 1e-10)
    buf = np.zeros(64, np.uint64)
    p = me.addr(buf)
    rcs = []
    for chi in (2**61, 2**62, 2**63 - 1, 2**60 + 1, 2**59):
        rcs.append(lib.mbe_import_shot(e._h(), 0, chi, 0, 1, 0.0, 1e-10, *[p] * 8))
    print(json.dumps({"rcs": rcs}))
else:
    e, name, thunk = make(case)
    Spy, state = hook(name)
    state["k"] = 10**9  # counts, never trips
    e._lib = Spy(e._lib)
    thunk()
    n_allocs = state["n"]
    results = [trip(case, k) for k in range(n_allocs)]
    print(json.dumps({"allocs": n_allocs, "results": results}))
"""


def _under_sanitizer() -> bool:
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__asan_init") or hasattr(libc, "__tsan_init")
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    _under_sanitizer() or shutil.which("c++") is None,
    reason="needs a C++ compiler and no sanitizer allocator (they own operator new)",
)


@pytest.fixture(scope="module")
def fi_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("fi")
    cpp, so = d / "fi_new.cpp", d / "libfi_new.so"
    cpp.write_text(_FI_CPP)
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)
    return so


def _child(fi_so: Path, case: str, timeout: int = 900) -> dict[str, Any]:
    env = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LD_PRELOAD": str(fi_so),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_SRC), str(fi_so), case],
        capture_output=True, text=True, env=env, timeout=timeout, check=False,
    )
    msg = f"{case}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2500:]}"
    assert proc.returncode == 0, msg
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


_ALLOCATING = [
    "reset_after_chi0", "pauli_mask_d300", "import_shot", "inject_cliffords", "block_pauli_sum",
    "block_sequential", "partition",
]


@pytest.mark.parametrize("case", _ALLOCATING)
def test_every_allocation_failure_inside_one_native_call_becomes_memory_error(
    case: str, fi_so: Path
) -> None:
    out = _child(fi_so, case)
    assert out["allocs"] >= 1, out  # the call really allocates (else the probe proves nothing)
    assert out["results"] == ["MemoryError"] * out["allocs"], out


def test_the_numpy_recipe_helpers_allocate_nothing(fi_so: Path) -> None:
    """`mbe_norm_sq` used to build a `std::vector` and abort the process on a failed allocation."""
    assert _child(fi_so, "norm_sq_allocates_nothing")["allocs"] == 0


def test_a_failed_create_returns_null(fi_so: Path) -> None:
    assert _child(fi_so, "create")["handle_is_null"] is True


def test_an_import_of_an_impossible_size_is_refused_instead_of_terminating(fi_so: Path) -> None:
    """`std::length_error` (and a wrapped size product) must never cross `extern "C"`."""
    rcs = _child(fi_so, "import_huge")["rcs"]
    assert all(rc in (1, 2) for rc in rcs), rcs
