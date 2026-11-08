"""Every raw size argument of every `mbe_*` entry point, with VALID pointers and huge values.

The raw ABI trusts its caller about pointers, but a size is just a number: `n_steps = 2**62`
made `p + n_steps` wrap (`build_block_job` in S4b-2: a SEGV / ASan heap-buffer-overflow) while
the older `n_axes` check was already there. So the rule tested here is general: for each size
argument (`n_axes`, `n_lists`, `n_steps`, `chi`, `d`, shot indices, `batch_size`, `n_qubits`,
`n` of the complex-multiply helpers) a value above what any array can have is refused BEFORE
it is used (error code -2, 1 or a NULL handle; NaN for `mbe_norm_sq`), the engine state is
untouched, and nothing is allocated, so these cases run under ASan/TSan too (no skip).

Every case runs in a child process: the failure mode is a dead process, and a dead child is one
failed test rather than a dead pytest. The `fuzz` seeds draw random values from [2**60, 2**64).
"""

from __future__ import annotations

import pytest

from stimside._native import magic_engine as me
from stimside._native.magic_engine_hostile_test import _NO_ASAN, _run_child

_CHILD = r"""
import random, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr, w_addr

which, seed, rounds = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
FIXED = [2**60, 2**62, 2**62 + 1, 2**63, 2**63 + 1, 2**63 + 2**62, 2**64 - 1]
rng = random.Random(seed)


def values():
    if seed < 0:
        return FIXED
    out = []
    for _ in range(rounds):
        bits = rng.randrange(61, 65)
        out.append(rng.getrandbits(bits - 1) | (1 << (bits - 1)))  # in [2**(bits-1), 2**bits)
    return out


N, NW, B = 70, 2, 3
lib = me.get_magic_engine_lib()
eng = me.MagicBatchEngine(B, N, 1e-10)
h = eng._h()
live0 = lib.mbe_live_engines()


def shot(chi, d):
    r = np.random.default_rng(chi * 7 + d)
    return me.ShotArrays(
        chi, d, True, 0.25, 1e-10, r.standard_normal(chi) + 1j * r.standard_normal(chi),
        r.integers(0, 2**63, (chi, NW), dtype=np.uint64),
        r.integers(0, 4, (chi, d), dtype=np.uint8),
        r.integers(0, 2**63, (d, NW), dtype=np.uint64), r.integers(0, 2**63, NW, dtype=np.uint64),
        r.integers(0, 2, (d, d), dtype=np.uint8), r.integers(0, 70, d, dtype=np.int32),
        r.integers(0, 2, d, dtype=np.uint8),
    )


for b in range(B):
    eng.import_shot(b, shot(3, 2))


def state():
    out = []
    for b in range(B):
        s = eng.export_shot(b)
        out.append(tuple(np.asarray(x).tobytes() for x in s[5:]) + tuple(s[:5]))
    return out


before = state()
z32, kap = np.zeros(4, np.int32), np.zeros(2, np.uint8)
a2, b2, w2 = np.zeros((2, NW), np.uint64), np.zeros((2, NW), np.uint64), np.zeros((2, 2))
b2[0, 0] = 1
i32, u8 = np.zeros(2, np.int32), np.zeros(2, np.uint8)
err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
shots = np.full(B, -1, np.int32)
checks = 0


def expect(label, got, want):
    global checks
    checks += 1
    assert got == want, (label, got, want)


def apply_block(mode, n_steps, n_axes):
    return lib.mbe_apply_block(
        h, mode, n_steps, n_axes, 1.0, 0.0, addr(z32), addr(z32), addr(z32), addr(a2), addr(b2),
        addr(kap), addr(w2), 1, 2**63 - 1, addr(err), addr(errd), addr(info),
    )


def inject(n_lists, n_axes):
    return lib.mbe_inject_cliffords(
        h, n_lists, n_axes, addr(u8), addr(i32), addr(i32), addr(i32), addr(i32), addr(a2),
        addr(b2), addr(kap), addr(shots), 0, addr(err), addr(errd),
    )


def import_args(b, chi, d):
    s = shot(3, 2)
    ptrs = [addr(x) for x in (s.h_words, s.ell, s.A_words, s.a_union, s.Gamma, s.pivots,
                              s.ell_odd)]
    return (h, b, chi, d, 1, 0.0, 1e-10, w_addr(s.weights), *ptrs), s


def import_shot(b, chi, d):
    args, keep = import_args(b, chi, d)
    rc = lib.mbe_import_shot(*args)
    del keep
    return rc


def run(v):
    if which in ("create", "all"):
        for args in ((v, N), (B, v), (v, v)):
            expect(("create", args), lib.mbe_create(args[0], args[1], 1e-10), None)
        expect("live engines", lib.mbe_live_engines(), live0)
    if which in ("import_shot", "all"):
        expect(("import chi", v), import_shot(0, v, 2), 1)
        expect(("import d", v), import_shot(0, 3, v), 1)
        expect(("import chi,d", v), import_shot(0, v, v), 1)
    if which in ("shot_index", "all"):
        expect(("import b", v), import_shot(v, 3, 2), 1)
        cd = np.zeros(3, np.uint64)
        expect(("header b", v), lib.mbe_shot_header(h, v, addr(cd), addr(np.zeros(2))), 1)
        s = shot(3, 2)
        ptrs = [addr(x) for x in (s.h_words, s.ell, s.A_words, s.a_union, s.Gamma, s.pivots,
                                  s.ell_odd)]
        expect(("export b", v), lib.mbe_export_shot(h, v, w_addr(s.weights), *ptrs), 1)
    if which in ("inject", "all"):
        for args in ((v, 2), (1, v), (v, v)):
            expect(("inject", args), inject(*args), -2)
    if which in ("apply_block", "all"):
        for args in ((1, v, 2), (1, 0, v), (1, v, v), (1, 0, 2**63 + 2**61)):
            expect(("apply_block", args), apply_block(*args), -2)
    if which in ("block_mode0", "all"):
        expect(("apply_block mode 0 with steps", v), apply_block(0, v, 2), -2)
    if which in ("cmul_norm", "all"):
        w = np.array([1.5 + 2.5j, -3.0 + 0.5j] * 4)
        keep = w.copy()
        lib.mbe_cmul_scalar(v, w_addr(w), 0.5, 0.25)
        expect(("cmul_scalar", v), bool(np.array_equal(w, keep)), True)
        for form in (0, 1, 2):
            lib.mbe_cmul_form(v, w_addr(w), 0.5, 0.25, form)
            expect(("cmul_form", v, form), bool(np.array_equal(w, keep)), True)
        expect(("norm_sq", v), bool(np.isnan(lib.mbe_norm_sq(v, w_addr(w)))), True)


for v in values():
    run(v)
assert state() == before, "engine state changed"
assert lib.mbe_live_engines() == live0
eng.close()
print("SIZES OK", checks)
"""


def _run(which: str, seed: int = -1, rounds: int = 0) -> None:
    run = _run_child(_CHILD, which, str(seed), str(rounds))
    assert run.returncode == 0 and "SIZES OK" in run.stdout, (
        which, run.returncode, run.stdout[-300:], run.stderr[-1200:],
    )


@pytest.mark.parametrize(
    "which", ["create", "import_shot", "shot_index", "inject", "apply_block", "cmul_norm"]
)
def test_an_impossible_size_argument_is_refused_before_use_with_valid_pointers(which: str) -> None:
    _run(which)


def test_a_pauli_sum_block_with_steps_and_a_huge_n_steps_is_refused() -> None:
    _run("block_mode0")


@pytest.mark.parametrize("seed", range(3))
def test_random_huge_sizes_are_refused_on_every_entry_point_without_touching_the_state(
    seed: int,
) -> None:
    _run("all", seed, 150)


@pytest.mark.parametrize("big", [2**60, 2**62, 2**63 - 1])
def test_an_engine_with_an_impossible_n_qubits_or_batch_is_a_memory_error(big: int) -> None:
    with pytest.raises(MemoryError):
        me.MagicBatchEngine(2, big, 1e-10)
    with pytest.raises(MemoryError):
        me.MagicBatchEngine(big, 70, 1e-10)
    assert me.get_magic_engine_lib().mbe_live_engines() == 0


@_NO_ASAN
@pytest.mark.parametrize("n_steps", [2**59, 2**60 - 1])
def test_the_largest_n_steps_that_passes_the_bound_is_out_of_memory_not_a_crash(
    n_steps: int,
) -> None:
    """Just below the bound the arrays are addressable but cannot be allocated: -3 (the ASan
    allocator aborts on such a request instead of throwing, so only the plain build runs it)."""
    script = (
        "import sys, numpy as np\nsys.path.insert(0, sys.argv[1])\n"
        "from stimside._native import magic_engine as me\n"
        "from stimside._native.magic_engine import addr\n"
        "e = me.MagicBatchEngine(2, 70, 1e-10)\nz = np.zeros(4, np.int32)\n"
        "a = np.zeros((1, 2), np.uint64); k = np.zeros(1, np.uint8); w = np.zeros((1, 2))\n"
        "o = [np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)]\n"
        f"rc = e._lib.mbe_apply_block(e._h(), 1, {n_steps}, 1, 1.0, 0.0, addr(z), addr(z),"
        " addr(z), addr(a), addr(a), addr(k), addr(w), 1, 2**63 - 1, *(addr(x) for x in o))\n"
        "print('RC', rc)\n"
    )
    run = _run_child(script)
    assert run.returncode == 0 and run.stdout.split()[-1] == "-3", (run.stdout, run.stderr[-800:])
