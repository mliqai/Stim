"""Carry-forward tests from the S4b-2 gate (WP-25 stage S5a, part 1).

Each test pins something the engine already does right (or the one wrapper gap, int32 table
fields), found missing by the S4b-2 reviewers / auditor / challenger; every one was checked
against a mutant of the engine (see the S5a handoff):

* A10: two shots that differ ONLY in the last byte of one array (the sign of the last weight's
  imaginary part, above all) must stay separate groups: `same_vec` compares its whole length;
* A7 / A8: the raw `mbe_import_shot` refuses `b == B` and `d > n_qubits` (the wrapper hides both);
* Z04-Z06, Z15: each size product of `mbe_import_shot` is bounded on its own (a value that passes
  the other products and fails only this one is refused with rc 1 before anything is allocated;
  without the bound it is an out-of-memory rc 2, or a read past the buffers);
* LOW: the wrapper used to wrap int32 table fields silently (`BlockTable.mode = 2**32 + 1` ran as
  mode 1): such a table is now refused with a clear error and the engine state is untouched.

The raw-ABI cases run in a child process: their failure mode is a dead process.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
import stimside.simulator_magic as sm
from stimside._magic.channels import CompiledCliffordFactorList
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, block_table, factor_table
from stimside._magic.live_store_block_diff_test import commuting_axes, make_shots
from stimside._magic.live_store_test import _nonzero_axis, random_state
from stimside._native.magic_engine_hostile_test import _NO_ASAN, _run_child

_ATOL = 1e-10

# --------------------------------------------------------------------------------------------
# A10: the last byte of every array takes part in the partition key
# --------------------------------------------------------------------------------------------


def _two_shot_partition(a: Any, b: Any) -> list[int]:
    store = LiveExpansions(2, a.n_qubits, _ATOL)
    store[0], store[1] = a, b
    rep = store.engine.partition(True).tolist()
    store.close()
    return rep


def _with_last_weight(exp: Any, value: complex) -> Any:
    out = exp.copy()
    w = out.weights.copy()
    w[-1] = value
    out.weights = w
    return out


@pytest.mark.parametrize("chi", [1, 2, 4, 9])
def test_shots_differing_only_in_the_sign_of_the_last_weights_imaginary_zero_stay_apart(
    chi: int,
) -> None:
    """The A10 case: +0.0 vs -0.0 in the imaginary part of the LAST weight, nothing else."""
    base = random_state(np.random.default_rng(chi), 70, chi, "plain")
    assert base.chi >= 1
    first = _with_last_weight(base, complex(0.5, 0.0))
    second = _with_last_weight(base, complex(0.5, -0.0))
    assert first.weights[-1].imag.tobytes() != second.weights[-1].imag.tobytes()
    assert _two_shot_partition(first, second) == [0, 1]
    assert _two_shot_partition(first, first.copy()) == [0, 0]  # the control: equal shots merge
    assert sorted(sm._group_shots_by_state([first, second])) == [[0], [1]]  # as the B3 grouping


@pytest.mark.parametrize(
    "field", ["weights", "h_words", "ell", "A_words", "a_union", "Gamma", "pivots", "ell_odd"]
)
def test_shots_differing_only_in_the_last_byte_of_one_array_stay_apart(field: str) -> None:
    base = random_state(np.random.default_rng(11), 70, 4, "d")
    assert base.d > 0 and base.chi > 1  # every array is non-empty
    other = base.copy()
    arr = getattr(other, field)
    assert arr.size > 0 and arr.flags.c_contiguous
    arr.view(np.uint8).reshape(-1)[-1] ^= np.uint8(0x80)  # the very last byte of the array
    assert arr.tobytes() != getattr(base, field).tobytes()
    assert _two_shot_partition(base, other) == [0, 1], field


# --------------------------------------------------------------------------------------------
# Raw `mbe_import_shot` (A7, A8, Z04-Z06, Z15), each case in a child process
# --------------------------------------------------------------------------------------------

_CHILD = r"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr, w_addr

case = sys.argv[2]
lib = me.get_magic_engine_lib()


def honest(nw, chi, d):
    r = np.random.default_rng(5)
    return me.ShotArrays(
        chi, d, True, 0.25, 1e-10, r.standard_normal(chi) + 1j * r.standard_normal(chi),
        r.integers(0, 2**60, (chi, nw), dtype=np.uint64),
        r.integers(0, 4, (chi, d), dtype=np.uint8),
        r.integers(0, 2**60, (d, nw), dtype=np.uint64), r.integers(0, 2**60, nw, dtype=np.uint64),
        r.integers(0, 2, (d, d), dtype=np.uint8), r.integers(0, 9, d, dtype=np.int32),
        r.integers(0, 2, d, dtype=np.uint8),
    )


def raw_import(eng, b, chi, d, buf):
    ptrs = [addr(x) for x in (buf.h_words, buf.ell, buf.A_words, buf.a_union, buf.Gamma,
                              buf.pivots, buf.ell_odd)]
    return lib.mbe_import_shot(eng._h(), b, chi, d, 1, 0.0, 1e-10, w_addr(buf.weights), *ptrs)


def snapshot(eng):
    out = []
    for b in range(eng.batch_size):
        s = eng.export_shot(b)
        out.append(tuple(np.asarray(x).tobytes() for x in s[5:]) + tuple(s[:5]))
    return out


def check(label, got, want, eng=None, before=None):
    assert got == want, (label, got, want)
    if before is not None:
        assert snapshot(eng) == before, (label, "state changed")


if case == "index":
    B, N = 3, 70
    eng = me.MagicBatchEngine(B, N, 1e-10)
    buf = honest(2, 3, 2)
    before = snapshot(eng)
    for b in (B, B + 1, 2**32, 2**63, 2**64 - 1):  # b == B is the untested boundary (A7)
        check(("b", b), raw_import(eng, b, 3, 2, buf), 1, eng, before)
    check(("b", B - 1), raw_import(eng, B - 1, 3, 2, buf), 0)  # the last valid shot works
    assert snapshot(eng) != before
elif case == "depth":
    N = 5
    eng = me.MagicBatchEngine(2, N, 1e-10)
    before = snapshot(eng)
    for d in (N + 1, N + 2, 3 * N, 70):  # d > n_qubits at the C++ level (A8 / Z16)
        check(("d", d), raw_import(eng, 0, 1, d, honest(1, 1, d)), 1, eng, before)
    check(("d == n", N), raw_import(eng, 0, 1, N, honest(1, 1, N)), 0)
elif case.startswith("prod_"):
    _, nq, chi, d, want = case.split("_")
    nq, chi, d, want = int(nq), int(chi), int(d), int(want)
    try:
        eng = me.MagicBatchEngine(1, nq, 1e-10)
    except MemoryError:
        print("SKIP create")
        sys.exit(0)
    before = snapshot(eng) if nq < 1000 else None
    buf = honest(1, 1, 0)  # tiny buffers: a bound that holds never reads them
    check(case, raw_import(eng, 0, chi, d, buf), want, eng, before)
print("OK")
"""


def _child(case: str) -> str:
    run = _run_child(_CHILD, case, timeout=600)
    assert run.returncode == 0, (case, run.returncode, run.stdout[-300:], run.stderr[-800:])
    return run.stdout.strip().splitlines()[-1]


def test_raw_import_refuses_a_shot_index_of_b_or_more_and_accepts_b_minus_1() -> None:
    assert _child("index") == "OK"


def test_raw_import_refuses_d_above_n_qubits_and_accepts_d_equal_n_qubits() -> None:
    assert _child("depth") == "OK"


# (n_qubits, chi, d): each passes every bound but the one it names (kMaxElems = 2**60 - 1).
_PRODUCTS = {
    "chi_times_2": (64, 2**59, 0),  # nw = 1: chi * nw = 2**59 is fine, chi * 2 = 2**60 is not
    "chi_times_nw": (130, 2**59 - 1, 0),  # nw = 3: chi * 2 is fine, chi * 3 is not
    "chi_times_d": (64, 2**58, 64),  # chi * 2 and chi * nw are fine; chi * d = 2**64 wraps
    "d_times_d": (2**30, 1, 2**30),  # chi * d and d * nw (2**54) are fine; d * d = 2**60 is not
}


@pytest.mark.parametrize("name", sorted(_PRODUCTS))
def test_each_raw_import_size_product_is_bounded_on_its_own(name: str) -> None:
    """rc 1 (arguments refused) before anything is allocated or read. Without its bound the
    call goes on to an impossible allocation (rc 2) or reads past the buffers."""
    nq, chi, d = _PRODUCTS[name]
    out = _child(f"prod_{nq}_{chi}_{d}_1")
    if out == "SKIP create":
        pytest.skip("this host cannot create an engine with that many qubits")
    assert out == "OK"


@_NO_ASAN
@pytest.mark.parametrize(("nq", "chi"), [(64, 2**59 - 1), (130, 2**58 - 1)])
def test_the_largest_import_that_passes_the_bounds_is_out_of_memory_not_a_crash(
    nq: int, chi: int
) -> None:
    """Just inside the bounds the arrays are addressable but cannot be allocated: rc 2. This
    puts the cases above exactly on the boundary (a `>=` for `>` would refuse these)."""
    assert _child(f"prod_{nq}_{chi}_0_2") == "OK"


# --------------------------------------------------------------------------------------------
# LOW: int32 table fields are never wrapped silently
# --------------------------------------------------------------------------------------------


def _cap_batch(seed: int, n: int = 70) -> tuple[list[Any], Any]:
    """Three shots and a SEQUENTIAL block (N1, P2, P1 steps: three int32 columns to widen)."""
    rng = np.random.default_rng(seed)
    axes = commuting_axes(rng, n, 3)
    steps = tuple(
        pq.CompiledPauliBlockStep(kind=k, paulis=(a.as_tuple(),), sign=s)  # type: ignore[arg-type]
        for k, a, s in (("N1", axes[0], 1), ("P2", axes[1], -1), ("P1", axes[2], 1))
    )
    return make_shots(rng, n, 4, 3), pq.CompiledPauliBlock(
        n_qubits=n, mode="sequential", steps=steps
    )


def _loaded_store(shots: list[Any]) -> LiveExpansions:
    store = LiveExpansions(len(shots), shots[0].n_qubits, _ATOL)
    for i, s in enumerate(shots):
        store[i] = s
    return store


def _state(store: LiveExpansions) -> list[tuple[Any, ...]]:
    out = []
    for b in range(store.batch_size):
        s = store.engine.export_shot(b)
        out.append(tuple(np.asarray(x).tobytes() for x in s[5:]) + tuple(s[:5]))
    return out


_TOO_WIDE = (2**32, 2**32 + 1, 2**33 + 1, 2**31, -(2**31) - 1, 2**40)


def _widen(col: Any, pos: int, big: int) -> Any:
    out = np.asarray(col).astype(np.int64)
    out[pos] = out[pos] + big if big > 0 else big
    return out


def test_a_block_table_with_an_int32_field_out_of_range_is_refused_and_state_untouched() -> None:
    shots, blk = _cap_batch(3)
    table = block_table(blk, 70)
    assert table is not None and table.mode == 1 and table.step_kind.shape[0] > 1
    last = table.step_kind.shape[0] - 1
    cases: list[tuple[str, Any]] = []
    for big in _TOO_WIDE:
        cases.append((f"mode={big}", table._replace(mode=big)))
        for name in ("step_kind", "step_sign", "step_start"):
            for pos in (0, last):
                col = _widen(getattr(table, name), pos, big)
                cases.append((f"{name}[{pos}]+{big}", table._replace(**{name: col})))
    for label, bad in cases:
        store = _loaded_store(shots)
        before = _state(store)
        with pytest.raises((ValueError, OverflowError)):
            store.apply_block(bad, None)
        assert _state(store) == before, label
        store.close()


def test_the_same_block_with_in_range_fields_still_runs() -> None:
    """The control: the guard refuses wide values only (else the tests around it are vacuous)."""
    shots, blk = _cap_batch(3)
    table = block_table(blk, 70)
    assert table is not None
    store = _loaded_store(shots)
    widened = table._replace(  # int64 dtype, in-range values: accepted and identical
        step_kind=table.step_kind.astype(np.int64), step_sign=table.step_sign.astype(np.int64),
        step_start=table.step_start.astype(np.int64),
    )
    store.apply_block(widened, None)
    ref = _loaded_store(shots)
    ref.apply_block(table, None)
    assert _state(store) == _state(ref)
    store.close()
    ref.close()


def test_a_factor_table_with_an_int32_field_out_of_range_is_refused_and_state_untouched() -> None:
    rng = np.random.default_rng(3)
    shots = [random_state(rng, 70, 4, "plain") for _ in range(3)]
    lists = [
        CompiledCliffordFactorList(3, None, (CompiledPauliAxis(*_nonzero_axis(rng, 70)),)),
        CompiledCliffordFactorList(1, None, (CompiledPauliAxis(*_nonzero_axis(rng, 70)),)),
    ]
    table = factor_table(lists, 2)
    shot_list = np.array([0, 1, 0], dtype=np.int32)
    for name in ("n_r", "v_fwd", "v_inv", "axis_start"):
        for big in _TOO_WIDE:
            col = _widen(getattr(table, name), len(lists) - 1, big)
            store = _loaded_store(shots)
            before = _state(store)
            with pytest.raises((ValueError, OverflowError)):
                store.engine.inject_cliffords(table._replace(**{name: col}), shot_list, False)
            assert _state(store) == before, (name, big)
            store.close()


@pytest.mark.parametrize("mode", [2**32 + 1, 2**64 + 1, 2**33 + 1, 2**32, -(2**32) + 1])
def test_the_wrapper_refuses_a_table_mode_that_only_wraps_to_a_valid_one(mode: int) -> None:
    """The headline case of the LOW finding, at the engine wrapper itself."""
    shots, blk = _cap_batch(3)
    table = block_table(blk, 70)
    assert table is not None
    store = _loaded_store(shots)
    before = _state(store)
    with pytest.raises((ValueError, OverflowError)):
        store.engine.apply_block(table._replace(mode=mode), True, None)
    assert _state(store) == before
    store.close()
