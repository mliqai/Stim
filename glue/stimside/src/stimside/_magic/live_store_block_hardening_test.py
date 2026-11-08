"""Hardening tests of `mbe_apply_block` / `mbe_partition` found missing by the S4b-2 review gate.

Each test pins something the engine already does right, so that an edit which broke it would
be seen (every one was checked against a mutant of the engine, see the S4b-2 fix handoff):

* the partition key: two shots that differ in exactly ONE field must stay separate groups,
  in the engine, in B3's `_group_shots_by_state` and in a pairwise comparison (a field dropped
  from both the exact comparison and the cheap fingerprint would otherwise merge them);
* `failed_shot` is reset when `apply_block` starts and names the FIRST member of a failing group;
* `max_components` that is negative, huge (>= 2**63) or hit by an empty result, through the
  wrapper and at the raw engine;
* non-finite inputs and overflow inside a block raise the reference's error at the reference's
  place, with the reference's partial state;
* `g * coeffs` operand order with a non-unit scalar, N2 followed by another step, an
  all-zero block's `a_union`.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
import stimside.simulator_magic as sm
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, block_table
from stimside._magic.live_store_block_diff_test import (
    commuting_axes,
    make_shots,
    random_records,
    reference_groups,
)
from stimside._magic.live_store_test import _first_difference, random_state
from stimside._magic.plan import CompiledOpenBlockRecord, compile_open_block
from stimside._magic.query import QueryBudgetExceededError
from stimside._native import magic_engine as me

_ATOL = 1e-10
_ERRORS = (FloatingPointError, ValueError, QueryBudgetExceededError)


# --------------------------------------------------------------------------------------------
# One block, reference vs engine
# --------------------------------------------------------------------------------------------


def reference_run(
    shots: list[pq.ShotExpansion], blk: pq.CompiledPauliBlock, atol: float, cap: int | None
) -> tuple[list[pq.ShotExpansion], Exception | None]:
    """`_flush_magic`'s loop body for one block: groups by pairwise comparison, per-group
    `apply_compiled_block`, copies to the members, then the `max_components` bound."""
    exps = [s.copy() for s in shots]
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for members in reference_groups(exps):
                out = pq.apply_compiled_block(
                    exps[members[0]], blk, check_unitary=True, atol=atol
                )
                for s in members[1:]:
                    exps[s] = out.copy()
                exps[members[0]] = out
                if cap is not None and out.chi > cap:
                    raise QueryBudgetExceededError(
                        f"Expansion exceeded max_components={cap} (got chi={out.chi})"
                    )
    except _ERRORS as exc:
        return exps, exc
    return exps, None


def engine_run(
    shots: list[pq.ShotExpansion], blk: pq.CompiledPauliBlock, atol: float, cap: int | None,
    dedupe: bool = True,
) -> tuple[list[pq.ShotExpansion], Exception | None, int | None]:
    n = shots[0].n_qubits
    store = LiveExpansions(len(shots), n, atol)
    for i, s in enumerate(shots):
        store[i] = s
    table = block_table(blk, n)
    assert table is not None
    exc: Exception | None = None
    try:
        store.apply_block(table, cap, dedupe)
    except _ERRORS as err:
        exc = err
    out = [store[i] for i in range(len(shots))]
    failed = store.failed_shot
    store.close()
    return out, exc, failed


def assert_engine_equals_reference(
    shots: list[pq.ShotExpansion], blk: pq.CompiledPauliBlock, atol: float = _ATOL,
    cap: int | None = None,
) -> Exception | None:
    """Same exception (type and text) and same bytes in every shot; returns the exception."""
    want, wexc = reference_run(shots, blk, atol, cap)
    got, gexc, _failed = engine_run(shots, blk, atol, cap)
    assert (type(gexc), str(gexc)) == (type(wexc), str(wexc))
    for i in range(len(shots)):
        assert _first_difference(want[i], got[i]) is None, (i, _first_difference(want[i], got[i]))
    return wexc


def _blocks(rng: np.random.Generator, n: int, sequential: bool) -> list[pq.CompiledPauliBlock]:
    out = []
    for _ in range(4):
        recs = compile_open_block(
            random_records(rng, n), n_qubits=n, budget=1 if sequential else 64
        )
        out += [r.block for r in recs if isinstance(r, CompiledOpenBlockRecord)]
    return [b for b in out if (b.mode == "sequential") == sequential]


def _state(rng: np.random.Generator, n: int, chi: int = 4, kind: str = "d") -> pq.ShotExpansion:
    while True:
        st = random_state(rng, n, chi, kind)
        if st.chi > 0:
            return st


# --------------------------------------------------------------------------------------------
# The partition key: one differing field keeps two shots apart
# --------------------------------------------------------------------------------------------


def _with(a: pq.ShotExpansion, **fields: Any) -> pq.ShotExpansion:
    e = a.copy()
    for name, arr in fields.items():
        setattr(e, name, arr)
    return e


def _flip(arr: Any, idx: tuple[int, ...]) -> Any:
    out = arr.copy()
    out[idx] = out[idx] ^ out.dtype.type(1)
    return out


def _one_field_variants(a: pq.ShotExpansion) -> dict[str, pq.ShotExpansion]:
    """Copies of `a` that differ from it in exactly one field (and one position of it)."""
    mid = a.chi // 2
    w = a.weights.copy()
    w.view(np.uint64)[2 * 1] ^= np.uint64(1)  # row 1: neither a strided sample nor the last row
    wm = a.weights.copy()
    wm.view(np.uint64)[2 * mid] ^= np.uint64(1)
    ell_mid = a.ell.copy()
    ell_mid[mid, 0] = (ell_mid[mid, 0] + 2) & 3
    ell_last = a.ell.copy()
    ell_last[-1, 0] = (ell_last[-1, 0] + 2) & 3
    d = a.d - 1
    assert a.ell_odd is not None
    return {
        "weights_row1": _with(a, weights=w),
        "weights_mid_row": _with(a, weights=wm),
        "h_row1": _with(a, h_words=_flip(a.h_words, (1, 0))),
        "h_last_row": _with(a, h_words=_flip(a.h_words, (a.chi - 1, 0))),
        "ell_mid_row": _with(a, ell=ell_mid),
        "ell_last_row": _with(a, ell=ell_last),
        "A_words": _with(a, A_words=_flip(a.A_words, (0, 0))),
        "a_union": _with(a, a_union=_flip(a.a_union, (0,))),
        "Gamma": _with(a, Gamma=_flip(a.Gamma, (0, 0))),
        "pivots": _with(a, pivots=_flip(a.pivots, (0,))),
        "ell_odd": _with(a, ell_odd=_flip(a.ell_odd, (0,))),
        "atol": _with(a, atol=a.atol * 2),
        "log_mass": _with(a, log_mass=float(np.nextafter(a.log_mass, 1e9))),
        "canonical_flag": _with(a, _is_canonical_merged=not a._is_canonical_merged),
        "chi": _with(a, weights=a.weights[:-1].copy(), h_words=a.h_words[:-1].copy(),
                     ell=a.ell[:-1].copy()),
        "d": _with(a, d=d, A_words=a.A_words[:d].copy(), Gamma=a.Gamma[:d, :d].copy(),
                   pivots=a.pivots[:d].copy(), ell_odd=a.ell_odd[:d].copy(),
                   ell=a.ell[:, :d].copy()),
    }


def _groups_of(rep: list[int]) -> list[list[int]]:
    got: dict[int, list[int]] = {}
    for s, r in enumerate(rep):
        if r >= 0:
            got.setdefault(r, []).append(s)
    return sorted(got.values())


_PARTITION_BASE = random_state(np.random.default_rng(5), 70, 64, "d")
assert _PARTITION_BASE.d >= 2 and _PARTITION_BASE.chi >= 24
_PARTITION_FIELDS = sorted(_one_field_variants(_PARTITION_BASE))


@pytest.mark.parametrize("field", _PARTITION_FIELDS)
def test_two_shots_differing_in_one_field_stay_apart_in_the_engine_b3_and_pairwise(
    field: str,
) -> None:
    a = _PARTITION_BASE.copy()
    v = _one_field_variants(a)[field]
    shots = [a.copy(), v, a.copy(), v.copy()]
    store = LiveExpansions(4, 70, a.atol)
    for i, s in enumerate(shots):
        store[i] = s
    rep = store.engine.partition(True).tolist()
    store.close()
    want = [[0, 2], [1, 3]]
    assert sm._group_shots_by_state(shots) == want  # B3 tells them apart
    assert reference_groups(shots) == want  # and so does a pairwise byte comparison
    assert _groups_of(rep) == want, (field, rep)  # the engine too, field by field
    assert rep == [0, 1, 0, 1]


def test_the_engine_partition_equals_b3_on_a_mixed_batch_of_one_field_variants() -> None:
    a = _PARTITION_BASE.copy()
    variants = _one_field_variants(a)
    shots = [a.copy()]
    for name in sorted(variants):  # every variant twice, base copies in between
        shots += [variants[name].copy(), a.copy(), variants[name].copy()]
    store = LiveExpansions(len(shots), 70, a.atol)
    for i, s in enumerate(shots):
        store[i] = s
    got = _groups_of(store.engine.partition(True).tolist())
    store.close()
    assert got == sorted(sm._group_shots_by_state(shots)) == sorted(reference_groups(shots))
    assert len(got) == 1 + len(variants)  # the base group and one group per variant


# --------------------------------------------------------------------------------------------
# failed_shot
# --------------------------------------------------------------------------------------------


def _nan_state(st: pq.ShotExpansion) -> pq.ShotExpansion:
    bad = st.copy()
    w = bad.weights.copy()
    w[0] = complex("nan")
    bad.weights = w
    return bad


def _block(rng: np.random.Generator, n: int, sequential: bool = False) -> pq.CompiledPauliBlock:
    return _blocks(rng, n, sequential)[0]


def test_failed_shot_is_the_first_member_of_a_failing_multi_member_group() -> None:
    n, rng = 5, np.random.default_rng(1)
    good, bad = _state(rng, n), _nan_state(_state(rng, n))
    blk = _block(rng, n)
    shots = [good, bad, bad.copy(), bad.copy(), bad.copy()]
    got, exc, failed = engine_run(shots, blk, _ATOL, None)
    assert isinstance(exc, FloatingPointError) and "Non-finite weights" in str(exc)
    assert failed == 1  # the lowest member of the failing group of four, not 4 (the last)
    want, wexc = reference_run(shots, blk, _ATOL, None)
    assert str(wexc) == str(exc)
    for i in range(len(shots)):
        assert _first_difference(want[i], got[i]) is None


def test_failed_shot_is_reset_when_apply_block_starts() -> None:
    """A failure leaves `failed_shot` set; the next, successful `apply_block` clears it (and so
    does one that fails elsewhere: it reports its own shot, not the stale one)."""
    n, rng = 5, np.random.default_rng(2)
    good, bad = _state(rng, n), _nan_state(_state(rng, n))
    blk = _block(rng, n)
    store = LiveExpansions(3, n, _ATOL)
    for i, s in enumerate((good, bad, good)):
        store[i] = s
    table = block_table(blk, n)
    assert table is not None
    with pytest.raises(FloatingPointError):
        store.apply_block(table, None)
    assert store.failed_shot == 1
    store[1] = good  # repaired: the next block succeeds
    store.apply_block(table, None)
    assert store.failed_shot is None
    store[0] = bad
    with pytest.raises(FloatingPointError):
        store.apply_block(table, None)
    assert store.failed_shot == 0  # its own failing shot, not the stale 1
    store.close()


# --------------------------------------------------------------------------------------------
# max_components: negative, huge, an empty result
# --------------------------------------------------------------------------------------------


def _cap_batch(seed: int, n: int = 70) -> tuple[list[pq.ShotExpansion], pq.CompiledPauliBlock]:
    rng = np.random.default_rng(seed)
    return make_shots(rng, n, 4, 3), _block(rng, n)


@pytest.mark.parametrize("cap", [-1, -5, -(2**62), -(2**63), -(10**30), 0])
def test_a_negative_or_zero_max_components_is_exceeded_by_every_group_like_the_reference(
    cap: int,
) -> None:
    shots, blk = _cap_batch(3)
    exc = assert_engine_equals_reference(shots, blk, cap=cap)
    assert isinstance(exc, QueryBudgetExceededError), cap
    assert f"max_components={cap}" in str(exc)


@pytest.mark.parametrize("cap", [2**63 - 1, 2**63, 2**64 - 1, 2**64, 2**70, 10**30])
def test_a_max_components_of_two_to_the_63_or_more_means_no_bound_through_the_wrapper(
    cap: int,
) -> None:
    """`coerce_strict_int` accepts 10**30; ctypes would raise or wrap it as an int64."""
    shots, blk = _cap_batch(4)
    assert assert_engine_equals_reference(shots, blk, cap=cap) is None
    store = LiveExpansions(len(shots), 70, _ATOL)
    for i, s in enumerate(shots):
        store[i] = s
    table = block_table(blk, 70)
    assert table is not None
    store.apply_block(table, cap)  # no exception
    store.close()


def _empty_result_case() -> tuple[list[pq.ShotExpansion], pq.CompiledPauliBlock]:
    """An all-zero pauli_sum on tiny states: it passes the unitarity check (norm**2 <= atol)
    and leaves chi == 0, so `0 > cap` decides the bound."""
    n, rng = 70, np.random.default_rng(6)
    nw = pq.n_words_for_qubits(n)
    blk = pq.CompiledPauliBlock(
        n_qubits=n, mode="pauli_sum", coeffs=np.zeros(2, dtype=np.complex128),
        a_words=np.zeros((2, nw), dtype=np.uint64), b_words=np.zeros((2, nw), dtype=np.uint64),
        kappas=np.zeros(2, dtype=np.uint8),
    )
    shots = []
    for _ in range(3):
        st = _state(rng, n)
        st.weights = st.weights * 1e-9
        shots.append(st)
    return shots, blk


def test_a_negative_bound_is_exceeded_even_by_an_empty_result() -> None:
    """chi == 0 after the block: with cap -1 the bound is still hit (0 > -1), which a bound
    clamped to 0 (the wrong clamp) would miss; cap 0 and larger are not exceeded by chi == 0."""
    shots, blk = _empty_result_case()
    for neg in (-1, -(10**30)):
        exc = assert_engine_equals_reference(shots, blk, cap=neg)
        assert isinstance(exc, QueryBudgetExceededError) and "got chi=0" in str(exc), neg
    for ok_cap in (0, 1, None):
        assert assert_engine_equals_reference(shots, blk, cap=ok_cap) is None, ok_cap


def test_the_raw_engine_treats_every_negative_bound_as_always_exceeded() -> None:
    """Below the wrapper: `mbe_apply_block` with INT64 values < 0 fails the first group (the
    group is updated first, as the reference does) and with INT64_MAX it never fails."""
    shots, blk = _cap_batch(7)
    n = 70
    table = block_table(blk, n)
    assert table is not None
    for raw_cap in (-1, -5, -(2**63)):
        store = LiveExpansions(len(shots), n, _ATOL)
        for i, s in enumerate(shots):
            store[i] = s
        eng = store.engine
        res = eng._lib  # the bound is an int64 argument: bypass the wrapper's clamp
        err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
        t = (table.step_kind, table.step_sign, table.step_start, table.axes_a, table.axes_b,
             table.axes_kappa, table.axes_w_ri)
        bad = res.mbe_apply_block(
            eng._h(), table.mode, int(table.step_kind.shape[0]), int(table.axes_kappa.shape[0]),
            table.g_re, table.g_im, *(me.addr(np.ascontiguousarray(x)) for x in t), 1, raw_cap,
            me.addr(err), me.addr(errd), me.addr(info),
        )
        assert bad >= 0 and err[0] == me.FAIL_MAX_COMPONENTS, (raw_cap, bad, err)
        store.close()


# --------------------------------------------------------------------------------------------
# Non-finite inputs and overflow inside a block
# --------------------------------------------------------------------------------------------


def _scaled_state(st: pq.ShotExpansion, k: float) -> pq.ShotExpansion:
    out = st.copy()
    out.weights = out.weights * k
    return out


def _with_scalar(blk: pq.CompiledPauliBlock, g: complex) -> pq.CompiledPauliBlock:
    return dataclasses.replace(blk, global_scalar=g)


def _step_block(n: int, kind: str, axes: list[CompiledPauliAxis]) -> pq.CompiledPauliBlock:
    paulis = tuple(a.as_tuple() for a in axes)
    steps = (pq.CompiledPauliBlockStep(kind=kind, paulis=paulis),)  # type: ignore[arg-type]
    return pq.CompiledPauliBlock(n_qubits=n, mode="sequential", steps=steps)


def test_non_finite_input_norm_and_non_finite_weights_raise_the_reference_errors() -> None:
    """Weights that are finite but whose squared norm overflows ("Non-finite input norm"), and
    NaN / +-inf weights ("Non-finite weights in apply_compiled_block"), in both block modes."""
    n, seen = 5, Counter[str]()
    for seed, sequential in ((s, q) for s in range(3) for q in (False, True)):
        rng = np.random.default_rng([seed, 11])
        st = _state(rng, n)
        for blk in _blocks(rng, n, sequential)[:2]:
            for bad_w in (float("nan"), float("inf"), -float("inf")):
                w = st.weights.copy()
                w[1] = complex(bad_w, 0.0)
                bad = st.copy()
                bad.weights = w
                exc = assert_engine_equals_reference([bad, st.copy(), bad.copy()], blk)
                seen[str(exc)] += 1
            exc = assert_engine_equals_reference([_scaled_state(st, 1e160), st.copy()], blk)
            seen[str(exc)] += 1
    assert seen["Non-finite weights in apply_compiled_block"] >= 12, seen
    assert seen["Non-finite input norm in apply_compiled_block"] >= 6, seen


def test_a_global_scalar_that_overflows_raises_the_reference_error_in_both_modes() -> None:
    """pauli_sum: the products overflow to inf and `merge_frames` refuses ("Non-finite weights
    encountered in merge_frames"); 1e308 is the edge where the norm, not a weight, overflows.
    sequential: the first step that checks its input names itself."""
    n, seen = 5, Counter[str]()
    for seed in range(4):
        rng = np.random.default_rng([seed, 12])
        st = _state(rng, n)
        for sequential in (False, True):
            for blk in _blocks(rng, n, sequential)[:2]:
                for scale, g in ((1e100, 1e250), (1.0, 1e308), (3.0, 1e308), (1e-5, 1e308)):
                    exc = assert_engine_equals_reference(
                        [_scaled_state(st, scale), st.copy()], _with_scalar(blk, complex(g, 0.0))
                    )
                    seen[("seq " if sequential else "sum ") + str(exc)] += 1
    assert seen["sum Non-finite weights encountered in merge_frames"] >= 4, seen
    assert any(k.startswith("seq Non-finite weights in apply_") for k in seen), seen
    assert any("unitary norm conservation violated" in k for k in seen), seen


@pytest.mark.parametrize("kind,n_axes,text", [
    ("N1", 1, "Non-finite weights in apply_pauli_t"),
    ("N2", 3, "Non-finite weights in apply_ccz"),
])
def test_a_branching_step_refuses_non_finite_weights_by_its_own_name(
    kind: str, n_axes: int, text: str
) -> None:
    """An N1 / N2 step is the first thing that sees the overflowed weights (`weights *= g`
    does not check): the error names apply_pauli_t / apply_ccz, as the reference's does."""
    n, rng = 5, np.random.default_rng(13)
    axes = commuting_axes(rng, n, 3)[:n_axes]
    blk = _with_scalar(_step_block(n, kind, axes), complex(1e250, 0.0))
    st = _scaled_state(_state(rng, n), 1e100)
    exc = assert_engine_equals_reference([st, st.copy(), _state(rng, n)], blk)
    assert isinstance(exc, FloatingPointError) and text in str(exc), exc


# --------------------------------------------------------------------------------------------
# Operand order, N2 then another step, r == 0
# --------------------------------------------------------------------------------------------


def _random_pauli_sum(
    rng: np.random.Generator, n: int, r: int, g: complex
) -> pq.CompiledPauliBlock:
    nw = pq.n_words_for_qubits(n)
    a = np.zeros((r, nw), np.uint64)
    b = np.zeros((r, nw), np.uint64)
    for m in range(r):
        b[m] = pq.pack_bit_vector((rng.random(n) < 0.3).astype(np.uint8), n)
    coeffs = (rng.standard_normal(r) + 1j * rng.standard_normal(r)) * 0.3
    return pq.CompiledPauliBlock(
        n_qubits=n, mode="pauli_sum", coeffs=coeffs.astype(np.complex128), a_words=a,
        b_words=b, kappas=np.zeros(r, np.uint8), global_scalar=g, is_unitary=False,
    )


def test_a_non_unit_global_scalar_multiplies_the_coefficients_from_the_left() -> None:
    """numpy rounds `g * coeffs` (Python complex on the left) differently from `coeffs * g`: random
    complex coefficients and scalar expose the operand order (atol 1e300 lets every block pass the
    unitarity check, so the states are compared)."""
    n, differing = 5, 0
    rng = np.random.default_rng(14)
    for _ in range(60):
        st = _state(rng, n, int(rng.choice([1, 2, 4, 8])))
        g = complex(*rng.standard_normal(2))
        blk = _random_pauli_sum(rng, n, int(rng.integers(1, 5)), g)
        assert assert_engine_equals_reference([st, st.copy()], blk, atol=1e300) is None
        coeffs = blk.coeffs
        assert coeffs is not None
        swapped = np.asarray(coeffs * g, dtype=np.complex128)
        differing += not np.array_equal(
            swapped.view(np.uint64), np.asarray(g * coeffs, dtype=np.complex128).view(np.uint64)
        )
    assert differing >= 10, differing  # the order matters on this data: the test is not vacuous


def test_a_sequential_block_applies_a_non_unit_scalar_before_the_steps() -> None:
    n, rng = 70, np.random.default_rng(15)
    axes = commuting_axes(rng, n, 3)
    steps = tuple(
        pq.CompiledPauliBlockStep(kind=k, paulis=(a.as_tuple(),), sign=s)  # type: ignore[arg-type]
        for k, a, s in (("N1", axes[0], 1), ("P2", axes[1], -1), ("P1", axes[2], 1))
    )
    for chi in (1, 2, 4):
        g = complex(*rng.standard_normal(2))
        blk = pq.CompiledPauliBlock(n_qubits=n, mode="sequential", steps=steps, global_scalar=g)
        shots = [_state(rng, n, chi, "plain"), _state(rng, n, chi, "d")]
        assert assert_engine_equals_reference(shots, blk, atol=1e300) is None


def test_an_n2_step_followed_by_other_steps_reads_the_rows_after_its_eight() -> None:
    """An N2 step owns 8 rows of the axes table; the step after it starts at row 8. A table that
    advanced by 1 would run the next step on a row of the N2 step (a different Pauli)."""
    n, rng = 70, np.random.default_rng(16)
    axes = commuting_axes(rng, n, 4)
    triple = tuple(a.as_tuple() for a in axes[:3])
    last = axes[3].as_tuple()
    for shape in (("N2", "N1"), ("N2", "P1"), ("N2", "P2"), ("N1", "N2", "P2"),
                  ("N2", "N2", "N1")):
        steps = []
        for i, name in enumerate(shape):
            kind: Any = name
            paulis = triple if kind == "N2" else (last,)
            steps.append(pq.CompiledPauliBlockStep(
                kind=kind, paulis=paulis, sign=-1 if (kind != "P1" and i % 2) else 1))
        blk = pq.CompiledPauliBlock(n_qubits=n, mode="sequential", steps=tuple(steps))
        table = block_table(blk, n)
        assert table is not None and table.axes_a.shape[0] == sum(
            8 if k == "N2" else 1 for k in shape)
        starts, row = table.step_start.tolist(), 0
        for kind, start in zip(shape, starts):  # each start is the sum of the earlier row counts
            assert start == row, (shape, starts)
            row += 8 if kind == "N2" else 1
        shots = [_state(rng, n, 2, "d"), _state(rng, n, 1, "plain"), _state(rng, n, 4, "d")]
        assert assert_engine_equals_reference(shots, blk) is None, shape


def test_an_all_zero_block_clears_a_union_like_the_reference() -> None:
    """r == 0 after dropping zero coefficients: the empty expansion has a zero `a_union`
    (the old one, non-zero for d > 0, must not survive in the engine)."""
    shots, blk = _empty_result_case()
    assert all(s.d > 0 and s.a_union.any() for s in shots)  # the premise
    want, exc = reference_run(shots, blk, _ATOL, None)
    assert exc is None
    got, gexc, _ = engine_run(shots, blk, _ATOL, None)
    assert gexc is None
    for i in range(len(shots)):
        assert want[i].chi == got[i].chi == 0 and not got[i].a_union.any()
        assert _first_difference(want[i], got[i]) is None
