"""Error paths of `mbe_inject_cliffords` against `apply_clifford_factors(check_unitary=True)`.

For every failure the exception type and message must equal the per-shot function's, and the set
of shots that were already updated when it was raised must be the same (shots before the failing
one are committed, the failing one and the later ones are untouched).
"""

from __future__ import annotations

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import CompiledCliffordFactorList, apply_clifford_factors
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_clifford_diff_test import random_factor_list
from stimside._magic.live_store_test import (
    _first_difference,
    _nonzero_axis,
    random_state,
)


def _both(states, lists, which, inverse, atol, n):
    """Run oracle and engine; return (oracle exc, engine exc, oracle states, engine states)."""
    oracle = [s.copy() for s in states]
    exc_o: BaseException | None = None
    try:
        for s, w in enumerate(which):
            if w >= 0:
                oracle[s] = apply_clifford_factors(
                    oracle[s], lists[w], inverse=inverse, check_unitary=True, atol=atol
                )
    except (FloatingPointError, ValueError, RuntimeError) as exc:  # the observable
        exc_o = exc
    store = LiveExpansions(len(states), n, atol)
    for b, s in enumerate(states):
        store[b] = s
    exc_e: BaseException | None = None
    try:
        store.inject_factor_lists(lists, np.asarray(which, dtype=np.int32), inverse)
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        exc_e = exc
    got = [store[i] for i in range(len(states))]
    store.close()
    return exc_o, exc_e, oracle, got


def _same(exc_o, exc_e, oracle, got) -> None:
    assert exc_o is not None and exc_e is not None, (exc_o, exc_e)
    assert type(exc_o) is type(exc_e) and str(exc_o) == str(exc_e), (exc_o, exc_e)
    for i, (x, y) in enumerate(zip(oracle, got, strict=True)):
        assert _first_difference(x, y) is None, f"shot {i}: {_first_difference(x, y)}"


@pytest.mark.parametrize("n,chi", [(5, 4), (5, 64), (6, 4), (7, 64)])
@pytest.mark.parametrize("inverse", [False, True])
def test_a_violated_unitarity_check_raises_the_same_message_and_leaves_the_same_shots(
    n: int, chi: int, inverse: bool
) -> None:
    rng = np.random.default_rng([n, chi, inverse])
    atol = 1e-300  # no rounding error can pass: the first rotated shot fails
    hits = 0
    for trial in range(40):
        states = [random_state(rng, n, chi, k) for k in ("plain", "d", "raw", "plain", "d")]
        for st in states:  # generic weights: the 1/sqrt(2) rotation then rounds the norm
            st.weights = rng.standard_normal(st.chi) + 1j * rng.standard_normal(st.chi)
        lists = [random_factor_list(rng, n) for _ in range(3)]
        lists[0] = CompiledCliffordFactorList(1, None, (CompiledPauliAxis(*_nonzero_axis(rng, n)),))
        which = [-1, 0, int(rng.integers(0, 3)), -1, int(rng.integers(0, 3))]
        exc_o, exc_e, oracle, got = _both(states, lists, which, inverse, atol, n)
        if exc_o is None:
            assert exc_e is None
            continue
        hits += 1
        assert "unitary norm conservation violated" in str(exc_o)
        _same(exc_o, exc_e, oracle, got)
    assert hits >= 3, hits


def test_the_failing_shot_is_untouched_and_the_earlier_ones_are_updated() -> None:
    n, rng = 70, np.random.default_rng(3)
    states = [random_state(rng, n, 4, "plain") for _ in range(6)]
    for s in states[:3]:  # shots 0-2 can rotate; shot 3 has non-finite weights
        assert np.all(np.isfinite(s.weights))
    bad = states[3].copy()
    bad.weights = bad.weights.copy()
    bad.weights[0] = complex(np.nan, 0.0)
    states[3] = bad
    ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
    lists = [CompiledCliffordFactorList(3, None, (ax,))]
    which = [0] * 6
    exc_o, exc_e, oracle, got = _both(states, lists, which, False, 1e-10, n)
    assert isinstance(exc_o, FloatingPointError) and "Non-finite weights" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)
    for i in (0, 1, 2):
        assert _first_difference(got[i], states[i]) is not None  # updated
    for i in (3, 4, 5):
        assert _first_difference(got[i], states[i]) is None  # untouched


@pytest.mark.parametrize("where", ["p0", "r_axis"])
def test_non_finite_weights_raise_the_reference_message_for_each_step_kind(where: str) -> None:
    n, rng = 5, np.random.default_rng(4)
    st = random_state(rng, n, 4, "plain")
    st.weights = st.weights.copy()
    st.weights[1] = complex(np.inf, 0.0)
    ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
    fac = (
        CompiledCliffordFactorList(0, ax, ())
        if where == "p0"
        else CompiledCliffordFactorList(0, None, (ax,))
    )
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
    op = "apply_pauli" if where == "p0" else "apply_quarter_turn"
    assert str(exc_o) == f"Non-finite weights in {op}"
    _same(exc_o, exc_e, oracle, got)


def test_a_non_hermitian_axis_raises_the_reference_value_error() -> None:
    n, rng = 5, np.random.default_rng(6)
    st = random_state(rng, n, 1, "plain")
    a = pq.pack_bit_vector(np.array([1, 0, 0, 0, 0], dtype=np.uint8), n)
    b = pq.pack_bit_vector(np.array([1, 0, 0, 0, 0], dtype=np.uint8), n)
    bad = CompiledPauliAxis(a, b, 0)  # <a,b> = 1 needs odd kappa
    fac = CompiledCliffordFactorList(0, None, (bad,))
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
    assert isinstance(exc_o, ValueError) and "must be Hermitian" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)


def test_padding_bits_in_an_axis_raise_the_reference_value_error() -> None:
    n, rng = 5, np.random.default_rng(7)
    st = random_state(rng, n, 1, "plain")
    a = np.array([1 << 40], dtype=np.uint64)
    z = np.zeros(1, dtype=np.uint64)
    fac = CompiledCliffordFactorList(0, CompiledPauliAxis(a, z, 0), ())
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
    assert isinstance(exc_o, ValueError) and "padding" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)


def test_a_chi_zero_shot_is_skipped_even_with_a_bad_factor_list() -> None:
    n, rng = 5, np.random.default_rng(8)
    dead = random_state(rng, n, 1, "zero")
    a = pq.pack_bit_vector(np.array([1, 0, 0, 0, 0], dtype=np.uint8), n)
    fac = CompiledCliffordFactorList(0, None, (CompiledPauliAxis(a, a, 0),))  # non-Hermitian
    out = apply_clifford_factors(dead, fac, check_unitary=True, atol=1e-10)
    assert out.chi == 0  # the reference returns it untouched, without validating the axis
    store = LiveExpansions(1, n, 1e-10)
    store[0] = dead
    store.inject_factor_lists([fac], np.array([0], dtype=np.int32), False)
    assert store[0].chi == 0
    store.close()


def test_a_shot_list_entry_outside_the_table_is_refused_before_anything_changes() -> None:
    n, rng = 5, np.random.default_rng(9)
    states = [random_state(rng, n, 4, "plain") for _ in range(3)]
    fac = CompiledCliffordFactorList(1, None, ())
    store = LiveExpansions(3, n, 1e-10)
    for b, s in enumerate(states):
        store[b] = s
    with pytest.raises(RuntimeError, match="failed before running"):
        store.inject_factor_lists([fac], np.array([0, 5, 0], dtype=np.int32), False)
    for i in range(3):
        assert _first_difference(states[i], store[i]) is None
    store.close()


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_an_overflowing_norm_raises_the_reference_non_finite_norm_message() -> None:
    n, rng = 5, np.random.default_rng(10)
    st = random_state(rng, n, 4, "plain")
    st.weights = st.weights * 1e200  # finite weights whose squared norm overflows
    ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
    fac = CompiledCliffordFactorList(0, None, (ax,))
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
    assert str(exc_o) == "Non-finite input norm in apply_quarter_turn"
    _same(exc_o, exc_e, oracle, got)


def _rounding_case(rng: np.random.Generator, scale: float):
    """A (state, factor list, delta) where one quarter turn changes the squared norm by delta > 0.

    `delta` is measured with the oracle itself (check_unitary off), so it is the exact rounding
    error the check sees; `pre` is the pre-norm (scaled below 1 by `scale`).
    """
    n = 5
    while True:
        st = random_state(rng, n, 4, "plain")
        raw = rng.standard_normal(st.chi) + 1j * rng.standard_normal(st.chi)
        st.weights = np.asarray(raw * scale, dtype=np.complex128)
        fac = CompiledCliffordFactorList(0, None, (CompiledPauliAxis(*_nonzero_axis(rng, n)),))
        pre = st.norm_sq()
        post = apply_clifford_factors(st.copy(), fac, check_unitary=False, atol=1.0).norm_sq()
        if post != pre:
            return st, fac, abs(post - pre), pre


def _tolerance_outcome(st, fac, atol: float):
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, atol, 5)
    assert (exc_o is None) == (exc_e is None), (atol, exc_o, exc_e)
    if exc_o is not None:
        _same(exc_o, exc_e, oracle, got)
    return exc_o is not None


@pytest.mark.parametrize("scale", [1.0, 0.3, 0.05])
def test_the_unitarity_tolerance_boundary_matches_the_reference_exactly(scale: float) -> None:
    """|post - pre| <= atol * max(1, pre): on, just below and just above the edge, pre < 1 too."""
    rng = np.random.default_rng(int(scale * 1000))
    checked = 0
    for _ in range(25):
        st, fac, delta, pre = _rounding_case(rng, scale)
        edge = delta / max(1.0, pre)  # atol at which `<=` flips
        for atol, fails in ((edge * 0.5, True), (edge * 2.0, False), (edge * 1e-3, True)):
            assert _tolerance_outcome(st, fac, atol) is fails, (scale, atol, edge)
        if pre < 1.0:  # between delta and delta / pre a `tol * pre` check would wrongly fail
            assert _tolerance_outcome(st, fac, delta / np.sqrt(pre)) is False
        checked += 1
    assert checked == 25


def test_padding_bits_in_a_quarter_turn_axis_raise_the_reference_value_error() -> None:
    """Not the kernel's error code 1: the reference validates the words first (ValueError)."""
    n, rng = 5, np.random.default_rng(12)
    st = random_state(rng, n, 4, "plain")
    a = np.array([1 << 40], dtype=np.uint64)
    z = np.zeros(1, dtype=np.uint64)
    fac = CompiledCliffordFactorList(0, None, (CompiledPauliAxis(a, z, 0),))
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], True, 1e-10, n)
    assert isinstance(exc_o, ValueError) and "padding" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)


def _table(**override):
    from stimside._native.magic_engine import FactorTable

    base = {
        "has_p0": np.array([1], np.uint8), "n_r": np.array([1], np.int32),
        "v_fwd": np.array([-1], np.int32), "v_inv": np.array([-1], np.int32),
        "axis_start": np.array([0], np.int32), "axes_a": np.zeros((2, 1), np.uint64),
        "axes_b": np.zeros((2, 1), np.uint64), "axes_kappa": np.zeros(2, np.uint8),
    }
    return FactorTable(**{**base, **override})


@pytest.mark.parametrize(
    "bad",
    [
        {"axis_start": np.array([1], np.int32)},  # p0 + 1 axis from index 1 needs 3 axes, has 2
        {"axis_start": np.array([-1], np.int32)},
        {"n_r": np.array([5], np.int32)},
        {"n_r": np.array([-1], np.int32)},
    ],
)
def test_an_inconsistent_factor_table_is_refused_before_anything_is_read_or_changed(bad) -> None:
    store = LiveExpansions(2, 5, 1e-10)
    eng = store.engine
    before = [store[i] for i in range(2)]
    with pytest.raises(RuntimeError, match="failed before running"):
        eng.inject_cliffords(_table(**bad), np.array([0, 0], np.int32), False)
    for i in range(2):
        assert _first_difference(before[i], store[i]) is None
    store.close()
