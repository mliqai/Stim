"""Regression tests for the S4b-1 review findings on `mbe_inject_cliffords` (fix stage).

Each test compares the engine with the per-shot reference `apply_clifford_factors` on a case the
first S4b-1 tests missed, or pins a property found by a mutant that survived them: malformed axis
shapes, a non-finite value at the *last* weight or in an imaginary part, padding bits in the `b`
word only, the exact unitarity-tolerance tie and its scaling by the pre-norm, the K1 merge
tolerance and the `chi > 2` merge condition, and the kappa in the Hermitian error message.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import CompiledCliffordFactorList, apply_clifford_factors
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, factor_table
from stimside._magic.live_store_clifford_errors_test import _both, _rounding_case, _same
from stimside._magic.live_store_test import (
    _first_difference,
    _nonzero_axis,
    random_state,
)


def _list_with(p0: Any = None, r: tuple[Any, ...] = (), v: int = 0) -> CompiledCliffordFactorList:
    return CompiledCliffordFactorList(v, p0, r)


# ---- B: axis shapes -------------------------------------------------------------------------


def _bad_axes(n: int, rng: np.random.Generator) -> dict[str, CompiledPauliAxis]:
    a, b, kap = _nonzero_axis(rng, n)
    z1 = np.zeros(1, np.uint64)
    short = {} if pq.n_words_for_qubits(n) == 1 else {  # one word is the right shape for n <= 64
        "length one": CompiledPauliAxis(np.array([1], np.uint64), z1, 0)
    }
    return {
        **short,
        "two dimensional": CompiledPauliAxis(a[None, :], b[None, :], kap),
        "empty": CompiledPauliAxis(np.zeros(0, np.uint64), np.zeros(0, np.uint64), 0),
        "too long": CompiledPauliAxis(np.append(a, a), np.append(b, b), kap),
        "b only": CompiledPauliAxis(a, b[None, :], kap),
    }


@pytest.mark.parametrize("n", [5, 70])
@pytest.mark.parametrize("where", ["p0", "r_axis"])
def test_a_malformed_axis_shape_raises_what_the_reference_raises_and_changes_nothing(
    n: int, where: str
) -> None:
    rng = np.random.default_rng(n)
    for name, bad in _bad_axes(n, rng).items():
        fac = _list_with(bad) if where == "p0" else _list_with(None, (bad,))
        states = [random_state(rng, n, 4, "plain") for _ in range(3)]
        exc_o, exc_e, oracle, got = _both(states, [fac], [0, 0, 0], False, 1e-10, n)
        assert exc_o is not None, name
        _same(exc_o, exc_e, oracle, got)
        for i in range(3):
            assert _first_difference(states[i], got[i]) is None, (name, i)


def test_a_malformed_axis_on_a_shot_the_list_never_reaches_is_ignored_like_the_reference() -> None:
    n, rng = 70, np.random.default_rng(1)
    bad = _bad_axes(n, rng)["length one"]
    assert bad.a_words.shape == (1,)
    states = [random_state(rng, n, 4, "plain") for _ in range(2)]
    exc_o, exc_e, oracle, got = _both(states, [_list_with(bad)], [-1, -1], False, 1e-10, n)
    assert exc_o is None and exc_e is None
    for i in range(2):
        assert _first_difference(oracle[i], got[i]) is None


@pytest.mark.parametrize("kappa", [np.array([1, 2]), "x", None])
def test_an_axis_whose_kappa_is_not_an_integer_scalar_is_not_encoded(kappa: Any) -> None:
    n, rng = 5, np.random.default_rng(2)
    a, b, _ = _nonzero_axis(rng, n)
    fac = _list_with(None, (CompiledPauliAxis(a, b, kappa),))
    with pytest.raises(ValueError, match="uint64 word arrays of shape"):
        factor_table([fac], 1)


def test_a_float_or_missing_word_array_is_refused_like_the_reference() -> None:
    n, rng = 5, np.random.default_rng(3)
    a, b, kap = _nonzero_axis(rng, n)
    float_axis = CompiledPauliAxis(a.astype(np.float64), b, kap)  # type: ignore[arg-type]
    for bad in (float_axis, CompiledPauliAxis(None, b, kap)):  # type: ignore[arg-type]
        states = [random_state(rng, n, 4, "plain")]
        lists = [_list_with(None, (bad,))]
        exc_o, exc_e, oracle, got = _both(states, lists, [0], False, 1e-10, n)
        assert exc_o is not None
        _same(exc_o, exc_e, oracle, got)


def test_a_non_integer_v_takes_the_reference_error_and_changes_nothing() -> None:
    n, rng = 5, np.random.default_rng(4)
    states = [random_state(rng, n, 4, "plain") for _ in range(2)]
    fac = _list_with(CompiledPauliAxis(*_nonzero_axis(rng, n)), v=3.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError) as ref:  # first shot: the reference stops at the omega multiply
        apply_clifford_factors(states[0].copy(), fac, check_unitary=True, atol=1e-10)
    store = LiveExpansions(2, n, 1e-10)
    for b, s in enumerate(states):
        store[b] = s
    with pytest.raises(TypeError) as eng:
        store.inject_factor_lists([fac], np.zeros(2, np.int32), False)
    assert str(eng.value) == str(ref.value)
    assert store.failed_shot == 0
    for b in range(2):
        assert _first_difference(states[b], store[b]) is None
    store.close()


@pytest.mark.parametrize("v", [0, 8, -8, 3, -5, 15])
def test_well_formed_lists_still_run_in_the_engine_and_match_the_reference(
    v: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    n, rng = 70, np.random.default_rng(5)
    states = [random_state(rng, n, 4, "plain") for _ in range(3)]
    fac = _list_with(CompiledPauliAxis(*_nonzero_axis(rng, n)), (), v=v)
    exp = [apply_clifford_factors(s.copy(), fac, check_unitary=True, atol=1e-10) for s in states]
    store = LiveExpansions(3, n, 1e-10)
    for b, s in enumerate(states):
        store[b] = s
    calls: list[int] = []
    real = store.engine.inject_cliffords

    def counting(*args: Any) -> Any:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(store.engine, "inject_cliffords", counting)
    store.inject_factor_lists([fac], np.zeros(3, np.int32), False)
    assert calls == [1]  # the native command, not the per-shot fallback
    for i in range(3):
        assert _first_difference(exp[i], store[i]) is None
    store.close()


# ---- F: non-finite values anywhere in the weights ---------------------------------------------


@pytest.mark.parametrize("where", ["p0", "r_axis"])
@pytest.mark.parametrize("pos", ["first", "middle", "last"])
@pytest.mark.parametrize("part", ["real", "imag"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_weights_are_found_at_every_position_and_in_both_parts(
    where: str, pos: str, part: str, bad: float
) -> None:
    n, rng = 5, np.random.default_rng(6)
    st = random_state(rng, n, 8, "plain")
    assert st.chi >= 4
    k = {"first": 0, "middle": st.chi // 2, "last": st.chi - 1}[pos]
    w = st.weights.copy()
    w[k] = complex(bad, 0.5) if part == "real" else complex(0.5, bad)
    st.weights = w
    ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
    fac = _list_with(ax) if where == "p0" else _list_with(None, (ax,))
    exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
    assert isinstance(exc_o, FloatingPointError) and "Non-finite weights" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)


# ---- F: padding bits in the `b` word only -----------------------------------------------------


@pytest.mark.parametrize("where", ["p0", "r_axis"])
@pytest.mark.parametrize("inverse", [False, True])
def test_padding_bits_in_the_b_word_only_raise_the_reference_value_error(
    where: str, inverse: bool
) -> None:
    n, rng = 5, np.random.default_rng(7)
    a = np.array([1], np.uint64)
    bad = CompiledPauliAxis(a, np.array([1 << 40], np.uint64), 0)
    fac = _list_with(bad) if where == "p0" else _list_with(None, (bad,))
    states = [random_state(rng, n, 4, "plain")]
    exc_o, exc_e, oracle, got = _both(states, [fac], [0], inverse, 1e-10, n)
    assert isinstance(exc_o, ValueError) and "padding" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)


# ---- F: unitarity tolerance -------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 0.3, 0.05])
def test_a_tolerance_exactly_equal_to_the_norm_change_passes_like_the_reference(
    scale: float,
) -> None:
    """|post - pre| == atol * max(1, pre) is allowed (`<=`): the tie is where `<` would flip."""
    rng = np.random.default_rng(int(scale * 100))
    ties = 0
    for _ in range(30):
        st, fac, delta, pre = _rounding_case(rng, scale)
        if pre < 1.0 and delta != 0.0:  # then max(1, pre) == 1, so atol == delta is exactly the tie
            exc_o, exc_e, _o, _g = _both([st], [fac], [0], False, delta, 5)
            assert exc_o is None and exc_e is None, (exc_o, exc_e)
            ties += 1
    assert ties >= 5 or scale == 1.0


def test_a_tolerance_exactly_equal_to_the_norm_change_passes_for_pre_above_one() -> None:
    rng = np.random.default_rng(8)
    ties = 0
    for _ in range(60):
        st, fac, delta, pre = _rounding_case(rng, 3.0)
        if pre > 1.0:
            atol = delta / pre  # the scaled edge: |post - pre| == atol * pre
            for tol, fails in ((atol, False), (np.nextafter(atol, 0.0) * 0.5, True)):
                exc_o, exc_e, _o, _g = _both([st], [fac], [0], False, tol, 5)
                assert (exc_o is not None) == (exc_e is not None), (tol, exc_o, exc_e)
                assert (exc_o is not None) is fails or tol == atol
            ties += 1
    assert ties >= 10


def test_the_tolerance_scales_with_the_pre_norm_not_the_post_norm() -> None:
    """For pre != post the two scalings differ: the verdicts must be the reference's (pre)."""
    rng = np.random.default_rng(9)
    checked = 0
    for _ in range(80):
        st, fac, delta, pre = _rounding_case(rng, 8.0)
        post = apply_clifford_factors(st.copy(), fac, check_unitary=False, atol=1.0).norm_sq()
        if pre <= 1.0 or post == pre:
            continue
        for atol in (delta / pre, delta / post, np.nextafter(delta / max(pre, post), 0.0)):
            exc_o, exc_e, _o, _g = _both([st], [fac], [0], False, atol, 5)
            assert (exc_o is None) == (exc_e is None), (atol, exc_o, exc_e)
        checked += 1
    assert checked >= 10


# ---- F: K1 merge tolerance and the chi > 2 condition ------------------------------------------


def _cancelling_state(n: int, rel: float, extra_rows: int = 1) -> pq.ShotExpansion:
    """Non-canonical rows 0 and 1 share a key and cancel to `rel` of their size."""
    st = pq.ShotExpansion.zero(n, atol=1e-10)
    nw = st.n_words
    w = 0.5 + 0.25j
    rows = [w, -w * (1 - rel)] + [0.7071 * (k + 1) for k in range(extra_rows)]
    st.weights = np.array(rows, dtype=np.complex128)
    st.h_words = np.zeros((len(rows), nw), dtype=np.uint64)
    for k in range(extra_rows):
        st.h_words[2 + k, 0] = k + 1  # distinct keys
    st.ell = np.zeros((len(rows), 0), dtype=np.uint8)
    st._is_canonical_merged = False
    return st


def _x_z_axis(n: int) -> CompiledPauliAxis:
    nw = pq.n_words_for_qubits(n)
    a, b = np.zeros(nw, np.uint64), np.zeros(nw, np.uint64)
    a[0], b[0] = 1, 2  # X0 Z1: Hermitian with kappa 0
    return CompiledPauliAxis(a, b, 0)


def test_the_merge_cancellation_tolerance_is_the_one_phase_qfe_uses() -> None:
    """Rows cancelling to just below / above K1_CANCEL_REL_TOL: engine and reference agree, and
    the verdict really flips at the constant (so a different C++ constant is caught)."""
    n = 3
    fac = _list_with(None, (_x_z_axis(n),))
    tol = pq.K1_CANCEL_REL_TOL
    chis = set()
    for rel in (tol * 0.1, tol * 0.5, tol * 0.9, tol * 1.1, tol * 2.0, tol * 8.0, tol * 12.0):
        st = _cancelling_state(n, rel)
        ref = apply_clifford_factors(st.copy(), fac, check_unitary=False, atol=1.0)
        store = LiveExpansions(1, n, 1.0)
        store[0] = st
        store.inject_factor_lists([fac], np.zeros(1, np.int32), False)
        assert _first_difference(ref, store[0]) is None, rel
        chis.add((rel > tol, ref.chi))
        store.close()
    assert len({c for _, c in chis}) == 2  # both outcomes occur: the boundary is exercised
    assert {c for below, c in chis if not below} != {c for below, c in chis if below}


@pytest.mark.parametrize("extra_rows", [0, 1, 2])
@pytest.mark.parametrize("canonical", [False, True])
def test_the_pre_norm_merge_happens_only_for_non_canonical_states_with_more_than_one_row(
    extra_rows: int, canonical: bool
) -> None:
    """chi = 2 + extra rows (2, 3, 4), canonical or not: unitarity verdicts equal the reference's
    (a merge threshold of `chi > 2` would skip the merge at chi == 2 and raise spuriously)."""
    n = 3
    fac = _list_with(None, (_x_z_axis(n),))
    for rel in (1.0, 0.5, 1e-14):
        st = _cancelling_state(n, rel, extra_rows)
        st._is_canonical_merged = canonical
        exc_o, exc_e, oracle, got = _both([st], [fac], [0], False, 1e-10, n)
        assert (exc_o is None) == (exc_e is None), (rel, exc_o, exc_e)
        if exc_o is None:
            assert _first_difference(oracle[0], got[0]) is None, rel
        else:
            _same(exc_o, exc_e, oracle, got)


# ---- F/O04: the Hermitian message carries kappa -----------------------------------------------


@pytest.mark.parametrize("kappa", [1, 2, 3])
def test_the_hermitian_error_message_reports_the_axis_kappa(kappa: int) -> None:
    n, rng = 5, np.random.default_rng(10)
    a = pq.pack_bit_vector(np.array([1, 0, 0, 0, 0], dtype=np.uint8), n)
    par = int(pq.word_dot_parity(a, a))  # 1: <a, b> with b == a
    bad_kappa = kappa if (kappa - par) & 1 else (kappa + 1) & 3  # of the wrong parity
    fac = _list_with(None, (CompiledPauliAxis(a, a, bad_kappa),))
    states = [random_state(rng, n, 2, "plain")]
    exc_o, exc_e, oracle, got = _both(states, [fac], [0], False, 1e-10, n)
    assert isinstance(exc_o, ValueError) and f"kappa={bad_kappa}" in str(exc_o)
    _same(exc_o, exc_e, oracle, got)
