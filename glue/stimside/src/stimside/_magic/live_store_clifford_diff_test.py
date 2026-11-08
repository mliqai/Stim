"""Bit-exact differential tests of `mbe_inject_cliffords` against `apply_clifford_factors`.

Oracle: today's per-shot `channels.apply_clifford_factors(check_unitary=True)` applied shot by
shot (what `_apply_factor_list` loops over). Subject: `LiveExpansions.inject_factor_lists`
(C++ `MagicBatchEngine`). Nothing here uses engine code to build an expected value. Everything is
compared as bytes (weights through a uint64 view, so -0.0 != 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import CompiledCliffordFactorList, apply_clifford_factors
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_test import (
    _CELLS,
    _KINDS,
    _first_difference,
    _nonzero_axis,
    random_state,
)

_ATOL = 1e-10


def random_factor_list(rng: np.random.Generator, n: int) -> CompiledCliffordFactorList:
    """n_r in 0..3 quarter-turn axes, v in 0..7 (also negative), p0 present or not."""
    def ax() -> CompiledPauliAxis:
        a, b, kap = _nonzero_axis(rng, n, density=float(rng.choice([0.1, 0.3, 0.8])))
        return CompiledPauliAxis(a, b, kap)

    n_r = int(rng.integers(0, 4))
    p0 = ax() if rng.random() < 0.6 else None
    return CompiledCliffordFactorList(
        int(rng.integers(-9, 17)) if rng.random() < 0.9 else 0, p0,
        tuple(ax() for _ in range(n_r)),
    )


@dataclass
class Coverage:
    cases: int = 0
    inverse: int = 0
    v_odd: int = 0
    v_zero: int = 0
    p0: int = 0
    n_r: dict[int, int] = field(default_factory=dict)
    d_positive: int = 0
    non_canonical: int = 0
    chi_zero: int = 0
    signed_zero: int = 0
    chi_max: int = 0
    d_grew: int = 0


def _oracle(exps: list[pq.ShotExpansion], lists, which, inverse: bool) -> None:
    for s, w in enumerate(which):
        if w >= 0:
            exps[s] = apply_clifford_factors(
                exps[s], lists[w], inverse=inverse, check_unitary=True, atol=_ATOL
            )


def run_cell(n: int, chi: int, seed: int, cov: Coverage, steps: int = 4) -> int:
    rng = np.random.default_rng([seed, n, chi, 4])
    batch = len(_KINDS) * 6
    kinds = [_KINDS[i % len(_KINDS)] for i in rng.permutation(batch)]
    states = [random_state(rng, n, chi, k) for k in kinds]
    oracle = [x.copy() for x in states]
    store = LiveExpansions(batch, n, _ATOL)
    for b, st in enumerate(states):
        store[b] = st
    compared = 0
    for step in range(steps):
        lists = [random_factor_list(rng, n) for _ in range(int(rng.integers(1, 4)))]
        density = float(rng.choice([0.0, 0.05, 0.5, 0.95, 1.0]))
        which = np.where(rng.random(batch) < density, rng.integers(0, len(lists), batch), -1)
        inverse = bool(rng.integers(0, 2))
        for shot, w in enumerate(which):  # one case = one (shot, factor list) application
            if w >= 0:
                cov.cases += 1
                cov.non_canonical += not oracle[shot]._is_canonical_merged
        _oracle(oracle, lists, which, inverse)
        store.inject_factor_lists(lists, which.astype(np.int32), inverse)
        for i in range(batch):
            diff = _first_difference(oracle[i], store[i])
            assert diff is None, f"n={n} chi={chi} step={step} shot={i} inverse={inverse}: {diff}"
            compared += 1
        for ln in lists:
            cov.inverse += inverse
            cov.v_odd += bool(ln.v & 1)
            cov.v_zero += (ln.v == 0)
            cov.p0 += ln.p0 is not None
            cov.n_r[len(ln.r_axes)] = cov.n_r.get(len(ln.r_axes), 0) + 1
    for fin, init in zip(oracle, states, strict=True):
        cov.d_positive += fin.d > 0
        cov.d_grew += fin.d > init.d
        cov.chi_zero += fin.chi == 0
        cov.chi_max = max(cov.chi_max, fin.chi)
        w = fin.weights
        cov.signed_zero += bool(np.any(np.signbit(w.real) | np.signbit(w.imag)))
    store.close()
    return compared


def test_inject_cliffords_is_bit_identical_to_the_per_shot_function_on_many_cases() -> None:
    cov, compared = Coverage(), 0
    for n, chi in _CELLS:
        for seed in range(8):
            compared += run_cell(n, chi, seed, cov)
    assert compared >= 10_000, compared  # (shot, command) states compared bytewise
    assert cov.cases >= 2000, cov.cases  # (shot, factor list) applications
    assert cov.inverse >= 200 and cov.v_odd >= 200 and cov.v_zero >= 20 and cov.p0 >= 200
    assert all(cov.n_r.get(k, 0) >= 60 for k in range(4)), cov.n_r
    assert cov.d_positive >= 500 and cov.d_grew >= 300 and cov.non_canonical >= 100
    assert cov.chi_zero >= 100 and cov.signed_zero >= 200 and cov.chi_max >= 64, cov


@pytest.mark.parametrize("inverse", [False, True])
def test_every_omega_power_matches_numpy_in_both_directions(inverse: bool) -> None:
    rng = np.random.default_rng(5)
    for n, chi in ((5, 1), (5, 4), (70, 64)):
        st = [random_state(rng, n, chi, k) for k in ("plain", "signed0", "raw", "d")] * 2
        for v in range(-9, 17):
            lists = [CompiledCliffordFactorList(v, None, ())]
            oracle = [s.copy() for s in st]
            store = LiveExpansions(len(st), n, _ATOL)
            for b, s in enumerate(st):
                store[b] = s
            which = np.zeros(len(st), dtype=np.int32)
            _oracle(oracle, lists, which, inverse)
            store.inject_factor_lists(lists, which, inverse)
            for i in range(len(st)):
                assert _first_difference(oracle[i], store[i]) is None, (n, chi, v, i)
            store.close()


def test_a_chi_one_shot_takes_numpys_unfused_in_place_path_nowhere() -> None:
    """The omega multiply is `weights * OMEGA` (out of place, FMA form), also for chi == 1."""
    rng = np.random.default_rng(9)
    st = [random_state(rng, 5, 1, "plain") for _ in range(40)]
    for v in range(1, 8):
        lists = [CompiledCliffordFactorList(v, None, ())]
        oracle = [s.copy() for s in st]
        store = LiveExpansions(len(st), 5, _ATOL)
        for b, s in enumerate(st):
            store[b] = s
        _oracle(oracle, lists, np.zeros(len(st), dtype=int), False)
        store.inject_factor_lists(lists, np.zeros(len(st), dtype=np.int32), False)
        assert all(_first_difference(oracle[i], store[i]) is None for i in range(len(st))), v
        store.close()


def test_a_table_with_no_axes_at_all_is_applied_like_the_reference() -> None:
    """Only v multiplies (or nothing): the axis table is empty, so its data pointer is null and
    the offset of every list is one past it (UBSan once saw a reference formed there)."""
    n, rng = 5, np.random.default_rng(21)
    states = [random_state(rng, n, 4, "plain") for _ in range(4)]
    lists = [CompiledCliffordFactorList(3, None, ()), CompiledCliffordFactorList(0, None, ())]
    for inverse in (False, True):
        which = np.array([0, 1, -1, 0], dtype=np.int32)
        oracle = [x.copy() for x in states]
        _oracle(oracle, lists, which, inverse)
        store = LiveExpansions(4, n, _ATOL)
        for b, st in enumerate(states):
            store[b] = st
        store.inject_factor_lists(lists, which, inverse)
        for i in range(4):
            assert _first_difference(oracle[i], store[i]) is None, (inverse, i)
        store.close()
