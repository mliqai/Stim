"""Strength tests for the engine `peek` / `postselect` / unmatched reset (WP-25 S5b).

Each test here exists because a mutant of the C++ or of the simulator wiring survived the
differential and edge tests (see the S5b handoff); every one is a boundary the random cases do
not hit: values exactly ON the 1e-12 snap, a mass exactly ON the zero floor, a conservation
deviation exactly equal to `atol * max(1, z)`, a failure of the measuring half after a draw, two
failures competing for the error, and the raw engine contract of `mbe_postselect` /
`mbe_unmatched_apply` (flags, padding, pending split). Expected values come from the reference
list store (today's per-shot code); the engine is compared with it as bytes.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_born_edges_test import _call, _masks, _same
from stimside._magic.live_store_measure_diff_test import _assert_same, _pair
from stimside._magic.live_store_measure_edges_test import (
    _ATOL,
    _N,
    _engine_with,
    _t_state,
    _two_branch_state,
    _x,
    _z,
)
from stimside._magic.live_store_test import _nonzero_axis
from stimside._native import magic_engine as me

_K = pq.K1_CANCEL_REL_TOL


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Simulators and their spies form reference cycles that hold a native engine until the
    cyclic gc runs; free them here so the live-engine counts other tests assert on stay exact."""
    yield
    gc.collect()



def _both(states: list[pq.ShotExpansion], atol: float = _ATOL, seed: int = 4, sync: bool = False):
    return _pair(states, _N, seed, sync, atol)


# --------------------------------------------------------------------------------------------
# States ON a boundary
# --------------------------------------------------------------------------------------------


def _peek_of(w: np.ndarray) -> float:
    ref = _both([_two_branch_state(w)])[0]
    return float(ref.peek_pauli_born_batch([0], "Z")[0, 0])


def _state_with_p1(target: float) -> pq.ShotExpansion:
    """A two-branch state whose peek of Z0 is EXACTLY `target`: the likelier branch gets a
    pseudo-random weight, the other one is scanned ulp by ulp around the ideal ratio (the set of
    reachable floats near `target` is sparse, so several big weights are tried)."""
    small_is_one = target < 0.5  # p1 is the probability of the |1> branch
    ratio = math.sqrt(target / (1.0 - target) if small_is_one else (1.0 - target) / target)
    for big in np.random.default_rng(0).uniform(0.3, 1.0, 600):
        base = float(big) * ratio
        for i in range(-40, 40):
            s = base + i * math.ulp(base)
            w = np.array([big, s] if small_is_one else [s, big], np.complex128)
            if _peek_of(w) == target:
                return _two_branch_state(w)
    raise AssertionError(f"no state with peek {target!r}")


@pytest.mark.parametrize("target", [1e-12, 1.0 - 1e-12])
def test_peek_values_exactly_on_the_snap_boundary_are_pinned_not_drawn(target: float) -> None:
    """`pb <= 1e-12` and `pb >= 1 - 1e-12` are inclusive: no draw there, the state is kept."""
    state = _state_with_p1(target)
    assert _peek_of(state.weights) == target
    for n_shots in (1, 3):
        pair = _both([state] * n_shots)
        rz, ro = _masks(n_shots, {0: list(range(n_shots))}), _masks(n_shots, {0: [0]})
        before = pair[1].np_rng.bit_generator.state
        assert _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0]) is None
        assert pair[1].np_rng.bit_generator.state == before  # nothing was drawn


@pytest.mark.parametrize("outward", [False, True])
@pytest.mark.parametrize("target", [1e-12, 1.0 - 1e-12])
def test_a_peek_one_ulp_inside_the_snap_boundary_is_drawn(target: float, outward: bool) -> None:
    """One ulp towards 1/2 is drawn (the reference consumes one `random()` per shot), one ulp
    away from it is pinned (nothing consumed); the engine does exactly the same."""
    toward_half = 1.0 if target < 0.5 else 0.0
    value = float(np.nextafter(target, 1.0 - toward_half if outward else toward_half))
    pair = _both([_state_with_p1(value)] * 3)
    rz, ro = _masks(3, {0: [0, 1, 2]}), _masks(3, {0: [1]})
    before = pair[0].np_rng.bit_generator.state
    _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    drew = pair[0].np_rng.bit_generator.state != before
    assert drew == (not outward)  # the reference drew iff the value is strictly inside the snap
    assert pair[1].np_rng.bit_generator.state == pair[0].np_rng.bit_generator.state


class _Fixed(np.random.Generator):
    """`random(n)` returns the given values (the generator state is otherwise real)."""

    def __init__(self, values: list[float]) -> None:
        super().__init__(np.random.PCG64(0))
        self.values = values

    def random(self, n: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """A scalar draw (the reference) or `random(k)` (the engine path): one value per shot."""
        if n is None:
            assert len(self.values) == 1
            return self.values[0]
        assert n == len(self.values), (n, self.values)
        return np.array(self.values)


@pytest.mark.parametrize("below", [True, False])
def test_a_draw_exactly_equal_to_p1_is_not_below_it(below: bool) -> None:
    """`u < p1`, not `u <= p1`: with u == p1 the outcome is 0, one ulp lower it is 1."""
    state = _two_branch_state(np.array([math.sqrt(0.7), 1j * math.sqrt(0.3)], np.complex128))
    pair = _both([state])
    p1 = float(pair[0].peek_pauli_born_batch([0], "Z")[0, 0])
    u = float(np.nextafter(p1, 0.0)) if below else p1
    for sim in pair[:2]:
        sim.np_rng = _Fixed([u])
    rz, ro = _masks(1, {0: [0]}), _masks(1, {0: []})
    assert _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0]) is None
    # outcome 1 keeps the branch of weight i * sqrt(.3) (renormalised to ~ i), outcome 0 the
    # real one
    kept = pair[1]._live_exps[0].weights
    assert (abs(kept[0].imag) > 0.5) == below


# --------------------------------------------------------------------------------------------
# A mass exactly ON the zero floor, a deviation exactly equal to the tolerance
# --------------------------------------------------------------------------------------------


def _floor_state() -> tuple[pq.ShotExpansion, float]:
    """z0 == K1 * z EXACTLY (floats found by search): requested bit 0 is refused, 1 is fine."""
    rng = np.random.default_rng(1)
    for _ in range(200000):
        z = float(rng.uniform(0.5, 1.0))
        a = _K * z
        w0 = math.sqrt(a)
        b = z - a
        w1 = math.sqrt(b)
        if w0 * w0 == a and w1 * w1 == b and _K * (a + b) == a:
            return _two_branch_state(np.array([w0, w1], np.complex128)), a
    raise AssertionError("no exact floor state")


def test_a_requested_outcome_with_a_mass_exactly_on_the_zero_floor_is_refused() -> None:
    state, z0 = _floor_state()
    assert state.norm_sq() * _K == z0  # exactly on the floor
    pair = _both([state, state])
    exc = _same(pair, "postselect_z_batch", 0, np.array([1, 0], bool), np.array([0, 0], np.uint8))
    assert isinstance(exc, pq.ImpossiblePostselectionError)  # `chosen_z <= floor`, not `<`
    assert _same(_both([state]), "postselect_z_batch", 0, np.array([1], bool),
                 np.array([1], np.uint8)) is None  # the other outcome has the mass


def _deviating_state() -> tuple[pq.ShotExpansion, float, float]:
    """A state with z0 + z1 != z by one ulp-ish: (state, deviation, z), z < 1."""
    ax = _z(0)
    for k in range(2, 12):
        for seed in range(60):
            r = np.random.default_rng([k, seed])
            s = pq.ShotExpansion.zero(_N)
            for _ in range(k):
                s = pq.apply_pauli_t(s, *_nonzero_axis(r, _N))
            s.weights = s.weights * 0.5
            _, z0, _, z1, z, *_ = pq.born_split_expansion(
                s, ax.a_words, ax.b_words, ax.kappa, atol=1.0
            )
            if abs((z0 + z1) - z) > 0.0 and z < 1.0 and z0 > 0.0:
                return s, abs((z0 + z1) - z), z
    raise AssertionError("no deviating state")


def test_the_conservation_check_is_inclusive_and_scaled_by_max_one_z() -> None:
    state, dev, z = _deviating_state()
    assert z < 1.0
    edge = float(np.nextafter(dev, 0.0))
    # deviation == atol * 1: accepted (<=); one ulp below the deviation: refused
    assert _same(_both([state], atol=dev), "postselect_z_batch", 0, np.array([1], bool),
                 np.array([0], np.uint8), label="at tolerance") is None
    refused = _same(_both([state], atol=edge), "postselect_z_batch", 0, np.array([1], bool),
                    np.array([0], np.uint8), label="below tolerance")
    assert isinstance(refused, FloatingPointError) and "Born mass conservation" in str(refused)
    # scale is max(1, z), not z: atol = 2 * dev passes although 2 * dev * z < dev
    assert z * 2 * dev < dev
    assert _same(_both([state], atol=2 * dev), "postselect_z_batch", 0, np.array([1], bool),
                 np.array([0], np.uint8), label="scale") is None


# --------------------------------------------------------------------------------------------
# A failure of the measuring half AFTER a draw, and two failures competing for the error
# --------------------------------------------------------------------------------------------


def test_the_draws_stop_at_the_shot_whose_projection_failed() -> None:
    """Peek has no conservation check, the projection has: with a tiny atol every shot draws,
    then the first projection fails. The reference drew exactly once; so must the engine."""
    state, _dev, _z0 = _deviating_state()
    pair = _both([state] * 5, atol=1e-300)
    peeked = pair[0].peek_pauli_born_batch([0], "Z")[0]
    assert np.all((peeked > 1e-12) & (peeked < 1 - 1e-12))  # every shot would draw
    rz, ro = _masks(5, {0: [0, 1, 2, 3, 4]}), _masks(5, {0: [1, 3]})
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    assert isinstance(exc, FloatingPointError) and "Born mass conservation" in str(exc)


def _nan_zero() -> pq.ShotExpansion:
    e = pq.ShotExpansion.zero(_N)
    e.weights = e.weights * complex("nan")
    return e


def test_a_trivial_shot_whose_kick_fails_stops_the_reset_and_cuts_the_draws() -> None:
    """A chi = 1, d = 0 shot is peeked as a plain bit (its weights are never looked at), so a
    non-finite weight shows only when the kick is applied: the draws of the shots after it must
    not have happened."""
    states = [_t_state(), _t_state(_N, 3), _nan_zero(), _t_state(), _t_state(_N, 3)]
    pair = _both(states)
    rz, ro = _masks(5, {0: [0, 1, 3, 4]}), _masks(5, {0: [2]})  # shot 2: bit 0 != reset_one 1
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    assert isinstance(exc, FloatingPointError) and "Non-finite weights in apply_pauli" in str(exc)


def test_the_error_of_the_earlier_shot_wins_over_a_later_failing_peek() -> None:
    """Shot 0 fails when kicked (non-finite trivial shot), shot 3 fails to peek (non-finite
    weights, chi > 1): the reference reaches shot 0's kick first."""
    bad_peek = _t_state()
    bad_peek.weights = bad_peek.weights * complex("nan")
    states = [_nan_zero(), _t_state(), _t_state(_N, 3), bad_peek]
    pair = _both(states)
    rz, ro = _masks(4, {0: [1, 2, 3]}), _masks(4, {0: [0]})
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    assert isinstance(exc, FloatingPointError) and "apply_pauli" in str(exc)


def test_a_failing_peek_applies_the_shots_before_it_first() -> None:
    bad_peek = _t_state()
    bad_peek.weights = bad_peek.weights * complex("nan")
    pair = _both([_t_state(), _t_state(_N, 3), bad_peek, _t_state()])
    rz, ro = _masks(4, {0: [0, 1, 2, 3]}), _masks(4, {0: [1]})
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    assert isinstance(exc, FloatingPointError)


# --------------------------------------------------------------------------------------------
# The raw engine contract
# --------------------------------------------------------------------------------------------

_ALL = np.arange(4, dtype=np.int32)


def _states() -> list[pq.ShotExpansion]:
    dead = pq.ShotExpansion.from_components((), n_qubits=_N, merge=False)
    return [_t_state(), dead, _t_state(_N, 3), pq.ShotExpansion.zero(_N)]


def _t(ax: Any) -> tuple[Any, Any, int]:
    return ax.a_words, ax.b_words, ax.kappa


def test_postselect_reports_done_only_for_shots_that_were_projected() -> None:
    store = _engine_with(_states())
    res = store.postselect(_t(_x(1)), _ALL, np.array([0, 0, 0, 0], np.uint8))
    assert res.failure is None
    assert res.done.tolist() == [1, 0, 1, 1]  # the dead shot is skipped, not done
    assert not res.z01[1].any()  # and has no masses


def test_postselect_desired_uses_only_the_lowest_bit_at_the_engine_level() -> None:
    ref, store = _engine_with(_states()), _engine_with(_states())
    want = ref.postselect(_t(_x(1)), _ALL, np.array([0, 1, 0, 1], np.uint8))
    got = store.postselect(_t(_x(1)), _ALL, np.array([2, 3, 4, 255], np.uint8))
    assert got.z01.tobytes() == want.z01.tobytes() and got.done.tobytes() == want.done.tobytes()
    for b in range(4):
        assert ref.engine.export_shot(b).weights.tobytes() == (
            store.engine.export_shot(b).weights.tobytes()
        )


def test_postselect_refuses_a_non_hermitian_observable_at_the_first_live_shot() -> None:
    store = _engine_with(_states())
    ax = _z(0)
    res = store.postselect((ax.a_words, ax.b_words, 1), _ALL, np.zeros(4, np.uint8))  # kappa odd
    assert res.failure is not None and res.failure.kind == me.FAIL_NOT_HERMITIAN
    assert res.failure.shot == 0 and not res.done.any()


def test_unmatched_apply_does_not_look_at_the_kick_of_a_dead_shot() -> None:
    store = _engine_with(_states())
    bad_kick = _x(0)
    bad_kick.a_words[:] = np.uint64(1) << np.uint64(_N)  # padding bit: a kick on a live shot fails
    res = store.unmatched_apply(_t(_z(0)), _t(bad_kick), np.array([1], np.int32), [0], [0], [1])
    assert res.failure is None  # the shot is dead: the reference skips the kick (`chi > 0`)
    res = store.unmatched_apply(_t(_z(0)), _t(bad_kick), np.array([0], np.int32), [0], [0], [1])
    assert res.failure is not None and res.failure.kind == me.FAIL_PADDING


def test_unmatched_apply_reports_phase_2_for_a_kick_and_phase_1_for_a_projection() -> None:
    store = _engine_with(_states())
    bad_kick = _x(0)
    bad_kick.a_words[:] = np.uint64(1) << np.uint64(_N)
    res = store.unmatched_apply(_t(_z(0)), _t(bad_kick), np.array([0], np.int32), [0], [0], [1])
    assert (res.failure is not None, res.phase) == (True, 2)  # the kick failed
    ax = _z(0)
    res = store.unmatched_apply(
        (ax.a_words, ax.b_words, 1), _t(_x(1)), np.array([0], np.int32), [0], [1], [0]
    )
    assert (res.failure is not None, res.phase) == (True, 1)  # the projection failed


def test_the_sync_unmatched_reset_matches_one_shot_at_a_time_including_a_failure() -> None:
    states = [_t_state(), _nan_zero(), _t_state(_N, 3), _t_state()]
    pair = _both(states, sync=True)
    rz, ro = _masks(4, {0: [0, 1, 2, 3]}), _masks(4, {0: [1]})
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
    assert isinstance(exc, FloatingPointError)
    assert isinstance(pair[1]._live_exps, LiveExpansions)


def test_an_engine_postselect_error_leaves_the_same_masses_as_the_reference() -> None:
    state, z0 = _floor_state()
    states = [_t_state(), _t_state(_N, 3), state, _t_state()]
    pair = _both(states)
    want = np.array([0, 1, 0, 0], np.uint8)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(4, bool), want)
    assert isinstance(exc, pq.ImpossiblePostselectionError)
    assert pair[1]._log_branch_masses[0] != 0.0 and pair[1]._log_branch_masses[3] == 0.0
    assert z0 > 0.0


# --------------------------------------------------------------------------------------------
# Exception texts of failures the engine cannot produce from a real state
# --------------------------------------------------------------------------------------------
# Projected masses are sums of squares, so a negative or non-finite mass cannot come out of a
# state; the mapping of those failures is checked against the reference's own text by feeding the
# reference (`born_split_expansion`) the same masses through a patched projection.


def _reference_text(z_raw: tuple[float, float], z: float, atol: float) -> tuple[type, str]:
    state = _t_state()
    masses = iter(z_raw)
    real = pq.project_diagonal

    def fake(exp: Any, b: Any, sigma: int = 0, outcome: int = 0) -> Any:
        out, _ = real(exp, b, sigma, outcome)
        return out, next(masses)

    ax = _z(0)
    pq.project_diagonal = fake  # type: ignore[assignment]
    try:
        with pytest.raises(FloatingPointError) as info:
            pq.born_split_expansion(state, ax.a_words, ax.b_words, ax.kappa, atol=atol, z_in=z)
    finally:
        pq.project_diagonal = real  # type: ignore[assignment]
    return type(info.value), str(info.value)


@pytest.mark.parametrize(
    ("kind", "z_raw", "z", "pre", "post"),
    [
        ("NEGATIVE", (-1e-9, 1.0), 1.0, -1e-9, 1.0),
        ("NEGATIVE", (0.5, -2e-12), 1.5, 0.5, -2e-12),
        ("NON_FINITE", (float("nan"), 1.0), 1.0, float("nan"), 1.0),
        ("NON_FINITE", (0.5, float("inf")), 1.0, 0.5, float("inf")),
    ],
)
def test_the_text_of_a_negative_or_non_finite_mass_is_the_references(
    kind: str, z_raw: tuple[float, float], z: float, pre: float, post: float
) -> None:
    from stimside._magic.live_store import raise_born_failure

    code = me.FAIL_NEGATIVE_MASS if kind == "NEGATIVE" else me.FAIL_NON_FINITE_MASS
    fail = me.FactorFailure(0, code, 6, 0, 0, pre, post, z)
    want_type, want_text = _reference_text(z_raw, z, _ATOL)
    with pytest.raises(want_type) as info:
        raise_born_failure(fail, _N, _ATOL)
    assert str(info.value) == want_text


def test_the_text_of_a_conservation_failure_is_the_references() -> None:
    from stimside._magic.live_store import raise_born_failure

    fail = me.FactorFailure(0, me.FAIL_CONSERVATION, 6, 0, 0, 0.25, 0.5, 1.0)
    want_type, want_text = _reference_text((0.25, 0.5), 1.0, 1e-3)
    with pytest.raises(want_type) as info:
        raise_born_failure(fail, _N, 1e-3)
    assert str(info.value) == want_text


# --------------------------------------------------------------------------------------------
# Masks that name a shot outside the batch
# --------------------------------------------------------------------------------------------


def _index_error_on_both(fn: str, *args: Any) -> None:
    """Both stores raise IndexError (the text is the store's: a list says 'list index out of
    range', the engine proxy 'shot 3 out of range ...', as for every other access) and the
    shots before the missing one are handled alike."""
    ref, eng, rc, ec = _both([_t_state(), _t_state(_N, 3)])
    _, want = _call(ref, fn, *args)
    _, got = _call(eng, fn, *args)
    assert isinstance(want, IndexError) and isinstance(got, IndexError), (want, got)
    _assert_same(ref, eng, fn, rc, ec)


def test_a_postselect_mask_longer_than_the_batch_raises_index_error_on_both_stores() -> None:
    _index_error_on_both(
        "postselect_z_batch", 0, np.array([1, 0, 0, 1], bool), np.zeros(4, np.uint8)
    )  # shot 3 does not exist: the engine path must not answer with another exception type


def test_an_unmatched_reset_mask_longer_than_the_batch_raises_index_error_on_both_stores() -> None:
    _index_error_on_both(
        "do_unmatched_noisy_reset_z", _masks(4, {0: [0, 3]}), _masks(4, {0: [1]}), "Z", [0]
    )
