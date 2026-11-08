"""Sensitivity tests added after the second WP-25 S5b gate: engine peek / unmatched reset.

Each test runs the same call on the REFERENCE (a plain list of `ShotExpansion`s) and on the ENGINE
store and compares bytes (results, state, branch masses, random streams, exception type and text);
no expected value is written by hand. They pin:

* the cast of an integer `reset_one_mask` (values outside {0, 1} never equal a sampled bit: a
  cast to bool or to int8 changes which shots are kicked);
* a `reset_one_mask` whose shape is not the one of `reset_zero_mask | reset_one_mask`: the reference
  reads `one[q, b]` shot by shot, so an `IndexError` leaves the earlier shots projected and the
  generator advanced; the engine path must leave exactly the same state (it reads the mask up front
  only for a full-shape mask);
* `peek_pauli_born_batch` really runs on the engine store;
* the upper clip `min(1, .)` of a peek (a projected state whose `z1 / z` is 1 + 1 ulp);
* the zero floor of a postselection (`cz <= K * z`, inclusive, at exact equality).
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_born_edges_test import _both, _masks, _same
from stimside._magic.live_store_measure_edges_test import (
    _N,
    _t_state,
    _two_branch_state,
    _z,
)
from stimside._magic.live_store_s5b_gate_test import _seeded_t_state


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Free the simulators' reference cycles (native engines) so live-engine counts stay exact."""
    yield
    gc.collect()


def _states() -> list[pq.ShotExpansion]:
    return [_t_state(), _t_state(_N, 3), _t_state(_N, 4), _t_state(_N, 2)]


@pytest.mark.parametrize(
    ("val", "dt"),
    [(2, np.int8), (-1, np.int8), (3, np.uint8), (2**62, np.int64), (2**63 + 1, np.uint64),
     (256, np.int64), (-(2**63), np.int64)],
)
@pytest.mark.parametrize("sync", [False, True])
def test_integer_reset_one_mask_values_outside_0_1_match_the_reference(
    val: int, dt: type, sync: bool
) -> None:
    pair = _both(_states(), sync=sync)
    zero = _masks(4, {0: [0, 1, 2, 3]})
    one = np.zeros(zero.shape, np.dtype(dt))
    one[0, :] = np.array([val, 0, val, 1], dtype=dt)  # != m_b for both m_b in {0, 1} at shots 0, 2
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is None


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("val", [2, -1])
def test_int8_reset_one_mask_2_and_minus1_never_equal_a_sampled_bit_over_many_shots(
    seed: int, val: int
) -> None:
    states = [_t_state(_N, 1 + (i + seed) % 5) for i in range(16)]
    pair = _both(states, seed=seed)
    zero = _masks(16, {0: list(range(16))})
    one = np.zeros(zero.shape, np.int8)
    one[0, :] = val
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is None


@pytest.mark.parametrize("dt", [np.bool_, np.int64])
@pytest.mark.parametrize("shape", [(1, 4), (_N, 1), (1, 1), (4,), ()])
@pytest.mark.parametrize("sync", [False, True])
def test_wrong_shape_reset_one_mask_fails_like_the_reference_with_the_same_partial_state(
    shape: tuple[int, ...], sync: bool, dt: type
) -> None:
    """`zero | one` broadcasts but `one[q, shots]` is out of range: the reference has projected the
    first shot by then, so the engine path must not read the mask before touching any shot."""
    pair = _both(_states(), sync=sync)
    zero = _masks(4, {1: [0, 1, 2, 3]})  # qubit 1: `one[1, shots]` is out of range for every shape
    exc = _same(pair, "do_unmatched_noisy_reset_z", zero, np.zeros(shape, dt), "Z", [1])
    assert exc is not None


@pytest.mark.parametrize("shape", [(1, 4), (_N, 1), (1, 1)])
def test_wrong_shape_reset_one_mask_that_happens_to_be_in_range_is_accepted_like_the_reference(
    shape: tuple[int, ...],
) -> None:
    pair = _both(_states())
    zero = _masks(4, {0: [0, 1, 2, 3] if shape[1] == 4 else [0]})  # shots inside the mask's width
    one = np.zeros(shape, bool)
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is None


def test_peek_born_batch_really_uses_the_engine_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing pin: `peek_pauli_born_batch` on the engine store calls `LiveExpansions.peek`."""
    pair = _both(_states())
    calls: list[int] = []
    real = LiveExpansions.peek

    def spy(self: LiveExpansions, *a: object, **k: object) -> object:
        calls.append(1)
        return real(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(LiveExpansions, "peek", spy)
    pair[1].peek_pauli_born_batch([0], "Z")
    assert calls
    assert _same(pair, "peek_pauli_born_batch", [0], "Z") is None


# (seed, k, qubit): projecting `_seeded_t_state(seed, k)` onto outcome 1 of Z(qubit) gives a state
# whose raw `z1 / z_in` is 1 + 2**-52; the reference clips it to exactly 1.0.
_OVERSHOOT = [(0, 7, 1), (0, 7, 3), (0, 8, 4)]


@pytest.mark.parametrize(("seed", "k", "q"), _OVERSHOOT)
def test_peek_of_a_projected_state_is_clipped_to_one_like_the_reference(
    seed: int, k: int, q: int
) -> None:
    ax = _z(q)
    post, _ = pq.project_diagonal(_seeded_t_state(seed, k), ax.b_words, 0, 1)
    z1 = pq.project_diagonal(post, ax.b_words, 0, 1)[1]
    assert z1 / post.norm_sq() > 1.0  # vacuity guard: unclipped, the ratio overshoots
    pair = _both([post, _t_state()])
    assert _same(pair, "peek_pauli_born_batch", [q], "Z") is None
    assert pair[0].peek_pauli_born_batch([q], "Z")[0, 0] == 1.0


# (big, small): amplitudes `small |0> + big |1>` whose z0 is EXACTLY `K1_CANCEL_REL_TOL * z`
# (found by an ulp scan around the ideal ratio); the zero-floor test is `cz <= K * z`.
_ON_THE_FLOOR = [(0.4888506996347092, 1.5458816466126006e-07),
                 (0.31156934486997034, 9.85268778875654e-08)]


@pytest.mark.parametrize(("big", "small"), _ON_THE_FLOOR)
def test_postselecting_a_mass_exactly_on_the_zero_floor_raises_like_the_reference(
    big: float, small: float
) -> None:
    ax = _z(0)
    state = _two_branch_state(np.array([small, big], np.complex128))
    _, z0, _, _, z, _, _ = pq.born_split_expansion(state, ax.a_words, ax.b_words, 0, atol=1e-9)
    assert z0 == pq.K1_CANCEL_REL_TOL * z  # vacuity guard: exactly on the floor
    pair = _both([state])
    exc = _same(pair, "postselect_z_batch", 0, np.array([True]), np.array([0], np.uint8), "Z")
    assert exc is not None  # the reference refuses a postselection onto a mass at the floor
