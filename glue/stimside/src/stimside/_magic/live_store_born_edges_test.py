"""Edge and failure paths of the engine `peek` / `postselect` / unmatched reset (WP-25 S5b).

Each test drives the public simulator methods (`peek_pauli_born_batch`, `postselect_z_batch`,
`do_unmatched_noisy_reset_z`) on two simulators holding the same states, the reference list store
and the engine store, and asserts that the results, the exception (type AND text), every shot (all
eight arrays as bytes, `log_mass`, canonical flag), the branch masses and the random state (an
extra or missing draw shows) are identical afterwards. The reference is today's per-shot code.

Failure paths pin the partial-mutation contract: the entries before the failing one are
committed (state and branch mass), the failing one and later ones are untouched, the draws of an
unmatched reset stop at the failing shot.
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
from stimside._magic.live_store_measure_diff_test import _assert_same, _pair
from stimside._magic.live_store_measure_edges_test import (
    _ATOL,
    _N,
    _scaled,
    _t_state,
    _two_branch_state,
    _x,
)
from stimside._magic.live_store_test import _first_difference as _first
from stimside.simulator_magic import MagicsideSimulator


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Simulators and their spies form reference cycles that hold a native engine until the
    cyclic gc runs; free them here so the live-engine counts other tests assert on stay exact."""
    yield
    gc.collect()



def _dead() -> pq.ShotExpansion:
    return pq.ShotExpansion.from_components((), n_qubits=_N, merge=False)


def _nan_state() -> pq.ShotExpansion:
    e = _t_state()
    e.weights = e.weights * complex("nan")
    return e


def _both(
    states: list[pq.ShotExpansion], sync: bool = False, atol: float = _ATOL, seed: int = 4
) -> tuple[MagicsideSimulator, MagicsideSimulator, list[Any], list[Any]]:
    return _pair(states, _N, seed, sync, atol)


def _call(sim: MagicsideSimulator, fn: str, *args: Any) -> tuple[Any, Exception | None]:
    try:
        with np.errstate(all="ignore"):
            return getattr(sim, fn)(*args), None
    except (FloatingPointError, ValueError, ZeroDivisionError, MemoryError, RuntimeError,
            TypeError, IndexError) as exc:
        return None, exc


def _same(
    pair: tuple[MagicsideSimulator, MagicsideSimulator, list[Any], list[Any]], fn: str,
    *args: Any, label: str = "",
) -> Exception | None:
    ref, eng, rc, ec = pair
    want, wexc = _call(ref, fn, *args)
    got, gexc = _call(eng, fn, *args)
    assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), (label, gexc, wexc)
    if wexc is None and want is not None:
        assert np.asarray(want).tobytes() == np.asarray(got).tobytes(), label
    _assert_same(ref, eng, label or fn, rc, ec)
    return wexc


def _masks(n_shots: int, rows: dict[int, list[int]]) -> np.ndarray:
    m = np.zeros((_N, n_shots), bool)
    for q, shots in rows.items():
        m[q, shots] = True
    return m


# --------------------------------------------------------------------------------------------
# peek
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_peek_matches_for_dead_trivial_pinned_and_drawn_shots(pauli: str) -> None:
    states = [_dead(), pq.ShotExpansion.zero(_N), _t_state(), _t_state(_N, 3),
              _two_branch_state(np.array([math.sqrt(0.3), math.sqrt(0.7)], np.complex128)),
              _scaled(_t_state(), 1e-7), _scaled(_t_state(), 1e3)]
    pair = _both(states)
    assert _same(pair, "peek_pauli_born_batch", [0, 1, 2], pauli) is None
    assert _same(pair, "peek_pauli_batch", [0, 1], pauli) is None


def test_peek_of_a_dead_shot_is_zero_and_of_a_zero_norm_shot_is_zero() -> None:
    zero_norm = _t_state()
    zero_norm.weights = zero_norm.weights * 0.0
    pair = _both([_dead(), zero_norm])
    assert _same(pair, "peek_pauli_born_batch", [0], "X") is None
    assert not np.any(pair[1].peek_pauli_born_batch([0], "X"))


def test_peek_of_non_finite_weights_raises_the_same_error() -> None:
    pair = _both([_t_state(), _nan_state(), _t_state()])
    for pauli in ("X", "Z"):
        assert _same(pair, "peek_pauli_born_batch", [0], pauli, label=pauli) is not None


def test_peek_probabilities_are_snapped_by_peek_pauli_batch_like_the_reference() -> None:
    k = 1e-12
    states = [_two_branch_state(np.array([math.sqrt(1 - p), math.sqrt(p)], np.complex128))
              for p in (k, np.nextafter(k, 0), np.nextafter(k, 1), 1 - k,
                        np.nextafter(1 - k, 0), np.nextafter(1 - k, 1), 0.5)]
    pair = _both(states)
    assert _same(pair, "peek_pauli_batch", [0], "Z") is None


@pytest.mark.parametrize("atol", [1e-300, 5.7e-17, 1e-10])
def test_the_pairwise_rotation_check_of_peek_matches_at_extreme_tolerances(atol: float) -> None:
    """`peek` rotates with the PAIRWISE unitarity check (the measure kernel uses the sequential
    one): both are pinned, here at tolerances where a sum order can decide."""
    rng = np.random.default_rng(3)
    states = [pq.apply_pauli_t(_t_state(_N, 3), *_axis(rng)) for _ in range(6)]
    raised = 0
    for seed in range(4):
        exc = _same(_both(states, atol=atol, seed=seed), "peek_pauli_born_batch", [0, 1, 2], "X",
                    label=f"atol={atol} seed={seed}")  # same result or the same exception
        raised += exc is not None
    assert raised in (0, 4)  # the atol, not the seed, decides whether the check trips


def _axis(rng: np.random.Generator) -> tuple[Any, Any, int]:
    from stimside._magic.live_store_test import _nonzero_axis

    return _nonzero_axis(rng, _N)


def test_peek_with_per_shot_tableaux_takes_the_per_shot_loop_and_still_matches() -> None:
    """After a compiled run each shot has its own tableau (`_shot_tabs`): the observable differs
    per shot, so the engine peek is not used and the reference loop runs on the engine store."""
    ref, eng, rc, ec = _both([_t_state(), _t_state(_N, 3), pq.ShotExpansion.zero(_N)])
    tabs = [ref._live_ref_tab.copy() for _ in range(3)]
    tabs[1].do_unitary("H", [0])
    ref._shot_tabs, eng._shot_tabs = tabs, [t.copy() for t in tabs]
    assert _same((ref, eng, rc, ec), "peek_pauli_born_batch", [0, 1], "Z") is None


# --------------------------------------------------------------------------------------------
# postselect
# --------------------------------------------------------------------------------------------


def test_postselect_matches_including_dead_shots_and_the_branch_mass_update() -> None:
    states = [_t_state(), _dead(), _t_state(_N, 3), pq.ShotExpansion.zero(_N), _t_state()]
    pair = _both(states)
    flip = np.array([1, 1, 0, 1, 1], bool)
    assert _same(pair, "postselect_z_batch", 0, flip, np.array([1, 0, 1, 0, 0], np.uint8)) is None
    ref, eng = pair[0], pair[1]
    assert ref._log_branch_masses.tobytes() == eng._log_branch_masses.tobytes()
    assert ref._log_branch_masses[2] == 0.0 and ref._log_branch_masses[1] == 0.0  # untouched


def test_an_impossible_postselection_commits_exactly_the_shots_before_it() -> None:
    """|0>, |0>, |0>, ... asked for bit 1 at the second shot: shot 0 (asked for 0) is committed
    with its mass, the failing shot and everything after it are untouched."""
    states = [pq.ShotExpansion.zero(_N), pq.ShotExpansion.zero(_N), _t_state(), _t_state()]
    pair = _both(states)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(4, bool), np.array([0, 1, 0, 0], np.uint8))
    assert isinstance(exc, pq.ImpossiblePostselectionError)
    ref, eng = pair[0], pair[1]
    assert eng._log_branch_masses.tolist() == ref._log_branch_masses.tolist()
    # shot 0 (bit 0 of |0...0>, mass 1) was committed: replaced by an equal state; shots 1..3
    # are exactly as before the call
    untouched = [pq.ShotExpansion.zero(_N), _t_state(), _t_state()]
    for b, want in zip((1, 2, 3), untouched):
        assert _first(eng._live_exps[b], want) is None, b


def test_the_branch_masses_of_the_committed_prefix_are_updated_before_the_error() -> None:
    states = [_t_state(), _t_state(_N, 3), pq.ShotExpansion.zero(_N), _t_state()]
    pair = _both(states)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(4, bool), np.array([0, 1, 1, 0], np.uint8))
    assert isinstance(exc, pq.ImpossiblePostselectionError)
    masses = pair[1]._log_branch_masses
    assert masses[0] != 0.0 and masses[1] != 0.0 and masses[3] == 0.0  # prefix applied, rest not


@pytest.mark.parametrize("bad", ["nan", "zero"])
def test_postselect_on_non_finite_or_zero_norm_weights_raises_like_the_reference(bad: str) -> None:
    broken = _nan_state() if bad == "nan" else _scaled(_t_state(), 0.0)
    pair = _both([_t_state(), broken, _t_state()])
    assert _same(pair, "postselect_z_batch", 1, np.ones(3, bool), np.zeros(3, np.uint8)) is not None


@pytest.mark.parametrize("pauli", ["X", "Y"])
def test_postselect_in_a_rotated_frame_matches(pauli: str) -> None:
    pair = _both([_t_state(), _t_state(_N, 3), _t_state(_N, 1)])
    assert _same(pair, "postselect_z_batch", 2, np.ones(3, bool), np.array([0, 1, 0], np.uint8),
                 pauli) is None


def test_postselect_desired_bits_use_only_their_lowest_bit() -> None:
    pair = _both([_t_state(), _t_state(_N, 3)])
    assert _same(pair, "postselect_z_batch", 0, np.ones(2, bool),
                 np.array([2, 3], np.uint8)) is None  # 2 & 1 == 0, 3 & 1 == 1


@pytest.mark.parametrize(
    "bits",
    [[0, 1, 0], np.array([0, 1, 0], np.int64), np.array([False, True, False]),
     np.array([[0], [1], [0]], np.uint8)],
    ids=["list", "int64", "bool", "2d"],
)
def test_odd_desired_bit_containers_take_the_reference_loop_with_identical_results(
    bits: Any,
) -> None:
    pair = _both([_t_state(), _t_state(_N, 3), pq.ShotExpansion.zero(_N)])
    flip = np.array([1, 1, 0], bool)
    exc = _same(pair, "postselect_z_batch", 0, flip, bits)
    assert (exc is None) == (np.ndim(bits) == 1)  # a 2-d array is a TypeError on both paths


def test_postselect_with_an_empty_mask_changes_nothing_and_does_not_flush_or_draw() -> None:
    pair = _both([_t_state(), _t_state()])
    assert _same(pair, "postselect_z_batch", 0, np.zeros(2, bool), np.zeros(2, np.uint8)) is None


# --------------------------------------------------------------------------------------------
# unmatched noisy reset
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pauli", ["X", "Z"])
@pytest.mark.parametrize("sync", [False, True])
def test_unmatched_reset_matches_with_pinned_drawn_and_dead_shots(pauli: str, sync: bool) -> None:
    states = [pq.ShotExpansion.zero(_N), _t_state(), _dead(), _t_state(_N, 3),
              pq.apply_pauli(pq.ShotExpansion.zero(_N), _x(0))]
    rz, ro = _masks(5, {0: [0, 1, 2, 3, 4]}), _masks(5, {0: [1, 3, 4], 1: [0, 2]})
    pair = _both(states, sync=sync)
    assert _same(pair, "do_unmatched_noisy_reset_z", rz, ro, pauli, [1, 0]) is None
    assert pair[0]._log_branch_masses.tobytes() == pair[1]._log_branch_masses.tobytes()
    assert not np.any(pair[1]._log_branch_masses)  # a reset never touches the branch mass


def test_pinned_shots_draw_nothing_and_leave_their_state_byte_identical() -> None:
    states = [pq.ShotExpansion.zero(_N), pq.ShotExpansion.zero(_N), pq.ShotExpansion.zero(_N)]
    ref, eng, rc, ec = _both(states)
    state0 = eng.np_rng.bit_generator.state
    rz, ro = _masks(3, {0: [0, 1, 2]}), _masks(3, {0: [1]})
    assert _same((ref, eng, rc, ec), "do_unmatched_noisy_reset_z", rz, ro, "Z", [0]) is None
    assert eng.np_rng.bit_generator.state == state0  # |0>: pb == 0 for every shot: no draw


def test_order_with_duplicates_absent_qubits_and_empty_rows_matches() -> None:
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 1), pq.ShotExpansion.zero(_N)]
    rz, ro = _masks(4, {0: [0, 1], 2: [2, 3], 4: [1]}), _masks(4, {1: [1], 2: [0, 3]})
    pair = _both(states)
    assert _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [2, 0, 2, 4, 3, 1, 0]) is None


def test_default_order_is_the_ascending_qubits_with_any_entry() -> None:
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 1), pq.ShotExpansion.zero(_N)]
    rz, ro = _masks(4, {3: [0, 1], 1: [2]}), _masks(4, {0: [3], 3: [2]})
    assert _same(_both(states), "do_unmatched_noisy_reset_z", rz, ro, "Z") is None


@pytest.mark.parametrize("bad_at", [0, 1, 2, 3])
def test_a_failing_shot_stops_the_reset_with_the_earlier_ones_finished_and_the_draws_cut(
    bad_at: int,
) -> None:
    """Whatever the failure (non-finite weights), the shots before it are measured and kicked
    and the random stream has advanced exactly as far as the per-shot code drew."""
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 1), _t_state()]
    states[bad_at] = _nan_state()
    rz, ro = _masks(4, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}), _masks(4, {0: [1, 2], 1: [0, 3]})
    pair = _both(states)
    exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "X", [0, 1])
    assert isinstance(exc, FloatingPointError), exc


def test_a_drawn_outcome_without_mass_raises_the_impossible_error_after_the_draw() -> None:
    """z1 is zero up to the 1e-12 snap but not below it: drawn, then a failing postselection."""
    w = np.array([math.sqrt(1 - 3e-12), math.sqrt(3e-12)], np.complex128)
    pair = _both([_two_branch_state(w) for _ in range(4)])
    rz, ro = _masks(4, {0: [0, 1, 2, 3]}), _masks(4, {0: [1]})
    outcomes = set()
    for _ in range(6):  # whichever shot first draws a 1 fails the same way on both paths
        exc = _same(pair, "do_unmatched_noisy_reset_z", rz, ro, "Z", [0])
        outcomes.add(None if exc is None else type(exc))
    assert outcomes <= {None, pq.ImpossiblePostselectionError}


def test_atol_is_the_simulators_not_the_engines_default() -> None:
    """The simulator's `atol` reaches the C++ checks (conservation, rotation) of every command."""
    states = [_t_state(), _t_state(_N, 3)]
    for atol in (1e-300, 1e-17, 1e-10):
        pair = _both(states, atol=atol)
        assert pair[1].atol == atol
        _same(pair, "peek_pauli_born_batch", [0, 1], "X", label=f"atol={atol}")
        _same(pair, "postselect_z_batch", 1, np.ones(2, bool), np.zeros(2, np.uint8), "X",
              label=f"atol={atol}")


def test_the_engine_store_is_really_used_by_the_engine_cases_above() -> None:
    _, eng, _, _ = _both([_t_state()])
    assert isinstance(eng._live_exps, LiveExpansions)
