"""Fixes after the WP-25 S5a gate (iteration 1): int32 table columns and the split token.

F1. `_as_i32` (the int32 table fields: `BlockTable.step_*`, `FactorTable.n_r/v_fwd/v_inv/
    axis_start`, `ShotArrays.pivots`) refuses everything but an integer that fits an int32. Before,
    a float column was truncated by numpy (NaN / inf / 1e30 became INT_MIN, 1.9 became 1) and run.
F2. `measure_finish(outcome, chosen, split)` is refused when `split` is not the split the engine
    holds: split A, split B, finish(A's outcomes) used to commit B's branches with A's masses.
"""

from __future__ import annotations

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import MeasurementRecord
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_measure_diff_test import _pair
from stimside._magic.live_store_measure_edges_test import (
    _before,
    _engine_with,
    _t_state,
    _x,
    _z,
)
from stimside._native import magic_engine as me

_N = 5

# ---------------------------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------------------------


def _i32(arr: object) -> list[int]:
    return me._as_i32(arr, (len(arr),), "x").tolist()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1.0]), np.array([1.9]), np.array([-0.5]), np.array([np.nan]),
        np.array([np.inf]), np.array([-np.inf]), np.array([1e30]), np.array([2.0**33]),
        np.array([3e9], np.float32), np.array([1 + 0j]), np.array([True]),
        np.array([2**40], dtype=object), np.array(["1"]), [1.0], [True, False],
    ],
    ids=lambda a: repr(a)[:30],
)
def test_f1_a_non_integer_column_is_refused_not_truncated_to_int32(arr: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        _i32(arr)


@pytest.mark.parametrize(
    "arr", [np.array([2**31], np.int64), np.array([-(2**31) - 1], np.int64),
            np.array([2**32 + 1], np.uint64), np.array([2**31], np.uint32), [2**40]],
)
def test_f1_an_integer_that_does_not_fit_an_int32_is_still_refused(arr: object) -> None:
    with pytest.raises(ValueError, match="int32"):
        _i32(arr)


def test_f1_integers_that_fit_and_empty_arrays_are_accepted() -> None:
    assert _i32(np.array([-(2**31), 0, 2**31 - 1], np.int64)) == [-(2**31), 0, 2**31 - 1]
    assert _i32(np.array([0, 7], np.uint8)) == [0, 7]
    assert _i32(np.array([3, -1], np.int32)) == [3, -1]
    assert _i32([5, 6]) == [5, 6]
    assert me._as_i32(np.zeros(0), (0,), "x").shape == (0,)  # `np.asarray([])` is float64
    assert me._as_i32([], (0,), "x").dtype == np.int32


def _factor_table(**over: object) -> me.FactorTable:
    nw = 1
    a, b = np.zeros((1, nw), np.uint64), np.zeros((1, nw), np.uint64)
    a[0, 0] = 2
    cols = {
        "has_p0": np.array([0], np.uint8), "n_r": np.array([1], np.int32),
        "v_fwd": np.array([-1], np.int32), "v_inv": np.array([-1], np.int32),
        "axis_start": np.array([0], np.int32), "axes_a": a, "axes_b": b,
        "axes_kappa": np.array([0], np.uint8),
    }
    cols.update(over)
    return me.FactorTable(**cols)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["n_r", "v_fwd", "v_inv", "axis_start"])
@pytest.mark.parametrize("value", [float(2**32 + 1), float("nan"), 1.9, -0.5, 1e30])
def test_f1_a_float_factor_table_column_is_refused_and_no_shot_changes(
    field: str, value: float
) -> None:
    rng = np.random.default_rng(1)
    from stimside._magic.live_store_test import random_state

    store = _engine_with([random_state(rng, _N, 4, "d") for _ in range(3)])
    before = _before(store)
    which = np.array([0, -1, 0], np.int32)
    with pytest.raises(TypeError, match="integer"):
        store.engine.inject_cliffords(_factor_table(**{field: np.array([value])}), which, False)
    assert _before(store) == before
    store.close()


def test_f1_a_float_pivots_column_is_refused_on_import() -> None:
    from stimside._magic.live_store_test import random_state

    store = _engine_with([random_state(np.random.default_rng(2), _N, 4, "d")])
    s = store.engine.export_shot(0)
    assert s.d > 0  # there are pivots to corrupt
    for bad in (np.full(s.d, np.nan), np.full(s.d, 1.0), np.full(s.d, 2.0**33)):
        with pytest.raises(TypeError, match="integer"):
            store.engine.import_shot(0, s._replace(pivots=bad))
    store.engine.import_shot(0, s._replace(pivots=s.pivots.astype(np.int64)))  # integers still pass
    store.close()


# ---------------------------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------------------------


def _outcomes(res: me.SplitResult) -> tuple[np.ndarray, np.ndarray]:
    out = np.arange(res.n_active) % 2 == 1
    z01 = res.z01[res.cls == me.CLS_ACTIVE]
    return out, np.where(out, z01[:, 1], z01[:, 0])


def _norms(store: LiveExpansions) -> list[float]:
    out = []
    for b in range(store.batch_size):
        w = store.engine.export_shot(b).weights
        out.append(float(np.sum(np.abs(w) ** 2)))
    return out


def _two_splits(store: LiveExpansions) -> tuple[me.SplitResult, me.SplitResult]:
    a, b = _x(0), _x(1)
    first = store.measure_split((a.a_words, a.b_words, a.kappa), True, None)
    second = store.measure_split((b.a_words, b.b_words, b.kappa), True, None)
    assert first.n_active == second.n_active  # the same sizes: only the token tells them apart
    return first, second


def test_f2_finishing_a_replaced_split_is_refused_and_changes_nothing() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3), _t_state(_N, 4)])
    before = _before(store)
    first, second = _two_splits(store)
    out, chosen = _outcomes(first)
    with pytest.raises(RuntimeError, match="no pending"):
        store.measure_finish(out, chosen, first)
    assert _before(store) == before  # nothing was committed (before the fix: B's branches were)
    out2, chosen2 = _outcomes(second)  # the replacing split is still pending and finishes normally
    assert store.measure_finish(out2, chosen2, second).failure is None
    assert _before(store) != before
    store.close()


def test_f2_the_committed_state_after_a_refused_finish_is_the_state_of_a_clean_run() -> None:
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 4)]
    store = _engine_with(states)
    first, second = _two_splits(store)
    with pytest.raises(RuntimeError):
        store.measure_finish(*_outcomes(first), first)
    store.measure_finish(*_outcomes(second), second)
    clean = _engine_with(states)
    b = _x(1)
    only = clean.measure_split((b.a_words, b.b_words, b.kappa), True, None)
    clean.measure_finish(*_outcomes(only), only)
    assert _before(store) == _before(clean) and _norms(store) == _norms(clean)
    store.close()
    clean.close()


def test_f2_a_split_can_be_finished_only_once_and_not_after_an_empty_split() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    a = _x(0)
    first = store.measure_split((a.a_words, a.b_words, a.kappa), False, None)
    store.measure_finish(*_outcomes(first), first)
    after = _before(store)
    with pytest.raises(RuntimeError, match="no pending"):  # used up
        store.measure_finish(*_outcomes(first), first)
    third = store.measure_split((a.a_words, a.b_words, a.kappa), False, None)
    z = _z(0)  # a split with no active shot left behind: a Z measurement of the shots
    empty = _engine_with([pq.ShotExpansion.zero(_N)] * 2)
    none = empty.measure_split((z.a_words, z.b_words, z.kappa), False, None)
    assert none.n_active == 0 and none.token != third.token
    assert _before(store) == after
    with pytest.raises(RuntimeError, match="no pending"):  # `none` was never pending
        empty.measure_finish(np.zeros(0, bool), np.zeros(0), none)
    store.close()
    empty.close()


def test_f2_split_tokens_are_unique_and_a_foreign_engines_token_is_refused() -> None:
    s1 = _engine_with([_t_state(), _t_state(_N, 3)])
    s2 = _engine_with([_t_state(), _t_state(_N, 3)])
    a = _x(0)
    r1 = s1.measure_split((a.a_words, a.b_words, a.kappa), False, None)
    r2 = s2.measure_split((a.a_words, a.b_words, a.kappa), False, None)
    assert r1.token != r2.token
    before = _before(s2)
    with pytest.raises(RuntimeError, match="no pending"):
        s2.measure_finish(*_outcomes(r1), r1)  # engine 1's split on engine 2
    assert _before(s2) == before
    assert s2.measure_finish(*_outcomes(r2), r2).failure is None
    s1.close()
    s2.close()


def test_f2_the_simulator_refuses_to_commit_outcomes_drawn_from_a_replaced_split() -> None:
    """Between the simulator's split and its finish another split replaces the pending one (what
    a second thread on the same engine could do): the substep raises, no shot is committed and no
    branch mass is counted."""
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 4)]
    _ref, eng, _rc, _ec = _pair(states, _N, 1, False)
    store = eng._live_exps
    assert isinstance(store, LiveExpansions)
    before, masses = _before(store), eng._log_branch_masses.copy()
    other = _x(1)
    real = store.measure_finish

    def finish_after_another_split(outcome, chosen, split=None):  # type: ignore[no-untyped-def]
        store.measure_split((other.a_words, other.b_words, other.kappa), True, None)
        return real(outcome, chosen, split)

    store.measure_finish = finish_after_another_split  # type: ignore[method-assign]
    rec = MeasurementRecord(0, 0.0, False, _x(0), _x(0))
    with pytest.raises(RuntimeError, match="no pending"):
        eng._project_substep_batch(rec)
    assert _before(store) == before
    assert eng._log_branch_masses.tobytes() == masses.tobytes()
