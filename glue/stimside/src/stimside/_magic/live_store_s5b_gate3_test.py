"""Sensitivity tests added after the third WP-25 S5b gate: routing of `do_unmatched_noisy_reset_z`.

Each behavioural test runs the same call on the REFERENCE (a plain list of `ShotExpansion`s) and
on the ENGINE store and compares bytes (results, state, branch masses, random streams, exception
type and text); no expected value is written by hand. They pin:

* masks that are not 2-D but have equal shapes (`zero | one` keeps that shape, so the shape
  guard alone cannot tell them apart): the reference reads `one[q, b]` shot by shot, after the
  earlier shots are projected, so the engine path (which reads `one[q, shots]` up front) must
  not be taken. This holds with and without sync mode;
* a full-shape 2-D mask still runs on the engine store, in both modes: the reset reaches
  `LiveExpansions.peek` and `LiveExpansions.unmatched_apply` (a routing spy; the reference store
  is a plain list and never calls them).
"""

from __future__ import annotations

import gc
from collections.abc import Iterator

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_born_edges_test import _both, _masks, _same
from stimside._magic.live_store_measure_edges_test import _N, _t_state

_B = 4


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Free the simulators' reference cycles (native engines) so live-engine counts stay exact."""
    yield
    gc.collect()


def _states() -> list[pq.ShotExpansion]:
    return [_t_state(), _t_state(_N, 3), _t_state(_N, 4), _t_state(_N, 2)]


@pytest.mark.parametrize("dt", [np.bool_, np.int64])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("shape", [(_B,), (_N, _B, 1), (1, _N, _B), ()])
def test_equal_shape_masks_that_are_not_2d_fail_like_the_reference(
    shape: tuple[int, ...], sync: bool, dt: type
) -> None:
    """`zero` and `one` share `shape`, so `zero | one` has it too; but `one[q, shots]` is not a
    (shots,) read of a 2-D mask: the engine path must leave the reference's partial state."""
    pair = _both(_states(), sync=sync)
    zero = np.ones(shape, np.bool_)
    one: np.ndarray = np.zeros(shape, dt)
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [1, 2]) is not None


@pytest.mark.parametrize("pauli", ["X", "Z"])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("shape", [(1, _N, _B), (_N, _B, 1)])
def test_equal_shape_3d_masks_with_set_rows_match_the_reference(
    shape: tuple[int, ...], sync: bool, pauli: str
) -> None:
    """Same, with only some shots set and a mixed one-mask: whatever the reference does (raise
    after projecting the first shot), the engine must do."""
    pair = _both(_states(), sync=sync)
    zero = np.zeros(shape, np.bool_)
    zero.reshape(-1, _B)[1] = [True, False, True, True]
    one = np.zeros(shape, np.int64)
    one.reshape(-1, _B)[1] = [0, 1, 1, 0]
    _same(pair, "do_unmatched_noisy_reset_z", zero, one, pauli, [1])


@pytest.mark.parametrize("dt", [np.bool_, np.int64])
@pytest.mark.parametrize("sync", [False, True])
def test_one_mask_of_equal_size_but_lower_rank_fails_like_the_reference(
    sync: bool, dt: type
) -> None:
    """`zero` is (1, B) and `one` is (B,): `zero | one` is (1, B), 2-D, with as many elements as
    `one`, yet `one[0, shots]` is a 2-index read of a 1-D array. Only the shape test sees it."""
    pair = _both(_states(), sync=sync)
    zero = np.ones((1, _B), np.bool_)
    one: np.ndarray = np.zeros((_B,), dt)
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is not None


@pytest.mark.parametrize("sync", [False, True])
def test_full_shape_2d_masks_reach_the_engine_store(
    sync: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing pin: the reference is a list store, so every call counted here is the engine's."""
    calls = {"peek": 0, "unmatched_apply": 0}
    for name in calls:
        real = getattr(LiveExpansions, name)

        def spy(self: LiveExpansions, *a: object, _r: object = real, _k: str = name,
                **k: object) -> object:
            calls[_k] += 1
            return _r(self, *a, **k)  # type: ignore[operator]

        monkeypatch.setattr(LiveExpansions, name, spy)
    pair = _both(_states(), sync=sync)
    zero = _masks(_B, {1: [0, 1, 2, 3], 2: [0, 2]})
    one = _masks(_B, {1: [1, 3]})
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [1, 2]) is None
    assert calls["peek"] >= 1
    assert calls["unmatched_apply"] >= 1
