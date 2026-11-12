"""`_plain_bits` gates the engine postselect on the shape of `desired_bits` (WP-25 S5b, finding n9).

The reference `postselect_z_batch` reads `desired_bits[b]` shot by shot and skips a shot whose
expansion is empty (`chi == 0`) without reading it. The engine path reads every entry up front,
so it may only be taken for a 1-d array of exactly one entry per shot. A `(1, B)` or `(1, 1, B)`
array has `size == B` but is not that: it must take the reference loop, where a `chi == 0` shot
never reads it (no error, nothing changes) and the first live shot raises IndexError on it.

Nothing is hardcoded: each case runs on the reference (a plain list) and on the engine store and
compares results, exception type AND text, every shot, masses and random streams.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_born_edges_test import _both, _dead, _t_state
from stimside._magic.live_store_s5b_gate_test import _same

_B = 3


class _Hooked(np.ndarray):
    """A 1-d ndarray subclass: its `__getitem__` hook is what the reference loop would run shot
    by shot, and an up-front engine read would skip (so it must take the reference loop)."""

    def __getitem__(self, item: Any) -> Any:
        return super().__getitem__(item)


class _Spy:
    """Counts the engine `postselect` commands issued by the stores it is installed on."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = LiveExpansions.postselect
        outer = self

        def postselect(store: Any, *a: Any, **k: Any) -> Any:
            outer.calls += 1
            return real(store, *a, **k)

        monkeypatch.setattr(LiveExpansions, "postselect", postselect)


def test_plain_bits_accepts_exactly_a_1d_integer_array_with_one_entry_per_shot() -> None:
    sim = _both([_dead()] * _B)[1]
    for good in (np.zeros(_B, np.uint8), np.zeros(_B, bool), np.zeros(_B, np.int64)):
        assert sim._plain_bits(good), good.dtype
    for bad in (
        np.zeros((1, _B), np.uint8),  # size == B, but 2-d
        np.zeros((1, 1, _B), np.uint8),  # size == B, but 3-d
        np.zeros((_B, 1), np.uint8),
        np.zeros((_B + 1,), np.uint8),
        np.zeros((_B - 1,), np.uint8),
        np.zeros((), np.uint8),
        np.zeros(_B, np.float64),
        np.zeros(_B, np.uint8).view(np.matrix),  # an ndarray subclass (2-d: the shape rejects it)
        np.zeros(_B, np.uint8).view(_Hooked),  # 1-d, right dtype and shape: only `type(a) is`
        [0] * _B,
    ):
        assert not sim._plain_bits(bad), (type(bad), getattr(bad, "shape", None))


@pytest.mark.parametrize("shape", [(1, _B), (1, 1, _B)])
def test_wrong_rank_desired_bits_is_never_read_for_chi0_shots_on_both_stores(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All shots empty: the reference reads nothing, so there is no error and no change. An
    engine path taken for `size == B` would read `desired_bits[0]` (a row) and raise TypeError."""
    spy = _Spy(monkeypatch)
    pair = _both([_dead() for _ in range(_B)])
    bits = np.zeros(shape, np.uint8)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(_B, bool), bits, "Z")
    assert exc is None
    assert spy.calls == 0  # wrong rank: the reference loop ran on the engine store too
    assert not np.any(pair[1]._log_branch_masses)


@pytest.mark.parametrize("shape", [(1, _B), (1, 1, _B)])
def test_wrong_rank_desired_bits_fails_at_the_first_live_shot_like_the_reference(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shot 0 is empty (skipped, never read); shot 1 is live and `desired_bits[1]` does not
    exist for an array whose first axis has length 1: IndexError, on both stores, same text."""
    spy = _Spy(monkeypatch)
    pair = _both([_dead(), _t_state(), _dead()])
    bits = np.zeros(shape, np.uint8)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(_B, bool), bits, "Z")
    assert isinstance(exc, IndexError), exc
    assert spy.calls == 0
    assert not np.any(pair[1]._log_branch_masses)  # nothing was committed before the failure


def test_a_1d_desired_bits_array_takes_the_engine_path_with_chi0_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the right shape is accepted, runs as one engine command and equals the
    reference (empty shots untouched). Guards the tests above against a gate that never opens."""
    spy = _Spy(monkeypatch)
    pair = _both([_dead(), _t_state(), _dead()])
    bits = np.array([1, 0, 1], np.uint8)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(_B, bool), bits, "Z")
    assert exc is None
    assert spy.calls == 1
    masses = pair[1]._log_branch_masses
    assert masses[0] == 0.0 and masses[2] == 0.0 and masses[1] != 0.0


def test_a_1d_ndarray_subclass_never_takes_the_engine_path_and_its_hook_runs_per_live_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behaviour, not just `_plain_bits`: a subclass with the right dtype and shape (1-d, `_B`
    entries) goes through the reference loop on both stores: no engine command, and the hook
    is called for the live shot only (a chi == 0 shot is skipped without being read)."""
    seen: list[int] = []

    class _Counting(np.ndarray):
        def __getitem__(self, item: Any) -> Any:
            seen.append(int(item))
            return super().__getitem__(item)

    spy = _Spy(monkeypatch)
    pair = _both([_dead(), _t_state(), _dead()])
    bits = np.array([1, 0, 1], np.uint8).view(_Counting)
    exc = _same(pair, "postselect_z_batch", 0, np.ones(_B, bool), bits, "Z")
    assert exc is None
    assert spy.calls == 0
    assert seen and set(seen) == {1}  # shot 1 only, on each of the two stores' reference loops
