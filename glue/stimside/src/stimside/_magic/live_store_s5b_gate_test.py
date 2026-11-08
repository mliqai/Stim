"""Sensitivity tests added after the WP-25 S5b gate (iteration 1): engine peek / postselect /
unmatched reset vs reference.

The S5b suites let a set of mutants of the C++ or of the simulator wiring survive; each test here
pins one behaviour by running the same call on the REFERENCE (a plain list of `ShotExpansion`s) and
on the ENGINE store and comparing bytes (results, state, branch masses, random streams, exception
type and text). Nothing is hardcoded: the expected result is whatever the reference path does.

* underflow: weights ~1e-162 make `norm_sq()` underflow to 0.0 although the rotation / projection
  recombines the rows into a non-zero mass: `peek` must return 0 before any other check and a
  postselection must raise ImpossiblePostselection (the `z_in <= 0` and `z <= 0` guards);
* clip order: a norm of +inf makes `z1 / z_in` NaN, which `min(1, max(0, .))` turns into 0;
* the finite check of a non-rotating peek looks at EVERY weight;
* raw ABI: a shot-list length just above `kMaxElems / 4` is refused (-2) before anything is read;
* simulator wiring: a mask naming shot `batch_size`, a short `desired_bits`, a sync reset at
  exactly p1 = 1e-12, `oom_at == 0`, a non-integer or hostile mask object;
* pending split: `postselect` and `unmatched_apply` drop it at the Python wrapper too.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_born_edges_test import _both, _masks
from stimside._magic.live_store_born_strength_test import _state_with_p1
from stimside._magic.live_store_measure_diff_test import _assert_same
from stimside._magic.live_store_measure_edges_test import (
    _N,
    _engine_with,
    _t_state,
    _two_branch_state,
    _x,
    _z,
)
from stimside._magic.live_store_test import _nonzero_axis
from stimside._native import magic_engine as me
from stimside._native.magic_engine import addr


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Free the simulators' reference cycles (native engines) so live-engine counts stay exact."""
    yield
    gc.collect()


Pair = tuple[Any, Any, list[Any], list[Any]]


def _run(sim: Any, fn: str, *args: Any) -> tuple[Any, Exception | None]:
    """Like `live_store_born_edges_test._call` but for ANY exception (OverflowError too)."""
    try:
        with np.errstate(all="ignore"):
            return getattr(sim, fn)(*args), None
    except (ArithmeticError, ValueError, MemoryError, RuntimeError, TypeError, IndexError) as exc:
        return None, exc  # ArithmeticError: FloatingPointError, OverflowError, ZeroDivisionError


def _same(pair: Pair, fn: str, *args: Any, label: str = "") -> Exception | None:
    """Run `fn` on both simulators; assert same results, exception (type AND text) and state."""
    return _same_fresh(pair, fn, lambda: args, label=label)


def _same_fresh(pair: Pair, fn: str, make: Callable[[], tuple[Any, ...]], label: str = "") -> Any:
    """As `_same`, with fresh arguments for each simulator (stateful argument objects)."""
    ref, eng, rc, ec = pair
    want, wexc = _run(ref, fn, *make())
    got, gexc = _run(eng, fn, *make())
    assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), (label, gexc, wexc)
    if wexc is None and want is not None:
        assert np.asarray(want).tobytes() == np.asarray(got).tobytes(), label
    _assert_same(ref, eng, label or fn, rc, ec)
    return wexc


def _underflowing(k: int) -> pq.ShotExpansion:
    """A T-state of `k` rotations whose weights are scaled to ~1.5e-162, so `norm_sq()` is 0.0."""
    e = _t_state(_N, k)
    e.weights = e.weights * (1.5e-162 / np.abs(e.weights).max())
    assert e.norm_sq() == 0.0 and e.chi > 0  # the witness really underflows
    return e


# --------------------------------------------------------------------------------------------
# pk_z_in_le / mr_z_le0_dropped: the squares of the weights underflow
# --------------------------------------------------------------------------------------------


def _seeded_t_state(seed: int, k: int) -> pq.ShotExpansion:
    rng = np.random.default_rng(seed)
    e = pq.ShotExpansion.zero(_N)
    for _ in range(k):
        e = pq.apply_pauli_t(e, *_nonzero_axis(rng, _N))
    return e


def _scaled_to(e: pq.ShotExpansion, top: float) -> pq.ShotExpansion:
    out = e.copy()
    out.weights = out.weights * (top / np.abs(out.weights).max())
    return out


# (seed, k, pauli, qubit): states found by scanning seeds 0..39 with k = 3..6 where the
# rotation + projection of a state with norm_sq() == 0.0 recombines rows into a non-zero mass, so
# that a `z_in < 0` guard (instead of `<= 0`) would return that mass instead of 0.
_UNDERFLOW_WITNESSES = [(21, 4, "Y", 2), (21, 6, "X", 2), (28, 5, "Y", 2), (19, 6, "Y", 0)]


@pytest.mark.parametrize(("seed", "k", "pauli", "q"), _UNDERFLOW_WITNESSES)
def test_peek_of_a_state_whose_norm_underflows_is_zero_like_the_reference(
    seed: int, k: int, pauli: str, q: int
) -> None:
    """`_obs_p1` returns 0 as soon as `norm_sq()` is 0.0 (before rotating / projecting); an engine
    that used `z_in < 0` would go on and report the recombined projected mass."""
    small = _scaled_to(_seeded_t_state(seed, k), 1.5e-162)
    assert small.norm_sq() == 0.0 and small.chi >= 4
    pair = _both([small, _t_state()])
    assert _same(pair, "peek_pauli_born_batch", [q], pauli) is None
    assert pair[0].peek_pauli_born_batch([q], pauli)[0, 0] == 0.0
    # vacuity guard: the SAME state with weights big enough not to underflow peeks above 0
    big = _scaled_to(small, 1e-3)
    assert big.norm_sq() > 0.0
    assert _both([big])[0].peek_pauli_born_batch([q], pauli)[0, 0] > 0.0


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_peek_of_the_underflowing_t_states_matches_the_reference_on_every_axis(
    k: int, pauli: str
) -> None:
    pair = _both([_underflowing(k), _t_state()])
    for q in range(_N):
        assert _same(pair, "peek_pauli_born_batch", [q], pauli, label=f"{pauli}{q}") is None


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_postselect_on_a_state_whose_norm_underflows_matches_the_reference(
    k: int, bit: int
) -> None:
    """With the squares underflowing, total z is 0.0: the reference raises (a projected mass
    above 0 does not make the postselection possible); `z <= 0` is part of that test."""
    raised = 0
    for pauli in ("X", "Y", "Z"):
        pair = _both([_underflowing(k)])
        exc = _same(pair, "postselect_z_batch", 0, np.array([True]), np.array([bit], np.uint8),
                    pauli, label=pauli)
        raised += exc is not None
    assert raised  # a witness that no longer raises on the reference would pin nothing


@pytest.mark.parametrize("amp", [1.2e-162, 1.5e-162])
@pytest.mark.parametrize("bit", [0, 1])
def test_postselect_of_an_underflowing_two_branch_x_state_raises_like_the_reference(
    amp: float, bit: int
) -> None:
    state = _two_branch_state(np.array([amp, amp], np.complex128))
    assert state.norm_sq() == 0.0
    exc = _same(_both([state]), "postselect_z_batch", 0, np.array([True]),
                np.array([bit], np.uint8), "X")
    assert exc is not None


# --------------------------------------------------------------------------------------------
# r01 / R15: the clip of the peek
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [[1e200, 1e200], [1e200, 0.5], [1e160, 1e160]])
def test_peek_of_a_state_with_an_infinite_norm_is_the_references_clip(w: list[float]) -> None:
    """`z_in` is +inf with finite weights, `z1 / z_in` is NaN or 0: `min(1, max(0, .))` gives 0,
    `max(0, min(1, .))` would give 1 for NaN. (X and Y rotate first and fail identically on both
    stores with a non-finite norm; only the non-rotating Z peek reaches the clip.)"""
    state = _two_branch_state(np.array(w, np.complex128))
    assert math.isinf(state.norm_sq())
    pair = _both([state, _t_state()])
    for pauli in ("X", "Y"):
        assert _same(pair, "peek_pauli_born_batch", [0], pauli, label=pauli) is not None
    assert _same(pair, "peek_pauli_born_batch", [0], "Z") is None
    assert pair[0].peek_pauli_born_batch([0], "Z")[0, 0] == 0.0


# --------------------------------------------------------------------------------------------
# R08: the non-rotating peek checks every weight
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pos", [1, -1])
@pytest.mark.parametrize("bad", [complex("nan"), complex("inf")])
def test_a_non_finite_weight_anywhere_fails_the_non_rotating_peek_like_the_reference(
    pos: int, bad: complex
) -> None:
    state = _two_branch_state(np.array([0.6 + 0j, 0.8 + 0j]))
    w = state.weights.copy()
    w[pos] = bad
    state.weights = w
    exc = _same(_both([state, _t_state()]), "peek_pauli_born_batch", [0], "Z")
    assert exc is not None


@pytest.mark.parametrize("axis", [_z, _x])
def test_a_peek_that_fails_in_the_merge_reports_the_merge_frames_op(axis: Any) -> None:
    """A non-canonical shot with NaN weights fails in `merge_frames` (before rotating or
    projecting): the raw failure names that op, as the reference's exception does."""
    bad = _t_state(_N, 3)
    bad.weights = bad.weights * complex("nan")
    bad._is_canonical_merged = False
    store = _engine_with([bad, _t_state()])
    ax = axis(0)
    _, fail = store.peek((ax.a_words, ax.b_words, ax.kappa), np.array([0, 1], np.int32))
    assert fail is not None and fail.shot == 0 and fail.kind == me.FAIL_MERGE_NON_FINITE
    assert me.OP_NAMES[fail.op] == "merge_frames"


# --------------------------------------------------------------------------------------------
# r04 / R16: the raw shot-list bound
# --------------------------------------------------------------------------------------------


def _raw_calls(store: LiveExpansions) -> dict[str, Callable[[int], int]]:
    """The three raw entry points with valid arguments and a caller-chosen list length."""
    eng, lib = store.engine, me.get_magic_engine_lib()
    ax, kick = _x(0), _z(2)
    lst, one = np.zeros(1, np.int32), np.zeros(1, np.uint8)
    p1, done, z01 = np.zeros(1), np.zeros(1, np.uint8), np.zeros((1, 2))
    err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(2, np.int64)
    o = (addr(ax.a_words), addr(ax.b_words), 0)
    tail = (addr(err), addr(errd), addr(info))

    def peek(n: int) -> int:
        return lib.mbe_peek(eng._h(), *o, addr(lst), n, addr(p1), *tail)

    def post(n: int) -> int:
        return lib.mbe_postselect(eng._h(), *o, addr(lst), n, addr(one), addr(done), addr(z01),
                                  *tail)

    def um(n: int) -> int:
        return lib.mbe_unmatched_apply(eng._h(), *o, addr(kick.a_words), addr(kick.b_words), 0,
                                       addr(lst), n, addr(one), addr(one), addr(one), *tail)

    return {"peek": peek, "postselect": post, "unmatched": um}


@pytest.mark.parametrize("name", ["peek", "postselect", "unmatched"])
def test_a_shot_list_length_above_the_quarter_bound_is_refused_before_it_is_read(
    name: str,
) -> None:
    """`n > kMaxElems / 4` is refused with -2 (nothing runs); a looser bound lets `n` through to an
    allocation of that many entries (-3, out of memory)."""
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    kmax = (2**63 - 1) // 8  # kMaxElems = PTRDIFF_MAX / 8
    before = [store.engine.export_shot(b)[5:] for b in range(2)]
    call = _raw_calls(store)[name]
    for n in (kmax // 4 + 1, kmax // 2, kmax, kmax + 1):
        assert call(n) == -2, n
    after = [store.engine.export_shot(b)[5:] for b in range(2)]
    assert all(np.array_equal(x, y) for bb, aa in zip(before, after) for x, y in zip(bb, aa))


# --------------------------------------------------------------------------------------------
# r11 / R19 / r12 / R21: the engine / reference routing of the simulator wiring
# --------------------------------------------------------------------------------------------


def test_a_postselect_mask_naming_exactly_the_batch_size_is_an_index_error_on_both() -> None:
    """Shot `batch_size` does not exist: the reference code raises IndexError (the engine is not
    asked); the committed prefix and the state equal the list store's."""
    pair = _both([_t_state(), _t_state(_N, 3)])
    mask, bits = np.array([1, 0, 1], bool), np.zeros(2, np.uint8)  # bits: one entry per shot
    for sim in pair[:2]:
        exc = _run(sim, "postselect_z_batch", 0, mask, bits)[1]
        assert isinstance(exc, IndexError), exc
    _assert_same(pair[0], pair[1], "postselect", pair[2], pair[3])


def test_an_unmatched_mask_naming_exactly_the_batch_size_is_an_index_error_on_both() -> None:
    pair = _both([_t_state(), _t_state(_N, 3)])
    zero, one = _masks(3, {0: [0, 2]}), _masks(3, {0: [1]})
    for sim in pair[:2]:
        exc = _run(sim, "do_unmatched_noisy_reset_z", zero, one, "Z", [0])[1]
        assert isinstance(exc, IndexError), exc
    _assert_same(pair[0], pair[1], "unmatched", pair[2], pair[3])


def test_a_short_desired_bits_array_commits_the_shots_before_the_missing_entry() -> None:
    """The reference reads `desired_bits[b]` shot by shot: shot 0 is committed, then IndexError.
    The engine path would read all entries first (and commit nothing)."""
    pair = _both([_t_state(), _t_state(_N, 3), _t_state(_N, 4)])
    exc = _same(pair, "postselect_z_batch", 0, np.array([1, 1, 1], bool), np.array([1], np.uint8),
                "X")
    assert isinstance(exc, IndexError), exc
    assert pair[0]._log_branch_masses[0] != 0.0  # shot 0 really was committed on the reference


# --------------------------------------------------------------------------------------------
# r16 / R24 / R25: a postselect or an unmatched apply drops the pending split in Python too
# --------------------------------------------------------------------------------------------


def _pending_split(store: LiveExpansions) -> Any:
    ax = _x(0)
    split = store.measure_split((ax.a_words, ax.b_words, ax.kappa), True, None)
    assert split.n_active > 0 and store.engine._pending_token == split.token
    return split


def _finish_message(store: LiveExpansions, split: Any) -> str:
    with pytest.raises(RuntimeError) as info:
        store.measure_finish(np.zeros(split.n_active, bool), np.ones(split.n_active), split)
    return str(info.value)


def test_a_finish_after_a_postselect_is_refused_with_the_no_pending_split_message() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    split = _pending_split(store)
    zz = _z(1)
    store.postselect((zz.a_words, zz.b_words, zz.kappa), np.array([0], np.int32),
                     np.array([0], np.uint8))
    assert store.engine._pending_token == 0
    assert "matches this split" in _finish_message(store, split)


def test_a_finish_after_an_unmatched_apply_is_refused_with_the_no_pending_split_message() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    split = _pending_split(store)
    o, kick = _z(1), _x(1)
    store.unmatched_apply((o.a_words, o.b_words, o.kappa), (kick.a_words, kick.b_words, kick.kappa),
                          np.array([0], np.int32), [0], [0], [0])
    assert store.engine._pending_token == 0
    assert "matches this split" in _finish_message(store, split)


def test_a_peek_keeps_the_pending_split_so_the_finish_still_works() -> None:
    """The counterpart: a peek does not drop the split (the other two wrappers do)."""
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    split = _pending_split(store)
    zz = _z(1)
    store.peek((zz.a_words, zz.b_words, zz.kappa), np.array([0], np.int32))
    assert store.engine._pending_token == split.token
    store.measure_finish(np.zeros(split.n_active, bool), np.ones(split.n_active), split)


# --------------------------------------------------------------------------------------------
# r17: the sync reset snaps at exactly 1e-12
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1e-12, 1.0 - 1e-12])
def test_a_sync_reset_at_exactly_the_snap_value_is_pinned_and_makes_no_sampler_call(
    target: float,
) -> None:
    pair = _both([_state_with_p1(target)] * 2, sync=True)
    zero, one = _masks(2, {0: [0, 1]}), _masks(2, {0: [0]})
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is None
    assert pair[2] == pair[3] == []  # the reference drew nothing from the shots' streams


# --------------------------------------------------------------------------------------------
# R23: an out-of-memory at list entry 0
# --------------------------------------------------------------------------------------------


def test_an_out_of_memory_at_list_entry_zero_is_raised_as_memory_error() -> None:
    sim = _both([_t_state()])[1]
    with pytest.raises(MemoryError, match="list entry 0"):
        sim._engine_error(None, 0, "mbe_unmatched_apply")
    sim._engine_error(None, None, "mbe_unmatched_apply")  # no failure, no oom: nothing raised


# --------------------------------------------------------------------------------------------
# L1: mask objects that violate the declared dtype take the reference loop
# --------------------------------------------------------------------------------------------


class _Hostile(np.ndarray):
    """An ndarray whose third scalar read of ITSELF raises: the reference loop reads lazily (a
    prefix of the shots is committed before it fails), an eager reader would read all first."""

    owner: Any = None
    reads = 0

    def __getitem__(self, idx: Any) -> Any:
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        if self is type(self).owner and all(isinstance(i, (int, np.integer)) for i in idx_t):
            type(self).reads += 1
            if type(self).reads == 3:
                raise RuntimeError("hostile read")
        return super().__getitem__(idx)


def _hostile(arr: np.ndarray) -> np.ndarray:
    h = np.array(arr).view(_Hostile)
    _Hostile.owner, _Hostile.reads = h, 0
    return h


def _drawn_states(n: int = 6) -> list[pq.ShotExpansion]:
    """Shots whose unmatched reset really draws (p1 away from 0 and 1)."""
    return [_two_branch_state(np.array([math.sqrt(0.3), math.sqrt(0.7)], np.complex128))
            for _ in range(n)]


@pytest.mark.parametrize("big", [2**70, 2**64, 2**63])
def test_an_object_dtype_reset_one_mask_gets_the_reference_result(big: int) -> None:
    """Python ints >= 2**63 do not fit the engine's int64 conversion; the reference compares them
    shot by shot (`m_b != int(mask[q, b])`): the engine store must give the same result."""
    pair = _both(_drawn_states())
    zero = _masks(6, {0: list(range(6))})
    one = np.zeros((_N, 6), dtype=object)
    one[0] = [big if b % 2 else 0 for b in range(6)]
    assert _same(pair, "do_unmatched_noisy_reset_z", zero, one, "Z", [0]) is None


def test_a_hostile_reset_one_mask_is_read_lazily_like_the_reference() -> None:
    """The third scalar read raises: the reference has committed shots 0 and 1 by then, and the
    engine store must have the same prefix, state and random stream."""
    pair = _both(_drawn_states())
    zero, one = _masks(6, {0: list(range(6))}), np.zeros((_N, 6), bool)
    exc = _same_fresh(pair, "do_unmatched_noisy_reset_z", lambda: (zero, _hostile(one), "Z", [0]))
    assert isinstance(exc, RuntimeError)


def test_a_hostile_desired_bits_array_is_read_lazily_like_the_reference() -> None:
    pair = _both([_t_state(_N, 2 + (b % 3)) for b in range(6)])
    mask, bits = np.ones(6, bool), np.zeros(6, np.uint8)
    exc = _same_fresh(pair, "postselect_z_batch", lambda: (0, mask, _hostile(bits), "Z"))
    assert isinstance(exc, RuntimeError)
