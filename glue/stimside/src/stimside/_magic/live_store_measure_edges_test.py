"""Edge cases and error paths of the engine measurement (WP-25 stage S5a): `measure_split` and
`measure_finish` against today's reference path, case by case.

The random corpus of `live_store_measure_diff_test` has no way to hit an exact boundary (a mass
of exactly the zero floor, a probability of exactly 0.5 +- 1e-12, a one-ulp difference) or most
error paths, so each is built here by hand. Every case is run on a reference simulator (plain
list of `ShotExpansion`s: `measure_expansions_batch`) and on an engine simulator; the type and text
of an exception, the partial state it leaves (shots, branch masses, random streams) and the
sampler calls must be identical. Bytes only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
import stimside._native.magic_engine as me
import stimside.simulator_magic as sm
from stimside._magic.channels import MeasurementRecord, ResetRecord
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, born_outcomes
from stimside._magic.live_store_measure_diff_test import _assert_same, _pair, _step
from stimside._magic.live_store_test import _nonzero_axis, random_state

_ATOL = 1e-10
_N = 5


def _z(q: int, n: int = _N, kappa: int = 0) -> CompiledPauliAxis:
    nw = pq.n_words_for_qubits(n)
    b = np.zeros(nw, np.uint64)
    b[q >> 6] = np.uint64(1) << np.uint64(q & 63)
    return CompiledPauliAxis(np.zeros(nw, np.uint64), b, kappa)


def _x(q: int, n: int = _N) -> CompiledPauliAxis:
    nw = pq.n_words_for_qubits(n)
    a = np.zeros(nw, np.uint64)
    a[q >> 6] = np.uint64(1) << np.uint64(q & 63)
    return CompiledPauliAxis(a, np.zeros(nw, np.uint64), 0)


def _meas(ax: CompiledPauliAxis, keep: bool = False) -> MeasurementRecord:
    return MeasurementRecord(0, 0.0, False, ax, ax if (keep or ax.a_words.any()) else None)


def _reset(ax: CompiledPauliAxis, kick: CompiledPauliAxis, flip_ref: bool) -> ResetRecord:
    return ResetRecord(None, 0.0, False, ax, ax if ax.a_words.any() else None, flip_ref, kick)


def _t_state(n: int = _N, k: int = 2) -> pq.ShotExpansion:
    rng = np.random.default_rng(3)
    e = pq.ShotExpansion.zero(n)
    for _ in range(k):
        e = pq.apply_pauli_t(e, *_nonzero_axis(rng, n))
    return e


def _scaled(e: pq.ShotExpansion, k: complex) -> pq.ShotExpansion:
    out = e.copy()
    out.weights = out.weights * k
    return out


def _both(
    states: list[pq.ShotExpansion], recs: list[Any], sync: bool = False, atol: float = _ATOL,
    seed: int = 4,
) -> tuple[Any, Any]:
    """Run `recs` in order on both paths; assert identical results, exceptions and state."""
    ref, eng, rc, ec = _pair(states, _N, seed, sync, atol)
    first: Exception | None = None
    for i, rec in enumerate(recs):
        want, wexc = _step(ref, rec)
        got, gexc = _step(eng, rec)
        assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), (i, gexc, wexc)
        _assert_same(ref, eng, f"record {i}", rc, ec)
        if wexc is not None:
            first = first or wexc
            break
        assert want.tobytes() == got.tobytes(), i
    return first, (ref, eng)


# --------------------------------------------------------------------------------------------
# Classification and kicks
# --------------------------------------------------------------------------------------------


def test_dead_shots_get_minus_infinity_and_no_draw_alongside_live_ones() -> None:
    dead = pq.ShotExpansion.from_components((), n_qubits=_N, merge=False)
    states = [dead, _t_state(), pq.ShotExpansion.zero(_N), dead, _t_state(_N, 3)]
    exc, (_ref, eng) = _both(states, [_meas(_x(0)), _meas(_z(1))])
    assert exc is None
    assert eng._log_branch_masses[0] == -math.inf and eng._log_branch_masses[3] == -math.inf


def test_every_shot_dead_is_a_noop_that_draws_nothing() -> None:
    dead = pq.ShotExpansion.from_components((), n_qubits=_N, merge=False)
    exc, (_ref, eng) = _both([dead, dead], [_meas(_z(0)), _reset(_z(0), _x(1), True)])
    assert exc is None and (eng._log_branch_masses == -math.inf).all()


@pytest.mark.parametrize("flip_ref", [False, True])
def test_trivial_shots_are_kicked_in_a_reset_exactly_when_the_bit_differs_from_flip_ref(
    flip_ref: bool,
) -> None:
    ones = pq.apply_pauli(pq.ShotExpansion.zero(_N), _x(0))  # |1> on qubit 0: bit 1 in Z0
    states = [pq.ShotExpansion.zero(_N), ones, pq.ShotExpansion.zero(_N), ones]
    rec = _reset(_z(0), _x(2), flip_ref)
    exc, (_ref, eng) = _both(states, [rec])
    assert exc is None
    kicked = [not np.array_equal(eng._live_exps[b].h_words, states[b].h_words) for b in range(4)]
    assert kicked == [flip_ref is True, flip_ref is False, flip_ref is True, flip_ref is False]


def test_trivial_shot_kicks_are_applied_before_a_later_shot_raises() -> None:
    """The reference kicks trivial shots in its classification loop, before the batch call that
    raises: the engine must leave the same partially kicked state."""
    bad = _scaled(_t_state(), complex(float("nan"), 0.0))
    states = [pq.ShotExpansion.zero(_N), bad, pq.ShotExpansion.zero(_N)]
    rec = _reset(_z(0), _x(2), True)
    exc, _ = _both(states, [rec])
    assert isinstance(exc, FloatingPointError)
    assert exc is not None  # and the shot before the failing one was kicked, as in the reference


def test_a_rotation_record_makes_a_chi1_d0_shot_active_not_trivial() -> None:
    states = [pq.ShotExpansion.zero(_N)] * 3
    exc, _ = _both(states, [_meas(_x(0)), _reset(_x(1), _z(2), False)])
    assert exc is None


def test_a_rot_axis_that_disagrees_with_the_observable_words_matches_the_reference() -> None:
    """`keep_rot` is the record's own `rot_axis is not None`, whatever the words say."""
    states = [_t_state(), pq.ShotExpansion.zero(_N), _t_state(_N, 3)]
    for ax, keep in ((_x(0), False), (_z(1), True)):
        rec = MeasurementRecord(0, 0.0, False, ax, ax if keep else None)
        exc, _ = _both(states, [rec])
        assert exc is None


# --------------------------------------------------------------------------------------------
# Draws: one per active shot (pinned ones too), snapping boundaries, sync callback
# --------------------------------------------------------------------------------------------


def test_pinned_shots_consume_a_draw_like_the_others() -> None:
    """|0>, |0>, T|+>-like: the first two have z1 == 0 (pinned to outcome 0) but still draw."""
    states = [pq.ShotExpansion.zero(_N), _t_state(), pq.ShotExpansion.zero(_N), _t_state()]
    rec = _meas(_x(1))  # x-type: the rotation makes every shot active
    exc, _ = _both(states, [rec, rec])
    assert exc is None


@pytest.mark.parametrize("sync", [False, True])
def test_the_probability_snapping_boundaries_match_to_the_ulp(sync: bool) -> None:
    """z1 / z at 1e-13 and 1 - 1e-13 +- 1 ulp, 0.5 +- 1e-12 +- 1 ulp, for every weight scale."""
    base = pq.ShotExpansion.zero(_N)
    plus = pq.apply_pauli_t(pq.apply_quarter_turn(base, *_x(0).as_tuple()), *_z(0).as_tuple())
    states = []
    for p1 in (1e-13, np.nextafter(1e-13, 0), np.nextafter(1e-13, 1), 1 - 1e-13,
               np.nextafter(1 - 1e-13, 0), np.nextafter(1 - 1e-13, 1), 0.5 - 1e-12, 0.5 + 1e-12,
               np.nextafter(0.5 + 1e-12, 1), np.nextafter(0.5 - 1e-12, 0), 0.5, 1e-12, 1 - 1e-12):
        w = np.array([math.sqrt(1 - p1), math.sqrt(p1)], dtype=np.complex128)
        st = _two_branch_state(w)
        states.append(st)
    del plus
    exc, _ = _both(states, [_meas(_z(0))], sync=sync, seed=6)
    assert exc is None


def _two_branch_state(w: Any) -> pq.ShotExpansion:
    """chi = 2, d = 0, h = (0, e_0): amplitudes w[0] |0> + w[1] |1> on qubit 0."""
    nw = pq.n_words_for_qubits(_N)
    h = np.zeros((2, nw), np.uint64)
    h[1, 0] = 1
    return pq.ShotExpansion(
        _N, 0, np.zeros((0, nw), np.uint64), np.zeros(nw, np.uint64), np.zeros((0, 0), np.uint8),
        np.zeros(0, np.int32), np.asarray(w, np.complex128), h, np.zeros((2, 0), np.uint8),
        np.zeros(0, np.uint8), 0.0, _ATOL, True,
    )


def test_a_mass_exactly_at_the_zero_floor_and_one_ulp_either_side() -> None:
    """z0 <= K1 * z is pinned (outcome 1) but 1 ulp above it is drawn: same on both paths."""
    k = pq.K1_CANCEL_REL_TOL
    states = []
    for p0 in (k, np.nextafter(k, 0), np.nextafter(k, 1), 2 * k, k / 2):
        w = np.array([math.sqrt(p0), math.sqrt(1 - p0)], dtype=np.complex128)
        states.append(_two_branch_state(w))
    exc, _ = _both(states, [_meas(_z(0))])
    assert exc is None


def test_sync_callback_gets_z1_over_z_not_z1_over_z0_plus_z1() -> None:
    """With z0 + z1 != z (within the conservation tolerance) the two ratios differ in the last
    bits; the sampler calls are compared as hex floats by `_assert_same`."""
    states = [_two_branch_state(np.array([0.6, 0.8 + 3e-11], np.complex128)) for _ in range(3)]
    exc, _ = _both(states, [_meas(_z(0))], sync=True, atol=1e-9)
    assert exc is None


# --------------------------------------------------------------------------------------------
# Error paths (type and text, partial state, random streams)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("what", ["nan", "inf", "overflow", "zero", "tiny", "huge"])
def test_non_finite_zero_tiny_and_huge_weights_raise_or_pass_like_the_reference(
    what: str, sync: bool
) -> None:
    k = {"nan": complex("nan"), "inf": complex("inf"), "overflow": 1e200, "zero": 0.0,
         "tiny": 1e-14, "huge": 1e3}[what]
    states = [_t_state(), _scaled(_t_state(_N, 3), k), pq.ShotExpansion.zero(_N)]
    for ax in (_x(0), _z(1)):
        exc, _ = _both(states, [_meas(ax)], sync=sync)
        if what in ("nan", "inf", "overflow", "zero"):
            assert exc is not None, (what, ax)


def test_a_non_canonical_shot_with_non_finite_weights_raises_the_merge_error() -> None:
    raw = random_state(np.random.default_rng(8), _N, 4, "raw")
    assert not raw._is_canonical_merged and raw.chi > 1
    bad = raw.copy()
    w = bad.weights.copy()
    w[1] = complex("nan")
    bad.weights = w
    exc, _ = _both([_t_state(), bad], [_meas(_x(0))])
    assert exc is not None


def test_a_shot_that_cancels_to_nothing_raises_the_zero_norm_error() -> None:
    cancel = _two_branch_state(np.array([1.0, 0.0], np.complex128))
    cancel.weights = np.zeros(2, np.complex128)
    exc, _ = _both([_t_state(), cancel], [_meas(_z(0))])
    assert isinstance(exc, pq.ImpossiblePostselectionError)


def test_the_rotation_unitarity_checks_at_absurd_tolerances_match_the_reference() -> None:
    """The batch kernel's SEQUENTIAL post-sum (in the split) and numpy's PAIRWISE one (the
    un-rotation in the finish) differ in the last bits (F6): between their two error levels one
    check passes and the other fails. Both must fail as the reference does, with the
    reference's partial state. The sweep reaches the passing case, the split's sequential check and
    the M2 conservation check; the un-rotation check of the finish (the pairwise flavour) sits in a
    window no random sweep reached (the checks before it fire first): see the S5a handoff."""
    texts: dict[str, int] = {}
    for atol in (5.7e-17, 1e-16, 1.5e-16, 2.2e-16, 3e-16, 4e-16, 1e-15, 1e-14, 1e-300):
        for seed in range(30):
            rng = np.random.default_rng([seed, 17])
            n = int(rng.choice([6, 9]))
            states = [random_state(rng, n, int(rng.choice([8, 64])), "plain") for _ in range(3)]
            ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
            if not ax.a_words.any():
                continue
            rec = MeasurementRecord(0, 0.0, bool(seed % 2), ax, None if seed % 3 == 0 else ax)
            ref, eng, rc, ec = _pair(states, n, 1, False, atol)
            _, wexc = _step(ref, rec)
            _, gexc = _step(eng, rec)
            assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), (atol, seed)
            _assert_same(ref, eng, f"atol={atol} seed={seed}", rc, ec)
            kind = "ok" if wexc is None else str(wexc).split(" ")[0] + " " + str(wexc).split(" ")[1]
            texts[kind] = texts.get(kind, 0) + 1
    assert texts.get("ok", 0) >= 30, texts
    assert texts.get("apply_quarter_turn failed", 0) >= 5, texts  # the split's sequential check
    assert texts.get("M2 Born", 0) >= 3, texts  # the conservation check right after the split


@pytest.mark.parametrize("seed", [0, 9, 18])
def test_a_failed_rotation_names_the_same_shot_index_among_the_active_ones(seed: int) -> None:
    """`apply_quarter_turn failed while measuring shot {s}`: s counts the ACTIVE shots only (a
    dead shot in front does not move it). At atol 5.7e-17 the kernel's sequential unitarity check
    fails where numpy's pairwise one would not (F6): the reference rule decides on both paths."""
    n = 6
    rng = np.random.default_rng(seed)
    states = [random_state(rng, n, 64, "plain") for _ in range(3)]
    ax = CompiledPauliAxis(*_nonzero_axis(rng, n))
    rec = MeasurementRecord(0, 0.0, False, ax, ax)
    dead = pq.ShotExpansion.from_components((), n_qubits=n, merge=False)
    texts = set()
    for lead in ([], [dead], [dead, dead]):
        ref, eng, rc, ec = _pair(lead + states, n, 1, False, 5.7e-17)
        _, wexc = _step(ref, rec)
        _, gexc = _step(eng, rec)
        assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), seed
        assert isinstance(wexc, FloatingPointError) and "while measuring shot" in str(wexc)
        _assert_same(ref, eng, f"lead={len(lead)}", rc, ec)
        texts.add(str(wexc))
    assert len(texts) == 1  # the index did not move with the dead shots in front


@pytest.mark.parametrize("pick", [0, 1])
def test_a_sampler_that_picks_an_outcome_without_mass_raises_like_the_reference(pick: int) -> None:
    """Through the simulator the pinning of the sync lambda makes this unreachable, so it is
    tested where it lives: a custom `sample_outcome` into `measure_expansions_batch` and into the
    engine's split + `born_outcomes`, after the callback ran (same calls, same text)."""
    states = [_two_branch_state(np.array([1.0, 0.0], np.complex128)),
              _two_branch_state(np.array([0.0, 1.0], np.complex128))]
    z = _z(0)
    seen_ref: list[Any] = []
    seen_eng: list[Any] = []

    def make(seen: list[Any]) -> Any:
        def sampler(i: int, p1: float) -> int:
            seen.append((i, float(p1).hex()))
            return pick

        return sampler

    ref_exc: Exception | None = None
    try:
        pq.measure_expansions_batch(states, z, rng=np.random.default_rng(1),
                                    sample_outcome=make(seen_ref), atol=_ATOL)
    except pq.ImpossiblePostselectionError as exc:
        ref_exc = exc
    store = _engine_with(states)
    res = store.measure_split((z.a_words, z.b_words, z.kappa), False, None)
    eng_exc: Exception | None = None
    try:
        born_outcomes(res, _ATOL, np.random.default_rng(1), make(seen_eng))
    except pq.ImpossiblePostselectionError as exc:
        eng_exc = exc
    assert ref_exc is not None and (type(eng_exc), str(eng_exc)) == (type(ref_exc), str(ref_exc))
    assert seen_ref == seen_eng and len(seen_ref) == 2
    store.close()


def test_a_failing_reset_kick_after_the_commit_matches_the_reference_partial_state() -> None:
    """A kick whose padding bits are set: the reference raises in its commit loop after the
    shots before it were assigned; the engine's finish has the same partial commit."""
    nw = pq.n_words_for_qubits(_N)
    bad_a = np.zeros(nw, np.uint64)
    bad_a[-1] = np.uint64(1) << np.uint64(60)  # a padding bit (n = 5)
    kick = CompiledPauliAxis(bad_a, np.zeros(nw, np.uint64), 0)
    states = [_t_state(), _t_state(_N, 3), pq.ShotExpansion.zero(_N)]
    rec = ResetRecord(None, 0.0, False, _x(0), _x(0), False, kick)
    exc, _ = _both(states, [rec])
    assert exc is not None


# --------------------------------------------------------------------------------------------
# The engine API itself
# --------------------------------------------------------------------------------------------


def _engine_with(states: list[pq.ShotExpansion]) -> LiveExpansions:
    store = LiveExpansions(len(states), states[0].n_qubits, _ATOL)
    for b, s in enumerate(states):
        store[b] = s
    return store


def _before(store: Any) -> list[bytes]:
    return [b"".join(np.asarray(x).tobytes() for x in store.engine.export_shot(b)[5:])
            for b in range(store.batch_size)]


def test_a_split_changes_no_shot_unless_a_trivial_reset_shot_is_kicked() -> None:
    store = _engine_with([_t_state(), pq.ShotExpansion.zero(_N), _t_state(_N, 3)])
    before = _before(store)
    ax = _x(0)
    res = store.measure_split((ax.a_words, ax.b_words, ax.kappa), True, None)
    assert res.n_active == 3 and res.failure is None and not res.oom
    assert _before(store) == before  # the projected branches live in scratch only
    store.close()


def test_finish_without_a_pending_split_or_twice_is_refused_and_changes_nothing() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    eng = store.engine
    before = _before(store)
    with pytest.raises(RuntimeError, match="no pending"):
        eng.measure_finish(np.zeros(2, bool), np.ones(2))
    ax = _x(0)
    res = eng.measure_split((ax.a_words, ax.b_words, ax.kappa), False, None)
    chosen = np.where(res.z01[:, 0] > 0, res.z01[:, 0], res.z01[:, 1])
    eng.measure_finish(np.zeros(2, bool), chosen)
    after = _before(store)
    assert after != before
    with pytest.raises(RuntimeError, match="no pending"):  # used up by the first finish
        eng.measure_finish(np.zeros(2, bool), chosen)
    assert _before(store) == after
    store.close()


def _mutating_commands() -> dict[str, Any]:
    from stimside._magic.channels import CompiledCliffordFactorList
    from stimside._magic.live_store import block_table
    from stimside._magic.live_store_block_diff_test import commuting_axes

    def block(s: LiveExpansions) -> None:
        axes = commuting_axes(np.random.default_rng(1), _N, 3)
        steps = (pq.CompiledPauliBlockStep(kind="P1", paulis=(axes[0].as_tuple(),), sign=1),)
        tab = block_table(pq.CompiledPauliBlock(n_qubits=_N, mode="sequential", steps=steps), _N)
        assert tab is not None
        s.apply_block(tab, None)

    def factors(s: LiveExpansions) -> None:
        fac = CompiledCliffordFactorList(1, _z(1), (_x(0),))
        s.inject_factor_lists([fac], np.zeros(s.batch_size, np.int32), False)

    return {
        "reset": lambda s: s.engine.reset(),
        "import_shot": lambda s: s.engine.import_shot(0, s.engine.export_shot(1)),
        "apply_pauli_mask": lambda s: s.apply_pauli_mask(_z(1), np.ones(s.batch_size, bool)),
        "inject_cliffords": factors,
        "apply_block": block,
    }


@pytest.mark.parametrize("command", sorted(_mutating_commands()))
def test_every_command_that_changes_a_shot_drops_the_pending_split(command: str) -> None:
    """A finish after one of them would commit branches projected from an OLD state: refused."""
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    ax = _x(0)
    res = store.engine.measure_split((ax.a_words, ax.b_words, ax.kappa), False, None)
    chosen = np.where(res.z01[:, 0] > 0, res.z01[:, 0], res.z01[:, 1])
    _mutating_commands()[command](store)
    after = _before(store)
    with pytest.raises(RuntimeError, match="no pending"):
        store.engine.measure_finish(np.zeros(2, bool), chosen)
    assert _before(store) == after  # and the refused finish changed nothing
    store.close()


def test_read_only_commands_keep_the_pending_split() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    ax = _x(0)
    res = store.engine.measure_split((ax.a_words, ax.b_words, ax.kappa), False, None)
    chosen = np.where(res.z01[:, 0] > 0, res.z01[:, 0], res.z01[:, 1])
    store.engine.partition(True)
    store.engine.export_shot(0)
    store.engine.live_shots()
    fin = store.engine.measure_finish(np.zeros(2, bool), chosen)
    assert fin.failure is None
    store.close()


def test_a_second_split_replaces_the_pending_one_and_an_empty_split_leaves_none() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    eng = store.engine
    x0, x1 = _x(0), _x(1)
    eng.measure_split((x0.a_words, x0.b_words, x0.kappa), False, None)
    res = eng.measure_split((x1.a_words, x1.b_words, x1.kappa), False, None)
    chosen = np.where(res.z01[:, 0] > 0, res.z01[:, 0], res.z01[:, 1])
    assert eng.measure_finish(np.zeros(2, bool), chosen).failure is None  # the second one
    z0 = _z(0)  # trivial for the |0..0> shots: nothing active, nothing pending
    fresh = _engine_with([pq.ShotExpansion.zero(_N)] * 2)
    res0 = fresh.engine.measure_split((z0.a_words, z0.b_words, z0.kappa), False, None)
    assert res0.n_active == 0
    with pytest.raises(RuntimeError, match="no pending"):
        fresh.engine.measure_finish(np.zeros(0, bool), np.zeros(0))
    store.close()
    fresh.close()


def test_finish_refuses_the_wrong_number_of_outcomes_and_keeps_the_state() -> None:
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    ax = _x(0)
    res = store.engine.measure_split((ax.a_words, ax.b_words, ax.kappa), False, None)
    before = _before(store)
    with pytest.raises(ValueError, match="do not match"):
        store.engine.measure_finish(np.zeros(1, bool), np.ones(1))
    assert _before(store) == before and res.n_active == 2
    store.close()


def test_finish_with_a_zero_chosen_mass_raises_division_by_zero_and_commits_nothing() -> None:
    from stimside._magic.live_store import raise_factor_failure

    store = _engine_with([_t_state(), _t_state(_N, 3)])
    ax = _x(0)
    store.engine.measure_split((ax.a_words, ax.b_words, ax.kappa), False, None)
    before = _before(store)
    fin = store.engine.measure_finish(np.zeros(2, bool), np.array([1.0, 0.0]))
    assert fin.failure is not None and fin.phase == 1 and fin.failure.shot == 1
    with pytest.raises(ZeroDivisionError):
        raise_factor_failure(fin.failure, _N, _ATOL)
    assert _before(store) == before
    store.close()


def test_the_engine_does_not_validate_a_bad_observable_before_classifying_but_reports_it() -> None:
    """A non-Hermitian observable: the reference raises ValueError from the batch call, after the
    classification (trivial shots with a reset kick are already kicked)."""
    nw = pq.n_words_for_qubits(_N)
    a = np.zeros(nw, np.uint64)
    b = np.zeros(nw, np.uint64)
    a[0], b[0] = 1, 1  # <a, b> = 1, kappa 0: not Hermitian
    bad = CompiledPauliAxis(a, b, 0)
    rec = MeasurementRecord(0, 0.0, False, bad, bad)
    exc, _ = _both([_t_state(), pq.ShotExpansion.zero(_N)], [rec])
    assert isinstance(exc, ValueError)
    # all trivial: the batch call is never made, so no validation happens, on either path
    exc2, _ = _both([pq.ShotExpansion.zero(_N)], [MeasurementRecord(0, 0.0, False, _z(0), None)])
    assert exc2 is None


def test_an_observable_with_padding_bits_matches_the_reference_error() -> None:
    nw = pq.n_words_for_qubits(_N)
    a = np.zeros(nw, np.uint64)
    a[-1] = np.uint64(1) << np.uint64(60)
    bad = CompiledPauliAxis(a, np.zeros(nw, np.uint64), 0)
    exc, _ = _both([_t_state(), _t_state(_N, 3)], [MeasurementRecord(0, 0.0, False, bad, bad)])
    assert exc is not None


def test_a_non_encodable_axis_takes_the_reference_path_with_an_engine_store() -> None:
    """An axis whose words are not uint64 arrays of shape (n_words,) never reaches the engine."""
    store = _engine_with([_t_state(), _t_state(_N, 3)])
    ok = _x(0)
    odd = CompiledPauliAxis(ok.a_words.astype(np.int64), ok.b_words, 0)  # type: ignore[arg-type]
    assert store.axis_is_encodable(ok) and not store.axis_is_encodable(odd)
    store.close()


def test_the_split_result_types_and_abi_constants() -> None:
    assert me.CLS_DEAD == 0 and me.CLS_TRIVIAL == 1 and me.CLS_ACTIVE == 2 and me.CLS_UNSEEN == 3
    assert (me.FAIL_BAD_NORM, me.FAIL_ROTATION, me.FAIL_ZERO_CHOSEN) == (9, 10, 11)


def _split_result(z: float, z0: float, z1: float, chi2: tuple[int, int] = (1, 1)) -> Any:
    one = np.array([me.CLS_ACTIVE], np.uint8)
    return me.SplitResult(
        one, np.zeros(1, np.uint8), np.array([z]), np.array([[z0, z1]]),
        np.array([chi2], np.int64), 1, None, False,
    )


class _FixedRng(np.random.Generator):
    """A generator stand-in whose `random(n)` returns the given values (and counts the calls)."""

    def __init__(self, values: list[float]) -> None:
        super().__init__(np.random.PCG64(0))
        self.values, self.calls = values, 0

    def random(self, n: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls += 1
        assert n == len(self.values)
        return np.array(self.values)


def test_the_born_outcome_compares_u_with_z1_over_z0_plus_z1_not_over_z() -> None:
    """With z0 + z1 != z (within the conservation tolerance) the two ratios differ by 1e-12; a
    draw exactly at z1 / (z0 + z1) is NOT below it, one just under it is."""
    z, z0, z1 = 1.0, 0.5, 0.5 + 1e-12
    p1 = z1 / (z0 + z1)  # plain float arithmetic: the reference formula
    assert z1 / z > p1
    for u, want in ((p1, False), (float(np.nextafter(p1, 0.0)), True)):
        out, g0, g1, chosen = born_outcomes(_split_result(z, z0, z1), 1e-9, _FixedRng([u]), None)
        assert out.tolist() == [want] and (g0[0], g1[0]) == (z0, z1)
        assert chosen[0] == (z1 if want else z0)


def test_one_draw_per_active_shot_even_when_every_outcome_is_pinned() -> None:
    rng = _FixedRng([0.3])
    out, _, _, _ = born_outcomes(_split_result(1.0, 1.0, 0.0), 1e-9, rng, None)  # z1 == 0
    assert rng.calls == 1 and out.tolist() == [False]
    rng = _FixedRng([0.3])
    out, _, _, _ = born_outcomes(_split_result(1.0, 0.0, 1.0), 1e-9, rng, None)  # z0 == 0
    assert rng.calls == 1 and out.tolist() == [True]


def test_the_zero_floor_decides_pinning_with_the_pre_norm_z() -> None:
    """z0 <= K1 * z pins the outcome to 1 whatever the draw (u = 0 would otherwise give 0 here is
    not reachable: p1 ~ 1); one ulp above the floor the draw decides (u just under p1 -> 1, a
    large u -> 0 is impossible, so use u = 1 - tiny >= p1 -> 0)."""
    k = pq.K1_CANCEL_REL_TOL
    z = 4.0
    pinned_z0 = k * z
    free_z0 = float(np.nextafter(pinned_z0, 1.0))
    for z0, u, want in (
        (pinned_z0, 0.9999999999999999, True),  # pinned: the draw is ignored
        (free_z0, 0.9999999999999999, False),  # not pinned: u >= p1 -> outcome 0
        (free_z0, 0.0, True),  # not pinned: u < p1 -> outcome 1
    ):
        out, _, _, _ = born_outcomes(_split_result(z, z0, z - z0), 1e-9, _FixedRng([u]), None)
        assert out.tolist() == [want], (z0, u)


def test_shots_whose_atol_differs_from_the_engine_atol_keep_their_own() -> None:
    """The engine stores atol per shot; a branch inherits the atol of ITS shot, bitwise."""
    rng = np.random.default_rng(12)
    states = []
    for atol in (1e-10, 3e-9, 7e-11, 1e-10):
        st = random_state(rng, _N, 4, "d")
        st.atol = atol
        states.append(st)
    ref, eng, rc, ec = _pair(states, _N, 3, False)
    for q in (0, 2):
        rec = _meas(_x(q))
        want, wexc = _step(ref, rec)
        got, gexc = _step(eng, rec)
        assert wexc is None and gexc is None and want.tobytes() == got.tobytes()
        _assert_same(ref, eng, f"x{q}", rc, ec)
    assert [eng._live_exps[b].atol for b in range(4)] == [1e-10, 3e-9, 7e-11, 1e-10]


def test_a_record_with_odd_word_arrays_gets_the_reference_result_with_an_engine_store() -> None:
    """Word arrays that are not uint64 of shape (n_words,) never reach the engine; the reference
    code then raises or accepts exactly as it does for a plain list."""
    ok = _x(0)
    odd = CompiledPauliAxis(ok.a_words.astype(np.int32), ok.b_words, 0)  # type: ignore[arg-type]
    rec = MeasurementRecord(0, 0.0, False, odd, odd)
    exc, _ = _both([_t_state(), _t_state(_N, 3)], [rec])
    assert exc is None or isinstance(exc, Exception)


# --------------------------------------------------------------------------------------------
# Gaps found by the S5a mutants
# --------------------------------------------------------------------------------------------


def _duplicate_row_state(n: int, seed: int) -> pq.ShotExpansion:
    """A non-canonical shot whose rows repeat keys: T then T_DAG of overlapping axes, unmerged.
    Merging first (before the rotation) changes the order of the floating-point sums, so a split
    that skipped that merge would differ from the reference in the last bits."""
    rng = np.random.default_rng(seed)
    e = pq.ShotExpansion.zero(n)
    axes = [_nonzero_axis(rng, n) for _ in range(2)]
    for _ in range(2):
        for ax in axes:
            e = pq.apply_pauli_t(e, *ax, dag=bool(rng.integers(0, 2)), merge=False)
    for i in range(len(e.weights)):  # irrational-ish weights so that duplicate sums round
        e.weights[i] *= complex(0.1 + 0.07 * i, 0.3 - 0.011 * i)
    assert not e._is_canonical_merged and e.chi >= 4
    keys = {(bytes(e.h_words[r]), bytes(e.ell[r])) for r in range(e.chi)}
    assert len(keys) < e.chi  # some keys really repeat
    return e


@pytest.mark.parametrize("n", [5, 70])
def test_a_non_canonical_shot_with_repeated_rows_is_merged_before_the_rotation(n: int) -> None:
    states = [_duplicate_row_state(n, s) for s in range(6)]
    rng = np.random.default_rng(n)
    for k in range(6):
        ax = CompiledPauliAxis(*_nonzero_axis(rng, n, density=0.4))
        if not ax.a_words.any():
            continue
        ref, eng, rc, ec = _pair(states, n, 1, False)
        for rec in (MeasurementRecord(0, 0.0, False, ax, ax), _reset(ax, _z(1, n), False)):
            _, wexc = _step(ref, rec)
            _, gexc = _step(eng, rec)
            assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), k
            _assert_same(ref, eng, f"n={n} axis {k}", rc, ec)


def test_the_failing_shot_of_a_split_with_several_bad_norms_is_the_first_one() -> None:
    bad = _scaled(_t_state(_N, 3), complex("nan"))
    states = [_t_state(), bad, _t_state(_N, 3), bad, pq.ShotExpansion.zero(_N)]
    store = _engine_with(states)
    ax = _x(0)
    res = store.engine.measure_split((ax.a_words, ax.b_words, 0), False, None)
    assert res.failure is not None and res.failure.kind == me.FAIL_BAD_NORM
    assert res.failure.shot == 1
    store.close()


def test_a_split_failure_leaves_the_unreached_shots_marked_unseen() -> None:
    dead = pq.ShotExpansion.from_components((), n_qubits=_N, merge=False)
    store = _engine_with([dead, _t_state(), pq.ShotExpansion.zero(_N)])
    ax = _x(0)
    res = store.engine.measure_split((ax.a_words, ax.b_words, 0), True, None)  # rotating
    assert res.failure is None and res.cls.tolist() == [me.CLS_DEAD, me.CLS_ACTIVE, me.CLS_ACTIVE]
    # a kick that cannot run (padding bits) stops the classification at the first kicked shot
    nw = pq.n_words_for_qubits(_N)
    bad_a = np.zeros(nw, np.uint64)
    bad_a[-1] = np.uint64(1) << np.uint64(60)
    z0 = _z(0)
    ones = pq.apply_pauli(pq.ShotExpansion.zero(_N), _x(0))
    store2 = _engine_with([pq.ShotExpansion.zero(_N), ones, ones, ones])
    res2 = store2.engine.measure_split(
        (z0.a_words, z0.b_words, 0), False, (bad_a, np.zeros(nw, np.uint64), 0, False)
    )
    assert res2.failure is not None and res2.failure.shot == 1
    assert res2.cls.tolist() == [me.CLS_TRIVIAL, me.CLS_TRIVIAL, me.CLS_UNSEEN, me.CLS_UNSEEN]
    store.close()
    store2.close()


def test_a_failed_finish_changes_no_shot_and_no_branch_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine contract (not reachable from `born_outcomes`, which never yields a zero chosen mass):
    a finish that fails in its first phase leaves every shot AND the branch masses as they were,
    and raises. Forced by feeding a zero mass into the finish through the simulator."""
    import stimside._magic.live_store as ls

    real = ls.born_outcomes

    def zero_chosen(*a: Any, **k: Any) -> Any:
        out, z0, z1, chosen = real(*a, **k)
        chosen = chosen.copy()
        chosen[-1] = 0.0
        return out, z0, z1, chosen

    monkeypatch.setattr(sm, "born_outcomes", zero_chosen)
    states = [_t_state(), _t_state(_N, 3), _t_state(_N, 2)]
    _ref, eng, _rc, _ec = _pair(states, _N, 5, False)
    before = _before(eng._live_exps)
    masses = eng._log_branch_masses.copy()
    rng_state = eng.np_rng.bit_generator.state
    _, exc = _step(eng, _meas(_x(0)))
    assert isinstance(exc, ZeroDivisionError)
    assert _before(eng._live_exps) == before  # nothing committed
    assert eng._log_branch_masses.tobytes() == masses.tobytes()  # nothing counted
    assert eng.np_rng.bit_generator.state != rng_state  # (the draw happened, as in the reference)


def test_a_split_that_ran_out_of_memory_is_raised_as_memory_error() -> None:
    from stimside._magic.live_store import raise_split_failure

    res = me.SplitResult(
        np.full(2, me.CLS_UNSEEN, np.uint8), np.zeros(2, np.uint8), np.zeros(2), np.zeros((2, 2)),
        np.zeros((2, 2), np.int64), 0, None, True,
    )
    with pytest.raises(MemoryError, match="no shot was measured"):
        raise_split_failure(res, _N, _ATOL)


def test_the_wrapper_binds_the_current_sentinel_and_every_earlier_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_get_lib(mod: Any, stem: str, sentinel: str, specs: Any, **kw: Any) -> str:
        seen.update(stem=stem, sentinel=sentinel, specs=specs)
        return "lib"

    monkeypatch.setattr(me, "get_lib", fake_get_lib)
    assert me.get_magic_engine_lib() == "lib"
    assert seen["sentinel"] == "mbe_abi_v6" and seen["stem"] == "magic_kernels"
    names = [n for n, _, _ in seen["specs"]]
    assert {"mbe_abi_v4", "mbe_measure_split", "mbe_measure_finish"} <= set(names)  # S5a kept
    assert {"mbe_peek", "mbe_postselect", "mbe_unmatched_apply"} <= set(names)  # S5b added
    assert all(f"mbe_abi_v{k}" in names for k in (1, 2, 3, 4, 5))


def test_a_negative_projected_mass_beyond_tolerance_raises_the_reference_error() -> None:
    """`born_outcomes` against `measure_expansions_batch`'s own message for the same masses."""
    from stimside._magic.live_store import born_outcomes

    res = _split_result(1.0, -1e-9, 1.0)
    with pytest.raises(FloatingPointError, match="Significantly negative projected Born mass"):
        born_outcomes(res, _ATOL, np.random.default_rng(0), None)
    ok = _split_result(1.0, -1e-14, 1.0)  # within -1e-12 * scale: clipped to 0, not an error
    out, z0, _, _ = born_outcomes(ok, _ATOL, np.random.default_rng(0), None)
    assert z0[0] == 0.0 and out.tolist() == [True]


def test_a_sampled_outcome_with_zero_rows_raises_even_when_its_mass_is_above_the_floor() -> None:
    """Sync only: the branch chosen by the callback has chi == 0 (an empty projection) although
    the masses look fine; the reference reads that from the kernel's `outs[1][:, 0]`."""
    from stimside._magic.live_store import born_outcomes

    res = _split_result(1.0, 0.5, 0.5, chi2=(2, 0))
    with pytest.raises(pq.ImpossiblePostselectionError, match="zero Born probability"):
        born_outcomes(res, _ATOL, np.random.default_rng(0), lambda i, p1: 1)
    out, _, _, _ = born_outcomes(res, _ATOL, np.random.default_rng(0), lambda i, p1: 0)
    assert out.tolist() == [False]


def test_the_oom_flag_wins_over_a_missing_failure_in_the_simulator_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the engine reports out-of-memory the simulator raises MemoryError (nothing is drawn)."""
    states = [_t_state(), _t_state(_N, 3)]
    _ref, eng, _rc, _ec = _pair(states, _N, 5, False)
    real = me.MagicBatchEngine.measure_split

    def oom(self: Any, *a: Any, **k: Any) -> Any:
        res = real(self, *a, **k)
        return res._replace(oom=True)

    monkeypatch.setattr(me.MagicBatchEngine, "measure_split", oom)
    before = eng.np_rng.bit_generator.state
    _, exc = _step(eng, _meas(_x(0)))
    assert isinstance(exc, MemoryError)
    assert eng.np_rng.bit_generator.state == before
