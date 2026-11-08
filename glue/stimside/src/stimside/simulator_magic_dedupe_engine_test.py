"""Engine counterpart of `simulator_magic_dedupe_test.py` (WP-25 S5b).

The dedupe tests of WP-24 B3 spy on the Python seam `_group_shots_by_state` and edit the stored
`ShotExpansion`s in place; with the shots in the C++ engine the grouping is done by `mbe_partition`
/ `mbe_apply_block` and a stored expansion is a copy, so those white-box assertions run against the
pinned list store (fixture `list_store` there). This file keeps the same BEHAVIOURAL assertions
for the engine store, by comparing it with the reference list store:

* every magic golden pair gives the committed digest with the engine store (never re-blessed),
  and many `magic_ds` flushes really merge shots (counted on the engine's own result);
* a noisy circuit with equal and unequal shots at every flush: engine == list, bytes of every
  shot, branch masses and the random state;
* shots that differ in exactly one field are not merged with the rest, per field, through
  `_flush_magic`: engine == list (and engine == list with identity grouping);
* each shot keeps its own `log_mass` through a merged group; a fanned-out result shares no memory
  and an edit of one shot (read, change, assign back) leaves the others unchanged.
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside.simulator_magic as sm
from stimside import golden_test as gt
from stimside._magic.live_store import LiveExpansions
from stimside._magic.phase_qfe import ShotExpansion
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_magic_dedupe_test import (
    _ARRAY_FIELDS,
    _D2_CIRCUIT,
    _EXPECTED,
    _MAGIC_PAIRS,
    _NOISY_CIRCUIT,
    _exp_digest,
    _identity_grouping,
    _pauli_x0_in_place,
    _perturbations,
    _state_digest,
)


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Simulators and their spies form reference cycles that hold a native engine until the
    cyclic gc runs; free them here so the live-engine counts other tests assert on stay exact."""
    yield
    gc.collect()



def _engine(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", on)


def _queued(on: bool, monkeypatch: pytest.MonkeyPatch, batch: int = 6,
            perturb: Callable[[ShotExpansion], None] | None = None,
            shots: tuple[int, ...] = (1, 4)) -> MagicsideSimulator:
    """Identical d=2 states on every shot, `perturb` applied to `shots` (read, change, assign
    back: works for both stores) and one pending block."""
    _engine(monkeypatch, on)
    circ = stim.Circuit(_D2_CIRCUIT)
    sim = MagicsideSimulator(circ, batch_size=batch, seed=1)
    assert isinstance(sim._live_exps, LiveExpansions) == on
    sim._do(circ)
    sim._flush_magic()
    sim._do(stim.Circuit("I[T] 1"))
    assert len(sim._pending_magic) == 1
    assert {(e.d, e.chi) for e in sim._live_exps} == {(2, 8)}
    for s in shots if perturb is not None else ():
        e = sim._live_exps[s].copy()
        assert perturb is not None
        perturb(e)
        sim._live_exps[s] = e
    return sim


def _digests(sim: MagicsideSimulator) -> list[str]:
    return [_exp_digest(sim._live_exps[b]) for b in range(sim.batch_size)]


@pytest.mark.parametrize("case", _MAGIC_PAIRS, ids=[f"{c[0]}-{c[1]}" for c in _MAGIC_PAIRS])
def test_every_magic_golden_pair_keeps_its_committed_digest_with_the_engine_store(
    case: tuple[str, str, bool, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine(monkeypatch, True)
    name, mode, has_leak, text = case
    mb = name == "c16_sparse_multibatch"
    digest = gt._run_case(
        stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH, multibatch_packed=mb
    )
    assert digest == _EXPECTED[name][mode]


def test_the_engine_dedupe_merges_shots_on_the_magic_golden_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the engine-vs-reference comparisons against being vacuous: the engine's own
    `apply_block` results (groups run, live shots) show many flushes that do merge shots."""
    _engine(monkeypatch, True)
    seen: list[tuple[int, int]] = []
    real = LiveExpansions.apply_block

    def spy(self: LiveExpansions, *a: Any, **k: Any) -> tuple[int, int]:
        out = real(self, *a, **k)
        seen.append(out)
        return out

    monkeypatch.setattr(LiveExpansions, "apply_block", spy)
    for case in _MAGIC_PAIRS:
        if case[1] == "magic_ds":
            name, mode, has_leak, text = case
            mb = name == "c16_sparse_multibatch"
            gt._run_case(stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH,
                         multibatch_packed=mb)
    assert seen, "the engine path never ran a block"
    assert sum(g for g, _ in seen) < sum(live for _, live in seen)


def _run_noisy(seed: int, on: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[str, bytes]:
    _engine(monkeypatch, on)
    sim = MagicsideSimulator(stim.Circuit(_NOISY_CIRCUIT), batch_size=16, seed=seed)
    sim.run()
    return _state_digest(sim), sim.get_final_measurement_records().tobytes()


@pytest.mark.parametrize("seed", range(6))
def test_engine_dedupe_matches_the_list_store_on_a_noisy_circuit_with_mixed_group_sizes(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _run_noisy(seed, False, monkeypatch)
    assert _run_noisy(seed, True, monkeypatch) == ref
    monkeypatch.setattr(sm, "_group_shots_by_state", _identity_grouping)
    assert _run_noisy(seed, False, monkeypatch) == ref  # the list store without dedupe agrees


@pytest.mark.parametrize("which", sorted(_perturbations()))
def test_engine_flush_matches_the_list_store_when_two_shots_differ_in_exactly_one_field(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    perturb = _perturbations()[which]
    on = _queued(True, monkeypatch, perturb=perturb)
    off = _queued(False, monkeypatch, perturb=perturb)
    assert _digests(on) == _digests(off)  # the perturbation is the same on both stores
    rep = on._live_exps.engine.partition(True).tolist()
    assert rep == [0, 1, 0, 0, 1, 0]  # shots 1 and 4 are their own group, the rest merge
    on._flush_magic()
    off._flush_magic()
    got = _digests(on)
    assert got == _digests(off)
    if which != "canon_flag":  # the block canonicalises, which erases only the flag difference
        assert got[0] != got[1]  # the perturbation survived the block: the groups really differ


def test_engine_flush_keeps_each_shots_own_log_mass_through_a_merged_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _queued(True, monkeypatch)
    want = [-0.25 * s for s in range(6)]
    for s, lm in enumerate(want):
        e = sim._live_exps[s].copy()
        e.log_mass = lm
        sim._live_exps[s] = e
    assert sim._live_exps.engine.partition(True).tolist() == list(range(6))  # log_mass is key
    sim._flush_magic()
    assert [sim._live_exps[b].log_mass for b in range(6)] == want
    assert len(set(_digests(sim))) == 6


def test_engine_fan_out_gives_each_shot_its_own_expansion_with_no_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _queued(True, monkeypatch)
    assert set(sim._live_exps.engine.partition(True).tolist()) == {0}  # one group of six
    sim._flush_magic()
    exps = [sim._live_exps[b] for b in range(6)]
    assert len({id(e) for e in exps}) == 6
    for a, b in itertools.combinations(exps, 2):
        assert _exp_digest(a) == _exp_digest(b)
        for name in _ARRAY_FIELDS:
            assert not np.shares_memory(getattr(a, name), getattr(b, name)), name


def test_engine_edit_of_one_shot_after_dedupe_leaves_other_shots_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _queued(True, monkeypatch)
    sim._flush_magic()
    before = _digests(sim)
    store = sim._live_exps
    e2 = store[2]
    _pauli_x0_in_place(e2)
    store[2] = e2  # an engine shot is a copy: the edit reaches it only when assigned back
    assert _digests(sim)[2] != before[2]
    e4 = store[4]
    e4.weights *= 2.0
    e5 = store[5]
    e5.Gamma ^= np.uint8(1)
    assert _digests(sim) == [*before[:2], _digests(sim)[2], *before[3:]]  # unassigned: no effect
    store[4], store[5] = e4, e5
    after = _digests(sim)
    assert [s for s in range(6) if after[s] != before[s]] == [2, 4, 5]
