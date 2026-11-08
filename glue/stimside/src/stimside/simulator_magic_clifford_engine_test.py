"""Simulator-level checks of the engine's masked Clifford command (switch on vs off).

With `_LIVE_STORE_ENGINE` on, `_apply_conditional_ref_step` and `_fire_fault_channel_live` issue
one `inject_factor_lists` engine command. Everything observable must equal the reference list
path bit for bit: records, expansions, branch masses, RNG streams, the reference tableau and the
frame events. Both reference-policy branches are exercised (apply to the minority and keep the
reference, or advance it and apply the inverse to the rest), including the `2 * n_act == B` tie.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside.simulator_magic as sm
from stimside import golden_test
from stimside._magic.gates import CliffordFaultChannel, FaultBranch
from stimside._magic.live_store import LiveExpansions
from stimside.simulator_magic import MagicsideSimulator


class _CountingStore(LiveExpansions):
    """Counts engine Clifford commands and the shots each one touched."""

    commands = 0
    shots = 0
    inverse = 0

    def inject_factor_lists(self, lists: Any, shot_list: Any, inverse: bool) -> None:
        type(self).commands += 1
        type(self).shots += int(np.count_nonzero(np.asarray(shot_list) >= 0))
        type(self).inverse += bool(inverse)
        super().inject_factor_lists(lists, shot_list, inverse)


def _engine(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", on)
    monkeypatch.setattr(sm, "LiveExpansions", _CountingStore)
    _CountingStore.commands = _CountingStore.shots = _CountingStore.inverse = 0


def test_every_golden_digest_is_unchanged_with_masked_cliffords_in_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, True)
    actual = golden_test._compute_all_digests()
    expected = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
    assert _CountingStore.commands >= 10 and _CountingStore.shots >= 20
    assert _CountingStore.inverse >= 1  # the majority branch ran through the engine too


def _tab_bytes(sim: MagicsideSimulator) -> bytes:
    return repr(sim._live_ref_tab.to_stim_tableau()).encode()


def _snapshot(sim: MagicsideSimulator, batch: int) -> list[bytes]:
    out = [c.tobytes() for c in sim._measurement_columns]
    out += [sim._log_branch_masses.tobytes(), repr(sim.np_rng.bit_generator.state).encode()]
    out += [_tab_bytes(sim), repr(sim._live_frame_events).encode()]
    for b in range(batch):
        _, e = sim.get_qfe_expansion(b)
        out += [e.weights.view(np.uint64).tobytes(), e.h_words.tobytes(), e.ell.tobytes()]
        out += [e.A_words.tobytes(), e.Gamma.tobytes(), e.pivots.tobytes()]
        out += [repr((e.d, e.chi)).encode()]
        out += [float(e.log_mass).hex().encode(), bytes([e._is_canonical_merged])]
    return out


# A heralded/conditioned two-qubit Clifford on a subset of shots, after a T-gate, with feedback.
_COND_CIRCUIT = """
R 0 1 2 3
H 0 1
I[T] 0
I[T] 1
CX 0 2
X_ERROR(0.35) 0 1 2 3
M 0 1
CX rec[-1] 2
CZ rec[-2] 3
S 0
M 2 3
DETECTOR rec[-1]
"""


def _run(circ: str, batch: int, seed: int, on: bool, mp: pytest.MonkeyPatch, **kw: Any):
    _engine(mp, on)
    sim = MagicsideSimulator(stim.Circuit(circ), batch_size=batch, seed=seed, **kw)
    sim.interactive_do(stim.Circuit(circ))
    sim.finish_interactive_run()
    return sim


@pytest.mark.parametrize("batch", [1, 2, 3, 7, 64])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_conditional_cliffords_agree_with_the_reference_for_any_batch_size(
    batch: int, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _run(_COND_CIRCUIT, batch, seed, False, monkeypatch)
    eng = _run(_COND_CIRCUIT, batch, seed, True, monkeypatch)
    assert isinstance(eng._live_exps, LiveExpansions)
    assert _snapshot(ref, batch) == _snapshot(eng, batch)


@pytest.mark.parametrize("n_act,batch", [(1, 2), (2, 4), (1, 3), (2, 3), (3, 6), (4, 6), (1, 64)])
def test_minority_majority_and_tie_decisions_agree_with_the_reference(
    n_act: int, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A masked CZ with exactly `n_act` active shots: `2*n_act <= B` is the minority branch."""
    results = []
    for on in (False, True):
        _engine(monkeypatch, on)
        sim = MagicsideSimulator(
            stim.Circuit("R 0 1\nH 0 1\nI[T] 0\nI[T] 1\n"), batch_size=batch, seed=4
        )
        sim.interactive_do(stim.Circuit("R 0 1\nH 0 1\nI[T] 0\nI[T] 1\n"))
        mask = np.zeros(batch, dtype=bool)
        mask[:n_act] = True
        sim._apply_unitary_group("CZ", (0, 1), mask)
        sim._apply_unitary_group("S", (0,), mask[::-1].copy())
        sim.finish_interactive_run()
        results.append((_snapshot(sim, batch), _CountingStore.commands, _CountingStore.inverse))
    assert results[0][0] == results[1][0]
    assert results[1][1] == 2  # one engine command per masked group
    minority = 2 * n_act <= batch
    assert results[1][2] == (0 if minority else 2)  # both masks have n_act active shots


def test_the_reference_advances_only_in_the_majority_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the policy of the code (not the plan text): 2*n_act <= B keeps the reference."""
    _engine(monkeypatch, True)
    circ = stim.Circuit("R 0 1\nH 0 1\nI[T] 0\n")
    for n_act, advanced in ((1, False), (2, False), (3, True)):
        sim = MagicsideSimulator(circ, batch_size=4, seed=1)
        sim.interactive_do(circ)
        before = len(sim._live_frame_events)
        mask = np.zeros(4, dtype=bool)
        mask[:n_act] = True
        sim._apply_unitary_group("CX", (0, 1), mask)
        assert (len(sim._live_frame_events) > before) is advanced, n_act
        sim.finish_interactive_run()


def _channel(slot: int) -> CliffordFaultChannel:
    return CliffordFaultChannel(
        0,
        (FaultBranch(0.3, "H", (0,)), FaultBranch(0.2, "CX", (0, 1)),
         FaultBranch(0.1, "S", (1,))),
        slot,
    )


@pytest.mark.parametrize("batch", [1, 5, 64])
@pytest.mark.parametrize("sync", [False, True])
def test_a_live_fault_channel_agrees_with_the_reference(
    batch: int, sync: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    snaps = []
    for on in (False, True):
        _engine(monkeypatch, on)
        circ = stim.Circuit("R 0 1\nH 0 1\nI[T] 0\nI[T] 1\n")
        sim = MagicsideSimulator(circ, batch_size=batch, seed=9, sync_tableside_rng=sync)
        sim.interactive_do(circ)
        for slot in range(4):
            sim._fire_fault_channel_live(slot, _channel(slot))
        sim.finish_interactive_run()
        snaps.append((_snapshot(sim, batch), _CountingStore.commands))
    assert snaps[0][0] == snaps[1][0]
    assert snaps[1][1] == 4  # one engine command per fired channel, not one per branch


def test_forced_fault_branches_agree_with_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    snaps = []
    for on in (False, True):
        _engine(monkeypatch, on)
        circ = stim.Circuit("R 0 1\nH 0 1\nI[T] 0\n")
        sim = MagicsideSimulator(circ, batch_size=6, seed=2)
        sim.interactive_do(circ)
        forced = [0, 1, 2, 3, 1, 2]
        monkeypatch.setattr(sim, "_forced_fault_branch", lambda s, slot, f=forced: f[s])
        sim._fire_fault_channel_live(0, _channel(0))
        sim.finish_interactive_run()
        snaps.append(_snapshot(sim, 6))
    assert snaps[0] == snaps[1]
