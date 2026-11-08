"""RNG streams after a *caught* failure inside a fault channel, engine on vs off.

The list store draws a branch and applies it shot by shot, so when shot `k` raises it has drawn
for shots `0..k` only. The engine draws for every shot, then issues one command; on a failure it
must give back the draws made for the shots after the failing one, so that a caller that catches
the exception and goes on sees the same random streams (`np_rng` and the per-shot generators)
with the engine on or off. Success paths cost nothing extra.
"""

from __future__ import annotations

import traceback
from typing import Any, ClassVar

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside.simulator_magic as sm
from stimside._magic.live_store import LiveExpansions
from stimside.simulator_magic import MagicsideSimulator

_CIRCUIT = """R 0 1 2 3
H 0 1 2 3
I[T] 0
I[T] 1
I[T_DAG] 3
I[CLIFFORD_FAULT: (0.124, C_NZYX) (0.226, C_XYZ) (0.228, H_NXY)] 3
I[CLIFFORD_FAULT: (0.154, C_XYZ)] 3
M 0 1 2 3
DETECTOR rec[-1] rec[-2]
"""


class _Recording(LiveExpansions):
    """Remembers where each Clifford command failed."""

    failures: ClassVar[list[int | None]] = []

    def inject_factor_lists(self, lists: Any, shot_list: Any, inverse: bool) -> None:
        try:
            super().inject_factor_lists(lists, shot_list, inverse)
        except Exception:
            type(self).failures.append(self.failed_shot)
            raise


def _rng_state(sim: MagicsideSimulator) -> tuple[str, str, float]:
    shots = "".join(repr(g.bit_generator.state) for g in (sim._shot_np_rngs or []))
    probe = np.random.default_rng()
    probe.bit_generator.state = sim.np_rng.bit_generator.state
    return repr(sim.np_rng.bit_generator.state), shots, float(probe.random())


def _run_until_failure(
    on: bool, batch: int, sync: bool, mp: pytest.MonkeyPatch, seed: int = 1
) -> Any:
    mp.setattr(sm, "_LIVE_STORE_ENGINE", on)
    mp.setattr(sm, "LiveExpansions", _Recording)
    _Recording.failures = []
    sim = MagicsideSimulator(
        stim.Circuit(_CIRCUIT), batch_size=batch, seed=seed, atol=1e-300, sync_tableside_rng=sync
    )
    try:
        sim.run()
    except FloatingPointError as exc:
        frames = [f.name for f in traceback.extract_tb(exc.__traceback__)]
        return sim, str(exc), "_fire_fault_channel_live" in frames
    return sim, "", False


# Seeds whose run fails inside a fault channel, per (batch, sync_tableside_rng): found by running
# the circuit (a batch of one fails for seed 4 only; for 7 the sets differ by mode).
_SEEDS = {
    (1, False): (4,), (1, True): (4,), (7, False): (1, 3, 7, 9), (7, True): (1, 4, 5, 6),
    (33, False): (0, 1, 2, 3), (33, True): (0, 1, 2, 3),
    (64, False): (0, 1, 2, 3), (64, True): (0, 1, 2, 3),
}


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("batch", [1, 7, 33, 64])
def test_the_rng_streams_after_a_failure_in_a_fault_channel_equal_the_list_stores(
    batch: int, sync: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    for seed in _SEEDS[batch, sync]:
        off, msg_off, in_channel = _run_until_failure(False, batch, sync, monkeypatch, seed)
        on, msg_on, _ = _run_until_failure(True, batch, sync, monkeypatch, seed)
        assert msg_off and msg_off == msg_on, (batch, seed)
        assert in_channel, f"seed {seed}: the case must fail inside a fault channel"
        assert isinstance(on._live_exps, LiveExpansions)
        assert _rng_state(off) == _rng_state(on), (batch, seed, sync)


def test_the_failures_really_leave_draws_to_give_back_for_some_batch_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the test above against passing vacuously: the failing shot is not always the last."""
    early = 0
    for batch in (7, 33, 64):
        for seed in _SEEDS[batch, False]:
            _run_until_failure(True, batch, False, monkeypatch, seed)
            early += any(f is not None and f < batch - 1 for f in _Recording.failures)
    assert early >= 6


@pytest.mark.parametrize("batch", [7, 64])
def test_a_success_path_makes_no_extra_draws_and_no_rewind(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = MagicsideSimulator._undo_excess_fault_draws

    def counting(self: MagicsideSimulator, *args: Any) -> None:
        calls.append(1)
        real(self, *args)

    monkeypatch.setattr(MagicsideSimulator, "_undo_excess_fault_draws", counting)
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_CIRCUIT), batch_size=batch, seed=1)  # default atol
    sim.run()
    assert calls == []


def _gen_at(sim: MagicsideSimulator, s: int) -> np.random.Generator:
    return sim._shot_np_rngs[s] if sim._shot_np_rngs is not None else sim.np_rng


@pytest.mark.parametrize("sync", [False, True])
def test_the_rewind_skips_forced_branches_and_keeps_a_buffered_uint32(
    sync: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced branch drew nothing; `advance` clears PCG64's buffered 32-bit word: put back.

    Five shots, shot 3 forced, the command failed at shot 1: the list store would have drawn for
    shots 0 and 1 only. The twin makes exactly those draws; after the rewind both must be equal.
    """
    circuit = stim.Circuit("H 0\nM 0")
    sim = MagicsideSimulator(circuit, batch_size=5, seed=3, sync_tableside_rng=sync)
    twin = MagicsideSimulator(circuit, batch_size=5, seed=3, sync_tableside_rng=sync)
    for model in (sim, twin):
        for s in range(5):
            _gen_at(model, s).integers(0, 5, dtype=np.uint32)  # leaves a buffered 32-bit word
    forced = {3: 1}
    monkeypatch.setattr(sim, "_forced_fault_branch", lambda s, slot_idx: forced.get(s))
    for s in (0, 1, 2, 4):  # what the engine path drew: every unforced shot
        _gen_at(sim, s).random()
    for s in (0, 1):  # what the list store drew before failing at shot 1
        _gen_at(twin, s).random()
    sim._undo_excess_fault_draws(0, 1)
    for s in range(5):
        assert _gen_at(sim, s).bit_generator.state == _gen_at(twin, s).bit_generator.state, s


def _sim_with_a_bad_shot(on: bool, mp: pytest.MonkeyPatch, bad_shot: int, batch: int = 6) -> Any:
    mp.setattr(sm, "_LIVE_STORE_ENGINE", on)
    sim = MagicsideSimulator(stim.Circuit("R 0 1\nH 0\nI[T] 0\nM 0"), batch_size=batch, seed=2)
    sim.interactive_do(stim.Circuit("R 0 1\nH 0\nI[T] 0"))
    sim._flush_magic()
    bad = sim._live_exps[bad_shot]
    bad.weights = bad.weights.copy()
    bad.weights[-1] = complex(np.nan, 0.0)
    sim._live_exps[bad_shot] = bad
    return sim


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("bad_shot", [0, 3, 5])
def test_a_failing_shot_leaves_the_earlier_shots_updated_and_the_later_ones_not(
    on: bool, bad_shot: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-shot fallback loop must run in ascending shot order (a reversed loop would update
    the shots *after* the failing one instead); the engine must leave the same shots."""
    from stimside._magic.channels import CompiledCliffordFactorList
    from stimside._magic.live_store_test import _first_difference

    sim = _sim_with_a_bad_shot(on, monkeypatch, bad_shot)
    n = sim._eff_nq
    fac = CompiledCliffordFactorList(3, None, ())  # multiplies every shot by omega^3
    before = [sim._live_exps[b].copy() for b in range(6)]
    with pytest.raises(FloatingPointError, match="Non-finite"):
        # the omega multiply alone does not check finiteness: add a quarter turn that does
        sim._apply_factor_lists([_with_quarter_turn(n, fac)], np.zeros(6, dtype=np.int32), False)
    for b in range(6):
        changed = _first_difference(before[b], sim._live_exps[b]) is not None
        assert changed == (b < bad_shot), (on, b, bad_shot)


def _with_quarter_turn(n: int, fac: Any) -> Any:
    from stimside._magic.channels import CompiledCliffordFactorList
    from stimside._magic.gates import CompiledPauliAxis
    from stimside._magic.live_store_test import _nonzero_axis

    ax = CompiledPauliAxis(*_nonzero_axis(np.random.default_rng(0), n))
    return CompiledCliffordFactorList(fac.v, None, (ax,))
