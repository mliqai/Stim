"""The private engine switch of `simulator_magic`: results must not depend on it.

With `_LIVE_STORE_ENGINE` on, the live per-shot expansions live in the C++ MagicBatchEngine behind
the `_live_exps` proxy. Every golden case is recomputed in that mode and compared with the blessed
digests, and the engine commands are checked to really run (a switch that silently fell back to
the Python list would pass the digest comparison vacuously).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside._native.magic_engine as me
import stimside.simulator_magic as sm
from stimside import golden_test
from stimside._magic.live_store import LiveExpansions
from stimside._magic.phase_qfe import ShotExpansion
from stimside.simulator_magic import MagicsideSimulator

_PAULI_CIRCUIT = "R 0 1 2\nH 0\nI[T] 0\nCX 0 1\nX_ERROR(0.3) 0 1\nM 0 1 2\nDETECTOR rec[-1]\n"


class _CountingStore(LiveExpansions):
    """A `LiveExpansions` that counts how often the engine Pauli command ran."""

    calls = 0

    def apply_pauli_mask(self, axis, mask) -> None:  # type: ignore[no-untyped-def]
        type(self).calls += 1
        super().apply_pauli_mask(axis, mask)


def _engine_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    monkeypatch.setattr(sm, "LiveExpansions", _CountingStore)
    _CountingStore.calls = 0


def test_every_golden_digest_is_unchanged_with_the_engine_switch_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine_on(monkeypatch)
    actual = golden_test._compute_all_digests()
    expected = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
    assert sum("magic_ds" in modes for modes in expected.values()) >= 10
    assert _CountingStore.calls >= 20  # heralded noise / feedback really went through the engine


def _noisy_run(engine: bool, monkeypatch: pytest.MonkeyPatch, seed: int = 5):
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=16, seed=seed)
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
    sim.finish_interactive_run()
    return sim, [sim.get_qfe_expansion(b)[1] for b in range(16)]


def test_a_noisy_magic_run_has_identical_expansions_and_rng_state_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref, ref_exps = _noisy_run(False, monkeypatch)
    eng, eng_exps = _noisy_run(True, monkeypatch)
    assert isinstance(eng._live_exps, LiveExpansions) and isinstance(ref._live_exps, list)
    assert [c.tobytes() for c in eng._measurement_columns] == [
        c.tobytes() for c in ref._measurement_columns
    ]
    assert eng._log_branch_masses.tobytes() == ref._log_branch_masses.tobytes()
    assert eng.np_rng.bit_generator.state == ref.np_rng.bit_generator.state
    for a, b in zip(ref_exps, eng_exps):
        assert a.weights.view(np.uint64).tobytes() == b.weights.view(np.uint64).tobytes()
        assert a.h_words.tobytes() == b.h_words.tobytes() and a.ell.tobytes() == b.ell.tobytes()
        assert a.A_words.tobytes() == b.A_words.tobytes() and a.d == b.d and a.chi == b.chi


def test_clear_reuses_one_engine_and_resets_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=1)
    assert isinstance(sim._live_exps, LiveExpansions)
    engine = sim._live_exps.engine
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
    sim.finish_interactive_run()
    sim.clear()
    assert isinstance(sim._live_exps, LiveExpansions)
    assert sim._live_exps.engine is engine  # reset in place, not recreated
    zero = ShotExpansion.zero(sim._eff_nq, atol=sim.atol)
    assert all(
        e.weights.tobytes() == zero.weights.tobytes() and e.chi == 1 and e.d == 0
        for e in sim._live_exps
    )


@pytest.mark.skipif(
    sm._LIVE_STORE_ENGINE is not False,
    reason="pins the OFF default: forcing the switch ON (to run the suite on the engine) or "
    "flipping the default is the owner's decision and retires this pin, not a regression",
)
def test_the_switch_is_private_and_off_by_default() -> None:
    import stimside

    assert sm._LIVE_STORE_ENGINE is False
    assert not any("live" in n.lower() or "engine" in n.lower() for n in stimside.__all__)
    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0\n"), batch_size=2, seed=1)
    assert isinstance(sim._live_exps, list)  # the reference path is the default


def test_with_the_switch_on_the_realization_style_chi_zero_assignment_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The white-box flow of simulator_magic_realization_test.py, on the engine-backed store.
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0"), batch_size=2, seed=1)
    sim.interactive_do(stim.Circuit("H 0\nI[T] 0\nH 0"))
    sim._flush_magic()
    assert isinstance(sim._live_exps, LiveExpansions)
    sim._live_exps[0] = ShotExpansion.from_components((), n_qubits=sim._eff_nq, merge=False)
    sim.interactive_do(stim.Circuit("M 0"))
    sim.finish_interactive_run()
    assert sim.get_branch_mass(0) == 0.0
    assert sim.get_branch_mass(1) > 0.0


_FEEDBACK_CIRCUIT = stim.Circuit(
    "R 0 1\nH 0\nI[T] 0\nH 0\nH 1\nI[T] 1\nH 1\nM 1\nCX rec[-1] 0\n"
)


def _feedback_run(
    engine: bool, monkeypatch: pytest.MonkeyPatch, seen: list[np.ndarray]
) -> tuple[MagicsideSimulator, np.ndarray, np.ndarray]:
    """Run to the feedback gate, then apply it through a per-group mask (`mask_g`)."""
    real = MagicsideSimulator._apply_axis_to_shots

    def spy(self: MagicsideSimulator, ax: Any, mask: np.ndarray) -> None:
        seen.append(np.array(mask, copy=True))
        real(self, ax, mask)

    with monkeypatch.context() as patch:  # each run wraps the real method, never an older spy
        patch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        patch.setattr(MagicsideSimulator, "_apply_axis_to_shots", spy)
        sim = MagicsideSimulator(_FEEDBACK_CIRCUIT, batch_size=8, seed=3)
        sim.interactive_do(_FEEDBACK_CIRCUIT[:-1])
        measured = sim._measurement_columns[-1].copy()
        mask_g = np.array([[True, False] * 4])  # one group (one CX pair), conditioned per shot
        seen.clear()
        sim.do_conditional_clifford(_FEEDBACK_CIRCUIT[-1], mask_g)
        sim._flush_magic()
    return sim, measured, mask_g[0]


def test_the_mask_g_conditioned_feedback_pauli_reaches_the_engine_with_the_anded_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_seen: list[np.ndarray] = []
    eng_seen: list[np.ndarray] = []
    ref, measured, mask_g = _feedback_run(False, monkeypatch, ref_seen)
    eng, measured_eng, _ = _feedback_run(True, monkeypatch, eng_seen)
    assert measured.tobytes() == measured_eng.tobytes()
    want = measured & mask_g
    # The path is only exercised if mask_g really removes shots the measurement alone selects.
    assert (measured & ~mask_g).any() and want.any() and not want.all()
    assert len(ref_seen) == len(eng_seen) == 1
    assert ref_seen[0].tolist() == eng_seen[0].tolist() == want.tolist()
    assert isinstance(eng._live_exps, LiveExpansions) and isinstance(ref._live_exps, list)
    assert all(_same_bytes(ref._live_exps[b], eng._live_exps[b]) for b in range(8))


def _same_bytes(x: ShotExpansion, y: ShotExpansion) -> bool:
    return (
        x.weights.view(np.uint64).tobytes() == y.weights.view(np.uint64).tobytes()
        and x.h_words.tobytes() == y.h_words.tobytes() and x.ell.tobytes() == y.ell.tobytes()
        and x.log_mass.hex() == y.log_mass.hex() and x.chi == y.chi and x.d == y.d
    )


def test_copy_copy_with_the_switch_on_gives_the_copy_its_own_engine_and_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=2)
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT)[:4])
    sim._flush_magic()
    twin = copy.copy(sim)
    assert isinstance(sim._live_exps, LiveExpansions)
    assert isinstance(twin._live_exps, LiveExpansions)
    assert twin._live_exps is twin._live_store and twin._live_store is not sim._live_store
    assert twin._live_exps.engine is not sim._live_exps.engine
    before = [sim._live_exps[b] for b in range(4)]
    assert any(e.chi > 1 or e.h_words.any() for e in before)  # not just the zero state
    assert all(_same_bytes(a, twin._live_exps[b]) for b, a in enumerate(before))
    twin._live_exps.close()  # the copy lets go of its engine ...
    twin.clear()  # ... and is cleared, which resets its own engine only
    assert all(_same_bytes(a, sim._live_exps[b]) for b, a in enumerate(before))
    assert sim._live_exps.engine is not twin._live_exps.engine


def test_copy_copy_with_the_switch_off_is_still_the_default_shallow_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", False)  # pinned: the default may flip to ON
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=2, seed=2)
    twin = copy.copy(sim)
    assert type(twin) is type(sim)
    assert twin.__dict__.keys() == sim.__dict__.keys()
    assert all(twin.__dict__[k] is v for k, v in sim.__dict__.items())  # nothing is rebuilt


def test_copy_copy_after_a_compiled_run_replaced_the_store_by_a_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=2, seed=2)
    sim._live_exps = [sim._live_exps[b] for b in range(2)]  # what the compiled paths leave
    twin = copy.copy(sim)
    assert twin._live_exps is sim._live_exps  # the list is shared, as in the default copy
    assert twin._live_store is not sim._live_store  # but never the native store
    assert isinstance(twin._live_store, LiveExpansions)


def test_a_library_that_cannot_load_makes_the_switch_fall_back_to_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(force_recompile: bool = False) -> Any:
        raise OSError("libmagic_kernels.so: cannot open shared object file")

    me.numerics_match_numpy.cache_clear()
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    monkeypatch.setattr(me, "get_magic_engine_lib", broken)
    try:
        sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=5)
        assert isinstance(sim._live_exps, list)  # silent fallback, not an OSError
        sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
        sim.finish_interactive_run()
        sim.clear()
        assert isinstance(sim._live_exps, list)
    finally:
        monkeypatch.undo()
        me.numerics_match_numpy.cache_clear()
    assert me.numerics_match_numpy()
