"""MagicsideSimulator realization routing, strict faults and branch-mass pins (WP-22)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.simulator_magic import MagicsideSimulator

_EMPTY_FORMS = [
    {}, [{}], {"faults": {}}, {"paulis": {}}, {"sweep_bits": []}, {"sweep_bits": {}},
    {"resets": {}}, {"measurements": {}}, {"quantum_measurements": {}}, {"pauli_noise": {}},
    {"sweep": []}, {"faults": []}, {"faults": None},
    {"faults": {}, "paulis": {}, "resets": {}, "sweep_bits": []},
    {"resets": []}, {"resets": ()}, {"paulis": []}, {"paulis": ()},
    {"faults": [], "paulis": (), "resets": [], "sweep_bits": ()},
]
_P0 = (1.0 + 0.5**0.5) / 2.0  # P(Z = 0) after H, T, H.


@pytest.mark.parametrize("fr", _EMPTY_FORMS, ids=repr)
def test_all_empty_realization_forms_behave_like_none(fr) -> None:
    circ = stim.Circuit("H 0 1\nI[T] 0\nCX 0 1\nI[T] 1\nX_ERROR(0.3) 0 1\nM 0 1")
    b = 64
    if isinstance(fr, list):
        fr = fr * b

    def recs(real, **kw):
        sim = MagicsideSimulator(circ, batch_size=b, seed=5, fault_realizations=real, **kw)
        sim.run()
        return sim.get_final_measurement_records()

    np.testing.assert_array_equal(recs(fr), recs(None))
    # Empty non-fault sections also must not trip the trivial-handler / sync restrictions.
    np.testing.assert_array_equal(
        recs(fr, sync_tableside_rng=True), recs(None, sync_tableside_rng=True)
    )


_EMPTY_LISTS = [{"resets": []}, {"paulis": ()}, {"paulis": [], "resets": ()}]


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
@pytest.mark.parametrize("backend", ["STREAM", "QUERY"])
@pytest.mark.parametrize("fr", _EMPTY_LISTS, ids=repr)
def test_empty_list_sections_equal_none_in_every_fault_mode_and_backend(
    fr, backend: str, fault_mode: str
) -> None:
    circ = stim.Circuit("H 0 1\nI[T] 0\nCX 0 1\nI[T] 1\nX_ERROR(0.3) 0 1\nM 0 1")

    def run(real):
        sim = MagicsideSimulator(
            circ, batch_size=64, seed=5, fault_mode=fault_mode, backend=backend,
            fault_realizations=real,
        )
        sim.run()
        return sim.get_final_measurement_records(), [sim.get_branch_mass(s) for s in range(64)]

    for shaped in (fr, [fr] * 64):
        (rec, mass), (rec0, mass0) = run(shaped), run(None)
        np.testing.assert_array_equal(rec, rec0)
        np.testing.assert_array_equal(mass, mass0)


@pytest.mark.parametrize("section", ["paulis", "resets"])
def test_zero_dimensional_array_section_keeps_its_one_dimensional_error(section: str) -> None:
    # A 0-d array is not an empty section: it must not be dropped (nor raise a bare TypeError
    # from len()); the downstream validation says "array must be 1D".
    circ = stim.Circuit("H 0\nI[T] 0\nM 0")
    with pytest.raises(ValueError, match="array must be 1D"):
        sim = MagicsideSimulator(circ, batch_size=1, fault_realizations={section: np.array(0)})
        sim.run()


def test_zero_dimensional_sweep_bits_is_still_rejected() -> None:
    # Pre-existing: the sweep normaliser rejects it (with a TypeError) before the empty filter.
    circ = stim.Circuit("H 0\nI[T] 0\nM 0")
    with pytest.raises((TypeError, ValueError)):
        MagicsideSimulator(circ, batch_size=1, fault_realizations={"sweep_bits": np.array(0)})


def test_empty_realization_section_keeps_constructor_sweep_bits() -> None:
    circ = stim.Circuit("CX sweep[0] 0\nCX sweep[1] 1\nM 0 1")
    sim = MagicsideSimulator(
        circ, batch_size=2, seed=1, sweep_bits=[1, 1],
        fault_realizations=[{"sweep_bits": []}, {"sweep_bits": [0, 1]}],
    )
    sim.run()
    assert sim.get_final_measurement_records().tolist() == [[True, True], [False, True]]


def test_simulator_fault_realization_list_must_match_slot_count_exactly() -> None:
    # The simulator is strict (strict_faults=True): a list longer than the slot count is an
    # error, unlike query.EvaluateQueries, which tolerates it (see _magic/query_strict_test.py).
    circ = stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.2, Z)] 0\nH 0\nM 0")
    sim = MagicsideSimulator(circ, batch_size=1, seed=1, fault_realizations=[{"faults": [0]}])
    sim.run()
    assert sim.get_branch_mass(0) == 1.0
    for faults in ([0, 0], [0, 1, 0, 0]):
        with pytest.raises(ValueError, match="invalid for 1 slots"):
            MagicsideSimulator(circ, batch_size=1, fault_realizations=[{"faults": faults}])


def test_postselect_z_batch_scales_branch_mass_by_the_chosen_outcome_probability() -> None:
    for desired, want in ((0, _P0), (1, 1.0 - _P0)):
        sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0"), batch_size=2, seed=1)
        sim.interactive_do(stim.Circuit("H 0\nI[T] 0\nH 0"))
        sim.postselect_z_batch(0, np.array([True, False]), np.array([desired, desired], np.uint8))
        assert math.isclose(sim.get_branch_mass(0), want, rel_tol=1e-12)
        assert sim.get_branch_mass(1) == 1.0  # Not postselected.


def test_compiled_branch_mass_includes_the_forced_herald_probability() -> None:
    # A forced herald outcome contributes P(herald) = 0.1 (or 0.9) through mass_scale, which the
    # compiled DIRECT/SELECTOR runs fold in via _log0; QUERY computes it independently.
    circ = stim.Circuit("X_ERROR(0.2) 1\nHERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0) 0\nM 0 1")
    reals = [{"measurements": {0: h}, "paulis": {0: 0}} for h in (0, 1)]
    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            sim = MagicsideSimulator(
                circ, batch_size=2, seed=3, fault_mode=fm, backend=be, fault_realizations=reals
            )
            sim.run()
            np.testing.assert_allclose(
                [sim.get_branch_mass(0), sim.get_branch_mass(1)], [0.9, 0.1], atol=1e-12
            )


def test_live_branch_mass_of_an_empty_expansion_is_zero() -> None:
    # WHITE-BOX: chi == 0 cannot arise from unitary evolution, and no public call or random
    # circuit reaches this defensive branch, so the empty expansion is injected directly. A
    # mutant that leaves the mass at 1.0 would pass every public-API test.
    from stimside._magic.phase_qfe import ShotExpansion

    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0"), batch_size=2, seed=1)
    sim.interactive_do(stim.Circuit("H 0\nI[T] 0\nH 0"))
    sim._flush_magic()
    sim._live_exps[0] = ShotExpansion.from_components((), n_qubits=sim._eff_nq, merge=False)
    sim.interactive_do(stim.Circuit("M 0"))
    sim.finish_interactive_run()
    bit = bool(sim.get_final_measurement_records()[1, 0])
    assert sim.get_branch_mass(0) == 0.0
    assert math.isclose(sim.get_branch_mass(1), 1.0 - _P0 if bit else _P0, rel_tol=1e-12)


def test_unmatched_noisy_reset_z_is_trace_preserving_and_leaves_branch_mass_alone() -> None:
    # D-06: unlike postselect_z_batch (and a live R, whose Born-sampled outcome scales the mass),
    # an unmatched noisy reset is a channel, so it must not touch the branch mass.
    for seed in (1, 2, 3):
        sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0"), batch_size=2, seed=seed)
        sim.interactive_do(stim.Circuit("H 0\nI[T] 0\nH 0"))
        zero, one = np.array([[True, False]]), np.array([[False, True]])
        sim.do_unmatched_noisy_reset_z(zero, one)
        assert [sim.get_branch_mass(0), sim.get_branch_mass(1)] == [1.0, 1.0]
        sim.interactive_do(stim.Circuit("M 0"))
        sim.finish_interactive_run()
        assert sim.get_final_measurement_records().tolist() == [[False], [True]]
