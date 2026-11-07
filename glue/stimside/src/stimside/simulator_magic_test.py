"""Unit tests for MagicsideSimulator."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import stim

from stimside._magic.compile import CliffordFaultChannel, FaultBranch
from stimside._magic.query import (
    EvaluateQueries,
    PrepareQueries,
    QueryBudget,
    QueryBudgetExceededError,
)
from stimside.golden_test import CORPUS
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside.sampler_magic import MagicsideSampler
from stimside.simulator_magic import MagicsideSimulator


def test_statevector_and_branch_mass_across_fault_modes_and_backends() -> None:
    circ = stim.Circuit("H 0 1 2\nI[T] 0\nI[T_DAG] 1\nI[CCZ] 0 1 2\nCX 0 2")
    want = None
    for fm in ("DIRECT", "SELECTOR", "direct", "selector"):
        for be in ("STREAM", "QUERY", "stream", "query"):
            sim = MagicsideSimulator(circ, batch_size=1, fault_mode=fm, backend=be)
            assert sim.run() is None
            psi = sim.get_statevector(0)
            psi_big = sim.state_vector(0, endian="big")
            assert psi.shape == (8,) and abs(np.linalg.norm(psi) - 1.0) < 1e-12
            assert abs(sim.get_branch_mass(0) - 1.0) < 1e-12
            tab, exp = sim.get_qfe_expansion(0, trace_selectors=True)
            assert isinstance(tab, stim.Tableau)
            assert len(tab) == 3 and exp.n_qubits == 3
            assert len(sim.get_state_expansions()) == 1
            want_big = psi.reshape((2, 2, 2)).transpose((2, 1, 0)).ravel()
            np.testing.assert_allclose(psi_big, want_big, atol=1e-12)
            if want is None:
                want = psi
            else:
                ov = abs(np.vdot(want, psi))
                assert abs(ov - 1.0) < 1e-12


def test_m2d_reference_modes_handle_deterministic_magic_interference() -> None:
    circ = stim.Circuit(
        "H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\nH 0\nM 0\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    for mode in ("auto", "magicside"):
        sim = MagicsideSimulator(circ, batch_size=8, seed=11, m2d_reference_mode=mode)
        sim.run()
        assert np.all(sim.get_final_measurement_records())
        np.testing.assert_array_equal(sim.get_reference_sample(), np.array([True]))
        assert not np.any(sim.get_detector_flips(append_observables=True))
        assert not np.any(sim.get_observable_flips())

    sim_zero = MagicsideSimulator(circ, batch_size=4, seed=11, m2d_reference_mode="zero")
    sim_zero.run()
    assert np.all(sim_zero.get_detector_flips())
    assert np.all(sim_zero.get_observable_flips())

    sim_exp = MagicsideSimulator(circ, batch_size=4, seed=11, reference_sample=[1])
    sim_exp.run()
    assert not np.any(sim_exp.get_detector_flips())


def test_sweep_feedback_and_correlated_error_multi_target_behavior() -> None:
    circ = stim.Circuit(
        "CX sweep[0] 0\n"
        "H 1\n"
        "CORRELATED_ERROR(1.0) X0 Z1\n"
        "ELSE_CORRELATED_ERROR(1.0) X1\n"
        "H 1\n"
        "M 0 1\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    sw = np.array([[False], [True], [False], [True]], dtype=np.bool_)
    sim = MagicsideSimulator(circ, batch_size=4, seed=21, sweep_bits=sw)
    sim.run()
    recs = sim.get_final_measurement_records()
    want = np.array([[True, True], [False, True], [True, True], [False, True]], dtype=np.bool_)
    np.testing.assert_array_equal(recs, want)
    # Correlated error flipped both X0 and Z1 -> both detector and observable flip relative to noiseless sweep ref
    assert np.all(sim.get_detector_flips())
    assert np.all(sim.get_observable_flips())


def test_word_straddling_n66_matches_small_isomorphic_circuit() -> None:
    def _make_circ(q0: int, q1: int, q2: int, q3: int) -> stim.Circuit:
        return stim.Circuit(
            f"H {q0} {q1} {q2} {q3}\n"
            f"CX {q0} {q1} {q1} {q2} {q2} {q3}\n"
            f"I[T] {q1} {q2}\n"
            f"I[CCZ] {q0} {q1} {q3}\n"
            f"X_ERROR(0.2) {q1} {q2}\n"
            f"H {q0} {q1} {q2} {q3}\n"
            f"M {q0} {q1} {q2} {q3}"
        )

    c_big = _make_circ(0, 63, 64, 65)
    c_ref = _make_circ(0, 1, 2, 3)
    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            s_big = MagicsideSimulator(c_big, batch_size=16, seed=77, fault_mode=fm, backend=be)
            s_ref = MagicsideSimulator(c_ref, batch_size=16, seed=77, fault_mode=fm, backend=be)
            s_big.run()
            s_ref.run()
            np.testing.assert_array_equal(
                s_big.get_final_measurement_records(), s_ref.get_final_measurement_records()
            )


def test_interactive_do_seed_stream_parity_coords_and_forced_faults() -> None:
    circ = stim.Circuit(
        "QUBIT_COORDS[data](1, 2) 0\n"
        "SHIFT_COORDS(3, 0)\n"
        "QUBIT_COORDS(1, 2) 1\n"
        "H 0\n"
        "I[CLIFFORD_FAULT: (0.5, S)] 0\n"
        "I[T] 0\n"
        "I[T] 0\n"
        "H 0\n"
        "M 0"
    )
    sim_i = MagicsideSimulator(circ, batch_size=4, seed=99)
    sim_b = MagicsideSimulator(circ, batch_size=4, seed=99)
    assert sim_i.current_tableau_measurement_record().shape == (0,)
    for batch_idx in range(2):
        for op in circ:
            sim_i.interactive_do(op)
        sim_i.finish_interactive_run()
        sim_b.run()
        np.testing.assert_array_equal(
            sim_i.get_final_measurement_records(),
            sim_b.get_final_measurement_records(),
            err_msg=f"batch {batch_idx}",
        )
        assert sim_i.qubit_coords == {0: [1.0, 2.0], 1: [4.0, 2.0]}
        assert sim_i.qubit_tags[0] == "data"
        assert sim_i.get_current_circuit_time() == len(circ)
        sim_i.clear()
        sim_b.clear()

    # Forced inline fault slot realization in interactive mode + post-end magic extension
    sim_f = MagicsideSimulator(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.5, S)] 0"), batch_size=1, fault_realizations=[{0: 1}])
    sim_f.interactive_do(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.5, S)] 0"))
    sim_f.interactive_do(stim.Circuit("S 0\nH 0\nM 0"))
    sim_f.finish_interactive_run()
    assert bool(sim_f.get_final_measurement_records()[0, 0]) is True


def test_programmatic_clifford_fault_channels_and_budget_enforcement() -> None:
    ch = CliffordFaultChannel(step_index=1, branches=(FaultBranch(prob=1.0, gate_name="Z", targets=(0,)),), slot_id=7)
    circ = stim.Circuit("H 0\nTICK\nH 0\nM 0")
    for be in ("STREAM", "QUERY"):
        sim = MagicsideSimulator(circ, batch_size=2, seed=5, backend=be, fault_channels=[ch], fault_realizations={7: 1})
        sim.run()
        assert np.all(sim.get_final_measurement_records())

    hard = stim.Circuit("H 0 1 2\nI[CCZ] 0 1 2\nH 0 1 2\nM 0")
    tight = QueryBudget(max_terms=1, on_budget_exceeded="raise")
    sim_err = MagicsideSimulator(hard, backend="QUERY", query_budget=tight)
    with pytest.raises(QueryBudgetExceededError):
        sim_err.run()
    sim_fb = MagicsideSimulator(hard, batch_size=4, seed=3, backend="QUERY", query_budget=tight, on_budget_exceeded="fallback")
    sim_fb.run()
    assert sim_fb.get_final_measurement_records().shape == (4, 1)


def test_invalid_inputs_and_lifecycle_errors_raise() -> None:
    c = stim.Circuit("H 0\nM 0")
    with pytest.raises(TypeError):
        MagicsideSimulator("H 0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_cpp_kernels"):
        MagicsideSimulator(c, use_cpp_kernels=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="use_cpp_kernels"):
        MagicsideSampler(dem_decoder=None, use_cpp_kernels=False)  # type: ignore[arg-type,call-arg]
    with pytest.raises(ValueError):
        MagicsideSimulator(c, batch_size=0)
    with pytest.raises(ValueError):
        MagicsideSimulator(c, batch_size=True)
    with pytest.raises(ValueError):
        MagicsideSimulator(c, fault_mode="BAD")
    with pytest.raises(ValueError):
        MagicsideSimulator(c, backend="BAD")
    with pytest.raises(ValueError):
        MagicsideSimulator(c, m2d_reference_mode="BAD")
    with pytest.raises(ValueError):
        MagicsideSimulator(c, on_budget_exceeded="ignore")
    with pytest.raises(ValueError):
        MagicsideSimulator(c, reference_sample=[0, 1])

    sim = MagicsideSimulator(c, batch_size=2, seed=1)
    with pytest.raises(RuntimeError):
        sim.get_final_measurement_records()
    sim.run()
    with pytest.raises(ValueError):
        sim.run()
    with pytest.raises(IndexError, match="out of range"):
        sim.get_qfe_expansion(2)
    with pytest.raises(IndexError, match="out of range"):
        sim.get_qfe_expansion(-3)
    with pytest.raises(ValueError):
        sim.get_statevector(0, endian="middle")  # type: ignore[arg-type]
    sim.clear()
    with pytest.raises(ValueError):
        sim.interactive_do(stim.CircuitInstruction("X", [0]))
    sim.clear()
    sim.interactive_do(c)
    with pytest.raises(ValueError):
        sim.interactive_do(stim.CircuitInstruction("X", [1]))
    with pytest.raises(ValueError):
        sim.interactive_do(stim.CircuitInstruction("DETECTOR", [stim.target_rec(-1)]))
    sim.finish_interactive_run()
    with pytest.raises(RuntimeError):
        sim.interactive_do(stim.CircuitInstruction("I", [0]))


def test_incompatible_modes_realization_validation_and_stream_pinning() -> None:
    c = stim.Circuit("H 0\nM 0")
    lh = LeakageUint8Magic()
    clh = lh.compile_op_handler(circuit=c, batch_size=2)
    for kw in (
        {"backend": "QUERY", "op_handler": lh},
        {"backend": "QUERY", "compiled_op_handler": clh},
        {"backend": "QUERY", "sync_tableside_rng": True},
        {"fault_mode": "SELECTOR", "op_handler": lh},
        {"fault_mode": "SELECTOR", "compiled_op_handler": clh},
        {"fault_mode": "SELECTOR", "sync_tableside_rng": True},
    ):
        with pytest.raises(ValueError):
            MagicsideSimulator(c, batch_size=2, **kw)  # type: ignore[arg-type]

    for kw_i in ({"backend": "QUERY"}, {"fault_mode": "SELECTOR"}):
        sim_i = MagicsideSimulator(c, batch_size=2, **kw_i)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sim_i.interactive_do(stim.CircuitInstruction("H", [0]))
        with pytest.raises(ValueError):
            sim_i.finish_interactive_run()

    for bad_real in ({"measurements": {0: 1}}, {"paulis": {0: 1}}, {"resets": {0: 1}}, {"bogus": 1}):
        with pytest.raises(ValueError):
            MagicsideSimulator(c, batch_size=1, op_handler=lh, fault_realizations=bad_real)
    sim_ir = MagicsideSimulator(c, batch_size=1, fault_realizations={"measurements": {0: 1}})
    with pytest.raises(ValueError):
        sim_ir.interactive_do(c)

    cpin = stim.Circuit("H 0\nDEPOLARIZE1(0.3) 1\nM 0 1")
    for real in (
        {"measurements": {0: 1, 1: 1}, "paulis": {0: 1}},
        {"measurements": [0, 1], "paulis": {0: "X"}},
    ):
        want_r = np.array([[real["measurements"][0] == 1, True]] * 2, dtype=np.bool_)  # type: ignore[index]
        ref_m = None
        for fm, be in (("DIRECT", "STREAM"), ("SELECTOR", "STREAM"), ("DIRECT", "QUERY"), ("SELECTOR", "QUERY")):
            sp = MagicsideSimulator(cpin, batch_size=2, fault_mode=fm, backend=be, fault_realizations=real)
            sp.run()
            np.testing.assert_array_equal(sp.get_final_measurement_records(), want_r)
            m0 = sp.get_branch_mass(0)
            if ref_m is None:
                ref_m = m0
                assert abs(m0 - 0.5) < 1e-12
            else:
                assert abs(m0 - ref_m) < 1e-12


def test_trace_selectors_normalize_unleaked_records_and_statevector_cap() -> None:
    circ = stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.4, X)] 1")
    for be in ("STREAM", "QUERY"):
        sim = MagicsideSimulator(
            circ,
            batch_size=2,
            fault_mode="SELECTOR",
            backend=be,
            fault_realizations={0: 1},
            record_unleaked_to_leaked=True,
        )
        assert sim.num_qubits_in_new_circuit() == 2
        u2l = sim.run()
        assert u2l is not None and len(u2l) == 2 and all(len(a) == 0 for a in u2l)
        tab_t, exp_t = sim.get_qfe_expansion(0, trace_selectors=True)
        tab_f, exp_f = sim.get_qfe_expansion(0, trace_selectors=False)
        assert isinstance(tab_t, stim.Tableau) and isinstance(tab_f, stim.Tableau)
        assert len(tab_t) == 2 and exp_t.n_qubits == 2
        assert len(tab_f) == 3 and exp_f.n_qubits == 3
        assert len(sim.get_state_expansions(trace_selectors=False)) == 2
        psi_t = sim.get_statevector(0, trace_selectors=True)
        psi_f = sim.get_statevector(0, trace_selectors=False)
        assert psi_t.shape == (4,) and psi_f.shape == (8,)
        assert abs(np.linalg.norm(psi_f) - 1.0) < 1e-12
        np.testing.assert_allclose(psi_f[:4], np.zeros(4), atol=1e-12)
        assert abs(abs(np.vdot(psi_t, psi_f[4:])) - 1.0) < 1e-12

    sn = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=1, fault_realizations={"measurements": {0: 1}})
    sn.run()
    psi_u = sn.get_statevector(0, normalize=False)
    psi_n = sn.get_statevector(0, normalize=True)
    assert abs(float(np.vdot(psi_u, psi_u).real) - 0.5) < 1e-12
    assert abs(float(np.vdot(psi_n, psi_n).real) - 1.0) < 1e-12

    big_sel = stim.Circuit("H 0 25\nI[CLIFFORD_FAULT: (0.2, Z)] 0\nI[CLIFFORD_FAULT: (0.2, Z)] 1")
    sim_cap = MagicsideSimulator(big_sel, batch_size=1, fault_mode="SELECTOR")
    sim_cap.run()
    with pytest.raises(ValueError):
        sim_cap.get_statevector(0, trace_selectors=False)


def test_fault_channels_beyond_circuit_qubits_and_heralded_invert_ignored() -> None:
    ch = CliffordFaultChannel(step_index=0, branches=(FaultBranch(prob=1.0, gate_name="X", targets=(2,)),))
    for base, want_nq in ((stim.Circuit(), 0), (stim.Circuit("I 0"), 1)):
        for fm in ("DIRECT", "SELECTOR"):
            for be in ("STREAM", "QUERY"):
                sim = MagicsideSimulator(base, batch_size=1, fault_mode=fm, backend=be, fault_channels=[ch])
                assert sim.num_qubits == want_nq
                sim.run()
                psi = sim.get_statevector(0)
                assert psi.shape == (8,) and abs(abs(psi[4]) - 1.0) < 1e-12

    hc = stim.Circuit(
        "HERALDED_ERASE(0.0) 0 !1\n"
        "HERALDED_ERASE(1.0) 2 !3\n"
        "HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0) 0 !1\n"
        "HERALDED_PAULI_CHANNEL_1(0.25, 0.25, 0.25, 0.25) 2 !3\n"
        "DETECTOR rec[-8]\nDETECTOR rec[-7]\nDETECTOR rec[-6]\nDETECTOR rec[-5]\n"
        "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]"
    )
    stim_recs = hc.compile_sampler(seed=1).sample(shots=8)
    stim_dets = hc.compile_m2d_converter().convert(measurements=stim_recs, separate_observables=False)
    for kw in (
        {"fault_mode": "DIRECT", "backend": "STREAM"},
        {"fault_mode": "SELECTOR", "backend": "STREAM"},
        {"fault_mode": "DIRECT", "backend": "QUERY"},
        {"fault_mode": "SELECTOR", "backend": "QUERY"},
        {"sync_tableside_rng": True},
        {"op_handler": LeakageUint8Magic(), "sync_tableside_rng": False},
        {"op_handler": LeakageUint8Magic(), "sync_tableside_rng": True},
    ):
        sh = MagicsideSimulator(hc, batch_size=8, seed=19, **kw)  # type: ignore[arg-type]
        sh.run()
        np.testing.assert_array_equal(sh.get_final_measurement_records(), stim_recs)
        np.testing.assert_array_equal(sh.get_detector_flips(), stim_dets)

    sh_i = MagicsideSimulator(hc, batch_size=8, seed=19)
    for op in hc:
        sh_i.interactive_do(op)
    sh_i.finish_interactive_run()
    np.testing.assert_array_equal(sh_i.get_final_measurement_records(), stim_recs)
    np.testing.assert_array_equal(sh_i.get_detector_flips(), stim_dets)


def test_partial_realizations_custom_slot_ids_empty_circuit_and_max_components() -> None:
    cp = stim.Circuit("I[CLIFFORD_FAULT: (0.9, X)] 0\nI[CLIFFORD_FAULT: (0.5, X)] 1\nX_ERROR(0.9) 2\nX_ERROR(0.5) 3\nM 0 1 2 3")
    for kw_p in (
        {"fault_mode": "DIRECT", "backend": "STREAM", "fault_realizations": {"faults": {0: 0}, "paulis": {0: 0}}},
        {"fault_mode": "SELECTOR", "backend": "STREAM", "fault_realizations": {"faults": {0: 0}, "paulis": {0: 0}}},
        {"op_handler": LeakageUint8Magic(), "fault_realizations": {0: 0}},
    ):
        sp = MagicsideSimulator(cp, batch_size=128, seed=31, **kw_p)  # type: ignore[arg-type]
        sp.run()
        rp = sp.get_final_measurement_records()
        assert not np.any(rp[:, 0]) and 0.25 < float(np.mean(rp[:, 1])) < 0.75
        if "op_handler" not in kw_p:
            assert not np.any(rp[:, 2]) and 0.25 < float(np.mean(rp[:, 3])) < 0.75

    chs = [
        CliffordFaultChannel(step_index=0, branches=(FaultBranch(1.0, "X", (0,)),), slot_id=10),
        CliffordFaultChannel(step_index=0, branches=(FaultBranch(1.0, "X", (1,)),), slot_id=20),
    ]
    cs = stim.Circuit("M 0 1")
    for kw_s in ({"backend": "STREAM"}, {"backend": "QUERY"}, {"op_handler": LeakageUint8Magic()}):
        ss = MagicsideSimulator(
            cs, batch_size=2, fault_channels=chs, fault_realizations=[None, {10: 0, 20: 1}], **kw_s  # type: ignore[arg-type]
        )
        ss.run()
        np.testing.assert_array_equal(ss.get_final_measurement_records(), np.array([[True, True], [False, True]]))
        with pytest.raises(ValueError):
            MagicsideSimulator(cs, batch_size=1, fault_channels=chs, fault_realizations={999: 1}, **kw_s)  # type: ignore[arg-type]

    fc0 = [CliffordFaultChannel(step_index=0, branches=(FaultBranch(1.0, "X", (0,)),))]
    se = MagicsideSimulator(stim.Circuit(), batch_size=1, op_handler=LeakageUint8Magic(), fault_channels=fc0)
    se.run()
    assert abs(abs(se.get_statevector(0)[1]) - 1.0) < 1e-12
    se_i = MagicsideSimulator(stim.Circuit(), batch_size=1, fault_channels=fc0)
    se_i.finish_interactive_run()
    assert abs(abs(se_i.get_statevector(0)[1]) - 1.0) < 1e-12

    hard = stim.Circuit("H 0 1 2\nI[CCZ] 0 1 2\nH 0 1 2\nM 0")
    for kw_m in (
        {"fault_mode": "DIRECT", "backend": "STREAM"},
        {"fault_mode": "SELECTOR", "backend": "STREAM"},
        {"fault_mode": "DIRECT", "backend": "QUERY"},
        {"op_handler": LeakageUint8Magic()},
    ):
        with pytest.raises(QueryBudgetExceededError):
            MagicsideSimulator(hard, batch_size=2, max_components=1, on_budget_exceeded="error", **kw_m).run()  # type: ignore[arg-type]
    sf = MagicsideSimulator(hard, batch_size=2, backend="QUERY", max_components=1, on_budget_exceeded="fallback_stream")
    sf.run()
    assert sf.get_final_measurement_records().shape == (2, 1)


def test_zero_born_reset_and_quantum_measurement_stream_parity() -> None:
    c_det = stim.Circuit("I[CLIFFORD_FAULT: (0.2, Z)] 0\nX_ERROR(0.2) 0\nR 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    c_mr, c_qm = stim.Circuit("H 0\nM(0.15) 0\nR 0\nM 0"), stim.Circuit("M(0.25) 0\nDETECTOR rec[-1]")
    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            sd = MagicsideSimulator(c_det, batch_size=2, fault_mode=fm, backend=be, fault_realizations=[{"faults": {0: 0}, "paulis": {0: 0}, "resets": {0: 1}}, {"faults": {0: 0}, "paulis": {0: 0}, "resets": {0: 0}}], seed=0)
            sd.run()
            assert [sd.get_branch_mass(0), sd.get_branch_mass(1)] == [0.0, 1.0] and sd.state_vector(0).shape == (2,) and not np.any(sd.state_vector(0))
            assert np.allclose(sd.peek_pauli_born_batch([0], "X"), [0.0, 0.5]) and not np.any(sd.get_detector_flips(append_observables=True))
        for seed, exp_m in ((0, 0.0), (12, 0.5)):
            runs = [MagicsideSimulator(c_mr, fault_mode=fm, backend=be, seed=seed, fault_realizations=[{"measurements": {0: 0}, "resets": {0: 1}}]) for be in ("STREAM", "QUERY")]
            for s_run in runs:
                s_run.run()
            assert all(abs(s_run.get_branch_mass(0) - exp_m) < 1e-12 for s_run in runs)
            np.testing.assert_array_equal(runs[0].get_final_measurement_records(), runs[1].get_final_measurement_records())
        for be in ("STREAM", "QUERY"):
            sqm = MagicsideSimulator(c_qm, fault_mode=fm, backend=be, fault_realizations=[{"quantum_measurements": {0: 1}}])
            sqm.run()
            assert sqm.get_branch_mass(0) == 0.0 and not np.any(sqm.state_vector(0))
        c_qmp = stim.Circuit("H 0\nM(0.25) 0\nM 0")
        reals_qmp = [{"quantum_measurements": {0: 0}, "measurements": {0: 0, 1: 0}}, {"quantum_measurements": {0: 1}, "measurements": {0: 0, 1: 1}}]
        pair_qmp = [MagicsideSimulator(c_qmp, batch_size=2, fault_mode=fm, backend=be, fault_realizations=reals_qmp, seed=5) for be in ("STREAM", "QUERY")]
        for sq_sim in pair_qmp:
            sq_sim.run()
        np.testing.assert_array_equal(pair_qmp[0].get_final_measurement_records(), pair_qmp[1].get_final_measurement_records())
        np.testing.assert_allclose([pair_qmp[1].get_branch_mass(0), pair_qmp[1].get_branch_mass(1)], [0.375, 0.125], atol=1e-12)


def test_m2d_per_shot_fault_realization_sweep_bits_and_override() -> None:
    c_sw = stim.Circuit("CX sweep[0] 0\nCX sweep[1] 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    cases = ((None, [{"sweep_bits": [1]}, {"sweep_bits": [0]}], [[True, False], [False, False]]), ([0, 1], [{"sweep_bits": [1]}, {}, {"sweep_bits": {1: 0, 3: 1}}], [[True, False], [False, True], [False, False]]))
    for ctor_sw, reals, exp_r in cases:
        for mode in ("auto", "stim", "magicside"):
            for be in ("STREAM", "QUERY"):
                sm = MagicsideSimulator(c_sw, batch_size=len(reals), backend=be, m2d_reference_mode=mode, sweep_bits=ctor_sw, fault_realizations=reals)
                sm.run()
                np.testing.assert_array_equal(sm.get_final_measurement_records(), exp_r)
                assert not np.any(sm.get_detector_flips()) and not np.any(sm.get_observable_flips())
    c_magic_sw = stim.Circuit("CX sweep[0] 0\nH 0\nI[T] 0\nI[T] 0\nS_DAG 0\nH 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            sm_m = MagicsideSimulator(c_magic_sw, batch_size=2, fault_mode=fm, backend=be, fault_realizations=[{"sweep_bits": [0]}, {"sweep_bits": [1]}], seed=1)
            sm_m.run()
            np.testing.assert_array_equal(sm_m.get_final_measurement_records(), [[False], [True]])
            assert not np.any(sm_m.get_detector_flips()) and not np.any(sm_m.get_observable_flips())


def test_noiseless_ref_conditioning_collapse_semantics() -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
    from stimside.simulator_coset import CosetsideSimulator
    from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import _strip_circuit_for_noiseless_ref

    for cond_op in (
        "I[CONDITIONED_ON_SELF: 0 1] 0",
        "I[CONDITIONED_ON_OTHER: 0 1 : 0] 1",
        "CZ[CONDITIONED_ON_PAIR: (0, U) (1, U)] 0 1",
        "CZ[CONDITIONED_ON_PAIR: (0, 2)] 0 1",
    ):
        cliff = stim.Circuit(f"H 0\n{cond_op}\nH 0\nM 0 1")
        cos = CosetsideSimulator(
            cliff,
            compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=cliff, batch_size=256),
            batch_size=256,
            seed=7,
        )
        cos.run()
        # Z-collapsing q0 in |+> destroys X coherence -> M 0 is ~50/50, not deterministic 0.
        assert 64 < int(np.count_nonzero(cos.get_final_measurement_records()[:, 0])) < 192

        magic = stim.Circuit(f"H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\n{cond_op}\nH 0\nM 0 1")
        stripped, needs_leak = _strip_circuit_for_noiseless_ref(magic)
        assert needs_leak
        assert cond_op in str(stripped)
        sim = MagicsideSimulator(magic, op_handler=LeakageUint8Magic(), batch_size=1, seed=0, m2d_reference_mode="magicside")
        sim.run()
        np.testing.assert_array_equal(sim.get_reference_sample(), sim.get_final_measurement_records()[0])

    for u_op, bare in (
        ("Z[CONDITIONED_ON_SELF: U] 0", "Z 0"),
        ("Z[CONDITIONED_ON_OTHER: U : 0] 1", "Z 1"),
        ("CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1", "CZ 0 1"),
        ("CZ[CONDITIONED_ON_PAIR: (2, U) (U, U)] 0 1", "CZ 0 1"),
    ):
        magic_u = stim.Circuit(f"H 0\nI[T] 0\nI[T_DAG] 0\n{u_op}\nH 0\nM 0 1")
        stripped_u, needs_leak_u = _strip_circuit_for_noiseless_ref(magic_u)
        assert not needs_leak_u
        assert bare in str(stripped_u) and "CONDITIONED_ON" not in str(stripped_u)

    stripped_drop, needs_leak_drop = _strip_circuit_for_noiseless_ref(
        stim.Circuit("H 0\nX[CONDITIONED_ON_SELF: 2] 0\nCZ[CONDITIONED_ON_PAIR: (2, U)] 0 1\nH 0\nM 0 1")
    )
    assert not needs_leak_drop and "CONDITIONED_ON" not in str(stripped_drop) and "CZ" not in str(stripped_drop)


@pytest.mark.parametrize("gate", ["SPP", "SPP_DAG"])
def test_spp_terms_on_leaked_qubits_align_across_magic_and_tableside(gate: str) -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8 as LeakageUint8Tableau
    from stimside.simulator_tableau import TablesideSimulator

    # 1. Stochastic lockstep parity with TablesideSimulator across minority & majority active fractions + `!`.
    for p_leak in (0.25, 0.75):
        circ = stim.Circuit(
            f"R 0 1 2 3\nH 1\n"
            f"I[LEAKAGE_TRANSITION_1: ({p_leak}, U-->2)] 0\n"
            f"{gate} Z0*X2 !Y1*Z3\n"
            f"{gate} Z0*X2 X3\n"
            f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2): 0] 0\n"
            f"H 1\nM 1 2 3"
        )
        tab = TablesideSimulator(
            circuit=circ,
            compiled_op_handler=LeakageUint8Tableau().compile_op_handler(circuit=circ, batch_size=1),
            batch_size=32,
            seed=19,
            sync_tableside_rng=True,
        )
        mag = MagicsideSimulator(
            circ, op_handler=LeakageUint8Magic(), batch_size=32, seed=19, sync_tableside_rng=True
        )
        tab.run()
        mag.run()
        t_rec, m_rec = tab.get_final_measurement_records(), mag.get_final_measurement_records()
        assert 0 < int(np.count_nonzero(t_rec[:, 0])) < 32
        np.testing.assert_array_equal(m_rec, t_rec)

        # 2. Magic statevector check on both minority and majority leakage batches.
        m_circ = stim.Circuit(
            f"R 0 1 2\nH 1 2\nI[T] 1\nI[T_DAG] 2\n"
            f"I[LEAKAGE_TRANSITION_1: ({p_leak}, U-->2)] 0\n"
            f"{gate} !Z0*X1 Y1*Z2\n"
            f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2): 0] 0"
        )
        msim = MagicsideSimulator(m_circ, op_handler=LeakageUint8Magic(), batch_size=16, seed=23)
        msim.run()
        leaked = msim.get_final_measurement_records()[:, 0]
        assert 0 < int(np.count_nonzero(leaked)) < 16
        ref_kept = MagicsideSimulator(stim.Circuit(f"R 0 1 2\nH 1 2\nI[T] 1\nI[T_DAG] 2\n{gate} !Z0*X1 Y1*Z2"))
        ref_skip = MagicsideSimulator(stim.Circuit(f"R 0 1 2\nH 1 2\nI[T] 1\nI[T_DAG] 2\n{gate} Y1*Z2"))
        ref_kept.run()
        ref_skip.run()
        psi_kept, psi_skip = ref_kept.get_statevector(0), ref_skip.get_statevector(0)
        for s in range(16):
            psi_s = msim.get_statevector(s)
            want = psi_skip if leaked[s] else psi_kept
            assert np.allclose(psi_s, want, atol=1e-12)

    # 3. unconditional_condition_on_U=False still applies all SPP terms unconditionally.
    c_u_false = stim.Circuit(f"R 0 1\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n{gate} Z0*X1 Z0*X1\nM 1")
    sim_uf = MagicsideSimulator(c_u_false, op_handler=LeakageUint8Magic(unconditional_condition_on_U=False), batch_size=4, seed=3)
    sim_uf.run()
    assert np.all(sim_uf.get_final_measurement_records()[:, 0])


def test_get_qfe_expansion_returns_forward_stim_tableau_and_roundtrips_into_queries() -> None:
    empty = MagicsideSimulator(stim.Circuit(), batch_size=1)
    empty.run()
    tab0, exp0 = empty.get_qfe_expansion(0)
    assert isinstance(tab0, stim.Tableau) and len(tab0) == exp0.n_qubits == 1

    circ = stim.Circuit("H 0\nCX 0 1\nS 0\nI[T] 0")
    sim = MagicsideSimulator(circ, batch_size=1)
    sim.run()
    qfe = sim.get_qfe_expansion(0)
    assert isinstance(qfe[0], stim.Tableau)
    assert qfe[0] == stim.Tableau.from_circuit(stim.Circuit("H 0\nCX 0 1\nS 0"))
    res = EvaluateQueries(PrepareQueries(circ, target_spec=qfe), queries=("fidelity",))
    assert res.fidelity_mass == pytest.approx(1.0, abs=1e-12)


def test_get_detector_flips_positional_append_observables() -> None:
    circ = stim.Circuit("X 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]")
    sim = MagicsideSimulator(circ, batch_size=3, seed=5)
    sim.run()
    np.testing.assert_array_equal(sim.get_detector_flips(True), sim.get_detector_flips(append_observables=True))
    np.testing.assert_array_equal(
        sim.get_detector_flips(True, bit_packed=True),
        sim.get_detector_flips(append_observables=True, bit_packed=True),
    )
    assert sim.get_detector_flips(True).shape == (3, 3)


def test_presample_stream_maps_heralded_and_correlated_branches() -> None:
    circ = stim.Circuit(
        "X_ERROR(0.2) 0\n"
        "HERALDED_PAULI_CHANNEL_1(0.0, 0.25, 0.0, 0.0) 0 1\n"
        "CORRELATED_ERROR(1.0) X2\n"
        "ELSE_CORRELATED_ERROR(1.0) X3\n"
        "M 0 1 2 3"
    )
    reals = [
        {"paulis": {0: 1}, "measurements": {0: 1, 1: 0}},
        {"paulis": {0: 0, 3: 0}},
    ]
    sim = MagicsideSimulator(circ, batch_size=2, seed=19, fault_realizations=reals)
    sim.run()
    recs = sim.get_final_measurement_records()
    # Shot 0: X_ERROR forced to X, Heralded target 0 forced herald=1 (applies X -> cancels X on q0),
    # Heralded target 1 forced herald=0 (no Pauli), CORRELATED_ERROR(1.0) fires on q2 so ELSE skips q3.
    assert recs[0].tolist() == [True, False, False, False, True, False]
    # Shot 1: CORRELATED_ERROR forced to 0 via paulis[3]=0, so ELSE_CORRELATED_ERROR(1.0) fires on q3.
    assert recs[1, 4:6].tolist() == [False, True]


def test_empty_fault_realizations_match_none() -> None:
    circ = stim.Circuit("H 0 1\nI[T] 0\nCX 0 1\nI[T] 1\nX_ERROR(0.3) 0 1\nM 0 1")
    b = 64

    def recs(fr: object) -> np.ndarray:
        sim = MagicsideSimulator(circ, batch_size=b, seed=5, fault_realizations=fr)
        sim.run()
        return sim.get_final_measurement_records()

    want = recs(None)
    for fr in ({}, [{}] * b):
        np.testing.assert_array_equal(recs(fr), want)


def test_magic_chi_envelope_commuting_t_and_scaling_pattern() -> None:
    """Chi growth and frame merging, with a loose time bound (about 0.2 s locally)."""
    t0 = time.perf_counter()
    qs = " ".join(map(str, range(12)))
    t_layer = "\n".join(f"I[T] {q}" for q in range(12))
    t_dag_layer = "\n".join(f"I[T_DAG] {q}" for q in range(12))
    # 12 commuting T gates: chi = 2^12 and the state is (|0> + e^{i pi/4}|1>)^{x12} / 2^6.
    sim = MagicsideSimulator(stim.Circuit(f"H {qs}\n{t_layer}"), batch_size=1, seed=1)
    sim.run()
    assert sim.get_qfe_expansion(0)[1].chi == 4096
    one = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    ref = one
    for _ in range(11):
        ref = np.kron(ref, one)
    assert abs(abs(np.vdot(ref, sim.state_vector(0))) - 1) < 1e-9
    # The M on spare qubit 12 splits the open block, so T_DAG must merge the 4096 frames back to 1.
    sim = MagicsideSimulator(
        stim.Circuit(f"H {qs}\n{t_layer}\nM 12\n{t_dag_layer}"), batch_size=1, seed=1
    )
    sim.run()
    exp = sim.get_qfe_expansion(0)[1]
    assert exp.chi == 1
    assert abs(exp.norm_sq() - 1) < 1e-9
    # magic_scaling.py's pattern: 8 qubits, k=8 layers of {I[T] j % 8; CX j % 8, (j + 1) % 8}.
    c = stim.Circuit()
    c.append("H", range(8))
    for j in range(8):
        c.append("I", [j % 8], tag="T")
        c.append("CX", [j % 8, (j + 1) % 8])
        c.append("TICK")
    sim = MagicsideSimulator(c, batch_size=4, seed=1)
    sim.run()
    assert [sim.get_qfe_expansion(s)[1].chi for s in range(4)] == [256] * 4
    assert time.perf_counter() - t0 < 10.0


def test_live_branch_mass_is_exact_product_and_survives_underflow() -> None:
    # Each block measures through the live Born path with P(0) = cos^2(pi/8), so the exact mass
    # is the product of P(outcome) over the records. n=2000 takes it below e^-745 (float minimum).
    p0 = (1.0 + 0.5**0.5) / 2.0
    for n in (40, 2000):
        circ = stim.Circuit("R 0\nH 0\nI[T] 0\nH 0\nM 0\n" * n)
        sim = MagicsideSimulator(circ, batch_size=2, seed=7)
        sim.run()
        assert sim._active_plan is None  # The live engine ran, not the compiled one.
        for s, row in enumerate(np.reshape(sim.get_final_measurement_records(), (2, n))):
            log_m = math.fsum(math.log(1.0 - p0 if m else p0) for m in row)
            assert math.isclose(sim.get_branch_mass(s), math.exp(log_m), rel_tol=1e-12)
            amp = float(np.max(np.abs(sim.state_vector(s, normalize=False))))
            assert math.isclose(amp, math.exp(0.5 * log_m), rel_tol=1e-9)
            assert n == 40 or log_m < -745.0


def _engine_chunk(job: tuple[str, int, bool]) -> np.ndarray:
    """Records of one fixed-seed 2048-shot chunk from the live or the forced compiled engine."""
    text, seed, compiled = job
    sim = MagicsideSimulator(stim.Circuit(text), batch_size=2048, seed=seed)
    if compiled:
        sim._run_compiled_stream()  # The engine run() picks only for SELECTOR / realizations.
    else:
        sim.run()  # DIRECT, trivial handler, no realizations: the live interpreter.
    assert (sim._active_plan is not None) == compiled  # Only the compiled engine sets a plan.
    return np.stack([sim.measurement_column(k) for k in range(sim.num_measurements)], axis=1)


def _max_z(a: np.ndarray, b: np.ndarray) -> float:
    """Largest pooled two-proportion z over all bit means and all pairwise XOR parities."""
    i, j = np.triu_indices(a.shape[1], 1)
    a, b = np.hstack([a, a[:, i] ^ a[:, j]]), np.hstack([b, b[:, i] ^ b[:, j]])
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    p = (pa + pb) / 2
    # Pooled p in {0, 1} means sigma = 0 and pa == pb exactly (both engines deterministic there).
    sigma = np.sqrt(p * (1 - p) * 2 / len(a))
    return float(np.max(np.where(sigma > 0, np.abs(pa - pb) / np.where(sigma > 0, sigma, 1), 0)))


@pytest.mark.slow
def test_live_and_compiled_stream_engines_sample_the_same_distribution() -> None:
    """Means and pairwise XOR parities of all measurements agree within 5 sigma (n = 8192).

    XOR parities plus means fix every pairwise correlation. At most 105 comparisons per circuit
    (168 in total), so a false failure is about 1e-4 family-wise. Each engine's 8192 shots are
    4 fixed-seed chunks of 2048; the 48 chunks run on 8 processes (about 30 s wall).
    """
    names = [
        "c01_clifford_noise_feedback",
        "c02_mpp_mxx_heralded_corr",
        "c02b_spp_clifford",
        "c06_magic_t_ccz_clifford",
        "c07_magic_inline_faults",
        "c10_magic_multiword_68q",
    ]
    texts = [next(text for c, _, _, text in CORPUS if c == name) for name in names]
    jobs = [(t, 100 * c + 10 * comp + k, comp) for c, t in enumerate(texts)
            for comp in (False, True) for k in range(4)]
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(_engine_chunk, jobs))
    z = {n: _max_z(np.vstack(res[8 * c:8 * c + 4]), np.vstack(res[8 * c + 4:8 * c + 8]))
         for c, n in enumerate(names)}
    assert max(z.values()) < 5, z
