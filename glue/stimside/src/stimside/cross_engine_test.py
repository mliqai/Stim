"""Cross-engine behavioral parity and sync_tableside_rng lockstep regression tests."""

import numpy as np
import pytest
import stim

from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import (
    LeakageUint8Coset,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import (
    LeakageUint8 as LeakageUint8Flip,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
    LeakageUint8 as LeakageUint8Tableau,
)
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_flip import FlipsideSimulator
from stimside.simulator_tableau import TablesideSimulator


def _run_sim(sim_cls, handler_cls, circuit, *, seed=42, batch_size=4, sync=True):
    handler = handler_cls()
    compiled = handler.compile_op_handler(circuit=circuit, batch_size=batch_size)
    sim = sim_cls(
        circuit,
        compiled_op_handler=compiled,
        seed=seed,
        batch_size=batch_size,
        record_unleaked_to_leaked=True,
        **({"sync_tableside_rng": True} if sync else {}),
    )
    unleaked_to_leaked = sim.run()
    rec = sim.get_final_measurement_records()
    det = sim.get_detector_flips()
    obs = sim.get_observable_flips()
    if sim_cls is FlipsideSimulator:
        det = det.T
        obs = obs.T
    return sim, rec, det, obs, unleaked_to_leaked


@pytest.mark.parametrize(
    "tagged_op",
    [
        "I[CONDITIONED_ON_SELF: 0 1] 0",
        "I[LEAKAGE_TRANSITION_1: (0.0, 1-->2)] 0",
        "I[LEAKAGE_TRANSITION_1: (0.001, 1-->2)] 0",
        "II[LEAKAGE_TRANSITION_2: (0.0, 0_0-->2_2)] 0 1",
        "MPAD[LEAKAGE_MEASUREMENT: (0.0, 0) (0.0, 1) (1.0, 2) : 0] 0",
    ],
)
def test_conditioning_on_0_or_1_collapses_superposition(tagged_op: str):
    """Conditioning on 0 or 1 collapses Z-basis superpositions across Tableside and Cosetside."""
    circuit = stim.Circuit(
        f"""
        H 0
        {tagged_op}
        H 0
        M 0
        """
    )
    num_shots = 200

    tab_handler = LeakageUint8Tableau().compile_op_handler(
        circuit=circuit, batch_size=1
    )
    tab_sim = TablesideSimulator(
        circuit, compiled_op_handler=tab_handler, seed=777
    )
    tab_outcomes = []
    for _ in range(num_shots):
        tab_sim.clear()
        tab_sim.run()
        tab_outcomes.append(bool(tab_sim.get_final_measurement_records()[0, -1]))
    tab_rate = np.mean(tab_outcomes)
    assert 0.3 < tab_rate < 0.7, (
        f"Expected ~0.5 after Z-collapse of |+> in Tableside for {tagged_op}, got {tab_rate}"
    )

    cos_handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=num_shots
    )
    cos_sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=cos_handler,
        batch_size=num_shots,
        seed=777,
    )
    cos_sim.run()
    cos_rate = float(np.mean(cos_sim.get_final_measurement_records()[:, -1]))
    assert 0.3 < cos_rate < 0.7, (
        f"Expected ~0.5 after Z-collapse of |+> in Cosetside for {tagged_op}, got {cos_rate}"
    )


def test_repeated_qubit_targets_processed_in_order():
    """Instructions with repeated targets are processed sequentially in target order across engines."""
    c_cx = stim.Circuit(
        """
        X 0
        CX 0 1 1 0
        H 0 0
        S 1 1
        M 0 1
        """
    )
    stim_sample = c_cx.compile_sampler().sample(1)[0].tolist()
    assert stim_sample == [False, True]

    for sim_cls, factory_cls in [
        (FlipsideSimulator, LeakageUint8Flip),
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
    ]:
        b_size = 1 if sim_cls is TablesideSimulator else 4
        handler = factory_cls().compile_op_handler(
            circuit=c_cx, batch_size=b_size
        )
        sim = sim_cls(
            c_cx, compiled_op_handler=handler, batch_size=b_size, seed=11
        )
        sim.run()
        recs = sim.get_final_measurement_records()
        assert np.all(recs[:, 0] == False)
        assert np.all(recs[:, 1] == True)

    c_trans = stim.Circuit(
        """
        I[LEAKAGE_TRANSITION_1: (1.0, U-->2) (1.0, 2-->3)] 0 0
        M[LEAKAGE_PROJECTION_Z: (0.0, 0) (0.0, 1) (0.0, 2) (1.0, 3)] 0
        """
    )
    for sim_cls, factory_cls in [
        (FlipsideSimulator, LeakageUint8Flip),
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
    ]:
        b_size = 1 if sim_cls is TablesideSimulator else 4
        handler = factory_cls().compile_op_handler(
            circuit=c_trans, batch_size=b_size
        )
        sim = sim_cls(
            c_trans, compiled_op_handler=handler, batch_size=b_size, seed=11
        )
        sim.run()
        recs = sim.get_final_measurement_records()
        assert np.all(recs[:, 0] == True), (
            f"Expected state 3 (meas=1) for {sim_cls.__name__}"
        )


def test_conditioning_01_with_repeated_targets():
    """0/1 conditioning combined with repeated/overlapping targets operates sequentially per group."""
    c1 = stim.Circuit(
        """
        X[CONDITIONED_ON_SELF: 0] 0 0
        M 0
        """
    )
    for sim_cls, factory_cls in [
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
    ]:
        b_size = 1 if sim_cls is TablesideSimulator else 8
        handler = factory_cls().compile_op_handler(
            circuit=c1, batch_size=b_size
        )
        sim = sim_cls(
            c1, compiled_op_handler=handler, batch_size=b_size, seed=42
        )
        sim.run()
        recs = sim.get_final_measurement_records()
        assert np.all(recs[:, 0] == True), (
            f"Expected |1> for X[CONDITIONED_ON_SELF: 0] 0 0 in {sim_cls.__name__}"
        )

    c2 = stim.Circuit(
        """
        X 0
        H[CONDITIONED_ON_SELF: 1] 0 0
        M 0
        """
    )
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=c2, batch_size=64
    )
    sim = CosetsideSimulator(
        c2,
        compiled_op_handler=handler,
        batch_size=64,
        seed=123,
    )
    sim.run()
    recs = sim.get_final_measurement_records()[:, 0]
    assert np.any(recs == False) and np.any(recs == True), (
        "Expected non-deterministic outcome for H[CONDITIONED_ON_SELF: 1] 0 0"
    )

    c3 = stim.Circuit(
        """
        X[CONDITIONED_ON_OTHER: 0 : 0 0] 0 1
        M 0 1
        """
    )
    for sim_cls, factory_cls in [
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
    ]:
        b_size = 1 if sim_cls is TablesideSimulator else 8
        handler = factory_cls().compile_op_handler(
            circuit=c3, batch_size=b_size
        )
        sim = sim_cls(
            c3, compiled_op_handler=handler, batch_size=b_size, seed=42
        )
        sim.run()
        recs = sim.get_final_measurement_records()
        assert np.all(recs[:, 0] == True)
        assert np.all(recs[:, 1] == False)


def test_leakage_transition_2_single_leg_collapse():
    """LEAKAGE_TRANSITION_2 only collapses the leg conditioned on 0 or 1."""
    circuit = stim.Circuit(
        """
        H 0 1
        II[LEAKAGE_TRANSITION_2: (0.0, 1_U-->2_U)] 0 1
        MX 1
        """
    )
    for sim_cls, factory_cls in [
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
    ]:
        b_size = 1 if sim_cls is TablesideSimulator else 32
        handler = factory_cls().compile_op_handler(
            circuit=circuit, batch_size=b_size
        )
        sim = sim_cls(
            circuit, compiled_op_handler=handler, batch_size=b_size, seed=77
        )
        for _ in range(16 if sim_cls is TablesideSimulator else 1):
            sim.clear()
            sim.run()
            recs = sim.get_final_measurement_records()
            assert np.all(recs[:, 0] == False), (
                f"Leg 1 conditioned only on U should remain |+> in {sim_cls.__name__}"
            )


def test_1a_leakage_projection_z_all_bases_and_mpad():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0 1 2 3
        H 0
        H_YZ 1
        X 2
        MX[LEAKAGE_PROJECTION_Z: (0.1, 0) (0.9, 1) (0.75, 2)] 0 !1
        MY[LEAKAGE_PROJECTION_Z: (0.1, 0) (0.9, 1) (0.75, 2)] 1
        MR[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 2 !3
        MRX[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 0
        MRY[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 1
        MPAD 0 1 1 0
        MPAD(0.25) 0 1 1
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2) : 0 2] 0 1
        MPAD[LEAKAGE_MEASUREMENT: (0.1, 0) (0.8, 1) (0.95, 2) : 1 3] 1 0
        M 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-5]
        """
    )
    _, t_rec, t_det, t_obs, t_ev = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=101, batch_size=8
    )
    _, c_rec, c_det, c_obs, c_ev = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=101, batch_size=8
    )

    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)
    for b in range(8):
        np.testing.assert_array_equal(t_ev[b], c_ev[b])


def test_1b_heralded_and_product_measurements_with_leakage():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1
        CX 0 2 1 3
        HERALDED_ERASE(0.3) 0 !1
        HERALDED_PAULI_CHANNEL_1(0.05, 0.1, 0.1, 0.15) 2 !3
        I[LEAKAGE_TRANSITION_1: (0.4, U-->2)] 0 2
        MXX(0.1) 0 1 !2 3
        MYY(0.1) 0 !1 2 3
        MZZ(0.1) 0 2 !1 3
        MPP(0.15) X0*X1 !Y1*Z2 Z0*X2*Y3
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.85, 2)] 0 1 2 3
        DETECTOR rec[-1] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-6]
        """
    )
    _, t_rec, t_det, t_obs, _ = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=202, batch_size=8
    )
    _, c_rec, c_det, c_obs, _ = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=202, batch_size=8
    )

    assert t_rec.shape == (8, 17)
    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)


def test_7_end_to_end_2way_shot_for_shot_equivalence():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        RX 1
        RY 2
        I[LEAKAGE_TRANSITION_1: (0.35, U-->2)] 0 1 2
        HERALDED_ERASE(0.2) 0 1
        HERALDED_PAULI_CHANNEL_1(0.05, 0.05, 0.05, 0.05) 2 3
        MXX(0.05) 0 1 2 3
        MPP(0.05) X0*Z1 Y2*X3
        MRX[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 0 !1
        MRY[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 2 !3
        I[LEAKAGE_TRANSITION_1: (1.0, 2-->U)] 0 1 2 3
        R 0 1 2 3
        MPAD(0.1) 0 1 1
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2) : 0 1] 0 1
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.85, 2)] 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )
    _, t_rec, t_det, t_obs, t_ev = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=303, batch_size=10
    )
    _, c_rec, c_det, c_obs, c_ev = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=303, batch_size=10
    )

    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)
    for b in range(10):
        np.testing.assert_array_equal(t_ev[b], c_ev[b])


def test_non_flip_compatible_x_y_reset_leakage_projection_z_equivalence():
    circuit = stim.Circuit(
        """
        R 0 1 2
        H 0 1
        X_ERROR(0.3) 0 1 2
        I[LEAKAGE_TRANSITION_1: (0.25, 0-->2) (0.45, 1-->2)] 0 1 2
        MX[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 !1
        MY[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 1
        MR[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 !1 2
        MRX[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 !1 2
        MRY[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 1 2
        I[LEAKAGE_TRANSITION_1: (1.0, 2-->U)] 0 1 2
        MX[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 1 2
        MR[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.7, 2)] 0 1 2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )
    _, t_rec, t_det, t_obs, t_ev = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=99, batch_size=24
    )
    _, c_rec, c_det, c_obs, c_ev = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=99, batch_size=24
    )
    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)
    for b in range(24):
        np.testing.assert_array_equal(t_ev[b], c_ev[b])


def test_noiseless_leakage_projection_z_sync_all_bases():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        H 0
        I[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 1 2 3
        MR[LEAKAGE_PROJECTION_Z: (0.8, 2)] 0
        MX[LEAKAGE_PROJECTION_Z: (0.8, 2)] 1
        MY[LEAKAGE_PROJECTION_Z: (0.8, 2)] 2
        MRX[LEAKAGE_PROJECTION_Z: (0.8, 2)] 3
        MRY[LEAKAGE_PROJECTION_Z: (0.8, 2)] 1
        MRZ[LEAKAGE_PROJECTION_Z: (0.8, 2)] 2
        I[LEAKAGE_TRANSITION_1: (1.0, 2-->U)] 0 1 2 3
        M 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )
    _, t_rec, t_det, t_obs, t_ev = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=12, batch_size=16
    )
    _, c_rec, c_det, c_obs, c_ev = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=12, batch_size=16
    )
    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)
    for b in range(16):
        np.testing.assert_array_equal(t_ev[b], c_ev[b])


def test_leaked_targets_reset_in_target_order_sync():
    circuit = stim.Circuit(
        """
        R 0 1 2 3 4 5 6 7
        H 0 1 4 5
        CX 0 2 1 3 4 6 5 7
        I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0 1 4 5
        M[LEAKAGE_PROJECTION_Z: (0.8, 2)] 1 0
        MX[LEAKAGE_PROJECTION_Z: (0.1, 0) (0.9, 1) (0.8, 2)] 5 4
        M 2 3
        MX 6 7
        """
    )
    _, t_rec, _, _, t_ev = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, circuit, seed=12, batch_size=16
    )
    _, c_rec, _, _, c_ev = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, circuit, seed=12, batch_size=16
    )
    np.testing.assert_array_equal(t_rec, c_rec)
    for b in range(16):
        np.testing.assert_array_equal(t_ev[b], c_ev[b])


def test_duplicate_leaked_target_leakage_projection_z_sync_raises():
    circuit = stim.Circuit(
        """
        R 0
        I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0
        M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 0
        """
    )
    with pytest.raises(NotImplementedError):
        _run_sim(CosetsideSimulator, LeakageUint8Coset, circuit, sync=True)


def test_repeated_targets_in_leakage_projection_z_and_mr():
    c_sync = stim.Circuit(
        """
        R 0
        X_ERROR(1.0) 0
        MR[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.8, 2)] 0 0
        X_ERROR(1.0) 0
        MR(0.1) 0 0
        MX[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.8, 2)] 0 0
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-6]
        """
    )
    _, t_rec, t_det, t_obs, _ = _run_sim(
        TablesideSimulator, LeakageUint8Tableau, c_sync, seed=12, batch_size=16, sync=True
    )
    _, c_rec, c_det, c_obs, _ = _run_sim(
        CosetsideSimulator, LeakageUint8Coset, c_sync, seed=12, batch_size=16, sync=True
    )
    np.testing.assert_array_equal(t_rec, c_rec)
    np.testing.assert_array_equal(t_det, c_det)
    np.testing.assert_array_equal(t_obs, c_obs)

    c_nosync = stim.Circuit(
        """
        R 0
        X 0
        MR[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.8, 2)] 0 0
        H 0
        Z 0
        MX[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.8, 2)] 0 0
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )
    for sim_cls, h_cls in [
        (TablesideSimulator, LeakageUint8Tableau),
        (CosetsideSimulator, LeakageUint8Coset),
        (FlipsideSimulator, LeakageUint8Flip),
    ]:
        _, rec, _, _, _ = _run_sim(
            sim_cls, h_cls, c_nosync, seed=12, batch_size=4, sync=False
        )
        for b in range(4):
            assert rec[b].tolist() == [True, False, True, True]
