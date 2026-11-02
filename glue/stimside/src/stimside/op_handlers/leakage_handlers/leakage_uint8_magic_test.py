"""Unit tests for LeakageUint8Magic and CompiledLeakageUint8Magic."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import stim

from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import CompiledLeakageUint8Magic, LeakageUint8Magic
from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_tableau import TablesideSimulator


def test_u_basis_mpad_leakage_measurement_preserves_magic_coherence() -> None:
    circ = stim.Circuit(
        "H 0\n"
        "I[T] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, U) (0.0, 2) : 0] 0\n"
        "I[T_DAG] 0\n"
        "H 0\n"
        "M 0"
    )
    handler = LeakageUint8Magic().compile_op_handler(circuit=circ, batch_size=32)
    assert isinstance(handler, CompiledLeakageUint8Magic)
    sim = MagicsideSimulator(circ, compiled_op_handler=handler, batch_size=32, seed=13)
    sim.run()
    recs = sim.get_final_measurement_records()
    assert np.all(recs[:, 0])
    assert not np.any(recs[:, 1])


def test_skewed_born_collapse_on_state_selective_transition() -> None:
    # R_Y(pi/4)|0>: P(|1>) = sin^2(pi/8) = (2 - sqrt(2))/4 ~= 0.1464466
    circ = stim.Circuit(
        "H 0\nI[T] 0\nH 0\nS 0\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (1.0, 2)] 0"
    )
    sim = MagicsideSimulator(circ, op_handler=LeakageUint8Magic(), batch_size=2048, seed=42)
    sim.run()
    recs = sim.get_final_measurement_records()
    np.testing.assert_array_equal(recs[:, 0], recs[:, 1])
    p_want = (2.0 - math.sqrt(2.0)) / 4.0
    assert abs(float(np.mean(recs[:, 0])) - p_want) < 0.03


def test_sync_tableside_rng_clifford_leakage_lockstep_with_cosetside() -> None:
    circ = stim.Circuit(
        "H 0 1\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.5, 1-->2)] 0\n"
        "TICK\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.8, 2) : 0] 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 1"
    )
    sc = CosetsideSimulator(
        circ, compiled_op_handler=LeakageUint8Coset(False).compile_op_handler(circuit=circ, batch_size=16),
        batch_size=16, seed=77, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sm = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(False), batch_size=16, seed=77,
        sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    assert [a.tolist() for a in sm.run()] == [a.tolist() for a in sc.run()]  # type: ignore[union-attr]
    np.testing.assert_array_equal(sm.get_final_measurement_records(), sc.get_final_measurement_records())
    assert sm.get_leakage_events() == sc.get_leakage_events()


def test_magic_gates_on_leaked_targets_warn_and_act_unconditionally() -> None:
    circ = stim.Circuit(
        "H 0 4\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0 4\n"
        "I[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\n"
        "I[T_DAG] 4\nI[T_DAG] 4\nI[T_DAG] 4\nI[T_DAG] 4\n"
        "MPP X0 X4\n"
        "X 1 2\nH 3\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 1\n"
        "I[CCZ] 1 2 3\n"
        "H 3\nM 3"
    )
    sim = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(unconditional_condition_on_U=True), batch_size=8, seed=17
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run()
    u_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(u_warns) == 9
    assert np.all(sim.get_final_measurement_records())

    for bad in ("X[T] 0", "I[T](0.1) 0", "I[CCZ] 0 1", "I[CCZ] 0 0 1"):
        with pytest.raises(ValueError):
            LeakageUint8Magic().compile_op_handler(circuit=stim.Circuit(bad), batch_size=1)


def test_sync_rng_nonclifford_born_stats_mpad_entangled_and_speed() -> None:
    p_sin2 = (2.0 - math.sqrt(2.0)) / 4.0
    sigma = math.sqrt(p_sin2 * (1.0 - p_sin2) / 4096.0)
    circ_mpad = stim.Circuit(
        "H 2\nI[T] 2\nH 2\nS 2\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.0, 0) (1.0, 1) : 2] 0 0\n"
        "M 0 2"
    )
    circ_cx = stim.Circuit(
        "H 0 1\nI[T] 1\nH 1\nS 1\nCX 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1)] 0 1\n"
        "M 0 1"
    )
    for sync in (False, True):
        sim_m = MagicsideSimulator(
            circ_mpad, op_handler=LeakageUint8Magic(), batch_size=4096, seed=42, sync_tableside_rng=sync
        )
        sim_m.run()
        rec_m = sim_m.get_final_measurement_records()
        assert not np.any(rec_m[:, 2])
        np.testing.assert_array_equal(rec_m[:, 0], rec_m[:, 1])
        np.testing.assert_array_equal(rec_m[:, 0], rec_m[:, 3])
        assert abs(float(np.mean(rec_m[:, 0])) - p_sin2) < 3.0 * sigma

        sim_cx = MagicsideSimulator(
            circ_cx, op_handler=LeakageUint8Magic(), batch_size=4096, seed=84, sync_tableside_rng=sync
        )
        sim_cx.run()
        rec_cx = sim_cx.get_final_measurement_records()
        assert abs(float(np.mean(rec_cx[:, 2] ^ rec_cx[:, 3])) - p_sin2) < 3.0 * sigma

    circ_ent = stim.Circuit(
        "H 0 1 2\nI[CCZ] 0 1 2\nH 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (0.9999, 1) (0.5, 2)] 0 1"
    )
    sim_e = MagicsideSimulator(
        circ_ent, op_handler=LeakageUint8Magic(), batch_size=512, seed=91, sync_tableside_rng=True
    )
    sim_e.run()
    recs_e = sim_e.get_final_measurement_records()
    p00 = float(np.mean((~recs_e[:, 0]) & (~recs_e[:, 1])))
    p01 = float(np.mean((~recs_e[:, 0]) & recs_e[:, 1]))
    p10 = float(np.mean(recs_e[:, 0] & (~recs_e[:, 1])))
    p11 = float(np.mean(recs_e[:, 0] & recs_e[:, 1]))
    assert abs(p00 - 0.50) < 0.07 and p01 < 0.01 and abs(p10 - 0.25) < 0.07 and abs(p11 - 0.25) < 0.07


def test_sync_rng_three_way_clifford_lockstep() -> None:
    circ = stim.Circuit(
        "H 0 1 2\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.35, 1-->2)] 0 2\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.9, 2) : 0] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.3, 0) (0.3, 1) : 1] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.2, 0) (0.8, 1) (0.9, 2) : 2] 0 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.1, 0) (0.85, 1) (0.95, 2)] 0 1 2"
    )
    st = TablesideSimulator(
        circ,
        compiled_op_handler=LeakageUint8(False).compile_op_handler(circuit=circ, batch_size=64),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sc = CosetsideSimulator(
        circ,
        compiled_op_handler=LeakageUint8Coset(False).compile_op_handler(circuit=circ, batch_size=64),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sm = MagicsideSimulator(
        circ,
        op_handler=LeakageUint8Magic(False),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    u2l_t, u2l_c, u2l_m = st.run(), sc.run(), sm.run()
    assert [a.tolist() for a in u2l_m] == [a.tolist() for a in u2l_c] == [a.tolist() for a in u2l_t]  # type: ignore[union-attr]
    np.testing.assert_array_equal(sm.get_final_measurement_records(), sc.get_final_measurement_records())
    np.testing.assert_array_equal(sm.get_final_measurement_records(), st.get_final_measurement_records())
    assert sm.get_leakage_events() == sc.get_leakage_events() == st.get_leakage_events()
