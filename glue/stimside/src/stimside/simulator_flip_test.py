import numpy as np
import pytest
import stim # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.simulator_flip import FlipsideSimulator


class MockCompiledOpHandler(CompiledOpHandler[FlipsideSimulator]):
    def claim_ops(self) -> set[str]:
        return {"MOCK"}

    def handle_op(self, op: stim.CircuitInstruction, sss: "FlipsideSimulator") -> None:
        if op in self.claim_ops():
            raise NotImplementedError("mock handle_op called and somehow matched")
        else:
            sss.do_on_flip_simulator(op)
            return

    def clear(self) -> None:
        raise NotImplementedError("mock clear called")


class TestFlipsideSimulator:

    batch_size = 128
    coh = MockCompiledOpHandler()

    def make_simulator(self, circuit):
        return FlipsideSimulator(
            circuit=circuit, compiled_op_handler=self.coh, batch_size=self.batch_size
        )

    def test_coords_tracking(self):
        """test qubit_coords, qubit_tags, coords_shifts."""
        circuit = stim.Circuit(
            """
        QUBIT_COORDS(0,0) 0
        QUBIT_COORDS(1,0) 1
        SHIFT_COORDS(5, 5)
        QUBIT_COORDS(0) 5
        QUBIT_COORDS(1, 0) 6
        SHIFT_COORDS(5)
        SHIFT_COORDS(5,5,5)
        QUBIT_COORDS[QUBIT_TAG](0,0) 10
        """
        )
        fss = self.make_simulator(circuit)
        fss.interactive_do(circuit[0])  # QUBIT_COORDS(0,0) 0
        fss.interactive_do(circuit[1])  # QUBIT_COORDS(1,0) 1
        assert fss.qubit_coords == {0: [0, 0], 1: [1, 0]}
        fss.interactive_do(circuit[2])  # SHIFT_COORDS(5, 5)
        assert fss.qubit_coords == {0: [0, 0], 1: [1, 0]}
        assert fss.coords_shifts == [5, 5]
        fss.interactive_do(circuit[3])  # QUBIT_COORDS(0) 5
        fss.interactive_do(circuit[4])  # QUBIT_COORDS(1, 0) 6
        assert fss.qubit_coords == {0: [0, 0], 1: [1, 0], 5: [5], 6: [6, 5]}
        fss.interactive_do(circuit[5])  # SHIFT_COORDS(5)
        assert fss.coords_shifts == [10, 5]
        fss.interactive_do(circuit[6])  # SHIFT_COORDS(5,5,5)
        assert fss.coords_shifts == [15, 10, 5]
        fss.interactive_do(circuit[7])  # QUBIT_COORDS[QUBIT_TAG](0, 0)
        assert fss.qubit_coords == {0: [0, 0], 1: [1, 0], 5: [5], 6: [6, 5], 10: [15, 10]}

    def test_run_and_clear(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
        """
        )
        fss = self.make_simulator(circuit)

        fss.run()

        with pytest.raises(ValueError, match=r"prepare the simulator for a new run"):
            fss.run()

        with pytest.raises(NotImplementedError, match="mock clear called"):
            fss.clear()

        assert fss.qubit_coords == {}
        assert fss.qubit_tags == {}
        assert fss.coords_shifts == []

    def test_flip_simulator_lookthroughs(self):
        circuit = stim.Circuit(
            """
            M 0 1 2 3 4 5 6 7
            DETECTOR(0) rec[-8]
            DETECTOR(1) rec[-7]
            DETECTOR(2) rec[-6]
            DETECTOR(3) rec[-5]
            OBSERVABLE_INCLUDE(0) rec[-4] rec[-3] rec[-2] rec[-1]
        """
        )
        circuit_num_qubits = 8
        circuit_num_measurements = 8
        circuit_num_detectors = 4
        circuit_num_observables = 1

        fss = self.make_simulator(circuit)
        fss.run()

        assert fss.batch_size == self.batch_size
        assert fss.num_qubits == circuit_num_qubits

        assert fss.get_detector_flips().shape == (circuit_num_detectors, self.batch_size)
        assert fss.get_measurement_flips().shape == (circuit_num_measurements, self.batch_size)
        assert fss.get_observable_flips().shape == (circuit_num_observables, self.batch_size)

    def test_broadcast_pauli_errors(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
        """
        )
        circuit_num_qubits = 4
        fss = self.make_simulator(circuit)
        fss.run()

        mask = np.zeros((circuit_num_qubits, self.batch_size), dtype=bool)
        mask[1, :] = True
        mask[3, :] = True
        before_xs, _, _, _, _ = fss._flip_simulator.to_numpy(output_xs=True)
        fss.broadcast_pauli_errors(error_mask=mask, p=1, pauli="X")
        after_xs, _, _, _, _ = fss._flip_simulator.to_numpy(output_xs=True)
        assert np.all(before_xs[0] == after_xs[0])
        assert np.all(before_xs[1] == np.logical_not(after_xs[1]))
        assert np.all(before_xs[2] == after_xs[2])
        assert np.all(before_xs[3] == np.logical_not(after_xs[3]))

        # jankily check probability
        mask = np.zeros((circuit_num_qubits, self.batch_size), dtype=bool)
        mask[1, :] = True
        mask[3, :] = True
        before_xs, _, _, _, _ = fss._flip_simulator.to_numpy(output_xs=True)
        fss.broadcast_pauli_errors(error_mask=mask, p=0.5, pauli="X")
        after_xs, _, _, _, _ = fss._flip_simulator.to_numpy(output_xs=True)
        assert np.all(before_xs[0] == after_xs[0])
        assert np.all(before_xs[2] == after_xs[2])

        # our batch size is massive, so we check that  the pauli error has done _something_
        # this should fail accidentally around 0.5**batch_size, which is vanishing
        assert np.any(before_xs[1] != after_xs[1])
        assert np.any(before_xs[3] != after_xs[3])
        # but also check it's not just been inverted unconditionally
        assert np.any(before_xs[1] != np.logical_not(before_xs[1]))
        assert np.any(before_xs[3] != np.logical_not(before_xs[3]))

    def test_get_clean_known_states(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
            X 1
            H 2 3
            M 3
        """
        )
        fss = self.make_simulator(circuit)
        fss.run()

        pX, mX = fss._get_clean_known_states(pauli="X", circuit_time=4)
        assert pX.tolist() == [False, False, True, False]
        assert mX.tolist() == [False, False, False, False]

        pZ, mZ = fss._get_clean_known_states(pauli="Z")  # circuit time should default to 4
        assert pZ.tolist() == [True, False, False, False]
        assert mZ.tolist() == [False, True, False, False]

    def test_all_targets_in_known_state(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
            X 1
            H 2 3
            M 3
        """
        )
        fss = self.make_simulator(circuit)
        fss.run()

        assert fss._all_targets_in_known_state(targets=[0, 1], pauli="Z")
        assert not fss._all_targets_in_known_state(targets=[0, 1, 2], pauli="Z")
        assert not fss._all_targets_in_known_state(targets=[0, 1, 3], pauli="Z", circuit_time=4)
        assert fss._all_targets_in_known_state(targets=[2], pauli="X", circuit_time=4)

    def test_get_current_known_state_masks(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
            X 1
            H 2 3
            M 3
        """
        )
        fss = self.make_simulator(circuit)
        fss.run()

        # No errors, no flips
        plus_mask, minus_mask = fss._get_current_known_state_masks(pauli="Z")
        assert plus_mask.shape == (fss.num_qubits, fss.batch_size)
        assert minus_mask.shape == (fss.num_qubits, fss.batch_size)

        # qubit 0 is in the 0 state, so in +Z
        assert np.all(plus_mask[0, :])
        assert not np.any(minus_mask[0, :])

        # qubit 1 is in the 1 state, so in Z
        assert np.all(minus_mask[1, :])
        assert not np.any(plus_mask[1, :])

        # qubit 2 is in +X, so shouldn't be in +Z or -Z
        assert not np.any(plus_mask[2, :])
        assert not np.any(minus_mask[2, :])

        # qubit 3 is randomly measured into +Z or -Z, so it is not in a known state
        assert not np.any(plus_mask[3, :])
        assert not np.any(minus_mask[3, :])

        # With errors, everyone gets a flip
        mask = np.ones((fss.num_qubits, fss.batch_size), dtype=bool)
        fss.broadcast_pauli_errors(mask, p=1, pauli="X")
        plus_mask, minus_mask = fss._get_current_known_state_masks(pauli="Z")
        # qubit 0 is in the 1 state, so in -Z
        assert np.all(minus_mask[0, :])
        assert not np.any(plus_mask[0, :])

        # qubit 1 is in the 0 state, so in +Z
        assert np.all(plus_mask[1, :])
        assert not np.any(minus_mask[1, :])

        # qubit 2 is in +X, so shouldn't be in +Z or -Z
        assert not np.any(plus_mask[2, :])
        assert not np.any(minus_mask[2, :])

        # qubit 3 is randomly measured into +Z or -Z, so it is not in a known state
        assert not np.any(plus_mask[3, :])
        assert not np.any(minus_mask[3, :])

    def test_circuit_time_zero_and_shift_coords_sync(self):
        """Regression tests for Bugs 8, 9, 10 and Optimization 4."""
        circuit = stim.Circuit(
            """
            R 0 1
            SHIFT_COORDS(0, 0, 1)
            X 0
            REPEAT 2 {
                SHIFT_COORDS(0, 0, 1)
                I 0 1
            }
            M 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        )
        fss = self.make_simulator(circuit)
        fss.run()
        # Bug 9: circuit_time=0 must return initial known state (+Z on both qubits), not final state (-Z on q0)
        pZ_0, mZ_0 = fss._get_clean_known_states("Z", circuit_time=0)
        assert pZ_0.tolist() == [True, True]
        assert mZ_0.tolist() == [False, False]
        pZ_end, mZ_end = fss._get_clean_known_states("Z")
        assert pZ_end.tolist() == [False, True]
        assert mZ_end.tolist() == [True, False]

        # Optimization 4: bit_packed=True matches packbits(det_flips.T, bitorder="little", axis=1)
        det_bool = fss.get_detector_flips(bit_packed=False)
        det_packed = fss.get_detector_flips(bit_packed=True)
        assert np.array_equal(
            det_packed, np.packbits(det_bool.T, axis=1, bitorder="little")
        )

    def test_interactive_do_must_replay_the_circuit(self):
        circuit = stim.Circuit("X 0\nREPEAT 2 {\n    M 0\n}\nM 1")
        fss = self.make_simulator(circuit)
        fss.interactive_do(circuit[0])
        fss.interactive_do(circuit[1])  # a REPEAT block replays as its unrolled ops
        fss.interactive_do(stim.Circuit("M 1"))  # an equal op from another circuit object is fine
        assert fss.get_final_measurement_records().tolist() == [[True, True, False]] * self.batch_size
        with pytest.raises(ValueError, match="ends after 4 operations"):
            fss.interactive_do(stim.Circuit("M 0"))
        # Any other op raises, even one that only changes the reference sample (I 0 for X 0 used to
        # give M 0 = 1, the construction circuit's reference value).
        for other in ["I 0\nM 0", "X 1", "X 0\nM 1", "X 0\nM !0"]:
            with pytest.raises(ValueError, match="doesn't match the circuit"):
                self.make_simulator(circuit).interactive_do(stim.Circuit(other))
        # flattened() drops the SHIFT_COORDS (operation 2 here), so it doesn't replay the circuit.
        rounds = stim.Circuit("REPEAT 2 {\n    M 0\n    DETECTOR(0) rec[-1]\n    SHIFT_COORDS(1)\n}")
        with pytest.raises(ValueError, match=r"not `circuit\.flattened\(\)` or `circuit \+ more`"):
            self.make_simulator(rounds).interactive_do(rounds.flattened())

    def test_interactive_do_noise_strength_may_differ(self):
        # Only an untagged noisy gate's strength may differ: it is applied live, and everything
        # precomputed from the construction circuit is noiseless.
        fss = self.make_simulator(stim.Circuit("X_ERROR(0) 0\nM(0) 0 1"))
        fss.interactive_do(stim.Circuit("X_ERROR(1) 0\nM(1) 0 1"))
        assert fss.get_final_measurement_records().tolist() == [[False, True]] * self.batch_size
        for a, b in [
            ("X_ERROR(0.1) 0", "Z_ERROR(0.1) 0"),
            ("X_ERROR[t](0.1) 0", "X_ERROR[t](0.2) 0"),
            ("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]", "M 0\nOBSERVABLE_INCLUDE(1) rec[-1]"),
        ]:
            with pytest.raises(ValueError, match="doesn't match the circuit"):
                self.make_simulator(stim.Circuit(a)).interactive_do(stim.Circuit(b))

    def test_final_measurement_records_after_partial_replay(self):
        circuit = stim.Circuit("X 0\nM 0\nI 0\nM 0 1")
        fss = self.make_simulator(circuit)
        fss.interactive_do(circuit[:2])  # used to broadcast the 3 reference bits over 1 measurement
        assert fss.get_final_measurement_records().tolist() == [[True]] * self.batch_size
        fss.interactive_do(circuit[2:])
        assert fss.get_final_measurement_records().tolist() == [[True, True, False]] * self.batch_size

    def test_multi_target_qubit_coords_and_shift_coords_across_engines(self) -> None:
        from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
        from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
        from stimside.simulator_coset import CosetsideSimulator
        from stimside.simulator_magic import MagicsideSimulator
        from stimside.simulator_tableau import TablesideSimulator

        circuit = stim.Circuit(
            "QUBIT_COORDS[data](1, 2) 0 1\n"
            "SHIFT_COORDS(10, 0, 5)\n"
            "QUBIT_COORDS[anc](3, 4, 5) 2 3\n"
            "M 0 1 2 3"
        )
        triv = TrivialOpHandler()

        tss_int = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=triv.compile_op_handler(circuit=circuit, batch_size=1),
            batch_size=1,
            seed=1,
        )
        tss_int.interactive_do(circuit)
        tss_int.finish_interactive_run()

        sims = [
            self.make_simulator(circuit),
            TablesideSimulator(
                circuit=circuit,
                compiled_op_handler=triv.compile_op_handler(circuit=circuit, batch_size=1),
                batch_size=2,
                seed=1,
            ),
            tss_int,
            CosetsideSimulator(
                circuit=circuit,
                compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=2),
                batch_size=2,
                seed=1,
            ),
            MagicsideSimulator(circuit=circuit, batch_size=2, seed=1),
        ]
        want_coords = {
            0: [1.0, 2.0],
            1: [1.0, 2.0],
            2: [13.0, 4.0, 10.0],
            3: [13.0, 4.0, 10.0],
        }
        for sim in sims:
            if sim is not tss_int:
                sim.run()
            assert sim.qubit_coords == want_coords
            assert sim.qubit_tags == {0: "data", 1: "data", 2: "anc", 3: "anc"}
            assert sim.coords_shifts == [10.0, 0.0, 5.0]

        q_sim = MagicsideSimulator(
            circuit=stim.Circuit(
                "QUBIT_COORDS(1, 2) 0 1\nSHIFT_COORDS(10, 0, 5)\nQUBIT_COORDS(3, 4, 5) 2 3\nM 0 1 2 3"
            ),
            batch_size=2,
            seed=1,
            backend="QUERY",
        )
        q_sim.run()
        assert q_sim.qubit_coords == want_coords
        assert q_sim.coords_shifts == [10.0, 0.0, 5.0]

    def test_flip_getters_support_index_and_instance_slicing(self):
        from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8

        # Non-deterministic H 0 1; M 0 1 produces mixed False/True _stim_reference_offsets()
        # because ref_ts (seed=0) collapses [False, True] while Stim's m2d reference is [False, False].
        circuit = stim.Circuit(
            "H 0 1\n"
            "M 0 1\n"
            "DETECTOR rec[-2]\n"
            "DETECTOR rec[-1]\n"
            "DETECTOR rec[-2] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-2]\n"
            "OBSERVABLE_INCLUDE(1) rec[-1]\n"
        )
        fss = FlipsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8().compile_op_handler(circuit=circuit, batch_size=12),
            batch_size=12,
            seed=19,
        )
        fss.run()
        det_off, obs_off = fss._stim_reference_offsets()
        assert det_off is not None and obs_off is not None

        full_dets = fss.get_detector_flips(bit_packed=False)
        full_dets_bp = fss.get_detector_flips(bit_packed=True)
        full_obs = fss.get_observable_flips(bit_packed=False)
        full_obs_bp = fss.get_observable_flips(bit_packed=True)

        for d_idx in (0, 1, 2, -1):
            np.testing.assert_array_equal(fss.get_detector_flips(detector_index=d_idx), full_dets[d_idx])
            np.testing.assert_array_equal(
                fss.get_detector_flips(detector_index=d_idx, bit_packed=True),
                np.packbits(full_dets[d_idx], bitorder="little"),
            )
        for o_idx in (0, 1, -1):
            np.testing.assert_array_equal(fss.get_observable_flips(observable_index=o_idx), full_obs[o_idx])
            np.testing.assert_array_equal(
                fss.get_observable_flips(observable_index=o_idx, bit_packed=True),
                np.packbits(full_obs[o_idx], bitorder="little"),
            )
        for s_idx in (0, 5, -1):
            np.testing.assert_array_equal(fss.get_detector_flips(instance_index=s_idx), full_dets[:, s_idx])
            np.testing.assert_array_equal(fss.get_detector_flips(instance_index=s_idx, bit_packed=True), full_dets_bp[s_idx])
            np.testing.assert_array_equal(fss.get_observable_flips(instance_index=s_idx), full_obs[:, s_idx])
            np.testing.assert_array_equal(fss.get_observable_flips(instance_index=s_idx, bit_packed=True), full_obs_bp[s_idx])
            assert bool(fss.get_detector_flips(detector_index=-1, instance_index=s_idx)) == bool(full_dets[-1, s_idx])
            assert int(fss.get_detector_flips(detector_index=-1, instance_index=s_idx, bit_packed=True)) == int(full_dets[-1, s_idx])
            assert bool(fss.get_observable_flips(observable_index=0, instance_index=s_idx)) == bool(full_obs[0, s_idx])
            assert int(fss.get_observable_flips(observable_index=0, instance_index=s_idx, bit_packed=True)) == int(full_obs[0, s_idx])

    def test_flip_sync_reference_handles_repeated_qubit_mpp_terms(self):
        from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8

        circuit = stim.Circuit(
            "H 0 1\n"
            "MPP X0*X0*X1 !X0*X0 Z0*Z0\n"
            "R 0 1\n"
            "MPP X0*Y0*X1*Y1\n"
            "DETECTOR rec[-4]\n"
            "DETECTOR rec[-3]\n"
            "DETECTOR rec[-2]\n"
            "DETECTOR rec[-1]\n"
        )
        fss = FlipsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8().compile_op_handler(circuit=circuit, batch_size=32),
            batch_size=32,
            seed=7,
        )
        fss.run()
        recs = fss.get_final_measurement_records()
        assert recs.shape == (32, 4)
        np.testing.assert_array_equal(recs[:, 0], False)
        np.testing.assert_array_equal(recs[:, 1], True)
        np.testing.assert_array_equal(recs[:, 2], False)
        np.testing.assert_array_equal(recs[:, 3], True)
        assert not np.any(fss.get_detector_flips())

    def test_get_flips_bit_packed_rejects_extra_args_and_kwargs(self):
        from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8

        circuit = stim.Circuit("R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
        fss = FlipsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8().compile_op_handler(circuit=circuit, batch_size=8),
            batch_size=8,
            seed=3,
        )
        fss.run()
        with pytest.raises(TypeError):
            fss.get_detector_flips(bit_packed=True, append_observables=True)
        with pytest.raises(TypeError):
            fss.get_detector_flips(bit_packed=True, detector_idx=0)
        with pytest.raises(TypeError):
            fss.get_observable_flips(bit_packed=True, observable_idx=0)


@pytest.mark.parametrize("bad_bs", [0, -1, True])
def test_batch_size_validation_across_simulators_and_samplers(bad_bs):
    from stimside import (
        CosetsideSampler,
        CosetsideSimulator,
        FlipsideSampler,
        MagicsideSampler,
        MagicsideSimulator,
        TablesideSampler,
        TablesideSimulator,
    )
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
    from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8 as LeakageUint8Flip
    from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8 as LeakageUint8Tab

    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    for sampler_cls, handler in (
        (FlipsideSampler, LeakageUint8Flip()),
        (TablesideSampler, LeakageUint8Tab()),
        (CosetsideSampler, LeakageUint8Coset()),
        (MagicsideSampler, LeakageUint8Magic()),
    ):
        with pytest.raises(ValueError, match="batch_size"):
            sampler_cls(op_handler=handler, dem_decoder="pymatching", batch_size=bad_bs)

    coh_f = LeakageUint8Flip().compile_op_handler(circuit=circuit, batch_size=1)
    coh_t = LeakageUint8Tab().compile_op_handler(circuit=circuit, batch_size=1)
    coh_c = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=1)
    coh_m = LeakageUint8Magic().compile_op_handler(circuit=circuit, batch_size=1)
    for sim_cls, coh in (
        (FlipsideSimulator, coh_f),
        (TablesideSimulator, coh_t),
        (CosetsideSimulator, coh_c),
        (MagicsideSimulator, coh_m),
    ):
        with pytest.raises(ValueError, match="batch_size"):
            sim_cls(circuit=circuit, compiled_op_handler=coh, batch_size=bad_bs, seed=1)


def test_getter_outputs_are_isolated_from_in_place_mutation():
    from stimside import CosetsideSimulator, MagicsideSimulator, TablesideSimulator
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
    from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8 as LeakageUint8Flip
    from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8 as LeakageUint8Tab

    circuit = stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sims = [
        FlipsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8Flip().compile_op_handler(circuit=circuit, batch_size=4),
            batch_size=4,
            seed=5,
        ),
        TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8Tab().compile_op_handler(circuit=circuit, batch_size=4),
            batch_size=4,
            seed=5,
        ),
        CosetsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=4),
            batch_size=4,
            seed=5,
        ),
        MagicsideSimulator(
            circuit=circuit,
            compiled_op_handler=LeakageUint8Magic().compile_op_handler(circuit=circuit, batch_size=4),
            batch_size=4,
            seed=5,
        ),
    ]
    for sim in sims:
        sim.run()
        orig_recs = sim.get_final_measurement_records().copy()
        recs = sim.get_final_measurement_records()
        recs ^= True
        np.testing.assert_array_equal(sim.get_final_measurement_records(), orig_recs)

        for bp in (False, True):
            orig_dets = sim.get_detector_flips(bit_packed=bp).copy()
            dets = sim.get_detector_flips(bit_packed=bp)
            dets ^= 1 if bp else True
            np.testing.assert_array_equal(sim.get_detector_flips(bit_packed=bp), orig_dets)

            orig_obs = sim.get_observable_flips(bit_packed=bp).copy()
            obs = sim.get_observable_flips(bit_packed=bp)
            obs ^= 1 if bp else True
            np.testing.assert_array_equal(sim.get_observable_flips(bit_packed=bp), orig_obs)
