import numpy as np
import pytest
import stim # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.simulator_tableau import TablesideSimulator


class MockCompiledOpHandler(CompiledOpHandler[TablesideSimulator]):
    def claim_ops(self) -> set[str]:
        return {"MOCK"}

    def handle_op(self, op: stim.CircuitInstruction, sss: "TablesideSimulator") -> None:
        if op in self.claim_ops():
            raise NotImplementedError("mock handle_op called and somehow matched")
        else:
            sss._do_bare_instruction(op)
            return

    def clear(self) -> None:
        raise NotImplementedError("mock clear called")


class TestTablesideSimulator:

    batch_size = 128
    coh = MockCompiledOpHandler()

    def make_simulator(self, circuit):
        return TablesideSimulator(
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
        tss = self.make_simulator(circuit)
        tss.interactive_do(circuit[0])  # QUBIT_COORDS(0,0) 0
        tss.interactive_do(circuit[1])  # QUBIT_COORDS(1,0) 1
        assert tss.qubit_coords == {0: [0, 0], 1: [1, 0]}
        tss.interactive_do(circuit[2])  # SHIFT_COORDS(5, 5)
        assert tss.qubit_coords == {0: [0, 0], 1: [1, 0]}
        assert tss.coords_shifts == [5, 5]
        tss.interactive_do(circuit[3])  # QUBIT_COORDS(0) 5
        tss.interactive_do(circuit[4])  # QUBIT_COORDS(1, 0) 6
        assert tss.qubit_coords == {0: [0, 0], 1: [1, 0], 5: [5], 6: [6, 5]}
        tss.interactive_do(circuit[5])  # SHIFT_COORDS(5)
        assert tss.coords_shifts == [10, 5]
        tss.interactive_do(circuit[6])  # SHIFT_COORDS(5,5,5)
        assert tss.coords_shifts == [15, 10, 5]
        tss.interactive_do(circuit[7])  # QUBIT_COORDS[QUBIT_TAG](0, 0)
        assert tss.qubit_coords == {0: [0, 0], 1: [1, 0], 5: [5], 6: [6, 5], 10: [15, 10]}

    def test_run_and_clear(self):
        circuit = stim.Circuit(
            """
            R 0 1 2 3
        """
        )
        tss = self.make_simulator(circuit)

        tss.run()

        with pytest.raises(ValueError, match=r"prepare the simulator for a new run"):
            tss.run()

        with pytest.raises(NotImplementedError, match="mock clear called"):
            tss.clear()

        assert tss.qubit_coords == {}
        assert tss.qubit_tags == {}
        assert tss.coords_shifts == []

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

        tss = self.make_simulator(circuit)
        tss.run()

        assert tss.batch_size == self.batch_size
        assert tss.num_qubits == circuit_num_qubits

        assert tss.get_detector_flips().shape == (self.batch_size, circuit_num_detectors)
        assert tss.get_final_measurement_records().shape == (self.batch_size, circuit_num_measurements)
        assert tss.get_observable_flips().shape == (self.batch_size, circuit_num_observables)

    def test_batch_size_1_detector_and_observable_flips(self):
        circuit = stim.Circuit(
            """
            X_ERROR(1.0) 0 4
            M 0 1 2 3 4 5 6 7
            DETECTOR(0) rec[-8]
            DETECTOR(1) rec[-7]
            OBSERVABLE_INCLUDE(0) rec[-4] rec[-3]
        """
        )
        from stimside.op_handlers.abstract_op_handler import TrivialOpHandler

        coh = TrivialOpHandler().compile_op_handler(circuit=circuit, batch_size=1)
        tss = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=coh,
            batch_size=1,
            seed=42,
            construct_reference_circuit=True,
        )
        tss.run()
        det_flips = tss.get_detector_flips()
        obs_flips = tss.get_observable_flips()
        assert det_flips.shape == (1, 2)
        assert obs_flips.shape == (1, 1)
        assert det_flips[0, 0] == True
        assert det_flips[0, 1] == False
        assert obs_flips[0, 0] == True

        tss.clear()
        assert tss._detector_flips is None
        assert tss._observable_flips is None
        assert tss._finished_running_circuit is False
        assert len(tss._new_reference_circuit) == 0

    def test_deterministic_seeding_and_reference_coords(self):
        circuit = stim.Circuit(
            """
            QUBIT_COORDS(1, 2) 0
            SHIFT_COORDS(3, 4)
            H 0
            M 0
            DETECTOR(0) rec[-1]
        """
        )
        from stimside.op_handlers.abstract_op_handler import TrivialOpHandler

        coh1 = TrivialOpHandler().compile_op_handler(circuit=circuit, batch_size=1)
        coh2 = TrivialOpHandler().compile_op_handler(circuit=circuit, batch_size=1)
        tss1 = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=coh1,
            batch_size=1,
            seed=999,
            construct_reference_circuit=True,
        )
        tss2 = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=coh2,
            batch_size=1,
            seed=999,
            construct_reference_circuit=True,
        )
        outs1, outs2 = [], []
        for _ in range(10):
            tss1.clear()
            tss1.run()
            outs1.append(bool(tss1.get_detector_flips()[0, 0]))

            tss2.clear()
            tss2.run()
            outs2.append(bool(tss2.get_detector_flips()[0, 0]))

        assert outs1 == outs2
        ref_names = [inst.name for inst in tss1._new_reference_circuit]
        assert "QUBIT_COORDS" in ref_names
        assert "SHIFT_COORDS" in ref_names

    def test_cpp_vs_python_kernels_mode_a_and_mode_b_up_to_d15(self):
        from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
            LeakageUint8,
        )

        for d, mode_b in [(5, False), (5, True), (15, False), (15, True)]:
            base = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=d,
                after_clifford_depolarization=0.001,
                before_measure_flip_probability=0.001,
                after_reset_flip_probability=0.001,
            ).flattened()
            mr_qubits = {
                t.value
                for inst in base
                if inst.name == "MR"
                for t in inst.targets_copy()
            }
            data_qubits = sorted(set(range(base.num_qubits)) - mr_qubits)
            circ = stim.Circuit()
            for inst in base:
                if inst.name == "CX":
                    targets = inst.targets_copy()
                    t_qs = [targets[i] for i in range(1, len(targets), 2)]
                    circ.append("H", t_qs)
                    circ.append(
                        "CZ", targets, [], tag="CONDITIONED_ON_PAIR: (U, U)"
                    )
                    circ.append(
                        "II_ERROR",
                        targets,
                        [0.0],
                        tag="LEAKAGE_TRANSITION_2: (0.002, U_U-->2_D)",
                    )
                    circ.append("H", t_qs)
                elif inst.name == "MR":
                    targets = inst.targets_copy()
                    if mode_b:
                        circ.append(
                            "I",
                            targets,
                            [],
                            tag="LEAKAGE_TRANSITION_1: (0.01, 1-->2)",
                        )
                    circ.append(
                        "M",
                        targets,
                        inst.gate_args_copy(),
                        tag="LEAKAGE_PROJECTION_Z: (1.0, 2)",
                    )
                    circ.append("R", targets)
                    circ.append(
                        "I",
                        targets,
                        [],
                        tag="LEAKAGE_TRANSITION_1: (0.5, 2-->0) (0.5, 2-->U)",
                    )
                    circ.append(
                        "I",
                        data_qubits,
                        [],
                        tag="LEAKAGE_TRANSITION_1: (1.0, 2-->U)",
                    )
                elif inst.name == "M":
                    circ.append(
                        "M",
                        inst.targets_copy(),
                        inst.gate_args_copy(),
                        tag="LEAKAGE_PROJECTION_Z: (1.0, 2)",
                    )
                else:
                    circ.append(inst)

            coh = LeakageUint8(
                unconditional_condition_on_U=True
            ).compile_op_handler(circuit=circ, batch_size=1)
            tss = TablesideSimulator(
                circuit=circ,
                compiled_op_handler=coh,
                batch_size=1,
                seed=1234,
                construct_reference_circuit=True,
            )
            assert tss._cpp_runner is not None
            assert tss._running_tableau is mode_b
            tss.run()
            recs = tss.get_final_measurement_records()
            dets = tss.get_detector_flips()
            obs = tss.get_observable_flips()
            assert recs.shape == (1, circ.num_measurements)
            assert dets.shape == (1, circ.num_detectors)
            assert obs.shape == (1, circ.num_observables)
            assert len(tss._new_circuit) > 0
            assert len(coh.state) == circ.num_qubits
            with pytest.raises(TypeError):
                TablesideSimulator(
                    circuit=circ,
                    compiled_op_handler=coh,
                    use_cpp_kernels=False,  # type: ignore[call-arg]
                )


def test_concurrent_simulators_and_fused_mode_b_new_circuit():
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        LeakageUint8,
    )

    circ = stim.Circuit(
        """
        X 0
        I[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0 1
        M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    coh1 = LeakageUint8(unconditional_condition_on_U=True).compile_op_handler(
        circuit=circ, batch_size=1
    )
    coh2 = LeakageUint8(unconditional_condition_on_U=True).compile_op_handler(
        circuit=circ, batch_size=1
    )
    sim1 = TablesideSimulator(
        circuit=circ, compiled_op_handler=coh1, batch_size=8, seed=101
    )
    sim2 = TablesideSimulator(
        circuit=circ, compiled_op_handler=coh2, batch_size=8, seed=202
    )
    # Concurrent instances on the same circuit must not share state buffers
    assert coh1.state is not coh2.state
    sim1.run()
    assert list(coh1.state) == [2, 0]
    assert list(coh2.state) == [0, 0]
    sim2.run()
    assert list(coh1.state) == [2, 0]
    assert list(coh2.state) == [2, 0]

    # Verify fused Mode B injects R and X_ERROR(1) BEFORE M in _new_circuit so all batch_size=8 shots measure [1, 0]
    coh = LeakageUint8(unconditional_condition_on_U=True).compile_op_handler(
        circuit=circ, batch_size=1
    )
    sim = TablesideSimulator(
        circuit=circ,
        compiled_op_handler=coh,
        batch_size=8,
        seed=303,
    )
    sim.run()
    recs = sim.get_final_measurement_records()
    assert recs.shape == (8, 2)
    assert np.all(recs[:, 0] == True)
    assert np.all(recs[:, 1] == False)
    assert str(sim.peek_bloch(1)) == "+Z"


def test_exotic_circuit_end_to_end_tableside_vs_cosetside_cross_validation():
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import (
        LeakageUint8Coset,
    )
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        LeakageUint8 as LeakageUint8Tableau,
    )
    from stimside.simulator_coset import CosetsideSimulator

    circ = stim.Circuit(
        """
        R 0 1 2 3
        REPEAT 2 {
            X 0
            CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1 2 3
            II_ERROR[LEAKAGE_TRANSITION_2: (0.25, U_U-->2_3) (0.25, 2_3-->U_Y)] 0 1
            I[LEAKAGE_TRANSITION_1: (0.3, 1-->2) (0.4, 2-->U) (0.4, 3-->0)] 0 1 2 3
            PAULI_CHANNEL_1(0.05, 0.05, 0.05) 0 1 2 3
            PAULI_CHANNEL_2(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01) 0 1 2 3
            CORRELATED_ERROR(0.2) X0 Z1
            ELSE_CORRELATED_ERROR(0.3) Y2 X3
            MPAD[LEAKAGE_MEASUREMENT: (0.0, 0) (0.0, 1) (1.0, 2) (1.0, 3) : 0 1] 0 0
            CX rec[-2] 2 rec[-1] 3
            M[LEAKAGE_PROJECTION_Z: (1.0, 2) (1.0, 3)] 0 1 2 3
            SHIFT_COORDS(0, 1)
            DETECTOR(0, 0) rec[-4] rec[-3]
        }
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
        """
    )

    for seed in (10, 20, 30):
        coh_t = LeakageUint8Tableau(
            unconditional_condition_on_U=True
        ).compile_op_handler(circuit=circ, batch_size=1)
        tss = TablesideSimulator(
            circuit=circ,
            compiled_op_handler=coh_t,
            batch_size=1,
            seed=seed,
        )
        assert tss._cpp_runner is not None
        tss.run()
        assert tss.get_final_measurement_records().shape == (1, circ.num_measurements)
        assert tss.get_detector_flips().shape == (1, circ.num_detectors)
        assert tss.get_observable_flips().shape == (1, circ.num_observables)

        coh_t_sync = LeakageUint8Tableau(
            unconditional_condition_on_U=True
        ).compile_op_handler(circuit=circ, batch_size=1)
        tss_sync = TablesideSimulator(
            circuit=circ,
            compiled_op_handler=coh_t_sync,
            batch_size=1,
            seed=seed,
            sync_tableside_rng=True,
        )
        tss_sync.run()

        coh_c_sync = LeakageUint8Coset(
            unconditional_condition_on_U=True
        ).compile_op_handler(circuit=circ, batch_size=1)
        css_sync = CosetsideSimulator(
            circuit=circ,
            compiled_op_handler=coh_c_sync,
            batch_size=1,
            seed=seed,
            sync_tableside_rng=True,
        )
        css_sync.run()

        np.testing.assert_array_equal(
            tss_sync.get_final_measurement_records(),
            css_sync.get_final_measurement_records(),
        )
        np.testing.assert_array_equal(
            tss_sync.get_detector_flips(),
            css_sync.get_detector_flips(),
        )
        np.testing.assert_array_equal(
            tss_sync.get_observable_flips(),
            css_sync.get_observable_flips(),
        )
        np.testing.assert_array_equal(
            coh_t_sync.state.reshape(-1),
            coh_c_sync.state.reshape(-1),
        )


@pytest.mark.parametrize(
    "text",
    [
        "H 0\nREPEAT 3 {\nMPAD[LEAKAGE_MEASUREMENT: (0, 0), (1, 1) : 0] 0\n}\nM 0",
        "REPEAT 3 {\nX_ERROR(0.1) 0\n}\nM 0",
        "REPEAT 3 {\nI[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0\n}\nM 0",
        "REPEAT 3 {\nM[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (1.0, 2)] 0\n}",
        "H 0\nREPEAT 2 {\nH 0\nSHIFT_COORDS(1)\nH 0\n}\nM 0",
    ],
)
def test_repeat_body_with_identical_consecutive_ops(text):
    """Unrolled REPEAT bodies that stim's flattened() would fuse (previously IndexError at precompile)."""
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import (
        LeakageUint8Coset,
    )
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        LeakageUint8 as LeakageUint8Tableau,
    )
    from stimside.simulator_coset import CosetsideSimulator

    circ = stim.Circuit(text)
    for batch_size in (1, 4):
        coh = LeakageUint8Tableau().compile_op_handler(circuit=circ, batch_size=batch_size)
        tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, seed=5, batch_size=batch_size)
        tss.run()
        recs = tss.get_final_measurement_records()
        assert recs.shape == (batch_size, circ.num_measurements)
        if text.startswith("H 0\nREPEAT 3 {\nMPAD"):
            # MPAD with (0, 0), (1, 1) reads out the qubit's Z value, so it matches the final M.
            assert np.all(recs[:, :3] == recs[:, 3:4])

    # Exact RNG lockstep with Cosetside.
    for seed in (1, 2, 3):
        coh_t = LeakageUint8Tableau().compile_op_handler(circuit=circ, batch_size=1)
        tss = TablesideSimulator(
            circuit=circ, compiled_op_handler=coh_t, batch_size=1, seed=seed, sync_tableside_rng=True
        )
        tss.run()
        coh_c = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=1)
        css = CosetsideSimulator(
            circuit=circ, compiled_op_handler=coh_c, batch_size=1, seed=seed, sync_tableside_rng=True
        )
        css.run()
        np.testing.assert_array_equal(
            tss.get_final_measurement_records(), css.get_final_measurement_records()
        )
        np.testing.assert_array_equal(coh_t.state.reshape(-1), coh_c.state.reshape(-1))
        assert tss.coords_shifts == css.coords_shifts


def test_repeat_body_fusion_leak_event_op_indices():
    """Deterministic leaks after a fusable REPEAT body: unrolled op indices must match Cosetside."""
    from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import (
        LeakageUint8Coset,
    )
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        LeakageUint8 as LeakageUint8Tableau,
    )
    from stimside.simulator_coset import CosetsideSimulator

    up = "I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0"
    dn = "I[LEAKAGE_TRANSITION_1: (1.0, 2-->U)] 0"
    circ = stim.Circuit(
        f"REPEAT 3 {{\nX_ERROR(0.1) 2\n}}\nREPEAT 2 {{\n{up}\nTICK\n{dn}\nTICK\n}}\n"
        f"REPEAT 2 {{\n{up}\n}}\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0"
    )
    coh_c = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=1)
    css = CosetsideSimulator(circuit=circ, compiled_op_handler=coh_c, batch_size=1, seed=1,
                             record_unleaked_to_leaked=True, record_leakage_events=True)
    expected_u2l = [list(x) for x in css.run()]
    assert expected_u2l == [[3, 7, 11]]
    coh = LeakageUint8Tableau().compile_op_handler(circuit=circ, batch_size=1)
    tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, batch_size=1, seed=1,
                             record_unleaked_to_leaked=True, record_leakage_events=True)
    assert [list(x) for x in tss.run()] == expected_u2l
    assert tss._leakage_events == css._leakage_events


def _leakage_tss(circuit: stim.Circuit, batch_size: int) -> TablesideSimulator:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    coh = LeakageUint8().compile_op_handler(circuit=circuit, batch_size=1)
    return TablesideSimulator(circuit=circuit, compiled_op_handler=coh, batch_size=batch_size, seed=5)


def _outputs(tss: TablesideSimulator) -> list[np.ndarray]:
    return [tss.get_final_measurement_records(), tss.get_detector_flips(), tss.get_observable_flips()]


@pytest.mark.parametrize("batch_size", [1, 8])
def test_interactive_do_exact_replay_and_noise_strength_match_truth(batch_size):
    # Deterministic circuits: run() may take the kernel path, which draws randomness differently.
    def circuit(p):
        return stim.Circuit(
            f"X 0\nX_ERROR({p}) 1\nM 0\nCX 0 1 rec[-1] 2\nM({p}) 1 2\nDETECTOR rec[-1] rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-3]\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nX 0 1\nM 0 1"
        )

    truth = _leakage_tss(circuit(1), batch_size)
    truth.run()
    for construct in (circuit(1), circuit(0)):  # exact replay; an untagged noisy gate's strength
        tss = _leakage_tss(construct, batch_size)
        tss.interactive_do(circuit(1))
        tss.finish_interactive_run()
        for x, y in zip(_outputs(truth), _outputs(tss)):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_interactive_do_checks_tags_and_arguments(batch_size):
    leak = "I[LEAKAGE_TRANSITION_1: ({}, U-->2)] 0"
    for a, b in [
        ("X 0\nCX 0 1\nM 1", "X 0\nCX[foo] 0 1\nM 1"),  # only a tag added
        ("R 0\nI 0\nX 0\nM 0", f"R 0\n{leak.format(1.0)}\nX 0\nM 0"),
        (f"R 0\n{leak.format(0.0)}\nX 0\nM 0", f"R 0\n{leak.format(1.0)}\nX 0\nM 0"),
        ("X 0\nM 0\nDETECTOR rec[-1]", "X 0\nM !0\nDETECTOR rec[-1]"),
        ("I 0\nM 0\nDETECTOR rec[-1]", "X 0\nM 0\nDETECTOR rec[-1]"),
        ("MPAD 0\nDETECTOR rec[-1]", "MPAD 1\nDETECTOR rec[-1]"),
        (
            "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]",
            "M 0 1\nOBSERVABLE_INCLUDE(1) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
        ),
    ]:
        # The op handler's tag map and the detector/observable converter are the construction
        # circuit's, so these used to be accepted with the old tag / reference / definition.
        with pytest.raises(ValueError, match="doesn't match the circuit"):
            _leakage_tss(stim.Circuit(a), batch_size).interactive_do(stim.Circuit(b))
    # flattened() drops the SHIFT_COORDS (operation 2 here), so it doesn't replay the circuit.
    rounds = stim.Circuit("REPEAT 2 {\n    M 0\n    DETECTOR(0) rec[-1]\n    SHIFT_COORDS(1)\n}")
    with pytest.raises(ValueError, match=r"not `circuit\.flattened\(\)` or `circuit \+ more`"):
        _leakage_tss(rounds, batch_size).interactive_do(rounds.flattened())


@pytest.mark.parametrize("batch_size", [1, 8])
def test_interactive_do_past_the_end(batch_size):
    circuit = stim.Circuit("X 0\nM 0\nDETECTOR rec[-1]")
    extension = stim.Circuit("X 1\nCX 1 0\nTICK\nM 0 1")
    truth = _leakage_tss(circuit + extension, batch_size)
    truth.run()
    tss = _leakage_tss(circuit, batch_size)
    tss.interactive_do(circuit)
    tss.interactive_do(extension)
    tss.finish_interactive_run()
    np.testing.assert_array_equal(
        tss.get_final_measurement_records(), truth.get_final_measurement_records()
    )
    with pytest.raises(ValueError):  # the construction circuit's converter: 1 measurement
        tss.get_detector_flips()
    for op in [
        "DETECTOR rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
        "I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0",
        "X[foo] 0",
    ]:
        tss = _leakage_tss(circuit, batch_size)
        tss.interactive_do(circuit)
        with pytest.raises(ValueError, match="past the end of the circuit"):
            tss.interactive_do(stim.Circuit(op))


@pytest.mark.parametrize("records_first", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        # The leaked qubit skips the X: M 0 is 0, the circuit's reference is 1.
        "R 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nX 0\nM 0\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]",
        # The leak resets the qubit to 0: M 0 is 0, the circuit's reference is 1.
        "X 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nI[LEAKAGE_TRANSITION_1: (1.0, 2-->0)] 0\n"
        "M 0\nDETECTOR rec[-1]",
    ],
)
def test_batched_detection_events_are_relative_to_the_circuit_reference(
    text, records_first
):
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circuit = stim.Circuit(text)
    tss = TablesideSimulator(
        circuit=circuit,
        compiled_op_handler=LeakageUint8().compile_op_handler(circuit=circuit, batch_size=4),
        batch_size=4,
        seed=0,
    )
    tss.run()
    if records_first:
        tss.get_final_measurement_records()
    assert tss.get_detector_flips()[:, 0].tolist() == [True] * 4
    assert tss.get_final_measurement_records()[:, 0].tolist() == [False] * 4


def test_cpp_format_op_round_trips_high_precision_gate_args():
    p_filt = 0.12345678901234568
    p_meas = 0.09876543210987654
    circuit = stim.Circuit(
        f"I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
        f"X_ERROR({p_filt!r}) 0 1\n"
        f"M[LEAKAGE_PROJECTION_Z: ({p_meas!r}, 2)] 0 1"
    )
    tss = _leakage_tss(circuit, batch_size=1)
    tss.run()
    x_err_args = [
        inst.gate_args_copy()[0]
        for inst in tss._new_circuit
        if inst.name == "X_ERROR"
    ]
    assert x_err_args == [p_filt, p_meas]


def test_retained_handler_state_survives_gc_and_cache_eviction():
    import gc
    import weakref
    from stimside._tableau.runner import _PRECOMPILED_CACHE
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    c = stim.Circuit("I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0 1")
    coh1 = LeakageUint8().compile_op_handler(circuit=c, batch_size=1)
    sim1 = TablesideSimulator(circuit=c, compiled_op_handler=coh1, batch_size=1, seed=11)
    owner_ref = weakref.ref(sim1._cpp_runner._raw_state.base.obj._engine_owner)
    sim1.run()
    view1 = coh1.state[:2]
    base1 = coh1.state.base
    assert view1.tolist() == [2, 0]

    # A second simulator while coh1/view1/base1 are alive gets a fresh runner and cannot clobber view1.
    del sim1
    gc.collect()
    coh2 = LeakageUint8().compile_op_handler(circuit=c, batch_size=1)
    sim2 = TablesideSimulator(circuit=c, compiled_op_handler=coh2, batch_size=1, seed=12)
    sim2.clear()
    assert view1.tolist() == [2, 0]

    # Cache eviction + GC check: FastEngine buffer must stay valid until coh1, view1, & base1 are dropped.
    del sim2, coh2
    _PRECOMPILED_CACHE.clear()
    gc.collect()
    assert owner_ref() is not None
    assert view1.tolist() == [2, 0]
    coh1.clear()
    assert view1.tolist() == [0, 0]

    del coh1, view1, base1
    gc.collect()
    assert owner_ref() is None


def test_ultra_low_prob_next_skip_does_not_overflow():
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circuit = stim.Circuit(
        "I[LEAKAGE_TRANSITION_1: (1e-9, U-->2)] 0 1 2 3\n"
        "II[LEAKAGE_TRANSITION_2: (1e-9, U_U-->2_2)] 0 1 2 3\n"
        "I[LEAKAGE_TRANSITION_1: (1e-12, 1-->2)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1"
    )
    tss = TablesideSimulator(
        circuit=circuit,
        compiled_op_handler=LeakageUint8().compile_op_handler(circuit=circuit, batch_size=1),
        batch_size=1,
        seed=42,
    )
    for _ in range(1000):
        tss.clear()
        tss.run()
    assert tss.get_final_measurement_records().shape == (1, 2)


def test_in_place_circuit_mutation_and_custom_ops_to_params_isolation():
    from stimside.op_handlers.leakage_handlers.leakage_parameters import LeakageTransition1Params
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        CompiledLeakageUint8,
        LeakageUint8,
    )

    # 1. In-place instruction fusion keeps len(c) == 2 while adding leaked target 1.
    c = stim.Circuit("I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0 1")
    s1 = TablesideSimulator(
        circuit=c,
        compiled_op_handler=LeakageUint8().compile_op_handler(circuit=c, batch_size=1),
        batch_size=1,
        seed=7,
    )
    s1.run()
    assert s1._compiled_op_handler.state.tolist() == [2, 0]

    c.pop()
    c.append("I", [1], tag="LEAKAGE_TRANSITION_1: (1.0, U-->2)")
    c.append("M", [0, 1])
    assert len(c) == 2
    s2 = TablesideSimulator(
        circuit=c,
        compiled_op_handler=LeakageUint8().compile_op_handler(circuit=c, batch_size=1),
        batch_size=1,
        seed=7,
    )
    s2.run()
    assert s2._compiled_op_handler.state.tolist() == [2, 2]

    # 2. Custom ops_to_params must not poison subsequent default compilations on the same circuit.
    c_custom = stim.Circuit("I[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nM 0")
    op0 = c_custom[0]
    custom_coh = CompiledLeakageUint8(
        num_qubits=1,
        batch_size=1,
        unconditional_condition_on_U=True,
        ops_to_params={
            op0: LeakageTransition1Params(((1.0, "U", 2),), from_tag=op0.tag),
        },
    )
    s_custom = TablesideSimulator(circuit=c_custom, compiled_op_handler=custom_coh, batch_size=1, seed=3)
    s_custom.run()
    assert custom_coh.state.tolist() == [2]

    default_coh = LeakageUint8().compile_op_handler(circuit=c_custom, batch_size=1)
    s_default = TablesideSimulator(circuit=c_custom, compiled_op_handler=default_coh, batch_size=1, seed=3)
    s_default.run()
    assert default_coh.state.tolist() == [0]


def test_sample_tableau_transitions_sparse_multibranch() -> None:
    from stimside.op_handlers.leakage_handlers.leakage_parameters import LeakageTransition1Params
    from stimside.op_handlers.leakage_handlers.transition_effects import sample_tableau_transitions

    params = LeakageTransition1Params(((0.01, "U", 2), (0.02, "U", 3)), from_tag="LEAKAGE_TRANSITION_1")
    rng = np.random.default_rng(42)
    res = sample_tableau_transitions(params, "U", 1000, rng, sync_rng=False)
    assert res is not None
    chosen, outs = res
    assert isinstance(chosen, np.ndarray) and len(chosen) == len(outs) > 0
    assert set(outs) == {"2", "3"} or set(outs) == {2, 3}
    single = LeakageTransition1Params(((0.03, "U", 2),), from_tag="LEAKAGE_TRANSITION_1")
    res_single = sample_tableau_transitions(single, "U", 1000, rng, sync_rng=False)
    assert res_single is not None and len(res_single[0]) == len(res_single[1]) > 0


@pytest.mark.parametrize("p", [1e-17, 1e-300])
def test_sub_epsilon_leakage_transition_skip_no_crash(p: float) -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    # Exercises OP_TRANSITION_1_U (kind 3), OP_TRANSITION_2_UU (kind 4), and OP_FUSED_TRANS1_MEAS (kind 7)
    # when 0 < p <= DBL_EPSILON / 2 (where 1.0 - p == 1.0 in IEEE-754 float64).
    circuits = [
        (3, stim.Circuit(f"I[LEAKAGE_TRANSITION_1: ({p}, U-->2)] 0 1 2 3\nM 0 1 2 3")),
        (4, stim.Circuit(f"II[LEAKAGE_TRANSITION_2: ({p}, U_U-->2_U)] 0 1 2 3\nM 0 1 2 3")),
        (
            7,
            stim.Circuit(
                f"I[LEAKAGE_TRANSITION_1: ({p}, 1-->2)] 0 1 2 3\n"
                "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1 2 3"
            ),
        ),
    ]
    for expected_kind, circ in circuits:
        coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
        tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, batch_size=1, seed=1)
        assert int(tss._cpp_runner.pre.steps_py[0][0]) == expected_kind
        tss.run()
        assert tss.get_final_measurement_records().shape == (1, 4)
        assert coh.state.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("prog", "exact"),
    [
        ("H 0\nX_ERROR[CONDITIONED_ON_SELF: 1](1.0) 0\nM 0", [0.0]),
        (
            "H 0\nI[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0",
            [0.5],
        ),
        (
            "H 0\nI[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0\nX 1\nM[LEAKAGE_PROJECTION_Z: (0.0, 2)] 0",
            [0.0],
        ),
        (
            "X_ERROR(0.5) 0\nI[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0\nM[LEAKAGE_PROJECTION_Z: (0.0, 2)] 0",
            [0.0],
        ),
        (
            "H 0\nII[LEAKAGE_TRANSITION_2: (1.0, 1_U-->2_U)] 0 1\nM[LEAKAGE_PROJECTION_Z: (0.0, 2)] 0",
            [0.0],
        ),
    ],
)
def test_batch_size_gt1_zero_one_collapse_marginals(prog: str, exact: list[float]) -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circ = stim.Circuit(prog)
    coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=64)
    tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, batch_size=64, seed=7)
    acc = np.zeros(circ.num_measurements)
    n_shots = 0
    while n_shots < 4096:
        tss.clear()
        tss.run()
        recs = tss.get_final_measurement_records()
        acc += recs.sum(axis=0)
        n_shots += recs.shape[0]
    got = acc / n_shots
    for g, e in zip(got, exact):
        if e in (0.0, 1.0):
            assert g == e
        else:
            assert abs(g - e) < 5.0 * np.sqrt(e * (1.0 - e) / n_shots)


def test_interactive_batch_leakage_events_and_sync_guard() -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circ = stim.Circuit(
        "R 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0"
    )
    coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=4)
    tss = TablesideSimulator(
        circuit=circ,
        compiled_op_handler=coh,
        batch_size=4,
        seed=1,
        record_unleaked_to_leaked=True,
        record_leakage_events=True,
    )
    tss.interactive_do(circ)
    tss.finish_interactive_run()
    u2l = tss.get_unleaked_to_leaked_records()
    lev = tss.get_leakage_events()
    assert [len(x) for x in u2l] == [1, 1, 1, 1]
    assert [len(x) for x in lev] == [1, 1, 1, 1]

    coh_sync = LeakageUint8().compile_op_handler(circuit=circ, batch_size=4)
    tss_sync = TablesideSimulator(
        circuit=circ,
        compiled_op_handler=coh_sync,
        batch_size=4,
        seed=1,
        sync_tableside_rng=True,
    )
    with pytest.raises(ValueError, match="sync_tableside_rng=True"):
        tss_sync.interactive_do(circ)


@pytest.mark.parametrize("sync", [False, True])
def test_subclassed_compiled_leakage_uint8_is_not_bypassed_by_fast_path(sync: bool) -> None:
    import dataclasses
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        CompiledLeakageUint8,
        LeakageUint8,
    )

    @dataclasses.dataclass
    class _LocationNoise(CompiledLeakageUint8):
        calls: int = 0
        coords_seen: list[dict[int, list[float]]] = dataclasses.field(default_factory=list)

        def handle_op(self, op, sss):
            self.calls += 1
            if op.name == "H":
                self.coords_seen.append(dict(sss.qubit_coords))
                far = [
                    t.value
                    for t in op.targets_copy()
                    if sss.qubit_coords.get(t.value, [0])[0] >= 1
                ]
                super().handle_op(op, sss)
                if far:
                    sss._do_bare_instruction(stim.CircuitInstruction("Z", sorted(set(far))))
                return
            super().handle_op(op, sss)

    circ = stim.Circuit(
        "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 0) 1\n"
        "I[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nH 0 1\nTICK\nH 0 1\nM 0 1"
    )
    base = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
    coh = _LocationNoise(
        num_qubits=base.num_qubits,
        batch_size=1,
        unconditional_condition_on_U=True,
        ops_to_params=base.ops_to_params,
    )
    tss = TablesideSimulator(
        circuit=circ, compiled_op_handler=coh, batch_size=1, seed=1, sync_tableside_rng=sync
    )
    tss.run()
    assert tss.get_final_measurement_records()[0].astype(int).tolist() == [0, 1]
    assert coh.calls == 5
    assert coh.coords_seen[0] == {0: [0.0, 0.0], 1: [1.0, 0.0]}


def test_slice_circuit_blocks_concurrent_first_run_is_thread_safe() -> None:
    import threading
    from stimside._tableau.runner import get_precompiled_circuit
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circ = stim.Circuit("\n".join(["H 0\nTICK"] * 300 + ["I[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0", "M 0"]))
    coh0 = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
    pre = get_precompiled_circuit(circ, True, coh0.ops_to_params)
    assert not pre._full_blocks
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        try:
            coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
            tss = TablesideSimulator(
                circuit=circ, compiled_op_handler=coh, batch_size=1, seed=seed, sync_tableside_rng=True
            )
            barrier.wait()
            tss.run()
            assert tss.get_final_measurement_records().tolist() == [[False]]
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i + 1,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_precompiled_cache_cap_and_metadata_mutation_isolation() -> None:
    from stimside._tableau.runner import _PRECOMPILED_CACHE
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    for k in range(6):
        c = stim.Circuit(f"QUBIT_COORDS({k}, 0) 0\nSHIFT_COORDS(1, 0)\nI[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nM 0")
        coh = LeakageUint8().compile_op_handler(circuit=c, batch_size=1)
        TablesideSimulator(circuit=c, compiled_op_handler=coh, batch_size=1, seed=1).run()
    assert len(_PRECOMPILED_CACHE) <= 4

    c_meta = stim.Circuit(
        "QUBIT_COORDS[my_tag](2, 3) 0\nSHIFT_COORDS(4, 5)\nI[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nM 0"
    )
    coh1 = LeakageUint8().compile_op_handler(circuit=c_meta, batch_size=1)
    tss1 = TablesideSimulator(circuit=c_meta, compiled_op_handler=coh1, batch_size=1, seed=1)
    tss1.run()
    tss1.qubit_coords[0].append(99.0)
    tss1.qubit_coords[99] = [1.0]
    tss1.qubit_tags[99] = "poisoned"
    tss1.coords_shifts.append(99.0)

    coh2 = LeakageUint8().compile_op_handler(circuit=c_meta, batch_size=1)
    tss2 = TablesideSimulator(circuit=c_meta, compiled_op_handler=coh2, batch_size=1, seed=1)
    tss2.run()
    assert tss2.qubit_coords == {0: [2.0, 3.0]}
    assert 99 not in tss2.qubit_tags
    assert tss2.coords_shifts == [4.0, 5.0]


@pytest.mark.parametrize("extra", ["", " (0.001, 1-->4)"], ids=["cpp", "python"])
def test_sparse_leakage_transition_rates_cpp_and_python(extra: str) -> None:
    import math
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    n_q, shots = 8, 4000
    qs = " ".join(str(q) for q in range(n_q))
    circ = stim.Circuit(
        f"R {qs}\nI[LEAKAGE_TRANSITION_1: (0.01, U-->2) (0.02, U-->3){extra}] {qs}\nM {qs}"
    )
    coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
    sim = TablesideSimulator(circ, compiled_op_handler=coh, batch_size=1, seed=7, record_leakage_events=True)
    hits = np.zeros((n_q, 5), dtype=np.int64)
    for s in range(shots):
        if s:
            sim.clear()
        sim.run()
        for e in sim.get_leakage_events()[0]:
            if e.old_state < 2 and e.new_state >= 2:
                hits[e.qubit, e.new_state] += 1
    for st, p in ((2, 0.01), (3, 0.02)):
        tol = 5.0 * math.sqrt(p * (1.0 - p) / shots)
        assert np.all(np.abs(hits[:, st] / shots - p) < tol)


@pytest.mark.parametrize("pre_leak", ["", "I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 1\n"])
def test_fused_trans1_meas_records_leakage_events_and_frame_switch(pre_leak: str) -> None:
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circ = stim.Circuit(
        f"X 0\n{pre_leak}I[LEAKAGE_TRANSITION_1: (0.5, 1-->2)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1"
    )
    coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=1)
    sim = TablesideSimulator(
        circ,
        compiled_op_handler=coh,
        batch_size=1,
        seed=11,
        record_leakage_events=True,
        record_unleaked_to_leaked=True,
    )
    q0_leaks = 0
    for s in range(32):
        if s:
            sim.clear()
        sim.run()
        evs = sim.get_leakage_events()[0]
        if any(e.qubit == 0 and e.new_state == 2 for e in evs):
            q0_leaks += 1
            assert int(sim.get_unleaked_to_leaked_records()[0][-1]) == (2 if pre_leak else 1)
    assert 5 <= q0_leaks <= 27

