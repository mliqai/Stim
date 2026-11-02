from concurrent.futures import ThreadPoolExecutor
import dataclasses

import numpy as np
import pytest
import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit, MeasStepMeta
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator


def test_shared_compiled_reference_circuit_no_per_batch_scratch_or_undeclared_attrs() -> None:
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0
        CZ[CONDITIONED_ON_PAIR: (U, U)] 0 2 1 3
        ISWAP[CONDITIONED_ON_PAIR: (U, U)] 0 3 1 2
        M 0
        R 1 2 3
        X_ERROR(0.3) 1 2 3
        M 1 2 3
        """
    )

    def run_once(
        batch_size: int, seed: int, compiled_ref: CompiledReferenceCircuit | None
    ) -> np.ndarray:
        handler = LeakageUint8Coset(unconditional_condition_on_U=True).compile_op_handler(
            circuit=circuit, batch_size=batch_size
        )
        sim = CosetsideSimulator(
            circuit=circuit,
            compiled_op_handler=handler,
            batch_size=batch_size,
            seed=seed,
            compiled_ref=compiled_ref,
        )
        sim.run()
        return sim.get_final_measurement_records()

    expected_a = run_once(5, 101, None)
    expected_b = run_once(17, 202, None)

    shared_ref = CompiledReferenceCircuit(circuit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs_a = [pool.submit(run_once, 5, 101, shared_ref) for _ in range(8)]
        futs_b = [pool.submit(run_once, 17, 202, shared_ref) for _ in range(8)]
        for fut in futs_a:
            np.testing.assert_array_equal(fut.result(), expected_a)
        for fut in futs_b:
            np.testing.assert_array_equal(fut.result(), expected_b)

    for step in shared_ref.steps:
        assert not hasattr(step, "_cpp_m_mat")
        declared = {f.name for f in dataclasses.fields(step)}
        assert set(vars(step)) <= declared, set(vars(step)) - declared
        if isinstance(step, MeasStepMeta):
            for sub in step.substeps:
                sub_declared = {f.name for f in dataclasses.fields(sub.meta)}
                assert set(vars(sub.meta)) <= sub_declared, (
                    set(vars(sub.meta)) - sub_declared
                )


def test_correlated_error_occurred_initialized_and_reset_on_clear() -> None:
    circuit = stim.Circuit("ELSE_CORRELATED_ERROR(1.0) X0\nM 0\nCORRELATED_ERROR(1.0) Z0")
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=8)
    sim = CosetsideSimulator(circuit=circuit, compiled_op_handler=handler, batch_size=8, seed=5)
    assert sim._correlated_error_occurred.shape == (8,)
    assert not np.any(sim._correlated_error_occurred)

    sim.run()
    assert np.all(sim.get_final_measurement_records()[:, 0])
    assert np.all(sim._correlated_error_occurred)

    sim.clear()
    assert not np.any(sim._correlated_error_occurred)
    sim.run()
    assert np.all(sim.get_final_measurement_records()[:, 0])


@pytest.mark.parametrize("sync_rng", [False, True])
@pytest.mark.parametrize(
    "prep_ops,reset_op,meas_op,expected_meas",
    [
        ("X 0", "R 0", "M 0", [0]),
        ("X 0", "MR 0", "M 0", [1, 0]),
        ("H 0\nZ 0", "RX 0", "MX 0", [0]),
        ("H 0\nZ 0", "MRX 0", "MX 0", [1, 0]),
        ("H 0\nS 0\nZ 0", "RY 0", "MY 0", [0]),
        ("H 0\nS 0\nZ 0", "MRY 0", "MY 0", [1, 0]),
    ],
)
def test_coset_resets_when_m_ref_is_one(
    sync_rng: bool,
    prep_ops: str,
    reset_op: str,
    meas_op: str,
    expected_meas: list[int],
):
    """Regression test for N1 & N6: CosetsideSimulator resets and measure-resets when m_ref == 1."""
    circuit = stim.Circuit(f"{prep_ops}\n{reset_op}\n{meas_op}")
    batch_size = 16
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=batch_size
    )
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch_size,
        seed=123,
        sync_tableside_rng=sync_rng,
    )
    sim.run()
    records = sim.get_final_measurement_records()
    assert records.shape == (batch_size, len(expected_meas))
    for idx, exp_val in enumerate(expected_meas):
        assert np.all(records[:, idx] == bool(exp_val))


def test_coset_resets_when_m_ref_is_one_with_active_mrqf():
    """Verify resets when m_ref == 1 and shots have active MRQF states (d >= 1)."""
    circuit = stim.Circuit(
        """
        I[LEAKAGE_TRANSITION_1: (0.5, 0-->2)] 0
        H 1
        CZ 0 1
        X 0
        MR 0
        M 0
        """
    )
    batch_size = 64
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=batch_size
    )
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch_size,
        seed=42,
    )
    sim.run()
    records = sim.get_final_measurement_records()
    assert np.all(records[:, 1] == False)


def test_conditional_pauli_noise_duplicate_targets():
    """Regression test for Bug 4: conditional X_ERROR(1.0) 0 0 cancels out on unleaked shots."""
    circuit = stim.Circuit(
        """
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 1
        X_ERROR[CONDITIONED_ON_OTHER: U : 1 1](1.0) 0 0
        M 0
        """
    )
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=64
    )
    sim = CosetsideSimulator(
        circuit, compiled_op_handler=handler, batch_size=64, seed=99
    )
    sim.run()
    recs = sim.get_final_measurement_records()[:, 0]
    assert np.all(recs == False), (
        "X_ERROR(1.0) 0 0 should apply X^2 = I on unleaked shots"
    )


def test_coset_in_place_circuit_mutation_re_parses_tags():
    c = stim.Circuit("I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0 1")
    h1 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=4)
    s1 = CosetsideSimulator(c, compiled_op_handler=h1, batch_size=4, seed=5)
    s1.run()
    assert np.all(h1.state[0] == 2) and np.all(h1.state[1] == 0)

    # Stim fuses the two identical tagged I instructions, keeping len(c) == 2.
    c.pop()
    c.append("I", [1], tag="LEAKAGE_TRANSITION_1: (1.0, U-->2)")
    c.append("M", [0, 1])
    assert len(c) == 2
    h2 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=4)
    s2 = CosetsideSimulator(c, compiled_op_handler=h2, batch_size=4, seed=5)
    s2.run()
    assert np.all(h2.state[0] == 2) and np.all(h2.state[1] == 2)
