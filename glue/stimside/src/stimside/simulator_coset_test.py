from concurrent.futures import ThreadPoolExecutor
import dataclasses

import numpy as np
import stim

from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator
from stimside.util.reference_chp import CompiledReferenceCircuit, MeasStepMeta


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
