"""TablesideSimulator per-shot routing pins: MPAD `has_01` and the generic (handle_op) path."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
    CompiledLeakageUint8,
    LeakageUint8,
)
from stimside.simulator_tableau import TablesideSimulator


def test_mpad_zero_one_keyed_readout_routes_per_shot_in_multibatch_run() -> None:
    # p0 == p1, so only the 0/1-keyed has_01 rule routes this readout to per-shot trajectories.
    # The later rec[-1] feedback must then copy the recorded bit in every shot.
    circ = stim.Circuit(
        "X_ERROR(0.5) 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.5, 0) (0.5, 1) : 0] 0\n"
        "CX rec[-1] 1\n"
        "M 1"
    )
    coh = LeakageUint8().compile_op_handler(circuit=circ, batch_size=64)
    tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, batch_size=64, seed=3)
    assert tss._collapses_z()
    tss.run()
    recs = tss.get_final_measurement_records()
    np.testing.assert_array_equal(recs[:, 0], recs[:, 1])
    assert 0 < recs[:, 0].sum() < 64


def test_generic_path_subclass_routes_zero_one_keys_per_shot_in_multibatch_run() -> None:
    # A handle_op override disables the precompiled runner, so run() takes the generic path and
    # must work out the per-shot routing itself (the field-only subclass test stays on the fast
    # path since WP-09, see test_subclassed_handler_routes_... in simulator_tableau_test.py).
    @dataclasses.dataclass
    class _PassThrough(CompiledLeakageUint8):
        def handle_op(self, op, sss):
            super().handle_op(op, sss)

    circ = stim.Circuit("H 0\nX_ERROR[CONDITIONED_ON_SELF: 1](1.0) 0\nM 0")
    base = LeakageUint8().compile_op_handler(circuit=circ, batch_size=64)
    coh = _PassThrough(
        num_qubits=base.num_qubits,
        batch_size=64,
        unconditional_condition_on_U=True,
        ops_to_params=base.ops_to_params,
    )
    with pytest.warns(UserWarning, match="handle_op"):
        tss = TablesideSimulator(circuit=circ, compiled_op_handler=coh, batch_size=64, seed=7)
    assert tss._precompiled_circuit is None
    for _ in range(8):
        tss.clear()
        tss.run()
        assert not np.any(tss.get_final_measurement_records())
