"""Randomized and exhaustive-gate tests for `CompiledReferenceCircuit.compile_instruction` (C-2)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import stim

from stimside._coset.reference_compile import (
    CompiledReferenceCircuit,
    MeasStepMeta,
    UnitaryStepMeta,
)


def _pauli_targets(rng: random.Random, num_qubits: int, combiner: bool) -> list[Any]:
    paulis = (stim.target_x, stim.target_y, stim.target_z)
    out: list[Any] = []
    for g_idx in range(rng.randint(1, 2)):
        if g_idx and combiner:
            out.append(stim.target_combiner())
        for j, q in enumerate(rng.sample(range(num_qubits), rng.randint(1, 2))):
            if j and combiner:
                out.append(stim.target_combiner())
            out.append(rng.choice(paulis)(q))
    return out


def _random_op_for_gate(
    name: str, rng: random.Random, num_qubits: int
) -> stim.CircuitInstruction | None:
    gate = stim.gate_data(name)
    arg_range = gate.num_parens_arguments_range
    n_args = max(0, arg_range.start)
    args: list[float] = [0] * n_args if name == "OBSERVABLE_INCLUDE" else [0.01] * n_args
    if name in ("SPP", "SPP_DAG", "MPP", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "E"):
        targets = _pauli_targets(rng, num_qubits, combiner=name in ("SPP", "SPP_DAG", "MPP"))
        return stim.CircuitInstruction(name, targets, args)
    if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
        return stim.CircuitInstruction(name, [stim.target_rec(-rng.randint(1, 3))], args)
    if name in ("QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
        return stim.CircuitInstruction(name, [], [])
    if name in ("MXX", "MYY", "MZZ"):
        q0, q1 = rng.sample(range(num_qubits), 2)
        t0: Any = stim.target_inv(q0) if rng.random() < 0.3 else q0
        return stim.CircuitInstruction(name, [t0, q1], args)
    arity = 2 if gate.is_two_qubit_gate else 1
    ng = rng.randint(0, 4)
    count = ng * arity if (gate.is_unitary or gate.is_two_qubit_gate) else rng.randint(0, 3)
    targets = []
    for _ in range(count):
        roll = rng.random()
        if gate.is_two_qubit_gate and roll < 0.25:
            targets.append(stim.target_rec(-rng.randint(1, 3)))
        elif gate.is_two_qubit_gate and roll < 0.35:
            targets.append(stim.target_sweep_bit(rng.randint(0, 2)))
        elif gate.produces_measurements and roll < 0.45:
            targets.append(stim.target_inv(rng.randrange(num_qubits)))
        else:
            targets.append(rng.randrange(num_qubits))
    try:
        return stim.CircuitInstruction(name, targets, args)
    except (ValueError, TypeError):
        return None


def _snap_eq(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_compile_instruction_and_append_agree_with_reference_compiler_across_all_gates() -> None:
    """Seeded fuzz across every `stim.gate_data` name: `compile_instruction` never mutates
    the reference, and `append_compiled_step` matches `compile_and_append_instruction`."""
    rng = random.Random(2026)
    all_gates = sorted(stim.gate_data().keys())
    num_qubits = 6
    base = stim.Circuit("R 0 1 2 3 4 5\nM 0 1 2 3")
    ref = CompiledReferenceCircuit(base)
    ref2 = CompiledReferenceCircuit(base)
    seen: set[str] = set()
    checked = 0
    schedule = list(all_gates) * 2 + [rng.choice(all_gates) for _ in range(3500)]
    for idx, name in enumerate(schedule):
        op = _random_op_for_gate(name, rng, num_qubits)
        if op is None:
            continue
        if idx % 100 == 0:
            ref = CompiledReferenceCircuit(base)
            ref2 = CompiledReferenceCircuit(base)
        steps_before = len(ref.steps)
        chp_before = ref.ref_chp.snapshot_packed()
        last_before = ref._last_packed_snapshot
        try:
            step, next_chp, next_snap = ref.compile_instruction(op)
        except (NotImplementedError, ValueError) as exc1:
            assert len(ref.steps) == steps_before
            assert _snap_eq(ref.ref_chp.snapshot_packed(), chp_before)
            assert _snap_eq(ref._last_packed_snapshot, last_before)
            try:
                ref2.compile_and_append_instruction(op)
            except type(exc1):
                seen.add(name)
                continue
            raise AssertionError(f"{op}: compile_instruction raised {exc1}, append did not")
        assert len(ref.steps) == steps_before
        assert _snap_eq(ref.ref_chp.snapshot_packed(), chp_before)
        assert _snap_eq(ref._last_packed_snapshot, last_before)
        ref.append_compiled_step(step, next_chp, next_snap)
        step2 = ref2.compile_and_append_instruction(op)
        assert type(step) is type(step2) and step.op == step2.op
        if isinstance(step, UnitaryStepMeta) and isinstance(step2, UnitaryStepMeta):
            assert step.target_groups == step2.target_groups
        elif isinstance(step, MeasStepMeta) and isinstance(step2, MeasStepMeta):
            assert len(step.substeps) == len(step2.substeps)
        assert _snap_eq(ref.ref_chp.snapshot_packed(), ref2.ref_chp.snapshot_packed())
        assert _snap_eq(ref._last_packed_snapshot, ref2._last_packed_snapshot)
        seen.add(name)
        checked += 1
    assert seen == set(all_gates)
    assert checked >= 2000
