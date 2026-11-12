"""Randomized and exhaustive-gate tests for `_compiled_unitary_rows` mirror parity (R3-F16)."""

from __future__ import annotations

import random
from typing import Any

import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit, UnitaryStepMeta
from stimside.simulator_coset import _compiled_unitary_rows


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


def test_compiled_unitary_rows_agrees_with_reference_compiler_across_all_gates() -> None:
    """Seeded fuzz across every `stim.gate_data` name: `_compiled_unitary_rows` matches
    `CompiledReferenceCircuit.compile_and_append_instruction` (and returns None if it raises)."""
    rng = random.Random(2026)
    all_gates = sorted(stim.gate_data().keys())
    num_qubits = 6
    base = stim.Circuit("R 0 1 2 3 4 5\nM 0 1 2 3")
    ref = CompiledReferenceCircuit(base)
    seen: set[str] = set()
    checked = 0
    schedule = list(all_gates) * 2 + [rng.choice(all_gates) for _ in range(3500)]
    for idx, name in enumerate(schedule):
        op = _random_op_for_gate(name, rng, num_qubits)
        if op is None:
            continue
        if idx % 100 == 0:
            ref = CompiledReferenceCircuit(base)
        pred = _compiled_unitary_rows(op)
        try:
            step = ref.compile_and_append_instruction(op)
        except (NotImplementedError, ValueError):
            assert pred is None, f"{op}: predicted {pred}, compiler raised"
            seen.add(name)
            continue
        expect = len(step.target_groups) if isinstance(step, UnitaryStepMeta) else None
        assert pred == expect, f"{op}: predicted {pred}, compiled {expect}"
        seen.add(name)
        checked += 1
    assert seen == set(all_gates)
    assert checked >= 2000
