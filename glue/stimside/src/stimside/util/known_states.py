from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside.util.circuit import qubit_targets
from stimside.util.circuit_ir import Category, CircuitIR, build_ir, classify_category
from stimside.util.numpy_types import Bool1DArray

_SKIP_KNOWN_STATE_CATS = frozenset(
    {Category.ANNOTATION, Category.NOISE, Category.MPAD, Category.HERALD, Category.OTHER}
)
# Measurement/reset Pauli basis code (+X=1, +Y=3, default +Z=5).
_BASIS_BASE_CODE: dict[str | None, int] = {"X": 1, "Y": 3}


def _without_noise_preserving_heralds(circuit: stim.Circuit) -> stim.Circuit:
    out = stim.Circuit()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    item.repeat_count,
                    _without_noise_preserving_heralds(item.body_copy()),
                )
            )
            continue
        cat = classify_category(item.name)
        if cat is Category.HERALD:
            mpad_targets = [
                1 if t.is_inverted_result_target else 0
                for t in item.targets_copy()
            ]
            out.append(stim.CircuitInstruction("MPAD", mpad_targets))
        elif cat is Category.NOISE:
            continue
        elif (
            cat in (Category.MEASURE, Category.MEASURE_RESET, Category.MPAD)
            and item.gate_args_copy()
        ):
            out.append(
                stim.CircuitInstruction(item.name, item.targets_copy(), [])
            )
        else:
            out.append(item)
    return out


def compute_known_measurements(circuit: stim.Circuit) -> list[int | None]:
    """Make a sample describing each deterministic measurement outcome.

    Basically, like a reference sample, except for non-deterministic measurements,
    which are registered as None instead of receiving any valid value.
    """
    reference_measurement_sampler = _without_noise_preserving_heralds(
        circuit
    ).compile_sampler()

    reference_samples = reference_measurement_sampler.sample(
        shots=128, bit_packed=False
    )
    # shape is (shots, #measurements)
    # shots sets how likely a spurious failure is
    # the likelihood of a 50:50 process flipping N heads in a row is (1/2)**N
    # at N=128 (which is the AVX instruction width), P=3E-39
    # i.e. not going to happen to you

    determined = np.all(reference_samples == reference_samples[0], axis=0)
    # determined has shape (#measurements)

    # np.where does `if determined, copy form reference_sample, else None`
    determined_sample = np.where(determined, reference_samples[0], cast(Any, None))

    return list(determined_sample)


# State encoding in uint8:
# 0: _ (unknown)
# 1: +X, 2: -X
# 3: +Y, 4: -Y
# 5: +Z, 6: -Z
_CODE_TO_PAULI: tuple[stim.PauliString, ...] = (
    stim.PauliString("_"),
    stim.PauliString("+X"),
    stim.PauliString("-X"),
    stim.PauliString("+Y"),
    stim.PauliString("-Y"),
    stim.PauliString("+Z"),
    stim.PauliString("-Z"),
)

_STR_TO_CODE: dict[str, int] = {
    "_": 0,
    "+_": 0,
    "-_": 0,
    "+X": 1,
    "-X": 2,
    "+Y": 3,
    "-Y": 4,
    "+Z": 5,
    "-Z": 6,
}

_GATE_FWD_LUT: dict[str, np.ndarray] = {}
_GATE_BWD_LUT: dict[str, np.ndarray] = {}


def _get_gate_luts(op_name: str) -> tuple[np.ndarray, np.ndarray]:
    if op_name in _GATE_FWD_LUT:
        return _GATE_FWD_LUT[op_name], _GATE_BWD_LUT[op_name]
    fwd = np.zeros(7, dtype=np.uint8)
    bwd = np.zeros(7, dtype=np.uint8)
    inst = stim.CircuitInstruction(name=op_name, targets=[0])
    for code in range(1, 7):
        ps = _CODE_TO_PAULI[code]
        fwd[code] = _STR_TO_CODE[str(ps.after(inst))]
        bwd[code] = _STR_TO_CODE[str(ps.before(inst))]
    _GATE_FWD_LUT[op_name] = fwd
    _GATE_BWD_LUT[op_name] = bwd
    return fwd, bwd


def _forward_known_states(
    ir: CircuitIR,
    known_measurements: list[int | None],
    num_qubits: int,
) -> np.ndarray:
    num_ops = len(ir.instrs)
    states_uint8 = np.empty((num_ops + 1, num_qubits), dtype=np.uint8)
    states_uint8[0, :] = 5  # All qubits start in +Z (|0>)

    for instr in ir.instrs:
        i = instr.unrolled_index
        states_uint8[i + 1] = states_uint8[i]
        cat = instr.category
        if cat in _SKIP_KNOWN_STATE_CATS:
            continue

        op = instr.op
        raw_targets = op.targets_copy()
        if cat in (Category.RESET, Category.MEASURE_RESET):
            q_targets = qubit_targets(raw_targets)
            reset_code = _BASIS_BASE_CODE.get(instr.basis, 5)
            states_uint8[i + 1, q_targets] = reset_code
        elif cat is Category.MEASURE:
            if not instr.is_product_measurement:
                base_code = _BASIS_BASE_CODE.get(instr.basis, 5)
                meas_base = instr.first_measurement
                for offset, t in enumerate(raw_targets):
                    this_measurement = known_measurements[meas_base + offset]
                    q = t.qubit_value
                    if this_measurement is not None:
                        actual_bit = bool(this_measurement) ^ bool(
                            t.is_inverted_result_target
                        )
                        states_uint8[i + 1, q] = base_code + (1 if actual_bit else 0)
                    else:
                        states_uint8[i + 1, q] = 0
            else:
                states_uint8[i + 1, qubit_targets(raw_targets)] = 0
        elif cat is Category.UNITARY and instr.arity == 1:
            q_targets = qubit_targets(raw_targets)
            fwd_lut, _ = _get_gate_luts(op.name)
            if len(set(q_targets.tolist())) != len(q_targets):
                for q in q_targets:
                    states_uint8[i + 1, q] = fwd_lut[states_uint8[i + 1, q]]
            else:
                states_uint8[i + 1, q_targets] = fwd_lut[
                    states_uint8[i + 1, q_targets]
                ]
        else:
            states_uint8[i + 1, qubit_targets(raw_targets)] = 0

    return states_uint8


def _backward_known_states(
    ir: CircuitIR,
    known_measurements: list[int | None],
    states_uint8: np.ndarray,
) -> None:
    shift = len(known_measurements) - ir.num_measurements
    for instr in reversed(ir.instrs):
        i = instr.unrolled_index
        op = instr.op
        cat = instr.category
        new_known_state = states_uint8[i + 1].copy()

        if cat in _SKIP_KNOWN_STATE_CATS:
            pass
        elif cat is Category.RESET:
            new_known_state[qubit_targets(op.targets_copy())] = 0
        elif cat in (Category.MEASURE, Category.MEASURE_RESET):
            raw_targets = op.targets_copy()
            if not instr.is_product_measurement:
                base_code = _BASIS_BASE_CODE.get(instr.basis, 5)
                meas_base = instr.first_measurement + shift
                for offset in range(len(raw_targets) - 1, -1, -1):
                    t = raw_targets[offset]
                    this_measurement = known_measurements[meas_base + offset]
                    q = t.qubit_value
                    if this_measurement is not None:
                        actual_bit = bool(this_measurement) ^ bool(
                            t.is_inverted_result_target
                        )
                        new_known_state[q] = base_code + (1 if actual_bit else 0)
                    else:
                        new_known_state[q] = 0
            else:
                new_known_state[qubit_targets(raw_targets)] = 0
        elif cat is Category.UNITARY and instr.arity == 1:
            q_targets = qubit_targets(op.targets_copy())
            _, bwd_lut = _get_gate_luts(op.name)
            if len(set(q_targets.tolist())) != len(q_targets):
                for q in q_targets[::-1]:
                    new_known_state[q] = bwd_lut[new_known_state[q]]
            else:
                new_known_state[q_targets] = bwd_lut[new_known_state[q_targets]]
        else:
            new_known_state[qubit_targets(op.targets_copy())] = 0

        already_known = states_uint8[i]
        conflict = (
            (new_known_state != 0)
            & (already_known != 0)
            & (already_known != new_known_state)
        )
        if np.any(conflict):
            raise ValueError(
                "Backprop known states disagreed with forward prop known states. "
                "This is on us, file a bug. "
                f"Across instruction {i}: {op} "
            )
        fill_mask = (already_known == 0) & (new_known_state != 0)
        states_uint8[i, fill_mask] = new_known_state[fill_mask]


def compute_known_states_uint8(
    circuit: stim.Circuit,
    unrolled_ops: Sequence[stim.CircuitInstruction] | None = None,
    *,
    ir: CircuitIR | None = None,
) -> np.ndarray:
    """Compute known states as a fast (num_ops + 1, num_qubits) uint8 array.

    Codes:
        0: _ (unknown)
        1: +X, 2: -X
        3: +Y, 4: -Y
        5: +Z, 6: -Z
    """
    if unrolled_ops is not None and ir is not None:
        raise ValueError("Cannot pass both unrolled_ops and ir.")
    known_measurements = compute_known_measurements(circuit)
    if ir is None:
        ir = build_ir(circuit) if unrolled_ops is None else build_ir(unrolled_ops)
    states_uint8 = _forward_known_states(ir, known_measurements, circuit.num_qubits)
    _backward_known_states(ir, known_measurements, states_uint8)
    return states_uint8


def compute_known_states(circuit: stim.Circuit) -> list[list[stim.PauliString]]:
    """Find circuit locations with known single qubit states."""
    states_uint8 = compute_known_states_uint8(circuit)
    return [
        [_CODE_TO_PAULI[int(code)].copy() for code in row] for row in states_uint8
    ]


def convert_paulis_to_arrays(
    known_states: list[stim.PauliString] | np.ndarray,
) -> tuple[
    Bool1DArray, Bool1DArray, Bool1DArray, Bool1DArray, Bool1DArray, Bool1DArray
]:
    """convert a list of known state PauliStrings into a dense metrix representation for fast
    lookups.

    Args:
        known_states: a list of length num_qubits
        of length-1 PauliStrings indicating the known qubit state.

    Returns:
        Returns a set of bool arrays of shape (num_qubits, )
        In order these are:
            in_pX: True if qubit is in +X
            in_mX: True if qubit is in -X
            in_pY: True if qubit is in +Y
            in_mY: True if qubit is in -Y
            in_pZ: True if qubit is in +Z
            in_mZ: True if qubit is in -Z
    """
    if isinstance(known_states, np.ndarray):
        return (
            known_states == 1,
            known_states == 2,
            known_states == 3,
            known_states == 4,
            known_states == 5,
            known_states == 6,
        )
    num_qubits = len(known_states)
    codes = np.fromiter(
        (_STR_TO_CODE.get(str(ps), 0) for ps in known_states),
        dtype=np.uint8,
        count=num_qubits,
    )
    return (
        codes == 1,
        codes == 2,
        codes == 3,
        codes == 4,
        codes == 5,
        codes == 6,
    )
