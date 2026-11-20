"""Noiseless synchronized reference walk for FlipsideSimulator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside.util.circuit_ir import Category, CircuitIR, IRInstr, build_ir
from stimside.util.stim_workarounds import split_fused_instruction

_SKIP_CATS = frozenset({Category.ANNOTATION, Category.NOISE, Category.OTHER})


def _record_op_ref_z(
    ref_ts: stim.TableauSimulator,
    op_idx: int,
    z_qs: tuple[list[int], list[int]],
    op_ref_z_map: dict[int, tuple[dict[int, bool | None], list[int]]],
) -> None:
    op_ref_z: dict[int, bool | None] = {}
    op_collapsed: list[int] = []
    for q in z_qs[0]:
        if q not in op_ref_z:
            if ref_ts.peek_z(q) == 0:
                ref_ts.postselect_z(q, desired_value=False)
                op_collapsed.append(q)
            op_ref_z[q] = ref_ts.peek_z(q) == -1
    for q in z_qs[1]:
        if q not in op_ref_z:
            pz = ref_ts.peek_z(q)
            op_ref_z[q] = None if pz == 0 else pz == -1
    op_ref_z_map[op_idx] = (op_ref_z, op_collapsed)


def _step_mpad(
    ref_ts: stim.TableauSimulator,
    instr: IRInstr,
    mpad_z_targets: Any,
    mpad_ref_z: dict[int, tuple[np.ndarray, np.ndarray]],
    ref_bits: list[bool],
) -> None:
    op = instr.op
    z_targets = mpad_z_targets(op) if mpad_z_targets is not None else None
    copies = split_fused_instruction(op, len(z_targets)) if z_targets else [op]
    off = 0
    for op_copy in copies:
        crs = [
            bool(gt.value) ^ bool(gt.is_inverted_result_target)
            for gt in op_copy.targets_copy()
        ]
        if z_targets is not None:
            # Collapse superposed targets to 0 (postselect: no reference RNG use) and
            # record each target's reference Z value and whether it was collapsed here.
            ref_z = np.zeros(len(z_targets), dtype=np.bool_)
            collapsed = np.zeros(len(z_targets), dtype=np.bool_)
            for k, q in enumerate(z_targets):
                if ref_ts.peek_z(q) == 0:
                    ref_ts.postselect_z(q, desired_value=False)
                    collapsed[k] = True
                ref_z[k] = ref_ts.peek_z(q) == -1
            mpad_ref_z[instr.first_measurement + off] = (ref_z, collapsed)
        ref_bits.extend(crs)
        off += len(crs)


def _step_product_meas(
    ref_ts: stim.TableauSimulator,
    num_qubits: int,
    op: stim.CircuitInstruction,
    basis: str | None,
    ref_bits: list[bool],
) -> None:
    for term in op.target_groups():
        inv_res = False
        term_paulis: list[tuple[int, str]] = []
        for gt in term:
            q_r = gt.qubit_value
            assert q_r is not None
            inv_res ^= bool(gt.is_inverted_result_target)
            p_str = gt.pauli_type if basis is None else basis
            term_paulis.append((q_r, p_str))
        if len({q_r for q_r, _ in term_paulis}) < len(term_paulis):
            ps = stim.PauliString(num_qubits)
            for q_r, p_str in term_paulis:
                single = stim.PauliString(num_qubits)
                single[q_r] = p_str
                ps *= single
            m_val = ref_ts.measure_observable(ps)
        else:
            for q_r, p_str in term_paulis:
                if p_str == "X":
                    ref_ts.h(q_r)
                elif p_str == "Y":
                    ref_ts.h_yz(q_r)
            q0 = term_paulis[0][0]
            for q_r, _ in term_paulis[1:]:
                ref_ts.cnot(q_r, q0)
            m_val = ref_ts.measure(q0)
            for q_r, _ in reversed(term_paulis[1:]):
                ref_ts.cnot(q_r, q0)
            for q_r, p_str in reversed(term_paulis):
                if p_str == "X":
                    ref_ts.h(q_r)
                elif p_str == "Y":
                    ref_ts.h_yz(q_r)
        ref_bits.append(bool(m_val) ^ inv_res)


def _step_controlled_pauli(
    ref_ts: stim.TableauSimulator,
    op: stim.CircuitInstruction,
    name: str,
    ref_bits: list[bool],
) -> None:
    targets = op.targets_copy()
    # Classical control (CX/CY/CZ/XCZ/YCZ): read ref_bits (stim's record),
    # not ref_ts's record (no MPAD/herald bits or inversions, has resets).
    for a, b in zip(targets[::2], targets[1::2]):
        if a.is_qubit_target and b.is_qubit_target:
            ref_ts.do(stim.CircuitInstruction(name, [a, b]))
        elif a.is_qubit_target:  # CZ/XCZ/YCZ q rec
            if b.is_measurement_record_target and ref_bits[b.value]:
                pauli = "Z" if name == "CZ" else name[0]
                ref_ts.do(stim.CircuitInstruction(pauli, [a]))
        elif a.is_measurement_record_target and ref_bits[a.value]:  # CX/CY/CZ rec q
            ref_ts.do(stim.CircuitInstruction(name[-1], [b]))
        # sweep bits are 0, as in stim's m2d reference sample


def _resolve_ir(
    unrolled_ops: Iterable[stim.CircuitInstruction] | stim.Circuit | None,
    ir: CircuitIR | None,
) -> CircuitIR:
    if unrolled_ops is not None and ir is not None:
        raise ValueError("Cannot pass both unrolled_ops and ir.")
    if ir is not None:
        return ir
    if unrolled_ops is None:
        raise ValueError("Either unrolled_ops or ir must be provided.")
    src = (
        unrolled_ops
        if isinstance(unrolled_ops, (stim.Circuit, Sequence))
        else list(unrolled_ops)
    )
    return build_ir(src)


def compute_sync_reference(
    num_qubits: int,
    unrolled_ops: Iterable[stim.CircuitInstruction] | stim.Circuit | None = None,
    compiled_op_handler: Any = None,
    *,
    ir: CircuitIR | None = None,
) -> tuple[
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[dict[int, bool | None], list[int]]],
]:
    """Compute FlipsideSimulator's reference sample and Z-collapse metadata."""
    ir = _resolve_ir(unrolled_ops, ir)
    mpad_ref_z: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    op_ref_z_map: dict[int, tuple[dict[int, bool | None], list[int]]] = {}
    ref_ts = stim.TableauSimulator(seed=0)
    ref_ts.set_num_qubits(num_qubits)
    ref_bits: list[bool] = []
    # Handler hook: qubits an MPAD Z-collapses (e.g. 0/1-keyed LEAKAGE_MEASUREMENT), or None.
    mpad_z_targets = getattr(compiled_op_handler, "mpad_z_collapse_targets", None)
    # Handler hook: (qubits a non-MPAD op reads the Z value of, qubits it may reset to |0>/|1>),
    # or None (e.g. 0/1-keyed LEAKAGE_TRANSITION_1/2). Read qubits are Z-collapsed like MPAD
    # targets; reset-only qubits are not (only shots that reset them collapse), so superposed
    # ones get None.
    op_z_targets = getattr(compiled_op_handler, "op_z_collapse_targets", None)

    for instr in ir.instrs:
        op = instr.op
        cat = instr.category
        if op_z_targets is not None:
            z_qs = op_z_targets(op)
            if z_qs is not None:
                _record_op_ref_z(ref_ts, instr.unrolled_index, z_qs, op_ref_z_map)
        if cat in _SKIP_CATS:
            continue
        name = op.name
        if cat is Category.MPAD:
            _step_mpad(ref_ts, instr, mpad_z_targets, mpad_ref_z, ref_bits)
        elif cat is Category.HERALD:
            ref_bits.extend([False] * instr.num_measurements)
        elif instr.is_product_measurement:
            _step_product_meas(ref_ts, num_qubits, op, instr.basis, ref_bits)
        elif cat in (Category.MEASURE, Category.MEASURE_RESET, Category.RESET):
            basis = instr.basis
            is_x, is_y = basis == "X", basis == "Y"
            is_rst = cat is not Category.MEASURE
            is_meas = cat is not Category.RESET
            for gt in op.targets_copy():
                q = gt.qubit_value
                assert q is not None
                if is_x:
                    ref_ts.h(q)
                elif is_y:
                    ref_ts.h_yz(q)
                m_val = ref_ts.measure(q)
                if is_rst:
                    ref_ts.reset_z(q)
                if is_x:
                    ref_ts.h(q)
                elif is_y:
                    ref_ts.h_yz(q)
                if is_meas:
                    ref_bits.append(bool(m_val) ^ bool(gt.is_inverted_result_target))
        elif cat is Category.UNITARY and name not in ("I", "II"):
            if instr.has_classical_control:
                _step_controlled_pauli(ref_ts, op, name, ref_bits)
                continue
            ref_ts.do(op)
    ref_sample = np.asarray(ref_bits, dtype=np.bool_)
    return ref_sample, mpad_ref_z, op_ref_z_map
