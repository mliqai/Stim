"""Noiseless synchronized reference walk for FlipsideSimulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside.util.stim_workarounds import split_fused_instruction


def compute_sync_reference(
    num_qubits: int,
    unrolled_ops: list[stim.CircuitInstruction],
    compiled_op_handler: Any,
) -> tuple[
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[dict[int, bool | None], list[int]]],
]:
    """Compute FlipsideSimulator's reference sample and Z-collapse metadata."""
    mpad_ref_z: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    op_ref_z_map: dict[int, tuple[dict[int, bool | None], list[int]]] = {}
    ref_ts = stim.TableauSimulator(seed=0)
    ref_ts.set_num_qubits(num_qubits)
    ref_chunks: list[np.ndarray] = []
    # Handler hook: qubits an MPAD Z-collapses (e.g. 0/1-keyed LEAKAGE_MEASUREMENT), or None.
    mpad_z_targets = getattr(compiled_op_handler, "mpad_z_collapse_targets", None)
    # Handler hook: (qubits a non-MPAD op reads the Z value of, qubits it may reset to |0>/|1>), or
    # None (e.g. 0/1-keyed LEAKAGE_TRANSITION_1/2). Read qubits are Z-collapsed like MPAD targets;
    # reset-only qubits are not (only shots that reset them collapse), so superposed ones get None.
    op_z_targets = getattr(compiled_op_handler, "op_z_collapse_targets", None)
    num_meas, num_chunks_counted = 0, 0
    for op_idx, op in enumerate(unrolled_ops):
        name = op.name
        z_qs = op_z_targets(op) if op_z_targets is not None else None
        if z_qs is not None:
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
        if name == "MPAD":
            z_targets = mpad_z_targets(op) if mpad_z_targets is not None else None
            copies = split_fused_instruction(op, len(z_targets)) if z_targets else [op]
            for op_copy in copies:
                crs = np.array(
                    [
                        bool(gt.value) ^ bool(gt.is_inverted_result_target)
                        for gt in op_copy.targets_copy()
                    ],
                    dtype=np.bool_,
                )
                if z_targets is not None:
                    num_meas += sum(len(c) for c in ref_chunks[num_chunks_counted:])
                    num_chunks_counted = len(ref_chunks)
                    # Collapse superposed targets to 0 (postselect: no reference RNG use) and record
                    # each target's reference Z value and whether it was collapsed here.
                    ref_z = np.zeros(len(z_targets), dtype=np.bool_)
                    collapsed = np.zeros(len(z_targets), dtype=np.bool_)
                    for k, q in enumerate(z_targets):
                        if ref_ts.peek_z(q) == 0:
                            ref_ts.postselect_z(q, desired_value=False)
                            collapsed[k] = True
                        ref_z[k] = ref_ts.peek_z(q) == -1
                    mpad_ref_z[num_meas] = (ref_z, collapsed)
                ref_chunks.append(crs)
        elif name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            ref_chunks.append(np.zeros(len(op.targets_copy()), dtype=np.bool_))
        elif name in ("MXX", "MYY", "MZZ", "MPP"):
            crs_list: list[bool] = []
            for term in op.target_groups():
                inv_res = False
                term_paulis: list[tuple[int, str]] = []
                for gt in term:
                    q_r = gt.qubit_value
                    inv_res ^= bool(gt.is_inverted_result_target)
                    p_str = gt.pauli_type if name == "MPP" else name[-1]
                    term_paulis.append((q_r, p_str))
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
                crs_list.append(bool(m_val) ^ inv_res)
            ref_chunks.append(np.array(crs_list, dtype=np.bool_))
        else:
            gd = stim.gate_data(name)
            if gd.is_single_qubit_gate and (gd.produces_measurements or gd.is_reset):
                is_x, is_y = "X" in name, "Y" in name
                crs_list = []
                for gt in op.targets_copy():
                    q = gt.qubit_value if gt.qubit_value is not None else gt.value
                    if is_x:
                        ref_ts.h(q)
                    elif is_y:
                        ref_ts.h_yz(q)
                    m_val = ref_ts.measure(q)
                    if gd.is_reset:
                        ref_ts.reset_z(q)
                    if is_x:
                        ref_ts.h(q)
                    elif is_y:
                        ref_ts.h_yz(q)
                    crs_list.append(bool(m_val) ^ bool(gt.is_inverted_result_target))
                if gd.produces_measurements:
                    ref_chunks.append(np.array(crs_list, dtype=np.bool_))
            elif gd.is_unitary or gd.is_reset:
                if name not in ("I", "II", "I_ERROR", "II_ERROR"):
                    if name in ("CX", "CY", "CZ", "XCZ", "YCZ") and (
                        "rec[" in str(op) or "sweep[" in str(op)
                    ):
                        targets = op.targets_copy()
                        # Classical control (CX/CY/CZ/XCZ/YCZ): read ref_chunks (stim's record),
                        # not ref_ts's record (no MPAD/herald bits or inversions, has resets).
                        rec = np.concatenate(ref_chunks) if ref_chunks else np.zeros(0, bool)
                        for a, b in zip(targets[::2], targets[1::2]):
                            if a.is_qubit_target and b.is_qubit_target:
                                ref_ts.do(stim.CircuitInstruction(name, [a, b]))
                            elif a.is_qubit_target:  # CZ/XCZ/YCZ q rec
                                if b.is_measurement_record_target and rec[b.value]:
                                    pauli = "Z" if name == "CZ" else name[0]
                                    ref_ts.do(stim.CircuitInstruction(pauli, [a]))
                            elif a.is_measurement_record_target and rec[a.value]:  # CX/CY/CZ rec q
                                ref_ts.do(stim.CircuitInstruction(name[-1], [b]))
                            # sweep bits are 0, as in stim's m2d reference sample
                    else:
                        ref_ts.do(op)
    ref_sample = np.concatenate(ref_chunks) if ref_chunks else np.empty(0, dtype=np.bool_)
    return ref_sample, mpad_ref_z, op_ref_z_map
