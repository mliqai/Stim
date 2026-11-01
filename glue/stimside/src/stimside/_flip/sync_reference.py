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
            crs = np.zeros(len(op.targets_copy()), dtype=np.bool_)
            ref_chunks.append(crs)
        elif name in (
            "M",
            "MZ",
            "MR",
            "MRZ",
            "MX",
            "MRX",
            "MY",
            "MRY",
            "R",
            "RZ",
            "RX",
            "RY",
        ):
            is_x = name in ("MX", "MRX", "RX")
            is_y = name in ("MY", "MRY", "RY")
            is_r = name in (
                "MR",
                "MRZ",
                "MRX",
                "MRY",
                "R",
                "RZ",
                "RX",
                "RY",
            )
            produces_m = name in (
                "M",
                "MZ",
                "MR",
                "MRZ",
                "MX",
                "MRX",
                "MY",
                "MRY",
            )
            crs_list: list[bool] = []
            for gt in op.targets_copy():
                q = gt.qubit_value if gt.qubit_value is not None else gt.value
                if is_x:
                    ref_ts.h(q)
                elif is_y:
                    ref_ts.h_yz(q)
                m_val = ref_ts.measure(q)
                if is_r:
                    ref_ts.reset_z(q)
                if is_x:
                    ref_ts.h(q)
                elif is_y:
                    ref_ts.h_yz(q)
                crs_list.append(bool(m_val) ^ bool(gt.is_inverted_result_target))
            if produces_m:
                ref_chunks.append(np.array(crs_list, dtype=np.bool_))
        elif name in ("MXX", "MYY", "MZZ", "MPP"):
            raw_targets = op.targets_copy()
            terms: list[list[stim.GateTarget]] = []
            if name == "MPP":
                t_i = 0
                while t_i < len(raw_targets):
                    term = [raw_targets[t_i]]
                    t_i += 1
                    while t_i < len(raw_targets) and raw_targets[t_i].is_combiner:
                        term.append(raw_targets[t_i + 1])
                        t_i += 2
                    terms.append(term)
            else:
                for t_i in range(0, len(raw_targets), 2):
                    terms.append([raw_targets[t_i], raw_targets[t_i + 1]])

            crs_list = []
            for term in terms:
                inv_res = False
                pre_ops: list[stim.CircuitInstruction] = []
                post_ops: list[stim.CircuitInstruction] = []
                term_paulis: list[tuple[int, str]] = []
                if name == "MPP":
                    for gt in term:
                        q_r = gt.qubit_value
                        if gt.is_inverted_result_target:
                            inv_res = not inv_res
                        if gt.is_x_target:
                            p_str = "X"
                            h_op = stim.CircuitInstruction("H", [q_r])
                            pre_ops.append(h_op)
                            post_ops.insert(0, h_op)
                        elif gt.is_y_target:
                            p_str = "Y"
                            hyz_op = stim.CircuitInstruction("H_YZ", [q_r])
                            pre_ops.append(hyz_op)
                            post_ops.insert(0, hyz_op)
                        else:
                            p_str = "Z"
                        term_paulis.append((q_r, p_str))
                else:
                    p_str = name[-1]
                    q0, q1 = term[0].qubit_value, term[1].qubit_value
                    inv_res = bool(
                        term[0].is_inverted_result_target
                        ^ term[1].is_inverted_result_target
                    )
                    term_paulis = [(q0, p_str), (q1, p_str)]
                    if p_str == "X":
                        h_op = stim.CircuitInstruction("H", [q0, q1])
                        pre_ops.append(h_op)
                        post_ops.append(h_op)
                    elif p_str == "Y":
                        hyz_op = stim.CircuitInstruction("H_YZ", [q0, q1])
                        pre_ops.append(hyz_op)
                        post_ops.append(hyz_op)

                q0 = term_paulis[0][0]
                cx_ops: list[stim.CircuitInstruction] = []
                for q_r, _ in term_paulis[1:]:
                    cx_ops.append(stim.CircuitInstruction("CX", [q_r, q0]))
                pre_ops.extend(cx_ops)
                post_ops = list(reversed(cx_ops)) + post_ops

                for p_op in pre_ops:
                    ref_ts.do(p_op)
                m_val = ref_ts.measure(q0)
                for p_op in post_ops:
                    ref_ts.do(p_op)
                crs_list.append(bool(m_val) ^ inv_res)

            ref_chunks.append(np.array(crs_list, dtype=np.bool_))
        else:
            gd = stim.gate_data(name)
            if gd.is_unitary or gd.is_reset:
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
