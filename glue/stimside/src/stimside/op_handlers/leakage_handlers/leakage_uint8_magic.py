"""Batched uint8 leakage OpHandler for MagicsideSimulator."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
import warnings

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageParams,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import CompiledLeakageUint8Coset, LeakageUint8Coset
from stimside.op_handlers.leakage_handlers.tag_registry import (
    MAGIC_TAGS as _MAGIC_TAGS,
    check_tag_qubits,
    parse_leakage_tag_magic as _parse_leakage_tag_magic,
    validate_magic_op,
)
from stimside.op_handlers.leakage_handlers.transition_effects import (
    check_late_frozen_read,
    meas_basis,
    qualify_fast_1q,
)
from stimside.util.stim_workarounds import split_fused_instruction

if TYPE_CHECKING:
    from stimside.simulator_magic import MagicsideSimulator

_PURE_NOISE_GATES: frozenset[str] = frozenset({
    "X_ERROR", "Y_ERROR", "Z_ERROR", "I_ERROR", "II_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR",
})


def _strip_circuit_for_noiseless_ref(circuit: stim.Circuit) -> tuple[stim.Circuit, bool]:
    needs_leakage = False

    def _strip(blk: stim.Circuit) -> stim.Circuit:
        nonlocal needs_leakage
        out = stim.Circuit()
        for op in blk:
            if isinstance(op, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(op.repeat_count, _strip(op.body_copy())))
                continue
            tag, name, tgs = op.tag, op.name, op.targets_copy()
            if tag.startswith(("CLIFFORD_FAULT", "LEAKAGE_TRANSITION", "LEAKAGE_CONTROLLED")):
                continue
            if tag == "LEAKAGE_SWAP":
                out.append(stim.CircuitInstruction("SWAP", tgs))
            elif tag.startswith(("LEAKAGE_MEASUREMENT", "LEAKAGE_PROJECTION_Z")):
                params, _ = _parse_leakage_tag_magic(op)
                assert isinstance(params, LeakageMeasurementParams)
                is_mpad = tag.startswith("LEAKAGE_MEASUREMENT")
                b0 = params.p0 > 0.5
                b1 = params.p1 > 0.5
                if is_mpad and b0 == b1:
                    out.append(stim.CircuitInstruction("MPAD", [int((bool(t.value) ^ t.is_inverted_result_target) ^ b0) for t in tgs]))
                elif not is_mpad and (b0, b1) == (False, True):
                    out.append(stim.CircuitInstruction(name, tgs))
                elif not is_mpad and (b0, b1) == (True, False):
                    out.append(stim.CircuitInstruction(name, [stim.target_inv(t.qubit_value) if not t.is_inverted_result_target else t.qubit_value for t in tgs]))
                else:
                    sfx = f" : {' '.join(str(q) for q in (params.targets or ()))}" if is_mpad else ""
                    out.append(stim.CircuitInstruction(name, tgs, [], tag=f"{'LEAKAGE_MEASUREMENT' if is_mpad else 'LEAKAGE_PROJECTION_Z'}: ({float(b0)}, 0) ({float(b1)}, 1){sfx}"))
                    needs_leakage = True
            elif tag.startswith("CONDITIONED_ON"):
                if name in _PURE_NOISE_GATES:
                    continue
                params, _ = _parse_leakage_tag_magic(op)
                assert isinstance(params, LeakageConditioningParams)
                has_comp_proj = any(0 in g or 1 in g for g in params.args)
                if has_comp_proj:
                    out.append(op)
                    needs_leakage = True
                elif ("U",) * len(params.args) in zip(*params.args):
                    out.append(stim.CircuitInstruction(name, tgs, op.gate_args_copy()))
            elif tag == "" and name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
                out.append(stim.CircuitInstruction("MPAD", [0] * len(tgs)))
            elif tag in _MAGIC_TAGS:
                out.append(op)
            elif name not in _PURE_NOISE_GATES:
                gd = stim.gate_data(name)
                out.append(stim.CircuitInstruction(name, tgs, [], tag=tag) if (gd.produces_measurements and op.gate_args_copy()) else op)
        return out

    return _strip(circuit), needs_leakage


def _parse_leakage_in_circuit_magic(
    circuit: stim.Circuit, num_qubits: int
) -> tuple[dict[stim.CircuitInstruction, LeakageParams], set[stim.CircuitInstruction], dict[stim.CircuitInstruction, list[int]]]:
    ops_to_params: dict[stim.CircuitInstruction, LeakageParams] = {}
    u_meas_ops: set[stim.CircuitInstruction] = set()
    magic_ops: dict[stim.CircuitInstruction, list[int]] = {}

    def _walk(blk: stim.Circuit) -> None:
        for item in blk:
            if isinstance(item, stim.CircuitRepeatBlock):
                _walk(item.body_copy())
            elif item.tag in _MAGIC_TAGS:
                if item not in magic_ops:
                    magic_ops[item] = validate_magic_op(item, num_qubits)
            elif item.tag != "":
                parsed, is_u = _parse_leakage_tag_magic(item)
                if parsed is None:
                    continue
                ops_to_params[item] = parsed
                if is_u:
                    u_meas_ops.add(item)

    _walk(circuit)
    check_tag_qubits(ops_to_params, num_qubits)
    return ops_to_params, u_meas_ops, magic_ops


@dataclasses.dataclass
class LeakageUint8Magic(LeakageUint8Coset):
    """Batched uint8 leakage OpHandler for MagicsideSimulator (inherits ``fast_leakage``)."""

    def compile_op_handler(self, *, circuit: stim.Circuit, batch_size: int) -> CompiledLeakageUint8Magic:
        ops_to_params, u_meas_ops, magic_ops = _parse_leakage_in_circuit_magic(circuit, circuit.num_qubits)
        fast_1q, compiled_len = qualify_fast_1q(
            circuit, ops_to_params, self.unconditional_condition_on_U, self.fast_leakage
        )
        compiled = CompiledLeakageUint8Magic(
            num_qubits=circuit.num_qubits, batch_size=batch_size,
            unconditional_condition_on_U=self.unconditional_condition_on_U,
            ops_to_params=ops_to_params, fast_leakage=self.fast_leakage,
            u_measurement_ops=u_meas_ops, magic_ops=magic_ops,
        )
        compiled._set_fast_1q(fast_1q, compiled_len)
        return compiled


@dataclasses.dataclass
class CompiledLeakageUint8Magic(CompiledLeakageUint8Coset):
    """Compiled batched uint8 leakage OpHandler for MagicsideSimulator."""

    u_measurement_ops: set[stim.CircuitInstruction] = dataclasses.field(default_factory=set)
    magic_ops: dict[stim.CircuitInstruction, list[int]] = dataclasses.field(default_factory=dict)

    def handle_op(self, op: stim.CircuitInstruction, sss: MagicsideSimulator) -> None:  # type: ignore[override]
        self._ensure_batch_size(sss.batch_size)
        if op.tag in _MAGIC_TAGS:
            check_late_frozen_read(self._fast_1q, self._compiled_len, op, sss, self.ops_to_params)
            qs = self.magic_ops.get(op)
            if qs is None:
                qs = validate_magic_op(op, sss.num_qubits)
                self.magic_ops[op] = qs
            if op.tag == "CCZ" and self.unconditional_condition_on_U:
                # Like an untagged Clifford gate: a triple is skipped in the shots where any of
                # its three qubits is leaked.
                if np.any(self._state[qs, :] >= 2):
                    triples = [qs[i : i + 3] for i in range(0, len(qs), 3)]
                    acts = np.array(
                        [np.all(self._state[t, :] < 2, axis=0) for t in triples], np.bool_
                    )
                    sss.do_conditional_magic(op, qs, acts)
                else:
                    sss.emit_magic_op(op, qs)
                return
            for grp in ([qs[i : i + 3] for i in range(0, len(qs), 3)] if op.tag == "CCZ" else [[q] for q in qs]):
                if np.any(self._state[grp, :] >= 2):
                    warnings.warn(f"Magic gate I[{op.tag}] applied to leaked group {tuple(grp)}", UserWarning, stacklevel=2)
            sss.emit_magic_op(op, qs)
            return
        if op.tag != "" and op not in self.ops_to_params and not op.tag.startswith("CLIFFORD_FAULT"):
            parsed, is_u = _parse_leakage_tag_magic(op)
            if parsed is not None:
                check_tag_qubits({op: parsed}, sss.num_qubits)
                self.ops_to_params[op] = parsed
                self.claimed_ops_keys.add(op)
                if is_u:
                    self.u_measurement_ops.add(op)
        super().handle_op(op, sss)

    def leakage_measurement(self, op: stim.CircuitInstruction, tss: MagicsideSimulator, params: LeakageMeasurementParams) -> None:  # type: ignore[override]
        if op.name == "MPAD" and op in self.u_measurement_ops:
            for s in split_fused_instruction(op, len(params.targets or ())):
                self._eval_u_mpad(s, tss, params)
            return
        raw_t, is_mpad = op.targets_copy(), params.targets is not None
        p0, p1 = params.p0, params.p1
        need_slow = (0 in params.prob_for_input_state or 1 in params.prob_for_input_state) if is_mpad else (p0 != 0.0 or p1 != 1.0)
        if (sync := tss.sync_rngs()) is not None and need_slow:
            qs = list(dict.fromkeys(params.targets if is_mpad else (int(t.qubit_value) for t in raw_t if t.is_qubit_target)))  # type: ignore[arg-type]
            valid_op = (op.name == "MPAD" and len(raw_t) == len(params.targets or ())) if is_mpad else (op.name in ("M", "MZ", "MR", "MRZ", "MX", "MY", "MRX", "MRY") and len(qs) == len(raw_t))
            if valid_op:
                self._ensure_batch_size(tss.batch_size)
                basis = meas_basis(op.name)
                for q_i in qs:
                    pb_row = tss.peek_pauli_born_batch([q_i], pauli=basis)[0]  # type: ignore[arg-type]
                    sup = (self._state[q_i] < 2) & (pb_row > 1e-12) & (pb_row < 1.0 - 1e-12)
                    if np.any(sup):
                        des = np.zeros(tss.batch_size, dtype=np.uint8)
                        for b in np.flatnonzero(sup):
                            des[b] = sync.shot_np_rngs[b].binomial(1, 0.5 if abs(pb_row[b] - 0.5) <= 1e-12 else min(1.0, max(0.0, float(pb_row[b]))))
                        tss.postselect_z_batch(q_i, sup, des, pauli=basis)  # type: ignore[arg-type]
        super().leakage_measurement(op, tss, params)

    def _eval_u_mpad(self, op: stim.CircuitInstruction, tss: MagicsideSimulator, params: LeakageMeasurementParams) -> None:
        raw_t, t_qubits = op.targets_copy(), list(params.targets or ())
        if len(raw_t) != len(t_qubits):
            raise ValueError(f"{op} has {len(raw_t)} pad targets, expected multiple of {len(t_qubits)}")
        states, p_u = self._state[t_qubits], params.p0
        probs = np.where(states < 2, p_u, 0.0).astype(np.float64)
        for st, p_val in params.prob_for_input_state.items():
            if st >= 2 and p_val > 0.0:
                probs[states == st] = p_val
        outcomes = np.zeros((len(t_qubits), self.batch_size), dtype=np.bool_)
        if tss.sync_rngs() is not None:
            for b in range(self.batch_size):
                st_b = states[:, b]
                for mask_g, pr in [(st_b < 2, p_u)] + [(st_b == n, params.prob_for_input_state[n]) for n in range(2, 10) if n in params.prob_for_input_state]:
                    idx_g = np.flatnonzero(mask_g)
                    if len(idx_g) and pr > 0.0:
                        outcomes[idx_g, b] = self._tab_flips(tss, b, [t_qubits[k] for k in idx_g], pr)
        elif np.any(probs > 0.0):
            pos = probs > 0.0
            outcomes[pos] = tss.np_rng.random(int(np.count_nonzero(pos))) < probs[pos]
        outcomes ^= np.array([bool(t.value) ^ t.is_inverted_result_target for t in raw_t], dtype=np.bool_)[:, None]
        tss.record_direct_measurements(outcomes)
