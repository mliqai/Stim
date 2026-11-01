"""Batched uint8 leakage OpHandler for MagicsideSimulator."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageParams,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import CompiledLeakageUint8Coset, LeakageUint8Coset, _do_copies_on_fused_step
from stimside.op_handlers.leakage_handlers.tag_registry import (
    MAGIC_TAGS as _MAGIC_TAGS,
    check_tag_qubits,
    parse_leakage_tag_magic as _parse_leakage_tag_magic,
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
                b0 = params.prob_for_input_state.get(0, 0.0) > 0.5
                b1 = params.prob_for_input_state.get(1, 0.0 if is_mpad else 1.0) > 0.5
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
                has_comp_proj = any(0 in g or 1 in g for g in params.args) or (
                    len(params.args) == 1
                    and any(tok in ("0", "1") for tok in (params.from_tag or ":").split(":")[1].split())
                )
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


def _validate_magic_instruction(op: stim.CircuitInstruction, num_qubits: int) -> list[int]:
    if op.name != "I" or op.gate_args_copy():
        raise ValueError(f"Magic gate I[{op.tag}] must be gate 'I' with no gate_args, got {op!r}")
    qs = [int(t.qubit_value) if (t.is_qubit_target and not t.is_inverted_result_target and 0 <= int(t.qubit_value) < num_qubits) else -1 for t in op.targets_copy()]
    if any(q < 0 for q in qs):
        raise ValueError(f"Invalid target in magic gate I[{op.tag}] for num_qubits={num_qubits}: {op!r}")
    if op.tag == "CCZ" and (len(qs) % 3 != 0 or any(len(set(qs[i : i + 3])) != 3 for i in range(0, len(qs), 3))):
        raise ValueError(f"I[CCZ] requires distinct triples of targets, got {qs}")
    return qs


def _parse_leakage_in_circuit_magic(circuit: stim.Circuit, num_qubits: int) -> tuple[dict[stim.CircuitInstruction, LeakageParams], set[stim.CircuitInstruction]]:
    ops_to_params: dict[stim.CircuitInstruction, LeakageParams] = {}
    u_meas_ops: set[stim.CircuitInstruction] = set()

    def _walk(blk: stim.Circuit) -> None:
        for item in blk:
            if isinstance(item, stim.CircuitRepeatBlock):
                _walk(item.body_copy())
            elif item.tag in _MAGIC_TAGS:
                _validate_magic_instruction(item, num_qubits)
            elif item.tag != "":
                parsed, is_u = _parse_leakage_tag_magic(item)
                if parsed is None:
                    continue
                ops_to_params[item] = parsed
                if is_u:
                    u_meas_ops.add(item)
                tag_t, raw_t = getattr(parsed, "targets", None), item.targets_copy()
                if tag_t and item.tag.startswith(("LEAKAGE_MEASUREMENT", "CONDITIONED_ON_OTHER")) and (not raw_t or len(raw_t) % len(tag_t) != 0):
                    raise ValueError(f"{item} target count not a multiple of {len(tag_t)}")

    _walk(circuit)
    check_tag_qubits(ops_to_params, num_qubits)
    return ops_to_params, u_meas_ops


@dataclasses.dataclass
class LeakageUint8Magic(LeakageUint8Coset):
    """Batched uint8 leakage OpHandler for MagicsideSimulator."""

    def compile_op_handler(self, *, circuit: stim.Circuit, batch_size: int) -> CompiledLeakageUint8Magic:
        ops_to_params, u_meas_ops = _parse_leakage_in_circuit_magic(circuit, circuit.num_qubits)
        return CompiledLeakageUint8Magic(num_qubits=circuit.num_qubits, batch_size=batch_size, unconditional_condition_on_U=self.unconditional_condition_on_U, ops_to_params=ops_to_params, u_measurement_ops=u_meas_ops)


@dataclasses.dataclass
class CompiledLeakageUint8Magic(CompiledLeakageUint8Coset):
    """Compiled batched uint8 leakage OpHandler for MagicsideSimulator."""

    u_measurement_ops: set[stim.CircuitInstruction] = dataclasses.field(default_factory=set)

    def handle_op(self, op: stim.CircuitInstruction, sss: MagicsideSimulator) -> None:  # type: ignore[override]
        self._ensure_batch_size(sss.batch_size)
        if op.tag in _MAGIC_TAGS:
            qs = _validate_magic_instruction(op, sss.num_qubits)
            for grp in ([qs[i : i + 3] for i in range(0, len(qs), 3)] if op.tag == "CCZ" else [[q] for q in qs]):
                if np.any(self._state[grp, :] >= 2):
                    warnings.warn(f"Magic gate I[{op.tag}] applied to leaked group {tuple(grp)}", UserWarning, stacklevel=2)
            sss._do_bare_instruction(op)
            return
        if op.tag != "" and op not in self.ops_to_params and not op.tag.startswith("CLIFFORD_FAULT"):
            parsed, is_u = _parse_leakage_tag_magic(op)
            if parsed is not None:
                check_tag_qubits({op: parsed}, sss.num_qubits)
                self.ops_to_params[op] = parsed
                self.claimed_ops_keys.add(op)
                if is_u:
                    self.u_measurement_ops.add(op)
        super().handle_op(op, sss)  # type: ignore[arg-type]

    def _update_state_from_peek_Z(self, targets: Sequence[int] | NDArray[np.int_], css: MagicsideSimulator) -> None:  # type: ignore[override]
        self._ensure_batch_size(css.batch_size)
        unique_t = list(dict.fromkeys(int(t) for t in targets if int(t) >= 0))
        if not unique_t:
            return
        p1_mat, any_projected = css.peek_pauli_born_batch(unique_t, pauli="Z"), False
        for idx, q in enumerate(unique_t):
            comp_mask = self._state[q] < 2
            if not np.any(comp_mask):
                continue
            p1_row = css.peek_pauli_born_batch([q], pauli="Z")[0] if any_projected else p1_mat[idx]
            new_row, det_one = np.zeros(css.batch_size, dtype=np.uint8), p1_row >= 1.0 - 1e-12
            new_row[det_one] = 1
            flip_mask = comp_mask & (p1_row > 1e-12) & ~det_one
            num_flips = int(np.count_nonzero(flip_mask))
            if num_flips > 0:
                if css._shot_np_rngs is not None:
                    new_row[flip_mask] = np.asarray([css._shot_np_rngs[b].binomial(1, 0.5 if abs(p1_row[b] - 0.5) <= 1e-12 else float(p1_row[b])) for b in np.flatnonzero(flip_mask)], dtype=np.uint8)
                else:
                    p_sub = p1_row[flip_mask]
                    new_row[flip_mask] = css.np_rng.binomial(1, 0.5 if np.allclose(p_sub, 0.5, atol=1e-12) else p_sub, size=num_flips).astype(np.uint8)
                css.postselect_z_batch(q, flip_mask, new_row)
                any_projected = True
            self._state[q, comp_mask] = new_row[comp_mask]

    def leakage_measurement(self, op: stim.CircuitInstruction, tss: MagicsideSimulator, params: LeakageMeasurementParams) -> None:  # type: ignore[override]
        if op.name == "MPAD" and op in self.u_measurement_ops:
            copies = split_fused_instruction(op, len(params.targets or ()))
            if len(copies) > 1:
                _do_copies_on_fused_step(tss, op, copies, lambda s: self._eval_u_mpad(s, tss, params))  # type: ignore[arg-type]
            else:
                self._eval_u_mpad(op, tss, params)
            return
        raw_t, is_mpad = op.targets_copy(), params.targets is not None
        p0, p1 = params.prob_for_input_state.get(0, 0.0), params.prob_for_input_state.get(1, 0.0 if is_mpad else 1.0)
        need_slow = (0 in params.prob_for_input_state or 1 in params.prob_for_input_state) if is_mpad else (p0 != 0.0 or p1 != 1.0)
        if tss._tab_rngs is not None and need_slow:
            qs = list(dict.fromkeys(params.targets if is_mpad else (int(t.qubit_value) for t in raw_t if t.is_qubit_target)))  # type: ignore[arg-type]
            valid_op = (op.name == "MPAD" and len(raw_t) == len(params.targets or ())) if is_mpad else (op.name in ("M", "MZ", "MR", "MRZ", "MX", "MY", "MRX", "MRY") and len(qs) == len(raw_t))
            if valid_op:
                self._ensure_batch_size(tss.batch_size)
                basis = "X" if op.name in ("MX", "MRX") else ("Y" if op.name in ("MY", "MRY") else "Z")
                for q_i in qs:
                    pb_row = tss.peek_pauli_born_batch([q_i], pauli=basis)[0]  # type: ignore[arg-type]
                    sup = (self._state[q_i] < 2) & (pb_row > 1e-12) & (pb_row < 1.0 - 1e-12)
                    if np.any(sup):
                        des = np.zeros(tss.batch_size, dtype=np.uint8)
                        for b in np.flatnonzero(sup):
                            des[b] = tss._shot_np_rngs[b].binomial(1, 0.5 if abs(pb_row[b] - 0.5) <= 1e-12 else min(1.0, max(0.0, float(pb_row[b]))))  # type: ignore[index]
                        tss.postselect_z_batch(q_i, sup, des, pauli=basis)  # type: ignore[arg-type]
        super().leakage_measurement(op, tss, params)  # type: ignore[arg-type]

    def _eval_u_mpad(self, op: stim.CircuitInstruction, tss: MagicsideSimulator, params: LeakageMeasurementParams) -> None:
        raw_t, t_qubits = op.targets_copy(), list(params.targets or ())
        if len(raw_t) != len(t_qubits):
            raise ValueError(f"{op} has {len(raw_t)} pad targets, expected multiple of {len(t_qubits)}")
        tss._next_step_meta(op)
        states, p_u = self._state[t_qubits], params.prob_for_input_state.get(0, 0.0)
        probs = np.where(states < 2, p_u, 0.0).astype(np.float64)
        for st, p_val in params.prob_for_input_state.items():
            if st >= 2 and p_val > 0.0:
                probs[states == st] = p_val
        outcomes = np.zeros((len(t_qubits), self.batch_size), dtype=np.bool_)
        if tss._tab_rngs is not None:
            for b in range(self.batch_size):
                tr, tr_x, st_b = tss._tab_rngs[b], tss._tab_rngs_x[b], states[:, b]  # type: ignore[index]
                for mask_g, pr in [(st_b < 2, p_u)] + [(st_b == n, params.prob_for_input_state[n]) for n in range(2, 10) if n in params.prob_for_input_state]:
                    q_list = [int(t_qubits[k]) for k in np.flatnonzero(mask_g)]
                    if q_list and pr > 0.0:
                        inst_g = stim.CircuitInstruction("X_ERROR", q_list, [pr])
                        tr.do(inst_g)
                        tr_x.do(inst_g)
                        for k_i, q_g in zip(np.flatnonzero(mask_g), q_list):
                            if tr.peek_z(q_g) == -1:
                                tr.x(q_g)
                                outcomes[int(k_i), b] = True
        elif np.any(probs > 0.0):
            pos = probs > 0.0
            outcomes[pos] = tss.np_rng.random(int(np.count_nonzero(pos))) < probs[pos]
        outcomes ^= np.array([bool(t.value) ^ t.is_inverted_result_target for t in raw_t], dtype=np.bool_)[:, None]
        tss.record_direct_measurements(outcomes)
