from __future__ import annotations

import dataclasses
from typing import Any, Literal, NamedTuple, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
)
from stimside.op_handlers.leakage_handlers.leakage_tag_parsing_tableau import (
    parse_leakage_in_circuit,
)
from stimside.op_handlers.leakage_handlers.tag_registry import check_tag_qubits
from stimside.op_handlers.leakage_handlers.transition_effects import (
    TransitionEffectMasks,
    apply_output_effect,
    check_late_frozen_read,
    dispatch_tagged_leakage_op,
    eval_allowed_state_mask,
    exec_leakage_swap,
    meas_basis,
    qualify_fast_1q,
    sample_batched_group_transitions,
    unpack_leakage_measurement_targets,
)
from stimside.op_handlers.seam import BatchedLeakageSimulator
from stimside.util.stim_workarounds import split_fused_instruction

_REC_BARE, _REC_PLAIN_UNIT, _REC_PLAIN_NOISE, _REC_MASK_UNIT_U = 0, 1, 2, 3
_REC_MASK_NOISE_U, _REC_MASK_CORR_U, _REC_MASK_SPP_U, _REC_TAGGED = 4, 5, 6, 7
_NOISE_CH: dict[str, tuple[int, int]] = {
    "X_ERROR": (0, 1), "Y_ERROR": (1, 1), "Z_ERROR": (2, 1),
    "DEPOLARIZE1": (3, 1), "PAULI_CHANNEL_1": (3, 1),
    "DEPOLARIZE2": (4, 2), "PAULI_CHANNEL_2": (5, 2),
}
_FAST_UNIT_SKIP = frozenset({"I", "II", "I_ERROR", "II_ERROR", "X", "Y", "Z", "SPP", "SPP_DAG"})


class _OpRec(NamedTuple):
    kind: int
    arity: int
    q_check: NDArray[np.intp] | None
    qs_arr: NDArray[np.int64] | None
    spp_terms: tuple[list[int], ...] | None
    noise_spec: tuple[int, int, NDArray[np.float64], bool] | None
    fast_unitary_ok: bool
    claimed: bool
    params: LeakageParams | None


def _make_noise_spec(
    op_name: str, op: stim.CircuitInstruction, n_q: int, has_classical: bool
) -> tuple[int, int, NDArray[np.float64], bool] | None:
    if has_classical or op_name not in _NOISE_CH:
        return None
    ch_code, arity = _NOISE_CH[op_name]
    args = op.gate_args_copy()
    is_zero = len(args) == 0 or max(args) <= 0.0
    if is_zero:
        probs = np.zeros(1, dtype=np.float64)
    elif op_name == "DEPOLARIZE1":
        p3 = args[0] / 3.0
        probs = np.array([p3, p3, p3], dtype=np.float64)
    elif op_name == "PAULI_CHANNEL_1":
        probs = np.array(args[:3], dtype=np.float64)
    elif op_name == "PAULI_CHANNEL_2":
        probs = np.ascontiguousarray(np.cumsum(args, dtype=np.float64), dtype=np.float64)
    else:
        probs = np.array([args[0]], dtype=np.float64)
    return (ch_code, n_q // arity, probs, is_zero)


@dataclasses.dataclass
class LeakageUint8Coset(OpHandler[BatchedLeakageSimulator]):
    """Batched uint8 leakage OpHandler for CosetsideSimulator reusing tableside tag parsers.

    Args:
        unconditional_condition_on_U: if True (default), untagged unitary gates and
            noise channels are treated as conditioned on U, i.e. they skip leaked
            qubits, whose computational state stays frozen from when they leaked.
            If False, they act on that frozen state as if the qubits were unleaked.
            Instructions that produce measurement results (including heralded noise)
            and resets are never filtered: they act on, or read, the frozen state.
            Details are in "Untagged Instructions on Leaked Qubits" in the top-level
            README.
        fast_leakage: if True, enable distribution-preserving leakage speedups
            (seeded outputs differ from default). See README for details.
    """

    unconditional_condition_on_U: bool = True
    fast_leakage: bool = False

    def compile_op_handler(
        self, *, circuit: stim.Circuit, batch_size: int
    ) -> "CompiledLeakageUint8Coset":
        parsed_ops = parse_leakage_in_circuit(circuit)
        check_tag_qubits(parsed_ops, circuit.num_qubits)
        fast_1q, compiled_len = qualify_fast_1q(
            circuit, parsed_ops, self.unconditional_condition_on_U, self.fast_leakage
        )
        compiled = CompiledLeakageUint8Coset(
            num_qubits=circuit.num_qubits, batch_size=batch_size, ops_to_params=parsed_ops,
            unconditional_condition_on_U=self.unconditional_condition_on_U,
            fast_leakage=self.fast_leakage,
        )
        compiled._set_fast_1q(fast_1q, compiled_len)
        return compiled


@dataclasses.dataclass
class CompiledLeakageUint8Coset(CompiledOpHandler[BatchedLeakageSimulator]):
    """Compiled batched uint8 leakage OpHandler for CosetsideSimulator."""

    num_qubits: int
    batch_size: int
    unconditional_condition_on_U: bool
    ops_to_params: dict[stim.CircuitInstruction, LeakageParams]
    # Keyword-only, so that positional calls keep the parameter order they had before this option.
    fast_leakage: bool = dataclasses.field(default=False, kw_only=True)
    # Set by compile_op_handler from the circuit it compiles; not constructor parameters.
    _fast_1q: bool = dataclasses.field(default=False, init=False)
    _compiled_len: int = dataclasses.field(default=0, init=False)

    _state: NDArray[np.uint8] = dataclasses.field(init=False)
    _scratch_depolarize: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_pauli_X: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_pauli_Y: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_pauli_Z: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_set_one: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_set_zero: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_rand_x: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_rand_phase: NDArray[np.bool_] = dataclasses.field(init=False)
    _scratch_zero_mask: NDArray[np.bool_] = dataclasses.field(init=False)
    _op_records: dict[stim.CircuitInstruction, _OpRec] = dataclasses.field(
        init=False, repr=False, compare=False)
    _op_rec_cfg: tuple[bool, bool] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.claimed_ops_keys = set(self.ops_to_params.keys())
        self._alloc_scratch(self.batch_size)
        self.clear()

    def _set_fast_1q(self, fast_1q: bool, compiled_len: int) -> None:
        """Record whether change 1 of `fast_leakage` applies, and the compiled circuit's length."""
        self._fast_1q, self._compiled_len = fast_1q, compiled_len
        self._op_records, self._op_rec_cfg = {}, (self.unconditional_condition_on_U, fast_1q)

    def _alloc_scratch(self, b_size: int) -> None:
        self._op_records, self._op_rec_cfg = {}, (self.unconditional_condition_on_U, self._fast_1q)
        sh = (self.num_qubits, b_size)
        for nm in (
            "depolarize", "pauli_X", "pauli_Y", "pauli_Z",
            "set_one", "set_zero", "rand_x", "rand_phase", "zero_mask",
        ):
            setattr(self, f"_scratch_{nm}", np.zeros(sh, dtype=np.bool_))

    @property
    def state(self) -> NDArray[np.uint8]:
        return self._state[:, 0] if self.batch_size == 1 else self._state

    @state.setter
    def state(self, val: NDArray[np.uint8]) -> None:
        arr = np.asarray(val, dtype=np.uint8)
        self._state = (
            np.broadcast_to(arr[:, None], (self.num_qubits, self.batch_size)).copy()
            if arr.ndim == 1 else arr.copy()
        )

    def _ensure_batch_size(self, sim_batch_size: int) -> None:
        if self._state.shape[1] != sim_batch_size:
            old_col = self._state[:, :1].copy()
            self.batch_size = sim_batch_size
            self._state = np.broadcast_to(old_col, (self.num_qubits, sim_batch_size)).copy()
            self._alloc_scratch(sim_batch_size)

    def clear(self) -> None:
        self._state = np.zeros((self.num_qubits, self.batch_size), dtype=np.uint8)

    def _update_state_from_peek_Z(
        self, targets: Sequence[int] | NDArray[np.int_], css: BatchedLeakageSimulator
    ) -> None:
        self._ensure_batch_size(css.batch_size)
        unique_targets = list(dict.fromkeys(int(t) for t in targets if int(t) >= 0))
        if not unique_targets:
            return
        p1_mat, any_projected = css.peek_pauli_born_batch(unique_targets, pauli="Z"), False
        for idx, q in enumerate(unique_targets):
            comp_mask = self._state[q, :] < 2
            if not np.any(comp_mask):
                continue
            p1_row = css.peek_pauli_born_batch([q], pauli="Z")[0] if any_projected else p1_mat[idx]
            new_row = np.zeros(css.batch_size, dtype=np.uint8)
            det_one = p1_row >= 1.0 - 1e-12
            new_row[det_one] = 1
            flip_mask = comp_mask & (p1_row > 1e-12) & ~det_one
            num_flips = int(np.count_nonzero(flip_mask))
            if num_flips > 0:
                if (rngs := css.sync_rngs()) is not None:
                    new_row[flip_mask] = np.asarray([
                        rngs.shot_np_rngs[int(b)].binomial(
                            1, 0.5 if abs(p1_row[b] - 0.5) <= 1e-12 else float(p1_row[b])
                        ) for b in np.flatnonzero(flip_mask)
                    ], dtype=np.uint8)
                else:
                    p_sub = p1_row[flip_mask]
                    new_row[flip_mask] = css.np_rng.binomial(
                        1, 0.5 if np.allclose(p_sub, 0.5, atol=1e-12) else p_sub, size=num_flips,
                    ).astype(np.uint8)
                css.postselect_z_batch(q, flip_mask, new_row)
                any_projected = True
            self._state[q, comp_mask] = new_row[comp_mask]

    def _clear_transient_ones(self, qs: Sequence[int] | NDArray[np.int_]) -> None:
        for q in dict.fromkeys(int(t) for t in qs if int(t) >= 0):
            row = self._state[q, :]
            row[row == 1] = 0

    def _filter_targets_1q_mask(
        self, targets: Sequence[int] | NDArray[np.int_],
        allowed_st_list: tuple[tuple[int | str, ...], ...], css: BatchedLeakageSimulator,
    ) -> NDArray[np.bool_]:
        self._ensure_batch_size(css.batch_size)
        targets_arr, allowed_group = np.asarray(targets, dtype=np.intp), allowed_st_list[0]
        valid_qs = targets_arr[targets_arr >= 0]
        peeked = len(valid_qs) > 0 and (1 in allowed_group or 0 in allowed_group)
        if peeked:
            self._update_state_from_peek_Z(valid_qs, css)
        is_cl = (targets_arr < 0)[:, None]
        st = np.where(is_cl, 0, self._state[np.where(targets_arr >= 0, targets_arr, 0), :].copy())
        if peeked:
            self._clear_transient_ones(valid_qs)
        return eval_allowed_state_mask(st, is_cl, allowed_group)

    def _filter_targets_2q_mask(
        self, targets: Sequence[Sequence[int]] | NDArray[np.int_],
        allowed_st_list: tuple[tuple[int | str, ...], tuple[int | str, ...]],
        css: BatchedLeakageSimulator,
    ) -> NDArray[np.bool_]:
        self._ensure_batch_size(css.batch_size)
        targets_np = np.asarray(targets, dtype=np.intp)
        if len(targets_np) == 0:
            return np.zeros((0, css.batch_size), dtype=np.bool_)
        peeked_qs: list[int] = []
        for i in (0, 1):
            if 1 in allowed_st_list[i] or 0 in allowed_st_list[i]:
                valid_qs = targets_np[targets_np[:, i] >= 0, i]
                if len(valid_qs) > 0:
                    self._update_state_from_peek_Z(valid_qs, css)
                    peeked_qs.extend(int(q) for q in valid_qs)
        t0, t1 = targets_np[:, 0], targets_np[:, 1]
        is_c0, is_c1 = (t0 < 0)[:, None], (t1 < 0)[:, None]
        s0 = np.where(is_c0, 0, self._state[np.where(t0 >= 0, t0, 0), :])
        s1 = np.where(is_c1, 0, self._state[np.where(t1 >= 0, t1, 0), :])
        if peeked_qs:
            self._clear_transient_ones(peeked_qs)
        out = np.zeros((len(targets_np), css.batch_size), dtype=np.bool_)
        for n in range(len(allowed_st_list[0])):
            m0 = eval_allowed_state_mask(s0, is_c0, (allowed_st_list[0][n],))
            out |= m0 & eval_allowed_state_mask(s1, is_c1, (allowed_st_list[1][n],))
        return out

    def _make_u_rec(
        self, op: stim.CircuitInstruction, gd: Any, is_1q_or_corr: bool, claimed: bool,
        params: LeakageParams | None,
    ) -> _OpRec:
        op_name, raw_t = op.name, op.targets_copy()
        has_cl = any(t.qubit_value is None for t in raw_t)
        q_list = [int(t.qubit_value) for t in raw_t if t.qubit_value is not None]
        q_check = np.array(list(dict.fromkeys(q_list)), dtype=np.intp)
        qs_arr: NDArray[np.int64] | None = None
        spp_terms: tuple[list[int], ...] | None = None
        if is_1q_or_corr:
            arity = 1
            qs_arr = np.array(
                [int(t.qubit_value) if t.qubit_value is not None else -1 for t in raw_t], np.int64
            )
            kind = (
                _REC_MASK_UNIT_U if gd.is_unitary
                else (_REC_MASK_NOISE_U if gd.is_single_qubit_gate else _REC_MASK_CORR_U)
            )
        elif gd.is_two_qubit_gate:
            arity, kind = 2, (_REC_MASK_UNIT_U if gd.is_unitary else _REC_MASK_NOISE_U)
            qs_arr = np.array(
                [[int(g[0].qubit_value) if g[0].qubit_value is not None else -1,
                  int(g[1].qubit_value) if g[1].qubit_value is not None else -1]
                 for g in op.target_groups()], dtype=np.int64,
            ).reshape(-1, 2)
        else:
            arity, kind = 0, _REC_MASK_SPP_U
            spp_terms = tuple([int(t.qubit_value) for t in term] for term in op.target_groups())
        nspec = (
            _make_noise_spec(op_name, op, len(raw_t), has_cl) if kind == _REC_MASK_NOISE_U else None
        )
        fast_u = (
            kind == _REC_MASK_UNIT_U and arity in (1, 2) and not has_cl and len(q_list) > 0
            and len(q_check) == len(q_list) and op_name not in _FAST_UNIT_SKIP
        )
        return _OpRec(kind, arity, q_check, qs_arr, spp_terms, nspec, fast_u, claimed, params)

    def _build_op_record(
        self, op: stim.CircuitInstruction, claimed: bool, params: LeakageParams | None
    ) -> _OpRec:
        if not claimed:
            if self.unconditional_condition_on_U:
                op_name = op.name
                gd = stim.gate_data(op_name)
                if gd.produces_measurements or not (gd.is_noisy_gate or gd.is_unitary):
                    return _OpRec(_REC_BARE, 0, None, None, None, None, False, False, None)
                is_1q_or_corr = gd.is_single_qubit_gate or op_name in (
                    "E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR",
                )
                if not (is_1q_or_corr or gd.is_two_qubit_gate or op_name in ("SPP", "SPP_DAG")):
                    return _OpRec(_REC_BARE, 0, None, None, None, None, False, False, None)
                if gd.is_single_qubit_gate and self._fast_1q:
                    nspec = (
                        None if gd.is_unitary
                        else _make_noise_spec(op_name, op, len(op.targets_copy()), False)
                    )
                    return _OpRec(
                        _REC_PLAIN_UNIT if gd.is_unitary else _REC_PLAIN_NOISE,
                        1, None, None, None, nspec, False, False, None,
                    )
                return self._make_u_rec(op, gd, is_1q_or_corr, False, None)
            return _OpRec(_REC_BARE, 0, None, None, None, None, False, False, None)
        if (
            isinstance(params, LeakageConditioningParams) and not params.targets
            and type(self).leakage_conditioning is CompiledLeakageUint8Coset.leakage_conditioning
        ):
            gd = stim.gate_data(op.name)
            if not gd.produces_measurements and (gd.is_unitary or gd.is_noisy_gate) and (
                (gd.is_single_qubit_gate and params.args == (("U",),))
                or (gd.is_two_qubit_gate and params.args == (("U",), ("U",)))
            ):
                return self._make_u_rec(op, gd, gd.is_single_qubit_gate, True, params)
        return _OpRec(_REC_TAGGED, 0, None, None, None, None, False, True, params)

    def _get_op_record(self, op: stim.CircuitInstruction) -> _OpRec:
        cfg = (self.unconditional_condition_on_U, self._fast_1q)
        if cfg != self._op_rec_cfg:
            self._op_records, self._op_rec_cfg = {}, cfg
        claimed = op in self.claimed_ops_keys
        params = self.ops_to_params.get(op) if claimed else None
        rec = self._op_records.get(op)
        if rec is not None and rec.claimed == claimed and rec.params is params:
            return rec
        self._op_records[op] = rec = self._build_op_record(op, claimed, params)
        return rec

    def handle_op(self, op: stim.CircuitInstruction, sss: BatchedLeakageSimulator) -> None:
        self._ensure_batch_size(sss.batch_size)
        check_late_frozen_read(self._fast_1q, self._compiled_len, op, sss, self.ops_to_params)
        rec = self._get_op_record(op)
        kind = rec.kind
        if kind == _REC_BARE:
            sss._do_bare_instruction(op)
        elif kind == _REC_TAGGED:
            dispatch_tagged_leakage_op(self, op, sss)
        elif kind == _REC_PLAIN_UNIT:
            sss.do_conditional_clifford(op, None)
        elif kind == _REC_PLAIN_NOISE:
            sss.do_conditional_pauli_noise(op, None)
        else:
            do_cond = (
                sss.do_conditional_clifford if kind in (_REC_MASK_UNIT_U, _REC_MASK_SPP_U)
                else sss.do_conditional_pauli_noise
            )
            qc = rec.q_check
            if qc is None or len(qc) == 0 or not np.any(self._state[qc, :] >= 2):
                do_cond(op, None)
                return
            if rec.arity == 1:
                assert rec.qs_arr is not None
                cm = self._filter_targets_1q_mask(rec.qs_arr, (("U",),), sss)
                if kind == _REC_MASK_CORR_U:
                    cm[:] = cm.all(axis=0)
            elif rec.arity == 2:
                assert rec.qs_arr is not None
                cm = self._filter_targets_2q_mask(rec.qs_arr, (("U",), ("U",)), sss)
            else:
                terms = rec.spp_terms or ()
                cm = (
                    np.array([np.all(self._state[t, :] < 2, axis=0) for t in terms], np.bool_)
                    if terms else np.zeros((0, sss.batch_size), dtype=np.bool_)
                )
            do_cond(op, cm)

    def leakage_swap(self, op: stim.CircuitInstruction, tss: BatchedLeakageSimulator) -> None:
        """SWAP[LEAKAGE_SWAP]: SWAP the qubits in every shot, then swap leakage states."""
        exec_leakage_swap(op, tss, self._state)

    def leakage_conditioning(
        self,
        op: stim.CircuitInstruction,
        tss: BatchedLeakageSimulator,
        params: LeakageConditioningParams,
    ) -> None:
        cg, ot = params.args, params.targets
        gd = stim.gate_data(op.name)
        if ot:
            copies = split_fused_instruction(op, len(ot))
            if len(copies) > 1 and not gd.is_unitary:
                tss.do_copies_on_fused_step(
                    op, copies, lambda sub_op: self.leakage_conditioning(sub_op, tss, params)
                )
                return
            ot = ot * len(copies)
        raw_t = op.targets_copy()
        cg2 = cast(tuple[tuple[int | str, ...], tuple[int | str, ...]], cg)
        if gd.is_unitary and any(0 in g or 1 in g for g in cg):
            t_groups = op.target_groups()
            if len(t_groups) > 1:
                all_qs = [
                    t.qubit_value for grp in t_groups for t in grp if t.qubit_value is not None
                ]
                if ot:
                    all_qs.extend(ot)
                if len(set(all_qs)) < len(all_qs):
                    def _eval_group_cond(g_idx: int) -> NDArray[np.bool_]:
                        grp = t_groups[g_idx]
                        if ot:
                            step = len(ot) // len(t_groups)
                            return self._filter_targets_1q_mask(
                                list(ot[g_idx * step : (g_idx + 1) * step]), cg, tss
                            )[0]
                        if len(cg) == 1:
                            return self._filter_targets_1q_mask(
                                [t.qubit_value if t.qubit_value is not None else -1 for t in grp],
                                cg, tss,
                            )[0]
                        pair = [[
                            grp[0].qubit_value if grp[0].qubit_value is not None else -1,
                            grp[1].qubit_value if grp[1].qubit_value is not None else -1,
                        ]]
                        return self._filter_targets_2q_mask(pair, cg2, tss)[0]

                    tss.do_conditional_clifford(op, condition_group_fn=_eval_group_cond)
                    return
        if ot:
            if len(ot) != len(raw_t):
                raise ValueError(
                    f"The number of targets of the {op} is not the same as the "
                    "number of targets specified in the CONDITIONED_ON_OTHER tag."
                )
            cond_mask = self._filter_targets_1q_mask(list(ot), cg, tss)
        elif len(cg) == 1:
            cond_mask = self._filter_targets_1q_mask(
                [t.qubit_value if t.qubit_value is not None else -1 for t in raw_t], cg, tss
            )
        else:
            pairs = [
                [g[0].qubit_value if g[0].qubit_value is not None else -1,
                 g[1].qubit_value if g[1].qubit_value is not None else -1]
                for g in op.target_groups()
            ]
            cond_mask = self._filter_targets_2q_mask(pairs, cg2, tss)
        if gd.is_unitary:
            tss.do_conditional_clifford(op, cond_mask)
        else:
            tss.do_conditional_pauli_noise(op, cond_mask)

    def _tab_flips(
        self,
        tss: BatchedLeakageSimulator,
        b: int,
        qs: Sequence[int] | NDArray[np.int_],
        p: float,
    ) -> NDArray[np.bool_]:
        q_list = [int(q) for q in qs]
        flips = np.zeros(len(q_list), dtype=np.bool_)
        if q_list and p > 0.0:
            rngs = tss.sync_rngs()
            assert rngs is not None
            tr = rngs.tab_rngs[b]
            inst = stim.CircuitInstruction("X_ERROR", q_list, [p])
            tr.do(inst)
            rngs.tab_rngs_x[b].do(inst)
            for i, q in enumerate(q_list):
                if tr.peek_z(q) == -1:
                    tr.x(q)
                    flips[i] = True
        return flips

    @staticmethod
    def _targets_per_shot(mask: NDArray[np.bool_]) -> list[list[np.intp]]:
        return [list(np.where(mask[:, b])[0]) for b in range(mask.shape[1])]

    def _apply_transition_effects(
        self, tss: BatchedLeakageSimulator, effects: TransitionEffectMasks
    ) -> None:
        if np.any(effects.set_zero):
            tss.do_unmatched_noisy_reset_z(
                effects.set_zero, self._scratch_zero_mask
            )
        if np.any(effects.set_one):
            tss.do_unmatched_noisy_reset_z(
                self._scratch_zero_mask, effects.set_one
            )
        if np.any(effects.depolarize):
            if tss.sync_rngs() is not None:
                tss.sample_pauli_noise_sync(
                    "DEPOLARIZE1", self._targets_per_shot(effects.depolarize), [0.75]
                )
            else:
                tss.apply_pauli("X", effects.depolarize, 0.5)
                tss.apply_pauli("Z", effects.depolarize, 0.5)
        for pauli, arr in (
            ("X", effects.pauli_x),
            ("Y", effects.pauli_y),
            ("Z", effects.pauli_z),
        ):
            if arr is not None and np.any(arr):
                tss.apply_pauli(pauli, arr)
        for gate, pauli, arr in (
            ("X_ERROR", "X", effects.rand_x),
            ("Z_ERROR", "Z", effects.rand_phase),
        ):
            if arr is not None and np.any(arr):
                if tss.sync_rngs() is not None:
                    tss.sample_pauli_noise_sync(
                        gate, self._targets_per_shot(arr), [0.5]
                    )
                else:
                    tss.apply_pauli(pauli, arr, 0.5)

    def _attempt_project(
        self, targets: NDArray[np.intp], css: BatchedLeakageSimulator, p_star: float
    ) -> NDArray[np.bool_]:
        self._ensure_batch_size(css.batch_size)
        comp = self._state[targets, :] < 2
        attempt = np.zeros(comp.shape, dtype=np.bool_)
        n_comp = int(np.count_nonzero(comp))
        n_att = (
            n_comp if p_star >= 1.0 - 1e-12
            else (0 if p_star <= 0.0 else int(css.np_rng.binomial(n_comp, p_star)))
        )
        if n_att > 0:
            rows_c, shots_c = np.nonzero(comp)
            pick: slice | NDArray[np.intp] = (
                slice(None) if n_att == n_comp else css.np_rng.choice(n_comp, n_att, replace=False)
            )
            attempt[rows_c[pick], shots_c[pick]] = True
        if n_att < n_comp:
            self._scratch_pauli_Z.fill(False)
            self._scratch_pauli_Z[targets, :] = comp & ~attempt
            css.apply_pauli("Z", self._scratch_pauli_Z, 0.5)
        rows = np.flatnonzero(attempt.any(axis=1))
        if len(rows) == 0:
            return attempt
        p1_mat = css.peek_pauli_born_batch(targets[rows], pauli="Z")
        dirty = np.zeros(css.batch_size, dtype=np.bool_)
        for j, i in enumerate(rows):
            q, m = int(targets[i]), attempt[i]
            p1_row = (
                css.peek_pauli_born_batch([q], pauli="Z")[0]
                if np.any(m & dirty) else p1_mat[j]
            )
            det_one = p1_row >= 1.0 - 1e-12
            new_row = det_one.astype(np.uint8)
            flip = m & (p1_row > 1e-12) & ~det_one
            if np.any(flip):
                p_sub = p1_row[flip]
                p_arg = 0.5 if np.allclose(p_sub, 0.5, atol=1e-12) else p_sub
                new_row[flip] = css.np_rng.binomial(1, p_arg, size=int(flip.sum())).astype(np.uint8)
                css.postselect_z_batch(q, flip, new_row)
                dirty |= flip
            self._state[q, m] = new_row[m]
        return attempt

    def leakage_transition_1(
        self, op: stim.CircuitInstruction, tss: BatchedLeakageSimulator,
        params: LeakageTransition1Params, *, _advance_step: bool = True,
    ) -> None:
        raw_t = op.targets_copy()
        target_indices = [t.qubit_value if t.qubit_value is not None else t.value for t in raw_t]
        if _advance_step:
            tss._next_step_meta(op)
        if len(target_indices) > 1 and len(set(target_indices)) < len(target_indices):
            for t in raw_t:
                self.leakage_transition_1(
                    stim.CircuitInstruction(op.name, [t], op.gate_args_copy()),
                    tss, params, _advance_step=False,
                )
            return
        sorted_target_indices = np.array(sorted(set(target_indices)), dtype=np.intp)
        to_dep, to_one, to_zero = (
            self._scratch_depolarize, self._scratch_set_one, self._scratch_set_zero
        )
        to_dep.fill(False)
        to_one.fill(False)
        to_zero.fill(False)
        effects = TransitionEffectMasks(to_zero, to_one, to_dep)
        rngs = tss.sync_rngs()
        batched = rngs is None and not (tss.batch_size == 1 and tss.seed is not None)
        peeked_01 = any(k in (0, 1) for k in params.args_by_input_state.keys())
        attempt: NDArray[np.bool_] | None = None
        p_star = 0.0
        if peeked_01:
            if self.fast_leakage and batched and "U" not in params.args_by_input_state:
                p_star = max(
                    float(sum(p for _, p in tr))
                    for k, tr in params.args_by_input_state.items() if k in (0, 1)
                )
                attempt = self._attempt_project(sorted_target_indices, tss, p_star)
            else:
                self._update_state_from_peek_Z(sorted_target_indices, tss)
        target_states = self._state[sorted_target_indices, :].copy()
        if not batched:
            for input_state in params.args_by_input_state.keys():
                for b in range(tss.batch_size):
                    in_mask_b = (
                        target_states[:, b] < 2 if input_state == "U"
                        else target_states[:, b] == input_state
                    )
                    ow_qs = sorted_target_indices[in_mask_b]
                    if len(ow_qs) == 0:
                        continue
                    rng_b = rngs.shot_np_rngs[b] if rngs is not None else tss.np_rng
                    out_b = params.sample_transitions_from_state(
                        input_state=input_state, num_samples=len(ow_qs), np_rng=rng_b
                    ).astype(str)
                    for q_np, out_st in zip(ow_qs, out_b):
                        apply_output_effect(
                            self._state, effects, (int(q_np), b), input_state, out_st,
                            drop_depol_reset=self.fast_leakage,
                        )
        else:
            for input_state, transitions in params.args_by_input_state.items():
                total_p = float(sum(p for _, p in transitions))
                if total_p <= 0.0:
                    continue
                input_state_mask = (
                    target_states < 2 if input_state == "U" else target_states == input_state
                )
                if attempt is not None and input_state in (0, 1):
                    input_state_mask &= attempt
                    transitions = tuple((o, p / p_star) for o, p in transitions)
                    total_p = min(1.0, total_p / p_star)
                for out_s, r_sub, b_sub in sample_batched_group_transitions(
                    transitions, total_p, tss.np_rng, input_state_mask
                ):
                    if len(r_sub) > 0 and str(out_s) != str(input_state):
                        apply_output_effect(
                            self._state, effects, (sorted_target_indices[r_sub], b_sub),
                            input_state, out_s, drop_depol_reset=self.fast_leakage,
                        )
        if peeked_01:
            self._clear_transient_ones(sorted_target_indices)
        self._apply_transition_effects(tss, effects)

    def leakage_transition_2(
        self,
        op: stim.CircuitInstruction,
        tss: BatchedLeakageSimulator,
        params: LeakageTransition2Params,
        *,
        _advance_step: bool = True,
    ) -> None:
        if _advance_step:
            tss._next_step_meta(op)
        targets_list = [t.qubit_value for t in op.targets_copy()]
        if len(targets_list) > 2 and len(set(targets_list)) < len(targets_list):
            for grp in op.target_groups():
                sub_op = stim.CircuitInstruction(
                    op.name, grp, op.gate_args_copy()
                )
                self.leakage_transition_2(
                    sub_op, tss, params, _advance_step=False
                )
            return
        even_targets_list = targets_list[::2]
        odd_targets_list = targets_list[1::2]
        even_targets_arr = np.asarray(even_targets_list, dtype=np.intp)
        odd_targets_arr = np.asarray(odd_targets_list, dtype=np.intp)

        peeked_2q: list[int] = []
        if any(k[0] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(even_targets_list, tss)
            peeked_2q.extend(even_targets_list)
        if any(k[1] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(odd_targets_list, tss)
            peeked_2q.extend(odd_targets_list)

        even_target_states = self._state[even_targets_arr, :].copy()
        odd_target_states = self._state[odd_targets_arr, :].copy()

        effects = TransitionEffectMasks(
            set_zero=self._scratch_set_zero,
            set_one=self._scratch_set_one,
            depolarize=self._scratch_depolarize,
            pauli_x=self._scratch_pauli_X,
            pauli_y=self._scratch_pauli_Y,
            pauli_z=self._scratch_pauli_Z,
            rand_x=self._scratch_rand_x,
            rand_phase=self._scratch_rand_phase,
        )
        for arr in (
            effects.set_zero,
            effects.set_one,
            effects.depolarize,
            effects.pauli_x,
            effects.pauli_y,
            effects.pauli_z,
            effects.rand_x,
            effects.rand_phase,
        ):
            assert arr is not None
            arr.fill(False)

        if (rngs := tss.sync_rngs()) is not None or (
            tss.batch_size == 1 and tss.seed is not None
        ):
            for input_state in params.args_by_input_state.keys():
                even_checks = (
                    even_target_states < 2
                    if input_state[0] == "U"
                    else even_target_states == input_state[0]
                )
                odd_checks = (
                    odd_target_states < 2
                    if input_state[1] == "U"
                    else odd_target_states == input_state[1]
                )
                target_state_mask = even_checks & odd_checks
                for b in range(tss.batch_size):
                    t_mask_b = target_state_mask[:, b]
                    samples_b = int(np.count_nonzero(t_mask_b))
                    if samples_b == 0:
                        continue
                    trig_pairs_b = np.column_stack(
                        (even_targets_arr[t_mask_b], odd_targets_arr[t_mask_b])
                    )
                    rng_b = rngs.shot_np_rngs[b] if rngs is not None else tss.np_rng
                    out_b = params.sample_transitions_from_state(
                        input_state=input_state, num_samples=samples_b, np_rng=rng_b
                    ).astype(str)
                    for pair, out_pair in zip(trig_pairs_b, out_b):
                        for leg in (0, 1):
                            apply_output_effect(
                                self._state, effects, (int(pair[leg]), b),
                                input_state[leg], out_pair[leg], drop_depol_reset=self.fast_leakage,
                            )
        else:
            for input_state, transitions in params.args_by_input_state.items():
                total_p = float(sum(p for _, p in transitions))
                if total_p <= 0.0:
                    continue
                even_checks = (
                    even_target_states < 2
                    if input_state[0] == "U"
                    else even_target_states == input_state[0]
                )
                odd_checks = (
                    odd_target_states < 2
                    if input_state[1] == "U"
                    else odd_target_states == input_state[1]
                )
                for out_pair, p_sub, b_sub in sample_batched_group_transitions(
                    transitions, total_p, tss.np_rng, even_checks & odd_checks
                ):
                    if len(p_sub) == 0:
                        continue
                    for q_sub, in_s, out_s in (
                        (even_targets_arr[p_sub], input_state[0], out_pair[0]),
                        (odd_targets_arr[p_sub], input_state[1], out_pair[1]),
                    ):
                        apply_output_effect(
                            self._state, effects, (q_sub, b_sub), in_s, out_s,
                            drop_depol_reset=self.fast_leakage,
                        )

        if peeked_2q:
            self._clear_transient_ones(peeked_2q)
        self._apply_transition_effects(tss, effects)

    def _sample_leaked_tab_outcomes(
        self,
        tss: BatchedLeakageSimulator,
        targets_arr: NDArray[np.intp],
        target_leakage_state: NDArray[np.uint8],
        params: LeakageMeasurementParams,
        outcomes: NDArray[np.bool_],
    ) -> None:
        for b in range(tss.batch_size):
            st_b = target_leakage_state[:, b]
            if not np.any(st_b > 1):
                continue
            for n in range(2, int(np.max(st_b)) + 1):
                if n in st_b and n in params.prob_for_input_state:
                    idx_n = np.where(st_b == n)[0]
                    outcomes[idx_n, b] = self._tab_flips(
                        tss, b, targets_arr[idx_n], params.prob_for_input_state[n]
                    )

    def _finish_sync_proj_z(
        self,
        tss: BatchedLeakageSimulator,
        op: stim.CircuitInstruction,
        raw_op_targets: list[stim.GateTarget],
        targets: list[int],
        targets_arr: NDArray[np.intp],
        inv_flags: list[bool],
        basis: Literal["X", "Y", "Z"],
        target_leakage_state: NDArray[np.uint8],
        outcomes: NDArray[np.bool_],
        replace_mask: NDArray[np.bool_] | None,
        has_leaked: bool,
    ) -> None:
        tss._do_bare_instruction(stim.CircuitInstruction(op.name, raw_op_targets))
        num_m = len(targets)
        if replace_mask is not None or has_leaked:
            net_x_flip = np.zeros((self.num_qubits, tss.batch_size), dtype=np.bool_)
            for k_i, q in enumerate(targets):
                m_col = tss.measurement_column(-num_m + k_i)
                target_out = outcomes[k_i] ^ inv_flags[k_i]
                if replace_mask is not None:
                    mask_k = replace_mask[k_i]
                    if not np.any(mask_k):
                        continue
                    new_col = np.where(mask_k, target_out, m_col)
                else:
                    new_col = target_out
                diff = m_col ^ new_col
                if np.any(diff):
                    tss.overwrite_measurement_column(-num_m + k_i, new_col)
                    leaked_diff = diff & (target_leakage_state[k_i] > 1)
                    if np.any(leaked_diff):
                        net_x_flip[q, :] ^= leaked_diff
            if np.any(net_x_flip):
                tss.apply_pauli("Z" if basis == "X" else "X", net_x_flip)
        elif replace_mask is None:
            for k_i in range(num_m):
                m_col = tss.measurement_column(-num_m + k_i)
                target_out = outcomes[k_i] ^ inv_flags[k_i]
                if np.any(m_col ^ target_out):
                    tss.overwrite_measurement_column(-num_m + k_i, target_out)
        if has_leaked:
            leaked_per_shot = [
                list(targets_arr[target_leakage_state[:, b] > 1])
                for b in range(tss.batch_size)
            ]
            tss.sample_pauli_noise_sync("DEPOLARIZE1", leaked_per_shot, [0.75])

    def leakage_measurement(
        self,
        op: stim.CircuitInstruction,
        tss: BatchedLeakageSimulator,
        params: LeakageMeasurementParams,
    ) -> None:
        if params.targets:
            copies = split_fused_instruction(op, len(params.targets))
            if len(copies) > 1:
                tss.do_copies_on_fused_step(
                    op, copies, lambda sub_op: self.leakage_measurement(sub_op, tss, params)
                )
                return
        is_mpad, raw_op_targets, targets, inv_flags = unpack_leakage_measurement_targets(
            op, params
        )

        target_leakage_state = self._state[targets, :].copy()  # (len(targets), batch_size)
        targets_arr = np.asarray(targets, dtype=np.intp)
        p0, p1 = params.p0, params.p1
        p_super = 0.5 * p0 + 0.5 * p1
        basis: Literal["X", "Y", "Z"] = meas_basis(op.name)
        leaked_mask = target_leakage_state > 1
        has_leaked = bool(np.any(leaked_mask))
        sync = tss.sync_rngs()

        if not is_mpad and sync is not None and len(set(targets)) < len(targets):
            if p0 > 0.0 or p1 < 1.0 or has_leaked:
                raise NotImplementedError(
                    f"Duplicate targets in {op.name} with LEAKAGE_PROJECTION_Z are not supported with "
                    "sync_tableside_rng=True when readout is noisy or any target is leaked."
                )

        # Fast-path for LEAKAGE_PROJECTION_Z measurements with noiseless computational readout (p0=0, p1=1) and sync_tableside_rng
        if not is_mpad and sync is not None and p0 == 0.0 and p1 == 1.0:
            leaked_read_one = np.zeros((len(targets), tss.batch_size), dtype=np.bool_)
            if has_leaked:
                leaked_zero_mask = self._scratch_set_zero
                leaked_zero_mask.fill(False)
                leaked_zero_mask[targets_arr, :] = leaked_mask
                tss.do_unmatched_noisy_reset_z(
                    leaked_zero_mask, self._scratch_zero_mask, pauli=basis, order=targets
                )
                self._sample_leaked_tab_outcomes(
                    tss, targets_arr, target_leakage_state, params, leaked_read_one
                )
            self._finish_sync_proj_z(
                tss,
                op,
                raw_op_targets,
                targets,
                targets_arr,
                inv_flags,
                basis,
                target_leakage_state,
                leaked_read_one,
                leaked_mask if has_leaked else np.zeros_like(leaked_mask),
                has_leaked,
            )
            return

        is_reset_meas = op.name in ("MR", "MRZ", "MRX", "MRY")
        first_occ: dict[int, int] = {}
        has_dup_targets = False
        if not is_mpad:
            for k_i, q_val in enumerate(targets):
                q_int = int(q_val)
                if q_int in first_occ:
                    has_dup_targets = True
                else:
                    first_occ[q_int] = k_i

        has_01_cond = (0 in params.prob_for_input_state) or (
            1 in params.prob_for_input_state
        )
        if is_mpad and has_01_cond and sync is None:
            self._update_state_from_peek_Z(targets, tss)
            self._clear_transient_ones(targets)

        need_pauli_peek = sync is not None or is_mpad or has_01_cond
        pauli_states = (
            tss.peek_pauli_batch(targets, basis)
            if need_pauli_peek
            else np.zeros((len(targets), tss.batch_size), dtype=np.int8)
        )

        if sync is not None:
            final_outcomes = np.zeros((len(targets), tss.batch_size), dtype=np.bool_)
            # Resolve superpositions via _shot_np_rngs + postselect_z_batch when not (is_mpad and not has_01_cond and np.isclose(p0, p1))
            if not (is_mpad and not has_01_cond and np.isclose(p0, p1)):
                for b in range(tss.batch_size):
                    idx_in_q = np.where(target_leakage_state[:, b] < 2)[0]
                    superpos_local = np.where(pauli_states[idx_in_q, b] == 0)[0]
                    for s_i, loc_idx in enumerate(superpos_local):
                        k_i = int(idx_in_q[loc_idx])
                        q_int = int(targets_arr[k_i])
                        if has_dup_targets and first_occ[q_int] < k_i:
                            pauli_states[k_i, b] = (
                                1 if is_reset_meas else pauli_states[first_occ[q_int], b]
                            )
                            continue
                        pz = 0 if s_i == 0 else int(tss.peek_pauli_batch([q_int], basis)[0, b])
                        if pz == 0:
                            bit = int(sync.shot_np_rngs[b].binomial(1, 0.5))
                            flip_mask_1d = np.zeros(tss.batch_size, dtype=np.bool_)
                            flip_mask_1d[b] = True
                            desired_1d = np.zeros(tss.batch_size, dtype=np.uint8)
                            desired_1d[b] = bit
                            tss.postselect_z_batch(
                                q_int, flip_mask_1d, desired_1d, pauli=basis
                            )
                            pz = -1 if bit == 1 else 1
                        pauli_states[k_i, b] = pz
            if has_dup_targets:
                for k_i, q_val in enumerate(targets):
                    f_k = first_occ[int(q_val)]
                    if f_k < k_i:
                        in_c = target_leakage_state[k_i, :] < 2
                        pauli_states[k_i, in_c] = (
                            1 if is_reset_meas else pauli_states[f_k, in_c]
                        )

            for b in range(tss.batch_size):
                in_comp_b = target_leakage_state[:, b] < 2
                targets_in_q = targets_arr[in_comp_b]
                idx_in_q = np.where(in_comp_b)[0]
                ps_b = pauli_states[in_comp_b, b]

                if is_mpad and not has_01_cond and np.isclose(p0, p1):
                    final_outcomes[idx_in_q, b] = self._tab_flips(tss, b, targets_in_q, p0)
                else:
                    m0_mask = ps_b == 1
                    m1_mask = ps_b == -1
                    final_outcomes[idx_in_q[m0_mask], b] = self._tab_flips(
                        tss, b, targets_in_q[m0_mask], p0
                    )
                    final_outcomes[idx_in_q[m1_mask], b] = ~self._tab_flips(
                        tss, b, targets_in_q[m1_mask], 1.0 - p1
                    )

            if not is_mpad and has_leaked:
                leaked_zero_mask = self._scratch_set_zero
                leaked_zero_mask.fill(False)
                leaked_zero_mask[targets_arr, :] = leaked_mask
                tss.do_unmatched_noisy_reset_z(
                    leaked_zero_mask, self._scratch_zero_mask, pauli=basis, order=targets
                )
            if has_leaked:
                self._sample_leaked_tab_outcomes(
                    tss, targets_arr, target_leakage_state, params, final_outcomes
                )

            if is_mpad:
                tss._next_step_meta(op)
                inv_arr = np.asarray(inv_flags, dtype=np.bool_)[:, None]
                tss.record_direct_measurements(final_outcomes ^ inv_arr)
            else:
                self._finish_sync_proj_z(
                    tss,
                    op,
                    raw_op_targets,
                    targets,
                    targets_arr,
                    inv_flags,
                    basis,
                    target_leakage_state,
                    final_outcomes,
                    None,
                    has_leaked,
                )
            return

        if has_dup_targets:
            for k_i, q_val in enumerate(targets):
                f_k = first_occ[int(q_val)]
                if f_k < k_i:
                    in_c = target_leakage_state[k_i, :] < 2
                    pauli_states[k_i, in_c] = (
                        1 if is_reset_meas else pauli_states[f_k, in_c]
                    )

        if not is_mpad:
            if has_leaked and not self.fast_leakage:
                leaked_zero_mask = self._scratch_set_zero
                leaked_zero_mask.fill(False)
                leaked_zero_mask[targets_arr, :] = leaked_mask
                tss.do_unmatched_noisy_reset_z(
                    leaked_zero_mask, self._scratch_zero_mask, pauli=basis
                )
            # Execute the matched measurement first so the reference tableau and active cosets collapse in O(1) time
            tss._do_bare_instruction(stim.CircuitInstruction(op.name, raw_op_targets))

        prob_Read1 = np.zeros((len(targets), tss.batch_size), dtype=np.float64)
        in_comp = ~leaked_mask
        prob_Read1[in_comp & (pauli_states == 1)] = p0
        prob_Read1[in_comp & (pauli_states == -1)] = p1
        prob_Read1[in_comp & (pauli_states == 0)] = p_super
        for n in range(2, int(np.max(target_leakage_state, initial=0)) + 1):
            if n in params.prob_for_input_state:
                prob_Read1[target_leakage_state == n] = params.prob_for_input_state[n]

        if is_mpad:
            tss._next_step_meta(op)
            outcomes = tss.np_rng.random((len(targets), tss.batch_size)) < prob_Read1
            inv_arr = np.asarray(inv_flags, dtype=np.bool_)[:, None]
            tss.record_direct_measurements(outcomes ^ inv_arr)
            return

        num_m = len(targets)
        for k_i in range(num_m):
            m_col = tss.measurement_column(-num_m + k_i)
            m_phys = m_col ^ inv_flags[k_i]
            super_k = in_comp[k_i] & (pauli_states[k_i] == 0)
            if np.any(super_k):
                prob_Read1[k_i, super_k & (~m_phys)] = p0
                prob_Read1[k_i, super_k & m_phys] = p1
        need_resample = (
            (in_comp & (pauli_states == 1) & (p0 > 0.0))
            | (in_comp & (pauli_states == -1) & (p1 < 1.0))
            | (in_comp & (pauli_states == 0) & ((p0 > 0.0) | (p1 < 1.0)))
            | leaked_mask
        )
        if np.any(need_resample):
            sampled_outcomes = (
                tss.np_rng.random((len(targets), tss.batch_size)) < prob_Read1
            )
            for k_i in range(num_m):
                mask_k = need_resample[k_i]
                if not np.any(mask_k):
                    continue
                m_col = tss.measurement_column(-num_m + k_i)
                tss.overwrite_measurement_column(
                    -num_m + k_i, np.where(mask_k, sampled_outcomes[k_i] ^ inv_flags[k_i], m_col)
                )
        if has_leaked:
            to_dep = self._scratch_depolarize
            to_dep.fill(False)
            to_dep[targets_arr, :] = leaked_mask
            tss.apply_pauli("X", to_dep, 0.5)
            tss.apply_pauli("Z", to_dep, 0.5)
