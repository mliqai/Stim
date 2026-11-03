from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, Sequence, cast

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
    dispatch_tagged_leakage_op,
    eval_allowed_state_mask,
    exec_leakage_swap,
    meas_basis,
    sample_batched_group_transitions,
    unpack_leakage_measurement_targets,
)
from stimside.util.stim_workarounds import split_fused_instruction

if TYPE_CHECKING:
    from stimside.simulator_coset import CosetsideSimulator


@dataclasses.dataclass
class LeakageUint8Coset(OpHandler["CosetsideSimulator"]):
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
    """

    unconditional_condition_on_U: bool = True

    def compile_op_handler(
        self, *, circuit: stim.Circuit, batch_size: int
    ) -> "CompiledLeakageUint8Coset":
        parsed_ops = parse_leakage_in_circuit(circuit)
        check_tag_qubits(parsed_ops, circuit.num_qubits)
        return CompiledLeakageUint8Coset(
            num_qubits=circuit.num_qubits,
            batch_size=batch_size,
            ops_to_params=parsed_ops,
            unconditional_condition_on_U=self.unconditional_condition_on_U,
        )


@dataclasses.dataclass
class CompiledLeakageUint8Coset(CompiledOpHandler["CosetsideSimulator"]):
    """Compiled batched uint8 leakage OpHandler for CosetsideSimulator."""

    num_qubits: int
    batch_size: int
    unconditional_condition_on_U: bool
    ops_to_params: dict[stim.CircuitInstruction, LeakageParams]

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

    def __post_init__(self) -> None:
        self.claimed_ops_keys = set(self.ops_to_params.keys())
        self._alloc_scratch(self.batch_size)
        self.clear()

    def _alloc_scratch(self, b_size: int) -> None:
        shape = (self.num_qubits, b_size)
        self._scratch_depolarize = np.zeros(shape, dtype=np.bool_)
        self._scratch_pauli_X = np.zeros(shape, dtype=np.bool_)
        self._scratch_pauli_Y = np.zeros(shape, dtype=np.bool_)
        self._scratch_pauli_Z = np.zeros(shape, dtype=np.bool_)
        self._scratch_set_one = np.zeros(shape, dtype=np.bool_)
        self._scratch_set_zero = np.zeros(shape, dtype=np.bool_)
        self._scratch_rand_x = np.zeros(shape, dtype=np.bool_)
        self._scratch_rand_phase = np.zeros(shape, dtype=np.bool_)
        self._scratch_zero_mask = np.zeros(shape, dtype=np.bool_)

    @property
    def state(self) -> NDArray[np.uint8]:
        if self.batch_size == 1:
            return self._state[:, 0]
        return self._state

    @state.setter
    def state(self, val: NDArray[np.uint8]) -> None:
        arr = np.asarray(val, dtype=np.uint8)
        if arr.ndim == 1:
            self._state = np.broadcast_to(
                arr[:, None], (self.num_qubits, self.batch_size)
            ).copy()
        else:
            self._state = arr.copy()

    def _ensure_batch_size(self, sim_batch_size: int) -> None:
        if self._state.shape[1] != sim_batch_size:
            old_col = self._state[:, :1].copy()
            self.batch_size = sim_batch_size
            self._state = np.broadcast_to(
                old_col, (self.num_qubits, sim_batch_size)
            ).copy()
            self._alloc_scratch(sim_batch_size)

    def clear(self) -> None:
        self._state = np.zeros(
            (self.num_qubits, self.batch_size), dtype=np.uint8
        )

    def _update_state_from_peek_Z(
        self, targets: Sequence[int] | NDArray[np.int_], css: CosetsideSimulator
    ) -> None:
        self._ensure_batch_size(css.batch_size)
        unique_targets = list(dict.fromkeys(int(t) for t in targets if int(t) >= 0))
        if not unique_targets:
            return

        p1_mat = css.peek_pauli_born_batch(unique_targets, pauli="Z")
        any_projected = False
        for idx, q in enumerate(unique_targets):
            comp_mask = self._state[q, :] < 2
            if not np.any(comp_mask):
                continue
            p1_row = (
                css.peek_pauli_born_batch([q], pauli="Z")[0]
                if any_projected
                else p1_mat[idx]
            )
            new_row = np.zeros(css.batch_size, dtype=np.uint8)
            det_one = p1_row >= 1.0 - 1e-12
            new_row[det_one] = 1
            flip_mask = comp_mask & (p1_row > 1e-12) & ~det_one
            num_flips = int(np.count_nonzero(flip_mask))
            if num_flips > 0:
                if css._shot_np_rngs is not None:
                    new_row[flip_mask] = np.asarray(
                        [
                            css._shot_np_rngs[int(b)].binomial(
                                1,
                                0.5 if abs(p1_row[b] - 0.5) <= 1e-12 else float(p1_row[b]),
                            )
                            for b in np.flatnonzero(flip_mask)
                        ],
                        dtype=np.uint8,
                    )
                else:
                    p_sub = p1_row[flip_mask]
                    new_row[flip_mask] = css.np_rng.binomial(
                        1,
                        0.5 if np.allclose(p_sub, 0.5, atol=1e-12) else p_sub,
                        size=num_flips,
                    ).astype(np.uint8)
                css.postselect_z_batch(q, flip_mask, new_row)
                any_projected = True
            self._state[q, comp_mask] = new_row[comp_mask]

    def _clear_transient_ones(self, qs: Sequence[int] | NDArray[np.int_]) -> None:
        for q in dict.fromkeys(int(t) for t in qs if int(t) >= 0):
            row = self._state[q, :]
            row[row == 1] = 0

    def _filter_targets_1q_mask(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        allowed_st_list: tuple[tuple[int | str, ...], ...],
        css: CosetsideSimulator,
    ) -> NDArray[np.bool_]:
        self._ensure_batch_size(css.batch_size)
        targets_arr = np.asarray(targets, dtype=np.intp)
        allowed_group = allowed_st_list[0]
        valid_qs = targets_arr[targets_arr >= 0]
        peeked = len(valid_qs) > 0 and (1 in allowed_group or 0 in allowed_group)
        if peeked:
            self._update_state_from_peek_Z(valid_qs, css)

        safe_qs = np.where(targets_arr >= 0, targets_arr, 0)
        target_states = self._state[safe_qs, :].copy()  # (len(targets), batch_size)
        is_classical = (targets_arr < 0)[:, None]
        target_states = np.where(is_classical, 0, target_states)
        if peeked:
            self._clear_transient_ones(valid_qs)
        return eval_allowed_state_mask(
            target_states, is_classical, allowed_group, allowed_st_list
        )

    def _filter_targets_2q_mask(
        self,
        targets: Sequence[Sequence[int]] | NDArray[np.int_],
        allowed_st_list: tuple[tuple[int | str, ...], tuple[int | str, ...]],
        css: CosetsideSimulator,
    ) -> NDArray[np.bool_]:
        self._ensure_batch_size(css.batch_size)
        targets_np = np.asarray(targets, dtype=np.intp)  # (num_pairs, 2)
        if len(targets_np) == 0:
            return np.zeros((0, css.batch_size), dtype=np.bool_)
        peeked_qs: list[int] = []
        for i in (0, 1):
            if 1 in allowed_st_list[i] or 0 in allowed_st_list[i]:
                valid_qs = targets_np[targets_np[:, i] >= 0, i]
                if len(valid_qs) > 0:
                    self._update_state_from_peek_Z(valid_qs, css)
                    peeked_qs.extend(int(q) for q in valid_qs)

        is_c0 = (targets_np[:, 0] < 0)[:, None]
        is_c1 = (targets_np[:, 1] < 0)[:, None]
        q0_safe = np.where(targets_np[:, 0] >= 0, targets_np[:, 0], 0)
        q1_safe = np.where(targets_np[:, 1] >= 0, targets_np[:, 1], 0)
        s0 = np.where(is_c0, 0, self._state[q0_safe, :])
        s1 = np.where(is_c1, 0, self._state[q1_safe, :])
        if peeked_qs:
            self._clear_transient_ones(peeked_qs)
        allowed_mask = np.zeros((len(targets_np), css.batch_size), dtype=np.bool_)
        for n in range(len(allowed_st_list[0])):
            m0 = eval_allowed_state_mask(
                s0, is_c0, (allowed_st_list[0][n],), allowed_st_list
            )
            m1 = eval_allowed_state_mask(
                s1, is_c1, (allowed_st_list[1][n],), allowed_st_list
            )
            allowed_mask |= m0 & m1
        return allowed_mask

    def handle_op(
        self, op: stim.CircuitInstruction, sss: CosetsideSimulator
    ) -> None:
        self._ensure_batch_size(sss.batch_size)
        if op not in self.claimed_ops_keys:
            if self.unconditional_condition_on_U:
                op_name = op.name
                gd = stim.gate_data(op_name)
                if gd.produces_measurements or not (
                    gd.is_noisy_gate or gd.is_unitary
                ):
                    sss._do_bare_instruction(op)
                    return

                is_1q_or_corr = gd.is_single_qubit_gate or op_name in (
                    "E",
                    "CORRELATED_ERROR",
                    "ELSE_CORRELATED_ERROR",
                )
                if not (
                    is_1q_or_corr
                    or gd.is_two_qubit_gate
                    or op_name in ("SPP", "SPP_DAG")
                ):
                    sss._do_bare_instruction(op)
                    return

                qs = [
                    t.qubit_value
                    for t in op.targets_copy()
                    if t.qubit_value is not None
                ]
                if not qs or not np.any(self._state[qs, :] >= 2):
                    if gd.is_unitary:
                        sss.do_conditional_clifford(op, None)
                    else:
                        sss.do_conditional_pauli_noise(op, None)
                    return

                if is_1q_or_corr:
                    targets = [
                        t.qubit_value if t.qubit_value is not None else -1
                        for t in op.targets_copy()
                    ]
                    cond_mask = self._filter_targets_1q_mask(
                        targets, (("U",),), sss
                    )
                    if not gd.is_single_qubit_gate:
                        # A correlated error is one Pauli product (like a pair for 2q ops): skip all of it.
                        cond_mask[:] = cond_mask.all(axis=0)
                elif gd.is_two_qubit_gate:
                    target_pairs = [
                        [
                            grp[0].qubit_value
                            if grp[0].qubit_value is not None
                            else -1,
                            grp[1].qubit_value
                            if grp[1].qubit_value is not None
                            else -1,
                        ]
                        for grp in op.target_groups()
                    ]
                    cond_mask = self._filter_targets_2q_mask(
                        target_pairs, (("U",), ("U",)), sss
                    )
                else:
                    terms = op.target_groups()
                    cond_mask = (
                        np.array(
                            [
                                np.all(self._state[[t.qubit_value for t in term], :] < 2, axis=0)
                                for term in terms
                            ],
                            dtype=np.bool_,
                        )
                        if terms
                        else np.zeros((0, sss.batch_size), dtype=np.bool_)
                    )

                if gd.is_unitary:
                    sss.do_conditional_clifford(op, cond_mask)
                else:
                    sss.do_conditional_pauli_noise(op, cond_mask)
                return

            sss._do_bare_instruction(op)
            return

        dispatch_tagged_leakage_op(self, op, sss)

    def leakage_swap(self, op: stim.CircuitInstruction, tss: CosetsideSimulator) -> None:
        """SWAP[LEAKAGE_SWAP]: SWAP the qubits in every shot, then swap leakage states."""
        exec_leakage_swap(op, tss, self._state)

    def leakage_conditioning(
        self,
        op: stim.CircuitInstruction,
        tss: CosetsideSimulator,
        params: LeakageConditioningParams,
    ) -> None:
        condition_groups = params.args
        other_targets = params.targets
        if other_targets:
            # stim fuses k identical consecutive CONDITIONED_ON_OTHER lines into one op with k times
            # the targets; each copy reuses the tag's controls.
            copy_len = len(other_targets)
            copies = split_fused_instruction(op, copy_len)
            if len(copies) > 1 and not stim.gate_data(op.name).is_unitary:
                # Noise has a no-op reference step, so loop (RNG draws as for k separate ops).
                tss.do_copies_on_fused_step(
                    op, copies, lambda sub_op: self.leakage_conditioning(sub_op, tss, params)
                )
                return
            # A unitary's reference step is the fused op, so tile the controls instead of looping.
            other_targets = other_targets * len(copies)
        gd = stim.gate_data(op.name)
        raw_targets = op.targets_copy()
        if gd.is_unitary and any(0 in g or 1 in g for g in condition_groups):
            t_groups = op.target_groups()
            if len(t_groups) > 1:
                all_qs = [
                    t.qubit_value
                    for grp in t_groups
                    for t in grp
                    if t.qubit_value is not None
                ]
                if other_targets:
                    all_qs.extend(other_targets)
                if len(set(all_qs)) < len(all_qs):
                    def _eval_group_cond(g_idx: int) -> NDArray[np.bool_]:
                        grp = t_groups[g_idx]
                        if other_targets:
                            step_ot = len(other_targets) // len(t_groups)
                            sub_ot = list(
                                other_targets[
                                    g_idx * step_ot : (g_idx + 1) * step_ot
                                ]
                            )
                            return self._filter_targets_1q_mask(
                                sub_ot,
                                cast(
                                    tuple[tuple[int | str, ...], ...],
                                    condition_groups,
                                ),
                                tss,
                            )[0]
                        if len(condition_groups) == 1:
                            sub_qs = [
                                t.qubit_value if t.qubit_value is not None else -1
                                for t in grp
                            ]
                            return self._filter_targets_1q_mask(
                                sub_qs, condition_groups, tss
                            )[0]
                        pair = [
                            [
                                grp[0].qubit_value
                                if grp[0].qubit_value is not None
                                else -1,
                                grp[1].qubit_value
                                if grp[1].qubit_value is not None
                                else -1,
                            ]
                        ]
                        return self._filter_targets_2q_mask(
                            pair,
                            cast(
                                tuple[
                                    tuple[int | str, ...],
                                    tuple[int | str, ...],
                                ],
                                condition_groups,
                            ),
                            tss,
                        )[0]

                    tss.do_conditional_clifford(
                        op, condition_group_fn=_eval_group_cond
                    )
                    return
        if other_targets:
            if len(other_targets) != len(raw_targets):
                raise ValueError(
                    f"The number of targets of the {op} is not the same as the "
                    "number of targets specified in the CONDITIONED_ON_OTHER tag."
                )
            cond_mask = self._filter_targets_1q_mask(
                list(other_targets),
                cast(tuple[tuple[int | str, ...], ...], condition_groups),
                tss,
            )
        else:
            if len(condition_groups) == 1:
                targets = [
                    t.qubit_value if t.qubit_value is not None else -1
                    for t in raw_targets
                ]
                cond_mask = self._filter_targets_1q_mask(
                    targets, condition_groups, tss
                )
            else:
                target_pairs = [
                    [
                        grp[0].qubit_value
                        if grp[0].qubit_value is not None
                        else -1,
                        grp[1].qubit_value
                        if grp[1].qubit_value is not None
                        else -1,
                    ]
                    for grp in op.target_groups()
                ]
                cond_mask = self._filter_targets_2q_mask(
                    target_pairs,
                    cast(
                        tuple[tuple[int | str, ...], tuple[int | str, ...]],
                        condition_groups,
                    ),
                    tss,
                )

        if gd.is_unitary:
            tss.do_conditional_clifford(op, cond_mask)
        else:
            tss.do_conditional_pauli_noise(op, cond_mask)

    def _tab_flips(
        self, tss: CosetsideSimulator, b: int, qs: Sequence[int] | NDArray[np.int_], p: float
    ) -> NDArray[np.bool_]:
        q_list = [int(q) for q in qs]
        flips = np.zeros(len(q_list), dtype=np.bool_)
        if q_list and p > 0.0:
            assert tss._tab_rngs is not None and tss._tab_rngs_x is not None
            tr = tss._tab_rngs[b]
            inst = stim.CircuitInstruction("X_ERROR", q_list, [p])
            tr.do(inst)
            tss._tab_rngs_x[b].do(inst)
            for i, q in enumerate(q_list):
                if tr.peek_z(q) == -1:
                    tr.x(q)
                    flips[i] = True
        return flips

    @staticmethod
    def _targets_per_shot(mask: NDArray[np.bool_]) -> list[list[np.intp]]:
        return [list(np.where(mask[:, b])[0]) for b in range(mask.shape[1])]

    def _apply_transition_effects(
        self, tss: CosetsideSimulator, effects: TransitionEffectMasks
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
            if tss._tab_rngs is not None:
                tss._sample_pauli_noise_via_tab_rng(
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
                if tss._tab_rngs is not None:
                    tss._sample_pauli_noise_via_tab_rng(
                        gate, self._targets_per_shot(arr), [0.5]
                    )
                else:
                    tss.apply_pauli(pauli, arr, 0.5)

    def leakage_transition_1(
        self,
        op: stim.CircuitInstruction,
        tss: CosetsideSimulator,
        params: LeakageTransition1Params,
        *,
        _advance_step: bool = True,
    ) -> None:
        raw_t = op.targets_copy()
        target_indices = [
            t.qubit_value if t.qubit_value is not None else t.value
            for t in raw_t
        ]
        if _advance_step:
            tss._next_step_meta(op)
        if len(target_indices) > 1 and len(set(target_indices)) < len(
            target_indices
        ):
            for t in raw_t:
                sub_op = stim.CircuitInstruction(
                    op.name, [t], op.gate_args_copy()
                )
                self.leakage_transition_1(
                    sub_op, tss, params, _advance_step=False
                )
            return
        sorted_target_indices = np.array(sorted(set(target_indices)), dtype=np.intp)

        to_depolarize = self._scratch_depolarize
        to_set_to_one = self._scratch_set_one
        to_set_to_zero = self._scratch_set_zero
        to_depolarize.fill(False)
        to_set_to_one.fill(False)
        to_set_to_zero.fill(False)
        effects = TransitionEffectMasks(
            set_zero=to_set_to_zero, set_one=to_set_to_one, depolarize=to_depolarize
        )

        peeked_01 = any(k in (0, 1) for k in params.args_by_input_state.keys())
        if peeked_01:
            self._update_state_from_peek_Z(sorted_target_indices, tss)
        target_states = self._state[sorted_target_indices, :].copy()

        if tss._shot_np_rngs is not None or (
            tss.batch_size == 1 and tss.seed is not None
        ):
            for input_state in params.args_by_input_state.keys():
                for b in range(tss.batch_size):
                    in_mask_b = (
                        target_states[:, b] < 2
                        if input_state == "U"
                        else target_states[:, b] == input_state
                    )
                    ow_qs = sorted_target_indices[in_mask_b]
                    if len(ow_qs) == 0:
                        continue
                    rng_b = tss._shot_np_rngs[b] if tss._shot_np_rngs is not None else tss.np_rng
                    out_b = params.sample_transitions_from_state(
                        input_state=input_state, num_samples=len(ow_qs), np_rng=rng_b
                    ).astype(str)
                    for q_np, out_st in zip(ow_qs, out_b):
                        apply_output_effect(self._state, effects, (int(q_np), b), input_state, out_st)
        else:
            for input_state, transitions in params.args_by_input_state.items():
                total_p = float(sum(p for _, p in transitions))
                if total_p <= 0.0:
                    continue
                input_state_mask = (
                    target_states < 2 if input_state == "U" else target_states == input_state
                )
                for out_s, r_sub, b_sub in sample_batched_group_transitions(
                    transitions, total_p, tss.np_rng, input_state_mask
                ):
                    if len(r_sub) > 0 and str(out_s) != str(input_state):
                        apply_output_effect(
                            self._state, effects, (sorted_target_indices[r_sub], b_sub), input_state, out_s
                        )

        if peeked_01:
            self._clear_transient_ones(sorted_target_indices)
        self._apply_transition_effects(tss, effects)

    def leakage_transition_2(
        self,
        op: stim.CircuitInstruction,
        tss: CosetsideSimulator,
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

        if tss._shot_np_rngs is not None or (
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
                    rng_b = tss._shot_np_rngs[b] if tss._shot_np_rngs is not None else tss.np_rng
                    out_b = params.sample_transitions_from_state(
                        input_state=input_state, num_samples=samples_b, np_rng=rng_b
                    ).astype(str)
                    for pair, out_pair in zip(trig_pairs_b, out_b):
                        for leg in (0, 1):
                            apply_output_effect(
                                self._state,
                                effects,
                                (int(pair[leg]), b),
                                input_state[leg],
                                out_pair[leg],
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
                            self._state, effects, (q_sub, b_sub), in_s, out_s
                        )

        if peeked_2q:
            self._clear_transient_ones(peeked_2q)
        self._apply_transition_effects(tss, effects)

    def _sample_leaked_tab_outcomes(
        self,
        tss: CosetsideSimulator,
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
        tss: CosetsideSimulator,
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
                m_col = tss._measurement_columns[-num_m + k_i]
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
                    m_col[:] = new_col
                    leaked_diff = diff & (target_leakage_state[k_i] > 1)
                    if np.any(leaked_diff):
                        net_x_flip[q, :] ^= leaked_diff
            if np.any(net_x_flip):
                tss.apply_pauli("Z" if basis == "X" else "X", net_x_flip)
        elif replace_mask is None:
            for k_i in range(num_m):
                m_col = tss._measurement_columns[-num_m + k_i]
                target_out = outcomes[k_i] ^ inv_flags[k_i]
                if np.any(m_col ^ target_out):
                    m_col[:] = target_out
        if has_leaked:
            leaked_per_shot = [
                list(targets_arr[target_leakage_state[:, b] > 1])
                for b in range(tss.batch_size)
            ]
            tss._sample_pauli_noise_via_tab_rng("DEPOLARIZE1", leaked_per_shot, [0.75])

    def leakage_measurement(
        self,
        op: stim.CircuitInstruction,
        tss: CosetsideSimulator,
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

        if not is_mpad and tss._tab_rngs is not None and len(set(targets)) < len(targets):
            if p0 > 0.0 or p1 < 1.0 or has_leaked:
                raise NotImplementedError(
                    f"Duplicate targets in {op.name} with LEAKAGE_PROJECTION_Z are not supported with "
                    "sync_tableside_rng=True when readout is noisy or any target is leaked."
                )

        # Fast-path for LEAKAGE_PROJECTION_Z measurements with noiseless computational readout (p0=0, p1=1) and sync_tableside_rng
        if not is_mpad and tss._tab_rngs is not None and p0 == 0.0 and p1 == 1.0:
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
        if is_mpad and has_01_cond and tss._tab_rngs is None:
            self._update_state_from_peek_Z(targets, tss)
            self._clear_transient_ones(targets)

        need_pauli_peek = tss._tab_rngs is not None or is_mpad or has_01_cond
        pauli_states = (
            tss.peek_pauli_batch(targets, basis)
            if need_pauli_peek
            else np.zeros((len(targets), tss.batch_size), dtype=np.int8)
        )

        if tss._tab_rngs is not None:
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
                            bit = int(tss._shot_np_rngs[b].binomial(1, 0.5))
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
            if has_leaked:
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
        for n in range(2, int(np.max(target_leakage_state)) + 1):
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
            m_col = tss._measurement_columns[-num_m + k_i]
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
                m_col = tss._measurement_columns[-num_m + k_i]
                m_col[:] = np.where(mask_k, sampled_outcomes[k_i] ^ inv_flags[k_i], m_col)
        if has_leaked:
            to_dep = self._scratch_depolarize
            to_dep.fill(False)
            to_dep[targets_arr, :] = leaked_mask
            tss.apply_pauli("X", to_dep, 0.5)
            tss.apply_pauli("Z", to_dep, 0.5)
