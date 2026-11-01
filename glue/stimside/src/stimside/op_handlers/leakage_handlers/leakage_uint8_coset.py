from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageParams,
    LeakageSwapParams,
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
    sample_coset_batched_groups,
)
from stimside.util.stim_workarounds import broadcast_pauli_errors, split_fused_instruction

if TYPE_CHECKING:
    from stimside.simulator_coset import CosetsideSimulator


def _do_copies_on_fused_step(
    tss: CosetsideSimulator,
    op: stim.CircuitInstruction,
    copies: list[stim.CircuitInstruction],
    do_copy: Callable[[stim.CircuitInstruction], None],
) -> None:
    """Apply the copies of a stim-fused op in sequence on the op's one (no-op) reference step."""
    tss._next_step_meta(op)  # matched (or, interactively, appended) for the fused op
    step_cursor = tss._step_cursor - 1
    used_interactively, tss._used_interactively = tss._used_interactively, False
    try:
        for sub_op in copies:
            tss._step_cursor = step_cursor  # each copy re-reads the fused op's step
            do_copy(sub_op)
    finally:
        tss._used_interactively = used_interactively
    tss._step_cursor = step_cursor + 1


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
        targets_arr = np.asarray(targets, dtype=np.intp)
        if len(targets_arr) == 0:
            return

        unique_targets = list(dict.fromkeys(int(t) for t in targets_arr if t >= 0))
        if len(unique_targets) == 0:
            return

        z_exps = css.peek_pauli_batch(unique_targets, "Z")
        any_projected = False
        for idx, q in enumerate(unique_targets):
            comp_mask = self._state[q, :] < 2
            if not np.any(comp_mask):
                continue
            z_row = (
                css.peek_pauli_batch([q], "Z")[0]
                if any_projected
                else z_exps[idx]
            )
            new_row = np.zeros(css.batch_size, dtype=np.uint8)
            new_row[z_row == -1] = 1
            flip_mask = comp_mask & (z_row == 0)
            num_flips = int(np.count_nonzero(flip_mask))
            if num_flips > 0:
                if css._shot_np_rngs is not None:
                    for b_np in np.flatnonzero(flip_mask):
                        b_idx = int(b_np)
                        new_row[b_idx] = int(
                            css._shot_np_rngs[b_idx].binomial(1, 0.5)
                        )
                else:
                    new_row[flip_mask] = css.np_rng.binomial(
                        1, 0.5, size=num_flips
                    ).astype(np.uint8)
                css.postselect_z_batch(q, flip_mask, new_row)
                any_projected = True
            self._state[q, comp_mask] = new_row[comp_mask]

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
        if len(valid_qs) > 0 and (
            1 in allowed_group or 0 in allowed_group
        ):
            self._update_state_from_peek_Z(valid_qs, css)

        safe_qs = np.where(targets_arr >= 0, targets_arr, 0)
        target_states = self._state[safe_qs, :].copy()  # (len(targets), batch_size)
        is_classical = (targets_arr < 0)[:, None]
        target_states = np.where(is_classical, 0, target_states)
        self._state[self._state == 1] = 0

        allowed_mask = np.zeros(
            (len(targets_arr), css.batch_size), dtype=np.bool_
        )
        if "U" in allowed_group:
            allowed_mask |= is_classical | (target_states < 2)
        for st in allowed_group:
            if isinstance(st, int):
                if st < 2:
                    allowed_mask |= is_classical | (target_states == st)
                else:
                    allowed_mask |= (~is_classical) & (target_states == st)
            elif st != "U":
                raise ValueError(
                    f"Unrecognised allowed state {st} in allowed_st_list {allowed_st_list}"
                )
        return allowed_mask

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
        for i in (0, 1):
            if 1 in allowed_st_list[i] or 0 in allowed_st_list[i]:
                valid_qs = targets_np[targets_np[:, i] >= 0, i]
                if len(valid_qs) > 0:
                    self._update_state_from_peek_Z(valid_qs, css)

        is_c0 = (targets_np[:, 0] < 0)[:, None]
        is_c1 = (targets_np[:, 1] < 0)[:, None]
        q0_safe = np.where(targets_np[:, 0] >= 0, targets_np[:, 0], 0)
        q1_safe = np.where(targets_np[:, 1] >= 0, targets_np[:, 1], 0)
        s0 = np.where(is_c0, 0, self._state[q0_safe, :])
        s1 = np.where(is_c1, 0, self._state[q1_safe, :])
        self._state[self._state == 1] = 0
        allowed_mask = np.zeros((len(targets_np), css.batch_size), dtype=np.bool_)

        for n in range(len(allowed_st_list[0])):
            st0 = allowed_st_list[0][n]
            st1 = allowed_st_list[1][n]
            if st0 == "U":
                m0 = is_c0 | (s0 < 2)
            elif isinstance(st0, int):
                m0 = (is_c0 & (st0 < 2)) | ((~is_c0) & (s0 == st0))
            else:
                raise ValueError(
                    f"Unrecognised allowed state {st0} in allowed_st_list {allowed_st_list}"
                )

            if st1 == "U":
                m1 = is_c1 | (s1 < 2)
            elif isinstance(st1, int):
                m1 = (is_c1 & (st1 < 2)) | ((~is_c1) & (s1 == st1))
            else:
                raise ValueError(
                    f"Unrecognised allowed state {st1} in allowed_st_list {allowed_st_list}"
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

                if gd.is_single_qubit_gate or op_name in (
                    "E",
                    "CORRELATED_ERROR",
                    "ELSE_CORRELATED_ERROR",
                ):
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
                elif op_name in ("SPP", "SPP_DAG"):
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
                else:
                    sss._do_bare_instruction(op)
                    return

                if gd.is_unitary:
                    sss.do_conditional_clifford(op, cond_mask)
                else:
                    sss.do_conditional_pauli_noise(op, cond_mask)
                return

            sss._do_bare_instruction(op)
            return

        params = self.ops_to_params[op]
        is_transition = isinstance(
            params, (LeakageTransition1Params, LeakageTransition2Params, LeakageSwapParams)
        )
        rec_ev = is_transition and getattr(sss, "record_leakage_events", False)
        if (sss.record_unleaked_to_leaked or rec_ev) and is_transition:
            self._ensure_batch_size(sss.batch_size)
            target_indices = np.unique(
                [
                    t.qubit_value if t.qubit_value is not None else t.value
                    for t in op.targets_copy()
                    if (t.qubit_value if t.qubit_value is not None else t.value) >= 0
                ]
            )
            was_unleaked = self._state[target_indices, :] < 2
            old_states = self._state[target_indices, :].copy() if rec_ev else None
        else:
            target_indices = None
            was_unleaked = None
            old_states = None

        match params:
            case LeakageConditioningParams():
                self.leakage_conditioning(op=op, tss=sss, params=params)
            case LeakageTransition1Params():
                self.leakage_transition_1(op=op, tss=sss, params=params)
            case LeakageTransition2Params():
                self.leakage_transition_2(op=op, tss=sss, params=params)
            case LeakageMeasurementParams():
                self.leakage_measurement(op=op, tss=sss, params=params)
            case LeakageSwapParams():
                self.leakage_swap(op=op, tss=sss)
            case _:
                raise ValueError(f"Unrecognised LEAKAGE params: {params}")

        if (
            sss.record_unleaked_to_leaked
            and was_unleaked is not None
            and target_indices is not None
        ):
            now_leaked = self._state[target_indices, :] >= 2
            counts = np.count_nonzero(was_unleaked & now_leaked, axis=0)
            sss._record_unleaked_to_leaked_counts(counts, sss._circuit_time)
        if old_states is not None and target_indices is not None:
            new_states = self._state[target_indices, :]
            for i_q, b in zip(*np.nonzero(old_states != new_states)):
                sss._record_leakage_event(
                    sss._circuit_time,
                    target_indices[i_q],
                    old_states[i_q, b],
                    new_states[i_q, b],
                    shot_idx=int(b),
                )

    def leakage_swap(self, op: stim.CircuitInstruction, tss: CosetsideSimulator) -> None:
        """SWAP[LEAKAGE_SWAP]: SWAP the qubits in every shot, leaked ones included (their frozen
        computational states move with them), then swap their leakage states, pair by pair from
        left to right.
        """
        tss._do_bare_instruction(
            stim.CircuitInstruction(op.name, op.targets_copy(), op.gate_args_copy())
        )
        targets = [t.qubit_value for t in op.targets_copy()]
        for q0, q1 in zip(targets[::2], targets[1::2]):
            self._state[[q0, q1], :] = self._state[[q1, q0], :]

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
                _do_copies_on_fused_step(
                    tss, op, copies, lambda sub_op: self.leakage_conditioning(sub_op, tss, params)
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
                broadcast_pauli_errors(
                    tss._flip_simulator, pauli="X", mask=effects.depolarize, p=0.5, np_rng=tss.np_rng
                )
                broadcast_pauli_errors(
                    tss._flip_simulator, pauli="Z", mask=effects.depolarize, p=0.5, np_rng=tss.np_rng
                )
        for pauli, arr in (
            ("X", effects.pauli_x),
            ("Y", effects.pauli_y),
            ("Z", effects.pauli_z),
        ):
            if arr is not None and np.any(arr):
                tss._flip_simulator.broadcast_pauli_errors(pauli=pauli, mask=arr, p=1.0)
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
                    broadcast_pauli_errors(
                        tss._flip_simulator, pauli=pauli, mask=arr, p=0.5, np_rng=tss.np_rng
                    )

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
        target_mask_1d = np.zeros(self.num_qubits, dtype=np.bool_)
        target_mask_1d[target_indices] = True
        sorted_target_indices = np.where(target_mask_1d)[0]

        to_depolarize = self._scratch_depolarize
        to_set_to_one = self._scratch_set_one
        to_set_to_zero = self._scratch_set_zero
        to_depolarize.fill(False)
        to_set_to_one.fill(False)
        to_set_to_zero.fill(False)
        effects = TransitionEffectMasks(
            set_zero=to_set_to_zero, set_one=to_set_to_one, depolarize=to_depolarize
        )

        if any(k in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(sorted_target_indices, tss)
        initial_state = self._state.copy()

        if tss._shot_np_rngs is not None or (
            tss.batch_size == 1 and tss.seed is not None
        ):
            for input_state in params.args_by_input_state.keys():
                for b in range(tss.batch_size):
                    in_mask_b = (
                        initial_state[:, b] < 2
                        if input_state == "U"
                        else initial_state[:, b] == input_state
                    )
                    ow_qs = np.where(target_mask_1d & in_mask_b)[0]
                    if len(ow_qs) == 0:
                        continue
                    rng_b = tss._shot_np_rngs[b] if tss._shot_np_rngs is not None else tss.np_rng
                    out_b = params.sample_transitions_from_state(
                        input_state=input_state, num_samples=len(ow_qs), np_rng=rng_b
                    ).astype(str)
                    for q_np, out_st in zip(ow_qs, out_b):
                        apply_output_effect(self._state, effects, (int(q_np), b), input_state, out_st)
        else:
            target_mask = target_mask_1d[:, None]
            for input_state, transitions in params.args_by_input_state.items():
                total_p = float(sum(p for _, p in transitions))
                if total_p <= 0.0:
                    continue
                input_state_mask = (
                    initial_state < 2 if input_state == "U" else initial_state == input_state
                )
                for out_s, q_sub, b_sub in sample_coset_batched_groups(
                    transitions, target_mask & input_state_mask, total_p, tss.np_rng
                ):
                    if len(q_sub) > 0 and str(out_s) != str(input_state):
                        apply_output_effect(self._state, effects, (q_sub, b_sub), input_state, out_s)

        self._state[self._state == 1] = 0
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

        if any(k[0] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(even_targets_list, tss)
        if any(k[1] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(odd_targets_list, tss)

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
                for out_pair, p_sub, b_sub in sample_coset_batched_groups(
                    transitions, even_checks & odd_checks, total_p, tss.np_rng
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

        self._state[self._state == 1] = 0
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
                tss._flip_simulator.broadcast_pauli_errors(
                    pauli=("Z" if basis == "X" else "X"), mask=net_x_flip, p=1.0
                )
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
                _do_copies_on_fused_step(
                    tss, op, copies, lambda sub_op: self.leakage_measurement(sub_op, tss, params)
                )
                return
        raw_op_targets = op.targets_copy()
        if params.targets:
            targets = list(params.targets)
            inv_flags = [
                bool(t.value) ^ bool(t.is_inverted_result_target)
                for t in raw_op_targets
            ]
            if len(raw_op_targets) != len(targets):
                raise ValueError(
                    "The number of targets in the MPAD operation with a LEAKAGE_MEASUREMENT tag "
                    "does not equal to the number of targets specified in the tag."
                )
        else:
            targets = [t.qubit_value for t in raw_op_targets]
            inv_flags = [bool(t.is_inverted_result_target) for t in raw_op_targets]

        target_leakage_state = self._state[targets, :].copy()  # (len(targets), batch_size)
        targets_arr = np.asarray(targets, dtype=np.intp)
        is_mpad = params.targets is not None
        p0 = params.prob_for_input_state.get(0, 0.0)
        p1 = params.prob_for_input_state.get(1, 0.0 if is_mpad else 1.0)
        p_super = 0.5 * p0 + 0.5 * p1
        basis: Literal["X", "Y", "Z"] = (
            "X"
            if op.name in ("MX", "MRX")
            else ("Y" if op.name in ("MY", "MRY") else "Z")
        )
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
            self._state[self._state == 1] = 0

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
            broadcast_pauli_errors(
                tss._flip_simulator, pauli="X", mask=to_dep, p=0.5, np_rng=tss.np_rng
            )
            broadcast_pauli_errors(
                tss._flip_simulator, pauli="Z", mask=to_dep, p=0.5, np_rng=tss.np_rng
            )
