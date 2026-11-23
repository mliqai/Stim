from __future__ import annotations

import dataclasses
import types
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
    LeakageConditioningParams,
    LeakageMeasurementParams,
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
    emit_tableau_transition_effects,
    eval_allowed_state_mask,
    exec_leakage_swap,
    qualify_fast_1q,
    sample_tableau_transitions,
    unpack_leakage_measurement_targets,
)
from stimside.util.stim_workarounds import split_fused_instruction

if TYPE_CHECKING:
    from stimside.simulator_tableau import TablesideSimulator

# Instructions that never change a qubit's state; the interactive Z-determinism pre-filter skips them.
_Z_SCAN_SKIP = frozenset(
    {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"}
)
# Names that Python (3.10-3.14), abc, typing and dataclasses put in a class body; no behaviour.
_CLASS_BOOKKEEPING = frozenset(
    {
        "__module__", "__doc__", "__firstlineno__", "__static_attributes__", "__annotations__",
        "__annotate_func__", "__annotations_cache__", "__parameters__", "__orig_bases__",
        "__abstractmethods__", "_abc_impl", "__slots__", "__dict__", "__weakref__", "__slotnames__",
        "__dataclass_fields__", "__dataclass_params__", "__match_args__", "__type_params__",
    }
)


def _is_plain_member(name: str, value: object, is_field: bool) -> bool:
    """Whether a class-body name is bookkeeping, the base handler's own object, a method that
    dataclasses generated, or a new dataclass field with a plain-data default (or a slot)."""
    base = CompiledLeakageUint8Tableau
    if name in _CLASS_BOOKKEEPING or value is getattr(base, name, object()):
        return True
    if name == "__repr__":  # dataclasses wraps it in reprlib.recursive_repr
        value = getattr(value, "__wrapped__", value)
    code = getattr(value, "__code__", None)
    if code is not None:  # dataclasses exec()s __init__, __eq__, ... (co_qualname: Python 3.11+)
        qualname = getattr(code, "co_qualname", "__create_fn__.<locals>.")
        return (
            name[:2] == name[-2:] == "__"  # every generated method is a dunder (needed on 3.10)
            and code.co_filename == "<string>"
            and qualname.startswith("__create_fn__.<locals>.")
        )
    descriptor = any(hasattr(type(value), a) for a in ("__get__", "__set__", "__delete__"))
    return (
        is_field
        and not hasattr(base, name)
        and (isinstance(value, types.MemberDescriptorType) or not descriptor)
    )


@dataclasses.dataclass
class LeakageUint8Tableau(OpHandler["TablesideSimulator"]):
    """Implementing leakage using an array of 8bit Uints.

    Basically, we're straight up storing the leakage state for each qubit.
    Pro: hard to mess up
        Supports leakage states up to 9 (tags with higher states are rejected)
        Easy to store and work with using numpy
    Con: probably not as fast as using a bool array for each leakage level we care about
        We spend a reasonable amount of time computing masks like state==2

    We leave state == 0 and 1 alone so as to not be confusing. 2 means 2.

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
    ) -> "CompiledLeakageUint8Tableau":
        parsed_ops = parse_leakage_in_circuit(circuit)
        check_tag_qubits(parsed_ops, circuit.num_qubits)
        fast_1q, compiled_len = qualify_fast_1q(
            circuit, parsed_ops, self.unconditional_condition_on_U, self.fast_leakage
        )
        compiled = CompiledLeakageUint8Tableau(
            num_qubits=circuit.num_qubits, batch_size=batch_size, ops_to_params=parsed_ops,
            unconditional_condition_on_U=self.unconditional_condition_on_U,
            fast_leakage=self.fast_leakage,
        )
        compiled._fast_1q, compiled._compiled_len = fast_1q, compiled_len
        return compiled


@dataclasses.dataclass
class CompiledLeakageUint8Tableau(CompiledOpHandler["TablesideSimulator"]):

    num_qubits: int
    batch_size: int
    unconditional_condition_on_U: bool

    state: NDArray[np.uint8] = dataclasses.field(init=False)

    ops_to_params: dict[stim.CircuitInstruction, LeakageParams]
    # Keyword-only, so that positional calls keep the parameter order they had before this option.
    fast_leakage: bool = dataclasses.field(default=False, kw_only=True)
    # Set by compile_op_handler from the circuit it compiles; not constructor parameters.
    _fast_1q: bool = dataclasses.field(default=False, init=False)
    _compiled_len: int = dataclasses.field(default=0, init=False)

    def __post_init__(self):
        self.claimed_ops_keys = set(self.ops_to_params.keys())
        self.state = np.zeros(self.num_qubits, dtype=np.uint8)
        self._reset_z_scan()

    def clear(self):
        """Clear the leakage state array in place (preserving any C++ shared buffer binding)."""
        if self.state is not None and len(self.state) == self.num_qubits:
            self.state.fill(0)
        else:
            self.state = np.zeros(self.num_qubits, dtype=np.uint8)
        self._reset_z_scan()

    def _fast_path_overrides(self) -> list[str]:
        """Every name that this handler's class (or a class it mixes in) defines beyond plain
        dataclass fields, every instance attribute of a subclass that shadows a base member, plus
        "claimed_ops_keys" if it differs from ops_to_params (the runners read only the latter):
        the precompiled runners would silently skip any of these."""
        base = CompiledLeakageUint8Tableau
        base_mro = base.__mro__
        names = [
            n
            for k in type(self).__mro__
            if k not in base_mro
            for n, v in vars(k).items()
            if not _is_plain_member(n, v, n in vars(k).get("__dataclass_fields__", ()))
        ]
        if type(self) is not base:  # the base type stays exempt (as at HEAD)
            names += [
                n for n in vars(self) if hasattr(base, n) and n not in base.__dataclass_fields__
            ]
        if self.claimed_ops_keys != self.ops_to_params.keys():
            names.append("claimed_ops_keys")
        return list(dict.fromkeys(names))

    def supports_cpp_fast_path(self) -> bool:
        """True iff the precompiled (C++ and sync) runners would not skip any handler behaviour."""
        return not self._fast_path_overrides()

    def _reset_z_scan(self) -> None:
        self._z_scan_cursor = 0
        # True: no instruction has touched the qubit since its last R/MR (or the start).
        self._z_scan_fresh = np.ones(self.num_qubits, dtype=np.bool_)

    def _z_exact_without_dem_check(
        self, targets: NDArray[np.int_], tss: TablesideSimulator
    ) -> bool:
        """Whether every target's Z is known to be identical in all shots of the batch.

        A qubit's state only changes through instructions that target it, and R/MR leave
        it in |0> in every shot, so a target untouched since its last R/MR has a
        deterministic Z. Scans tss._new_circuit incrementally from the last call.
        """
        circ = tss._new_circuit
        n = len(circ)
        if n < self._z_scan_cursor:
            self._reset_z_scan()
        fresh = self._z_scan_fresh
        # Re-scan the last scanned instruction too: stim may have fused new targets into it.
        for k in range(max(self._z_scan_cursor - 1, 0), n):
            inst = circ[k]
            if isinstance(inst, stim.CircuitRepeatBlock):
                fresh[:] = False
                continue
            if inst.name in _Z_SCAN_SKIP:
                continue
            is_reset = inst.name in ("R", "MR")
            for t in inst.targets_copy():
                q = t.qubit_value
                if q is not None and q < len(fresh):
                    fresh[q] = is_reset
        self._z_scan_cursor = n
        return bool(np.all(fresh[targets]))

    def _update_state_from_peek_Z(
        self, targets: list[int] | NDArray[np.int_], tss: TablesideSimulator
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Update self.state for the targets based on the current Z pauli states.
        If a target is in a superposition, project its Z state in the tableau simulator
        to preserve multi-qubit stabilizer entanglement.
        """
        targets_arr = np.asarray(targets, dtype=np.intp)
        if len(targets_arr) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        _, unique_idx = np.unique(targets_arr, return_index=True)
        targets_arr = targets_arr[np.sort(unique_idx)]
        targets_in_qubit_space = targets_arr[self.state[targets_arr] < 2]
        pauli_states = np.empty(len(targets_in_qubit_space), dtype=np.intp)
        new_state = np.zeros(len(targets_in_qubit_space), dtype=np.uint8)

        for idx, q in enumerate(targets_in_qubit_space):
            pz = tss.peek_z(int(q))
            pauli_states[idx] = pz
            if pz == -1:
                new_state[idx] = 1

        superpos_indices = np.where(pauli_states == 0)[0]
        if tss.batch_size > 1 and len(targets_in_qubit_space) > 0:
            if len(superpos_indices) > 0:
                raise ValueError(
                    "interactive_do with batch_size > 1 cannot branch or collapse Z superposition "
                    "on 0/1-conditioned tags; use run() or batch_size=1."
                )
            if not self._z_exact_without_dem_check(targets_in_qubit_space, tss) and any(
                stim.gate_data(inst.name).is_noisy_gate
                or stim.gate_data(inst.name).produces_measurements
                for inst in tss._new_circuit
            ):
                # Real probabilities: any p in (0, 1) still gives a DEM error in (0, 1).
                chk = tss._new_circuit.copy()
                if chk.num_observables:
                    # Observables do not affect this check, but stim rejects non-deterministic ones.
                    chk = stim.Circuit()
                    for inst in tss._new_circuit:
                        if inst.name != "OBSERVABLE_INCLUDE":
                            chk.append(inst)
                d0 = chk.num_detectors
                n_chk = len(targets_in_qubit_space)
                chk.append("M", [int(q) for q in targets_in_qubit_space])
                for k in range(n_chk, 0, -1):
                    chk.append("DETECTOR", [stim.target_rec(-k)])
                try:
                    dem = chk.detector_error_model(
                        allow_gauge_detectors=True,
                        approximate_disjoint_errors=True,
                    )
                except ValueError:
                    raise ValueError(
                        "interactive_do with batch_size > 1 cannot apply 0/1-conditioned tags "
                        "to qubits with non-deterministic Z across the batch; use run() or batch_size=1."
                    ) from None
                if any(
                    inst.type == "error"
                    and 0.0 < inst.args_copy()[0] < 1.0
                    and any(
                        t.is_relative_detector_id() and t.val >= d0
                        for t in inst.targets_copy()
                    )
                    for inst in dem.flattened()
                ):
                    raise ValueError(
                        "interactive_do with batch_size > 1 cannot apply 0/1-conditioned tags "
                        "to qubits with non-deterministic Z across the batch; use run() or batch_size=1."
                    )

        for s_i, idx in enumerate(superpos_indices):
            q_int = int(targets_in_qubit_space[idx])
            pz = 0 if s_i == 0 else tss.peek_z(q_int)
            pauli_states[idx] = pz
            if pz == -1:
                new_state[idx] = 1
            elif pz == 0:
                bit = int(tss.np_rng.binomial(1, 0.5))
                new_state[idx] = bit
                tss._tableau_simulator.postselect_z(q_int, desired_value=bool(bit))

        self.state[targets_in_qubit_space] = new_state
        return targets_in_qubit_space, pauli_states

    def _filter_targets_1q_mask(
        self,
        targets: list[int] | NDArray[np.int_],
        allowed_st_list: tuple[tuple[int | str, ...]],
        tss: TablesideSimulator,
    ) -> NDArray[np.bool_]:
        """
        Filter single qubit targets based on allowed states.
        If 0 or 1 is in allowed_st_list, update the state from the current Z pauli states.
        Return only the targets that are in the allowed states.
        """
        if allowed_st_list == (("U",),) and not np.any(self.state >= 2):
            return np.ones(len(targets), dtype=np.bool_)

        targets_arr = np.asarray(targets, dtype=np.intp)
        valid_qs = targets_arr[targets_arr >= 0]
        if len(valid_qs) > 0 and (
            1 in allowed_st_list[0] or 0 in allowed_st_list[0]
        ):
            self._update_state_from_peek_Z(valid_qs, tss)

        safe_qs = np.where(targets_arr >= 0, targets_arr, 0)
        target_states = np.where(targets_arr < 0, 0, self.state[safe_qs])
        self.state[self.state == 1] = 0
        return eval_allowed_state_mask(
            target_states, targets_arr < 0, allowed_st_list[0]
        )

    def _filter_targets_2q(
        self,
        targets: list[list[stim.GateTarget]],
        allowed_st_list: tuple[tuple[int | str, ...], tuple[int | str, ...]],
        tss: TablesideSimulator,
    ) -> list[stim.GateTarget]:
        """
        Filter two-qubit op targets based on allowed states.
        If 0 or 1 is in allowed_st_list, update the state from the current Z pauli states.
        Return only the 2-qubit targets that are in the allowed states.
        """
        if len(targets) == 0:
            return []
        num_pairs = len(targets)
        q_pairs: list[tuple[int | None, int | None]] = [
            (pair[0].qubit_value, pair[1].qubit_value) for pair in targets
        ]

        qs_to_peek: list[int] = []
        for i in [0, 1]:
            if 1 in allowed_st_list[i] or 0 in allowed_st_list[i]:
                qs_to_peek.extend(
                    p[i] for p in q_pairs if p[i] is not None and p[i] >= 0
                )
        if len(qs_to_peek) > 0:
            self._update_state_from_peek_Z(qs_to_peek, tss)

        target_states = np.zeros((num_pairs, 2), dtype=np.intp)
        is_classical = np.zeros((num_pairs, 2), dtype=bool)
        for p_idx, (q0, q1) in enumerate(q_pairs):
            if q0 is None or q0 < 0:
                is_classical[p_idx, 0] = True
            else:
                target_states[p_idx, 0] = self.state[q0]
            if q1 is None or q1 < 0:
                is_classical[p_idx, 1] = True
            else:
                target_states[p_idx, 1] = self.state[q1]
        self.state[self.state == 1] = 0
        allowed_mask = np.zeros(num_pairs, dtype=bool)
        for n in range(len(allowed_st_list[0])):
            m0 = eval_allowed_state_mask(
                target_states[:, 0], is_classical[:, 0], (allowed_st_list[0][n],)
            )
            m1 = eval_allowed_state_mask(
                target_states[:, 1], is_classical[:, 1], (allowed_st_list[1][n],)
            )
            allowed_mask |= m0 & m1

        return [
            t
            for pair, keep in zip(targets, allowed_mask)
            if keep
            for t in pair
        ]

    def handle_op(self, op: stim.CircuitInstruction, sss: TablesideSimulator):
        """handle a single stim CircuitInstruction."""
        check_late_frozen_read(self._fast_1q, self._compiled_len, op, sss, self.ops_to_params)
        if op not in self.claimed_ops_keys:
            if self.unconditional_condition_on_U and np.any(self.state >= 2):
                op_name = op.name
                gate_data = stim.gate_data(op_name)
                if gate_data.produces_measurements or not (
                    gate_data.is_noisy_gate or gate_data.is_unitary
                ):
                    sss._do_bare_instruction(op)
                    return

                if gate_data.is_single_qubit_gate and self._fast_1q:
                    sss._do_bare_instruction(op)
                    return

                if gate_data.is_single_qubit_gate or op_name in (
                    "E",
                    "CORRELATED_ERROR",
                    "ELSE_CORRELATED_ERROR",
                ):
                    raw_targets = op.targets_copy()
                    qubit_indices = [
                        t.qubit_value if t.qubit_value is not None else -1
                        for t in raw_targets
                    ]
                    mask = self._filter_targets_1q_mask(qubit_indices, (("U",),), sss)
                    if not gate_data.is_single_qubit_gate and not np.all(mask):
                        # A correlated error is one Pauli product (like a pair for 2q ops): skip all of it.
                        mask = np.zeros_like(mask)
                    filtered_targets = [
                        t for t, keep in zip(raw_targets, mask) if keep
                    ]
                elif gate_data.is_two_qubit_gate:
                    target_groups = op.target_groups()
                    filtered_targets = self._filter_targets_2q(
                        target_groups, (("U",), ("U",)), sss
                    )
                else:
                    # SPP / SPP_DAG: keep each Pauli product term only if none of its qubits is leaked.
                    filtered_targets = []
                    for term in op.target_groups():
                        if all(self.state[t.qubit_value] < 2 for t in term):
                            for k, t in enumerate(term):
                                if k > 0:
                                    filtered_targets.append(stim.target_combiner())
                                filtered_targets.append(t)
                if len(filtered_targets) > 0:
                    sss._do_bare_instruction(
                        stim.CircuitInstruction(op_name, filtered_targets, op.gate_args_copy())
                    )
                elif op_name in (
                    "E",
                    "CORRELATED_ERROR",
                    "ELSE_CORRELATED_ERROR",
                ):
                    sss._do_bare_instruction(
                        stim.CircuitInstruction(op_name, [], op.gate_args_copy())
                    )
                return
            else:
                sss._do_bare_instruction(op)
                return

        dispatch_tagged_leakage_op(self, op, sss, state2d=self.state[:, None])

    def leakage_swap(self, op: stim.CircuitInstruction, tss: TablesideSimulator):
        """SWAP[LEAKAGE_SWAP]: SWAP the qubits, leaked ones included, then swap leakage states."""
        exec_leakage_swap(op, tss, self.state)

    def leakage_conditioning(
        self,
        op: stim.CircuitInstruction,
        tss: TablesideSimulator,
        params: LeakageConditioningParams,
    ):
        """implement conditioning on qubit (leakage) state"""

        if params.targets:
            copy_len = len(params.targets)
            copies = split_fused_instruction(op, copy_len)
            if len(copies) > 1:
                for sub_op in copies:
                    self.leakage_conditioning(sub_op, tss, params)
                return

        # controller state is an array where each qubit has the leakage state
        # of the qubit controlling it (or 0 if it's not being targeted)
        condition_groups = params.args
        if all(g == ("U",) for g in condition_groups) and not np.any(self.state >= 2):
            tss._do_bare_instruction(op)
            return

        other_targets = params.targets
        if (
            any(0 in g or 1 in g for g in condition_groups)
            and op.name
            not in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR")
        ):
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
                    for g_idx, grp in enumerate(t_groups):
                        sub_other: tuple[int, ...] | None = None
                        if other_targets:
                            step_ot = len(other_targets) // len(t_groups)
                            sub_other = tuple(
                                other_targets[
                                    g_idx * step_ot : (g_idx + 1) * step_ot
                                ]
                            )
                        sub_params = LeakageConditioningParams(
                            args=condition_groups,
                            targets=sub_other,
                            from_tag=params.from_tag,
                        )
                        sub_op = stim.CircuitInstruction(
                            op.name, grp, op.gate_args_copy()
                        )
                        self.leakage_conditioning(sub_op, tss, sub_params)
                    return
        if other_targets:
            # CONDITIONED_ON_OTHER case
            raw_targets = op.targets_copy()
            if len(other_targets) != len(raw_targets):
                raise ValueError(
                    f"The number of targets of the {op} is not the same as the"
                    "number of targets specified in the CONDITIONED_ON_OTHER tag."
                )
            mask = self._filter_targets_1q_mask(
                list(other_targets),
                cast(tuple[tuple[int | str, ...]], condition_groups),
                tss,
            )
            filtered_targets = [
                t for t, keep in zip(raw_targets, mask) if keep
            ]
        else:
            if len(condition_groups) == 1:
                # CONDITIONED_ON_SELF / single-group CONDITIONED_ON case
                raw_targets = op.targets_copy()
                qubit_indices = [
                    t.qubit_value if t.qubit_value is not None else -1
                    for t in raw_targets
                ]
                mask = self._filter_targets_1q_mask(
                    qubit_indices, condition_groups, tss
                )
                filtered_targets = [
                    t for t, keep in zip(raw_targets, mask) if keep
                ]
            else:
                # CONDITIONED_ON_PAIR case
                target_groups = op.target_groups()
                filtered_targets = self._filter_targets_2q(
                    target_groups, condition_groups, tss
                )

        if len(filtered_targets) > 0:
            tss._do_bare_instruction(
                stim.CircuitInstruction(op.name, filtered_targets, op.gate_args_copy())
            )
        elif op.name in (
            "E",
            "CORRELATED_ERROR",
            "ELSE_CORRELATED_ERROR",
        ):
            tss._do_bare_instruction(
                stim.CircuitInstruction(op.name, [], op.gate_args_copy())
            )

    def leakage_transition_1(
        self,
        op: stim.CircuitInstruction,
        tss: TablesideSimulator,
        params: LeakageTransition1Params,
    ):
        """implement leakage state transitions on single qubits.

        Allows depolarization when something comes unleaked and is specified by 'U'
        """

        raw_t = op.targets_copy()
        if len(raw_t) > 1:
            qs_list = [t.qubit_value for t in raw_t if t.qubit_value is not None]
            if len(set(qs_list)) < len(qs_list):
                for t in raw_t:
                    sub_op = stim.CircuitInstruction(
                        op.name, [t], op.gate_args_copy()
                    )
                    self.leakage_transition_1(sub_op, tss, params)
                return

        target_mask = np.zeros_like(self.state, dtype=bool)
        target_mask[[t.qubit_value if t.qubit_value is not None else t.value for t in raw_t]] = True
        sync_rng = getattr(tss, "_sync_tableside_rng", False)

        if 0 in params.args_by_input_state or 1 in params.args_by_input_state:
            self._update_state_from_peek_Z(np.where(target_mask)[0], tss)

        initial_state = self.state.copy()
        effects = TransitionEffectMasks.zeros(self.state.shape, two_qubit=False)

        for input_state in params.args_by_input_state.keys():
            in_mask = initial_state < 2 if input_state == "U" else initial_state == input_state
            target_qs = np.where(target_mask & in_mask)[0]
            if len(target_qs) == 0:
                continue
            sampled = sample_tableau_transitions(
                params, input_state, len(target_qs), tss.np_rng, sync_rng=sync_rng
            )
            if sampled is None:
                continue
            sel, output_states = sampled
            if not isinstance(sel, slice):
                target_qs = target_qs[np.sort(sel)]
            for q_np, out_st in zip(target_qs, output_states):
                apply_output_effect(
                    self.state, effects, int(q_np), input_state, out_st,
                    drop_depol_reset=self.fast_leakage,
                )

        self.state[self.state == 1] = 0
        emit_tableau_transition_effects(tss, effects)

    def leakage_transition_2(
        self,
        op: stim.CircuitInstruction,
        tss: TablesideSimulator,
        params: LeakageTransition2Params,
    ):
        """implement leakage state transitions on pairs of qubits.

        when something unleaks, fully depolarize it
        """

        targets_list = [
            target.qubit_value if target.qubit_value is not None else target.value
            for target in op.targets_copy()
        ]
        if len(targets_list) > 2 and len(set(targets_list)) < len(targets_list):
            for grp in op.target_groups():
                sub_op = stim.CircuitInstruction(
                    op.name, grp, op.gate_args_copy()
                )
                self.leakage_transition_2(sub_op, tss, params)
            return

        targets_even = np.asarray(targets_list[::2], dtype=np.intp)
        targets_odd = np.asarray(targets_list[1::2], dtype=np.intp)
        sync_rng = getattr(tss, "_sync_tableside_rng", False)

        if any(k[0] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(targets_list[::2], tss)
        if any(k[1] in (0, 1) for k in params.args_by_input_state.keys()):
            self._update_state_from_peek_Z(targets_list[1::2], tss)

        initial_state = self.state.copy()
        even_target_states = initial_state[targets_even]
        odd_target_states = initial_state[targets_odd]
        effects = TransitionEffectMasks.zeros(self.state.shape, two_qubit=True)

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
            trigger_mask = even_checks & odd_checks
            samples_to_take = int(np.count_nonzero(trigger_mask))
            if samples_to_take == 0:
                continue
            sampled = sample_tableau_transitions(
                params, input_state, samples_to_take, tss.np_rng, sync_rng=sync_rng
            )
            if sampled is None:
                continue
            sel, output_states = sampled
            triggered_pairs = np.column_stack(
                (targets_even[trigger_mask], targets_odd[trigger_mask])
            )[sel]
            for pair, out_pair in zip(triggered_pairs, output_states):
                for leg in (0, 1):
                    apply_output_effect(
                        self.state, effects, int(pair[leg]), input_state[leg], out_pair[leg],
                        drop_depol_reset=self.fast_leakage,
                    )

        self.state[self.state == 1] = 0
        emit_tableau_transition_effects(tss, effects)

    def leakage_measurement(
        self,
        op: stim.CircuitInstruction,
        tss: TablesideSimulator,
        params: LeakageMeasurementParams,
    ):
        """implement qubit state projection measurement that can be affected by leakage."""

        if params.targets:
            copies = split_fused_instruction(op, len(params.targets))
            if len(copies) > 1:
                for sub_op in copies:
                    self.leakage_measurement(sub_op, tss, params)
                return

        is_mpad, raw_op_targets, targets, inv_flags = unpack_leakage_measurement_targets(
            op, params
        )

        if not is_mpad and len(set(targets)) < len(targets):
            for t in raw_op_targets:
                sub_op = stim.CircuitInstruction(
                    op.name, [t], op.gate_args_copy()
                )
                self.leakage_measurement(sub_op, tss, params)
            return

        basis_rot_gate = (
            "H"
            if op.name in ("MX", "MRX")
            else ("H_YZ" if op.name in ("MY", "MRY") else None)
        )
        is_reset_meas = op.name in ("MR", "MRZ", "MRX", "MRY")
        if basis_rot_gate is not None:
            tss._do_bare_only_on_tableau(
                stim.CircuitInstruction(basis_rot_gate, targets)
            )

        n_qbt = max(
            tss.num_qubits, tss.num_qubits_in_new_circuit(), max(targets, default=-1) + 1
        )

        # get actual qubit states
        target_leakage_state = self.state[targets]

        p0, p1 = params.p0, params.p1

        targets_arr = np.asarray(targets, dtype=np.intp)
        idx_of_targets_in_qubit_space = np.where(target_leakage_state < 2)[0]
        targets_in_qubit_space = targets_arr[idx_of_targets_in_qubit_space]
        targets_new = targets_arr.copy()

        if len(targets_in_qubit_space) > 0:
            if not is_mpad and p0 == 0.0 and p1 == 1.0:
                # Standard Z measurement on unleaked qubits: measure directly without R or X_ERROR
                # so post-measurement state and multi-qubit entanglement are preserved.
                pass
            elif not is_mpad and np.isclose(p0, 1.0 - p1) and not getattr(tss, "_sync_tableside_rng", False):
                # Symmetric readout flip probability p0 on unleaked qubits:
                # Copy Z state coherently to fresh ancilla via CX and apply readout error on ancilla.
                anc_comp = list(
                    range(n_qbt, n_qbt + len(targets_in_qubit_space))
                )
                n_qbt += len(targets_in_qubit_space)
                targets_new[idx_of_targets_in_qubit_space] = anc_comp
                cx_targets = []
                for q_d, q_a in zip(targets_in_qubit_space, anc_comp):
                    cx_targets.extend([int(q_d), int(q_a)])
                tss._do_bare_only_on_tableau(
                    stim.CircuitInstruction("CX", cx_targets)
                )
                if p0 > 0:
                    tss._do_bare_only_on_tableau(
                        stim.CircuitInstruction("X_ERROR", anc_comp, [p0])
                    )
            else:
                # Asymmetric readout probabilities on 0 vs 1 (or MPAD)
                has_01_cond = (0 in params.prob_for_input_state) or (
                    1 in params.prob_for_input_state
                )
                if is_mpad and not has_01_cond and np.isclose(p0, p1):
                    anc_comp = list(
                        range(n_qbt, n_qbt + len(targets_in_qubit_space))
                    )
                    n_qbt += len(targets_in_qubit_space)
                    targets_new[idx_of_targets_in_qubit_space] = anc_comp
                    if p0 > 0:
                        tss._do_bare_only_on_tableau(
                            stim.CircuitInstruction("X_ERROR", anc_comp, [p0])
                        )
                elif tss.batch_size > 1 and hasattr(tss, "_asymmetric_readout_postproc"):
                    # batch_size > 1 frame-samples _new_circuit, so the readout must follow each shot's own Z value:
                    # copy it onto a fresh ancilla (measuring the ancilla Z-collapses a superposed target, as at
                    # batch_size == 1) and apply p(0)/p(1) per shot when the records are sampled.
                    anc_comp = list(
                        range(n_qbt, n_qbt + len(targets_in_qubit_space))
                    )
                    n_qbt += len(targets_in_qubit_space)
                    targets_new[idx_of_targets_in_qubit_space] = anc_comp
                    cx_targets = []
                    for q_d, q_a in zip(targets_in_qubit_space, anc_comp):
                        cx_targets.extend([int(q_d), int(q_a)])
                    tss._do_bare_only_on_tableau(
                        stim.CircuitInstruction("CX", cx_targets)
                    )
                    base_meas_idx = tss._new_circuit.num_measurements
                    for i_pos in idx_of_targets_in_qubit_space:
                        tss._asymmetric_readout_postproc.append(
                            (
                                int(base_meas_idx + i_pos),
                                float(p0),
                                float(p1),
                                inv_flags[int(i_pos)],
                            )
                        )
                else:
                    # Resolve superposition Z states in tableau_simulator when batch_size == 1
                    pauli_states = np.empty(
                        len(targets_in_qubit_space), dtype=np.intp
                    )
                    for i_q, q in enumerate(targets_in_qubit_space):
                        pauli_states[i_q] = tss.peek_z(int(q))
                    if tss.batch_size == 1:
                        superpos_indices = np.where(pauli_states == 0)[0]
                        for s_i, i_q in enumerate(superpos_indices):
                            q_int = int(targets_in_qubit_space[i_q])
                            pz = 0 if s_i == 0 else tss.peek_z(q_int)
                            if pz == 0:
                                bit = int(tss.np_rng.binomial(1, 0.5))
                                tss._tableau_simulator.postselect_z(
                                    q_int, desired_value=bool(bit)
                                )
                                pz = -1 if bit == 1 else 1
                            pauli_states[i_q] = pz

                    anc_comp = list(
                        range(n_qbt, n_qbt + len(targets_in_qubit_space))
                    )
                    n_qbt += len(targets_in_qubit_space)
                    targets_new[idx_of_targets_in_qubit_space] = anc_comp
                    anc_comp_arr = np.asarray(anc_comp, dtype=np.intp)

                    if not is_mpad:
                        cx_targets = []
                        for q_d, q_a in zip(targets_in_qubit_space, anc_comp):
                            cx_targets.extend([int(q_d), int(q_a)])
                        tss._do_bare_only_on_tableau(
                            stim.CircuitInstruction("CX", cx_targets)
                        )

                    m0_mask = pauli_states == 1
                    m1_mask = pauli_states == -1
                    mflip_mask = pauli_states == 0

                    if np.any(m0_mask) and p0 > 0:
                        tss._do_bare_only_on_tableau(
                            stim.CircuitInstruction(
                                "X_ERROR", anc_comp_arr[m0_mask], [p0]
                            )
                        )
                    if np.any(m1_mask):
                        if is_mpad:
                            tss._do_bare_only_on_tableau(
                                stim.CircuitInstruction("X", anc_comp_arr[m1_mask])
                            )
                        if 1.0 - p1 > 0:
                            tss._do_bare_only_on_tableau(
                                stim.CircuitInstruction(
                                    "X_ERROR", anc_comp_arr[m1_mask], [1.0 - p1]
                                )
                            )
                    if np.any(mflip_mask):
                        if p1 >= p0:
                            x_error_prob = 0.5 * p0 + 0.5 * (1.0 - p1)
                        else:
                            tss._do_bare_only_on_tableau(
                                stim.CircuitInstruction("X", anc_comp_arr[mflip_mask])
                            )
                            x_error_prob = 0.5 * (1.0 - p0) + 0.5 * p1
                        if x_error_prob > 0:
                            tss._do_bare_only_on_tableau(
                                stim.CircuitInstruction(
                                    "X_ERROR",
                                    anc_comp_arr[mflip_mask],
                                    [x_error_prob],
                                )
                            )

        # Handle qubits in leaked states (> 1)
        target_leaked_pos = np.where(target_leakage_state > 1)[0]
        leaked_qubits = targets_arr[target_leaked_pos]
        if len(target_leaked_pos) > 0:
            if is_mpad:
                leaked_meas_targets = np.asarray(
                    list(range(n_qbt, n_qbt + len(target_leaked_pos))),
                    dtype=np.intp,
                )
                n_qbt += len(target_leaked_pos)
                targets_new[target_leaked_pos] = leaked_meas_targets
            else:
                leaked_meas_targets = leaked_qubits
                tss._do_bare_only_on_tableau(
                    stim.CircuitInstruction("R", leaked_meas_targets)
                )
            for n in range(2, int(np.max(target_leakage_state)) + 1):
                if n in target_leakage_state and n in params.prob_for_input_state:
                    idx_n = leaked_meas_targets[
                        target_leakage_state[target_leaked_pos] == n
                    ]
                    if params.prob_for_input_state[n] > 0:
                        tss._do_bare_only_on_tableau(
                            stim.CircuitInstruction(
                                "X_ERROR", idx_n, [params.prob_for_input_state[n]]
                            )
                        )

        # Finally perform qubit space measurement, preserving any target inversion (!)
        m_targets: list[int | stim.GateTarget] = [
            stim.target_inv(int(q_new)) if inv_flags[i] else int(q_new)
            for i, q_new in enumerate(targets_new)
        ]

        tss._do_bare_only_on_tableau(stim.CircuitInstruction("M", m_targets))
        if is_reset_meas and len(targets_in_qubit_space) > 0:
            tss._do_bare_only_on_tableau(
                stim.CircuitInstruction("R", targets_in_qubit_space)
            )
        if basis_rot_gate is not None:
            tss._do_bare_only_on_tableau(
                stim.CircuitInstruction(basis_rot_gate, targets)
            )
        if is_mpad:
            tss._append_to_new_reference_circuit(op)
        else:
            ref_m_targets = [
                stim.target_inv(int(q)) if inv_flags[i] else int(q)
                for i, q in enumerate(targets)
            ]
            tss._append_to_new_reference_circuit(
                stim.CircuitInstruction(op.name, ref_m_targets)
            )
            if len(leaked_qubits) > 0:
                tss._do_bare_only_on_tableau(
                    stim.CircuitInstruction("DEPOLARIZE1", leaked_qubits, [0.75])
                )


LeakageUint8 = LeakageUint8Tableau
CompiledLeakageUint8 = CompiledLeakageUint8Tableau
