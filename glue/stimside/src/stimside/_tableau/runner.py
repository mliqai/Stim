"""Precompiler, synchronized Python runner, and C++ runner wrapper for `TablesideSimulator`."""

from __future__ import annotations

import ctypes
import sys
from typing import TYPE_CHECKING, Any
import weakref

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageParams,
    LeakageSwapParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
)
from stimside.op_handlers.leakage_handlers.tag_registry import (
    parse_leakage_tag,
)
from stimside._native.tableside_kernels import (
    EmittedActionC,
    StepDescC,
    TransBranchC,
    get_tableside_kernels_lib,
)
from stimside.util.circuit import apply_coords_op, unroll_circuit

if TYPE_CHECKING:
    from stimside.simulator_tableau import TablesideSimulator


_STATE_CODE_MAP = {
    "U": -1,
    "V": -2,
    "D": -3,
    "X": -4,
    "Y": -5,
    "Z": -6,
}


def _encode_state(st: int | str) -> int:
    if isinstance(st, int):
        return st
    return _STATE_CODE_MAP[st]


def format_stim_op_text(op_name: str, targets: np.ndarray) -> str:
    if targets.dtype == np.uint32 and np.any(targets & (1 << 31)):
        t_strs = [
            f"!{int(t & 0x7FFFFFFF)}" if (t & (1 << 31)) else str(int(t))
            for t in targets
        ]
    else:
        t_strs = [str(int(t)) for t in targets]
    return f"{op_name} {' '.join(t_strs)}\n"


_EMPTY_I32 = np.empty(0, dtype=np.int32)


def _circuit_of(insts: list[stim.CircuitInstruction]) -> stim.Circuit:
    """Circuit built by appending the instructions one at a time (as stim.Circuit.append would fuse them)."""
    c = stim.Circuit()
    for inst in insts:
        c.append(inst)
    return c


def _slice_circuit(
    insts: list[stim.CircuitInstruction],
    cache: dict[tuple[int, int], stim.Circuit],
    blocks: list[stim.Circuit],
    a: int,
    b: int,
) -> stim.Circuit:
    """Cached circuit of insts[a:b]; spans of >= 64 reuse lazily built 32-instruction blocks."""
    if a >= b:
        return stim.Circuit()
    sub = cache.get((a, b))
    if sub is None:
        if b - a >= 64:
            if not blocks:
                for i in range(0, len(insts), 32):
                    blocks.append(_circuit_of(insts[i : i + 32]))
            b0, b1 = (a + 31) >> 5, b >> 5
            sub = _circuit_of(insts[a : b0 << 5])
            for k in range(b0, b1):
                sub += blocks[k]
            for inst in insts[b1 << 5 : b]:
                sub.append(inst)
        else:
            sub = _circuit_of(insts[a:b])
        if len(cache) >= 512:
            cache.pop(next(iter(cache)))
        cache[(a, b)] = sub
    return sub


class PrecompiledTablesideCircuit:
    """Pre-compiled circuit structure for TablesideSimulator (C++ FastEngine and sync-RNG runner)."""

    def __init__(
        self,
        circuit: stim.Circuit,
        unconditional_condition_on_U: bool = True,
        parsed_ops: dict[stim.CircuitInstruction, LeakageParams] | None = None,
    ) -> None:
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.num_words = max(1, (self.num_qubits + 63) // 64)
        self.unconditional_condition_on_U = unconditional_condition_on_U
        self._m2d_converter: Any = None
        self._reference_circuit: stim.Circuit | None = None
        self._full_bare_circuit: stim.Circuit | None = None
        self._quantum_bare_circuit: stim.Circuit | None = None

        has_repeat = any(isinstance(inst, stim.CircuitRepeatBlock) for inst in circuit)
        # Not circuit.flattened(): it fuses identical consecutive instructions, desyncing steps from step_to_unrolled_idx.
        unrolled_ops = unroll_circuit(circuit) if has_repeat else list(circuit)
        use_supplied_ops = parsed_ops is not None and all(
            isinstance(
                v,
                (
                    LeakageConditioningParams,
                    LeakageTransition1Params,
                    LeakageTransition2Params,
                    LeakageMeasurementParams,
                    LeakageSwapParams,
                ),
            )
            for v in parsed_ops.values()
        )
        self.parsed_ops: dict[stim.CircuitInstruction, LeakageParams] = {}

        self.final_qubit_coords: dict[int, list[float]] = {}
        self.final_qubit_tags: dict[int, str] = {}
        self.final_coords_shifts: list[float] = []
        self.ref_insts: list[stim.CircuitInstruction] = []

        self.steps_py: list[tuple[Any, ...]] = []
        self.orig_ops: list[stim.CircuitInstruction] = []
        self.full_bare_insts: list[stim.CircuitInstruction] = []
        self.quantum_bare_insts: list[stim.CircuitInstruction] = []
        self.step_to_full_idx: list[int] = []
        self.step_to_quant_idx: list[int] = []
        self.step_has_bare_op: list[bool] = []
        self.requires_tableau = False

        c_steps: list[StepDescC] = []
        raw_t_chunks: list[np.ndarray] = []
        qubit_t_chunks: list[np.ndarray] = []
        t_running_len = 0
        all_step_masks: list[int] = [0] * self.num_words
        all_branches: list[TransBranchC] = []
        op_template_cache: dict[
            stim.CircuitInstruction,
            tuple[
                stim.CircuitInstruction | None,
                stim.CircuitInstruction | None,
                StepDescC,
                tuple[Any, ...],
            ],
        ] = {}

        def push(bare, ref, c_step: StepDescC, s_py: tuple[Any, ...]) -> None:
            """Append one step (bare/ref instruction may be None) to the per-step lists."""
            if bare is not None:
                self.full_bare_insts.append(bare)
                self.quantum_bare_insts.append(bare)
            if ref is not None:
                self.ref_insts.append(ref)
            self.step_has_bare_op.append(bare is not None)
            c_steps.append(c_step)
            self.steps_py.append(s_py)

        def emit(bare, ref, kind, text="", p_tot=0.0, br=(0, 0), p_leak=0.0) -> None:
            """Build, cache and push the current op's step (reads this iteration's locals)."""
            c_step = StepDescC(kind, t_off, t_cnt, m_off, p_tot, *br, p_leak)
            s_py = (kind, op_name, gate_args, text, qubit_i32, params)
            op_template_cache[op] = (bare, ref, c_step, s_py)
            push(bare, ref, c_step, s_py)

        for op in unrolled_ops:
            op_name = op.name
            if op_name in ("DETECTOR", "TICK", "OBSERVABLE_INCLUDE"):
                self.full_bare_insts.append(op)
                self.ref_insts.append(op)
                continue
            if apply_coords_op(
                op,
                self.final_qubit_coords,
                self.final_qubit_tags,
                self.final_coords_shifts,
            ):
                self.full_bare_insts.append(op)
                self.ref_insts.append(op)
                continue

            self.step_to_full_idx.append(len(self.full_bare_insts))
            self.step_to_quant_idx.append(len(self.quantum_bare_insts))
            self.orig_ops.append(op)

            tpl = op_template_cache.get(op)
            if tpl is not None:
                push(*tpl)
                continue

            params: LeakageParams | None = None
            if op.tag != "":
                params = (
                    parsed_ops.get(op)
                    if use_supplied_ops and parsed_ops is not None
                    else parse_leakage_tag(op, simulator="tableau")
                )
                if params is not None:
                    self.parsed_ops[op] = params
            is_claimed = params is not None

            if not is_claimed:
                gate_data = stim.gate_data(op_name)
                if (
                    not self.unconditional_condition_on_U
                    or gate_data.produces_measurements
                    or not (gate_data.is_noisy_gate or gate_data.is_unitary)
                ):
                    c_step_inst = StepDescC(0, 0, 0, 0, 0.0, 0, 0, 0.0)
                    s_py: tuple[Any, ...] = (0, op_name, [], "", _EMPTY_I32, None)
                    op_template_cache[op] = (op, op, c_step_inst, s_py)
                    push(op, op, c_step_inst, s_py)
                    continue

            raw_t = op.targets_copy()
            gate_args = op.gate_args_copy()
            has_special_targets = len(gate_args) > 1 or any(
                not (
                    t.is_qubit_target
                    or (t.is_inverted_result_target and t.qubit_value is not None)
                )
                for t in raw_t
            )
            if not has_special_targets:
                raw_u32 = np.array(
                    [
                        t.qubit_value | ((1 << 31) if t.is_inverted_result_target else 0)
                        for t in raw_t
                    ],
                    dtype=np.uint32,
                )
                qubit_i32 = (raw_u32 & 0x7FFFFFFF).astype(np.int32)
            else:
                raw_u32 = np.array(
                    [
                        (
                            (t.qubit_value if t.qubit_value is not None else max(0, t.value))
                            | ((1 << 31) if t.is_inverted_result_target else 0)
                        )
                        for t in raw_t
                    ],
                    dtype=np.uint32,
                )
                qubit_vals = [t.qubit_value for t in raw_t]
                qubit_i32 = np.array(
                    [
                        q if q is not None else t.value
                        for q, t in zip(qubit_vals, raw_t)
                    ],
                    dtype=np.int32,
                )
            bare_inst = (
                op
                if not is_claimed
                else stim.CircuitInstruction(op_name, raw_t, gate_args)
            )

            has_dup_targets = len(set(qubit_i32.tolist())) < len(qubit_i32)
            if isinstance(params, LeakageTransition1Params) and not has_dup_targets:
                unique_qs = np.unique(qubit_i32[qubit_i32 >= 0])
                qubit_i32 = unique_qs.astype(np.int32)
                raw_u32 = unique_qs.astype(np.uint32)

            mask_words = [0] * self.num_words
            # With special targets, qubit_i32 holds .value for a combiner (0) or sweep[k] (k), not a qubit.
            mask_source = (
                params.targets
                if (isinstance(params, LeakageConditioningParams) and params.targets)
                else qubit_i32
                if not has_special_targets
                else [q for q in qubit_vals if q is not None]
            )
            for q in mask_source:
                q_int = int(q)
                if 0 <= q_int < self.num_qubits:
                    mask_words[q_int >> 6] |= 1 << (q_int & 63)

            t_off = t_running_len
            t_cnt = len(raw_u32)
            t_running_len += t_cnt
            m_off = len(all_step_masks)
            if t_cnt > 0:
                raw_t_chunks.append(raw_u32)
                qubit_t_chunks.append(qubit_i32)
            all_step_masks.extend(mask_words)

            if not is_claimed:
                if has_special_targets or gate_data.produces_measurements:
                    kind = 8
                elif gate_data.is_single_qubit_gate:
                    kind = 1
                elif gate_data.is_two_qubit_gate:
                    kind = 2
                else:
                    kind = 8
                emit(bare_inst, bare_inst, kind)
            else:
                if isinstance(params, LeakageConditioningParams):
                    cond_groups = params.args
                    if any(0 in g or 1 in g for g in cond_groups):
                        self.requires_tableau = True
                        kind = 6
                    elif (
                        not params.targets
                        and not has_special_targets
                        and len(cond_groups) == 1
                        and cond_groups[0] == ("U",)
                    ):
                        kind = 2 if stim.gate_data(op_name).is_two_qubit_gate else 1
                    elif (
                        not params.targets
                        and not has_special_targets
                        and len(cond_groups) == 2
                        and cond_groups == (("U",), ("U",))
                    ):
                        kind = 2
                    elif all(g == ("U",) for g in cond_groups):
                        kind = 8
                    else:
                        kind = 6
                    emit(bare_inst, bare_inst, kind)

                elif isinstance(params, LeakageTransition1Params):
                    has_01 = (0 in params.args_by_input_state) or (
                        1 in params.args_by_input_state
                    )
                    if has_01 or has_dup_targets:
                        self.requires_tableau = True
                        p1_tot = sum(
                            p for _, p in params.args_by_input_state.get(1, ())
                        )
                        emit(None, None, 6, p_tot=p1_tot)
                    else:
                        br_off = len(all_branches)
                        u_args = params.args_by_input_state.get("U", ())
                        p_tot_u = sum(p for _, p in u_args)
                        cum = 0.0
                        for out_st, p in u_args:
                            cum += p / p_tot_u if p_tot_u > 0 else 0.0
                            all_branches.append(
                                TransBranchC(
                                    cum, -1, -1, _encode_state(out_st), 0, p
                                )
                            )
                        for in_st, br_list in params.args_by_input_state.items():
                            if in_st != "U":
                                for out_st, p in br_list:
                                    all_branches.append(
                                        TransBranchC(
                                            0.0,
                                            int(in_st),
                                            -1,
                                            _encode_state(out_st),
                                            0,
                                            p,
                                        )
                                    )
                        br = (br_off, len(all_branches) - br_off)
                        emit(None, None, 3, p_tot=p_tot_u, br=br)

                elif isinstance(params, LeakageTransition2Params):
                    has_01 = any(
                        0 in k or 1 in k for k in params.args_by_input_state.keys()
                    )
                    has_overlapping_pairs = len(set(qubit_i32.tolist())) < len(
                        qubit_i32
                    )
                    if has_01:
                        self.requires_tableau = True
                        emit(None, None, 6)
                    elif has_overlapping_pairs:
                        emit(None, None, 6)
                    else:
                        br_off = len(all_branches)
                        uu_args = params.args_by_input_state.get(("U", "U"), ())
                        p_tot_u = sum(p for _, p in uu_args)
                        cum = 0.0
                        for (out0, out1), p in uu_args:
                            cum += p / p_tot_u if p_tot_u > 0 else 0.0
                            all_branches.append(
                                TransBranchC(
                                    cum,
                                    -1,
                                    -1,
                                    _encode_state(out0),
                                    _encode_state(out1),
                                    p,
                                )
                            )
                        for in_pair, br_list in params.args_by_input_state.items():
                            if in_pair != ("U", "U"):
                                k0 = _encode_state(in_pair[0])
                                k1 = _encode_state(in_pair[1])
                                for (out0, out1), p in br_list:
                                    all_branches.append(
                                        TransBranchC(
                                            0.0,
                                            k0,
                                            k1,
                                            _encode_state(out0),
                                            _encode_state(out1),
                                            p,
                                        )
                                    )
                        br = (br_off, len(all_branches) - br_off)
                        emit(None, None, 4, p_tot=p_tot_u, br=br)

                elif isinstance(params, LeakageMeasurementParams):
                    is_mpad = params.targets is not None
                    p0 = params.prob_for_input_state.get(0, 0.0)
                    p1 = params.prob_for_input_state.get(
                        1, 0.0 if is_mpad else 1.0
                    )
                    p2 = params.prob_for_input_state.get(2, 0.0)
                    meas_inst = stim.CircuitInstruction(
                        "M" if is_mpad else op_name, raw_t, []
                    )
                    ref_inst = op if is_mpad else meas_inst
                    extra_states = set(params.prob_for_input_state.keys()) - {
                        0,
                        1,
                        2,
                    }
                    has_01_cond = (0 in params.prob_for_input_state) or (
                        1 in params.prob_for_input_state
                    )
                    if (is_mpad and (has_01_cond or not np.isclose(p0, p1))) or (
                        not is_mpad and not np.isclose(p0, 1.0 - p1)
                    ):
                        self.requires_tableau = True
                    if (
                        not is_mpad
                        and not has_dup_targets
                        and op_name in ("M", "MZ")
                        and p0 == 0.0
                        and p1 == 1.0
                        and not extra_states
                    ):
                        kind = 5
                        bare_op_text = format_stim_op_text("M", raw_u32)
                    else:
                        kind = 6
                        bare_op_text = ""
                    emit(meas_inst, ref_inst, kind, text=bare_op_text, p_leak=p2)

                elif isinstance(params, LeakageSwapParams):
                    # The runners yield to the handler (bare SWAP + leakage state swap) when a target
                    # may be leaked; otherwise the bare SWAP runs in the slice (state swap is a no-op).
                    emit(bare_inst, bare_inst, 8)

        self.step_to_full_idx.append(len(self.full_bare_insts))
        self.step_to_quant_idx.append(len(self.quantum_bare_insts))
        self.num_steps = len(self.steps_py)

        # Fuse adjacent step s (kind==6, LEAKAGE_TRANSITION_1 with 1-->2) + step s+1 (kind==5, M[LEAKAGE_PROJECTION_Z: (1.0, 2)])
        for s in range(self.num_steps - 1):
            if c_steps[s].kind == 6 and c_steps[s + 1].kind == 5:
                p_s = self.steps_py[s][5]
                p_m = self.steps_py[s + 1][5]
                if (
                    isinstance(p_s, LeakageTransition1Params)
                    and isinstance(p_m, LeakageMeasurementParams)
                    and set(p_s.args_by_input_state.keys()) == {1}
                    and len(p_s.args_by_input_state[1]) == 1
                    and isinstance(p_s.args_by_input_state[1][0][0], int)
                    and p_s.args_by_input_state[1][0][0] == 2
                    and p_m.prob_for_input_state.get(2, 0.0) == 1.0
                    and len(set(self.steps_py[s][4].tolist()))
                    == len(self.steps_py[s][4])
                    and np.array_equal(
                        np.sort(self.steps_py[s][4]),
                        np.sort(self.steps_py[s + 1][4]),
                    )
                ):
                    # Copy first: step descriptors are shared between identical ops via the template cache.
                    c_steps[s] = StepDescC.from_buffer_copy(c_steps[s])
                    c_steps[s].kind = 7
                    self.steps_py[s] = (7, *self.steps_py[s][1:])

        self._full_slice_cache: dict[tuple[int, int], stim.Circuit] = {}
        self._quant_slice_cache: dict[tuple[int, int], stim.Circuit] = {}
        self._full_blocks: list[stim.Circuit] = []
        self._quant_blocks: list[stim.Circuit] = []

        self.unrolled_num_ops = len(unrolled_ops)
        self.step_to_unrolled_idx = [
            idx
            for idx, u_op in enumerate(unrolled_ops)
            if u_op.name
            not in (
                "DETECTOR",
                "TICK",
                "OBSERVABLE_INCLUDE",
                "SHIFT_COORDS",
                "QUBIT_COORDS",
            )
        ]

        self._c_steps = (StepDescC * len(c_steps))(*c_steps) if c_steps else None
        if self._c_steps is not None:
            for s_i, u_idx in enumerate(self.step_to_unrolled_idx):
                self._c_steps[s_i].unrolled_idx = int(u_idx)
        self._raw_t_np = (
            np.concatenate(raw_t_chunks)
            if raw_t_chunks
            else np.empty(0, dtype=np.uint32)
        )
        self._qubit_t_np = (
            np.concatenate(qubit_t_chunks)
            if qubit_t_chunks
            else np.empty(0, dtype=np.int32)
        )
        self._masks_np = np.asarray(all_step_masks, dtype=np.uint64)
        self._c_raw_t = (
            self._raw_t_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            if len(self._raw_t_np) > 0
            else None
        )
        self._c_qubit_t = (
            self._qubit_t_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if len(self._qubit_t_np) > 0
            else None
        )
        self._c_masks = (
            self._masks_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            if len(self._masks_np) > 0
            else None
        )
        self._c_branches = (
            (TransBranchC * len(all_branches))(*all_branches)
            if all_branches
            else None
        )
        self._runner_pool: list[CppTablesideRunner] = []

    @property
    def m2d_converter(self) -> Any:
        if self._m2d_converter is None:
            self._m2d_converter = self.circuit.compile_m2d_converter()
        return self._m2d_converter

    @property
    def reference_circuit(self) -> stim.Circuit:
        if self._reference_circuit is None:
            self._reference_circuit = _circuit_of(self.ref_insts)
        return self._reference_circuit

    @property
    def full_bare_circuit(self) -> stim.Circuit:
        if self._full_bare_circuit is None:
            self._full_bare_circuit = _circuit_of(self.full_bare_insts)
        return self._full_bare_circuit

    @property
    def quantum_bare_circuit(self) -> stim.Circuit:
        if self._quantum_bare_circuit is None:
            self._quantum_bare_circuit = _circuit_of(self.quantum_bare_insts)
        return self._quantum_bare_circuit

    def acquire_cpp_runner(self, seed: int) -> "CppTablesideRunner":
        while self._runner_pool:
            runner = self._runner_pool.pop()
            if sys.getrefcount(runner._raw_state) == runner._idle_state_refs:
                runner.clear(seed)
                return runner
        return CppTablesideRunner(self, seed)

    def release_cpp_runner(self, runner: "CppTablesideRunner") -> None:
        if len(self._runner_pool) < 8:
            self._runner_pool.append(runner)

    def get_full_slice(self, f0: int, f1: int) -> stim.Circuit:
        if f0 == 0 < f1 == len(self.full_bare_insts):
            return self.full_bare_circuit
        return _slice_circuit(self.full_bare_insts, self._full_slice_cache, self._full_blocks, f0, f1)

    def get_quant_slice(self, q0: int, q1: int) -> stim.Circuit:
        if q0 == 0 < q1 == len(self.quantum_bare_insts):
            return self.quantum_bare_circuit
        return _slice_circuit(self.quantum_bare_insts, self._quant_slice_cache, self._quant_blocks, q0, q1)


_PRECOMPILED_CACHE: dict[
    tuple[int, int, bool],
    tuple[
        stim.Circuit,
        dict[stim.CircuitInstruction, LeakageParams] | None,
        PrecompiledTablesideCircuit,
    ],
] = {}


def get_precompiled_circuit(
    circuit: stim.Circuit,
    unconditional_condition_on_U: bool,
    parsed_ops: dict[stim.CircuitInstruction, LeakageParams] | None = None,
) -> PrecompiledTablesideCircuit:
    ucu = bool(unconditional_condition_on_U)
    key = (id(circuit), len(circuit), ucu)
    entry = _PRECOMPILED_CACHE.get(key)
    if (
        entry is not None
        and entry[0] == circuit
        and (parsed_ops is None and entry[1] is None or entry[1] == parsed_ops)
    ):
        return entry[2]
    snap = circuit.copy()
    ops_snap = dict(parsed_ops) if parsed_ops is not None else None
    pre = PrecompiledTablesideCircuit(snap, ucu, ops_snap)
    if len(_PRECOMPILED_CACHE) >= 16 and key not in _PRECOMPILED_CACHE:
        _PRECOMPILED_CACHE.pop(next(iter(_PRECOMPILED_CACHE)))
    _PRECOMPILED_CACHE[key] = (snap, ops_snap, pre)
    return pre


_ACTION_OP_MAP: dict[int, tuple[str, bool]] = {
    3: ("R", False),
    4: ("X", False),
    5: ("Y", False),
    6: ("Z", False),
    7: ("DEPOLARIZE1", True),
    8: ("X_ERROR", True),
    9: ("Z_ERROR", True),
}


def _leaked_meas_text(qs: np.ndarray, p2: float, bare_text: str) -> str:
    t = " ".join(str(int(q)) for q in qs)
    x = f"X_ERROR({p2}) {t}\n" if p2 > 0.0 else ""
    return f"R {t}\n{x}{bare_text}DEPOLARIZE1(0.75) {t}\n"


class _StepCursor:
    """Shared circuit/tableau slice-flush and handler-yield cursor for Tableside runners."""

    __slots__ = ("tss", "pre", "sync", "f", "q", "use_tab")

    def __init__(
        self,
        tss: "TablesideSimulator",
        pre: PrecompiledTablesideCircuit,
        *,
        sync: bool = False,
    ) -> None:
        self.tss, self.pre, self.sync = tss, pre, sync
        self.f = self.q = 0
        self.use_tab = tss._running_tableau

    def flush_full(self, f_end: int) -> None:
        if f_end > self.f:
            self.tss._new_circuit += self.pre.get_full_slice(self.f, f_end)
            self.f = f_end

    def flush_quant(self, q_end: int) -> None:
        if self.use_tab and q_end > self.q:
            tab = self.tss._tableau_simulator
            if self.sync:
                do = tab.do
                for inst in self.pre.quantum_bare_insts[self.q : q_end]:
                    do(inst)
            else:
                tab.do_circuit(self.pre.get_quant_slice(self.q, q_end))
            self.q = q_end

    def flush(self, f_end: int, q_end: int) -> None:
        self.flush_full(f_end)
        self.flush_quant(q_end)

    def emit(self, s_idx: int, txt: str, consumes: int) -> None:
        f_s = self.pre.step_to_full_idx[s_idx]
        self.flush_full(f_s)
        if txt:
            self.tss._new_circuit.append_from_stim_program_text(txt)
        self.f = f_s + consumes
        if self.use_tab:
            q_s = self.pre.step_to_quant_idx[s_idx]
            self.flush_quant(q_s)
            if txt:
                self.tss._tableau_simulator.do_circuit(stim.Circuit(txt))
            self.q = q_s + consumes

    def yield_step(self, s_idx: int) -> None:
        pre, tss = self.pre, self.tss
        f_s, q_s = pre.step_to_full_idx[s_idx], pre.step_to_quant_idx[s_idx]
        self.flush(f_s, q_s)
        tss._circuit_time = pre.step_to_unrolled_idx[s_idx]
        tss._compiled_op_handler.handle_op(op=pre.orig_ops[s_idx], sss=tss)
        self.use_tab = tss._running_tableau
        consumes = 1 if pre.step_has_bare_op[s_idx] else 0
        self.f, self.q = f_s + consumes, q_s + consumes

    def finish(self) -> None:
        self.flush(self.pre.step_to_full_idx[-1], self.pre.step_to_quant_idx[-1])
        self.tss.qubit_coords = self.pre.final_qubit_coords
        self.tss.qubit_tags = self.pre.final_qubit_tags
        self.tss.coords_shifts = self.pre.final_coords_shifts
        if self.tss._construct_reference_circuit:
            self.tss._new_reference_circuit = self.pre.reference_circuit.copy()
        self.tss._circuit_time = self.pre.unrolled_num_ops


def run_tableside_sync_rng(
    tss: "TablesideSimulator",
    pre: PrecompiledTablesideCircuit,
) -> None:
    """Tier 1 Python execution preserving 100% exact RNG lockstep with CosetsideSimulator(sync_tableside_rng=True)."""
    state = tss._compiled_op_handler.state
    steps = pre.steps_py
    cur = _StepCursor(tss, pre, sync=True)

    for s in range(pre.num_steps):
        kind, _, _, _, qubit_i32, params = steps[s]
        if kind == 0:
            continue
        if kind in (1, 2, 5, 8):
            is_other = isinstance(params, LeakageConditioningParams) and params.targets
            check_qs = (
                np.asarray(params.targets, dtype=np.int32)
                if is_other
                else qubit_i32[qubit_i32 >= 0]
            )
            # A stim-fused CONDITIONED_ON_OTHER op (more targets than the tag lists) goes to the
            # handler, which applies its copies one by one, drawing RNG like CosetsideSimulator.
            if not np.any(state[check_qs] >= 2) and not (
                is_other and len(qubit_i32) > len(params.targets)
            ):
                continue
        cur.yield_step(s)

    cur.finish()


class _EngineOwner:
    """Lifetime anchor ensuring FastEngine outlives all ctypes/NumPy views of its state buffer."""

    __slots__ = ("__weakref__",)


class CppTablesideRunner:
    """Tier 2 C++ FastEngine wrapper bound to a TablesideSimulator instance."""

    def __init__(
        self,
        pre: PrecompiledTablesideCircuit,
        seed: int,
    ) -> None:
        self.lib = get_tableside_kernels_lib()
        self.pre = pre
        self._shot_seed = int(seed) + 1000
        self._eng = self.lib.fast_engine_create(
            pre.num_qubits,
            pre.num_steps,
            ctypes.c_uint64(self._shot_seed),
            pre._c_steps,
            len(pre._raw_t_np),
            pre._c_raw_t,
            pre._c_qubit_t,
            len(pre._masks_np),
            pre._c_masks,
            len(pre._c_branches) if pre._c_branches is not None else 0,
            pre._c_branches,
        )
        owner = _EngineOwner()
        weakref.finalize(owner, self.lib.fast_engine_destroy, self._eng)
        state_ptr = self.lib.fast_engine_get_state_ptr(self._eng)
        n_alloc = max(1, pre.num_qubits)
        c_buf = (ctypes.c_uint8 * n_alloc).from_address(
            ctypes.cast(state_ptr, ctypes.c_void_p).value
        )
        c_buf._engine_owner = owner  # type: ignore[attr-defined]
        raw_state = np.ctypeslib.as_array(c_buf)
        self._raw_state = raw_state
        self.state = raw_state if pre.num_qubits == n_alloc else raw_state[: pre.num_qubits]
        del owner, c_buf, raw_state
        self._idle_state_refs = sys.getrefcount(self._raw_state)
        self._fmt_buf = ctypes.create_string_buffer(32768)

    def clear(self, new_seed: int) -> None:
        self._shot_seed = int(new_seed) + 1000
        self.lib.fast_engine_clear(self._eng, ctypes.c_uint64(self._shot_seed))

    def _cpp_format_op(
        self,
        op_name: str,
        gate_args: list[float],
        targets_ptr: Any,
        offset: int,
        count: int,
    ) -> str:
        if count <= 0:
            return ""
        if len(gate_args) > 1:
            op_prefix = f"{op_name}(" + ",".join(str(a) for a in gate_args) + ")"
            has_arg = 0
            arg0 = 0.0
        else:
            op_prefix = op_name
            has_arg = 1 if len(gate_args) == 1 else 0
            arg0 = float(gate_args[0]) if has_arg else 0.0
        needed = len(op_prefix) + 64 + count * 12
        if len(self._fmt_buf) < needed:
            self._fmt_buf = ctypes.create_string_buffer(max(65536, needed))
        n_bytes = self.lib.fast_format_stim_op(
            op_prefix.encode("ascii"),
            has_arg,
            arg0,
            ctypes.byref(targets_ptr.contents, offset * 4),
            count,
            self._fmt_buf,
        )
        return self._fmt_buf.raw[:n_bytes].decode("ascii")

    def run(self, tss: "TablesideSimulator") -> None:
        pre = self.pre
        cur = _StepCursor(tss, pre)
        if pre.num_steps == 0:
            cur.finish()
            return

        # Sync leaked mask in case user/test manually modified self._compiled_op_handler.state before run()
        if tss._compiled_op_handler.state is not self.state:
            np.copyto(self.state, tss._compiled_op_handler.state)
            tss._compiled_op_handler.state = self.state
        self.lib.fast_engine_sync_leaked_mask(self._eng)

        rec_leak = tss.record_unleaked_to_leaked
        self.lib.fast_engine_set_record_leakage(self._eng, 1 if rec_leak else 0)
        leaked_ops_ptr = ctypes.POINTER(ctypes.c_int32)() if rec_leak else None
        rec_ev = bool(getattr(tss, "record_leakage_events", False))
        self.lib.fast_engine_set_record_events(self._eng, 1 if rec_ev else 0)
        events_ptr = ctypes.POINTER(ctypes.c_int32)()

        actions_ptr = ctypes.POINTER(EmittedActionC)()
        targets_ptr = ctypes.POINTER(ctypes.c_uint32)()
        steps_py = pre.steps_py
        step_to_full = pre.step_to_full_idx
        step_to_quant = pre.step_to_quant_idx
        cur_step = 0
        num_steps = pre.num_steps

        while cur_step < num_steps:
            n_act = self.lib.fast_engine_run_segment(
                self._eng,
                cur_step,
                ctypes.byref(actions_ptr),
                ctypes.byref(targets_ptr),
            )
            if rec_leak and leaked_ops_ptr is not None:
                n_leaked_ops = self.lib.fast_engine_get_segment_leaked_ops(
                    self._eng, ctypes.byref(leaked_ops_ptr)
                )
                if n_leaked_ops > 0:
                    ev_list = tss._unleaked_to_leaked_events[0]
                    for k_ev in range(n_leaked_ops):
                        ev_list.append(int(leaked_ops_ptr[k_ev]))
            if rec_ev:
                n_events = self.lib.fast_engine_get_segment_events(
                    self._eng, ctypes.byref(events_ptr)
                )
                for k_ev in range(n_events):
                    tss._record_leakage_event(
                        events_ptr[4 * k_ev],
                        events_ptr[4 * k_ev + 1],
                        events_ptr[4 * k_ev + 2],
                        events_ptr[4 * k_ev + 3],
                    )
            next_step = num_steps
            for i in range(n_act):
                act = actions_ptr[i]
                kind = act.kind
                if kind == 0:  # ACT_RUN_SLICE
                    cur.flush(step_to_full[act.slice_end], step_to_quant[act.slice_end])
                elif kind in (1, 2):  # ACT_FILTERED_1Q / ACT_FILTERED_2Q
                    s_idx = act.step_idx
                    _, op_name, gate_args, _, _, _ = steps_py[s_idx]
                    txt = self._cpp_format_op(
                        op_name, gate_args, targets_ptr, act.target_offset, act.target_count
                    )
                    cur.emit(s_idx, txt, 1)
                elif kind in _ACTION_OP_MAP:
                    g_name, has_p = _ACTION_OP_MAP[kind]
                    txt = self._cpp_format_op(
                        g_name,
                        [act.prob] if has_p else [],
                        targets_ptr,
                        act.target_offset,
                        act.target_count,
                    )
                    cur.emit(act.step_idx, txt, 0)
                elif kind == 10:  # ACT_MODIFIED_MEAS
                    s_idx = act.step_idx
                    _, _, _, bare_op_text, _, _ = steps_py[s_idx]
                    off, cnt = act.target_offset, act.target_count
                    r_str = self._cpp_format_op("R", [], targets_ptr, off, cnt)
                    x_str = (
                        self._cpp_format_op("X_ERROR", [act.prob], targets_ptr, off, cnt)
                        if act.prob > 0.0
                        else ""
                    )
                    d_str = self._cpp_format_op("DEPOLARIZE1", [0.75], targets_ptr, off, cnt)
                    cur.emit(s_idx, r_str + x_str + bare_op_text + d_str, 1)
                elif kind == 12:  # ACT_YIELD_FUSED_MEAS
                    s_trans = act.step_idx
                    s_meas = act.slice_end
                    if not cur.use_tab:
                        cur.flush_full(step_to_full[s_trans])
                        tss._run_tableau()
                        cur.use_tab = True
                        cur.q = step_to_quant[s_trans]

                    _, _, _, meas_bare_text, meas_qs, _ = steps_py[s_meas]
                    p2 = act.prob
                    pre_leaked_qs = meas_qs[self.state[meas_qs] >= 2]
                    if len(pre_leaked_qs) == 0:
                        cur.flush_quant(step_to_quant[s_meas + 1])
                    else:
                        q_m = step_to_quant[s_meas]
                        cur.flush_quant(q_m)
                        tss._tableau_simulator.do_circuit(
                            stim.Circuit(_leaked_meas_text(pre_leaked_qs, p2, meas_bare_text))
                        )
                        cur.q = q_m + 1

                    newly_leaked = [
                        int(targets_ptr[act.target_offset + k])
                        for k in range(act.target_count)
                        if tss._tableau_simulator.peek_z(int(targets_ptr[act.target_offset + k])) == -1
                    ]
                    if newly_leaked:
                        out_state = int(steps_py[s_trans][5].args_by_input_state[1][0][0])
                        if rec_ev:
                            u_idx = pre.step_to_unrolled_idx[s_trans]
                            for q_new in newly_leaked:
                                tss._record_leakage_event(u_idx, q_new, self.state[q_new], out_state)
                        self.state[newly_leaked] = out_state
                        if rec_leak:
                            tss._record_unleaked_to_leaked_count(
                                len(newly_leaked), pre.step_to_unrolled_idx[s_trans]
                            )
                        self.lib.fast_engine_sync_leaked_mask(self._eng)
                        t_new = " ".join(str(q) for q in newly_leaked)
                        tss._tableau_simulator.do_circuit(
                            stim.Circuit(f"R {t_new}\nDEPOLARIZE1(0.75) {t_new}\n")
                        )

                    all_leaked_qs = meas_qs[self.state[meas_qs] >= 2]
                    if len(all_leaked_qs) > 0:
                        f_m = step_to_full[s_meas]
                        cur.flush_full(f_m)
                        tss._new_circuit.append_from_stim_program_text(
                            _leaked_meas_text(all_leaked_qs, p2, meas_bare_text)
                        )
                        cur.f = f_m + 1

                    next_step = s_meas + 1
                    break
                elif kind == 11:  # ACT_YIELD_PYTHON
                    cur.yield_step(act.step_idx)
                    self.lib.fast_engine_sync_leaked_mask(self._eng)
                    next_step = act.step_idx + 1
                    break
            cur_step = next_step

        cur.finish()
