"""Precompiler, synchronized Python runner, and C++ runner wrapper for `TablesideSimulator`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import ctypes
from dataclasses import dataclass
import os
import threading
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

STANDARD_LEAKAGE_PARAM_TYPES = (
    LeakageConditioningParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
    LeakageMeasurementParams,
    LeakageSwapParams,
)


def _is_asymmetric_readout(p: LeakageMeasurementParams) -> bool:
    if p.targets is not None:
        has_01 = 0 in p.prob_for_input_state or 1 in p.prob_for_input_state
        return has_01 or not np.isclose(p.p0, p.p1)
    return not np.isclose(p.p0, 1.0 - p.p1)


def collapses_z(
    ops: Iterable[stim.CircuitInstruction], ops_to_params: Mapping[Any, object]
) -> bool:
    """True if shots cannot share one trajectory: a 0/1-conditioned tag, or rec feedback
    after an asymmetric readout."""
    any_asym = False
    for p in ops_to_params.values():
        if isinstance(p, LeakageConditioningParams) and any(0 in g or 1 in g for g in p.args):
            return True
        if isinstance(p, LeakageTransition1Params) and (
            0 in p.args_by_input_state or 1 in p.args_by_input_state
        ):
            return True
        if isinstance(p, LeakageTransition2Params) and any(
            0 in k or 1 in k for k in p.args_by_input_state
        ):
            return True
        if isinstance(p, LeakageMeasurementParams) and _is_asymmetric_readout(p):
            any_asym = True
    if not any_asym:
        return False
    found_asym = False
    for op in ops:
        if op.name in ("DETECTOR", "TICK", "OBSERVABLE_INCLUDE"):
            continue
        if found_asym and any(t.is_measurement_record_target for t in op.targets_copy()):
            return True
        p = ops_to_params.get(op)
        if isinstance(p, LeakageMeasurementParams) and _is_asymmetric_readout(p):
            found_asym = True
    return False


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


_CACHE_LOCK = threading.Lock()


def _reinit_cache_lock() -> None:
    # A forked child inherits the lock in whatever state another parent thread left it.
    global _CACHE_LOCK
    _CACHE_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_cache_lock)


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
    with _CACHE_LOCK:
        sub = cache.get((a, b))
        need_blocks = sub is None and b - a >= 64 and not blocks
    if sub is None:
        if b - a >= 64:
            if need_blocks:
                built_blocks = [_circuit_of(insts[i : i + 32]) for i in range(0, len(insts), 32)]
                with _CACHE_LOCK:
                    if not blocks:
                        blocks[:] = built_blocks
            b0, b1 = (a + 31) >> 5, b >> 5
            sub = _circuit_of(insts[a : b0 << 5])
            for k in range(b0, b1):
                sub += blocks[k]
            for inst in insts[b1 << 5 : b]:
                sub.append(inst)
        else:
            sub = _circuit_of(insts[a:b])
        with _CACHE_LOCK:
            if len(cache) >= 512 and (a, b) not in cache:
                cache.pop(next(iter(cache)))
            cache[(a, b)] = sub
    return sub


_StepTemplate = tuple[
    stim.CircuitInstruction | None,
    stim.CircuitInstruction | None,
    StepDescC,
    tuple[Any, ...],
]


@dataclass(frozen=True)
class PrecompiledTablesideData:
    """Immutable pure precompilation output for ``TablesideSimulator``."""

    num_qubits: int
    num_words: int
    unconditional_condition_on_U: bool
    parsed_ops: dict[stim.CircuitInstruction, LeakageParams]
    final_qubit_coords: dict[int, list[float]]
    final_qubit_tags: dict[int, str]
    final_coords_shifts: list[float]
    ref_insts: list[stim.CircuitInstruction]
    steps_py: list[tuple[Any, ...]]
    orig_ops: list[stim.CircuitInstruction]
    full_bare_insts: list[stim.CircuitInstruction]
    quantum_bare_insts: list[stim.CircuitInstruction]
    step_to_full_idx: list[int]
    step_to_quant_idx: list[int]
    step_has_bare_op: list[bool]
    requires_tableau: bool
    num_steps: int
    collapses_z: bool
    unrolled_num_ops: int
    step_to_unrolled_idx: list[int]
    c_steps: Any
    raw_t_np: np.ndarray
    qubit_t_np: np.ndarray
    masks_np: np.ndarray
    c_branches: Any


class _PrecompileBuffers:
    """Mutable chunk, mask, and branch accumulator local to one precompile call."""

    __slots__ = (
        "all_branches",
        "all_step_masks",
        "final_coords_shifts",
        "final_qubit_coords",
        "final_qubit_tags",
        "num_qubits",
        "num_words",
        "qubit_t_chunks",
        "raw_t_chunks",
        "requires_tableau",
        "t_running_len",
        "uc",
    )

    def __init__(self, num_qubits: int, num_words: int, uc: bool) -> None:
        self.num_qubits = num_qubits
        self.num_words = num_words
        self.uc = uc
        self.requires_tableau = False
        self.final_qubit_coords: dict[int, list[float]] = {}
        self.final_qubit_tags: dict[int, str] = {}
        self.final_coords_shifts: list[float] = []
        self.raw_t_chunks: list[np.ndarray] = []
        self.qubit_t_chunks: list[np.ndarray] = []
        self.t_running_len = 0
        self.all_step_masks: list[int] = [0] * num_words
        self.all_branches: list[TransBranchC] = []


def _decode_op_targets(
    raw_t: list[Any],
    gate_args: list[float],
) -> tuple[np.ndarray, np.ndarray, list[int | None], bool]:
    # Only plain qubit targets (`0`, `!0`) take the fast decoding; a Pauli target stays special
    # when it is inverted (`!X0`).
    has_special_targets = len(gate_args) > 1 or any(not t.is_qubit_target for t in raw_t)
    if not has_special_targets:
        raw_u32 = np.array(
            [
                t.qubit_value | ((1 << 31) if t.is_inverted_result_target else 0)
                for t in raw_t
            ],
            dtype=np.uint32,
        )
        qubit_i32 = (raw_u32 & 0x7FFFFFFF).astype(np.int32)
        return raw_u32, qubit_i32, [], False
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
    qubit_vals: list[int | None] = [t.qubit_value for t in raw_t]
    qubit_i32 = np.array(
        [q if q is not None else t.value for q, t in zip(qubit_vals, raw_t)],
        dtype=np.int32,
    )
    return raw_u32, qubit_i32, qubit_vals, True


def _extract_op_targets_and_mask(
    op: stim.CircuitInstruction,
    op_name: str,
    params: LeakageParams | None,
    is_claimed: bool,
    buf: _PrecompileBuffers,
) -> tuple[
    stim.CircuitInstruction,
    list[Any],
    list[float],
    np.ndarray,
    np.ndarray,
    bool,
    bool,
    int,
    int,
    int,
]:
    raw_t = op.targets_copy()
    gate_args = op.gate_args_copy()
    raw_u32, qubit_i32, qubit_vals, has_special_targets = _decode_op_targets(
        raw_t, gate_args
    )
    bare_inst = (
        op if not is_claimed else stim.CircuitInstruction(op_name, raw_t, gate_args)
    )

    has_dup_targets = len(set(qubit_i32.tolist())) < len(qubit_i32)
    if isinstance(params, LeakageTransition1Params) and not has_dup_targets:
        unique_qs = np.unique(qubit_i32[qubit_i32 >= 0])
        qubit_i32 = unique_qs.astype(np.int32)
        raw_u32 = unique_qs.astype(np.uint32)

    mask_words = [0] * buf.num_words
    # With special targets, qubit_i32 holds .value for a combiner (0) or sweep[k] (k).
    mask_source: Iterable[Any] = (
        params.targets
        if (isinstance(params, LeakageConditioningParams) and params.targets)
        else qubit_i32
        if not has_special_targets
        else [q for q in qubit_vals if q is not None]
    )
    for q in mask_source:
        q_int = int(q)
        if 0 <= q_int < buf.num_qubits:
            mask_words[q_int >> 6] |= 1 << (q_int & 63)

    t_off = buf.t_running_len
    t_cnt = len(raw_u32)
    buf.t_running_len += t_cnt
    m_off = len(buf.all_step_masks)
    if t_cnt > 0:
        buf.raw_t_chunks.append(raw_u32)
        buf.qubit_t_chunks.append(qubit_i32)
    buf.all_step_masks.extend(mask_words)
    return (
        bare_inst, raw_t, gate_args, raw_u32, qubit_i32,
        has_special_targets, has_dup_targets, t_off, t_cnt, m_off,
    )


def _classify_conditioning_op(
    params: LeakageConditioningParams,
    has_special_targets: bool,
    target_count: int,
) -> tuple[int, bool]:
    cond_groups = params.args
    if any(0 in g or 1 in g for g in cond_groups):
        return 6, True
    if (
        not params.targets
        and not has_special_targets
        and len(cond_groups) == 1
        and cond_groups[0] == ("U",)
    ):
        return 1, False
    if (
        not params.targets
        and not has_special_targets
        and target_count % 2 == 0  # the pair filter reads the targets two at a time
        and len(cond_groups) == 2
        and cond_groups == (("U",), ("U",))
    ):
        return 2, False
    if all(g == ("U",) for g in cond_groups):
        return 8, False
    return 6, False


def _pack_lt1_step(
    params: LeakageTransition1Params,
    has_dup_targets: bool,
    all_branches: list[TransBranchC],
) -> tuple[int, bool, float, tuple[int, int]]:
    has_01 = (0 in params.args_by_input_state) or (1 in params.args_by_input_state)
    if has_01 or has_dup_targets:
        p1_tot = sum(p for _, p in params.args_by_input_state.get(1, ()))
        return 6, True, p1_tot, (0, 0)
    br_off = len(all_branches)
    u_args = params.args_by_input_state.get("U", ())
    p_tot_u = sum(p for _, p in u_args)
    cum = 0.0
    for out_st, p in u_args:
        cum += p / p_tot_u if p_tot_u > 0 else 0.0
        all_branches.append(TransBranchC(cum, -1, -1, _encode_state(out_st), 0, p))
    for in_st, br_list in params.args_by_input_state.items():
        if in_st != "U":
            for out_st, p in br_list:
                all_branches.append(
                    TransBranchC(0.0, int(in_st), -1, _encode_state(out_st), 0, p)
                )
    return 3, False, p_tot_u, (br_off, len(all_branches) - br_off)


def _pack_lt2_step(
    params: LeakageTransition2Params,
    has_overlapping_pairs: bool,
    all_branches: list[TransBranchC],
) -> tuple[int, bool, float, tuple[int, int]]:
    has_01 = any(0 in k or 1 in k for k in params.args_by_input_state)
    if has_01:
        return 6, True, 0.0, (0, 0)
    if has_overlapping_pairs:
        return 6, False, 0.0, (0, 0)
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
    return 4, False, p_tot_u, (br_off, len(all_branches) - br_off)


def _classify_meas_op(
    op: stim.CircuitInstruction,
    op_name: str,
    raw_t: list[Any],
    raw_u32: np.ndarray,
    params: LeakageMeasurementParams,
    has_dup_targets: bool,
) -> tuple[stim.CircuitInstruction, stim.CircuitInstruction, int, bool, str, float]:
    is_mpad = params.targets is not None
    p0, p1 = params.p0, params.p1
    p2 = params.prob_for_input_state.get(2, 0.0)
    meas_inst = stim.CircuitInstruction("M" if is_mpad else op_name, raw_t, [])
    ref_inst = op if is_mpad else meas_inst
    extra_states = set(params.prob_for_input_state) - {0, 1, 2}
    needs_tableau = _is_asymmetric_readout(params)
    if (
        not is_mpad
        and not has_dup_targets
        and op_name in ("M", "MZ")
        and p0 == 0.0
        and p1 == 1.0
        and not extra_states
    ):
        return meas_inst, ref_inst, 5, needs_tableau, format_stim_op_text("M", raw_u32), p2
    return meas_inst, ref_inst, 6, needs_tableau, "", p2


def _compile_uncached_op(
    op: stim.CircuitInstruction,
    op_name: str,
    buf: _PrecompileBuffers,
    use_supplied_ops: bool,
    supplied_ops: dict[stim.CircuitInstruction, LeakageParams] | None,
    out_parsed_ops: dict[stim.CircuitInstruction, LeakageParams],
) -> _StepTemplate:
    params: LeakageParams | None = None
    if op.tag != "":
        params = (
            supplied_ops.get(op)
            if use_supplied_ops and supplied_ops is not None
            else parse_leakage_tag(op, simulator="tableau")
        )
        if params is not None:
            out_parsed_ops[op] = params
    is_claimed = params is not None

    gate_data: Any = None
    if not is_claimed:
        gate_data = stim.gate_data(op_name)
        if (
            not buf.uc
            or gate_data.produces_measurements
            or not (gate_data.is_noisy_gate or gate_data.is_unitary)
        ):
            c_step_inst = StepDescC(0, 0, 0, 0, 0.0, 0, 0, 0.0)
            s_py: tuple[Any, ...] = (0, op_name, [], "", _EMPTY_I32, None)
            return op, op, c_step_inst, s_py

    (
        bare_inst, raw_t, gate_args, raw_u32, qubit_i32,
        has_special_targets, has_dup_targets, t_off, t_cnt, m_off,
    ) = _extract_op_targets_and_mask(op, op_name, params, is_claimed, buf)

    bare: stim.CircuitInstruction | None = bare_inst
    ref: stim.CircuitInstruction | None = bare_inst
    text, p_tot, br, p_leak = "", 0.0, (0, 0), 0.0
    if not is_claimed:
        if has_special_targets:
            kind = 8
        elif gate_data.is_single_qubit_gate:
            kind = 1
        else:
            kind = 2
    elif isinstance(params, LeakageConditioningParams):
        kind, needs_tab = _classify_conditioning_op(params, has_special_targets, t_cnt)
        buf.requires_tableau = buf.requires_tableau or needs_tab
    elif isinstance(params, LeakageTransition1Params):
        bare, ref = None, None
        kind, needs_tab, p_tot, br = _pack_lt1_step(params, has_dup_targets, buf.all_branches)
        buf.requires_tableau = buf.requires_tableau or needs_tab
    elif isinstance(params, LeakageTransition2Params):
        bare, ref = None, None
        kind, needs_tab, p_tot, br = _pack_lt2_step(params, has_dup_targets, buf.all_branches)
        buf.requires_tableau = buf.requires_tableau or needs_tab
    elif isinstance(params, LeakageMeasurementParams):
        bare, ref, kind, needs_tab, text, p_leak = _classify_meas_op(
            op, op_name, raw_t, raw_u32, params, has_dup_targets
        )
        buf.requires_tableau = buf.requires_tableau or needs_tab
    elif isinstance(params, LeakageSwapParams):
        # Runners yield to the handler when a target may be leaked; else run bare SWAP.
        kind = 8
    else:
        raise TypeError(f"Unsupported leakage parameter type: {type(params).__name__}")

    c_step = StepDescC(kind, t_off, t_cnt, m_off, p_tot, br[0], br[1], p_leak)
    s_py = (kind, op_name, gate_args, text, qubit_i32, params)
    return bare, ref, c_step, s_py


def _fuse_lt1_meas_steps(
    c_steps: list[StepDescC],
    steps_py: list[tuple[Any, ...]],
) -> None:
    """Fuse adjacent LT1(1-->2) and M[LEAKAGE_PROJECTION_Z: (1.0, 2)] steps (copy-on-write)."""
    for s in range(len(steps_py) - 1):
        if c_steps[s].kind == 6 and c_steps[s + 1].kind == 5:
            p_s = steps_py[s][5]
            p_m = steps_py[s + 1][5]
            if (
                isinstance(p_s, LeakageTransition1Params)
                and isinstance(p_m, LeakageMeasurementParams)
                and set(p_s.args_by_input_state) == {1}
                and len(p_s.args_by_input_state[1]) == 1
                and isinstance(p_s.args_by_input_state[1][0][0], int)
                and p_s.args_by_input_state[1][0][0] == 2
                and p_m.prob_for_input_state.get(2, 0.0) == 1.0
                and len(set(steps_py[s][4].tolist())) == len(steps_py[s][4])
                and np.array_equal(
                    np.sort(steps_py[s][4]),
                    np.sort(steps_py[s + 1][4]),
                )
            ):
                # Copy first: step descriptors are shared between identical ops via the cache.
                c_steps[s] = StepDescC.from_buffer_copy(c_steps[s])
                c_steps[s].kind = 7
                steps_py[s] = (7, *steps_py[s][1:])


def _pack_ctypes_buffers(
    c_steps: list[StepDescC],
    step_to_unrolled_idx: list[int],
    buf: _PrecompileBuffers,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any]:
    c_steps_arr = (StepDescC * len(c_steps))(*c_steps) if c_steps else None
    if c_steps_arr is not None:
        for s_i, u_idx in enumerate(step_to_unrolled_idx):
            c_steps_arr[s_i].unrolled_idx = u_idx
    raw_t_np = (
        np.concatenate(buf.raw_t_chunks)
        if buf.raw_t_chunks
        else np.empty(0, dtype=np.uint32)
    )
    qubit_t_np = (
        np.concatenate(buf.qubit_t_chunks)
        if buf.qubit_t_chunks
        else np.empty(0, dtype=np.int32)
    )
    masks_np = np.asarray(buf.all_step_masks, dtype=np.uint64)
    c_branches_arr = (
        (TransBranchC * len(buf.all_branches))(*buf.all_branches)
        if buf.all_branches
        else None
    )
    return c_steps_arr, raw_t_np, qubit_t_np, masks_np, c_branches_arr


def _ctypes_ptr(arr: np.ndarray, ctype: Any) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctype)) if len(arr) > 0 else None


def _finalize_precompiled_data(
    buf: _PrecompileBuffers,
    out_parsed_ops: dict[stim.CircuitInstruction, LeakageParams],
    ref_insts: list[stim.CircuitInstruction],
    steps_py: list[tuple[Any, ...]],
    orig_ops: list[stim.CircuitInstruction],
    full_bare_insts: list[stim.CircuitInstruction],
    quantum_bare_insts: list[stim.CircuitInstruction],
    step_to_full_idx: list[int],
    step_to_quant_idx: list[int],
    step_to_unrolled_idx: list[int],
    step_has_bare_op: list[bool],
    c_steps: list[StepDescC],
    unrolled_num_ops: int,
) -> PrecompiledTablesideData:
    num_steps = len(steps_py)
    z_collapses = collapses_z(orig_ops, out_parsed_ops)
    _fuse_lt1_meas_steps(c_steps, steps_py)
    c_steps_arr, raw_t_np, qubit_t_np, masks_np, c_branches_arr = _pack_ctypes_buffers(
        c_steps, step_to_unrolled_idx, buf
    )
    return PrecompiledTablesideData(
        num_qubits=buf.num_qubits,
        num_words=buf.num_words,
        unconditional_condition_on_U=buf.uc,
        parsed_ops=out_parsed_ops,
        final_qubit_coords=buf.final_qubit_coords,
        final_qubit_tags=buf.final_qubit_tags,
        final_coords_shifts=buf.final_coords_shifts,
        ref_insts=ref_insts,
        steps_py=steps_py,
        orig_ops=orig_ops,
        full_bare_insts=full_bare_insts,
        quantum_bare_insts=quantum_bare_insts,
        step_to_full_idx=step_to_full_idx,
        step_to_quant_idx=step_to_quant_idx,
        step_has_bare_op=step_has_bare_op,
        requires_tableau=buf.requires_tableau,
        num_steps=num_steps,
        collapses_z=z_collapses,
        unrolled_num_ops=unrolled_num_ops,
        step_to_unrolled_idx=step_to_unrolled_idx,
        c_steps=c_steps_arr,
        raw_t_np=raw_t_np,
        qubit_t_np=qubit_t_np,
        masks_np=masks_np,
        c_branches=c_branches_arr,
    )


def precompile_tableside_circuit(
    circuit: stim.Circuit,
    unconditional_condition_on_U: bool = True,
    parsed_ops: dict[stim.CircuitInstruction, LeakageParams] | None = None,
) -> PrecompiledTablesideData:
    """Purely precompile ``circuit`` into immutable step descriptors and backing buffers."""
    num_qubits = circuit.num_qubits
    num_words = max(1, (num_qubits + 63) // 64)
    has_repeat = any(isinstance(inst, stim.CircuitRepeatBlock) for inst in circuit)
    # Not circuit.flattened(): it fuses across repeat boundaries, desyncing step_to_unrolled_idx.
    unrolled_ops = unroll_circuit(circuit) if has_repeat else list(circuit)
    use_supplied_ops = parsed_ops is not None and all(
        isinstance(v, STANDARD_LEAKAGE_PARAM_TYPES) for v in parsed_ops.values()
    )

    out_parsed_ops: dict[stim.CircuitInstruction, LeakageParams] = {}
    ref_insts: list[stim.CircuitInstruction] = []
    steps_py: list[tuple[Any, ...]] = []
    orig_ops: list[stim.CircuitInstruction] = []
    full_bare_insts: list[stim.CircuitInstruction] = []
    quantum_bare_insts: list[stim.CircuitInstruction] = []
    step_to_full_idx: list[int] = []
    step_to_quant_idx: list[int] = []
    step_to_unrolled_idx: list[int] = []
    step_has_bare_op: list[bool] = []
    c_steps: list[StepDescC] = []
    buf = _PrecompileBuffers(num_qubits, num_words, unconditional_condition_on_U)
    op_template_cache: dict[stim.CircuitInstruction, _StepTemplate] = {}

    full_append, quant_append = full_bare_insts.append, quantum_bare_insts.append
    ref_append, orig_append = ref_insts.append, orig_ops.append
    s2f_append, s2q_append = step_to_full_idx.append, step_to_quant_idx.append
    s2u_append, has_bare_append = step_to_unrolled_idx.append, step_has_bare_op.append
    c_steps_append, steps_py_append = c_steps.append, steps_py.append
    cache_get = op_template_cache.get

    for u_idx, op in enumerate(unrolled_ops):
        op_name = op.name
        if op_name in ("DETECTOR", "TICK", "OBSERVABLE_INCLUDE"):
            full_append(op)
            ref_append(op)
            continue
        if (op_name == "QUBIT_COORDS" or op_name == "SHIFT_COORDS") and apply_coords_op(
            op, buf.final_qubit_coords, buf.final_qubit_tags, buf.final_coords_shifts
        ):
            full_append(op)
            ref_append(op)
            continue

        s2f_append(len(full_bare_insts))
        s2q_append(len(quantum_bare_insts))
        s2u_append(u_idx)
        orig_append(op)

        tpl = cache_get(op)
        if tpl is None:
            tpl = _compile_uncached_op(
                op, op_name, buf, use_supplied_ops, parsed_ops, out_parsed_ops
            )
            op_template_cache[op] = tpl
        bare, ref, c_step, s_py = tpl
        if bare is not None:
            full_append(bare)
            quant_append(bare)
        if ref is not None:
            ref_append(ref)
        has_bare_append(bare is not None)
        c_steps_append(c_step)
        steps_py_append(s_py)

    step_to_full_idx.append(len(full_bare_insts))
    step_to_quant_idx.append(len(quantum_bare_insts))
    return _finalize_precompiled_data(
        buf, out_parsed_ops, ref_insts, steps_py, orig_ops, full_bare_insts,
        quantum_bare_insts, step_to_full_idx, step_to_quant_idx, step_to_unrolled_idx,
        step_has_bare_op, c_steps, len(unrolled_ops),
    )


class PrecompiledTablesideCircuit:
    """Pre-compiled circuit structure for TablesideSimulator runners."""

    def __init__(
        self,
        circuit: stim.Circuit,
        unconditional_condition_on_U: bool = True,
        parsed_ops: dict[stim.CircuitInstruction, LeakageParams] | None = None,
    ) -> None:
        data = precompile_tableside_circuit(
            circuit, unconditional_condition_on_U=unconditional_condition_on_U,
            parsed_ops=parsed_ops,
        )
        self.circuit, self.parsed_ops = circuit, data.parsed_ops
        self.num_qubits, self.num_words = data.num_qubits, data.num_words
        self.unconditional_condition_on_U = data.unconditional_condition_on_U
        self.final_qubit_coords = data.final_qubit_coords
        self.final_qubit_tags = data.final_qubit_tags
        self.final_coords_shifts = data.final_coords_shifts
        self.ref_insts, self.steps_py, self.orig_ops = data.ref_insts, data.steps_py, data.orig_ops
        self.full_bare_insts = data.full_bare_insts
        self.quantum_bare_insts = data.quantum_bare_insts
        self.step_to_full_idx = data.step_to_full_idx
        self.step_to_quant_idx = data.step_to_quant_idx
        self.step_has_bare_op, self.requires_tableau = data.step_has_bare_op, data.requires_tableau
        self.num_steps, self.collapses_z = data.num_steps, data.collapses_z
        self.unrolled_num_ops = data.unrolled_num_ops
        self.step_to_unrolled_idx = data.step_to_unrolled_idx
        self._c_steps, self._c_branches = data.c_steps, data.c_branches
        self._raw_t_np, self._qubit_t_np = data.raw_t_np, data.qubit_t_np
        self._masks_np = data.masks_np
        self._c_raw_t = _ctypes_ptr(self._raw_t_np, ctypes.c_uint32)
        self._c_qubit_t = _ctypes_ptr(self._qubit_t_np, ctypes.c_int32)
        self._c_masks = _ctypes_ptr(self._masks_np, ctypes.c_uint64)
        self._m2d_converter: Any = None
        self._reference_circuit: stim.Circuit | None = None
        self._full_bare_circuit: stim.Circuit | None = None
        self._quantum_bare_circuit: stim.Circuit | None = None
        self._full_slice_cache: dict[tuple[int, int], stim.Circuit] = {}
        self._quant_slice_cache: dict[tuple[int, int], stim.Circuit] = {}
        self._full_blocks: list[stim.Circuit] = []
        self._quant_blocks: list[stim.Circuit] = []

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
    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
        if len(_PRECOMPILED_CACHE) >= 4 and key not in _PRECOMPILED_CACHE:
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


_XY_BASIS_COLLAPSE = frozenset({"MX", "MY", "MRX", "MRY", "RX", "RY", "MXX", "MYY"})


def do_sync_bare_inst(tab: stim.TableauSimulator, inst: stim.CircuitInstruction) -> None:
    args = inst.gate_args_copy()
    if inst.name in _XY_BASIS_COLLAPSE and (not args or args[0] == 0.0):
        for g in inst.target_groups():  # stim collapses fused X/Y targets in sorted order
            tab.do(stim.CircuitInstruction(inst.name, g))
    else:
        tab.do(inst)


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
                for inst in self.pre.quantum_bare_insts[self.q : q_end]:
                    do_sync_bare_inst(tab, inst)
            else:
                tab.do_circuit(self.pre.get_quant_slice(self.q, q_end))
            self.q = q_end

    def flush(self, f_end: int, q_end: int) -> None:
        if f_end > self.f:
            self.flush_full(f_end)
        if self.use_tab:
            self.flush_quant(q_end)

    def emit(self, s_idx: int, txt: str, consumes: int) -> None:
        f_s = self.pre.step_to_full_idx[s_idx]
        if f_s > self.f:
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
        self.tss.qubit_coords = {q: list(v) for q, v in self.pre.final_qubit_coords.items()}
        self.tss.qubit_tags = dict(self.pre.final_qubit_tags)
        self.tss.coords_shifts = list(self.pre.final_coords_shifts)
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
        state_addr = ctypes.cast(state_ptr, ctypes.c_void_p).value
        assert state_addr is not None
        c_buf = (ctypes.c_uint8 * n_alloc).from_address(state_addr)
        c_buf._engine_owner = owner  # type: ignore[attr-defined]
        raw_state = np.ctypeslib.as_array(c_buf)
        self._raw_state = raw_state
        self.state = raw_state if pre.num_qubits == n_alloc else raw_state[: pre.num_qubits]
        del owner, c_buf, raw_state
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
        has_arg = 1 if len(gate_args) == 1 else 0
        arg0 = float(gate_args[0]) if has_arg else 0.0
        needed = len(op_name) + 64 + count * 12
        if len(self._fmt_buf) < needed:
            self._fmt_buf = ctypes.create_string_buffer(max(65536, needed))
        n_bytes = self.lib.fast_format_stim_op(
            op_name.encode("ascii"),
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
                            tss._record_unleaked_to_leaked_counts(
                                np.array([len(newly_leaked)]), pre.step_to_unrolled_idx[s_trans]
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
