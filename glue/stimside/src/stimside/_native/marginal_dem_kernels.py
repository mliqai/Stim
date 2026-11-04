"""Shared-library loader for `marginal_dem_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

from stimside._native._kernel_loader import (
    AbiSpec,
    c_ptr,
    f64_p,
    get_lib,
    i32_p,
    resolve_so,
    u8_p,
    void_p,
)

MOP_NONE = 0
MOP_MEAS_FLAG = 1
MOP_MEAS_PROJ_Z = 2
MOP_TRANS_1 = 3
MOP_TRANS_2 = 4
MOP_UNTAGGED_1Q_UNITARY = 5
MOP_UNTAGGED_2Q_UNITARY = 6
MOP_BARE_NOOP = 7
MOP_BARE_1Q_UNITARY = 8
MOP_BARE_RESET = 9
MOP_BARE_OTHER = 10
MOP_CONDITIONED = 11


class OpDescC(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("target_offset", ctypes.c_int32),
        ("target_count", ctypes.c_int32),
        ("aux_offset", ctypes.c_int32),
        ("aux_count", ctypes.c_int32),
        ("gate_id", ctypes.c_int32),
        ("reset_basis", ctypes.c_int32),
        ("ctrl_leg0_basis", ctypes.c_int32),
        ("ctrl_leg1_basis", ctypes.c_int32),
        ("cond_base_fires", ctypes.c_int32),
        ("cond_subkind", ctypes.c_int32),
        ("cond_channel_id", ctypes.c_int32),
    ]


class SourceDescC(ctypes.Structure):
    _fields_ = [
        ("op_index", ctypes.c_int32),
        ("group", ctypes.c_int32),
        ("qubit", ctypes.c_int32),
        ("state", ctypes.c_int32),
        ("weight", ctypes.c_double),
        ("partner", ctypes.c_int32),
        ("partner_channel_id", ctypes.c_int32),
        ("init_tracked", ctypes.c_int32),
    ]


class Trans1RuleC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int32),
        ("p_unleak", ctypes.c_double),
        ("next_state", ctypes.c_int32),
        ("fork_offset", ctypes.c_int32),
        ("fork_count", ctypes.c_int32),
    ]


class Trans2RuleC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int32),
        ("leg", ctypes.c_int32),
        ("p_clear", ctypes.c_double),
        ("p_move", ctypes.c_double),
        ("move_offset", ctypes.c_int32),
        ("move_count", ctypes.c_int32),
        ("max_move_state", ctypes.c_int32),
        ("stay_next_state", ctypes.c_int32),
        ("channel_id", ctypes.c_int32),
        ("fork_offset", ctypes.c_int32),
        ("fork_count", ctypes.c_int32),
        ("fork_channel_id", ctypes.c_int32),
        ("spread_offset", ctypes.c_int32),
        ("spread_count", ctypes.c_int32),
    ]


class MoveBranchC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int32),
        ("prob", ctypes.c_double),
        ("channel_id", ctypes.c_int32),
    ]


class MeasStateProbC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int32),
        ("prob", ctypes.c_double),
    ]


class CondFireRuleC(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int32),
        ("mask", ctypes.c_int32),
        ("fires", ctypes.c_int32),
    ]


_LIB_INSTANCE: ctypes.CDLL | None = None
_c_i32 = ctypes.c_int32

_TRACE_FLOWS_ARGS = [
    _c_i32,
    _c_i32,
    _c_i32,
    ctypes.POINTER(OpDescC),
    i32_p,
    i32_p,
    i32_p,
    i32_p,
    ctypes.POINTER(MeasStateProbC),
    ctypes.POINTER(Trans1RuleC),
    ctypes.POINTER(Trans2RuleC),
    ctypes.POINTER(MoveBranchC),
    ctypes.POINTER(CondFireRuleC),
    _c_i32,
    ctypes.POINTER(SourceDescC),
]

_SPECS: tuple[AbiSpec, ...] = (
    ("marginal_trace_flows", _TRACE_FLOWS_ARGS, void_p),
    ("marginal_trace_source_flows", _TRACE_FLOWS_ARGS, void_p),
    ("marginal_trace_result_get_counts", [void_p, i32_p, i32_p, i32_p, i32_p, i32_p], None),
    (
        "marginal_trace_result_copy_data",
        [void_p, i32_p, i32_p, i32_p, i32_p, i32_p, f64_p, i32_p, i32_p],
        None,
    ),
    ("marginal_free_trace_result", [void_p], None),
    (
        "marginal_build_envelopes",
        [_c_i32, _c_i32, _c_i32, i32_p, i32_p, f64_p, _c_i32, i32_p, i32_p, _c_i32, i32_p, i32_p, f64_p],
        void_p,
    ),
    ("marginal_envelope_result_get_total_entries", [void_p], _c_i32),
    ("marginal_envelope_result_copy_data", [void_p, i32_p, i32_p, f64_p], None),
    ("marginal_free_envelope_result", [void_p], None),
    (
        "marginal_process_shots",
        [
            _c_i32, _c_i32, u8_p, i32_p, i32_p, i32_p, i32_p, f64_p,
            _c_i32, _c_i32, f64_p, _c_i32, i32_p, i32_p, i32_p, f64_p,
        ],
        void_p,
    ),
    ("marginal_batch_result_get_counts", [void_p, i32_p, i32_p], None),
    ("marginal_batch_result_copy_data", [void_p, i32_p, i32_p, i32_p, f64_p, f64_p], None),
    ("marginal_free_batch_result", [void_p], None),
)


def _resolve_marginal_dem_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "marginal_dem_kernels", force_recompile)


def get_marginal_dem_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for marginal_dem_kernels.cpp."""
    return get_lib(sys.modules[__name__], "marginal_dem_kernels", "marginal_dem_kernels_abi_v5", _SPECS, force_recompile=force_recompile)


def trace_flows_cpp(
    num_ops: int,
    num_qubits: int,
    num_flags: int,
    ops_arr: ctypes.Array[OpDescC],
    target_qubits: NDArray[np.int32],
    touch_offsets: NDArray[np.int32],
    touch_indices: NDArray[np.int32],
    conj_table: NDArray[np.int32],
    meas_probs_arr: ctypes.Array[MeasStateProbC],
    trans1_rules_arr: ctypes.Array[Trans1RuleC],
    trans2_rules_arr: ctypes.Array[Trans2RuleC],
    move_branches_arr: ctypes.Array[MoveBranchC],
    cond_rules_arr: ctypes.Array[CondFireRuleC],
    sources_arr: ctypes.Array[SourceDescC],
    source_mode: bool = False,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Trace the leakage flows (``marginal_trace_flows``).

    With ``source_mode=True`` calls ``marginal_trace_source_flows`` instead: the
    returned "flag" arrays are per source, and ``num_flags`` must be the number of sources.
    """
    lib = get_marginal_dem_kernels_lib()
    trace_fn = lib.marginal_trace_source_flows if source_mode else lib.marginal_trace_flows
    res_ptr = trace_fn(
        ctypes.c_int32(num_ops),
        ctypes.c_int32(num_qubits),
        ctypes.c_int32(num_flags),
        ops_arr,
        c_ptr(target_qubits, i32_p),
        c_ptr(touch_offsets, i32_p),
        c_ptr(touch_indices, i32_p),
        c_ptr(conj_table, i32_p),
        meas_probs_arr,
        trans1_rules_arr,
        trans2_rules_arr,
        move_branches_arr,
        cond_rules_arr,
        ctypes.c_int32(len(sources_arr)),
        sources_arr,
    )
    try:
        c_num_sites = ctypes.c_int32(0)
        c_num_used_flows = ctypes.c_int32(0)
        c_total_flow_sites = ctypes.c_int32(0)
        c_total_cands = ctypes.c_int32(0)
        c_total_preds = ctypes.c_int32(0)
        lib.marginal_trace_result_get_counts(
            res_ptr,
            ctypes.byref(c_num_sites),
            ctypes.byref(c_num_used_flows),
            ctypes.byref(c_total_flow_sites),
            ctypes.byref(c_total_cands),
            ctypes.byref(c_total_preds),
        )
        num_sites = int(c_num_sites.value)
        num_used_flows = int(c_num_used_flows.value)
        total_flow_sites = int(c_total_flow_sites.value)
        total_cands = int(c_total_cands.value)
        total_preds = int(c_total_preds.value)

        sites = np.empty((num_sites, 5), dtype=np.int32)
        flow_site_offsets = np.empty(num_used_flows + 1, dtype=np.int32)
        flow_site_ids = np.empty(total_flow_sites, dtype=np.int32)
        flag_cand_offsets = np.empty(num_flags + 1, dtype=np.int32)
        flag_cand_flows = np.empty(total_cands, dtype=np.int32)
        flag_cand_weights = np.empty(total_cands, dtype=np.float64)
        pred_offsets = np.empty(num_flags + 1, dtype=np.int32)
        pred_flags = np.empty(total_preds, dtype=np.int32)

        lib.marginal_trace_result_copy_data(
            res_ptr,
            c_ptr(sites, i32_p),
            c_ptr(flow_site_offsets, i32_p),
            c_ptr(flow_site_ids, i32_p),
            c_ptr(flag_cand_offsets, i32_p),
            c_ptr(flag_cand_flows, i32_p),
            c_ptr(flag_cand_weights, f64_p),
            c_ptr(pred_offsets, i32_p),
            c_ptr(pred_flags, i32_p),
        )
        return (
            sites,
            flow_site_offsets,
            flow_site_ids,
            flag_cand_offsets,
            flag_cand_flows,
            flag_cand_weights,
            pred_offsets,
            pred_flags,
        )
    finally:
        lib.marginal_free_trace_result(res_ptr)


def build_envelopes_cpp(
    num_sites: int,
    num_symptoms: int,
    entry_site_ids: NDArray[np.int32],
    entry_sym_ids: NDArray[np.int32],
    entry_probs: NDArray[np.float64],
    num_used_flows: int,
    flow_site_offsets: NDArray[np.int32],
    flow_site_ids: NDArray[np.int32],
    num_flags: int,
    flag_cand_offsets: NDArray[np.int32],
    flag_cand_flows: NDArray[np.int32],
    flag_cand_weights: NDArray[np.float64],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    lib = get_marginal_dem_kernels_lib()
    res_ptr = lib.marginal_build_envelopes(
        ctypes.c_int32(num_sites),
        ctypes.c_int32(num_symptoms),
        ctypes.c_int32(entry_site_ids.size),
        c_ptr(entry_site_ids, i32_p),
        c_ptr(entry_sym_ids, i32_p),
        c_ptr(entry_probs, f64_p),
        ctypes.c_int32(num_used_flows),
        c_ptr(flow_site_offsets, i32_p),
        c_ptr(flow_site_ids, i32_p),
        ctypes.c_int32(num_flags),
        c_ptr(flag_cand_offsets, i32_p),
        c_ptr(flag_cand_flows, i32_p),
        c_ptr(flag_cand_weights, f64_p),
    )
    try:
        total_entries = int(lib.marginal_envelope_result_get_total_entries(res_ptr))
        flag_env_offsets = np.empty(num_flags + 1, dtype=np.int32)
        flag_env_sym_ids = np.empty(total_entries, dtype=np.int32)
        flag_env_probs = np.empty(total_entries, dtype=np.float64)
        lib.marginal_envelope_result_copy_data(
            res_ptr,
            c_ptr(flag_env_offsets, i32_p),
            c_ptr(flag_env_sym_ids, i32_p),
            c_ptr(flag_env_probs, f64_p),
        )
        return flag_env_offsets, flag_env_sym_ids, flag_env_probs
    finally:
        lib.marginal_free_envelope_result(res_ptr)


def process_shots_cpp(
    raised: NDArray[np.uint8],
    pred_offsets: NDArray[np.int32],
    pred_flags: NDArray[np.int32],
    flag_env_offsets: NDArray[np.int32],
    flag_env_sym_ids: NDArray[np.int32],
    flag_env_probs: NDArray[np.float64],
    num_symptoms: int,
    mode: int,
    base_sym_probs: NDArray[np.float64] | None = None,
    sym_edge_offsets: NDArray[np.int32] | None = None,
    sym_edge_ids: NDArray[np.int32] | None = None,
    edge_nodes: NDArray[np.int32] | None = None,
    base_edge_probs: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    lib = get_marginal_dem_kernels_lib()
    raised_c = np.ascontiguousarray(raised, dtype=np.uint8)
    num_shots, num_flags = raised_c.shape
    if num_flags + 1 != pred_offsets.size or num_flags + 1 != flag_env_offsets.size:
        raise ValueError(
            f"Expected raised with {pred_offsets.size - 1} flag columns, got {num_flags}."
        )
    num_edges = 0 if base_edge_probs is None else int(base_edge_probs.size)

    res_ptr = lib.marginal_process_shots(
        ctypes.c_int32(num_shots),
        ctypes.c_int32(num_flags),
        c_ptr(raised_c, u8_p),
        c_ptr(pred_offsets, i32_p),
        c_ptr(pred_flags, i32_p),
        c_ptr(flag_env_offsets, i32_p),
        c_ptr(flag_env_sym_ids, i32_p),
        c_ptr(flag_env_probs, f64_p),
        ctypes.c_int32(num_symptoms),
        ctypes.c_int32(mode),
        c_ptr(base_sym_probs, f64_p),
        ctypes.c_int32(num_edges),
        c_ptr(sym_edge_offsets, i32_p),
        c_ptr(sym_edge_ids, i32_p),
        c_ptr(edge_nodes, i32_p),
        c_ptr(base_edge_probs, f64_p),
    )
    try:
        c_num_groups = ctypes.c_int32(0)
        c_total_items = ctypes.c_int32(0)
        lib.marginal_batch_result_get_counts(
            res_ptr,
            ctypes.byref(c_num_groups),
            ctypes.byref(c_total_items),
        )
        num_groups = int(c_num_groups.value)
        total_items = int(c_total_items.value)
        shot_group_ids = np.empty(num_shots, dtype=np.int32)
        group_item_offsets = np.empty(num_groups + 1, dtype=np.int32)
        group_sym_ids = None if mode == 2 else np.empty(total_items, dtype=np.int32)
        group_probs = None if mode == 2 else np.empty(total_items, dtype=np.float64)
        group_reweights = np.empty((total_items, 3), dtype=np.float64) if mode == 2 else None
        lib.marginal_batch_result_copy_data(
            res_ptr,
            c_ptr(shot_group_ids, i32_p),
            c_ptr(group_item_offsets, i32_p),
            c_ptr(group_sym_ids, i32_p),
            c_ptr(group_probs, f64_p),
            c_ptr(group_reweights, f64_p),
        )
        return shot_group_ids, group_item_offsets, group_sym_ids, group_probs, group_reweights
    finally:
        lib.marginal_free_batch_result(res_ptr)
