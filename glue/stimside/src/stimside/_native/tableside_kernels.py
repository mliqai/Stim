"""Ctypes step descriptors and shared-library loader for `tableside_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from stimside._native._kernel_loader import (
    AbiSpec,
    find_prebuilt_so as _find_prebuilt_so_candidate,
    get_lib,
    i32_p,
    resolve_so,
    u8_p,
    u32_p,
    u64_p,
    void_p,
)


class StepDescC(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("target_offset", ctypes.c_int32),
        ("target_count", ctypes.c_int32),
        ("mask_offset", ctypes.c_int32),
        ("p_total_u", ctypes.c_double),
        ("trans_offset", ctypes.c_int32),
        ("trans_count", ctypes.c_int32),
        ("meas_leak_prob", ctypes.c_double),
        ("unrolled_idx", ctypes.c_int32),
    ]


class TransBranchC(ctypes.Structure):
    _fields_ = [
        ("cum_prob", ctypes.c_double),
        ("in0", ctypes.c_int8),
        ("in1", ctypes.c_int8),
        ("out0", ctypes.c_int8),
        ("out1", ctypes.c_int8),
        ("raw_prob", ctypes.c_double),
    ]


class EmittedActionC(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("step_idx", ctypes.c_int32),
        ("slice_end", ctypes.c_int32),
        ("target_offset", ctypes.c_int32),
        ("target_count", ctypes.c_int32),
        ("prob", ctypes.c_double),
    ]


_LIB_INSTANCE: ctypes.CDLL | None = None
_i32_pp = ctypes.POINTER(i32_p)

_SPECS: tuple[AbiSpec, ...] = (
    (
        "fast_engine_create",
        [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(StepDescC),
            ctypes.c_int,
            u32_p,
            i32_p,
            ctypes.c_int,
            u64_p,
            ctypes.c_int,
            ctypes.POINTER(TransBranchC),
        ],
        void_p,
    ),
    ("fast_engine_destroy", [void_p], None),
    ("fast_engine_clear", [void_p, ctypes.c_uint64], None),
    ("fast_engine_set_record_leakage", [void_p, ctypes.c_int], None),
    ("fast_engine_get_segment_leaked_ops", [void_p, _i32_pp], ctypes.c_int),
    ("fast_engine_set_record_events", [void_p, ctypes.c_int], None),
    ("fast_engine_get_segment_events", [void_p, _i32_pp], ctypes.c_int),
    ("fast_engine_get_state_ptr", [void_p], u8_p),
    ("fast_engine_sync_leaked_mask", [void_p], None),
    (
        "fast_engine_run_segment",
        [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(EmittedActionC)),
            ctypes.POINTER(u32_p),
        ],
        ctypes.c_int,
    ),
    (
        "fast_format_stim_op",
        [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, void_p, ctypes.c_int, ctypes.c_char_p],
        ctypes.c_int,
    ),
)


def _resolve_tableside_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "tableside_kernels", force_recompile, _find_prebuilt_so_candidate)


def get_tableside_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for tableside_kernels.cpp."""
    return get_lib(
        sys.modules[__name__],
        "tableside_kernels",
        "tableside_kernels_abi_v1",
        _SPECS,
        force_recompile=force_recompile,
    )
