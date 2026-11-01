"""Ctypes step descriptors and shared-library loader for `tableside_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path

from stimside._native._kernel_loader import (
    find_prebuilt_so as _find_prebuilt_so_candidate,
    load_cdll,
    resolve_so,
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


def _resolve_tableside_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "tableside_kernels", force_recompile, _find_prebuilt_so_candidate)


def get_tableside_kernels_lib() -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for tableside_kernels.cpp."""
    global _LIB_INSTANCE
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    lib = load_cdll(_resolve_tableside_kernels_so, "fast_engine_get_segment_events")
    lib.fast_engine_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(StepDescC),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.POINTER(TransBranchC),
    ]
    lib.fast_engine_create.restype = ctypes.c_void_p
    lib.fast_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.fast_engine_destroy.restype = None
    lib.fast_engine_clear.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.fast_engine_clear.restype = None
    lib.fast_engine_set_record_leakage.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fast_engine_set_record_leakage.restype = None
    lib.fast_engine_get_segment_leaked_ops.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ]
    lib.fast_engine_get_segment_leaked_ops.restype = ctypes.c_int
    lib.fast_engine_set_record_events.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fast_engine_set_record_events.restype = None
    lib.fast_engine_get_segment_events.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    ]
    lib.fast_engine_get_segment_events.restype = ctypes.c_int
    lib.fast_engine_get_state_ptr.argtypes = [ctypes.c_void_p]
    lib.fast_engine_get_state_ptr.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.fast_engine_sync_leaked_mask.argtypes = [ctypes.c_void_p]
    lib.fast_engine_sync_leaked_mask.restype = None
    lib.fast_engine_run_segment.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(EmittedActionC)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
    ]
    lib.fast_engine_run_segment.restype = ctypes.c_int
    lib.fast_format_stim_op.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.fast_format_stim_op.restype = ctypes.c_int
    _LIB_INSTANCE = lib
    return lib
