"""Shared-library loader for `coset_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from stimside._native._kernel_loader import (
    AbiSpec,
    get_lib,
    i8_p,
    i32_p,
    i64_p,
    int_p,
    resolve_so,
    u8_p,
    u64_p,
    void_p,
)

_c_int = ctypes.c_int

_LIB_INSTANCE: ctypes.CDLL | None = None

_SPECS: tuple[AbiSpec, ...] = (
    ("engine_create", [_c_int, _c_int, u64_p], void_p),
    ("engine_destroy", [void_p], None),
    ("engine_set_rng_states", [void_p, u64_p], None),
    ("engine_clear", [void_p], None),
    ("engine_get_active_shot_indices", [void_p, i32_p, i32_p], _c_int),
    ("engine_export_shot", [void_p, _c_int, u64_p, u64_p, u64_p, u8_p, u8_p, i32_p], None),
    ("engine_import_shot", [void_p, _c_int, _c_int, u64_p, u64_p, u64_p, u8_p, u8_p, i32_p], None),
    (
        "engine_batch_inject_general_clifford",
        [
            void_p, _c_int, _c_int, _c_int, i64_p, u8_p, u8_p, u8_p, u8_p, u8_p, u8_p,
            u64_p, u64_p, u8_p, u8_p, u8_p, u64_p, u64_p, u8_p, u64_p, u64_p, u8_p, u8_p,
            int_p, int_p,
        ],
        None,
    ),
    (
        "engine_batch_det_meas_step",
        [void_p, _c_int, i64_p, u8_p, u64_p, u8_p, u8_p, u64_p, u64_p, u8_p, u8_p, int_p, int_p],
        None,
    ),
    (
        "engine_batch_mixed_substep",
        [
            void_p, _c_int, ctypes.c_int64, ctypes.c_uint8, ctypes.c_int64,
            u64_p, u64_p, u64_p, _c_int, _c_int, i64_p, _c_int, i64_p,
            u8_p, u8_p, u8_p, u8_p, int_p, int_p,
        ],
        None,
    ),
    ("engine_batch_evict", [void_p, u64_p, u64_p, u8_p, u8_p, u8_p, u8_p, int_p, int_p], None),
    (
        "engine_batch_unmatched_noisy_reset_z",
        [void_p, u8_p, u8_p, u8_p, u64_p, u64_p, u8_p, u64_p, u8_p, u8_p, int_p, int_p],
        None,
    ),
    (
        "engine_peek_pauli_batch",
        [void_p, _c_int, i64_p, _c_int, u8_p, u8_p, u64_p, u64_p, u64_p, u64_p, u8_p, u8_p, i8_p],
        None,
    ),
)


def _resolve_coset_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "coset_kernels", force_recompile)


def get_coset_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for coset_kernels.cpp."""
    return get_lib(
        sys.modules[__name__],
        "coset_kernels",
        "coset_kernels_abi_v2",
        _SPECS,
        force_recompile=force_recompile,
    )
