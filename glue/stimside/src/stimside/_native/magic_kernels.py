"""Shared-library loader and ctypes ABI bindings for `magic_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from stimside._native._kernel_loader import (
    AbiSpec,
    f64_p as _c_double_p,
    find_prebuilt_so as _find_prebuilt_so_candidate,
    get_lib,
    i32_p as _c_i32_p,
    resolve_so,
    size_p as _c_size_p,
    u8_p as _c_u8_p,
    u64_p as _c_u64_p,
)

_LIB_INSTANCE: ctypes.CDLL | None = None

_sz, _c_u8, _c_int = ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int
_c_i32, _c_dbl = ctypes.c_int32, ctypes.c_double

_SPECS: tuple[AbiSpec, ...] = (
    ("magic_abi_v1", [], _c_i32),
    (
        "magic_apply_pauli_batch",
        [_sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u64_p, _c_u64_p, _c_u8],
        None,
    ),
    (
        "magic_apply_quarter_turn",
        [
            _sz, _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _c_u64_p, _c_u64_p, _c_u8, _c_size_p, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p,
            _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
        ],
        _c_int,
    ),
    (
        "magic_canonicalize_and_merge",
        [
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
        ],
        None,
    ),
    (
        "magic_branch_pauli_t",
        [
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _c_u64_p, _c_u64_p, _c_u8, _c_int, _c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
        ],
        None,
    ),
    (
        "magic_branch_ccz",
        [
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _c_double_p, _c_u64_p, _c_u64_p, _c_u8_p, _c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
        ],
        None,
    ),
    (
        "magic_project_diagonal",
        [
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _c_u64_p, _c_u8, _c_u8, _c_dbl, _c_size_p, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p,
            _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p, _c_double_p,
        ],
        None,
    ),
    ("magic_abi_v2", [], _c_i32),
    (
        "magic_contract_choi",
        [
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            _sz, _c_i32_p, _sz, _c_int, _c_dbl,
            _c_size_p, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
        ],
        _c_int,
    ),
)


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile, _find_prebuilt_so_candidate)


def get_magic_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for `magic_kernels.cpp`."""
    return get_lib(
        sys.modules[__name__],
        "magic_kernels",
        "magic_abi_v2",
        _SPECS,
        force_recompile=force_recompile,
    )
