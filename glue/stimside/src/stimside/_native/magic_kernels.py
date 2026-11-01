"""Shared-library loader and ctypes ABI bindings for `magic_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from stimside._native._kernel_loader import (
    find_prebuilt_so as _find_prebuilt_so_candidate,
    load_cdll,
    resolve_so,
)

_LIB_INSTANCE: ctypes.CDLL | None = None

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_u64_p = ctypes.POINTER(ctypes.c_uint64)
_c_u8_p = ctypes.POINTER(ctypes.c_uint8)
_c_i32_p = ctypes.POINTER(ctypes.c_int32)
_c_size_p = ctypes.POINTER(ctypes.c_size_t)


def _configure_abi(lib: ctypes.CDLL) -> None:
    sz, c_u8, c_int = ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int
    c_i32, c_dbl = ctypes.c_int32, ctypes.c_double
    specs: tuple[tuple[str, list[Any], Any], ...] = (
        ("magic_abi_v1", [], c_i32),
        (
            "magic_apply_pauli_batch",
            [sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u64_p, _c_u64_p, c_u8],
            None,
        ),
        (
            "magic_apply_quarter_turn",
            [
                sz, sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
                _c_u64_p, _c_u64_p, c_u8, _c_size_p, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p,
                _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
            ],
            c_int,
        ),
        (
            "magic_canonicalize_and_merge",
            [
                sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
                c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
            ],
            None,
        ),
        (
            "magic_branch_pauli_t",
            [
                sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
                _c_u64_p, _c_u64_p, c_u8, c_int, c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
            ],
            None,
        ),
        (
            "magic_branch_ccz",
            [
                sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
                _c_double_p, _c_u64_p, _c_u64_p, _c_u8_p, c_dbl, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p, _c_double_p,
            ],
            None,
        ),
        (
            "magic_project_diagonal",
            [
                sz, sz, sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
                _c_u64_p, c_u8, c_u8, c_dbl, _c_size_p, _c_size_p, _c_double_p, _c_u64_p, _c_u8_p,
                _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p, _c_double_p,
            ],
            None,
        ),
    )
    for fn_name, argtypes, restype in specs:
        fn = getattr(lib, fn_name)
        fn.argtypes = argtypes
        fn.restype = restype


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile, _find_prebuilt_so_candidate)


def get_magic_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for `magic_kernels.cpp`."""
    global _LIB_INSTANCE
    if not force_recompile and _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    if force_recompile:
        _LIB_INSTANCE = None

    resolver = (
        (lambda force_recompile=True: _resolve_magic_kernels_so(force_recompile=True))
        if force_recompile
        else _resolve_magic_kernels_so
    )
    lib = load_cdll(resolver, "magic_abi_v1")
    _configure_abi(lib)
    _LIB_INSTANCE = lib
    return lib
