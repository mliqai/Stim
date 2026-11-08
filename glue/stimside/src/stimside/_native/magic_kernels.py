"""Shared-library loader and ctypes ABI bindings for `magic_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from typing import Any

from stimside._native._kernel_loader import AbiSpec, get_lib, resolve_so

_LIB_INSTANCE: ctypes.CDLL | None = None
_find_prebuilt_so_candidate = None  # kept for e2e T1-F11-01 / T2-F11-03 compatibility

# Every pointer parameter is bound as c_void_p: callers pass integer addresses (`addr`/`w_addr`),
# which skips building a typed ctypes pointer per argument per call. Typed pointers and `byref`
# objects are still accepted. The C symbols and the ABI sentinel are unchanged.
_c_double_p = _c_u64_p = _c_u8_p = _c_i32_p = _c_i64_p = _c_size_p = ctypes.c_void_p
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
        "magic_apply_quarter_turn_tol",
        [
            _sz, _sz, _sz, _sz, _c_double_p, _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p,
            _c_u8_p, _c_u64_p, _c_u64_p, _c_u8, _c_dbl, _c_size_p, _c_size_p, _c_double_p,
            _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p,
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
    ("magic_abi_v3", [], _c_i32),
    (
        "magic_measure_batch",
        [
            _sz, _sz, _sz, _c_i64_p, _c_i64_p, _c_u8_p, _c_double_p, _c_double_p, _c_u64_p,
            _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p, _c_int, _c_u64_p, _c_u64_p, _c_u8,
            _c_u64_p, _c_u8, _c_dbl, _c_dbl, _c_u8_p, _c_double_p, _c_i64_p, _c_double_p,
            _c_u64_p, _c_u8_p, _c_u64_p, _c_u8_p, _c_i32_p, _c_u8_p, _c_u64_p,
        ],
        ctypes.c_int64,
    ),
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


def addr(arr: Any) -> int:
    """Address of `arr`'s first element (0 for an empty array). The caller keeps `arr` alive."""
    return int(arr.ctypes.data) if arr.size > 0 else 0


def w_addr(w: Any) -> int:
    """Like `addr` for a weights array, with the same strict checks as `c_w_ptr`.

    The check is not optional: the kernels read `w` as interleaved doubles, so anything but a
    native, aligned, C-contiguous complex128 array with an axis would be read out of bounds.
    """
    if (
        w.dtype.char != "D"  # "D" is complex128
        or not w.dtype.isnative
        or w.ndim == 0
        or not (w.flags.c_contiguous and w.flags.aligned)
    ):
        raise TypeError(
            "weights must be a native-endian, aligned, C-contiguous complex128 array with at "
            f"least one axis, got dtype {w.dtype}, shape {w.shape}"
        )
    return int(w.ctypes.data) if w.size > 0 else 0


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile)


def get_magic_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for `magic_kernels.cpp`."""
    return get_lib(
        sys.modules[__name__], "magic_kernels", "magic_abi_v3", _SPECS,
        force_recompile=force_recompile,
    )


def get_magic_measure_batch() -> Any:
    """Return `magic_measure_batch` from the cached library."""
    return get_magic_kernels_lib().magic_measure_batch
