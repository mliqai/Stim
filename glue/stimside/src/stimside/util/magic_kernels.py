"""Shared-library loader and ctypes ABI bindings for `magic_kernels.cpp`."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from typing import Any

import numpy as np

from stimside.util._kernel_loader import (
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
    c_i32, c_i64, c_dbl = ctypes.c_int32, ctypes.c_int64, ctypes.c_double
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
        ("magic_quadratic_gauss_sum", [sz, _c_u8_p, _c_u8_p, c_i32, c_i64, _c_double_p], None),
        (
            "magic_overlap",
            [sz, sz, _c_u64_p, _c_u64_p, _c_u8_p, _c_u8_p, sz, _c_u64_p, _c_u64_p, _c_u8_p, _c_u8_p, _c_double_p],
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
        pq_mod = sys.modules.get("stimside.util.phase_qfe")
        if pq_mod is not None and hasattr(pq_mod, "_reset_cpp_kernel_cache"):
            pq_mod._reset_cpp_kernel_cache()

    resolver = (
        (lambda force_recompile=True: _resolve_magic_kernels_so(force_recompile=True))
        if force_recompile
        else _resolve_magic_kernels_so
    )
    lib = load_cdll(resolver, "magic_abi_v1")
    _configure_abi(lib)
    _LIB_INSTANCE = lib
    return lib


def _contig_ptr(
    arr: Any, dtype: Any, ptr_t: Any, shape: tuple[int, ...]
) -> tuple[np.ndarray, Any]:
    if not isinstance(arr, np.ndarray) or arr.dtype.kind not in ("u", "i"):
        raise TypeError("cpp_overlap component buffers must be integer numpy arrays")
    if arr.shape != shape:
        raise ValueError(f"Expected buffer shape {shape}, got {arr.shape}")
    a = arr if (arr.dtype == dtype and arr.flags.c_contiguous) else np.ascontiguousarray(arr, dtype=dtype)
    return a, (ctypes.cast(a.ctypes.data, ptr_t) if a.size > 0 else ctypes.cast(0, ptr_t))


def cpp_overlap(bra: Any, ket: Any) -> complex:
    """Contiguity/dtype/shape-safe dispatch wrapper invoked by `phase_qfe.overlap`."""
    if hasattr(bra, "weights") or hasattr(ket, "weights"):
        raise TypeError("cpp_overlap expects single QFE components, not weighted expansions")
    try:
        dims = (bra.n_qubits, ket.n_qubits, bra.d, ket.d)
        bh0, bA0, be0, bG0 = bra.h_words, bra.A_words, bra.ell, bra.Gamma
        kh0, kA0, ke0, kG0 = ket.h_words, ket.A_words, ket.ell, ket.Gamma
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in dims):
            raise TypeError("cpp_overlap requires integer component metadata")
        nq, kq, db, dk = (int(v) for v in dims)
        if nq < 1 or nq != kq or db < 0 or dk < 0:
            raise ValueError("Invalid or mismatched n_qubits or d in cpp_overlap")
        nws = (bra.n_words, ket.n_words)
    except AttributeError as exc:
        raise TypeError("cpp_overlap operands lack required PhaseQFEComponent attributes") from exc
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in nws):
        raise TypeError("cpp_overlap requires integer n_words metadata")
    nw = (nq + 63) // 64
    if int(nws[0]) != nw or int(nws[1]) != nw:
        raise ValueError("Invalid or mismatched n_words in cpp_overlap")
    bh, p_bh = _contig_ptr(bh0, np.uint64, _c_u64_p, (nw,))
    bA, p_bA = _contig_ptr(bA0, np.uint64, _c_u64_p, (db, nw))
    be, p_be = _contig_ptr(be0, np.uint8, _c_u8_p, (db,))
    bG, p_bG = _contig_ptr(bG0, np.uint8, _c_u8_p, (db, db))
    kh, p_kh = _contig_ptr(kh0, np.uint64, _c_u64_p, (nw,))
    kA, p_kA = _contig_ptr(kA0, np.uint64, _c_u64_p, (dk, nw))
    ke, p_ke = _contig_ptr(ke0, np.uint8, _c_u8_p, (dk,))
    kG, p_kG = _contig_ptr(kG0, np.uint8, _c_u8_p, (dk, dk))
    out_ri = (ctypes.c_double * 2)(0.0, 0.0)
    get_magic_kernels_lib().magic_overlap(
        nw, db, p_bh, p_bA, p_be, p_bG,
        dk, p_kh, p_kA, p_ke, p_kG, out_ri,
    )
    _ = (bh, bA, be, bG, kh, kA, ke, kG)
    return complex(float(out_ri[0]), float(out_ri[1]))
