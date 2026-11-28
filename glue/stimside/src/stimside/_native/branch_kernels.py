"""Thin ctypes ABI bindings and RAII handle for the `mbr_*` stabilizer-branch kernels.

Compiled into `libmagic_kernels.so` via `branch_kernels.hpp`. All stabilizer-state math executes
in C++; this module provides only ABI validation, status-code translation, and a thin RAII
`BranchState` wrapper with duck-typed conversion helpers.
"""

from __future__ import annotations

import ctypes
import functools
import math
import numbers
import operator
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stimside._native._kernel_loader import AbiSpec, get_lib, resolve_so
from stimside._native.magic_kernels import addr

_LIB_INSTANCE: ctypes.CDLL | None = None

_VP = ctypes.c_void_p
_SZ = ctypes.c_size_t
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_U8 = ctypes.c_uint8
_U32 = ctypes.c_uint32
_U64 = ctypes.c_uint64

MBR_OK = 0
MBR_ERR_INVALID_ARG = -1
MBR_ERR_OOM = -2
MBR_ERR_FLOAT_DRIFT = -3
MBR_ERR_BUDGET = -4

MBR_KIND_T = 0
MBR_KIND_CCZ = 1
MBR_KIND_SKIP = 2
MBR_KIND_CFAULT = 3
MBR_KIND_MEAS = 4
MBR_KIND_PEEK = 5
MBR_KIND_POST = 6
MBR_KIND_URESET = 7
MBR_KIND_NAMES: tuple[str, ...] = (
    "T", "CCZ", "SKIP", "CFAULT", "MEAS", "PEEK", "POST", "URESET"
)

MBR_MEAS_DEFINITE = 0
MBR_MEAS_ABSORBED = 1
MBR_MEAS_FORKED = 2

_BRANCH_STORE_MAX_NODES = 300_000
_MAX_QUBITS = 0x7FFFFFFF
_SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class BranchBudgetError(RuntimeError):
    """Raised when a native branch operation exceeds `max_components` (status -4)."""


def _check_rc(rc: int, op: str) -> None:
    if rc == MBR_OK:
        return
    if rc == MBR_ERR_INVALID_ARG:
        raise ValueError(f"{op}: invalid arguments or state invariants (rc={rc})")
    if rc == MBR_ERR_OOM:
        raise MemoryError(f"{op}: out of memory (rc={rc})")
    if rc == MBR_ERR_FLOAT_DRIFT:
        raise FloatingPointError(f"{op}: numerical drift exceeded tolerance (rc={rc})")
    if rc == MBR_ERR_BUDGET:
        raise BranchBudgetError(f"{op}: branch count exceeded max_components (rc={rc})")
    raise RuntimeError(f"{op}: unexpected native status {rc}")


_SPECS: tuple[AbiSpec, ...] = (
    ("mbr_abi_v1", [], _I32),
    ("mbr_live_states", [], _SZ),
    ("mbr_state_create_zero", [_SZ, _VP], _I32),
    ("mbr_state_from_arrays", [_SZ, _SZ, _VP, _VP, _VP, _VP, _VP, _VP], _I32),
    ("mbr_state_copy", [_VP, _VP], _I32),
    ("mbr_state_destroy", [_VP], None),
    ("mbr_state_header", [_VP, _VP, _VP, _VP], _I32),
    ("mbr_state_export_arrays", [_VP, _VP, _VP, _VP, _VP, _VP], _I32),
    ("mbr_state_pauli", [_VP, _VP, _VP, _U8, _VP], _I32),
    ("mbr_state_qturn", [_VP, _VP, _VP, _U8, _VP], _I32),
    ("mbr_state_zproject", [_VP, _VP, _U8, _U8, _VP], _I32),
    ("mbr_state_project", [_VP, _VP, _VP, _U8, _U8, _VP], _I32),
    ("mbr_state_definite", [_VP, _VP, _VP, _U8, _VP], _I32),
    ("mbr_state_generators", [_VP, _VP, _VP, _VP], _I32),
    ("mbr_state_canon", [_VP, _VP], _I32),
    ("mbr_state_canonical_key_size", [_VP, _VP], _I32),
    ("mbr_state_canonical_key", [_VP, _VP, _SZ, _VP], _I32),
    ("mbr_state_from_canonical_key", [_VP, _SZ, _VP], _I32),
    ("mbr_state_overlap", [_VP, _VP, _VP], _I32),
    ("mbr_state_same_phase", [_VP, _VP, _VP, _VP], _I32),
    ("mbr_state_to_dense", [_VP, _I32, _VP, _SZ], _I32),
    ("mbr_is_clifford", [_VP, _SZ, _SZ, _VP], _I32),
    ("mbr_choose_split", [_VP, _SZ, _SZ, _U32, _VP, _VP], _I32),
    ("mbr_live_stores", [], _SZ),
    ("mbr_store_create", [_SZ, _SZ, _VP], _I32),
    ("mbr_store_destroy", [_VP], None),
    ("mbr_store_clear", [_VP], _I32),
    ("mbr_store_set_max_components", [_VP, _SZ], _I32),
    ("mbr_store_node_count", [_VP, _VP], _I32),
    ("mbr_store_node_info", [_VP, _I64, _VP, _VP, _VP], _I32),
    ("mbr_store_node_branch", [_VP, _I64, _SZ, _VP, _VP], _I32),
    ("mbr_store_node_pending", [_VP, _I64, _VP, _VP, _VP, _VP], _I32),
    ("mbr_store_node_dense", [_VP, _I64, _I32, _VP, _SZ], _I32),
    ("mbr_store_node_from_branches", [_VP, _SZ, _VP, _VP, _SZ, _VP, _VP, _VP, _VP, _VP], _I32),
    ("mbr_store_resolve_node", [_VP, _I64, _VP], _I32),
    ("mbr_store_step_rotations", [_VP, _I64, _U64, _U8, _U32, _SZ, _VP, _VP, _VP, _VP, _VP], _I32),
    ("mbr_store_step_qturns", [_VP, _I64, _U64, _U8, _U32, _SZ, _VP, _VP, _VP, _VP], _I32),
    ("mbr_store_step_meas", [_VP, _I64, _U64, _U8, _VP, _VP, _U8, _VP, _VP, _VP, _VP, _VP, _VP], _I32),
    ("mbr_store_step_peek", [_VP, _I64, _U64, _VP, _VP, _U8, _VP], _I32),
    ("mbr_store_counters", [_VP, _VP], _I32),
    ("mbr_store_branches_hist", [_VP, _VP, _VP, _SZ, _VP], _I32),
    (
        "mbr_batch_step_rotations",
        [_VP, _SZ, _VP, _U64, _U8, _SZ, _VP, _VP, _SZ, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP],
        _I32,
    ),
    (
        "mbr_batch_step_qturns",
        [_VP, _SZ, _VP, _U64, _U8, _SZ, _VP, _VP, _VP, _VP, _VP, _VP, _VP],
        _I32,
    ),
    (
        "mbr_batch_step_meas",
        [
            _VP, _SZ, _VP, _U64, _U8, _VP, _VP, _U8, _VP, _VP, _VP, _VP, _U8,
            _VP, _VP, _VP, _VP, _VP, _VP, _VP,
        ],
        _I32,
    ),
    ("mbr_batch_step_peek", [_VP, _SZ, _U64, _VP, _VP, _U8, _VP, _VP, _VP, _VP], _I32),
    (
        "mbr_batch_step_postselect",
        [_VP, _SZ, _VP, _U64, _VP, _VP, _U8, _VP, _VP, _VP, _VP, _VP, _VP, _VP],
        _I32,
    ),
)


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile)


def get_branch_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Return the cached shared library with the `mbr_*` ABI bound."""
    return get_lib(
        sys.modules[__name__],
        "magic_kernels",
        "mbr_abi_v1",
        _SPECS,
        force_recompile=force_recompile,
    )


def _as_index(val: Any, name: str) -> int:
    if isinstance(val, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    return operator.index(val)


def _as_contiguous(arr: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    target = np.dtype(dtype)
    if isinstance(arr, np.ndarray):
        a = arr
    else:
        a = np.asarray(arr, dtype=object)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if a.dtype == target and a.dtype.isnative and a.flags.c_contiguous and a.flags.aligned:
        return a
    info = np.iinfo(target)
    if a.dtype == object:
        vals: list[int] = []
        for x in a.flat:
            if isinstance(x, (bool, np.bool_)):
                raise TypeError(f"{name} elements must be integers, got bool")
            v = operator.index(x)
            if v < info.min or v > info.max:
                raise ValueError(
                    f"{name} value {v} out of range for {target} [{info.min}, {info.max}]"
                )
            vals.append(v)
        return np.array(vals, dtype=target).reshape(shape)
    if a.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must have an integer dtype, got {a.dtype}")
    if a.size > 0 and a.dtype != target:
        lo = int(a.min())
        hi = int(a.max())
        if lo < info.min or hi > info.max:
            raise ValueError(
                f"{name} values [{lo}, {hi}] out of range for {target} [{info.min}, {info.max}]"
            )
    return np.require(a, dtype=target, requirements=("C", "A"))


def _as_rw_contiguous(arr: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Validate an in/out caller ndarray (zero-copy; must be native, C-contiguous, aligned, writeable)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    target = np.dtype(dtype)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if arr.dtype != target or not arr.dtype.isnative:
        raise TypeError(f"{name} must have native dtype {target}, got {arr.dtype}")
    if not (arr.flags.c_contiguous and arr.flags.aligned):
        raise ValueError(f"{name} must be C-contiguous and aligned")
    if not arr.flags.writeable:
        raise ValueError(f"{name} must be writeable")
    return arr


def _as_u8_mask(arr: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert bool or uint8 mask to contiguous uint8 (zero-copy when already contiguous bool/uint8)."""
    if isinstance(arr, np.ndarray):
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        if arr.dtype == np.uint8:
            if arr.flags.c_contiguous and arr.flags.aligned:
                return arr
            return np.require(arr, dtype=np.uint8, requirements=("C", "A"))
        if arr.dtype == np.bool_:
            if arr.flags.c_contiguous and arr.flags.aligned:
                return arr.view(np.uint8)
            return np.ascontiguousarray(arr, dtype=np.bool_).view(np.uint8)
        raise TypeError(f"{name} must have dtype bool or uint8, got {arr.dtype}")
    a = np.asarray(arr)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if a.dtype == np.bool_:
        return np.ascontiguousarray(a, dtype=np.bool_).view(np.uint8)
    if a.dtype == np.uint8:
        return np.require(a, dtype=np.uint8, requirements=("C", "A"))
    raise TypeError(f"{name} must have dtype bool or uint8, got {a.dtype}")


def _as_f64_contiguous(arr: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        a = arr
    else:
        a = np.asarray(arr, dtype=object)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if a.dtype == np.float64 and a.dtype.isnative and a.flags.c_contiguous and a.flags.aligned:
        return a
    if a.dtype == object:
        vals: list[float] = []
        for x in a.flat:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (numbers.Real, np.floating, np.integer)):
                raise TypeError(f"{name} elements must be real numbers, got {type(x).__name__}")
            vals.append(float(x))
        return np.array(vals, dtype=np.float64).reshape(shape)
    if a.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"{name} must have a real numeric dtype, got {a.dtype}")
    return np.require(a, dtype=np.float64, requirements=("C", "A"))


def _as_u64_words(arr: Any, nw: int, name: str) -> np.ndarray:
    return _as_contiguous(arr, np.uint64, (nw,), name)


def _as_u8_flag(val: Any, name: str) -> int:
    v = _as_index(val, name)
    if v not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {v}")
    return v


def _as_kappa(val: Any) -> int:
    v = _as_index(val, "kappa")
    return v & 3


def _as_u64_scalar(val: Any, name: str) -> int:
    v = _as_index(val, name)
    if v < 0 or v > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} out of uint64 range: {v}")
    return v


def _as_u32_scalar(val: Any, name: str) -> int:
    v = _as_index(val, name)
    if v < 0 or v > 0xFFFFFFFF:
        raise ValueError(f"{name} out of uint32 range: {v}")
    return v


def _as_i64_scalar(val: Any, name: str) -> int:
    v = _as_index(val, name)
    if v < -(1 << 63) or v > (1 << 63) - 1:
        raise ValueError(f"{name} out of int64 range: {v}")
    return v


def _as_size_t_scalar(val: Any, name: str) -> int:
    v = _as_index(val, name)
    if v < 0 or v > _SIZE_MAX:
        raise ValueError(f"{name} out of size_t range [0, {_SIZE_MAX}]: {v}")
    return v


def _as_complex128_1d(arr: Any, length: int, name: str) -> np.ndarray:
    raw = arr if isinstance(arr, np.ndarray) else np.asarray(arr, dtype=object)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {raw.shape}")
    if (
        raw.dtype == np.complex128
        and raw.dtype.isnative
        and raw.flags.c_contiguous
        and raw.flags.aligned
    ):
        return raw
    if raw.dtype == object:
        vals: list[complex] = []
        for x in raw.flat:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (numbers.Number, np.number)):
                raise TypeError(
                    f"{name} elements must be numeric (not bool/str), got {type(x).__name__}"
                )
            vals.append(complex(x))
        return np.array(vals, dtype=np.complex128)
    if raw.dtype.kind == "b" or raw.dtype.kind not in ("i", "u", "f", "c"):
        raise TypeError(f"{name} must have a numeric dtype, got {raw.dtype}")
    return np.require(raw, dtype=np.complex128, requirements=("C", "A"))


def _pack_parities(parities: Any, n_words: int | None) -> tuple[np.ndarray, int, int]:
    """Pack integer or word-matrix parities into contiguous `(n_parities, nw) uint64`."""
    if n_words is not None:
        nw = _as_index(n_words, "n_words")
        if nw < 1:
            raise ValueError(f"n_words must be >= 1, got {nw}")
    else:
        nw = 0

    if isinstance(parities, np.ndarray) and parities.ndim == 2:
        n_par = int(parities.shape[0])
        in_nw = int(parities.shape[1])
        if in_nw < 1:
            raise ValueError(f"parities 2D array must have shape[1] >= 1, got {parities.shape}")
        if nw == 0:
            nw = in_nw
        elif in_nw != nw:
            raise ValueError(f"parities must have shape ({n_par}, {nw}), got {parities.shape}")
        if n_par == 0:
            return np.empty((0, nw), dtype=np.uint64), 0, nw
        arr = _as_contiguous(parities, np.uint64, (n_par, nw), "parities")
        return arr, n_par, nw

    items = list(parities)
    n_par = len(items)
    if n_par == 0:
        eff_nw = nw if nw > 0 else 1
        return np.empty((0, eff_nw), dtype=np.uint64), 0, eff_nw

    ints: list[int] = []
    max_bits = 0
    for x in items:
        if isinstance(x, (bool, np.bool_)):
            raise TypeError("parities elements must be integers, got bool")
        v = operator.index(x)
        if v <= 0:
            raise ValueError(f"parity bitmask must be > 0, got {v}")
        ints.append(v)
        bl = v.bit_length()
        if bl > max_bits:
            max_bits = bl
    needed_nw = max(1, (max_bits + 63) // 64)
    if nw == 0:
        nw = needed_nw
    elif needed_nw > nw:
        raise ValueError(f"parity requires {needed_nw} uint64 words, exceeds n_words={nw}")

    out = np.zeros((n_par, nw), dtype=np.uint64)
    mask64 = (1 << 64) - 1
    for i, v in enumerate(ints):
        cur = v
        for w in range(nw):
            out[i, w] = cur & mask64
            cur >>= 64
    return out, n_par, nw


def is_clifford(parities: Any, n_words: int | None = None) -> bool:
    """Native Mobius Clifford test (`r <= 20`; returns `False` for `r > 20`)."""
    arr, n_par, nw = _pack_parities(parities, n_words)
    lib = get_branch_kernels_lib()
    out = ctypes.c_int32()
    _check_rc(
        lib.mbr_is_clifford(addr(arr), n_par, nw, ctypes.addressof(out)),
        "mbr_is_clifford",
    )
    return bool(out.value)


def choose_split(
    parities: Any, max_depth: int = 5, n_words: int | None = None
) -> tuple[int, int]:
    """Native iterative-deepening split choice returning `(lam_int, depth)`."""
    md = _as_u32_scalar(max_depth, "max_depth")
    arr, n_par, nw = _pack_parities(parities, n_words)
    if n_par == 0:
        raise ValueError("parities must be non-empty for choose_split")
    lib = get_branch_kernels_lib()
    out_lam = np.zeros(nw, dtype=np.uint64)
    out_depth = ctypes.c_int32()
    _check_rc(
        lib.mbr_choose_split(
            addr(arr), n_par, nw, md, addr(out_lam), ctypes.addressof(out_depth)
        ),
        "mbr_choose_split",
    )
    lam_int = 0
    for w in range(nw - 1, -1, -1):
        lam_int = (lam_int << 64) | int(out_lam[w])
    return lam_int, int(out_depth.value)


class BranchState:
    """Thin RAII handle owning a native `MbrState` instance."""

    __slots__ = ("_lib", "_handle", "_n_qubits", "_nw")

    def __init__(self, lib: ctypes.CDLL, handle: int, n_qubits: int, nw: int) -> None:
        self._lib = lib
        self._handle = handle
        self._n_qubits = n_qubits
        self._nw = nw

    @staticmethod
    def live_states() -> int:
        return int(get_branch_kernels_lib().mbr_live_states())

    @classmethod
    def zero(cls, n_qubits: int) -> BranchState:
        n = _as_index(n_qubits, "n_qubits")
        if n < 1 or n > _MAX_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {_MAX_QUBITS}], got {n}")
        lib = get_branch_kernels_lib()
        out = ctypes.c_void_p()
        _check_rc(lib.mbr_state_create_zero(n, ctypes.addressof(out)), "mbr_state_create_zero")
        return cls(lib, int(out.value), n, (n + 63) // 64)

    @classmethod
    def from_arrays(
        cls,
        n_qubits: int,
        h_words: Any,
        A_words: Any,
        ell: Any,
        Gamma: Any,
        pivots: Any,
    ) -> BranchState:
        n = _as_index(n_qubits, "n_qubits")
        if n < 1 or n > _MAX_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {_MAX_QUBITS}], got {n}")
        nw = (n + 63) // 64
        ell_a = ell if isinstance(ell, np.ndarray) else np.asarray(ell, dtype=object)
        if ell_a.ndim != 1:
            raise ValueError(f"ell must be 1D, got shape {ell_a.shape}")
        d = int(ell_a.shape[0])
        h_c = _as_u64_words(h_words, nw, "h_words")
        A_c = _as_contiguous(A_words, np.uint64, (d, nw), "A_words")
        ell_c = _as_contiguous(ell_a, np.uint8, (d,), "ell")
        G_c = _as_contiguous(Gamma, np.uint8, (d, d), "Gamma")
        piv_c = _as_contiguous(pivots, np.int32, (d,), "pivots")
        lib = get_branch_kernels_lib()
        out = ctypes.c_void_p()
        _check_rc(
            lib.mbr_state_from_arrays(
                n, d, addr(h_c), addr(A_c), addr(ell_c), addr(G_c), addr(piv_c),
                ctypes.addressof(out),
            ),
            "mbr_state_from_arrays",
        )
        return cls(lib, int(out.value), n, nw)

    @classmethod
    def from_phase_qfe(cls, comp: Any) -> BranchState:
        """Construct from any duck-typed Phase-QFE component."""
        n = _as_index(comp.n_qubits, "n_qubits")
        d = _as_index(comp.d, "d")
        ell_a = comp.ell if isinstance(comp.ell, np.ndarray) else np.asarray(comp.ell, dtype=object)
        if ell_a.shape != (d,):
            raise ValueError(f"component ell shape {ell_a.shape} != ({d},)")
        return cls.from_arrays(n, comp.h_words, comp.A_words, ell_a, comp.Gamma, comp.pivots)

    @classmethod
    def from_canonical_key(cls, key: bytes | bytearray | memoryview) -> BranchState:
        mv = memoryview(key)
        if mv.ndim == 1 and mv.c_contiguous and mv.itemsize == 1:
            buf = np.frombuffer(mv, dtype=np.uint8)
        else:
            buf = np.frombuffer(mv.tobytes(), dtype=np.uint8)
        if not (buf.flags.c_contiguous and buf.flags.aligned):
            buf = np.require(buf, dtype=np.uint8, requirements=("C", "A"))
        lib = get_branch_kernels_lib()
        out = ctypes.c_void_p()
        _check_rc(
            lib.mbr_state_from_canonical_key(addr(buf), buf.size, ctypes.addressof(out)),
            "mbr_state_from_canonical_key",
        )
        h = int(out.value)
        n_c, d_c, nw_c = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        rc = lib.mbr_state_header(
            h, ctypes.addressof(n_c), ctypes.addressof(d_c), ctypes.addressof(nw_c)
        )
        if rc != MBR_OK:
            lib.mbr_state_destroy(h)
            _check_rc(rc, "mbr_state_header")
        return cls(lib, h, int(n_c.value), int(nw_c.value))

    def _require_open(self) -> int:
        if self._handle == 0:
            raise ValueError("BranchState handle is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle == 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_words(self) -> int:
        return self._nw

    @property
    def d(self) -> int:
        h = self._require_open()
        n_c, d_c, nw_c = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        _check_rc(
            self._lib.mbr_state_header(
                h, ctypes.addressof(n_c), ctypes.addressof(d_c), ctypes.addressof(nw_c)
            ),
            "mbr_state_header",
        )
        return int(d_c.value)

    def close(self) -> None:
        h = self._handle
        if h != 0:
            self._handle = 0
            self._lib.mbr_state_destroy(h)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> BranchState:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def copy(self) -> BranchState:
        h = self._require_open()
        out = ctypes.c_void_p()
        _check_rc(self._lib.mbr_state_copy(h, ctypes.addressof(out)), "mbr_state_copy")
        return BranchState(self._lib, int(out.value), self._n_qubits, self._nw)

    def to_arrays(
        self,
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = self._require_open()
        n_c, d_c, nw_c = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        _check_rc(
            self._lib.mbr_state_header(
                h, ctypes.addressof(n_c), ctypes.addressof(d_c), ctypes.addressof(nw_c)
            ),
            "mbr_state_header",
        )
        n, d, nw = int(n_c.value), int(d_c.value), int(nw_c.value)
        h_out = np.empty(nw, dtype=np.uint64)
        A_out = np.empty((d, nw), dtype=np.uint64)
        ell_out = np.empty(d, dtype=np.uint8)
        G_out = np.empty((d, d), dtype=np.uint8)
        piv_out = np.empty(d, dtype=np.int32)
        _check_rc(
            self._lib.mbr_state_export_arrays(
                h, addr(h_out), addr(A_out), addr(ell_out), addr(G_out), addr(piv_out)
            ),
            "mbr_state_export_arrays",
        )
        return (n, d, h_out, A_out, ell_out, G_out, piv_out)

    def to_phase_qfe(self, component_cls: Any) -> Any:
        """Materialize a fresh duck-typed Phase-QFE component via `component_cls`."""
        n, d, h_out, A_out, ell_out, G_out, piv_out = self.to_arrays()
        return component_cls(
            n_qubits=n,
            d=d,
            h_words=h_out,
            A_words=A_out,
            ell=ell_out,
            Gamma=G_out,
            pivots=piv_out,
        )

    def pauli(self, x: Any, z: Any, kappa: int = 0) -> complex:
        h = self._require_open()
        xa = _as_u64_words(x, self._nw, "x")
        za = _as_u64_words(z, self._nw, "z")
        k = _as_kappa(kappa)
        out = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_pauli(h, addr(xa), addr(za), k, ctypes.addressof(out)),
            "mbr_state_pauli",
        )
        return complex(out[0], out[1])

    def qturn(self, x: Any, z: Any, kappa: int) -> complex:
        h = self._require_open()
        xa = _as_u64_words(x, self._nw, "x")
        za = _as_u64_words(z, self._nw, "z")
        k = _as_kappa(kappa)
        out = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_qturn(h, addr(xa), addr(za), k, ctypes.addressof(out)),
            "mbr_state_qturn",
        )
        return complex(out[0], out[1])

    def zproject(self, z: Any, sigma: int, m: int) -> complex:
        h = self._require_open()
        za = _as_u64_words(z, self._nw, "z")
        sg = _as_u8_flag(sigma, "sigma")
        mv = _as_u8_flag(m, "m")
        out = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_zproject(h, addr(za), sg, mv, ctypes.addressof(out)),
            "mbr_state_zproject",
        )
        return complex(out[0], out[1])

    def project(self, x: Any, z: Any, kappa: int, m: int) -> complex:
        h = self._require_open()
        xa = _as_u64_words(x, self._nw, "x")
        za = _as_u64_words(z, self._nw, "z")
        k = _as_kappa(kappa)
        mv = _as_u8_flag(m, "m")
        out = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_project(h, addr(xa), addr(za), k, mv, ctypes.addressof(out)),
            "mbr_state_project",
        )
        return complex(out[0], out[1])

    def definite(self, x: Any, z: Any, kappa: int) -> int:
        h = self._require_open()
        xa = _as_u64_words(x, self._nw, "x")
        za = _as_u64_words(z, self._nw, "z")
        k = _as_kappa(kappa)
        ev = ctypes.c_int32()
        _check_rc(
            self._lib.mbr_state_definite(h, addr(xa), addr(za), k, ctypes.addressof(ev)),
            "mbr_state_definite",
        )
        return int(ev.value)

    def generators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._require_open()
        gx = np.empty((self._n_qubits, self._nw), dtype=np.uint64)
        gz = np.empty((self._n_qubits, self._nw), dtype=np.uint64)
        gk = np.empty(self._n_qubits, dtype=np.uint8)
        _check_rc(
            self._lib.mbr_state_generators(h, addr(gx), addr(gz), addr(gk)),
            "mbr_state_generators",
        )
        return gx, gz, gk

    def canon(self) -> complex:
        h = self._require_open()
        out = (ctypes.c_double * 2)()
        _check_rc(self._lib.mbr_state_canon(h, ctypes.addressof(out)), "mbr_state_canon")
        return complex(out[0], out[1])

    def canonical_key(self) -> bytes:
        h = self._require_open()
        needed = ctypes.c_size_t()
        _check_rc(
            self._lib.mbr_state_canonical_key_size(h, ctypes.addressof(needed)),
            "mbr_state_canonical_key_size",
        )
        buf = np.empty(int(needed.value), dtype=np.uint8)
        written = ctypes.c_size_t()
        _check_rc(
            self._lib.mbr_state_canonical_key(h, addr(buf), buf.size, ctypes.addressof(written)),
            "mbr_state_canonical_key",
        )
        return buf[: int(written.value)].tobytes()

    def overlap(self, other: BranchState) -> complex:
        ha = self._require_open()
        if not isinstance(other, BranchState):
            raise TypeError(f"other must be BranchState, got {type(other).__name__}")
        hb = other._require_open()
        out = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_overlap(ha, hb, ctypes.addressof(out)),
            "mbr_state_overlap",
        )
        return complex(out[0], out[1])

    def same_phase(self, other: BranchState) -> tuple[bool, complex]:
        ha = self._require_open()
        if not isinstance(other, BranchState):
            raise TypeError(f"other must be BranchState, got {type(other).__name__}")
        hb = other._require_open()
        eq = ctypes.c_int32()
        ratio = (ctypes.c_double * 2)()
        _check_rc(
            self._lib.mbr_state_same_phase(ha, hb, ctypes.addressof(eq), ctypes.addressof(ratio)),
            "mbr_state_same_phase",
        )
        return (bool(eq.value), complex(ratio[0], ratio[1]))

    def dense(self, endian: str = "little") -> np.ndarray:
        h = self._require_open()
        if endian == "little":
            big = 0
        elif endian == "big":
            big = 1
        else:
            raise ValueError(f"endian must be 'little' or 'big', got {endian!r}")
        if self._n_qubits > 26:
            raise ValueError(f"dense requires n_qubits <= 26, got {self._n_qubits}")
        dim = 1 << self._n_qubits
        out = np.empty(dim, dtype=np.complex128)
        _check_rc(
            self._lib.mbr_state_to_dense(h, big, addr(out), dim),
            "mbr_state_to_dense",
        )
        return out


class BranchStore:
    """Thin RAII handle owning a native `MbrStore` trie, sub-caches, and Section 7 counters."""

    __slots__ = ("_lib", "_handle", "_n_qubits", "_nw")

    def __init__(self, n_qubits: int, max_components: int = 0) -> None:
        n = _as_index(n_qubits, "n_qubits")
        if n < 1 or n > _MAX_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {_MAX_QUBITS}], got {n}")
        mc = _as_size_t_scalar(max_components, "max_components")
        lib = get_branch_kernels_lib()
        out = ctypes.c_void_p()
        _check_rc(lib.mbr_store_create(n, mc, ctypes.addressof(out)), "mbr_store_create")
        self._lib = lib
        self._handle = int(out.value)
        self._n_qubits = n
        self._nw = (n + 63) // 64

    @staticmethod
    def live_stores() -> int:
        return int(get_branch_kernels_lib().mbr_live_stores())

    def _require_open(self) -> int:
        if self._handle == 0:
            raise ValueError("BranchStore handle is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle == 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_words(self) -> int:
        return self._nw

    def close(self) -> None:
        h = self._handle
        if h != 0:
            self._handle = 0
            self._lib.mbr_store_destroy(h)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> BranchStore:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def clear(self) -> None:
        h = self._require_open()
        _check_rc(self._lib.mbr_store_clear(h), "mbr_store_clear")

    def set_max_components(self, max_components: int) -> None:
        h = self._require_open()
        mc = _as_size_t_scalar(max_components, "max_components")
        _check_rc(self._lib.mbr_store_set_max_components(h, mc), "mbr_store_set_max_components")

    @property
    def node_count(self) -> int:
        h = self._require_open()
        cnt = ctypes.c_size_t()
        _check_rc(self._lib.mbr_store_node_count(h, ctypes.addressof(cnt)), "mbr_store_node_count")
        return int(cnt.value)

    def node_info(self, node_id: int) -> tuple[int, int, int]:
        """Return `(n_branches, n_pend, transient_peak)` for `node_id`."""
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        nb = ctypes.c_size_t()
        npd = ctypes.c_size_t()
        pk = ctypes.c_uint32()
        _check_rc(
            self._lib.mbr_store_node_info(
                h, nid, ctypes.addressof(nb), ctypes.addressof(npd), ctypes.addressof(pk)
            ),
            "mbr_store_node_info",
        )
        return int(nb.value), int(npd.value), int(pk.value)

    def node_branches(self, node_id: int) -> list[tuple[complex, BranchState]]:
        """Return a fresh list of `(weight, BranchState)` copies for `node_id`."""
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        n_branches, _, _ = self.node_info(nid)
        out: list[tuple[complex, BranchState]] = []
        w_buf = (ctypes.c_double * 2)()
        st_ptr = ctypes.c_void_p()
        try:
            for i in range(n_branches):
                _check_rc(
                    self._lib.mbr_store_node_branch(
                        h, nid, i, ctypes.addressof(w_buf), ctypes.addressof(st_ptr)
                    ),
                    "mbr_store_node_branch",
                )
                st = BranchState(self._lib, int(st_ptr.value), self._n_qubits, self._nw)
                out.append((complex(w_buf[0], w_buf[1]), st))
        except Exception:
            for _, st in out:
                st.close()
            raise
        return out

    def node_pending(self, node_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(x, z, kappa, e)` arrays of pending turns on `node_id`."""
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        _, n_pend, _ = self.node_info(nid)
        px = np.empty((n_pend, self._nw), dtype=np.uint64)
        pz = np.empty((n_pend, self._nw), dtype=np.uint64)
        pk = np.empty(n_pend, dtype=np.uint8)
        pe = np.empty(n_pend, dtype=np.uint8)
        if n_pend > 0:
            _check_rc(
                self._lib.mbr_store_node_pending(h, nid, addr(px), addr(pz), addr(pk), addr(pe)),
                "mbr_store_node_pending",
            )
        return px, pz, pk, pe

    def node_dense(self, node_id: int, endian: str = "little") -> np.ndarray:
        """Compute exact `D sum_b w_b |S_b>` dense vector for `n_qubits <= 26`."""
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        if endian == "little":
            big = 0
        elif endian == "big":
            big = 1
        else:
            raise ValueError(f"endian must be 'little' or 'big', got {endian!r}")
        if self._n_qubits > 26:
            raise ValueError(f"node_dense requires n_qubits <= 26, got {self._n_qubits}")
        dim = 1 << self._n_qubits
        out = np.empty(dim, dtype=np.complex128)
        _check_rc(
            self._lib.mbr_store_node_dense(h, nid, big, addr(out), dim),
            "mbr_store_node_dense",
        )
        return out

    def node_from_branches(
        self,
        weights: Any,
        states: Any,
        pend_x: Any = None,
        pend_z: Any = None,
        pend_kappa: Any = None,
        pend_e: Any = None,
    ) -> int:
        """Create a node from explicit `(weights, states)` and optional pending turns."""
        h = self._require_open()
        st_list = list(states)
        n_b = len(st_list)
        if n_b == 0:
            raise ValueError("states must be non-empty")
        w_c = _as_complex128_1d(weights, n_b, "weights")
        ptrs = np.empty(n_b, dtype=np.uintp)
        for i, st in enumerate(st_list):
            if not isinstance(st, BranchState):
                raise TypeError(f"states[{i}] must be BranchState, got {type(st).__name__}")
            ptrs[i] = st._require_open()

        if pend_x is None and pend_z is None and pend_kappa is None and pend_e is None:
            n_p = 0
            px_c = np.empty((0, self._nw), dtype=np.uint64)
            pz_c = np.empty((0, self._nw), dtype=np.uint64)
            pk_c = np.empty(0, dtype=np.uint8)
            pe_c = np.empty(0, dtype=np.uint8)
        else:
            if pend_x is None or pend_z is None or pend_kappa is None or pend_e is None:
                raise ValueError("pend_x, pend_z, pend_kappa, pend_e must all be provided together")
            pk_raw = pend_kappa if isinstance(pend_kappa, np.ndarray) else np.asarray(pend_kappa, dtype=object)
            if pk_raw.ndim != 1:
                raise ValueError(f"pend_kappa must be 1D, got shape {pk_raw.shape}")
            n_p = int(pk_raw.shape[0])
            px_c = _as_contiguous(pend_x, np.uint64, (n_p, self._nw), "pend_x")
            pz_c = _as_contiguous(pend_z, np.uint64, (n_p, self._nw), "pend_z")
            pk_c = _as_contiguous(pk_raw, np.uint8, (n_p,), "pend_kappa")
            pe_c = _as_contiguous(pend_e, np.uint8, (n_p,), "pend_e")

        out_nid = ctypes.c_int64()
        _check_rc(
            self._lib.mbr_store_node_from_branches(
                h, n_b, addr(w_c), addr(ptrs),
                n_p, addr(px_c), addr(pz_c), addr(pk_c), addr(pe_c),
                ctypes.addressof(out_nid),
            ),
            "mbr_store_node_from_branches",
        )
        return int(out_nid.value)

    def resolve_node(self, node_id: int) -> int:
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        out_nid = ctypes.c_int64()
        _check_rc(
            self._lib.mbr_store_resolve_node(h, nid, ctypes.addressof(out_nid)),
            "mbr_store_resolve_node",
        )
        return int(out_nid.value)

    def step_rotations(
        self,
        node_id: int,
        event_id: int,
        kind: int,
        sign_bits: int,
        ax_x: Any,
        ax_z: Any,
        ax_kappa: Any,
        ax_e: Any,
    ) -> int:
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_T, MBR_KIND_CCZ):
            raise ValueError(f"kind must be MBR_KIND_T or MBR_KIND_CCZ, got {kd}")
        sb = _as_u32_scalar(sign_bits, "sign_bits")
        ak_raw = ax_kappa if isinstance(ax_kappa, np.ndarray) else np.asarray(ax_kappa, dtype=object)
        if ak_raw.ndim != 1:
            raise ValueError(f"ax_kappa must be 1D, got shape {ak_raw.shape}")
        n_ax = int(ak_raw.shape[0])
        x_c = _as_contiguous(ax_x, np.uint64, (n_ax, self._nw), "ax_x")
        z_c = _as_contiguous(ax_z, np.uint64, (n_ax, self._nw), "ax_z")
        k_c = _as_contiguous(ak_raw, np.uint8, (n_ax,), "ax_kappa")
        e_c = _as_contiguous(ax_e, np.uint8, (n_ax,), "ax_e")
        out_child = ctypes.c_int64()
        _check_rc(
            self._lib.mbr_store_step_rotations(
                h, nid, eid, kd, sb, n_ax, addr(x_c), addr(z_c), addr(k_c), addr(e_c),
                ctypes.addressof(out_child),
            ),
            "mbr_store_step_rotations",
        )
        return int(out_child.value)

    def step_qturns(
        self,
        node_id: int,
        event_id: int,
        kind: int,
        sign_bits: int,
        qt_x: Any,
        qt_z: Any,
        qt_kappa: Any,
    ) -> int:
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_SKIP, MBR_KIND_CFAULT):
            raise ValueError(f"kind must be MBR_KIND_SKIP or MBR_KIND_CFAULT, got {kd}")
        sb = _as_u32_scalar(sign_bits, "sign_bits")
        qk_raw = qt_kappa if isinstance(qt_kappa, np.ndarray) else np.asarray(qt_kappa, dtype=object)
        if qk_raw.ndim != 1:
            raise ValueError(f"qt_kappa must be 1D, got shape {qk_raw.shape}")
        n_qt = int(qk_raw.shape[0])
        x_c = _as_contiguous(qt_x, np.uint64, (n_qt, self._nw), "qt_x")
        z_c = _as_contiguous(qt_z, np.uint64, (n_qt, self._nw), "qt_z")
        k_c = _as_contiguous(qk_raw, np.uint8, (n_qt,), "qt_kappa")
        out_child = ctypes.c_int64()
        _check_rc(
            self._lib.mbr_store_step_qturns(
                h, nid, eid, kd, sb, n_qt, addr(x_c), addr(z_c), addr(k_c),
                ctypes.addressof(out_child),
            ),
            "mbr_store_step_qturns",
        )
        return int(out_child.value)

    def step_meas(
        self,
        node_id: int,
        event_id: int,
        kind: int,
        mx: Any,
        mz: Any,
        mkappa: int,
    ) -> tuple[float, int, int, int, np.ndarray, np.ndarray]:
        """Step measurement/postselect/ureset.

        Returns `(p0, child0, child1, meas_type, absorb_x, absorb_z)`.
        """
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_MEAS, MBR_KIND_POST, MBR_KIND_URESET):
            raise ValueError(
                f"kind must be MBR_KIND_MEAS, MBR_KIND_POST, or MBR_KIND_URESET, got {kd}"
            )
        mx_c = _as_u64_words(mx, self._nw, "mx")
        mz_c = _as_u64_words(mz, self._nw, "mz")
        mk = _as_kappa(mkappa)
        out_p0 = ctypes.c_double()
        out_c0 = ctypes.c_int64()
        out_c1 = ctypes.c_int64()
        out_mt = ctypes.c_uint8()
        abs_x = np.zeros(self._nw, dtype=np.uint64)
        abs_z = np.zeros(self._nw, dtype=np.uint64)
        _check_rc(
            self._lib.mbr_store_step_meas(
                h, nid, eid, kd, addr(mx_c), addr(mz_c), mk,
                ctypes.addressof(out_p0), ctypes.addressof(out_c0), ctypes.addressof(out_c1),
                ctypes.addressof(out_mt), addr(abs_x), addr(abs_z),
            ),
            "mbr_store_step_meas",
        )
        return (
            float(out_p0.value),
            int(out_c0.value),
            int(out_c1.value),
            int(out_mt.value),
            abs_x,
            abs_z,
        )

    def step_peek(
        self,
        node_id: int,
        event_id: int,
        mx: Any,
        mz: Any,
        mkappa: int,
    ) -> float:
        """Read-only Born probability `p0` for inner outcome 0 (`+1` eigenvalue)."""
        h = self._require_open()
        nid = _as_i64_scalar(node_id, "node_id")
        eid = _as_u64_scalar(event_id, "event_id")
        mx_c = _as_u64_words(mx, self._nw, "mx")
        mz_c = _as_u64_words(mz, self._nw, "mz")
        mk = _as_kappa(mkappa)
        out_p0 = ctypes.c_double()
        _check_rc(
            self._lib.mbr_store_step_peek(
                h, nid, eid, addr(mx_c), addr(mz_c), mk, ctypes.addressof(out_p0)
            ),
            "mbr_store_step_peek",
        )
        return float(out_p0.value)

    def batch_step_rotations(
        self,
        event_id: int,
        kind: int,
        gen_x: Any,
        gen_z: Any,
        base_x: Any,
        base_z: Any,
        base_kappa: Any,
        ax_e: Any,
        flip_masks: Any,
        fx: np.ndarray,
        fz: np.ndarray,
        nodes: np.ndarray,
        peaks: np.ndarray,
        *,
        sel_mask: Any = None,
    ) -> None:
        h = self._require_open()
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_T, MBR_KIND_CCZ):
            raise ValueError(f"kind must be MBR_KIND_T or MBR_KIND_CCZ, got {kd}")
        if not isinstance(nodes, np.ndarray) or nodes.ndim != 1:
            raise TypeError("nodes must be a 1D numpy.ndarray")
        bsz = int(nodes.shape[0])
        nodes_c = _as_rw_contiguous(nodes, np.int64, (bsz,), "nodes")
        peaks_c = _as_rw_contiguous(peaks, np.int16, (bsz,), "peaks")
        fx_c = _as_contiguous(fx, np.uint64, (bsz, self._nw), "fx")
        fz_c = _as_contiguous(fz, np.uint64, (bsz, self._nw), "fz")
        sel_c = None if sel_mask is None else _as_u8_mask(sel_mask, (bsz,), "sel_mask")

        gx_raw = gen_x if isinstance(gen_x, np.ndarray) else np.asarray(gen_x, dtype=object)
        if gx_raw.ndim != 2:
            raise ValueError(f"gen_x must be 2D, got shape {gx_raw.shape}")
        n_gens = int(gx_raw.shape[0])
        gx_c = _as_contiguous(gx_raw, np.uint64, (n_gens, self._nw), "gen_x")
        gz_c = _as_contiguous(gen_z, np.uint64, (n_gens, self._nw), "gen_z")

        bk_raw = base_kappa if isinstance(base_kappa, np.ndarray) else np.asarray(base_kappa, dtype=object)
        if bk_raw.ndim != 1:
            raise ValueError(f"base_kappa must be 1D, got shape {bk_raw.shape}")
        n_ax = int(bk_raw.shape[0])
        bx_c = _as_contiguous(base_x, np.uint64, (n_ax, self._nw), "base_x")
        bz_c = _as_contiguous(base_z, np.uint64, (n_ax, self._nw), "base_z")
        bk_c = _as_contiguous(bk_raw, np.uint8, (n_ax,), "base_kappa")
        ae_c = _as_contiguous(ax_e, np.uint8, (n_ax,), "ax_e")
        fm_c = _as_contiguous(flip_masks, np.uint32, (n_ax,), "flip_masks")

        _check_rc(
            self._lib.mbr_batch_step_rotations(
                h, bsz, 0 if sel_c is None else addr(sel_c), eid, kd,
                n_gens, addr(gx_c), addr(gz_c),
                n_ax, addr(bx_c), addr(bz_c), addr(bk_c), addr(ae_c), addr(fm_c),
                addr(fx_c), addr(fz_c), addr(nodes_c), addr(peaks_c),
            ),
            "mbr_batch_step_rotations",
        )

    def batch_step_qturns(
        self,
        event_id: int,
        kind: int,
        base_x: Any,
        base_z: Any,
        base_kappa: Any,
        fx: np.ndarray,
        fz: np.ndarray,
        nodes: np.ndarray,
        peaks: np.ndarray,
        *,
        sel_mask: Any = None,
    ) -> None:
        h = self._require_open()
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_SKIP, MBR_KIND_CFAULT):
            raise ValueError(f"kind must be MBR_KIND_SKIP or MBR_KIND_CFAULT, got {kd}")
        if not isinstance(nodes, np.ndarray) or nodes.ndim != 1:
            raise TypeError("nodes must be a 1D numpy.ndarray")
        bsz = int(nodes.shape[0])
        nodes_c = _as_rw_contiguous(nodes, np.int64, (bsz,), "nodes")
        peaks_c = _as_rw_contiguous(peaks, np.int16, (bsz,), "peaks")
        fx_c = _as_contiguous(fx, np.uint64, (bsz, self._nw), "fx")
        fz_c = _as_contiguous(fz, np.uint64, (bsz, self._nw), "fz")
        sel_c = None if sel_mask is None else _as_u8_mask(sel_mask, (bsz,), "sel_mask")

        bk_raw = base_kappa if isinstance(base_kappa, np.ndarray) else np.asarray(base_kappa, dtype=object)
        if bk_raw.ndim != 1:
            raise ValueError(f"base_kappa must be 1D, got shape {bk_raw.shape}")
        n_qt = int(bk_raw.shape[0])
        bx_c = _as_contiguous(base_x, np.uint64, (n_qt, self._nw), "base_x")
        bz_c = _as_contiguous(base_z, np.uint64, (n_qt, self._nw), "base_z")
        bk_c = _as_contiguous(bk_raw, np.uint8, (n_qt,), "base_kappa")

        _check_rc(
            self._lib.mbr_batch_step_qturns(
                h, bsz, 0 if sel_c is None else addr(sel_c), eid, kd,
                n_qt, addr(bx_c), addr(bz_c), addr(bk_c),
                addr(fx_c), addr(fz_c), addr(nodes_c), addr(peaks_c),
            ),
            "mbr_batch_step_qturns",
        )

    def batch_step_meas(
        self,
        event_id: int,
        kind: int,
        mx: Any,
        mz: Any,
        mkappa: int,
        uniforms: Any,
        fx: np.ndarray,
        fz: np.ndarray,
        nodes: np.ndarray,
        peaks: np.ndarray,
        *,
        log_mass: np.ndarray | None = None,
        update_log_mass: bool = True,
        sel_mask: Any = None,
        kick_x: Any = None,
        kick_z: Any = None,
        want_one_mask: Any = None,
        out_true_m: np.ndarray | None = None,
        out_p_chosen: np.ndarray | None = None,
    ) -> None:
        h = self._require_open()
        eid = _as_u64_scalar(event_id, "event_id")
        kd = _as_index(kind, "kind")
        if kd not in (MBR_KIND_MEAS, MBR_KIND_URESET):
            raise ValueError(f"kind must be MBR_KIND_MEAS or MBR_KIND_URESET, got {kd}")
        if not isinstance(nodes, np.ndarray) or nodes.ndim != 1:
            raise TypeError("nodes must be a 1D numpy.ndarray")
        bsz = int(nodes.shape[0])
        mx_c = _as_u64_words(mx, self._nw, "mx")
        mz_c = _as_u64_words(mz, self._nw, "mz")
        mk = _as_kappa(mkappa)
        u_c = _as_f64_contiguous(uniforms, (bsz,), "uniforms")
        fx_c = _as_rw_contiguous(fx, np.uint64, (bsz, self._nw), "fx")
        fz_c = _as_rw_contiguous(fz, np.uint64, (bsz, self._nw), "fz")
        nodes_c = _as_rw_contiguous(nodes, np.int64, (bsz,), "nodes")
        peaks_c = _as_rw_contiguous(peaks, np.int16, (bsz,), "peaks")
        sel_c = None if sel_mask is None else _as_u8_mask(sel_mask, (bsz,), "sel_mask")
        if (kick_x is None) != (kick_z is None):
            raise ValueError("kick_x and kick_z must either both be None or both provided")
        kx_c = None if kick_x is None else _as_u64_words(kick_x, self._nw, "kick_x")
        kz_c = None if kick_z is None else _as_u64_words(kick_z, self._nw, "kick_z")
        w1_c = None if want_one_mask is None else _as_u8_mask(want_one_mask, (bsz,), "want_one_mask")
        upd_lm = 1 if update_log_mass else 0
        if upd_lm:
            if log_mass is None:
                raise ValueError("log_mass must be provided when update_log_mass is True")
            lm_c = _as_rw_contiguous(log_mass, np.float64, (bsz,), "log_mass")
        else:
            lm_c = None
        if out_true_m is None:
            tm_c = None
        elif isinstance(out_true_m, np.ndarray) and out_true_m.dtype == np.bool_:
            tm_c = _as_rw_contiguous(out_true_m, np.bool_, (bsz,), "out_true_m").view(np.uint8)
        else:
            tm_c = _as_rw_contiguous(out_true_m, np.uint8, (bsz,), "out_true_m")
        pc_c = (
            None
            if out_p_chosen is None
            else _as_rw_contiguous(out_p_chosen, np.float64, (bsz,), "out_p_chosen")
        )

        _check_rc(
            self._lib.mbr_batch_step_meas(
                h, bsz, 0 if sel_c is None else addr(sel_c), eid, kd,
                addr(mx_c), addr(mz_c), mk, addr(u_c),
                0 if kx_c is None else addr(kx_c),
                0 if kz_c is None else addr(kz_c),
                0 if w1_c is None else addr(w1_c),
                upd_lm,
                addr(fx_c), addr(fz_c), addr(nodes_c), addr(peaks_c),
                0 if lm_c is None else addr(lm_c),
                0 if tm_c is None else addr(tm_c),
                0 if pc_c is None else addr(pc_c),
            ),
            "mbr_batch_step_meas",
        )

    def batch_step_peek(
        self,
        event_id: int,
        mx: Any,
        mz: Any,
        mkappa: int,
        fx: np.ndarray,
        fz: np.ndarray,
        nodes: np.ndarray,
        out_p1: np.ndarray,
    ) -> None:
        h = self._require_open()
        eid = _as_u64_scalar(event_id, "event_id")
        if not isinstance(nodes, np.ndarray) or nodes.ndim != 1:
            raise TypeError("nodes must be a 1D numpy.ndarray")
        bsz = int(nodes.shape[0])
        mx_c = _as_u64_words(mx, self._nw, "mx")
        mz_c = _as_u64_words(mz, self._nw, "mz")
        mk = _as_kappa(mkappa)
        fx_c = _as_contiguous(fx, np.uint64, (bsz, self._nw), "fx")
        fz_c = _as_contiguous(fz, np.uint64, (bsz, self._nw), "fz")
        nodes_c = _as_contiguous(nodes, np.int64, (bsz,), "nodes")
        p1_c = _as_rw_contiguous(out_p1, np.float64, (bsz,), "out_p1")

        _check_rc(
            self._lib.mbr_batch_step_peek(
                h, bsz, eid, addr(mx_c), addr(mz_c), mk,
                addr(fx_c), addr(fz_c), addr(nodes_c), addr(p1_c),
            ),
            "mbr_batch_step_peek",
        )

    def batch_step_postselect(
        self,
        event_id: int,
        mx: Any,
        mz: Any,
        mkappa: int,
        desired_bits: Any,
        fx: np.ndarray,
        fz: np.ndarray,
        nodes: np.ndarray,
        peaks: np.ndarray,
        log_mass: np.ndarray,
        *,
        sel_mask: Any = None,
    ) -> int:
        """Apply postselection across active shots atomically; returns `-1` on success or first failing shot index."""
        h = self._require_open()
        eid = _as_u64_scalar(event_id, "event_id")
        if not isinstance(nodes, np.ndarray) or nodes.ndim != 1:
            raise TypeError("nodes must be a 1D numpy.ndarray")
        bsz = int(nodes.shape[0])
        mx_c = _as_u64_words(mx, self._nw, "mx")
        mz_c = _as_u64_words(mz, self._nw, "mz")
        mk = _as_kappa(mkappa)
        des_c = _as_u8_mask(desired_bits, (bsz,), "desired_bits")
        fx_c = _as_rw_contiguous(fx, np.uint64, (bsz, self._nw), "fx")
        fz_c = _as_rw_contiguous(fz, np.uint64, (bsz, self._nw), "fz")
        nodes_c = _as_rw_contiguous(nodes, np.int64, (bsz,), "nodes")
        peaks_c = _as_rw_contiguous(peaks, np.int16, (bsz,), "peaks")
        lm_c = _as_rw_contiguous(log_mass, np.float64, (bsz,), "log_mass")
        sel_c = None if sel_mask is None else _as_u8_mask(sel_mask, (bsz,), "sel_mask")
        out_fail = ctypes.c_int64(-1)

        _check_rc(
            self._lib.mbr_batch_step_postselect(
                h, bsz, 0 if sel_c is None else addr(sel_c), eid,
                addr(mx_c), addr(mz_c), mk, addr(des_c),
                addr(fx_c), addr(fz_c), addr(nodes_c), addr(peaks_c), addr(lm_c),
                ctypes.addressof(out_fail),
            ),
            "mbr_batch_step_postselect",
        )
        return int(out_fail.value)

    def counters(self) -> dict[str, Any]:
        """Return Section 7 diagnostic counters owned by the native store."""
        h = self._require_open()
        raw = np.empty(26, dtype=np.uint64)
        _check_rc(self._lib.mbr_store_counters(h, addr(raw)), "mbr_store_counters")

        n_hist = ctypes.c_size_t()
        _check_rc(
            self._lib.mbr_store_branches_hist(h, 0, 0, 0, ctypes.addressof(n_hist)),
            "mbr_store_branches_hist",
        )
        nh = int(n_hist.value)
        hist: dict[int, int] = {}
        if nh > 0:
            hk = np.empty(nh, dtype=np.uint64)
            hc = np.empty(nh, dtype=np.uint64)
            _check_rc(
                self._lib.mbr_store_branches_hist(h, addr(hk), addr(hc), nh, ctypes.addressof(n_hist)),
                "mbr_store_branches_hist",
            )
            for i in range(int(n_hist.value)):
                hist[int(hk[i])] = int(hc[i])

        return {
            "store_nodes": int(raw[0]),
            "store_bytes_estimate": int(raw[1]),
            "store_clears": int(raw[2]),
            "transitions_new": {
                name: int(raw[3 + i]) for i, name in enumerate(MBR_KIND_NAMES)
            },
            "transitions_hit": {
                name: int(raw[11 + i]) for i, name in enumerate(MBR_KIND_NAMES)
            },
            "resolve_calls": int(raw[19]),
            "splits": int(raw[20]),
            "subcache_hits": int(raw[21]),
            "subcache_misses": int(raw[22]),
            "meas_definite": int(raw[23]),
            "meas_absorbed": int(raw[24]),
            "meas_forked": int(raw[25]),
            "branches_hist": hist,
        }


@functools.cache
def branch_abi_ok() -> bool:
    """Verify `mbr_abi_v1() == 1` and run a fast deterministic native round-trip self-check."""
    lib = get_branch_kernels_lib()
    if int(lib.mbr_abi_v1()) != 1:
        raise RuntimeError(f"unexpected mbr_abi_v1 value: {lib.mbr_abi_v1()}")
    x1 = np.array([1], dtype=np.uint64)
    z0 = np.array([0], dtype=np.uint64)
    z1 = np.array([1], dtype=np.uint64)
    with BranchState.zero(1) as s0, BranchState.zero(1) as s1:
        # R(-Y)|0> = (I - iY)/sqrt(2)|0> = |+>
        f_qt = s1.qturn(x1, z1, 3)  # -Y = i^3 X Z
        f_c = s1.canon()
        if abs(f_qt - 1.0) > 1e-15 or abs(f_c - 1.0) > 1e-15:
            raise RuntimeError(f"branch self-check scalar mismatch: {f_qt}, {f_c}")
        if s1.definite(x1, z0, 0) != 1 or s1.definite(z0, z1, 0) != 0:
            raise RuntimeError("branch self-check definite mismatch")
        ov = s0.overlap(s1)
        if abs(ov - math.sqrt(0.5)) > 1e-15:
            raise RuntimeError(f"branch self-check overlap mismatch: {ov}")
    return True
