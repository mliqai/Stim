"""Thin ctypes ABI bindings and RAII handle for the `mbr_*` stabilizer-branch kernels.

Compiled into `libmagic_kernels.so` via `branch_kernels.hpp`. All stabilizer-state math executes
in C++; this module provides only ABI validation, status-code translation, and a thin RAII
`BranchState` wrapper with duck-typed conversion helpers.
"""

from __future__ import annotations

import ctypes
import functools
import math
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
_U8 = ctypes.c_uint8

MBR_OK = 0
MBR_ERR_INVALID_ARG = -1
MBR_ERR_OOM = -2
MBR_ERR_FLOAT_DRIFT = -3
MBR_ERR_BUDGET = -4

_MAX_QUBITS = 0x7FFFFFFF


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
    if isinstance(val, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    return operator.index(val)


def _as_contiguous(arr: Any, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    target = np.dtype(dtype)
    info = np.iinfo(target)
    if isinstance(arr, np.ndarray):
        a = arr
    else:
        a = np.asarray(arr, dtype=object)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if a.dtype == target and a.dtype.isnative and a.flags.c_contiguous and a.flags.aligned:
        return a
    if a.dtype == object:
        vals: list[int] = []
        for x in a.flat:
            if isinstance(x, bool):
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
