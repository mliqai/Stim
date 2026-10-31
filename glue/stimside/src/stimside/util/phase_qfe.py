"""Phase-preserving Moving-Reference Quadratic-Form Expansion (QFE) engine.

Implements Recipe v3 primitives for `magicside`: `C1` (`PhaseQFEComponent`,
`ShotExpansion`), `H0`–`H4`, `P1`–`P2`, `K1`, `N1`–`N3`, `M1`–`M2`, and `O1`–`O2`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from types import ModuleType
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.util.reference_chp import ReferenceCHPTableau

__all__ = [
    "DEFAULT_ATOL", "K1_CANCEL_REL_TOL", "ImpossiblePostselectionError",
    "n_words_for_qubits", "tail_mask", "pack_bit_vector", "unpack_bit_vector",
    "coerce_pauli_words", "unit_bit_words", "get_bit_in_words", "get_bits_at_qubits",
    "word_dot_parity", "word_dot_parity_batch", "words_any_nonzero", "words_any_overlap",
    "first_set_bit_in_words", "pauli_multiply", "validate_hermitian_pauli",
    "PhaseQFEComponent", "ShotExpansion", "ComponentTerm", "ExpansionInput",
    "MeasurementResult", "CompiledPauliBlockStep", "CompiledPauliBlock",
    "reference_chp_to_stim_tableau", "apply_reference_tableau_to_statevector",
    "drop_variable", "add_column", "make_principal", "restrict_amplitude",
    "eliminate_zero_amplitude", "apply_pauli", "apply_quarter_turn", "canonicalize",
    "merge_frames", "expansion_norm_sq", "apply_pauli_t", "apply_ccz",
    "apply_compiled_block", "project_diagonal", "measure_expansion",
    "overlap", "quadratic_gauss_sum", "quadratic_gauss_sum_dyadic", "validate_atol",
]

# Exact IEEE-754 float64 / complex128 constants
_SQRT2: float = math.sqrt(2.0)
_INV_SQRT2: float = math.ldexp(1.0, -1) * _SQRT2
_OMEGA: complex = complex(_INV_SQRT2, _INV_SQRT2)
_OMEGA_CONJ: complex = complex(_INV_SQRT2, -_INV_SQRT2)
_OMEGA_POW: NDArray[np.complex128] = np.array(
    [1.0 + 0.0j, _OMEGA, 1.0j, complex(-_INV_SQRT2, _INV_SQRT2),
     -1.0 + 0.0j, complex(-_INV_SQRT2, -_INV_SQRT2), -1.0j, _OMEGA_CONJ],
    dtype=np.complex128,
)
_PHASE4: NDArray[np.complex128] = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j], dtype=np.complex128)
_T_C0: complex = complex(0.5 + 0.5 * _INV_SQRT2, 0.5 * _INV_SQRT2)
_T_C1: complex = complex(0.5 - 0.5 * _INV_SQRT2, -0.5 * _INV_SQRT2)
_TDAG_C0: complex = complex(0.5 + 0.5 * _INV_SQRT2, -0.5 * _INV_SQRT2)
_TDAG_C1: complex = complex(0.5 - 0.5 * _INV_SQRT2, 0.5 * _INV_SQRT2)
_CCZ_WEIGHTS: NDArray[np.complex128] = np.array(
    [0.75, 0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25], dtype=np.complex128
)
K1_CANCEL_REL_TOL: float = 1e-13
DEFAULT_ATOL: float = 1e-10


class ImpossiblePostselectionError(ValueError):
    """Raised when M2 postselection requests an outcome with zero Born mass."""


def _is_valid_int(v: Any) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, np.integer))


def _is_bad_scalar(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_)) or (
        isinstance(v, np.ndarray) and (v.ndim != 0 or np.issubdtype(v.dtype, np.bool_))
    )


def validate_atol(atol: float) -> float:
    """Validate that `atol` is a strictly positive finite float (rejecting `bool`)."""
    if _is_bad_scalar(atol):
        raise ValueError(f"atol must be a finite positive float, got {atol!r}")
    try:
        val = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"atol must be a finite positive float, got {atol!r}") from exc
    if not (math.isfinite(val) and val > 0.0):
        raise ValueError(f"atol must be a finite positive float, got {atol!r}")
    return val


def _coerce_finite_complex(val: Any, name: str = "weight") -> complex:
    if _is_bad_scalar(val):
        raise ValueError(f"{name} must be a complex scalar, got {val!r}")
    try:
        c = complex(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a complex scalar, got {val!r}") from exc
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        raise FloatingPointError(f"Non-finite {name}: {val!r}")
    return c


def _check_unitary_norm(pre_norm: float, post_norm: float, atol: float, op_name: str) -> None:
    """Enforce unitary squared-norm conservation, catching NaN/Inf safely."""
    tol = validate_atol(atol)
    if not (
        math.isfinite(pre_norm) and math.isfinite(post_norm)
        and pre_norm >= 0.0 and post_norm >= 0.0
        and abs(post_norm - pre_norm) <= tol * max(1.0, pre_norm)
    ):
        raise FloatingPointError(
            f"{op_name} unitary norm conservation violated: "
            f"before={pre_norm!r}, after={post_norm!r}, atol={tol!r}"
        )


# ---------------------------------------------------------------------------
# Lazy optional M2 C++17 shared-library dispatch hook (`magic_kernels.py`)
# ---------------------------------------------------------------------------

_CACHED_MAGIC_LIB: ctypes.CDLL | None = None
_CACHED_MAGIC_MOD: ModuleType | None = None

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_u64_p = ctypes.POINTER(ctypes.c_uint64)
_c_u8_p = ctypes.POINTER(ctypes.c_uint8)
_c_i32_p = ctypes.POINTER(ctypes.c_int32)


def _reset_cpp_kernel_cache() -> None:
    """Reset cached M2 C++ kernel library handle."""
    global _CACHED_MAGIC_LIB, _CACHED_MAGIC_MOD
    _CACHED_MAGIC_LIB = None
    _CACHED_MAGIC_MOD = None


def _get_cpp_lib_if_enabled(use_cpp_kernels: bool) -> ctypes.CDLL | None:
    if not use_cpp_kernels:
        return None
    global _CACHED_MAGIC_LIB, _CACHED_MAGIC_MOD
    if _CACHED_MAGIC_LIB is not None:
        return _CACHED_MAGIC_LIB
    try:
        from stimside.util import magic_kernels as mk  # type: ignore[import-not-found]

        lib = mk.get_magic_kernels_lib()
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None
    _CACHED_MAGIC_MOD = mk
    _CACHED_MAGIC_LIB = lib
    return lib


def _c_ptr(arr: NDArray[Any], ptr_t: Any) -> Any:
    return ctypes.cast(arr.ctypes.data, ptr_t) if arr.size > 0 else ctypes.cast(0, ptr_t)


def _c_w_ptr(w: NDArray[np.complex128]) -> Any:
    return _c_ptr(w.view(np.float64), _c_double_p)


# ---------------------------------------------------------------------------
# Multi-word uint64 bitpacking & Pauli helpers
# ---------------------------------------------------------------------------


def n_words_for_qubits(n_qubits: int) -> int:
    """Return number of uint64 words needed for `n_qubits` (`max(1, ceil(N / 64))`)."""
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be non-negative, got {n_qubits}")
    return max(1, (n_qubits + 63) // 64)


def tail_mask(n_qubits: int) -> np.uint64:
    """Valid bit mask for the final uint64 word `n_words - 1`."""
    if n_qubits <= 0:
        return np.uint64(0)
    rem = n_qubits & 63
    return np.uint64(0xFFFFFFFFFFFFFFFF) if rem == 0 else (np.uint64(1) << np.uint64(rem)) - np.uint64(1)


def pack_bit_vector(bits: Sequence[int | bool] | NDArray[Any], n_qubits: int) -> NDArray[np.uint64]:
    """Pack a binary length-`n_qubits` sequence into `uint64[n_words]`."""
    n_words = n_words_for_qubits(n_qubits)
    arr = np.asarray(bits)
    if arr.ndim != 1 or arr.shape[0] != n_qubits:
        raise ValueError(f"Expected 1D bit vector of length {n_qubits}, got shape {arr.shape}")
    if arr.dtype != np.bool_ and np.any((arr != 0) & (arr != 1)):
        raise ValueError("Bit vector entries must be binary (0 or 1)")
    b_u64 = arr.astype(np.uint64, copy=False)
    out = np.zeros(n_words, dtype=np.uint64)
    if n_qubits == 0:
        return out
    shifts = np.arange(64, dtype=np.uint64)
    full_words = n_qubits >> 6
    for w in range(full_words):
        out[w] = np.bitwise_or.reduce(b_u64[(w << 6) : ((w + 1) << 6)] << shifts)
    rem = n_qubits & 63
    if rem:
        out[full_words] = np.bitwise_or.reduce(b_u64[(full_words << 6) :] << shifts[:rem])
    return out


def unpack_bit_vector(words: NDArray[np.uint64], n_qubits: int) -> NDArray[np.uint8]:
    """Unpack `uint64[..., n_words]` into `uint8[..., n_qubits]`."""
    n_words = n_words_for_qubits(n_qubits)
    w_arr = np.asarray(words, dtype=np.uint64)
    if w_arr.ndim < 1 or w_arr.shape[-1] != n_words:
        raise ValueError(f"Expected trailing dimension n_words={n_words}, got shape {w_arr.shape}")
    if n_qubits == 0:
        return np.zeros(w_arr.shape[:-1] + (0,), dtype=np.uint8)
    q_idx = np.arange(n_qubits, dtype=np.intp)
    return ((w_arr[..., q_idx >> 6] >> (q_idx & 63).astype(np.uint64)) & np.uint64(1)).astype(np.uint8)


def coerce_pauli_words(vec: Sequence[int | bool] | NDArray[Any], n_qubits: int) -> NDArray[np.uint64]:
    """Coerce unpacked `(n_qubits,)` binary vector OR packed `uint64[n_words]`."""
    n_words = n_words_for_qubits(n_qubits)
    if isinstance(vec, np.ndarray) and vec.dtype == np.uint64 and vec.ndim == 1 and vec.shape[0] == n_words:
        if (vec[-1] & ~tail_mask(n_qubits)) != np.uint64(0):
            raise ValueError(f"Non-zero upper padding bits in packed uint64 word for n_qubits={n_qubits}")
        return np.ascontiguousarray(vec.copy(), dtype=np.uint64)
    return pack_bit_vector(vec, n_qubits)


def unit_bit_words(q: int, n_words: int) -> NDArray[np.uint64]:
    """Return packed basis vector `e_q` of shape `(n_words,)`."""
    if q < 0 or (q >> 6) >= n_words:
        raise ValueError(f"Qubit index {q} out of range for n_words={n_words}")
    out = np.zeros(n_words, dtype=np.uint64)
    out[q >> 6] = np.uint64(1) << np.uint64(q & 63)
    return out


def get_bit_in_words(words: NDArray[np.uint64], q: int) -> int:
    """Extract bit `q` (`0` or `1`) from 1D `uint64` array `words`."""
    return int((int(words[q >> 6]) >> (q & 63)) & 1)


def get_bits_at_qubits(words_2d: NDArray[np.uint64], qubits: NDArray[np.int32]) -> NDArray[np.uint8]:
    """Extract bits at `qubits` `(K,)` across `words_2d` `(M, n_words)` -> `uint8[M, K]`."""
    M, K = words_2d.shape[0], qubits.shape[0]
    if M == 0 or K == 0:
        return np.zeros((M, K), dtype=np.uint8)
    shifts = (qubits & 63).astype(np.uint64)[None, :]
    return np.ascontiguousarray(((words_2d[:, qubits >> 6] >> shifts) & np.uint64(1)).astype(np.uint8))


def word_dot_parity(u_words: NDArray[np.uint64], v_words: NDArray[np.uint64]) -> int:
    """Compute `<u, v> mod 2` across packed `uint64` word arrays."""
    if u_words.shape[0] == 1:
        return (int(u_words[0]) & int(v_words[0])).bit_count() & 1
    acc = 0
    for a, b in zip(u_words.tolist(), v_words.tolist()):
        acc ^= a & b
    return acc.bit_count() & 1


def word_dot_parity_batch(mat_words: NDArray[np.uint64], vec_words: NDArray[np.uint64]) -> NDArray[np.uint8]:
    """Compute `(mat_words @ vec_words) mod 2` -> `uint8[M]`."""
    if mat_words.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    if mat_words.shape[1] == 1:
        par = np.bitwise_count(mat_words[:, 0] & vec_words[0]) & 1
    else:
        par = np.bitwise_count(mat_words & vec_words[None, :]).sum(axis=1) & 1
    return np.ascontiguousarray(par.astype(np.uint8))


def words_any_nonzero(words: NDArray[np.uint64]) -> bool:
    """Return True iff at least one bit in `words` is non-zero."""
    return bool(int(words[0]) != 0) if words.shape[0] == 1 else bool(np.any(words != 0))


def words_any_overlap(u_words: NDArray[np.uint64], v_words: NDArray[np.uint64]) -> bool:
    """Return True iff `u_words & v_words` has at least one set bit."""
    if u_words.shape[0] == 1:
        return bool((int(u_words[0]) & int(v_words[0])) != 0)
    return bool(np.any((u_words & v_words) != 0))


def first_set_bit_in_words(words: NDArray[np.uint64]) -> int:
    """Return smallest bit index `q >= 0` set in `words`, or `-1` if all zero."""
    for w_idx, val in enumerate(words.tolist()):
        if val != 0:
            return (w_idx << 6) + (val & -val).bit_length() - 1
    return -1


def pauli_multiply(
    a1_w: NDArray[np.uint64], b1_w: NDArray[np.uint64], kap1: int,
    a2_w: NDArray[np.uint64], b2_w: NDArray[np.uint64], kap2: int,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
    """Multiply `P(a1, b1, kap1) * P(a2, b2, kap2)` -> `(a12, b12, kap12)`."""
    kap12 = (int(kap1) + int(kap2) + (word_dot_parity(b1_w, a2_w) << 1)) & 3
    return np.ascontiguousarray(a1_w ^ a2_w), np.ascontiguousarray(b1_w ^ b2_w), kap12


def validate_hermitian_pauli(a_w: NDArray[np.uint64], b_w: NDArray[np.uint64], kappa: int) -> int:
    """Validate `(kappa - <a, b>) % 2 == 0` and return `kappa & 3`."""
    kap = int(kappa) & 3
    ab = word_dot_parity(a_w, b_w)
    if ((kap - ab) & 1) != 0:
        raise ValueError(f"Pauli operator P(a, b, kappa={kap}) must be Hermitian (<a,b>={ab})")
    return kap


def _unpack_pauli_args(
    a: Any, b: Any, kappa: int, n_qubits: int
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
    if b is None:
        if hasattr(a, "a_words") and hasattr(a, "b_words") and hasattr(a, "kappa"):
            a, b, kappa = a.a_words, a.b_words, a.kappa
        elif isinstance(a, tuple) and len(a) == 3:
            a, b, kappa = a
        else:
            raise ValueError("Expected (a, b, kappa) or a Pauli axis object")
    return coerce_pauli_words(a, n_qubits), coerce_pauli_words(b, n_qubits), int(kappa) & 3


# ---------------------------------------------------------------------------
# Core Data Structures (`C1`)
# ---------------------------------------------------------------------------


def _compute_a_union(A_words: NDArray[np.uint64], d: int, n_words: int) -> NDArray[np.uint64]:
    if d == 0:
        return np.zeros(n_words, dtype=np.uint64)
    if d == 1:
        return np.ascontiguousarray(A_words[0].copy())
    return np.ascontiguousarray(np.bitwise_or.reduce(A_words, axis=0))


def _check_shared_geometry(
    n_qubits: int, d: int, A_words: Any, Gamma: Any, pivots: Any, *, require_c_contiguous: bool = False
) -> tuple[int, np.uint64]:
    if not _is_valid_int(n_qubits) or n_qubits < 1:
        raise ValueError(f"n_qubits must be an int >= 1, got {n_qubits!r}")
    if not _is_valid_int(d) or d < 0 or d > n_qubits:
        raise ValueError(f"Invalid d={d!r} for n_qubits={n_qubits}")
    nw = n_words_for_qubits(int(n_qubits))
    for name, arr, dt, shp in (
        ("A_words", A_words, np.uint64, (d, nw)),
        ("Gamma", Gamma, np.uint8, (d, d)),
        ("pivots", pivots, np.int32, (d,)),
    ):
        if not isinstance(arr, np.ndarray) or arr.dtype != dt or arr.shape != shp:
            raise ValueError(f"Invalid {name} dtype/shape: {getattr(arr, 'dtype', type(arr))}, {getattr(arr, 'shape', None)}")
    if require_c_contiguous and not (
        A_words.flags.c_contiguous and Gamma.flags.c_contiguous and pivots.flags.c_contiguous
    ):
        raise ValueError("Geometry arrays must be C-contiguous")
    tm = tail_mask(int(n_qubits))
    if d > 0:
        if np.any((A_words[:, -1] & ~tm) != np.uint64(0)):
            raise ValueError("A_words has non-zero upper tail padding bits")
        piv_list = pivots.tolist()
        if any(p < 0 or p >= n_qubits for p in piv_list) or len(set(piv_list)) != d:
            raise ValueError(f"Invalid or duplicate pivots: {piv_list}")
        if np.any((Gamma != 0) & (Gamma != 1)) or not np.array_equal(Gamma, Gamma.T) or np.any(np.diag(Gamma) != 0):
            raise ValueError("Gamma must be binary, symmetric, and hollow")
        if not np.array_equal(get_bits_at_qubits(A_words, pivots), np.eye(d, dtype=np.uint8)):
            raise ValueError("Principal submatrix A[pivots, :] != I_d")
    return nw, tm


def _canonicalize_rows_inplace(
    h_2d: NDArray[np.uint64], ell_2d: NDArray[np.uint8],
    A_words: NDArray[np.uint64], Gamma: NDArray[np.uint8],
    pivots: NDArray[np.int32], ell_odd: NDArray[np.uint8],
) -> tuple[NDArray[np.intp], NDArray[np.complex128]]:
    """Clear `h_2d[:, pivots]` in-place; return `(dirty_indices, phase_factors)`."""
    if h_2d.shape[0] == 0 or pivots.shape[0] == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.complex128)
    s = get_bits_at_qubits(h_2d, pivots)
    idx = np.flatnonzero(np.any(s != 0, axis=1))
    if len(idx) == 0:
        return idx, np.zeros(0, dtype=np.complex128)
    s_d, ell_d = s[idx], ell_2d[idx]
    s_i = s_d.astype(np.int32)
    q_mod4 = (
        np.sum(s_i * ell_d.astype(np.int32), axis=1)
        + 2 * np.sum((s_i @ np.triu(Gamma, 1).astype(np.int32)) * s_i, axis=1)
    ) & 3
    for r in range(pivots.shape[0]):
        sel_r = idx[s_d[:, r] != 0]
        if len(sel_r) > 0:
            h_2d[sel_r] ^= A_words[r]
    polar = ((s_d @ Gamma) ^ (s_d & ell_odd[None, :])) & np.uint8(1)
    ell_2d[idx] = (ell_d + (polar << np.uint8(1))) & np.uint8(3)
    return idx, _PHASE4[q_mod4]


@dataclass
class PhaseQFEComponent:
    """Normalized Phase-QFE component on `n_qubits` qubits with dimension `d`."""

    n_qubits: int
    d: int
    h_words: NDArray[np.uint64]
    A_words: NDArray[np.uint64]
    ell: NDArray[np.uint8]
    Gamma: NDArray[np.uint8]
    pivots: NDArray[np.int32]

    @property
    def n_words(self) -> int:
        return n_words_for_qubits(self.n_qubits)

    @property
    def a_union(self) -> NDArray[np.uint64]:
        return _compute_a_union(self.A_words, self.d, self.n_words)

    @classmethod
    def zero_state(cls, n_qubits: int) -> PhaseQFEComponent:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        nw = n_words_for_qubits(n_qubits)
        return cls(
            int(n_qubits), 0, np.zeros(nw, dtype=np.uint64), np.zeros((0, nw), dtype=np.uint64),
            np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
        )

    @classmethod
    def zero(cls, n_qubits: int) -> PhaseQFEComponent:
        return cls.zero_state(n_qubits)

    @classmethod
    def computational_basis_state(cls, n_qubits: int, bits: Sequence[int | bool] | NDArray[Any]) -> PhaseQFEComponent:
        comp = cls.zero_state(n_qubits)
        comp.h_words = coerce_pauli_words(bits, n_qubits)
        return comp

    @classmethod
    def from_unpacked(
        cls,
        h: Sequence[int] | NDArray[Any],
        A: NDArray[Any],
        ell: Sequence[int] | NDArray[Any],
        Gamma: NDArray[Any],
        pivots: Sequence[int] | NDArray[Any],
    ) -> PhaseQFEComponent:
        h_arr = np.asarray(h, dtype=np.uint8)
        n_qubits = int(h_arr.shape[0])
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        A_arr = np.asarray(A, dtype=np.uint8)
        if A_arr.ndim != 2 or A_arr.shape[0] != n_qubits:
            raise ValueError(f"Expected unpacked A shape ({n_qubits}, d), got {A_arr.shape}")
        d = int(A_arr.shape[1])
        A_words = np.zeros((d, n_words_for_qubits(n_qubits)), dtype=np.uint64)
        for j in range(d):
            A_words[j] = pack_bit_vector(A_arr[:, j], n_qubits)
        return cls(
            n_qubits=n_qubits, d=d, h_words=pack_bit_vector(h_arr, n_qubits), A_words=A_words,
            ell=np.ascontiguousarray(np.asarray(ell, dtype=np.uint8)),
            Gamma=np.ascontiguousarray(np.asarray(Gamma, dtype=np.uint8)),
            pivots=np.ascontiguousarray(np.asarray(pivots, dtype=np.int32)),
        )

    def copy(self) -> PhaseQFEComponent:
        return PhaseQFEComponent(
            self.n_qubits, self.d,
            np.ascontiguousarray(self.h_words.copy()),
            np.ascontiguousarray(self.A_words.copy()),
            np.ascontiguousarray(self.ell.copy()),
            np.ascontiguousarray(self.Gamma.copy()),
            np.ascontiguousarray(self.pivots.copy()),
        )

    def check_invariants(self, *, require_canonical_offset: bool = False) -> None:
        nw, tm = _check_shared_geometry(
            self.n_qubits, self.d, self.A_words, self.Gamma, self.pivots, require_c_contiguous=True
        )
        if not isinstance(self.h_words, np.ndarray) or self.h_words.dtype != np.uint64 or self.h_words.shape != (nw,):
            raise ValueError(f"Invalid h_words dtype/shape: {getattr(self.h_words, 'dtype', type(self.h_words))}, {getattr(self.h_words, 'shape', None)}")
        if not isinstance(self.ell, np.ndarray) or self.ell.dtype != np.uint8 or self.ell.shape != (self.d,):
            raise ValueError(f"Invalid ell dtype/shape: {getattr(self.ell, 'dtype', type(self.ell))}, {getattr(self.ell, 'shape', None)}")
        if not (self.h_words.flags.c_contiguous and self.ell.flags.c_contiguous):
            raise ValueError("PhaseQFEComponent arrays must be C-contiguous")
        if (self.h_words[-1] & ~tm) != np.uint64(0):
            raise ValueError("h_words has non-zero upper tail padding bits")
        if self.d > 0:
            if np.any(self.ell >= 4):
                raise ValueError(f"ell entries must be in {{0,1,2,3}}, got {self.ell}")
            if require_canonical_offset and np.any(get_bits_at_qubits(self.h_words[None, :], self.pivots) != 0):
                raise ValueError("Non-canonical offset: h_p != 0 at pivot")

    def canonical_offset(self) -> complex:
        """Clear `h[pivots]` in-place and return exact scalar phase factor in `{1, i, -1, -i}`."""
        if self.d == 0:
            return 1.0 + 0.0j
        ell_2d = self.ell.copy()[None, :]
        idx, phases = _canonicalize_rows_inplace(
            self.h_words[None, :], ell_2d, self.A_words, self.Gamma, self.pivots, self.ell & np.uint8(1)
        )
        if len(idx) == 0:
            return 1.0 + 0.0j
        self.ell = np.ascontiguousarray(ell_2d[0])
        return complex(phases[0])

    def geometry_key(self) -> bytes:
        header = np.array([self.n_qubits, self.d], dtype=np.int64).tobytes()
        ell_par = np.ascontiguousarray(self.ell & np.uint8(1)).tobytes()
        return header + self.A_words.tobytes() + self.Gamma.tobytes() + ell_par + self.pivots.tobytes()

    def key(self) -> tuple[bytes, bytes, bytes]:
        return (self.geometry_key(), self.h_words.tobytes(), self.ell.tobytes())

    def to_statevector(self, *, endian: Literal["little", "big"] = "little") -> NDArray[np.complex128]:
        """Materialize dense `2**n_qubits` statevector (for small-N verification)."""
        if self.n_qubits > 26:
            raise ValueError(f"to_statevector requires n_qubits <= 26, got {self.n_qubits}")
        out = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        if self.d == 0:
            idx = int(self.h_words[0])
            if endian == "big":
                idx = int(f"{idx:0{self.n_qubits}b}"[::-1], 2)
            out[idx] = 1.0 + 0.0j
            return out
        u = (
            (np.arange(1 << self.d, dtype=np.uint64)[:, None] >> np.arange(self.d, dtype=np.uint64)[None, :])
            & np.uint64(1)
        ).astype(np.uint8)
        u_i = u.astype(np.int32)
        quad = np.sum((u_i @ np.triu(self.Gamma, 1).astype(np.int32)) * u_i, axis=1)
        q_mod4 = (u_i @ self.ell.astype(np.int32) + 2 * quad) & 3
        x_idx = np.bitwise_xor.reduce(np.where(u != 0, self.A_words[:, 0][None, :], np.uint64(0)), axis=1) ^ self.h_words[0]
        if endian == "big":
            rev = np.zeros_like(x_idx)
            for b in range(self.n_qubits):
                rev |= ((x_idx >> np.uint64(b)) & np.uint64(1)) << np.uint64(self.n_qubits - 1 - b)
            x_idx = rev
        scale = math.ldexp(1.0, -(self.d >> 1)) * (_INV_SQRT2 if (self.d & 1) else 1.0)
        np.add.at(out, x_idx.astype(np.intp), scale * _PHASE4[q_mod4])
        return out


@dataclass
class ShotExpansion:
    """Shared-geometry vectorized Phase-QFE expansion across `chi` components."""

    n_qubits: int
    d: int
    A_words: NDArray[np.uint64]
    a_union: NDArray[np.uint64]
    Gamma: NDArray[np.uint8]
    pivots: NDArray[np.int32]
    weights: NDArray[np.complex128]
    h_words: NDArray[np.uint64]
    ell: NDArray[np.uint8]
    ell_odd: NDArray[np.uint8] | None = None
    log_mass: float = 0.0
    use_cpp_kernels: bool = True
    atol: float = DEFAULT_ATOL
    _is_canonical_merged: bool = True

    def __post_init__(self) -> None:
        self.atol = validate_atol(self.atol)
        self._is_canonical_merged = bool(self._is_canonical_merged)
        self.A_words = np.ascontiguousarray(self.A_words, dtype=np.uint64)
        self.a_union = np.ascontiguousarray(self.a_union, dtype=np.uint64)
        self.Gamma = np.ascontiguousarray(self.Gamma, dtype=np.uint8)
        self.pivots = np.ascontiguousarray(self.pivots, dtype=np.int32)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.complex128)
        self.h_words = np.ascontiguousarray(self.h_words, dtype=np.uint64)
        self.ell = np.ascontiguousarray(self.ell, dtype=np.uint8)
        if self.ell_odd is None:
            self.ell_odd = (
                np.ascontiguousarray(self.ell[0] & np.uint8(1), dtype=np.uint8)
                if self.weights.shape[0] > 0
                else np.zeros(self.d, dtype=np.uint8)
            )
        else:
            self.ell_odd = np.ascontiguousarray(self.ell_odd, dtype=np.uint8)

    @property
    def chi(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_words(self) -> int:
        return n_words_for_qubits(self.n_qubits)

    def __len__(self) -> int:
        return int(self.weights.shape[0])

    def _refresh_a_union(self) -> None:
        self.a_union = _compute_a_union(self.A_words, self.d, self.n_words)

    @classmethod
    def zero_state(
        cls, n_qubits: int, *, use_cpp_kernels: bool = True, atol: float = DEFAULT_ATOL
    ) -> ShotExpansion:
        if not _is_valid_int(n_qubits) or n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits!r}")
        nw = n_words_for_qubits(n_qubits)
        return cls(
            int(n_qubits), 0, np.zeros((0, nw), dtype=np.uint64), np.zeros(nw, dtype=np.uint64),
            np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
            np.array([1.0 + 0.0j], dtype=np.complex128), np.zeros((1, nw), dtype=np.uint64),
            np.zeros((1, 0), dtype=np.uint8), np.zeros(0, dtype=np.uint8),
            0.0, use_cpp_kernels, atol,
        )

    @classmethod
    def zero(
        cls, n_qubits: int, *, use_cpp_kernels: bool = True, atol: float = DEFAULT_ATOL
    ) -> ShotExpansion:
        return cls.zero_state(n_qubits, use_cpp_kernels=use_cpp_kernels, atol=atol)

    @classmethod
    def from_component(
        cls, comp: PhaseQFEComponent, weight: complex = 1.0 + 0.0j, *,
        use_cpp_kernels: bool = True, atol: float = DEFAULT_ATOL,
    ) -> ShotExpansion:
        if not isinstance(comp, PhaseQFEComponent):
            raise ValueError(f"Expected PhaseQFEComponent, got {type(comp)!r}")
        return cls(
            comp.n_qubits, comp.d, comp.A_words.copy(), comp.a_union,
            comp.Gamma.copy(), comp.pivots.copy(),
            np.array([complex(weight)], dtype=np.complex128),
            comp.h_words.copy()[None, :], comp.ell.copy()[None, :],
            (comp.ell & np.uint8(1)).copy(), 0.0, use_cpp_kernels, atol,
        )

    @classmethod
    def from_components(
        cls,
        terms: Sequence[tuple[complex, PhaseQFEComponent]],
        *,
        n_qubits: int | None = None,
        merge: bool = True,
        use_cpp_kernels: bool = True,
        atol: float = DEFAULT_ATOL,
    ) -> ShotExpansion:
        if len(terms) == 0:
            if n_qubits is None or not _is_valid_int(n_qubits) or n_qubits < 1:
                raise ValueError("n_qubits >= 1 must be provided when constructing an empty ShotExpansion")
            return _empty_shot_expansion(int(n_qubits), use_cpp_kernels=use_cpp_kernels, atol=atol)
        n_q = _validate_component_terms(terms)
        if n_qubits is not None and (not _is_valid_int(n_qubits) or n_q != n_qubits):
            raise ValueError(f"Component n_qubits={n_q} != requested {n_qubits!r}")
        ref_comp = terms[0][1]
        g_key = ref_comp.geometry_key()
        if any(comp.geometry_key() != g_key for _, comp in terms[1:]):
            raise ValueError("ShotExpansion.from_components requires common geometry across all terms")
        ell = np.vstack([c.ell for _, c in terms]) if ref_comp.d > 0 else np.zeros((len(terms), 0), dtype=np.uint8)
        exp = cls(
            ref_comp.n_qubits, ref_comp.d, ref_comp.A_words.copy(), ref_comp.a_union,
            ref_comp.Gamma.copy(), ref_comp.pivots.copy(),
            np.array([complex(w) for w, _ in terms], dtype=np.complex128),
            np.vstack([c.h_words for _, c in terms]), ell,
            (ref_comp.ell & np.uint8(1)).copy(), 0.0, use_cpp_kernels, atol,
            _is_canonical_merged=(len(terms) <= 1),
        )
        return merge_frames(exp, use_cpp_kernels=use_cpp_kernels) if merge else exp  # type: ignore[return-value]

    def copy(self) -> ShotExpansion:
        assert self.ell_odd is not None
        return ShotExpansion(
            self.n_qubits, self.d, self.A_words.copy(), self.a_union.copy(),
            self.Gamma.copy(), self.pivots.copy(), self.weights.copy(),
            self.h_words.copy(), self.ell.copy(), self.ell_odd.copy(),
            self.log_mass, self.use_cpp_kernels, self.atol, self._is_canonical_merged,
        )

    def check_invariants(self, *, require_canonical_offset: bool = False) -> None:
        validate_atol(self.atol)
        nw, tm = _check_shared_geometry(self.n_qubits, self.d, self.A_words, self.Gamma, self.pivots)
        chi = self.chi
        for name, arr, dt, shp in (
            ("a_union", self.a_union, np.uint64, (nw,)),
            ("weights", self.weights, np.complex128, (chi,)),
            ("h_words", self.h_words, np.uint64, (chi, nw)),
            ("ell", self.ell, np.uint8, (chi, self.d)),
        ):
            if arr.dtype != dt or arr.shape != shp:
                raise ValueError(f"Invalid {name} shape/dtype: {arr.shape}, {arr.dtype}")
        if self.ell_odd is None or self.ell_odd.dtype != np.uint8 or self.ell_odd.shape != (self.d,):
            raise ValueError("Invalid ell_odd shape/dtype")
        if (self.a_union[-1] & ~tm) != np.uint64(0):
            raise ValueError("a_union has non-zero upper tail padding bits")
        if chi > 0 and np.any((self.h_words[:, -1] & ~tm) != np.uint64(0)):
            raise ValueError("h_words has non-zero upper tail padding bits")
        if not np.array_equal(self.a_union, _compute_a_union(self.A_words, self.d, nw)):
            raise ValueError("a_union != bitwise OR of A_words rows")
        if self.d > 0:
            if np.any((self.ell_odd != 0) & (self.ell_odd != 1)):
                raise ValueError("ell_odd must be binary")
            if chi > 0:
                if np.any(self.ell >= 4):
                    raise ValueError("ell entries must be in {0, 1, 2, 3}")
                if not np.array_equal(self.ell & np.uint8(1), np.broadcast_to(self.ell_odd[None, :], (chi, self.d))):
                    raise ValueError("ell & 1 must match shared ell_odd across all components")
                if require_canonical_offset and np.any(get_bits_at_qubits(self.h_words, self.pivots) != 0):
                    raise ValueError("Non-canonical h_words on pivots")

    def to_components(self) -> list[tuple[complex, PhaseQFEComponent]]:
        return [
            (
                complex(self.weights[i]),
                PhaseQFEComponent(
                    self.n_qubits, self.d, self.h_words[i].copy(), self.A_words.copy(),
                    self.ell[i].copy(), self.Gamma.copy(), self.pivots.copy(),
                ),
            )
            for i in range(self.chi)
        ]

    def norm_sq(self, *, canonicalize_first: bool | None = None) -> float:
        if self.chi == 0:
            return 0.0
        do_canon = (not self._is_canonical_merged) if canonicalize_first is None else bool(canonicalize_first)
        exp = merge_frames(self, use_cpp_kernels=self.use_cpp_kernels) if (do_canon and self.chi > 1) else self
        assert isinstance(exp, ShotExpansion)
        if exp.chi == 0:
            return 0.0
        re, im = exp.weights.real, exp.weights.imag
        return float(np.sum(re * re + im * im))

    def branch_mass(self) -> float:
        nu = self.norm_sq()
        if nu <= 0.0 or self.log_mass < -745.0:
            return 0.0
        return float(nu * math.exp(self.log_mass))

    def log_branch_mass(self) -> float:
        nu = self.norm_sq()
        return -math.inf if nu <= 0.0 else float(math.log(nu) + self.log_mass)

    def to_statevector(
        self,
        *,
        ref_tableau: ReferenceCHPTableau | stim.Tableau | None = None,
        endian: Literal["little", "big"] = "little",
    ) -> NDArray[np.complex128]:
        if self.n_qubits > 26:
            raise ValueError(f"to_statevector requires n_qubits <= 26, got {self.n_qubits}")
        psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        for w, comp in self.to_components():
            if w != 0.0j:
                psi += w * comp.to_statevector(endian=endian)
        return apply_reference_tableau_to_statevector(psi, ref_tableau, endian=endian) if ref_tableau is not None else psi


def _empty_shot_expansion(
    n_qubits: int, *, log_mass: float = 0.0, use_cpp_kernels: bool = True, atol: float = DEFAULT_ATOL
) -> ShotExpansion:
    nw = n_words_for_qubits(n_qubits)
    return ShotExpansion(
        n_qubits, 0, np.zeros((0, nw), dtype=np.uint64), np.zeros(nw, dtype=np.uint64),
        np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.complex128), np.zeros((0, nw), dtype=np.uint64),
        np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.uint8),
        log_mass, use_cpp_kernels, atol,
    )


ComponentTerm: TypeAlias = tuple[complex, PhaseQFEComponent]
ExpansionInput: TypeAlias = ShotExpansion | Sequence[ComponentTerm]


@dataclass(frozen=True)
class MeasurementResult:
    """Result of `measure_expansion` (`M2`)."""

    outcome: int
    reported_outcome: int
    readout_flipped: bool
    z0: float
    z1: float
    input_norm_sq: float
    chosen_mass: float
    log_branch_mass: float
    post_state: ShotExpansion | list[ComponentTerm]


@dataclass(frozen=True)
class CompiledPauliBlockStep:
    """Single unflattened step inside a `CompiledPauliBlock` (`mode='sequential'`)."""

    kind: Literal["P1", "P2", "N1", "N2"]
    paulis: tuple[tuple[NDArray[np.uint64], NDArray[np.uint64], int], ...]
    sign: int = 1
    weights: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class CompiledPauliBlock:
    """Unified compiled open non-Clifford block for `C2` (`compile_open_block`) and `N3` (`apply_compiled_block`)."""

    n_qubits: int
    mode: Literal["pauli_sum", "sequential"] = "pauli_sum"
    coeffs: NDArray[np.complex128] | None = None
    a_words: NDArray[np.uint64] | None = None
    b_words: NDArray[np.uint64] | None = None
    kappas: NDArray[np.uint8] | None = None
    steps: tuple[CompiledPauliBlockStep | CompiledPauliBlock, ...] = ()
    global_scalar: complex = 1.0 + 0.0j
    is_unitary: bool = True
    is_identity: bool = False
    cache_key: tuple[Any, ...] = ()
    n_qubits_exact: bool = True


# ---------------------------------------------------------------------------
# Exact complex128 ReferenceCHPTableau -> Statevector helper
# ---------------------------------------------------------------------------

_ELIM_1Q_MATS: dict[str, NDArray[np.complex128]] = {
    "H": np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=np.complex128),
    "S": np.array([[1.0 + 0.0j, 0.0j], [0.0j, 1.0j]], dtype=np.complex128),
    "X": np.array([[0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0j]], dtype=np.complex128),
    "Y": np.array([[0.0j, -1.0j], [1.0j, 0.0j]], dtype=np.complex128),
    "Z": np.array([[1.0 + 0.0j, 0.0j], [0.0j, -1.0 + 0.0j]], dtype=np.complex128),
}
_C128_CX: NDArray[np.complex128] = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    dtype=np.complex128,
)


def reference_chp_to_stim_tableau(ref_chp: ReferenceCHPTableau) -> stim.Tableau:
    """Convert `ReferenceCHPTableau` inverse symplectic tables into forward `stim.Tableau`."""
    n = ref_chp.num_qubits
    xs: list[stim.PauliString] = []
    zs: list[stim.PauliString] = []
    pauli_chars = "IZXY"
    for arr_x, arr_z, arr_k, out_list in (
        (ref_chp.x2x, ref_chp.x2z, ref_chp.x_kap, xs),
        (ref_chp.z2x, ref_chp.z2z, ref_chp.z_kap, zs),
    ):
        for q in range(n):
            aw, bw = arr_x[q], arr_z[q]
            exp4 = (int(arr_k[q]) - int(np.bitwise_count(aw & bw).sum())) & 3
            bx, bz = unpack_bit_vector(aw, n), unpack_bit_vector(bw, n)
            chars = "".join(pauli_chars[(int(bx[i]) << 1) | int(bz[i])] for i in range(n))
            out_list.append(stim.PauliString(("-" if exp4 == 2 else "+") + chars))
    return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs).inverse()


def _apply_local_unitary_c128(
    psi: NDArray[np.complex128],
    U_le: NDArray[np.complex128],
    targets: Sequence[int],
    n_qubits: int,
    endian: Literal["little", "big"],
) -> NDArray[np.complex128]:
    """Contract a 1Q or 2Q little-endian `complex128` unitary into `psi` in `O(2^N)`."""
    k = len(targets)
    axes = [(n_qubits - 1 - q) if endian == "little" else q for q in targets]
    U_t = U_le.reshape((2,) * (2 * k), order="F")
    contracted = np.tensordot(U_t, psi.reshape((2,) * n_qubits), axes=(list(range(k, 2 * k)), axes))
    cur_order = list(axes) + [ax for ax in range(n_qubits) if ax not in axes]
    inv_perm = [0] * n_qubits
    for pos, orig_ax in enumerate(cur_order):
        inv_perm[orig_ax] = pos
    return np.ascontiguousarray(np.transpose(contracted, inv_perm).reshape(1 << n_qubits))


def apply_reference_tableau_to_statevector(
    psi_ref: NDArray[np.complex128],
    ref_tableau: ReferenceCHPTableau | stim.Tableau,
    *,
    endian: Literal["little", "big"] = "little",
) -> NDArray[np.complex128]:
    """Transform `psi_ref` from reference coordinates to physical coordinates in `complex128`."""
    tab = reference_chp_to_stim_tableau(ref_tableau) if isinstance(ref_tableau, ReferenceCHPTableau) else ref_tableau
    n = len(tab)
    psi = np.ascontiguousarray(psi_ref, dtype=np.complex128).copy()
    if n == 0:
        return psi
    for inst in tab.to_circuit(method="elimination"):
        name = inst.name
        targs = [t.value for t in inst.targets_copy()]
        if name in _ELIM_1Q_MATS:
            U1 = _ELIM_1Q_MATS[name]
            for q in targs:
                psi = _apply_local_unitary_c128(psi, U1, (q,), n, endian)
        elif name == "CX":
            for i in range(0, len(targs), 2):
                psi = _apply_local_unitary_c128(psi, _C128_CX, (targs[i], targs[i + 1]), n, endian)
        else:
            raise ValueError(f"Unexpected gate {name!r} in tableau elimination circuit")
    return psi


# ---------------------------------------------------------------------------
# Structural Helpers: H0 - H4
# ---------------------------------------------------------------------------


def drop_variable(target: PhaseQFEComponent | ShotExpansion, col: int) -> None:
    """`H0`: Delete binary generator variable `col` in-place (`d -> d - 1`)."""
    c = int(col)
    if c < 0 or c >= target.d:
        raise ValueError(f"Column index {c} out of range for d={target.d}")
    target.A_words = np.ascontiguousarray(np.delete(target.A_words, c, axis=0))
    target.Gamma = np.ascontiguousarray(np.delete(np.delete(target.Gamma, c, axis=0), c, axis=1))
    target.pivots = np.ascontiguousarray(np.delete(target.pivots, c))
    if isinstance(target, PhaseQFEComponent):
        target.ell = np.ascontiguousarray(np.delete(target.ell, c))
        target.d -= 1
    elif isinstance(target, ShotExpansion):
        target.ell = np.ascontiguousarray(np.delete(target.ell, c, axis=1))
        assert target.ell_odd is not None
        target.ell_odd = np.ascontiguousarray(np.delete(target.ell_odd, c))
        target.d -= 1
        target._refresh_a_union()
        if target.chi > 1:
            target._is_canonical_merged = False
    else:
        raise TypeError(f"Unsupported target type: {type(target)!r}")


def add_column(target: PhaseQFEComponent | ShotExpansion, src: int, dst: int) -> None:
    """`H1`: Column addition `A_dst ^= A_src` with exact mod-4 `Q` pullback in-place."""
    r, s = int(src), int(dst)
    if r < 0 or r >= target.d or s < 0 or s >= target.d:
        raise ValueError(f"Column indices ({r}, {s}) out of range for d={target.d}")
    if r == s:
        raise ValueError("add_column requires distinct columns (src != dst)")
    old_edge = int(target.Gamma[r, s])
    src_row = target.Gamma[r, :].copy()
    target.A_words[s] ^= target.A_words[r]

    if isinstance(target, PhaseQFEComponent):
        parity = int(target.ell[r]) & 1
        target.ell[s] = np.uint8((int(target.ell[s]) + int(target.ell[r]) + 2 * old_edge) & 3)
    elif isinstance(target, ShotExpansion):
        assert target.ell_odd is not None
        parity = int(target.ell_odd[r]) & 1
        if target.chi > 0:
            target.ell[:, s] = (target.ell[:, s] + target.ell[:, r] + np.uint8(2 * old_edge)) & np.uint8(3)
        target.ell_odd[s] ^= np.uint8(parity)
        target._refresh_a_union()
    else:
        raise TypeError(f"Unsupported target type: {type(target)!r}")

    mask = np.ones(target.d, dtype=bool)
    mask[r] = False
    mask[s] = False
    new_edges = target.Gamma[s, mask] ^ src_row[mask]
    target.Gamma[s, mask] = new_edges
    target.Gamma[mask, s] = new_edges
    new_rs = np.uint8(old_edge ^ parity)
    target.Gamma[r, s] = new_rs
    target.Gamma[s, r] = new_rs


def make_principal(
    target: PhaseQFEComponent | ShotExpansion, col: int, pivot_qubit: int | None = None
) -> int:
    """`H2`: Clear bit `pivot_qubit` from all columns `!= col` and record pivot."""
    c = int(col)
    if c < 0 or c >= target.d:
        raise ValueError(f"Column index {c} out of range for d={target.d}")
    if pivot_qubit is None:
        p = first_set_bit_in_words(target.A_words[c])
        if p < 0:
            raise ValueError(f"Column {c} is identically zero; cannot make principal")
    else:
        p = int(pivot_qubit)
        if p < 0 or p >= target.n_qubits:
            raise ValueError(f"pivot_qubit={p} out of range for n_qubits={target.n_qubits}")
        if get_bit_in_words(target.A_words[c], p) == 0:
            raise ValueError(f"Column {c} has 0 at requested pivot_qubit={p}")
    if target.d > 1:
        col_bits = get_bits_at_qubits(target.A_words, np.array([p], dtype=np.int32))[:, 0]
        for j in range(target.d):
            if j != c and col_bits[j] != 0:
                add_column(target, c, j)
    target.pivots[c] = np.int32(p)
    return p


def restrict_amplitude(
    target: PhaseQFEComponent | ShotExpansion,
    lam: Sequence[int] | NDArray[np.integer],
    eps: int | Sequence[int] | NDArray[np.integer],
    *,
    scale: float = _INV_SQRT2,
) -> complex | None:
    """`H3`: Restrict to affine hyperplane `lam . u = eps (mod 2)` in-place."""
    if not isinstance(target, (PhaseQFEComponent, ShotExpansion)):
        raise TypeError(f"Unsupported target type: {type(target)!r}")
    lam_arr = (np.asarray(lam, dtype=np.uint8) & np.uint8(1)).ravel()
    if lam_arr.shape != (target.d,):
        raise ValueError(f"lam length {lam_arr.shape[0]} does not match d={target.d}")
    nz = np.flatnonzero(lam_arr)

    if isinstance(target, PhaseQFEComponent):
        eps_bit = int(eps) & 1  # type: ignore[arg-type]
        if len(nz) == 0:
            return (1.0 + 0.0j) if eps_bit == 0 else 0.0j
        f = int(nz[0])
        for k in nz[1:]:
            add_column(target, f, int(k))
        factor = complex(_PHASE4[(eps_bit * int(target.ell[f])) & 3]) * scale
        if eps_bit == 1:
            target.h_words ^= target.A_words[f]
            target.ell = np.ascontiguousarray((target.ell + (target.Gamma[f, :] << np.uint8(1))) & np.uint8(3))
        drop_variable(target, f)
        return factor

    if np.ndim(eps) == 0:
        eps_vec = np.full(target.chi, int(eps) & 1, dtype=np.uint8)  # type: ignore[arg-type]
    else:
        eps_vec = (np.asarray(eps, dtype=np.uint8) & np.uint8(1)).ravel()
        if eps_vec.shape != (target.chi,):
            raise ValueError(f"eps length {eps_vec.shape[0]} does not match chi={target.chi}")
    if len(nz) == 0:
        if target.chi > 0:
            keep = eps_vec == np.uint8(0)
            target.weights = np.ascontiguousarray(target.weights[keep])
            target.h_words = np.ascontiguousarray(target.h_words[keep])
            target.ell = np.ascontiguousarray(target.ell[keep])
        return None
    f = int(nz[0])
    for k in nz[1:]:
        add_column(target, f, int(k))
    if target.chi > 0:
        target.weights *= _PHASE4[(eps_vec * target.ell[:, f]) & np.uint8(3)] * scale
        ones = eps_vec != 0
        if np.any(ones):
            target.h_words[ones] ^= target.A_words[f]
            target.ell[ones] = (target.ell[ones] + (target.Gamma[f, :] << np.uint8(1))[None, :]) & np.uint8(3)
    drop_variable(target, f)
    return None


def _xor_hollow_outer_inplace(Gamma: NDArray[np.uint8], lam: NDArray[np.uint8]) -> None:
    upd = (lam[:, None] & lam[None, :]) & np.uint8(1)
    np.fill_diagonal(upd, 0)
    Gamma ^= upd


def eliminate_zero_amplitude(target: PhaseQFEComponent | ShotExpansion, col: int) -> complex | None:
    """`H4`: Analytically contract a zero amplitude column `A_words[col] == 0` in-place."""
    if not isinstance(target, (PhaseQFEComponent, ShotExpansion)):
        raise TypeError(f"Unsupported target type: {type(target)!r}")
    c = int(col)
    if c < 0 or c >= target.d:
        raise ValueError(f"Column index {c} out of range for d={target.d}")
    if words_any_nonzero(target.A_words[c]):
        raise ValueError(f"eliminate_zero_amplitude requires A_words[{c}] == 0, got non-zero column")
    lam = np.delete(target.Gamma[c, :], c).copy()

    if isinstance(target, PhaseQFEComponent):
        c0 = int(target.ell[c]) & 3
        drop_variable(target, c)
        if (c0 & 1) != 0:
            if target.d > 0:
                target.ell = np.ascontiguousarray((target.ell.astype(np.int16) - c0 * lam.astype(np.int16)) & 3, dtype=np.uint8)
                _xor_hollow_outer_inplace(target.Gamma, lam)
            return _OMEGA if c0 == 1 else _OMEGA_CONJ
        if np.any(lam != 0):
            return restrict_amplitude(target, lam, c0 >> 1, scale=1.0)
        return complex(_SQRT2, 0.0) if c0 == 0 else 0.0j

    assert target.ell_odd is not None
    c_odd = int(target.ell_odd[c]) & 1
    c_vec = target.ell[:, c].copy() if target.chi > 0 else np.zeros(0, dtype=np.uint8)
    drop_variable(target, c)
    if c_odd == 1:
        if target.chi > 0:
            target.weights *= np.where(c_vec == np.uint8(1), _OMEGA, _OMEGA_CONJ)
            if target.d > 0:
                neg_c = ((4 - c_vec.astype(np.int16)) & 3).astype(np.uint8)
                target.ell = np.ascontiguousarray((target.ell + neg_c[:, None] * lam[None, :]) & np.uint8(3))
        if target.d > 0:
            target.ell_odd ^= lam & np.uint8(1)
            _xor_hollow_outer_inplace(target.Gamma, lam)
        return None
    eps_vec = (c_vec >> np.uint8(1)) & np.uint8(1)
    if np.any(lam != 0):
        restrict_amplitude(target, lam, eps_vec, scale=1.0)
    elif target.chi > 0:
        keep = eps_vec == np.uint8(0)
        target.weights = np.ascontiguousarray(target.weights[keep] * _SQRT2)
        target.h_words = np.ascontiguousarray(target.h_words[keep])
        target.ell = np.ascontiguousarray(target.ell[keep])
    return None


# ---------------------------------------------------------------------------
# Unitary Frame Primitives: P1 (`apply_pauli`) and P2 (`apply_quarter_turn`)
# ---------------------------------------------------------------------------


def _validate_component_terms(terms: Sequence[ComponentTerm]) -> int:
    n_q = -1
    for idx, item in enumerate(terms):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Term {idx} must be a 2-tuple (weight, PhaseQFEComponent), got {item!r}")
        w, comp = item
        if not isinstance(comp, PhaseQFEComponent):
            raise ValueError(f"Term {idx} second element must be PhaseQFEComponent, got {type(comp)!r}")
        _coerce_finite_complex(w, f"term[{idx}] weight")
        comp.check_invariants(require_canonical_offset=False)
        if idx == 0:
            n_q = comp.n_qubits
        elif comp.n_qubits != n_q:
            raise ValueError(f"Inconsistent component n_qubits in expansion: term 0 has {n_q}, term {idx} has {comp.n_qubits}")
    return n_q


def _checked_shot_pre_norm(exp: ShotExpansion, check_unitary: bool, op_name: str) -> float:
    if exp.chi > 0 and not np.all(np.isfinite(exp.weights)):
        raise FloatingPointError(f"Non-finite weights in {op_name}")
    pre_norm = exp.norm_sq()
    if check_unitary and not math.isfinite(pre_norm):
        raise FloatingPointError(f"Non-finite input norm in {op_name}")
    return pre_norm


def _copy_shot_inplace(dst: ShotExpansion, src: ShotExpansion) -> ShotExpansion:
    dst.weights = src.weights
    dst.h_words = src.h_words
    dst.ell = src.ell
    if src.ell_odd is not None:
        dst.ell_odd = src.ell_odd
    dst._is_canonical_merged = src._is_canonical_merged
    return dst


def _clone_shot_with_rows(
    src: ShotExpansion,
    weights: NDArray[np.complex128],
    h_words: NDArray[np.uint64],
    ell: NDArray[np.uint8],
    *,
    is_canonical_merged: bool,
) -> ShotExpansion:
    assert src.ell_odd is not None
    return ShotExpansion(
        src.n_qubits, src.d, src.A_words.copy(), src.a_union.copy(),
        src.Gamma.copy(), src.pivots.copy(), weights, h_words, ell,
        src.ell_odd.copy(), src.log_mass, src.use_cpp_kernels, src.atol, is_canonical_merged,
    )


def _cpp_shot_in_ptrs(exp: ShotExpansion, *, include_a_union: bool = False) -> tuple[Any, ...]:
    assert exp.ell_odd is not None
    mid = (_c_ptr(exp.A_words, _c_u64_p), _c_ptr(exp.a_union, _c_u64_p)) if include_a_union else (_c_ptr(exp.A_words, _c_u64_p),)
    return (
        _c_w_ptr(exp.weights), _c_ptr(exp.h_words, _c_u64_p), _c_ptr(exp.ell, _c_u8_p),
        *mid, _c_ptr(exp.Gamma, _c_u8_p), _c_ptr(exp.pivots, _c_i32_p), _c_ptr(exp.ell_odd, _c_u8_p),
    )


def _alloc_cpp_row_bufs(cap: int, nw: int, d: int) -> tuple[Any, ...]:
    out_chi, out_norm = ctypes.c_size_t(0), ctypes.c_double(0.0)
    out_w = np.empty(cap, dtype=np.complex128)
    out_h = np.empty((cap, nw), dtype=np.uint64)
    out_ell = np.empty((cap, d), dtype=np.uint8)
    ptrs = (ctypes.byref(out_chi), _c_w_ptr(out_w), _c_ptr(out_h, _c_u64_p), _c_ptr(out_ell, _c_u8_p), ctypes.byref(out_norm))
    return out_chi, out_w, out_h, out_ell, ptrs


def _alloc_cpp_geom_bufs(chi: int, nw: int, d_cap: int) -> tuple[Any, ...]:
    out_chi, out_d = ctypes.c_size_t(0), ctypes.c_size_t(0)
    out_w = np.empty(chi, dtype=np.complex128)
    out_h = np.empty((chi, nw), dtype=np.uint64)
    out_ell = np.empty(chi * d_cap, dtype=np.uint8)
    out_A = np.empty((d_cap, nw), dtype=np.uint64)
    out_G = np.empty(d_cap * d_cap, dtype=np.uint8)
    out_piv = np.empty(d_cap, dtype=np.int32)
    out_eo = np.empty(d_cap, dtype=np.uint8)
    bufs = (out_w, out_h, out_ell, out_A, out_G, out_piv, out_eo)
    ptrs = (
        ctypes.byref(out_chi), ctypes.byref(out_d),
        _c_w_ptr(out_w), _c_ptr(out_h, _c_u64_p), _c_ptr(out_ell, _c_u8_p),
        _c_ptr(out_A, _c_u64_p), _c_ptr(out_G, _c_u8_p), _c_ptr(out_piv, _c_i32_p), _c_ptr(out_eo, _c_u8_p),
    )
    return out_chi, out_d, bufs, ptrs


def _assign_cpp_geom_shot(
    exp: ShotExpansion, oc: int, od: int, bufs: tuple[NDArray[Any], ...]
) -> ShotExpansion:
    out_w, out_h, out_ell, out_A, out_G, out_piv, out_eo = bufs
    exp.d = od
    exp.A_words = np.ascontiguousarray(out_A[:od])
    exp.Gamma = np.ascontiguousarray(out_G[: od * od].reshape(od, od))
    exp.pivots = np.ascontiguousarray(out_piv[:od])
    exp.ell_odd = np.ascontiguousarray(out_eo[:od])
    exp.weights = np.ascontiguousarray(out_w[:oc])
    exp.h_words = np.ascontiguousarray(out_h[:oc])
    exp.ell = np.ascontiguousarray(out_ell[: oc * od].reshape(oc, od))
    exp._refresh_a_union()
    exp._is_canonical_merged = True
    return exp


def _apply_pauli_component_inplace(
    comp: PhaseQFEComponent, a_w: NDArray[np.uint64], b_w: NDArray[np.uint64], kap: int
) -> complex:
    factor = complex(_PHASE4[(kap + (word_dot_parity(b_w, comp.h_words) << 1)) & 3])
    comp.h_words ^= a_w
    if comp.d > 0 and words_any_overlap(b_w, comp.a_union):
        comp.ell = np.ascontiguousarray((comp.ell + (word_dot_parity_batch(comp.A_words, b_w) << np.uint8(1))) & np.uint8(3))
    return factor


def apply_pauli(
    target: PhaseQFEComponent | ExpansionInput,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    in_place: bool = False,
    use_cpp_kernels: bool | None = None,
) -> Any:
    """`P1`: Apply `P = i**kappa X(a) Z(b)` (`O(chi * (n_words + d))`)."""
    if isinstance(target, PhaseQFEComponent):
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
        comp = target if in_place else target.copy()
        factor = _apply_pauli_component_inplace(comp, a_w, b_w, kap)
        return factor if in_place else (factor, comp)

    if isinstance(target, ShotExpansion):
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
        exp = target if in_place else target.copy()
        if exp.chi == 0:
            return exp
        if not np.all(np.isfinite(exp.weights)):
            raise FloatingPointError("Non-finite weights in apply_pauli")
        use_cpp = exp.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        lib = _get_cpp_lib_if_enabled(use_cpp)
        if lib is not None:
            lib.magic_apply_pauli_batch(
                exp.chi, exp.d, exp.n_words, _c_w_ptr(exp.weights),
                _c_ptr(exp.h_words, _c_u64_p), _c_ptr(exp.ell, _c_u8_p),
                _c_ptr(exp.A_words, _c_u64_p), _c_ptr(exp.a_union, _c_u64_p),
                _c_ptr(a_w, _c_u64_p), _c_ptr(b_w, _c_u64_p), ctypes.c_uint8(kap),
            )
            return exp
        bh = word_dot_parity_batch(exp.h_words, b_w)
        exp.weights *= _PHASE4[(np.uint8(kap) + (bh << np.uint8(1))) & np.uint8(3)]
        if words_any_nonzero(a_w):
            exp.h_words ^= a_w[None, :]
        if exp.d > 0 and words_any_overlap(b_w, exp.a_union):
            t = word_dot_parity_batch(exp.A_words, b_w)
            exp.ell = np.ascontiguousarray((exp.ell + (t[None, :] << np.uint8(1))) & np.uint8(3))
        return exp

    terms = list(target)
    if not terms:
        return []
    a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, _validate_component_terms(terms))
    out: list[ComponentTerm] = []
    for w, comp in terms:
        c_copy = comp.copy()
        out.append((complex(w) * _apply_pauli_component_inplace(c_copy, a_w, b_w, kap), c_copy))
    return out


def _quarter_turn_step_inplace(
    target: PhaseQFEComponent | ShotExpansion, a_w: NDArray[np.uint64], b_w: NDArray[np.uint64], kap: int
) -> complex | None:
    d0 = target.d
    t = word_dot_parity_batch(target.A_words, b_w) if d0 > 0 else np.zeros(0, dtype=np.uint8)
    s = get_bits_at_qubits(a_w[None, :], target.pivots)[0] if d0 > 0 else np.zeros(0, dtype=np.uint8)
    new_G = np.zeros((d0 + 1, d0 + 1), dtype=np.uint8)
    if d0 > 0:
        new_G[:d0, :d0] = target.Gamma
        new_G[:d0, d0] = t
        new_G[d0, :d0] = t
    target.A_words = np.ascontiguousarray(np.vstack([target.A_words, a_w[None, :]]))
    target.Gamma = new_G
    target.pivots = np.ascontiguousarray(np.append(target.pivots, np.int32(-1)))
    if isinstance(target, PhaseQFEComponent):
        c_val = np.uint8((kap + 1 + (word_dot_parity(b_w, target.h_words) << 1)) & 3)
        target.ell = np.ascontiguousarray(np.append(target.ell, c_val))
        target.d = d0 + 1
    else:
        assert target.ell_odd is not None
        c_col = (np.uint8((kap + 1) & 3) + (word_dot_parity_batch(target.h_words, b_w) << np.uint8(1))) & np.uint8(3)
        target.ell = np.ascontiguousarray(np.concatenate([target.ell, c_col[:, None]], axis=1))
        target.ell_odd = np.ascontiguousarray(np.append(target.ell_odd, np.uint8((kap + 1) & 1)))
        target.d = d0 + 1
        target._refresh_a_union()
    for i in np.flatnonzero(s):
        add_column(target, int(i), d0)
    p_new = first_set_bit_in_words(target.A_words[d0])
    if p_new >= 0:
        make_principal(target, d0, p_new)
        return (1.0 + 0.0j) if isinstance(target, PhaseQFEComponent) else None
    return eliminate_zero_amplitude(target, d0)


def apply_quarter_turn(
    target: PhaseQFEComponent | ExpansionInput,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    sign: int = 1,
    in_place: bool = False,
    check_unitary: bool = True,
    atol: float | None = None,
    use_cpp_kernels: bool | None = None,
) -> Any:
    """`P2`: Apply `R(sign * P) = (I + i * sign * P) / sqrt(2)` without rank growth (`chi' <= chi`)."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if atol is not None:
        validate_atol(atol)

    if isinstance(target, PhaseQFEComponent):
        tol_c = validate_atol(DEFAULT_ATOL if atol is None else atol)
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
        kap = (validate_hermitian_pauli(a_w, b_w, kap) + (2 if sign == -1 else 0)) & 3
        comp = target if in_place else target.copy()
        mult = _quarter_turn_step_inplace(comp, a_w, b_w, kap)
        assert isinstance(mult, complex)
        if check_unitary:
            _check_unitary_norm(1.0, abs(mult) ** 2, tol_c, "apply_quarter_turn")
        return mult if in_place else (mult, comp)

    if isinstance(target, ShotExpansion):
        tol_e = validate_atol(target.atol if atol is None else atol)
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
        kap = (validate_hermitian_pauli(a_w, b_w, kap) + (2 if sign == -1 else 0)) & 3
        exp = target if in_place else target.copy()
        pre_norm = _checked_shot_pre_norm(exp, check_unitary, "apply_quarter_turn")
        if exp.chi == 0:
            exp._is_canonical_merged = True
            return exp

        use_cpp = exp.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        lib = _get_cpp_lib_if_enabled(use_cpp)
        if lib is not None:
            if not exp._is_canonical_merged:
                merged_in = merge_frames(exp, use_cpp_kernels=use_cpp)
                assert isinstance(merged_in, ShotExpansion)
                exp = _copy_shot_inplace(exp, merged_in) if in_place else merged_in
                if exp.chi == 0:
                    if check_unitary:
                        _check_unitary_norm(pre_norm, 0.0, tol_e, "apply_quarter_turn")
                    return exp
            out_chi, out_d, bufs, out_ptrs = _alloc_cpp_geom_bufs(exp.chi, exp.n_words, exp.d + 1)
            rc = lib.magic_apply_quarter_turn(
                exp.chi, exp.d, exp.n_qubits, exp.n_words, *_cpp_shot_in_ptrs(exp),
                _c_ptr(a_w, _c_u64_p), _c_ptr(b_w, _c_u64_p), ctypes.c_uint8(kap), *out_ptrs,
            )
            if rc != 0:
                raise FloatingPointError(f"magic_apply_quarter_turn returned error code {rc}")
            _assign_cpp_geom_shot(exp, int(out_chi.value), int(out_d.value), bufs)
            if check_unitary:
                _check_unitary_norm(pre_norm, exp.norm_sq(), tol_e, "apply_quarter_turn")
            return exp

        _quarter_turn_step_inplace(exp, a_w, b_w, kap)
        merged = merge_frames(exp, use_cpp_kernels=use_cpp)
        assert isinstance(merged, ShotExpansion)
        res_out = _copy_shot_inplace(exp, merged) if in_place else merged
        if check_unitary:
            _check_unitary_norm(pre_norm, res_out.norm_sq(), tol_e, "apply_quarter_turn")
        return res_out

    tol_t = validate_atol(DEFAULT_ATOL if atol is None else atol)
    terms = list(target)
    if not terms:
        return []
    a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, _validate_component_terms(terms))
    kap = (validate_hermitian_pauli(a_w, b_w, kap) + (2 if sign == -1 else 0)) & 3
    pre_norm = expansion_norm_sq(terms, use_cpp_kernels=bool(use_cpp_kernels)) if check_unitary else 0.0
    out: list[ComponentTerm] = []
    for w, comp in terms:
        c_copy = comp.copy()
        mult = _quarter_turn_step_inplace(c_copy, a_w, b_w, kap)
        assert isinstance(mult, complex)
        if check_unitary:
            _check_unitary_norm(1.0, abs(mult) ** 2, tol_t, "apply_quarter_turn")
        w_new = complex(w) * mult
        if w_new != 0.0j:
            out.append((w_new, c_copy))
    merged_out = merge_frames(out, use_cpp_kernels=use_cpp_kernels)
    assert isinstance(merged_out, list)
    if check_unitary:
        _check_unitary_norm(pre_norm, expansion_norm_sq(merged_out, use_cpp_kernels=bool(use_cpp_kernels)), tol_t, "apply_quarter_turn")
    return merged_out


# ---------------------------------------------------------------------------
# Coherent Merging, Canonicalization & Norm (`K1`)
# ---------------------------------------------------------------------------


def canonicalize(
    target: PhaseQFEComponent | ShotExpansion, *, in_place: bool = False
) -> tuple[complex, PhaseQFEComponent] | ShotExpansion:
    """Apply `canonical_offset` (clearing `h[pivots]`) without deduplicating rows."""
    if isinstance(target, PhaseQFEComponent):
        comp = target if in_place else target.copy()
        return comp.canonical_offset(), comp
    if isinstance(target, ShotExpansion):
        exp = target if in_place else target.copy()
        if exp.chi == 0 or exp.d == 0:
            return exp
        assert exp.ell_odd is not None
        idx, phases = _canonicalize_rows_inplace(
            exp.h_words, exp.ell, exp.A_words, exp.Gamma, exp.pivots, exp.ell_odd
        )
        if len(idx) > 0:
            exp.weights[idx] *= phases
        return exp
    raise TypeError(f"Unsupported target type: {type(target)!r}")


def merge_frames(
    expansion: ExpansionInput,
    *,
    cancel_tol: float = K1_CANCEL_REL_TOL,
    use_cpp_kernels: bool | None = None,
) -> ShotExpansion | list[ComponentTerm]:
    """`K1`: Canonicalize coset offsets, coherently sum identical keys, and cancel roundoff zeros."""
    if isinstance(expansion, ShotExpansion):
        exp = expansion.copy()
        if exp.chi == 0:
            exp._is_canonical_merged = True
            return exp
        if not np.all(np.isfinite(exp.weights)):
            raise FloatingPointError("Non-finite weights encountered in merge_frames")
        use_cpp = exp.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        lib = _get_cpp_lib_if_enabled(use_cpp)
        if lib is not None:
            out_chi, out_w, out_h, out_ell, out_ptrs = _alloc_cpp_row_bufs(exp.chi, exp.n_words, exp.d)
            lib.magic_canonicalize_and_merge(
                exp.chi, exp.d, exp.n_words, *_cpp_shot_in_ptrs(exp),
                ctypes.c_double(cancel_tol), *out_ptrs,
            )
            oc = int(out_chi.value)
            exp.weights = np.ascontiguousarray(out_w[:oc])
            exp.h_words = np.ascontiguousarray(out_h[:oc])
            exp.ell = np.ascontiguousarray(out_ell[:oc])
            exp._is_canonical_merged = True
            return exp

        in_w_max = float(np.max(np.abs(exp.weights)))
        exp = canonicalize(exp, in_place=True)  # type: ignore[assignment]
        assert isinstance(exp, ShotExpansion)

        nw = exp.n_words
        row_bytes = 8 * nw + exp.d
        key_buf = np.empty((exp.chi, row_bytes), dtype=np.uint8)
        key_buf[:, : 8 * nw] = exp.h_words.view(np.uint8).reshape(exp.chi, 8 * nw)
        if exp.d > 0:
            key_buf[:, 8 * nw :] = exp.ell
        key_void = np.ascontiguousarray(key_buf).view(np.dtype((np.void, row_bytes))).ravel()
        _, first_idx, inv = np.unique(key_void, return_index=True, return_inverse=True)
        order = np.argsort(first_idx, kind="mergesort")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        group_id = rank[inv]
        first_rows = first_idx[order]

        n_groups = len(first_rows)
        merged_w = np.bincount(group_id, weights=exp.weights.real, minlength=n_groups) + (
            1j * np.bincount(group_id, weights=exp.weights.imag, minlength=n_groups)
        )
        keep = np.abs(merged_w) > (cancel_tol * min(1.0, in_w_max))
        sel = first_rows[keep]
        exp.weights = np.ascontiguousarray(merged_w[keep], dtype=np.complex128)
        exp.h_words = np.ascontiguousarray(exp.h_words[sel])
        exp.ell = np.ascontiguousarray(exp.ell[sel])
        exp._is_canonical_merged = True
        return exp

    terms = list(expansion)
    if not terms:
        return []
    _validate_component_terms(terms)
    raw_ws = np.array([complex(w) for w, _ in terms], dtype=np.complex128)
    if not np.all(np.isfinite(raw_ws)):
        raise FloatingPointError("Non-finite weights encountered in merge_frames")
    thresh = cancel_tol * min(1.0, float(np.max(np.abs(raw_ws))))

    groups: dict[bytes, list[ComponentTerm]] = {}
    for w, comp in terms:
        groups.setdefault(comp.geometry_key(), []).append((complex(w), comp))

    out_terms: list[ComponentTerm] = []
    use_cpp = True if use_cpp_kernels is None else use_cpp_kernels
    for g_terms in groups.values():
        g_exp = ShotExpansion.from_components(g_terms, merge=False, use_cpp_kernels=use_cpp)
        g_exp = merge_frames(g_exp, cancel_tol=0.0, use_cpp_kernels=use_cpp)  # type: ignore[assignment]
        assert isinstance(g_exp, ShotExpansion)
        for w_m, c_m in g_exp.to_components():
            if abs(w_m) > thresh:
                out_terms.append((w_m, c_m))
    return out_terms


def expansion_norm_sq(
    expansion: ExpansionInput, *, canonicalize_first: bool = True, use_cpp_kernels: bool = True
) -> float:
    """`K1.3`: Compute squared norm `||Psi||^2` via `O(chi)` shared-geometry fast path or `O1` Gram fallback."""
    if isinstance(expansion, ShotExpansion):
        return expansion.norm_sq(canonicalize_first=True if canonicalize_first else None)

    terms_in = list(expansion)
    if not terms_in:
        return 0.0
    _validate_component_terms(terms_in)
    terms = merge_frames(terms_in, use_cpp_kernels=use_cpp_kernels) if canonicalize_first else terms_in
    assert isinstance(terms, list)
    if not terms:
        return 0.0
    if canonicalize_first and len({comp.geometry_key() for _, comp in terms}) == 1:
        return float(sum(abs(w) ** 2 for w, _ in terms))

    acc = 0.0
    for i, (wi, ci) in enumerate(terms):
        acc += (wi.real * wi.real) + (wi.imag * wi.imag)
        for j in range(i + 1, len(terms)):
            wj, cj = terms[j]
            acc += 2.0 * ((wi.conjugate() * wj) * overlap(ci, cj, use_cpp_kernels=use_cpp_kernels)).real
    return 0.0 if (-1e-12 < acc < 0.0) else float(acc)


# ---------------------------------------------------------------------------
# Non-Clifford Branching Primitives: N1, N2, N3
# ---------------------------------------------------------------------------


def _branch_pauli_sum_shot(
    expansion: ShotExpansion,
    coeffs: NDArray[np.complex128],
    a_arr: NDArray[np.uint64],
    b_arr: NDArray[np.uint64],
    k_arr: NDArray[np.uint8],
    *,
    merge: bool,
    use_cpp: bool,
    check_unitary: bool,
    pre_norm: float,
    tol: float,
    op_name: str,
) -> ShotExpansion:
    r = int(coeffs.shape[0])
    if r == 0:
        empty = _empty_shot_expansion(
            expansion.n_qubits, log_mass=expansion.log_mass,
            use_cpp_kernels=expansion.use_cpp_kernels, atol=expansion.atol,
        )
        if check_unitary:
            _check_unitary_norm(pre_norm, 0.0, tol, op_name)
        return empty
    chi, d, nw = expansion.chi, expansion.d, expansion.n_words
    out_w = np.empty(r * chi, dtype=np.complex128)
    out_h = np.empty((r * chi, nw), dtype=np.uint64)
    out_ell = np.empty((r * chi, d), dtype=np.uint8)
    for m in range(r):
        sl = slice(m * chi, (m + 1) * chi)
        bh = word_dot_parity_batch(expansion.h_words, b_arr[m])
        out_w[sl] = (coeffs[m] * expansion.weights) * _PHASE4[(np.uint8(int(k_arr[m]) & 3) + (bh << np.uint8(1))) & np.uint8(3)]
        out_h[sl] = expansion.h_words ^ a_arr[m][None, :]
        if d > 0:
            if words_any_overlap(b_arr[m], expansion.a_union):
                t = word_dot_parity_batch(expansion.A_words, b_arr[m])
                out_ell[sl] = (expansion.ell + (t[None, :] << np.uint8(1))) & np.uint8(3)
            else:
                out_ell[sl] = expansion.ell
    raw = _clone_shot_with_rows(expansion, out_w, out_h, out_ell, is_canonical_merged=False)
    if not merge:
        return raw
    out = merge_frames(raw, use_cpp_kernels=use_cpp)
    assert isinstance(out, ShotExpansion)
    if check_unitary:
        _check_unitary_norm(pre_norm, out.norm_sq(), tol, op_name)
    return out


def _branch_pauli_sum_terms(
    terms: Sequence[ComponentTerm],
    coeffs: NDArray[np.complex128],
    a_arr: NDArray[np.uint64],
    b_arr: NDArray[np.uint64],
    k_arr: NDArray[np.uint8],
    *,
    merge: bool,
    use_cpp_kernels: bool | None,
    check_unitary: bool,
    pre_norm: float,
    tol: float,
    op_name: str,
) -> list[ComponentTerm]:
    out_terms: list[ComponentTerm] = []
    for m in range(int(coeffs.shape[0])):
        cm = complex(coeffs[m])
        am, bm, km = a_arr[m], b_arr[m], int(k_arr[m]) & 3
        is_id = not (words_any_nonzero(am) or words_any_nonzero(bm) or km != 0)
        for w, comp in terms:
            c_copy = comp.copy()
            f = 1.0 + 0.0j if is_id else _apply_pauli_component_inplace(c_copy, am, bm, km)
            out_terms.append((complex(w) * cm * f, c_copy))
    if not merge:
        return out_terms
    merged = merge_frames(out_terms, use_cpp_kernels=use_cpp_kernels)
    assert isinstance(merged, list)
    if check_unitary:
        _check_unitary_norm(pre_norm, expansion_norm_sq(merged, use_cpp_kernels=bool(use_cpp_kernels)), tol, op_name)
    return merged


def _apply_pauli_sum_op(
    expansion: ExpansionInput,
    prep_branches: Any,
    cpp_runner: Any,
    *,
    merge: bool,
    check_unitary: bool,
    atol: float | None,
    use_cpp_kernels: bool | None,
    op_name: str,
) -> ShotExpansion | list[ComponentTerm]:
    if atol is not None:
        validate_atol(atol)
    if isinstance(expansion, ShotExpansion):
        tol = validate_atol(expansion.atol if atol is None else atol)
        bw, ba, bb, bk = prep_branches(expansion.n_qubits)
        pre_norm = _checked_shot_pre_norm(expansion, check_unitary, op_name)
        if expansion.chi == 0:
            empty_out = expansion.copy()
            empty_out._is_canonical_merged = True
            return empty_out
        use_cpp = expansion.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        lib = _get_cpp_lib_if_enabled(use_cpp) if merge else None
        if lib is not None:
            r = int(bw.shape[0])
            out_chi, out_w, out_h, out_ell, out_ptrs = _alloc_cpp_row_bufs(
                r * expansion.chi, expansion.n_words, expansion.d
            )
            head = (expansion.chi, expansion.d, expansion.n_words, *_cpp_shot_in_ptrs(expansion, include_a_union=True))
            cpp_runner(lib, head, out_ptrs, bw, ba, bb, bk)
            oc = int(out_chi.value)
            res = _clone_shot_with_rows(
                expansion, out_w[:oc].copy(), out_h[:oc].copy(), out_ell[:oc].copy(), is_canonical_merged=True
            )
            if check_unitary:
                _check_unitary_norm(pre_norm, res.norm_sq(), tol, op_name)
            return res
        return _branch_pauli_sum_shot(
            expansion, bw, ba, bb, bk, merge=merge, use_cpp=use_cpp,
            check_unitary=check_unitary, pre_norm=pre_norm, tol=tol, op_name=op_name,
        )

    tol = validate_atol(DEFAULT_ATOL if atol is None else atol)
    terms = list(expansion)
    if not terms:
        return []
    bw, ba, bb, bk = prep_branches(_validate_component_terms(terms))
    pre_norm = (
        expansion_norm_sq(terms, canonicalize_first=False, use_cpp_kernels=bool(use_cpp_kernels))
        if check_unitary
        else 0.0
    )
    return _branch_pauli_sum_terms(
        terms, bw, ba, bb, bk, merge=merge, use_cpp_kernels=use_cpp_kernels,
        check_unitary=check_unitary, pre_norm=pre_norm, tol=tol, op_name=op_name,
    )


def apply_pauli_t(
    expansion: ExpansionInput,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    dag: bool = False,
    sign: int | None = None,
    merge: bool = True,
    check_unitary: bool = True,
    atol: float | None = None,
    use_cpp_kernels: bool | None = None,
) -> ShotExpansion | list[ComponentTerm]:
    """`N1`: Apply Pauli-axis T gate `((1 + omega^s)/2) I + ((1 - omega^s)/2) P`."""
    if atol is not None:
        validate_atol(atol)
    if sign is not None and sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    is_dag = (sign == -1) if sign is not None else bool(dag)
    coeffs = np.array([_TDAG_C0, _TDAG_C1] if is_dag else [_T_C0, _T_C1], dtype=np.complex128)

    def _prep(n_q: int) -> tuple[NDArray[np.complex128], NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]]:
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, n_q)
        kap = validate_hermitian_pauli(a_w, b_w, kap)
        z_w = np.zeros(n_words_for_qubits(n_q), dtype=np.uint64)
        return coeffs, np.vstack([z_w, a_w]), np.vstack([z_w, b_w]), np.array([0, kap], dtype=np.uint8)

    def _cpp(lib: Any, head: tuple[Any, ...], out_ptrs: tuple[Any, ...], _bw: Any, ba: Any, bb: Any, bk: Any) -> None:
        lib.magic_branch_pauli_t(
            *head, _c_ptr(ba[1], _c_u64_p), _c_ptr(bb[1], _c_u64_p),
            ctypes.c_uint8(int(bk[1])), ctypes.c_int(1 if is_dag else 0),
            ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs,
        )

    return _apply_pauli_sum_op(
        expansion, _prep, _cpp, merge=merge, check_unitary=check_unitary,
        atol=atol, use_cpp_kernels=use_cpp_kernels, op_name="apply_pauli_t",
    )


def _normalize_ccz_branches(
    paulis: Sequence[Any], n_qubits: int
) -> tuple[NDArray[np.complex128], NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint8]]:
    if not isinstance(paulis, Sequence) or isinstance(paulis, (str, bytes)):
        raise ValueError(f"apply_ccz expects a sequence of 3 or 8 Paulis, got {paulis!r}")
    nw = n_words_for_qubits(n_qubits)
    if len(paulis) == 3:
        axes: list[tuple[NDArray[np.uint64], NDArray[np.uint64], int]] = []
        for ax in paulis:
            a_w, b_w, kap = _unpack_pauli_args(ax, None, 0, n_qubits)
            axes.append((a_w, b_w, validate_hermitian_pauli(a_w, b_w, kap)))
        for i in range(3):
            for j in range(i + 1, 3):
                if (word_dot_parity(axes[i][0], axes[j][1]) ^ word_dot_parity(axes[i][1], axes[j][0])) != 0:
                    raise ValueError(f"apply_ccz requires pairwise commuting axes; axes {i} and {j} anticommute")
        a_arr = np.zeros((8, nw), dtype=np.uint64)
        b_arr = np.zeros((8, nw), dtype=np.uint64)
        k_arr = np.zeros(8, dtype=np.uint8)
        for mask in range(1, 8):
            ca = np.zeros(nw, dtype=np.uint64)
            cb = np.zeros(nw, dtype=np.uint64)
            ck = 0
            for bit in range(3):
                if (mask >> bit) & 1:
                    ca, cb, ck = pauli_multiply(ca, cb, ck, axes[bit][0], axes[bit][1], axes[bit][2])
            a_arr[mask], b_arr[mask], k_arr[mask] = ca, cb, np.uint8(ck)
        return _CCZ_WEIGHTS.copy(), a_arr, b_arr, k_arr

    if len(paulis) == 8:
        w_arr = np.empty(8, dtype=np.complex128)
        a_arr = np.empty((8, nw), dtype=np.uint64)
        b_arr = np.empty((8, nw), dtype=np.uint64)
        k_arr = np.empty(8, dtype=np.uint8)
        for m, item in enumerate(paulis):
            if isinstance(item, tuple) and len(item) == 4:
                w_arr[m] = _coerce_finite_complex(item[0], f"ccz branch[{m}] weight")
                a_arr[m] = coerce_pauli_words(item[1], n_qubits)
                b_arr[m] = coerce_pauli_words(item[2], n_qubits)
                k_arr[m] = np.uint8(int(item[3]) & 3)
            else:
                a_w, b_w, kap = _unpack_pauli_args(item, None, 0, n_qubits)
                w_arr[m], a_arr[m], b_arr[m], k_arr[m] = _CCZ_WEIGHTS[m], a_w, b_w, np.uint8(kap)
        return w_arr, a_arr, b_arr, k_arr

    raise ValueError(f"apply_ccz expects 3 commuting axes or 8 pre-expanded branches, got {len(paulis)}")


def apply_ccz(
    expansion: ExpansionInput,
    paulis: Sequence[Any],
    *,
    merge: bool = True,
    check_unitary: bool = True,
    atol: float | None = None,
    use_cpp_kernels: bool | None = None,
) -> ShotExpansion | list[ComponentTerm]:
    """`N2`: Apply generalized 3-axis `CCZ(Q1, Q2, Q3)` via 8-branch Pauli expansion."""

    def _cpp(lib: Any, head: tuple[Any, ...], out_ptrs: tuple[Any, ...], bw: Any, ba: Any, bb: Any, bk: Any) -> None:
        lib.magic_branch_ccz(
            *head, _c_w_ptr(bw), _c_ptr(ba, _c_u64_p), _c_ptr(bb, _c_u64_p), _c_ptr(bk, _c_u8_p),
            ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs,
        )

    return _apply_pauli_sum_op(
        expansion, lambda n_q: _normalize_ccz_branches(paulis, n_q), _cpp,
        merge=merge, check_unitary=check_unitary, atol=atol,
        use_cpp_kernels=use_cpp_kernels, op_name="apply_ccz",
    )


def _validate_compiled_block(block: CompiledPauliBlock, n_qubits: int | None) -> None:
    if not isinstance(block, CompiledPauliBlock):
        raise ValueError(f"Expected CompiledPauliBlock, got {type(block)!r}")
    if not _is_valid_int(block.n_qubits) or block.n_qubits < 1:
        raise ValueError(f"block.n_qubits must be >= 1, got {block.n_qubits!r}")
    eff_nq = int(block.n_qubits) if n_qubits is None else n_qubits
    nw = n_words_for_qubits(eff_nq)
    if n_qubits is not None and block.n_qubits != n_qubits:
        if block.n_qubits_exact or n_words_for_qubits(block.n_qubits) != nw:
            raise ValueError(f"Block n_qubits={block.n_qubits} != expansion n_qubits={n_qubits}")
    if block.mode not in ("pauli_sum", "sequential"):
        raise ValueError(f"Unknown CompiledPauliBlock mode: {block.mode!r}")
    _coerce_finite_complex(block.global_scalar, "block.global_scalar")
    if not isinstance(block.steps, Sequence) or isinstance(block.steps, (str, bytes)):
        raise ValueError(f"block.steps must be a sequence, got {block.steps!r}")

    tm_inv = ~tail_mask(eff_nq)
    has_pauli_arrays = any(arr is not None for arr in (block.coeffs, block.a_words, block.b_words, block.kappas))
    if (block.mode == "pauli_sum" and not block.is_identity) or has_pauli_arrays:
        if not all(isinstance(a, np.ndarray) for a in (block.coeffs, block.a_words, block.b_words, block.kappas)):
            raise ValueError("pauli_sum CompiledPauliBlock requires ndarray coeffs, a_words, b_words, kappas")
        assert block.coeffs is not None and block.a_words is not None and block.b_words is not None and block.kappas is not None
        if block.coeffs.ndim != 1 or np.issubdtype(block.coeffs.dtype, np.bool_) or not np.issubdtype(block.coeffs.dtype, np.number):
            raise ValueError(f"Invalid block.coeffs dtype/ndim: {block.coeffs.dtype}, ndim={block.coeffs.ndim}")
        if not np.all(np.isfinite(block.coeffs)):
            raise FloatingPointError("Non-finite entries in CompiledPauliBlock.coeffs")
        r = int(block.coeffs.shape[0])
        if (
            block.a_words.dtype != np.uint64 or block.a_words.shape != (r, nw)
            or block.b_words.dtype != np.uint64 or block.b_words.shape != (r, nw)
            or np.issubdtype(block.kappas.dtype, np.bool_) or not np.issubdtype(block.kappas.dtype, np.integer)
            or block.kappas.shape != (r,) or (r > 0 and bool(np.any((block.kappas < 0) | (block.kappas >= 4))))
        ):
            raise ValueError(f"Invalid pauli_sum array shapes/dtypes for r={r}, n_words={nw}")
        if r > 0 and (np.any((block.a_words[:, -1] & tm_inv) != 0) or np.any((block.b_words[:, -1] & tm_inv) != 0)):
            raise ValueError("Block Pauli support exceeds expansion n_qubits")

    if block.mode == "pauli_sum" and len(block.steps) > 0:
        raise ValueError("pauli_sum CompiledPauliBlock must not define sequential steps")

    for step in block.steps:
        if isinstance(step, CompiledPauliBlock):
            _validate_compiled_block(step, eff_nq)
        elif isinstance(step, CompiledPauliBlockStep):
            if step.weights is not None:
                if (
                    not isinstance(step.weights, np.ndarray)
                    or np.issubdtype(step.weights.dtype, np.bool_)
                    or not np.issubdtype(step.weights.dtype, np.number)
                ):
                    raise ValueError(f"Invalid step.weights: {step.weights!r}")
                if not np.all(np.isfinite(step.weights)):
                    raise FloatingPointError("Non-finite entries in CompiledPauliBlockStep.weights")
            if not isinstance(step.paulis, Sequence) or isinstance(step.paulis, (str, bytes)):
                raise ValueError(f"step.paulis must be a sequence, got {step.paulis!r}")
            if step.kind in ("P1", "P2", "N1"):
                if step.kind != "P1" and step.sign not in (1, -1):
                    raise ValueError(f"{step.kind} step sign must be +1 or -1, got {step.sign}")
                if len(step.paulis) != 1 or not isinstance(step.paulis[0], tuple) or len(step.paulis[0]) != 3:
                    raise ValueError(f"{step.kind} step expects 1 Pauli 3-tuple, got {step.paulis!r}")
                a_w, b_w, kap = _unpack_pauli_args(*step.paulis[0], eff_nq)
                if step.kind != "P1":
                    validate_hermitian_pauli(a_w, b_w, kap)
            elif step.kind == "N2":
                _normalize_ccz_branches(step.paulis, eff_nq)
            else:
                raise ValueError(f"Unknown step kind: {step.kind!r}")
        else:
            raise ValueError(f"Unsupported step in CompiledPauliBlock: {step!r}")


def _execute_sequential_steps(
    cur: ShotExpansion | list[ComponentTerm],
    steps: Sequence[CompiledPauliBlockStep | CompiledPauliBlock],
    *,
    tol: float,
    use_cpp_kernels: bool | None,
) -> ShotExpansion | list[ComponentTerm]:
    for step in steps:
        if isinstance(step, CompiledPauliBlock):
            cur = apply_compiled_block(cur, step, check_unitary=False, atol=tol, use_cpp_kernels=use_cpp_kernels)
        elif step.kind == "P1":
            cur = apply_pauli(cur, *step.paulis[0], use_cpp_kernels=use_cpp_kernels)
        elif step.kind == "P2":
            cur = apply_quarter_turn(
                cur, *step.paulis[0], sign=step.sign, check_unitary=False, atol=tol, use_cpp_kernels=use_cpp_kernels
            )
        elif step.kind == "N1":
            cur = apply_pauli_t(
                cur, *step.paulis[0], sign=step.sign, merge=True,
                check_unitary=False, atol=tol, use_cpp_kernels=use_cpp_kernels,
            )
        elif step.kind == "N2":
            cur = apply_ccz(cur, step.paulis, merge=True, check_unitary=False, atol=tol, use_cpp_kernels=use_cpp_kernels)
        else:
            raise ValueError(f"Unknown step kind: {step.kind!r}")
    return merge_frames(cur, use_cpp_kernels=use_cpp_kernels)  # type: ignore[return-value]


def apply_compiled_block(
    expansion: ExpansionInput,
    block: CompiledPauliBlock,
    *,
    check_unitary: bool | None = None,
    atol: float | None = None,
    use_cpp_kernels: bool | None = None,
) -> ShotExpansion | list[ComponentTerm]:
    """`N3`: Execute a `CompiledPauliBlock` (`pauli_sum` or `sequential` mode)."""
    if atol is not None:
        validate_atol(atol)
    if not isinstance(block, CompiledPauliBlock):
        raise ValueError(f"Expected CompiledPauliBlock, got {type(block)!r}")
    do_check = block.is_unitary if check_unitary is None else bool(check_unitary)

    if isinstance(expansion, ShotExpansion):
        tol = validate_atol(expansion.atol if atol is None else atol)
        _validate_compiled_block(block, expansion.n_qubits)
        pre_norm = _checked_shot_pre_norm(expansion, do_check, "apply_compiled_block")
        use_cpp = expansion.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        if expansion.chi == 0:
            empty_out = expansion.copy()
            empty_out._is_canonical_merged = True
            return empty_out

        g_sc = _coerce_finite_complex(block.global_scalar, "block.global_scalar")
        if block.mode == "pauli_sum" and not block.is_identity:
            assert block.coeffs is not None and block.a_words is not None and block.b_words is not None and block.kappas is not None
            nz = np.flatnonzero(np.abs(block.coeffs) > 0.0)
            return _branch_pauli_sum_shot(
                expansion, np.asarray(g_sc * block.coeffs[nz], dtype=np.complex128),
                block.a_words[nz], block.b_words[nz], block.kappas[nz],
                merge=True, use_cpp=use_cpp, check_unitary=do_check,
                pre_norm=pre_norm, tol=tol, op_name="apply_compiled_block",
            )

        out = expansion.copy()
        out.weights *= g_sc
        if block.is_identity:
            out = merge_frames(out, use_cpp_kernels=use_cpp)  # type: ignore[assignment]
        else:
            out = _execute_sequential_steps(out, block.steps, tol=tol, use_cpp_kernels=use_cpp)  # type: ignore[assignment]
        assert isinstance(out, ShotExpansion)
        if do_check:
            _check_unitary_norm(pre_norm, out.norm_sq(), tol, "apply_compiled_block")
        return out

    tol = validate_atol(DEFAULT_ATOL if atol is None else atol)
    terms = list(expansion)
    if not terms:
        _validate_compiled_block(block, None)
        return []
    _validate_compiled_block(block, _validate_component_terms(terms))
    g_sc = _coerce_finite_complex(block.global_scalar, "block.global_scalar")
    pre_norm = expansion_norm_sq(terms, canonicalize_first=False, use_cpp_kernels=bool(use_cpp_kernels)) if do_check else 0.0

    if block.mode == "pauli_sum" and not block.is_identity:
        assert block.coeffs is not None and block.a_words is not None and block.b_words is not None and block.kappas is not None
        nz = np.flatnonzero(np.abs(block.coeffs) > 0.0)
        return _branch_pauli_sum_terms(
            terms, np.asarray(g_sc * block.coeffs[nz], dtype=np.complex128),
            block.a_words[nz], block.b_words[nz], block.kappas[nz],
            merge=True, use_cpp_kernels=use_cpp_kernels, check_unitary=do_check,
            pre_norm=pre_norm, tol=tol, op_name="apply_compiled_block",
        )

    scaled: list[ComponentTerm] = [(g_sc * complex(w), c.copy()) for w, c in terms]
    out_terms = (
        merge_frames(scaled, use_cpp_kernels=use_cpp_kernels)
        if block.is_identity
        else _execute_sequential_steps(scaled, block.steps, tol=tol, use_cpp_kernels=use_cpp_kernels)
    )
    assert isinstance(out_terms, list)
    if do_check:
        _check_unitary_norm(pre_norm, expansion_norm_sq(out_terms, use_cpp_kernels=bool(use_cpp_kernels)), tol, "apply_compiled_block")
    return out_terms


# ---------------------------------------------------------------------------
# Projective Measurements & Born Sampling: M1 (`project_diagonal`) and M2 (`measure_expansion`)
# ---------------------------------------------------------------------------


def project_diagonal(
    expansion: ExpansionInput,
    b: Any,
    sigma: int = 0,
    outcome: int = 0,
    *,
    kappa: int | None = None,
    use_cpp_kernels: bool | None = None,
) -> tuple[ShotExpansion | list[ComponentTerm], float]:
    """`M1`: Project onto `Pi_m = (I + (-1)^m (-1)^sigma Z(b)) / 2` without renormalizing."""
    if kappa is not None:
        kap = int(kappa) & 3
        if kap not in (0, 2):
            raise ValueError(f"Diagonal Hermitian Pauli requires kappa in {{0, 2}}, got {kap}")
        sig = (kap >> 1) & 1
    else:
        sig = int(sigma)
        if sig not in (0, 1):
            raise ValueError(f"sigma must be 0 or 1, got {sigma}")
    m = int(outcome)
    if m not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome}")

    if isinstance(expansion, ShotExpansion):
        b_w = coerce_pauli_words(b, expansion.n_qubits)
        exp = expansion.copy()
        if exp.chi == 0:
            exp._is_canonical_merged = True
            return exp, 0.0
        if not np.all(np.isfinite(exp.weights)):
            raise FloatingPointError("Non-finite weights in project_diagonal")

        use_cpp = expansion.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        lib = _get_cpp_lib_if_enabled(use_cpp)
        if lib is not None:
            out_chi, out_d, bufs, out_ptrs = _alloc_cpp_geom_bufs(exp.chi, exp.n_words, max(1, exp.d))
            out_z = ctypes.c_double(0.0)
            lib.magic_project_diagonal(
                exp.chi, exp.d, exp.n_words, *_cpp_shot_in_ptrs(exp),
                _c_ptr(b_w, _c_u64_p), ctypes.c_uint8(sig << 1), ctypes.c_uint8(m),
                ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs, ctypes.byref(out_z),
            )
            _assign_cpp_geom_shot(exp, int(out_chi.value), int(out_d.value), bufs)
            return exp, float(out_z.value)

        rhs = (np.uint8(m ^ sig) ^ word_dot_parity_batch(exp.h_words, b_w)) & np.uint8(1)
        col_par = word_dot_parity_batch(exp.A_words, b_w) if exp.d > 0 else np.zeros(0, dtype=np.uint8)
        restrict_amplitude(exp, col_par, rhs, scale=_INV_SQRT2)
        exp = merge_frames(exp, use_cpp_kernels=use_cpp)  # type: ignore[assignment]
        assert isinstance(exp, ShotExpansion)
        return exp, exp.norm_sq()

    terms = list(expansion)
    if not terms:
        return [], 0.0
    b_w = coerce_pauli_words(b, _validate_component_terms(terms))
    out_terms: list[ComponentTerm] = []
    for w, comp in terms:
        c_copy = comp.copy()
        lam = word_dot_parity_batch(c_copy.A_words, b_w) if c_copy.d > 0 else np.zeros(0, dtype=np.uint8)
        eps = (m ^ sig ^ word_dot_parity(b_w, c_copy.h_words)) & 1
        g = restrict_amplitude(c_copy, lam, eps, scale=_INV_SQRT2)
        assert isinstance(g, complex)
        if g != 0.0j:
            out_terms.append((complex(w) * g, c_copy))
    merged = merge_frames(out_terms, use_cpp_kernels=use_cpp_kernels)
    assert isinstance(merged, list)
    return merged, expansion_norm_sq(merged, use_cpp_kernels=bool(use_cpp_kernels))


def measure_expansion(
    expansion: ExpansionInput,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    sigma: int | None = None,
    rng: np.random.Generator | None = None,
    requested_bit: int | None = None,
    p_flip: float = 0.0,
    invert_result: bool = False,
    renormalize: bool = True,
    keep_rotated_frame: bool = False,
    atol: float | None = None,
    use_cpp_kernels: bool | None = None,
) -> MeasurementResult:
    """`M2`: Measure Hermitian Pauli `P = i**kappa X(a) Z(b)` (or diagonal `(-1)^sigma Z(a)`)."""
    if not (0.0 <= p_flip <= 1.0):
        raise ValueError(f"p_flip must be in [0, 1], got {p_flip}")
    if requested_bit is not None and requested_bit not in (0, 1):
        raise ValueError(f"requested_bit must be 0, 1, or None, got {requested_bit}")

    if isinstance(expansion, ShotExpansion):
        n_q = expansion.n_qubits
        tol = validate_atol(expansion.atol if atol is None else atol)
        use_cpp = expansion.use_cpp_kernels if use_cpp_kernels is None else use_cpp_kernels
        z = expansion.norm_sq()
        in_log_mass = expansion.log_mass
    else:
        tol = validate_atol(DEFAULT_ATOL if atol is None else atol)
        terms_in = list(expansion)
        if not terms_in:
            raise ImpossiblePostselectionError("Cannot measure an empty component expansion")
        n_q = _validate_component_terms(terms_in)
        use_cpp = True if use_cpp_kernels is None else use_cpp_kernels
        z = expansion_norm_sq(terms_in, canonicalize_first=False, use_cpp_kernels=use_cpp)
        in_log_mass = 0.0

    if not (math.isfinite(z) and z >= 0.0):
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={z!r}")

    if b is None and sigma is not None:
        if sigma not in (0, 1):
            raise ValueError(f"sigma must be 0 or 1, got {sigma}")
        a_w = np.zeros(n_words_for_qubits(n_q), dtype=np.uint64)
        b_w = coerce_pauli_words(a, n_q)
        kap = (int(sigma) & 1) << 1
    else:
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, n_q)
        kap = validate_hermitian_pauli(a_w, b_w, kap)

    rotated = words_any_nonzero(a_w)
    if rotated:
        e_p_w = unit_bit_words(first_set_bit_in_words(a_w), n_words_for_qubits(n_q))
        a_rot, b_rot, kap_rot = a_w, np.ascontiguousarray(b_w ^ e_p_w), (kap + 1) & 3
        work_exp = apply_quarter_turn(
            expansion, a_rot, b_rot, kap_rot, sign=1, check_unitary=True, atol=tol, use_cpp_kernels=use_cpp
        )
        b_diag, sig_diag = e_p_w, 0
    else:
        a_rot, b_rot, kap_rot = a_w, b_w, 0
        work_exp, b_diag, sig_diag = expansion, b_w, (kap >> 1) & 1

    exp0, z0_raw = project_diagonal(work_exp, b_diag, sig_diag, 0, use_cpp_kernels=use_cpp)
    exp1, z1_raw = project_diagonal(work_exp, b_diag, sig_diag, 1, use_cpp_kernels=use_cpp)

    scale_z = max(1.0, z)
    if not (math.isfinite(z0_raw) and math.isfinite(z1_raw)):
        raise FloatingPointError(f"Non-finite projected Born masses: z0={z0_raw!r}, z1={z1_raw!r}")
    if z0_raw < -1e-12 * scale_z or z1_raw < -1e-12 * scale_z:
        raise FloatingPointError(f"Significantly negative projected Born mass: z0={z0_raw!r}, z1={z1_raw!r}, z={z!r}")
    z0 = 0.0 if z0_raw < 0.0 else float(z0_raw)
    z1 = 0.0 if z1_raw < 0.0 else float(z1_raw)
    if not (abs((z0 + z1) - z) <= tol * scale_z):
        raise FloatingPointError(
            f"M2 Born mass conservation violated: z0={z0:.16e}, z1={z1:.16e}, "
            f"z0+z1={z0 + z1:.16e}, z={z:.16e}, tol={tol:.3e}"
        )

    zero_floor = K1_CANCEL_REL_TOL * z
    if requested_bit is not None:
        outcome = int(requested_bit)
        chosen_z, chosen_exp = (z0, exp0) if outcome == 0 else (z1, exp1)
        chosen_empty = (chosen_exp.chi == 0) if isinstance(chosen_exp, ShotExpansion) else (len(chosen_exp) == 0)
        if z <= 0.0 or chosen_z <= zero_floor or chosen_empty:
            raise ImpossiblePostselectionError(
                f"Requested postselection outcome {outcome} has zero Born probability "
                f"(z_{outcome}={chosen_z:.3e}, total z={z:.3e})"
            )
    else:
        if z <= 0.0 or (z0 + z1) <= 0.0:
            raise ImpossiblePostselectionError("Cannot sample measurement from a zero-norm expansion")
        rng_obj = np.random.default_rng() if rng is None else rng
        u_born = float(rng_obj.random())
        if z0 <= zero_floor:
            outcome = 1
        elif z1 <= zero_floor:
            outcome = 0
        else:
            outcome = 1 if u_born < min(1.0, max(0.0, z1 / (z0 + z1))) else 0
        chosen_z, chosen_exp = (z1, exp1) if outcome == 1 else (z0, exp0)

    if p_flip > 0.0:
        rng_obj = np.random.default_rng() if rng is None else rng
        readout_flipped = bool(float(rng_obj.random()) < p_flip)
    else:
        readout_flipped = False
    reported_outcome = outcome ^ (1 if invert_result else 0) ^ (1 if readout_flipped else 0)

    if rotated and not keep_rotated_frame:
        chosen_exp = apply_quarter_turn(
            chosen_exp, a_rot, b_rot, kap_rot, sign=-1, check_unitary=True, atol=tol, use_cpp_kernels=use_cpp
        )

    if isinstance(chosen_exp, ShotExpansion):
        if renormalize:
            chosen_exp.weights = np.ascontiguousarray(chosen_exp.weights * (1.0 / math.sqrt(chosen_z)))
            chosen_exp.log_mass = in_log_mass + math.log(chosen_z)
            out_log_mass = chosen_exp.log_mass
        else:
            chosen_exp.log_mass = in_log_mass
            out_log_mass = in_log_mass + math.log(chosen_z)
    else:
        if renormalize:
            inv_sqrt_zm = 1.0 / math.sqrt(chosen_z)
            chosen_exp = [(w * inv_sqrt_zm, c) for w, c in chosen_exp]
        out_log_mass = math.log(chosen_z)

    return MeasurementResult(
        outcome, reported_outcome, readout_flipped, z0, z1, z, chosen_z, out_log_mass, chosen_exp
    )


# ---------------------------------------------------------------------------
# Coherent Contractions: O2 (`quadratic_gauss_sum`) and O1 (`overlap`)
# ---------------------------------------------------------------------------


def _parse_gauss_sum_args(
    arg1: Any, arg2: Any, arg3: Any, c0: int
) -> tuple[int, NDArray[np.uint8], NDArray[np.uint8]]:
    if arg3 is not None:
        q0 = int(arg1) & 3
        l_arr = (np.asarray(arg2, dtype=np.int64) & 3).astype(np.uint8)
        B_arr = (np.asarray(arg3, dtype=np.int64) & 1).astype(np.uint8)
    else:
        q0 = int(c0) & 3
        l_arr = (np.asarray(arg1, dtype=np.int64) & 3).astype(np.uint8)
        B_arr = (np.asarray(arg2, dtype=np.int64) & 1).astype(np.uint8)
    if l_arr.ndim != 1:
        raise ValueError(f"lin/ell must be 1D, got shape {l_arr.shape}")
    g = int(l_arr.shape[0])
    if B_arr.shape != (g, g):
        raise ValueError(f"B/Gamma shape {B_arr.shape} does not match (g, g)=({g}, {g})")
    if g > 0 and (not np.array_equal(B_arr, B_arr.T) or np.any(np.diag(B_arr) != 0)):
        raise ValueError("B/Gamma must be symmetric with zero diagonal")
    return q0, np.ascontiguousarray(l_arr.copy()), np.ascontiguousarray(B_arr.copy())


def quadratic_gauss_sum_dyadic(
    arg1: Any, arg2: Any, arg3: Any = None, *, c0: int = 0
) -> tuple[bool, int, int]:
    """`O2` exact dyadic evaluator: returns `(nonzero, k2, v8)` for `2^(k2 / 2) * omega^v8`."""
    q0, l, M = _parse_gauss_sum_args(arg1, arg2, arg3, c0)
    k = int(l.shape[0])
    k2, v8 = 0, (2 * q0) & 7

    def _swap_var(i: int, j: int, cur_k: int) -> None:
        if i == j:
            return
        l[i], l[j] = l[j], l[i]
        row_i = M[i, :cur_k].copy()
        M[i, :cur_k] = M[j, :cur_k]
        M[j, :cur_k] = row_i
        col_i = M[:cur_k, i].copy()
        M[:cur_k, i] = M[:cur_k, j]
        M[:cur_k, j] = col_i

    while k > 0:
        odd_idx = np.flatnonzero(l[:k] & np.uint8(1))
        if len(odd_idx) > 0:
            _swap_var(int(odd_idx[0]), k - 1, k)
            c = int(l[k - 1])
            k2 += 1
            v8 = (v8 + (1 if c == 1 else 7)) & 7
            if k > 1:
                m = M[k - 1, : k - 1]
                l[: k - 1] = (l[: k - 1] + np.uint8(3 if c == 1 else 1) * m) & np.uint8(3)
                M[: k - 1, : k - 1] ^= m[:, None] & m[None, :]
                np.fill_diagonal(M[: k - 1, : k - 1], 0)
            k -= 1
            continue

        row_any = np.any(M[:k, :k] != 0, axis=1)
        if np.any(row_any):
            f = int(np.argmax(row_any))
            p = int(np.argmax(M[f, :k]))
            if f > p:
                f, p = p, f
            _swap_var(p, k - 1, k)
            _swap_var(f, k - 2, k)
            lf, lp = int(l[k - 2]) >> 1, int(l[k - 1]) >> 1
            k2 += 2
            v8 = (v8 + 4 * (lf & lp)) & 7
            if k > 2:
                mf, mp = M[k - 2, : k - 2], M[k - 1, : k - 2]
                lin_upd = ((np.uint8(lp) * mf) ^ (np.uint8(lf) * mp) ^ (mf & mp)) & np.uint8(1)
                l[: k - 2] = (l[: k - 2] + (lin_upd << np.uint8(1))) & np.uint8(3)
                M[: k - 2, : k - 2] ^= (mf[:, None] & mp[None, :]) ^ (mp[:, None] & mf[None, :])
                np.fill_diagonal(M[: k - 2, : k - 2], 0)
            k -= 2
            continue

        if np.any(l[:k] == np.uint8(2)):
            return False, 0, 0
        k2 += 2 * k
        k = 0

    return True, k2, v8


def quadratic_gauss_sum(
    arg1: Any, arg2: Any, arg3: Any = None, *, c0: int = 0, scale_log2: int = 0, use_cpp_kernels: bool = True
) -> complex:
    """`O2`: Evaluate `2^(scale_log2 / 2) * sum_{r in F_2^g} i^(q0 + ell.r + 2 r^T triu(Gamma,1) r)`."""
    q0, l_arr, B_arr = _parse_gauss_sum_args(arg1, arg2, arg3, c0)
    lib = _get_cpp_lib_if_enabled(use_cpp_kernels)
    if lib is not None:
        out_ri = (ctypes.c_double * 2)(0.0, 0.0)
        lib.magic_quadratic_gauss_sum(
            l_arr.shape[0], _c_ptr(l_arr, _c_u8_p), _c_ptr(B_arr, _c_u8_p),
            ctypes.c_int32(q0), ctypes.c_int64(int(scale_log2)), out_ri,
        )
        return complex(float(out_ri[0]), float(out_ri[1]))
    nonzero, k2, v8 = quadratic_gauss_sum_dyadic(l_arr, B_arr, c0=q0)
    if not nonzero:
        return 0.0 + 0.0j
    total_k2 = k2 + int(scale_log2)
    unit = complex(_OMEGA_POW[v8]) * (_SQRT2 if (total_k2 & 1) else 1.0)
    return complex(math.ldexp(unit.real, total_k2 >> 1), math.ldexp(unit.imag, total_k2 >> 1))


def overlap(bra: PhaseQFEComponent, ket: PhaseQFEComponent, *, use_cpp_kernels: bool = True) -> complex:
    """`O1`: Exact inner product `<phi(bra) | phi(ket)>` (`np.vdot(bra, ket)`)."""
    if not (isinstance(bra, PhaseQFEComponent) and isinstance(ket, PhaseQFEComponent)):
        raise TypeError("overlap expects PhaseQFEComponent operands")
    for c in (bra, ket):
        if not (_is_valid_int(c.n_qubits) and _is_valid_int(c.d)) or any(
            not isinstance(a, np.ndarray) or a.dtype.kind not in ("u", "i")
            for a in (c.h_words, c.A_words, c.ell, c.Gamma)
        ):
            raise TypeError("overlap requires integer metadata and integer ndarray buffers")
        nw, cd = (int(c.n_qubits) + 63) // 64, int(c.d)
        if c.n_qubits < 1 or cd < 0 or c.n_words != nw or (
            c.h_words.shape, c.A_words.shape, c.ell.shape, c.Gamma.shape
        ) != ((nw,), (cd, nw), (cd,), (cd, cd)):
            raise ValueError("Invalid PhaseQFEComponent dimensions or buffer shapes in overlap")
    if bra.n_qubits != ket.n_qubits:
        raise ValueError(f"Qubit count mismatch in overlap: {bra.n_qubits} vs {ket.n_qubits}")

    d_bra, d_ket = int(bra.d), int(ket.d)
    D = d_ket + d_bra
    if D == 0:
        same_h = np.array_equal(np.asarray(bra.h_words, dtype=np.uint64), np.asarray(ket.h_words, dtype=np.uint64))
        return (1.0 + 0.0j) if same_h else 0.0j

    if _get_cpp_lib_if_enabled(use_cpp_kernels) is not None and _CACHED_MAGIC_MOD is not None:
        return complex(_CACHED_MAGIC_MOD.cpp_overlap(bra, ket))

    V = np.vstack([np.asarray(ket.A_words, dtype=np.uint64), np.asarray(bra.A_words, dtype=np.uint64)]).copy()
    C = np.eye(D, dtype=np.uint8)
    rhs = (np.asarray(ket.h_words, dtype=np.uint64) ^ np.asarray(bra.h_words, dtype=np.uint64)).copy()
    t0 = np.zeros(D, dtype=np.uint8)
    null_cols: list[NDArray[np.uint8]] = []

    for col in range(D):
        p = first_set_bit_in_words(V[col])
        if p >= 0:
            w_idx, mask = p >> 6, np.uint64(1) << np.uint64(p & 63)
            if col + 1 < D:
                has_p = (V[col + 1 :, w_idx] & mask) != np.uint64(0)
                if np.any(has_p):
                    idx = np.flatnonzero(has_p) + (col + 1)
                    V[idx] ^= V[col]
                    C[idx] ^= C[col]
            if (rhs[w_idx] & mask) != np.uint64(0):
                rhs ^= V[col]
                t0 ^= C[col]
        else:
            null_cols.append(C[col].copy())

    if words_any_nonzero(rhs):
        return 0.0 + 0.0j

    tilde_ell = np.concatenate([ket.ell.astype(np.int32), (-bra.ell.astype(np.int32)) & 3])
    tilde_G = np.zeros((D, D), dtype=np.uint8)
    if d_ket > 0:
        tilde_G[:d_ket, :d_ket] = np.asarray(ket.Gamma, dtype=np.uint8) & np.uint8(1)
    if d_bra > 0:
        tilde_G[d_ket:, d_ket:] = np.asarray(bra.Gamma, dtype=np.uint8) & np.uint8(1)

    t0_i, t0_f = t0.astype(np.int32), t0.astype(np.float64)
    U = np.triu(tilde_G, 1)
    quad_c0 = int(t0_f @ (U.astype(np.float64) @ t0_f)) & 1
    c0 = int((int(t0_i @ tilde_ell) + 2 * quad_c0) & 3)

    g = len(null_cols)
    if g == 0:
        return quadratic_gauss_sum(
            np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8),
            c0=c0, scale_log2=-D, use_cpp_kernels=False,
        )

    L = np.column_stack(null_cols)
    L_f, U_f, G_f = L.astype(np.float64), U.astype(np.float64), tilde_G.astype(np.float64)
    signed_ell = tilde_ell * (1 - 2 * t0_i)
    quad_from_t0 = ((G_f @ t0_f).astype(np.int64) & 1).astype(np.int32)
    diag_quad = (np.sum((U_f @ L_f).astype(np.int64) * L.astype(np.int64), axis=0) & 1).astype(np.int32)
    lin = ((L.astype(np.int32).T @ (signed_ell + 2 * quad_from_t0)) + 2 * diag_quad) & 3

    odd_mask = (tilde_ell & 1) != 0
    B1 = ((L_f[odd_mask].T @ L_f[odd_mask]).astype(np.int64) & 1).astype(np.uint8) if np.any(odd_mask) else np.zeros((g, g), dtype=np.uint8)
    B2 = ((L_f.T @ ((G_f @ L_f).astype(np.int64) & 1).astype(np.float64)).astype(np.int64) & 1).astype(np.uint8)
    B = B1 ^ B2
    np.fill_diagonal(B, 0)
    return quadratic_gauss_sum(lin.astype(np.uint8), B, c0=c0, scale_log2=-D, use_cpp_kernels=use_cpp_kernels)
