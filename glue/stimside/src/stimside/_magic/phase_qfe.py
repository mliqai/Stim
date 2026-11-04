"""Phase-preserving Moving-Reference Quadratic-Form Expansion (QFE) engine.

Implements Recipe v3 primitives for `magicside`: `C1` (`PhaseQFEComponent`,
`ShotExpansion`), `P1`–`P2`, `K1`, `N1`–`N3`, and `M1`–`M2`. The `H0`–`H4` local
surgery lives in the C++ `WorkShot` (`_native/magic_kernels.cpp`).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._native._kernel_loader import (
    c_ptr as _c_ptr,
    c_w_ptr as _c_w_ptr,
    i32_p as _c_i32_p,
    u8_p as _c_u8_p,
    u64_p as _c_u64_p,
)
from stimside._native.magic_kernels import get_magic_kernels_lib as _get_cpp_lib
from stimside.util.bitwords import (
    coerce_pauli_words,
    first_set_bit_in_words,
    get_bit_in_words,
    get_bits_at_qubits,
    n_words_for_qubits,
    pack_bit_vector,
    pauli_multiply,
    tail_mask,
    unit_bit_words,
    validate_hermitian_pauli,
    word_dot_parity,
    word_dot_parity_batch,
    words_any_nonzero,
    words_any_overlap,
)
from stimside.util.reference_chp import ReferenceCHPTableau

SQRT2: float = math.sqrt(2.0)
INV_SQRT2: float = math.ldexp(1.0, -1) * SQRT2
OMEGA: complex = complex(INV_SQRT2, INV_SQRT2)
OMEGA_CONJ: complex = complex(INV_SQRT2, -INV_SQRT2)
OMEGA_POWERS: tuple[complex, ...] = (
    1.0 + 0.0j,
    OMEGA,
    0.0 + 1.0j,
    complex(-INV_SQRT2, INV_SQRT2),
    -1.0 + 0.0j,
    complex(-INV_SQRT2, -INV_SQRT2),
    0.0 - 1.0j,
    OMEGA_CONJ,
)
OMEGA_ARR: NDArray[np.complex128] = np.array(OMEGA_POWERS, dtype=np.complex128)
I_POWERS: tuple[complex, ...] = (1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j)
I_POWERS_ARR: NDArray[np.complex128] = np.array(I_POWERS, dtype=np.complex128)
_INV_SQRT2, _PHASE4 = INV_SQRT2, I_POWERS_ARR

__all__ = [
    "DEFAULT_ATOL", "K1_CANCEL_REL_TOL", "ImpossiblePostselectionError",
    "n_words_for_qubits", "tail_mask", "pack_bit_vector",
    "coerce_pauli_words", "unit_bit_words", "get_bit_in_words", "get_bits_at_qubits",
    "word_dot_parity", "word_dot_parity_batch", "words_any_nonzero", "words_any_overlap",
    "first_set_bit_in_words", "pauli_multiply", "validate_hermitian_pauli",
    "PhaseQFEComponent", "ShotExpansion",
    "MeasurementResult", "CompiledPauliBlockStep", "CompiledPauliBlock",
    "apply_reference_tableau_to_statevector",
    "apply_pauli", "apply_quarter_turn",
    "merge_frames", "apply_pauli_t", "apply_ccz",
    "apply_compiled_block", "project_diagonal", "contract_choi_expansions",
    "born_split_expansion", "measure_expansion", "validate_atol",
]

# Exact IEEE-754 float64 / complex128 constants
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


def check_unitary_norm(pre_norm: float, post_norm: float, atol: float, op_name: str) -> None:
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
    def zero(cls, n_qubits: int) -> PhaseQFEComponent:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        nw = n_words_for_qubits(n_qubits)
        return cls(
            int(n_qubits), 0, np.zeros(nw, dtype=np.uint64), np.zeros((0, nw), dtype=np.uint64),
            np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
        )

    @classmethod
    def computational_basis_state(cls, n_qubits: int, bits: Sequence[int | bool] | NDArray[Any]) -> PhaseQFEComponent:
        comp = cls.zero(n_qubits)
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

    def geometry_key(self) -> bytes:
        header = np.array([self.n_qubits, self.d], dtype=np.int64).tobytes()
        ell_par = np.ascontiguousarray(self.ell & np.uint8(1)).tobytes()
        return header + self.A_words.tobytes() + self.Gamma.tobytes() + ell_par + self.pivots.tobytes()

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
    def zero(
        cls, n_qubits: int, *, atol: float = DEFAULT_ATOL
    ) -> ShotExpansion:
        if not _is_valid_int(n_qubits) or n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits!r}")
        nw = n_words_for_qubits(n_qubits)
        return cls(
            int(n_qubits), 0, np.zeros((0, nw), dtype=np.uint64), np.zeros(nw, dtype=np.uint64),
            np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
            np.array([1.0 + 0.0j], dtype=np.complex128), np.zeros((1, nw), dtype=np.uint64),
            np.zeros((1, 0), dtype=np.uint8), np.zeros(0, dtype=np.uint8),
            0.0, atol,
        )

    @classmethod
    def from_component(
        cls, comp: PhaseQFEComponent, weight: complex = 1.0 + 0.0j, *,
        atol: float = DEFAULT_ATOL,
    ) -> ShotExpansion:
        if not isinstance(comp, PhaseQFEComponent):
            raise ValueError(f"Expected PhaseQFEComponent, got {type(comp)!r}")
        return cls(
            comp.n_qubits, comp.d, comp.A_words.copy(), comp.a_union,
            comp.Gamma.copy(), comp.pivots.copy(),
            np.array([complex(weight)], dtype=np.complex128),
            comp.h_words.copy()[None, :], comp.ell.copy()[None, :],
            (comp.ell & np.uint8(1)).copy(), 0.0, atol,
        )

    @classmethod
    def from_components(
        cls,
        terms: Sequence[tuple[complex, PhaseQFEComponent]],
        *,
        n_qubits: int | None = None,
        merge: bool = True,
        atol: float = DEFAULT_ATOL,
    ) -> ShotExpansion:
        if len(terms) == 0:
            if n_qubits is None or not _is_valid_int(n_qubits) or n_qubits < 1:
                raise ValueError("n_qubits >= 1 must be provided when constructing an empty ShotExpansion")
            return _empty_shot_expansion(int(n_qubits), atol=atol)
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
            (ref_comp.ell & np.uint8(1)).copy(), 0.0, atol,
            _is_canonical_merged=(len(terms) <= 1),
        )
        return merge_frames(exp) if merge else exp

    def copy(self) -> ShotExpansion:
        assert self.ell_odd is not None
        return ShotExpansion(
            self.n_qubits, self.d, self.A_words.copy(), self.a_union.copy(),
            self.Gamma.copy(), self.pivots.copy(), self.weights.copy(),
            self.h_words.copy(), self.ell.copy(), self.ell_odd.copy(),
            self.log_mass, self.atol, self._is_canonical_merged,
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
        exp = merge_frames(self) if (do_canon and self.chi > 1) else self
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
    n_qubits: int, *, log_mass: float = 0.0, atol: float = DEFAULT_ATOL
) -> ShotExpansion:
    nw = n_words_for_qubits(n_qubits)
    return ShotExpansion(
        n_qubits, 0, np.zeros((0, nw), dtype=np.uint64), np.zeros(nw, dtype=np.uint64),
        np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.complex128), np.zeros((0, nw), dtype=np.uint64),
        np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.uint8),
        log_mass, atol,
    )


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
    post_state: ShotExpansion


@dataclass(frozen=True)
class CompiledPauliBlockStep:
    """Single unflattened step inside a `CompiledPauliBlock` (`mode='sequential'`)."""

    kind: Literal["P1", "P2", "N1", "N2"]
    paulis: tuple[tuple[NDArray[np.uint64], NDArray[np.uint64], int], ...]
    sign: int = 1


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


def apply_local_unitary_c128(
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
    tab = ref_tableau.to_stim_tableau() if isinstance(ref_tableau, ReferenceCHPTableau) else ref_tableau
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
                psi = apply_local_unitary_c128(psi, U1, (q,), n, endian)
        elif name == "CX":
            for i in range(0, len(targs), 2):
                psi = apply_local_unitary_c128(psi, _C128_CX, (targs[i], targs[i + 1]), n, endian)
        else:
            raise ValueError(f"Unexpected gate {name!r} in tableau elimination circuit")
    return psi


# ---------------------------------------------------------------------------
# Unitary Frame Primitives: P1 (`apply_pauli`) and P2 (`apply_quarter_turn`)
# ---------------------------------------------------------------------------


def _validate_component_terms(terms: Sequence[tuple[complex, PhaseQFEComponent]]) -> int:
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
        src.ell_odd.copy(), src.log_mass, src.atol, is_canonical_merged,
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


def apply_pauli(
    target: ShotExpansion,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    in_place: bool = False,
) -> ShotExpansion:
    """`P1`: Apply `P = i**kappa X(a) Z(b)` (`O(chi * (n_words + d))`)."""
    if not isinstance(target, ShotExpansion):
        raise TypeError(f"apply_pauli expects ShotExpansion, got {type(target)!r}")
    a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
    exp = target if in_place else target.copy()
    if exp.chi == 0:
        return exp
    if not np.all(np.isfinite(exp.weights)):
        raise FloatingPointError("Non-finite weights in apply_pauli")
    _get_cpp_lib().magic_apply_pauli_batch(
        exp.chi, exp.d, exp.n_words, _c_w_ptr(exp.weights),
        _c_ptr(exp.h_words, _c_u64_p), _c_ptr(exp.ell, _c_u8_p),
        _c_ptr(exp.A_words, _c_u64_p), _c_ptr(exp.a_union, _c_u64_p),
        _c_ptr(a_w, _c_u64_p), _c_ptr(b_w, _c_u64_p), ctypes.c_uint8(kap),
    )
    return exp


def apply_quarter_turn(
    target: ShotExpansion,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    sign: int = 1,
    in_place: bool = False,
    check_unitary: bool = True,
    atol: float | None = None,
) -> ShotExpansion:
    """`P2`: Apply `R(sign * P) = (I + i * sign * P) / sqrt(2)` without rank growth (`chi' <= chi`)."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if atol is not None:
        validate_atol(atol)
    if not isinstance(target, ShotExpansion):
        raise TypeError(f"apply_quarter_turn expects ShotExpansion, got {type(target)!r}")
    tol_e = validate_atol(target.atol if atol is None else atol)
    a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, target.n_qubits)
    kap = (validate_hermitian_pauli(a_w, b_w, kap) + (2 if sign == -1 else 0)) & 3
    exp = target if in_place else target.copy()
    pre_norm = _checked_shot_pre_norm(exp, check_unitary, "apply_quarter_turn")
    if exp.chi == 0:
        exp._is_canonical_merged = True
        return exp

    lib = _get_cpp_lib()
    if not exp._is_canonical_merged:
        merged_in = merge_frames(exp)
        exp = _copy_shot_inplace(exp, merged_in) if in_place else merged_in
        if exp.chi == 0:
            if check_unitary:
                check_unitary_norm(pre_norm, 0.0, tol_e, "apply_quarter_turn")
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
        check_unitary_norm(pre_norm, exp.norm_sq(), tol_e, "apply_quarter_turn")
    return exp


# ---------------------------------------------------------------------------
# Coherent Merging (`K1`)
# ---------------------------------------------------------------------------


def merge_frames(
    expansion: ShotExpansion,
    *,
    cancel_tol: float = K1_CANCEL_REL_TOL,
) -> ShotExpansion:
    """`K1`: Canonicalize coset offsets, coherently sum identical keys, and cancel roundoff zeros."""
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"merge_frames expects ShotExpansion, got {type(expansion)!r}")
    exp = expansion.copy()
    if exp.chi == 0:
        exp._is_canonical_merged = True
        return exp
    if not np.all(np.isfinite(exp.weights)):
        raise FloatingPointError("Non-finite weights encountered in merge_frames")
    out_chi, out_w, out_h, out_ell, out_ptrs = _alloc_cpp_row_bufs(exp.chi, exp.n_words, exp.d)
    _get_cpp_lib().magic_canonicalize_and_merge(
        exp.chi, exp.d, exp.n_words, *_cpp_shot_in_ptrs(exp),
        ctypes.c_double(cancel_tol), *out_ptrs,
    )
    oc = int(out_chi.value)
    exp.weights = np.ascontiguousarray(out_w[:oc])
    exp.h_words = np.ascontiguousarray(out_h[:oc])
    exp.ell = np.ascontiguousarray(out_ell[:oc])
    exp._is_canonical_merged = True
    return exp


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
    check_unitary: bool,
    pre_norm: float,
    tol: float,
    op_name: str,
) -> ShotExpansion:
    r = int(coeffs.shape[0])
    if r == 0:
        empty = _empty_shot_expansion(
            expansion.n_qubits, log_mass=expansion.log_mass, atol=expansion.atol,
        )
        if check_unitary:
            check_unitary_norm(pre_norm, 0.0, tol, op_name)
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
    out = merge_frames(raw)
    if check_unitary:
        check_unitary_norm(pre_norm, out.norm_sq(), tol, op_name)
    return out


def _apply_pauli_sum_op(
    expansion: ShotExpansion,
    prep_branches: Any,
    cpp_runner: Any,
    *,
    merge: bool,
    check_unitary: bool,
    atol: float | None,
    op_name: str,
) -> ShotExpansion:
    if atol is not None:
        validate_atol(atol)
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"{op_name} expects ShotExpansion, got {type(expansion)!r}")
    tol = validate_atol(expansion.atol if atol is None else atol)
    bw, ba, bb, bk = prep_branches(expansion.n_qubits)
    pre_norm = _checked_shot_pre_norm(expansion, check_unitary, op_name)
    if expansion.chi == 0:
        empty_out = expansion.copy()
        empty_out._is_canonical_merged = True
        return empty_out
    if merge:
        r = int(bw.shape[0])
        out_chi, out_w, out_h, out_ell, out_ptrs = _alloc_cpp_row_bufs(
            r * expansion.chi, expansion.n_words, expansion.d
        )
        head = (expansion.chi, expansion.d, expansion.n_words, *_cpp_shot_in_ptrs(expansion, include_a_union=True))
        cpp_runner(_get_cpp_lib(), head, out_ptrs, bw, ba, bb, bk)
        oc = int(out_chi.value)
        res = _clone_shot_with_rows(
            expansion, out_w[:oc].copy(), out_h[:oc].copy(), out_ell[:oc].copy(), is_canonical_merged=True
        )
        if check_unitary:
            check_unitary_norm(pre_norm, res.norm_sq(), tol, op_name)
        return res
    return _branch_pauli_sum_shot(
        expansion, bw, ba, bb, bk, merge=False,
        check_unitary=check_unitary, pre_norm=pre_norm, tol=tol, op_name=op_name,
    )


def apply_pauli_t(
    expansion: ShotExpansion,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    dag: bool = False,
    sign: int | None = None,
    merge: bool = True,
    check_unitary: bool = True,
    atol: float | None = None,
) -> ShotExpansion:
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
        atol=atol, op_name="apply_pauli_t",
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
    expansion: ShotExpansion,
    paulis: Sequence[Any],
    *,
    merge: bool = True,
    check_unitary: bool = True,
    atol: float | None = None,
) -> ShotExpansion:
    """`N2`: Apply generalized 3-axis `CCZ(Q1, Q2, Q3)` via 8-branch Pauli expansion."""

    def _cpp(lib: Any, head: tuple[Any, ...], out_ptrs: tuple[Any, ...], bw: Any, ba: Any, bb: Any, bk: Any) -> None:
        lib.magic_branch_ccz(
            *head, _c_w_ptr(bw), _c_ptr(ba, _c_u64_p), _c_ptr(bb, _c_u64_p), _c_ptr(bk, _c_u8_p),
            ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs,
        )

    return _apply_pauli_sum_op(
        expansion, lambda n_q: _normalize_ccz_branches(paulis, n_q), _cpp,
        merge=merge, check_unitary=check_unitary, atol=atol, op_name="apply_ccz",
    )


def _validate_compiled_block(block: CompiledPauliBlock, n_qubits: int | None) -> None:
    if not isinstance(block, CompiledPauliBlock):
        raise ValueError(f"Expected CompiledPauliBlock, got {type(block)!r}")
    if not _is_valid_int(block.n_qubits) or block.n_qubits < 1:
        raise ValueError(f"block.n_qubits must be >= 1, got {block.n_qubits!r}")
    eff_nq = int(block.n_qubits) if n_qubits is None else n_qubits
    nw = n_words_for_qubits(eff_nq)
    if n_qubits is not None and block.n_qubits != n_qubits:
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
    cur: ShotExpansion,
    steps: Sequence[CompiledPauliBlockStep | CompiledPauliBlock],
    *,
    tol: float,
) -> ShotExpansion:
    for step in steps:
        if isinstance(step, CompiledPauliBlock):
            cur = apply_compiled_block(cur, step, check_unitary=False, atol=tol)
        elif step.kind == "P1":
            cur = apply_pauli(cur, *step.paulis[0])
        elif step.kind == "P2":
            cur = apply_quarter_turn(
                cur, *step.paulis[0], sign=step.sign, check_unitary=False, atol=tol
            )
        elif step.kind == "N1":
            cur = apply_pauli_t(
                cur, *step.paulis[0], sign=step.sign, merge=True,
                check_unitary=False, atol=tol,
            )
        elif step.kind == "N2":
            cur = apply_ccz(cur, step.paulis, merge=True, check_unitary=False, atol=tol)
        else:
            raise ValueError(f"Unknown step kind: {step.kind!r}")
    return merge_frames(cur)


def apply_compiled_block(
    expansion: ShotExpansion,
    block: CompiledPauliBlock,
    *,
    check_unitary: bool | None = None,
    atol: float | None = None,
) -> ShotExpansion:
    """`N3`: Execute a `CompiledPauliBlock` (`pauli_sum` or `sequential` mode)."""
    if atol is not None:
        validate_atol(atol)
    if not isinstance(block, CompiledPauliBlock):
        raise ValueError(f"Expected CompiledPauliBlock, got {type(block)!r}")
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"apply_compiled_block expects ShotExpansion, got {type(expansion)!r}")
    do_check = block.is_unitary if check_unitary is None else bool(check_unitary)
    tol = validate_atol(expansion.atol if atol is None else atol)
    _validate_compiled_block(block, expansion.n_qubits)
    pre_norm = _checked_shot_pre_norm(expansion, do_check, "apply_compiled_block")
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
            merge=True, check_unitary=do_check,
            pre_norm=pre_norm, tol=tol, op_name="apply_compiled_block",
        )

    out = expansion.copy()
    out.weights *= g_sc
    out = merge_frames(out) if block.is_identity else _execute_sequential_steps(out, block.steps, tol=tol)
    if do_check:
        check_unitary_norm(pre_norm, out.norm_sq(), tol, "apply_compiled_block")
    return out


# ---------------------------------------------------------------------------
# Projective Measurements & Born Sampling: M1 (`project_diagonal`) and M2 (`measure_expansion`)
# ---------------------------------------------------------------------------


def project_diagonal(
    expansion: ShotExpansion,
    b: Any,
    sigma: int = 0,
    outcome: int = 0,
) -> tuple[ShotExpansion, float]:
    """`M1`: Project onto `Pi_m = (I + (-1)^m (-1)^sigma Z(b)) / 2` without renormalizing."""
    sig = int(sigma)
    if sig not in (0, 1):
        raise ValueError(f"sigma must be 0 or 1, got {sigma}")
    m = int(outcome)
    if m not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome}")
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"project_diagonal expects ShotExpansion, got {type(expansion)!r}")

    b_w = coerce_pauli_words(b, expansion.n_qubits)
    exp = expansion.copy()
    if exp.chi == 0:
        exp._is_canonical_merged = True
        return exp, 0.0
    if not np.all(np.isfinite(exp.weights)):
        raise FloatingPointError("Non-finite weights in project_diagonal")

    out_chi, out_d, bufs, out_ptrs = _alloc_cpp_geom_bufs(exp.chi, exp.n_words, max(1, exp.d))
    out_z = ctypes.c_double(0.0)
    _get_cpp_lib().magic_project_diagonal(
        exp.chi, exp.d, exp.n_words, *_cpp_shot_in_ptrs(exp),
        _c_ptr(b_w, _c_u64_p), ctypes.c_uint8(sig << 1), ctypes.c_uint8(m),
        ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs, ctypes.byref(out_z),
    )
    _assign_cpp_geom_shot(exp, int(out_chi.value), int(out_d.value), bufs)
    return exp, float(out_z.value)


def contract_choi_expansions(
    left: ShotExpansion,
    right: ShotExpansion,
    n_op: int,
    q_act: Sequence[int],
    right_is_choi: bool,
    *,
    atol: float = DEFAULT_ATOL,
) -> ShotExpansion:
    """Contract Choi `left` with Choi/ket `right` across `q_act` (scaled by `2**-(len(q_act)//2)`)."""
    m = len(q_act)
    if not _is_valid_int(n_op) or n_op < 0 or m > n_op:
        raise ValueError(f"Invalid n_op={n_op!r} or len(q_act)={m} in contract_choi_expansions")
    seen_q: set[int] = set()
    q_list: list[int] = []
    for q in q_act:
        if not _is_valid_int(q):
            raise ValueError(f"q_act entries must be integers, got {q!r}")
        qi = int(q)
        if qi < 0 or qi >= n_op or qi in seen_q:
            raise ValueError(f"Invalid or duplicate q_act entry {q!r} for n_op={n_op}")
        seen_q.add(qi)
        q_list.append(qi)
    want_left = max(1, n_op + m)
    eff_keep = max(1, (n_op + m) if right_is_choi else n_op)
    if left.n_qubits != want_left or right.n_qubits != eff_keep:
        raise ValueError(
            f"Operand n_qubits mismatch in contract_choi_expansions: got ({left.n_qubits}, {right.n_qubits}), "
            f"expected ({want_left}, {eff_keep}) for n_op={n_op}, len(q_act)={m}, right_is_choi={right_is_choi}"
        )
    prod_chi = left.chi * right.chi
    if prod_chi == 0:
        return ShotExpansion.from_components((), n_qubits=eff_keep, atol=atol)
    if not (np.all(np.isfinite(left.weights)) and np.all(np.isfinite(right.weights))):
        raise FloatingPointError("Non-finite weights in contract_choi_expansions")
    q_arr = np.asarray(q_list, dtype=np.int32)
    out_chi, out_d, bufs, out_ptrs = _alloc_cpp_geom_bufs(
        prod_chi, n_words_for_qubits(eff_keep), max(1, left.d + right.d)
    )
    rc = int(
        _get_cpp_lib().magic_contract_choi(
            left.chi, left.d, left.n_words, *_cpp_shot_in_ptrs(left),
            right.chi, right.d, right.n_words, *_cpp_shot_in_ptrs(right),
            n_op, _c_ptr(q_arr, _c_i32_p), m, int(bool(right_is_choi)),
            ctypes.c_double(K1_CANCEL_REL_TOL), *out_ptrs,
        )
    )
    if rc != 0:
        if rc == 1:
            raise ValueError("Invalid q_act or Choi pivot support in contract_choi_expansions")
        if rc == 2:
            raise MemoryError("Out of memory in magic_contract_choi")
        raise RuntimeError(f"magic_contract_choi failed with status {rc}")
    oc = int(out_chi.value)
    if oc == 0:
        return ShotExpansion.from_components((), n_qubits=eff_keep, atol=atol)
    return _assign_cpp_geom_shot(ShotExpansion.zero(eff_keep, atol=atol), oc, int(out_d.value), bufs)


def born_split_expansion(
    expansion: ShotExpansion,
    a_w: NDArray[np.uint64],
    b_w: NDArray[np.uint64],
    kap: int,
    *,
    atol: float,
    z_in: float | None = None,
) -> tuple[
    ShotExpansion, float, ShotExpansion, float, float, ShotExpansion,
    tuple[NDArray[np.uint64], NDArray[np.uint64], int] | None,
]:
    """Rotate if needed, project onto both outcomes, and verify Born mass conservation."""
    z = expansion.norm_sq() if z_in is None else z_in
    if not (math.isfinite(z) and z >= 0.0):
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={z!r}")
    if words_any_nonzero(a_w):
        e_p_w = unit_bit_words(first_set_bit_in_words(a_w), expansion.n_words)
        rot: tuple[NDArray[np.uint64], NDArray[np.uint64], int] | None = (
            a_w, np.ascontiguousarray(b_w ^ e_p_w), (kap + 1) & 3
        )
        work_exp = apply_quarter_turn(expansion, *rot, sign=1, check_unitary=True, atol=atol)
        b_diag, sig_diag = e_p_w, 0
    else:
        rot = None
        work_exp, b_diag, sig_diag = expansion, b_w, (kap >> 1) & 1

    exp0, z0_raw = project_diagonal(work_exp, b_diag, sig_diag, 0)
    exp1, z1_raw = project_diagonal(work_exp, b_diag, sig_diag, 1)
    scale_z = max(1.0, z)
    if not (math.isfinite(z0_raw) and math.isfinite(z1_raw)):
        raise FloatingPointError(f"Non-finite projected Born masses: z0={z0_raw!r}, z1={z1_raw!r}")
    if z0_raw < -1e-12 * scale_z or z1_raw < -1e-12 * scale_z:
        raise FloatingPointError(f"Significantly negative projected Born mass: z0={z0_raw!r}, z1={z1_raw!r}, z={z!r}")
    z0, z1 = max(0.0, float(z0_raw)), max(0.0, float(z1_raw))
    if not (abs((z0 + z1) - z) <= atol * scale_z):
        raise FloatingPointError(
            f"M2 Born mass conservation violated: z0={z0:.16e}, z1={z1:.16e}, "
            f"z0+z1={z0 + z1:.16e}, z={z:.16e}, tol={atol:.3e}"
        )
    return exp0, z0, exp1, z1, z, work_exp, rot


def measure_expansion(
    expansion: ShotExpansion,
    a: Any,
    b: Any = None,
    kappa: int = 0,
    *,
    sigma: int | None = None,
    rng: np.random.Generator | None = None,
    requested_bit: int | None = None,
    sample_outcome: Callable[[float], int] | None = None,
    p_flip: float = 0.0,
    invert_result: bool = False,
    renormalize: bool = True,
    keep_rotated_frame: bool = False,
    atol: float | None = None,
) -> MeasurementResult:
    """`M2`: Measure Hermitian Pauli `P = i**kappa X(a) Z(b)` (or diagonal `(-1)^sigma Z(a)`)."""
    if not (0.0 <= p_flip <= 1.0):
        raise ValueError(f"p_flip must be in [0, 1], got {p_flip}")
    if requested_bit is not None and requested_bit not in (0, 1):
        raise ValueError(f"requested_bit must be 0, 1, or None, got {requested_bit}")
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"measure_expansion expects ShotExpansion, got {type(expansion)!r}")

    n_q = expansion.n_qubits
    tol = validate_atol(expansion.atol if atol is None else atol)
    in_log_mass = expansion.log_mass

    if b is None and sigma is not None:
        if sigma not in (0, 1):
            raise ValueError(f"sigma must be 0 or 1, got {sigma}")
        a_w = np.zeros(n_words_for_qubits(n_q), dtype=np.uint64)
        b_w = coerce_pauli_words(a, n_q)
        kap = (int(sigma) & 1) << 1
    else:
        a_w, b_w, kap = _unpack_pauli_args(a, b, kappa, n_q)
        kap = validate_hermitian_pauli(a_w, b_w, kap)

    exp0, z0, exp1, z1, z, _, rot = born_split_expansion(expansion, a_w, b_w, kap, atol=tol)

    zero_floor = K1_CANCEL_REL_TOL * z
    if requested_bit is not None or sample_outcome is not None:
        if requested_bit is None:
            if z <= 0.0 or (z0 + z1) <= 0.0:
                raise ImpossiblePostselectionError("Cannot sample measurement from a zero-norm expansion")
            assert sample_outcome is not None
            outcome = int(sample_outcome(min(1.0, max(0.0, z1 / z))))
        else:
            outcome = int(requested_bit)
        chosen_z, chosen_exp = (z0, exp0) if outcome == 0 else (z1, exp1)
        if z <= 0.0 or chosen_z <= zero_floor or chosen_exp.chi == 0:
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

    if rot is not None and not keep_rotated_frame:
        chosen_exp = apply_quarter_turn(chosen_exp, *rot, sign=-1, check_unitary=True, atol=tol)

    if renormalize:
        chosen_exp.weights = np.ascontiguousarray(chosen_exp.weights * (1.0 / math.sqrt(chosen_z)))
        chosen_exp.log_mass = in_log_mass + math.log(chosen_z)
        out_log_mass = chosen_exp.log_mass
    else:
        chosen_exp.log_mass = in_log_mass
        out_log_mass = in_log_mass + math.log(chosen_z)

    return MeasurementResult(
        outcome, reported_outcome, readout_flipped, z0, z1, z, chosen_z, out_log_mass, chosen_exp
    )
