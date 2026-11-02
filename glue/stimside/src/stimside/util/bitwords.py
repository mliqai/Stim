"""Shared multi-word uint64 bitpacking and Pauli parity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
    """Compute `<u, v> mod 2` across packed `uint64` word arrays without allocating."""
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
    if words.shape[0] == 1:
        return int(words[0]) != 0
    return any(words.tolist())


def words_any_overlap(u_words: NDArray[np.uint64], v_words: NDArray[np.uint64]) -> bool:
    """Return True iff `u_words & v_words` has at least one set bit."""
    if u_words.shape[0] == 1:
        return (int(u_words[0]) & int(v_words[0])) != 0
    return any(a & b for a, b in zip(u_words.tolist(), v_words.tolist()))


def first_set_bit_in_words(words: NDArray[np.uint64]) -> int:
    """Return smallest bit index `q >= 0` set in `words`, or `-1` if all zero."""
    for w_idx, val in enumerate(words.tolist()):
        if val:
            return (w_idx << 6) + (val & -val).bit_length() - 1
    return -1


def pauli_multiply(
    a1_w: NDArray[np.uint64],
    b1_w: NDArray[np.uint64],
    kap1: int,
    a2_w: NDArray[np.uint64],
    b2_w: NDArray[np.uint64],
    kap2: int,
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
