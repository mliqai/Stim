"""Unit and property tests for `stimside._magic.phase_qfe` (Milestone M1: `V1.1`, `V1.2a`, multi-word, tolerance guards)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import stim

from stimside._magic import phase_qfe as pq
from stimside._magic.phase_qfe import (
    CompiledPauliBlock,
    CompiledPauliBlockStep,
    ImpossiblePostselectionError,
    PhaseQFEComponent,
    ShotExpansion,
    add_column,
    apply_ccz,
    apply_compiled_block,
    apply_pauli,
    apply_pauli_t,
    apply_quarter_turn,
    coerce_pauli_words,
    drop_variable,
    eliminate_zero_amplitude,
    get_bits_at_qubits,
    make_principal,
    measure_expansion,
    merge_frames,
    n_words_for_qubits,
    pauli_multiply,
    project_diagonal,
    restrict_amplitude,
    unit_bit_words,
)
from stimside.util.bitwords import unpack_bit_vector
from stimside.util.reference_chp import ReferenceCHPTableau


# ---------------------------------------------------------------------------
# Independent Dense Little-Endian Oracle Helpers (complex128, zero phase fudge)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2
_OMEGA = complex(_INV_SQRT2, _INV_SQRT2)


def formal_qfe_ket_le(
    n_qubits: int,
    h_words: np.ndarray,
    A_words: np.ndarray,
    ell: np.ndarray,
    Gamma: np.ndarray,
) -> np.ndarray:
    """Evaluate formal ket `2^(-d/2) sum_{u in F_2^d} i^(ell.u + 2 u^T triu(Gamma,1) u) |h ^ A u>` directly."""
    d = int(ell.shape[0])
    out = np.zeros(1 << n_qubits, dtype=np.complex128)
    h_bits = unpack_bit_vector(h_words, n_qubits)
    A_bits = (
        unpack_bit_vector(A_words, n_qubits)
        if d > 0
        else np.zeros((0, n_qubits), dtype=np.uint8)
    )
    scale = 2.0 ** (-0.5 * d)
    for mask in range(1 << d):
        u = np.array([(mask >> j) & 1 for j in range(d)], dtype=np.int64)
        q = int((ell.astype(np.int64) @ u + 2 * (u @ np.triu(Gamma, 1).astype(np.int64) @ u)) & 3)
        x_bits = h_bits.copy()
        for j in range(d):
            if u[j]:
                x_bits ^= A_bits[j]
        idx = sum(int(x_bits[q_i]) << q_i for q_i in range(n_qubits))
        out[idx] += scale * ((1j) ** q)
    return out


def dense_pauli_le(
    n_qubits: int, a_bits: np.ndarray, b_bits: np.ndarray, kappa: int
) -> np.ndarray:
    """Build exact `2^n x 2^n` little-endian matrix for `P = i**kappa X(a) Z(b)`."""
    dim = 1 << n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex128)
    a_mask = sum(int(a_bits[q]) << q for q in range(n_qubits))
    b_mask = sum(int(b_bits[q]) << q for q in range(n_qubits))
    pref = (1j) ** (int(kappa) & 3)
    for x in range(dim):
        sgn = -1.0 if ((x & b_mask).bit_count() & 1) else 1.0
        mat[x ^ a_mask, x] = pref * sgn
    return mat


def dense_quarter_turn_le(
    n_qubits: int, a_bits: np.ndarray, b_bits: np.ndarray, kappa: int, sign: int = 1
) -> np.ndarray:
    P = dense_pauli_le(n_qubits, a_bits, b_bits, kappa)
    return (np.eye(1 << n_qubits, dtype=np.complex128) + (1j * sign) * P) * _INV_SQRT2


def dense_pauli_t_le(
    n_qubits: int, a_bits: np.ndarray, b_bits: np.ndarray, kappa: int, sign: int = 1
) -> np.ndarray:
    P = dense_pauli_le(n_qubits, a_bits, b_bits, kappa)
    w = _OMEGA if sign == 1 else _OMEGA.conjugate()
    I_mat = np.eye(1 << n_qubits, dtype=np.complex128)
    return 0.5 * (1.0 + w) * I_mat + 0.5 * (1.0 - w) * P


def dense_ccz_axes_le(
    n_qubits: int,
    axes_bits: list[tuple[np.ndarray, np.ndarray, int]],
) -> np.ndarray:
    dim = 1 << n_qubits
    I_mat = np.eye(dim, dtype=np.complex128)
    mats = [dense_pauli_le(n_qubits, a, b, k) for a, b, k in axes_bits]
    # Projector onto (-1, -1, -1) joint eigenspace gets -1, else +1: I - 2 * ((I-P0)/2)((I-P1)/2)((I-P2)/2)
    proj = (I_mat - mats[0]) @ (I_mat - mats[1]) @ (I_mat - mats[2]) * 0.125
    return I_mat - 2.0 * proj


def random_valid_component(
    rng: np.random.Generator, n_qubits: int, d: int
) -> PhaseQFEComponent:
    pivots = np.sort(rng.choice(n_qubits, size=d, replace=False)).astype(np.int32)
    A = rng.integers(0, 2, size=(n_qubits, d), dtype=np.uint8)
    if d > 0:
        A[pivots, :] = np.eye(d, dtype=np.uint8)
    h = rng.integers(0, 2, size=n_qubits, dtype=np.uint8)
    ell = rng.integers(0, 4, size=d, dtype=np.uint8)
    G = rng.integers(0, 2, size=(d, d), dtype=np.uint8)
    G = np.triu(G, 1)
    G = G ^ G.T
    comp = PhaseQFEComponent.from_unpacked(h, A, ell, G, pivots)
    comp.check_invariants()
    return comp


def random_hermitian_pauli_bits(
    rng: np.random.Generator, n_qubits: int, *, force_nonzero_a: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    while True:
        a = rng.integers(0, 2, size=n_qubits, dtype=np.uint8)
        if not force_nonzero_a or np.any(a):
            break
    b = rng.integers(0, 2, size=n_qubits, dtype=np.uint8)
    ab = int(a @ b) & 1
    kap = (ab + 2 * int(rng.integers(0, 2))) & 3
    return a, b, kap


# ---------------------------------------------------------------------------
# Group A: Bitpacking, Invariants & H0-H4 Support Helpers (V1.1)
# ---------------------------------------------------------------------------


def test_component_zero_and_invariants() -> None:
    for n in [1, 2, 5, 63, 64, 65, 128, 130]:
        comp = PhaseQFEComponent.zero(n)
        comp.check_invariants(require_canonical_offset=True)
        assert comp.d == 0
        assert comp.n_words == (n + 63) // 64
        exp = ShotExpansion.zero(n)
        exp.check_invariants(require_canonical_offset=True)
        if n <= 6:
            sv = comp.to_statevector()
            expected = np.zeros(1 << n, dtype=np.complex128)
            expected[0] = 1.0
            assert np.max(np.abs(sv - expected)) < 1e-15

    with pytest.raises(ValueError):
        PhaseQFEComponent.zero(0)

    # Non-zero upper padding bit in h_words[-1] and A_words[:, -1]
    for n_pad, bad_bit in [(5, 5), (63, 63), (65, 1)]:
        c_bad = PhaseQFEComponent.zero(n_pad)
        c_bad.h_words[-1] |= np.uint64(1) << np.uint64(bad_bit)
        with pytest.raises(ValueError):
            c_bad.check_invariants()

        rng = np.random.default_rng(100 + n_pad)
        c_a = random_valid_component(rng, n_pad, 1)
        c_a.A_words[0, -1] |= np.uint64(1) << np.uint64(bad_bit)
        with pytest.raises(ValueError):
            c_a.check_invariants()

    rng = np.random.default_rng(42)
    base = random_valid_component(rng, 5, 2)

    dup = base.copy()
    dup.pivots[1] = dup.pivots[0]
    with pytest.raises(ValueError):
        dup.check_invariants()

    non_princ = base.copy()
    non_princ.A_words[1] ^= unit_bit_words(int(non_princ.pivots[0]), non_princ.n_words)
    with pytest.raises(ValueError):
        non_princ.check_invariants()

    bad_ell = base.copy()
    bad_ell.ell[0] = np.uint8(4)
    with pytest.raises(ValueError):
        bad_ell.check_invariants()

    bad_sym = base.copy()
    bad_sym.Gamma[0, 1] = np.uint8(1)
    bad_sym.Gamma[1, 0] = np.uint8(0)
    with pytest.raises(ValueError):
        bad_sym.check_invariants()

    bad_diag = base.copy()
    bad_diag.Gamma[0, 0] = np.uint8(1)
    with pytest.raises(ValueError):
        bad_diag.check_invariants()


def test_h0_drop_variable() -> None:
    rng = np.random.default_rng(101)
    n = 5
    for d in [1, 2, 4]:
        for j in sorted({0, d // 2, d - 1}):
            comp = random_valid_component(rng, n, d)
            # Zero out generator j so formal sum over u_j gives exact factor 2 * 2^(-1/2) = sqrt(2)
            comp.A_words[j] = np.zeros(comp.n_words, dtype=np.uint64)
            comp.ell[j] = np.uint8(0)
            comp.Gamma[j, :] = 0
            comp.Gamma[:, j] = 0
            ket_before = formal_qfe_ket_le(
                n, comp.h_words, comp.A_words, comp.ell, comp.Gamma
            )
            expected_pivots = np.delete(comp.pivots, j)

            exp = ShotExpansion.from_component(comp, weight=0.7 - 0.3j)
            drop_variable(exp, j)

            exp.check_invariants()
            assert np.array_equal(exp.pivots, expected_pivots)
            assert (
                np.max(
                    np.abs(
                        _SQRT2 * exp.to_statevector() - (0.7 - 0.3j) * ket_before
                    )
                )
                < 1e-14
            )


def test_h1_add_column_all_mod4_cases() -> None:
    rng = np.random.default_rng(102)
    n, d = 5, 3
    base = random_valid_component(rng, n, d)
    with pytest.raises(ValueError):
        add_column(ShotExpansion.from_component(base), 1, 1)

    for r, s in [(0, 1), (2, 0)]:
        k = 3 - r - s
        for ell_r, ell_s, g_rs, g_rk, g_sk in itertools.product(
            range(4), range(4), range(2), range(2), range(2)
        ):
            comp = base.copy()
            comp.ell[r] = np.uint8(ell_r)
            comp.ell[s] = np.uint8(ell_s)
            comp.Gamma[:, :] = 0
            comp.Gamma[r, s] = comp.Gamma[s, r] = np.uint8(g_rs)
            comp.Gamma[r, k] = comp.Gamma[k, r] = np.uint8(g_rk)
            comp.Gamma[s, k] = comp.Gamma[k, s] = np.uint8(g_sk)

            ket_before = formal_qfe_ket_le(
                n, comp.h_words, comp.A_words, comp.ell, comp.Gamma
            )
            exp = ShotExpansion.from_component(comp)
            add_column(exp, r, s)

            assert int(exp.Gamma[r, s]) == (g_rs ^ (ell_r & 1))
            assert int(exp.ell[0, s]) == ((ell_s + ell_r + 2 * g_rs) & 3)
            ket_exp = formal_qfe_ket_le(
                n, exp.h_words[0], exp.A_words, exp.ell[0], exp.Gamma
            )
            assert np.max(np.abs(ket_exp - ket_before)) < 1e-14


def test_h2_make_principal() -> None:
    rng = np.random.default_rng(103)
    n = 6
    for d in [1, 3, 5]:
        comp = random_valid_component(rng, n, d)
        j = d - 1
        # Pick non-pivot qubit or force bit p = 1 in column j and several other columns
        non_pivs = [q for q in range(n) if q not in set(comp.pivots[:j].tolist())]
        p = non_pivs[-1]
        comp.A_words[j] |= unit_bit_words(p, comp.n_words)
        for col_s in range(j):
            comp.A_words[col_s] |= unit_bit_words(p, comp.n_words)

        ket_before = formal_qfe_ket_le(
            n, comp.h_words, comp.A_words, comp.ell, comp.Gamma
        )
        exp = ShotExpansion.from_component(comp)
        make_principal(exp, j, p)

        exp.check_invariants()
        assert int(exp.pivots[j]) == p
        assert np.max(np.abs(exp.to_statevector() - ket_before)) < 1e-14

    comp_zero_bit = PhaseQFEComponent.zero(4)
    comp_zero_bit.A_words = np.zeros((1, 1), dtype=np.uint64)
    comp_zero_bit.ell = np.zeros(1, dtype=np.uint8)
    comp_zero_bit.Gamma = np.zeros((1, 1), dtype=np.uint8)
    comp_zero_bit.pivots = np.array([-1], dtype=np.int32)
    comp_zero_bit.d = 1
    with pytest.raises(ValueError):
        make_principal(ShotExpansion.from_component(comp_zero_bit), 0, 2)


def test_h3_restrict_amplitude_all_three_cases() -> None:
    rng = np.random.default_rng(104)
    n = 5
    for d in range(5):
        comp = random_valid_component(rng, n, d)
        lam0 = np.zeros(d, dtype=np.uint8)

        e1 = ShotExpansion.from_component(comp)
        restrict_amplitude(e1, lam0, 0)
        assert e1.chi == 1 and complex(e1.weights[0]) == 1.0 + 0.0j and e1.d == d

        e2 = ShotExpansion.from_component(comp)
        restrict_amplitude(e2, lam0, 1)
        assert e2.chi == 0 and e2.d == d

    for d in [1, 2, 4]:
        for weight_target in [w for w in (1, 2, 4) if w <= d]:
            for eps in (0, 1):
                for ell_val in range(4):
                    comp = random_valid_component(rng, n, d)
                    comp.ell[:] = rng.integers(0, 4, size=d, dtype=np.uint8)
                    lam = np.zeros(d, dtype=np.uint8)
                    chosen = rng.choice(d, size=weight_target, replace=False)
                    lam[chosen] = 1
                    comp.ell[int(np.min(chosen))] = np.uint8(ell_val)

                    before = comp.to_statevector()
                    # Oracle: restrict basis index x in coset where lam . ((x ^ h)[pivots]) == eps
                    h_bits = unpack_bit_vector(comp.h_words, n)
                    expected = np.zeros_like(before)
                    for x in range(1 << n):
                        if abs(before[x]) == 0.0:
                            continue
                        u = np.array(
                            [((x >> int(p)) & 1) ^ int(h_bits[int(p)]) for p in comp.pivots],
                            dtype=np.uint8,
                        )
                        if (int(lam @ u) & 1) == eps:
                            expected[x] = before[x]

                    exp = ShotExpansion.from_component(comp)
                    restrict_amplitude(exp, lam, eps)

                    assert exp.chi == 1 and abs(abs(complex(exp.weights[0])) - _INV_SQRT2) < 1e-14
                    assert exp.d == d - 1
                    exp.check_invariants()
                    assert np.max(np.abs(exp.to_statevector() - expected)) < 1e-14


def test_h4_eliminate_zero_amplitude_all_four_cases() -> None:
    rng = np.random.default_rng(105)
    n = 5
    for d0 in [0, 1, 2, 3]:
        for j in sorted({0, d0}):
            for c0 in range(4):
                lam_choices = [np.zeros(d0, dtype=np.uint8)]
                for w in [1, 2, 3]:
                    if w <= d0:
                        lv = np.zeros(d0, dtype=np.uint8)
                        lv[:w] = 1
                        rng.shuffle(lv)
                        lam_choices.append(lv)
                for lam in lam_choices:
                    base = random_valid_component(rng, n, d0)
                    # Insert a zero column at position j
                    A_new = np.insert(
                        base.A_words, j, np.zeros(base.n_words, dtype=np.uint64), axis=0
                    )
                    ell_new = np.insert(base.ell, j, np.uint8(c0))
                    G_new = np.insert(np.insert(base.Gamma, j, 0, axis=0), j, 0, axis=1)
                    rem_idx = [idx for idx in range(d0 + 1) if idx != j]
                    G_new[j, rem_idx] = lam
                    G_new[rem_idx, j] = lam
                    piv_new = np.insert(base.pivots, j, np.int32(-1))

                    comp = PhaseQFEComponent(
                        n_qubits=n,
                        d=d0 + 1,
                        h_words=base.h_words.copy(),
                        A_words=np.ascontiguousarray(A_new),
                        ell=np.ascontiguousarray(ell_new),
                        Gamma=np.ascontiguousarray(G_new),
                        pivots=np.ascontiguousarray(piv_new),
                    )
                    ket_before = formal_qfe_ket_le(
                        n, comp.h_words, comp.A_words, comp.ell, comp.Gamma
                    )
                    exp = ShotExpansion.from_component(comp)
                    eliminate_zero_amplitude(exp, j)
                    exp.check_invariants()

                    if c0 & 1:
                        assert exp.chi == 1 and abs(abs(exp.weights[0]) - 1.0) < 1e-14
                    elif np.any(lam):
                        assert exp.chi == 1 and abs(abs(exp.weights[0]) - 1.0) < 1e-14
                        assert exp.d == d0 - 1
                    elif c0 == 0:
                        assert exp.chi == 1 and abs(complex(exp.weights[0]) - _SQRT2) < 1e-14
                    else:
                        assert exp.chi == 0

                    assert np.max(np.abs(exp.to_statevector() - ket_before)) < 1e-14

    non_zero_col = ShotExpansion.from_component(random_valid_component(rng, 4, 2))
    with pytest.raises(ValueError):
        eliminate_zero_amplitude(non_zero_col, 0)


# ---------------------------------------------------------------------------
# Group B (Part 1): P1 and P2 Primitives
# ---------------------------------------------------------------------------


def test_p1_apply_pauli_hermitian_and_non_hermitian() -> None:
    rng = np.random.default_rng(106)
    for n in range(1, 6):
        for d in range(min(n + 1, 5)):
            comp = random_valid_component(rng, n, d)
            before = comp.to_statevector()
            for kap in range(4):
                for overlap_mode in (False, True):
                    a_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
                    if not overlap_mode:
                        a_un_bits = unpack_bit_vector(comp.a_union, n)
                        b_bits = rng.integers(0, 2, size=n, dtype=np.uint8) & (
                            1 - a_un_bits
                        )
                    else:
                        b_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
                    A_orig = comp.A_words.copy()
                    G_orig = comp.Gamma.copy()
                    piv_orig = comp.pivots.copy()
                    ell_par_orig = (comp.ell & 1).copy()

                    exp_unit = apply_pauli(ShotExpansion.from_component(comp), a_bits, b_bits, kap)
                    exp_in = ShotExpansion.from_component(comp, weight=0.6 + 0.8j)
                    exp_out = apply_pauli(exp_in, a_bits, b_bits, kap)

                    g = complex(exp_unit.weights[0])
                    assert any(abs(g - u) < 1e-15 for u in (1, 1j, -1, -1j))
                    assert np.array_equal(exp_unit.A_words, A_orig)
                    assert np.array_equal(exp_unit.Gamma, G_orig)
                    assert np.array_equal(exp_unit.pivots, piv_orig)
                    assert np.array_equal(exp_unit.ell[0] & 1, ell_par_orig)
                    expected = dense_pauli_le(n, a_bits, b_bits, kap) @ before
                    assert np.max(np.abs(exp_unit.to_statevector() - expected)) < 1e-14
                    assert (
                        np.max(
                            np.abs(exp_out.to_statevector() - (0.6 + 0.8j) * expected)
                        )
                        < 1e-14
                    )


def test_p2_apply_quarter_turn_all_three_cases_and_errors() -> None:
    # Non-Hermitian axes must raise ValueError
    e0 = ShotExpansion.zero(2)
    with pytest.raises(ValueError):
        apply_quarter_turn(e0, [1, 0], [0, 0], 1)
    with pytest.raises(ValueError):
        apply_quarter_turn(e0, [1, 0], [1, 0], 0)

    # Case 1: d -> d + 1 (R(X0) on |0>, then R(X0 X1))
    e1 = apply_quarter_turn(e0, [1, 0], [0, 0], 0)
    assert e1.d == 1 and abs(e1.norm_sq() - 1.0) < 1e-14
    assert (
        np.max(
            np.abs(
                e1.to_statevector()
                - dense_quarter_turn_le(2, np.array([1, 0]), np.array([0, 0]), 0)
                @ e0.to_statevector()
            )
        )
        < 1e-14
    )
    e2 = apply_quarter_turn(e1, [1, 1], [0, 0], 0)
    assert e2.d == 2 and abs(e2.norm_sq() - 1.0) < 1e-14

    # Case 2: dependent column + odd c0 (R(Z0) on |0>, R(Z0 Z1) on e2, d' == d)
    ez = apply_quarter_turn(e0, [0, 0], [1, 0], 0)
    assert ez.d == 0 and abs(ez.weights[0] - _OMEGA) < 1e-14
    ezz = apply_quarter_turn(e2, [0, 0], [1, 1], 0)
    assert ezz.d == 2 and abs(ezz.norm_sq() - 1.0) < 1e-14

    # Case 3: dependent column + even c0, lam != 0 (R(-X0)R(+X0)|0> = |0>, R(+X0)^2|0> = i|1>, d: 1 -> 0)
    e_inv = apply_quarter_turn(e1, [1, 0], [0, 0], 2)
    assert e_inv.d == 0
    assert np.max(np.abs(e_inv.to_statevector() - e0.to_statevector())) < 1e-14

    e_sq = apply_quarter_turn(e1, [1, 0], [0, 0], 0)
    assert e_sq.d == 0
    expected_sq = np.array([0.0, 1.0j, 0.0, 0.0], dtype=np.complex128)
    assert np.max(np.abs(e_sq.to_statevector() - expected_sq)) < 1e-14

    # 60 random Hermitian axes on random d in [0..5] states + ShotExpansion rank bound
    rng = np.random.default_rng(107)
    n = 5
    for _ in range(60):
        d = int(rng.integers(0, 6))
        comp = random_valid_component(rng, n, d)
        a_b, b_b, kap = random_hermitian_pauli_bits(rng, n)
        before = comp.to_statevector()
        after = apply_quarter_turn(ShotExpansion.from_component(comp), a_b, b_b, kap)
        after.check_invariants(require_canonical_offset=True)
        assert abs(after.norm_sq() - 1.0) < 1e-12
        expected = dense_quarter_turn_le(n, a_b, b_b, kap) @ before
        assert np.max(np.abs(after.to_statevector() - expected)) < 1e-13

        exp_in = ShotExpansion.from_component(comp)
        exp_in = apply_pauli_t(exp_in, [0, 0, 0, 0, 0], [1, 0, 1, 0, 0], 0)
        chi_before = exp_in.chi
        psi_exp_before = exp_in.to_statevector()
        exp_out = apply_quarter_turn(exp_in, a_b, b_b, kap)
        exp_out.check_invariants(require_canonical_offset=True)
        assert exp_out.chi <= chi_before
        assert (
            np.max(
                np.abs(
                    exp_out.to_statevector()
                    - dense_quarter_turn_le(n, a_b, b_b, kap) @ psi_exp_before
                )
            )
            < 1e-13
        )


# ---------------------------------------------------------------------------
# Group B (Part 2): K1, N1-N3, M1-M2, O2, O1 Primitives (V1.1)
# ---------------------------------------------------------------------------


def test_k1_canonical_offset_and_interference() -> None:
    rng = np.random.default_rng(108)
    for _ in range(30):
        comp = random_valid_component(rng, 5, 3)
        comp.ell = np.array([1, 3, 1], dtype=np.uint8)
        comp.Gamma = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
        comp.h_words[0] = np.uint64(0b11111)
        before = comp.to_statevector()
        exp = merge_frames(ShotExpansion.from_components([(1.0, comp)], merge=False))
        exp.check_invariants(require_canonical_offset=True)
        g = complex(exp.weights[0])
        assert any(abs(g - u) < 1e-15 for u in (1, 1j, -1, -1j))
        assert np.max(np.abs(exp.to_statevector() - before)) < 1e-14

    # Constructive interference on identical cosets represented with shifted h
    c_a_raw = random_valid_component(rng, 4, 2)
    c_a = merge_frames(ShotExpansion.from_components([(1.0, c_a_raw)], merge=False)).to_components()[0][1]
    c_b = c_a.copy()
    # Shift c_b by A_words[0] and compensate phase so |c_b> == |c_a> up to canonical offset
    c_b.h_words ^= c_b.A_words[0]
    g_shift = complex(merge_frames(ShotExpansion.from_components([(1.0, c_b.copy())], merge=False)).weights[0])
    exp_con = ShotExpansion.from_components(
        [(0.6, c_a), (0.4j * g_shift.conjugate(), c_b)], merge=True
    )
    assert exp_con.chi == 1
    assert np.max(np.abs(exp_con.to_statevector() - (0.6 + 0.4j) * c_a.to_statevector())) < 1e-14

    # Exact and near-exact destructive interference -> chi == 0, nu == 0.0
    for w_neg in (-0.7, -1.0 + 1e-15j):
        w_pos = 0.7 if w_neg == -0.7 else 1.0
        exp_des = ShotExpansion.from_components(
            [(w_pos, c_a.copy()), (w_neg, c_a.copy())], merge=True
        )
        assert exp_des.chi == 0
        assert exp_des.norm_sq() == 0.0

    # Scale-covariant small-weight retention (1e-14 retained because 1e-13 * min(1, 1e-14) = 1e-27)
    exp_tiny = ShotExpansion.from_components([(1e-14 + 0.0j, c_a.copy())], merge=True)
    assert exp_tiny.chi == 1
    assert abs(exp_tiny.norm_sq() - 1e-28) < 1e-40


def test_k1_common_geometry_norm_sq() -> None:
    n = 4
    exp = apply_quarter_turn(ShotExpansion.zero(n), [1, 1, 0, 0], [0, 0, 0, 0], 0)
    for q in range(4):
        z_b = np.zeros(n, dtype=np.uint8)
        z_b[q] = 1
        x_b = np.zeros(n, dtype=np.uint8)
        x_b[(q + 1) % n] = 1
        exp = apply_pauli_t(exp, x_b, z_b, int(x_b @ z_b) & 1)
    assert exp.chi >= 8
    psi_common = exp.to_statevector()
    assert abs(exp.norm_sq() - float(np.vdot(psi_common, psi_common).real)) < 1e-14


def test_n1_n2_n3_nonclifford_gates_and_c2_budgets() -> None:
    n = 4
    # Build incoming d=2, chi=4 state
    exp = ShotExpansion.zero(n)
    exp = apply_quarter_turn(exp, [1, 1, 0, 0], [0, 0, 0, 0], 0)
    exp = apply_quarter_turn(exp, [0, 0, 1, 1], [0, 0, 0, 0], 0)
    exp = apply_pauli_t(exp, [0, 0, 0, 0], [1, 0, 0, 0], 0)  # type: ignore[assignment]
    exp = apply_pauli_t(exp, [0, 0, 0, 0], [0, 0, 1, 0], 0)  # type: ignore[assignment]
    assert exp.d == 2 and exp.chi == 4
    psi0 = exp.to_statevector()

    # N1: T and T_DAG on multi-qubit Hermitian axis, T^8 = I, T_DAG @ T = I
    ax_a = np.array([1, 0, 1, 0], dtype=np.uint8)
    ax_b = np.array([0, 1, 1, 0], dtype=np.uint8)
    ax_k = 3  # <a, b> = 1 -> Hermitian
    exp_t = apply_pauli_t(exp, ax_a, ax_b, ax_k, dag=False)
    assert isinstance(exp_t, ShotExpansion)
    assert (
        np.max(np.abs(exp_t.to_statevector() - dense_pauli_t_le(n, ax_a, ax_b, ax_k, 1) @ psi0))
        < 1e-13
    )
    exp_rt = apply_pauli_t(exp_t, ax_a, ax_b, ax_k, dag=True)
    assert np.max(np.abs(exp_rt.to_statevector() - psi0)) < 1e-13  # type: ignore[union-attr]

    cur8 = exp.copy()
    for _ in range(8):
        cur8 = apply_pauli_t(cur8, ax_a, ax_b, ax_k)  # type: ignore[assignment]
    assert np.max(np.abs(cur8.to_statevector() - psi0)) < 1e-12

    # N2: CCZ on standard Z axes & commuting multi-qubit axes (X0 X1, Z0 Z1, -Y2 X3), CCZ^2 = I
    comm_axes_bits = [
        (np.array([1, 1, 0, 0], np.uint8), np.array([0, 0, 0, 0], np.uint8), 0),
        (np.array([0, 0, 0, 0], np.uint8), np.array([1, 1, 0, 0], np.uint8), 0),
        (np.array([0, 0, 1, 1], np.uint8), np.array([0, 0, 1, 0], np.uint8), 3),
    ]
    exp_ccz = apply_ccz(exp, comm_axes_bits)
    assert isinstance(exp_ccz, ShotExpansion)
    assert (
        np.max(np.abs(exp_ccz.to_statevector() - dense_ccz_axes_le(n, comm_axes_bits) @ psi0))
        < 1e-13
    )
    exp_ccz2 = apply_ccz(exp_ccz, comm_axes_bits)
    assert np.max(np.abs(exp_ccz2.to_statevector() - psi0)) < 1e-13  # type: ignore[union-attr]

    # Non-commuting CCZ axes raise ValueError
    with pytest.raises(ValueError):
        apply_ccz(exp, [([1, 0, 0, 0], [0, 0, 0, 0], 0), ([0, 0, 0, 0], [1, 0, 0, 0], 0), ([0, 0, 0, 0], [0, 1, 0, 0], 0)])

    # N3: CompiledPauliBlock sequential vs pauli_sum (equivalent to budgets 1, 4, 64)
    p1_ax = (coerce_pauli_words([0, 0, 0, 0], n), coerce_pauli_words([1, 0, 0, 0], n), 0)
    p2_ax = (coerce_pauli_words([0, 0, 0, 0], n), coerce_pauli_words([0, 1, 0, 0], n), 0)
    seq_blk = CompiledPauliBlock(
        n_qubits=n,
        mode="sequential",
        steps=(
            CompiledPauliBlockStep(kind="N1", paulis=(p1_ax,), sign=1),
            CompiledPauliBlockStep(kind="N1", paulis=(p2_ax,), sign=-1),
        ),
        is_unitary=True,
    )
    # Expanded 4-term Pauli sum of G_{+1}(Z0) * G_{-1}(Z1)
    c_t0, c_t1 = pq._T_C0, pq._T_C1
    c_d0, c_d1 = pq._TDAG_C0, pq._TDAG_C1
    a12, b12, k12 = pauli_multiply(*p1_ax, *p2_ax)
    sum_blk = CompiledPauliBlock(
        n_qubits=n,
        mode="pauli_sum",
        coeffs=np.array([c_t0 * c_d0, c_t1 * c_d0, c_t0 * c_d1, c_t1 * c_d1], dtype=np.complex128),
        a_words=np.vstack([np.zeros(1, np.uint64), p1_ax[0], p2_ax[0], a12]),
        b_words=np.vstack([np.zeros(1, np.uint64), p1_ax[1], p2_ax[1], b12]),
        kappas=np.array([0, p1_ax[2], p2_ax[2], k12], dtype=np.uint8),
        is_unitary=True,
    )
    split_blk = CompiledPauliBlock(
        n_qubits=n,
        mode="sequential",
        steps=(
            CompiledPauliBlock(
                n_qubits=n,
                mode="pauli_sum",
                coeffs=np.array([c_t0, c_t1], dtype=np.complex128),
                a_words=np.vstack([np.zeros(1, np.uint64), p1_ax[0]]),
                b_words=np.vstack([np.zeros(1, np.uint64), p1_ax[1]]),
                kappas=np.array([0, p1_ax[2]], dtype=np.uint8),
            ),
            CompiledPauliBlock(
                n_qubits=n,
                mode="pauli_sum",
                coeffs=np.array([c_d0, c_d1], dtype=np.complex128),
                a_words=np.vstack([np.zeros(1, np.uint64), p2_ax[0]]),
                b_words=np.vstack([np.zeros(1, np.uint64), p2_ax[1]]),
                kappas=np.array([0, p2_ax[2]], dtype=np.uint8),
            ),
        ),
        is_unitary=True,
    )
    dense_ref = (
        dense_pauli_t_le(n, np.zeros(n, np.uint8), np.array([0, 1, 0, 0], np.uint8), 0, -1)
        @ dense_pauli_t_le(n, np.zeros(n, np.uint8), np.array([1, 0, 0, 0], np.uint8), 0, 1)
        @ psi0
    )
    sv_b1 = apply_compiled_block(exp, seq_blk).to_statevector()  # type: ignore[union-attr]
    sv_b4 = apply_compiled_block(exp, split_blk).to_statevector()  # type: ignore[union-attr]
    sv_b64 = apply_compiled_block(exp, sum_blk).to_statevector()  # type: ignore[union-attr]
    assert np.max(np.abs(sv_b1 - dense_ref)) < 1e-13
    assert np.max(np.abs(sv_b4 - dense_ref)) < 1e-13
    assert np.max(np.abs(sv_b64 - dense_ref)) < 1e-13


def test_m1_m2_born_postselection_readout_flip_and_reset() -> None:
    # H T |+> = H T R(-Y0)|0> ? Wait: |+> = R(+Y0)|0>, T|+> = apply_pauli_t(Z0), H(T|+>):
    # Measuring X0 on T|+> directly equals measuring Z0 on H T|+>!
    # Even better: test BOTH diagonal Z0 on H T|+> and non-diagonal X0 on T|+>:
    # T|+> = (|0> + e^{i*pi/4}|1>)/sqrt(2); <+|T|+> = (1 + e^{i*pi/4})/2 -> |<+|T|+>|^2 = (2 + sqrt(2))/4 = cos^2(pi/8)!
    exp_plus = apply_quarter_turn(ShotExpansion.zero(1), [1], [1], 1)  # R(+Y0)|0> = (|0> - |1>)/sqrt(2)? Check: R(-Y0)|0> = |+>!
    exp_plus = apply_quarter_turn(ShotExpansion.zero(1), [1], [1], 3)  # R(-Y0)|0> = (|0> + |1>)/sqrt(2) = |+>
    exp_magic = apply_pauli_t(exp_plus, [0], [1], 0)
    assert isinstance(exp_magic, ShotExpansion)
    psi_m = exp_magic.to_statevector()
    assert np.max(np.abs(psi_m - np.array([1.0, _OMEGA]) * _INV_SQRT2)) < 1e-14

    p0_exact = math.cos(math.pi / 8.0) ** 2
    p1_exact = math.sin(math.pi / 8.0) ** 2

    res_x0 = measure_expansion(exp_magic, [1], [0], 0, requested_bit=0)
    assert abs(res_x0.z0 - p0_exact) < 1e-14
    assert abs(res_x0.z1 - p1_exact) < 1e-14
    assert abs(res_x0.z0 + res_x0.z1 - 1.0) < 1e-14

    # M1 sigma in {0, 1} unnormalized kets on 2-qubit superposition
    exp2 = apply_quarter_turn(ShotExpansion.zero(2), [1, 1], [0, 0], 0)
    psi2 = exp2.to_statevector()
    P_diag = dense_pauli_le(2, np.array([0, 0]), np.array([1, 0]), 2)  # -Z0 (sigma=1)
    for m in (0, 1):
        proj_exp, z_m = project_diagonal(exp2, [1, 0], 1, m)
        expected_unnorm = 0.5 * (np.eye(4) + ((-1) ** m) * P_diag) @ psi2
        assert np.max(np.abs(proj_exp.to_statevector() - expected_unnorm)) < 1e-14  # type: ignore[union-attr]
        assert abs(z_m - float(np.vdot(expected_unnorm, expected_unnorm).real)) < 1e-14

    # 4,000 Born samples on X0 of T|+>
    rng = np.random.default_rng(2026)
    ones = sum(
        measure_expansion(exp_magic, [1], [0], 0, rng=rng).outcome for _ in range(4000)
    )
    p1_hat = ones / 4000.0
    assert abs(p1_hat - p1_exact) < 0.025

    # Impossible postselection raises ImpossiblePostselectionError (subclass of ValueError)
    assert issubclass(ImpossiblePostselectionError, ValueError)
    with pytest.raises(ImpossiblePostselectionError):
        measure_expansion(ShotExpansion.zero(1), [0], [1], 0, requested_bit=1)

    # p_flip=1.0 and invert_result=True report flipped classical bit while keeping |0>
    r_flip = measure_expansion(ShotExpansion.zero(1), [0], [1], 0, p_flip=1.0, rng=rng)
    assert r_flip.outcome == 0 and r_flip.reported_outcome == 1 and r_flip.readout_flipped
    assert np.max(np.abs(r_flip.post_state.to_statevector() - np.array([1.0, 0.0]))) < 1e-15  # type: ignore[union-attr]

    r_inv = measure_expansion(ShotExpansion.zero(1), [0], [1], 0, invert_result=True)
    assert r_inv.outcome == 0 and r_inv.reported_outcome == 1 and not r_inv.readout_flipped
    assert np.max(np.abs(r_inv.post_state.to_statevector() - np.array([1.0, 0.0]))) < 1e-15  # type: ignore[union-attr]

    # MR(1.0) on |1>: true outcome m=1 resets quantum state via X^m to |0>, while reported_outcome == 0!
    ket1 = apply_pauli(ShotExpansion.zero(1), [1], [0], 0)
    r_mr = measure_expansion(ket1, [0], [1], 0, p_flip=1.0, rng=rng)
    assert r_mr.outcome == 1 and r_mr.reported_outcome == 0
    post_reset = (
        apply_pauli(r_mr.post_state, [1], [0], 0)
        if r_mr.outcome == 1
        else r_mr.post_state
    )
    assert np.max(np.abs(post_reset.to_statevector() - np.array([1.0, 0.0]))) < 1e-15  # type: ignore[union-attr]


def test_m2_deep_circuit_underflow_prevention() -> None:
    rng = np.random.default_rng(112)
    exp = ShotExpansion.zero(1)
    # Prepare |+> first so step 0 (measure Z) is also 50/50, then alternate X <-> Z for 1,200 steps!
    exp = apply_quarter_turn(exp, [1], [1], 3)  # |+>
    for step in range(1200):
        if step % 2 == 0:
            res = measure_expansion(exp, [0], [1], 0, rng=rng, renormalize=True)
        else:
            res = measure_expansion(exp, [1], [0], 0, rng=rng, renormalize=True)
        exp = res.post_state  # type: ignore[assignment]
        assert abs(exp.norm_sq() - 1.0) < 1e-12
    assert np.all(np.isfinite(exp.weights))
    assert abs(exp.log_mass - (-1200.0 * math.log(2.0))) < 1e-10
    assert exp.branch_mass() == 0.0  # 2^-1200 underflows float64 cleanly without error


# ---------------------------------------------------------------------------
# Group C: V1.2a Moving-Reference Trajectories, Multi-Word & Tolerance Guards
# ---------------------------------------------------------------------------


def _pullback_pauli_via_ref_chp(
    ref_chp: ReferenceCHPTableau,
    a_bits: np.ndarray,
    b_bits: np.ndarray,
    kappa: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute `B_t^dagger P(a, b, kappa) B_t` from `ReferenceCHPTableau`."""
    nw = ref_chp.n_words
    aw = np.zeros(nw, dtype=np.uint64)
    bw = np.zeros(nw, dtype=np.uint64)
    kap = int(kappa) & 3
    for q in np.flatnonzero(a_bits):
        aw, bw, kap = pauli_multiply(
            aw, bw, kap, ref_chp.x2x[q], ref_chp.x2z[q], int(ref_chp.x_kap[q])
        )
    for q in np.flatnonzero(b_bits):
        aw, bw, kap = pauli_multiply(
            aw, bw, kap, ref_chp.z2x[q], ref_chp.z2z[q], int(ref_chp.z_kap[q])
        )
    return aw, bw, kap


def _exact_stim_gate_c128_le(
    gate_name: str, targets: tuple[int, ...], n_qubits: int
) -> np.ndarray:
    """Embed a 1Q or 2Q Stim Clifford gate in `complex128` little-endian without float32 error."""
    u_raw = stim.gate_data(gate_name).unitary_matrix.astype(np.complex128)
    # Snap float32 cyclotomic entries {0, +-1, +-1/2, +-1/sqrt(2)} to exact float64!
    grid = np.array(
        [0.0, 1.0, -1.0, 0.5, -0.5, _INV_SQRT2, -_INV_SQRT2], dtype=np.float64
    )
    re = grid[np.argmin(np.abs(u_raw.real[..., None] - grid), axis=-1)]
    im = grid[np.argmin(np.abs(u_raw.imag[..., None] - grid), axis=-1)]
    u_local = (re + 1j * im).astype(np.complex128)

    dim = 1 << n_qubits
    out = np.zeros((dim, dim), dtype=np.complex128)
    k = len(targets)
    clear_mask = ~sum(1 << q for q in targets)
    for x in range(dim):
        in_loc = sum(((x >> targets[i]) & 1) << i for i in range(k))
        base = x & clear_mask
        for out_loc in range(1 << k):
            val = u_local[out_loc, in_loc]
            if val != 0.0j:
                y = base | sum(((out_loc >> i) & 1) << targets[i] for i in range(k))
                out[y, x] = val
    return out


def test_v1_2a_moving_reference_random_trajectories() -> None:
    rng = np.random.default_rng(20261029)
    gates_1q = ("H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "C_XYZ", "C_ZYX")
    gates_2q = ("CX", "CZ", "CY", "XCY", "SWAP", "ISWAP", "SQRT_XX", "SQRT_ZZ_DAG")

    for trial in range(100):
        n = int(rng.integers(1, 6))
        ref_chp = ReferenceCHPTableau(n)
        B_mat = np.eye(1 << n, dtype=np.complex128)
        psi_dense = np.zeros(1 << n, dtype=np.complex128)
        psi_dense[0] = 1.0 + 0.0j
        exp = ShotExpansion.zero(n)

        for _ in range(36):
            choices = ["C1", "P1", "P2", "T", "TD", "M", "R"]
            probs = [0.18, 0.12, 0.14, 0.16, 0.10, 0.18, 0.12]
            if n >= 2:
                choices.append("C2")
                probs = [p * 0.88 for p in probs] + [0.12]
            if n >= 3:
                choices.append("CCZ")
                probs = [p * 0.90 for p in probs] + [0.10]
            probs_arr = np.array(probs, dtype=np.float64)
            probs_arr /= probs_arr.sum()
            kind = str(rng.choice(choices, p=probs_arr))

            if kind == "C1":
                g_name = str(rng.choice(gates_1q))
                q = int(rng.integers(0, n))
                ref_chp.do_unitary(g_name, [q])
                U = _exact_stim_gate_c128_le(g_name, (q,), n)
                B_mat = U @ B_mat
                psi_dense = U @ psi_dense
            elif kind == "C2":
                g_name = str(rng.choice(gates_2q))
                q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
                ref_chp.do_unitary(g_name, [q0, q1])
                U = _exact_stim_gate_c128_le(g_name, (q0, q1), n)
                B_mat = U @ B_mat
                psi_dense = U @ psi_dense
            elif kind == "P1":
                a_b, b_b, kap = random_hermitian_pauli_bits(rng, n)
                aw, bw, k_pb = _pullback_pauli_via_ref_chp(ref_chp, a_b, b_b, kap)
                exp = apply_pauli(exp, aw, bw, k_pb)
                psi_dense = dense_pauli_le(n, a_b, b_b, kap) @ psi_dense
            elif kind == "P2":
                a_b, b_b, kap = random_hermitian_pauli_bits(rng, n)
                aw, bw, k_pb = _pullback_pauli_via_ref_chp(ref_chp, a_b, b_b, kap)
                chi_before = exp.chi
                exp = apply_quarter_turn(exp, aw, bw, k_pb)
                assert exp.chi <= chi_before
                psi_dense = dense_quarter_turn_le(n, a_b, b_b, kap) @ psi_dense
            elif kind in ("T", "TD"):
                q = int(rng.integers(0, n))
                z_b = np.zeros(n, dtype=np.uint8)
                z_b[q] = 1
                aw, bw, k_pb = _pullback_pauli_via_ref_chp(
                    ref_chp, np.zeros(n, dtype=np.uint8), z_b, 0
                )
                sgn = 1 if kind == "T" else -1
                exp = apply_pauli_t(exp, aw, bw, k_pb, sign=sgn)  # type: ignore[assignment]
                psi_dense = dense_pauli_t_le(
                    n, np.zeros(n, dtype=np.uint8), z_b, 0, sgn
                ) @ psi_dense
            elif kind == "CCZ":
                qs = [int(x) for x in rng.choice(n, size=3, replace=False)]
                pb_axes = []
                phys_axes = []
                for q in qs:
                    z_b = np.zeros(n, dtype=np.uint8)
                    z_b[q] = 1
                    phys_axes.append((np.zeros(n, dtype=np.uint8), z_b, 0))
                    pb_axes.append(
                        _pullback_pauli_via_ref_chp(
                            ref_chp, np.zeros(n, dtype=np.uint8), z_b, 0
                        )
                    )
                exp = apply_ccz(exp, pb_axes)  # type: ignore[assignment]
                psi_dense = dense_ccz_axes_le(n, phys_axes) @ psi_dense
            elif kind == "M":
                # General multi-qubit or single-qubit Hermitian Pauli measurement!
                a_b, b_b, kap = random_hermitian_pauli_bits(rng, n, force_nonzero_a=False)
                if not np.any(a_b) and not np.any(b_b):
                    b_b[0] = 1
                    kap = 0
                aw, bw, k_pb = _pullback_pauli_via_ref_chp(ref_chp, a_b, b_b, kap)
                P_phys = dense_pauli_le(n, a_b, b_b, kap)
                I_mat = np.eye(1 << n, dtype=np.complex128)
                v0 = 0.5 * (I_mat + P_phys) @ psi_dense
                v1 = 0.5 * (I_mat - P_phys) @ psi_dense
                z0_dense = float(np.vdot(v0, v0).real)
                z1_dense = float(np.vdot(v1, v1).real)

                # Verify BOTH branches unnormalized before sampling!
                for m_branch, v_branch, z_branch in ((0, v0, z0_dense), (1, v1, z1_dense)):
                    if z_branch > 1e-12:
                        rb = measure_expansion(
                            exp, aw, bw, k_pb, requested_bit=m_branch, renormalize=False
                        )
                        psi_b = B_mat @ rb.post_state.to_statevector()  # type: ignore[union-attr]
                        assert np.max(np.abs(psi_b - v_branch)) < 1e-10
                        assert abs(rb.chosen_mass - z_branch) < 1e-10
                        assert abs((rb.z0 + rb.z1) - exp.norm_sq()) < 1e-10

                res = measure_expansion(exp, aw, bw, k_pb, rng=rng, renormalize=True)
                exp = res.post_state  # type: ignore[assignment]
                v_chosen = v1 if res.outcome == 1 else v0
                psi_dense = v_chosen / math.sqrt(float(np.vdot(v_chosen, v_chosen).real))
            elif kind == "R":
                # Reset qubit q in Z basis via C1.5 ref_chp.measure_z(q, is_reset=True)
                q = int(rng.integers(0, n))
                meta = ref_chp.measure_z(q, is_reset=True)
                if meta.is_random:
                    exp = apply_quarter_turn(
                        exp, meta.a_words, meta.b_rot_words, meta.kappa_rot
                    )
                    # Update B_mat <- B_mat @ R(-P_rot)
                    a_rot_u = unpack_bit_vector(meta.a_words, n)
                    b_rot_u = unpack_bit_vector(meta.b_rot_words, n)
                    B_mat = B_mat @ dense_quarter_turn_le(
                        n, a_rot_u, b_rot_u, meta.kappa_rot, sign=-1
                    )
                    res_r = measure_expansion(
                        exp, meta.e_p_words, sigma=0, rng=rng, renormalize=True
                    )
                else:
                    res_r = measure_expansion(
                        exp, meta.beta_Z_words, sigma=meta.m_ref, rng=rng, renormalize=True
                    )
                exp = res_r.post_state  # type: ignore[assignment]
                # Apply physical X_q if true outcome == 1 XOR reset_flip_ref!
                # Wait: in reference_chp.measure_z(is_reset=True), if m_ref == 1 it flipped z_kap[q]
                # (which corresponds to B_t <- X_q B_t!), so apply X_q to B_mat when reset_flip_ref,
                # and apply pullback(X_q) to exp iff (res_r.outcome ^ int(meta.reset_flip_ref)) == 1!
                x_b = np.zeros(n, dtype=np.uint8)
                x_b[q] = 1
                X_q_mat = dense_pauli_le(n, x_b, np.zeros(n, dtype=np.uint8), 0)
                if meta.reset_flip_ref:
                    B_mat = X_q_mat @ B_mat
                if (res_r.outcome ^ int(meta.reset_flip_ref)) == 1:
                    aw_x, bw_x, k_x = _pullback_pauli_via_ref_chp(
                        ref_chp, x_b, np.zeros(n, dtype=np.uint8), 0
                    )
                    exp = apply_pauli(exp, aw_x, bw_x, k_x)

                # Dense oracle reset of qubit q to |0>:
                z_b = np.zeros(n, dtype=np.uint8)
                z_b[q] = 1
                Z_q_mat = dense_pauli_le(n, np.zeros(n, dtype=np.uint8), z_b, 0)
                I_mat = np.eye(1 << n, dtype=np.complex128)
                v_r = 0.5 * (I_mat + ((-1) ** res_r.outcome) * Z_q_mat) @ psi_dense
                if res_r.outcome == 1:
                    v_r = X_q_mat @ v_r
                psi_dense = v_r / math.sqrt(float(np.vdot(v_r, v_r).real))

            exp.check_invariants(
                require_canonical_offset=(kind in ("P2", "T", "TD", "CCZ", "M"))
            )
            psi_qfe = B_mat @ exp.to_statevector()
            assert np.max(np.abs(psi_qfe - psi_dense)) < 1e-10
            assert abs(exp.norm_sq() - 1.0) < 1e-10


@pytest.mark.parametrize("n_big", [63, 64, 65, 70, 128, 130])
def test_multiword_primitives_remap_invariance(n_big: int) -> None:
    rng = np.random.default_rng(5000 + n_big)
    n_small = 5
    # Two embeddings: monotone straddling 64-bit boundaries, and reversed/scrambled across words!
    if n_big == 63:
        maps = [[0, 1, 31, 61, 62], [62, 31, 61, 0, 1]]
    elif n_big == 64:
        maps = [[0, 1, 32, 62, 63], [63, 0, 62, 1, 32]]
    elif n_big == 65:
        maps = [[0, 1, 62, 63, 64], [64, 63, 62, 1, 0]]
    elif n_big == 70:
        maps = [[0, 62, 63, 64, 69], [69, 64, 63, 62, 0]]
    elif n_big == 128:
        maps = [[0, 1, 63, 64, 127], [127, 64, 63, 1, 0]]
    else:
        maps = [[0, 63, 64, 127, n_big - 1], [n_big - 1, 64, 127, 63, 0]]
    for qm in maps:
        assert len(set(qm)) == n_small

    def embed_bits(small_bits: np.ndarray, qmap: list[int]) -> np.ndarray:
        aw = np.zeros(n_words_for_qubits(n_big), dtype=np.uint64)
        for s_idx, b_val in enumerate(small_bits):
            if b_val:
                q = qmap[s_idx]
                aw[q >> 6] |= np.uint64(1) << np.uint64(q & 63)
        return aw

    def restrict_to_qmap(e_big: ShotExpansion, qmap: list[int]) -> ShotExpansion:
        qm_arr = np.asarray(qmap, dtype=np.int32)
        loc = {q: j for j, q in enumerate(qmap)}
        shifts = (np.uint64(1) << np.arange(n_small, dtype=np.uint64))[None, :]
        a_b = get_bits_at_qubits(e_big.A_words, qm_arr) if e_big.d > 0 else np.zeros((0, n_small), dtype=np.uint8)
        h_b = get_bits_at_qubits(e_big.h_words, qm_arr) if e_big.chi > 0 else np.zeros((0, n_small), dtype=np.uint8)
        a_s = np.sum(a_b.astype(np.uint64) * shifts, axis=1, keepdims=True, dtype=np.uint64)
        h_s = np.sum(h_b.astype(np.uint64) * shifts, axis=1, keepdims=True, dtype=np.uint64)
        piv_s = np.array([loc[int(p)] for p in e_big.pivots], dtype=np.int32)
        au_s = np.array([np.bitwise_or.reduce(a_s[:, 0])] if e_big.d > 0 else [0], dtype=np.uint64)
        return ShotExpansion(
            n_small, e_big.d, a_s, au_s, e_big.Gamma, piv_s,
            e_big.weights, h_s, e_big.ell, e_big.ell_odd,
            e_big.log_mass, e_big.atol, e_big._is_canonical_merged,
        )

    for qmap in maps:
        exp_small = ShotExpansion.zero(n_small)
        exp_big = ShotExpansion.zero(n_big)
        checkpoints_small: list[ShotExpansion] = []
        checkpoints_big: list[ShotExpansion] = []

        for step in range(30):
            op = step % 5
            if op == 0:
                a_s, b_s, k_s = random_hermitian_pauli_bits(rng, n_small)
                exp_small = apply_pauli(exp_small, a_s, b_s, k_s)
                exp_big = apply_pauli(
                    exp_big, embed_bits(a_s, qmap), embed_bits(b_s, qmap), k_s
                )
            elif op == 1:
                a_s, b_s, k_s = random_hermitian_pauli_bits(rng, n_small)
                exp_small = apply_quarter_turn(exp_small, a_s, b_s, k_s)
                exp_big = apply_quarter_turn(
                    exp_big, embed_bits(a_s, qmap), embed_bits(b_s, qmap), k_s
                )
            elif op == 2:
                a_s, b_s, k_s = random_hermitian_pauli_bits(rng, n_small)
                sgn = 1 if rng.random() < 0.5 else -1
                exp_small = apply_pauli_t(exp_small, a_s, b_s, k_s, sign=sgn)  # type: ignore[assignment]
                exp_big = apply_pauli_t(
                    exp_big,
                    embed_bits(a_s, qmap),
                    embed_bits(b_s, qmap),
                    k_s,
                    sign=sgn,
                )  # type: ignore[assignment]
            elif op == 3:
                qs = [int(x) for x in rng.choice(n_small, size=3, replace=False)]
                axes_s = []
                axes_b = []
                for q in qs:
                    zb = np.zeros(n_small, dtype=np.uint8)
                    zb[q] = 1
                    axes_s.append((np.zeros(n_small, dtype=np.uint8), zb, 0))
                    axes_b.append(
                        (
                            np.zeros(n_words_for_qubits(n_big), dtype=np.uint64),
                            embed_bits(zb, qmap),
                            0,
                        )
                    )
                exp_small = apply_ccz(exp_small, axes_s)  # type: ignore[assignment]
                exp_big = apply_ccz(exp_big, axes_b)  # type: ignore[assignment]
            else:
                a_s, b_s, k_s = random_hermitian_pauli_bits(rng, n_small, force_nonzero_a=True)
                r_probe = measure_expansion(exp_small, a_s, b_s, k_s, rng=rng)
                r_big = measure_expansion(
                    exp_big,
                    embed_bits(a_s, qmap),
                    embed_bits(b_s, qmap),
                    k_s,
                    requested_bit=r_probe.outcome,
                )
                assert abs(r_big.z0 - r_probe.z0) < 1e-10
                assert abs(r_big.z1 - r_probe.z1) < 1e-10
                exp_small = r_probe.post_state  # type: ignore[assignment]
                exp_big = r_big.post_state  # type: ignore[assignment]

            exp_big.check_invariants(require_canonical_offset=(op != 0))
            assert abs(exp_big.norm_sq() - exp_small.norm_sq()) < 1e-10
            if step in (4, 9, 14, 19, 29):
                checkpoints_small.append(exp_small.copy())
                checkpoints_big.append(exp_big.copy())

        # Verify active inner products AND exact 32-amplitude statevector!
        for i in range(len(checkpoints_small) - 1):
            ov_s = complex(np.vdot(checkpoints_small[i].to_statevector(), checkpoints_small[i + 1].to_statevector()))
            ov_b = complex(
                np.vdot(
                    restrict_to_qmap(checkpoints_big[i], qmap).to_statevector(),
                    restrict_to_qmap(checkpoints_big[i + 1], qmap).to_statevector(),
                )
            )
            assert abs(ov_b - ov_s) < 1e-10

        psi_small = exp_small.to_statevector()
        psi_big_active = restrict_to_qmap(exp_big, qmap).to_statevector()
        assert np.max(np.abs(psi_big_active - psi_small)) < 1e-10


def test_deliberate_norm_and_mass_tolerance_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (1) Corrupted raw ShotExpansion claiming _is_canonical_merged=True on duplicate rows:
    #     fast O(chi) norm_sq() == 1.0, whereas coherent merged norm == 2.0!
    unmerged = ShotExpansion(
        n_qubits=3,
        d=0,
        A_words=np.zeros((0, 1), dtype=np.uint64),
        a_union=np.zeros(1, dtype=np.uint64),
        Gamma=np.zeros((0, 0), dtype=np.uint8),
        pivots=np.zeros(0, dtype=np.int32),
        weights=np.array([_INV_SQRT2 + 0.0j, _INV_SQRT2 + 0.0j], dtype=np.complex128),
        h_words=np.zeros((2, 1), dtype=np.uint64),
        ell=np.zeros((2, 0), dtype=np.uint8),
    )
    assert unmerged._is_canonical_merged is True
    assert abs(unmerged.norm_sq() - 1.0) < 1e-15

    with pytest.raises((FloatingPointError, ValueError)):
        apply_pauli_t(unmerged, [0, 0, 0], [1, 0, 0], 0)

    with pytest.raises((FloatingPointError, ValueError)):
        apply_ccz(
            unmerged,
            [
                ([0, 0, 0], [1, 0, 0], 0),
                ([0, 0, 0], [0, 1, 0], 0),
                ([0, 0, 0], [0, 0, 1], 0),
            ],
        )

    blk = CompiledPauliBlock(
        n_qubits=3,
        mode="pauli_sum",
        coeffs=np.array([pq._T_C0, pq._T_C1], dtype=np.complex128),
        a_words=np.zeros((2, 1), dtype=np.uint64),
        b_words=np.array([[0], [1]], dtype=np.uint64),
        kappas=np.array([0, 0], dtype=np.uint8),
        is_unitary=True,
    )
    with pytest.raises((FloatingPointError, ValueError)):
        apply_compiled_block(unmerged, blk)

    with pytest.raises((FloatingPointError, ValueError)):
        measure_expansion(unmerged, [0, 0, 0], [1, 0, 0], 0)

    with pytest.raises((FloatingPointError, ValueError)):
        apply_quarter_turn(unmerged, [1, 0, 0], [0, 0, 0], 0)

    # (2) Targeted perturbation via monkeypatch on project_diagonal in measure_expansion
    orig_proj = pq.project_diagonal
    valid = ShotExpansion.zero(1)

    def drifted_proj(exp, b, sigma=0, outcome=0, **kw):  # type: ignore[no-untyped-def]
        st, z_m = orig_proj(exp, b, sigma, outcome, **kw)
        return st, (z_m + 1e-6 if outcome == 1 else z_m)

    monkeypatch.setattr(pq, "project_diagonal", drifted_proj)
    with pytest.raises((FloatingPointError, ValueError)):
        measure_expansion(valid, [0], [1], 0)

    def neg_bad_proj(exp, b, sigma=0, outcome=0, **kw):  # type: ignore[no-untyped-def]
        st, z_m = orig_proj(exp, b, sigma, outcome, **kw)
        return st, (-1e-6 if outcome == 1 else z_m)

    monkeypatch.setattr(pq, "project_diagonal", neg_bad_proj)
    with pytest.raises((FloatingPointError, ValueError)):
        measure_expansion(valid, [0], [1], 0)

    def neg_tiny_proj(exp, b, sigma=0, outcome=0, **kw):  # type: ignore[no-untyped-def]
        st, z_m = orig_proj(exp, b, sigma, outcome, **kw)
        return st, (-1e-14 if outcome == 1 else z_m)

    monkeypatch.setattr(pq, "project_diagonal", neg_tiny_proj)
    res_clamped = measure_expansion(valid, [0], [1], 0)
    assert res_clamped.z1 == 0.0 and res_clamped.outcome == 0
    monkeypatch.setattr(pq, "project_diagonal", orig_proj)

    # (3) NaN / Inf weights actively raise FloatingPointError / ValueError
    for bad_val in (complex(float("nan"), 0.0), complex(float("inf"), 0.0)):
        nan_exp = ShotExpansion.zero(2)
        nan_exp.weights[0] = bad_val
        with pytest.raises((FloatingPointError, ValueError)):
            merge_frames(nan_exp)
        with pytest.raises((FloatingPointError, ValueError)):
            apply_pauli_t(nan_exp, [0, 0], [1, 0], 0)
        with pytest.raises((FloatingPointError, ValueError)):
            measure_expansion(nan_exp, [0, 0], [1, 0], 0)

    # (4) Invalid atol values (0.0, negative, NaN, +/-Inf) actively raise ValueError immediately
    for bad_atol in (0.0, -1e-10, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            ShotExpansion.zero(2, atol=bad_atol)
        with pytest.raises(ValueError):
            ShotExpansion.from_components([(1.0 + 0.0j, PhaseQFEComponent.zero(2))], atol=bad_atol)
        with pytest.raises(ValueError):
            measure_expansion(valid, [0], [1], 0, atol=bad_atol)
        with pytest.raises(ValueError):
            apply_quarter_turn(valid, [1], [0], 0, atol=bad_atol)
        mutated = ShotExpansion.zero(2)
        mutated.atol = bad_atol
        with pytest.raises(ValueError):
            mutated.check_invariants()
        with pytest.raises(ValueError):
            apply_pauli_t(mutated, [0, 0], [1, 0], 0)
        with pytest.raises(ValueError):
            measure_expansion(mutated, [0, 0], [1, 0], 0)


def test_m1_p1_sequence_terms_and_compiled_block_validation() -> None:
    terms_3q = [(1.0 + 0.0j, PhaseQFEComponent.zero(3))]
    exp_3q = ShotExpansion.zero(3)
    mixed_terms = [
        (0.5 + 0.0j, PhaseQFEComponent.zero(5)),
        (0.5 + 0.0j, PhaseQFEComponent.zero(3)),
    ]

    with pytest.raises(ValueError):
        ShotExpansion.from_components(mixed_terms)

    # Non-ShotExpansion targets must raise TypeError on every primitive
    for bad_target in (terms_3q, PhaseQFEComponent.zero(3)):
        for bad_call in (
            lambda: apply_pauli(bad_target, [0, 0, 1], [0, 0, 0], 0),  # type: ignore[arg-type]
            lambda: apply_quarter_turn(bad_target, [0, 0, 1], [0, 0, 0], 0),  # type: ignore[arg-type]
            lambda: merge_frames(bad_target),  # type: ignore[arg-type]
            lambda: apply_pauli_t(bad_target, [0, 0, 1], [0, 0, 0], 0),  # type: ignore[arg-type]
            lambda: apply_ccz(
                bad_target,  # type: ignore[arg-type]
                [([0, 0, 0], [1, 0, 0], 0), ([0, 0, 0], [0, 1, 0], 0), ([0, 0, 0], [0, 0, 1], 0)],
            ),
            lambda: apply_compiled_block(
                bad_target, CompiledPauliBlock(n_qubits=3, is_identity=True)  # type: ignore[arg-type]
            ),
            lambda: project_diagonal(bad_target, [0, 0, 1]),  # type: ignore[arg-type]
            lambda: measure_expansion(bad_target, [0, 0, 1], [0, 0, 0], 0),  # type: ignore[arg-type]
        ):
            with pytest.raises(TypeError):
                bad_call()

    # Dirty upper tail bits on a component in from_components (multi-word N=65)
    dirty_65 = PhaseQFEComponent.zero(65)
    dirty_65.h_words[1] = np.uint64(1) << np.uint64(6)  # q=70 >= 65
    with pytest.raises(ValueError):
        ShotExpansion.from_components([(1.0 + 0.0j, dirty_65)])

    # Mismatched block.n_qubits, dirty step/pauli_sum tail masks, invalid mode, invalid step.kind
    blk_5q = CompiledPauliBlock(
        n_qubits=5,
        mode="pauli_sum",
        coeffs=np.array([1.0 + 0.0j], dtype=np.complex128),
        a_words=np.array([[1 << 4]], dtype=np.uint64),
        b_words=np.array([[0]], dtype=np.uint64),
        kappas=np.array([0], dtype=np.uint8),
    )
    blk_dirty_tail = CompiledPauliBlock(
        n_qubits=3,
        mode="pauli_sum",
        coeffs=np.array([1.0 + 0.0j], dtype=np.complex128),
        a_words=np.array([[1 << 4]], dtype=np.uint64),
        b_words=np.array([[0]], dtype=np.uint64),
        kappas=np.array([0], dtype=np.uint8),
    )
    blk_seq_dirty = CompiledPauliBlock(
        n_qubits=3,
        mode="sequential",
        steps=(
            CompiledPauliBlockStep(
                kind="P1",
                paulis=((np.array([1 << 4], dtype=np.uint64), np.array([0], dtype=np.uint64), 0),),
            ),
        ),
    )
    blk_bad_mode = CompiledPauliBlock(n_qubits=3, mode="bogus")  # type: ignore[arg-type]
    blk_bad_step = CompiledPauliBlock(
        n_qubits=3,
        mode="sequential",
        steps=(CompiledPauliBlockStep(kind="BOGUS", paulis=()),),  # type: ignore[arg-type]
    )
    for bad_blk in (blk_5q, blk_dirty_tail, blk_seq_dirty, blk_bad_mode, blk_bad_step):
        with pytest.raises(ValueError):
            apply_compiled_block(exp_3q, bad_blk)


def test_m1_p2_explicit_atol_override_precedence_all_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp1 = ShotExpansion.zero(1)

    # (1) apply_compiled_block: is_identity, pauli_sum, and sequential modes
    for mode_blk_e11, mode_blk_e8 in (
        (
            CompiledPauliBlock(
                n_qubits=1, is_identity=True, global_scalar=math.sqrt(1.0 + 1e-11), is_unitary=True
            ),
            CompiledPauliBlock(
                n_qubits=1, is_identity=True, global_scalar=math.sqrt(1.0 + 1e-8), is_unitary=True
            ),
        ),
        (
            CompiledPauliBlock(
                n_qubits=1,
                mode="pauli_sum",
                coeffs=np.array([math.sqrt(1.0 + 1e-11) + 0.0j], dtype=np.complex128),
                a_words=np.zeros((1, 1), dtype=np.uint64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.zeros(1, dtype=np.uint8),
                is_unitary=True,
            ),
            CompiledPauliBlock(
                n_qubits=1,
                mode="pauli_sum",
                coeffs=np.array([math.sqrt(1.0 + 1e-8) + 0.0j], dtype=np.complex128),
                a_words=np.zeros((1, 1), dtype=np.uint64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.zeros(1, dtype=np.uint8),
                is_unitary=True,
            ),
        ),
        (
            CompiledPauliBlock(
                n_qubits=1,
                mode="sequential",
                global_scalar=math.sqrt(1.0 + 1e-11),
                steps=(CompiledPauliBlockStep(kind="P1", paulis=(([1], [0], 0),)),),
                is_unitary=True,
            ),
            CompiledPauliBlock(
                n_qubits=1,
                mode="sequential",
                global_scalar=math.sqrt(1.0 + 1e-8),
                steps=(CompiledPauliBlockStep(kind="P1", paulis=(([1], [0], 0),)),),
                is_unitary=True,
            ),
        ),
    ):
        with pytest.raises(FloatingPointError):
            apply_compiled_block(exp1, mode_blk_e11, atol=1e-13)
        _ = apply_compiled_block(exp1, mode_blk_e8, atol=1e-6)

    # (2) apply_ccz with drifted 8-branch weights
    exp3 = ShotExpansion.zero(3)
    ccz_e11 = [
        (
            math.sqrt(1.0 + 1e-11) if m == 0 else 0.0j,
            np.zeros(1, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            0,
        )
        for m in range(8)
    ]
    ccz_e8 = [
        (
            math.sqrt(1.0 + 1e-8) if m == 0 else 0.0j,
            np.zeros(1, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            0,
        )
        for m in range(8)
    ]
    with pytest.raises(FloatingPointError):
        apply_ccz(exp3, ccz_e11, atol=1e-13)
    _ = apply_ccz(exp3, ccz_e8, atol=1e-6)

    # (3) apply_quarter_turn and apply_pauli_t explicit atol overrides under norm drift
    orig_merge = pq.merge_frames
    orig_pre = pq._checked_shot_pre_norm

    def make_drifted_merge(delta: float):  # type: ignore[no-untyped-def]
        def _drifted_merge(expansion, **kw):  # type: ignore[no-untyped-def]
            res = orig_merge(expansion, **kw)
            res.weights = np.ascontiguousarray(res.weights * math.sqrt(1.0 + delta))
            return res

        return _drifted_merge

    monkeypatch.setattr(pq, "merge_frames", make_drifted_merge(1e-11))
    monkeypatch.setattr(pq, "_checked_shot_pre_norm", lambda e, c, op, _p=orig_pre: _p(e, c, op) + 1e-11)
    with pytest.raises(FloatingPointError):
        apply_quarter_turn(exp1, [1], [0], 0, atol=1e-13)
    with pytest.raises(FloatingPointError):
        apply_pauli_t(exp1, [0], [1], 0, atol=1e-13)

    monkeypatch.setattr(pq, "merge_frames", make_drifted_merge(1e-8))
    monkeypatch.setattr(pq, "_checked_shot_pre_norm", lambda e, c, op, _p=orig_pre: _p(e, c, op) + 1e-8)
    _ = apply_quarter_turn(exp1, [1], [0], 0, atol=1e-6)
    _ = apply_pauli_t(exp1, [0], [1], 0, atol=1e-6)
    monkeypatch.setattr(pq, "merge_frames", orig_merge)
    monkeypatch.setattr(pq, "_checked_shot_pre_norm", orig_pre)

    # (4) measure_expansion forwards explicit atol=tol to both basis-rotation apply_quarter_turn calls
    recorded_atols: list[float | None] = []
    orig_qt = pq.apply_quarter_turn

    def _spy_qt(*args, **kwargs):  # type: ignore[no-untyped-def]
        recorded_atols.append(kwargs.get("atol"))
        return orig_qt(*args, **kwargs)

    monkeypatch.setattr(pq, "apply_quarter_turn", _spy_qt)
    _ = measure_expansion(exp1, [1], [0], 0, atol=2.5e-6, rng=np.random.default_rng(7))
    monkeypatch.setattr(pq, "apply_quarter_turn", orig_qt)
    assert recorded_atols == [2.5e-6, 2.5e-6]


def test_m1_p3_unmerged_shot_expansion_physical_norm_and_primitives() -> None:
    c0 = PhaseQFEComponent.zero(3)
    terms_u = [(0.5 + 0.0j, c0.copy()), (0.5 + 0.0j, c0.copy())]
    exp_u = ShotExpansion.from_components(terms_u, merge=False)
    exp_m = merge_frames(exp_u)
    assert exp_u.chi == 2
    assert exp_u._is_canonical_merged is False
    assert abs(exp_u.norm_sq() - 1.0) < 1e-14
    assert abs(exp_u.norm_sq(canonicalize_first=False) - 0.5) < 1e-14
    assert abs(exp_u.branch_mass() - 1.0) < 1e-14
    assert abs(exp_u.log_branch_mass() - 0.0) < 1e-14

    # All unitary & measurement primitives must accept unmerged ShotExpansion and match merged ShotExpansion
    q_exp = apply_quarter_turn(exp_u, [1, 0, 0], [0, 0, 0], 0)
    q_ref = apply_quarter_turn(exp_m, [1, 0, 0], [0, 0, 0], 0)
    assert q_exp._is_canonical_merged is True
    assert np.max(np.abs(q_exp.to_statevector() - q_ref.to_statevector())) < 1e-14

    t_exp = apply_pauli_t(exp_u, [1, 0, 0], [0, 0, 0], 0)
    t_ref = apply_pauli_t(exp_m, [1, 0, 0], [0, 0, 0], 0)
    assert t_exp._is_canonical_merged is True
    assert np.max(np.abs(t_exp.to_statevector() - t_ref.to_statevector())) < 1e-14

    ccz_axes = [
        ([0, 0, 0], [1, 0, 0], 0),
        ([0, 0, 0], [0, 1, 0], 0),
        ([0, 0, 0], [0, 0, 1], 0),
    ]
    c_exp = apply_ccz(exp_u, ccz_axes)
    assert abs(c_exp.norm_sq() - 1.0) < 1e-14

    b_exp = apply_compiled_block(
        exp_u, CompiledPauliBlock(n_qubits=3, is_identity=True, global_scalar=_OMEGA)
    )
    assert abs(b_exp.norm_sq() - 1.0) < 1e-14

    m_exp = measure_expansion(exp_u, [1, 0, 0], [0, 0, 0], 0, rng=np.random.default_rng(11))
    m_ref = measure_expansion(exp_m, [1, 0, 0], [0, 0, 0], 0, rng=np.random.default_rng(11))
    assert abs((m_exp.z0 + m_exp.z1) - 1.0) < 1e-14
    assert abs(m_exp.z0 - m_ref.z0) < 1e-14 and abs(m_exp.z1 - m_ref.z1) < 1e-14
    assert m_exp.outcome == m_ref.outcome

    # Unmerged raw output of apply_ccz(..., merge=False)
    plus_3 = apply_quarter_turn(ShotExpansion.zero(3), [1, 0, 0], [0, 0, 0], 0)
    unmerged_ccz = apply_ccz(
        plus_3,
        [([1, 0, 0], [0, 0, 0], 0), ([0, 0, 0], [0, 1, 0], 0), ([0, 0, 0], [0, 0, 1], 0)],
        merge=False,
    )
    assert unmerged_ccz._is_canonical_merged is False
    assert abs(unmerged_ccz.norm_sq() - 1.0) < 1e-13
    assert abs(unmerged_ccz.branch_mass() - 1.0) < 1e-13


@pytest.mark.parametrize(
    "bad_atol", [True, False, np.bool_(True), np.bool_(False), np.array(True)]
)
def test_m1_r3_rejects_boolean_atol(bad_atol: object) -> None:
    """Bug 1: `bool` / `np.bool_` / bool ndarray must never coerce to `1.0` in `validate_atol` or primitives."""
    exp = ShotExpansion.zero(1)
    with pytest.raises(ValueError):
        pq.validate_atol(bad_atol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ShotExpansion.zero(1, atol=bad_atol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        apply_quarter_turn(exp, [1], [0], 0, atol=bad_atol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        measure_expansion(exp, [0], [1], 0, atol=bad_atol)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("blk_kwargs", "exc_types"),
    [
        # Bug 2: NaN / Inf in coeffs or global_scalar must never be silently dropped
        (
            dict(
                coeffs=np.array([1.0 + 0j, complex(float("nan"), 0.0)]),
                a_words=np.array([[0], [1]], dtype=np.uint64),
                b_words=np.zeros((2, 1), dtype=np.uint64),
                kappas=np.zeros(2, dtype=np.uint8),
            ),
            FloatingPointError,
        ),
        (
            dict(is_identity=True, global_scalar=complex(float("nan"), 0.0), is_unitary=False),
            FloatingPointError,
        ),
        # Bug 4: 0D coeffs, non-uint64 words, bad kappas, and is_identity=True with bogus steps/dirty tail
        (
            dict(
                coeffs=np.array(1.0 + 0j),
                a_words=np.zeros((1, 1), dtype=np.uint64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.zeros(1, dtype=np.uint8),
            ),
            ValueError,
        ),
        (
            dict(
                coeffs=np.array([1.0 + 0j]),
                a_words=np.array([[1]], dtype=np.int64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.zeros(1, dtype=np.uint8),
            ),
            ValueError,
        ),
        (
            dict(
                coeffs=np.array([1.0 + 0j]),
                a_words=np.zeros((1, 1), dtype=np.uint64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.array([4], dtype=np.uint8),
            ),
            ValueError,
        ),
        (
            dict(is_identity=True, steps=(CompiledPauliBlockStep("BOGUS", ()),)),  # type: ignore[arg-type]
            ValueError,
        ),
        (
            dict(
                is_identity=True,
                coeffs=np.array([1.0 + 0j]),
                a_words=np.array([[2]], dtype=np.uint64),
                b_words=np.zeros((1, 1), dtype=np.uint64),
                kappas=np.zeros(1, dtype=np.uint8),
            ),
            ValueError,
        ),
    ],
)
def test_m1_r3_compiled_block_nan_ndim_dtype_and_identity_validation(
    blk_kwargs: dict[str, object], exc_types: type[Exception] | tuple[type[Exception], ...]
) -> None:
    """Bugs 2 & 4: `apply_compiled_block` validates finiteness, ndim, dtypes, and `is_identity=True`."""
    blk = CompiledPauliBlock(n_qubits=1, **blk_kwargs)  # type: ignore[arg-type]
    for tgt in (ShotExpansion.zero(1), ShotExpansion.from_components([], n_qubits=1)):
        with pytest.raises(exc_types):
            apply_compiled_block(tgt, blk, check_unitary=False)


def test_m1_r3_component_terms_invariants_and_in_place_quarter_turn() -> None:
    """Bugs 3 & 5: `_validate_component_terms` enforces invariants; `apply_quarter_turn(in_place=True)` merges in-place."""
    c0 = PhaseQFEComponent.zero(2)
    bad_piv = PhaseQFEComponent(
        n_qubits=2,
        d=1,
        h_words=np.array([0], dtype=np.uint64),
        A_words=np.array([[1]], dtype=np.uint64),
        ell=np.array([0], dtype=np.uint8),
        Gamma=np.zeros((1, 1), dtype=np.uint8),
        pivots=np.array([1], dtype=np.int32),
    )
    bad_ell = PhaseQFEComponent(
        n_qubits=2,
        d=1,
        h_words=np.array([0], dtype=np.uint64),
        A_words=np.array([[1]], dtype=np.uint64),
        ell=np.array([4], dtype=np.uint8),
        Gamma=np.zeros((1, 1), dtype=np.uint8),
        pivots=np.array([0], dtype=np.int32),
    )
    for bad_terms, exc in (
        ([c0], ValueError),
        ([(1.0 + 0j, c0, 0)], ValueError),
        ([(1.0 + 0j, "not_a_comp")], ValueError),
        ([(complex(float("nan"), 0.0), c0)], FloatingPointError),
        ([(0.5 + 0j, bad_piv), (0.5 + 0j, bad_piv)], ValueError),
        ([(1.0 + 0j, bad_ell)], ValueError),
    ):
        with pytest.raises(exc):  # type: ignore[arg-type]
            ShotExpansion.from_components(bad_terms)  # type: ignore[arg-type]

    # Bug 5: in_place=True on ShotExpansion must mutate & merge `exp_ip` in-place (`ret is exp_ip`)
    exp_ip = ShotExpansion.from_components(
        [(0.5 + 0j, c0.copy()), (0.5 + 0j, c0.copy())], merge=False
    )
    exp_ref = apply_quarter_turn(exp_ip.copy(), [1, 0], [0, 0], 0, in_place=False)
    ret = apply_quarter_turn(exp_ip, [1, 0], [0, 0], 0, in_place=True)
    assert ret is exp_ip
    assert exp_ip._is_canonical_merged is True and exp_ip.chi == 1
    exp_ip.check_invariants(require_canonical_offset=True)
    assert np.max(np.abs(exp_ip.to_statevector() - exp_ref.to_statevector())) < 1e-15


def test_contract_choi_expansions_validates_operand_qubit_counts() -> None:
    good_left = ShotExpansion.zero(2)
    good_right = ShotExpansion.zero(2)
    out = pq.contract_choi_expansions(good_left, good_right, n_op=1, q_act=[0], right_is_choi=True)
    assert out.n_qubits == 2

    with pytest.raises(ValueError, match="Operand n_qubits mismatch"):
        pq.contract_choi_expansions(ShotExpansion.zero(3), good_right, n_op=1, q_act=[0], right_is_choi=True)
    with pytest.raises(ValueError, match="Operand n_qubits mismatch"):
        pq.contract_choi_expansions(good_left, ShotExpansion.zero(3), n_op=1, q_act=[0], right_is_choi=True)
    with pytest.raises(ValueError, match="Operand n_qubits mismatch"):
        pq.contract_choi_expansions(good_left, ShotExpansion.zero(2), n_op=1, q_act=[0], right_is_choi=False)

