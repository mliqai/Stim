from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from stimside.util.bitwords import (
    first_set_bit_in_words as _first_set_bit_in_words,
    word_dot_parity as _word_dot_parity,
    words_any_overlap as _words_any_overlap,
)


class MRQFState:
    """Compressed Moving-Reference Quadratic-Form (MRQF) state D = (h, A, ell, Gamma, P).

    Represents the residual stabilizer state in reference coordinates B_t:
        |chi> = 2^{-d/2} sum_{u in F_2^d} i^{Q(u)} |h xor A u>,
        Q(u) = sum_{j=0}^{d-1} ell_j u_j + 2 sum_{0 <= j < k < d} Gamma_{jk} u_j u_k (mod 4),
    maintaining the Principal-Row Invariant A_{P, :} = I_d in O(n(d + 1) + d^2) bits.
    """

    def __init__(
        self,
        num_qubits: int,
        n_words: int | None = None,
        _init_cap: int = 16,
    ) -> None:
        self.num_qubits = num_qubits
        self.n_words = (
            max(1, (num_qubits + 63) // 64) if n_words is None else n_words
        )
        self._cap = max(4, _init_cap)
        self._d = 0
        self.h_words: NDArray[np.uint64] = np.zeros(
            self.n_words, dtype=np.uint64
        )
        self.a_union: NDArray[np.uint64] = np.zeros(
            self.n_words, dtype=np.uint64
        )
        self._A_buf: NDArray[np.uint64] = np.zeros(
            (self._cap, self.n_words), dtype=np.uint64
        )
        self._ell_buf: NDArray[np.uint8] = np.zeros(
            (self._cap,), dtype=np.uint8
        )
        self._Gamma_buf: NDArray[np.uint8] = np.zeros(
            (self._cap, self._cap), dtype=np.uint8
        )
        self._A_view: NDArray[np.uint64] = self._A_buf[:0]
        self._ell_view: NDArray[np.uint8] = self._ell_buf[:0]
        self._Gamma_view: NDArray[np.uint8] = self._Gamma_buf[:0, :0]
        self.P: list[int] = []

    def _refresh_views(self) -> None:
        d = self._d
        self._A_view = self._A_buf[:d]
        self._ell_view = self._ell_buf[:d]
        self._Gamma_view = self._Gamma_buf[:d, :d]

    @property
    def A_words(self) -> NDArray[np.uint64]:
        return self._A_view

    @property
    def ell(self) -> NDArray[np.uint8]:
        return self._ell_view

    @property
    def Gamma(self) -> NDArray[np.uint8]:
        return self._Gamma_view

    @classmethod
    def from_arrays(
        cls,
        num_qubits: int,
        *,
        A_words: NDArray[np.uint64],
        ell: NDArray[np.uint8],
        Gamma: NDArray[np.uint8],
        P: Sequence[int],
        h_words: NDArray[np.uint64] | None = None,
    ) -> "MRQFState":
        d = len(P)
        st = cls(num_qubits, _init_cap=max(4, d))
        st._d = d
        st.P = list(P)
        if h_words is not None:
            np.copyto(st.h_words, np.asarray(h_words, dtype=np.uint64))
        if d > 0:
            st._A_buf[:d] = np.asarray(A_words, dtype=np.uint64)
            st._ell_buf[:d] = np.asarray(ell, dtype=np.uint8)
            st._Gamma_buf[:d, :d] = np.asarray(Gamma, dtype=np.uint8)
        st._refresh_views()
        st._recompute_a_union()
        return st

    def _ensure_capacity(self, need_d: int) -> None:
        if need_d <= self._cap:
            return
        new_cap = self._cap
        while new_cap < need_d:
            new_cap *= 2
        new_A = np.zeros((new_cap, self.n_words), dtype=np.uint64)
        new_ell = np.zeros((new_cap,), dtype=np.uint8)
        new_Gamma = np.zeros((new_cap, new_cap), dtype=np.uint8)
        d = self._d
        if d > 0:
            new_A[:d] = self._A_buf[:d]
            new_ell[:d] = self._ell_buf[:d]
            new_Gamma[:d, :d] = self._Gamma_buf[:d, :d]
        self._cap = new_cap
        self._A_buf = new_A
        self._ell_buf = new_ell
        self._Gamma_buf = new_Gamma
        self._refresh_views()

    def _recompute_a_union(self) -> None:
        d = self._d
        if d == 0:
            self.a_union.fill(0)
        elif d == 1:
            np.copyto(self.a_union, self._A_buf[0])
        else:
            np.bitwise_or.reduce(self._A_buf[:d], axis=0, out=self.a_union)

    @property
    def d(self) -> int:
        return self._d

    @property
    def num_frames(self) -> int:
        return 1 << self._d

    def copy(self) -> "MRQFState":
        clone = MRQFState(self.num_qubits, self.n_words, _init_cap=self._cap)
        clone._d = self._d
        np.copyto(clone.h_words, self.h_words)
        np.copyto(clone.a_union, self.a_union)
        if self._d > 0:
            clone._A_buf[: self._d] = self._A_buf[: self._d]
            clone._ell_buf[: self._d] = self._ell_buf[: self._d]
            clone._Gamma_buf[: self._d, : self._d] = self._Gamma_buf[
                : self._d, : self._d
            ]
        clone.P = list(self.P)
        clone._refresh_views()
        return clone

    def check_invariants(self) -> None:
        """Verify Check C.1: shapes, principal-row identity A_{P, :} = I_d, symmetric zero-diagonal Gamma, ell in Z_4."""
        d = len(self.P)
        assert self._d == d
        assert self.h_words.shape == (self.n_words,)
        assert self.A_words.shape == (d, self.n_words)
        assert self.ell.shape == (d,)
        assert self.Gamma.shape == (d, d)
        assert len(set(self.P)) == d
        if d == 0:
            assert not np.any(self.a_union)
        else:
            expected_union = np.bitwise_or.reduce(self.A_words, axis=0)
            assert np.array_equal(self.a_union, expected_union)
            assert np.all(self.ell < 4)
            assert np.all((self.Gamma == 0) | (self.Gamma == 1))
            assert np.array_equal(self.Gamma, self.Gamma.T)
            assert np.all(np.diag(self.Gamma) == 0)
            for j, p_j in enumerate(self.P):
                p_word = p_j >> 6
                p_mask = np.uint64(1 << (p_j & 63))
                col_bits = (self.A_words[:, p_word] & p_mask) != 0
                expected = np.zeros(d, dtype=np.bool_)
                expected[j] = True
                assert np.array_equal(col_bits, expected)

    def _matvec_A_T(self, b_words: NDArray[np.uint64]) -> NDArray[np.uint8]:
        """Compute t = A^T b in F_2^d."""
        d = self._d
        if d <= 8:
            b_list = b_words.tolist()
            au_list = self.a_union.tolist()
            nz_w = [
                (w, b_list[w])
                for w in range(self.n_words)
                if b_list[w] & au_list[w]
            ]
            out = np.zeros(d, dtype=np.uint8)
            if not nz_w:
                return out
            A_list = self._A_buf[:d].tolist()
            for r in range(d):
                row = A_list[r]
                acc = 0
                for w, bw in nz_w:
                    acc ^= row[w] & bw
                out[r] = acc.bit_count() & 1
            return out
        if self.n_words == 1:
            return (
                np.bitwise_count(self.A_words[:, 0] & b_words[0]) & 1
            ).astype(np.uint8)
        return (
            np.bitwise_count(self.A_words & b_words[None, :]).sum(axis=1) & 1
        ).astype(np.uint8)

    # S1. AddColumn(r, s)
    def s1_add_column(self, r: int, s: int) -> None:
        """Add column r into column s and update ell and Gamma consistently (Eq. 15)."""
        assert r != s
        d = self._d
        ell_r = int(self._ell_buf[r])
        ell_s = int(self._ell_buf[s])
        gamma_rs = int(self._Gamma_buf[r, s])

        self._A_buf[s] ^= self._A_buf[r]
        self._ell_buf[s] = (ell_s + ell_r + (gamma_rs << 1)) & 3
        new_gamma_rs = gamma_rs ^ (ell_r & 1)
        self._Gamma_buf[s, :d] ^= self._Gamma_buf[r, :d]
        self._Gamma_buf[:d, s] = self._Gamma_buf[s, :d]
        self._Gamma_buf[r, s] = new_gamma_rs
        self._Gamma_buf[s, r] = new_gamma_rs
        self._Gamma_buf[s, s] = 0

    # S2. MakePrincipal(j, p)
    def s2_make_principal(self, j: int, p: int) -> None:
        """Restore designated identity row p for column j (Section 6, S2)."""
        p_word = p >> 6
        p_bit = p & 63
        A_buf = self._A_buf
        for s in range(self._d):
            if s != j and ((int(A_buf[s, p_word]) >> p_bit) & 1):
                self.s1_add_column(j, s)
        self.P[j] = p

    # S4. DropVariable(j)
    def s4_drop_variable(self, j: int) -> None:
        """Remove variable slot j from A_words, ell, Gamma, and P in-place (Eq. 17)."""
        k = self._d
        if k == 1:
            self._d = 0
            self.P.clear()
            self.a_union.fill(0)
            self._refresh_views()
            return
        if j < k - 1:
            self._A_buf[j : k - 1] = self._A_buf[j + 1 : k]
            self._ell_buf[j : k - 1] = self._ell_buf[j + 1 : k]
            self._Gamma_buf[j : k - 1, :k] = self._Gamma_buf[j + 1 : k, :k]
            self._Gamma_buf[: k - 1, j : k - 1] = self._Gamma_buf[
                : k - 1, j + 1 : k
            ]
        self._d = k - 1
        self.P.pop(j)
        self._refresh_views()
        self._recompute_a_union()

    # S3. Restrict(lambda, epsilon)
    def s3_restrict(
        self, lam: NDArray[np.uint8] | Sequence[int], eps: int
    ) -> tuple[bool, float]:
        """Impose affine equation lam^T u = eps (mod 2) on the support (Eq. 16)."""
        d = self._d
        first_nz = -1
        for idx in range(d):
            if lam[idx]:
                if first_nz == -1:
                    first_nz = idx
                else:
                    self.s1_add_column(first_nz, idx)
        eps_bit = int(eps) & 1
        if first_nz == -1:
            return (True, 1.0) if eps_bit == 0 else (False, 0.0)
        if eps_bit:
            self.h_words ^= self._A_buf[first_nz]
            self._ell_buf[:d] = (
                self._ell_buf[:d] + (self._Gamma_buf[first_nz, :d] << 1)
            ) & 3
        self.s4_drop_variable(first_nz)
        return True, 0.5

    # S5. EliminateZero(j)
    def s5_eliminate_zero(self, j: int) -> tuple[bool, int]:
        """Sum out redundant variable j with A_j = 0 coherently (Eq. 19-20)."""
        c = int(self._ell_buf[j])
        k = self._d
        lam = np.empty(k - 1, dtype=np.uint8)
        if j > 0:
            lam[:j] = self._Gamma_buf[:j, j]
        if j < k - 1:
            lam[j:] = self._Gamma_buf[j + 1 : k, j]
        self.s4_drop_variable(j)

        if c & 1:
            if len(lam) > 0:
                self.ell[:] = (
                    (self.ell.astype(np.int16) - c * lam.astype(np.int16)) & 3
                ).astype(np.uint8)
                outer = lam[:, None] & lam[None, :]
                np.fill_diagonal(outer, 0)
                self.Gamma[:, :] ^= outer
            return True, 1

        if np.any(lam):
            ok, _ = self.s3_restrict(lam, c >> 1)
            return ok, (1 if ok else 0)

        if c == 0:
            return True, 2
        return False, 0

    # P1. ApplyPauli(a, b)
    def p1_apply_pauli(
        self,
        a_words: NDArray[np.uint64],
        b_words: NDArray[np.uint64],
    ) -> None:
        """Apply P(a, b, kappa) to |chi> in O(d * n_words) time (Eq. 7)."""
        if self._d > 0 and _words_any_overlap(b_words, self.a_union):
            t = self._matvec_A_T(b_words)
            self.ell[:] = (self.ell + (t << 1)) & 3
        self.h_words ^= a_words

    def _normalize_column(self, idx: int) -> None:
        """Normalize newly added column idx against principal rows 0..idx-1 via S1, then S2 or S5."""
        A_buf = self._A_buf
        for r in range(idx):
            p_r = self.P[r]
            if (int(A_buf[idx, p_r >> 6]) >> (p_r & 63)) & 1:
                self.s1_add_column(r, idx)
        p_new = _first_set_bit_in_words(A_buf[idx])
        if p_new >= 0:
            self.s2_make_principal(idx, p_new)
        else:
            ok, eta = self.s5_eliminate_zero(idx)
            assert ok and eta == 1, (
                f"Unitary EliminateZero failed: ok={ok}, eta={eta}"
            )

    # P2. ApplyQuarterTurn(a, b, kappa)
    def p2_apply_quarter_turn(
        self,
        a_words: NDArray[np.uint64],
        b_words: NDArray[np.uint64],
        kappa: int,
    ) -> None:
        """Apply quarter-turn R(P(a, b, kappa)) = (I + i P(a, b, kappa))/sqrt(2) to |chi> (Eq. 8-9)."""
        d0 = self._d
        self._ensure_capacity(d0 + 1)
        if d0 > 0:
            if _words_any_overlap(b_words, self.a_union):
                t = self._matvec_A_T(b_words)
                self._Gamma_buf[:d0, d0] = t
                self._Gamma_buf[d0, :d0] = t
            else:
                self._Gamma_buf[:d0, d0] = 0
                self._Gamma_buf[d0, :d0] = 0
        self._Gamma_buf[d0, d0] = 0
        bh = _word_dot_parity(b_words, self.h_words)
        c = (kappa + 1 + (bh << 1)) & 3

        self._A_buf[d0] = a_words
        self.a_union |= a_words
        self._ell_buf[d0] = c
        self.P.append(-1)
        self._d = d0 + 1
        self._refresh_views()
        self._normalize_column(d0)

    # PCP. ApplyControlledPauli(a1, b1, kappa1, a2, b2, kappa2)
    def p2_apply_controlled_pauli(
        self,
        a1_words: NDArray[np.uint64],
        b1_words: NDArray[np.uint64],
        kappa1: int,
        a2_words: NDArray[np.uint64],
        b2_words: NDArray[np.uint64],
        kappa2: int,
    ) -> None:
        """Apply controlled-Pauli U = (I + P_1 + P_2 - P_1 P_2) / 2 via direct 2-column rank-2 injection."""
        d0 = self._d
        self._ensure_capacity(d0 + 2)
        if d0 > 0:
            if _words_any_overlap(b1_words, self.a_union):
                t1 = self._matvec_A_T(b1_words)
                self._Gamma_buf[:d0, d0] = t1
                self._Gamma_buf[d0, :d0] = t1
            else:
                self._Gamma_buf[:d0, d0] = 0
                self._Gamma_buf[d0, :d0] = 0
            if _words_any_overlap(b2_words, self.a_union):
                t2 = self._matvec_A_T(b2_words)
                self._Gamma_buf[:d0, d0 + 1] = t2
                self._Gamma_buf[d0 + 1, :d0] = t2
            else:
                self._Gamma_buf[:d0, d0 + 1] = 0
                self._Gamma_buf[d0 + 1, :d0] = 0

        bh1 = _word_dot_parity(b1_words, self.h_words)
        bh2 = _word_dot_parity(b2_words, self.h_words)
        c1 = (kappa1 + (bh1 << 1)) & 3
        c2 = (kappa2 + (bh2 << 1)) & 3
        gamma12 = 1 ^ _word_dot_parity(b1_words, a2_words)

        self._A_buf[d0] = a1_words
        self._A_buf[d0 + 1] = a2_words
        self.a_union |= a1_words
        self.a_union |= a2_words
        self._ell_buf[d0] = c1
        self._ell_buf[d0 + 1] = c2
        self._Gamma_buf[d0, d0] = 0
        self._Gamma_buf[d0 + 1, d0 + 1] = 0
        self._Gamma_buf[d0, d0 + 1] = gamma12
        self._Gamma_buf[d0 + 1, d0] = gamma12
        self.P.extend([-1, -1])
        self._d = d0 + 2
        self._refresh_views()

        self._normalize_column(d0)
        if self._d > 0 and self.P[-1] == -1:
            self._normalize_column(self._d - 1)

    # P3. MeasureDiagonal(b, sigma; optional requested bit)
    def p3_measure_diagonal(
        self,
        b_words: NDArray[np.uint64],
        sigma: int,
        draw_bit_fn: Callable[[], int] | None = None,
        requested_bit: int | None = None,
    ) -> tuple[int, float]:
        """Measure diagonal observable (-1)^sigma Z(b) on |chi> (Eq. 11)."""
        b_list = b_words.tolist()
        h_list = self.h_words.tolist()
        bh_acc = 0
        for bw, hw in zip(b_list, h_list):
            bh_acc ^= bw & hw
        bh = bh_acc.bit_count() & 1
        beta = (sigma ^ bh) & 1
        d = self._d
        if d == 0:
            if requested_bit is not None and (requested_bit & 1) != beta:
                return int(requested_bit) & 1, 0.0
            return beta, 1.0

        au_list = self.a_union.tolist()
        nz_w = [
            (w, b_list[w])
            for w in range(self.n_words)
            if b_list[w] & au_list[w]
        ]
        if not nz_w:
            if requested_bit is not None and (requested_bit & 1) != beta:
                return int(requested_bit) & 1, 0.0
            return beta, 1.0

        A_list = self._A_buf[:d].tolist()
        first_nz = -1
        other_nz: list[int] = []
        for r in range(d):
            row = A_list[r]
            acc = 0
            for w, bw in nz_w:
                acc ^= row[w] & bw
            if acc.bit_count() & 1:
                if first_nz == -1:
                    first_nz = r
                else:
                    other_nz.append(r)

        if first_nz == -1:
            if requested_bit is not None and (requested_bit & 1) != beta:
                return int(requested_bit) & 1, 0.0
            return beta, 1.0

        if requested_bit is not None:
            m = int(requested_bit) & 1
        elif draw_bit_fn is not None:
            m = int(draw_bit_fn()) & 1
        else:
            raise ValueError(
                "Either requested_bit or draw_bit_fn must be provided when lam != 0."
            )
        eps_bit = (m ^ beta) & 1
        for idx in other_nz:
            self.s1_add_column(first_nz, idx)
        if eps_bit:
            self.h_words ^= self._A_buf[first_nz]
            self._ell_buf[:d] = (
                self._ell_buf[:d] + (self._Gamma_buf[first_nz, :d] << 1)
            ) & 3
        self.s4_drop_variable(first_nz)
        return m, 0.5

    # Q1. PauliExpectation(a, b, kappa)
    def q1_pauli_expectation(
        self,
        a_words: NDArray[np.uint64],
        b_words: NDArray[np.uint64],
        kappa: int,
    ) -> int:
        """Evaluate exact read-only Pauli expectation <chi|P(a, b, kappa)|chi> in {0, +1, -1} (Eq. 14)."""
        d = self._d
        if d == 0:
            if np.any(a_words):
                return 0
            bh = _word_dot_parity(b_words, self.h_words)
            exp_mod4 = (kappa + (bh << 1)) & 3
            return 1 if exp_mod4 == 0 else -1

        if np.any(a_words & ~self.a_union):
            return 0

        s = np.array(
            [
                int((a_words[p_r >> 6] >> np.uint64(p_r & 63)) & np.uint64(1))
                for p_r in self.P
            ],
            dtype=np.uint8,
        )
        s_nz = np.flatnonzero(s)
        if len(s_nz) == 0:
            if np.any(a_words):
                return 0
        else:
            a_recon = np.bitwise_xor.reduce(self.A_words[s_nz], axis=0)
            if not np.array_equal(a_recon, a_words):
                return 0

        at_b = self._matvec_A_T(b_words)
        ws = ((self.ell & 1) & s) ^ ((self.Gamma @ s) & 1)
        if not np.array_equal(at_b, ws):
            return 0

        s_i32 = s.astype(np.int32)
        lin = int(np.dot(self.ell.astype(np.int32), s_i32))
        quad = int(s_i32 @ np.triu(self.Gamma, k=1).astype(np.int32) @ s_i32)
        q_s = (lin + (quad << 1)) & 3
        bh = _word_dot_parity(b_words, self.h_words)
        exp_mod4 = (kappa + (bh << 1) - q_s) & 3
        assert exp_mod4 in (0, 2)
        return 1 if exp_mod4 == 0 else -1
