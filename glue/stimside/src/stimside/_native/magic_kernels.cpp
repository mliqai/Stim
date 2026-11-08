// Copyright 2026 Google LLC
//
// C++17 shared-library kernels accelerating coherent Phase-QFE state evolution,
// canonical coset merging, non-Clifford branching, and diagonal projection
// across single- and multi-word bitpackings.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include "qfe_core.hpp"

extern "C" {

int32_t magic_abi_v1() {
    return 1;
}

void magic_apply_pauli_batch(
    size_t chi,
    size_t d,
    size_t n_words,
    double* weights_ri,
    uint64_t* h_words,
    uint8_t* ell,
    const uint64_t* A_words,
    const uint64_t* a_union,
    const uint64_t* pauli_a,
    const uint64_t* pauli_b,
    uint8_t kappa
) {
    if (chi == 0 || n_words == 0) return;
    const uint8_t kap = kappa & 3U;
    const bool has_a = words_any_nonzero(pauli_a, n_words);
    const bool has_b = words_any_nonzero(pauli_b, n_words);

    uint8_t t_stack[256];
    std::vector<uint8_t> t_heap;
    uint8_t* t_shift = nullptr;
    bool any_t = false;
    if (d > 0 && has_b && (!a_union || words_any_overlap(pauli_b, a_union, n_words))) {
        t_shift = (d <= 256) ? t_stack : (t_heap.resize(d), t_heap.data());
        for (size_t r = 0; r < d; ++r) {
            const uint8_t tr = static_cast<uint8_t>(
                word_dot_parity(&A_words[r * n_words], pauli_b, n_words) << 1
            );
            t_shift[r] = tr;
            any_t |= (tr != 0);
        }
    }

    for (size_t a = 0; a < chi; ++a) {
        const uint8_t bh = has_b ? word_dot_parity(&h_words[a * n_words], pauli_b, n_words) : 0U;
        mul_phase4(
            weights_ri[2 * a],
            weights_ri[2 * a + 1],
            static_cast<uint8_t>((kap + (bh << 1)) & 3U)
        );
        if (has_a) {
            xor_words_inplace(&h_words[a * n_words], pauli_a, n_words);
        }
        if (any_t) {
            uint8_t* row_ell = &ell[a * d];
            for (size_t r = 0; r < d; ++r) {
                row_ell[r] = static_cast<uint8_t>((row_ell[r] + t_shift[r]) & 3U);
            }
        }
    }
}

int magic_apply_quarter_turn_tol(
    size_t chi,
    size_t d0,
    size_t n_qubits,
    size_t n_words,
    const double* weights_ri,
    const uint64_t* h_words,
    const uint8_t* ell,
    const uint64_t* A_words,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    const uint64_t* pauli_a,
    const uint64_t* pauli_b,
    uint8_t kappa,
    double cancel_tol,
    size_t* out_chi,
    size_t* out_d,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    uint64_t* out_A_words,
    uint8_t* out_Gamma,
    int32_t* out_pivots,
    uint8_t* out_ell_odd
) {
    const size_t expected_nw = std::max<size_t>(1, (n_qubits + 63) >> 6);
    if (n_words != expected_nw || d0 > n_qubits || !pauli_a || !pauli_b) {
        return 1;
    }
    const uint64_t tm = tail_mask(n_qubits);
    if (((pauli_a[n_words - 1] | pauli_b[n_words - 1]) & ~tm) != 0ULL) {
        return 1;
    }
    const uint8_t kap = kappa & 3U;
    if (((kap ^ word_dot_parity(pauli_a, pauli_b, n_words)) & 1U) != 0U) {
        return 1;
    }

    const size_t cap = d0 + 1;
    WorkShot work(
        chi, d0, cap, n_words,
        weights_ri, h_words, ell, A_words, Gamma, pivots, ell_odd
    );

    std::memcpy(&work.A[d0 * n_words], pauli_a, n_words * sizeof(uint64_t));
    work.A[d0 * n_words + (n_words - 1)] &= tm;
    const bool has_b = words_any_nonzero(pauli_b, n_words);
    for (size_t r = 0; r < d0; ++r) {
        const uint8_t t_r = has_b ? word_dot_parity(&work.A[r * n_words], pauli_b, n_words) : 0U;
        work.G[r * cap + d0] = t_r;
        work.G[d0 * cap + r] = t_r;
    }
    work.G[d0 * cap + d0] = 0U;
    work.piv[d0] = -1;
    work.eo[d0] = static_cast<uint8_t>((kap + 1U) & 1U);
    const uint8_t base_c = static_cast<uint8_t>((kap + 1U) & 3U);
    for (size_t a = 0; a < chi; ++a) {
        const uint8_t bh = has_b ? word_dot_parity(&work.h[a * n_words], pauli_b, n_words) : 0U;
        work.ell[a * cap + d0] = static_cast<uint8_t>((base_c + (bh << 1)) & 3U);
    }
    work.d = d0 + 1;

    for (size_t i = 0; i < d0; ++i) {
        if (get_bit(pauli_a, work.piv[i]) != 0) {
            work.add_column(i, d0);
        }
    }

    const int32_t p_new = first_set_bit_in_words(&work.A[d0 * n_words], n_words);
    if (p_new >= 0) {
        work.make_principal(d0, p_new);
    } else {
        work.eliminate_zero_amplitude(d0);
    }

    work.export_and_merge(
        cancel_tol, out_chi, out_d,
        out_weights_ri, out_h_words, out_ell,
        out_A_words, out_Gamma, out_pivots, out_ell_odd, nullptr
    );
    return 0;
}

// Legacy ABI-v2 entry point with the tolerance fixed at 1e-13; callers pass cancel_tol to _tol.
int magic_apply_quarter_turn(
    size_t chi, size_t d0, size_t n_qubits, size_t n_words, const double* weights_ri,
    const uint64_t* h_words, const uint8_t* ell, const uint64_t* A_words, const uint8_t* Gamma,
    const int32_t* pivots, const uint8_t* ell_odd, const uint64_t* pauli_a,
    const uint64_t* pauli_b, uint8_t kappa, size_t* out_chi, size_t* out_d,
    double* out_weights_ri, uint64_t* out_h_words, uint8_t* out_ell, uint64_t* out_A_words,
    uint8_t* out_Gamma, int32_t* out_pivots, uint8_t* out_ell_odd
) {
    return magic_apply_quarter_turn_tol(
        chi, d0, n_qubits, n_words, weights_ri, h_words, ell, A_words, Gamma, pivots, ell_odd,
        pauli_a, pauli_b, kappa, 1e-13, out_chi, out_d, out_weights_ri, out_h_words, out_ell,
        out_A_words, out_Gamma, out_pivots, out_ell_odd
    );
}

void magic_canonicalize_and_merge(
    size_t chi,
    size_t d,
    size_t n_words,
    const double* weights_ri,
    const uint64_t* h_words,
    const uint8_t* ell,
    const uint64_t* A_words,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    double cancel_tol,
    size_t* out_chi,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    double* out_norm_sq
) {
    if (chi == 0 || n_words == 0) {
        if (out_chi) *out_chi = 0;
        if (out_norm_sq) *out_norm_sq = 0.0;
        return;
    }
    FrameMerger merger(
        chi, d, n_words,
        A_words, Gamma, pivots, ell_odd,
        out_weights_ri, out_h_words, out_ell
    );
    for (size_t a = 0; a < chi; ++a) {
        merger.push_copy(
            weights_ri[2 * a], weights_ri[2 * a + 1],
            &h_words[a * n_words], d > 0 ? &ell[a * d] : nullptr
        );
    }
    merger.finalize(cancel_tol, out_chi, out_norm_sq);
}

void magic_branch_pauli_t(
    size_t chi,
    size_t d,
    size_t n_words,
    const double* weights_ri,
    const uint64_t* h_words,
    const uint8_t* ell,
    const uint64_t* A_words,
    const uint64_t* a_union,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    const uint64_t* pauli_a,
    const uint64_t* pauli_b,
    uint8_t kappa,
    int is_dag,
    double cancel_tol,
    size_t* out_chi,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    double* out_norm_sq
) {
    if (chi == 0 || n_words == 0 || !pauli_a || !pauli_b) {
        if (out_chi) *out_chi = 0;
        if (out_norm_sq) *out_norm_sq = 0.0;
        return;
    }
    const double bw[4] = {
        is_dag ? kTDagC0Re : kTC0Re,
        is_dag ? kTDagC0Im : kTC0Im,
        is_dag ? kTDagC1Re : kTC1Re,
        is_dag ? kTDagC1Im : kTC1Im,
    };
    const uint8_t bk[2] = {0U, static_cast<uint8_t>(kappa & 3U)};

    uint64_t ab_stack[64];
    std::vector<uint64_t> ab_heap;
    uint64_t* ba = (n_words <= 16) ? ab_stack : (ab_heap.assign(4 * n_words, 0ULL), ab_heap.data());
    uint64_t* bb = ba + 2 * n_words;
    if (n_words <= 16) {
        std::memset(ab_stack, 0, 4 * n_words * sizeof(uint64_t));
    }
    std::memcpy(ba + n_words, pauli_a, n_words * sizeof(uint64_t));
    std::memcpy(bb + n_words, pauli_b, n_words * sizeof(uint64_t));

    branch_pauli_sum_and_merge(
        2, chi, d, n_words,
        weights_ri, h_words, ell, A_words, a_union, Gamma, pivots, ell_odd,
        bw, ba, bb, bk, cancel_tol,
        out_chi, out_weights_ri, out_h_words, out_ell, out_norm_sq
    );
}

void magic_branch_ccz(
    size_t chi,
    size_t d,
    size_t n_words,
    const double* weights_ri,
    const uint64_t* h_words,
    const uint8_t* ell,
    const uint64_t* A_words,
    const uint64_t* a_union,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    const double* branch_weights_ri,
    const uint64_t* branch_a_words,
    const uint64_t* branch_b_words,
    const uint8_t* branch_kappas,
    double cancel_tol,
    size_t* out_chi,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    double* out_norm_sq
) {
    branch_pauli_sum_and_merge(
        8, chi, d, n_words,
        weights_ri, h_words, ell, A_words, a_union, Gamma, pivots, ell_odd,
        branch_weights_ri, branch_a_words, branch_b_words, branch_kappas, cancel_tol,
        out_chi, out_weights_ri, out_h_words, out_ell, out_norm_sq
    );
}

void magic_project_diagonal(
    size_t chi,
    size_t d0,
    size_t n_words,
    const double* weights_ri,
    const uint64_t* h_words,
    const uint8_t* ell,
    const uint64_t* A_words,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    const uint64_t* b_words,
    uint8_t kappa,
    uint8_t outcome,
    double cancel_tol,
    size_t* out_chi,
    size_t* out_d,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    uint64_t* out_A_words,
    uint8_t* out_Gamma,
    int32_t* out_pivots,
    uint8_t* out_ell_odd,
    double* out_z_m
) {
    WorkShot work(
        chi, d0, std::max<size_t>(1, d0), n_words,
        weights_ri, h_words, ell, A_words, Gamma, pivots, ell_odd
    );
    const uint8_t sig = (kappa >> 1) & 1U;
    const uint8_t m_bit = outcome & 1U;
    const bool has_b = words_any_nonzero(b_words, n_words);
    std::vector<uint8_t> rhs(chi, 0U);
    for (size_t a = 0; a < chi; ++a) {
        const uint8_t hb = has_b ? word_dot_parity(&work.h[a * n_words], b_words, n_words) : 0U;
        rhs[a] = static_cast<uint8_t>((m_bit ^ sig ^ hb) & 1U);
    }
    std::vector<uint8_t> col_par(d0, 0U);
    if (has_b) {
        for (size_t r = 0; r < d0; ++r) {
            col_par[r] = word_dot_parity(&work.A[r * n_words], b_words, n_words);
        }
    }
    work.restrict_amplitude(col_par.data(), rhs.data(), kInvSqrt2);
    work.export_and_merge(
        cancel_tol, out_chi, out_d, out_weights_ri, out_h_words, out_ell,
        out_A_words, out_Gamma, out_pivots, out_ell_odd, out_z_m
    );
}

int32_t magic_abi_v2() {
    return 2;
}

int32_t magic_abi_v3() {
    return 3;
}

int magic_contract_choi(
    size_t chi_L,
    size_t d_L,
    size_t nw_L,
    const double* w_ri_L,
    const uint64_t* h_L,
    const uint8_t* ell_L,
    const uint64_t* A_L,
    const uint8_t* G_L,
    const int32_t* piv_L,
    const uint8_t* eo_L,
    size_t chi_R,
    size_t d_R,
    size_t nw_R,
    const double* w_ri_R,
    const uint64_t* h_R,
    const uint8_t* ell_R,
    const uint64_t* A_R,
    const uint8_t* G_R,
    const int32_t* piv_R,
    const uint8_t* eo_R,
    size_t n_op,
    const int32_t* q_act,
    size_t m,
    int right_is_choi,
    double cancel_tol,
    size_t* out_chi,
    size_t* out_d,
    double* out_weights_ri,
    uint64_t* out_h_words,
    uint8_t* out_ell,
    uint64_t* out_A_words,
    uint8_t* out_Gamma,
    int32_t* out_pivots,
    uint8_t* out_ell_odd
) {
    if (out_chi) *out_chi = 0;
    if (out_d) *out_d = 0;
    if (m > n_op || (m > 0 && !q_act)) {
        return 1;
    }
    try {
        std::vector<uint8_t> is_act(n_op, 0U);
        for (size_t r = 0; r < m; ++r) {
            const int32_t q = q_act[r];
            if (q < 0 || static_cast<size_t>(q) >= n_op || is_act[static_cast<size_t>(q)]) {
                return 1;
            }
            is_act[static_cast<size_t>(q)] = 1U;
        }
        if (chi_L == 0 || chi_R == 0) {
            return 0;
        }
        if (!w_ri_L || !w_ri_R || (d_L > 0 && (!piv_L || !eo_L)) || (d_R > 0 && (!piv_R || !eo_R))) {
            return 1;
        }
        if (chi_R > (~size_t(0)) / chi_L) {
            return 2;
        }
        const size_t prod_chi = chi_L * chi_R;
        const size_t n_keep = right_is_choi ? (n_op + m) : n_op;
        const size_t eff_keep = std::max<size_t>(1, n_keep);
        const size_t nw_keep = std::max<size_t>(1, (eff_keep + 63) >> 6);
        const size_t n_tot = std::max<size_t>(1, n_keep + 2 * m);
        const size_t nw_tot = std::max<size_t>(1, (n_tot + 63) >> 6);
        const size_t d_tot = d_L + d_R;

        const size_t n_pL = 2 * m;
        const size_t n_pR = right_is_choi ? (2 * m) : n_op;
        std::vector<int32_t> src_L(n_pL), dst_L(n_pL), src_R(n_pR), dst_R(n_pR);
        for (size_t r = 0; r < m; ++r) {
            src_L[r] = q_act[r];
            dst_L[r] = q_act[r];
            src_L[m + r] = static_cast<int32_t>(n_op + r);
            dst_L[m + r] = static_cast<int32_t>(n_keep + r);
            src_R[r] = q_act[r];
            dst_R[r] = static_cast<int32_t>(n_keep + m + r);
            if (right_is_choi) {
                src_R[m + r] = static_cast<int32_t>(n_op + r);
                dst_R[m + r] = static_cast<int32_t>(n_op + r);
            }
        }
        if (!right_is_choi) {
            size_t idle_pos = m;
            for (size_t q = 0; q < n_op; ++q) {
                if (!is_act[q]) {
                    src_R[idle_pos] = static_cast<int32_t>(q);
                    dst_R[idle_pos] = static_cast<int32_t>(q);
                    ++idle_pos;
                }
            }
        }

        auto scatter_row = [](
            const uint64_t* src, size_t nw_src, uint64_t* dst,
            const int32_t* s_bits, const int32_t* d_bits, size_t n_pairs
        ) noexcept {
            for (size_t i = 0; i < n_pairs; ++i) {
                const size_t sb = static_cast<size_t>(s_bits[i]);
                const size_t db = static_cast<size_t>(d_bits[i]);
                if ((sb >> 6) < nw_src && ((src[sb >> 6] >> (sb & 63U)) & 1ULL)) {
                    dst[db >> 6] ^= (uint64_t(1) << (db & 63U));
                }
            }
        };
        auto map_piv = [](
            int32_t p, const int32_t* s_bits, const int32_t* d_bits, size_t n_pairs
        ) noexcept -> int32_t {
            for (size_t i = n_pairs; i > 0; --i) {
                if (s_bits[i - 1] == p) return d_bits[i - 1];
            }
            return -1;
        };

        WorkShot work(
            prod_chi, d_tot, std::max<size_t>(1, d_tot), nw_tot,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
        );
        for (size_t r = 0; r < d_L; ++r) {
            if (A_L) scatter_row(&A_L[r * nw_L], nw_L, &work.A[r * nw_tot], src_L.data(), dst_L.data(), n_pL);
            if (G_L) std::memcpy(&work.G[r * work.cap], &G_L[r * d_L], d_L * sizeof(uint8_t));
            const int32_t mp = map_piv(piv_L[r], src_L.data(), dst_L.data(), n_pL);
            if (mp < 0) return 1;
            work.piv[r] = mp;
            work.eo[r] = eo_L[r];
        }
        for (size_t r = 0; r < d_R; ++r) {
            if (A_R) scatter_row(&A_R[r * nw_R], nw_R, &work.A[(d_L + r) * nw_tot], src_R.data(), dst_R.data(), n_pR);
            if (G_R) std::memcpy(&work.G[(d_L + r) * work.cap + d_L], &G_R[r * d_R], d_R * sizeof(uint8_t));
            const int32_t mp = map_piv(piv_R[r], src_R.data(), dst_R.data(), n_pR);
            if (mp < 0) return 1;
            work.piv[d_L + r] = mp;
            work.eo[d_L + r] = eo_R[r];
        }

        std::vector<uint64_t> hL_scat(chi_L * nw_tot, 0ULL), hR_scat(chi_R * nw_tot, 0ULL);
        if (h_L) {
            for (size_t a = 0; a < chi_L; ++a) {
                scatter_row(&h_L[a * nw_L], nw_L, &hL_scat[a * nw_tot], src_L.data(), dst_L.data(), n_pL);
            }
        }
        if (h_R) {
            for (size_t a = 0; a < chi_R; ++a) {
                scatter_row(&h_R[a * nw_R], nw_R, &hR_scat[a * nw_tot], src_R.data(), dst_R.data(), n_pR);
            }
        }
        for (size_t aL = 0; aL < chi_L; ++aL) {
            const double wL_re = w_ri_L[2 * aL];
            const double wL_im = w_ri_L[2 * aL + 1];
            const uint64_t* hL_row = &hL_scat[aL * nw_tot];
            for (size_t aR = 0; aR < chi_R; ++aR) {
                const size_t idx = aL * chi_R + aR;
                const double wR_re = w_ri_R[2 * aR];
                const double wR_im = w_ri_R[2 * aR + 1];
                work.w_ri[2 * idx] = wL_re * wR_re - wL_im * wR_im;
                work.w_ri[2 * idx + 1] = wL_re * wR_im + wL_im * wR_re;
                uint64_t* dst_h = &work.h[idx * nw_tot];
                const uint64_t* hR_row = &hR_scat[aR * nw_tot];
                for (size_t w = 0; w < nw_tot; ++w) {
                    dst_h[w] = hL_row[w] ^ hR_row[w];
                }
                if (d_L > 0 && ell_L) {
                    std::memcpy(&work.ell[idx * work.cap], &ell_L[aL * d_L], d_L * sizeof(uint8_t));
                }
                if (d_R > 0 && ell_R) {
                    std::memcpy(&work.ell[idx * work.cap + d_L], &ell_R[aR * d_R], d_R * sizeof(uint8_t));
                }
            }
        }

        std::vector<uint8_t> lam_buf(std::max<size_t>(1, d_tot), 0U);
        std::vector<uint8_t> eps_buf(prod_chi, 0U);
        for (size_t j = 0; j < m; ++j) {
            const int32_t qL = static_cast<int32_t>(n_keep + j);
            const int32_t qR = static_cast<int32_t>(n_keep + m + j);
            for (size_t r = 0; r < work.d; ++r) {
                lam_buf[r] = static_cast<uint8_t>(
                    get_bit(&work.A[r * nw_tot], qL) ^ get_bit(&work.A[r * nw_tot], qR)
                );
            }
            for (size_t a = 0; a < work.chi; ++a) {
                eps_buf[a] = static_cast<uint8_t>(
                    get_bit(&work.h[a * nw_tot], qL) ^ get_bit(&work.h[a * nw_tot], qR)
                );
            }
            work.restrict_amplitude(lam_buf.data(), eps_buf.data(), kInvSqrt2);
            if (work.chi == 0) {
                if (out_chi) *out_chi = 0;
                if (out_d) *out_d = work.d;
                return 0;
            }
            const size_t wL = static_cast<size_t>(qL) >> 6, sL = static_cast<size_t>(qL) & 63U;
            const size_t wR = static_cast<size_t>(qR) >> 6, sR = static_cast<size_t>(qR) & 63U;
            const uint64_t maskL = ~(uint64_t(1) << sL);
            const uint64_t maskR = ~(uint64_t(1) << sR);
            for (size_t r = 0; r < work.d; ++r) {
                work.A[r * nw_tot + wL] &= maskL;
                work.A[r * nw_tot + wR] &= maskR;
            }
            for (size_t a = 0; a < work.chi; ++a) {
                work.h[a * nw_tot + wL] &= maskL;
                work.h[a * nw_tot + wR] &= maskR;
            }
            for (size_t r0 = 0; r0 < work.d; ++r0) {
                if (work.piv[r0] == qL || work.piv[r0] == qR) {
                    const int32_t p_new = first_set_bit_in_words(&work.A[r0 * nw_tot], nw_tot);
                    if (p_new >= 0) {
                        work.make_principal(r0, p_new);
                    } else {
                        work.eliminate_zero_amplitude(r0);
                    }
                    break;
                }
            }
            if (work.chi == 0) {
                if (out_chi) *out_chi = 0;
                if (out_d) *out_d = work.d;
                return 0;
            }
        }

        const uint64_t tm_keep = tail_mask(eff_keep);
        if (nw_keep < nw_tot) {
            for (size_t r = 0; r < work.d; ++r) {
                std::memmove(&work.A[r * nw_keep], &work.A[r * nw_tot], nw_keep * sizeof(uint64_t));
            }
            for (size_t a = 0; a < work.chi; ++a) {
                std::memmove(&work.h[a * nw_keep], &work.h[a * nw_tot], nw_keep * sizeof(uint64_t));
            }
            work.nw = nw_keep;
        }
        for (size_t r = 0; r < work.d; ++r) {
            work.A[r * nw_keep + (nw_keep - 1)] &= tm_keep;
        }
        for (size_t a = 0; a < work.chi; ++a) {
            work.h[a * nw_keep + (nw_keep - 1)] &= tm_keep;
        }
        work.export_and_merge(
            cancel_tol, out_chi, out_d,
            out_weights_ri, out_h_words, out_ell,
            out_A_words, out_Gamma, out_pivots, out_ell_odd, nullptr
        );
        if ((m & 1U) != 0U && out_chi && *out_chi > 0) {
            const size_t oc = *out_chi;
            for (size_t a = 0; a < 2 * oc; ++a) {
                out_weights_ri[a] *= kSqrt2;
            }
        }
        return 0;
    } catch (const std::bad_alloc&) {
        // OOM policy: report failure code 2 with empty outputs; this handler allocates nothing,
        // so it is safe to run while the allocator is exhausted.
        if (out_chi) *out_chi = 0;
        if (out_d) *out_d = 0;
        return 2;
    } catch (...) {
        if (out_chi) *out_chi = 0;
        if (out_d) *out_d = 0;
        return 3;
    }
}

// The C++ half of `measure_expansion` (phase_qfe.py) for many shots measuring one observable,
// calling the per-shot entry points above. Shot s is packed tightly (shot_chi[s] rows, geometry
// shot_d[s]); output slot s has room for shot_chi[s] rows and geometry shot_d[s] + 1. Per shot:
// if has_rot, merge (non-canonical input) and quarter-turn, then check unitarity against z_in[s];
// then project onto outcome m (out_z[2s + m] = z_m). outcome == nullptr projects both outcomes
// (slot s keeps m = 1); otherwise only outcome[s]. out_cd[2s..2s+1] = (chi, d) of slot s.
// Returns -1, or the index of the first shot whose rotation fails.
int64_t magic_measure_batch(
    size_t n_shots, size_t n_qubits, size_t nw, const int64_t* shot_chi, const int64_t* shot_d,
    const uint8_t* shot_canon, const double* z_in, const double* w, const uint64_t* h,
    const uint8_t* ell, const uint64_t* A, const uint8_t* G, const int32_t* piv, const uint8_t* eo,
    int has_rot, const uint64_t* rot_a, const uint64_t* rot_b, uint8_t rot_kappa,
    const uint64_t* diag_b, uint8_t diag_kappa, double cancel_tol, double atol,
    const uint8_t* outcome, double* out_z, int64_t* out_cd, double* out_w, uint64_t* out_h,
    uint8_t* out_ell, uint64_t* out_A, uint8_t* out_G, int32_t* out_piv, uint8_t* out_eo,
    uint64_t* out_a_union
) {
    std::vector<double> mw, rw;
    std::vector<uint64_t> mh, rh, rA;
    std::vector<uint8_t> mell, rell, rG, reo;
    std::vector<int32_t> rpiv;
    for (size_t s = 0; s < n_shots; ++s) {
        const size_t chi0 = static_cast<size_t>(shot_chi[s]), d0 = static_cast<size_t>(shot_d[s]);
        const size_t cap = d0 + 1;
        size_t chi = chi0, d = d0;
        const double* cw = w;
        const uint64_t *ch = h, *cA = A;
        const uint8_t *cl = ell, *cG = G, *ce = eo;
        const int32_t* cp = piv;
        w += 2 * chi0, h += chi0 * nw, ell += chi0 * d0, A += d0 * nw, G += d0 * d0;
        piv += d0, eo += d0;
        if (has_rot) {
            if (!shot_canon[s]) {
                mw.resize(2 * chi0), mh.resize(chi0 * nw), mell.resize(chi0 * d0);
                magic_canonicalize_and_merge(
                    chi0, d0, nw, cw, ch, cl, cA, cG, cp, ce, cancel_tol, &chi,
                    mw.data(), mh.data(), mell.data(), nullptr
                );
                cw = mw.data(), ch = mh.data(), cl = mell.data();
            }
            double post = 0.0;  // ShotExpansion.norm_sq of the rotated state (sequential sum)
            if (chi > 0) {
                rw.resize(2 * chi), rh.resize(chi * nw), rell.resize(chi * cap);
                rA.resize(cap * nw), rG.resize(cap * cap), rpiv.resize(cap), reo.resize(cap);
                if (magic_apply_quarter_turn_tol(
                        chi, d, n_qubits, nw, cw, ch, cl, cA, cG, cp, ce, rot_a, rot_b, rot_kappa,
                        cancel_tol,
                        &chi, &d, rw.data(), rh.data(), rell.data(), rA.data(), rG.data(),
                        rpiv.data(), reo.data()
                    ) != 0) {
                    return static_cast<int64_t>(s);
                }
                cw = rw.data(), ch = rh.data(), cl = rell.data(), cA = rA.data(), cG = rG.data();
                cp = rpiv.data(), ce = reo.data();
                for (size_t i = 0; i < 2 * chi; i += 2) {
                    post += cw[i] * cw[i] + cw[i + 1] * cw[i + 1];
                }
            }
            const double pre = z_in[s];  // check_unitary_norm
            if (!(std::isfinite(post) && std::abs(post - pre) <= atol * std::max(1.0, pre))) {
                return static_cast<int64_t>(s);
            }
        }
        size_t oc = 0, od = 0;
        for (uint8_t m = 0; m < 2; ++m) {
            if (outcome != nullptr && outcome[s] != m) continue;
            double z = 0.0;
            oc = od = 0;
            if (chi > 0) {
                magic_project_diagonal(
                    chi, d, nw, cw, ch, cl, cA, cG, cp, ce, diag_b, diag_kappa, m, cancel_tol,
                    &oc, &od, out_w, out_h, out_ell, out_A, out_G, out_piv, out_eo, &z
                );
            }
            out_z[2 * s + m] = z;
        }
        out_cd[2 * s] = static_cast<int64_t>(oc), out_cd[2 * s + 1] = static_cast<int64_t>(od);
        for (size_t i = 0; i < nw; ++i) {
            uint64_t u = 0;
            for (size_t r = 0; r < od; ++r) u |= out_A[r * nw + i];
            out_a_union[i] = u;
        }
        out_w += 2 * chi0, out_h += chi0 * nw, out_ell += chi0 * cap, out_A += cap * nw;
        out_G += cap * cap, out_piv += cap, out_eo += cap, out_a_union += nw;
    }
    return -1;
}

}  // extern "C"

// MagicBatchEngine (WP-25): mbe_* entry points; included last, it calls the kernels above.
#include "magic_engine.hpp"
