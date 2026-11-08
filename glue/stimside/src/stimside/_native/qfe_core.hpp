// Copyright 2026 Google LLC
//
// Quadratic-form-expansion (QFE) core of the Magicside C++ kernels: bit-word helpers,
// canonical-row hashing / frame merging, and the per-shot WorkShot algebra (geometry
// d/A/Gamma/pivots/eo and the H0-H4 updates). Included by magic_kernels.cpp.
//
// Everything lives in an ANONYMOUS namespace on purpose: each translation unit gets its own
// internal-linkage copy, so the header adds no exported symbols and the compiled
// libmagic_kernels.so is byte-identical to the one built before this code was moved here
// (verified with cmp). Do not rename the namespace, add `inline`, or edit the moved text
// without re-checking that.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

inline int popcount64(uint64_t x) noexcept {
    return __builtin_popcountll(x);
}

inline int ctz64(uint64_t x) noexcept {
    return __builtin_ctzll(x);
}

// Exact IEEE-754 float64 constants matching phase_qfe.py:91-118 to 0 ULPs.
const double kSqrt2 = std::sqrt(2.0);
const double kInvSqrt2 = 0.5 * kSqrt2;

const double kTC0Re = 0.5 + 0.5 * kInvSqrt2;
const double kTC0Im = 0.5 * kInvSqrt2;
const double kTC1Re = 0.5 - 0.5 * kInvSqrt2;
const double kTC1Im = -0.5 * kInvSqrt2;

const double kTDagC0Re = 0.5 + 0.5 * kInvSqrt2;
const double kTDagC0Im = -0.5 * kInvSqrt2;
const double kTDagC1Re = 0.5 - 0.5 * kInvSqrt2;
const double kTDagC1Im = 0.5 * kInvSqrt2;

inline void mul_phase4(double& re, double& im, uint8_t q) noexcept {
    const double r = re;
    const double i = im;
    switch (q & 3U) {
        case 0: break;
        case 1: re = -i; im =  r; break;
        case 2: re = -r; im = -i; break;
        case 3: re =  i; im = -r; break;
    }
}

inline uint64_t tail_mask(size_t n_qubits) noexcept {
    if (n_qubits == 0) return 0ULL;
    const size_t rem = n_qubits & 63U;
    return rem == 0 ? ~uint64_t(0) : ((uint64_t(1) << rem) - 1ULL);
}

inline uint8_t get_bit(const uint64_t* words, int32_t q) noexcept {
    const size_t uq = static_cast<size_t>(q);
    return static_cast<uint8_t>((words[uq >> 6] >> (uq & 63U)) & 1ULL);
}

inline uint8_t word_dot_parity(const uint64_t* u, const uint64_t* v, size_t nw) noexcept {
    if (!u || !v || nw == 0) return 0U;
    if (nw == 1) return static_cast<uint8_t>(popcount64(u[0] & v[0]) & 1);
    uint64_t acc = 0ULL;
    for (size_t w = 0; w < nw; ++w) {
        acc ^= (u[w] & v[w]);
    }
    return static_cast<uint8_t>(popcount64(acc) & 1);
}

inline bool words_any_nonzero(const uint64_t* u, size_t nw) noexcept {
    if (!u || nw == 0) return false;
    if (nw == 1) return u[0] != 0ULL;
    for (size_t w = 0; w < nw; ++w) {
        if (u[w] != 0ULL) return true;
    }
    return false;
}

inline bool words_any_overlap(const uint64_t* u, const uint64_t* v, size_t nw) noexcept {
    if (!u || !v || nw == 0) return false;
    if (nw == 1) return (u[0] & v[0]) != 0ULL;
    for (size_t w = 0; w < nw; ++w) {
        if ((u[w] & v[w]) != 0ULL) return true;
    }
    return false;
}

inline void xor_words_inplace(uint64_t* dst, const uint64_t* src, size_t nw) noexcept {
    if (nw == 1) {
        dst[0] ^= src[0];
        return;
    }
    for (size_t w = 0; w < nw; ++w) {
        dst[w] ^= src[w];
    }
}

inline int32_t first_set_bit_in_words(const uint64_t* words, size_t nw) noexcept {
    for (size_t w = 0; w < nw; ++w) {
        if (words[w] != 0ULL) {
            return static_cast<int32_t>((w << 6) + static_cast<size_t>(ctz64(words[w])));
        }
    }
    return -1;
}

inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t hash_frame_key(
    const uint64_t* h, size_t nw, const uint8_t* ell, size_t d
) noexcept {
    uint64_t acc = splitmix64(nw ^ (d << 32));
    for (size_t w = 0; w < nw; ++w) {
        acc ^= splitmix64(h[w] + 0x9e3779b97f4a7c15ULL * (w + 1));
    }
    size_t idx = 0;
    while (idx + 8 <= d) {
        uint64_t chunk = 0;
        std::memcpy(&chunk, ell + idx, 8);
        acc ^= splitmix64(chunk + idx);
        idx += 8;
    }
    if (idx < d) {
        uint64_t tail = 0;
        for (size_t r = idx; r < d; ++r) {
            tail |= static_cast<uint64_t>(ell[r]) << ((r - idx) * 8);
        }
        acc ^= splitmix64(tail + idx + 1);
    }
    return acc;
}

// K1 row canonicalization clearing dirty pivot bits on h_words (phase_qfe.py:1908-1927).
inline void canonicalize_row(
    double& re,
    double& im,
    uint64_t* cur_h,
    uint8_t* cur_ell,
    size_t nw,
    size_t d,
    const uint64_t* A_words,
    const uint8_t* Gamma,
    const int32_t* pivots,
    const uint8_t* ell_odd,
    uint8_t* s_buf,
    uint8_t* polar_buf
) noexcept {
    if (d == 0) return;
    bool any_dirty = false;
    for (size_t r = 0; r < d; ++r) {
        const uint8_t bit = get_bit(cur_h, pivots[r]);
        s_buf[r] = bit;
        any_dirty |= (bit != 0);
    }
    if (!any_dirty) return;

    for (size_t c = 0; c < d; ++c) {
        polar_buf[c] = static_cast<uint8_t>(s_buf[c] & (ell_odd[c] & 1U));
    }
    uint32_t lin = 0;
    uint8_t quad_par = 0;
    for (size_t r = 0; r < d; ++r) {
        if (!s_buf[r]) continue;
        lin += cur_ell[r];
        const uint8_t* g_row = &Gamma[r * d];
        for (size_t c = r + 1; c < d; ++c) {
            quad_par ^= static_cast<uint8_t>(s_buf[c] & g_row[c]);
        }
        xor_words_inplace(cur_h, &A_words[r * nw], nw);
        for (size_t c = 0; c < d; ++c) {
            polar_buf[c] ^= g_row[c];
        }
    }
    const uint8_t q_mod4 = static_cast<uint8_t>((lin + (uint32_t(quad_par & 1U) << 1)) & 3U);
    mul_phase4(re, im, q_mod4);
    for (size_t c = 0; c < d; ++c) {
        cur_ell[c] = static_cast<uint8_t>((cur_ell[c] + ((polar_buf[c] & 1U) << 1)) & 3U);
    }
}

// Open-addressed linear-probing K1 merger preserving exact first-seen group order.
struct FrameMerger {
    static constexpr size_t kEmpty = ~size_t(0);

    size_t nw;
    size_t d;
    const uint64_t* A_words;
    const uint8_t* Gamma;
    const int32_t* pivots;
    const uint8_t* ell_odd;

    double* out_w_ri;
    uint64_t* out_h;
    uint8_t* out_ell;

    size_t mask = 0;
    size_t n_groups = 0;
    double w_max = 0.0;

    size_t slots_stack[512];
    std::vector<size_t> slots_heap;
    size_t* slots = nullptr;

    uint64_t h_stack[16];
    std::vector<uint64_t> h_heap;
    uint64_t* tmp_h = nullptr;

    uint8_t d_stack[768];
    std::vector<uint8_t> d_heap;
    uint8_t* tmp_ell = nullptr;
    uint8_t* s_buf = nullptr;
    uint8_t* polar_buf = nullptr;

    FrameMerger(
        size_t max_rows,
        size_t d_,
        size_t nw_,
        const uint64_t* A_words_,
        const uint8_t* Gamma_,
        const int32_t* pivots_,
        const uint8_t* ell_odd_,
        double* out_w_ri_,
        uint64_t* out_h_,
        uint8_t* out_ell_
    )
        : nw(nw_),
          d(d_),
          A_words(A_words_),
          Gamma(Gamma_),
          pivots(pivots_),
          ell_odd(ell_odd_),
          out_w_ri(out_w_ri_),
          out_h(out_h_),
          out_ell(out_ell_) {
        if (max_rows == 0) return;
        size_t cap = 16;
        while (cap < 2 * max_rows) cap <<= 1;
        mask = cap - 1;
        if (cap <= 512) {
            slots = slots_stack;
            std::fill_n(slots, cap, kEmpty);
        } else {
            slots_heap.assign(cap, kEmpty);
            slots = slots_heap.data();
        }
        if (nw <= 16) {
            tmp_h = h_stack;
        } else {
            h_heap.resize(nw);
            tmp_h = h_heap.data();
        }
        if (d > 0) {
            if (d <= 256) {
                tmp_ell = d_stack;
            } else {
                d_heap.resize(3 * d);
                tmp_ell = d_heap.data();
            }
            s_buf = tmp_ell + d;
            polar_buf = tmp_ell + 2 * d;
        }
    }

    inline void push_staged(double re, double im) noexcept {
        w_max = std::max(w_max, std::hypot(re, im));
        canonicalize_row(
            re, im, tmp_h, tmp_ell, nw, d,
            A_words, Gamma, pivots, ell_odd, s_buf, polar_buf
        );
        size_t slot = static_cast<size_t>(hash_frame_key(tmp_h, nw, tmp_ell, d)) & mask;
        for (;;) {
            const size_t g = slots[slot];
            if (g == kEmpty) {
                const size_t ng = n_groups++;
                slots[slot] = ng;
                std::memcpy(&out_h[ng * nw], tmp_h, nw * sizeof(uint64_t));
                if (d > 0) {
                    std::memcpy(&out_ell[ng * d], tmp_ell, d * sizeof(uint8_t));
                }
                out_w_ri[2 * ng] = re;
                out_w_ri[2 * ng + 1] = im;
                return;
            }
            if (std::memcmp(&out_h[g * nw], tmp_h, nw * sizeof(uint64_t)) == 0 &&
                (d == 0 || std::memcmp(&out_ell[g * d], tmp_ell, d * sizeof(uint8_t)) == 0)) {
                out_w_ri[2 * g] += re;
                out_w_ri[2 * g + 1] += im;
                return;
            }
            slot = (slot + 1) & mask;
        }
    }

    inline void push_copy(
        double re, double im, const uint64_t* row_h, const uint8_t* row_ell
    ) noexcept {
        std::memcpy(tmp_h, row_h, nw * sizeof(uint64_t));
        if (d > 0) {
            std::memcpy(tmp_ell, row_ell, d * sizeof(uint8_t));
        }
        push_staged(re, im);
    }

    inline void finalize(double cancel_tol, size_t* out_chi, double* out_norm_sq) noexcept {
        const double thresh = cancel_tol * std::min(1.0, w_max);
        size_t keep = 0;
        double norm_sq = 0.0;
        for (size_t g = 0; g < n_groups; ++g) {
            const double re = out_w_ri[2 * g];
            const double im = out_w_ri[2 * g + 1];
            if (std::hypot(re, im) > thresh) {
                if (keep != g) {
                    out_w_ri[2 * keep] = re;
                    out_w_ri[2 * keep + 1] = im;
                    std::memcpy(&out_h[keep * nw], &out_h[g * nw], nw * sizeof(uint64_t));
                    if (d > 0) {
                        std::memcpy(&out_ell[keep * d], &out_ell[g * d], d * sizeof(uint8_t));
                    }
                }
                norm_sq += re * re + im * im;
                ++keep;
            }
        }
        if (out_chi) *out_chi = keep;
        if (out_norm_sq) *out_norm_sq = norm_sq;
    }
};

// Shared Pauli-sum branching + K1 merger in branch-major (m, a) order (N1 & N2).
void branch_pauli_sum_and_merge(
    size_t r_branches,
    size_t chi,
    size_t d,
    size_t nw,
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
    if (chi == 0 || r_branches == 0 || nw == 0 ||
        !branch_weights_ri || !branch_a_words || !branch_b_words || !branch_kappas) {
        if (out_chi) *out_chi = 0;
        if (out_norm_sq) *out_norm_sq = 0.0;
        return;
    }
    FrameMerger merger(
        r_branches * chi, d, nw,
        A_words, Gamma, pivots, ell_odd,
        out_weights_ri, out_h_words, out_ell
    );
    uint8_t t_stack[256];
    std::vector<uint8_t> t_heap;
    uint8_t* t_shift = nullptr;
    if (d > 0) {
        if (d <= 256) {
            t_shift = t_stack;
        } else {
            t_heap.resize(d);
            t_shift = t_heap.data();
        }
    }

    for (size_t m = 0; m < r_branches; ++m) {
        const double cm_re = branch_weights_ri[2 * m];
        const double cm_im = branch_weights_ri[2 * m + 1];
        const uint64_t* am = &branch_a_words[m * nw];
        const uint64_t* bm = &branch_b_words[m * nw];
        const uint8_t km = branch_kappas[m] & 3U;
        const bool has_a = words_any_nonzero(am, nw);
        const bool has_b = words_any_nonzero(bm, nw);

        bool any_t = false;
        if (d > 0 && has_b && (!a_union || words_any_overlap(bm, a_union, nw))) {
            for (size_t r = 0; r < d; ++r) {
                const uint8_t tr = static_cast<uint8_t>(
                    word_dot_parity(&A_words[r * nw], bm, nw) << 1
                );
                t_shift[r] = tr;
                any_t |= (tr != 0);
            }
        }

        for (size_t a = 0; a < chi; ++a) {
            const uint64_t* ha = &h_words[a * nw];
            const uint8_t bh = has_b ? word_dot_parity(ha, bm, nw) : 0U;
            const uint8_t p_idx = static_cast<uint8_t>((km + (bh << 1)) & 3U);

            const double wa_re = weights_ri[2 * a];
            const double wa_im = weights_ri[2 * a + 1];
            double re = cm_re * wa_re - cm_im * wa_im;
            double im = cm_re * wa_im + cm_im * wa_re;
            mul_phase4(re, im, p_idx);

            if (has_a) {
                for (size_t w = 0; w < nw; ++w) merger.tmp_h[w] = ha[w] ^ am[w];
            } else {
                std::memcpy(merger.tmp_h, ha, nw * sizeof(uint64_t));
            }
            if (d > 0) {
                const uint8_t* ea = &ell[a * d];
                if (any_t) {
                    for (size_t r = 0; r < d; ++r) {
                        merger.tmp_ell[r] = static_cast<uint8_t>((ea[r] + t_shift[r]) & 3U);
                    }
                } else {
                    std::memcpy(merger.tmp_ell, ea, d * sizeof(uint8_t));
                }
            }
            merger.push_staged(re, im);
        }
    }
    merger.finalize(cancel_tol, out_chi, out_norm_sq);
}

// Structural QFE geometry mutator implementing H0-H4 + tight export & K1 merge (shared by P2 & M1).
struct WorkShot {
    size_t chi;
    size_t d;
    size_t cap;
    size_t nw;

    std::vector<double> w_ri;
    std::vector<uint64_t> h;
    std::vector<uint8_t> ell;
    std::vector<uint64_t> A;
    std::vector<uint8_t> G;
    std::vector<int32_t> piv;
    std::vector<uint8_t> eo;

    WorkShot(
        size_t chi_,
        size_t d_,
        size_t cap_,
        size_t nw_,
        const double* in_w_ri,
        const uint64_t* in_h,
        const uint8_t* in_ell,
        const uint64_t* in_A,
        const uint8_t* in_G,
        const int32_t* in_piv,
        const uint8_t* in_eo
    )
        : chi(chi_),
          d(d_),
          cap(std::max<size_t>(1, cap_)),
          nw(nw_),
          w_ri(2 * chi_, 0.0),
          h(chi_ * nw_, 0ULL),
          ell(chi_ * std::max<size_t>(1, cap_), 0U),
          A(std::max<size_t>(1, cap_) * nw_, 0ULL),
          G(std::max<size_t>(1, cap_) * std::max<size_t>(1, cap_), 0U),
          piv(std::max<size_t>(1, cap_), -1),
          eo(std::max<size_t>(1, cap_), 0U) {
        if (chi > 0) {
            if (in_w_ri) {
                std::memcpy(w_ri.data(), in_w_ri, 2 * chi * sizeof(double));
            }
            if (nw > 0 && in_h) {
                std::memcpy(h.data(), in_h, chi * nw * sizeof(uint64_t));
            }
            if (d > 0 && in_ell) {
                for (size_t a = 0; a < chi; ++a) {
                    std::memcpy(&ell[a * cap], &in_ell[a * d], d * sizeof(uint8_t));
                }
            }
        }
        if (d > 0) {
            if (nw > 0 && in_A) {
                std::memcpy(A.data(), in_A, d * nw * sizeof(uint64_t));
            }
            if (in_G) {
                for (size_t r = 0; r < d; ++r) {
                    std::memcpy(&G[r * cap], &in_G[r * d], d * sizeof(uint8_t));
                }
            }
            if (in_piv) {
                std::memcpy(piv.data(), in_piv, d * sizeof(int32_t));
            }
            if (in_eo) {
                std::memcpy(eo.data(), in_eo, d * sizeof(uint8_t));
            }
        }
    }

    // H0: Delete binary generator column `col` in-place (d -> d - 1).
    void drop_variable(size_t col) noexcept {
        for (size_t r = col; r + 1 < d; ++r) {
            if (nw > 0) {
                std::memcpy(&A[r * nw], &A[(r + 1) * nw], nw * sizeof(uint64_t));
            }
            piv[r] = piv[r + 1];
            eo[r] = eo[r + 1];
            for (size_t c = 0; c < d; ++c) {
                G[r * cap + c] = G[(r + 1) * cap + c];
            }
        }
        for (size_t r = 0; r + 1 < d; ++r) {
            for (size_t c = col; c + 1 < d; ++c) {
                G[r * cap + c] = G[r * cap + c + 1];
            }
        }
        for (size_t a = 0; a < chi; ++a) {
            uint8_t* row = &ell[a * cap];
            for (size_t c = col; c + 1 < d; ++c) {
                row[c] = row[c + 1];
            }
        }
        --d;
    }

    // H1: Column addition A_s ^= A_r with exact mod-4 Q pullback.
    void add_column(size_t r, size_t s) noexcept {
        const uint8_t old_edge = G[r * cap + s] & 1U;
        const uint8_t parity = eo[r] & 1U;
        if (nw > 0) {
            xor_words_inplace(&A[s * nw], &A[r * nw], nw);
        }
        const uint8_t two_old = static_cast<uint8_t>(old_edge << 1);
        for (size_t a = 0; a < chi; ++a) {
            uint8_t* row = &ell[a * cap];
            row[s] = static_cast<uint8_t>((row[s] + row[r] + two_old) & 3U);
        }
        eo[s] ^= parity;
        for (size_t k = 0; k < d; ++k) {
            if (k != r && k != s) {
                const uint8_t ne = static_cast<uint8_t>((G[s * cap + k] ^ G[r * cap + k]) & 1U);
                G[s * cap + k] = ne;
                G[k * cap + s] = ne;
            }
        }
        const uint8_t new_rs = static_cast<uint8_t>(old_edge ^ parity);
        G[r * cap + s] = new_rs;
        G[s * cap + r] = new_rs;
    }

    // H2: Clear bit `p` from all columns != col and record pivot.
    void make_principal(size_t col, int32_t p) noexcept {
        for (size_t j = 0; j < d; ++j) {
            if (j != col && get_bit(&A[j * nw], p) != 0) {
                add_column(col, j);
            }
        }
        piv[col] = p;
    }

    void filter_eps_zero(const uint8_t* eps_vec, double keep_scale) noexcept {
        size_t keep = 0;
        for (size_t a = 0; a < chi; ++a) {
            if ((eps_vec[a] & 1U) == 0U) {
                if (keep != a) {
                    w_ri[2 * keep] = w_ri[2 * a];
                    w_ri[2 * keep + 1] = w_ri[2 * a + 1];
                    if (nw > 0) {
                        std::memcpy(&h[keep * nw], &h[a * nw], nw * sizeof(uint64_t));
                    }
                    if (d > 0) {
                        std::memcpy(&ell[keep * cap], &ell[a * cap], d * sizeof(uint8_t));
                    }
                }
                w_ri[2 * keep] *= keep_scale;
                w_ri[2 * keep + 1] *= keep_scale;
                ++keep;
            }
        }
        chi = keep;
    }

    // H3: Restrict to affine hyperplane lam . u = eps_vec[a] (mod 2).
    void restrict_amplitude(const uint8_t* lam, const uint8_t* eps_vec, double scale) noexcept {
        size_t f = d;
        for (size_t r = 0; r < d; ++r) {
            if (lam[r] & 1U) {
                f = r;
                break;
            }
        }
        if (f == d) {
            filter_eps_zero(eps_vec, 1.0);
            return;
        }
        for (size_t k = f + 1; k < d; ++k) {
            if (lam[k] & 1U) {
                add_column(f, k);
            }
        }
        const uint64_t* Af = &A[f * nw];
        const uint8_t* Gf = &G[f * cap];
        for (size_t a = 0; a < chi; ++a) {
            uint8_t* row = &ell[a * cap];
            const uint8_t eps_a = eps_vec[a] & 1U;
            const uint8_t p_idx = static_cast<uint8_t>((eps_a * row[f]) & 3U);
            double re = w_ri[2 * a];
            double im = w_ri[2 * a + 1];
            mul_phase4(re, im, p_idx);
            w_ri[2 * a] = re * scale;
            w_ri[2 * a + 1] = im * scale;
            if (eps_a) {
                if (nw > 0) {
                    xor_words_inplace(&h[a * nw], Af, nw);
                }
                for (size_t c = 0; c < d; ++c) {
                    row[c] = static_cast<uint8_t>((row[c] + ((Gf[c] & 1U) << 1)) & 3U);
                }
            }
        }
        drop_variable(f);
    }

    // H4: Analytically contract zero column `col` (A[col] == 0) in-place.
    void eliminate_zero_amplitude(size_t col) {
        std::vector<uint8_t> lam(d > 0 ? d - 1 : 0, 0U);
        size_t pos = 0;
        for (size_t c = 0; c < d; ++c) {
            if (c != col) lam[pos++] = G[col * cap + c] & 1U;
        }
        const uint8_t c_odd = eo[col] & 1U;
        std::vector<uint8_t> c_vec(chi, 0U);
        for (size_t a = 0; a < chi; ++a) {
            c_vec[a] = ell[a * cap + col] & 3U;
        }
        drop_variable(col);

        if (c_odd == 1U) {
            for (size_t a = 0; a < chi; ++a) {
                const double fr = kInvSqrt2;
                const double fi = (c_vec[a] == 1U) ? kInvSqrt2 : -kInvSqrt2;
                const double re = w_ri[2 * a];
                const double im = w_ri[2 * a + 1];
                w_ri[2 * a] = re * fr - im * fi;
                w_ri[2 * a + 1] = re * fi + im * fr;
                if (d > 0) {
                    const uint8_t neg_c = static_cast<uint8_t>((4U - c_vec[a]) & 3U);
                    uint8_t* row = &ell[a * cap];
                    for (size_t r = 0; r < d; ++r) {
                        row[r] = static_cast<uint8_t>((row[r] + neg_c * lam[r]) & 3U);
                    }
                }
            }
            for (size_t r = 0; r < d; ++r) {
                eo[r] ^= lam[r];
                if (lam[r]) {
                    for (size_t s = 0; s < d; ++s) {
                        if (s != r && lam[s]) G[r * cap + s] ^= 1U;
                    }
                }
            }
            return;
        }

        for (size_t a = 0; a < chi; ++a) {
            c_vec[a] = (c_vec[a] >> 1) & 1U;
        }
        bool any_lam = false;
        for (size_t r = 0; r < d; ++r) any_lam |= (lam[r] != 0U);
        if (any_lam) {
            restrict_amplitude(lam.data(), c_vec.data(), 1.0);
        } else {
            filter_eps_zero(c_vec.data(), kSqrt2);
        }
    }

    // Export tightly-strided geometry (stride = out_d) + K1 canonicalization & frame merge.
    void export_and_merge(
        double cancel_tol,
        size_t* out_chi,
        size_t* out_d,
        double* out_w_ri,
        uint64_t* out_h,
        uint8_t* out_ell,
        uint64_t* out_A,
        uint8_t* out_G,
        int32_t* out_piv,
        uint8_t* out_eo,
        double* out_norm_sq
    ) {
        if (out_d) *out_d = d;
        if (d > 0) {
            if (nw > 0 && out_A) {
                std::memcpy(out_A, A.data(), d * nw * sizeof(uint64_t));
            }
            if (out_G) {
                for (size_t r = 0; r < d; ++r) {
                    std::memcpy(&out_G[r * d], &G[r * cap], d * sizeof(uint8_t));
                }
            }
            if (out_piv) {
                std::memcpy(out_piv, piv.data(), d * sizeof(int32_t));
            }
            if (out_eo) {
                std::memcpy(out_eo, eo.data(), d * sizeof(uint8_t));
            }
        }
        if (chi == 0 || nw == 0) {
            if (out_chi) *out_chi = 0;
            if (out_norm_sq) *out_norm_sq = 0.0;
            return;
        }
        FrameMerger merger(
            chi, d, nw, out_A, out_G, out_piv, out_eo, out_w_ri, out_h, out_ell
        );
        for (size_t a = 0; a < chi; ++a) {
            merger.push_copy(
                w_ri[2 * a], w_ri[2 * a + 1],
                &h[a * nw], d > 0 ? &ell[a * cap] : nullptr
            );
        }
        merger.finalize(cancel_tol, out_chi, out_norm_sq);
    }
};

}  // namespace
