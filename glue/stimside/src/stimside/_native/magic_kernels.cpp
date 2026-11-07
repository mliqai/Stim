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

int magic_apply_quarter_turn(
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
        1e-13, out_chi, out_d,  // Must equal K1_CANCEL_REL_TOL (_magic/phase_qfe.py).
        out_weights_ri, out_h_words, out_ell,
        out_A_words, out_Gamma, out_pivots, out_ell_odd, nullptr
    );
    return 0;
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
                if (magic_apply_quarter_turn(
                        chi, d, n_qubits, nw, cw, ch, cl, cA, cG, cp, ce, rot_a, rot_b, rot_kappa,
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
