// Copyright 2026 Google LLC
//
// C++17 shared-library kernels accelerating coherent Phase-QFE state evolution,
// canonical coset merging, non-Clifford branching, diagonal projection, and
// exact Z_4 Gauss-sum / overlap contractions across single- and multi-word bitpackings.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

inline int popcount64(uint64_t x) noexcept {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

inline int ctz64(uint64_t x) noexcept {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(x);
#endif
}

// Exact IEEE-754 float64 constants matching phase_qfe.py:91-118 to 0 ULPs.
const double kSqrt2 = std::sqrt(2.0);
const double kInvSqrt2 = 0.5 * kSqrt2;
const double kOmegaRe[8] = {1.0, kInvSqrt2, 0.0, -kInvSqrt2, -1.0, -kInvSqrt2, 0.0, kInvSqrt2};
const double kOmegaIm[8] = {0.0, kInvSqrt2, 1.0, kInvSqrt2, 0.0, -kInvSqrt2, -1.0, -kInvSqrt2};

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

// O2 internal evaluator on bitpacked symmetric hollow binary matrix M (d rows x mw uint64_t words).
void gauss_sum_bitpacked(
    size_t d,
    uint8_t* l,
    uint64_t* M,
    size_t mw,
    int32_t q0,
    int64_t scale_log2,
    double* out_ri
) noexcept {
    size_t k = d;
    int64_t k2 = 0;
    int v8 = static_cast<int>(((static_cast<uint32_t>(q0) & 3U) << 1) & 7U);

    auto swap_var = [&](size_t i, size_t j, size_t cur_k) noexcept {
        if (i == j) return;
        std::swap(l[i], l[j]);
        const size_t aw = (cur_k + 63) >> 6;
        uint64_t* ri = &M[i * mw];
        uint64_t* rj = &M[j * mw];
        for (size_t w = 0; w < aw; ++w) {
            std::swap(ri[w], rj[w]);
        }
        const size_t wi = i >> 6, bi = i & 63U;
        const size_t wj = j >> 6, bj = j & 63U;
        if (wi == wj) {
            for (size_t r = 0; r < cur_k; ++r) {
                uint64_t word = M[r * mw + wi];
                const uint64_t diff = ((word >> bi) ^ (word >> bj)) & 1ULL;
                M[r * mw + wi] = word ^ (diff << bi) ^ (diff << bj);
            }
        } else {
            for (size_t r = 0; r < cur_k; ++r) {
                uint64_t& a = M[r * mw + wi];
                uint64_t& b = M[r * mw + wj];
                const uint64_t diff = ((a >> bi) ^ (b >> bj)) & 1ULL;
                a ^= (diff << bi);
                b ^= (diff << bj);
            }
        }
    };

    while (k > 0) {
        size_t f_odd = k;
        for (size_t i = 0; i < k; ++i) {
            if (l[i] & 1U) {
                f_odd = i;
                break;
            }
        }
        if (f_odd < k) {
            swap_var(f_odd, k - 1, k);
            const uint8_t c = l[k - 1] & 3U;
            k2 += 1;
            v8 = (v8 + (c == 1U ? 1 : 7)) & 7;
            if (k > 1) {
                const size_t k1 = k - 1;
                const size_t nw1 = (k1 + 63) >> 6;
                const uint64_t* m_row = &M[k1 * mw];
                const uint8_t neg_c = (c == 1U) ? 3U : 1U;
                for (size_t r = 0; r < k1; ++r) {
                    if ((m_row[r >> 6] >> (r & 63U)) & 1ULL) {
                        l[r] = static_cast<uint8_t>((l[r] + neg_c) & 3U);
                        uint64_t* row_r = &M[r * mw];
                        for (size_t w = 0; w < nw1; ++w) {
                            row_r[w] ^= m_row[w];
                        }
                        row_r[r >> 6] &= ~(1ULL << (r & 63U));
                    }
                }
            }
            k -= 1;
            continue;
        }

        const size_t nw_k = (k + 63) >> 6;
        const uint64_t last_mask = (k & 63U) ? ((1ULL << (k & 63U)) - 1ULL) : ~0ULL;
        size_t f_nz = k, p_nz = k;
        for (size_t r = 0; r < k; ++r) {
            const uint64_t* row_r = &M[r * mw];
            for (size_t w = 0; w < nw_k; ++w) {
                const uint64_t val = row_r[w] & ((w + 1 == nw_k) ? last_mask : ~0ULL);
                if (val != 0ULL) {
                    f_nz = r;
                    p_nz = (w << 6) + static_cast<size_t>(ctz64(val));
                    break;
                }
            }
            if (f_nz < k) break;
        }
        if (f_nz < k) {
            if (f_nz > p_nz) std::swap(f_nz, p_nz);
            // Collision-free swap of (f_nz < p_nz) into (k - 2, k - 1): swap p_nz <-> k-1 FIRST!
            swap_var(p_nz, k - 1, k);
            swap_var(f_nz, k - 2, k);
            const uint64_t lf = (l[k - 2] >> 1) & 1ULL;
            const uint64_t lp = (l[k - 1] >> 1) & 1ULL;
            k2 += 2;
            v8 = (v8 + static_cast<int>(4 * (lf & lp))) & 7;
            if (k > 2) {
                const size_t k2_len = k - 2;
                const size_t nw2 = (k2_len + 63) >> 6;
                const uint64_t* mf_row = &M[(k - 2) * mw];
                const uint64_t* mp_row = &M[(k - 1) * mw];
                for (size_t r = 0; r < k2_len; ++r) {
                    const uint64_t mfr = (mf_row[r >> 6] >> (r & 63U)) & 1ULL;
                    const uint64_t mpr = (mp_row[r >> 6] >> (r & 63U)) & 1ULL;
                    const uint64_t lin_u = ((lp & mfr) ^ (lf & mpr) ^ (mfr & mpr)) & 1ULL;
                    l[r] = static_cast<uint8_t>((l[r] + (lin_u << 1)) & 3U);
                    if (mfr | mpr) {
                        uint64_t* row_r = &M[r * mw];
                        if (mfr & mpr) {
                            for (size_t w = 0; w < nw2; ++w) row_r[w] ^= mf_row[w] ^ mp_row[w];
                        } else if (mfr) {
                            for (size_t w = 0; w < nw2; ++w) row_r[w] ^= mp_row[w];
                        } else {
                            for (size_t w = 0; w < nw2; ++w) row_r[w] ^= mf_row[w];
                        }
                        row_r[r >> 6] &= ~(1ULL << (r & 63U));
                    }
                }
            }
            k -= 2;
            continue;
        }

        for (size_t r = 0; r < k; ++r) {
            if ((l[r] & 3U) == 2U) {
                out_ri[0] = 0.0;
                out_ri[1] = 0.0;
                return;
            }
        }
        k2 += 2 * static_cast<int64_t>(k);
        k = 0;
    }

    const int64_t total_k2 = k2 + scale_log2;
    const int64_t odd_rem = (total_k2 % 2 != 0) ? 1 : 0;
    const int64_t exp2_64 = (total_k2 - odd_rem) / 2;
    const int exp2 = (exp2_64 < -2000) ? -2000 : ((exp2_64 > 2000) ? 2000 : static_cast<int>(exp2_64));
    const double sc = odd_rem ? kSqrt2 : 1.0;
    out_ri[0] = std::ldexp(kOmegaRe[v8] * sc, exp2);
    out_ri[1] = std::ldexp(kOmegaIm[v8] * sc, exp2);
}

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
        1e-13, out_chi, out_d,
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

void magic_quadratic_gauss_sum(
    size_t d,
    const uint8_t* ell,
    const uint8_t* Gamma,
    int32_t q0,
    int64_t scale_log2,
    double* out_ri
) {
    if (d == 0) {
        gauss_sum_bitpacked(0, nullptr, nullptr, 0, q0, scale_log2, out_ri);
        return;
    }
    const size_t mw = (d + 63) >> 6;
    std::vector<uint8_t> l(d);
    std::vector<uint64_t> M(d * mw, 0ULL);
    for (size_t i = 0; i < d; ++i) {
        l[i] = ell[i] & 3U;
        const uint8_t* g_row = &Gamma[i * d];
        uint64_t* m_row = &M[i * mw];
        for (size_t j = 0; j < d; ++j) {
            if (i != j && (g_row[j] & 1U)) {
                m_row[j >> 6] |= (1ULL << (j & 63U));
            }
        }
    }
    gauss_sum_bitpacked(d, l.data(), M.data(), mw, q0, scale_log2, out_ri);
}

void magic_overlap(
    size_t n_words,
    size_t d_bra,
    const uint64_t* bra_h_words,
    const uint64_t* bra_A_words,
    const uint8_t* bra_ell,
    const uint8_t* bra_Gamma,
    size_t d_ket,
    const uint64_t* ket_h_words,
    const uint64_t* ket_A_words,
    const uint8_t* ket_ell,
    const uint8_t* ket_Gamma,
    double* out_ri
) {
    const size_t D = d_ket + d_bra;
    if (D == 0) {
        for (size_t w = 0; w < n_words; ++w) {
            if (bra_h_words[w] != ket_h_words[w]) {
                out_ri[0] = 0.0;
                out_ri[1] = 0.0;
                return;
            }
        }
        out_ri[0] = 1.0;
        out_ri[1] = 0.0;
        return;
    }

    const size_t cw = (D + 63) >> 6;
    std::vector<uint64_t> V(D * n_words, 0ULL);
    if (d_ket > 0 && n_words > 0) {
        std::memcpy(V.data(), ket_A_words, d_ket * n_words * sizeof(uint64_t));
    }
    if (d_bra > 0 && n_words > 0) {
        std::memcpy(&V[d_ket * n_words], bra_A_words, d_bra * n_words * sizeof(uint64_t));
    }

    std::vector<uint64_t> C(D * cw, 0ULL);
    for (size_t r = 0; r < D; ++r) {
        C[r * cw + (r >> 6)] = (1ULL << (r & 63U));
    }

    std::vector<uint64_t> rhs(n_words, 0ULL);
    for (size_t w = 0; w < n_words; ++w) {
        rhs[w] = ket_h_words[w] ^ bra_h_words[w];
    }
    std::vector<uint64_t> t0(cw, 0ULL);
    std::vector<size_t> null_rows;
    null_rows.reserve(D);

    for (size_t col = 0; col < D; ++col) {
        const uint64_t* v_col = &V[col * n_words];
        const uint64_t* c_col = &C[col * cw];
        size_t p_word = n_words;
        uint64_t mask = 0ULL;
        for (size_t w = 0; w < n_words; ++w) {
            if (v_col[w] != 0ULL) {
                p_word = w;
                mask = (1ULL << ctz64(v_col[w]));
                break;
            }
        }
        if (p_word < n_words) {
            for (size_t idx = col + 1; idx < D; ++idx) {
                uint64_t* v_idx = &V[idx * n_words];
                if ((v_idx[p_word] & mask) != 0ULL) {
                    for (size_t w = 0; w < n_words; ++w) v_idx[w] ^= v_col[w];
                    uint64_t* c_idx = &C[idx * cw];
                    for (size_t w = 0; w < cw; ++w) c_idx[w] ^= c_col[w];
                }
            }
            if ((rhs[p_word] & mask) != 0ULL) {
                for (size_t w = 0; w < n_words; ++w) rhs[w] ^= v_col[w];
                for (size_t w = 0; w < cw; ++w) t0[w] ^= c_col[w];
            }
        } else {
            null_rows.push_back(col);
        }
    }

    for (size_t w = 0; w < n_words; ++w) {
        if (rhs[w] != 0ULL) {
            out_ri[0] = 0.0;
            out_ri[1] = 0.0;
            return;
        }
    }

    std::vector<uint8_t> tilde_ell(D);
    for (size_t a = 0; a < d_ket; ++a) tilde_ell[a] = ket_ell[a] & 3U;
    for (size_t a = 0; a < d_bra; ++a) {
        tilde_ell[d_ket + a] = static_cast<uint8_t>((4U - (bra_ell[a] & 3U)) & 3U);
    }

    std::vector<uint64_t> G_words(D * cw, 0ULL);
    std::vector<uint64_t> U_words(D * cw, 0ULL);
    for (size_t a = 0; a < d_ket; ++a) {
        const uint8_t* g_row = &ket_Gamma[a * d_ket];
        for (size_t b = 0; b < d_ket; ++b) {
            if (a != b && (g_row[b] & 1U)) {
                G_words[a * cw + (b >> 6)] |= (1ULL << (b & 63U));
                if (b > a) U_words[a * cw + (b >> 6)] |= (1ULL << (b & 63U));
            }
        }
    }
    for (size_t a = 0; a < d_bra; ++a) {
        const uint8_t* g_row = &bra_Gamma[a * d_bra];
        const size_t ga = d_ket + a;
        for (size_t b = 0; b < d_bra; ++b) {
            if (a != b && (g_row[b] & 1U)) {
                const size_t gb = d_ket + b;
                G_words[ga * cw + (gb >> 6)] |= (1ULL << (gb & 63U));
                if (gb > ga) U_words[ga * cw + (gb >> 6)] |= (1ULL << (gb & 63U));
            }
        }
    }

    int32_t lin_c0 = 0;
    uint8_t quad_c0 = 0;
    for (size_t a = 0; a < D; ++a) {
        if ((t0[a >> 6] >> (a & 63U)) & 1ULL) {
            lin_c0 += tilde_ell[a];
            quad_c0 ^= word_dot_parity(&U_words[a * cw], t0.data(), cw);
        }
    }
    const int32_t c0 = (lin_c0 + 2 * static_cast<int32_t>(quad_c0)) & 3;
    const int64_t scale_log2 = -static_cast<int64_t>(D);

    const size_t g = null_rows.size();
    if (g == 0) {
        gauss_sum_bitpacked(0, nullptr, nullptr, 0, c0, scale_log2, out_ri);
        return;
    }

    std::vector<uint8_t> vec_mod4(D);
    std::vector<uint64_t> odd_ell_words(cw, 0ULL);
    for (size_t a = 0; a < D; ++a) {
        const uint8_t e = tilde_ell[a];
        if (e & 1U) odd_ell_words[a >> 6] |= (1ULL << (a & 63U));
        const uint64_t t0_a = (t0[a >> 6] >> (a & 63U)) & 1ULL;
        const uint8_t signed_e = t0_a ? static_cast<uint8_t>((4U - e) & 3U) : e;
        const uint8_t qt0 = word_dot_parity(&G_words[a * cw], t0.data(), cw);
        vec_mod4[a] = static_cast<uint8_t>((signed_e + (qt0 << 1)) & 3U);
    }

    std::vector<uint8_t> lin(g);
    std::vector<uint64_t> GL_words(g * cw, 0ULL);
    for (size_t u = 0; u < g; ++u) {
        const uint64_t* L_u = &C[null_rows[u] * cw];
        uint64_t* GL_u = &GL_words[u * cw];
        int32_t acc_lin = 0;
        uint8_t diag_q = 0;
        for (size_t w = 0; w < cw; ++w) {
            uint64_t word = L_u[w];
            while (word != 0ULL) {
                const size_t a = (w << 6) + static_cast<size_t>(ctz64(word));
                acc_lin += vec_mod4[a];
                diag_q ^= word_dot_parity(&U_words[a * cw], L_u, cw);
                const uint64_t* G_a = &G_words[a * cw];
                for (size_t kw = 0; kw < cw; ++kw) GL_u[kw] ^= G_a[kw];
                word &= (word - 1ULL);
            }
        }
        lin[u] = static_cast<uint8_t>((acc_lin + 2 * static_cast<int32_t>(diag_q)) & 3);
    }

    const size_t mw = (g + 63) >> 6;
    std::vector<uint64_t> M(g * mw, 0ULL);
    for (size_t u = 0; u < g; ++u) {
        const uint64_t* L_u = &C[null_rows[u] * cw];
        for (size_t v = u + 1; v < g; ++v) {
            const uint64_t* L_v = &C[null_rows[v] * cw];
            const uint64_t* GL_v = &GL_words[v * cw];
            uint64_t acc = 0ULL;
            for (size_t w = 0; w < cw; ++w) {
                acc ^= (L_u[w] & L_v[w] & odd_ell_words[w]) ^ (L_u[w] & GL_v[w]);
            }
            if (popcount64(acc) & 1) {
                M[u * mw + (v >> 6)] |= (1ULL << (v & 63U));
                M[v * mw + (u >> 6)] |= (1ULL << (u & 63U));
            }
        }
    }

    gauss_sum_bitpacked(g, lin.data(), M.data(), mw, c0, scale_log2, out_ri);
}

}  // extern "C"
