// Copyright 2026 Google LLC
//
// Stabilizer-branch engine (Method C) C++17 primitives and C ABI (`mbr_*`).
// Included ONCE at the end of magic_kernels.cpp (after magic_engine.hpp).
// All internal structs and helpers live in `namespace { namespace mbr { ... } }`
// for internal linkage and complete isolation from qfe_core.hpp and magic_engine.hpp.
// Reuses qfe_core.hpp bit-word helpers and magic_engine.hpp overflow guards without
// modifying qfe_core.hpp.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace mbr {

constexpr int32_t kOk = 0;
constexpr int32_t kErrInvalidArg = -1;
constexpr int32_t kErrOom = -2;
constexpr int32_t kErrFloatDrift = -3;
constexpr int32_t kErrBudget = -4;

constexpr size_t kMaxQubits = static_cast<size_t>(std::numeric_limits<int32_t>::max());
constexpr size_t kMaxDenseQubits = 26U;

struct MbrComplex {
    double re = 1.0;
    double im = 0.0;
};

inline MbrComplex i_pow(uint8_t p) noexcept {
    switch (p & 3U) {
        case 0: return { 1.0,  0.0};
        case 1: return { 0.0,  1.0};
        case 2: return {-1.0,  0.0};
        default: return {0.0, -1.0};
    }
}

inline MbrComplex omega_pow(uint8_t k) noexcept {
    static const double kRe[8] = {
        1.0,  kInvSqrt2, 0.0, -kInvSqrt2, -1.0, -kInvSqrt2,  0.0,  kInvSqrt2
    };
    static const double kIm[8] = {
        0.0,  kInvSqrt2, 1.0,  kInvSqrt2,  0.0, -kInvSqrt2, -1.0, -kInvSqrt2
    };
    const uint8_t idx = k & 7U;
    return {kRe[idx], kIm[idx]};
}

inline MbrComplex cmul(MbrComplex a, MbrComplex b) noexcept {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline MbrComplex cscale(MbrComplex a, double s) noexcept {
    return {a.re * s, a.im * s};
}

inline size_t n_words_for_qubits(size_t n_qubits) noexcept {
    return (n_qubits + 63U) >> 6U;
}

inline bool tail_clear(const uint64_t* words, size_t n_qubits, size_t nw) noexcept {
    return (words[nw - 1U] & ~tail_mask(n_qubits)) == 0ULL;
}

inline void set_bit(uint64_t* words, int32_t q) noexcept {
    const size_t uq = static_cast<size_t>(q);
    words[uq >> 6U] |= (uint64_t(1) << (uq & 63U));
}

inline void write_u32_le(uint8_t* dst, uint32_t v) noexcept {
    dst[0] = static_cast<uint8_t>(v & 0xFFU);
    dst[1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
    dst[2] = static_cast<uint8_t>((v >> 16U) & 0xFFU);
    dst[3] = static_cast<uint8_t>((v >> 24U) & 0xFFU);
}

inline uint32_t read_u32_le(const uint8_t* src) noexcept {
    return static_cast<uint32_t>(src[0])
        | (static_cast<uint32_t>(src[1]) << 8U)
        | (static_cast<uint32_t>(src[2]) << 16U)
        | (static_cast<uint32_t>(src[3]) << 24U);
}

inline void write_u64_le(uint8_t* dst, uint64_t v) noexcept {
    for (size_t i = 0; i < 8U; ++i) {
        dst[i] = static_cast<uint8_t>((v >> (i * 8U)) & 0xFFULL);
    }
}

inline uint64_t read_u64_le(const uint8_t* src) noexcept {
    uint64_t v = 0ULL;
    for (size_t i = 0; i < 8U; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (i * 8U);
    }
    return v;
}

// Multi-word Pauli helpers (directly reused by WP3 resolve / absorption / store).
struct MbrPauli {
    std::vector<uint64_t> x;  // nw words, tail bits 0
    std::vector<uint64_t> z;  // nw words, tail bits 0
    uint8_t kappa = 0;        // in {0, 1, 2, 3}
};

// Exact popcount(a & b) mod 4 across words (never XOR-then-popcount for mod-4!).
inline uint8_t popcount_and_mod4(const uint64_t* a, const uint64_t* b, size_t nw) noexcept {
    uint32_t sum = 0U;
    for (size_t w = 0; w < nw; ++w) {
        sum += static_cast<uint32_t>(popcount64(a[w] & b[w]));
    }
    return static_cast<uint8_t>(sum & 3U);
}

inline bool is_hermitian(const uint64_t* x, const uint64_t* z, uint8_t kappa, size_t nw) noexcept {
    return ((kappa ^ word_dot_parity(x, z, nw)) & 1U) == 0U;
}

inline uint8_t pauli_anti(
    const uint64_t* x1, const uint64_t* z1,
    const uint64_t* x2, const uint64_t* z2, size_t nw
) noexcept {
    uint64_t acc = 0ULL;
    for (size_t w = 0; w < nw; ++w) {
        acc ^= (x1[w] & z2[w]) ^ (z1[w] & x2[w]);
    }
    return static_cast<uint8_t>(popcount64(acc) & 1);
}

inline void pauli_mul_inplace(
    uint64_t* x1, uint64_t* z1, uint8_t& k1,
    const uint64_t* x2, const uint64_t* z2, uint8_t k2, size_t nw
) noexcept {
    const uint8_t s = word_dot_parity(z1, x2, nw);
    k1 = static_cast<uint8_t>((k1 + k2 + (s << 1U)) & 3U);
    for (size_t w = 0; w < nw; ++w) {
        x1[w] ^= x2[w];
        z1[w] ^= z2[w];
    }
}

// Normalize Hermitian Pauli to sign +1; returns true if negated.
inline bool pauli_norm_sign_plus(
    const uint64_t* x, const uint64_t* z, uint8_t& kappa, size_t nw
) noexcept {
    const uint8_t pc = popcount_and_mod4(x, z, nw);
    if (((kappa - pc) & 3U) == 2U) {
        kappa = static_cast<uint8_t>((kappa + 2U) & 3U);
        return true;
    }
    kappa &= 3U;
    return false;
}

// Arbitrary-n, arbitrary-d (including n > 64, d > 64) stabilizer state.
struct MbrState {
    size_t n_qubits = 0;
    size_t nw = 0;
    size_t d = 0;
    std::vector<uint64_t> h;     // nw
    std::vector<uint64_t> A;     // d * nw (row j = column j of A)
    std::vector<uint8_t> ell;    // d, entries in {0, 1, 2, 3}
    std::vector<uint8_t> Gamma;  // d * d, binary symmetric hollow {0, 1}
    std::vector<int32_t> piv;    // d, distinct in [0, n_qubits), A[piv[j], k] = delta_{jk}
};

std::atomic<size_t> g_live_states{0};

inline MbrState make_zero(size_t n_qubits) {
    MbrState s;
    s.n_qubits = n_qubits;
    s.nw = n_words_for_qubits(n_qubits);
    s.d = 0;
    s.h.assign(s.nw, 0ULL);
    return s;
}

inline bool validate_state(const MbrState& s) noexcept {
    if (s.n_qubits == 0U || s.n_qubits > kMaxQubits) return false;
    if (s.nw != n_words_for_qubits(s.n_qubits)) return false;
    if (s.d > s.n_qubits) return false;
    size_t a_elems = 0U, g_elems = 0U;
    if (!mul_within(s.d, s.nw, kMaxElems, a_elems)) return false;
    if (!mul_within(s.d, s.d, kMaxElems, g_elems)) return false;
    if (s.h.size() != s.nw || s.A.size() != a_elems || s.ell.size() != s.d
        || s.Gamma.size() != g_elems || s.piv.size() != s.d) {
        return false;
    }
    if (!tail_clear(s.h.data(), s.n_qubits, s.nw)) return false;
    const size_t d = s.d, nw = s.nw;
    for (size_t j = 0; j < d; ++j) {
        if (!tail_clear(&s.A[j * nw], s.n_qubits, nw)) return false;
        if (s.ell[j] >= 4U) return false;
        if (s.piv[j] < 0 || static_cast<size_t>(s.piv[j]) >= s.n_qubits) return false;
        if (s.Gamma[j * d + j] != 0U) return false;
        for (size_t k = j + 1U; k < d; ++k) {
            const uint8_t gjk = s.Gamma[j * d + k];
            if (gjk > 1U || gjk != s.Gamma[k * d + j]) return false;
        }
    }
    // Checking A[piv[j], k] == delta_{jk} for all j, k also implies piv[j] are distinct.
    for (size_t j = 0; j < d; ++j) {
        const int32_t pj = s.piv[j];
        for (size_t k = 0; k < d; ++k) {
            const uint8_t expected = (j == k) ? 1U : 0U;
            if (get_bit(&s.A[k * nw], pj) != expected) return false;
        }
    }
    return true;
}

inline bool is_rcef(const MbrState& s) noexcept {
    const size_t d = s.d, nw = s.nw;
    for (size_t j = 0; j < d; ++j) {
        if (j > 0U && s.piv[j - 1U] >= s.piv[j]) return false;
        if (first_set_bit_in_words(&s.A[j * nw], nw) != s.piv[j]) return false;
        for (size_t k = 0; k < d; ++k) {
            const uint8_t expected = (j == k) ? 1U : 0U;
            if (get_bit(&s.A[j * nw], s.piv[k]) != expected) return false;
        }
    }
    return true;
}

inline bool is_canonical(const MbrState& s) noexcept {
    if (!is_rcef(s)) return false;
    for (size_t j = 0; j < s.d; ++j) {
        if (get_bit(s.h.data(), s.piv[j]) != 0U) return false;
    }
    return true;
}

// Column b += column a, pulling Q back under u_a -> u_a + u_b (state unchanged).
inline void col_add(MbrState& s, size_t a, size_t b) noexcept {
    const size_t d = s.d, nw = s.nw;
    xor_words_inplace(&s.A[b * nw], &s.A[a * nw], nw);
    s.ell[b] = static_cast<uint8_t>((s.ell[b] + s.ell[a] + (s.Gamma[a * d + b] << 1U)) & 3U);
    const uint8_t ell_a_odd = s.ell[a] & 1U;
    for (size_t k = 0; k < d; ++k) {
        if (k != a && k != b && s.Gamma[a * d + k] != 0U) {
            s.Gamma[b * d + k] ^= 1U;
            s.Gamma[k * d + b] ^= 1U;
        }
    }
    if (ell_a_odd != 0U) {
        s.Gamma[a * d + b] ^= 1U;
        s.Gamma[b * d + a] ^= 1U;
    }
}

inline void make_principal(MbrState& s, size_t col, int32_t row) noexcept {
    const size_t d = s.d, nw = s.nw;
    for (size_t j = 0; j < d; ++j) {
        if (j != col && get_bit(&s.A[j * nw], row) != 0U) {
            col_add(s, col, j);
        }
    }
    s.piv[col] = row;
}

inline void delete_col(MbrState& s, size_t j) noexcept {
    const size_t d = s.d, nw = s.nw, nd = d - 1U;
    if (j < nd) {
        std::memmove(&s.A[j * nw], &s.A[(j + 1U) * nw], (nd - j) * nw * sizeof(uint64_t));
        std::memmove(&s.ell[j], &s.ell[j + 1U], (nd - j) * sizeof(uint8_t));
        std::memmove(&s.piv[j], &s.piv[j + 1U], (nd - j) * sizeof(int32_t));
    }
    s.A.resize(nd * nw);
    s.ell.resize(nd);
    s.piv.resize(nd);
    // In-place compaction: for all 0 <= r, c < nd, r*nd + c <= src_r*d + src_c.
    for (size_t r = 0; r < nd; ++r) {
        const size_t src_r = (r < j) ? r : r + 1U;
        for (size_t c = 0; c < nd; ++c) {
            const size_t src_c = (c < j) ? c : c + 1U;
            s.Gamma[r * nd + c] = s.Gamma[src_r * d + src_c];
        }
    }
    s.Gamma.resize(nd * nd);
    s.d = nd;
}

// Condition on lam . u = eps with direct amplitude scale (avoids kSqrt2 * kInvSqrt2 drift).
inline MbrComplex restrict_state(
    MbrState& s, const uint8_t* lam, uint8_t eps, double scale
) noexcept {
    const size_t d = s.d, nw = s.nw;
    size_t j = 0U;
    while (j < d && lam[j] == 0U) ++j;
    for (size_t k = j + 1U; k < d; ++k) {
        if (lam[k] != 0U) col_add(s, j, k);
    }
    const MbrComplex f = cscale(i_pow(eps != 0U ? s.ell[j] : 0U), scale);
    if (eps != 0U) {
        xor_words_inplace(s.h.data(), &s.A[j * nw], nw);
        for (size_t k = 0; k < d; ++k) {
            if (s.Gamma[j * d + k] != 0U) {
                s.ell[k] = static_cast<uint8_t>((s.ell[k] + 2U) & 3U);
            }
        }
    }
    delete_col(s, j);
    return f;
}

inline void add_parity(
    MbrState& s, uint8_t alpha, uint8_t const_bit, const uint8_t* r
) noexcept {
    const size_t d = s.d;
    const uint8_t delta = static_cast<uint8_t>(
        (const_bit != 0U ? (4U - (alpha & 3U)) : (alpha & 3U)) & 3U
    );
    for (size_t k = 0; k < d; ++k) {
        if (r[k] != 0U) {
            s.ell[k] = static_cast<uint8_t>((s.ell[k] + delta) & 3U);
        }
    }
    if ((alpha & 1U) != 0U) {
        for (size_t j = 0; j < d; ++j) {
            if (r[j] == 0U) continue;
            for (size_t k = j + 1U; k < d; ++k) {
                if (r[k] != 0U) {
                    s.Gamma[j * d + k] ^= 1U;
                    s.Gamma[k * d + j] ^= 1U;
                }
            }
        }
    }
}

inline MbrComplex eliminate_zero(MbrState& s, size_t j) {
    const size_t d = s.d;
    const uint8_t c = s.ell[j];
    std::vector<uint8_t> lam(d - 1U, 0U);
    bool any_lam = false;
    for (size_t k = 0; k < d; ++k) {
        if (k == j) continue;
        const uint8_t bit = s.Gamma[j * d + k];
        lam[k < j ? k : k - 1U] = bit;
        any_lam = any_lam || (bit != 0U);
    }
    delete_col(s, j);
    if ((c & 1U) != 0U) {
        add_parity(s, static_cast<uint8_t>((4U - c) & 3U), 0U, lam.data());
        // (1 + i^c) / sqrt(2): omega^1 for c == 1, omega^7 for c == 3.
        return omega_pow(c == 1U ? 1U : 7U);
    }
    if (any_lam) {
        return restrict_state(s, lam.data(), static_cast<uint8_t>(c >> 1U), 1.0);
    }
    return (c == 0U) ? MbrComplex{kSqrt2, 0.0} : MbrComplex{0.0, 0.0};
}

// Apply i^(kappa & 3) X(x) Z(z); returns exact phase i^p.
inline MbrComplex apply_pauli(
    MbrState& s, const uint64_t* x, const uint64_t* z, uint8_t kappa
) noexcept {
    const size_t nw = s.nw, d = s.d;
    const uint8_t p = static_cast<uint8_t>((kappa + (word_dot_parity(z, s.h.data(), nw) << 1U)) & 3U);
    if (words_any_nonzero(z, nw)) {
        for (size_t j = 0; j < d; ++j) {
            if (word_dot_parity(&s.A[j * nw], z, nw) != 0U) {
                s.ell[j] = static_cast<uint8_t>((s.ell[j] + 2U) & 3U);
            }
        }
    }
    xor_words_inplace(s.h.data(), x, nw);
    return i_pow(p);
}

// Apply R(P) = exp(+i*pi*P/4) = (I + iP)/sqrt(2) for Hermitian P = i^kappa X(x) Z(z).
// Note: temporarily extends d to d0 + 1 (even when d0 == n_qubits) before reducing/eliminating.
inline MbrComplex apply_qturn(
    MbrState& s, const uint64_t* x, const uint64_t* z, uint8_t kappa
) {
    const size_t nw = s.nw, d0 = s.d, d1 = d0 + 1U;
    std::vector<uint8_t> t(d0, 0U);
    if (words_any_nonzero(z, nw)) {
        for (size_t j = 0; j < d0; ++j) {
            t[j] = word_dot_parity(&s.A[j * nw], z, nw);
        }
    }
    std::vector<uint8_t> sv(d0, 0U);
    if (words_any_nonzero(x, nw)) {
        for (size_t j = 0; j < d0; ++j) {
            sv[j] = get_bit(x, s.piv[j]);
        }
    }
    std::vector<uint8_t> ng(d1 * d1, 0U);
    for (size_t j = 0; j < d0; ++j) {
        if (d0 > 0U) {
            std::memcpy(&ng[j * d1], &s.Gamma[j * d0], d0);
        }
        ng[j * d1 + d0] = t[j];
        ng[d0 * d1 + j] = t[j];
    }
    s.A.resize(d1 * nw);
    std::memcpy(&s.A[d0 * nw], x, nw * sizeof(uint64_t));
    s.ell.push_back(static_cast<uint8_t>(
        (kappa + 1U + (word_dot_parity(z, s.h.data(), nw) << 1U)) & 3U
    ));
    s.piv.push_back(-1);
    s.Gamma.swap(ng);
    s.d = d1;

    for (size_t j = 0; j < d0; ++j) {
        if (sv[j] != 0U) col_add(s, j, d0);
    }
    const int32_t row = first_set_bit_in_words(&s.A[d0 * nw], nw);
    if (row >= 0) {
        make_principal(s, d0, row);
        return {1.0, 0.0};
    }
    return eliminate_zero(s, d0);
}

// Project onto (-1)^sigma Z(z) = (-1)^m. Returns 0 on impossible outcome (state untouched).
inline MbrComplex apply_zproject(MbrState& s, const uint64_t* z, uint8_t sigma, uint8_t m) {
    const size_t nw = s.nw, d = s.d;
    std::vector<uint8_t> lam(d, 0U);
    bool any_lam = false;
    for (size_t j = 0; j < d; ++j) {
        const uint8_t bit = word_dot_parity(&s.A[j * nw], z, nw);
        lam[j] = bit;
        any_lam = any_lam || (bit != 0U);
    }
    const uint8_t eps = static_cast<uint8_t>((m ^ sigma ^ word_dot_parity(z, s.h.data(), nw)) & 1U);
    if (!any_lam) {
        return (eps == 0U) ? MbrComplex{1.0, 0.0} : MbrComplex{0.0, 0.0};
    }
    return restrict_state(s, lam.data(), eps, kInvSqrt2);
}

// Project onto Hermitian P = (-1)^m. Pi|old> = c|new>.
inline MbrComplex apply_project(
    MbrState& s, const uint64_t* x, const uint64_t* z, uint8_t kappa, uint8_t m
) {
    const size_t nw = s.nw;
    const int32_t p = first_set_bit_in_words(x, nw);
    if (p < 0) {
        return apply_zproject(s, z, static_cast<uint8_t>((kappa >> 1U) & 1U), m);
    }
    std::vector<uint64_t> zp(z, z + nw);
    zp[static_cast<size_t>(p) >> 6U] ^= (uint64_t(1) << (static_cast<size_t>(p) & 63U));
    std::vector<uint64_t> ep(nw, 0ULL);
    set_bit(ep.data(), p);
    const MbrComplex f = apply_qturn(s, x, zp.data(), static_cast<uint8_t>((kappa + 1U) & 3U));
    const MbrComplex g = apply_zproject(s, ep.data(), 0U, m);
    if (g.re == 0.0 && g.im == 0.0) {
        return {0.0, 0.0};
    }
    const MbrComplex h = apply_qturn(s, x, zp.data(), static_cast<uint8_t>((kappa + 3U) & 3U));
    return cmul(cmul(f, g), h);
}

inline uint8_t qphase(const MbrState& s, const uint8_t* v) noexcept {
    const size_t d = s.d;
    uint32_t e = 0U;
    for (size_t j = 0; j < d; ++j) {
        if (v[j] == 0U) continue;
        uint32_t gv = 0U;
        const uint8_t* row = &s.Gamma[j * d];
        for (size_t k = 0; k < d; ++k) {
            gv += static_cast<uint32_t>(row[k] & v[k]);
        }
        e += static_cast<uint32_t>(s.ell[j]) + gv;
    }
    return static_cast<uint8_t>(e & 3U);
}

// +1 or -1 if Hermitian Pauli P has definite eigenvalue on s, else 0.
inline int8_t definite(
    const MbrState& s, const uint64_t* x, const uint64_t* z, uint8_t kappa
) {
    const size_t nw = s.nw, d = s.d;
    std::vector<uint8_t> v(d, 0U);
    if (words_any_nonzero(x, nw)) {
        std::vector<uint64_t> av(nw, 0ULL);
        for (size_t j = 0; j < d; ++j) {
            if (get_bit(x, s.piv[j]) != 0U) {
                v[j] = 1U;
                xor_words_inplace(av.data(), &s.A[j * nw], nw);
            }
        }
        if (std::memcmp(av.data(), x, nw * sizeof(uint64_t)) != 0) {
            return 0;
        }
    }
    uint32_t e = static_cast<uint32_t>(kappa)
        + (static_cast<uint32_t>(word_dot_parity(z, s.h.data(), nw)) << 1U);
    for (size_t j = 0; j < d; ++j) {
        const uint8_t tj = word_dot_parity(&s.A[j * nw], z, nw);
        const uint8_t vj = v[j];
        uint32_t gv = 0U;
        const uint8_t* row = &s.Gamma[j * d];
        for (size_t k = 0; k < d; ++k) {
            gv += static_cast<uint32_t>(row[k] & v[k]);
        }
        // Unreduced gv MUST be added to e (mod-4 sum, NOT mod-2!).
        if (((tj ^ static_cast<uint8_t>(gv & 1U) ^ static_cast<uint8_t>(s.ell[j] & vj)) & 1U) != 0U) {
            return 0;
        }
        if (vj != 0U) {
            e += static_cast<uint32_t>(s.ell[j]) + gv + (static_cast<uint32_t>(tj) << 1U);
        }
    }
    return ((e & 3U) == 0U) ? 1 : -1;
}

inline void generators(
    const MbrState& s,
    std::vector<uint64_t>& out_x,
    std::vector<uint64_t>& out_z,
    std::vector<uint8_t>& out_kappa
) {
    const size_t n = s.n_qubits, nw = s.nw, d = s.d;
    out_x.assign(n * nw, 0ULL);
    out_z.assign(n * nw, 0ULL);
    out_kappa.assign(n, 0U);
    std::vector<uint8_t> is_piv(n, 0U);
    for (size_t j = 0; j < d; ++j) {
        is_piv[static_cast<size_t>(s.piv[j])] = 1U;
    }
    size_t idx = 0U;
    for (size_t r = 0; r < n; ++r) {
        if (is_piv[r] != 0U) continue;
        uint64_t* bz = &out_z[idx * nw];
        set_bit(bz, static_cast<int32_t>(r));
        for (size_t j = 0; j < d; ++j) {
            if (get_bit(&s.A[j * nw], static_cast<int32_t>(r)) != 0U) {
                set_bit(bz, s.piv[j]);
            }
        }
        out_kappa[idx] = static_cast<uint8_t>(word_dot_parity(bz, s.h.data(), nw) << 1U);
        ++idx;
    }
    for (size_t j = 0; j < d; ++j) {
        uint64_t* cx = &out_x[idx * nw];
        uint64_t* cz = &out_z[idx * nw];
        std::memcpy(cx, &s.A[j * nw], nw * sizeof(uint64_t));
        for (size_t k = 0; k < d; ++k) {
            uint8_t bit = s.Gamma[k * d + j];
            if (k == j) bit ^= (s.ell[j] & 1U);
            if (bit != 0U) set_bit(cz, s.piv[k]);
        }
        const uint32_t sum = static_cast<uint32_t>(s.ell[j])
            + (static_cast<uint32_t>(word_dot_parity(cz, s.h.data(), nw)) << 1U)
            + (static_cast<uint32_t>(s.ell[j] & 1U) << 1U);
        out_kappa[idx] = static_cast<uint8_t>((4U - (sum & 3U)) & 3U);
        ++idx;
    }
}

// Canonicalize in-place; returns p in {0..3} such that |old> = i^p |new>.
inline uint8_t canon_phase4(MbrState& s) {
    const size_t n = s.n_qubits, nw = s.nw, d = s.d;
    if (d == 0U) return 0U;
    if (!is_rcef(s)) {
        std::vector<uint8_t> used(d, 0U);
        std::vector<int32_t> newpiv(d, -1);
        size_t found = 0U;
        for (size_t r = 0; r < n && found < d; ++r) {
            size_t jj = d;
            for (size_t j = 0; j < d; ++j) {
                if (used[j] == 0U && get_bit(&s.A[j * nw], static_cast<int32_t>(r)) != 0U) {
                    jj = j;
                    break;
                }
            }
            if (jj == d) continue;
            used[jj] = 1U;
            newpiv[jj] = static_cast<int32_t>(r);
            ++found;
            for (size_t k = 0; k < d; ++k) {
                if (k != jj && get_bit(&s.A[k * nw], static_cast<int32_t>(r)) != 0U) {
                    col_add(s, jj, k);
                }
            }
        }
        std::vector<size_t> order(d);
        for (size_t j = 0; j < d; ++j) order[j] = j;
        std::sort(order.begin(), order.end(), [&newpiv](size_t a, size_t b) {
            return newpiv[a] < newpiv[b];
        });
        bool identity_order = true;
        for (size_t j = 0; j < d; ++j) {
            if (order[j] != j) {
                identity_order = false;
                break;
            }
        }
        if (!identity_order) {
            std::vector<uint64_t> nA(d * nw);
            std::vector<uint8_t> nell(d);
            std::vector<uint8_t> nG(d * d);
            for (size_t j = 0; j < d; ++j) {
                const size_t oj = order[j];
                std::memcpy(&nA[j * nw], &s.A[oj * nw], nw * sizeof(uint64_t));
                nell[j] = s.ell[oj];
                for (size_t k = 0; k < d; ++k) {
                    nG[j * d + k] = s.Gamma[oj * d + order[k]];
                }
            }
            s.A.swap(nA);
            s.ell.swap(nell);
            s.Gamma.swap(nG);
        }
        for (size_t j = 0; j < d; ++j) {
            s.piv[j] = newpiv[order[j]];
        }
    }
    std::vector<uint8_t> v(d, 0U);
    bool any_v = false;
    for (size_t j = 0; j < d; ++j) {
        if (get_bit(s.h.data(), s.piv[j]) != 0U) {
            v[j] = 1U;
            any_v = true;
        }
    }
    if (!any_v) return 0U;
    const uint8_t e = qphase(s, v.data());
    for (size_t j = 0; j < d; ++j) {
        if (v[j] != 0U) {
            xor_words_inplace(s.h.data(), &s.A[j * nw], nw);
        }
        uint8_t gv_par = 0U;
        const uint8_t* row = &s.Gamma[j * d];
        for (size_t k = 0; k < d; ++k) {
            gv_par ^= static_cast<uint8_t>(row[k] & v[k]);
        }
        if (((gv_par ^ (s.ell[j] & v[j])) & 1U) != 0U) {
            s.ell[j] = static_cast<uint8_t>((s.ell[j] + 2U) & 3U);
        }
    }
    return e;
}

inline MbrComplex canon(MbrState& s) {
    return i_pow(canon_phase4(s));
}

inline size_t canonical_key_size(size_t n_qubits, size_t d) noexcept {
    const size_t nw = n_words_for_qubits(n_qubits);
    // Guard against unsigned underflow when d == 0.
    const size_t n_tri = (d >= 2U) ? ((d * (d - 1U)) / 2U) : 0U;
    const size_t g_bytes = (n_tri + 7U) >> 3U;
    return 8U + (1U + d) * nw * 8U + d + g_bytes;
}

inline void serialize_canonical_unchecked(const MbrState& s, uint8_t* out_buf) noexcept {
    const size_t nw = s.nw, d = s.d;
    uint8_t* p = out_buf;
    write_u32_le(p, static_cast<uint32_t>(s.n_qubits));
    p += 4U;
    write_u32_le(p, static_cast<uint32_t>(d));
    p += 4U;
    for (size_t w = 0; w < nw; ++w) {
        write_u64_le(p, s.h[w]);
        p += 8U;
    }
    for (size_t i = 0; i < d * nw; ++i) {
        write_u64_le(p, s.A[i]);
        p += 8U;
    }
    if (d > 0U) {
        std::memcpy(p, s.ell.data(), d);
        p += d;
    }
    const size_t n_tri = (d >= 2U) ? ((d * (d - 1U)) / 2U) : 0U;
    const size_t g_bytes = (n_tri + 7U) >> 3U;
    if (g_bytes > 0U) {
        std::memset(p, 0, g_bytes);
        size_t bit = 0U;
        for (size_t j = 0; j < d; ++j) {
            const uint8_t* row = &s.Gamma[j * d];
            for (size_t k = j + 1U; k < d; ++k) {
                if (row[k] != 0U) {
                    p[bit >> 3U] |= static_cast<uint8_t>(1U << (bit & 7U));
                }
                ++bit;
            }
        }
    }
}

inline std::string serialize_canonical_str(const MbrState& s) {
    std::string out(canonical_key_size(s.n_qubits, s.d), '\0');
    serialize_canonical_unchecked(s, reinterpret_cast<uint8_t*>(&out[0]));
    return out;
}

inline bool deserialize_canonical(const uint8_t* buf, size_t len, MbrState& out) {
    if (buf == nullptr || len < 8U) return false;
    const uint32_t n32 = read_u32_le(buf);
    const uint32_t d32 = read_u32_le(buf + 4U);
    if (n32 == 0U || static_cast<size_t>(n32) > kMaxQubits || d32 > n32) return false;
    const size_t n = static_cast<size_t>(n32);
    const size_t d = static_cast<size_t>(d32);
    const size_t nw = n_words_for_qubits(n);
    size_t a_elems = 0U, g_elems = 0U;
    if (!mul_within(d, nw, kMaxElems, a_elems)) return false;
    if (!mul_within(d, d, kMaxElems, g_elems)) return false;
    const size_t expected = canonical_key_size(n, d);
    if (len != expected) return false;

    MbrState st;
    st.n_qubits = n;
    st.nw = nw;
    st.d = d;
    st.h.resize(nw);
    st.A.resize(a_elems);
    st.ell.resize(d);
    st.Gamma.assign(g_elems, 0U);
    st.piv.resize(d);

    const uint8_t* p = buf + 8U;
    for (size_t w = 0; w < nw; ++w) {
        st.h[w] = read_u64_le(p);
        p += 8U;
    }
    if (!tail_clear(st.h.data(), n, nw)) return false;

    for (size_t i = 0; i < a_elems; ++i) {
        st.A[i] = read_u64_le(p);
        p += 8U;
    }
    for (size_t j = 0; j < d; ++j) {
        if (!tail_clear(&st.A[j * nw], n, nw)) return false;
        const int32_t piv_j = first_set_bit_in_words(&st.A[j * nw], nw);
        if (piv_j < 0) return false;
        if (j > 0U && st.piv[j - 1U] >= piv_j) return false;
        st.piv[j] = piv_j;
    }
    if (d > 0U) {
        std::memcpy(st.ell.data(), p, d);
        p += d;
        for (size_t j = 0; j < d; ++j) {
            if (st.ell[j] >= 4U) return false;
        }
    }
    const size_t n_tri = (d >= 2U) ? ((d * (d - 1U)) / 2U) : 0U;
    const size_t g_bytes = (n_tri + 7U) >> 3U;
    if (g_bytes > 0U) {
        const size_t rem_bits = n_tri & 7U;
        if (rem_bits != 0U && (p[g_bytes - 1U] >> rem_bits) != 0U) return false;
        size_t bit = 0U;
        for (size_t j = 0; j < d; ++j) {
            for (size_t k = j + 1U; k < d; ++k) {
                const uint8_t val = static_cast<uint8_t>((p[bit >> 3U] >> (bit & 7U)) & 1U);
                st.Gamma[j * d + k] = val;
                st.Gamma[k * d + j] = val;
                ++bit;
            }
        }
    }
    if (!is_canonical(st)) return false;
    out = std::move(st);
    return true;
}

// Assumes |a> and |b> are equal up to global phase; returns <a|b> = (-i)^{Q_a(v)}.
inline MbrComplex same_phase_assumed(const MbrState& a, const MbrState& b) {
    const size_t d = a.d;
    std::vector<uint8_t> v(d, 0U);
    for (size_t j = 0; j < d; ++j) {
        const int32_t pj = a.piv[j];
        const uint8_t bit = static_cast<uint8_t>(
            get_bit(a.h.data(), pj) ^ get_bit(b.h.data(), pj)
        );
        v[j] = bit;
    }
    const uint8_t q = qphase(a, v.data());
    return i_pow(static_cast<uint8_t>((4U - q) & 3U));
}

// True iff |a> and |b> represent the same quantum state up to global phase;
// if true, out_ratio = <a|b> in {1, i, -1, -i}.
inline bool same_phase(const MbrState& a, const MbrState& b, MbrComplex& out_ratio) {
    if (a.n_qubits != b.n_qubits || a.d != b.d) {
        out_ratio = {0.0, 0.0};
        return false;
    }
    MbrState ca_buf, cb_buf;
    const MbrState* ca = &a;
    const MbrState* cb = &b;
    uint8_t pa = 0U, pb = 0U;
    if (!is_canonical(a)) {
        ca_buf = a;
        pa = canon_phase4(ca_buf);
        ca = &ca_buf;
    }
    if (!is_canonical(b)) {
        cb_buf = b;
        pb = canon_phase4(cb_buf);
        cb = &cb_buf;
    }
    if (ca->h == cb->h && ca->A == cb->A && ca->ell == cb->ell && ca->Gamma == cb->Gamma) {
        out_ratio = i_pow(static_cast<uint8_t>((pb + 4U - pa) & 3U));
        return true;
    }
    out_ratio = {0.0, 0.0};
    return false;
}

// Compute <a|b> (conjugate-linear in a, linear in b).
inline MbrComplex overlap(const MbrState& a, const MbrState& b) {
    const size_t n = a.n_qubits, nw = a.nw;
    // Sound disjoint-coset shortcut (valid for BOTH canonical and non-canonical states).
    if (a.d == b.d && a.A == b.A && a.h != b.h) {
        std::vector<uint64_t> dlt(nw);
        for (size_t w = 0; w < nw; ++w) dlt[w] = a.h[w] ^ b.h[w];
        std::vector<uint64_t> av(nw, 0ULL);
        for (size_t j = 0; j < a.d; ++j) {
            if (get_bit(dlt.data(), a.piv[j]) != 0U) {
                xor_words_inplace(av.data(), &a.A[j * nw], nw);
            }
        }
        if (av != dlt) return {0.0, 0.0};
    }
    std::vector<uint64_t> gx, gz;
    std::vector<uint8_t> gk;
    generators(a, gx, gz, gk);
    MbrState c_copy;
    bool copied = false;
    MbrComplex f{1.0, 0.0};
    for (size_t g = 0; g < n; ++g) {
        const MbrState& cur = copied ? c_copy : b;
        const int8_t ev = definite(cur, &gx[g * nw], &gz[g * nw], gk[g]);
        if (ev < 0) return {0.0, 0.0};
        if (ev == 0) {
            if (!copied) {
                c_copy = b;
                copied = true;
            }
            const MbrComplex step = apply_project(c_copy, &gx[g * nw], &gz[g * nw], gk[g], 0U);
            f = cmul(f, step);
            if (f.re == 0.0 && f.im == 0.0) return {0.0, 0.0};
        }
    }
    return cmul(f, same_phase_assumed(a, copied ? c_copy : b));
}

inline bool to_dense(const MbrState& s, bool big_endian, double* out_psi_ri) noexcept {
    if (s.n_qubits > kMaxDenseQubits || out_psi_ri == nullptr) return false;
    const size_t n = s.n_qubits, d = s.d;
    const size_t dim = size_t(1) << n;
    std::memset(out_psi_ri, 0, dim * 2U * sizeof(double));

    auto bitrev = [n](uint32_t x) noexcept -> uint32_t {
        uint32_t r = 0U;
        for (size_t b = 0; b < n; ++b) {
            r = (r << 1U) | (x & 1U);
            x >>= 1U;
        }
        return r;
    };

    uint32_t x = static_cast<uint32_t>(s.h[0]);
    if (big_endian) x = bitrev(x);
    const double amp = std::ldexp(1.0, -static_cast<int>(d >> 1U)) * ((d & 1U) ? kInvSqrt2 : 1.0);
    if (d == 0U) {
        out_psi_ri[2U * x] = 1.0;
        return true;
    }

    uint32_t A_cur[kMaxDenseQubits];
    uint32_t gmask[kMaxDenseQubits];
    for (size_t j = 0; j < d; ++j) {
        uint32_t col = static_cast<uint32_t>(s.A[j * s.nw]);
        A_cur[j] = big_endian ? bitrev(col) : col;
        uint32_t gm = 0U;
        const uint8_t* row = &s.Gamma[j * d];
        for (size_t k = 0; k < d; ++k) {
            if (row[k] != 0U) gm |= (uint32_t(1) << k);
        }
        gmask[j] = gm;
    }

    uint32_t u = 0U;
    uint8_t q = 0U;
    out_psi_ri[2U * x] = amp;
    const uint32_t total = uint32_t(1) << d;
    for (uint32_t t = 1U; t < total; ++t) {
        const uint32_t j = static_cast<uint32_t>(__builtin_ctz(t));
        const uint32_t bit = uint32_t(1) << j;
        const uint32_t par = static_cast<uint32_t>(__builtin_popcount(gmask[j] & u) & 1);
        if ((u & bit) == 0U) {
            q = static_cast<uint8_t>((q + s.ell[j] + (par << 1U)) & 3U);
        } else {
            q = static_cast<uint8_t>((q + (4U - s.ell[j]) + (par << 1U)) & 3U);
        }
        u ^= bit;
        x ^= A_cur[j];
        const MbrComplex ph = i_pow(q);
        out_psi_ri[2U * x] = amp * ph.re;
        out_psi_ri[2U * x + 1U] = amp * ph.im;
    }
    return true;
}

inline MbrState* state_of(void* handle) noexcept {
    return static_cast<MbrState*>(handle);
}

inline const MbrState* state_of(const void* handle) noexcept {
    return static_cast<const MbrState*>(handle);
}

}  // namespace mbr
}  // namespace

extern "C" {

int32_t mbr_abi_v1(void) {
    return 1;
}

size_t mbr_live_states(void) {
    return mbr::g_live_states.load(std::memory_order_relaxed);
}

int32_t mbr_state_create_zero(size_t n_qubits, void** out_handle) {
    try {
        if (out_handle == nullptr || n_qubits == 0U || n_qubits > mbr::kMaxQubits) {
            return mbr::kErrInvalidArg;
        }
        const size_t nw = mbr::n_words_for_qubits(n_qubits);
        if (nw > kMaxElems) return mbr::kErrInvalidArg;
        mbr::MbrState* st = new mbr::MbrState(mbr::make_zero(n_qubits));
        mbr::g_live_states.fetch_add(1U, std::memory_order_relaxed);
        *out_handle = st;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_from_arrays(
    size_t n_qubits, size_t d,
    const uint64_t* h, const uint64_t* A, const uint8_t* ell,
    const uint8_t* Gamma, const int32_t* piv, void** out_handle
) {
    try {
        if (out_handle == nullptr || n_qubits == 0U || n_qubits > mbr::kMaxQubits
            || d > n_qubits || h == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const size_t nw = mbr::n_words_for_qubits(n_qubits);
        size_t a_elems = 0U, g_elems = 0U;
        if (nw > kMaxElems || !mul_within(d, nw, kMaxElems, a_elems)
            || !mul_within(d, d, kMaxElems, g_elems)) {
            return mbr::kErrInvalidArg;
        }
        if (d > 0U && (A == nullptr || ell == nullptr || Gamma == nullptr || piv == nullptr)) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrState st;
        st.n_qubits = n_qubits;
        st.nw = nw;
        st.d = d;
        st.h.assign(h, h + nw);
        if (d > 0U) {
            st.A.assign(A, A + a_elems);
            st.ell.assign(ell, ell + d);
            st.Gamma.assign(Gamma, Gamma + g_elems);
            st.piv.assign(piv, piv + d);
        }
        if (!mbr::validate_state(st)) return mbr::kErrInvalidArg;
        mbr::MbrState* heap = new mbr::MbrState(std::move(st));
        mbr::g_live_states.fetch_add(1U, std::memory_order_relaxed);
        *out_handle = heap;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_copy(const void* handle, void** out_handle) {
    try {
        if (handle == nullptr || out_handle == nullptr) return mbr::kErrInvalidArg;
        mbr::MbrState* copy = new mbr::MbrState(*mbr::state_of(handle));
        mbr::g_live_states.fetch_add(1U, std::memory_order_relaxed);
        *out_handle = copy;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

void mbr_state_destroy(void* handle) {
    if (handle != nullptr) {
        delete mbr::state_of(handle);
        mbr::g_live_states.fetch_sub(1U, std::memory_order_relaxed);
    }
}

int32_t mbr_state_header(const void* handle, size_t* out_n, size_t* out_d, size_t* out_nw) {
    try {
        if (handle == nullptr || out_n == nullptr || out_d == nullptr || out_nw == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* st = mbr::state_of(handle);
        *out_n = st->n_qubits;
        *out_d = st->d;
        *out_nw = st->nw;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_export_arrays(
    const void* handle,
    uint64_t* out_h, uint64_t* out_A, uint8_t* out_ell, uint8_t* out_Gamma, int32_t* out_piv
) {
    try {
        if (handle == nullptr || out_h == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrState* st = mbr::state_of(handle);
        if (st->d > 0U && (out_A == nullptr || out_ell == nullptr
                           || out_Gamma == nullptr || out_piv == nullptr)) {
            return mbr::kErrInvalidArg;
        }
        std::memcpy(out_h, st->h.data(), st->nw * sizeof(uint64_t));
        if (st->d > 0U) {
            std::memcpy(out_A, st->A.data(), st->A.size() * sizeof(uint64_t));
            std::memcpy(out_ell, st->ell.data(), st->d * sizeof(uint8_t));
            std::memcpy(out_Gamma, st->Gamma.data(), st->Gamma.size() * sizeof(uint8_t));
            std::memcpy(out_piv, st->piv.data(), st->d * sizeof(int32_t));
        }
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_pauli(
    void* handle, const uint64_t* x, const uint64_t* z, uint8_t kappa, double* out_scalar_ri
) {
    try {
        if (handle == nullptr || x == nullptr || z == nullptr || out_scalar_ri == nullptr) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrState* st = mbr::state_of(handle);
        if (!mbr::tail_clear(x, st->n_qubits, st->nw) || !mbr::tail_clear(z, st->n_qubits, st->nw)) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrComplex f = mbr::apply_pauli(*st, x, z, kappa);
        out_scalar_ri[0] = f.re;
        out_scalar_ri[1] = f.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_qturn(
    void* handle, const uint64_t* x, const uint64_t* z, uint8_t kappa, double* out_scalar_ri
) {
    try {
        if (handle == nullptr || x == nullptr || z == nullptr || out_scalar_ri == nullptr) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrState* st = mbr::state_of(handle);
        if (!mbr::tail_clear(x, st->n_qubits, st->nw) || !mbr::tail_clear(z, st->n_qubits, st->nw)) {
            return mbr::kErrInvalidArg;
        }
        if (!mbr::is_hermitian(x, z, kappa, st->nw)) return mbr::kErrInvalidArg;
        mbr::MbrState tmp = *st;
        const mbr::MbrComplex f = mbr::apply_qturn(tmp, x, z, kappa);
        *st = std::move(tmp);
        out_scalar_ri[0] = f.re;
        out_scalar_ri[1] = f.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_zproject(
    void* handle, const uint64_t* z, uint8_t sigma, uint8_t m, double* out_scalar_ri
) {
    try {
        if (handle == nullptr || z == nullptr || out_scalar_ri == nullptr
            || sigma > 1U || m > 1U) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrState* st = mbr::state_of(handle);
        if (!mbr::tail_clear(z, st->n_qubits, st->nw)) return mbr::kErrInvalidArg;
        mbr::MbrState tmp = *st;
        const mbr::MbrComplex f = mbr::apply_zproject(tmp, z, sigma, m);
        if (f.re != 0.0 || f.im != 0.0) {
            *st = std::move(tmp);
        }
        out_scalar_ri[0] = f.re;
        out_scalar_ri[1] = f.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_project(
    void* handle, const uint64_t* x, const uint64_t* z, uint8_t kappa, uint8_t m,
    double* out_scalar_ri
) {
    try {
        if (handle == nullptr || x == nullptr || z == nullptr || out_scalar_ri == nullptr
            || m > 1U) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrState* st = mbr::state_of(handle);
        if (!mbr::tail_clear(x, st->n_qubits, st->nw) || !mbr::tail_clear(z, st->n_qubits, st->nw)) {
            return mbr::kErrInvalidArg;
        }
        if (!mbr::is_hermitian(x, z, kappa, st->nw)) return mbr::kErrInvalidArg;
        mbr::MbrState tmp = *st;
        const mbr::MbrComplex f = mbr::apply_project(tmp, x, z, kappa, m);
        if (f.re != 0.0 || f.im != 0.0) {
            *st = std::move(tmp);
        }
        out_scalar_ri[0] = f.re;
        out_scalar_ri[1] = f.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_definite(
    const void* handle, const uint64_t* x, const uint64_t* z, uint8_t kappa, int32_t* out_ev
) {
    try {
        if (handle == nullptr || x == nullptr || z == nullptr || out_ev == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* st = mbr::state_of(handle);
        if (!mbr::tail_clear(x, st->n_qubits, st->nw) || !mbr::tail_clear(z, st->n_qubits, st->nw)) {
            return mbr::kErrInvalidArg;
        }
        if (!mbr::is_hermitian(x, z, kappa, st->nw)) return mbr::kErrInvalidArg;
        *out_ev = static_cast<int32_t>(mbr::definite(*st, x, z, kappa));
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_generators(
    const void* handle, uint64_t* out_x, uint64_t* out_z, uint8_t* out_kappa
) {
    try {
        if (handle == nullptr || out_x == nullptr || out_z == nullptr || out_kappa == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* st = mbr::state_of(handle);
        size_t total_words = 0U;
        if (!mul_within(st->n_qubits, st->nw, kMaxElems, total_words)) {
            return mbr::kErrInvalidArg;
        }
        std::vector<uint64_t> gx, gz;
        std::vector<uint8_t> gk;
        mbr::generators(*st, gx, gz, gk);
        std::memcpy(out_x, gx.data(), total_words * sizeof(uint64_t));
        std::memcpy(out_z, gz.data(), total_words * sizeof(uint64_t));
        std::memcpy(out_kappa, gk.data(), st->n_qubits * sizeof(uint8_t));
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_canon(void* handle, double* out_scalar_ri) {
    try {
        if (handle == nullptr || out_scalar_ri == nullptr) return mbr::kErrInvalidArg;
        mbr::MbrState* st = mbr::state_of(handle);
        mbr::MbrState tmp = *st;
        const mbr::MbrComplex f = mbr::canon(tmp);
        *st = std::move(tmp);
        out_scalar_ri[0] = f.re;
        out_scalar_ri[1] = f.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_canonical_key_size(const void* handle, size_t* out_nbytes) {
    try {
        if (handle == nullptr || out_nbytes == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrState* st = mbr::state_of(handle);
        *out_nbytes = mbr::canonical_key_size(st->n_qubits, st->d);
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_canonical_key(
    const void* handle, uint8_t* out_buf, size_t buf_len, size_t* out_nbytes
) {
    try {
        if (handle == nullptr || out_nbytes == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrState* st = mbr::state_of(handle);
        const size_t needed = mbr::canonical_key_size(st->n_qubits, st->d);
        *out_nbytes = needed;
        if (out_buf == nullptr || buf_len < needed) return mbr::kErrInvalidArg;
        if (mbr::is_canonical(*st)) {
            mbr::serialize_canonical_unchecked(*st, out_buf);
        } else {
            mbr::MbrState tmp = *st;
            mbr::canon_phase4(tmp);
            mbr::serialize_canonical_unchecked(tmp, out_buf);
        }
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_from_canonical_key(
    const uint8_t* buf, size_t buf_len, void** out_handle
) {
    try {
        if (buf == nullptr || out_handle == nullptr) return mbr::kErrInvalidArg;
        mbr::MbrState st;
        if (!mbr::deserialize_canonical(buf, buf_len, st)) return mbr::kErrInvalidArg;
        mbr::MbrState* heap = new mbr::MbrState(std::move(st));
        mbr::g_live_states.fetch_add(1U, std::memory_order_relaxed);
        *out_handle = heap;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_overlap(const void* a, const void* b, double* out_overlap_ri) {
    try {
        if (a == nullptr || b == nullptr || out_overlap_ri == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* sa = mbr::state_of(a);
        const mbr::MbrState* sb = mbr::state_of(b);
        if (sa->n_qubits != sb->n_qubits) return mbr::kErrInvalidArg;
        const mbr::MbrComplex ov = mbr::overlap(*sa, *sb);
        out_overlap_ri[0] = ov.re;
        out_overlap_ri[1] = ov.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_same_phase(
    const void* a, const void* b, int32_t* out_equal, double* out_ratio_ri
) {
    try {
        if (a == nullptr || b == nullptr || out_equal == nullptr || out_ratio_ri == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* sa = mbr::state_of(a);
        const mbr::MbrState* sb = mbr::state_of(b);
        if (sa->n_qubits != sb->n_qubits) return mbr::kErrInvalidArg;
        mbr::MbrComplex ratio{0.0, 0.0};
        const bool eq = mbr::same_phase(*sa, *sb, ratio);
        *out_equal = eq ? 1 : 0;
        out_ratio_ri[0] = ratio.re;
        out_ratio_ri[1] = ratio.im;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_state_to_dense(
    const void* handle, int32_t big_endian, double* out_psi_ri, size_t out_complex_len
) {
    try {
        if (handle == nullptr || out_psi_ri == nullptr
            || (big_endian != 0 && big_endian != 1)) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrState* st = mbr::state_of(handle);
        if (st->n_qubits > mbr::kMaxDenseQubits) return mbr::kErrInvalidArg;
        const size_t expected = size_t(1) << st->n_qubits;
        if (out_complex_len != expected) return mbr::kErrInvalidArg;
        if (!mbr::to_dense(*st, big_endian != 0, out_psi_ri)) return mbr::kErrInvalidArg;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

}  // extern "C"
