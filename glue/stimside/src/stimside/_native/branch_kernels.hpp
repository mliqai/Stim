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
#include <map>
#include <new>
#include <string>
#include <unordered_map>
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

// ---------------------------------------------------------------------------
// WP3: Resolve step, Mobius is_clifford, choose_split, MbrStore & GF(2) absorption
// ---------------------------------------------------------------------------

constexpr uint8_t kKindT = 0U;
constexpr uint8_t kKindCcz = 1U;
constexpr uint8_t kKindSkip = 2U;
constexpr uint8_t kKindCfault = 3U;
constexpr uint8_t kKindMeas = 4U;
constexpr uint8_t kKindPeek = 5U;
constexpr uint8_t kKindPost = 6U;
constexpr uint8_t kKindUreset = 7U;
constexpr size_t kNumKinds = 8U;

constexpr uint8_t kMeasDefinite = 0U;
constexpr uint8_t kMeasAbsorbed = 1U;
constexpr uint8_t kMeasForked = 2U;

constexpr uint32_t kInvalidStateId = std::numeric_limits<uint32_t>::max();
constexpr double kPruneRelTol = 1e-13;
constexpr double kImpossibleProb = 1e-13;
constexpr double kDriftTol = 1e-8;
constexpr size_t kOver20SearchCap = 10000U;

inline uint32_t clz64(uint64_t x) noexcept {
    return static_cast<uint32_t>(__builtin_clzll(x));
}

inline MbrComplex cadd(MbrComplex a, MbrComplex b) noexcept {
    return {a.re + b.re, a.im + b.im};
}

inline MbrComplex csub(MbrComplex a, MbrComplex b) noexcept {
    return {a.re - b.re, a.im - b.im};
}

inline MbrComplex cconj(MbrComplex a) noexcept {
    return {a.re, -a.im};
}

inline double cabs2(MbrComplex a) noexcept {
    return a.re * a.re + a.im * a.im;
}

inline double cabs(MbrComplex a) noexcept {
    return std::hypot(a.re, a.im);
}

inline bool cfinite(MbrComplex a) noexcept {
    return std::isfinite(a.re) && std::isfinite(a.im);
}

inline uint8_t pauli_neg_kappa(uint8_t kappa) noexcept {
    return static_cast<uint8_t>((kappa + 2U) & 3U);
}

inline bool pauli_is_identity(const uint64_t* x, const uint64_t* z, size_t nw) noexcept {
    return !words_any_nonzero(x, nw) && !words_any_nonzero(z, nw);
}

inline bool pauli_xz_equal(
    const uint64_t* x1, const uint64_t* z1,
    const uint64_t* x2, const uint64_t* z2, size_t nw
) noexcept {
    return std::memcmp(x1, x2, nw * sizeof(uint64_t)) == 0
        && std::memcmp(z1, z2, nw * sizeof(uint64_t)) == 0;
}

// Left-multiplying quarter-turn conjugation: R(A) P R(A)^dag = i * A * P when anti(A, P) == 1.
inline void pauli_conj_by_qturn_inplace(
    uint64_t* px, uint64_t* pz, uint8_t& pk,
    const uint64_t* ax, const uint64_t* az, uint8_t ak, size_t nw
) noexcept {
    if (pauli_anti(ax, az, px, pz, nw) == 0U) return;
    const uint8_t s = word_dot_parity(az, px, nw);
    pk = static_cast<uint8_t>((ak + pk + (s << 1U) + 1U) & 3U);
    xor_words_inplace(px, ax, nw);
    xor_words_inplace(pz, az, nw);
}

inline void hash_combine_u64(size_t& h, uint64_t v) noexcept {
    h ^= static_cast<size_t>(splitmix64(v + 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(h)));
}

// Arbitrary-width non-negative bitmask with canonical trimmed storage and big-endian ordering
// matching Python `int` comparisons identically.
struct ParityVec {
    uint64_t w0 = 0ULL;
    std::vector<uint64_t> extra;  // words 1.., trimmed (no trailing zeros)

    inline bool is_zero() const noexcept {
        return w0 == 0ULL && extra.empty();
    }

    inline void trim() noexcept {
        while (!extra.empty() && extra.back() == 0ULL) {
            extra.pop_back();
        }
    }

    inline uint64_t word(size_t i) const noexcept {
        if (i == 0U) return w0;
        const size_t ei = i - 1U;
        return (ei < extra.size()) ? extra[ei] : 0ULL;
    }

    inline size_t n_words() const noexcept {
        return 1U + extra.size();
    }

    inline uint8_t get_bit(size_t b) const noexcept {
        return static_cast<uint8_t>((word(b >> 6U) >> (b & 63U)) & 1ULL);
    }

    inline void xor_bit(size_t b) {
        const size_t wi = b >> 6U;
        const uint64_t mask = uint64_t(1) << (b & 63U);
        if (wi == 0U) {
            w0 ^= mask;
            return;
        }
        if (wi > extra.size()) {
            extra.resize(wi, 0ULL);
        }
        extra[wi - 1U] ^= mask;
        trim();
    }

    inline void xor_with(const ParityVec& o) {
        w0 ^= o.w0;
        if (o.extra.size() > extra.size()) {
            extra.resize(o.extra.size(), 0ULL);
        }
        for (size_t i = 0; i < o.extra.size(); ++i) {
            extra[i] ^= o.extra[i];
        }
        trim();
    }

    inline size_t bit_length() const noexcept {
        if (!extra.empty()) {
            return 64U * extra.size() + static_cast<size_t>(64U - clz64(extra.back()));
        }
        if (w0 != 0ULL) {
            return static_cast<size_t>(64U - clz64(w0));
        }
        return 0U;
    }

    inline size_t lowest_bit() const noexcept {
        if (w0 != 0ULL) {
            return static_cast<size_t>(ctz64(w0));
        }
        for (size_t i = 0; i < extra.size(); ++i) {
            if (extra[i] != 0ULL) {
                return 64U * (i + 1U) + static_cast<size_t>(ctz64(extra[i]));
            }
        }
        return 0U;
    }

    inline bool operator==(const ParityVec& o) const noexcept {
        return w0 == o.w0 && extra == o.extra;
    }

    inline bool operator!=(const ParityVec& o) const noexcept {
        return !(*this == o);
    }

    inline bool operator<(const ParityVec& o) const noexcept {
        if (extra.size() != o.extra.size()) return extra.size() < o.extra.size();
        for (size_t i = extra.size(); i > 0U; --i) {
            if (extra[i - 1U] != o.extra[i - 1U]) return extra[i - 1U] < o.extra[i - 1U];
        }
        return w0 < o.w0;
    }
};

struct ParityVecHash {
    inline size_t operator()(const ParityVec& p) const noexcept {
        size_t h = static_cast<size_t>(splitmix64(p.w0));
        for (uint64_t w : p.extra) {
            hash_combine_u64(h, w);
        }
        return h;
    }
};

inline ParityVec parity_from_words(const uint64_t* words, size_t nw) {
    ParityVec p;
    if (nw == 0U) return p;
    p.w0 = words[0];
    if (nw > 1U) {
        size_t last = nw;
        while (last > 1U && words[last - 1U] == 0ULL) --last;
        if (last > 1U) {
            p.extra.assign(words + 1U, words + last);
        }
    }
    return p;
}

inline std::string serialize_parity_set(const std::vector<ParityVec>& O) {
    size_t total_words = 0U;
    for (const ParityVec& p : O) total_words += p.n_words();
    std::string out(4U + O.size() * 4U + total_words * 8U, '\0');
    uint8_t* dst = reinterpret_cast<uint8_t*>(&out[0]);
    write_u32_le(dst, static_cast<uint32_t>(O.size()));
    dst += 4U;
    for (const ParityVec& p : O) {
        const size_t nw = p.n_words();
        write_u32_le(dst, static_cast<uint32_t>(nw));
        dst += 4U;
        write_u64_le(dst, p.w0);
        dst += 8U;
        for (uint64_t w : p.extra) {
            write_u64_le(dst, w);
            dst += 8U;
        }
    }
    return out;
}

// Precondition: O is sorted ascending (big-endian ParityVec order), deduplicated, nonzero.
inline bool is_clifford_sorted(const std::vector<ParityVec>& O) {
    if (O.empty()) return true;
    if (O.size() < 15U) return false;
    const size_t r_eff = O.back().bit_length();
    if (r_eff > 20U) return false;

    const size_t nw_f = (r_eff <= 6U) ? 1U : (size_t(1) << (r_eff - 6U));
    uint64_t f_stack[16] = {};
    std::vector<uint64_t> f_heap;
    uint64_t* F = f_stack;
    if (nw_f > 16U) {
        f_heap.assign(nw_f, 0ULL);
        F = f_heap.data();
    }
    for (const ParityVec& x : O) {
        const uint32_t idx = static_cast<uint32_t>(x.w0);
        F[idx >> 6U] |= (uint64_t(1) << (idx & 63U));
    }

    static constexpr uint64_t kIntraMask[6] = {
        0x5555555555555555ULL,
        0x3333333333333333ULL,
        0x0F0F0F0F0F0F0F0FULL,
        0x00FF00FF00FF00FFULL,
        0x0000FFFF0000FFFFULL,
        0x00000000FFFFFFFFULL,
    };
    const size_t intra_end = std::min<size_t>(r_eff, 6U);
    for (size_t j = 0; j < intra_end; ++j) {
        const uint32_t shift = uint32_t(1) << j;
        const uint64_t mask = kIntraMask[j];
        for (size_t w = 0; w < nw_f; ++w) {
            F[w] ^= (F[w] >> shift) & mask;
        }
    }
    for (size_t j = 6U; j < r_eff; ++j) {
        const size_t step = size_t(1) << (j - 6U);
        const size_t stride = step << 1U;
        for (size_t base = 0; base < nw_f; base += stride) {
            for (size_t k = 0; k < step; ++k) {
                F[base + k] ^= F[base + k + step];
            }
        }
    }

    auto test_bit = [F](uint32_t idx) noexcept -> bool {
        return ((F[idx >> 6U] >> (idx & 63U)) & 1ULL) != 0ULL;
    };
    for (size_t a = 0; a < r_eff; ++a) {
        const uint32_t ba = uint32_t(1) << a;
        if (test_bit(ba)) return false;
        for (size_t b = a + 1U; b < r_eff; ++b) {
            const uint32_t bab = ba | (uint32_t(1) << b);
            if (test_bit(bab)) return false;
            for (size_t c = b + 1U; c < r_eff; ++c) {
                if (test_bit(bab | (uint32_t(1) << c))) return false;
            }
        }
    }
    return true;
}

inline std::vector<ParityVec> quotient_parity_set(
    const std::vector<ParityVec>& O, const ParityVec& lam
) {
    const size_t low = lam.lowest_bit();
    std::vector<ParityVec> tmp;
    tmp.reserve(O.size());
    for (const ParityVec& x : O) {
        ParityVec y = x;
        if (y.get_bit(low) != 0U) {
            y.xor_with(lam);
        }
        if (!y.is_zero()) {
            tmp.push_back(std::move(y));
        }
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<ParityVec> out;
    out.reserve(tmp.size());
    size_t i = 0U;
    while (i < tmp.size()) {
        size_t j = i + 1U;
        while (j < tmp.size() && tmp[j] == tmp[i]) ++j;
        if (((j - i) & 1U) != 0U) {
            out.push_back(std::move(tmp[i]));
        }
        i = j;
    }
    return out;
}

struct SplitEntry {
    ParityVec lam;
    int32_t depth = 0;
};

inline bool split_chain_dfs(
    const std::vector<ParityVec>& O,
    uint32_t depth,
    size_t& budget,
    std::unordered_map<std::string, uint32_t>& max_failed,
    ParityVec* out_first_lam
) {
    if (is_clifford_sorted(O)) return true;
    if (depth == 0U || budget == 0U) return false;
    --budget;
    const std::string key = serialize_parity_set(O);
    auto it = max_failed.find(key);
    if (it != max_failed.end() && it->second >= depth) return false;

    const ParityVec& x0 = O[0];
    const size_t m = O.size();
    for (size_t idx = 0; idx < m; ++idx) {
        ParityVec lam = x0;
        if (idx > 0U) {
            lam.xor_with(O[idx]);
        }
        std::vector<ParityVec> qO = quotient_parity_set(O, lam);
        if (split_chain_dfs(qO, depth - 1U, budget, max_failed, nullptr)) {
            if (out_first_lam != nullptr) *out_first_lam = std::move(lam);
            return true;
        }
        if (budget == 0U) return false;
    }
    max_failed[key] = depth;
    return false;
}

// Precondition: O is non-empty, sorted ascending, deduplicated, nonzero, and !is_clifford_sorted(O).
inline SplitEntry choose_split_sorted(
    const std::vector<ParityVec>& O,
    uint32_t max_depth,
    std::unordered_map<std::string, SplitEntry>* cache,
    uint64_t* subcache_hits,
    uint64_t* subcache_misses,
    uint64_t* bytes_estimate
) {
    std::string cache_key;
    if (cache != nullptr) {
        cache_key = serialize_parity_set(O);
        auto it = cache->find(cache_key);
        if (it != cache->end()) {
            if (subcache_hits != nullptr) *subcache_hits += 1U;
            return it->second;
        }
        if (subcache_misses != nullptr) *subcache_misses += 1U;
    }

    const bool over_20 = (O.back().bit_length() > 20U);
    size_t budget = over_20 ? kOver20SearchCap : std::numeric_limits<size_t>::max();
    std::unordered_map<std::string, uint32_t> max_failed;
    SplitEntry res{O[0], 0};
    for (uint32_t d = 1U; d <= max_depth; ++d) {
        ParityVec chosen;
        if (split_chain_dfs(O, d, budget, max_failed, &chosen)) {
            res.lam = std::move(chosen);
            res.depth = static_cast<int32_t>(d);
            break;
        }
        if (budget == 0U) break;
    }

    if (cache != nullptr) {
        const size_t entry_bytes = 64U + cache_key.capacity()
            + sizeof(SplitEntry) + res.lam.extra.capacity() * sizeof(uint64_t);
        cache->emplace(std::move(cache_key), res);
        if (bytes_estimate != nullptr) *bytes_estimate += entry_bytes;
    }
    return res;
}

// ---------------------------------------------------------------------------
// MbrStore: sequential-ID path trie, canonical state pool, 5 sub-caches, counters
// ---------------------------------------------------------------------------

struct MbrBranch {
    MbrComplex w{1.0, 0.0};
    uint32_t state_id = 0U;
};

struct MbrPending {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;
    uint8_t kappa = 0U;  // normalized to sign +1
    uint8_t e = 0U;      // in {1..7}
};

struct MbrNode {
    std::vector<MbrBranch> branches;  // sorted by state_pool[state_id].key
    std::vector<MbrPending> pend;     // insertion/update order matching mfe.py
    uint32_t transient_peak = 1U;
};

struct StateEntry {
    MbrState state;
    std::string key;
};

struct TransKey {
    int64_t node_id = 0;
    uint64_t event_id = 0U;
    uint32_t sign_bits = 0U;
    uint8_t kind = 0U;

    inline bool operator==(const TransKey& o) const noexcept {
        return node_id == o.node_id && event_id == o.event_id
            && sign_bits == o.sign_bits && kind == o.kind;
    }
};

struct TransKeyHash {
    inline size_t operator()(const TransKey& k) const noexcept {
        size_t h = static_cast<size_t>(splitmix64(static_cast<uint64_t>(k.node_id)));
        hash_combine_u64(h, k.event_id);
        hash_combine_u64(h, (static_cast<uint64_t>(k.sign_bits) << 8U) | k.kind);
        return h;
    }
};

struct MeasKey {
    int64_t node_id = 0;
    uint64_t event_id = 0U;
    uint8_t kind = 0U;

    inline bool operator==(const MeasKey& o) const noexcept {
        return node_id == o.node_id && event_id == o.event_id && kind == o.kind;
    }
};

struct MeasKeyHash {
    inline size_t operator()(const MeasKey& k) const noexcept {
        size_t h = static_cast<size_t>(splitmix64(static_cast<uint64_t>(k.node_id)));
        hash_combine_u64(h, k.event_id);
        hash_combine_u64(h, k.kind);
        return h;
    }
};

struct MeasEntry {
    double p0 = 0.0;
    int64_t child0 = -1;
    int64_t child1 = -1;
    uint8_t meas_type = kMeasDefinite;
    std::vector<uint64_t> absorb_x;
    std::vector<uint64_t> absorb_z;
};

struct PeekKey {
    int64_t node_id = 0;
    uint64_t event_id = 0U;

    inline bool operator==(const PeekKey& o) const noexcept {
        return node_id == o.node_id && event_id == o.event_id;
    }
};

struct PeekKeyHash {
    inline size_t operator()(const PeekKey& k) const noexcept {
        size_t h = static_cast<size_t>(splitmix64(static_cast<uint64_t>(k.node_id)));
        hash_combine_u64(h, k.event_id);
        return h;
    }
};

struct StateStrKey {
    uint32_t state_id = 0U;
    std::string aux;

    inline bool operator==(const StateStrKey& o) const noexcept {
        return state_id == o.state_id && aux == o.aux;
    }
};

struct StateStrKeyHash {
    inline size_t operator()(const StateStrKey& k) const noexcept {
        size_t h = static_cast<size_t>(splitmix64(k.state_id));
        hash_combine_u64(h, std::hash<std::string>{}(k.aux));
        return h;
    }
};

struct ProjEntry {
    MbrComplex f{0.0, 0.0};
    uint32_t state_id = kInvalidStateId;
    bool valid = false;
};

struct MbrStore {
    size_t n_qubits = 0U;
    size_t nw = 0U;
    size_t max_components = 0U;  // 0 = unlimited

    std::vector<StateEntry> state_pool;
    std::unordered_map<std::string, uint32_t> state_intern;
    std::vector<MbrNode> nodes;
    std::map<uint64_t, uint64_t> node_branches_hist;

    std::unordered_map<TransKey, int64_t, TransKeyHash> trans;
    std::unordered_map<MeasKey, MeasEntry, MeasKeyHash> meas;
    std::unordered_map<PeekKey, double, PeekKeyHash> peek;

    // Five exact-key sub-caches
    std::unordered_map<StateStrKey, std::vector<MbrBranch>, StateStrKeyHash> resolve_cache;
    std::unordered_map<StateStrKey, ProjEntry, StateStrKeyHash> project_cache;
    std::unordered_map<uint64_t, MbrComplex> overlap_cache;
    std::unordered_map<StateStrKey, int8_t, StateStrKeyHash> definite_cache;
    std::unordered_map<std::string, SplitEntry> split_cache;

    // Cumulative counters (preserved across clear())
    uint64_t store_clears = 0U;
    uint64_t transitions_new[kNumKinds] = {};
    uint64_t transitions_hit[kNumKinds] = {};
    uint64_t resolve_calls = 0U;
    uint64_t splits = 0U;
    uint64_t subcache_hits = 0U;
    uint64_t subcache_misses = 0U;
    uint64_t meas_definite = 0U;
    uint64_t meas_absorbed = 0U;
    uint64_t meas_forked = 0U;

    // Current heap estimate (recomputed on clear())
    uint64_t bytes_estimate = 0U;

    inline uint64_t state_bytes(const MbrState& s, const std::string& key) const noexcept {
        return sizeof(StateEntry)
            + s.h.capacity() * sizeof(uint64_t)
            + s.A.capacity() * sizeof(uint64_t)
            + s.ell.capacity() * sizeof(uint8_t)
            + s.Gamma.capacity() * sizeof(uint8_t)
            + s.piv.capacity() * sizeof(int32_t)
            + key.capacity();
    }

    inline uint64_t node_bytes(const MbrNode& nd) const noexcept {
        uint64_t b = sizeof(MbrNode)
            + nd.branches.capacity() * sizeof(MbrBranch)
            + nd.pend.capacity() * sizeof(MbrPending);
        for (const MbrPending& p : nd.pend) {
            b += (p.x.capacity() + p.z.capacity()) * sizeof(uint64_t);
        }
        return b;
    }

    inline uint32_t intern_canonical(MbrState&& st) {
        std::string key = serialize_canonical_str(st);
        auto it = state_intern.find(key);
        if (it != state_intern.end()) return it->second;
        const uint32_t id = static_cast<uint32_t>(state_pool.size());
        const uint64_t add_bytes = state_bytes(st, key) + 64U + key.capacity();
        state_intern.emplace(key, id);
        state_pool.push_back({std::move(st), std::move(key)});
        bytes_estimate += add_bytes;
        return id;
    }

    inline int64_t push_node(
        std::vector<MbrBranch> branches,
        std::vector<MbrPending> pend,
        uint32_t transient_peak
    ) {
        const uint32_t nb = static_cast<uint32_t>(branches.size());
        if (transient_peak < nb) transient_peak = nb;
        const int64_t nid = static_cast<int64_t>(nodes.size());
        MbrNode nd{std::move(branches), std::move(pend), transient_peak};
        bytes_estimate += node_bytes(nd);
        nodes.push_back(std::move(nd));
        node_branches_hist[static_cast<uint64_t>(nb)] += 1U;
        return nid;
    }

    inline void seed_root() {
        state_pool.clear();
        state_intern.clear();
        nodes.clear();
        node_branches_hist.clear();
        trans.clear();
        meas.clear();
        peek.clear();
        resolve_cache.clear();
        project_cache.clear();
        overlap_cache.clear();
        definite_cache.clear();
        split_cache.clear();
        bytes_estimate = sizeof(MbrStore);
        const uint32_t z0 = intern_canonical(make_zero(n_qubits));
        push_node({{MbrComplex{1.0, 0.0}, z0}}, {}, 1U);
    }

    inline std::string serialize_pauli_key(
        const uint64_t* x, const uint64_t* z, uint8_t kappa
    ) const {
        std::string out(nw * 16U + 1U, '\0');
        uint8_t* p = reinterpret_cast<uint8_t*>(&out[0]);
        for (size_t w = 0; w < nw; ++w) {
            write_u64_le(p, x[w]);
            p += 8U;
        }
        for (size_t w = 0; w < nw; ++w) {
            write_u64_le(p, z[w]);
            p += 8U;
        }
        *p = static_cast<uint8_t>(kappa & 3U);
        return out;
    }

    inline std::string serialize_pauli_m_key(
        const uint64_t* x, const uint64_t* z, uint8_t kappa, uint8_t m
    ) const {
        std::string out(nw * 16U + 2U, '\0');
        uint8_t* p = reinterpret_cast<uint8_t*>(&out[0]);
        for (size_t w = 0; w < nw; ++w) {
            write_u64_le(p, x[w]);
            p += 8U;
        }
        for (size_t w = 0; w < nw; ++w) {
            write_u64_le(p, z[w]);
            p += 8U;
        }
        p[0] = static_cast<uint8_t>(kappa & 3U);
        p[1] = static_cast<uint8_t>(m & 1U);
        return out;
    }

    inline std::string serialize_pending_key(const std::vector<MbrPending>& pend) const {
        std::string out(4U + pend.size() * (nw * 16U + 2U), '\0');
        uint8_t* p = reinterpret_cast<uint8_t*>(&out[0]);
        write_u32_le(p, static_cast<uint32_t>(pend.size()));
        p += 4U;
        for (const MbrPending& it : pend) {
            for (size_t w = 0; w < nw; ++w) {
                write_u64_le(p, it.x[w]);
                p += 8U;
            }
            for (size_t w = 0; w < nw; ++w) {
                write_u64_le(p, it.z[w]);
                p += 8U;
            }
            p[0] = static_cast<uint8_t>(it.kappa & 3U);
            p[1] = static_cast<uint8_t>(it.e & 7U);
            p += 2U;
        }
        return out;
    }

    inline int8_t cached_definite(
        uint32_t state_id, const uint64_t* x, const uint64_t* z, uint8_t kappa
    ) {
        StateStrKey key{state_id, serialize_pauli_key(x, z, kappa)};
        auto it = definite_cache.find(key);
        if (it != definite_cache.end()) {
            subcache_hits += 1U;
            return it->second;
        }
        subcache_misses += 1U;
        const int8_t ev = definite(state_pool[state_id].state, x, z, static_cast<uint8_t>(kappa & 3U));
        const uint64_t add_bytes = 64U + key.aux.capacity();
        definite_cache.emplace(std::move(key), ev);
        bytes_estimate += add_bytes;
        return ev;
    }

    inline ProjEntry cached_project(
        uint32_t state_id, const uint64_t* x, const uint64_t* z, uint8_t kappa, uint8_t m
    ) {
        StateStrKey key{state_id, serialize_pauli_m_key(x, z, kappa, m)};
        auto it = project_cache.find(key);
        if (it != project_cache.end()) {
            subcache_hits += 1U;
            return it->second;
        }
        subcache_misses += 1U;
        MbrState qc = state_pool[state_id].state;
        const MbrComplex f = apply_project(qc, x, z, static_cast<uint8_t>(kappa & 3U), m);
        ProjEntry entry;
        if (f.re != 0.0 || f.im != 0.0) {
            const MbrComplex fc = cmul(f, canon(qc));
            const uint32_t cid = intern_canonical(std::move(qc));
            entry = {fc, cid, true};
        }
        const uint64_t add_bytes = 64U + key.aux.capacity() + sizeof(ProjEntry);
        project_cache.emplace(std::move(key), entry);
        bytes_estimate += add_bytes;
        return entry;
    }

    inline MbrComplex cached_overlap(uint32_t id_a, uint32_t id_b) {
        if (id_a == id_b) return {1.0, 0.0};
        const StateEntry& ea = state_pool[id_a];
        const StateEntry& eb = state_pool[id_b];
        // Both states in state_pool are canonical, so equal d & A with distinct h guarantees
        // disjoint affine cosets without needing cache lookup or generator projection.
        if (ea.state.d == eb.state.d && ea.state.A == eb.state.A && ea.state.h != eb.state.h) {
            return {0.0, 0.0};
        }
        bool swapped = false;
        uint32_t lo = id_a;
        uint32_t hi = id_b;
        if (eb.key < ea.key) {
            lo = id_b;
            hi = id_a;
            swapped = true;
        }
        const uint64_t okey = (static_cast<uint64_t>(lo) << 32U) | static_cast<uint64_t>(hi);
        auto it = overlap_cache.find(okey);
        if (it != overlap_cache.end()) {
            subcache_hits += 1U;
            return swapped ? cconj(it->second) : it->second;
        }
        subcache_misses += 1U;
        const MbrComplex ov = overlap(state_pool[lo].state, state_pool[hi].state);
        overlap_cache.emplace(okey, ov);
        bytes_estimate += 64U + sizeof(MbrComplex);
        return swapped ? cconj(ov) : ov;
    }
};

std::atomic<size_t> g_live_stores{0};

inline MbrStore* store_of(void* handle) noexcept {
    return static_cast<MbrStore*>(handle);
}

inline const MbrStore* store_of(const void* handle) noexcept {
    return static_cast<const MbrStore*>(handle);
}

// Merge branches by state_id, prune at K1 threshold 1e-13 * min(1.0, w_max), and sort by
// canonical byte key (unsigned-byte lexicographic order, NEVER by state_id).
inline int32_t merge_and_sort_branches(
    const MbrStore& store,
    const std::vector<MbrBranch>& raw,
    std::vector<MbrBranch>& out
) {
    out.clear();
    if (raw.empty()) return kOk;
    double w_max = 0.0;
    for (const MbrBranch& b : raw) {
        if (!cfinite(b.w)) return kErrFloatDrift;
        const double mag = cabs(b.w);
        if (mag > w_max) w_max = mag;
    }
    const double thresh = kPruneRelTol * std::min(1.0, w_max);
    std::unordered_map<uint32_t, MbrComplex> groups;
    groups.reserve(raw.size());
    for (const MbrBranch& b : raw) {
        auto it = groups.find(b.state_id);
        if (it == groups.end()) {
            groups.emplace(b.state_id, b.w);
        } else {
            it->second = cadd(it->second, b.w);
        }
    }
    out.reserve(groups.size());
    for (const auto& kv : groups) {
        if (!cfinite(kv.second)) return kErrFloatDrift;
        if (cabs(kv.second) > thresh) {
            out.push_back({kv.second, kv.first});
        }
    }
    std::sort(out.begin(), out.end(), [&store](const MbrBranch& a, const MbrBranch& b) {
        return store.state_pool[a.state_id].key < store.state_pool[b.state_id].key;
    });
    return kOk;
}

inline int32_t first_slot_in_3nw(const uint64_t* v, size_t n, size_t nw) noexcept {
    for (size_t seg = 0; seg < 3U; ++seg) {
        const uint64_t* s = v + seg * nw;
        for (size_t w = 0; w < nw; ++w) {
            if (s[w] != 0ULL) {
                return static_cast<int32_t>(seg * n + (w << 6U) + static_cast<size_t>(ctz64(s[w])));
            }
        }
    }
    return -1;
}

inline MbrComplex apply_spow(MbrState& s, const MbrPauli& B, uint8_t k) {
    switch (k & 3U) {
        case 0U:
            return {1.0, 0.0};
        case 1U:
            return cmul(
                omega_pow(1U),
                apply_qturn(s, B.x.data(), B.z.data(), pauli_neg_kappa(B.kappa))
            );
        case 2U:
            return apply_pauli(s, B.x.data(), B.z.data(), B.kappa);
        default:
            return cmul(
                omega_pow(7U),
                apply_qturn(s, B.x.data(), B.z.data(), B.kappa)
            );
    }
}

// Resolve a single canonical stabilizer state against `pend` (preserving input `pend` order).
// Never inserts aborted subtrees into `store.resolve_cache`.
inline int32_t resolve_one(
    MbrStore& store,
    uint32_t state_id,
    const std::vector<MbrPending>& pend,
    const std::string& pk,
    size_t leaf_limit,
    std::vector<MbrBranch>& out_branches,
    size_t& out_peak
) {
    const size_t n = store.n_qubits;
    const size_t nw = store.nw;

    std::vector<uint64_t> gx, gz;
    std::vector<uint8_t> gk;
    generators(store.state_pool[state_id].state, gx, gz, gk);

    std::vector<uint64_t> GX(n * nw, 0ULL);
    std::vector<uint64_t> GZ(n * nw, 0ULL);
    for (size_t i = 0; i < n; ++i) {
        const size_t iw = i >> 6U;
        const uint64_t ibit = uint64_t(1) << (i & 63U);
        const uint64_t* gxi = &gx[i * nw];
        const uint64_t* gzi = &gz[i * nw];
        for (size_t w = 0; w < nw; ++w) {
            uint64_t m = gxi[w];
            while (m != 0ULL) {
                const size_t qb = (w << 6U) + static_cast<size_t>(ctz64(m));
                GX[qb * nw + iw] |= ibit;
                m &= (m - 1ULL);
            }
            m = gzi[w];
            while (m != 0ULL) {
                const size_t qb = (w << 6U) + static_cast<size_t>(ctz64(m));
                GZ[qb * nw + iw] |= ibit;
                m &= (m - 1ULL);
            }
        }
    }

    struct BasisRow {
        std::vector<uint64_t> v;  // 3 * nw
        MbrPauli R;
        int32_t var_idx = -1;
        uint8_t val = 0U;
    };
    std::vector<int32_t> basis_at(3U * n, -1);
    std::vector<BasisRow> basis_rows;
    basis_rows.reserve(pend.size());
    std::vector<MbrPauli> indef;
    std::unordered_map<ParityVec, uint8_t, ParityVecHash> coef;
    uint8_t const_ph = 0U;

    std::vector<uint64_t> v(3U * nw, 0ULL);
    for (const MbrPending& item : pend) {
        std::fill(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(nw), 0ULL);
        for (size_t w = 0; w < nw; ++w) {
            uint64_t m = item.x[w];
            while (m != 0ULL) {
                const size_t qb = (w << 6U) + static_cast<size_t>(ctz64(m));
                xor_words_inplace(v.data(), &GZ[qb * nw], nw);
                m &= (m - 1ULL);
            }
            m = item.z[w];
            while (m != 0ULL) {
                const size_t qb = (w << 6U) + static_cast<size_t>(ctz64(m));
                xor_words_inplace(v.data(), &GX[qb * nw], nw);
                m &= (m - 1ULL);
            }
        }
        std::memcpy(&v[nw], item.x.data(), nw * sizeof(uint64_t));
        std::memcpy(&v[2U * nw], item.z.data(), nw * sizeof(uint64_t));

        MbrPauli R{item.x, item.z, item.kappa};
        ParityVec xx;
        uint8_t kap = 0U;
        while (true) {
            const int32_t slot = first_slot_in_3nw(v.data(), n, nw);
            if (slot < 0) break;
            const int32_t bi = basis_at[static_cast<size_t>(slot)];
            if (bi < 0) break;
            const BasisRow& hit = basis_rows[static_cast<size_t>(bi)];
            xor_words_inplace(v.data(), hit.v.data(), 3U * nw);
            pauli_mul_inplace(
                R.x.data(), R.z.data(), R.kappa,
                hit.R.x.data(), hit.R.z.data(), hit.R.kappa, nw
            );
            if (hit.var_idx < 0) {
                kap ^= hit.val;
            } else {
                xx.xor_bit(static_cast<size_t>(hit.var_idx));
            }
        }
        const int32_t slot = first_slot_in_3nw(v.data(), n, nw);
        if (slot >= 0) {
            if (static_cast<size_t>(slot) < n) {
                const int32_t r_idx = static_cast<int32_t>(indef.size());
                basis_at[static_cast<size_t>(slot)] = static_cast<int32_t>(basis_rows.size());
                basis_rows.push_back({v, R, r_idx, 0U});
                xx.xor_bit(static_cast<size_t>(r_idx));
                indef.push_back(std::move(R));
            } else {
                const int8_t ev = store.cached_definite(state_id, R.x.data(), R.z.data(), R.kappa);
                if (ev == 0) return kErrInvalidArg;
                const uint8_t val = (ev < 0) ? 1U : 0U;
                basis_at[static_cast<size_t>(slot)] = static_cast<int32_t>(basis_rows.size());
                basis_rows.push_back({v, std::move(R), -1, val});
                kap ^= val;
            }
        } else {
            if (!pauli_is_identity(R.x.data(), R.z.data(), nw) || ((R.kappa & 1U) != 0U)) {
                return kErrInvalidArg;
            }
            kap ^= static_cast<uint8_t>((R.kappa >> 1U) & 1U);
        }
        const uint8_t e = static_cast<uint8_t>(item.e & 7U);
        const_ph = static_cast<uint8_t>((const_ph + (kap ? e : 0U)) & 7U);
        if (!xx.is_zero()) {
            const uint8_t delta = kap ? static_cast<uint8_t>((8U - e) & 7U) : e;
            uint8_t& c = coef[xx];
            c = static_cast<uint8_t>((c + delta) & 7U);
        }
    }

    std::vector<ParityVec> O;
    O.reserve(coef.size());
    for (const auto& kv : coef) {
        if ((kv.second & 1U) != 0U) {
            O.push_back(kv.first);
        }
    }
    std::sort(O.begin(), O.end());

    if (!is_clifford_sorted(O)) {
        store.splits += 1U;
        const SplitEntry sp = choose_split_sorted(
            O, 5U, &store.split_cache,
            &store.subcache_hits, &store.subcache_misses, &store.bytes_estimate
        );
        MbrPauli A;
        bool has_A = false;
        for (size_t j = 0; j < indef.size(); ++j) {
            if (sp.lam.get_bit(j) != 0U) {
                if (!has_A) {
                    A = indef[j];
                    has_A = true;
                } else {
                    pauli_mul_inplace(
                        A.x.data(), A.z.data(), A.kappa,
                        indef[j].x.data(), indef[j].z.data(), indef[j].kappa, nw
                    );
                }
            }
        }
        if (!has_A) return kErrInvalidArg;

        std::vector<MbrBranch> res;
        size_t peak = 0U;
        for (uint8_t mm = 0U; mm < 2U; ++mm) {
            if (leaf_limit > 0U && res.size() >= leaf_limit) return kErrBudget;
            MbrState qc = store.state_pool[state_id].state;
            MbrComplex f = apply_project(qc, A.x.data(), A.z.data(), A.kappa, mm);
            if (std::fabs(cabs(f) * kSqrt2 - 1.0) >= 1e-9) {
                return kErrFloatDrift;
            }
            f = cmul(f, canon(qc));
            const uint32_t qc_id = store.intern_canonical(std::move(qc));
            StateStrKey rkey{qc_id, pk};
            auto it = store.resolve_cache.find(rkey);
            if (it != store.resolve_cache.end()) {
                store.subcache_hits += 1U;
                const std::vector<MbrBranch>& sub = it->second;
                if (leaf_limit > 0U && res.size() + sub.size() > leaf_limit) {
                    return kErrBudget;
                }
                for (const MbrBranch& b : sub) {
                    res.push_back({cmul(f, b.w), b.state_id});
                }
                if (sub.size() > peak) peak = sub.size();
            } else {
                store.subcache_misses += 1U;
                const size_t rem = (leaf_limit == 0U) ? 0U : (leaf_limit - res.size());
                std::vector<MbrBranch> sub;
                size_t sub_peak = 0U;
                const int32_t rc = resolve_one(store, qc_id, pend, pk, rem, sub, sub_peak);
                if (rc != kOk) return rc;
                if (leaf_limit > 0U && res.size() + sub.size() > leaf_limit) {
                    return kErrBudget;
                }
                const uint64_t add_bytes = 64U + rkey.aux.capacity()
                    + sub.capacity() * sizeof(MbrBranch);
                auto ins = store.resolve_cache.emplace(std::move(rkey), std::move(sub));
                store.bytes_estimate += add_bytes;
                const std::vector<MbrBranch>& cached_sub = ins.first->second;
                for (const MbrBranch& b : cached_sub) {
                    res.push_back({cmul(f, b.w), b.state_id});
                }
                if (sub_peak > peak) peak = sub_peak;
            }
        }
        if (res.size() > peak) peak = res.size();
        out_branches = std::move(res);
        out_peak = peak;
        return kOk;
    }

    // Clifford base case
    const size_t r = indef.size();
    std::vector<uint8_t> mu1(r, 0U);
    std::map<std::pair<size_t, size_t>, uint8_t> mu2;
    std::vector<size_t> bits;
    for (const auto& kv : coef) {
        const uint8_t c = static_cast<uint8_t>(kv.second & 7U);
        if (c == 0U) continue;
        bits.clear();
        const ParityVec& xv = kv.first;
        const size_t nw_x = xv.n_words();
        for (size_t wi = 0; wi < nw_x; ++wi) {
            uint64_t m = xv.word(wi);
            while (m != 0ULL) {
                bits.push_back((wi << 6U) + static_cast<size_t>(ctz64(m)));
                m &= (m - 1ULL);
            }
        }
        for (size_t j : bits) {
            mu1[j] = static_cast<uint8_t>((mu1[j] + c) & 7U);
        }
        for (size_t a = 0; a < bits.size(); ++a) {
            for (size_t b = a + 1U; b < bits.size(); ++b) {
                uint8_t& m2 = mu2[{bits[a], bits[b]}];
                m2 = static_cast<uint8_t>((m2 + 6U * c) & 7U);
            }
        }
    }

    MbrState st = store.state_pool[state_id].state;
    MbrComplex f = omega_pow(const_ph);
    for (size_t j = 0; j < r; ++j) {
        if ((mu1[j] & 1U) != 0U) return kErrFloatDrift;
        if (mu1[j] != 0U) {
            f = cmul(f, apply_spow(st, indef[j], static_cast<uint8_t>(mu1[j] >> 1U)));
        }
    }
    for (const auto& kv : mu2) {
        if ((kv.second & 3U) != 0U) return kErrFloatDrift;
        if (kv.second != 0U) {
            const MbrPauli& pi = indef[kv.first.first];
            const MbrPauli& pj = indef[kv.first.second];
            MbrPauli pij = pi;
            pauli_mul_inplace(
                pij.x.data(), pij.z.data(), pij.kappa,
                pj.x.data(), pj.z.data(), pj.kappa, nw
            );
            f = cmul(f, apply_spow(st, pi, 1U));
            f = cmul(f, apply_spow(st, pj, 1U));
            f = cmul(f, apply_spow(st, pij, 3U));
        }
    }
    f = cmul(f, canon(st));
    const uint32_t out_id = store.intern_canonical(std::move(st));
    out_branches.clear();
    out_branches.push_back({f, out_id});
    out_peak = 1U;
    return kOk;
}

inline int32_t resolve_branches(
    MbrStore& store,
    const std::vector<MbrBranch>& in_branches,
    const std::vector<MbrPending>& in_pend,
    std::vector<MbrBranch>& out_merged,
    uint32_t& out_peak
) {
    if (in_pend.empty()) {
        std::vector<MbrBranch>(in_branches).swap(out_merged);
        out_peak = static_cast<uint32_t>(in_branches.size());
        return kOk;
    }
    store.resolve_calls += 1U;
    const std::string pk = store.serialize_pending_key(in_pend);

    size_t leaf_limit = 0U;
    if (store.max_components > 0U) {
        if (in_branches.size() <= 1U) {
            leaf_limit = store.max_components;
        } else {
            leaf_limit = std::min<size_t>(
                std::numeric_limits<size_t>::max() / 4U, store.max_components
            ) * 4U;
        }
    }

    std::vector<MbrBranch> raw_out;
    size_t peak = in_branches.size();
    for (const MbrBranch& br : in_branches) {
        StateStrKey rkey{br.state_id, pk};
        auto it = store.resolve_cache.find(rkey);
        if (it != store.resolve_cache.end()) {
            store.subcache_hits += 1U;
            const std::vector<MbrBranch>& sub = it->second;
            for (const MbrBranch& sb : sub) {
                raw_out.push_back({cmul(br.w, sb.w), sb.state_id});
            }
            if (sub.size() > peak) peak = sub.size();
        } else {
            store.subcache_misses += 1U;
            std::vector<MbrBranch> sub;
            size_t sub_peak = 0U;
            const int32_t rc = resolve_one(store, br.state_id, in_pend, pk, leaf_limit, sub, sub_peak);
            if (rc != kOk) return rc;
            const uint64_t add_bytes = 64U + rkey.aux.capacity()
                + sub.capacity() * sizeof(MbrBranch);
            auto ins = store.resolve_cache.emplace(std::move(rkey), std::move(sub));
            store.bytes_estimate += add_bytes;
            const std::vector<MbrBranch>& cached_sub = ins.first->second;
            for (const MbrBranch& sb : cached_sub) {
                raw_out.push_back({cmul(br.w, sb.w), sb.state_id});
            }
            if (sub_peak > peak) peak = sub_peak;
        }
    }
    if (raw_out.size() > peak) peak = raw_out.size();
    const int32_t mrc = merge_and_sort_branches(store, raw_out, out_merged);
    if (mrc != kOk) return mrc;
    if (!in_branches.empty() && out_merged.empty()) return kErrFloatDrift;
    if (store.max_components > 0U && out_merged.size() > store.max_components) {
        return kErrBudget;
    }
    out_peak = static_cast<uint32_t>(std::min<size_t>(peak, std::numeric_limits<uint32_t>::max()));
    return kOk;
}

// Multi-branch GF(2) measurement absorption search:
// Finds Hermitian Q_bar stabilizing every branch S_b with ONE common sign (+1 on S_0..S_{K-1}),
// commuting with all remaining working_pend axes, and anticommuting with M_bar.
inline bool find_absorb_pauli(
    MbrStore& store,
    const std::vector<MbrBranch>& branches,
    const std::vector<MbrPending>& working_pend,
    const uint64_t* mx, const uint64_t* mz,
    std::vector<uint64_t>& out_qx, std::vector<uint64_t>& out_qz
) {
    if (branches.empty()) return false;
    const size_t n = store.n_qubits;
    const size_t nw = store.nw;

    std::vector<uint64_t> gx, gz;
    std::vector<uint8_t> gk;
    generators(store.state_pool[branches[0].state_id].state, gx, gz, gk);

    std::vector<MbrPauli> basis;
    basis.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        MbrPauli p;
        p.x.assign(&gx[i * nw], &gx[(i + 1U) * nw]);
        p.z.assign(&gz[i * nw], &gz[(i + 1U) * nw]);
        p.kappa = gk[i];
        basis.push_back(std::move(p));
    }

    auto intersect_commuting = [&](const uint64_t* ax, const uint64_t* az) {
        size_t piv = basis.size();
        for (size_t i = 0; i < basis.size(); ++i) {
            if (pauli_anti(basis[i].x.data(), basis[i].z.data(), ax, az, nw) != 0U) {
                piv = i;
                break;
            }
        }
        if (piv == basis.size()) return;
        for (size_t i = piv + 1U; i < basis.size(); ++i) {
            if (pauli_anti(basis[i].x.data(), basis[i].z.data(), ax, az, nw) != 0U) {
                pauli_mul_inplace(
                    basis[i].x.data(), basis[i].z.data(), basis[i].kappa,
                    basis[piv].x.data(), basis[piv].z.data(), basis[piv].kappa, nw
                );
            }
        }
        basis.erase(basis.begin() + static_cast<std::ptrdiff_t>(piv));
    };

    // 1. Intersect with commutation kernel of remaining pending turns.
    for (const MbrPending& item : working_pend) {
        intersect_commuting(item.x.data(), item.z.data());
        if (basis.empty()) return false;
    }

    // 2. Intersect with each subsequent branch S_b (commutation nullspace FIRST, then +1 sign kernel).
    for (size_t b = 1U; b < branches.size(); ++b) {
        const uint32_t st_id = branches[b].state_id;
        generators(store.state_pool[st_id].state, gx, gz, gk);
        for (size_t r = 0; r < n; ++r) {
            intersect_commuting(&gx[r * nw], &gz[r * nw]);
            if (basis.empty()) return false;
        }
        // Now every g in basis commutes with S_b, so definite(S_b, g) in {+1, -1}.
        size_t neg_piv = basis.size();
        std::vector<int8_t> evs(basis.size(), 0);
        for (size_t i = 0; i < basis.size(); ++i) {
            evs[i] = store.cached_definite(st_id, basis[i].x.data(), basis[i].z.data(), basis[i].kappa);
            if (evs[i] < 0 && neg_piv == basis.size()) {
                neg_piv = i;
            }
        }
        if (neg_piv < basis.size()) {
            for (size_t i = neg_piv + 1U; i < basis.size(); ++i) {
                if (evs[i] < 0) {
                    pauli_mul_inplace(
                        basis[i].x.data(), basis[i].z.data(), basis[i].kappa,
                        basis[neg_piv].x.data(), basis[neg_piv].z.data(), basis[neg_piv].kappa, nw
                    );
                }
            }
            basis.erase(basis.begin() + static_cast<std::ptrdiff_t>(neg_piv));
            if (basis.empty()) return false;
        }
    }

    // 3. Pick first surviving common stabilizer anticommuting with M_bar.
    for (const MbrPauli& g : basis) {
        if (pauli_anti(g.x.data(), g.z.data(), mx, mz, nw) != 0U) {
            out_qx = g.x;
            out_qz = g.z;
            return true;
        }
    }
    return false;
}

inline double branches_norm2(MbrStore& store, const std::vector<MbrBranch>& lst) {
    double tot = 0.0;
    for (const MbrBranch& b : lst) {
        tot += cabs2(b.w);
    }
    const size_t k = lst.size();
    for (size_t i = 0; i < k; ++i) {
        const MbrComplex wi_conj = cconj(lst[i].w);
        for (size_t j = i + 1U; j < k; ++j) {
            const MbrComplex ov = store.cached_overlap(lst[i].state_id, lst[j].state_id);
            if (ov.re != 0.0 || ov.im != 0.0) {
                const MbrComplex prod = cmul(cmul(wi_conj, lst[j].w), ov);
                tot += 2.0 * prod.re;
            }
        }
    }
    return tot;
}

inline int32_t store_meas_or_peek(
    MbrStore& store,
    int64_t node_id,
    uint64_t event_id,
    uint8_t kind,
    bool is_peek,
    const uint64_t* mx, const uint64_t* mz, uint8_t mkappa,
    double& out_p0,
    int64_t* out_child0, int64_t* out_child1,
    uint8_t* out_meas_type,
    uint64_t* out_absorb_x, uint64_t* out_absorb_z
) {
    const size_t nw = store.nw;
    if (node_id < 0 || static_cast<size_t>(node_id) >= store.nodes.size()) {
        return kErrInvalidArg;
    }
    if (mx == nullptr || mz == nullptr) return kErrInvalidArg;
    if (!tail_clear(mx, store.n_qubits, nw) || !tail_clear(mz, store.n_qubits, nw)) {
        return kErrInvalidArg;
    }
    const uint8_t mk = static_cast<uint8_t>(mkappa & 3U);
    if (!is_hermitian(mx, mz, mk, nw)) return kErrInvalidArg;

    if (is_peek) {
        PeekKey pkey{node_id, event_id};
        auto pit = store.peek.find(pkey);
        if (pit != store.peek.end()) {
            store.transitions_hit[kKindPeek] += 1U;
            out_p0 = pit->second;
            return kOk;
        }
    } else {
        MeasKey mkey{node_id, event_id, kind};
        auto mit = store.meas.find(mkey);
        if (mit != store.meas.end()) {
            store.transitions_hit[kind] += 1U;
            const MeasEntry& me = mit->second;
            out_p0 = me.p0;
            *out_child0 = me.child0;
            *out_child1 = me.child1;
            *out_meas_type = me.meas_type;
            if (me.meas_type == kMeasAbsorbed) {
                std::memcpy(out_absorb_x, me.absorb_x.data(), nw * sizeof(uint64_t));
                std::memcpy(out_absorb_z, me.absorb_z.data(), nw * sizeof(uint64_t));
            } else {
                std::memset(out_absorb_x, 0, nw * sizeof(uint64_t));
                std::memset(out_absorb_z, 0, nw * sizeof(uint64_t));
            }
            return kOk;
        }
    }

    // Step 1: Resolve if M_bar anticommutes with any pending turn.
    std::vector<MbrBranch> working_branches;
    std::vector<MbrPending> working_pend;
    bool did_resolve = false;
    uint32_t res_peak = 0U;
    {
        const MbrNode& nd = store.nodes[static_cast<size_t>(node_id)];
        bool anti_pend = false;
        for (const MbrPending& it : nd.pend) {
            if (pauli_anti(it.x.data(), it.z.data(), mx, mz, nw) != 0U) {
                anti_pend = true;
                break;
            }
        }
        if (anti_pend) {
            std::vector<MbrBranch> nb = nd.branches;
            std::vector<MbrPending> np = nd.pend;
            const int32_t rrc = resolve_branches(store, nb, np, working_branches, res_peak);
            if (rrc != kOk) return rrc;
            did_resolve = true;
        } else {
            std::vector<MbrBranch>(nd.branches).swap(working_branches);
            std::vector<MbrPending>(nd.pend).swap(working_pend);
            res_peak = static_cast<uint32_t>(working_branches.size());
        }
    }
    if (working_branches.empty()) return kErrFloatDrift;

    // Step 2: Definite check across all branches.
    const size_t K = working_branches.size();
    std::vector<int8_t> evs(K, 0);
    bool all_pos = true;
    bool all_neg = true;
    for (size_t b = 0; b < K; ++b) {
        evs[b] = store.cached_definite(working_branches[b].state_id, mx, mz, mk);
        if (evs[b] != 1) all_pos = false;
        if (evs[b] != -1) all_neg = false;
    }
    if (all_pos || all_neg) {
        const double p0 = all_pos ? 1.0 : 0.0;
        if (is_peek) {
            store.peek.emplace(PeekKey{node_id, event_id}, p0);
            store.bytes_estimate += 64U;
            store.transitions_new[kKindPeek] += 1U;
            out_p0 = p0;
            return kOk;
        }
        const int64_t target_nid = did_resolve
            ? store.push_node(std::move(working_branches), {}, res_peak)
            : node_id;
        MeasEntry me;
        me.p0 = p0;
        me.child0 = all_pos ? target_nid : -1;
        me.child1 = all_pos ? -1 : target_nid;
        me.meas_type = kMeasDefinite;
        store.meas_definite += 1U;
        store.transitions_new[kind] += 1U;
        store.meas.emplace(MeasKey{node_id, event_id, kind}, me);
        store.bytes_estimate += 64U + sizeof(MeasEntry);
        out_p0 = me.p0;
        *out_child0 = me.child0;
        *out_child1 = me.child1;
        *out_meas_type = kMeasDefinite;
        std::memset(out_absorb_x, 0, nw * sizeof(uint64_t));
        std::memset(out_absorb_z, 0, nw * sizeof(uint64_t));
        return kOk;
    }

    // Step 3: Multi-branch GF(2) absorption check.
    std::vector<uint64_t> qx, qz;
    if (find_absorb_pauli(store, working_branches, working_pend, mx, mz, qx, qz)) {
        if (is_peek) {
            store.peek.emplace(PeekKey{node_id, event_id}, 0.5);
            store.bytes_estimate += 64U;
            store.transitions_new[kKindPeek] += 1U;
            out_p0 = 0.5;
            return kOk;
        }
        std::vector<MbrBranch> raw0;
        raw0.reserve(K);
        for (size_t b = 0; b < K; ++b) {
            if (evs[b] == 0) {
                const ProjEntry pr = store.cached_project(
                    working_branches[b].state_id, mx, mz, mk, 0U
                );
                if (pr.valid) {
                    raw0.push_back({cscale(cmul(working_branches[b].w, pr.f), kSqrt2), pr.state_id});
                }
            } else if (evs[b] == 1) {
                raw0.push_back({cscale(working_branches[b].w, kSqrt2), working_branches[b].state_id});
            }
        }
        std::vector<MbrBranch> b0;
        const int32_t mrc = merge_and_sort_branches(store, raw0, b0);
        if (mrc != kOk) return mrc;
        if (b0.empty()) return kErrFloatDrift;
        if (store.max_components > 0U && b0.size() > store.max_components) {
            return kErrBudget;
        }
        const uint32_t peak0 = std::max(res_peak, static_cast<uint32_t>(b0.size()));
        const int64_t c0 = store.push_node(std::move(b0), std::move(working_pend), peak0);
        MeasEntry me;
        me.p0 = 0.5;
        me.child0 = c0;
        me.child1 = c0;
        me.meas_type = kMeasAbsorbed;
        me.absorb_x = qx;
        me.absorb_z = qz;
        store.meas_absorbed += 1U;
        store.transitions_new[kind] += 1U;
        const uint64_t add_bytes = 64U + sizeof(MeasEntry) + 2U * nw * sizeof(uint64_t);
        store.meas.emplace(MeasKey{node_id, event_id, kind}, std::move(me));
        store.bytes_estimate += add_bytes;
        out_p0 = 0.5;
        *out_child0 = c0;
        *out_child1 = c0;
        *out_meas_type = kMeasAbsorbed;
        std::memcpy(out_absorb_x, qx.data(), nw * sizeof(uint64_t));
        std::memcpy(out_absorb_z, qz.data(), nw * sizeof(uint64_t));
        return kOk;
    }

    // Step 4: Forked measurement (project m = 0, 1; overlap norm2; drift & impossible guards).
    std::vector<MbrBranch> b0, b1;
    for (uint8_t m = 0U; m < 2U; ++m) {
        std::vector<MbrBranch> raw_m;
        raw_m.reserve(K);
        for (size_t b = 0; b < K; ++b) {
            if (evs[b] == 0) {
                const ProjEntry pr = store.cached_project(
                    working_branches[b].state_id, mx, mz, mk, m
                );
                if (pr.valid) {
                    raw_m.push_back({cmul(working_branches[b].w, pr.f), pr.state_id});
                }
            } else if ((evs[b] == 1) == (m == 0U)) {
                raw_m.push_back(working_branches[b]);
            }
        }
        const int32_t mrc = merge_and_sort_branches(store, raw_m, (m == 0U) ? b0 : b1);
        if (mrc != kOk) return mrc;
    }

    const double p0 = branches_norm2(store, b0);
    const double p1 = branches_norm2(store, b1);
    if (!std::isfinite(p0) || !std::isfinite(p1) || std::fabs(p0 + p1 - 1.0) > kDriftTol) {
        return kErrFloatDrift;
    }
    if (p0 <= kImpossibleProb && p1 <= kImpossibleProb) {
        return kErrFloatDrift;
    }

    const bool has0 = (p0 > kImpossibleProb);
    const bool has1 = (p1 > kImpossibleProb);
    const double p0_out = !has0 ? 0.0 : (!has1 ? 1.0 : std::clamp(p0, 0.0, 1.0));

    if (is_peek) {
        store.peek.emplace(PeekKey{node_id, event_id}, p0_out);
        store.bytes_estimate += 64U;
        store.transitions_new[kKindPeek] += 1U;
        out_p0 = p0_out;
        return kOk;
    }

    if ((has0 && b0.empty()) || (has1 && b1.empty())) return kErrFloatDrift;
    if (store.max_components > 0U) {
        if ((has0 && b0.size() > store.max_components)
            || (has1 && b1.size() > store.max_components)) {
            return kErrBudget;
        }
    }

    if (has0) {
        const double inv0 = 1.0 / std::sqrt(p0);
        for (MbrBranch& br : b0) br.w = cscale(br.w, inv0);
    }
    if (has1) {
        const double inv1 = 1.0 / std::sqrt(p1);
        for (MbrBranch& br : b1) br.w = cscale(br.w, inv1);
    }

    const uint32_t peak0 = std::max(res_peak, static_cast<uint32_t>(b0.size()));
    const uint32_t peak1 = std::max(res_peak, static_cast<uint32_t>(b1.size()));
    std::vector<MbrPending> pend0;
    if (has0 && has1) {
        pend0 = working_pend;
    }
    const int64_t c0 = has0
        ? store.push_node(
              std::move(b0),
              has1 ? std::move(pend0) : std::move(working_pend),
              peak0
          )
        : -1;
    const int64_t c1 = has1
        ? store.push_node(
              std::move(b1),
              std::move(working_pend),
              peak1
          )
        : -1;

    MeasEntry me;
    me.p0 = p0_out;
    me.child0 = c0;
    me.child1 = c1;
    me.meas_type = kMeasForked;
    store.meas_forked += 1U;
    store.transitions_new[kind] += 1U;
    store.meas.emplace(MeasKey{node_id, event_id, kind}, me);
    store.bytes_estimate += 64U + sizeof(MeasEntry);
    out_p0 = p0_out;
    *out_child0 = c0;
    *out_child1 = c1;
    *out_meas_type = kMeasForked;
    std::memset(out_absorb_x, 0, nw * sizeof(uint64_t));
    std::memset(out_absorb_z, 0, nw * sizeof(uint64_t));
    return kOk;
}

inline int32_t normalize_commuting_turns(
    size_t n_qubits, size_t nw,
    size_t n_axes,
    const uint64_t* ax_x, const uint64_t* ax_z,
    const uint8_t* ax_kappa, const uint8_t* ax_e,
    std::vector<MbrPending>& out_turns,
    uint8_t& out_ph
) {
    out_turns.clear();
    out_ph = 0U;
    if (n_axes == 0U) return kOk;
    if (ax_x == nullptr || ax_z == nullptr || ax_kappa == nullptr || ax_e == nullptr) {
        return kErrInvalidArg;
    }
    size_t total_words = 0U;
    if (!mul_within(n_axes, nw, kMaxElems, total_words)) {
        return kErrInvalidArg;
    }
    out_turns.reserve(n_axes);
    for (size_t i = 0; i < n_axes; ++i) {
        const uint64_t* xi = &ax_x[i * nw];
        const uint64_t* zi = &ax_z[i * nw];
        uint8_t ki = static_cast<uint8_t>(ax_kappa[i] & 3U);
        if (!tail_clear(xi, n_qubits, nw) || !tail_clear(zi, n_qubits, nw)) {
            return kErrInvalidArg;
        }
        if (!is_hermitian(xi, zi, ki, nw)) return kErrInvalidArg;
        uint8_t e = static_cast<uint8_t>(ax_e[i] & 7U);
        if (e == 0U) continue;
        if (pauli_norm_sign_plus(xi, zi, ki, nw)) {
            out_ph = static_cast<uint8_t>((out_ph + e) & 7U);
            e = static_cast<uint8_t>((8U - e) & 7U);
        }
        if (pauli_is_identity(xi, zi, nw)) continue;
        MbrPending t;
        t.x.assign(xi, xi + nw);
        t.z.assign(zi, zi + nw);
        t.kappa = ki;
        t.e = e;
        out_turns.push_back(std::move(t));
    }
    for (size_t i = 0; i < out_turns.size(); ++i) {
        for (size_t j = i + 1U; j < out_turns.size(); ++j) {
            if (pauli_anti(
                    out_turns[i].x.data(), out_turns[i].z.data(),
                    out_turns[j].x.data(), out_turns[j].z.data(), nw
                ) != 0U) {
                return kErrInvalidArg;
            }
        }
    }
    return kOk;
}

inline void accumulate_pending_inplace(
    std::vector<MbrPending>& pend,
    const std::vector<MbrPending>& incoming,
    size_t nw
) {
    for (const MbrPending& t : incoming) {
        bool merged = false;
        for (MbrPending& p : pend) {
            if (pauli_xz_equal(p.x.data(), p.z.data(), t.x.data(), t.z.data(), nw)) {
                p.e = static_cast<uint8_t>((p.e + t.e) & 7U);
                merged = true;
                break;
            }
        }
        if (!merged) {
            pend.push_back(t);
        }
    }
    pend.erase(
        std::remove_if(pend.begin(), pend.end(), [](const MbrPending& p) { return p.e == 0U; }),
        pend.end()
    );
}

inline uint32_t bitrev_u32(uint32_t x, size_t n) noexcept {
    uint32_t r = 0U;
    for (size_t b = 0; b < n; ++b) {
        r = (r << 1U) | (x & 1U);
        x >>= 1U;
    }
    return r;
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

// ---------------------------------------------------------------------------
// WP3 C ABI: is_clifford, choose_split, MbrStore trie, transitions & counters
// ---------------------------------------------------------------------------

int32_t mbr_is_clifford(
    const uint64_t* parities, size_t n_parities, size_t n_words_per_parity,
    int32_t* out_is_clifford
) {
    try {
        if (out_is_clifford == nullptr || n_words_per_parity == 0U) {
            return mbr::kErrInvalidArg;
        }
        if (n_parities == 0U) {
            *out_is_clifford = 1;
            return mbr::kOk;
        }
        if (parities == nullptr) return mbr::kErrInvalidArg;
        size_t total_words = 0U;
        if (!mul_within(n_parities, n_words_per_parity, kMaxElems, total_words)) {
            return mbr::kErrInvalidArg;
        }
        std::vector<mbr::ParityVec> O;
        O.reserve(n_parities);
        for (size_t i = 0; i < n_parities; ++i) {
            mbr::ParityVec p = mbr::parity_from_words(
                &parities[i * n_words_per_parity], n_words_per_parity
            );
            if (p.is_zero()) return mbr::kErrInvalidArg;
            O.push_back(std::move(p));
        }
        std::sort(O.begin(), O.end());
        O.erase(std::unique(O.begin(), O.end()), O.end());
        *out_is_clifford = mbr::is_clifford_sorted(O) ? 1 : 0;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_choose_split(
    const uint64_t* parities, size_t n_parities, size_t n_words_per_parity,
    uint32_t max_depth, uint64_t* out_lam_words, int32_t* out_depth
) {
    try {
        if (parities == nullptr || n_parities == 0U || n_words_per_parity == 0U
            || out_lam_words == nullptr || out_depth == nullptr) {
            return mbr::kErrInvalidArg;
        }
        size_t total_words = 0U;
        if (!mul_within(n_parities, n_words_per_parity, kMaxElems, total_words)) {
            return mbr::kErrInvalidArg;
        }
        std::vector<mbr::ParityVec> O;
        O.reserve(n_parities);
        for (size_t i = 0; i < n_parities; ++i) {
            mbr::ParityVec p = mbr::parity_from_words(
                &parities[i * n_words_per_parity], n_words_per_parity
            );
            if (p.is_zero()) return mbr::kErrInvalidArg;
            O.push_back(std::move(p));
        }
        std::sort(O.begin(), O.end());
        O.erase(std::unique(O.begin(), O.end()), O.end());
        if (mbr::is_clifford_sorted(O)) {
            std::memset(out_lam_words, 0, n_words_per_parity * sizeof(uint64_t));
            *out_depth = 0;
            return mbr::kOk;
        }
        const mbr::SplitEntry sp = mbr::choose_split_sorted(
            O, max_depth, nullptr, nullptr, nullptr, nullptr
        );
        for (size_t w = 0; w < n_words_per_parity; ++w) {
            out_lam_words[w] = sp.lam.word(w);
        }
        *out_depth = sp.depth;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

size_t mbr_live_stores(void) {
    return mbr::g_live_stores.load(std::memory_order_relaxed);
}

int32_t mbr_store_create(size_t n_qubits, size_t max_components, void** out_store) {
    try {
        if (out_store == nullptr || n_qubits == 0U || n_qubits > mbr::kMaxQubits) {
            return mbr::kErrInvalidArg;
        }
        const size_t nw = mbr::n_words_for_qubits(n_qubits);
        if (nw > kMaxElems) return mbr::kErrInvalidArg;
        mbr::MbrStore* st = new mbr::MbrStore();
        st->n_qubits = n_qubits;
        st->nw = nw;
        st->max_components = max_components;
        try {
            st->seed_root();
        } catch (...) {
            delete st;
            throw;
        }
        mbr::g_live_stores.fetch_add(1U, std::memory_order_relaxed);
        *out_store = st;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

void mbr_store_destroy(void* store) {
    if (store != nullptr) {
        delete mbr::store_of(store);
        mbr::g_live_stores.fetch_sub(1U, std::memory_order_relaxed);
    }
}

int32_t mbr_store_clear(void* store) {
    try {
        if (store == nullptr) return mbr::kErrInvalidArg;
        mbr::MbrStore* st = mbr::store_of(store);
        st->seed_root();
        st->store_clears += 1U;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_set_max_components(void* store, size_t max_components) {
    try {
        if (store == nullptr) return mbr::kErrInvalidArg;
        mbr::store_of(store)->max_components = max_components;
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_count(const void* store, size_t* out_count) {
    try {
        if (store == nullptr || out_count == nullptr) return mbr::kErrInvalidArg;
        *out_count = mbr::store_of(store)->nodes.size();
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_info(
    const void* store, int64_t node_id,
    size_t* out_n_branches, size_t* out_n_pend, uint32_t* out_transient_peak
) {
    try {
        if (store == nullptr || out_n_branches == nullptr || out_n_pend == nullptr
            || out_transient_peak == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrNode& nd = st->nodes[static_cast<size_t>(node_id)];
        *out_n_branches = nd.branches.size();
        *out_n_pend = nd.pend.size();
        *out_transient_peak = nd.transient_peak;
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_branch(
    const void* store, int64_t node_id, size_t branch_idx,
    double* out_w_ri, void** out_state_handle
) {
    try {
        if (store == nullptr || out_w_ri == nullptr || out_state_handle == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrNode& nd = st->nodes[static_cast<size_t>(node_id)];
        if (branch_idx >= nd.branches.size()) return mbr::kErrInvalidArg;
        const mbr::MbrBranch& br = nd.branches[branch_idx];
        mbr::MbrState* copy = new mbr::MbrState(st->state_pool[br.state_id].state);
        mbr::g_live_states.fetch_add(1U, std::memory_order_relaxed);
        out_w_ri[0] = br.w.re;
        out_w_ri[1] = br.w.im;
        *out_state_handle = copy;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_pending(
    const void* store, int64_t node_id,
    uint64_t* out_x, uint64_t* out_z, uint8_t* out_kappa, uint8_t* out_e
) {
    try {
        if (store == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrNode& nd = st->nodes[static_cast<size_t>(node_id)];
        const size_t np = nd.pend.size();
        if (np == 0U) return mbr::kOk;
        if (out_x == nullptr || out_z == nullptr || out_kappa == nullptr || out_e == nullptr) {
            return mbr::kErrInvalidArg;
        }
        const size_t nw = st->nw;
        for (size_t i = 0; i < np; ++i) {
            std::memcpy(&out_x[i * nw], nd.pend[i].x.data(), nw * sizeof(uint64_t));
            std::memcpy(&out_z[i * nw], nd.pend[i].z.data(), nw * sizeof(uint64_t));
            out_kappa[i] = nd.pend[i].kappa;
            out_e[i] = nd.pend[i].e;
        }
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_dense(
    const void* store, int64_t node_id, int32_t big_endian,
    double* out_psi_ri, size_t out_complex_len
) {
    try {
        if (store == nullptr || out_psi_ri == nullptr
            || (big_endian != 0 && big_endian != 1)) {
            return mbr::kErrInvalidArg;
        }
        const mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const size_t n = st->n_qubits;
        if (n > mbr::kMaxDenseQubits) return mbr::kErrInvalidArg;
        const size_t dim = size_t(1) << n;
        if (out_complex_len != dim) return mbr::kErrInvalidArg;

        const mbr::MbrNode& nd = st->nodes[static_cast<size_t>(node_id)];
        std::memset(out_psi_ri, 0, dim * 2U * sizeof(double));
        std::vector<double> tmp(dim * 2U, 0.0);
        for (const mbr::MbrBranch& br : nd.branches) {
            if (!mbr::to_dense(st->state_pool[br.state_id].state, big_endian != 0, tmp.data())) {
                return mbr::kErrInvalidArg;
            }
            for (size_t x = 0; x < dim; ++x) {
                const mbr::MbrComplex c = mbr::cmul(br.w, {tmp[2U * x], tmp[2U * x + 1U]});
                out_psi_ri[2U * x] += c.re;
                out_psi_ri[2U * x + 1U] += c.im;
            }
        }

        // Apply pending diagonal/Pauli eighth turns D = prod omega^{e(1-P)/2}.
        for (const mbr::MbrPending& p : nd.pend) {
            uint32_t px = static_cast<uint32_t>(p.x[0]);
            uint32_t pz = static_cast<uint32_t>(p.z[0]);
            if (big_endian != 0) {
                px = mbr::bitrev_u32(px, n);
                pz = mbr::bitrev_u32(pz, n);
            }
            const mbr::MbrComplex we = mbr::omega_pow(p.e);
            if (px == 0U) {
                const uint8_t sgn_base = static_cast<uint8_t>((p.kappa >> 1U) & 1U);
                for (uint32_t y = 0; y < static_cast<uint32_t>(dim); ++y) {
                    const uint8_t par = static_cast<uint8_t>(
                        (sgn_base ^ static_cast<uint8_t>(__builtin_popcount(pz & y) & 1)) & 1U
                    );
                    if (par != 0U) {
                        const mbr::MbrComplex v{out_psi_ri[2U * y], out_psi_ri[2U * y + 1U]};
                        const mbr::MbrComplex nv = mbr::cmul(we, v);
                        out_psi_ri[2U * y] = nv.re;
                        out_psi_ri[2U * y + 1U] = nv.im;
                    }
                }
            } else {
                const mbr::MbrComplex a = mbr::cscale(mbr::cadd({1.0, 0.0}, we), 0.5);
                const mbr::MbrComplex b = mbr::cscale(mbr::csub({1.0, 0.0}, we), 0.5);
                for (uint32_t y = 0; y < static_cast<uint32_t>(dim); ++y) {
                    const uint32_t y2 = y ^ px;
                    if (y >= y2) continue;
                    const mbr::MbrComplex c0{out_psi_ri[2U * y], out_psi_ri[2U * y + 1U]};
                    const mbr::MbrComplex c1{out_psi_ri[2U * y2], out_psi_ri[2U * y2 + 1U]};
                    const uint8_t p0 = static_cast<uint8_t>(
                        (p.kappa + (static_cast<uint8_t>(__builtin_popcount(pz & y) & 1) << 1U)) & 3U
                    );
                    const uint8_t p1 = static_cast<uint8_t>(
                        (p.kappa + (static_cast<uint8_t>(__builtin_popcount(pz & y2) & 1) << 1U)) & 3U
                    );
                    const mbr::MbrComplex s0 = mbr::i_pow(p0);
                    const mbr::MbrComplex s1 = mbr::i_pow(p1);
                    const mbr::MbrComplex n0 = mbr::cadd(
                        mbr::cmul(a, c0), mbr::cmul(b, mbr::cmul(s1, c1))
                    );
                    const mbr::MbrComplex n1 = mbr::cadd(
                        mbr::cmul(a, c1), mbr::cmul(b, mbr::cmul(s0, c0))
                    );
                    out_psi_ri[2U * y] = n0.re;
                    out_psi_ri[2U * y + 1U] = n0.im;
                    out_psi_ri[2U * y2] = n1.re;
                    out_psi_ri[2U * y2 + 1U] = n1.im;
                }
            }
        }
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_node_from_branches(
    void* store,
    size_t n_branches, const double* w_ri, const void* const* state_handles,
    size_t n_pend, const uint64_t* pend_x, const uint64_t* pend_z,
    const uint8_t* pend_kappa, const uint8_t* pend_e,
    int64_t* out_node_id
) {
    try {
        if (store == nullptr || out_node_id == nullptr || n_branches == 0U
            || w_ri == nullptr || state_handles == nullptr) {
            return mbr::kErrInvalidArg;
        }
        mbr::MbrStore* st = mbr::store_of(store);
        const size_t n = st->n_qubits;
        const size_t nw = st->nw;
        size_t total_w = 0U;
        if (!mul_within(n_branches, size_t(2), kMaxElems, total_w)) {
            return mbr::kErrInvalidArg;
        }

        std::vector<mbr::MbrPending> norm_pend;
        uint8_t ph = 0U;
        const int32_t prc = mbr::normalize_commuting_turns(
            n, nw, n_pend, pend_x, pend_z, pend_kappa, pend_e, norm_pend, ph
        );
        if (prc != mbr::kOk) return prc;

        std::vector<mbr::MbrPending> pend;
        mbr::accumulate_pending_inplace(pend, norm_pend, nw);

        for (size_t i = 0; i < n_branches; ++i) {
            if (state_handles[i] == nullptr) return mbr::kErrInvalidArg;
            const mbr::MbrState* s = mbr::state_of(state_handles[i]);
            if (s->n_qubits != n) return mbr::kErrInvalidArg;
            if (!mbr::cfinite({w_ri[2U * i], w_ri[2U * i + 1U]})) return mbr::kErrInvalidArg;
        }

        const mbr::MbrComplex ph_c = mbr::omega_pow(ph);
        std::vector<mbr::MbrBranch> raw;
        raw.reserve(n_branches);
        for (size_t i = 0; i < n_branches; ++i) {
            mbr::MbrState sc = *mbr::state_of(state_handles[i]);
            const mbr::MbrComplex fc = mbr::canon(sc);
            const mbr::MbrComplex w = mbr::cmul(
                mbr::cmul({w_ri[2U * i], w_ri[2U * i + 1U]}, fc), ph_c
            );
            const uint32_t sid = st->intern_canonical(std::move(sc));
            raw.push_back({w, sid});
        }

        std::vector<mbr::MbrBranch> merged;
        const int32_t mrc = mbr::merge_and_sort_branches(*st, raw, merged);
        if (mrc != mbr::kOk) return mrc;
        if (merged.empty()) return mbr::kErrInvalidArg;
        if (st->max_components > 0U && merged.size() > st->max_components) {
            return mbr::kErrBudget;
        }
        const uint32_t peak = static_cast<uint32_t>(merged.size());
        *out_node_id = st->push_node(std::move(merged), std::move(pend), peak);
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_resolve_node(void* store, int64_t node_id, int64_t* out_node_id) {
    try {
        if (store == nullptr || out_node_id == nullptr) return mbr::kErrInvalidArg;
        mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        if (st->nodes[static_cast<size_t>(node_id)].pend.empty()) {
            *out_node_id = node_id;
            return mbr::kOk;
        }
        std::vector<mbr::MbrBranch> in_b = st->nodes[static_cast<size_t>(node_id)].branches;
        std::vector<mbr::MbrPending> in_p = st->nodes[static_cast<size_t>(node_id)].pend;
        std::vector<mbr::MbrBranch> resolved;
        uint32_t peak = 0U;
        const int32_t rc = mbr::resolve_branches(*st, in_b, in_p, resolved, peak);
        if (rc != mbr::kOk) return rc;
        *out_node_id = st->push_node(std::move(resolved), {}, peak);
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_step_rotations(
    void* store, int64_t node_id, uint64_t event_id, uint8_t kind, uint32_t sign_bits,
    size_t n_axes, const uint64_t* ax_x, const uint64_t* ax_z,
    const uint8_t* ax_kappa, const uint8_t* ax_e,
    int64_t* out_child
) {
    try {
        if (store == nullptr || out_child == nullptr) return mbr::kErrInvalidArg;
        if (kind != mbr::kKindT && kind != mbr::kKindCcz) return mbr::kErrInvalidArg;
        mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const mbr::TransKey tkey{node_id, event_id, sign_bits, kind};
        auto tit = st->trans.find(tkey);
        if (tit != st->trans.end()) {
            st->transitions_hit[kind] += 1U;
            *out_child = tit->second;
            return mbr::kOk;
        }

        const size_t nw = st->nw;
        std::vector<mbr::MbrPending> incoming;
        uint8_t ph = 0U;
        const int32_t nrc = mbr::normalize_commuting_turns(
            st->n_qubits, nw, n_axes, ax_x, ax_z, ax_kappa, ax_e, incoming, ph
        );
        if (nrc != mbr::kOk) return nrc;

        std::vector<mbr::MbrBranch> branches = st->nodes[static_cast<size_t>(node_id)].branches;
        std::vector<mbr::MbrPending> pend = st->nodes[static_cast<size_t>(node_id)].pend;
        uint32_t peak = static_cast<uint32_t>(branches.size());

        if ((ph & 7U) != 0U) {
            const mbr::MbrComplex sc = mbr::omega_pow(ph);
            for (mbr::MbrBranch& br : branches) br.w = mbr::cmul(br.w, sc);
        }

        bool need_resolve = false;
        if (!pend.empty() && !incoming.empty()) {
            for (const mbr::MbrPending& inc : incoming) {
                for (const mbr::MbrPending& cur : pend) {
                    if (mbr::pauli_anti(inc.x.data(), inc.z.data(), cur.x.data(), cur.z.data(), nw) != 0U) {
                        need_resolve = true;
                        break;
                    }
                }
                if (need_resolve) break;
            }
        }
        if (need_resolve) {
            std::vector<mbr::MbrBranch> resolved;
            uint32_t rpeak = 0U;
            const int32_t rrc = mbr::resolve_branches(*st, branches, pend, resolved, rpeak);
            if (rrc != mbr::kOk) return rrc;
            branches = std::move(resolved);
            pend.clear();
            if (rpeak > peak) peak = rpeak;
        }

        mbr::accumulate_pending_inplace(pend, incoming, nw);

        const int64_t cid = st->push_node(std::move(branches), std::move(pend), peak);
        st->trans.emplace(tkey, cid);
        st->bytes_estimate += 64U;
        st->transitions_new[kind] += 1U;
        *out_child = cid;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_step_qturns(
    void* store, int64_t node_id, uint64_t event_id, uint8_t kind, uint32_t sign_bits,
    size_t n_qturns, const uint64_t* qt_x, const uint64_t* qt_z, const uint8_t* qt_kappa,
    int64_t* out_child
) {
    try {
        if (store == nullptr || out_child == nullptr) return mbr::kErrInvalidArg;
        if (kind != mbr::kKindSkip && kind != mbr::kKindCfault) return mbr::kErrInvalidArg;
        mbr::MbrStore* st = mbr::store_of(store);
        if (node_id < 0 || static_cast<size_t>(node_id) >= st->nodes.size()) {
            return mbr::kErrInvalidArg;
        }
        const mbr::TransKey tkey{node_id, event_id, sign_bits, kind};
        auto tit = st->trans.find(tkey);
        if (tit != st->trans.end()) {
            st->transitions_hit[kind] += 1U;
            *out_child = tit->second;
            return mbr::kOk;
        }

        const size_t n = st->n_qubits;
        const size_t nw = st->nw;
        if (n_qturns > 0U && (qt_x == nullptr || qt_z == nullptr || qt_kappa == nullptr)) {
            return mbr::kErrInvalidArg;
        }
        size_t total_qwords = 0U;
        if (!mul_within(n_qturns, nw, kMaxElems, total_qwords)) {
            return mbr::kErrInvalidArg;
        }
        for (size_t q = 0; q < n_qturns; ++q) {
            const uint64_t* qx = &qt_x[q * nw];
            const uint64_t* qz = &qt_z[q * nw];
            const uint8_t qk = static_cast<uint8_t>(qt_kappa[q] & 3U);
            if (!mbr::tail_clear(qx, n, nw) || !mbr::tail_clear(qz, n, nw)) {
                return mbr::kErrInvalidArg;
            }
            if (!mbr::is_hermitian(qx, qz, qk, nw)) return mbr::kErrInvalidArg;
        }

        std::vector<mbr::MbrBranch> branches = st->nodes[static_cast<size_t>(node_id)].branches;
        std::vector<mbr::MbrPending> pend = st->nodes[static_cast<size_t>(node_id)].pend;
        const uint32_t peak = static_cast<uint32_t>(branches.size());

        std::vector<mbr::MbrBranch> updated;
        updated.reserve(branches.size());
        for (const mbr::MbrBranch& br : branches) {
            mbr::MbrState qc = st->state_pool[br.state_id].state;
            mbr::MbrComplex f = br.w;
            for (size_t q = 0; q < n_qturns; ++q) {
                const mbr::MbrComplex fq = mbr::apply_qturn(
                    qc, &qt_x[q * nw], &qt_z[q * nw], static_cast<uint8_t>(qt_kappa[q] & 3U)
                );
                f = mbr::cmul(f, fq);
            }
            f = mbr::cmul(f, mbr::canon(qc));
            const uint32_t cid_st = st->intern_canonical(std::move(qc));
            updated.push_back({f, cid_st});
        }

        if (!pend.empty() && n_qturns > 0U) {
            uint8_t ph = 0U;
            for (mbr::MbrPending& p : pend) {
                for (size_t q = 0; q < n_qturns; ++q) {
                    mbr::pauli_conj_by_qturn_inplace(
                        p.x.data(), p.z.data(), p.kappa,
                        &qt_x[q * nw], &qt_z[q * nw], static_cast<uint8_t>(qt_kappa[q] & 3U), nw
                    );
                }
                if (mbr::pauli_norm_sign_plus(p.x.data(), p.z.data(), p.kappa, nw)) {
                    ph = static_cast<uint8_t>((ph + p.e) & 7U);
                    p.e = static_cast<uint8_t>((8U - p.e) & 7U);
                }
            }
            if ((ph & 7U) != 0U) {
                const mbr::MbrComplex sc = mbr::omega_pow(ph);
                for (mbr::MbrBranch& br : updated) br.w = mbr::cmul(br.w, sc);
            }
        }

        std::vector<mbr::MbrBranch> merged;
        const int32_t mrc = mbr::merge_and_sort_branches(*st, updated, merged);
        if (mrc != mbr::kOk) return mrc;
        if (!branches.empty() && merged.empty()) return mbr::kErrFloatDrift;

        const int64_t cid = st->push_node(std::move(merged), std::move(pend), peak);
        st->trans.emplace(tkey, cid);
        st->bytes_estimate += 64U;
        st->transitions_new[kind] += 1U;
        *out_child = cid;
        return mbr::kOk;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_step_meas(
    void* store, int64_t node_id, uint64_t event_id, uint8_t kind,
    const uint64_t* mx, const uint64_t* mz, uint8_t mkappa,
    double* out_p0, int64_t* out_child0, int64_t* out_child1,
    uint8_t* out_meas_type, uint64_t* out_absorb_x, uint64_t* out_absorb_z
) {
    try {
        if (store == nullptr || out_p0 == nullptr || out_child0 == nullptr
            || out_child1 == nullptr || out_meas_type == nullptr
            || out_absorb_x == nullptr || out_absorb_z == nullptr) {
            return mbr::kErrInvalidArg;
        }
        if (kind != mbr::kKindMeas && kind != mbr::kKindPost && kind != mbr::kKindUreset) {
            return mbr::kErrInvalidArg;
        }
        double p0 = 0.0;
        const int32_t rc = mbr::store_meas_or_peek(
            *mbr::store_of(store), node_id, event_id, kind, false,
            mx, mz, mkappa, p0, out_child0, out_child1, out_meas_type, out_absorb_x, out_absorb_z
        );
        if (rc == mbr::kOk) *out_p0 = p0;
        return rc;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_step_peek(
    void* store, int64_t node_id, uint64_t event_id,
    const uint64_t* mx, const uint64_t* mz, uint8_t mkappa,
    double* out_p0
) {
    try {
        if (store == nullptr || out_p0 == nullptr) return mbr::kErrInvalidArg;
        double p0 = 0.0;
        const int32_t rc = mbr::store_meas_or_peek(
            *mbr::store_of(store), node_id, event_id, mbr::kKindPeek, true,
            mx, mz, mkappa, p0, nullptr, nullptr, nullptr, nullptr, nullptr
        );
        if (rc == mbr::kOk) *out_p0 = p0;
        return rc;
    } catch (const std::bad_alloc&) {
        return mbr::kErrOom;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_counters(const void* store, uint64_t* out_counters_26) {
    try {
        if (store == nullptr || out_counters_26 == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrStore* st = mbr::store_of(store);
        out_counters_26[0] = static_cast<uint64_t>(st->nodes.size());
        out_counters_26[1] = st->bytes_estimate;
        out_counters_26[2] = st->store_clears;
        for (size_t k = 0; k < mbr::kNumKinds; ++k) {
            out_counters_26[3U + k] = st->transitions_new[k];
            out_counters_26[11U + k] = st->transitions_hit[k];
        }
        out_counters_26[19] = st->resolve_calls;
        out_counters_26[20] = st->splits;
        out_counters_26[21] = st->subcache_hits;
        out_counters_26[22] = st->subcache_misses;
        out_counters_26[23] = st->meas_definite;
        out_counters_26[24] = st->meas_absorbed;
        out_counters_26[25] = st->meas_forked;
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

int32_t mbr_store_branches_hist(
    const void* store, uint64_t* out_keys, uint64_t* out_counts, size_t cap, size_t* out_len
) {
    try {
        if (store == nullptr || out_len == nullptr) return mbr::kErrInvalidArg;
        const mbr::MbrStore* st = mbr::store_of(store);
        const size_t n_entries = st->node_branches_hist.size();
        *out_len = n_entries;
        if (cap == 0U) return mbr::kOk;
        if (out_keys == nullptr || out_counts == nullptr || cap < n_entries) {
            return mbr::kErrInvalidArg;
        }
        size_t idx = 0U;
        for (const auto& kv : st->node_branches_hist) {
            out_keys[idx] = kv.first;
            out_counts[idx] = kv.second;
            ++idx;
        }
        return mbr::kOk;
    } catch (...) {
        return mbr::kErrInvalidArg;
    }
}

}  // extern "C"
