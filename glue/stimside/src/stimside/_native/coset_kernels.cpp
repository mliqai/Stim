#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

extern "C" {

static inline int word_dot_parity(const uint64_t* u, const uint64_t* v, int n_words) {
    uint64_t acc = 0;
    for (int w = 0; w < n_words; ++w) {
        acc ^= (u[w] & v[w]);
    }
    return __builtin_popcountll(acc) & 1;
}

static inline bool words_any_overlap(const uint64_t* u, const uint64_t* v, int n_words) {
    for (int w = 0; w < n_words; ++w) {
        if (u[w] & v[w]) return true;
    }
    return false;
}

static inline bool words_any_nonzero(const uint64_t* u, int n_words) {
    for (int w = 0; w < n_words; ++w) {
        if (u[w]) return true;
    }
    return false;
}

static inline int first_set_bit_in_words(const uint64_t* col_words, int n_words) {
    for (int w = 0; w < n_words; ++w) {
        uint64_t val = col_words[w];
        if (val) {
            return __builtin_ctzll(val) + (w << 6);
        }
    }
    return -1;
}

static inline int splitmix64_draw_bit(uint64_t* rng_state) {
    uint64_t z = (*rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return int((z ^ (z >> 31)) & 1ULL);
}

constexpr size_t kCosetMaxElems = size_t(PTRDIFF_MAX) / 8;

static inline bool coset_mul_within(size_t a, size_t b, size_t limit, size_t& out) {
    if (a != 0 && b > limit / a) return false;
    out = a * b;
    return true;
}

static inline int compute_n_words(int nq) {
    int64_t nw = (int64_t(nq) + 63) / 64;
    return nw > 0 ? int(nw) : 1;
}

struct PauliProjInfo {
    bool a_is_zero;
    int sign_bool;
    int ref_exp;
    int kappa_rot;
};

static inline PauliProjInfo prepare_pauli_proj(
    const uint64_t* aw,
    const uint64_t* bw,
    int kap,
    int nw,
    uint64_t* b_rot_words,
    uint64_t* e_p_words
) {
    bool a_is_zero = !words_any_nonzero(aw, nw);
    int ab_pop = 0;
    for (int w = 0; w < nw; ++w) ab_pop += __builtin_popcountll(aw[w] & bw[w]);
    int sign_bool = (((kap - ab_pop) & 3) == 2) ? 1 : 0;
    int ref_exp = sign_bool ? -1 : 1;
    int kappa_rot = 0;
    if (!a_is_zero) {
        int p = first_set_bit_in_words(aw, nw);
        kappa_rot = (kap + 1) & 3;
        for (int w = 0; w < nw; ++w) {
            b_rot_words[w] = bw[w];
            e_p_words[w] = 0ULL;
        }
        b_rot_words[p >> 6] ^= (1ULL << (p & 63));
        e_p_words[p >> 6] = (1ULL << (p & 63));
    }
    return {a_is_zero, sign_bool, ref_exp, kappa_rot};
}

static inline int pauli_frame_comm_bit(
    int pauli_kind, const uint8_t* xs_q, const uint8_t* zs_q, int b
) {
    if (pauli_kind == 2) return int(xs_q[b]);
    if (pauli_kind == 0) return int(zs_q[b]);
    return int(xs_q[b] ^ zs_q[b]);
}

struct CppMRQFShot {
    int d = 0;
    int cap = 0;
    int n_words = 0;
    std::vector<uint64_t> h_words;
    std::vector<uint64_t> a_union;
    std::vector<uint64_t> A_buf;
    std::vector<uint8_t> ell_buf;
    std::vector<uint8_t> Gamma_buf;
    std::vector<int32_t> P;

    void init(int nw, int init_cap = 64) {
        n_words = nw;
        cap = init_cap;
        d = 0;
        h_words.assign(size_t(nw), 0);
        a_union.assign(size_t(nw), 0);
        A_buf.assign(size_t(cap) * size_t(nw), 0);
        ell_buf.assign(size_t(cap), 0);
        Gamma_buf.assign(size_t(cap) * size_t(cap), 0);
        P.assign(size_t(cap), -1);
    }

    void clear() {
        d = 0;
        std::fill(h_words.begin(), h_words.end(), 0ULL);
        std::fill(a_union.begin(), a_union.end(), 0ULL);
    }

    void ensure_capacity(int need_d) {
        if (need_d <= cap) return;
        int new_cap = cap > 0 ? cap : 16;
        while (new_cap < need_d) {
            if (new_cap > INT32_MAX / 2) {
                new_cap = need_d;
                break;
            }
            new_cap *= 2;
        }
        std::vector<uint64_t> new_A(size_t(new_cap) * size_t(n_words), 0);
        std::vector<uint8_t> new_ell(size_t(new_cap), 0);
        std::vector<uint8_t> new_Gamma(size_t(new_cap) * size_t(new_cap), 0);
        std::vector<int32_t> new_P(size_t(new_cap), -1);
        if (d > 0) {
            std::memcpy(new_A.data(), A_buf.data(), size_t(d) * size_t(n_words) * sizeof(uint64_t));
            std::memcpy(new_ell.data(), ell_buf.data(), size_t(d) * sizeof(uint8_t));
            for (int r = 0; r < d; ++r) {
                std::memcpy(
                    &new_Gamma[size_t(r) * size_t(new_cap)],
                    &Gamma_buf[size_t(r) * size_t(cap)],
                    size_t(d) * sizeof(uint8_t)
                );
            }
            std::memcpy(new_P.data(), P.data(), size_t(d) * sizeof(int32_t));
        }
        cap = new_cap;
        A_buf.swap(new_A);
        ell_buf.swap(new_ell);
        Gamma_buf.swap(new_Gamma);
        P.swap(new_P);
    }

    void recompute_a_union() {
        for (int w = 0; w < n_words; ++w) {
            uint64_t acc = 0;
            for (int r = 0; r < d; ++r) {
                acc |= A_buf[r * n_words + w];
            }
            a_union[w] = acc;
        }
    }

    void s1_add_column(int r, int s) {
        uint8_t ell_r = ell_buf[r];
        uint8_t ell_s = ell_buf[s];
        uint8_t gamma_rs = Gamma_buf[r * cap + s];

        uint64_t* A_s = &A_buf[s * n_words];
        const uint64_t* A_r = &A_buf[r * n_words];
        for (int w = 0; w < n_words; ++w) {
            A_s[w] ^= A_r[w];
        }

        ell_buf[s] = (ell_s + ell_r + (gamma_rs << 1)) & 3;
        uint8_t new_gamma_rs = gamma_rs ^ (ell_r & 1);

        uint8_t* G_s = &Gamma_buf[s * cap];
        const uint8_t* G_r = &Gamma_buf[r * cap];
        for (int k = 0; k < d; ++k) {
            uint8_t v = G_s[k] ^ G_r[k];
            G_s[k] = v;
            Gamma_buf[k * cap + s] = v;
        }
        Gamma_buf[r * cap + s] = new_gamma_rs;
        Gamma_buf[s * cap + r] = new_gamma_rs;
        Gamma_buf[s * cap + s] = 0;
    }

    void s4_drop_variable(int j, bool recompute_union = true) {
        int k = d;
        if (k == 1) {
            d = 0;
            std::fill(a_union.begin(), a_union.end(), 0ULL);
            return;
        }
        if (j < k - 1) {
            std::memmove(&A_buf[j * n_words], &A_buf[(j + 1) * n_words], (k - 1 - j) * n_words * sizeof(uint64_t));
            std::memmove(&ell_buf[j], &ell_buf[j + 1], (k - 1 - j) * sizeof(uint8_t));
            for (int r = j; r < k - 1; ++r) {
                std::memcpy(&Gamma_buf[r * cap], &Gamma_buf[(r + 1) * cap], k * sizeof(uint8_t));
            }
            for (int r = 0; r < k - 1; ++r) {
                std::memmove(&Gamma_buf[r * cap + j], &Gamma_buf[r * cap + j + 1], (k - 1 - j) * sizeof(uint8_t));
            }
            std::memmove(&P[j], &P[j + 1], (k - 1 - j) * sizeof(int32_t));
        }
        d = k - 1;
        if (recompute_union) {
            recompute_a_union();
        }
    }

    void s3_restrict(const uint8_t* lam, int eps) {
        int first_nz = -1;
        for (int idx = 0; idx < d; ++idx) {
            if (lam[idx]) {
                if (first_nz == -1) first_nz = idx;
                else s1_add_column(first_nz, idx);
            }
        }
        if (first_nz == -1) return;
        apply_eps_and_drop(first_nz, eps & 1);
    }

    void s5_eliminate_zero(int j) {
        uint8_t c = ell_buf[j];
        int k = d;
        uint8_t lam_stack[256];
        std::vector<uint8_t> lam_vec;
        uint8_t* lam = lam_stack;
        if (k > 256) {
            lam_vec.resize(k);
            lam = lam_vec.data();
        }
        for (int r = 0; r < j; ++r) lam[r] = Gamma_buf[r * cap + j];
        for (int r = j + 1; r < k; ++r) lam[r - 1] = Gamma_buf[r * cap + j];
        s4_drop_variable(j, false);
        int d_new = d;
        if (c & 1) {
            for (int r = 0; r < d_new; ++r) {
                ell_buf[r] = (ell_buf[r] - c * lam[r]) & 3;
            }
            for (int r = 0; r < d_new; ++r) {
                if (!lam[r]) continue;
                for (int s = 0; s < d_new; ++s) {
                    if (r != s && lam[s]) Gamma_buf[r * cap + s] ^= 1;
                }
            }
            return;
        }
        bool any_lam = false;
        for (int r = 0; r < d_new; ++r) if (lam[r]) { any_lam = true; break; }
        if (any_lam) s3_restrict(lam, c >> 1);
    }

    void normalize_column(int idx) {
        for (int r = 0; r < idx; ++r) {
            int32_t p_r = P[r];
            if ((A_buf[idx * n_words + (p_r >> 6)] >> (p_r & 63)) & 1ULL) {
                s1_add_column(r, idx);
            }
        }
        int p_new = first_set_bit_in_words(&A_buf[idx * n_words], n_words);
        if (p_new >= 0) {
            int p_word = p_new >> 6;
            int p_bit = p_new & 63;
            for (int s = 0; s < d; ++s) {
                if (s != idx && ((A_buf[s * n_words + p_word] >> p_bit) & 1ULL)) {
                    s1_add_column(idx, s);
                }
            }
            P[idx] = p_new;
        } else {
            s5_eliminate_zero(idx);
        }
    }

    // Gamma[r][col] = Gamma[col][r] = <A_r, b> for r < d0 (all zero when b misses every A_r).
    void fill_gamma_col(int col, int d0, const uint64_t* b_words) {
        bool overlap = words_any_overlap(b_words, a_union.data(), n_words);
        for (int r = 0; r < d0; ++r) {
            uint8_t t = overlap ? uint8_t(word_dot_parity(&A_buf[r * n_words], b_words, n_words)) : 0;
            Gamma_buf[r * cap + col] = t;
            Gamma_buf[col * cap + r] = t;
        }
    }

    // Fold the eps_bit phase of variable f into h/ell, then drop f.
    void apply_eps_and_drop(int f, int eps_bit) {
        if (eps_bit) {
            const uint64_t* A_f = &A_buf[f * n_words];
            for (int w = 0; w < n_words; ++w) h_words[w] ^= A_f[w];
            const uint8_t* G_f = &Gamma_buf[f * cap];
            for (int idx = 0; idx < d; ++idx) {
                ell_buf[idx] = (ell_buf[idx] + (G_f[idx] << 1)) & 3;
            }
        }
        s4_drop_variable(f);
    }

    void p1_apply_pauli(const uint64_t* a_words, const uint64_t* b_words) {
        if (d > 0 && words_any_overlap(b_words, a_union.data(), n_words)) {
            for (int r = 0; r < d; ++r) {
                int t = word_dot_parity(&A_buf[r * n_words], b_words, n_words);
                ell_buf[r] = (ell_buf[r] + (t << 1)) & 3;
            }
        }
        for (int w = 0; w < n_words; ++w) h_words[w] ^= a_words[w];
    }

    void p2_apply_quarter_turn(const uint64_t* a_words, const uint64_t* b_words, int kappa) {
        int d0 = d;
        ensure_capacity(d0 + 1);
        if (d0 > 0) fill_gamma_col(d0, d0, b_words);
        Gamma_buf[d0 * cap + d0] = 0;
        int bh = word_dot_parity(b_words, h_words.data(), n_words);
        uint8_t c = (kappa + 1 + (bh << 1)) & 3;
        for (int w = 0; w < n_words; ++w) {
            A_buf[d0 * n_words + w] = a_words[w];
            a_union[w] |= a_words[w];
        }
        ell_buf[d0] = c;
        P[d0] = -1;
        d = d0 + 1;
        normalize_column(d0);
    }

    void p2_apply_controlled_pauli(
        const uint64_t* a1_words, const uint64_t* b1_words, int kappa1,
        const uint64_t* a2_words, const uint64_t* b2_words, int kappa2
    ) {
        int d0 = d;
        ensure_capacity(d0 + 2);
        if (d0 > 0) {
            fill_gamma_col(d0, d0, b1_words);
            fill_gamma_col(d0 + 1, d0, b2_words);
        }
        int bh1 = word_dot_parity(b1_words, h_words.data(), n_words);
        int bh2 = word_dot_parity(b2_words, h_words.data(), n_words);
        uint8_t c1 = (kappa1 + (bh1 << 1)) & 3;
        uint8_t c2 = (kappa2 + (bh2 << 1)) & 3;
        uint8_t gamma12 = 1 ^ word_dot_parity(b1_words, a2_words, n_words);

        for (int w = 0; w < n_words; ++w) {
            A_buf[d0 * n_words + w] = a1_words[w];
            A_buf[(d0 + 1) * n_words + w] = a2_words[w];
            a_union[w] |= a1_words[w] | a2_words[w];
        }
        ell_buf[d0] = c1;
        ell_buf[d0 + 1] = c2;
        Gamma_buf[d0 * cap + d0] = 0;
        Gamma_buf[(d0 + 1) * cap + (d0 + 1)] = 0;
        Gamma_buf[d0 * cap + (d0 + 1)] = gamma12;
        Gamma_buf[(d0 + 1) * cap + d0] = gamma12;
        P[d0] = -1;
        P[d0 + 1] = -1;
        d = d0 + 2;

        normalize_column(d0);
        if (d > 0 && P[d - 1] == -1) {
            normalize_column(d - 1);
        }
    }

    int p3_measure_diagonal(const uint64_t* b_words, int sigma, int requested_bit, uint64_t* rng_state) {
        int bh = word_dot_parity(b_words, h_words.data(), n_words);
        int beta = (sigma ^ bh) & 1;
        if (d == 0 || !words_any_overlap(b_words, a_union.data(), n_words)) {
            return beta;
        }
        int first_nz = -1;
        int other_nz_stack[256];
        std::vector<int> other_nz_vec;
        int* other_nz = other_nz_stack;
        if (d > 256) {
            other_nz_vec.resize(d);
            other_nz = other_nz_vec.data();
        }
        int n_other = 0;
        for (int r = 0; r < d; ++r) {
            if (word_dot_parity(&A_buf[r * n_words], b_words, n_words)) {
                if (first_nz == -1) first_nz = r;
                else other_nz[n_other++] = r;
            }
        }
        if (first_nz == -1) {
            return beta;
        }
        int m;
        if (requested_bit >= 0) {
            m = requested_bit & 1;
        } else {
            m = splitmix64_draw_bit(rng_state);
        }

        int eps_bit = (m ^ beta) & 1;
        for (int i = 0; i < n_other; ++i) {
            s1_add_column(first_nz, other_nz[i]);
        }
        apply_eps_and_drop(first_nz, eps_bit);
        return m;
    }

    int q1_pauli_expectation(const uint64_t* a_words, const uint64_t* b_words, int kappa) const {
        if (d == 0) {
            for (int w = 0; w < n_words; ++w) {
                if (a_words[w]) return 0;
            }
            int bh = word_dot_parity(b_words, h_words.data(), n_words);
            int exp_mod4 = (kappa + (bh << 1)) & 3;
            return (exp_mod4 == 0) ? 1 : -1;
        }
        for (int w = 0; w < n_words; ++w) {
            if (a_words[w] & ~a_union[w]) return 0;
        }
        uint8_t s_stack[256];
        std::vector<uint8_t> s_vec;
        uint8_t* s = s_stack;
        if (d > 256) {
            s_vec.resize(d);
            s = s_vec.data();
        }
        int n_nz = 0;
        for (int r = 0; r < d; ++r) {
            int32_t p_r = P[r];
            uint8_t bit = uint8_t((a_words[p_r >> 6] >> (p_r & 63)) & 1ULL);
            s[r] = bit;
            if (bit) n_nz++;
        }
        if (n_nz == 0) {
            for (int w = 0; w < n_words; ++w) {
                if (a_words[w]) return 0;
            }
        } else {
            for (int w = 0; w < n_words; ++w) {
                uint64_t recon = 0;
                for (int r = 0; r < d; ++r) {
                    if (s[r]) recon ^= A_buf[r * n_words + w];
                }
                if (recon != a_words[w]) return 0;
            }
        }
        for (int r = 0; r < d; ++r) {
            int at_b_r = word_dot_parity(&A_buf[r * n_words], b_words, n_words);
            int gs = 0;
            const uint8_t* G_r = &Gamma_buf[r * cap];
            for (int c = 0; c < d; ++c) {
                gs ^= (G_r[c] & s[c]);
            }
            int ws_r = ((ell_buf[r] & 1) & s[r]) ^ (gs & 1);
            if (at_b_r != ws_r) return 0;
        }
        int lin = 0;
        int quad = 0;
        for (int r = 0; r < d; ++r) {
            if (s[r]) {
                lin += ell_buf[r];
                const uint8_t* G_r = &Gamma_buf[r * cap];
                for (int c = r + 1; c < d; ++c) {
                    if (s[c] && G_r[c]) quad++;
                }
            }
        }
        int q_s = (lin + (quad << 1)) & 3;
        int bh = word_dot_parity(b_words, h_words.data(), n_words);
        int exp_mod4 = (kappa + (bh << 1) - q_s) & 3;
        return (exp_mod4 == 0) ? 1 : -1;
    }
};

struct CppUnitaryStep {
    int num_groups = 0;
    int arity = 0;
    int n_factors = 0;
    std::vector<int64_t> target_groups;
    std::vector<uint8_t> factor_kinds;
    std::vector<uint8_t> factor_lx1;
    std::vector<uint8_t> factor_lz1;
    std::vector<uint64_t> factor_a1_w;
    std::vector<uint64_t> factor_b1_w;
    std::vector<uint8_t> factor_kap1;
    std::vector<uint8_t> factor_lx2;
    std::vector<uint8_t> factor_lz2;
    std::vector<uint64_t> factor_a2_w;
    std::vector<uint64_t> factor_b2_w;
    std::vector<uint8_t> factor_kap2;
    std::vector<uint64_t> x2z_w;
    std::vector<uint64_t> z2z_w;
    bool has_group_evict = false;
    std::vector<uint64_t> group_x2z_w;
    std::vector<uint64_t> group_z2z_w;
};

static inline void compute_pauli_image(
    int pauli_kind,
    int64_t q,
    int nw,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    uint64_t* aw_buf,
    uint64_t* bw_buf,
    const uint64_t*& aw,
    const uint64_t*& bw,
    int& kap
) {
    size_t off = size_t(q) * size_t(nw);
    if (pauli_kind == 2) {
        aw = z2x_w + off;
        bw = z2z_w + off;
        kap = int(z_kappa[q]);
    } else if (pauli_kind == 0) {
        aw = x2x_w + off;
        bw = x2z_w + off;
        kap = int(x_kappa[q]);
    } else {
        const uint64_t* x_a = x2x_w + off;
        const uint64_t* x_b = x2z_w + off;
        const uint64_t* z_a = z2x_w + off;
        const uint64_t* z_b = z2z_w + off;
        int dp = word_dot_parity(x_b, z_a, nw);
        for (int w = 0; w < nw; ++w) {
            aw_buf[w] = x_a[w] ^ z_a[w];
            bw_buf[w] = x_b[w] ^ z_b[w];
        }
        aw = aw_buf;
        bw = bw_buf;
        kap = (1 + int(x_kappa[q]) + int(z_kappa[q]) + (dp << 1)) & 3;
    }
}

struct BatchedMRQFEngine {
    int num_qubits;
    int batch_size;
    int n_words;
    uint64_t* shot_rng_states;
    std::vector<CppMRQFShot> shots;
    std::vector<uint8_t> is_active;
    std::vector<uint8_t> scratch_delta;
    std::vector<uint64_t> tmp_words_1;
    std::vector<uint64_t> tmp_words_2;
    std::vector<uint64_t> tmp_words_3;
    std::vector<uint64_t> tmp_words_4;
    std::vector<CppUnitaryStep> unitary_steps;

    BatchedMRQFEngine(int nq, int bs, uint64_t* ext_rng_states)
        : num_qubits(nq),
          batch_size(bs),
          n_words(compute_n_words(nq)),
          shot_rng_states(ext_rng_states),
          shots(bs),
          is_active(bs, 0),
          scratch_delta(bs, 0),
          tmp_words_1(n_words, 0),
          tmp_words_2(n_words, 0),
          tmp_words_3(n_words, 0),
          tmp_words_4(n_words, 0) {
        for (int b = 0; b < bs; ++b) {
            shots[b].init(n_words, 64);
        }
    }

    int draw_bit(int b) {
        return splitmix64_draw_bit(&shot_rng_states[b]);
    }

    void inject_group_raw(
        int g,
        int arity,
        int n_factors,
        const int64_t* target_groups,
        const uint8_t* cm_row,
        uint8_t* xs_base,
        uint8_t* zs_base,
        const uint8_t* factor_kinds,
        const uint8_t* factor_lx1,
        const uint8_t* factor_lz1,
        const uint64_t* factor_a1_w,
        const uint64_t* factor_b1_w,
        const uint8_t* factor_kap1,
        const uint8_t* factor_lx2,
        const uint8_t* factor_lz2,
        const uint64_t* factor_a2_w,
        const uint64_t* factor_b2_w,
        const uint8_t* factor_kap2
    ) {
        int B = batch_size;
        int nw = n_words;
        bool any_failed = false;
        for (int b = 0; b < B; ++b) {
            if (!cm_row[b]) { any_failed = true; break; }
        }
        if (!any_failed) return;

        const int64_t* grp = target_groups + size_t(g) * size_t(arity);
        for (int b = 0; b < B; ++b) {
            if (cm_row[b]) continue;
            is_active[b] = 1;
            CppMRQFShot& mrqf = shots[b];

            for (int f = 0; f < n_factors; ++f) {
                size_t gf = size_t(g) * size_t(n_factors) + size_t(f);
                uint8_t kind = factor_kinds[gf];
                const uint64_t* aw1 = factor_a1_w + gf * size_t(nw);
                const uint64_t* bw1 = factor_b1_w + gf * size_t(nw);

                if (kind == 0) {
                    mrqf.p1_apply_pauli(aw1, bw1);
                    continue;
                }

                const uint8_t* lx1 = factor_lx1 + gf * size_t(arity);
                const uint8_t* lz1 = factor_lz1 + gf * size_t(arity);
                int gamma1 = 0;
                for (int r = 0; r < arity; ++r) {
                    int64_t q_r = grp[r];
                    uint8_t fx = xs_base[size_t(q_r) * size_t(B) + size_t(b)];
                    uint8_t fz = zs_base[size_t(q_r) * size_t(B) + size_t(b)];
                    gamma1 ^= (lx1[r] & fz) ^ (lz1[r] & fx);
                }
                int kap1_shot = (int(factor_kap1[gf]) + ((gamma1 & 1) << 1)) & 3;

                if (kind == 1) {
                    mrqf.p2_apply_quarter_turn(aw1, bw1, kap1_shot);
                } else {
                    const uint8_t* lx2 = factor_lx2 + gf * size_t(arity);
                    const uint8_t* lz2 = factor_lz2 + gf * size_t(arity);
                    int gamma2 = 0;
                    for (int r = 0; r < arity; ++r) {
                        int64_t q_r = grp[r];
                        uint8_t fx = xs_base[size_t(q_r) * size_t(B) + size_t(b)];
                        uint8_t fz = zs_base[size_t(q_r) * size_t(B) + size_t(b)];
                        gamma2 ^= (lx2[r] & fz) ^ (lz2[r] & fx);
                    }
                    int kap2_shot = (int(factor_kap2[gf]) + ((gamma2 & 1) << 1)) & 3;
                    const uint64_t* aw2 = factor_a2_w + gf * size_t(nw);
                    const uint64_t* bw2 = factor_b2_w + gf * size_t(nw);
                    mrqf.p2_apply_controlled_pauli(aw1, bw1, kap1_shot, aw2, bw2, kap2_shot);
                }
            }
        }
    }

    void evict_single_shot(
        int b,
        const uint64_t* x2z_w,
        const uint64_t* z2z_w,
        uint8_t* xs_base,
        uint8_t* zs_base,
        uint8_t* out_evict_x,
        uint8_t* out_evict_z,
        int* any_evict_x,
        int* any_evict_z
    ) {
        if (!is_active[b] || shots[b].d != 0) return;
        is_active[b] = 0;
        int B = batch_size;
        int nw = n_words;
        const uint64_t* h = shots[b].h_words.data();
        if (words_any_nonzero(h, nw)) {
            for (int q = 0; q < num_qubits; ++q) {
                size_t qw = size_t(q) * size_t(nw);
                size_t qb = size_t(q) * size_t(B) + size_t(b);
                int xp = word_dot_parity(z2z_w + qw, h, nw);
                int zp = word_dot_parity(x2z_w + qw, h, nw);
                if (xp) {
                    out_evict_x[qb] ^= 1;
                    if (xs_base) xs_base[qb] ^= 1;
                    *any_evict_x = 1;
                }
                if (zp) {
                    out_evict_z[qb] ^= 1;
                    if (zs_base) zs_base[qb] ^= 1;
                    *any_evict_z = 1;
                }
            }
            shots[b].clear();
        }
    }

    void evict_healed_shots(
        const uint64_t* x2z_w,
        const uint64_t* z2z_w,
        uint8_t* xs_base,
        uint8_t* zs_base,
        uint8_t* out_evict_x,
        uint8_t* out_evict_z,
        int* any_evict_x,
        int* any_evict_z
    ) {
        for (int b = 0; b < batch_size; ++b) {
            if (is_active[b] && shots[b].d == 0) {
                evict_single_shot(
                    b, x2z_w, z2z_w, xs_base, zs_base,
                    out_evict_x, out_evict_z, any_evict_x, any_evict_z
                );
            }
        }
    }

    bool shot_has_state(int b) const {
        return is_active[b] &&
            (shots[b].d > 0 || words_any_nonzero(shots[b].h_words.data(), n_words));
    }

    int shot_pauli_exp(
        int b,
        const uint64_t* aw,
        const uint64_t* bw,
        int kap,
        const PauliProjInfo& info,
        int f_q
    ) const {
        if (shot_has_state(b)) {
            int mu = shots[b].q1_pauli_expectation(aw, bw, kap);
            return f_q ? -mu : mu;
        }
        return info.a_is_zero ? (f_q ? -info.ref_exp : info.ref_exp) : 0;
    }

    void project_shot_pauli(
        int b,
        const uint64_t* aw,
        const uint64_t* bw,
        const PauliProjInfo& info,
        int f_q,
        int m,
        bool activate_if_a_zero
    ) {
        CppMRQFShot& mrqf = shots[b];
        if (info.a_is_zero) {
            if (is_active[b] || activate_if_a_zero) {
                if (activate_if_a_zero) is_active[b] = 1;
                mrqf.p3_measure_diagonal(bw, (f_q ^ info.sign_bool) & 1, m, &shot_rng_states[b]);
            }
        } else {
            is_active[b] = 1;
            const uint64_t* b_rot = tmp_words_1.data();
            const uint64_t* e_p = tmp_words_2.data();
            mrqf.p2_apply_quarter_turn(aw, b_rot, info.kappa_rot);
            mrqf.p3_measure_diagonal(e_p, 0, (m ^ f_q) & 1, &shot_rng_states[b]);
            mrqf.p2_apply_quarter_turn(aw, b_rot, (info.kappa_rot + 2) & 3);
        }
    }
};

void* engine_create(int num_qubits, int batch_size, uint64_t* ext_rng_states) {
    if (num_qubits < 0 || batch_size <= 0 || !ext_rng_states) return nullptr;
    int64_t nw64 = (int64_t(num_qubits) + 63) / 64;
    if (nw64 <= 0) nw64 = 1;
    if (nw64 > INT32_MAX / 64) return nullptr;
    if (int64_t(num_qubits) * int64_t(batch_size) > INT32_MAX) return nullptr;
    if (int64_t(batch_size) * nw64 > (1LL << 22)) return nullptr;
    try {
        return new BatchedMRQFEngine(num_qubits, batch_size, ext_rng_states);
    } catch (...) {
        return nullptr;
    }
}

void engine_destroy(void* ptr) {
    delete static_cast<BatchedMRQFEngine*>(ptr);
}

void engine_set_rng_states(void* ptr, uint64_t* ext_rng_states) {
    if (!ptr || !ext_rng_states) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    eng->shot_rng_states = ext_rng_states;
}

void engine_clear(void* ptr) {
    if (!ptr) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    for (int b = 0; b < eng->batch_size; ++b) {
        if (eng->is_active[b]) {
            eng->shots[b].clear();
            eng->is_active[b] = 0;
        }
    }
}

int engine_get_active_shot_indices(void* ptr, int32_t* out_indices, int32_t* out_d) {
    if (!ptr) return -1;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int count = 0;
    for (int b = 0; b < eng->batch_size; ++b) {
        if (eng->is_active[b] && (eng->shots[b].d > 0 || words_any_nonzero(eng->shots[b].h_words.data(), eng->n_words))) {
            if (out_indices) out_indices[count] = b;
            if (out_d) out_d[count] = eng->shots[b].d;
            count++;
        }
    }
    return count;
}

void engine_export_shot(
    void* ptr,
    int b,
    uint64_t* out_h,
    uint64_t* out_au,
    uint64_t* out_A,
    uint8_t* out_ell,
    uint8_t* out_Gamma,
    int32_t* out_P
) {
    if (!ptr || !out_h || !out_au) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (b < 0 || b >= eng->batch_size) return;
    const CppMRQFShot& s = eng->shots[b];
    int nw = eng->n_words;
    int d = s.d;
    if (d > 0 && (!out_A || !out_ell || !out_Gamma || !out_P)) return;
    std::memcpy(out_h, s.h_words.data(), size_t(nw) * sizeof(uint64_t));
    std::memcpy(out_au, s.a_union.data(), size_t(nw) * sizeof(uint64_t));
    if (d > 0) {
        std::memcpy(out_A, s.A_buf.data(), size_t(d) * size_t(nw) * sizeof(uint64_t));
        std::memcpy(out_ell, s.ell_buf.data(), size_t(d) * sizeof(uint8_t));
        for (int r = 0; r < d; ++r) {
            std::memcpy(
                out_Gamma + size_t(r) * size_t(d),
                &s.Gamma_buf[size_t(r) * size_t(s.cap)],
                size_t(d) * sizeof(uint8_t)
            );
        }
        std::memcpy(out_P, s.P.data(), size_t(d) * sizeof(int32_t));
    }
}

void engine_import_shot(
    void* ptr,
    int b,
    int d,
    const uint64_t* in_h,
    const uint64_t* in_au,
    const uint64_t* in_A,
    const uint8_t* in_ell,
    const uint8_t* in_Gamma,
    const int32_t* in_P
) {
    if (!ptr || !in_h || !in_au) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (b < 0 || b >= eng->batch_size || d < 0 || d > eng->num_qubits) return;
    if (d > 0) {
        if (!in_A || !in_ell || !in_Gamma || !in_P) return;
        for (int j = 0; j < d; ++j) {
            if (in_P[j] < 0 || in_P[j] >= eng->num_qubits) return;
        }
    }
    CppMRQFShot& s = eng->shots[b];
    int nw = eng->n_words;
    s.ensure_capacity(d);
    s.d = d;
    std::memcpy(s.h_words.data(), in_h, size_t(nw) * sizeof(uint64_t));
    std::memcpy(s.a_union.data(), in_au, size_t(nw) * sizeof(uint64_t));
    if (d > 0) {
        std::memcpy(s.A_buf.data(), in_A, size_t(d) * size_t(nw) * sizeof(uint64_t));
        std::memcpy(s.ell_buf.data(), in_ell, size_t(d) * sizeof(uint8_t));
        for (int r = 0; r < d; ++r) {
            std::memcpy(
                &s.Gamma_buf[size_t(r) * size_t(s.cap)],
                in_Gamma + size_t(r) * size_t(d),
                size_t(d) * sizeof(uint8_t)
            );
        }
        std::memcpy(s.P.data(), in_P, size_t(d) * sizeof(int32_t));
    }
    eng->is_active[b] = (d > 0 || words_any_nonzero(in_h, nw)) ? 1 : 0;
}

// 1. Fused general conditional Clifford cocycle injection (H, CX, CZ, S, etc.)
void engine_batch_inject_general_clifford(
    void* ptr,
    int num_groups,
    int arity,
    int n_factors,
    const int64_t* target_groups,     // [num_groups * arity]
    const uint8_t* cond_mask,         // [num_groups * B]
    uint8_t* xs_base,                 // [num_qubits * B]
    uint8_t* zs_base,                 // [num_qubits * B]
    const uint8_t* factor_kinds,      // [num_groups * n_factors]
    const uint8_t* factor_lx1,        // [num_groups * n_factors * arity]
    const uint8_t* factor_lz1,        // [num_groups * n_factors * arity]
    const uint64_t* factor_a1_w,      // [num_groups * n_factors * nw]
    const uint64_t* factor_b1_w,      // [num_groups * n_factors * nw]
    const uint8_t* factor_kap1,       // [num_groups * n_factors]
    const uint8_t* factor_lx2,        // [num_groups * n_factors * arity]
    const uint8_t* factor_lz2,        // [num_groups * n_factors * arity]
    const uint64_t* factor_a2_w,      // [num_groups * n_factors * nw]
    const uint64_t* factor_b2_w,      // [num_groups * n_factors * nw]
    const uint8_t* factor_kap2,       // [num_groups * n_factors]
    const uint64_t* x2z_w,
    const uint64_t* z2z_w,
    uint8_t* out_evict_x,
    uint8_t* out_evict_z,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !any_evict_x || !any_evict_z) return;
    *any_evict_x = 0;
    *any_evict_z = 0;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (num_groups < 0 || arity < 1 || arity > 2 || n_factors < 0) return;
    if (!x2z_w || !z2z_w || !out_evict_x || !out_evict_z) return;
    if (num_groups > 0) {
        if (!target_groups || !cond_mask || !xs_base || !zs_base) return;
        if (n_factors > 0 && (!factor_kinds || !factor_lx1 || !factor_lz1 || !factor_a1_w ||
            !factor_b1_w || !factor_kap1 || !factor_lx2 || !factor_lz2 ||
            !factor_a2_w || !factor_b2_w || !factor_kap2)) return;
        size_t tg_count = size_t(num_groups) * size_t(arity);
        size_t gf_count = size_t(num_groups) * size_t(n_factors);
        if (tg_count > size_t(INT32_MAX) || gf_count > size_t(INT32_MAX)) return;
        for (size_t i = 0; i < tg_count; ++i) {
            if (target_groups[i] < 0 || target_groups[i] >= eng->num_qubits) return;
        }
        for (size_t gf = 0; gf < gf_count; ++gf) {
            if (factor_kinds[gf] > 2) return;
        }
    }

    int B = eng->batch_size;
    for (int g = 0; g < num_groups; ++g) {
        eng->inject_group_raw(
            g, arity, n_factors, target_groups, cond_mask + size_t(g) * size_t(B),
            xs_base, zs_base, factor_kinds, factor_lx1, factor_lz1, factor_a1_w,
            factor_b1_w, factor_kap1, factor_lx2, factor_lz2, factor_a2_w,
            factor_b2_w, factor_kap2
        );
    }

    eng->evict_healed_shots(
        x2z_w, z2z_w, xs_base, zs_base, out_evict_x, out_evict_z, any_evict_x, any_evict_z
    );
}

// 2A. Fused batch deterministic Z measurement / reset step
void engine_batch_det_meas_step(
    void* ptr,
    int num_sub,
    const int64_t* q_arr,
    const uint8_t* m_ref_arr,
    const uint64_t* b_sub_words,
    uint8_t* xs_base,
    uint8_t* m_mat,
    const uint64_t* x2z_w,
    const uint64_t* z2z_w,
    uint8_t* accum_x_mask,
    uint8_t* accum_z_mask,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !any_evict_x || !any_evict_z) return;
    *any_evict_x = 0;
    *any_evict_z = 0;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (num_sub < 0 || !x2z_w || !z2z_w || !accum_x_mask || !accum_z_mask) return;
    if (num_sub > 0) {
        if (!q_arr || !m_ref_arr || !b_sub_words || !xs_base || !m_mat) return;
        for (int k = 0; k < num_sub; ++k) {
            if (q_arr[k] < 0 || q_arr[k] >= eng->num_qubits) return;
        }
    }
    int B = eng->batch_size;
    int nw = eng->n_words;

    for (int k = 0; k < num_sub; ++k) {
        const uint8_t* xs_row = xs_base + size_t(q_arr[k]) * size_t(B);
        uint8_t* m_row = m_mat + size_t(k) * size_t(B);
        uint8_t mr = m_ref_arr[k];
        for (int b = 0; b < B; ++b) {
            m_row[b] = xs_row[b] ^ mr;
        }
    }

    for (int b = 0; b < B; ++b) {
        if (!eng->is_active[b]) continue;
        CppMRQFShot& mrqf = eng->shots[b];
        if (mrqf.d == 0 && !words_any_nonzero(mrqf.h_words.data(), nw)) continue;

        for (int k = 0; k < num_sub; ++k) {
            const uint64_t* b_k = b_sub_words + size_t(k) * size_t(nw);
            size_t kb = size_t(k) * size_t(B) + size_t(b);
            if (mrqf.d == 0 || !words_any_overlap(b_k, mrqf.a_union.data(), nw)) {
                if (word_dot_parity(b_k, mrqf.h_words.data(), nw)) {
                    m_mat[kb] ^= 1;
                }
            } else {
                size_t qb = size_t(q_arr[k]) * size_t(B) + size_t(b);
                int sigma = (int(xs_base[qb]) ^ int(m_ref_arr[k])) & 1;
                int m_val = mrqf.p3_measure_diagonal(b_k, sigma, -1, &eng->shot_rng_states[b]);
                m_mat[kb] = uint8_t(m_val);
            }
        }
    }

    eng->evict_healed_shots(
        x2z_w, z2z_w, xs_base, nullptr, accum_x_mask, accum_z_mask, any_evict_x, any_evict_z
    );
}

// 2B. Fused mixed/random Z measurement step (handles Round 1 where is_random == True!)
static inline bool validate_mixed_substep_args(
    const BatchedMRQFEngine* eng,
    int is_random,
    int64_t q,
    int64_t p,
    const uint64_t* beta_or_a_words,
    const uint64_t* b_rot_words,
    const uint64_t* e_p_words,
    int num_gx,
    const int64_t* gx_indices,
    int num_gz,
    const int64_t* gz_indices,
    const uint8_t* accum_x_mask,
    const uint8_t* accum_z_mask,
    const int* any_x_accum,
    const int* any_z_accum
) {
    if (q < 0 || q >= eng->num_qubits) return false;
    if (!is_random) return beta_or_a_words != nullptr;
    if (p < 0 || p >= eng->num_qubits || !beta_or_a_words || !b_rot_words || !e_p_words) {
        return false;
    }
    if (num_gx < 0 || num_gz < 0) return false;
    if (num_gx > 0) {
        if (!gx_indices || !accum_x_mask || !any_x_accum) return false;
        for (int i = 0; i < num_gx; ++i) {
            if (gx_indices[i] < 0 || gx_indices[i] >= eng->num_qubits) return false;
        }
    }
    if (num_gz > 0) {
        if (!gz_indices || !accum_z_mask || !any_z_accum) return false;
        for (int i = 0; i < num_gz; ++i) {
            if (gz_indices[i] < 0 || gz_indices[i] >= eng->num_qubits) return false;
        }
    }
    return true;
}

static void mixed_substep_det(
    BatchedMRQFEngine* eng,
    int64_t q,
    uint8_t m_ref,
    const uint64_t* b_words,
    const uint8_t* xs_base,
    uint8_t* m_shots
) {
    int B = eng->batch_size;
    int nw = eng->n_words;
    const uint8_t* f_q_shots = xs_base + size_t(q) * size_t(B);
    for (int b = 0; b < B; ++b) {
        uint8_t m_b = f_q_shots[b] ^ m_ref;
        if (eng->is_active[b] && eng->shots[b].d > 0) {
            CppMRQFShot& mrqf = eng->shots[b];
            if (!words_any_overlap(b_words, mrqf.a_union.data(), nw)) {
                if (word_dot_parity(b_words, mrqf.h_words.data(), nw)) m_b ^= 1;
            } else {
                int sigma = (int(f_q_shots[b]) ^ int(m_ref)) & 1;
                m_b = uint8_t(
                    mrqf.p3_measure_diagonal(b_words, sigma, -1, &eng->shot_rng_states[b])
                );
            }
        } else if (eng->is_active[b] && eng->shots[b].d == 0) {
            if (word_dot_parity(b_words, eng->shots[b].h_words.data(), nw)) m_b ^= 1;
        }
        m_shots[b] = m_b;
    }
}

static void mixed_substep_rand(
    BatchedMRQFEngine* eng,
    int64_t q,
    uint8_t m_ref,
    int64_t p,
    const uint64_t* a_words,
    const uint64_t* b_rot_words,
    const uint64_t* e_p_words,
    int kappa_rot,
    int num_gx,
    const int64_t* gx_indices,
    int num_gz,
    const int64_t* gz_indices,
    uint8_t* xs_base,
    uint8_t* m_shots,
    uint8_t* accum_x_mask,
    uint8_t* accum_z_mask,
    int* any_x_accum,
    int* any_z_accum
) {
    int B = eng->batch_size;
    const uint8_t* f_q_shots = xs_base + size_t(q) * size_t(B);
    int p_word = int(p >> 6);
    uint64_t p_mask = 1ULL << int(p & 63);
    uint8_t* delta_shots = eng->scratch_delta.data();
    bool any_delta = false;
    for (int b = 0; b < B; ++b) {
        uint8_t m_b = m_shots[b];
        uint8_t fq_b = f_q_shots[b];
        uint8_t d_b = m_b ^ fq_b ^ m_ref;
        if (eng->is_active[b]) {
            CppMRQFShot& mrqf = eng->shots[b];
            if ((mrqf.a_union[p_word] & p_mask) == 0ULL) {
                if ((mrqf.h_words[p_word] & p_mask) != 0ULL) d_b ^= 1;
            } else {
                d_b = 0;
                mrqf.p2_apply_quarter_turn(a_words, b_rot_words, kappa_rot);
                int req = int(m_b ^ fq_b);
                int m_eff = mrqf.p3_measure_diagonal(
                    e_p_words, 0, req, &eng->shot_rng_states[b]
                );
                m_shots[b] = uint8_t(m_eff ^ fq_b);
            }
        }
        delta_shots[b] = d_b;
        if (d_b) any_delta = true;
    }
    if (!any_delta) return;
    if (num_gx > 0) {
        for (int i = 0; i < num_gx; ++i) {
            size_t off = size_t(gx_indices[i]) * size_t(B);
            for (int b = 0; b < B; ++b) {
                accum_x_mask[off + b] ^= delta_shots[b];
                xs_base[off + b] ^= delta_shots[b];
            }
        }
        *any_x_accum = 1;
    }
    if (num_gz > 0) {
        for (int i = 0; i < num_gz; ++i) {
            size_t off = size_t(gz_indices[i]) * size_t(B);
            for (int b = 0; b < B; ++b) accum_z_mask[off + b] ^= delta_shots[b];
        }
        *any_z_accum = 1;
    }
}

void engine_batch_mixed_substep(
    void* ptr,
    int is_random,
    int64_t q,
    uint8_t m_ref,
    int64_t p,
    const uint64_t* beta_or_a_words,
    const uint64_t* b_rot_words,
    const uint64_t* e_p_words,
    int kappa_rot,
    int num_gx,
    const int64_t* gx_indices,
    int num_gz,
    const int64_t* gz_indices,
    uint8_t* xs_base,
    uint8_t* m_shots,
    uint8_t* accum_x_mask,
    uint8_t* accum_z_mask,
    int* any_x_accum,
    int* any_z_accum
) {
    if (!ptr || !xs_base || !m_shots) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (!validate_mixed_substep_args(
            eng, is_random, q, p, beta_or_a_words, b_rot_words, e_p_words,
            num_gx, gx_indices, num_gz, gz_indices,
            accum_x_mask, accum_z_mask, any_x_accum, any_z_accum
        )) {
        return;
    }
    if (!is_random) {
        mixed_substep_det(eng, q, m_ref, beta_or_a_words, xs_base, m_shots);
        return;
    }
    mixed_substep_rand(
        eng, q, m_ref, p, beta_or_a_words, b_rot_words, e_p_words, kappa_rot,
        num_gx, gx_indices, num_gz, gz_indices, xs_base, m_shots,
        accum_x_mask, accum_z_mask, any_x_accum, any_z_accum
    );
}

void engine_batch_evict(
    void* ptr,
    const uint64_t* x2z_w,
    const uint64_t* z2z_w,
    uint8_t* xs_base,
    uint8_t* zs_base,
    uint8_t* accum_x_mask,
    uint8_t* accum_z_mask,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !x2z_w || !z2z_w || !accum_x_mask || !accum_z_mask ||
        !any_evict_x || !any_evict_z) {
        return;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    eng->evict_healed_shots(
        x2z_w, z2z_w, xs_base, zs_base, accum_x_mask, accum_z_mask, any_evict_x, any_evict_z
    );
}

// 3. Fused batch unmatched noisy reset Z
static void reset_z_qubit(
    BatchedMRQFEngine* eng,
    int q,
    const uint8_t* reset_zero_mask,
    const uint8_t* reset_one_mask,
    uint8_t* xs_base,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* z_kappa,
    const uint64_t* x2z_w,
    uint8_t* out_x_kick_mask,
    uint8_t* out_evict_z_mask,
    int* any_x_kick,
    int* any_evict_z
) {
    int B = eng->batch_size;
    int nw = eng->n_words;
    size_t qb_off = size_t(q) * size_t(B);
    const uint8_t* r0_row = reset_zero_mask + qb_off;
    const uint8_t* r1_row = reset_one_mask + qb_off;
    bool any_q = false;
    for (int b = 0; b < B; ++b) {
        if (r0_row[b] | r1_row[b]) { any_q = true; break; }
    }
    if (!any_q) return;

    const uint64_t* aw = z2x_w + size_t(q) * size_t(nw);
    const uint64_t* bw = z2z_w + size_t(q) * size_t(nw);
    int kap = int(z_kappa[q]);
    PauliProjInfo info = prepare_pauli_proj(
        aw, bw, kap, nw, eng->tmp_words_1.data(), eng->tmp_words_2.data()
    );
    uint8_t* xs_q = xs_base + qb_off;
    uint8_t* x_kick_q = out_x_kick_mask + qb_off;
    for (int b = 0; b < B; ++b) {
        if (!(r0_row[b] | r1_row[b])) continue;
        int target_state = r1_row[b] ? 1 : 0;
        int f_q = int(xs_q[b]);
        int z_exp = eng->shot_pauli_exp(b, aw, bw, kap, info, f_q);
        bool need_flip = false;
        if (z_exp == 1 || z_exp == -1) {
            need_flip = (z_exp == 1) ? (target_state == 1) : (target_state == 0);
        } else {
            int m = eng->draw_bit(b);
            eng->project_shot_pauli(b, aw, bw, info, f_q, m, true);
            need_flip = (m != target_state);
            if (eng->shots[b].d == 0) {
                eng->evict_single_shot(
                    b, x2z_w, z2z_w, xs_base, nullptr,
                    out_x_kick_mask, out_evict_z_mask, any_x_kick, any_evict_z
                );
            }
        }
        if (need_flip) {
            x_kick_q[b] ^= 1;
            xs_q[b] ^= 1;
            *any_x_kick = 1;
        }
    }
}

void engine_batch_unmatched_noisy_reset_z(
    void* ptr,
    const uint8_t* reset_zero_mask,  // [num_qubits * B]
    const uint8_t* reset_one_mask,   // [num_qubits * B]
    uint8_t* xs_base,                // [num_qubits * B]
    const uint64_t* z2x_w,           // [num_qubits * nw]
    const uint64_t* z2z_w,           // [num_qubits * nw]
    const uint8_t* z_kappa,          // [num_qubits]
    const uint64_t* x2z_w,           // [num_qubits * nw]
    uint8_t* out_x_kick_mask,        // [num_qubits * B]
    uint8_t* out_evict_z_mask,       // [num_qubits * B]
    int* any_x_kick,
    int* any_evict_z
) {
    if (!ptr || !any_x_kick || !any_evict_z) return;
    *any_x_kick = 0;
    *any_evict_z = 0;
    if (!reset_zero_mask || !reset_one_mask || !xs_base || !z2x_w || !z2z_w || !z_kappa ||
        !x2z_w || !out_x_kick_mask || !out_evict_z_mask) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    for (int q = 0; q < eng->num_qubits; ++q) {
        reset_z_qubit(
            eng, q, reset_zero_mask, reset_one_mask, xs_base, z2x_w, z2z_w, z_kappa,
            x2z_w, out_x_kick_mask, out_evict_z_mask, any_x_kick, any_evict_z
        );
    }
    eng->evict_healed_shots(
        x2z_w, z2z_w, xs_base, nullptr,
        out_x_kick_mask, out_evict_z_mask, any_x_kick, any_evict_z
    );
}

// 4. Fused peek_pauli_batch in C++
void engine_peek_pauli_batch(
    void* ptr,
    int num_targets,
    const int64_t* targets,
    int pauli_kind, // 0=X, 1=Y, 2=Z
    const uint8_t* xs_base,
    const uint8_t* zs_base,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    int8_t* out_exps
) {
    if (!ptr || num_targets < 0 || pauli_kind < 0 || pauli_kind > 2) return;
    if (num_targets == 0) return;
    if (!targets || !xs_base || !zs_base || !x2x_w || !x2z_w || !z2x_w || !z2z_w ||
        !x_kappa || !z_kappa || !out_exps) return;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    for (int idx = 0; idx < num_targets; ++idx) {
        if (targets[idx] < 0 || targets[idx] >= eng->num_qubits) return;
    }
    int B = eng->batch_size;
    int nw = eng->n_words;
    uint64_t* aw_buf = eng->tmp_words_1.data();
    uint64_t* bw_buf = eng->tmp_words_2.data();
    for (int idx = 0; idx < num_targets; ++idx) {
        int64_t q = targets[idx];
        const uint64_t* aw = nullptr;
        const uint64_t* bw = nullptr;
        int kap = 0;
        compute_pauli_image(
            pauli_kind, q, nw, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa,
            aw_buf, bw_buf, aw, bw, kap
        );
        bool a_is_zero = !words_any_nonzero(aw, nw);
        int ab_pop = 0;
        for (int w = 0; w < nw; ++w) ab_pop += __builtin_popcountll(aw[w] & bw[w]);
        int sign_bool = (((kap - ab_pop) & 3) == 2) ? 1 : 0;
        int8_t ref_exp = sign_bool ? -1 : 1;
        const uint8_t* xs_q = xs_base + size_t(q) * size_t(B);
        const uint8_t* zs_q = zs_base + size_t(q) * size_t(B);
        int8_t* out_row = out_exps + size_t(idx) * size_t(B);
        for (int b = 0; b < B; ++b) {
            int f_comm = pauli_frame_comm_bit(pauli_kind, xs_q, zs_q, b);
            if (eng->shot_has_state(b)) {
                int mu = eng->shots[b].q1_pauli_expectation(aw, bw, kap);
                out_row[b] = int8_t(f_comm ? -mu : mu);
            } else {
                out_row[b] = a_is_zero ? (f_comm ? -ref_exp : ref_exp) : 0;
            }
        }
    }
}

int coset_kernels_abi_v2() {
    return 2;
}

// ============================================================================
// ABI v3 Additive Entry Points (WP-26a + C-1 Engine-Sourced Size Checks)
// ============================================================================

int engine_num_qubits(void* ptr) {
    if (!ptr) return -1;
    return static_cast<BatchedMRQFEngine*>(ptr)->num_qubits;
}

int engine_batch_size(void* ptr) {
    if (!ptr) return -1;
    return static_cast<BatchedMRQFEngine*>(ptr)->batch_size;
}

int engine_n_words(void* ptr) {
    if (!ptr) return -1;
    return static_cast<BatchedMRQFEngine*>(ptr)->n_words;
}

static inline int validate_postselect_args(
    const BatchedMRQFEngine* eng,
    int64_t q,
    int pauli_kind,
    int flip_len,
    int desired_len,
    int xs_rows, int xs_cols, const uint8_t* xs_base,
    int zs_rows, int zs_cols, const uint8_t* zs_base,
    int pw_rows, int pw_cols,
    int out_x_rows, int out_x_cols,
    int out_z_rows, int out_z_cols
) {
    int N = eng->num_qubits;
    int B = eng->batch_size;
    int nw = eng->n_words;
    if (pauli_kind < 0 || pauli_kind > 2) return -2;
    if (q < 0 || q >= N) return -3;
    if (flip_len != B || desired_len != B) return -4;
    if (pw_rows != N || pw_cols != nw) return -5;
    if (!xs_base || xs_rows != N || xs_cols != B) return -6;
    if ((pauli_kind != 2 && !zs_base) || (zs_base && (zs_rows != N || zs_cols != B))) return -7;
    if (out_x_rows != N || out_x_cols != B || out_z_rows != N || out_z_cols != B) return -8;
    return 0;
}

int engine_postselect_batch(
    void* ptr,
    int64_t q,
    int pauli_kind,
    int flip_len,
    const uint8_t* flip_mask,
    int desired_len,
    const uint8_t* desired_bits,
    int xs_rows, int xs_cols, uint8_t* xs_base,
    int zs_rows, int zs_cols, uint8_t* zs_base,
    int pw_rows, int pw_cols,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    int out_x_rows, int out_x_cols, uint8_t* out_evict_x,
    int out_z_rows, int out_z_cols, uint8_t* out_evict_z,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !flip_mask || !desired_bits || !x2x_w || !x2z_w || !z2x_w || !z2z_w ||
        !x_kappa || !z_kappa || !out_evict_x || !out_evict_z || !any_evict_x || !any_evict_z) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int vrc = validate_postselect_args(
        eng, q, pauli_kind, flip_len, desired_len, xs_rows, xs_cols, xs_base,
        zs_rows, zs_cols, zs_base, pw_rows, pw_cols,
        out_x_rows, out_x_cols, out_z_rows, out_z_cols
    );
    if (vrc < 0) return vrc;
    *any_evict_x = 0;
    *any_evict_z = 0;
    int B = eng->batch_size;
    int nw = eng->n_words;
    const uint64_t* aw = nullptr;
    const uint64_t* bw = nullptr;
    int kap = 0;
    compute_pauli_image(
        pauli_kind, q, nw, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa,
        eng->tmp_words_3.data(), eng->tmp_words_4.data(), aw, bw, kap
    );
    PauliProjInfo info = prepare_pauli_proj(
        aw, bw, kap, nw, eng->tmp_words_1.data(), eng->tmp_words_2.data()
    );
    const uint8_t* xs_q = xs_base + size_t(q) * size_t(B);
    const uint8_t* zs_q = zs_base ? (zs_base + size_t(q) * size_t(B)) : nullptr;
    bool any_flipped = false;
    for (int b = 0; b < B; ++b) {
        if (!flip_mask[b]) continue;
        any_flipped = true;
        int m = int(desired_bits[b]) & 1;
        int f_q = pauli_frame_comm_bit(pauli_kind, xs_q, zs_q, b);
        eng->project_shot_pauli(b, aw, bw, info, f_q, m, false);
        if (eng->is_active[b] && eng->shots[b].d == 0) {
            eng->evict_single_shot(
                b, x2z_w, z2z_w, xs_base, zs_base,
                out_evict_x, out_evict_z, any_evict_x, any_evict_z
            );
        }
    }
    if (any_flipped) {
        eng->evict_healed_shots(
            x2z_w, z2z_w, xs_base, zs_base,
            out_evict_x, out_evict_z, any_evict_x, any_evict_z
        );
    }
    return 0;
}

static inline int validate_project_pauli_args(
    const BatchedMRQFEngine* eng,
    int pauli_kind,
    int r0_rows, int r0_cols,
    int r1_rows, int r1_cols,
    int num_order,
    const int64_t* order,
    int xs_rows, int xs_cols, const uint8_t* xs_base,
    int zs_rows, int zs_cols, const uint8_t* zs_base,
    int pw_rows, int pw_cols,
    int out_x_rows, int out_x_cols,
    int out_z_rows, int out_z_cols
) {
    int N = eng->num_qubits;
    int B = eng->batch_size;
    int nw = eng->n_words;
    if (pauli_kind < 0 || pauli_kind > 2) return -2;
    if (r0_rows != N || r0_cols != B || r1_rows != N || r1_cols != B) return -3;
    if (pw_rows != N || pw_cols != nw) return -4;
    if (!xs_base || xs_rows != N || xs_cols != B) return -5;
    if ((pauli_kind != 2 && !zs_base) || (zs_base && (zs_rows != N || zs_cols != B))) return -6;
    if (out_x_rows != N || out_x_cols != B || out_z_rows != N || out_z_cols != B) return -7;
    if (num_order > 0) {
        if (!order) return -8;
        for (int i = 0; i < num_order; ++i) {
            if (order[i] < 0 || order[i] >= N) return -9;
        }
    }
    return 0;
}

static bool project_pauli_qubit(
    BatchedMRQFEngine* eng,
    int pauli_kind,
    int64_t q,
    const uint8_t* reset_zero_mask,
    const uint8_t* reset_one_mask,
    uint8_t* xs_base,
    uint8_t* zs_base,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    uint8_t* out_x_kick_mask,
    uint8_t* out_z_kick_mask,
    int* any_x_kick,
    int* any_z_kick
) {
    int B = eng->batch_size;
    int nw = eng->n_words;
    size_t qb_off = size_t(q) * size_t(B);
    const uint8_t* r0_row = reset_zero_mask + qb_off;
    const uint8_t* r1_row = reset_one_mask + qb_off;
    bool any_q = false;
    for (int b = 0; b < B; ++b) {
        if (r0_row[b] | r1_row[b]) { any_q = true; break; }
    }
    if (!any_q) return false;
    const uint64_t* aw = nullptr;
    const uint64_t* bw = nullptr;
    int kap = 0;
    compute_pauli_image(
        pauli_kind, q, nw, x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa,
        eng->tmp_words_3.data(), eng->tmp_words_4.data(), aw, bw, kap
    );
    PauliProjInfo info = prepare_pauli_proj(
        aw, bw, kap, nw, eng->tmp_words_1.data(), eng->tmp_words_2.data()
    );
    uint8_t* xs_q = xs_base + qb_off;
    uint8_t* zs_q = zs_base ? (zs_base + qb_off) : nullptr;
    uint8_t* x_kick_q = out_x_kick_mask + qb_off;
    uint8_t* z_kick_q = out_z_kick_mask + qb_off;
    for (int b = 0; b < B; ++b) {
        if (!(r0_row[b] | r1_row[b])) continue;
        int target_state = r1_row[b] ? 1 : 0;
        int f_q = pauli_frame_comm_bit(pauli_kind, xs_q, zs_q, b);
        int z_exp = eng->shot_pauli_exp(b, aw, bw, kap, info, f_q);
        bool need_flip = false;
        if (z_exp == 1 || z_exp == -1) {
            need_flip = (z_exp == 1) ? (target_state == 1) : (target_state == 0);
        } else {
            int m = eng->draw_bit(b);
            eng->project_shot_pauli(b, aw, bw, info, f_q, m, true);
            need_flip = (m != target_state);
            if (eng->shots[b].d == 0) {
                eng->evict_single_shot(
                    b, x2z_w, z2z_w, xs_base, zs_base,
                    out_x_kick_mask, out_z_kick_mask, any_x_kick, any_z_kick
                );
            }
        }
        if (need_flip) {
            if (pauli_kind == 0) {
                z_kick_q[b] ^= 1;
                if (zs_q) zs_q[b] ^= 1;
                *any_z_kick = 1;
            } else {
                x_kick_q[b] ^= 1;
                xs_q[b] ^= 1;
                *any_x_kick = 1;
            }
        }
    }
    return true;
}

int engine_project_pauli(
    void* ptr,
    int pauli_kind,
    int r0_rows, int r0_cols, const uint8_t* reset_zero_mask,
    int r1_rows, int r1_cols, const uint8_t* reset_one_mask,
    int num_order,
    const int64_t* order,
    int xs_rows, int xs_cols, uint8_t* xs_base,
    int zs_rows, int zs_cols, uint8_t* zs_base,
    int pw_rows, int pw_cols,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    int out_x_rows, int out_x_cols, uint8_t* out_x_kick_mask,
    int out_z_rows, int out_z_cols, uint8_t* out_z_kick_mask,
    int* any_x_kick,
    int* any_z_kick
) {
    if (!ptr || !reset_zero_mask || !reset_one_mask || !x2x_w || !x2z_w || !z2x_w || !z2z_w ||
        !x_kappa || !z_kappa || !out_x_kick_mask || !out_z_kick_mask ||
        !any_x_kick || !any_z_kick) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int vrc = validate_project_pauli_args(
        eng, pauli_kind, r0_rows, r0_cols, r1_rows, r1_cols, num_order, order,
        xs_rows, xs_cols, xs_base, zs_rows, zs_cols, zs_base, pw_rows, pw_cols,
        out_x_rows, out_x_cols, out_z_rows, out_z_cols
    );
    if (vrc < 0) return vrc;
    *any_x_kick = 0;
    *any_z_kick = 0;
    if (num_order == 0) return 0;
    int q_count = (num_order < 0) ? eng->num_qubits : num_order;
    bool any_q_projected = false;
    for (int idx = 0; idx < q_count; ++idx) {
        int64_t q = (num_order < 0) ? int64_t(idx) : order[idx];
        any_q_projected |= project_pauli_qubit(
            eng, pauli_kind, q, reset_zero_mask, reset_one_mask, xs_base, zs_base,
            x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa,
            out_x_kick_mask, out_z_kick_mask, any_x_kick, any_z_kick
        );
    }
    if (any_q_projected) {
        eng->evict_healed_shots(
            x2z_w, z2z_w, xs_base, zs_base,
            out_x_kick_mask, out_z_kick_mask, any_x_kick, any_z_kick
        );
    }
    return 0;
}

static inline int validate_register_step_args(
    const BatchedMRQFEngine* eng,
    int num_groups, int arity, int n_factors,
    const int64_t* target_groups,
    const uint8_t* factor_kinds, const uint8_t* factor_lx1, const uint8_t* factor_lz1,
    const uint64_t* factor_a1_w, const uint64_t* factor_b1_w, const uint8_t* factor_kap1,
    const uint8_t* factor_lx2, const uint8_t* factor_lz2,
    const uint64_t* factor_a2_w, const uint64_t* factor_b2_w, const uint8_t* factor_kap2,
    int pw_rows, int pw_cols,
    size_t& tg_count, size_t& gf_count, size_t& pw_sz, size_t& gfa, size_t& gfw
) {
    int N = eng->num_qubits;
    int nw = eng->n_words;
    if (num_groups < 0 || arity < 1 || arity > 2 || n_factors < 0) return -2;
    if (pw_rows != N || pw_cols != nw) return -3;
    if (!coset_mul_within(size_t(num_groups), size_t(arity), kCosetMaxElems, tg_count) ||
        !coset_mul_within(size_t(num_groups), size_t(n_factors), kCosetMaxElems, gf_count) ||
        !coset_mul_within(size_t(N), size_t(nw), kCosetMaxElems, pw_sz) ||
        !coset_mul_within(gf_count, size_t(arity), kCosetMaxElems, gfa) ||
        !coset_mul_within(gf_count, size_t(nw), kCosetMaxElems, gfw)) {
        return -8;
    }
    if (num_groups > 0) {
        if (!target_groups) return -4;
        for (size_t i = 0; i < tg_count; ++i) {
            if (target_groups[i] < 0 || target_groups[i] >= N) return -5;
        }
    }
    if (gf_count > 0) {
        if (!factor_kinds || !factor_lx1 || !factor_lz1 || !factor_a1_w || !factor_b1_w ||
            !factor_kap1 || !factor_lx2 || !factor_lz2 || !factor_a2_w || !factor_b2_w ||
            !factor_kap2) {
            return -6;
        }
        for (size_t gf = 0; gf < gf_count; ++gf) {
            if (factor_kinds[gf] > 2) return -7;
        }
    }
    return 0;
}

int engine_register_unitary_step(
    void* ptr,
    int num_groups,
    int arity,
    int n_factors,
    const int64_t* target_groups,
    const uint8_t* factor_kinds,
    const uint8_t* factor_lx1,
    const uint8_t* factor_lz1,
    const uint64_t* factor_a1_w,
    const uint64_t* factor_b1_w,
    const uint8_t* factor_kap1,
    const uint8_t* factor_lx2,
    const uint8_t* factor_lz2,
    const uint64_t* factor_a2_w,
    const uint64_t* factor_b2_w,
    const uint8_t* factor_kap2,
    int pw_rows, int pw_cols,
    const uint64_t* x2z_w,
    const uint64_t* z2z_w
) {
    if (!ptr || !x2z_w || !z2z_w) return -1;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    size_t tg_count = 0, gf_count = 0, pw_sz = 0, gfa = 0, gfw = 0;
    int vrc = validate_register_step_args(
        eng, num_groups, arity, n_factors, target_groups,
        factor_kinds, factor_lx1, factor_lz1, factor_a1_w, factor_b1_w, factor_kap1,
        factor_lx2, factor_lz2, factor_a2_w, factor_b2_w, factor_kap2,
        pw_rows, pw_cols, tg_count, gf_count, pw_sz, gfa, gfw
    );
    if (vrc < 0) return vrc;
    try {
        CppUnitaryStep st;
        st.num_groups = num_groups;
        st.arity = arity;
        st.n_factors = n_factors;
        st.x2z_w.assign(x2z_w, x2z_w + pw_sz);
        st.z2z_w.assign(z2z_w, z2z_w + pw_sz);
        if (num_groups > 0) st.target_groups.assign(target_groups, target_groups + tg_count);
        if (gf_count > 0) {
            st.factor_kinds.assign(factor_kinds, factor_kinds + gf_count);
            st.factor_lx1.assign(factor_lx1, factor_lx1 + gfa);
            st.factor_lz1.assign(factor_lz1, factor_lz1 + gfa);
            st.factor_a1_w.assign(factor_a1_w, factor_a1_w + gfw);
            st.factor_b1_w.assign(factor_b1_w, factor_b1_w + gfw);
            st.factor_kap1.assign(factor_kap1, factor_kap1 + gf_count);
            st.factor_lx2.assign(factor_lx2, factor_lx2 + gfa);
            st.factor_lz2.assign(factor_lz2, factor_lz2 + gfa);
            st.factor_a2_w.assign(factor_a2_w, factor_a2_w + gfw);
            st.factor_b2_w.assign(factor_b2_w, factor_b2_w + gfw);
            st.factor_kap2.assign(factor_kap2, factor_kap2 + gf_count);
        }
        int step_id = int(eng->unitary_steps.size());
        eng->unitary_steps.push_back(std::move(st));
        return step_id;
    } catch (const std::bad_alloc&) {
        return -9;
    }
}

int engine_unitary_step_num_groups(void* ptr, int step_id) {
    if (!ptr) return -1;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (step_id < 0 || step_id >= int(eng->unitary_steps.size())) return -1;
    return eng->unitary_steps[step_id].num_groups;
}

int engine_register_unitary_group_evict(
    void* ptr,
    int step_id,
    int num_groups,
    int pw_rows, int pw_cols,
    const uint64_t* group_x2z_w,
    const uint64_t* group_z2z_w
) {
    if (!ptr) return -1;
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    if (step_id < 0 || step_id >= int(eng->unitary_steps.size())) return -2;
    CppUnitaryStep& st = eng->unitary_steps[step_id];
    if (num_groups != st.num_groups) return -3;
    if (pw_rows != eng->num_qubits || pw_cols != eng->n_words) return -4;
    if (num_groups > 0 && (!group_x2z_w || !group_z2z_w)) return -5;
    size_t nw_q = 0, total = 0;
    if (!coset_mul_within(size_t(eng->num_qubits), size_t(eng->n_words), kCosetMaxElems, nw_q) ||
        !coset_mul_within(size_t(num_groups), nw_q, kCosetMaxElems, total)) {
        return -6;
    }
    try {
        st.group_x2z_w.assign(group_x2z_w, group_x2z_w + total);
        st.group_z2z_w.assign(group_z2z_w, group_z2z_w + total);
        st.has_group_evict = true;
        return 0;
    } catch (const std::bad_alloc&) {
        return -7;
    }
}

int engine_inject_group_v3(
    void* ptr,
    int step_id,
    int g_idx,
    int cm_len, const uint8_t* cm_row,
    int xs_rows, int xs_cols, uint8_t* xs_base,
    int zs_rows, int zs_cols, uint8_t* zs_base,
    int out_x_rows, int out_x_cols, uint8_t* out_evict_x,
    int out_z_rows, int out_z_cols, uint8_t* out_evict_z,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !cm_row || !xs_base || !zs_base || !out_evict_x || !out_evict_z ||
        !any_evict_x || !any_evict_z) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int N = eng->num_qubits;
    int B = eng->batch_size;
    int nw = eng->n_words;
    if (step_id < 0 || step_id >= int(eng->unitary_steps.size())) return -2;
    const CppUnitaryStep& st = eng->unitary_steps[step_id];
    if (g_idx < 0 || g_idx >= st.num_groups) return -3;
    if (cm_len != B) return -4;
    if (xs_rows != N || xs_cols != B || zs_rows != N || zs_cols != B) return -5;
    if (out_x_rows != N || out_x_cols != B || out_z_rows != N || out_z_cols != B) return -6;

    *any_evict_x = 0;
    *any_evict_z = 0;

    eng->inject_group_raw(
        g_idx, st.arity, st.n_factors, st.target_groups.data(), cm_row, xs_base, zs_base,
        st.factor_kinds.data(), st.factor_lx1.data(), st.factor_lz1.data(),
        st.factor_a1_w.data(), st.factor_b1_w.data(), st.factor_kap1.data(),
        st.factor_lx2.data(), st.factor_lz2.data(), st.factor_a2_w.data(),
        st.factor_b2_w.data(), st.factor_kap2.data()
    );

    const uint64_t* ev_x2z = st.has_group_evict
        ? (st.group_x2z_w.data() + size_t(g_idx) * size_t(N) * size_t(nw))
        : st.x2z_w.data();
    const uint64_t* ev_z2z = st.has_group_evict
        ? (st.group_z2z_w.data() + size_t(g_idx) * size_t(N) * size_t(nw))
        : st.z2z_w.data();
    eng->evict_healed_shots(
        ev_x2z, ev_z2z, xs_base, zs_base, out_evict_x, out_evict_z, any_evict_x, any_evict_z
    );
    return 0;
}

int engine_batch_inject_general_clifford_v3(
    void* ptr,
    int step_id,
    int cm_rows, int cm_cols, const uint8_t* cond_mask,
    int xs_rows, int xs_cols, uint8_t* xs_base,
    int zs_rows, int zs_cols, uint8_t* zs_base,
    int out_x_rows, int out_x_cols, uint8_t* out_evict_x,
    int out_z_rows, int out_z_cols, uint8_t* out_evict_z,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !xs_base || !zs_base || !out_evict_x || !out_evict_z ||
        !any_evict_x || !any_evict_z) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int N = eng->num_qubits;
    int B = eng->batch_size;
    if (step_id < 0 || step_id >= int(eng->unitary_steps.size())) return -2;
    const CppUnitaryStep& st = eng->unitary_steps[step_id];
    if (cm_rows != st.num_groups || cm_cols != B) return -3;
    if (cm_rows > 0 && !cond_mask) return -4;
    if (xs_rows != N || xs_cols != B || zs_rows != N || zs_cols != B) return -5;
    if (out_x_rows != N || out_x_cols != B || out_z_rows != N || out_z_cols != B) return -6;

    *any_evict_x = 0;
    *any_evict_z = 0;

    for (int g = 0; g < st.num_groups; ++g) {
        eng->inject_group_raw(
            g, st.arity, st.n_factors, st.target_groups.data(),
            cond_mask + size_t(g) * size_t(B), xs_base, zs_base,
            st.factor_kinds.data(), st.factor_lx1.data(), st.factor_lz1.data(),
            st.factor_a1_w.data(), st.factor_b1_w.data(), st.factor_kap1.data(),
            st.factor_lx2.data(), st.factor_lz2.data(), st.factor_a2_w.data(),
            st.factor_b2_w.data(), st.factor_kap2.data()
        );
    }

    eng->evict_healed_shots(
        st.x2z_w.data(), st.z2z_w.data(), xs_base, zs_base,
        out_evict_x, out_evict_z, any_evict_x, any_evict_z
    );
    return 0;
}

int engine_peek_pauli_batch_v3(
    void* ptr,
    int num_targets,
    const int64_t* targets,
    int pauli_kind,
    int xs_rows, int xs_cols, const uint8_t* xs_base,
    int zs_rows, int zs_cols, const uint8_t* zs_base,
    int pw_rows, int pw_cols,
    const uint64_t* x2x_w,
    const uint64_t* x2z_w,
    const uint64_t* z2x_w,
    const uint64_t* z2z_w,
    const uint8_t* x_kappa,
    const uint8_t* z_kappa,
    int out_rows, int out_cols, int8_t* out_exps
) {
    if (!ptr || !xs_base || !zs_base || !x2x_w || !x2z_w || !z2x_w || !z2z_w ||
        !x_kappa || !z_kappa) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int N = eng->num_qubits;
    int B = eng->batch_size;
    int nw = eng->n_words;
    if (num_targets < 0 || pauli_kind < 0 || pauli_kind > 2) return -2;
    if (xs_rows != N || xs_cols != B || zs_rows != N || zs_cols != B) return -3;
    if (pw_rows != N || pw_cols != nw) return -4;
    if (out_rows != num_targets || out_cols != B) return -5;
    if (num_targets > 0) {
        if (!targets || !out_exps) return -6;
        for (int idx = 0; idx < num_targets; ++idx) {
            if (targets[idx] < 0 || targets[idx] >= N) return -7;
        }
    }
    if (num_targets == 0) return 0;
    engine_peek_pauli_batch(
        ptr, num_targets, targets, pauli_kind, xs_base, zs_base,
        x2x_w, x2z_w, z2x_w, z2z_w, x_kappa, z_kappa, out_exps
    );
    return 0;
}

int engine_batch_evict_v3(
    void* ptr,
    int pw_rows, int pw_cols,
    const uint64_t* x2z_w,
    const uint64_t* z2z_w,
    int xs_rows, int xs_cols, uint8_t* xs_base,
    int zs_rows, int zs_cols, uint8_t* zs_base,
    int out_x_rows, int out_x_cols, uint8_t* accum_x_mask,
    int out_z_rows, int out_z_cols, uint8_t* accum_z_mask,
    int* any_evict_x,
    int* any_evict_z
) {
    if (!ptr || !x2z_w || !z2z_w || !accum_x_mask || !accum_z_mask ||
        !any_evict_x || !any_evict_z) {
        return -1;
    }
    auto* eng = static_cast<BatchedMRQFEngine*>(ptr);
    int N = eng->num_qubits;
    int B = eng->batch_size;
    int nw = eng->n_words;
    if (pw_rows != N || pw_cols != nw) return -2;
    if (xs_base && (xs_rows != N || xs_cols != B)) return -3;
    if (zs_base && (zs_rows != N || zs_cols != B)) return -4;
    if (out_x_rows != N || out_x_cols != B || out_z_rows != N || out_z_cols != B) return -5;

    *any_evict_x = 0;
    *any_evict_z = 0;
    eng->evict_healed_shots(
        x2z_w, z2z_w, xs_base, zs_base, accum_x_mask, accum_z_mask, any_evict_x, any_evict_z
    );
    return 0;
}

int coset_kernels_abi_v3() {
    return 3;
}

} // extern "C"
