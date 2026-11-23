// Copyright 2026 Google LLC
//
// MagicBatchEngine (WP-25 stage S4a): a persistent, engine-owned store of the per-shot
// Phase-QFE expansions of one Magicside batch, driven by a few C commands that loop over the
// existing kernels. Included ONCE, at the END of magic_kernels.cpp, because it calls
// magic_apply_pauli_batch (defined above it) and, like qfe_core.hpp, keeps its helpers in an
// anonymous namespace. Entry points are prefixed `mbe_` (never `magic_`), so the `magic_*`
// kernel table and the tests that enumerate it are unaffected; the Python side declares them
// in _native/magic_engine.py with its own ABI sentinel.
//
// Storage per shot is the tight ShotExpansion layout (weights as interleaved re/im doubles,
// ell chi*d, G d*d, ...), so import/export are plain copies. The engine is the sole owner of
// the shots: nothing outside it holds a pointer into them.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct EngineShot {
    size_t chi = 1, d = 0;
    bool canon = true;
    double log_mass = 0.0, atol = 0.0;
    std::vector<double> w;                // 2*chi, interleaved re/im (complex128 memory)
    std::vector<uint64_t> h, A, a_union;  // chi*nw, d*nw, nw
    std::vector<uint8_t> ell, G, eo;      // chi*d, d*d, d
    std::vector<int32_t> piv;             // d
};

// The measurement S5a (mbe_measure_split / mbe_measure_finish): the observable of the command, as
// copied from the caller once, and the two projected branches of every shot that needs a Born
// draw, kept between the two calls. Any command that changes a shot drops it (valid = false).
struct MeasureJob {
    bool keep_rot = false, rotating = false, reset = false, flip_ref = false;
    uint8_t kappa = 0, diag_kappa = 0, rot_kappa = 0, kick_kappa = 0;
    std::vector<uint64_t> words;  // a, b, diagonal b, rotated b, kick a, kick b: nw words each
    const uint64_t* part(size_t k, size_t nw) const noexcept { return words.data() + k * nw; }
};

struct MeasurePending {
    bool valid = false;
    size_t n_active = 0;
    MeasureJob job;
    std::vector<int32_t> shots;      // the shots that need a draw, ascending
    std::vector<EngineShot> pool;    // branch m of active shot i is pool[2 * i + m]
    void release() noexcept {
        valid = false, n_active = 0;
        std::vector<int32_t>().swap(shots), std::vector<EngineShot>().swap(pool);
    }
};

struct MagicBatchEngine {
    size_t B, n_qubits, nw;
    double atol;
    std::vector<EngineShot> shots;
    MeasurePending pend;

    MagicBatchEngine(size_t batch, size_t nq, double tol)
        : B(batch), n_qubits(nq), nw(std::max<size_t>(1, (nq + 63) >> 6)), atol(tol),
          shots(batch) {
        reset();
    }

    // ShotExpansion.zero(): chi=1, d=0, weight 1+0j, h=0, canonical, log_mass 0. May throw
    // (out of memory): every shot is then either reset or untouched, never half reset (the
    // capacity it needs is reserved before anything is changed), earlier ones are reset.
    void reset() {
        pend.release();
        for (EngineShot& s : shots) {
            s.w.reserve(2), s.h.reserve(nw), s.a_union.reserve(nw);
            s.chi = 1, s.d = 0, s.canon = true, s.log_mass = 0.0, s.atol = atol;
            s.w.assign({1.0, 0.0});
            s.h.assign(nw, 0ULL);
            s.a_union.assign(nw, 0ULL);
            s.A.clear(), s.ell.clear(), s.G.clear(), s.eo.clear(), s.piv.clear();
        }
    }
};

std::atomic<size_t> g_live_engines{0};

inline MagicBatchEngine* engine_of(void* handle) noexcept {
    return static_cast<MagicBatchEngine*>(handle);
}

// Size arguments of the raw ABI are caller data. Every element count derived from one (a size
// itself, a product of sizes) is checked against kMaxElems BEFORE it is used in pointer
// arithmetic (`p + n`), in a multiplication or in an allocation: an array of at most kMaxElems
// items of up to 8 bytes is smaller than PTRDIFF_MAX bytes, so `p + n` cannot wrap and no
// vector refuses it with length_error. A size that passes may still be too big to allocate:
// that is the out-of-memory result (-3 / 2), never undefined behaviour. (That the arrays behind
// the pointers really hold that many elements stays the caller's contract, as for every raw
// entry point.)
constexpr size_t kMaxElems = static_cast<size_t>(PTRDIFF_MAX) / 8;

// a * b into `out`; false (out unset) when the product exceeds `limit`.
inline bool mul_within(size_t a, size_t b, size_t limit, size_t& out) noexcept {
    if (a != 0 && b > limit / a) return false;
    out = a * b;
    return true;
}

// Copies p[0:n) into v. false: null source. Throws std::length_error for a size no vector can
// hold (checked before any pointer arithmetic on the caller's address).
template <class T>
bool fill_vector(std::vector<T>& v, const T* p, size_t n) {
    if (n != 0 && p == nullptr) return false;
    if (n > v.max_size()) throw std::length_error("mbe: shot array too large");
    if (n == 0) v.clear();
    else v.assign(p, p + n);
    return true;
}

template <class T>
bool copy_out(const std::vector<T>& v, T* dst) {
    if (!v.empty() && dst == nullptr) return false;
    if (!v.empty()) std::copy(v.begin(), v.end(), dst);
    return true;
}

inline bool all_finite(const double* x, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) return false;
    }
    return true;
}

// numpy `weights * complex(cr, ci)` (array x python-complex scalar) is FMA-contracted:
//   re = fma(a, cr, -(b*ci)),  im = fma(a, ci, b*cr)
// including the signs of zeros. An unfused a*cr - b*ci differs by 1 ulp at odd powers of omega.
inline void cmul_scalar(double* w_ri, size_t n, double cr, double ci) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const double a = w_ri[2 * i], b = w_ri[2 * i + 1];
        w_ri[2 * i] = std::fma(a, cr, -(b * ci));
        w_ri[2 * i + 1] = std::fma(a, ci, b * cr);
    }
}

// numpy `np.complex128(c) * weights` (scalar on the left) is the other contraction, form B:
//   re = fma(a, cr, -(b*ci)),  im = fma(b, cr, a*ci).
inline void cmul_form_b(double a, double b, double cr, double ci, double& re, double& im) noexcept {
    re = std::fma(a, cr, -(b * ci)), im = std::fma(b, cr, a * ci);
}

// numpy's in-place `w *= c` on a ONE-element array is unfused: plain a*cr - b*ci.
inline void cmul_plain(double a, double b, double cr, double ci, double& re, double& im) noexcept {
    re = a * cr - b * ci, im = a * ci + b * cr;
}

// numpy's float64 pairwise sum (contiguous): below 8 sequential, up to 128 eight interleaved
// accumulators, above that a split at n/2 rounded down to a multiple of 8. A plain sequential
// sum differs from np.sum in the last bit for most lengths >= 8. `a(i)` is the i-th addend
// (computed on the fly: nothing is allocated, so no out-of-memory path exists).
template <class Addend>
double pairwise_sum(const Addend& a, size_t base, size_t n) noexcept {
    if (n < 8) {
        double r = -0.0;
        for (size_t i = 0; i < n; ++i) r += a(base + i);
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (size_t j = 0; j < 8; ++j) r[j] = a(base + j);
        size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (size_t j = 0; j < 8; ++j) r[j] += a(base + i + j);
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a(base + i);
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, base, n2) + pairwise_sum(a, base + n2, n - n2);
}

// ShotExpansion.norm_sq's final line: float(np.sum(re*re + im*im)) over the weights.
double norm_sq_pairwise(const double* w_ri, size_t chi) noexcept {
    if (chi == 0) return 0.0;
    const auto sq = [w_ri](size_t i) noexcept {
        const double re = w_ri[2 * i], im = w_ri[2 * i + 1];
        return re * re + im * im;  // built with -ffp-contract=off: never fused
    };
    return pairwise_sum(sq, 0, chi);
}


// ---------------------------------------------------------------------------------------------
// Masked Cliffords (S4b): the C++ half of `channels.apply_clifford_factors(check_unitary=True)`.
// Each shot is processed on a staged copy and committed only if its whole factor list succeeded,
// as `_live_exps[s] = apply_clifford_factors(...)` assigns only after the list: the failing shot
// and every later one are untouched, earlier ones are already updated.
// ---------------------------------------------------------------------------------------------

constexpr double kMergeCancelTol = 1e-13;  // phase_qfe.K1_CANCEL_REL_TOL

enum FactorErrorKind : int64_t {
    kErrNone = 0, kErrNonFiniteWeights = 1, kErrNonFiniteNorm = 2, kErrKernel = 3,
    kErrUnitarity = 4, kErrNotHermitian = 5, kErrPadding = 6, kErrMergeNonFinite = 7,
    kErrMaxComponents = 8, kErrBadNorm = 9, kErrRotation = 10, kErrZeroChosen = 11,
    // S5b: the Born-split checks of `born_split_expansion` / `measure_expansion` for one shot.
    kErrNonFiniteMass = 12, kErrNegativeMass = 13, kErrConservation = 14, kErrZeroPostselect = 15,
};

// Which phase_qfe function a failure is reported for (its name is in the exception text).
enum FactorOp : int64_t {
    kOpPauli = 0, kOpQuarterTurn = 1, kOpPauliT = 2, kOpCcz = 3, kOpBlock = 4, kOpMerge = 5,
    kOpProject = 6,
};

struct FactorError {
    int64_t kind = kErrNone, op = 0, aux = 0, aux2 = 0;  // op: 0 apply_pauli, 1 apply_quarter_turn
    double pre = 0.0, post = 0.0, z = 0.0;  // z: only the S5b Born-split failures set it
};

struct PauliView {
    const uint64_t *a, *b;
    uint8_t kappa;
};

struct CliffordScratch {
    EngineShot work, out;
    std::vector<double> mw;
    std::vector<uint64_t> mh;
    std::vector<uint8_t> mell;
};

inline void set_error(FactorError& e, int64_t kind, int64_t op, int64_t aux = 0, int64_t aux2 = 0,
                      double pre = 0.0, double post = 0.0) {
    e.kind = kind, e.op = op, e.aux = aux, e.aux2 = aux2, e.pre = pre, e.post = post;
}

inline bool padding_clear(const MagicBatchEngine& e, const PauliView& ax) noexcept {
    return ((ax.a[e.nw - 1] | ax.b[e.nw - 1]) & ~tail_mask(e.n_qubits)) == 0ULL;
}

// phase_qfe.check_unitary_norm: true when the squared norm is conserved.
inline bool unitary_ok(double pre, double post, double tol) noexcept {
    return std::isfinite(pre) && std::isfinite(post) && pre >= 0.0 && post >= 0.0
        && std::fabs(post - pre) <= tol * std::max(1.0, pre);
}

// `merge_frames(shot)` rows (K1 canonicalise + merge) into the scratch vectors; returns chi.
inline size_t merged_rows(const MagicBatchEngine& e, const EngineShot& s, CliffordScratch& sc) {
    size_t oc = 0;
    sc.mw.resize(2 * s.chi), sc.mh.resize(s.chi * e.nw), sc.mell.resize(s.chi * s.d);
    magic_canonicalize_and_merge(
        s.chi, s.d, e.nw, s.w.data(), s.h.data(), s.ell.data(), s.A.data(), s.G.data(),
        s.piv.data(), s.eo.data(), kMergeCancelTol, &oc, sc.mw.data(), sc.mh.data(),
        sc.mell.data(), nullptr
    );
    return oc;
}

// Adopts the first `oc` merged rows (see merged_rows) as the shot's rows; the shot is canonical.
inline void adopt_merged(const MagicBatchEngine& e, EngineShot& s, const CliffordScratch& sc,
                         size_t oc) {
    s.w.assign(sc.mw.begin(), sc.mw.begin() + 2 * oc);
    s.h.assign(sc.mh.begin(), sc.mh.begin() + oc * e.nw);
    s.ell.assign(sc.mell.begin(), sc.mell.begin() + oc * s.d);
    s.chi = oc, s.canon = true;
}

// ShotExpansion.norm_sq(): rows are merged first (for the value only) when not canonical, chi > 1.
inline double shot_norm_sq(const MagicBatchEngine& e, const EngineShot& s, CliffordScratch& sc) {
    if (s.chi == 0) return 0.0;
    if (!s.canon && s.chi > 1) {
        const size_t oc = merged_rows(e, s, sc);
        return oc == 0 ? 0.0 : norm_sq_pairwise(sc.mw.data(), oc);
    }
    return norm_sq_pairwise(s.w.data(), s.chi);
}

// phase_qfe.apply_pauli on the staged shot (chi == 0 is a no-op).
inline bool pauli_step(const MagicBatchEngine& e, EngineShot& s, const PauliView& ax,
                       FactorError& err) {
    if (!padding_clear(e, ax)) return set_error(err, kErrPadding, kOpPauli), false;
    if (s.chi == 0) return true;
    if (!all_finite(s.w.data(), 2 * s.chi)) {
        return set_error(err, kErrNonFiniteWeights, kOpPauli), false;
    }
    magic_apply_pauli_batch(
        s.chi, s.d, e.nw, s.w.data(), s.h.data(), s.ell.data(), s.A.data(), s.a_union.data(),
        ax.a, ax.b, ax.kappa
    );
    return true;
}

// Adopts the kernel's tight output (`_assign_cpp_geom_shot`): geometry, rows, canonical flag.
inline void adopt_rotated(const MagicBatchEngine& e, EngineShot& s, EngineShot& out, size_t oc,
                          size_t od) {
    out.w.resize(2 * oc), out.h.resize(oc * e.nw), out.ell.resize(oc * od);
    out.A.resize(od * e.nw), out.G.resize(od * od), out.piv.resize(od), out.eo.resize(od);
    out.a_union.assign(e.nw, 0ULL);
    for (size_t r = 0; r < od; ++r) {
        for (size_t i = 0; i < e.nw; ++i) out.a_union[i] |= out.A[r * e.nw + i];
    }
    out.chi = oc, out.d = od, out.canon = true, out.log_mass = s.log_mass, out.atol = s.atol;
    std::swap(s, out);
}

// phase_qfe.apply_quarter_turn(sign, check_unitary=check) on the staged shot. Without the check
// (the steps of a sequential block) the norms are not computed: the reference computes and
// ignores them, with no effect that can be seen.
bool quarter_turn_step(const MagicBatchEngine& e, CliffordScratch& sc, const PauliView& ax,
                       int sign, FactorError& err, bool check = true) {
    EngineShot& s = sc.work;
    if (!padding_clear(e, ax)) return set_error(err, kErrPadding, kOpQuarterTurn), false;
    const uint8_t parity = word_dot_parity(ax.a, ax.b, e.nw), kap0 = ax.kappa & 3U;
    if (((kap0 - parity) & 1) != 0) {
        return set_error(err, kErrNotHermitian, kOpQuarterTurn, parity, kap0), false;
    }
    const uint8_t kap = static_cast<uint8_t>((kap0 + (sign == -1 ? 2 : 0)) & 3U);
    if (s.chi > 0 && !all_finite(s.w.data(), 2 * s.chi)) {
        return set_error(err, kErrNonFiniteWeights, kOpQuarterTurn), false;
    }
    const double pre = check ? shot_norm_sq(e, s, sc) : 0.0;
    if (check && !std::isfinite(pre)) {
        return set_error(err, kErrNonFiniteNorm, kOpQuarterTurn), false;
    }
    if (s.chi == 0) return s.canon = true, true;
    if (!s.canon) {
        const size_t oc = merged_rows(e, s, sc);
        adopt_merged(e, s, sc, oc);
        if (oc == 0) {
            if (!check || unitary_ok(pre, 0.0, e.atol)) return true;
            return set_error(err, kErrUnitarity, kOpQuarterTurn, 0, 0, pre, 0.0), false;
        }
    }
    EngineShot& o = sc.out;
    const size_t cap = s.d + 1;
    o.w.resize(2 * s.chi), o.h.resize(s.chi * e.nw), o.ell.resize(s.chi * cap);
    o.A.resize(cap * e.nw), o.G.resize(cap * cap), o.piv.resize(cap), o.eo.resize(cap);
    size_t oc = 0, od = 0;
    const int rc = magic_apply_quarter_turn_tol(
        s.chi, s.d, e.n_qubits, e.nw, s.w.data(), s.h.data(), s.ell.data(), s.A.data(),
        s.G.data(), s.piv.data(), s.eo.data(), ax.a, ax.b, kap, kMergeCancelTol, &oc, &od,
        o.w.data(), o.h.data(), o.ell.data(), o.A.data(), o.G.data(), o.piv.data(), o.eo.data()
    );
    if (rc != 0) return set_error(err, kErrKernel, kOpQuarterTurn, rc), false;
    adopt_rotated(e, s, o, oc, od);
    if (!check) return true;
    const double post = norm_sq_pairwise(s.w.data(), s.chi);  // canonical: no merge first
    if (unitary_ok(pre, post, e.atol)) return true;
    return set_error(err, kErrUnitarity, kOpQuarterTurn, 0, 0, pre, post), false;
}

// The `weights * OMEGA_POWERS[idx]` of apply_clifford_factors (numpy array x python complex).
inline void omega_multiply(EngineShot& s, int32_t idx) noexcept {
    const double c = kInvSqrt2;
    const double re[8] = {1.0, c, 0.0, -c, -1.0, -c, 0.0, c};
    const double im[8] = {0.0, c, 1.0, c, 0.0, -c, -1.0, -c};
    cmul_scalar(s.w.data(), s.chi, re[idx & 7], im[idx & 7]);
}

}  // namespace

namespace {

struct FactorListView {
    bool has_p0;
    int32_t n_r, v_fwd, v_inv;  // v_*: omega index for each direction, -1 for "no multiply"
    const PauliView* axes;      // p0 first (if any), then the r axes
};

// apply_clifford_factors for one staged shot; false and `err` set on failure.
bool factor_list_steps(const MagicBatchEngine& e, CliffordScratch& sc, const FactorListView& fl,
                       bool inverse, FactorError& err) {
    EngineShot& s = sc.work;
    if (s.chi == 0) return true;
    const PauliView* p0 = fl.has_p0 ? fl.axes : nullptr;
    const PauliView* r = fl.axes + (fl.has_p0 ? 1 : 0);
    if (!inverse && p0 != nullptr && !pauli_step(e, s, *p0, err)) return false;
    for (int32_t k = 0; k < fl.n_r; ++k) {
        const PauliView& ax = inverse ? r[fl.n_r - 1 - k] : r[k];
        if (!quarter_turn_step(e, sc, ax, inverse ? -1 : 1, err)) return false;
    }
    if (inverse && p0 != nullptr && !pauli_step(e, s, *p0, err)) return false;
    const int32_t v = inverse ? fl.v_inv : fl.v_fwd;
    if (v >= 0 && s.chi > 0) omega_multiply(s, v);
    return true;
}

// The caller's buffers as `mbe_inject_cliffords` received them. They belong to the caller, who
// may change them while this runs (the GIL is released): each value is read ONCE, into a
// CliffordJob, and everything after `build_job` reads only that private copy.
struct RawFactorTable {
    size_t n_lists, n_axes;
    const uint8_t* has_p0;
    const int32_t *n_r, *v_fwd, *v_inv, *axis_start;
    const uint64_t *axes_a, *axes_b;
    const uint8_t* axes_kappa;
};

struct CliffordJob {
    std::vector<int32_t> shots;
    std::vector<FactorListView> lists;
    std::vector<PauliView> views;
    std::vector<uint64_t> words;  // the `a` rows of all axes, then the `b` rows
};

// 0 ok, -2 the table or the shot list is inconsistent (nothing ran). May throw (out of memory):
// the caller turns that into -3.
int64_t build_job(const MagicBatchEngine& e, const RawFactorTable& t, const int32_t* shot_list,
                  CliffordJob& job) {
    size_t half = 0;  // words of the `a` rows (and of the `b` rows) of all axes
    if (t.n_lists > kMaxElems / 4 || !mul_within(t.n_axes, e.nw, kMaxElems / 2, half)) return -2;
    job.shots.assign(shot_list, shot_list + e.B);
    for (const int32_t l : job.shots) {
        if (l >= static_cast<int64_t>(t.n_lists)) return -2;  // any negative entry means "skip"
    }
    job.words.resize(2 * half);
    std::copy(t.axes_a, t.axes_a + half, job.words.begin());
    std::copy(t.axes_b, t.axes_b + half, job.words.begin() + half);
    job.views.resize(t.n_axes);
    for (size_t i = 0; i < t.n_axes; ++i) {
        job.views[i] = {
            job.words.data() + i * e.nw, job.words.data() + half + i * e.nw,
            static_cast<uint8_t>(t.axes_kappa[i] & 3U)
        };
    }
    job.lists.resize(t.n_lists);
    for (size_t l = 0; l < t.n_lists; ++l) {
        const bool p0 = t.has_p0[l] != 0;
        const int32_t n_r = t.n_r[l], start = t.axis_start[l];
        const int64_t end = static_cast<int64_t>(start) + (p0 ? 1 : 0) + n_r;
        if (n_r < 0 || start < 0 || end > static_cast<int64_t>(t.n_axes)) return -2;
        job.lists[l] = {p0, n_r, t.v_fwd[l], t.v_inv[l], job.views.data() + start};
    }
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Blocks (S4b-2): `phase_qfe.apply_compiled_block(check_unitary=True)` on every group of
// bit-identical shots (the grouping of `simulator_magic._group_shots_by_state`), the result
// copied to every member, then the `max_components` check of `_flush_magic`. A group is run on a
// staged copy of its first shot and committed whole or not at all.
// ---------------------------------------------------------------------------------------------

namespace {

enum StepKind : int32_t { kStepP1 = 0, kStepP2 = 1, kStepN1 = 2, kStepN2 = 3 };

template <class T>
inline bool same_vec(const std::vector<T>& x, const std::vector<T>& y) noexcept {
    if (x.size() != y.size()) return false;
    return x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

inline bool same_bits(double x, double y) noexcept {
    return std::memcmp(&x, &y, sizeof(double)) == 0;
}

// `simulator_magic._state_key` equality: every field a block reads or passes through, as bytes.
inline bool same_shot(const EngineShot& a, const EngineShot& b) noexcept {
    return a.chi == b.chi && a.d == b.d && a.canon == b.canon && same_bits(a.log_mass, b.log_mass)
        && same_bits(a.atol, b.atol) && same_vec(a.w, b.w) && same_vec(a.h, b.h)
        && same_vec(a.ell, b.ell) && same_vec(a.A, b.A) && same_vec(a.a_union, b.a_union)
        && same_vec(a.G, b.G) && same_vec(a.piv, b.piv) && same_vec(a.eo, b.eo);
}

inline uint64_t mix(uint64_t h, uint64_t x) noexcept {
    h = (h ^ x) * 0x9E3779B97F4A7C15ULL;
    return h ^ (h >> 29);
}

template <class T>
inline uint64_t mix_vec(uint64_t h, const std::vector<T>& v) noexcept {
    for (const T& x : v) h = mix(h, static_cast<uint64_t>(x));
    return h;
}

// Cheap prefilter (`_state_fingerprint`): a pure function of the bytes `same_shot` compares, so
// equal shots always collide; it can only split a bucket, never merge two shots.
uint64_t fingerprint(const EngineShot& s) noexcept {
    uint64_t h = mix(mix(s.chi, s.d), s.canon ? 1 : 0);
    uint64_t bits;
    std::memcpy(&bits, &s.log_mass, sizeof bits), h = mix(h, bits);
    std::memcpy(&bits, &s.atol, sizeof bits), h = mix(h, bits);
    h = mix_vec(mix_vec(mix_vec(mix_vec(mix_vec(h, s.A), s.a_union), s.G), s.piv), s.eo);
    const size_t step = std::max<size_t>(1, s.chi / 8);
    for (size_t i = 0; i < s.chi; i += step) {
        std::memcpy(&bits, &s.w[2 * i], sizeof bits), h = mix(h, bits);
    }
    if (s.chi > 0) {
        std::memcpy(&bits, &s.w[2 * s.chi - 2], sizeof bits), h = mix(h, bits);
        const size_t nw = s.h.size() / s.chi;
        for (size_t i = 0; i < nw; ++i) h = mix(h, s.h[(s.chi - 1) * nw + i]);
        for (size_t i = 0; i < s.d; ++i) h = mix(h, s.ell[(s.chi - 1) * s.d + i]);
    }
    return h;
}

// Shots with chi > 0 grouped by identical state, ordered by lowest member (every live shot its
// own group when `dedupe` is off). `same_shot` alone decides membership.
std::vector<std::vector<size_t>> partition_shots(const MagicBatchEngine& e, bool dedupe) {
    std::vector<std::vector<size_t>> groups;
    std::unordered_map<uint64_t, std::vector<size_t>> buckets;
    for (size_t s = 0; s < e.B; ++s) {
        if (e.shots[s].chi == 0) continue;
        if (!dedupe) {
            groups.push_back({s});
            continue;
        }
        std::vector<size_t>& cand = buckets[fingerprint(e.shots[s])];
        bool placed = false;
        for (const size_t g : cand) {
            if (same_shot(e.shots[groups[g][0]], e.shots[s])) {
                groups[g].push_back(s), placed = true;
                break;
            }
        }
        if (!placed) cand.push_back(groups.size()), groups.push_back({s});
    }
    return groups;
}

// The caller's block as `mbe_apply_block` received it (the caller may change it meanwhile).
struct RawBlock {
    int32_t mode;  // 0: the axes rows are the branches of a pauli_sum, 1: sequential steps
    size_t n_steps, n_axes;
    double g_re, g_im;
    const int32_t *step_kind, *step_sign, *step_start;
    const uint64_t *axes_a, *axes_b;
    const uint8_t* axes_kappa;
    const double* axes_w_ri;  // pauli_sum coefficients, or the 8 weights of a CCZ step
};

struct BlockJob {
    int32_t mode = 0;
    double g_re = 1.0, g_im = 0.0;
    std::vector<int32_t> kind, sign, start;
    std::vector<uint64_t> words;  // the `a` rows of all axes, then the `b` rows
    std::vector<PauliView> views;
    std::vector<uint8_t> kappa;
    std::vector<double> w_ri;
};

struct BlockScratch : CliffordScratch {
    std::vector<double> bw;
    std::vector<uint64_t> bh;
    std::vector<uint8_t> bell, tsh;
};

// 0 ok, -2 inconsistent block (nothing ran). May throw (out of memory): the caller maps it to -3.
int64_t build_block_job(const MagicBatchEngine& e, const RawBlock& r, BlockJob& job) {
    if (r.mode != 0 && r.mode != 1) return -2;
    size_t half = 0;  // words of the `a` rows (and of the `b` rows) of all axes
    if (!mul_within(r.n_axes, e.nw, kMaxElems / 2, half) || r.n_steps > kMaxElems) return -2;
    if (r.mode == 0 && r.n_steps != 0) return -2;
    job.mode = r.mode, job.g_re = r.g_re, job.g_im = r.g_im;
    job.words.resize(2 * half);
    std::copy(r.axes_a, r.axes_a + half, job.words.begin());
    std::copy(r.axes_b, r.axes_b + half, job.words.begin() + half);
    job.kappa.assign(r.axes_kappa, r.axes_kappa + r.n_axes);
    job.w_ri.assign(r.axes_w_ri, r.axes_w_ri + 2 * r.n_axes);
    job.views.resize(r.n_axes);
    for (size_t i = 0; i < r.n_axes; ++i) {
        job.kappa[i] &= 3U;
        const uint64_t* wa = job.words.data() + i * e.nw;
        job.views[i] = {wa, wa + half, job.kappa[i]};
        if (!padding_clear(e, job.views[i])) return -2;
    }
    job.kind.assign(r.step_kind, r.step_kind + r.n_steps);
    job.sign.assign(r.step_sign, r.step_sign + r.n_steps);
    job.start.assign(r.step_start, r.step_start + r.n_steps);
    for (size_t i = 0; i < r.n_steps; ++i) {
        const int32_t k = job.kind[i];
        const int64_t len = k == kStepN2 ? 8 : 1, st = job.start[i];
        if (k < kStepP1 || k > kStepN2 || st < 0) return -2;
        if (st + len > static_cast<int64_t>(r.n_axes)) return -2;
        if (k != kStepP1 && job.sign[i] != 1 && job.sign[i] != -1) return -2;
        const PauliView& ax = job.views[st];
        const int par = ax.kappa - word_dot_parity(ax.a, ax.b, e.nw);
        if ((k == kStepP2 || k == kStepN1) && (par & 1) != 0) return -2;  // Python checked it
    }
    return 0;
}

// `r * chi` rows of an expansion (2 weights, nw words and d ell bytes each) must be addressable.
// r comes from a copied axis table and chi from a stored shot, so both are bounded by real
// memory; this makes the products below overflow-proof anyway (std::length_error, reported as
// out of memory, before any `2 * r * chi`-style product is formed).
inline void check_rows(size_t r, size_t chi, size_t nw, size_t d) {
    size_t rows = 0;
    if (!mul_within(r, chi, kMaxElems / (2 + nw + d), rows)) {
        throw std::length_error("mbe: block expansion too large");
    }
}

// The `merge_frames` that ends a block or a sequential path: finite check, K1 merge, canonical.
bool merge_step(const MagicBatchEngine& e, CliffordScratch& sc, FactorError& err) {
    EngineShot& s = sc.work;
    if (s.chi == 0) return s.canon = true, true;
    if (!all_finite(s.w.data(), 2 * s.chi)) {
        return set_error(err, kErrMergeNonFinite, kOpMerge), false;
    }
    adopt_merged(e, s, sc, merged_rows(e, s, sc));
    return true;
}

// N1 / N2 inside a sequential block: branch into r * chi rows and K1-merge them with the shared
// kernels (apply_pauli_t / apply_ccz with merge=True, check_unitary=False).
bool branch_step(const MagicBatchEngine& e, CliffordScratch& sc, const BlockJob& j, size_t i,
                 FactorError& err) {
    EngineShot& s = sc.work;
    const bool ccz = j.kind[i] == kStepN2;
    if (s.chi > 0 && !all_finite(s.w.data(), 2 * s.chi)) {
        return set_error(err, kErrNonFiniteWeights, ccz ? kOpCcz : kOpPauliT), false;
    }
    if (s.chi == 0) return s.canon = true, true;
    const size_t r = ccz ? 8 : 2, st = static_cast<size_t>(j.start[i]);
    check_rows(r, s.chi, e.nw, s.d);
    const size_t half = j.kappa.size() * e.nw;
    EngineShot& o = sc.out;
    o.w.resize(2 * r * s.chi), o.h.resize(r * s.chi * e.nw), o.ell.resize(r * s.chi * s.d);
    size_t oc = 0;
    if (ccz) {
        magic_branch_ccz(
            s.chi, s.d, e.nw, s.w.data(), s.h.data(), s.ell.data(), s.A.data(), s.a_union.data(),
            s.G.data(), s.piv.data(), s.eo.data(), j.w_ri.data() + 2 * st,
            j.words.data() + st * e.nw, j.words.data() + half + st * e.nw, j.kappa.data() + st,
            kMergeCancelTol, &oc, o.w.data(), o.h.data(), o.ell.data(), nullptr
        );
    } else {
        magic_branch_pauli_t(
            s.chi, s.d, e.nw, s.w.data(), s.h.data(), s.ell.data(), s.A.data(), s.a_union.data(),
            s.G.data(), s.piv.data(), s.eo.data(), j.views[st].a, j.views[st].b, j.kappa[st],
            j.sign[i] == -1 ? 1 : 0, kMergeCancelTol, &oc, o.w.data(), o.h.data(), o.ell.data(),
            nullptr
        );
    }
    s.w.swap(o.w), s.h.swap(o.h), s.ell.swap(o.ell);
    s.w.resize(2 * oc), s.h.resize(oc * e.nw), s.ell.resize(oc * s.d);
    s.chi = oc, s.canon = true;
    return true;
}

// numpy `weights *= g` (python complex): form A for two or more elements, but a single element is
// computed unfused (the quirk pinned by numpy_recipes_drift_test).
inline void scale_in_place(EngineShot& s, double gr, double gi) noexcept {
    if (s.chi != 1) return cmul_scalar(s.w.data(), s.chi, gr, gi);
    cmul_plain(s.w[0], s.w[1], gr, gi, s.w[0], s.w[1]);
}

bool sequential_block(const MagicBatchEngine& e, BlockScratch& sc, const BlockJob& j,
                      FactorError& err) {
    EngineShot& s = sc.work;
    scale_in_place(s, j.g_re, j.g_im);
    for (size_t i = 0; i < j.kind.size(); ++i) {
        const int32_t k = j.kind[i];
        const PauliView& ax = j.views[j.start[i]];
        const bool ok = k == kStepP1 ? pauli_step(e, s, ax, err)
            : k == kStepP2 ? quarter_turn_step(e, sc, ax, j.sign[i], err, false)
            : branch_step(e, sc, j, i, err);
        if (!ok) return false;
    }
    return merge_step(e, sc, err);
}

// `_branch_pauli_sum_shot` + merge_frames: (coeff[m] * weights) * i**(kappa + 2 <h, b>) with
// numpy's FMA forms (coefficient on the left: form B; the phase table is exact in every form).
bool pauli_sum_block(const MagicBatchEngine& e, BlockScratch& sc, const BlockJob& j, double pre,
                     FactorError& err) {
    EngineShot& s = sc.work;
    const size_t r = j.views.size(), chi = s.chi, nw = e.nw, d = s.d;
    if (r == 0) {  // _empty_shot_expansion: chi = 0, d = 0, same log_mass and atol
        s.chi = s.d = 0, s.canon = true;
        s.w.clear(), s.h.clear(), s.ell.clear(), s.A.clear();
        s.G.clear(), s.piv.clear(), s.eo.clear();
        s.a_union.assign(nw, 0ULL);
        if (unitary_ok(pre, 0.0, e.atol)) return true;
        return set_error(err, kErrUnitarity, kOpBlock, 0, 0, pre, 0.0), false;
    }
    check_rows(r, chi, nw, d);
    sc.bw.resize(2 * r * chi), sc.bh.resize(r * chi * nw);
    sc.bell.resize(r * chi * d), sc.tsh.resize(d);
    static const double pr[4] = {1.0, 0.0, -1.0, 0.0}, pi[4] = {0.0, 1.0, 0.0, -1.0};
    for (size_t m = 0; m < r; ++m) {
        const PauliView& ax = j.views[m];
        const double cr = j.w_ri[2 * m], ci = j.w_ri[2 * m + 1];
        const bool has_b = words_any_nonzero(ax.b, nw);
        const bool shift = d > 0 && has_b && words_any_overlap(ax.b, s.a_union.data(), nw);
        if (shift) {  // else the rows are copied as they are (the reference does not mask them)
            for (size_t q = 0; q < d; ++q) {
                sc.tsh[q] = static_cast<uint8_t>(word_dot_parity(&s.A[q * nw], ax.b, nw) << 1);
            }
        }
        for (size_t a = 0; a < chi; ++a) {
            const size_t row = m * chi + a;
            const uint8_t bh = has_b ? word_dot_parity(&s.h[a * nw], ax.b, nw) : 0U;
            const uint8_t idx = static_cast<uint8_t>((ax.kappa + (bh << 1)) & 3U);
            const double wa = s.w[2 * a], wb = s.w[2 * a + 1];
            double x, y;
            cmul_form_b(wa, wb, cr, ci, x, y);  // coeffs[m] * weights
            sc.bw[2 * row] = std::fma(x, pr[idx], -(y * pi[idx]));  // * _PHASE4[idx]: exact
            sc.bw[2 * row + 1] = std::fma(x, pi[idx], y * pr[idx]);
            for (size_t w = 0; w < nw; ++w) sc.bh[row * nw + w] = s.h[a * nw + w] ^ ax.a[w];
            for (size_t q = 0; q < d; ++q) {
                const uint8_t v = s.ell[a * d + q];
                sc.bell[row * d + q] = shift ? static_cast<uint8_t>((v + sc.tsh[q]) & 3U) : v;
            }
        }
    }
    if (!all_finite(sc.bw.data(), 2 * r * chi)) {
        return set_error(err, kErrMergeNonFinite, kOpMerge), false;
    }
    EngineShot& o = sc.out;
    o.w.resize(2 * r * chi), o.h.resize(r * chi * nw), o.ell.resize(r * chi * d);
    size_t oc = 0;
    magic_canonicalize_and_merge(
        r * chi, d, nw, sc.bw.data(), sc.bh.data(), sc.bell.data(), s.A.data(), s.G.data(),
        s.piv.data(), s.eo.data(), kMergeCancelTol, &oc, o.w.data(), o.h.data(), o.ell.data(),
        nullptr
    );
    s.w.swap(o.w), s.h.swap(o.h), s.ell.swap(o.ell);
    s.w.resize(2 * oc), s.h.resize(oc * nw), s.ell.resize(oc * d);
    s.chi = oc, s.canon = true;
    const double post = norm_sq_pairwise(s.w.data(), oc);
    if (unitary_ok(pre, post, e.atol)) return true;
    return set_error(err, kErrUnitarity, kOpBlock, 0, 0, pre, post), false;
}

// apply_compiled_block(check_unitary=True) on the staged shot `sc.work` (chi > 0).
bool block_on_shot(const MagicBatchEngine& e, BlockScratch& sc, const BlockJob& j,
                   FactorError& err) {
    EngineShot& s = sc.work;
    if (!all_finite(s.w.data(), 2 * s.chi)) {
        return set_error(err, kErrNonFiniteWeights, kOpBlock), false;
    }
    const double pre = shot_norm_sq(e, s, sc);
    if (!std::isfinite(pre)) return set_error(err, kErrNonFiniteNorm, kOpBlock), false;
    if (j.mode == 0) return pauli_sum_block(e, sc, j, pre, err);
    if (!sequential_block(e, sc, j, err)) return false;
    const double post = norm_sq_pairwise(s.w.data(), s.chi);
    if (unitary_ok(pre, post, e.atol)) return true;
    return set_error(err, kErrUnitarity, kOpBlock, 0, 0, pre, post), false;
}

// Runs the block on every group in order. -1 ok, else the first shot of the group that failed
// (earlier groups are fully updated, it and later ones are not; for the max_components error the
// failing group is updated too). `info` = {groups, live shots}.
int64_t run_block_groups(MagicBatchEngine& e, const BlockJob& j, bool dedupe, int64_t max_chi,
                         FactorError& err, int64_t* info) {
    const std::vector<std::vector<size_t>> groups = partition_shots(e, dedupe);
    info[0] = static_cast<int64_t>(groups.size()), info[1] = 0;
    for (const auto& g : groups) info[1] += static_cast<int64_t>(g.size());
    BlockScratch sc;
    for (const auto& g : groups) {
        sc.work = e.shots[g[0]];
        if (!block_on_shot(e, sc, j, err)) return static_cast<int64_t>(g[0]);
        std::vector<EngineShot> copies;  // every copy is made before any shot is touched
        copies.reserve(g.size() - 1);
        for (size_t k = 1; k < g.size(); ++k) copies.push_back(sc.work);
        std::swap(e.shots[g[0]], sc.work);
        for (size_t k = 1; k < g.size(); ++k) std::swap(e.shots[g[k]], copies[k - 1]);
        const uint64_t chi = e.shots[g[0]].chi;  // `chi > cap`: a negative cap is always exceeded
        if (max_chi < 0 || chi > static_cast<uint64_t>(max_chi)) {
            set_error(err, kErrMaxComponents, kOpBlock, static_cast<int64_t>(chi));
            return static_cast<int64_t>(g[0]);
        }
    }
    return -1;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Measurements (S5a): the C++ half of `simulator_magic._project_substep_batch` for the engine path.
// `mbe_measure_split` classifies every shot, kicks the trivial shots of a reset, and for the others
// rotates and projects BOTH outcomes into per-shot scratch (nothing of a shot changes); Python then
// checks the masses, draws the outcomes and calls `mbe_measure_finish`, which commits the chosen
// branches (un-rotation, renormalisation, reset kick) by swapping them in. The arithmetic is the
// one of `magic_measure_batch` / `_unpack_measured` / `apply_pauli` step by step (same kernels,
// same operation order, numpy-exact sums and multiplies), so every double comes out bit for bit.
// ---------------------------------------------------------------------------------------------

namespace {

enum ShotClass : uint8_t { kDead = 0, kTrivial = 1, kActive = 2, kUnseen = 3 };

struct RawMeasure {
    const uint64_t *a, *b, *kick_a, *kick_b;
    uint8_t kappa, kick_kappa;
    bool keep_rot, is_reset, flip_ref;
};

struct SplitOut {
    uint8_t *cls, *bit;
    double *z, *z01;
    int64_t* chi2;
};

// The observable and kick as copied from the caller (once: the GIL is released, the caller's
// buffers may change); `words` = a, b, diagonal b, rotated b, kick a, kick b, nw words each.
void build_measure_job(const MagicBatchEngine& e, const RawMeasure& r, MeasureJob& job) {
    const size_t nw = e.nw;
    job.words.assign(6 * nw, 0ULL);
    std::copy(r.a, r.a + nw, job.words.begin());
    std::copy(r.b, r.b + nw, job.words.begin() + nw);
    if (r.is_reset) {
        std::copy(r.kick_a, r.kick_a + nw, job.words.begin() + 4 * nw);
        std::copy(r.kick_b, r.kick_b + nw, job.words.begin() + 5 * nw);
    }
    job.keep_rot = r.keep_rot, job.reset = r.is_reset, job.flip_ref = r.flip_ref;
    job.kappa = r.kappa & 3U, job.kick_kappa = r.kick_kappa & 3U;
    job.rotating = false, job.diag_kappa = job.rot_kappa = 0;
}

// measure_expansions_batch's first lines: padding, then Hermiticity of the observable.
bool validate_observable(const MagicBatchEngine& e, const MeasureJob& job, FactorError& err) {
    const uint64_t *a = job.part(0, e.nw), *b = job.part(1, e.nw);
    if (!padding_clear(e, PauliView{a, b, job.kappa})) {
        return set_error(err, kErrPadding, kOpQuarterTurn), false;
    }
    const uint8_t parity = word_dot_parity(a, b, e.nw);
    if (((job.kappa - parity) & 1) != 0) {
        return set_error(err, kErrNotHermitian, kOpQuarterTurn, parity, job.kappa), false;
    }
    return true;
}

// Rotation by the first set bit p of a (the axis becomes Z_p), else the diagonal b as it is.
void prepare_rotation(const MagicBatchEngine& e, MeasureJob& job) {
    const size_t nw = e.nw;
    const uint64_t *a = job.part(0, nw), *b = job.part(1, nw);
    uint64_t *diag = job.words.data() + 2 * nw, *rot_b = job.words.data() + 3 * nw;
    job.rotating = words_any_nonzero(a, nw);
    if (!job.rotating) {
        std::copy(b, b + nw, diag);
        job.diag_kappa = static_cast<uint8_t>(((job.kappa >> 1) & 1U) << 1);
        return;
    }
    const size_t p = static_cast<size_t>(first_set_bit_in_words(a, nw));
    std::copy(b, b + nw, rot_b);
    rot_b[p >> 6] ^= uint64_t(1) << (p & 63U);
    diag[p >> 6] = uint64_t(1) << (p & 63U);
    job.rot_kappa = static_cast<uint8_t>((job.kappa + 1U) & 3U);
}

// The classification loop of `_project_substep_batch`, in shot order: chi == 0 is dead, a shot
// with no rotation, d == 0 and chi == 1 is trivial (its bit needs no draw; for a reset it is
// kicked at once), the rest is active. Returns -1, else the shot whose kick failed (the shots
// before it are classified and kicked, as in the Python loop that raised there).
int64_t classify_and_kick(MagicBatchEngine& e, const MeasureJob& job, const SplitOut& out,
                          size_t& n_active, FactorError& err) {
    const PauliView kick{job.part(4, e.nw), job.part(5, e.nw), job.kick_kappa};
    for (size_t s = 0; s < e.B; ++s) {
        EngineShot& sh = e.shots[s];
        if (sh.chi == 0) {
            out.cls[s] = kDead;
        } else if (!job.keep_rot && sh.d == 0 && sh.chi == 1) {
            const uint8_t bit = static_cast<uint8_t>(
                ((job.kappa >> 1) ^ word_dot_parity(job.part(1, e.nw), sh.h.data(), e.nw)) & 1U);
            out.cls[s] = kTrivial, out.bit[s] = bit;
            if (job.reset && ((bit != 0) != job.flip_ref) && !pauli_step(e, sh, kick, err)) {
                return static_cast<int64_t>(s);
            }
        } else {
            out.cls[s] = kActive, ++n_active;
        }
    }
    return -1;
}

// z of `measure_expansions_batch`: chi == 1 is re*re + im*im, else ShotExpansion.norm_sq() (a
// shot that is not canonical is merged for the value only, which refuses non-finite weights).
bool pre_measure_norm(const MagicBatchEngine& e, const EngineShot& s, CliffordScratch& sc,
                      double& z) {
    if (s.chi > 1 && !s.canon && !all_finite(s.w.data(), 2 * s.chi)) return false;
    return z = shot_norm_sq(e, s, sc), true;
}

struct StateView {  // a state to project: tight rows and geometry
    size_t chi, d;
    const double* w;
    const uint64_t *h, *A;
    const uint8_t *ell, *G, *eo;
    const int32_t* piv;
};

// The rotation of `magic_measure_batch` for one shot: merge a non-canonical shot, quarter turn,
// then the unitarity check against z_in with the SEQUENTIAL post-sum (not the pairwise one: the
// batch kernel does it so). False: the rotation failed or its check did.
bool rotate_for_measure(const MagicBatchEngine& e, const MeasureJob& job, const EngineShot& sh,
                        double z_in, CliffordScratch& sc, StateView& v) {
    const size_t nw = e.nw;
    v = {sh.chi, sh.d, sh.w.data(), sh.h.data(), sh.A.data(), sh.ell.data(), sh.G.data(),
         sh.eo.data(), sh.piv.data()};
    if (!job.rotating) return true;
    if (!sh.canon) {
        v.chi = merged_rows(e, sh, sc);
        v.w = sc.mw.data(), v.h = sc.mh.data(), v.ell = sc.mell.data();
    }
    double post = 0.0;
    if (v.chi > 0) {
        EngineShot& o = sc.out;
        const size_t cap = v.d + 1;
        check_rows(2, v.chi, nw, cap);
        o.w.resize(2 * v.chi), o.h.resize(v.chi * nw), o.ell.resize(v.chi * cap);
        o.A.resize(cap * nw), o.G.resize(cap * cap), o.piv.resize(cap), o.eo.resize(cap);
        size_t oc = 0, od = 0;
        const int rc = magic_apply_quarter_turn_tol(
            v.chi, v.d, e.n_qubits, nw, v.w, v.h, v.ell, v.A, v.G, v.piv, v.eo, job.part(0, nw),
            job.part(3, nw), job.rot_kappa, kMergeCancelTol, &oc, &od, o.w.data(), o.h.data(),
            o.ell.data(), o.A.data(), o.G.data(), o.piv.data(), o.eo.data()
        );
        if (rc != 0) return false;
        v = {oc, od, o.w.data(), o.h.data(), o.A.data(), o.ell.data(), o.G.data(), o.eo.data(),
             o.piv.data()};
        for (size_t i = 0; i < 2 * oc; i += 2) post += v.w[i] * v.w[i] + v.w[i + 1] * v.w[i + 1];
    }
    return std::isfinite(post) && std::abs(post - z_in) <= e.atol * std::max(1.0, z_in);
}

// One outcome of `magic_project_diagonal` into `br` (tight, canonical); z = its mass (0 if empty).
void project_branch(const MagicBatchEngine& e, const MeasureJob& job, const StateView& v, int m,
                    EngineShot& br, double& z) {
    const size_t nw = e.nw, cap = v.d + 1;
    check_rows(2, v.chi, nw, cap);  // a stored shot is real memory: this only rules out overflow
    br.w.resize(2 * v.chi), br.h.resize(v.chi * nw), br.ell.resize(v.chi * cap);
    br.A.resize(cap * nw), br.G.resize(cap * cap), br.piv.resize(cap), br.eo.resize(cap);
    size_t oc = 0, od = 0;
    z = 0.0;
    if (v.chi > 0) {
        magic_project_diagonal(
            v.chi, v.d, nw, v.w, v.h, v.ell, v.A, v.G, v.piv, v.eo, job.part(2, nw),
            job.diag_kappa, static_cast<uint8_t>(m), kMergeCancelTol, &oc, &od, br.w.data(),
            br.h.data(), br.ell.data(), br.A.data(), br.G.data(), br.piv.data(), br.eo.data(), &z
        );
    }
    br.w.resize(2 * oc), br.h.resize(oc * nw), br.ell.resize(oc * od);
    br.A.resize(od * nw), br.G.resize(od * od), br.piv.resize(od), br.eo.resize(od);
    br.a_union.assign(nw, 0ULL);
    for (size_t r = 0; r < od; ++r) {
        for (size_t i = 0; i < nw; ++i) br.a_union[i] |= br.A[r * nw + i];
    }
    br.chi = oc, br.d = od, br.canon = true;
}

// Step 3-4 of a split for the active shots: z of each, then both branches of each. -1 ok, else the
// failing shot with err set (kErrMergeNonFinite, kErrBadNorm: first shot with an invalid z,
// kErrRotation: aux = its position among the active shots).
int64_t project_active(MagicBatchEngine& e, MeasureJob& job, const SplitOut& out, size_t n_active,
                       FactorError& err) {
    MeasurePending& p = e.pend;
    p.shots.clear();
    if (p.pool.size() < 2 * n_active) p.pool.resize(2 * n_active);  // capacity is kept
    CliffordScratch sc;
    int64_t bad_z = -1;
    for (size_t s = 0; s < e.B; ++s) {
        if (out.cls[s] != kActive) continue;
        p.shots.push_back(static_cast<int32_t>(s));
        if (!pre_measure_norm(e, e.shots[s], sc, out.z[s])) {
            return set_error(err, kErrMergeNonFinite, kOpMerge), static_cast<int64_t>(s);
        }
        if (bad_z < 0 && !(std::isfinite(out.z[s]) && out.z[s] >= 0.0)) {
            bad_z = static_cast<int64_t>(s);
        }
    }
    if (bad_z >= 0) return set_error(err, kErrBadNorm, kOpQuarterTurn), bad_z;
    prepare_rotation(e, job);
    for (size_t i = 0; i < n_active; ++i) {
        const size_t s = static_cast<size_t>(p.shots[i]);
        StateView v;
        if (!rotate_for_measure(e, job, e.shots[s], out.z[s], sc, v)) {
            return set_error(err, kErrRotation, kOpQuarterTurn, static_cast<int64_t>(i)),
                   static_cast<int64_t>(s);
        }
        for (int m = 0; m < 2; ++m) {
            EngineShot& br = p.pool[2 * i + m];
            project_branch(e, job, v, m, br, out.z01[2 * s + m]);
            br.log_mass = e.shots[s].log_mass, br.atol = e.shots[s].atol;
            out.chi2[2 * s + m] = static_cast<int64_t>(br.chi);
        }
    }
    return -1;
}

// Phase 1 of a finish, per active shot in order: un-rotate (quarter turn with the reference's
// pairwise checks) unless the frame is kept, then renormalise exactly as numpy
// `weights * (1.0 / sqrt(z))` does (explicit multiply, signed zeros included) and add log z to
// the shot's log mass. -1, else the failing shot. Nothing of `e.shots` has changed yet.
int64_t finish_stage(MagicBatchEngine& e, const uint8_t* outc, const double* chosen,
                     FactorError& err) {
    MeasurePending& p = e.pend;
    const MeasureJob& job = p.job;
    const PauliView unrot{job.part(0, e.nw), job.part(3, e.nw), job.rot_kappa};
    CliffordScratch sc;
    for (size_t i = 0; i < p.n_active; ++i) {
        const size_t s = static_cast<size_t>(p.shots[i]);
        EngineShot& br = p.pool[2 * i + (outc[i] != 0 ? 1 : 0)];
        if (job.rotating && !job.keep_rot) {
            std::swap(sc.work, br);
            const bool ok = quarter_turn_step(e, sc, unrot, -1, err);
            std::swap(sc.work, br);
            if (!ok) return static_cast<int64_t>(s);
        }
        const double cz = chosen[i];
        if (!(cz > 0.0)) {  // aux 0: exactly 0 (1.0 / sqrt(0.0) raises), 1: negative or NaN
            return set_error(err, kErrZeroChosen, 0, cz == 0.0 ? 0 : 1), static_cast<int64_t>(s);
        }
        cmul_scalar(br.w.data(), br.chi, 1.0 / std::sqrt(cz), 0.0);
        br.log_mass = e.shots[s].log_mass + std::log(cz);
    }
    return -1;
}

// Phase 2: reset kicks and the commit, per active shot in order (a failing kick leaves the shots
// before it committed, as the Python loop that assigned them one by one).
int64_t finish_commit(MagicBatchEngine& e, const uint8_t* outc, FactorError& err) {
    MeasurePending& p = e.pend;
    const MeasureJob& job = p.job;
    const PauliView kick{job.part(4, e.nw), job.part(5, e.nw), job.kick_kappa};
    for (size_t i = 0; i < p.n_active; ++i) {
        const size_t s = static_cast<size_t>(p.shots[i]);
        EngineShot& br = p.pool[2 * i + (outc[i] != 0 ? 1 : 0)];
        if (job.reset && ((outc[i] != 0) != job.flip_ref) && !pauli_step(e, br, kick, err)) {
            return static_cast<int64_t>(s);
        }
        std::swap(e.shots[s], br);
    }
    return -1;
}

using NextDoubleFn = double (*)(void*);

// Pass 1 checks all active shots before any RNG draw/mutation; false => 0 draws for Python raise.
bool step_born_sample(
    const MagicBatchEngine& e, const SplitOut& out, size_t n_active, NextDoubleFn next_double,
    void* rng_state, std::vector<uint8_t>& outc, std::vector<double>& chosen, double* out_dlog
) {
    const auto& shots = e.pend.shots;
    std::vector<double> z01_c(2 * n_active);
    for (size_t i = 0; i < n_active; ++i) {
        const size_t s = static_cast<size_t>(shots[i]);
        const double z = out.z[s], z0r = out.z01[2 * s], z1r = out.z01[2 * s + 1];
        const double scale = z > 1.0 ? z : 1.0;
        if (!(std::isfinite(z0r) && std::isfinite(z1r))
            || z0r < -1e-12 * scale || z1r < -1e-12 * scale) return false;
        const double z0 = z0r > 0.0 ? z0r : 0.0, z1 = z1r > 0.0 ? z1r : 0.0, ztot = z0 + z1;
        if (!(std::fabs(ztot - z) <= e.atol * scale) || z <= 0.0 || ztot <= 0.0) return false;
        z01_c[2 * i] = z0, z01_c[2 * i + 1] = z1;
    }
    outc.resize(n_active), chosen.resize(n_active);
    for (size_t i = 0; i < n_active; ++i) {
        const size_t s = static_cast<size_t>(shots[i]);
        const double z = out.z[s], z0 = z01_c[2 * i], z1 = z01_c[2 * i + 1], ztot = z0 + z1;
        const double u = next_double(rng_state);  // one draw per active shot, pinned included
        const double p1r = z1 / ztot, p1 = p1r < 0.0 ? 0.0 : (p1r > 1.0 ? 1.0 : p1r);
        const double zf = kMergeCancelTol * z;
        const uint8_t b = (z0 <= zf) ? 1U : ((z1 <= zf) ? 0U : (u < p1 ? 1U : 0U));
        outc[i] = b, out.bit[s] = b, chosen[i] = b ? z1 : z0;
        out_dlog[s] = std::log(chosen[i] / ztot);
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Peeks, postselections and unmatched resets (S5b): the C++ half of `simulator_magic._obs_p1`,
// `postselect_z_batch` and `do_unmatched_noisy_reset_z`. All three are per-shot loops in the
// reference whose arithmetic is `_obs_p1` (peek: its own formula, with the pairwise rotation
// check of `apply_quarter_turn`) and `measure_expansion(requested_bit=...)` (postselect and the
// measuring half of an unmatched reset: pairwise checks, scalar renormalisation); the same
// kernels run in the same order, so every double comes out bit for bit. Python keeps the RNG
// draws of an unmatched reset between its two engine calls, the branch-mass logs and the
// exception texts.
// ---------------------------------------------------------------------------------------------

namespace {

struct BornScratch : CliffordScratch {
    EngineShot br[2];  // the two projected outcomes of one shot
};

inline StateView view_of(const EngineShot& s) noexcept {
    return {s.chi, s.d, s.w.data(), s.h.data(), s.A.data(), s.ell.data(), s.G.data(),
            s.eo.data(), s.piv.data()};
}

inline void export_error(const FactorError& err, int64_t* out_err, double* out_errd) noexcept {
    out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
    out_errd[0] = err.pre, out_errd[1] = err.post, out_errd[2] = err.z;
}

// The caller's shot list as a private copy; false when it is null, too long or names a shot that
// does not exist (nothing runs). May throw (out of memory).
bool copy_shots(const MagicBatchEngine& e, const int32_t* p, size_t n, std::vector<int32_t>& out) {
    if (n > kMaxElems / 4 || (n != 0 && p == nullptr)) return false;
    out.assign(p, p + n);
    for (const int32_t s : out) {
        if (s < 0 || static_cast<size_t>(s) >= e.B) return false;
    }
    return true;
}

// `_obs_p1` for one shot: chi == 0 gives 0, a trivial shot its bit, else z1 / z_in of the
// unrenormalised projection onto outcome 1 (after the rotation, if the observable has an X part).
bool peek_shot(const MagicBatchEngine& e, const MeasureJob& job, const EngineShot& sh,
               BornScratch& sc, double& p1, FactorError& err) {
    const size_t nw = e.nw;
    if (sh.chi == 0) return p1 = 0.0, true;
    if (!job.rotating && sh.d == 0 && sh.chi == 1) {
        p1 = static_cast<double>(
            ((job.kappa >> 1) ^ word_dot_parity(job.part(1, nw), sh.h.data(), nw)) & 1U);
        return true;
    }
    double z_in = 0.0;
    if (!pre_measure_norm(e, sh, sc, z_in)) {
        return set_error(err, kErrMergeNonFinite, kOpMerge), false;
    }
    if (z_in <= 0.0) return p1 = 0.0, true;
    StateView v = view_of(sh);
    if (job.rotating) {
        sc.work = sh;
        if (!quarter_turn_step(e, sc, PauliView{job.part(0, nw), job.part(3, nw), job.rot_kappa}, 1,
                               err)) {
            return false;
        }
        v = view_of(sc.work);
    } else {
        if (!padding_clear(e, PauliView{job.part(0, nw), job.part(1, nw), job.kappa})) {
            return set_error(err, kErrPadding, kOpProject), false;
        }
        if (!all_finite(sh.w.data(), 2 * sh.chi)) {
            return set_error(err, kErrNonFiniteWeights, kOpProject), false;
        }
    }
    double z1 = 0.0;
    project_branch(e, job, v, 1, sc.br[0], z1);
    p1 = std::min(1.0, std::max(0.0, z1 / z_in));
    return true;
}

// The Born-split checks of `born_split_expansion` on the raw masses of one shot, then the
// requested-outcome check of `measure_expansion`. z0 / z1 are the masses clipped at 0.
bool born_checks(const MagicBatchEngine& e, double z, const double* zr, int bit,
                 const EngineShot& br, double* z01, FactorError& err) {
    const double scale = std::max(1.0, z);
    err.z = z;
    if (!(std::isfinite(zr[0]) && std::isfinite(zr[1]))) {
        return set_error(err, kErrNonFiniteMass, kOpProject, 0, 0, zr[0], zr[1]), false;
    }
    if (zr[0] < -1e-12 * scale || zr[1] < -1e-12 * scale) {
        return set_error(err, kErrNegativeMass, kOpProject, 0, 0, zr[0], zr[1]), false;
    }
    z01[0] = zr[0] > 0.0 ? zr[0] : 0.0, z01[1] = zr[1] > 0.0 ? zr[1] : 0.0;
    if (!(std::fabs((z01[0] + z01[1]) - z) <= e.atol * scale)) {
        return set_error(err, kErrConservation, kOpProject, 0, 0, z01[0], z01[1]), false;
    }
    const double cz = bit != 0 ? z01[1] : z01[0];
    if (z <= 0.0 || cz <= kMergeCancelTol * z || br.chi == 0) {
        return set_error(err, kErrZeroPostselect, kOpProject, bit, 0, cz), false;
    }
    return true;
}

// `measure_expansion(shot, obs, requested_bit=bit, renormalize=True, keep_rotated_frame=False)`:
// the post state goes to `staged` (the shot is untouched), z01 gets the clipped masses.
bool measure_requested(const MagicBatchEngine& e, const MeasureJob& job, const EngineShot& sh,
                       int bit, BornScratch& sc, EngineShot& staged, double* z01,
                       FactorError& err) {
    const size_t nw = e.nw;
    if (!validate_observable(e, job, err)) return false;
    double z = 0.0;
    if (!pre_measure_norm(e, sh, sc, z)) {
        return set_error(err, kErrMergeNonFinite, kOpMerge), false;
    }
    if (!(std::isfinite(z) && z >= 0.0)) {
        return err.z = z, set_error(err, kErrBadNorm, kOpQuarterTurn), false;
    }
    const PauliView rot{job.part(0, nw), job.part(3, nw), job.rot_kappa};
    StateView v = view_of(sh);
    if (job.rotating) {
        sc.work = sh;
        if (!quarter_turn_step(e, sc, rot, 1, err)) return false;
        v = view_of(sc.work);
    }
    double zr[2] = {0.0, 0.0};
    for (int m = 0; m < 2; ++m) {
        project_branch(e, job, v, m, sc.br[m], zr[m]);
        sc.br[m].log_mass = sh.log_mass, sc.br[m].atol = sh.atol;
    }
    EngineShot& br = sc.br[bit != 0 ? 1 : 0];
    if (!born_checks(e, z, zr, bit, br, z01, err)) return false;
    if (job.rotating) {
        std::swap(sc.work, br);
        const bool ok = quarter_turn_step(e, sc, rot, -1, err);
        std::swap(sc.work, br);
        if (!ok) return false;
    }
    const double cz = bit != 0 ? z01[1] : z01[0];
    cmul_scalar(br.w.data(), br.chi, 1.0 / std::sqrt(cz), 0.0);
    br.log_mass = sh.log_mass + std::log(cz);
    std::swap(staged, br);
    return true;
}

// `postselect_z_batch`'s loop: the listed shots in order, dead ones skipped, each one committed
// before the next is started. Returns -1, else the list position of the shot that failed (`at`
// is the position being worked on, also when an exception leaves).
int64_t postselect_shots(MagicBatchEngine& e, const MeasureJob& job,
                         const std::vector<int32_t>& shots, const std::vector<uint8_t>& want,
                         uint8_t* done, double* z01, FactorError& err, size_t& at) {
    BornScratch sc;
    EngineShot staged;
    for (at = 0; at < shots.size(); ++at) {
        EngineShot& sh = e.shots[static_cast<size_t>(shots[at])];
        if (sh.chi == 0) continue;
        if (!measure_requested(e, job, sh, want[at] & 1, sc, staged, z01 + 2 * at, err)) {
            return static_cast<int64_t>(at);
        }
        std::swap(sh, staged);
        done[at] = 1;
    }
    return -1;
}

struct UnmatchedIn {
    const std::vector<int32_t>& shots;
    const std::vector<uint8_t>&bit, &measure, &kick;
};

// The second half of `do_unmatched_noisy_reset_z` for one qubit: per listed shot, in order, the
// projection onto the drawn bit (only where `measure`), then the reset kick (where `kick` and the
// shot is not dead). Returns -1, else the failing list position; `phase` 1 = the projection failed
// (the shot is as before), 2 = the kick failed (the projection is committed).
int64_t unmatched_shots(MagicBatchEngine& e, const MeasureJob& job, const UnmatchedIn& in,
                        FactorError& err, size_t& at, int64_t& phase) {
    const PauliView kick{job.part(4, e.nw), job.part(5, e.nw), job.kick_kappa};
    BornScratch sc;
    EngineShot staged;
    double z01[2];
    for (at = 0; at < in.shots.size(); ++at) {
        EngineShot& sh = e.shots[static_cast<size_t>(in.shots[at])];
        if (in.measure[at] != 0) {
            phase = 1;
            if (!measure_requested(e, job, sh, in.bit[at] & 1, sc, staged, z01, err)) {
                return static_cast<int64_t>(at);
            }
            std::swap(sh, staged);
        }
        if (in.kick[at] != 0 && sh.chi > 0) {
            phase = 2;
            if (!pauli_step(e, sh, kick, err)) return static_cast<int64_t>(at);
        }
    }
    return -1;
}

// Apply all non-zero rows of mask[n_q, n_b] in ascending qubit order, pulling each physical
// (q, P) back via (t_a[q], t_b[q], t_kap[q]). Reads each affected row once into local scratch.
int64_t apply_noise_half(
    MagicBatchEngine& e, const uint64_t* t_a, const uint64_t* t_b, const uint8_t* t_kap,
    const uint8_t* mask, int64_t* out_err
) {
    const size_t nq = e.n_qubits, B = e.B, nw = e.nw;
    if (B == 0) return -1;
    std::vector<uint64_t> aw(nw), bw(nw);
    std::vector<uint8_t> mr(B);
    for (size_t q = 0; q < nq; ++q) {
        const uint8_t* row = mask + q * B;
        bool any = false;
        for (size_t s = 0; s < B; ++s) { if (row[s] != 0) { any = true; break; } }
        if (!any) continue;
        std::copy(t_a + q * nw, t_a + (q + 1) * nw, aw.begin());
        std::copy(t_b + q * nw, t_b + (q + 1) * nw, bw.begin());
        const uint8_t kap = static_cast<uint8_t>(t_kap[q] & 3U);
        std::copy(row, row + B, mr.begin());
        const uint8_t par = word_dot_parity(aw.data(), bw.data(), nw);
        if (((kap - par) & 1U) != 0) {
            out_err[0] = kErrNotHermitian, out_err[1] = kOpPauli;
            return out_err[2] = par, out_err[3] = kap, 0;
        }
        if (!padding_clear(e, PauliView{aw.data(), bw.data(), kap})) {
            return out_err[0] = kErrPadding, out_err[1] = kOpPauli, 0;
        }
        for (size_t s = 0; s < B; ++s) {
            EngineShot& sh = e.shots[s];
            if (mr[s] == 0 || sh.chi == 0) continue;
            if (!all_finite(sh.w.data(), 2 * sh.chi)) {
                return out_err[0] = kErrNonFiniteWeights, out_err[1] = kOpPauli,
                       static_cast<int64_t>(s);
            }
            magic_apply_pauli_batch(
                sh.chi, sh.d, nw, sh.w.data(), sh.h.data(), sh.ell.data(), sh.A.data(),
                sh.a_union.data(), aw.data(), bw.data(), kap
            );
        }
    }
    return -1;
}

}  // namespace

extern "C" {

int32_t mbe_abi_v1() {
    return 1;
}

// v2 adds mbe_inject_cliffords (S4b-1).
int32_t mbe_abi_v2() {
    return 2;
}

// v3 adds mbe_apply_block and mbe_cmul_form (S4b-2); mbe_reset and mbe_apply_pauli_mask gain
// an out-of-memory result.
int32_t mbe_abi_v3() {
    return 3;
}

// v4 adds mbe_measure_split and mbe_measure_finish (S5a).
int32_t mbe_abi_v4() {
    return 4;
}

// v5 adds mbe_peek, mbe_postselect and mbe_unmatched_apply (S5b).
int32_t mbe_abi_v5() {
    return 5;
}

// v6 adds mbe_measure_step and mbe_apply_pauli_noise_batch.
int32_t mbe_abi_v6() { return 6; }

size_t mbe_live_engines() {
    return g_live_engines.load();
}

// nullptr: n_qubits == 0, an n_qubits whose `n_qubits + 63` would wrap or whose arrays cannot
// exist, or out of memory (a batch no vector can hold throws length_error, caught below).
void* mbe_create(size_t batch_size, size_t n_qubits, double atol) {
    if (n_qubits == 0 || n_qubits > kMaxElems) return nullptr;
    try {
        MagicBatchEngine* e = new MagicBatchEngine(batch_size, n_qubits, atol);
        g_live_engines.fetch_add(1);
        return e;
    } catch (...) {
        return nullptr;
    }
}

void mbe_destroy(void* handle) {
    if (handle == nullptr) return;
    delete engine_of(handle);
    g_live_engines.fetch_sub(1);
}

// 0 ok, 1 bad handle, 2 out of memory (nothing changed: the capacity is reserved first).
int mbe_reset(void* handle) {
    if (handle == nullptr) return 1;
    try {
        engine_of(handle)->reset();
        return 0;
    } catch (...) {
        return 2;
    }
}

// Copy-in of one shot. 0 ok, 1 invalid arguments (nothing changed), 2 out of memory (nothing
// changed). `a_union` is stored as given, so a ShotExpansion round-trips bit for bit.
int mbe_import_shot(
    void* handle, size_t b, size_t chi, size_t d, int32_t canon, double log_mass, double atol,
    const double* w_ri, const uint64_t* h_words, const uint8_t* ell, const uint64_t* A_words,
    const uint64_t* a_union, const uint8_t* G, const int32_t* pivots, const uint8_t* ell_odd
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || b >= e->B || d > e->n_qubits) return 1;
    e->pend.valid = false;
    // Every array size is a product of chi and d with small factors: refuse the ones that exceed
    // kMaxElems (or wrap) before any of them is used.
    const size_t nw = e->nw;
    size_t n_w = 0, n_h = 0, n_ell = 0, n_A = 0, n_G = 0;
    if (!mul_within(chi, 2, kMaxElems, n_w) || !mul_within(chi, nw, kMaxElems, n_h)
        || !mul_within(chi, d, kMaxElems, n_ell) || !mul_within(d, nw, kMaxElems, n_A)
        || !mul_within(d, d, kMaxElems, n_G)) {
        return 1;
    }
    try {
        EngineShot s;
        s.chi = chi, s.d = d, s.canon = canon != 0, s.log_mass = log_mass, s.atol = atol;
        const bool ok = fill_vector(s.w, w_ri, n_w) && fill_vector(s.h, h_words, n_h)
            && fill_vector(s.ell, ell, n_ell) && fill_vector(s.A, A_words, n_A)
            && fill_vector(s.a_union, a_union, nw) && fill_vector(s.G, G, n_G)
            && fill_vector(s.piv, pivots, d) && fill_vector(s.eo, ell_odd, d);
        if (!ok) return 1;
        e->shots[b] = std::move(s);
        return 0;
    } catch (...) {  // bad_alloc, or length_error for a size no vector can have
        return 2;
    }
}

// Scalars of shot `b`: out_cd = {chi, d, canonical}, out_lm_atol = {log_mass, atol}.
int mbe_shot_header(void* handle, size_t b, size_t* out_cd, double* out_lm_atol) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || b >= e->B || out_cd == nullptr || out_lm_atol == nullptr) return 1;
    const EngineShot& s = e->shots[b];
    out_cd[0] = s.chi, out_cd[1] = s.d, out_cd[2] = s.canon ? 1 : 0;
    out_lm_atol[0] = s.log_mass, out_lm_atol[1] = s.atol;
    return 0;
}

// Copy-out of the arrays of shot `b` into caller buffers sized from mbe_shot_header.
int mbe_export_shot(
    void* handle, size_t b, double* w_ri, uint64_t* h_words, uint8_t* ell, uint64_t* A_words,
    uint64_t* a_union, uint8_t* G, int32_t* pivots, uint8_t* ell_odd
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || b >= e->B) return 1;
    const EngineShot& s = e->shots[b];
    const bool ok = copy_out(s.w, w_ri) && copy_out(s.h, h_words) && copy_out(s.ell, ell)
        && copy_out(s.A, A_words) && copy_out(s.a_union, a_union) && copy_out(s.G, G)
        && copy_out(s.piv, pivots) && copy_out(s.eo, ell_odd);
    return ok ? 0 : 1;
}

// P1 on every shot with mask[s] != 0 and chi > 0 (the Python loop `apply_pauli` per masked
// shot), in shot order and in place. Returns -1 on success, else the index of the first shot
// whose weights are not finite: shots before it are already updated, it and later ones are not
// (the same partial update as the per-shot loop raising FloatingPointError). -2: bad handle or
// null argument. -3: out of memory (the kernel keeps a scratch row for d > 256): the shots before
// the failing one are updated, it and the later ones are not.
int64_t mbe_apply_pauli_mask(
    void* handle, const uint64_t* pauli_a, const uint64_t* pauli_b, uint8_t kappa,
    const uint8_t* mask
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || pauli_a == nullptr || pauli_b == nullptr || mask == nullptr) return -2;
    e->pend.valid = false;
    try {
        for (size_t s = 0; s < e->B; ++s) {
            EngineShot& sh = e->shots[s];
            if (mask[s] == 0 || sh.chi == 0) continue;
            if (!all_finite(sh.w.data(), 2 * sh.chi)) return static_cast<int64_t>(s);
            magic_apply_pauli_batch(
                sh.chi, sh.d, e->nw, sh.w.data(), sh.h.data(), sh.ell.data(), sh.A.data(),
                sh.a_union.data(), pauli_a, pauli_b, kappa
            );
        }
        return -1;
    } catch (...) {
        return -3;
    }
}

// Factor lists on shots (S4b-1). `shot_list[s]` is the list applied to shot s (< 0: skip), so one
// call serves a masked Clifford (0 / -1) and a fault channel (a branch per shot). List l has
// `has_p0[l] + n_r[l]` axes from `axis_start[l]` (p0 first): axes_a/b are [n_axes, n_words].
// `v_fwd/v_inv[l]` is the omega index of the final multiply, -1 for none. The table and the shot
// list are copied once before anything runs (the caller may change them meanwhile). Shots are
// processed in order on a staged copy and committed one by one: -1 on success, else the index of
// the first shot that failed (earlier shots are updated, it and later ones are not) with
// out_err = {kind, op, aux, aux2} and out_errd = {pre, post}. -2: bad handle, table or shot list
// (nothing ran). -3: out of memory (or a size no allocation can satisfy) at shot out_err[2]: the
// shots before it are updated, it and the later ones are not. Nothing unwinds through the ABI.
int64_t mbe_inject_cliffords(
    void* handle, size_t n_lists, size_t n_axes, const uint8_t* has_p0, const int32_t* n_r,
    const int32_t* v_fwd, const int32_t* v_inv, const int32_t* axis_start,
    const uint64_t* axes_a, const uint64_t* axes_b, const uint8_t* axes_kappa,
    const int32_t* shot_list, int32_t inverse, int64_t* out_err, double* out_errd
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || out_err == nullptr || out_errd == nullptr) return -2;
    e->pend.valid = false;
    size_t s = 0;
    try {
        CliffordJob job;
        const RawFactorTable raw = {
            n_lists, n_axes, has_p0, n_r, v_fwd, v_inv, axis_start, axes_a, axes_b, axes_kappa
        };
        if (build_job(*e, raw, shot_list, job) != 0) return -2;
        CliffordScratch sc;
        for (; s < e->B; ++s) {
            const int32_t l = job.shots[s];
            if (l < 0 || e->shots[s].chi == 0) continue;
            sc.work = e->shots[s];
            FactorError err;
            if (!factor_list_steps(*e, sc, job.lists[l], inverse != 0, err)) {
                out_err[0] = err.kind, out_err[1] = err.op;
                out_err[2] = err.aux, out_err[3] = err.aux2;
                out_errd[0] = err.pre, out_errd[1] = err.post;
                return static_cast<int64_t>(s);
            }
            std::swap(e->shots[s], sc.work);
        }
        return -1;
    } catch (...) {  // bad_alloc, length_error: never let an exception cross extern "C"
        out_err[0] = out_err[1] = out_err[3] = 0, out_err[2] = static_cast<int64_t>(s);
        return -3;
    }
}

// Number of shots with chi > 0 (-2: bad handle).
int64_t mbe_live_shots(void* handle) {
    const MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr) return -2;
    int64_t n = 0;
    for (const EngineShot& s : e->shots) n += s.chi > 0;
    return n;
}

// The grouping `mbe_apply_block` uses: out_rep[s] = lowest shot of shot s's group of identical
// states, -1 for a shot with chi == 0 (every live shot is its own group when `dedupe` is 0).
// 0 ok, -2 bad handle or null argument, -3 out of memory.
int mbe_partition(void* handle, int32_t dedupe, int64_t* out_rep) {
    const MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || out_rep == nullptr) return -2;
    try {
        for (size_t s = 0; s < e->B; ++s) out_rep[s] = -1;
        for (const auto& g : partition_shots(*e, dedupe != 0)) {
            for (const size_t s : g) out_rep[s] = static_cast<int64_t>(g[0]);
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

// One compiled block on every shot with chi > 0 (see run_block_groups). `mode` 0: the n_axes rows
// are the Walsh branches (coefficients axes_w_ri) of a pauli_sum; 1: `n_steps` steps, step i
// being kind (0 P1, 1 P2, 2 N1, 3 N2) and sign over axes rows start[i] (+ 7 more for N2, whose
// weights are axes_w_ri). (g_re, g_im) is the block's global scalar; the caller has applied it to
// the coefficients of a pauli_sum already. `max_components`: INT64_MAX means none (a negative
// bound is exceeded by every group, as in the reference). The block is copied
// once before anything runs. Returns -1 ok, else the first shot of the failing group with
// out_err = {kind, op, aux, aux2}, out_errd = {pre, post}; -2 bad handle or block (nothing ran);
// -3 out of memory (the groups before the failing one are updated). out_info = {groups, live}.
int64_t mbe_apply_block(
    void* handle, int32_t mode, size_t n_steps, size_t n_axes, double g_re, double g_im,
    const int32_t* step_kind, const int32_t* step_sign, const int32_t* step_start,
    const uint64_t* axes_a, const uint64_t* axes_b, const uint8_t* axes_kappa,
    const double* axes_w_ri, int32_t dedupe, int64_t max_components, int64_t* out_err,
    double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || out_err == nullptr || out_errd == nullptr || out_info == nullptr) return -2;
    out_info[0] = out_info[1] = 0;
    e->pend.valid = false;
    try {
        BlockJob job;
        const RawBlock raw = {
            mode, n_steps, n_axes, g_re, g_im, step_kind, step_sign, step_start, axes_a, axes_b,
            axes_kappa, axes_w_ri
        };
        if (build_block_job(*e, raw, job) != 0) return -2;
        FactorError err;
        const int64_t bad = run_block_groups(*e, job, dedupe != 0, max_components, err, out_info);
        if (bad >= 0) {
            out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
            out_errd[0] = err.pre, out_errd[1] = err.post;
        }
        return bad;
    } catch (...) {
        out_err[0] = out_err[1] = out_err[2] = out_err[3] = 0;
        return -3;
    }
}

// numpy recipes the later stages build on, exported so Python can compare them with numpy.
// These loop over a caller buffer of `n` complex values. A buffer of more than kMaxElems / 2 of
// them cannot exist, so such an `n` touches nothing (norm_sq: NaN).
void mbe_cmul_scalar(size_t n, double* w_ri, double cr, double ci) {
    if (n <= kMaxElems / 2) cmul_scalar(w_ri, n, cr, ci);
}

double mbe_norm_sq(size_t chi, const double* w_ri) {
    if (chi > kMaxElems / 2) return std::nan("");
    return norm_sq_pairwise(w_ri, chi);
}

// form 0: `w * c` (A, the scalar_in_place of two or more elements), 1: `np.complex128(c) * w`
// (B), 2: the unfused single-element in-place multiply.
void mbe_cmul_form(size_t n, double* w_ri, double cr, double ci, int32_t form) {
    if (n > kMaxElems / 2) return;
    if (form == 0) return cmul_scalar(w_ri, n, cr, ci);
    for (size_t i = 0; i < n; ++i) {
        double* p = w_ri + 2 * i;
        if (form == 1) cmul_form_b(p[0], p[1], cr, ci, p[0], p[1]);
        else cmul_plain(p[0], p[1], cr, ci, p[0], p[1]);
    }
}

// First half of a measurement of the observable (obs_a, obs_b, kappa) on every shot. Nothing of a
// shot changes, except that for a reset (`is_reset`) every trivial shot whose bit differs from
// `flip_ref` is kicked by (kick_a, kick_b, kick_kappa) at once. `keep_rot`: the caller keeps the
// rotated frame (no un-rotation in the finish). Per shot s the outputs are: out_cls[s] (0 dead,
// 1 trivial, 2 needs a draw, 3 not reached), out_bit[s] (trivial), out_z[s] (pre-norm, active),
// out_z01[2s + m] (the raw mass of outcome m), out_chi[2s + m] (the rows of that branch). All of
// them are B (or 2B) long, filled by the caller with 3 / 0 beforehand. out_info = {number of
// active shots, failing shot}. Returns 0 ok; 1 failure (out_err = {kind, op, aux, aux2},
// out_errd = {pre, post}; the shots before the failing one are classified and kicked); -2 bad
// handle or null argument; -3 out of memory.
int64_t mbe_measure_split(
    void* handle, const uint64_t* obs_a, const uint64_t* obs_b, uint8_t kappa, int32_t keep_rot,
    int32_t is_reset, int32_t flip_ref, const uint64_t* kick_a, const uint64_t* kick_b,
    uint8_t kick_kappa, uint8_t* out_cls, uint8_t* out_bit, double* out_z, double* out_z01,
    int64_t* out_chi, int64_t* out_err, double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || obs_a == nullptr || obs_b == nullptr || out_cls == nullptr
        || out_bit == nullptr || out_z == nullptr || out_z01 == nullptr || out_chi == nullptr
        || out_err == nullptr || out_errd == nullptr || out_info == nullptr
        || (is_reset != 0 && (kick_a == nullptr || kick_b == nullptr))) {
        return -2;
    }
    out_info[0] = out_info[1] = 0;
    e->pend.valid = false;
    try {
        MeasureJob job;
        build_measure_job(*e, {obs_a, obs_b, kick_a, kick_b, kappa, kick_kappa, keep_rot != 0,
                               is_reset != 0, flip_ref != 0}, job);
        const SplitOut out{out_cls, out_bit, out_z, out_z01, out_chi};
        FactorError err;
        size_t n_active = 0;
        int64_t bad = classify_and_kick(*e, job, out, n_active, err);
        out_info[0] = static_cast<int64_t>(n_active);
        if (err.kind == kErrNone && n_active > 0 && validate_observable(*e, job, err)) {
            bad = project_active(*e, job, out, n_active, err);
        }
        if (err.kind != kErrNone) {  // every failure of the steps above sets err
            out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
            out_errd[0] = err.pre, out_errd[1] = err.post, out_info[1] = bad;
            return 1;
        }
        if (n_active > 0) {
            e->pend.job = std::move(job), e->pend.n_active = n_active, e->pend.valid = true;
        }
        return 0;
    } catch (...) {
        e->pend.valid = false;
        out_err[0] = out_err[1] = out_err[2] = out_err[3] = 0;
        return -3;
    }
}

// Second half: commits outcome[i] (0 / 1) of the i-th of the `n` active shots of the last split,
// whose mass `chosen[i]` Python has checked. Un-rotates (unless the frame is kept), renormalises,
// adds log(chosen[i]) to the shot's log mass, kicks the shots of a reset whose outcome differs from
// `flip_ref`, and swaps the results in. Fails (1) without changing any shot, unless the failure
// is a kick (phase 2): then the shots before it are committed. out_info = {phase, failing shot};
// out_err / out_errd as in the split. -2: no pending split (an intervening command dropped it),
// bad handle or null argument; -4 `n` is not the number of active shots of the split (nothing
// read or changed); -3 out of memory (out_info[0] = the phase it happened in: 1 no
// shot changed, 2 the shots before the one whose kick could not run are committed). The pending
// split is used up in every case.
int64_t mbe_measure_finish(
    void* handle, size_t n, const uint8_t* outcome, const double* chosen, int64_t* out_err,
    double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || out_err == nullptr || out_errd == nullptr || out_info == nullptr) return -2;
    const bool pending = e->pend.valid;
    e->pend.valid = false;  // used up whatever happens next
    if (!pending) return -2;
    if (n != e->pend.n_active) return -4;
    if (n != 0 && (outcome == nullptr || chosen == nullptr)) return -2;
    out_info[0] = 1, out_info[1] = 0;  // phase 1 until the commit starts
    try {
        // The caller's arrays may change while this runs (the GIL is released): read each value
        // ONCE, here (n <= B), and let the two phases use only the private copies.
        const std::vector<uint8_t> outc(outcome, outcome + n);
        const std::vector<double> mass(chosen, chosen + n);
        FactorError err;
        int64_t bad = finish_stage(*e, outc.data(), mass.data(), err);
        if (bad < 0) out_info[0] = 2, bad = finish_commit(*e, outc.data(), err);
        if (bad < 0) return 0;
        out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
        out_errd[0] = err.pre, out_errd[1] = err.post, out_info[1] = bad;
        return 1;
    } catch (...) {
        out_err[0] = out_err[1] = out_err[2] = out_err[3] = 0;
        return -3;
    }
}

// Single-call measurement: split, mass check, Born draw, finish.
// rc: 0 ok (info={n_act,has_dead}); 1 split; 2 mass; 3/4 finish; -2 bad; -3 OOM (info[1]=phase).
int64_t mbe_measure_step(
    void* handle, size_t n_b, size_t n_w, const uint64_t* obs_a, const uint64_t* obs_b,
    uint8_t kappa, int32_t keep_rot, int32_t is_reset, int32_t flip_ref, const uint64_t* kick_a,
    const uint64_t* kick_b, uint8_t kick_kappa, void* next_double, void* rng_state,
    uint8_t* out_cls, uint8_t* out_bit, double* out_z, double* out_z01, int64_t* out_chi,
    double* out_dlog, int64_t* out_err, double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (!e || n_b != e->B || n_w != e->nw || !obs_a || !obs_b || !next_double || !rng_state
        || !out_err || !out_errd || !out_info || (is_reset != 0 && (!kick_a || !kick_b))
        || (n_b != 0 && (!out_cls || !out_bit || !out_z || !out_z01 || !out_chi || !out_dlog)))
        return -2;
    out_info[0] = out_info[1] = 0, e->pend.valid = false;
    int64_t oom_phase = 0;
    try {
        if (n_b != 0) std::fill_n(out_cls, n_b, kUnseen), std::fill_n(out_bit, n_b, uint8_t{0}),
            std::fill_n(out_dlog, n_b, 0.0);
        MeasureJob job;
        build_measure_job(*e, {obs_a, obs_b, kick_a, kick_b, kappa, kick_kappa, keep_rot != 0,
                               is_reset != 0, flip_ref != 0}, job);
        const SplitOut out{out_cls, out_bit, out_z, out_z01, out_chi};
        FactorError err;
        size_t n_active = 0;
        int64_t bad = classify_and_kick(*e, job, out, n_active, err);
        out_info[0] = static_cast<int64_t>(n_active);
        if (err.kind == kErrNone && n_active > 0 && validate_observable(*e, job, err))
            bad = project_active(*e, job, out, n_active, err);
        if (err.kind != kErrNone) {
            out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
            return out_errd[0] = err.pre, out_errd[1] = err.post, out_info[1] = bad, 1;
        }
        int64_t has_dead = 0;
        for (size_t s = 0; s < n_b; ++s) { if (out_cls[s] == kDead) { has_dead = 1; break; } }
        if (n_active == 0) return out_info[1] = has_dead, 0;
        std::vector<uint8_t> outc;
        std::vector<double> chosen;
        const auto nd = reinterpret_cast<NextDoubleFn>(next_double);
        if (!step_born_sample(*e, out, n_active, nd, rng_state, outc, chosen, out_dlog)) return 2;
        e->pend.job = std::move(job), e->pend.n_active = n_active, oom_phase = 1;
        bad = finish_stage(*e, outc.data(), chosen.data(), err);
        if (bad < 0) oom_phase = 2, bad = finish_commit(*e, outc.data(), err);
        if (bad < 0) return out_info[1] = has_dead, 0;
        out_err[0] = err.kind, out_err[1] = err.op, out_err[2] = err.aux, out_err[3] = err.aux2;
        return out_errd[0] = err.pre, out_errd[1] = err.post, out_info[1] = bad,
               oom_phase == 1 ? 3 : 4;
    } catch (...) {
        out_err[0] = out_err[1] = out_err[2] = out_err[3] = 0;
        return e->pend.valid = false, out_info[1] = oom_phase, -3;
    }
}


// `_obs_p1` of the observable (obs_a, obs_b, kappa) on the listed shots (a private copy of the
// list is read once): out_p1[i] for shots[i]. Nothing changes (a pending split stays). 0 ok;
// 1 failure at list position out_info[0] (out_p1[0 : out_info[0]] are valid; out_err =
// {kind, op, aux, aux2}, out_errd = {pre, post, z}); -2 bad handle, null argument or a shot
// outside the batch (nothing ran); -3 out of memory (out_info[0] = the position it was at).
int64_t mbe_peek(
    void* handle, const uint64_t* obs_a, const uint64_t* obs_b, uint8_t kappa,
    const int32_t* shots, size_t n, double* out_p1, int64_t* out_err, double* out_errd,
    int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || obs_a == nullptr || obs_b == nullptr || out_err == nullptr
        || out_errd == nullptr || out_info == nullptr || (n != 0 && out_p1 == nullptr)) {
        return -2;
    }
    out_info[0] = 0;
    size_t at = 0;
    try {
        std::vector<int32_t> list;
        if (!copy_shots(*e, shots, n, list)) return -2;
        MeasureJob job;
        build_measure_job(*e, {obs_a, obs_b, nullptr, nullptr, kappa, 0, false, false, false}, job);
        prepare_rotation(*e, job);
        BornScratch sc;
        FactorError err;
        for (; at < n; ++at) {
            if (!peek_shot(*e, job, e->shots[static_cast<size_t>(list[at])], sc, out_p1[at], err)) {
                export_error(err, out_err, out_errd), out_info[0] = static_cast<int64_t>(at);
                return 1;
            }
        }
        out_info[0] = static_cast<int64_t>(n);
        return 0;
    } catch (...) {
        out_info[0] = static_cast<int64_t>(at);
        return -3;
    }
}

// `postselect_z_batch` on the listed shots: shot shots[i] is projected onto bit desired[i] & 1
// (dead shots skipped), in list order, each committed before the next; the first failure stops
// the loop with the shots before it committed and the later ones untouched. out_done[i] = 1 when
// shot i was committed, out_z01[2i], [2i + 1] = its clipped masses z0, z1. Return and
// out_info[0] (the failing / out-of-memory list position) as for mbe_peek. Any pending split is
// dropped.
int64_t mbe_postselect(
    void* handle, const uint64_t* obs_a, const uint64_t* obs_b, uint8_t kappa,
    const int32_t* shots, size_t n, const uint8_t* desired, uint8_t* out_done, double* out_z01,
    int64_t* out_err, double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || obs_a == nullptr || obs_b == nullptr || out_err == nullptr
        || out_errd == nullptr || out_info == nullptr
        || (n != 0 && (desired == nullptr || out_done == nullptr || out_z01 == nullptr))) {
        return -2;
    }
    out_info[0] = 0;
    e->pend.valid = false;
    size_t at = 0;
    try {
        std::vector<int32_t> list;
        if (!copy_shots(*e, shots, n, list)) return -2;
        const std::vector<uint8_t> want(desired, desired + n);  // read once
        MeasureJob job;
        build_measure_job(*e, {obs_a, obs_b, nullptr, nullptr, kappa, 0, false, false, false}, job);
        prepare_rotation(*e, job);
        FactorError err;
        if (postselect_shots(*e, job, list, want, out_done, out_z01, err, at) < 0) return 0;
        export_error(err, out_err, out_errd), out_info[0] = static_cast<int64_t>(at);
        return 1;
    } catch (...) {
        out_info[0] = static_cast<int64_t>(at);
        return -3;
    }
}

// Second half of an unmatched noisy reset on the listed shots (the first half is mbe_peek plus
// the draws): where measure[i] != 0, shot shots[i] is projected onto bit[i] & 1 (as in
// mbe_postselect, no branch mass involved); then, where kick[i] != 0 and the shot is not dead, it
// is kicked by (kick_a, kick_b, kick_kappa). In list order, each shot finished before the next.
// 0 ok; 1 failure at list position out_info[0] with out_info[1] = phase (1: the projection
// failed, the shot is as before; 2: the kick failed, the projection is committed), earlier shots
// finished, later ones untouched; -2 / -3 as for mbe_peek (out_info[1] = phase when -3). Any
// pending split is dropped.
int64_t mbe_unmatched_apply(
    void* handle, const uint64_t* obs_a, const uint64_t* obs_b, uint8_t kappa,
    const uint64_t* kick_a, const uint64_t* kick_b, uint8_t kick_kappa, const int32_t* shots,
    size_t n, const uint8_t* bit, const uint8_t* measure, const uint8_t* kick, int64_t* out_err,
    double* out_errd, int64_t* out_info
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || obs_a == nullptr || obs_b == nullptr || kick_a == nullptr
        || kick_b == nullptr || out_err == nullptr || out_errd == nullptr || out_info == nullptr
        || (n != 0 && (bit == nullptr || measure == nullptr || kick == nullptr))) {
        return -2;
    }
    out_info[0] = out_info[1] = 0;
    e->pend.valid = false;
    size_t at = 0;
    int64_t phase = 0;
    try {
        std::vector<int32_t> list;
        if (!copy_shots(*e, shots, n, list)) return -2;
        const std::vector<uint8_t> bits(bit, bit + n), meas(measure, measure + n),
            kicks(kick, kick + n);  // read once
        MeasureJob job;
        build_measure_job(
            *e, {obs_a, obs_b, kick_a, kick_b, kappa, kick_kappa, false, true, false}, job
        );
        prepare_rotation(*e, job);
        FactorError err;
        const UnmatchedIn in{list, bits, meas, kicks};
        if (unmatched_shots(*e, job, in, err, at, phase) < 0) return 0;
        export_error(err, out_err, out_errd);
        out_info[0] = static_cast<int64_t>(at), out_info[1] = phase;
        return 1;
    } catch (...) {
        out_info[0] = static_cast<int64_t>(at), out_info[1] = phase;
        return -3;
    }
}

// Batch Pauli noise: ascending X qubits 0..n_q-1 then ascending Z qubits 0..n_q-1, shot order.
// -1 ok; >=0 shot/error index (out_err set); -2 bad args/dims/null; -3 OOM.
int64_t mbe_apply_pauli_noise_batch(
    void* handle, size_t n_q, size_t n_b, size_t n_w,
    const uint64_t* x2x, const uint64_t* x2z, const uint8_t* x_kap,
    const uint64_t* z2x, const uint64_t* z2z, const uint8_t* z_kap,
    const uint8_t* xm, const uint8_t* zm, int64_t* out_err
) {
    MagicBatchEngine* e = engine_of(handle);
    if (e == nullptr || out_err == nullptr || n_q != e->n_qubits || n_b != e->B || n_w != e->nw
        || x2x == nullptr || x2z == nullptr || x_kap == nullptr
        || z2x == nullptr || z2z == nullptr || z_kap == nullptr
        || (n_b != 0 && (xm == nullptr || zm == nullptr))) {
        return -2;
    }
    out_err[0] = out_err[1] = out_err[2] = out_err[3] = 0, e->pend.valid = false;
    try {
        int64_t rc = apply_noise_half(*e, x2x, x2z, x_kap, xm, out_err);
        if (rc < 0) rc = apply_noise_half(*e, z2x, z2z, z_kap, zm, out_err);
        return rc;
    } catch (...) {
        return -3;
    }
}

}  // extern "C"
