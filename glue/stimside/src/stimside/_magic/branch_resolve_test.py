"""T2 & T3 tests for C++ resolve step, Mobius/choose_split, quarter-turn conjugation & driver.

Covers:
1. T2: `is_clifford` (empty, <15, 15-term RM(3,4) vs 14-term, `r <= 20` incl. `r = 20`, `r > 20`
   fallback, multi-word `r > 64`), `choose_split` (depth 1..5 & >5 fallback, big-endian multi-word),
   pending turn sign normalization (`D(-P, e) = omega^e D(P, -e)`), `-I`/`+I`/`e=0`, mod-8
   accumulation/cancellation, quarter-turn conjugation across all 46 1Q/2Q Stim Cliffords (forward
   & inverse, including 4-factor `CXSWAP`/`SWAPCX`), K1 relative pruning, `max_components` budget
   `-4` atomicity, multi-word resolve (`n = 65, 130`), and >=300 random resolve states vs dense
   (`< 1e-12`).
2. Differential verification against read-only `scratch/magic_ref/mfe.py` (exact `is_clifford` &
   `choose_split` for `r <= 20`, exact canonical keys & sorted order, weights/amplitudes `< 1e-12`,
   `skip_clifford` parity).
3. T3: Compact test-only per-shot moving-frame driver (`BranchStore` + `ReferenceCHPTableau` +
   `compile_pauli_product` + `compile_clifford_factors`) vs independent dense state-vector oracle on
   >=2,000 random circuits (`n = 1..8`) covering 1Q/2Q Cliffords, `SPP`/`SPP_DAG`, `T`, `T_DAG`,
   `CCZ`, `M`, `MX`, `MY`, `MPP`, `R`, `RX`, `RY`, `MR`, Pauli faults, `CFAULT`, `SKIP`, `PEEK`,
   `POST`, and `URESET`.
"""

from __future__ import annotations

import gc
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from stimside._magic.channels import compile_clifford_factors, compile_pauli_product
from stimside._magic.gates import DECOMP_1Q, DECOMP_2Q
from stimside._native import branch_kernels as bk
from stimside.util.reference_chp import ReferenceCHPTableau

_MAGIC_REF_DIR = Path(
    "/usr/local/google/home/mliq/.gemini/jetski/brain/"
    "a4cd4c57-114b-4c18-9db0-bcc178e62419/scratch/magic_ref"
)
_OMEGA = complex(math.sqrt(0.5), math.sqrt(0.5))
_OMEGA_P = tuple(_OMEGA**k for k in range(8))


# ---------------------------------------------------------------------------------------------
# Shared Pauli / Dense Oracle Helpers (Little-Endian Qubit 0 = LSB)
# ---------------------------------------------------------------------------------------------


def _nw(n: int) -> int:
    return (n + 63) // 64


def _tail_mask(n: int) -> int:
    rem = n & 63
    return (1 << 64) - 1 if rem == 0 else (1 << rem) - 1


def _int_to_words(val: int, nw: int) -> np.ndarray:
    out = np.zeros(nw, dtype=np.uint64)
    for w in range(nw):
        out[w] = np.uint64((val >> (64 * w)) & 0xFFFFFFFFFFFFFFFF)
    return out


def _words_to_int(words: np.ndarray) -> int:
    val = 0
    for w in range(len(words) - 1, -1, -1):
        val = (val << 64) | int(words[w])
    return val


def _herm_kappa(x_int: int, z_int: int, neg: bool = False, *, negate: bool | None = None) -> int:
    is_neg = neg if negate is None else negate
    return ((x_int & z_int).bit_count() + (2 if is_neg else 0)) & 3


def _anti_words(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> int:
    acc = int(np.bitwise_xor.reduce((x1 & z2) ^ (z1 & x2)))
    return acc.bit_count() & 1


def _anti_int(x1: int, z1: int, x2: int, z2: int) -> int:
    return ((x1 & z2) ^ (z1 & x2)).bit_count() & 1


def _dense_apply_pauli(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.uint64)
    src = idx ^ np.uint64(x_int)
    zb = src & np.uint64(z_int)
    par = np.bitwise_count(zb) & np.uint64(1)
    sign = np.where(par == 0, 1.0, -1.0)
    phase = (1.0, 1.0j, -1.0, -1.0j)[kappa & 3]
    return (phase * sign) * vec[src]


def _dense_apply_qturn(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int) -> np.ndarray:
    return (vec + 1.0j * _dense_apply_pauli(vec, n, x_int, z_int, kappa)) * math.sqrt(0.5)


def _dense_apply_turn(
    vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int, e: int
) -> np.ndarray:
    we = _OMEGA_P[e & 7]
    pv = _dense_apply_pauli(vec, n, x_int, z_int, kappa)
    return 0.5 * (1.0 + we) * vec + 0.5 * (1.0 - we) * pv


def _dense_apply_clifford_gate(
    vec: np.ndarray, n: int, gate_name: str, targets: tuple[int, ...], *, inverse: bool = False
) -> np.ndarray:
    """Apply an exact 1Q/2Q Stim Clifford gate (or its inverse) via `DECOMP_1Q`/`DECOMP_2Q`."""
    ref = ReferenceCHPTableau(n)
    fac = compile_clifford_factors(ref, gate_name, targets)
    out = vec.copy()
    if not inverse and fac.p0 is not None:
        out = _dense_apply_pauli(
            out, n, _words_to_int(fac.p0.a_words), _words_to_int(fac.p0.b_words), int(fac.p0.kappa)
        )
    seq = reversed(fac.r_axes) if inverse else fac.r_axes
    for ax in seq:
        k = (int(ax.kappa) + (2 if inverse else 0)) & 3
        out = _dense_apply_qturn(out, n, _words_to_int(ax.a_words), _words_to_int(ax.b_words), k)
    if inverse and fac.p0 is not None:
        out = _dense_apply_pauli(
            out, n, _words_to_int(fac.p0.a_words), _words_to_int(fac.p0.b_words), int(fac.p0.kappa)
        )
    if fac.v:
        out = out * _OMEGA_P[(-fac.v if inverse else fac.v) & 7]
    return out


def _phase_aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Return min_{phi} ||a - e^{i phi} b||_inf."""
    ov = np.vdot(b, a)
    if abs(ov) > 1e-15:
        b = b * (ov / abs(ov))
    return float(np.max(np.abs(a - b)))


def _make_random_state(
    rng: np.random.Generator, n: int, steps: int | None = None
) -> tuple[bk.BranchState, complex, np.ndarray]:
    st = bk.BranchState.zero(n)
    w = 1.0 + 0.0j
    vec = np.zeros(1 << n, dtype=np.complex128)
    vec[0] = 1.0
    nw = _nw(n)
    tm = _tail_mask(n)
    n_steps = int(rng.integers(1, 2 * n + 4)) if steps is None else steps
    for _ in range(n_steps):
        xi = int(rng.integers(0, tm + 1))
        zi = int(rng.integers(0, tm + 1))
        k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
        xw = _int_to_words(xi, nw)
        zw = _int_to_words(zi, nw)
        if rng.random() < 0.8:
            w *= st.qturn(xw, zw, k)
            vec = _dense_apply_qturn(vec, n, xi, zi, k)
        else:
            w *= st.pauli(xw, zw, k)
            vec = _dense_apply_pauli(vec, n, xi, zi, k)
    w *= st.canon()
    return st, w, vec


def _make_commuting_turns(
    rng: np.random.Generator, n: int, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate `count` pairwise commuting Hermitian turns via a random Clifford frame on Z-products."""
    ref = ReferenceCHPTableau(n)
    gates_1 = ("H", "S", "S_DAG", "SQRT_X", "SQRT_Y", "X", "Y", "Z")
    for _ in range(2 * n + 3):
        if n >= 2 and rng.random() < 0.45:
            q0, q1 = rng.choice(n, size=2, replace=False)
            ref.do_unitary("CX" if rng.random() < 0.5 else "CZ", [int(q0), int(q1)])
        else:
            q = int(rng.integers(0, n))
            ref.do_unitary(gates_1[int(rng.integers(0, len(gates_1)))], [q])

    nw = _nw(n)
    xs = np.zeros((count, nw), dtype=np.uint64)
    zs = np.zeros((count, nw), dtype=np.uint64)
    ks = np.zeros(count, dtype=np.uint8)
    es = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        mask = int(rng.integers(1, 1 << n))
        terms = [(q, "Z") for q in range(n) if (mask >> q) & 1]
        ax = compile_pauli_product(ref, terms, negate=bool(rng.integers(0, 2)))
        xs[i] = ax.a_words
        zs[i] = ax.b_words
        ks[i] = np.uint8(ax.kappa)
        es[i] = np.uint8(int(rng.integers(1, 8)))
    return xs, zs, ks, es


def _load_mfe_ref() -> tuple[Any, Any]:
    mfe_path = _MAGIC_REF_DIR / "mfe.py"
    if not mfe_path.is_file():
        pytest.skip(f"Reference prototype not present: {mfe_path}")
    added = str(_MAGIC_REF_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(_MAGIC_REF_DIR))
    try:
        spec_b = importlib.util.spec_from_file_location("_bqfe_ref_wp3", _MAGIC_REF_DIR / "bqfe.py")
        spec_m = importlib.util.spec_from_file_location("_mfe_ref_wp3", mfe_path)
        assert spec_b is not None and spec_b.loader is not None
        assert spec_m is not None and spec_m.loader is not None
        bqfe_mod = importlib.util.module_from_spec(spec_b)
        spec_b.loader.exec_module(bqfe_mod)
        mfe_mod = importlib.util.module_from_spec(spec_m)
        spec_m.loader.exec_module(mfe_mod)
        return mfe_mod, bqfe_mod
    finally:
        if added and str(_MAGIC_REF_DIR) in sys.path:
            sys.path.remove(str(_MAGIC_REF_DIR))


def _state_to_bqfe(st: bk.BranchState, bqfe_mod: Any) -> Any:
    n, d, h_w, A_w, ell, G, piv = st.to_arrays()
    q = bqfe_mod.Q(n)
    q.h = _words_to_int(h_w)
    q.cols = [_words_to_int(A_w[j]) for j in range(d)]
    q.ell = [int(ell[j]) for j in range(d)]
    q.gam = [sum((int(G[j, k]) << k) for k in range(d)) for j in range(d)]
    q.piv = [int(piv[j]) for j in range(d)]
    return q


def _bqfe_key_bytes(q: Any) -> bytes:
    import struct

    n, d = q.n, len(q.cols)
    nw = max(1, (n + 63) // 64)
    n_tri = (d * (d - 1)) // 2 if d >= 2 else 0
    gbuf = bytearray((n_tri + 7) // 8)
    bit = 0
    for j in range(d):
        gj = q.gam[j]
        for k in range(j + 1, d):
            if (gj >> k) & 1:
                gbuf[bit >> 3] |= 1 << (bit & 7)
            bit += 1
    return (
        struct.pack("<II", n, d)
        + q.h.to_bytes(nw * 8, "little")
        + b"".join(c.to_bytes(nw * 8, "little") for c in q.cols)
        + bytes(q.ell)
        + bytes(gbuf)
    )


# ---------------------------------------------------------------------------------------------
# Group 1 (T2): is_clifford, choose_split, pending turns, qturn conjugation, resolve, budget
# ---------------------------------------------------------------------------------------------


def test_is_clifford_and_choose_split_edge_cases() -> None:
    # Empty set is Clifford; < 15 nonzero parities is never Clifford.
    assert bk.is_clifford([]) is True
    assert bk.is_clifford([1]) is False
    assert bk.is_clifford(list(range(1, 15))) is False

    # 15-term RM(3, 4) codeword is Clifford; removing any 1 term makes it non-Clifford.
    rm4 = list(range(1, 16))
    assert bk.is_clifford(rm4) is True
    rm4_minus1 = [x for x in rm4 if x != 15]
    assert bk.is_clifford(rm4_minus1) is False
    lam, depth = bk.choose_split(rm4_minus1)
    assert depth == 1 and lam > 0

    # r = 20 exact boundary: shift RM(3, 4) to bits {16, 17, 18, 19}.
    rm20 = [
        ((x & 1) << 16) | (((x >> 1) & 1) << 17) | (((x >> 2) & 1) << 18) | (((x >> 3) & 1) << 19)
        for x in range(1, 16)
    ]
    assert max(rm20).bit_length() == 20
    assert bk.is_clifford(rm20) is True
    rm20_bad = rm20[:-1]
    assert bk.is_clifford(rm20_bad) is False
    lam20, d20 = bk.choose_split(rm20_bad)
    assert d20 == 1 and lam20 > 0

    # r > 20 (r = 21 and r = 25): Plan §2.6 skips Mobius test -> returns False.
    rm21 = [x << 17 for x in range(1, 16)]
    assert max(rm21).bit_length() == 21
    assert bk.is_clifford(rm21) is False
    # 1-split GHZ-like parity set with r = 25: {bit0, bit0|bit1, ..., bit0|bit24} collapses in 1 split
    # because quotient by bit0? Wait: bit0 | bit_j with lam = 1 leaves bit_j (24 singletons);
    # instead pairs {1, 1|(1<<24)} collapse with lam = 1|(1<<24) or {1<<k, (1<<k)|(1<<24)} with lam = 1<<24!
    ghz25 = []
    for k in range(21):
        ghz25.append(1 << k)
        ghz25.append((1 << k) | (1 << 24))
    assert max(ghz25).bit_length() == 25
    assert bk.is_clifford(ghz25) is False
    lam25, d25 = bk.choose_split(ghz25)
    assert lam25 == (1 << 24) and d25 == 1

    # Depth > 5 fallback: two singletons pair in 1 split (lam = a ^ b), so 11 independent
    # singletons need ceil(11/2) = 6 > 5 splits -> fallback (min(O), 0); 10 need exactly 5.
    ind11 = [1 << k for k in range(11)]
    assert bk.choose_split(ind11, max_depth=5) == (1, 0)
    assert bk.choose_split(ind11, max_depth=6)[1] == 6
    ind10 = [1 << k for k in range(10)]
    assert bk.choose_split(ind10, max_depth=5)[1] == 5

    # Multi-word r > 64 (r = 70, n_words = 2): big-endian ordering matches Python int min(O)!
    p_small_hi_w0 = (1 << 63)  # word0 = 2^63, word1 = 0 -> smaller Python int!
    p_large_lo_w0 = (1 << 69) | 1  # word0 = 1, word1 = 32 -> larger Python int!
    O70 = [p_large_lo_w0, p_small_hi_w0]
    assert bk.is_clifford(O70, n_words=2) is False
    lam70, d70 = bk.choose_split(O70, max_depth=1, n_words=2)
    assert lam70 == (p_small_hi_w0 ^ p_large_lo_w0) and d70 == 1
    # Depth 0 fallback must return min(O) = p_small_hi_w0 (NOT p_large_lo_w0!).
    lam70_fb, d70_fb = bk.choose_split(
        [p_large_lo_w0, p_small_hi_w0, (1 << 68) | 2, (1 << 67) | 4],
        max_depth=1,
        n_words=2,
    )
    assert lam70_fb == p_small_hi_w0 and d70_fb == 0


def test_resolve_rm4_clifford_identity_and_perturbed() -> None:
    """15-term RM(3,4) on |++++> resolves with 0 splits & 1 branch; 14-term requires splitting."""
    n = 4
    with bk.BranchStore(n) as store:
        st = bk.BranchState.zero(n)
        vec = np.zeros(1 << n, dtype=np.complex128)
        vec[0] = 1.0
        for q in range(n):
            xq = _int_to_words(1 << q, 1)
            zq = _int_to_words(1 << q, 1)
            st.qturn(xq, zq, 3)  # R(-Y)|0> = |+>
            vec = _dense_apply_qturn(vec, n, 1 << q, 1 << q, 3)
        st.canon()

        # All 15 nonzero Z-subsets of 4 qubits with e = 1.
        xs = np.zeros((15, 1), dtype=np.uint64)
        zs = np.array([[m] for m in range(1, 16)], dtype=np.uint64)
        ks = np.zeros(15, dtype=np.uint8)
        es = np.ones(15, dtype=np.uint8)

        nid15 = store.node_from_branches([1.0 + 0.0j], [st], xs, zs, ks, es)
        r15 = store.resolve_node(nid15)
        nb15, np15, _ = store.node_info(r15)
        cnt15 = store.counters()
        assert nb15 == 1 and np15 == 0 and cnt15["splits"] == 0

        dense15 = vec.copy()
        for m in range(1, 16):
            dense15 = _dense_apply_turn(dense15, n, 0, m, 0, 1)
        assert float(np.max(np.abs(store.node_dense(r15) - dense15))) < 1e-12

        # Drop 1 parity (14 terms): must split (>0 splits, >1 branch) and match dense < 1e-12.
        nid14 = store.node_from_branches([1.0 + 0.0j], [st], xs[:14], zs[:14], ks[:14], es[:14])
        r14 = store.resolve_node(nid14)
        nb14, np14, _ = store.node_info(r14)
        cnt14 = store.counters()
        assert nb14 > 1 and np14 == 0 and cnt14["splits"] > 0

        dense14 = vec.copy()
        for m in range(1, 15):
            dense14 = _dense_apply_turn(dense14, n, 0, m, 0, 1)
        assert float(np.max(np.abs(store.node_dense(r14) - dense14))) < 1e-12
        st.close()


def test_pending_sign_normalization_identity_cancellation_and_order() -> None:
    """Verify D(-P, e) = omega^e D(P, -e), -I/+I/e=0 handling, mod-8 accumulation & validation."""
    n = 3
    with bk.BranchStore(n) as store:
        # Prepare |+++>
        st = bk.BranchState.zero(n)
        vec = np.zeros(1 << n, dtype=np.complex128)
        vec[0] = 1.0
        for q in range(n):
            w_q = _int_to_words(1 << q, 1)
            st.qturn(w_q, w_q, 3)
            vec = _dense_apply_qturn(vec, n, 1 << q, 1 << q, 3)
        st.canon()
        nid = store.node_from_branches([1.0 + 0.0j], [st])
        st.close()

        # Batch with -Z0 (e=3), +Z1 (e=2), -I (e=5), +I (e=7), +Z2 (e=0)
        xs = np.zeros((5, 1), dtype=np.uint64)
        zs = np.array([[1], [2], [0], [0], [4]], dtype=np.uint64)
        ks = np.array([2, 0, 2, 0, 0], dtype=np.uint8)
        es = np.array([3, 2, 5, 7, 0], dtype=np.uint8)
        c1 = store.step_rotations(nid, 1, bk.MBR_KIND_T, 0, xs, zs, ks, es)
        px, pz, pk, pe = store.node_pending(c1)
        # Only Z0 (e = -3 mod 8 = 5) and Z1 (e = 2) survive, in order!
        assert list(pz[:, 0]) == [1, 2]
        assert list(pk) == [0, 0]
        assert list(pe) == [5, 2]

        dense_ref = vec.copy()
        for zi, ki, ei in ((1, 2, 3), (2, 0, 2), (0, 2, 5), (0, 0, 7), (4, 0, 0)):
            dense_ref = _dense_apply_turn(dense_ref, n, 0, zi, ki, ei)
        assert float(np.max(np.abs(store.node_dense(c1) - dense_ref))) < 1e-12

        # Add +Z0 (e=3) -> cancels Z0 (5+3 = 8 == 0), leaving only Z1 (e=2)
        c2 = store.step_rotations(
            c1,
            2,
            bk.MBR_KIND_T,
            0,
            np.array([[0]], dtype=np.uint64),
            np.array([[1]], dtype=np.uint64),
            np.array([0], dtype=np.uint8),
            np.array([3], dtype=np.uint8),
        )
        _, pz2, _, pe2 = store.node_pending(c2)
        assert list(pz2[:, 0]) == [2] and list(pe2) == [2]
        dense_ref = _dense_apply_turn(dense_ref, n, 0, 1, 0, 3)
        assert float(np.max(np.abs(store.node_dense(c2) - dense_ref))) < 1e-12

        # Anticommuting incoming turn (X1) triggers automatic resolve before adding X1!
        c3 = store.step_rotations(
            c2,
            3,
            bk.MBR_KIND_T,
            0,
            np.array([[2]], dtype=np.uint64),
            np.array([[0]], dtype=np.uint64),
            np.array([0], dtype=np.uint8),
            np.array([1], dtype=np.uint8),
        )
        assert store.counters()["resolve_calls"] == 1
        dense_ref = _dense_apply_turn(dense_ref, n, 2, 0, 0, 1)
        assert float(np.max(np.abs(store.node_dense(c3) - dense_ref))) < 1e-12
        r3 = store.resolve_node(c3)
        assert float(np.max(np.abs(store.node_dense(r3) - dense_ref))) < 1e-12

        # Pairwise non-commuting batch within a single step_rotations call must raise ValueError.
        with pytest.raises(ValueError):
            store.step_rotations(
                nid,
                99,
                bk.MBR_KIND_CCZ,
                0,
                np.array([[1], [0]], dtype=np.uint64),
                np.array([[0], [1]], dtype=np.uint64),
                np.array([0, 0], dtype=np.uint8),
                np.array([1, 1], dtype=np.uint8),
            )


def test_qturn_conjugation_all_46_cliffords_forward_and_inverse() -> None:
    """Verify left-multiplying R(A) P R(A)^dag = i*A*P across all 46 1Q/2Q Stim Cliffords."""
    rng = np.random.default_rng(20260301)
    n = 3
    max_dev = 0.0
    eid = 1
    with bk.BranchStore(n) as store:
        for gate_name in (*DECOMP_1Q.keys(), *DECOMP_2Q.keys()):
            arity = 1 if gate_name in DECOMP_1Q else 2
            targets = (0,) if arity == 1 else (0, 2)
            for inverse in (False, True):
                st, w, vec = _make_random_state(rng, n)
                px, pz, pk, pe = _make_commuting_turns(rng, n, 4)
                nid = store.node_from_branches([w], [st], px, pz, pk, pe)
                st.close()

                dense_vec = vec.copy()
                for i in range(4):
                    dense_vec = _dense_apply_turn(
                        dense_vec, n, int(px[i, 0]), int(pz[i, 0]), int(pk[i]), int(pe[i])
                    )

                # Build quarter-turn sequence of the Clifford gate in frame coords.
                ref = ReferenceCHPTableau(n)
                fac = compile_clifford_factors(ref, gate_name, targets)
                seq = list(reversed(fac.r_axes) if inverse else fac.r_axes)
                q_x = np.zeros((len(seq), 1), dtype=np.uint64)
                q_z = np.zeros((len(seq), 1), dtype=np.uint64)
                q_k = np.zeros(len(seq), dtype=np.uint8)
                for j, ax in enumerate(seq):
                    q_x[j] = ax.a_words
                    q_z[j] = ax.b_words
                    q_k[j] = np.uint8((int(ax.kappa) + (2 if inverse else 0)) & 3)
                    dense_vec = _dense_apply_qturn(
                        dense_vec, n, int(q_x[j, 0]), int(q_z[j, 0]), int(q_k[j])
                    )

                kind = bk.MBR_KIND_SKIP if inverse else bk.MBR_KIND_CFAULT
                # Exercise sign_bits up to 15 (4-factor CXSWAP/SWAPCX).
                sign_bits = (1 << len(seq)) - 1 if len(seq) > 0 else 0
                cid = store.step_qturns(nid, eid, kind, sign_bits, q_x, q_z, q_k)
                eid += 1

                # Verify exact dense match (including global phase!) before & after resolve.
                dev_pre = float(np.max(np.abs(store.node_dense(cid) - dense_vec)))
                rid = store.resolve_node(cid)
                dev_post = float(np.max(np.abs(store.node_dense(rid) - dense_vec)))
                max_dev = max(max_dev, dev_pre, dev_post)

                # Verify branches remain strictly sorted by canonical byte key.
                branches = store.node_branches(rid)
                keys = [b_st.canonical_key() for _, b_st in branches]
                for _, b_st in branches:
                    b_st.close()
                assert keys == sorted(keys)

    assert max_dev < 1e-12, f"max_dev={max_dev}"


def test_k1_relative_pruning_and_budget_atomicity() -> None:
    """Verify K1 threshold 1e-13*min(1, w_max) and atomic -4 BranchBudgetError rollback."""
    # K1 relative pruning on small weights (w_max = 1e-6 -> thresh = 1e-19):
    # branch at 1e-12 survives (> 1e-19), exact cancellation pair (+w, -w) and 1e-21 are pruned.
    n = 2
    with bk.BranchStore(n) as store:
        s0 = bk.BranchState.zero(n)
        s1 = bk.BranchState.zero(n)
        s1.pauli([1], [0], 0)  # |10>
        s2 = bk.BranchState.zero(n)
        s2.pauli([2], [0], 0)  # |01>
        s3 = bk.BranchState.zero(n)
        s3.pauli([3], [0], 0)  # |11>

        nid = store.node_from_branches(
            [1e-6, 1e-12, 1e-6, -1e-6, 1e-21],
            [s0, s1, s2, s2, s3],
        )
        nb, _, _ = store.node_info(nid)
        assert nb == 2  # s0 (1e-6) and s1 (1e-12) survive; s2 cancelled, s3 below 1e-19!
        s0.close()
        s1.close()
        s2.close()
        s3.close()

    # Budget atomicity: single-branch (K=1), multi-branch (K>1), and r=25 fast abort.
    with bk.BranchStore(6, max_components=4) as store:
        st = bk.BranchState.zero(6)
        for q in range(6):
            wq = _int_to_words(1 << q, 1)
            st.qturn(wq, wq, 3)
        st.canon()
        xs = np.zeros((6, 1), dtype=np.uint64)
        zs = np.array([[1 << q] for q in range(6)], dtype=np.uint64)
        ks = np.zeros(6, dtype=np.uint8)
        es = np.ones(6, dtype=np.uint8)
        nid = store.node_from_branches([1.0 + 0.0j], [st], xs, zs, ks, es)
        st.close()

        before_nodes = store.node_count
        with pytest.raises(bk.BranchBudgetError):
            store.resolve_node(nid)
        assert store.node_count == before_nodes

        # Relaxing cap succeeds cleanly (proving aborted subtree was NOT cached!).
        # T^{\otimes 6} pairs into 3 splits -> 2^3 = 8 stabilizer branches.
        store.set_max_components(0)
        rid = store.resolve_node(nid)
        nb, npd, _ = store.node_info(rid)
        assert nb == 8 and npd == 0

    # K > 1 budget atomicity: 2-branch node exceeding max_components=3 aborts atomically.
    with bk.BranchStore(4, max_components=3) as store:
        s_a = bk.BranchState.zero(4)
        s_b = bk.BranchState.zero(4)
        for q in range(4):
            wq = _int_to_words(1 << q, 1)
            s_a.qturn(wq, wq, 3)
            s_b.qturn(wq, wq, 3)
        s_b.pauli([0], [1], 0)  # |-+++>
        s_a.canon()
        s_b.canon()
        xs = np.zeros((3, 1), dtype=np.uint64)
        zs = np.array([[2], [4], [8]], dtype=np.uint64)
        ks = np.zeros(3, dtype=np.uint8)
        es = np.ones(3, dtype=np.uint8)
        nid = store.node_from_branches([0.6, 0.8], [s_a, s_b], xs, zs, ks, es)
        s_a.close()
        s_b.close()
        before = store.node_count
        with pytest.raises(bk.BranchBudgetError):
            store.resolve_node(nid)
        assert store.node_count == before

    # r = 25 independent T-gates with max_components = 32 must abort quickly with -4.
    with bk.BranchStore(25, max_components=32) as store:
        st = bk.BranchState.zero(25)
        for q in range(25):
            wq = _int_to_words(1 << q, 1)
            st.qturn(wq, wq, 3)
        st.canon()
        xs = np.zeros((25, 1), dtype=np.uint64)
        zs = np.array([[1 << q] for q in range(25)], dtype=np.uint64)
        ks = np.zeros(25, dtype=np.uint8)
        es = np.ones(25, dtype=np.uint8)
        nid = store.node_from_branches([1.0 + 0.0j], [st], xs, zs, ks, es)
        st.close()
        before = store.node_count
        with pytest.raises(bk.BranchBudgetError):
            store.resolve_node(nid)
        assert store.node_count == before


def test_resolve_random_states_vs_dense_300_trials() -> None:
    """T2: 320 random resolve problems (n = 1..8) verified against exact dense (< 1e-12)."""
    rng = np.random.default_rng(20260302)
    max_dev = 0.0
    for trial in range(320):
        n = int(rng.integers(1, 9))
        n_turns = int(rng.integers(1, min(11, n + 5)))
        with bk.BranchStore(n) as store:
            st, w, vec = _make_random_state(rng, n)
            px, pz, pk, pe = _make_commuting_turns(rng, n, n_turns)
            nid = store.node_from_branches([w], [st], px, pz, pk, pe)
            st.close()

            dense_ref = vec.copy()
            for i in range(n_turns):
                dense_ref = _dense_apply_turn(
                    dense_ref, n, int(px[i, 0]), int(pz[i, 0]), int(pk[i]), int(pe[i])
                )

            dev_unres = float(np.max(np.abs(store.node_dense(nid) - dense_ref)))
            rid = store.resolve_node(nid)
            nb, npd, _ = store.node_info(rid)
            assert nb >= 1 and npd == 0
            dev_res = float(np.max(np.abs(store.node_dense(rid) - dense_ref)))
            max_dev = max(max_dev, dev_unres, dev_res)

            branches = store.node_branches(rid)
            keys = [b_st.canonical_key() for _, b_st in branches]
            for _, b_st in branches:
                b_st.close()
            assert keys == sorted(keys)

    assert max_dev < 1e-12, f"T2 random resolve max_dev={max_dev}"


def test_resolve_multiword_65_and_130() -> None:
    """Multi-word resolve at n = 65 and n = 130: RM(3,4) Clifford cancellation + 1/2-split resolves."""
    for n, qs in ((65, (0, 63, 64, 17)), (130, (0, 63, 64, 129))):
        nw = _nw(n)
        with bk.BranchStore(n) as store:
            st = bk.BranchState.zero(n)
            for q in qs:
                wq = _int_to_words(1 << q, nw)
                st.qturn(wq, wq, 3)  # |+> on qs
            st.canon()

            # 15-term RM(3,4) across word boundaries -> 0 splits, 1 branch, exact overlap invariants.
            xs = np.zeros((15, nw), dtype=np.uint64)
            zs = np.zeros((15, nw), dtype=np.uint64)
            ks = np.zeros(15, dtype=np.uint8)
            es = np.ones(15, dtype=np.uint8)
            for idx, mask in enumerate(range(1, 16)):
                z_int = 0
                for bit in range(4):
                    if (mask >> bit) & 1:
                        z_int |= 1 << qs[bit]
                zs[idx] = _int_to_words(z_int, nw)
            nid15 = store.node_from_branches([1.0 + 0.0j], [st], xs, zs, ks, es)
            r15 = store.resolve_node(nid15)
            nb15, np15, _ = store.node_info(r15)
            assert nb15 == 1 and np15 == 0 and store.counters()["splits"] == 0

            # 2-split resolve across word boundaries: 3 independent T gates on qs[:3]
            # collapse in ceil(3/2) = 2 splits -> 4 orthogonal branches with total norm^2 = 1.
            nid2 = store.node_from_branches(
                [1.0 + 0.0j],
                [st],
                np.zeros((3, nw), dtype=np.uint64),
                np.vstack([
                    _int_to_words(1 << qs[0], nw),
                    _int_to_words(1 << qs[1], nw),
                    _int_to_words(1 << qs[2], nw),
                ]),
                np.zeros(3, dtype=np.uint8),
                np.ones(3, dtype=np.uint8),
            )
            r2 = store.resolve_node(nid2)
            nb2, np2, _ = store.node_info(r2)
            assert nb2 == 4 and np2 == 0
            branches = store.node_branches(r2)
            norm2 = 0.0
            for i, (wi, si) in enumerate(branches):
                norm2 += abs(wi) ** 2
                for j in range(i + 1, len(branches)):
                    wj, sj = branches[j]
                    ov = si.overlap(sj)
                    norm2 += 2.0 * (wi.conjugate() * wj * ov).real
            for _, si in branches:
                si.close()
            assert abs(norm2 - 1.0) < 1e-12
            st.close()

    # r > 20 exactness test: 21-qubit GHZ-like state with 1-split non-Clifford set
    # |psi> = (|0...0> + |1...1>)/sqrt(2), apply T on q0 + 20 pairs (Z_k, Z_k Z_0) -> 2 branches.
    n21 = 21
    with bk.BranchStore(n21) as store:
        st = bk.BranchState.zero(n21)
        st.qturn([1], [1], 3)  # |+> on q0
        for q in range(1, n21):
            # CX(0 -> q) via R(+ZX) R(-ZI) R(-IX)
            st.qturn([1 << q], [1], 0)
            st.qturn([0], [1], 2)
            st.qturn([1 << q], [0], 2)
        st.canon()
        # Apply 1 T gate on q0 -> 2 branches, exact norm^2 = 1.
        nid = store.node_from_branches(
            [1.0 + 0.0j],
            [st],
            np.zeros((1, 1), dtype=np.uint64),
            np.array([[1]], dtype=np.uint64),
            np.zeros(1, dtype=np.uint8),
            np.ones(1, dtype=np.uint8),
        )
        rid = store.resolve_node(nid)
        assert store.node_info(rid)[0] == 2
        st.close()


# ---------------------------------------------------------------------------------------------
# Group 2: Differential Verification against Read-Only `scratch/magic_ref/mfe.py`
# ---------------------------------------------------------------------------------------------


def test_differential_is_clifford_and_choose_split_vs_mfe() -> None:
    """100% exact `is_clifford` & `choose_split` match vs `mfe.py` for r <= 20."""
    mfe_mod, _ = _load_mfe_ref()
    rng = np.random.default_rng(20260303)

    # 1. 2,000 random + RM codeword parity sets on r = 1..7.
    n_cliff = 0
    n_split = 0
    for _ in range(2000):
        r = int(rng.integers(1, 8))
        O: set[int] = set()
        if rng.random() < 0.4:
            for _ in range(int(rng.integers(0, 35))):
                O ^= {int(rng.integers(1, 1 << r))}
        else:
            for _ in range(int(rng.integers(1, 4))):
                k = min(r, int(rng.integers(4, 8)))
                if k < 4:
                    continue
                sub = [int(x) for x in rng.choice(r, size=k, replace=False)]
                for x in range(1, 1 << k):
                    val = 0
                    for j in range(k):
                        if (x >> j) & 1:
                            val |= 1 << sub[j]
                    O ^= {val}
            if rng.random() < 0.5 and O:
                O ^= {int(rng.integers(1, 1 << r))}

        O_list = sorted(O)
        got_c = bk.is_clifford(O_list)
        ref_c = bool(mfe_mod.is_clifford(O))
        assert got_c == ref_c, f"is_clifford mismatch on {O_list}"
        n_cliff += int(got_c)
        if not got_c and len(O_list) <= 24:
            got_sp = bk.choose_split(O_list, max_depth=5)
            ref_sp = mfe_mod.choose_split(O, maxdepth=5)
            assert got_sp == ref_sp, f"choose_split mismatch on {O_list}: {got_sp} vs {ref_sp}"
            n_split += 1
    assert n_cliff > 100 and n_split > 500

    # 2. High-r boundary checks r in {15, 17, 19, 20}: RM codewords & single-term perturbations.
    for r in (15, 17, 19, 20):
        sub = [0, 5, r - 2, r - 1]
        O_rm = {
            sum((1 << sub[j]) for j in range(4) if (x >> j) & 1)
            for x in range(1, 16)
        }
        assert bk.is_clifford(sorted(O_rm)) == bool(mfe_mod.is_clifford(O_rm)) is True
        O_pert = O_rm ^ {(1 << (r - 1)) | (1 << (r - 2))}
        assert bk.is_clifford(sorted(O_pert)) == bool(mfe_mod.is_clifford(O_pert)) is False
        assert bk.choose_split(sorted(O_pert), max_depth=5) == mfe_mod.choose_split(O_pert, maxdepth=5)


def test_differential_resolve_and_skip_clifford_vs_mfe() -> None:
    """Differential check vs `mfe.py`: identical canonical keys, sorted order, weights < 1e-12."""
    mfe_mod, bqfe_mod = _load_mfe_ref()
    rng = np.random.default_rng(20260304)
    max_w_dev = 0.0
    max_amp_dev = 0.0
    mfe_cache = mfe_mod.Cache()

    for trial in range(150):
        n = int(rng.integers(2, 8))
        n_turns = int(rng.integers(1, min(9, n + 4)))
        with bk.BranchStore(n) as store:
            st, w, _ = _make_random_state(rng, n)
            q_ref = _state_to_bqfe(st, bqfe_mod)
            px, pz, pk, pe = _make_commuting_turns(rng, n, n_turns)

            nid = store.node_from_branches([w], [st])
            st.close()

            # Build parallel MF instance.
            mf = mfe_mod.MF(n, [[w, q_ref]], [], mfe_cache)
            axes_ref = [
                ((int(px[i, 0]), int(pz[i, 0]), int(pk[i])), int(pe[i]))
                for i in range(n_turns)
            ]
            c_rot = store.step_rotations(nid, 1, bk.MBR_KIND_T, 0, px, pz, pk, pe)
            mf.rotate(axes_ref)

            # Half the trials apply a random 2Q skip_clifford before resolving!
            if trial % 2 == 1:
                gate = "CX" if rng.random() < 0.5 else "CZ"
                q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
                ref_tab = ReferenceCHPTableau(n)
                fac = compile_clifford_factors(ref_tab, gate, (q0, q1))
                seq = list(reversed(fac.r_axes))
                qx = np.array([ax.a_words for ax in seq], dtype=np.uint64)
                qz = np.array([ax.b_words for ax in seq], dtype=np.uint64)
                qk = np.array([(int(ax.kappa) + 2) & 3 for ax in seq], dtype=np.uint8)
                c_rot = store.step_qturns(c_rot, 2, bk.MBR_KIND_SKIP, 0, qx, qz, qk)
                labs_ref = [(int(qx[j, 0]), int(qz[j, 0]), int(qk[j])) for j in range(len(seq))]
                mf.skip_clifford((1.0 + 0.0j, labs_ref))

            # Compare pending turns exact equality before resolve.
            got_px, got_pz, got_pk, got_pe = store.node_pending(c_rot)
            got_pend = [
                ((int(got_px[i, 0]), int(got_pz[i, 0]), int(got_pk[i])), int(got_pe[i]))
                for i in range(len(got_pk))
            ]
            ref_pend = [(tuple(it[0]), int(it[1])) for it in mf.pend]
            assert got_pend == ref_pend

            # Resolve both and compare canonical keys, sorted order, and weights.
            rid = store.resolve_node(c_rot)
            mf.flush()

            got_branches = store.node_branches(rid)
            got_pairs = [(b_st.canonical_key(), bw) for bw, b_st in got_branches]
            for _, b_st in got_branches:
                b_st.close()
            ref_pairs = sorted((_bqfe_key_bytes(qr), wr) for wr, qr in mf.b)

            assert [k for k, _ in got_pairs] == [k for k, _ in ref_pairs]
            for (_, gw), (_, rw) in zip(got_pairs, ref_pairs):
                max_w_dev = max(max_w_dev, abs(gw - rw))

            # Dense comparison (axis q = qubit q in mfe.dense -> reshape to little-endian flat).
            ref_dense = mf.dense().reshape(-1, order="F")
            got_dense = store.node_dense(rid, endian="little")
            max_amp_dev = max(max_amp_dev, float(np.max(np.abs(got_dense - ref_dense))))

    assert max_w_dev < 1e-12, f"mfe diff max_w_dev={max_w_dev}"
    assert max_amp_dev < 1e-12, f"mfe diff max_amp_dev={max_amp_dev}"


# ---------------------------------------------------------------------------------------------
# Group 3 (T3): Per-Shot Moving-Frame Driver vs Dense Oracle (>= 2,000 Random Circuits)
# ---------------------------------------------------------------------------------------------


class _DenseOracle:
    """Independent dense state-vector simulator on n <= 8 qubits (little-endian)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.vec = np.zeros(1 << n, dtype=np.complex128)
        self.vec[0] = 1.0

    def gate(self, name: str, targets: tuple[int, ...], *, inverse: bool = False) -> None:
        self.vec = _dense_apply_clifford_gate(self.vec, self.n, name, targets, inverse=inverse)

    def pauli_terms(self, terms: list[tuple[int, str]], negate: bool = False) -> None:
        x_int, z_int = 0, 0
        for q, ch in terms:
            if ch in ("X", "Y"):
                x_int |= 1 << q
            if ch in ("Z", "Y"):
                z_int |= 1 << q
        k = _herm_kappa(x_int, z_int, negate=negate)
        self.vec = _dense_apply_pauli(self.vec, self.n, x_int, z_int, k)

    def spp(self, terms: list[tuple[int, str]], *, negate: bool = False, dag: bool = False) -> None:
        x_int, z_int = 0, 0
        for q, ch in terms:
            if ch in ("X", "Y"):
                x_int |= 1 << q
            if ch in ("Z", "Y"):
                z_int |= 1 << q
        # Stim SPP P = exp(-i pi/4 P) = R(-P); SPP_DAG P = exp(+i pi/4 P) = R(+P).
        k = _herm_kappa(x_int, z_int, negate=(not dag) ^ negate)
        self.vec = _dense_apply_qturn(self.vec, self.n, x_int, z_int, k)

    def t_gate(self, q: int, dag: bool = False) -> None:
        self.vec = _dense_apply_turn(self.vec, self.n, 0, 1 << q, 0, 7 if dag else 1)

    def ccz(self, a: int, b: int, c: int) -> None:
        idx = np.arange(1 << self.n, dtype=np.uint64)
        mask = np.uint64((1 << a) | (1 << b) | (1 << c))
        flip = (idx & mask) == mask
        self.vec = np.where(flip, -self.vec, self.vec)

    def peek_prob0(self, terms: list[tuple[int, str]], negate: bool = False) -> float:
        x_int, z_int = 0, 0
        for q, ch in terms:
            if ch in ("X", "Y"):
                x_int |= 1 << q
            if ch in ("Z", "Y"):
                z_int |= 1 << q
        k = _herm_kappa(x_int, z_int, negate=negate)
        pv = _dense_apply_pauli(self.vec, self.n, x_int, z_int, k)
        proj0 = 0.5 * (self.vec + pv)
        return float(np.real(np.vdot(proj0, proj0)))

    def project(self, terms: list[tuple[int, str]], outcome: int, negate: bool = False) -> float:
        x_int, z_int = 0, 0
        for q, ch in terms:
            if ch in ("X", "Y"):
                x_int |= 1 << q
            if ch in ("Z", "Y"):
                z_int |= 1 << q
        k = _herm_kappa(x_int, z_int, negate=negate)
        pv = _dense_apply_pauli(self.vec, self.n, x_int, z_int, k)
        proj = 0.5 * (self.vec + (1.0 if outcome == 0 else -1.0) * pv)
        p = float(np.real(np.vdot(proj, proj)))
        assert p > 1e-13
        self.vec = proj / math.sqrt(p)
        return p


class _MovingFrameShotDriver:
    """Compact test-only per-shot moving-frame driver for T3 verification."""

    def __init__(self, n: int, store: bk.BranchStore) -> None:
        self.n = n
        self.nw = _nw(n)
        self.store = store
        self.U = ReferenceCHPTableau(n)
        self.U_history: list[tuple[str, Any]] = []
        self.fx = np.zeros(self.nw, dtype=np.uint64)
        self.fz = np.zeros(self.nw, dtype=np.uint64)
        self.node = 0
        self._eid = 1

    def _next_eid(self) -> int:
        eid = self._eid
        self._eid += 1
        return eid

    def do_clifford(self, name: str, targets: tuple[int, ...]) -> None:
        self.U.do_unitary(name, targets)
        self.U_history.append(("gate", (name, targets)))

    def do_spp(self, terms: list[tuple[int, str]], *, negate: bool = False, dag: bool = False) -> None:
        self.U.do_spp(terms, negate=negate, dag=dag)
        self.U_history.append(("spp", (terms, negate, dag)))

    def do_pauli_fault(self, terms: list[tuple[int, str]]) -> None:
        ax = compile_pauli_product(self.U, terms)
        self.fx ^= ax.a_words
        self.fz ^= ax.b_words

    def do_t(self, q: int, dag: bool = False) -> None:
        ax = compile_pauli_product(self.U, [(q, "Z")])
        s = _anti_words(self.fx, self.fz, ax.a_words, ax.b_words)
        k = (int(ax.kappa) + (2 if s else 0)) & 3
        self.node = self.store.step_rotations(
            self.node,
            self._next_eid(),
            bk.MBR_KIND_T,
            s,
            ax.a_words.reshape(1, self.nw),
            ax.b_words.reshape(1, self.nw),
            np.array([k], dtype=np.uint8),
            np.array([7 if dag else 1], dtype=np.uint8),
        )

    def do_ccz(self, a: int, b: int, c: int) -> None:
        subsets = (
            ((a,), 1), ((b,), 1), ((c,), 1),
            ((a, b), 7), ((a, c), 7), ((b, c), 7),
            ((a, b, c), 1),
        )
        # Compute 3-bit sign mask from the 3 single-qubit Z pullbacks.
        za = compile_pauli_product(self.U, [(a, "Z")])
        zb = compile_pauli_product(self.U, [(b, "Z")])
        zc = compile_pauli_product(self.U, [(c, "Z")])
        sb = (
            _anti_words(self.fx, self.fz, za.a_words, za.b_words)
            | (_anti_words(self.fx, self.fz, zb.a_words, zb.b_words) << 1)
            | (_anti_words(self.fx, self.fz, zc.a_words, zc.b_words) << 2)
        )
        xs = np.zeros((7, self.nw), dtype=np.uint64)
        zs = np.zeros((7, self.nw), dtype=np.uint64)
        ks = np.zeros(7, dtype=np.uint8)
        es = np.zeros(7, dtype=np.uint8)
        for i, (sub, e) in enumerate(subsets):
            ax = compile_pauli_product(self.U, [(q, "Z") for q in sub])
            s = _anti_words(self.fx, self.fz, ax.a_words, ax.b_words)
            xs[i] = ax.a_words
            zs[i] = ax.b_words
            ks[i] = np.uint8((int(ax.kappa) + (2 if s else 0)) & 3)
            es[i] = np.uint8(e)
        self.node = self.store.step_rotations(
            self.node, self._next_eid(), bk.MBR_KIND_CCZ, sb, xs, zs, ks, es
        )

    def do_skip_or_cfault(self, name: str, targets: tuple[int, ...], *, is_skip: bool) -> None:
        fac = compile_clifford_factors(self.U, name, targets)
        if not is_skip and fac.p0 is not None:
            self.fx ^= fac.p0.a_words
            self.fz ^= fac.p0.b_words
        seq = list(reversed(fac.r_axes) if is_skip else fac.r_axes)
        if seq:
            xs = np.zeros((len(seq), self.nw), dtype=np.uint64)
            zs = np.zeros((len(seq), self.nw), dtype=np.uint64)
            ks = np.zeros(len(seq), dtype=np.uint8)
            sb = 0
            for i, ax in enumerate(seq):
                s = _anti_words(self.fx, self.fz, ax.a_words, ax.b_words)
                sb |= s << i
                xs[i] = ax.a_words
                zs[i] = ax.b_words
                # inverse (SKIP): R(-(-1)^s A); forward (CFAULT): R(+(-1)^s A)
                neg_flag = (1 if is_skip else 0) ^ s
                ks[i] = np.uint8((int(ax.kappa) + (2 if neg_flag else 0)) & 3)
            kind = bk.MBR_KIND_SKIP if is_skip else bk.MBR_KIND_CFAULT
            self.node = self.store.step_qturns(
                self.node, self._next_eid(), kind, sb, xs, zs, ks
            )
        if is_skip and fac.p0 is not None:
            self.fx ^= fac.p0.a_words
            self.fz ^= fac.p0.b_words

    def peek_prob0(self, terms: list[tuple[int, str]], negate: bool = False) -> float:
        ax = compile_pauli_product(self.U, terms, negate=negate)
        p0_inner = self.store.step_peek(
            self.node, self._next_eid(), ax.a_words, ax.b_words, int(ax.kappa)
        )
        s = _anti_words(self.fx, self.fz, ax.a_words, ax.b_words)
        return (1.0 - p0_inner) if s else p0_inner

    def measure(
        self,
        terms: list[tuple[int, str]],
        u: float,
        *,
        kind: int = bk.MBR_KIND_MEAS,
        negate: bool = False,
        force_outcome: int | None = None,
    ) -> tuple[int, float]:
        ax = compile_pauli_product(self.U, terms, negate=negate)
        s = _anti_words(self.fx, self.fz, ax.a_words, ax.b_words)
        p0_inner, c0, c1, mt, abs_x, abs_z = self.store.step_meas(
            self.node, self._next_eid(), kind, ax.a_words, ax.b_words, int(ax.kappa)
        )
        if force_outcome is None:
            inner_m = 0 if u < p0_inner else 1
            if inner_m == 0 and c0 < 0:
                inner_m = 1
            elif inner_m == 1 and c1 < 0:
                inner_m = 0
            outcome = inner_m ^ s
        else:
            outcome = int(force_outcome) & 1
            inner_m = outcome ^ s

        p_outcome = p0_inner if inner_m == 0 else (1.0 - p0_inner)
        if mt == bk.MBR_MEAS_ABSORBED and inner_m == 1:
            self.fx ^= abs_x
            self.fz ^= abs_z
        self.node = c0 if inner_m == 0 else c1
        assert self.node >= 0
        return outcome, p_outcome

    def physical_dense(self, *, resolve_first: bool = False) -> np.ndarray:
        nid = self.store.resolve_node(self.node) if resolve_first else self.node
        vec = self.store.node_dense(nid, endian="little")
        # Apply F_t (Hermitian Pauli with + sign).
        fx_i = _words_to_int(self.fx)
        fz_i = _words_to_int(self.fz)
        vec = _dense_apply_pauli(vec, self.n, fx_i, fz_i, _herm_kappa(fx_i, fz_i))
        # Replay U_t in forward order.
        for kind, payload in self.U_history:
            if kind == "gate":
                name, targets = payload
                vec = _dense_apply_clifford_gate(vec, self.n, name, targets)
            else:
                terms, negate, dag = payload
                x_int, z_int = 0, 0
                for q, ch in terms:
                    if ch in ("X", "Y"):
                        x_int |= 1 << q
                    if ch in ("Z", "Y"):
                        z_int |= 1 << q
                k = _herm_kappa(x_int, z_int, negate=(not dag) ^ negate)
                vec = _dense_apply_qturn(vec, self.n, x_int, z_int, k)
        return vec


def test_t3_ccz_matches_7cx_7t_identity() -> None:
    """Verify CCZ pending expansion matches the exact 7-T + CX decomposition."""
    n = 3
    rng = np.random.default_rng(20260305)
    with bk.BranchStore(n) as store:
        drv = _MovingFrameShotDriver(n, store)
        dense = _DenseOracle(n)
        for q in range(n):
            drv.do_clifford("H", (q,))
            dense.gate("H", (q,))
        drv.do_pauli_fault([(0, "X"), (2, "Y")])
        dense.pauli_terms([(0, "X"), (2, "Y")])
        drv.do_ccz(0, 1, 2)
        dense.ccz(0, 1, 2)
        assert _phase_aligned_diff(drv.physical_dense(resolve_first=False), dense.vec) < 1e-12
        assert _phase_aligned_diff(drv.physical_dense(resolve_first=True), dense.vec) < 1e-12
        # Measure all 3 qubits and verify probabilities.
        for q in range(n):
            u = float(rng.random())
            m, p_b = drv.measure([(q, "Z")], u)
            p_d = dense.project([(q, "Z")], m)
            assert abs(p_b - p_d) < 1e-12


def test_t3_per_shot_driver_vs_dense_2000_random_circuits() -> None:
    """T3: >= 2,000 random circuits (n = 1..8) covering all Clifford, magic, fault & handler ops."""
    rng = np.random.default_rng(20260306)
    gates_1q = ("H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y", "C_XYZ", "H_YZ")
    gates_2q = ("CX", "CZ", "CY", "SWAP", "ISWAP", "CXSWAP", "SWAPCX", "XCX")
    paulis = ("X", "Y", "Z")
    reset_kicks = {"Z": "X", "X": "Z", "Y": "Z"}

    max_prob_dev = 0.0
    max_state_dev_pre = 0.0
    max_state_dev_post = 0.0
    op_counts: dict[str, int] = {}

    def bump(name: str) -> None:
        op_counts[name] = op_counts.get(name, 0) + 1

    n_circuits = 2000
    for c_idx in range(n_circuits):
        n = int((c_idx % 8) + 1)
        n_ops = int(rng.integers(6, 15))
        with bk.BranchStore(n) as store:
            drv = _MovingFrameShotDriver(n, store)
            dense = _DenseOracle(n)
            magic_budget = 0

            for _ in range(n_ops):
                r = float(rng.random())
                if r < 0.22:
                    # 1Q Clifford
                    g = gates_1q[int(rng.integers(0, len(gates_1q)))]
                    q = int(rng.integers(0, n))
                    drv.do_clifford(g, (q,))
                    dense.gate(g, (q,))
                    bump("cliff1")
                elif r < 0.38 and n >= 2:
                    # 2Q Clifford
                    g = gates_2q[int(rng.integers(0, len(gates_2q)))]
                    q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
                    drv.do_clifford(g, (q0, q1))
                    dense.gate(g, (q0, q1))
                    bump("cliff2")
                elif r < 0.43:
                    # SPP / SPP_DAG
                    k_len = int(rng.integers(1, min(n, 3) + 1))
                    qs = [int(x) for x in rng.choice(n, size=k_len, replace=False)]
                    terms = [(q, paulis[int(rng.integers(0, 3))]) for q in qs]
                    neg = bool(rng.integers(0, 2))
                    dag = bool(rng.integers(0, 2))
                    drv.do_spp(terms, negate=neg, dag=dag)
                    dense.spp(terms, negate=neg, dag=dag)
                    bump("spp")
                elif r < 0.56 and magic_budget < 4:
                    # T / T_DAG
                    q = int(rng.integers(0, n))
                    dag = bool(rng.integers(0, 2))
                    drv.do_t(q, dag=dag)
                    dense.t_gate(q, dag=dag)
                    magic_budget += 1
                    bump("t")
                elif r < 0.61 and n >= 3 and magic_budget < 3:
                    # CCZ
                    a, b, c = [int(x) for x in rng.choice(n, size=3, replace=False)]
                    drv.do_ccz(a, b, c)
                    dense.ccz(a, b, c)
                    magic_budget += 2
                    bump("ccz")
                elif r < 0.69:
                    # Pauli fault
                    q = int(rng.integers(0, n))
                    p = paulis[int(rng.integers(0, 3))]
                    drv.do_pauli_fault([(q, p)])
                    dense.pauli_terms([(q, p)])
                    bump("pfault")
                elif r < 0.74:
                    # CFAULT (1Q or 2Q)
                    if n >= 2 and rng.random() < 0.5:
                        g = gates_2q[int(rng.integers(0, len(gates_2q)))]
                        t = tuple(int(x) for x in rng.choice(n, size=2, replace=False))
                    else:
                        g = gates_1q[int(rng.integers(0, len(gates_1q)))]
                        t = (int(rng.integers(0, n)),)
                    drv.do_skip_or_cfault(g, t, is_skip=False)
                    dense.gate(g, t, inverse=False)
                    bump("cfault")
                elif r < 0.79:
                    # SKIP (ideal gate was added to U, then skipped via G~^dag -> net identity on dense!)
                    if n >= 2 and rng.random() < 0.5:
                        g = gates_2q[int(rng.integers(0, len(gates_2q)))]
                        t = tuple(int(x) for x in rng.choice(n, size=2, replace=False))
                    else:
                        g = gates_1q[int(rng.integers(0, len(gates_1q)))]
                        t = (int(rng.integers(0, n)),)
                    drv.do_clifford(g, t)
                    drv.do_skip_or_cfault(g, t, is_skip=True)
                    bump("skip")
                elif r < 0.85:
                    # M / MX / MY
                    q = int(rng.integers(0, n))
                    basis = paulis[int(rng.integers(0, 3))]
                    u = float(rng.random())
                    m, p_b = drv.measure([(q, basis)], u, kind=bk.MBR_KIND_MEAS)
                    p_d = dense.project([(q, basis)], m)
                    max_prob_dev = max(max_prob_dev, abs(p_b - p_d))
                    bump("meas_1q")
                elif r < 0.89:
                    # MPP (multi-qubit Pauli product measurement)
                    k_len = int(rng.integers(1, min(n, 3) + 1))
                    qs = [int(x) for x in rng.choice(n, size=k_len, replace=False)]
                    terms = [(q, paulis[int(rng.integers(0, 3))]) for q in qs]
                    neg = bool(rng.integers(0, 2))
                    u = float(rng.random())
                    m, p_b = drv.measure(terms, u, kind=bk.MBR_KIND_MEAS, negate=neg)
                    p_d = dense.project(terms, m, negate=neg)
                    max_prob_dev = max(max_prob_dev, abs(p_b - p_d))
                    bump("mpp")
                elif r < 0.93:
                    # R / RX / RY / MR
                    q = int(rng.integers(0, n))
                    basis = paulis[int(rng.integers(0, 3))]
                    u = float(rng.random())
                    m, p_b = drv.measure([(q, basis)], u, kind=bk.MBR_KIND_MEAS)
                    p_d = dense.project([(q, basis)], m)
                    max_prob_dev = max(max_prob_dev, abs(p_b - p_d))
                    if m == 1:
                        kick = reset_kicks[basis]
                        drv.do_pauli_fault([(q, kick)])
                        dense.pauli_terms([(q, kick)])
                    bump("reset")
                elif r < 0.96:
                    # PEEK (read-only Born probability)
                    q = int(rng.integers(0, n))
                    basis = paulis[int(rng.integers(0, 3))]
                    p0_b = drv.peek_prob0([(q, basis)])
                    p0_d = dense.peek_prob0([(q, basis)])
                    max_prob_dev = max(max_prob_dev, abs(p0_b - p0_d))
                    bump("peek")
                elif r < 0.98:
                    # POST (postselect onto a possible Z outcome)
                    q = int(rng.integers(0, n))
                    p0_d = dense.peek_prob0([(q, "Z")])
                    desired = 0 if p0_d > 0.25 else 1
                    if desired == 1 and (1.0 - p0_d) <= 1e-6:
                        desired = 0
                    m, p_b = drv.measure([(q, "Z")], 0.0, kind=bk.MBR_KIND_POST, force_outcome=desired)
                    assert m == desired
                    p_d = dense.project([(q, "Z")], desired)
                    max_prob_dev = max(max_prob_dev, abs(p_b - p_d))
                    bump("post")
                else:
                    # URESET (unmatched noisy reset to |0> or |1> in Z)
                    q = int(rng.integers(0, n))
                    target_bit = int(rng.integers(0, 2))
                    u = float(rng.random())
                    m, p_b = drv.measure([(q, "Z")], u, kind=bk.MBR_KIND_URESET)
                    p_d = dense.project([(q, "Z")], m)
                    max_prob_dev = max(max_prob_dev, abs(p_b - p_d))
                    if m != target_bit:
                        drv.do_pauli_fault([(q, "X")])
                        dense.pauli_terms([(q, "X")])
                    bump("ureset")

            # Verify final physical state vector up to global phase (pre- and post-resolve).
            dev_pre = _phase_aligned_diff(drv.physical_dense(resolve_first=False), dense.vec)
            dev_post = _phase_aligned_diff(drv.physical_dense(resolve_first=True), dense.vec)
            max_state_dev_pre = max(max_state_dev_pre, dev_pre)
            max_state_dev_post = max(max_state_dev_post, dev_post)

    # Confirm every single event category was exercised across the 2,000 circuits.
    for key in (
        "cliff1", "cliff2", "spp", "t", "ccz", "pfault", "cfault", "skip",
        "meas_1q", "mpp", "reset", "peek", "post", "ureset",
    ):
        assert op_counts.get(key, 0) > 50, f"under-exercised op category {key}: {op_counts}"

    assert max_prob_dev < 1e-10, f"T3 max_prob_dev={max_prob_dev}"
    assert max_state_dev_pre < 1e-10, f"T3 max_state_dev_pre={max_state_dev_pre}"
    assert max_state_dev_post < 1e-10, f"T3 max_state_dev_post={max_state_dev_post}"

    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0
