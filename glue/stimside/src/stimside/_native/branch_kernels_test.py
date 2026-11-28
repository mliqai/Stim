"""T14 differential verification & T15 C++ kernel hardening tests for `_native.branch_kernels`.

Covers:
1. T14 Differential Verification against `scratch/magic_ref/{bqfe,mfe,bsm}.py` (with zero skips
   when `scratch/magic_ref` is present and a compact self-contained test-only fallback oracle if
   absent, plus a dedicated test exercising the fallback path):
   - T14a (`bqfe.py`): small `n = 1..8` (>=300 random states) and multi-word `n in {63, 64, 65, 130}`
     with `d > 64` (incl. `d = 65, 70, 130`): `pauli`, `qturn`, `zproject`, `project`, `definite`,
     `generators`, `canon`, exact canonical byte key, `overlap`, `same_phase`, `dense("big")`.
   - T14b (`mfe.py`): `is_clifford` & `choose_split(max_depth=5)` on >=1,000 random/RM/boundary
     `r in {15, 17, 19, 20}` parity sets (100% exact equality); resolve + `step_rotations` +
     `step_qturns` (`skip_clifford` with `scal=1`) on >=150 single/multi-branch states (`n = 2..7`),
     K1 canceling-branch pruning, and multi-word resolve (`n = 65, 130`).
   - T14c (`bsm.py`): `bsm.Machine` replay on `authors` (`n = 20`, 256 shots + `max_nodes=50` pass)
     and 20 random Clifford+T+CCZ+M+CORR+loss circuits (`n = 3..6`, also vs dense statevector),
     explicitly accounting for C++ GF(2) absorption kicks (`F ^= Q_bar`).
2. T15 Hostile Inputs, Multi-Word (130+, 193), Genuine C++ OOM, Threading, `os.fork()` & Warnings:
   - Exhaustive 41-entrypoint hostile C ABI sweep (asserting `len(bk._SPECS) == 41`) including
     dirty tail bits (`x` & `z`), cache-hit validation bypass regressions on `step_rotations` and
     `step_qturns`, `max_depth > 20`, overflows, NaN/Inf, `-3` drift, `-4` budget, and Python
     wrapper `np.bool_` / `(0, 0)` parities rejection.
   - Multi-word stress for `n in {63, 64, 65, 130, 193}` with `d = 65, 70, 130, 193` across all
     state primitives and `BranchStore` ops (including multi-word GF(2) absorption across words 0..3).
   - Genuine C++ OOM via `LD_PRELOAD` `operator new` thread-local countdown in subprocess across
     EVERY allocating `mbr_*` entrypoint (verifying `MemoryError`, zero handle leaks, pre-existing
     state/store bit-identical invariants, and clean post-OOM retry) plus zero-alloc pins.
   - 8-thread independent & `RLock`-shared stress + clean subprocess `os.fork()` safety (idle
     `BranchState`/`BranchStore`, lock-held idle `_SharedBranchStore`, and busy-thread poisoning).
   - Strict compiler check (`-Wall -Wextra -Werror -ffp-contract=off`).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stimside._native import branch_kernels as bk

_NATIVE_DIR = Path(bk.__file__).resolve().parent
_SRC_DIR = _NATIVE_DIR.parents[1]
_MAGIC_REF_DIR = Path(
    "/usr/local/google/home/mliq/.gemini/jetski/brain/"
    "a4cd4c57-114b-4c18-9db0-bcc178e62419/scratch/magic_ref"
)
_REF_MOD_NAMES = (
    "bqfe",
    "mfe",
    "bsm",
    "bhs",
    "mr_core",
    "fold3",
    "authors_circuit",
    "wbranch",
)
_OMEGA = complex(math.sqrt(0.5), math.sqrt(0.5))
_OMEGA_P = tuple(_OMEGA**k for k in range(8))
_IPOW = (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)


# =============================================================================================
# Shared Bit-Word & Canonical-Key Helpers
# =============================================================================================


def _nw(n: int) -> int:
    return (n + 63) // 64


def _int_to_words(val: int, nw: int) -> np.ndarray:
    out = np.zeros(nw, dtype=np.uint64)
    v = int(val)
    for w in range(nw):
        out[w] = np.uint64(v & 0xFFFFFFFFFFFFFFFF)
        v >>= 64
    return out


def _words_to_int(words: np.ndarray) -> int:
    val = 0
    for w in range(len(words) - 1, -1, -1):
        val = (val << 64) | int(words[w])
    return val


def _rand_mask(rng: np.random.Generator, n: int) -> int:
    nw = _nw(n)
    raw = rng.integers(0, 1 << 64, size=nw, dtype=np.uint64)
    rem = n & 63
    if rem:
        raw[-1] &= np.uint64((1 << rem) - 1)
    return _words_to_int(raw)


def _herm_kappa(x_int: int, z_int: int, neg: bool = False) -> int:
    return ((x_int & z_int).bit_count() + (2 if neg else 0)) & 3


def _anti_int(x1: int, z1: int, x2: int, z2: int) -> int:
    return ((x1 & z2) ^ (z1 & x2)).bit_count() & 1


def _pauli_mul_int(
    x1: int, z1: int, k1: int, x2: int, z2: int, k2: int
) -> tuple[int, int, int]:
    """Multiply (i^k1 X(x1) Z(z1)) (i^k2 X(x2) Z(z2)) (Z acts first)."""
    k = (k1 + k2 + 2 * (z1 & x2).bit_count()) & 3
    return x1 ^ x2, z1 ^ z2, k


def _norm_sign_plus_int(x: int, z: int, k: int) -> tuple[int, bool]:
    pc = (x & z).bit_count() & 3
    if ((k - pc) & 3) == 2:
        return (k + 2) & 3, True
    return k & 3, False


def _bqfe_key_bytes(q: Any) -> bytes:
    """Serialize a canonical `bqfe.Q` to the exact canonical byte key format of `BranchState`."""
    qc = q.copy()
    qc.canon()
    n = int(qc.n)
    d = len(qc.cols)
    nw = _nw(n)
    parts = [struct.pack("<II", n, d)]
    parts.append(int(qc.h).to_bytes(nw * 8, "little"))
    for col in qc.cols:
        parts.append(int(col).to_bytes(nw * 8, "little"))
    parts.append(bytes(int(x) & 0xFF for x in qc.ell))
    n_tri = (d * (d - 1)) // 2
    g_bytes = (n_tri + 7) // 8
    g_buf = bytearray(g_bytes)
    bit = 0
    for j in range(d):
        gj = int(qc.gam[j])
        for k in range(j + 1, d):
            if (gj >> k) & 1:
                g_buf[bit >> 3] |= 1 << (bit & 7)
            bit += 1
    parts.append(bytes(g_buf))
    return b"".join(parts)


def _phase_aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    ov = np.vdot(b, a)
    if abs(ov) > 1e-15:
        b = b * (ov / abs(ov))
    return float(np.max(np.abs(a - b)))


# =============================================================================================
# Section A — Dynamic `scratch/magic_ref` Loader & Compact Test-Only Fallback Oracle
# =============================================================================================


@contextlib.contextmanager
def _magic_ref_modules(ref_dir: Path | None = None):
    """Load `scratch/magic_ref/{bqfe,mfe,bsm,bhs}` cleanly, restoring `sys.path`/`sys.modules`."""
    target = _MAGIC_REF_DIR if ref_dir is None else ref_dir
    try:
        has_dir = target.is_dir()
    except OSError:
        has_dir = False
    if not has_dir:
        yield None
        return
    saved_path = list(sys.path)
    saved_mods = {name: sys.modules.get(name) for name in _REF_MOD_NAMES}
    for name in _REF_MOD_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(target))
    try:
        bqfe = importlib.import_module("bqfe")
        mfe = importlib.import_module("mfe")
        bsm = importlib.import_module("bsm")
        bhs = importlib.import_module("bhs")
        yield (bqfe, mfe, bsm, bhs)
    finally:
        sys.path[:] = saved_path
        for name, mod in saved_mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class _FallbackOracle:
    """Compact test-only oracle used ONLY when `scratch/magic_ref` is absent (zero skips)."""

    @staticmethod
    def _bitrev(val: int, n: int) -> int:
        out = 0
        for q in range(n):
            if (val >> q) & 1:
                out |= 1 << (n - 1 - q)
        return out

    @classmethod
    def dense_pauli_big(cls, vec: np.ndarray, n: int, x_i: int, z_i: int, k: int) -> np.ndarray:
        """Apply i^k X(x) Z(z) to a big-endian dense vector of shape (2**n,)."""
        xr = cls._bitrev(x_i, n)
        zr = cls._bitrev(z_i, n)
        idx = np.arange(1 << n, dtype=np.uint64)
        src = idx ^ np.uint64(xr)
        par = np.bitwise_count(src & np.uint64(zr)) & np.uint64(1)
        return (_IPOW[k & 3] * np.where(par == 0, 1.0, -1.0)) * vec[src]

    @classmethod
    def dense_qturn_big(cls, vec: np.ndarray, n: int, x_i: int, z_i: int, k: int) -> np.ndarray:
        return (vec + 1.0j * cls.dense_pauli_big(vec, n, x_i, z_i, k)) * math.sqrt(0.5)

    @classmethod
    def dense_project_big(
        cls, vec: np.ndarray, n: int, x_i: int, z_i: int, k: int, m: int
    ) -> tuple[float, np.ndarray]:
        pv = cls.dense_pauli_big(vec, n, x_i, z_i, k)
        proj = 0.5 * (vec + ((-1.0) ** m) * pv)
        norm = float(np.linalg.norm(proj))
        if norm < 1e-12:
            return 0.0, vec.copy()
        return norm, proj / norm

    @classmethod
    def dense_turn_big(
        cls, vec: np.ndarray, n: int, x_i: int, z_i: int, k: int, e: int
    ) -> np.ndarray:
        we = _OMEGA_P[e & 7]
        pv = cls.dense_pauli_big(vec, n, x_i, z_i, k)
        return 0.5 * (1.0 + we) * vec + 0.5 * (1.0 - we) * pv

    @classmethod
    def dense_definite_big(cls, vec: np.ndarray, n: int, x_i: int, z_i: int, k: int) -> int:
        pv = cls.dense_pauli_big(vec, n, x_i, z_i, k)
        ev = complex(np.vdot(vec, pv)).real
        if abs(ev - 1.0) < 1e-9:
            return 1
        if abs(ev + 1.0) < 1e-9:
            return -1
        return 0

    @staticmethod
    def is_clifford(parities: list[int]) -> bool:
        uniq = sorted(set(parities))
        if not uniq:
            return True
        if len(uniq) < 15:
            return False
        r = max(x.bit_length() for x in uniq)
        if r > 20:
            return False
        f = np.zeros(1 << r, dtype=np.uint8)
        f[uniq] = 1
        for b in range(r):
            step = 1 << b
            f = f.reshape(-1, 2 * step)
            f[:, :step] ^= f[:, step:]
            f = f.reshape(-1)
        idx = np.flatnonzero(f)
        for v in idx:
            if 1 <= int(v).bit_count() <= 3:
                return False
        return True

    @staticmethod
    def quotient(parities: list[int], lam: int) -> list[int]:
        piv = (lam & -lam).bit_length() - 1
        counts: dict[int, int] = {}
        for x in parities:
            y = x ^ lam if (x >> piv) & 1 else x
            if y:
                counts[y] = counts.get(y, 0) ^ 1
        return sorted(y for y, c in counts.items() if c)

    @classmethod
    def choose_split(cls, parities: list[int], max_depth: int = 5) -> tuple[int, int]:
        uniq = sorted(set(parities))
        if cls.is_clifford(uniq):
            return 0, 0

        def dfs(cur: list[int], depth: int) -> int | None:
            if cls.is_clifford(cur):
                return 0
            if depth == 0:
                return None
            x0 = cur[0]
            for lam in [x0] + [x0 ^ y for y in cur[1:]]:
                q = cls.quotient(cur, lam)
                if dfs(q, depth - 1) is not None:
                    return lam
            return None

        for d in range(1, max_depth + 1):
            hit = dfs(uniq, d)
            if hit is not None:
                return hit, d
        return uniq[0], 0


def test_magic_ref_available_when_dir_present() -> None:
    """Confirm real `scratch/magic_ref` loads cleanly whenever `_MAGIC_REF_DIR` exists."""
    try:
        has_dir = _MAGIC_REF_DIR.is_dir()
    except OSError:
        has_dir = False
    with _magic_ref_modules() as refs:
        if has_dir:
            assert refs is not None
            bqfe, mfe, bsm, bhs = refs
            assert hasattr(bqfe, "Q") and hasattr(mfe, "MF")
            assert hasattr(bsm, "Machine") and hasattr(bhs, "load")
        else:
            assert refs is None


# =============================================================================================
# Section B — T14 Differential Verification vs `scratch/magic_ref/{bqfe,mfe,bsm}.py`
# =============================================================================================


def _run_t14a_with_fallback(rng: np.random.Generator) -> float:
    max_err = 0.0
    for n in range(1, 7):
        nw = _nw(n)
        for _ in range(15):
            st = bk.BranchState.zero(n)
            vec = np.zeros(1 << n, dtype=np.complex128)
            vec[0] = 1.0
            try:
                for _ in range(n + 2):
                    xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                    k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                    xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                    if rng.random() < 0.7:
                        sc = st.qturn(xw, zw, k)
                        vec = _FallbackOracle.dense_qturn_big(vec, n, xi, zi, k) / sc
                    else:
                        sc = st.pauli(xw, zw, k)
                        vec = _FallbackOracle.dense_pauli_big(vec, n, xi, zi, k) / sc
                err = float(np.max(np.abs(st.dense("big") - vec)))
                max_err = max(max_err, err)
                assert err < 1e-12
                xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                assert st.definite(xw, zw, k) == _FallbackOracle.dense_definite_big(
                    vec, n, xi, zi, k
                )
                m = int(rng.integers(0, 2))
                f_cpp = st.project(xw, zw, k, m)
                norm_ref, vec_ref = _FallbackOracle.dense_project_big(vec, n, xi, zi, k, m)
                assert abs(abs(f_cpp) - norm_ref) < 1e-12
                if abs(f_cpp) > 0:
                    err_p = _phase_aligned_diff(st.dense("big"), vec_ref)
                    max_err = max(max_err, err_p)
                    assert err_p < 1e-12
            finally:
                st.close()
    return max_err


def _gf2_rank_pauli_gens(gx: np.ndarray, gz: np.ndarray, n: int) -> int:
    rows: list[int] = []
    for i in range(n):
        row = (_words_to_int(gx[i]) << n) | _words_to_int(gz[i])
        rows.append(row)
    pivs: dict[int, int] = {}
    for r in rows:
        cur = r
        while cur:
            p = (cur & -cur).bit_length() - 1
            if p in pivs:
                cur ^= pivs[p]
            else:
                pivs[p] = cur
                break
    return len(pivs)


def test_t14a_bqfe_differential_small_and_multiword() -> None:
    """T14a: Differential verification against `bqfe.py` on small `n=1..8` and multi-word `n<=130, d>64`."""
    rng = np.random.default_rng(20260501)
    with _magic_ref_modules() as refs:
        if refs is None:
            _run_t14a_with_fallback(rng)
            return
        bqfe, _, _, _ = refs
        max_amp_err = 0.0
        max_scalar_err = 0.0

        # 1. Small n = 1..8: 320 random states (40 per n)
        for n in range(1, 9):
            nw = _nw(n)
            for trial in range(40):
                st = bk.BranchState.zero(n)
                q = bqfe.Q(n)
                try:
                    n_steps = int(rng.integers(2, 2 * n + 5))
                    for _ in range(n_steps):
                        xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                        k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                        xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                        u = rng.random()
                        if u < 0.65:
                            f_cpp = st.qturn(xw, zw, k)
                            f_ref = q.qturn(xi, zi, k)
                            max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                            assert abs(f_cpp - f_ref) < 1e-12
                        elif u < 0.85:
                            f_cpp = st.pauli(xw, zw, k)
                            f_ref = q.pauli(xi, zi, k)
                            max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                            assert abs(f_cpp - f_ref) < 1e-12
                        elif u < 0.93:
                            sig = int(rng.integers(0, 2))
                            m = int(rng.integers(0, 2))
                            q_bak = q.copy()
                            f_cpp = st.zproject(zw, sig, m)
                            f_ref = q.zproject(zi, sig, m)
                            max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                            assert abs(f_cpp - f_ref) < 1e-12
                            if f_ref == 0:
                                q = q_bak
                        else:
                            m = int(rng.integers(0, 2))
                            q_bak = q.copy()
                            f_cpp = st.project(xw, zw, k, m)
                            f_ref = q.project((xi, zi, k), m)
                            max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                            assert abs(f_cpp - f_ref) < 1e-12
                            if f_ref == 0:
                                q = q_bak

                    # definite() check on 5 random Paulis
                    for _ in range(5):
                        xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                        k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                        assert st.definite(_int_to_words(xi, nw), _int_to_words(zi, nw), k) == (
                            q.definite((xi, zi, k))
                        )

                    # generators() check: all n stabilize q with +1 and GF(2) rank == n
                    gx, gz, gk = st.generators()
                    assert _gf2_rank_pauli_gens(gx, gz, n) == n
                    for gi in range(n):
                        g_lbl = (_words_to_int(gx[gi]), _words_to_int(gz[gi]), int(gk[gi]))
                        assert q.definite(g_lbl) == 1

                    # dense("big") check BEFORE canon
                    d_cpp = st.dense("big").reshape((2,) * n)
                    d_ref = q.dense()
                    err_d = float(np.max(np.abs(d_cpp - d_ref)))
                    max_amp_err = max(max_amp_err, err_d)
                    assert err_d < 1e-12

                    # same_phase() against a phase-equivalent copy rotated by stabilizer/canon
                    st_copy = st.copy()
                    q_copy = q.copy()
                    try:
                        # Apply a stabilizer generator (changes global phase by +1) + canon on copy
                        g0 = (_words_to_int(gx[0]), _words_to_int(gz[0]), int(gk[0]))
                        st_copy.pauli(gx[0], gz[0], int(gk[0]))
                        q_copy.pauli(*g0)
                        c_copy = st_copy.canon()
                        q_copy.canon()
                        eq, ratio = st.same_phase(st_copy)
                        assert eq is True
                        ref_ratio = bqfe.same_phase(q, q_copy)
                        max_scalar_err = max(max_scalar_err, abs(ratio - ref_ratio))
                        assert abs(ratio - ref_ratio) < 1e-12
                        # Also verify canon() scalar & exact canonical key bytes
                        fc_cpp = st.canon()
                        fc_ref = q.canon()
                        max_scalar_err = max(max_scalar_err, abs(fc_cpp - fc_ref))
                        assert abs(fc_cpp - fc_ref) < 1e-12
                        assert st.canonical_key() == _bqfe_key_bytes(q)
                        assert abs(c_copy) == pytest.approx(1.0, abs=1e-12)
                    finally:
                        st_copy.close()

                    # overlap() against a second independent random state
                    if trial % 4 == 0:
                        st2 = bk.BranchState.zero(n)
                        q2 = bqfe.Q(n)
                        try:
                            for _ in range(n + 2):
                                xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                                k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                                st2.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), k)
                                q2.qturn(xi, zi, k)
                            ov_cpp = st.overlap(st2)
                            ov_ref = bqfe.overlap(q, q2)
                            max_scalar_err = max(max_scalar_err, abs(ov_cpp - ov_ref))
                            assert abs(ov_cpp - ov_ref) < 1e-12
                        finally:
                            st2.close()
                finally:
                    st.close()

        # 2. Multi-word n in {63, 64, 65, 130} with d explicitly > 64 (incl. d = 65, 70, 130)
        for n, target_d in ((63, 63), (64, 64), (65, 65), (130, 70), (130, 130)):
            nw = _nw(n)
            st = bk.BranchState.zero(n)
            q = bqfe.Q(n)
            try:
                # Build exact affine dimension d = target_d by applying X_j quarter-turns
                for qb in range(target_d):
                    xi = 1 << qb
                    xw = _int_to_words(xi, nw)
                    zw = np.zeros(nw, dtype=np.uint64)
                    f_cpp = st.qturn(xw, zw, 0)
                    f_ref = q.qturn(xi, 0, 0)
                    max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                    assert abs(f_cpp - f_ref) < 1e-12
                assert st.d == target_d and len(q.cols) == target_d

                # Mix with multi-word Paulis, qturns, zproject, project across word boundaries
                for _ in range(6):
                    xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                    k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                    xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                    f_cpp = st.qturn(xw, zw, k)
                    f_ref = q.qturn(xi, zi, k)
                    max_scalar_err = max(max_scalar_err, abs(f_cpp - f_ref))
                    assert abs(f_cpp - f_ref) < 1e-12

                    xi2, zi2 = _rand_mask(rng, n), _rand_mask(rng, n)
                    k2 = _herm_kappa(xi2, zi2, bool(rng.integers(0, 2)))
                    f_cpp = st.pauli(_int_to_words(xi2, nw), _int_to_words(zi2, nw), k2)
                    f_ref = q.pauli(xi2, zi2, k2)
                    assert abs(f_cpp - f_ref) < 1e-12

                # Check definite, generators, canon, canonical key, project, zproject, overlap, same_phase
                for _ in range(5):
                    xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                    k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                    assert st.definite(_int_to_words(xi, nw), _int_to_words(zi, nw), k) == (
                        q.definite((xi, zi, k))
                    )

                gx, gz, gk = st.generators()
                assert _gf2_rank_pauli_gens(gx, gz, n) == n
                for gi in (0, n // 2, n - 1):
                    g_lbl = (_words_to_int(gx[gi]), _words_to_int(gz[gi]), int(gk[gi]))
                    assert q.definite(g_lbl) == 1

                # Clone for overlap & same_phase
                st_b = st.copy()
                q_b = q.copy()
                try:
                    # Exact canon & canonical key match
                    fc_cpp = st.canon()
                    fc_ref = q.canon()
                    assert abs(fc_cpp - fc_ref) < 1e-12
                    assert st.canonical_key() == _bqfe_key_bytes(q)

                    eq, ratio = st.same_phase(st_b)
                    assert eq is True
                    assert abs(ratio - bqfe.same_phase(q, q_b)) < 1e-12

                    # Perturb st_b slightly so overlap is nontrivial (not 0 or 1)
                    xi, zi = (1 << 0) | (1 << (n - 1)), (1 << (n // 2))
                    k = _herm_kappa(xi, zi, False)
                    st_b.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), k)
                    q_b.qturn(xi, zi, k)
                    ov_cpp = st.overlap(st_b)
                    ov_ref = bqfe.overlap(q, q_b)
                    max_scalar_err = max(max_scalar_err, abs(ov_cpp - ov_ref))
                    assert abs(ov_cpp - ov_ref) < 1e-12

                    # zproject and project on multi-word
                    z_mask = (1 << 0) | (1 << (n - 1))
                    q_bak = q_b.copy()
                    f_cpp = st_b.zproject(_int_to_words(z_mask, nw), 0, 0)
                    f_ref = q_b.zproject(z_mask, 0, 0)
                    assert abs(f_cpp - f_ref) < 1e-12
                    if f_ref == 0:
                        q_b = q_bak
                    q_bak = q_b.copy()
                    f_cpp = st_b.project(_int_to_words(xi, nw), _int_to_words(zi, nw), k, 0)
                    f_ref = q_b.project((xi, zi, k), 0)
                    assert abs(f_cpp - f_ref) < 1e-12
                    if f_ref == 0:
                        q_b = q_bak
                    assert st_b.canonical_key() == _bqfe_key_bytes(q_b)
                finally:
                    st_b.close()
            finally:
                st.close()

        assert max_amp_err < 1e-12 and max_scalar_err < 1e-12
        assert bk.BranchState.live_states() == 0


def _make_commuting_paulis_int(
    rng: np.random.Generator, n: int, count: int
) -> list[tuple[int, int, int, int]]:
    """Generate `count` pairwise commuting Hermitian Paulis + exponents e in 1..7."""
    # Start from commuting Z-type generators on n qubits and conjugate by random 1Q/2Q Cliffords.
    # For arbitrary n (including 65, 130), apply a sparse sequence of random H/S/CX gates directly
    # on bitmasks so this helper stays _native-only with zero upper-layer imports!
    xs = [0] * count
    zs = []
    used: set[int] = set()
    for _ in range(count):
        for _attempt in range(16):
            # Sparse support so we exercise both small and cross-word products
            k_bits = int(rng.integers(1, min(n, 5) + 1))
            qs = rng.choice(n, size=k_bits, replace=False)
            m = 0
            for q in qs:
                m |= 1 << int(q)
            if m not in used:
                used.add(m)
                zs.append(m)
                break
        else:
            m = 1 << int(rng.integers(0, n))
            zs.append(m)

    # Apply random H, S, CX layers (preserving commutation!)
    n_cliff = min(3 * n, 40)
    for _ in range(n_cliff):
        u = rng.random()
        if u < 0.4:
            q = int(rng.integers(0, n))
            bit = 1 << q
            for i in range(count):
                xb = (xs[i] >> q) & 1
                zb = (zs[i] >> q) & 1
                if xb != zb:
                    xs[i] ^= bit
                    zs[i] ^= bit
        elif u < 0.65:
            q = int(rng.integers(0, n))
            bit = 1 << q
            for i in range(count):
                if (xs[i] >> q) & 1:
                    zs[i] ^= bit
        elif n >= 2:
            c, t = rng.choice(n, size=2, replace=False)
            cb, tb = 1 << int(c), 1 << int(t)
            for i in range(count):
                if (xs[i] >> int(c)) & 1:
                    xs[i] ^= tb
                if (zs[i] >> int(t)) & 1:
                    zs[i] ^= cb

    out = []
    for i in range(count):
        k = _herm_kappa(xs[i], zs[i], bool(rng.integers(0, 2)))
        e = int(rng.integers(1, 8))
        out.append((xs[i], zs[i], k, e))
    return out


def _run_t14b_with_fallback(rng: np.random.Generator) -> None:
    # 1. is_clifford & choose_split on 100 random parity sets
    for _ in range(100):
        r = int(rng.integers(1, 7))
        sz = int(rng.integers(1, min((1 << r) - 1, 18) + 1))
        O = sorted(int(x) for x in rng.choice(np.arange(1, 1 << r), size=sz, replace=False))
        assert bk.is_clifford(O) == _FallbackOracle.is_clifford(O)
        assert bk.choose_split(O, max_depth=5) == _FallbackOracle.choose_split(O, max_depth=5)

    # 2. Resolve & step_qturns vs dense oracle on 20 states
    for _ in range(20):
        n = int(rng.integers(2, 6))
        nw = _nw(n)
        st = bk.BranchState.zero(n)
        vec = np.zeros(1 << n, dtype=np.complex128)
        vec[0] = 1.0
        store = bk.BranchStore(n)
        try:
            for _ in range(n + 1):
                xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                sc = st.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), k)
                vec = _FallbackOracle.dense_qturn_big(vec, n, xi, zi, k) / sc
            turns = _make_commuting_paulis_int(rng, n, int(rng.integers(1, 5)))
            ax_x = np.stack([_int_to_words(x, nw) for x, _, _, _ in turns])
            ax_z = np.stack([_int_to_words(z, nw) for _, z, _, _ in turns])
            ax_k = np.array([k for _, _, k, _ in turns], dtype=np.uint8)
            ax_e = np.array([e for _, _, _, e in turns], dtype=np.uint8)
            nid = store.node_from_branches([1.0 + 0j], [st], ax_x, ax_z, ax_k, ax_e)
            rid = store.resolve_node(nid)
            for x, z, k, e in turns:
                vec = _FallbackOracle.dense_turn_big(vec, n, x, z, k, e)
            assert float(np.max(np.abs(store.node_dense(rid, "big") - vec))) < 1e-12
        finally:
            st.close()
            store.close()


def test_t14b_mfe_differential_clifford_split_resolve_and_qturns() -> None:
    """T14b: Differential verification vs `mfe.py` (`is_clifford`, `choose_split`, resolve, qturns)."""
    rng = np.random.default_rng(20260502)
    with _magic_ref_modules() as refs:
        if refs is None:
            _run_t14b_with_fallback(rng)
            return
        bqfe, mfe, _, _ = refs

        # 1. >= 1,000 random + RM + boundary r in {15, 17, 19, 20} parity sets
        parity_cases: list[list[int]] = []
        # Full RM(1,4) / 15-term Clifford identity and perturbations
        rm15 = list(range(1, 16))
        parity_cases.append(rm15)
        for drop in range(1, 16):
            parity_cases.append([x for x in rm15 if x != drop])
        # Higher-r RM embeddings at r = 15, 17, 19, 20 (including exact 15-term Clifford in r=20!)
        for r in (15, 17, 19, 20):
            # Embed 4 random independent basis vectors in GF(2)^r and take their 15 nonzero sums
            basis = [1 << (r - 1), 1 << (r - 2), 1 << 3, 1 << 0]
            emb = []
            for mask in range(1, 16):
                val = 0
                for b in range(4):
                    if (mask >> b) & 1:
                        val ^= basis[b]
                emb.append(val)
            parity_cases.append(sorted(emb))
            # Perturbed (non-Clifford) with 1 extra term or 1 dropped term
            parity_cases.append(sorted(emb[:-1]))
            parity_cases.append(sorted(set(emb + [1 << (r // 2)])))
            # Random sparse sets at boundary r
            for _ in range(15):
                sz = int(rng.integers(1, 18))
                s = {1 << (r - 1)}
                while len(s) < sz:
                    s.add(int(rng.integers(1, 1 << r)))
                parity_cases.append(sorted(s))
        # 1,000 random parity sets for r = 1..8
        for _ in range(1000):
            r = int(rng.integers(1, 9))
            hi = 1 << r
            sz = int(rng.integers(1, min(hi - 1, 24) + 1))
            s = sorted(int(x) for x in rng.choice(np.arange(1, hi), size=sz, replace=False))
            parity_cases.append(s)

        assert len(parity_cases) >= 1050
        for O in parity_cases:
            O_set = frozenset(O)
            assert bk.is_clifford(O) == bool(mfe.is_clifford(O_set))
            if not mfe.is_clifford(O_set):
                ref_lam, ref_dep = mfe.choose_split(O_set, maxdepth=5)
                cpp_lam, cpp_dep = bk.choose_split(O, max_depth=5)
                assert (cpp_lam, cpp_dep) == (int(ref_lam), int(ref_dep))
            else:
                assert bk.choose_split(O, max_depth=5) == (0, 0)

        # 2. >= 160 random single-branch and multi-branch (K=1,2,3) resolve + step_rotations + step_qturns
        max_w_err = 0.0
        max_dense_err = 0.0
        for trial in range(160):
            n = int(rng.integers(2, 8))
            nw = _nw(n)
            K = 1 if trial < 90 else (2 if trial < 130 else 3)
            cpp_states: list[bk.BranchState] = []
            ref_branches: list[list[Any]] = []
            weights: list[complex] = []
            store = bk.BranchStore(n)
            try:
                for _ in range(K):
                    st = bk.BranchState.zero(n)
                    q = bqfe.Q(n)
                    for _ in range(int(rng.integers(1, n + 3))):
                        xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                        k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                        st.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), k)
                        q.qturn(xi, zi, k)
                    st.canon()
                    q_f = q.canon()
                    w = complex(rng.normal(), rng.normal())
                    if abs(w) < 0.2:
                        w = 0.7 + 0.3j
                    cpp_states.append(st)
                    ref_branches.append([w, q])
                    weights.append(w)
                    assert abs(q_f) == pytest.approx(1.0, abs=1e-12)

                # Normalize weights to O(1) so K1 relative threshold aligns with mfe.TINY
                norm_w = math.sqrt(sum(abs(w) ** 2 for w in weights))
                weights = [w / norm_w for w in weights]
                for i in range(K):
                    ref_branches[i][0] = weights[i]

                turns = _make_commuting_paulis_int(rng, n, int(rng.integers(1, 6)))
                ax_x = np.stack([_int_to_words(x, nw) for x, _, _, _ in turns])
                ax_z = np.stack([_int_to_words(z, nw) for _, z, _, _ in turns])
                ax_k = np.array([k for _, _, k, _ in turns], dtype=np.uint8)
                ax_e = np.array([e for _, _, _, e in turns], dtype=np.uint8)

                nid = store.node_from_branches(weights, cpp_states)
                mf = mfe.MF(n, [[w, q.copy()] for w, q in ref_branches], [], mfe.Cache())

                # Step rotations (adds pending turns)
                nid_rot = store.step_rotations(
                    nid, 1, bk.MBR_KIND_T, 0, ax_x, ax_z, ax_k, ax_e
                )
                mf.rotate([((x, z, k), e) for x, z, k, e in turns])

                # Also exercise skip_clifford / step_qturns on every 3rd trial BEFORE resolve
                if trial % 3 == 0:
                    nq_turns = int(rng.integers(1, 4))
                    q_list = []
                    for _ in range(nq_turns):
                        qx, qz = _rand_mask(rng, n), _rand_mask(rng, n)
                        qk = _herm_kappa(qx, qz, bool(rng.integers(0, 2)))
                        q_list.append((qx, qz, qk))
                    qt_x = np.stack([_int_to_words(x, nw) for x, _, _ in q_list])
                    qt_z = np.stack([_int_to_words(z, nw) for _, z, _ in q_list])
                    qt_k = np.array([k for _, _, k in q_list], dtype=np.uint8)
                    nid_rot = store.step_qturns(
                        nid_rot, 2, bk.MBR_KIND_SKIP, 0, qt_x, qt_z, qt_k
                    )
                    mf.skip_clifford((1.0 + 0.0j, q_list))

                # Resolve both
                nid_res = store.resolve_node(nid_rot)
                mf.flush()

                # Compare sorted canonical keys & weights
                cpp_br = store.node_branches(nid_res)
                try:
                    cpp_pairs = [(st.canonical_key(), w) for w, st in cpp_br]
                finally:
                    for _, st in cpp_br:
                        st.close()
                ref_pairs = sorted(
                    [(_bqfe_key_bytes(q), w) for w, q in mf.b], key=lambda it: it[0]
                )
                assert [k for k, _ in cpp_pairs] == [k for k, _ in ref_pairs]
                for (_, wc), (_, wr) in zip(cpp_pairs, ref_pairs):
                    max_w_err = max(max_w_err, abs(wc - wr))
                    assert abs(wc - wr) < 1e-12

                d_cpp = store.node_dense(nid_res, "big").reshape((2,) * n)
                d_ref = mf.dense()
                err_d = float(np.max(np.abs(d_cpp - d_ref)))
                max_dense_err = max(max_dense_err, err_d)
                assert err_d < 1e-12
            finally:
                for st in cpp_states:
                    st.close()
                store.close()

        # 3. K1 canceling-branch pruning check
        store = bk.BranchStore(2)
        s0 = bk.BranchState.zero(2)
        s1 = bk.BranchState.zero(2)
        s2 = bk.BranchState.zero(2)
        try:
            s2.pauli(_int_to_words(1, 1), _int_to_words(0, 1), 0)  # |10>
            nid = store.node_from_branches(
                [1.0 + 0j, -(1.0 - 1e-15), 0.5 + 0j], [s0, s1, s2]
            )
            nb, _, _ = store.node_info(nid)
            assert nb == 1  # |00> pair canceled below 1e-13 relative threshold!
        finally:
            s0.close()
            s1.close()
            s2.close()
            store.close()

        # 4. Multi-word resolve (n = 65, 130) vs mfe.MF
        for n in (65, 130):
            nw = _nw(n)
            for _ in range(5):
                st = bk.BranchState.zero(n)
                q = bqfe.Q(n)
                store = bk.BranchStore(n)
                try:
                    for qb in (0, 63, 64, n - 1):
                        xw = _int_to_words(1 << qb, nw)
                        zw = np.zeros(nw, dtype=np.uint64)
                        st.qturn(xw, zw, 0)
                        q.qturn(1 << qb, 0, 0)
                    turns = _make_commuting_paulis_int(rng, n, 4)
                    ax_x = np.stack([_int_to_words(x, nw) for x, _, _, _ in turns])
                    ax_z = np.stack([_int_to_words(z, nw) for _, z, _, _ in turns])
                    ax_k = np.array([k for _, _, k, _ in turns], dtype=np.uint8)
                    ax_e = np.array([e for _, _, _, e in turns], dtype=np.uint8)
                    nid = store.node_from_branches([1.0 + 0j], [st], ax_x, ax_z, ax_k, ax_e)
                    rid = store.resolve_node(nid)
                    mf = mfe.MF(n, [[1.0 + 0j, q]], [], mfe.Cache())
                    mf.rotate([((x, z, k), e) for x, z, k, e in turns])
                    mf.flush()
                    cpp_br = store.node_branches(rid)
                    try:
                        cpp_pairs = [(s.canonical_key(), w) for w, s in cpp_br]
                    finally:
                        for _, s in cpp_br:
                            s.close()
                    ref_pairs = sorted(
                        [(_bqfe_key_bytes(qr), wr) for wr, qr in mf.b], key=lambda it: it[0]
                    )
                    assert [k for k, _ in cpp_pairs] == [k for k, _ in ref_pairs]
                    for (_, wc), (_, wr) in zip(cpp_pairs, ref_pairs):
                        max_w_err = max(max_w_err, abs(wc - wr))
                        assert abs(wc - wr) < 1e-12
                finally:
                    st.close()
                    store.close()

        assert max_w_err < 1e-12 and max_dense_err < 1e-12
        assert bk.BranchState.live_states() == 0
        assert bk.BranchStore.live_stores() == 0


def _replay_bsm_batch_on_cpp(
    H: Any,
    D: dict[str, Any],
    bqfe: Any,
    store: bk.BranchStore,
    *,
    bsm_end_nodes: np.ndarray | None = None,
    bsm_end_fx: np.ndarray | None = None,
    bsm_end_fz: np.ndarray | None = None,
) -> tuple[float, float, dict[str, int]]:
    """Replay `bsm.Machine` faulty shots onto C++ `BranchStore` and verify `p_true` + end states."""
    n = H.n
    nw = _nw(n)
    faulty = D["faulty"]
    Nf = len(faulty)
    shot, site, opt = D["shot"], D["site"], D["opt"]
    rec_bsm = D["rec"]
    ptrue_bsm = D["ptrue"]

    loc = np.searchsorted(faulty, shot)
    hop = H.site_op[site]
    order = np.argsort(hop, kind="stable")
    bnd = np.searchsorted(hop[order], np.arange(H.T + 1))
    kind, lxs, lzs, aux = H.opt_kind[opt], H.opt_lx[opt], H.opt_lz[opt], H.opt_aux[opt]

    fx = np.zeros(Nf, dtype=np.uint64)
    fz = np.zeros(Nf, dtype=np.uint64)
    last = np.zeros(Nf, dtype=np.uint64)
    lost = np.zeros(Nf, dtype=np.uint64)
    node = np.zeros(Nf, dtype=np.int64)

    max_p_err = 0.0
    meas_counts = {"def": 0, "abs": 0, "fork": 0}
    U64 = np.uint64
    ONE = U64(1)

    for t in range(H.T):
        op = H.ops[t]
        k = op[0]
        skip = None
        if k in ("T", "T_DAG", "CCZ"):
            if k == "CCZ":
                skip = (lost & H.opmask[t]) != 0
                if not skip.any():
                    skip = None
            sel_idx = np.arange(Nf) if skip is None else np.flatnonzero(~skip)
            if sel_idx.size > 0:
                sb = H._signs(t, fx[sel_idx], fz[sel_idx])
                keys = node[sel_idx] * 8 + sb
                uk, inv = np.unique(keys, return_inverse=True)
                ax = H.frame.axes[t]
                base_labs = [l for l, _ in ax]
                ax_x = np.stack([_int_to_words(l[0], nw) for l in base_labs])
                ax_z = np.stack([_int_to_words(l[1], nw) for l in base_labs])
                ax_e = np.array([e & 7 for _, e in ax], dtype=np.uint8)
                mk_kind = bk.MBR_KIND_CCZ if k == "CCZ" else bk.MBR_KIND_T
                new_nids = np.empty(uk.size, dtype=np.int64)
                for ui, kv in enumerate(uk):
                    nid_u = int(kv) >> 3
                    sb_u = int(kv) & 7
                    fl = H._flip(t, base_labs, sb_u)
                    ax_k = np.array([l[2] & 3 for l in fl], dtype=np.uint8)
                    new_nids[ui] = store.step_rotations(
                        nid_u, t + 1, mk_kind, sb_u, ax_x, ax_z, ax_k, ax_e
                    )
                node[sel_idx] = new_nids[inv]
        elif k in ("CX", "CZ"):
            skip = (lost & H.opmask[t]) != 0
            if skip.any():
                sel_idx = np.flatnonzero(skip)
                sb = H._signs(t, fx[sel_idx], fz[sel_idx])
                keys = node[sel_idx] * 8 + sb
                uk, inv = np.unique(keys, return_inverse=True)
                _, base_labs = H.frame.skip[t]
                qt_x = np.stack([_int_to_words(l[0], nw) for l in base_labs])
                qt_z = np.stack([_int_to_words(l[1], nw) for l in base_labs])
                new_nids = np.empty(uk.size, dtype=np.int64)
                for ui, kv in enumerate(uk):
                    nid_u = int(kv) >> 3
                    sb_u = int(kv) & 7
                    fl = H._flip(t, base_labs, sb_u)
                    qt_k = np.array([l[2] & 3 for l in fl], dtype=np.uint8)
                    new_nids[ui] = store.step_qturns(
                        nid_u, t + 1, bk.MBR_KIND_SKIP, sb_u, qt_x, qt_z, qt_k
                    )
                node[sel_idx] = new_nids[inv]
            else:
                skip = None
        elif k == "M":
            j = H.midx[t]
            q = U64(op[1])
            mx_i, mz_i, mk_i = H.frame.meas[t]
            mx_w, mz_w = _int_to_words(mx_i, nw), _int_to_words(mz_i, nw)
            un, inv = np.unique(node, return_inverse=True)
            tab = [
                store.step_meas(int(u), t + 1, bk.MBR_KIND_MEAS, mx_w, mz_w, int(mk_i))
                for u in un
            ]
            s_bit = (
                np.bitwise_count((fx & U64(mz_i)) ^ (fz & U64(mx_i))) & U64(1)
            ).astype(np.uint8)
            m_true = rec_bsm[:, j]
            inner_m = m_true ^ s_bit
            for i in range(Nf):
                p0, c0, c1, mt, abs_x, abs_z = tab[int(inv[i])]
                if mt == bk.MBR_MEAS_DEFINITE:
                    meas_counts["def"] += 1
                elif mt == bk.MBR_MEAS_ABSORBED:
                    meas_counts["abs"] += 1
                else:
                    meas_counts["fork"] += 1
                mi = int(inner_m[i])
                p_cpp = (1.0 - p0) if mi else p0
                err_p = abs(p_cpp - float(ptrue_bsm[i, j]))
                max_p_err = max(max_p_err, err_p)
                assert err_p < 1e-12
                if mt == bk.MBR_MEAS_ABSORBED and mi == 1:
                    fx[i] ^= U64(_words_to_int(abs_x))
                    fz[i] ^= U64(_words_to_int(abs_z))
                nxt = c1 if mi else c0
                assert nxt >= 0
                node[i] = nxt
            last = (last & ~(ONE << q)) | (m_true.astype(U64) << q)
        elif k == "CORR":
            q = U64(op[1])
            sel = ((last >> q) & ONE).astype(bool)
            if sel.any():
                cx, cz = H.corr_xz[t]
                fx[sel] ^= cx
                fz[sel] ^= cz
            lost &= ~(ONE << q)

        lo, hi = bnd[t], bnd[t + 1]
        if hi > lo:
            ix = order[lo:hi]
            if skip is not None:
                ix = ix[~skip[loc[ix]]]
            kk = kind[ix]
            P = ix[kk == 0]  # KIND_P
            if len(P):
                np.bitwise_xor.at(fx, loc[P], lxs[P])
                np.bitwise_xor.at(fz, loc[P], lzs[P])
            L = ix[kk == 1]  # KIND_L
            if len(L):
                np.bitwise_or.at(lost, loc[L], ONE << aux[L].astype(U64))

    max_state_err = 0.0
    if bsm_end_nodes is not None and bsm_end_fx is not None and bsm_end_fz is not None:
        for i in range(Nf):
            dfx = int(fx[i] ^ bsm_end_fx[i])
            dfz = int(fz[i] ^ bsm_end_fz[i])
            dfk = _herm_kappa(dfx, dfz, False)
            # Transform bsm's inner state by delta_F = F_cpp ^ F_bsm so physical states align:
            # F_cpp |node_cpp> == F_bsm |node_bsm>  <=>  |node_cpp> == (F_cpp ^ F_bsm) |node_bsm>
            e_bsm = H.nodes[int(bsm_end_nodes[i])]
            ref_pend: list[tuple[int, int, int, int]] = []
            ph_pend = 0
            for (px, pz, pk), pe in e_bsm.pend:
                pk_new = (pk + (2 if _anti_int(dfx, dfz, px, pz) else 0)) & 3
                pk_norm, neg = _norm_sign_plus_int(px, pz, pk_new)
                e_eff = (8 - pe) & 7 if neg else pe & 7
                if neg:
                    ph_pend = (ph_pend + pe) & 7
                ref_pend.append((px, pz, pk_norm, e_eff))

            # Compare pending turns (order-independent set of (x, z, k, e))
            px_c, pz_c, pk_c, pe_c = store.node_pending(int(node[i]))
            cpp_pend = [
                (_words_to_int(px_c[j]), _words_to_int(pz_c[j]), int(pk_c[j]), int(pe_c[j]))
                for j in range(len(pe_c))
            ]
            assert sorted(cpp_pend) == sorted(ref_pend)

            # Transform bsm branches by delta_F + phase_pend, canonicalize & merge
            ref_map: dict[bytes, complex] = {}
            ph_c = _OMEGA_P[ph_pend]
            for wb, qb in e_bsm.b:
                qc = qb.copy()
                sc = qc.pauli(dfx, dfz, dfk)
                sc *= qc.canon()
                key = _bqfe_key_bytes(qc)
                ref_map[key] = ref_map.get(key, 0.0j) + wb * sc * ph_c
            ref_pairs = sorted(
                ((k, v) for k, v in ref_map.items() if abs(v) > 1e-12), key=lambda it: it[0]
            )

            cpp_br = store.node_branches(int(node[i]))
            try:
                cpp_pairs = [(s.canonical_key(), w) for w, s in cpp_br]
            finally:
                for _, s in cpp_br:
                    s.close()

            assert [k for k, _ in cpp_pairs] == [k for k, _ in ref_pairs]
            w_cpp = np.array([w for _, w in cpp_pairs], dtype=np.complex128)
            w_ref = np.array([w for _, w in ref_pairs], dtype=np.complex128)
            err_w = _phase_aligned_diff(w_cpp, w_ref)
            max_state_err = max(max_state_err, err_w)
            assert err_w < 1e-12
    return max_p_err, max_state_err, meas_counts


def _run_bsm_recording_end_state(H: Any, N: int, rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run `bsm.Machine.run_batch(N, rng, probs=True)` and also capture final `(node, fx, fz)`."""
    # Re-run `run_batch` on a cloned rng state is cleanest, or mirror end-state extraction from
    # `H.run_batch` deterministically by capturing rng state before calling `H.run_batch`.
    state0 = rng.bit_generator.state
    _, D = H.run_batch(N, rng, probs=True)
    # Replay bsm's own transition table on the exact sampled events to extract final (node, fx, fz)
    rng_check = np.random.default_rng()
    rng_check.bit_generator.state = state0
    # Consume exact same sample + measurement draws via bsm's own arrays
    faulty = D["faulty"]
    Nf = len(faulty)
    shot, site, opt = D["shot"], D["site"], D["opt"]
    rec = D["rec"]
    loc = np.searchsorted(faulty, shot)
    hop = H.site_op[site]
    order = np.argsort(hop, kind="stable")
    bnd = np.searchsorted(hop[order], np.arange(H.T + 1))
    kind, lxs, lzs, aux = H.opt_kind[opt], H.opt_lx[opt], H.opt_lz[opt], H.opt_aux[opt]
    U64, ONE = np.uint64, np.uint64(1)
    fx, fz = np.zeros(Nf, U64), np.zeros(Nf, U64)
    last, lost = np.zeros(Nf, U64), np.zeros(Nf, U64)
    node = np.zeros(Nf, np.int64)
    for t in range(H.T):
        op = H.ops[t]
        k = op[0]
        skip = None
        if k in ("T", "T_DAG", "CCZ"):
            if k == "CCZ":
                skip = (lost & H.opmask[t]) != 0
                if not skip.any():
                    skip = None
            if skip is None:
                node = H._map(node * 8 + H._signs(t, fx, fz), t, H._rot)
            else:
                sel = ~skip
                node[sel] = H._map(node[sel] * 8 + H._signs(t, fx[sel], fz[sel]), t, H._rot)
        elif k in ("CX", "CZ"):
            skip = (lost & H.opmask[t]) != 0
            if skip.any():
                node[skip] = H._map(node[skip] * 8 + H._signs(t, fx[skip], fz[skip]), t, H._skip)
            else:
                skip = None
        elif k == "M":
            j = H.midx[t]
            q = U64(op[1])
            mx, mz = H.meas_xz[t]
            s_bit = (np.bitwise_count((fx & mz) ^ (fz & mx)) & ONE).astype(np.uint8)
            mi = (rec[:, j] ^ s_bit).astype(bool)
            un, inv = np.unique(node, return_inverse=True)
            tab = [H._meas(int(u), t) for u in un]
            c0 = np.array([r[1] for r in tab], np.int64)[inv]
            c1 = np.array([r[2] for r in tab], np.int64)[inv]
            node = np.where(mi, c1, c0)
            last = (last & ~(ONE << q)) | (rec[:, j].astype(U64) << q)
        elif k == "CORR":
            q = U64(op[1])
            sel = ((last >> q) & ONE).astype(bool)
            if sel.any():
                cx, cz = H.corr_xz[t]
                fx[sel] ^= cx
                fz[sel] ^= cz
            lost &= ~(ONE << q)
        lo, hi = bnd[t], bnd[t + 1]
        if hi > lo:
            ix = order[lo:hi]
            if skip is not None:
                ix = ix[~skip[loc[ix]]]
            kk = kind[ix]
            P = ix[kk == 0]
            if len(P):
                np.bitwise_xor.at(fx, loc[P], lxs[P])
                np.bitwise_xor.at(fz, loc[P], lzs[P])
            L = ix[kk == 1]
            if len(L):
                np.bitwise_or.at(lost, loc[L], ONE << aux[L].astype(U64))
    return D, node, fx, fz


def _dense_replay_small_circuit(
    H: Any,
    D: dict[str, Any],
    shot_idx: int,
) -> tuple[list[float], np.ndarray]:
    """Independent dense big-endian statevector replay of one faulty shot in `bsm` op format."""
    n = H.n
    faulty = D["faulty"]
    shot, site, opt = D["shot"], D["site"], D["opt"]
    rec = D["rec"][shot_idx]
    loc = np.searchsorted(faulty, shot)
    hop = H.site_op[site]
    order = np.argsort(hop, kind="stable")
    bnd = np.searchsorted(hop[order], np.arange(H.T + 1))

    # Map op -> list of (kind, lab_in_frame, aux) for this shot
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    lost = 0
    last = 0
    probs_out: list[float] = []
    # Work in the exact moving frame using H.frame pulled-back operators (same physical state up to U)!
    for t in range(H.T):
        op = H.ops[t]
        k = op[0]
        skipped = False
        if k in ("T", "T_DAG", "CCZ"):
            if k == "CCZ" and (lost & int(H.opmask[t])):
                skipped = True
            else:
                for (px, pz, pk), e in H.frame.axes[t]:
                    psi = _FallbackOracle.dense_turn_big(psi, n, px, pz, pk, e)
        elif k in ("CX", "CZ"):
            if lost & int(H.opmask[t]):
                skipped = True
                scal, labs = H.frame.skip[t]
                psi = psi * scal
                for px, pz, pk in labs:
                    psi = _FallbackOracle.dense_qturn_big(psi, n, px, pz, pk)
        elif k == "M":
            j = H.midx[t]
            q = op[1]
            mx, mz, mk = H.frame.meas[t]
            m = int(rec[j])
            norm, psi = _FallbackOracle.dense_project_big(psi, n, mx, mz, mk, m)
            probs_out.append(norm * norm)
            last = (last & ~(1 << q)) | (m << q)
        elif k == "CORR":
            q = op[1]
            if (last >> q) & 1:
                cx, cz, ck = H.frame.corr[t][0]
                psi = _FallbackOracle.dense_pauli_big(psi, n, cx, cz, ck)
            lost &= ~(1 << q)

        lo, hi = bnd[t], bnd[t + 1]
        if hi > lo and not skipped:
            for idx in order[lo:hi]:
                if int(loc[idx]) != shot_idx:
                    continue
                kk = int(H.opt_kind[opt[idx]])
                if kk == 0:  # KIND_P
                    lx, lz, lk = H.opt_lab[opt[idx]]
                    psi = _FallbackOracle.dense_pauli_big(psi, n, lx, lz, lk)
                elif kk == 1:  # KIND_L
                    lost |= 1 << int(H.opt_aux[opt[idx]])
    return probs_out, psi


def _make_random_bsm_circuit(
    rng: np.random.Generator, n: int
) -> tuple[list[tuple[Any, ...]], list[tuple[int, str, tuple[int, ...]]]]:
    """Build a random Clifford+T+CCZ+M+CORR+loss circuit whose noiseless M outcomes are 0."""
    # Partition qubits into data (0..n-2) and ancilla (n-1); apply random Cliffords/T/CCZ on data,
    # uncompute ancilla-coupled parity checks (or conjugate Z-checks by C -> C^dag) so noiseless
    # measurements are deterministic 0 while Pauli & loss faults produce absorbed & forked outcomes!
    ops: list[tuple[Any, ...]] = []
    sites: list[tuple[int, str, tuple[int, ...]]] = []
    anc = n - 1
    data = list(range(n - 1))

    # Stage 1: prepare magic + entangled state on data qubits
    for q in data:
        if rng.random() < 0.7:
            ops.append(("H", q))
            sites.append((len(ops) - 1, "d1", (q,)))
        if rng.random() < 0.6:
            ops.append(("T" if rng.random() < 0.5 else "T_DAG", q))
            sites.append((len(ops) - 1, "d1", (q,)))

    for _ in range(4):
        if len(data) >= 3 and rng.random() < 0.45:
            a, b, c = [int(x) for x in rng.choice(data, size=3, replace=False)]
            ops.append(("CCZ", a, b, c))
            sites.append((len(ops) - 1, "d3", (a, b, c)))
            sites.append((len(ops) - 1, "loss", (a,)))
        elif len(data) >= 2:
            a, b = [int(x) for x in rng.choice(data, size=2, replace=False)]
            ops.append(("CX" if rng.random() < 0.5 else "CZ", a, b))
            sites.append((len(ops) - 1, "d2", (a, b)))
            sites.append((len(ops) - 1, "loss", (a,)))
            sites.append((len(ops) - 1, "loss", (b,)))
        q = int(rng.choice(data))
        ops.append(("T" if rng.random() < 0.5 else "T_DAG", q))
        sites.append((len(ops) - 1, "d1", (q,)))

    # Stage 2: two ancilla check rounds using CX/CZ/CCZ pairs that cancel noiselessly on anc,
    # followed by M + CORR on anc (noiseless outcome is deterministic 0, faulty shots absorb & fork!)
    for _ in range(2):
        a = int(rng.choice(data))
        b = int(rng.choice([x for x in data if x != a])) if len(data) >= 2 else a
        ops.append(("H", anc))
        sites.append((len(ops) - 1, "d1", (anc,)))
        ops.append(("CZ", a, anc))
        sites.append((len(ops) - 1, "d2", (a, anc)))
        sites.append((len(ops) - 1, "loss", (a,)))
        sites.append((len(ops) - 1, "loss", (anc,)))
        if len(data) >= 2:
            ops.append(("CCZ", a, b, anc))
            sites.append((len(ops) - 1, "d3", (a, b, anc)))
            sites.append((len(ops) - 1, "loss", (b,)))
            ops.append(("CCZ", a, b, anc))
            sites.append((len(ops) - 1, "d3", (a, b, anc)))
        ops.append(("CZ", a, anc))
        sites.append((len(ops) - 1, "d2", (a, anc)))
        ops.append(("H", anc))
        sites.append((len(ops) - 1, "d1", (anc,)))
        ops.append(("M", anc, "Z"))
        sites.append((len(ops) - 1, "flip", (anc,)))
        ops.append(("CORR", anc))
        sites.append((len(ops) - 1, "x", (anc,)))
        # Extra T/H layer on data between checks
        q = int(rng.choice(data))
        ops.append(("H", q))
        sites.append((len(ops) - 1, "d1", (q,)))
        ops.append(("T", q))
        sites.append((len(ops) - 1, "d1", (q,)))

    return ops, sites


def _run_t14c_with_fallback(rng: np.random.Generator) -> None:
    """Fallback batched store vs dense walk on 5 random circuits when `scratch/magic_ref` absent."""
    for _ in range(5):
        n = 4
        nw = 1
        store = bk.BranchStore(n)
        try:
            nid = 0
            fx, fz = 0, 0
            vec = np.zeros(1 << n, dtype=np.complex128)
            vec[0] = 1.0
            for step in range(8):
                xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                k = _herm_kappa(xi, zi, bool(rng.integers(0, 2)))
                if step % 3 == 0:
                    ax_x = np.array([[xi]], dtype=np.uint64)
                    ax_z = np.array([[zi]], dtype=np.uint64)
                    k_eff = (k + (2 if _anti_int(fx, fz, xi, zi) else 0)) & 3
                    nid = store.step_rotations(
                        nid, step + 1, bk.MBR_KIND_T, 0, ax_x, ax_z,
                        np.array([k_eff], dtype=np.uint8), np.array([1], dtype=np.uint8),
                    )
                    vec = _FallbackOracle.dense_turn_big(vec, n, xi, zi, k, 1)
                elif step % 3 == 1:
                    qt_x = np.array([[xi]], dtype=np.uint64)
                    qt_z = np.array([[zi]], dtype=np.uint64)
                    k_eff = (k + (2 if _anti_int(fx, fz, xi, zi) else 0)) & 3
                    nid = store.step_qturns(
                        nid, step + 1, bk.MBR_KIND_SKIP, 0, qt_x, qt_z,
                        np.array([k_eff], dtype=np.uint8),
                    )
                    vec = _FallbackOracle.dense_qturn_big(vec, n, xi, zi, k)
                else:
                    p0, c0, c1, mt, abs_x, abs_z = store.step_meas(
                        nid, step + 1, bk.MBR_KIND_MEAS,
                        _int_to_words(xi, nw), _int_to_words(zi, nw), k,
                    )
                    s_bit = _anti_int(fx, fz, xi, zi)
                    p_true_0 = (1.0 - p0) if s_bit else p0
                    m_true = 0 if p_true_0 >= 0.5 else 1
                    inner_m = m_true ^ s_bit
                    norm_ref, vec = _FallbackOracle.dense_project_big(vec, n, xi, zi, k, m_true)
                    p_cpp = (1.0 - p0) if inner_m else p0
                    assert abs(p_cpp - norm_ref * norm_ref) < 1e-12
                    if mt == bk.MBR_MEAS_ABSORBED and inner_m == 1:
                        fx ^= int(abs_x[0])
                        fz ^= int(abs_z[0])
                    nid = c1 if inner_m else c0
            psi_cpp = _FallbackOracle.dense_pauli_big(
                store.node_dense(nid, "big"), n, fx, fz, _herm_kappa(fx, fz, False)
            )
            assert _phase_aligned_diff(psi_cpp, vec) < 1e-12
        finally:
            store.close()


def test_t14c_bsm_differential_authors_and_random_circuits() -> None:
    """T14c: Replay `bsm.Machine` on `authors` (n=20) & 20 random circuits vs C++ `BranchStore`."""
    rng = np.random.default_rng(20260503)
    with _magic_ref_modules() as refs:
        if refs is None:
            _run_t14c_with_fallback(rng)
            return
        bqfe, _, bsm, bhs = refs

        # 1. Authors circuit (n=20): 256 shots at normal cap + 128 shots at max_nodes=50
        n_auth, ops_auth, sites_auth = bhs.load("authors")
        probs_auth = bhs.kinds_probs(2e-3, 2e-3, 2e-3)

        H_main = bsm.Machine(n_auth, ops_auth, sites_auth, probs_auth, max_nodes=300000)
        D_main, end_n, end_fx, end_fz = _run_bsm_recording_end_state(
            H_main, 256, np.random.default_rng(17)
        )
        assert len(D_main["faulty"]) >= 100
        store = bk.BranchStore(n_auth)
        try:
            max_p_err, max_s_err, mcnt = _replay_bsm_batch_on_cpp(
                H_main, D_main, bqfe, store,
                bsm_end_nodes=end_n, bsm_end_fx=end_fx, bsm_end_fz=end_fz,
            )
            assert max_p_err < 1e-12 and max_s_err < 1e-12
            assert mcnt["def"] > 0 and mcnt["abs"] > 0 and mcnt["fork"] > 0
        finally:
            store.close()

        # Small-cap (max_nodes=50) multi-batch pass forcing memo resets in bsm & clear() in C++
        H_cap = bsm.Machine(n_auth, ops_auth, sites_auth, probs_auth, max_nodes=50)
        store_cap = bk.BranchStore(n_auth)
        try:
            cap_rng = np.random.default_rng(29)
            for batch_i in range(3):
                if batch_i > 0 and store_cap.node_count > 50:
                    store_cap.clear()
                D_c, en_c, efx_c, efz_c = _run_bsm_recording_end_state(H_cap, 64, cap_rng)
                mp, ms, _ = _replay_bsm_batch_on_cpp(
                    H_cap, D_c, bqfe, store_cap,
                    bsm_end_nodes=en_c, bsm_end_fx=efx_c, bsm_end_fz=efz_c,
                )
                assert mp < 1e-12 and ms < 1e-12
            assert H_cap.cnt["memo_resets"] >= 1
            assert store_cap.counters()["store_clears"] >= 1
        finally:
            store_cap.close()

        # 2. 20 small random Clifford+T+CCZ+M+CORR+loss circuits (n = 3..6) vs bsm AND dense oracle
        tot_abs = 0
        tot_fork = 0
        for ci in range(20):
            n = int(rng.integers(3, 7))
            ops_r, sites_r = _make_random_bsm_circuit(rng, n)
            probs_r = bhs.kinds_probs(0.08, 0.08, 0.08)
            H_r = bsm.Machine(n, ops_r, sites_r, probs_r, max_nodes=300000)
            D_r, en_r, efx_r, efz_r = _run_bsm_recording_end_state(
                H_r, 12, np.random.default_rng(1000 + ci)
            )
            store_r = bk.BranchStore(n)
            try:
                mp, ms, mcnt = _replay_bsm_batch_on_cpp(
                    H_r, D_r, bqfe, store_r,
                    bsm_end_nodes=en_r, bsm_end_fx=efx_r, bsm_end_fz=efz_r,
                )
                assert mp < 1e-12 and ms < 1e-12
                tot_abs += mcnt["abs"]
                tot_fork += mcnt["fork"]
                # Verify first 3 faulty shots against independent dense statevector oracle
                for si in range(min(3, len(D_r["faulty"]))):
                    p_dense, psi_dense = _dense_replay_small_circuit(H_r, D_r, si)
                    for j, pd in enumerate(p_dense):
                        assert abs(pd - float(D_r["ptrue"][si, j])) < 1e-12
                    # Compare physical dense state F_bsm |node_bsm> against psi_dense
                    e_bsm = H_r.nodes[int(en_r[si])]
                    psi_bsm = e_bsm.dense().reshape(-1)
                    psi_bsm = _FallbackOracle.dense_pauli_big(
                        psi_bsm, n, int(efx_r[si]), int(efz_r[si]),
                        _herm_kappa(int(efx_r[si]), int(efz_r[si]), False),
                    )
                    assert _phase_aligned_diff(psi_bsm, psi_dense) < 1e-12
            finally:
                store_r.close()
        assert tot_abs > 0 and tot_fork > 0
        assert bk.BranchState.live_states() == 0
        assert bk.BranchStore.live_stores() == 0


def test_t14_fallback_oracle_exercised_when_ref_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the fallback oracle via monkeypatched absent path so it never rots."""
    monkeypatch.setattr(
        sys.modules[__name__], "_MAGIC_REF_DIR", tmp_path / "nonexistent_magic_ref"
    )
    test_magic_ref_available_when_dir_present()
    test_t14a_bqfe_differential_small_and_multiword()
    test_t14b_mfe_differential_clifford_split_resolve_and_qturns()
    test_t14c_bsm_differential_authors_and_random_circuits()


# =============================================================================================
# Section C — T15 Hostile Inputs (All 46 Symbols), Multi-Word (130+, 193), OOM, Threads & Fork
# =============================================================================================


def test_t15_exhaustive_46_symbol_hostile_abi_sweep() -> None:
    """Probe EVERY single one of the 46 `mbr_*` symbols with hostile C ABI inputs + wrapper checks."""
    assert len(bk._SPECS) == 46
    lib = bk.get_branch_kernels_lib()
    probed: set[str] = set()

    def call(name: str, *args: Any) -> Any:
        probed.add(name)
        return getattr(lib, name)(*args)

    # 1-2. abi & live_states
    assert call("mbr_abi_v1") == 1
    assert call("mbr_live_states") == 0

    # Reusable valid scratch buffers for n = 5 (nw = 1, dirty tail bit = 1 << 5)
    n, nw = 5, 1
    dirty = np.array([1 << n], dtype=np.uint64)
    z0 = np.zeros(nw, dtype=np.uint64)
    x_q0 = np.array([1], dtype=np.uint64)
    z_q0 = np.array([1], dtype=np.uint64)
    out_p = ctypes.c_void_p()
    out_i32 = ctypes.c_int32()
    out_i64 = ctypes.c_int64()
    out_sz = ctypes.c_size_t()
    out_u32 = ctypes.c_uint32()
    out_u8 = ctypes.c_uint8()
    out_f64 = ctypes.c_double()
    out_c = np.zeros(2, dtype=np.float64)
    out_cnt = np.zeros(26, dtype=np.uint64)

    # 3. mbr_state_create_zero
    assert call("mbr_state_create_zero", 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_create_zero", bk._MAX_QUBITS + 1, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_create_zero", n, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_create_zero", n, ctypes.addressof(out_p)) == bk.MBR_OK
    h_st = int(out_p.value)

    # 4. mbr_state_from_arrays
    h_ok = np.zeros(nw, dtype=np.uint64)
    A_ok = np.array([[1]], dtype=np.uint64)
    ell_ok = np.array([0], dtype=np.uint8)
    G_ok = np.zeros((1, 1), dtype=np.uint8)
    piv_ok = np.array([0], dtype=np.int32)
    assert call("mbr_state_from_arrays", 0, 0, bk.addr(h_ok), 0, 0, 0, 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, n + 1, bk.addr(h_ok), bk.addr(A_ok), bk.addr(ell_ok), bk.addr(G_ok), bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 0, 0, 0, 0, 0, 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), 0, bk.addr(ell_ok), bk.addr(G_ok), bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 0, bk.addr(dirty), 0, 0, 0, 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), bk.addr(dirty.reshape(1, 1)), bk.addr(ell_ok), bk.addr(G_ok), bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), bk.addr(A_ok), np.array([4], np.uint8).ctypes.data, bk.addr(G_ok), bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), bk.addr(A_ok), bk.addr(ell_ok), np.array([[1]], np.uint8).ctypes.data, bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), bk.addr(A_ok), bk.addr(ell_ok), bk.addr(G_ok), np.array([n], np.int32).ctypes.data, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Asymmetric Gamma and duplicate piv (d = 2)
    A2 = np.array([[1], [2]], dtype=np.uint64)
    ell2 = np.array([0, 0], dtype=np.uint8)
    G_asym = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    piv2 = np.array([0, 1], dtype=np.int32)
    piv_dup = np.array([0, 0], dtype=np.int32)
    assert call("mbr_state_from_arrays", n, 2, bk.addr(h_ok), bk.addr(A2), bk.addr(ell2), bk.addr(G_asym), bk.addr(piv2), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_arrays", n, 2, bk.addr(h_ok), bk.addr(A2), bk.addr(ell2), np.zeros((2, 2), np.uint8).ctypes.data, bk.addr(piv_dup), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Create a valid d=1 state handle for export/other tests
    assert call("mbr_state_from_arrays", n, 1, bk.addr(h_ok), bk.addr(A_ok), bk.addr(ell_ok), bk.addr(G_ok), bk.addr(piv_ok), ctypes.addressof(out_p)) == bk.MBR_OK
    h_d1 = int(out_p.value)

    # 5-6. mbr_state_copy, mbr_state_destroy
    assert call("mbr_state_copy", 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_copy", h_st, 0) == bk.MBR_ERR_INVALID_ARG
    call("mbr_state_destroy", 0)  # null-safe no-op

    # 7. mbr_state_header
    assert call("mbr_state_header", 0, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_header", h_st, 0, ctypes.addressof(out_sz), ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG

    # 8. mbr_state_export_arrays
    assert call("mbr_state_export_arrays", 0, bk.addr(h_ok), 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_export_arrays", h_st, 0, 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_export_arrays", h_st, bk.addr(h_ok), 0, 0, 0, 0) == bk.MBR_OK  # d=0 allows null A..piv
    assert call("mbr_state_export_arrays", h_d1, bk.addr(h_ok), 0, bk.addr(ell_ok), bk.addr(G_ok), bk.addr(piv_ok)) == bk.MBR_ERR_INVALID_ARG

    # 9. mbr_state_pauli (nulls + dirty x AND dirty z)
    assert call("mbr_state_pauli", 0, bk.addr(z0), bk.addr(z0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_pauli", h_st, 0, bk.addr(z0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_pauli", h_st, bk.addr(z0), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_pauli", h_st, bk.addr(z0), bk.addr(z0), 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_pauli", h_st, bk.addr(dirty), bk.addr(z0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_pauli", h_st, bk.addr(z0), bk.addr(dirty), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG

    # 10. mbr_state_qturn (nulls + dirty x/z + non-Hermitian)
    assert call("mbr_state_qturn", 0, bk.addr(x_q0), bk.addr(z0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_qturn", h_st, bk.addr(dirty), bk.addr(z0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_qturn", h_st, bk.addr(z0), bk.addr(dirty), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_qturn", h_st, bk.addr(x_q0), bk.addr(z_q0), 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG  # Y needs kappa in {1,3}

    # 11. mbr_state_zproject
    assert call("mbr_state_zproject", 0, bk.addr(z_q0), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_zproject", h_st, bk.addr(dirty), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_zproject", h_st, bk.addr(z_q0), 2, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_zproject", h_st, bk.addr(z_q0), 0, 2, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG

    # 12. mbr_state_project
    assert call("mbr_state_project", 0, bk.addr(x_q0), bk.addr(z0), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_project", h_st, bk.addr(dirty), bk.addr(z0), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_project", h_st, bk.addr(z0), bk.addr(dirty), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_project", h_st, bk.addr(x_q0), bk.addr(z_q0), 0, 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_project", h_st, bk.addr(x_q0), bk.addr(z0), 0, 2, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG

    # 13. mbr_state_definite
    assert call("mbr_state_definite", 0, bk.addr(x_q0), bk.addr(z0), 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_definite", h_st, bk.addr(dirty), bk.addr(z0), 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_definite", h_st, bk.addr(z0), bk.addr(dirty), 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_definite", h_st, bk.addr(x_q0), bk.addr(z_q0), 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG

    # 14. mbr_state_generators
    gx_b = np.zeros((n, nw), dtype=np.uint64)
    gz_b = np.zeros((n, nw), dtype=np.uint64)
    gk_b = np.zeros(n, dtype=np.uint8)
    assert call("mbr_state_generators", 0, bk.addr(gx_b), bk.addr(gz_b), bk.addr(gk_b)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_generators", h_st, 0, bk.addr(gz_b), bk.addr(gk_b)) == bk.MBR_ERR_INVALID_ARG

    # 15-18. canon, canonical_key_size, canonical_key, from_canonical_key
    assert call("mbr_state_canon", 0, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_canon", h_st, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_canonical_key_size", 0, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_canonical_key_size", h_d1, ctypes.addressof(out_sz)) == bk.MBR_OK
    kbuf = np.zeros(int(out_sz.value), dtype=np.uint8)
    assert call("mbr_state_canonical_key", h_d1, 0, len(kbuf), ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_canonical_key", h_d1, bk.addr(kbuf), len(kbuf) - 1, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_canonical_key", h_d1, bk.addr(kbuf), len(kbuf), ctypes.addressof(out_sz)) == bk.MBR_OK
    assert call("mbr_state_from_canonical_key", 0, len(kbuf), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_from_canonical_key", bk.addr(kbuf), 4, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Non-canonical h (pivot bit 0 set in h)
    bad_h_key = bytearray(kbuf.tobytes())
    bad_h_key[8] |= 1
    bad_h_arr = np.frombuffer(bad_h_key, dtype=np.uint8)
    assert call("mbr_state_from_canonical_key", bk.addr(bad_h_arr), len(bad_h_arr), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Non-RCEF A (d=2, col 1 has col 0's pivot bit set)
    non_rcef = struct.pack("<IIQQQBB", n, 2, 0, 1, 3, 0, 0) + bytes([0])
    nr_arr = np.frombuffer(non_rcef, dtype=np.uint8)
    assert call("mbr_state_from_canonical_key", bk.addr(nr_arr), len(nr_arr), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Dirty Gamma padding bit (d=2 uses 1 bit of Gamma byte, bit 1..7 must be 0)
    bad_gpad = struct.pack("<IIQQQBB", n, 2, 0, 1, 2, 0, 0) + bytes([0x02])
    gp_arr = np.frombuffer(bad_gpad, dtype=np.uint8)
    assert call("mbr_state_from_canonical_key", bk.addr(gp_arr), len(gp_arr), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    # Header with d > n
    bad_dn = struct.pack("<II", 4, 5)
    dn_arr = np.frombuffer(bad_dn, dtype=np.uint8)
    assert call("mbr_state_from_canonical_key", bk.addr(dn_arr), len(dn_arr), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG

    # 19-20. overlap & same_phase (including n mismatch)
    assert call("mbr_state_create_zero", 3, ctypes.addressof(out_p)) == bk.MBR_OK
    h_n3 = int(out_p.value)
    assert call("mbr_state_overlap", 0, h_st, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_overlap", h_st, h_n3, bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_same_phase", 0, h_st, ctypes.addressof(out_i32), bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_same_phase", h_st, h_n3, ctypes.addressof(out_i32), bk.addr(out_c)) == bk.MBR_ERR_INVALID_ARG

    # 21. mbr_state_to_dense (bad big_endian, wrong len, n=27)
    dense_buf = np.zeros((1 << n) * 2, dtype=np.float64)
    assert call("mbr_state_to_dense", 0, 0, bk.addr(dense_buf), 1 << n) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_to_dense", h_st, 2, bk.addr(dense_buf), 1 << n) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_to_dense", h_st, 0, bk.addr(dense_buf), (1 << n) - 1) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_state_create_zero", 27, ctypes.addressof(out_p)) == bk.MBR_OK
    h_n27 = int(out_p.value)
    assert call("mbr_state_to_dense", h_n27, 0, bk.addr(dense_buf), 1 << 27) == bk.MBR_ERR_INVALID_ARG

    # 22-23. mbr_is_clifford & mbr_choose_split (overflow, zero parity, max_depth > 20)
    p_ok = np.array([[1], [3]], dtype=np.uint64)
    p_zero = np.array([[0]], dtype=np.uint64)
    lam_out = np.zeros(1, dtype=np.uint64)
    assert call("mbr_is_clifford", bk.addr(p_ok), 2, 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_is_clifford", bk.addr(p_ok), 2, (1 << 62), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_is_clifford", 0, 0, (1 << 62), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_is_clifford", 0, 1, 1, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_is_clifford", bk.addr(p_zero), 1, 1, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_is_clifford", bk.addr(p_ok), (1 << 62) + 7, 2, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", 0, 2, 1, 5, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_ok), 0, 1, 5, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_ok), 2, 0, 5, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_ok), 2, (1 << 62), 5, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_zero), 1, 1, 5, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_ok), 2, 1, 21, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_choose_split", bk.addr(p_ok), 2, 1, 0xFFFFFFFF, bk.addr(lam_out), ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG

    # 24-28. live_stores, store_create, store_destroy, store_clear, store_set_max_components
    assert call("mbr_live_stores") == 0
    assert call("mbr_store_create", 0, 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_create", bk._MAX_QUBITS + 1, 0, ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_create", n, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_create", n, 0, ctypes.addressof(out_p)) == bk.MBR_OK
    h_store = int(out_p.value)
    call("mbr_store_destroy", 0)  # null-safe no-op
    assert call("mbr_store_clear", 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_set_max_components", 0, 4) == bk.MBR_ERR_INVALID_ARG

    # 29-33. node_count, node_info, node_branch, node_pending, node_dense
    assert call("mbr_store_node_count", 0, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_count", h_store, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_info", 0, 0, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_u32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_info", h_store, -1, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_u32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_info", h_store, 99, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_u32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_info", h_store, (1 << 63) - 1, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_u32)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_branch", 0, 0, 0, bk.addr(out_c), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_branch", h_store, 0, 1, bk.addr(out_c), ctypes.addressof(out_p)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_pending", 0, 0, 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_pending", h_store, -1, 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_pending", h_store, 0, 0, 0, 0, 0) == bk.MBR_OK  # np==0 allows null
    assert call("mbr_store_node_dense", 0, 0, 0, bk.addr(dense_buf), 1 << n) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_dense", h_store, -1, 0, bk.addr(dense_buf), 1 << n) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_dense", h_store, 0, 2, bk.addr(dense_buf), 1 << n) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_dense", h_store, 0, 0, bk.addr(dense_buf), (1 << n) - 1) == bk.MBR_ERR_INVALID_ARG
    # n=27 store rejects node_dense
    assert call("mbr_store_create", 27, 0, ctypes.addressof(out_p)) == bk.MBR_OK
    h_st27 = int(out_p.value)
    assert call("mbr_store_node_dense", h_st27, 0, 0, bk.addr(dense_buf), 1 << 27) == bk.MBR_ERR_INVALID_ARG
    call("mbr_store_destroy", h_st27)

    # 34. mbr_store_node_from_branches (NaN/Inf, n mismatch, overflow, dirty pend, non-commuting pend, -4 budget)
    w_ok = np.array([1.0, 0.0], dtype=np.float64)
    w_nan = np.array([math.nan, 0.0], dtype=np.float64)
    w_inf = np.array([math.inf, 0.0], dtype=np.float64)
    ptrs_ok = np.array([h_st], dtype=np.uintp)
    ptrs_bad_n = np.array([h_n3], dtype=np.uintp)
    ptrs_null = np.array([0], dtype=np.uintp)
    assert call("mbr_store_node_from_branches", 0, 1, bk.addr(w_ok), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, 0, bk.addr(w_ok), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, (1 << 62) + 7, bk.addr(w_ok), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, 1, bk.addr(w_nan), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, 1, bk.addr(w_inf), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, 1, bk.addr(w_ok), bk.addr(ptrs_bad_n), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_node_from_branches", h_store, 1, bk.addr(w_ok), bk.addr(ptrs_null), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    # -4 budget check on 2 distinct states with max_components=1
    assert call("mbr_store_set_max_components", h_store, 1) == bk.MBR_OK
    w2 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    p2 = np.array([h_st, h_d1], dtype=np.uintp)
    assert call("mbr_store_node_from_branches", h_store, 2, bk.addr(w2), bk.addr(p2), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_BUDGET
    assert call("mbr_store_set_max_components", h_store, 0) == bk.MBR_OK
    # Non-unit norm node -> triggers -3 float drift on forked step_meas!
    w_drift = np.array([2.0, 0.0], dtype=np.float64)
    assert call("mbr_store_node_from_branches", h_store, 1, bk.addr(w_drift), bk.addr(ptrs_ok), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_OK
    nid_drift = int(out_i64.value)
    # Create a node with 2 branches and non-commuting measurement to avoid absorption & hit forked norm2 check
    w_drift2 = np.array([2.0, 0.0, 0.5, 0.0], dtype=np.float64)
    assert call("mbr_store_node_from_branches", h_store, 2, bk.addr(w_drift2), bk.addr(p2), 0, 0, 0, 0, 0, ctypes.addressof(out_i64)) == bk.MBR_OK
    nid_drift2 = int(out_i64.value)

    # 35. mbr_store_resolve_node
    assert call("mbr_store_resolve_node", 0, 0, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_resolve_node", h_store, -1, ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_resolve_node", h_store, 0, 0) == bk.MBR_ERR_INVALID_ARG

    # 36. mbr_store_step_rotations (INCLUDING cache-hit validation bypass regression!)
    ax_x = np.array([[1]], dtype=np.uint64)
    ax_z = np.array([[0]], dtype=np.uint64)
    ax_k = np.array([0], dtype=np.uint8)
    ax_e = np.array([1], dtype=np.uint8)
    # Seed cache entry (node=0, eid=77, kind=T, sb=0)
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 1, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_OK
    nid_pend = int(out_i64.value)
    # Verify node_pending rejects null output when np > 0
    assert call("mbr_store_node_pending", h_store, nid_pend, 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    # CACHE-HIT HOSTILE PROBES on exact cached key (0, 77, T, 0): MUST ALL RETURN -1!
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 1, 0, bk.addr(ax_z), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 1, bk.addr(dirty.reshape(1, 1)), bk.addr(ax_z), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 1, bk.addr(ax_x), bk.addr(dirty.reshape(1, 1)), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 1, bk.addr(ax_x), bk.addr(ax_x), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG  # non-Hermitian Y with k=0
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, (1 << 62) + 7, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    # Non-commuting pair on cached key (0, 77, T, 0): X0 and Z0 anticommute!
    nc_x = np.array([[1], [0]], dtype=np.uint64)
    nc_z = np.array([[0], [1]], dtype=np.uint64)
    nc_k = np.array([0, 0], dtype=np.uint8)
    nc_e = np.array([1, 1], dtype=np.uint8)
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_T, 0, 2, bk.addr(nc_x), bk.addr(nc_z), bk.addr(nc_k), bk.addr(nc_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    # Wrong kind
    assert call("mbr_store_step_rotations", h_store, 0, 77, bk.MBR_KIND_SKIP, 0, 1, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), bk.addr(ax_e), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG

    # 37. mbr_store_step_qturns (INCLUDING cache-hit validation bypass regression!)
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, 1, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_OK
    # CACHE-HIT HOSTILE PROBES on exact cached key (0, 88, SKIP, 0): MUST ALL RETURN -1!
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, 1, 0, bk.addr(ax_z), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, 1, bk.addr(dirty.reshape(1, 1)), bk.addr(ax_z), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, 1, bk.addr(ax_x), bk.addr(dirty.reshape(1, 1)), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, 1, bk.addr(ax_x), bk.addr(ax_x), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_SKIP, 0, (1 << 62) + 7, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_qturns", h_store, 0, 88, bk.MBR_KIND_T, 0, 1, bk.addr(ax_x), bk.addr(ax_z), bk.addr(ax_k), ctypes.addressof(out_i64)) == bk.MBR_ERR_INVALID_ARG

    # 38-39. mbr_store_step_meas & mbr_store_step_peek (nulls, kinds, dirty x/z, non-Hermitian, -3 drift)
    c0_o, c1_o = ctypes.c_int64(), ctypes.c_int64()
    abs_x, abs_z = np.zeros(nw, dtype=np.uint64), np.zeros(nw, dtype=np.uint64)
    assert call("mbr_store_step_meas", 0, 0, 1, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_meas", h_store, 0, 1, bk.MBR_KIND_T, bk.addr(z0), bk.addr(z_q0), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_meas", h_store, 0, 1, bk.MBR_KIND_MEAS, bk.addr(dirty), bk.addr(z_q0), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_meas", h_store, 0, 1, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(dirty), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_meas", h_store, 0, 1, bk.MBR_KIND_MEAS, bk.addr(x_q0), bk.addr(z_q0), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_INVALID_ARG
    # -3 float drift on unnormalized forked state (nid_drift2 has |0> and |+>, measuring Z0 is forked!)
    assert call("mbr_store_step_meas", h_store, nid_drift2, 999, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, ctypes.addressof(out_f64), ctypes.addressof(c0_o), ctypes.addressof(c1_o), ctypes.addressof(out_u8), bk.addr(abs_x), bk.addr(abs_z)) == bk.MBR_ERR_FLOAT_DRIFT
    assert call("mbr_store_step_peek", 0, 0, 1, bk.addr(z0), bk.addr(z_q0), 0, ctypes.addressof(out_f64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_peek", h_store, 0, 1, bk.addr(dirty), bk.addr(z_q0), 0, ctypes.addressof(out_f64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_peek", h_store, 0, 1, bk.addr(z0), bk.addr(dirty), 0, ctypes.addressof(out_f64)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_step_peek", h_store, nid_drift2, 999, bk.addr(z0), bk.addr(z_q0), 0, ctypes.addressof(out_f64)) == bk.MBR_ERR_FLOAT_DRIFT
    assert nid_drift >= 1

    # 40-41. mbr_store_counters & mbr_store_branches_hist
    assert call("mbr_store_counters", 0, bk.addr(out_cnt)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_counters", h_store, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_counters", h_store, bk.addr(out_cnt)) == bk.MBR_OK
    assert call("mbr_store_branches_hist", 0, 0, 0, 0, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_branches_hist", h_store, 0, 0, 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_store_branches_hist", h_store, 0, 0, 0, ctypes.addressof(out_sz)) == bk.MBR_OK
    assert int(out_sz.value) >= 2
    hk = np.zeros(1, dtype=np.uint64)
    hc = np.zeros(1, dtype=np.uint64)
    assert call("mbr_store_branches_hist", h_store, bk.addr(hk), bk.addr(hc), 1, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG  # cap < n_entries
    assert call("mbr_store_branches_hist", h_store, 0, bk.addr(hc), int(out_sz.value), ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG

    # 42-46. WP6 mbr_batch_step_* hostile checks (nulls, kinds, >30 gens/qturns, dirty tails, OOB nodes,
    #        mismatched kick_x/z, update_log_mass>1, -3 drift, -4 budget atomicity)
    b_fx = np.zeros((4, nw), dtype=np.uint64)
    b_fz = np.zeros((4, nw), dtype=np.uint64)
    b_fx_dirty = b_fx.copy()
    b_fx_dirty[2, 0] = np.uint64(1 << n)
    b_nodes = np.zeros(4, dtype=np.int64)
    b_nodes_oob = np.array([0, 99999, 0, 0], dtype=np.int64)
    b_peaks = np.ones(4, dtype=np.int16)
    b_lm = np.zeros(4, dtype=np.float64)
    b_u = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    b_des = np.array([0, 1, 0, 1], dtype=np.uint8)
    b_sel = np.array([1, 0, 1, 0], dtype=np.uint8)
    b_p1 = np.zeros(4, dtype=np.float64)
    b_tm = np.zeros(4, dtype=np.uint8)
    b_pc = np.zeros(4, dtype=np.float64)
    b_fm = np.array([1], dtype=np.uint32)
    b_gx = np.array([[1]], dtype=np.uint64)
    b_gz = np.array([[0]], dtype=np.uint64)
    b_fail = ctypes.c_int64(-99)
    ax_k_ok = np.array([0], dtype=np.uint8)
    ax_e_ok = np.array([1], dtype=np.uint8)

    # 42. mbr_batch_step_rotations
    assert call("mbr_batch_step_rotations", 0, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_MEAS, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 31, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(dirty.reshape(1, 1)), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx_dirty), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_oob), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_rotations", h_store, 4, 0, 1, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(ax_e_ok), bk.addr(b_fm), 0, bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG

    # 43. mbr_batch_step_qturns
    assert call("mbr_batch_step_qturns", 0, 4, 0, 2, bk.MBR_KIND_SKIP, 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_qturns", h_store, 4, 0, 2, bk.MBR_KIND_T, 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_qturns", h_store, 4, 0, 2, bk.MBR_KIND_SKIP, 31, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_qturns", h_store, 4, 0, 2, bk.MBR_KIND_SKIP, 1, bk.addr(b_gx), bk.addr(b_gx), bk.addr(ax_k_ok), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG  # non-Hermitian
    assert call("mbr_batch_step_qturns", h_store, 4, 0, 2, bk.MBR_KIND_SKIP, 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(b_fx_dirty), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_qturns", h_store, 4, 0, 2, bk.MBR_KIND_SKIP, 1, bk.addr(b_gx), bk.addr(b_gz), bk.addr(ax_k_ok), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_oob), bk.addr(b_peaks)) == bk.MBR_ERR_INVALID_ARG

    # 44. mbr_batch_step_meas
    assert call("mbr_batch_step_meas", 0, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), bk.addr(b_tm), bk.addr(b_pc)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_T, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0, 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), bk.addr(x_q0), 0, 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0, 0) == bk.MBR_ERR_INVALID_ARG  # mismatched kick_x/z
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), bk.addr(dirty), bk.addr(z0), 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0, 0) == bk.MBR_ERR_INVALID_ARG  # dirty kick
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 2, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0, 0) == bk.MBR_ERR_INVALID_ARG  # bad update_log_mass
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), 0, 0, 0) == bk.MBR_ERR_INVALID_ARG  # null log_mass when update=1
    assert call("mbr_batch_step_meas", h_store, 4, 0, 3, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 1, bk.addr(b_fx_dirty), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0, 0) == bk.MBR_ERR_INVALID_ARG
    # -3 float drift on shot 2 leaves all buffers untouched (even shot 0 which succeeded in Pass 1!)
    b_nodes_drift = np.array([0, 0, nid_drift2, 0], dtype=np.int64)
    b_nodes_drift_orig = b_nodes_drift.copy()
    b_peaks_orig = b_peaks.copy()
    b_lm_orig = b_lm.copy()
    assert call("mbr_batch_step_meas", h_store, 4, 0, 9991, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_u), 0, 0, 0, 1, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_drift), bk.addr(b_peaks), bk.addr(b_lm), bk.addr(b_tm), bk.addr(b_pc)) == bk.MBR_ERR_FLOAT_DRIFT
    assert np.array_equal(b_nodes_drift, b_nodes_drift_orig)
    assert np.array_equal(b_peaks, b_peaks_orig)
    assert np.array_equal(b_lm, b_lm_orig)

    # 45. mbr_batch_step_peek
    assert call("mbr_batch_step_peek", 0, 4, 4, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_p1)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 4, bk.addr(dirty), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_p1)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 4, bk.addr(x_q0), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_p1)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 4, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_fx_dirty), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_p1)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 4, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_oob), bk.addr(b_p1)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 4, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_peek", h_store, 4, 9992, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_drift), bk.addr(b_p1)) == bk.MBR_ERR_FLOAT_DRIFT

    # 46. mbr_batch_step_postselect
    assert call("mbr_batch_step_postselect", 0, 4, bk.addr(b_sel), 5, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), ctypes.addressof(b_fail)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_postselect", h_store, 4, bk.addr(b_sel), 5, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), 0) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_postselect", h_store, 4, bk.addr(b_sel), 5, bk.addr(dirty), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), ctypes.addressof(b_fail)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_postselect", h_store, 4, bk.addr(b_sel), 5, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx_dirty), bk.addr(b_fz), bk.addr(b_nodes), bk.addr(b_peaks), bk.addr(b_lm), ctypes.addressof(b_fail)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_postselect", h_store, 4, 0, 5, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_oob), bk.addr(b_peaks), bk.addr(b_lm), ctypes.addressof(b_fail)) == bk.MBR_ERR_INVALID_ARG
    assert call("mbr_batch_step_postselect", h_store, 4, 0, 9993, bk.addr(z0), bk.addr(z_q0), 0, bk.addr(b_des), bk.addr(b_fx), bk.addr(b_fz), bk.addr(b_nodes_drift), bk.addr(b_peaks), bk.addr(b_lm), ctypes.addressof(b_fail)) == bk.MBR_ERR_FLOAT_DRIFT

    # Clean up raw handles
    for h in (h_st, h_d1, h_n3, h_n27):
        call("mbr_state_destroy", h)
    call("mbr_store_destroy", h_store)

    # Assert ALL 46 symbols were probed!
    assert probed == {name for name, _, _ in bk._SPECS}
    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0

    # Python wrapper checks for read-only arrays & budget atomicity on batch_step_*
    with bk.BranchStore(4, max_components=1) as bs_ro:
        ro_nodes = np.zeros(2, dtype=np.int64)
        ro_nodes.flags.writeable = False
        ok_nodes = np.zeros(2, dtype=np.int64)
        ok_peaks = np.ones(2, dtype=np.int16)
        ro_peaks = ok_peaks.copy()
        ro_peaks.flags.writeable = False
        ok_fx = np.zeros((2, 1), dtype=np.uint64)
        ok_fz = np.zeros((2, 1), dtype=np.uint64)
        ro_fx = ok_fx.copy()
        ro_fx.flags.writeable = False
        ok_lm = np.zeros(2, dtype=np.float64)
        ok_u = np.array([0.2, 0.7], dtype=np.float64)
        with pytest.raises(ValueError, match="writeable"):
            bs_ro.batch_step_rotations(
                1, bk.MBR_KIND_T, b_gx, b_gz, b_gx, b_gz, ax_k_ok, ax_e_ok, b_fm,
                ok_fx, ok_fz, ro_nodes, ok_peaks,
            )
        with pytest.raises(ValueError, match="writeable"):
            bs_ro.batch_step_qturns(
                2, bk.MBR_KIND_SKIP, b_gx, b_gz, ax_k_ok, ok_fx, ok_fz, ok_nodes, ro_peaks,
            )
        with pytest.raises(ValueError, match="writeable"):
            bs_ro.batch_step_meas(
                3, bk.MBR_KIND_MEAS, z0, z_q0, 0, ok_u, ro_fx, ok_fz, ok_nodes, ok_peaks,
                log_mass=ok_lm,
            )
        # Budget (-4) atomicity: shot 0 succeeds (commuting), shot 1 triggers resolve (>1 branch -> -4)
        # Both shots' nodes and peaks must remain 100% untouched!
        n_x = bs_ro.step_rotations(
            0, 10, bk.MBR_KIND_T, 0, b_gx, b_gz, ax_k_ok, ax_e_ok
        )
        bud_nodes = np.array([0, n_x], dtype=np.int64)
        bud_peaks = np.array([1, 1], dtype=np.int16)
        with pytest.raises(bk.BranchBudgetError):
            bs_ro.batch_step_rotations(
                11, bk.MBR_KIND_T, b_gz, b_gx, b_gz, b_gx, ax_k_ok, ax_e_ok, b_fm,
                ok_fx, ok_fz, bud_nodes, bud_peaks,
            )
        assert np.array_equal(bud_nodes, np.array([0, n_x], dtype=np.int64))
        assert np.array_equal(bud_peaks, np.array([1, 1], dtype=np.int16))

    # Python wrapper tightening checks: np.bool_ rejection & (0, 0) 2D parities rejection
    with pytest.raises(TypeError):
        bk.BranchState.zero(np.bool_(True))
    with pytest.raises(TypeError):
        bk.BranchState.from_arrays(
            2,
            np.array([np.bool_(True)], dtype=object),
            np.empty((0, 1), np.uint64),
            np.empty(0, np.uint8),
            np.empty((0, 0), np.uint8),
            np.empty(0, np.int32),
        )
    with pytest.raises(TypeError):
        bk.is_clifford([np.bool_(True)])
    with pytest.raises(ValueError):
        bk.is_clifford(np.empty((0, 0), dtype=np.uint64))
    with pytest.raises(ValueError):
        bk.choose_split(np.empty((0, 0), dtype=np.uint64))
    with pytest.raises(ValueError):
        bk.choose_split([1, 3], max_depth=21)


def test_t15_multiword_state_and_store_up_to_193() -> None:
    """T15: Multi-word stress for `n in {63, 64, 65, 130, 193}` with `d = 65, 70, 130, 193`."""
    rng = np.random.default_rng(20260515)
    for n, target_d in ((63, 63), (64, 64), (65, 65), (130, 70), (130, 130), (193, 193)):
        nw = _nw(n)
        st = bk.BranchState.zero(n)
        store = bk.BranchStore(n)
        try:
            # Build exact d = target_d (including d = 193 spanning 4 words!)
            for qb in range(target_d):
                st.qturn(_int_to_words(1 << qb, nw), np.zeros(nw, np.uint64), 0)
            assert st.d == target_d

            # Cross-word pauli, qturn, zproject, project, definite, generators, canon, key roundtrip
            x_span = (1 << 0) | (1 << (n // 2)) | (1 << (n - 1))
            z_span = (1 << (n - 1)) | (1 << 1)
            k_span = _herm_kappa(x_span, z_span, False)
            xw, zw = _int_to_words(x_span, nw), _int_to_words(z_span, nw)
            assert abs(st.pauli(xw, zw, k_span)) == pytest.approx(1.0, abs=1e-12)
            assert abs(st.qturn(xw, zw, k_span)) == pytest.approx(1.0, abs=1e-12)
            assert st.definite(xw, zw, k_span) in (-1, 0, 1)
            gx, gz, gk = st.generators()
            assert gx.shape == (n, nw) and gz.shape == (n, nw) and gk.shape == (n,)
            assert _gf2_rank_pauli_gens(gx, gz, n) == n
            for gi in (0, n // 3, n - 1):
                assert st.definite(gx[gi], gz[gi], int(gk[gi])) == 1

            st.canon()
            key = st.canonical_key()
            st_rt = bk.BranchState.from_canonical_key(key)
            try:
                eq, ratio = st.same_phase(st_rt)
                assert eq is True and abs(ratio - 1.0) < 1e-12
                assert abs(st.overlap(st_rt) - 1.0) < 1e-12
                f_zp = st_rt.zproject(_int_to_words(1 << (n - 1), nw), 0, 0)
                assert abs(f_zp) > 0.0
                f_pr = st_rt.project(xw, zw, k_span, 0)
                assert abs(f_pr) >= 0.0
            finally:
                st_rt.close()

            # BranchStore multi-word transitions: node_from_branches, step_rotations, step_qturns,
            # resolve_node across word boundaries, and step_meas (definite, absorbed across words 0..nw-1, forked)
            turns = _make_commuting_paulis_int(rng, n, 3)
            ax_x = np.stack([_int_to_words(x, nw) for x, _, _, _ in turns])
            ax_z = np.stack([_int_to_words(z, nw) for _, z, _, _ in turns])
            ax_k = np.array([k for _, _, k, _ in turns], dtype=np.uint8)
            ax_e = np.array([e for _, _, _, e in turns], dtype=np.uint8)

            # Build a multi-word state stabilized by X_0 X_{n-1} (spanning word 0 and word nw-1)
            # and apply T on Z_0 + Z_{n-1} (which commutes with X_0 X_{n-1}?! No, X_0 X_{n-1} commutes with Z_0 Z_{n-1}!)
            st_mw = bk.BranchState.zero(n)
            try:
                # Start from |0...0>, measure/project onto X_0 X_{n-1} = +1 -> GHZ pair on (0, n-1)
                pair_x = (1 << 0) | (1 << (n - 1))
                pair_z = (1 << 0) | (1 << (n - 1))
                st_mw.project(_int_to_words(pair_x, nw), np.zeros(nw, np.uint64), 0, 0)
                # Verify definite on X_0 X_{n-1} (+1) and Z_0 Z_{n-1} (+1)
                assert st_mw.definite(_int_to_words(pair_x, nw), np.zeros(nw, np.uint64), 0) == 1
                assert st_mw.definite(np.zeros(nw, np.uint64), _int_to_words(pair_z, nw), 0) == 1

                nid_mw = store.node_from_branches([1.0 + 0j], [st_mw])
                # Add a pending T on X_1 (commutes with X_0 X_{n-1} and Z_0)
                t_x = _int_to_words(1 << 1, nw).reshape(1, nw)
                t_z = np.zeros((1, nw), dtype=np.uint64)
                nid_t = store.step_rotations(
                    nid_mw, 1, bk.MBR_KIND_T, 0, t_x, t_z,
                    np.array([0], np.uint8), np.array([1], np.uint8),
                )
                # Resolve produces 2 branches (|0...> and |010...>), BOTH stabilized by X_0 X_{n-1}!
                nid_r = store.resolve_node(nid_t)
                nb_r, _, _ = store.node_info(nid_r)
                assert nb_r == 2

                # Definite measurement on Z_0 Z_{n-1} -> p0 == 1.0
                p0_d, c0_d, c1_d, mt_d, _, _ = store.step_meas(
                    nid_r, 2, bk.MBR_KIND_MEAS, np.zeros(nw, np.uint64), _int_to_words(pair_z, nw), 0
                )
                assert mt_d == bk.MBR_MEAS_DEFINITE and p0_d == 1.0 and c0_d == nid_r and c1_d == -1
                assert store.step_peek(
                    nid_r, 2, np.zeros(nw, np.uint64), _int_to_words(pair_z, nw), 0
                ) == 1.0

                # Multi-word GF(2) absorption: measure Z_{n-1} (anticommutes with common stabilizer X_0 X_{n-1}!)
                mz_hi = _int_to_words(1 << (n - 1), nw)
                p0_a, c0_a, c1_a, mt_a, abs_x, abs_z = store.step_meas(
                    nid_r, 3, bk.MBR_KIND_MEAS, np.zeros(nw, np.uint64), mz_hi, 0
                )
                assert mt_a == bk.MBR_MEAS_ABSORBED and p0_a == 0.5 and c0_a == c1_a
                # Verify Q_bar anticommutes with Z_{n-1} (so bit n-1 of abs_x is 1!) and spans word 0 & nw-1
                abs_x_i = _words_to_int(abs_x)
                abs_z_i = _words_to_int(abs_z)
                assert _anti_int(abs_x_i, abs_z_i, 0, 1 << (n - 1)) == 1
                assert (abs_x_i >> (n - 1)) & 1 == 1
                assert (abs_x_i & 1) == 1  # spans word 0 and word nw-1!
                assert abs_x.shape == (nw,) and abs_z.shape == (nw,)

                # Forked measurement: measure Z_1 on nid_r (superposition after T on X_1)
                mz_1 = _int_to_words(1 << 1, nw)
                # Apply qturn on Y_1 first so Z_1 has nontrivial non-50/50 or forked outcome
                nid_q = store.step_qturns(
                    nid_r, 4, bk.MBR_KIND_SKIP, 0,
                    _int_to_words(1 << 1, nw).reshape(1, nw),
                    _int_to_words(1 << 1, nw).reshape(1, nw),
                    np.array([1], np.uint8),
                )
                p0_f, c0_f, c1_f, mt_f, _, _ = store.step_meas(
                    nid_q, 5, bk.MBR_KIND_MEAS, np.zeros(nw, np.uint64), mz_1, 0
                )
                assert mt_f == bk.MBR_MEAS_FORKED and 0.0 < p0_f < 1.0
                assert c0_f >= 0 and c1_f >= 0 and c0_f != c1_f

                # Also exercise general commuting turns resolve on multi-word state
                nid_gen = store.step_rotations(
                    nid_mw, 6, bk.MBR_KIND_T, 0, ax_x, ax_z, ax_k, ax_e
                )
                nid_gen_r = store.resolve_node(nid_gen)
                assert nid_gen_r >= 0
                store.clear()
                assert store.node_count == 1
            finally:
                st_mw.close()
        finally:
            st.close()
            store.close()

    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0


# =============================================================================================
# Genuine C++ OOM Harness (`LD_PRELOAD` `operator new` thread-local countdown in subprocess)
# =============================================================================================

_FI_CPP = r"""
#include <cstdlib>
#include <new>
static thread_local long g_countdown = -1;
static thread_local long g_count = 0;
extern "C" void fi_arm(long n) { g_countdown = n; g_count = 0; }
extern "C" long fi_count() { return g_count; }
extern "C" void fi_disarm() { g_countdown = -1; }
static void* alloc_impl(size_t sz) {
    g_count++;
    if (g_countdown == 0) { g_countdown = -1; throw std::bad_alloc(); }
    if (g_countdown > 0) --g_countdown;
    void* p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new(size_t sz) { return alloc_impl(sz); }
void* operator new[](size_t sz) { return alloc_impl(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
"""

_OOM_CHILD = r"""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
fi = ctypes.CDLL(sys.argv[2])
fi.fi_arm.argtypes = [ctypes.c_long]
fi.fi_count.restype = ctypes.c_long
from stimside._native import branch_kernels as bk

lib = bk.get_branch_kernels_lib()

def arm_call(fn, k):
    fi.fi_arm(k)
    try:
        return fn()
    finally:
        fi.fi_disarm()

def count_allocs(fn):
    fi.fi_arm(10**9)
    try:
        fn()
    finally:
        n = int(fi.fi_count())
        fi.fi_disarm()
    return n

def sample_ks(n_allocs):
    if n_allocs <= 80:
        return list(range(n_allocs))
    ks = list(range(35)) + list(range(n_allocs - 35, n_allocs))
    step = max(1, (n_allocs - 70) // 20)
    ks.extend(range(35, n_allocs - 35, step))
    return sorted(set(ks))

def make_noncanon_state(n=4):
    # Create a valid non-canonical state (h has pivot bit set)
    h = np.array([1], dtype=np.uint64)
    A = np.array([[1], [2]], dtype=np.uint64)
    ell = np.array([1, 2], dtype=np.uint8)
    G = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    piv = np.array([0, 1], dtype=np.int32)
    return bk.BranchState.from_arrays(n, h, A, ell, G, piv)

def make_rich_store():
    # n=4 store with:
    # node 0: |0000>
    # node 1: pending T on X0 -> resolve gives 2 branches (|0000>, |1000>)
    # node 2: resolved 2-branch state (|0000>, |1000>)
    # node 3: GHZ(0,3) + T on X1 (resolved, 2 branches, absorbed on Z3, definite on Z0 Z3, forked after Y1 qturn)
    st = bk.BranchStore(4)
    x0 = np.array([[1]], np.uint64)
    z0 = np.array([[0]], np.uint64)
    k0 = np.array([0], np.uint8)
    e1 = np.array([1], np.uint8)
    n1 = st.step_rotations(0, 10, bk.MBR_KIND_T, 0, x0, z0, k0, e1)
    n2 = st.resolve_node(n1)
    s_ghz = bk.BranchState.zero(4)
    s_ghz.project(np.array([9], np.uint64), np.array([0], np.uint64), 0, 0)  # X0 X3 = +1
    n_ghz = st.node_from_branches([1.0 + 0j], [s_ghz])
    s_ghz.close()
    n_gt = st.step_rotations(n_ghz, 11, bk.MBR_KIND_T, 0, np.array([[2]], np.uint64), z0, k0, e1)
    n_gr = st.resolve_node(n_gt)  # 2 branches, both stabilized by X0 X3
    n_gy = st.step_qturns(n_gr, 12, bk.MBR_KIND_SKIP, 0, np.array([[2]], np.uint64), np.array([[2]], np.uint64), np.array([1], np.uint8))
    return st, {"n1": n1, "n2": n2, "n_gt": n_gt, "n_gr": n_gr, "n_gy": n_gy}

def snapshot_store(st):
    nc = st.node_count
    denses = [st.node_dense(i, "big").copy() for i in range(nc)]
    infos = [st.node_info(i) for i in range(nc)]
    cnt = st.counters()
    return nc, denses, infos, cnt

def assert_store_intact(st, snap):
    nc0, denses0, infos0, _ = snap
    assert st.node_count == nc0, f"node_count changed {st.node_count} != {nc0}"
    cnt = st.counters()
    assert sum(cnt["branches_hist"].values()) == nc0, f"hist sum mismatch: {cnt['branches_hist']} vs {nc0}"
    for i in range(nc0):
        assert st.node_info(i) == infos0[i], f"node_info changed for {i}"
        assert np.array_equal(st.node_dense(i, "big"), denses0[i]), f"node_dense changed for {i}"

report = {"allocating": {}, "zero_alloc": {}}

def test_case(name, setup_fn, call_fn, verify_pre_oom_intact, verify_post_retry, cleanup_fn):
    # 1. Measure baseline allocation count & expected output
    ctx0 = setup_fn()
    n_allocs = count_allocs(lambda: call_fn(ctx0))
    cleanup_fn(ctx0)
    assert n_allocs >= 1, f"{name} made 0 allocations"
    tested = 0
    for k in sample_ks(n_allocs):
        ctx = setup_fn()
        ls0 = bk.BranchState.live_states()
        lst0 = bk.BranchStore.live_stores()
        try:
            arm_call(lambda: call_fn(ctx), k)
        except MemoryError:
            pass
        else:
            raise AssertionError(f"{name} k={k}/{n_allocs} did not raise MemoryError")
        # Verify zero leaks & intact pre-existing state/store
        assert bk.BranchState.live_states() == ls0, f"{name} k={k} leaked states: {bk.BranchState.live_states()} vs {ls0}"
        assert bk.BranchStore.live_stores() == lst0, f"{name} k={k} leaked stores: {bk.BranchStore.live_stores()} vs {lst0}"
        verify_pre_oom_intact(ctx)
        # Re-run exact same call with OOM disarmed -> MUST succeed identically!
        res = call_fn(ctx)
        verify_post_retry(ctx, res)
        cleanup_fn(ctx)
        tested += 1
    report["allocating"][name] = {"allocs": n_allocs, "tested_ks": tested}

# --- State allocating entrypoints ---
def _check(cond, msg=""):
    if not cond:
        raise AssertionError(msg)

test_case(
    "mbr_state_create_zero",
    lambda: {},
    lambda c: bk.BranchState.zero(65),
    lambda c: None,
    lambda c, r: (_check(r.n_qubits == 65), r.close()),
    lambda c: None,
)
test_case(
    "mbr_state_from_arrays",
    lambda: {},
    lambda c: make_noncanon_state(4),
    lambda c: None,
    lambda c, r: (_check(r.d == 2), r.close()),
    lambda c: None,
)
test_case(
    "mbr_state_copy",
    lambda: {"s": make_noncanon_state(4)},
    lambda c: c["s"].copy(),
    lambda c: _check(c["s"].d == 2),
    lambda c, r: (_check(r.canonical_key() == c["s"].canonical_key()), r.close()),
    lambda c: c["s"].close(),
)
test_case(
    "mbr_state_from_canonical_key",
    lambda: {"key": (lambda s: (k := s.canonical_key(), s.close())[0])(make_noncanon_state(4))},
    lambda c: bk.BranchState.from_canonical_key(c["key"]),
    lambda c: None,
    lambda c, r: (_check(r.canonical_key() == c["key"]), r.close()),
    lambda c: None,
)
for op_name, op_fn in (
    ("mbr_state_qturn", lambda c: c["s"].qturn(np.array([5], np.uint64), np.array([2], np.uint64), 0)),
    ("mbr_state_zproject", lambda c: c["s"].zproject(np.array([1], np.uint64), 0, 0)),
    ("mbr_state_project", lambda c: c["s"].project(np.array([4], np.uint64), np.array([1], np.uint64), 0, 0)),
    ("mbr_state_definite", lambda c: c["s"].definite(np.array([5], np.uint64), np.array([2], np.uint64), 0)),
    ("mbr_state_generators", lambda c: c["s"].generators()),
    ("mbr_state_canon", lambda c: c["s"].canon()),
    ("mbr_state_canonical_key", lambda c: c["s"].canonical_key()),
):
    def _mk_setup():
        s = make_noncanon_state(4)
        k0 = s.canonical_key()
        return {"s": s, "k0": k0}
    test_case(
        op_name,
        _mk_setup,
        op_fn,
        lambda c: _check(c["s"].canonical_key() == c["k0"], "state mutated on OOM"),
        lambda c, r: _check(r is not None),
        lambda c: c["s"].close(),
    )

test_case(
    "mbr_state_overlap",
    lambda: {"a": make_noncanon_state(4), "b": bk.BranchState.zero(4)},
    lambda c: c["a"].overlap(c["b"]),
    lambda c: None,
    lambda c, r: _check(abs(r) <= 1.0 + 1e-12),
    lambda c: (c["a"].close(), c["b"].close()),
)
test_case(
    "mbr_state_same_phase",
    lambda: {"a": make_noncanon_state(4), "b": make_noncanon_state(4)},
    lambda c: c["a"].same_phase(c["b"]),
    lambda c: None,
    lambda c, r: _check(r[0] is True),
    lambda c: (c["a"].close(), c["b"].close()),
)
test_case(
    "mbr_is_clifford",
    lambda: {"O": list(range(1, 16)) + [1 << 15]},
    lambda c: bk.is_clifford(c["O"]),
    lambda c: None,
    lambda c, r: _check(r is False),
    lambda c: None,
)
test_case(
    "mbr_choose_split",
    lambda: {"O": [1, 2, 3, 4, 5, 6, 7]},
    lambda c: bk.choose_split(c["O"], max_depth=5),
    lambda c: None,
    lambda c, r: _check(r[1] >= 1),
    lambda c: None,
)

# --- Store allocating entrypoints ---
test_case(
    "mbr_store_create",
    lambda: {},
    lambda c: bk.BranchStore(4),
    lambda c: None,
    lambda c, r: (_check(r.node_count == 1), r.close()),
    lambda c: None,
)

def _setup_store():
    st, ids = make_rich_store()
    snap = snapshot_store(st)
    return {"st": st, "ids": ids, "snap": snap}

test_case(
    "mbr_store_clear",
    _setup_store,
    lambda c: c["st"].clear(),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(c["st"].node_count == 1 and c["st"].counters()["store_clears"] == 1),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_node_branch",
    _setup_store,
    lambda c: c["st"].node_branches(c["ids"]["n2"]),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: (_check(len(r) == 2), [s.close() for _, s in r]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_node_dense",
    _setup_store,
    lambda c: c["st"].node_dense(c["ids"]["n2"], "big"),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r.shape == (16,)),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_node_from_branches",
    lambda: {**_setup_store(), "s1": make_noncanon_state(4), "s2": bk.BranchState.zero(4)},
    lambda c: c["st"].node_from_branches(
        [0.6 + 0j, 0.8 + 0j], [c["s1"], c["s2"]],
        np.array([[1]], np.uint64), np.array([[0]], np.uint64),
        np.array([0], np.uint8), np.array([1], np.uint8),
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r == c["snap"][0]),
    lambda c: (c["s1"].close(), c["s2"].close(), c["st"].close()),
)
test_case(
    "mbr_store_resolve_node",
    _setup_store,
    lambda c: c["st"].resolve_node(c["ids"]["n_gt"]),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r == c["snap"][0]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_rotations",
    _setup_store,
    # Anticommuting Z0 on n1 (which has pending X0) -> triggers resolve + push_node + trans.emplace!
    lambda c: c["st"].step_rotations(
        c["ids"]["n1"], 700, bk.MBR_KIND_T, 0,
        np.array([[0]], np.uint64), np.array([[1]], np.uint64),
        np.array([0], np.uint8), np.array([1], np.uint8),
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r == c["snap"][0]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_qturns",
    _setup_store,
    lambda c: c["st"].step_qturns(
        c["ids"]["n1"], 701, bk.MBR_KIND_SKIP, 0,
        np.array([[2]], np.uint64), np.array([[1]], np.uint64),
        np.array([0], np.uint8),
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r == c["snap"][0]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_meas_definite",
    _setup_store,
    lambda c: c["st"].step_meas(
        c["ids"]["n_gr"], 702, bk.MBR_KIND_MEAS,
        np.array([0], np.uint64), np.array([9], np.uint64), 0,
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r[3] == bk.MBR_MEAS_DEFINITE and r[0] == 1.0),
    lambda c: c["st"].close(),
)

def _setup_def_res():
    st, ids = make_rich_store()
    nid_e4 = st.step_rotations(
        0, 50, bk.MBR_KIND_T, 0,
        np.array([[1]], np.uint64), np.array([[0]], np.uint64),
        np.array([0], np.uint8), np.array([4], np.uint8),
    )
    snap = snapshot_store(st)
    return {"st": st, "ids": ids, "nid_e4": nid_e4, "snap": snap}

test_case(
    "mbr_store_step_meas_definite_resolved",
    _setup_def_res,
    lambda c: c["st"].step_meas(
        c["nid_e4"], 703, bk.MBR_KIND_MEAS,
        np.array([0], np.uint64), np.array([1], np.uint64), 0,
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r[3] == bk.MBR_MEAS_DEFINITE and r[0] == 0.0 and r[2] == c["snap"][0]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_meas_absorbed",
    _setup_store,
    lambda c: c["st"].step_meas(
        c["ids"]["n_gr"], 704, bk.MBR_KIND_MEAS,
        np.array([0], np.uint64), np.array([8], np.uint64), 0,
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r[3] == bk.MBR_MEAS_ABSORBED and r[0] == 0.5 and r[1] == r[2] == c["snap"][0]),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_meas_forked",
    _setup_store,
    lambda c: c["st"].step_meas(
        c["ids"]["n_gy"], 705, bk.MBR_KIND_MEAS,
        np.array([0], np.uint64), np.array([2], np.uint64), 0,
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(r[3] == bk.MBR_MEAS_FORKED and 0.0 < r[0] < 1.0 and r[1] == c["snap"][0] and r[2] == c["snap"][0] + 1),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_store_step_peek",
    _setup_store,
    lambda c: c["st"].step_peek(
        c["ids"]["n_gy"], 706,
        np.array([0], np.uint64), np.array([2], np.uint64), 0,
    ),
    lambda c: assert_store_intact(c["st"], c["snap"]),
    lambda c, r: _check(0.0 < r < 1.0),
    lambda c: c["st"].close(),
)

# --- WP6 batch_step_* allocating OOM checks (verify zero buffer mutation & zero store corruption) ---
def _setup_batch_store():
    st, ids = make_rich_store()
    snap = snapshot_store(st)
    nodes = np.array([0, ids["n1"], ids["n_gr"], ids["n_gy"]], dtype=np.int64)
    peaks = np.array([1, 1, 1, 1], dtype=np.int16)
    fx = np.array([[0], [1], [2], [4]], dtype=np.uint64)
    fz = np.array([[0], [0], [1], [2]], dtype=np.uint64)
    lm = np.zeros(4, dtype=np.float64)
    tm = np.full(4, 7, dtype=np.uint8)
    pc = np.full(4, -1.0, dtype=np.float64)
    p1 = np.full(4, -1.0, dtype=np.float64)
    return {
        "st": st, "ids": ids, "snap": snap,
        "nodes": nodes, "nodes0": nodes.copy(),
        "peaks": peaks, "peaks0": peaks.copy(),
        "fx": fx, "fx0": fx.copy(),
        "fz": fz, "fz0": fz.copy(),
        "lm": lm, "lm0": lm.copy(),
        "tm": tm, "tm0": tm.copy(),
        "pc": pc, "pc0": pc.copy(),
        "p1": p1, "p10": p1.copy(),
        "u": np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
    }

def _assert_batch_untouched(c):
    # Pre-existing nodes must remain bit-identical, any already-committed prefix transitions
    # must leave the store completely valid (no torn nodes/pools/hist), and caller buffers 100% untouched.
    st = c["st"]
    nc0, denses0, infos0, _cnt0 = c["snap"]
    _check(st.node_count >= nc0, f"node_count shrank {st.node_count} < {nc0}")
    cnt = st.counters()
    _check(sum(cnt["branches_hist"].values()) == st.node_count, f"hist sum mismatch: {cnt['branches_hist']} vs {st.node_count}")
    for nid in range(nc0):
        _check(st.node_info(nid) == infos0[nid], f"pre-existing node {nid} info changed")
        _check(np.array_equal(st.node_dense(nid, "big"), denses0[nid]), f"pre-existing node {nid} dense changed")
    for nid in range(nc0, st.node_count):
        nb, npd, tpk = st.node_info(nid)
        _check(nb >= 1 and tpk >= nb, f"committed prefix node {nid} invalid")
        _check(st.node_dense(nid, "big").shape == (16,), f"committed prefix node {nid} dense invalid")
    _check(np.array_equal(c["nodes"], c["nodes0"]), "nodes mutated on OOM")
    _check(np.array_equal(c["peaks"], c["peaks0"]), "peaks mutated on OOM")
    _check(np.array_equal(c["fx"], c["fx0"]), "fx mutated on OOM")
    _check(np.array_equal(c["fz"], c["fz0"]), "fz mutated on OOM")
    _check(np.array_equal(c["lm"], c["lm0"]), "lm mutated on OOM")
    _check(np.array_equal(c["tm"], c["tm0"]), "tm mutated on OOM")
    _check(np.array_equal(c["pc"], c["pc0"]), "pc mutated on OOM")
    _check(np.array_equal(c["p1"], c["p10"]), "p1 mutated on OOM")

test_case(
    "mbr_batch_step_rotations",
    _setup_batch_store,
    lambda c: c["st"].batch_step_rotations(
        710, bk.MBR_KIND_T,
        np.array([[0]], np.uint64), np.array([[1]], np.uint64),
        np.array([[0]], np.uint64), np.array([[1]], np.uint64),
        np.array([0], np.uint8), np.array([1], np.uint8), np.array([1], np.uint32),
        c["fx"], c["fz"], c["nodes"], c["peaks"],
    ),
    _assert_batch_untouched,
    lambda c, r: _check(not np.array_equal(c["nodes"], c["nodes0"])),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_batch_step_qturns",
    _setup_batch_store,
    lambda c: c["st"].batch_step_qturns(
        711, bk.MBR_KIND_SKIP,
        np.array([[2]], np.uint64), np.array([[1]], np.uint64),
        np.array([0], np.uint8),
        c["fx"], c["fz"], c["nodes"], c["peaks"],
    ),
    _assert_batch_untouched,
    lambda c, r: _check(not np.array_equal(c["nodes"], c["nodes0"])),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_batch_step_meas",
    _setup_batch_store,
    lambda c: c["st"].batch_step_meas(
        712, bk.MBR_KIND_MEAS,
        np.array([0], np.uint64), np.array([2], np.uint64), 0,
        c["u"], c["fx"], c["fz"], c["nodes"], c["peaks"],
        log_mass=c["lm"], update_log_mass=True,
        out_true_m=c["tm"], out_p_chosen=c["pc"],
    ),
    _assert_batch_untouched,
    lambda c, r: _check(np.all((c["tm"] == 0) | (c["tm"] == 1))),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_batch_step_peek",
    _setup_batch_store,
    lambda c: c["st"].batch_step_peek(
        713,
        np.array([0], np.uint64), np.array([2], np.uint64), 0,
        c["fx"], c["fz"], c["nodes"], c["p1"],
    ),
    _assert_batch_untouched,
    lambda c, r: _check(np.all((c["p1"] >= 0.0) & (c["p1"] <= 1.0))),
    lambda c: c["st"].close(),
)
test_case(
    "mbr_batch_step_postselect",
    _setup_batch_store,
    lambda c: c["st"].batch_step_postselect(
        714,
        np.array([0], np.uint64), np.array([2], np.uint64), 0,
        np.array([0, 0, 0, 0], dtype=np.uint8),
        c["fx"], c["fz"], c["nodes"], c["peaks"], c["lm"],
    ),
    _assert_batch_untouched,
    lambda c, r: _check(r == -1 or (0 <= r < 4)),
    lambda c: c["st"].close(),
)

# --- Zero-alloc pins (including WP6 warm B=4096 batch cache hits) ---
st_z = bk.BranchState.zero(4)
store_z, ids_z = make_rich_store()
# Pre-warm scalar and batch cache hits (multi-node < 128 distinct keys, B = 4096)
x0 = np.array([[1]], np.uint64)
z0 = np.array([[0]], np.uint64)
k0 = np.array([0], np.uint8)
e1 = np.array([1], np.uint8)
fm1 = np.array([1], np.uint32)
store_z.step_rotations(0, 900, bk.MBR_KIND_T, 0, x0, z0, k0, e1)
store_z.step_qturns(0, 901, bk.MBR_KIND_SKIP, 0, x0, z0, k0)

B_Z = 4096
bz_fx = np.zeros((B_Z, 1), dtype=np.uint64)
bz_fz = np.zeros((B_Z, 1), dtype=np.uint64)
bz_fz[1::2, 0] = 1  # two distinct sign_bits (0 and 1)
bz_nodes_rot = np.zeros(B_Z, dtype=np.int64)
bz_nodes_rot[2::3] = ids_z["n1"]
bz_nodes_rot_orig = bz_nodes_rot.copy()
bz_peaks = np.ones(B_Z, dtype=np.int16)
bz_lm = np.zeros(B_Z, dtype=np.float64)
bz_u = np.full(B_Z, 0.25, dtype=np.float64)
bz_tm = np.zeros(B_Z, dtype=np.uint8)
bz_pc = np.zeros(B_Z, dtype=np.float64)
bz_p1 = np.zeros(B_Z, dtype=np.float64)
bz_des = np.zeros(B_Z, dtype=np.uint8)
bz_fail = ctypes.c_int64(-1)
# Pre-warm batch transitions on all present (node, sb) / node keys
store_z.batch_step_rotations(910, bk.MBR_KIND_T, x0, z0, x0, z0, k0, e1, fm1, bz_fx, bz_fz, bz_nodes_rot.copy(), bz_peaks.copy())
store_z.batch_step_qturns(911, bk.MBR_KIND_SKIP, x0, z0, k0, bz_fx, bz_fz, bz_nodes_rot.copy(), bz_peaks.copy())
store_z.batch_step_meas(912, bk.MBR_KIND_MEAS, z0[0], np.array([8], np.uint64), 0, bz_u, bz_fx.copy(), bz_fz.copy(), bz_nodes_rot.copy(), bz_peaks.copy(), log_mass=bz_lm.copy())
store_z.batch_step_peek(913, z0[0], np.array([8], np.uint64), 0, bz_fx, bz_fz, bz_nodes_rot, bz_p1)
store_z.batch_step_postselect(914, z0[0], np.array([8], np.uint64), 0, bz_des, bz_fx.copy(), bz_fz.copy(), bz_nodes_rot.copy(), bz_peaks.copy(), bz_lm.copy())
mz8 = np.array([8], np.uint64)

# Pre-allocate buffers so Python/NumPy wrappers don't skew native C++ alloc count
out_h = np.zeros(1, np.uint64)
out_dense = np.zeros(32, np.float64)
out_c = np.zeros(2, np.float64)
out_cnt = np.zeros(26, np.uint64)
out_hk = np.zeros(4, np.uint64)
out_hc = np.zeros(4, np.uint64)
out_sz = ctypes.c_size_t()
out_sz2 = ctypes.c_size_t()
out_u32 = ctypes.c_uint32()
out_i64 = ctypes.c_int64()
px_b = np.zeros((1, 1), np.uint64)
pz_b = np.zeros((1, 1), np.uint64)
pk_b = np.zeros(1, np.uint8)
pe_b = np.zeros(1, np.uint8)

zero_calls = {
    "mbr_abi_v1": lambda: lib.mbr_abi_v1(),
    "mbr_live_states": lambda: lib.mbr_live_states(),
    "mbr_live_stores": lambda: lib.mbr_live_stores(),
    "mbr_state_header": lambda: lib.mbr_state_header(st_z._handle, ctypes.addressof(out_sz), ctypes.addressof(out_sz2), ctypes.addressof(out_u32)),
    "mbr_state_export_arrays": lambda: lib.mbr_state_export_arrays(st_z._handle, bk.addr(out_h), 0, 0, 0, 0),
    "mbr_state_pauli": lambda: lib.mbr_state_pauli(st_z._handle, bk.addr(x0), bk.addr(z0), 0, bk.addr(out_c)),
    "mbr_state_to_dense": lambda: lib.mbr_state_to_dense(st_z._handle, 0, bk.addr(out_dense), 16),
    "mbr_store_set_max_components": lambda: lib.mbr_store_set_max_components(store_z._handle, 100),
    "mbr_store_node_count": lambda: lib.mbr_store_node_count(store_z._handle, ctypes.addressof(out_sz)),
    "mbr_store_node_info": lambda: lib.mbr_store_node_info(store_z._handle, 0, ctypes.addressof(out_sz), ctypes.addressof(out_sz2), ctypes.addressof(out_u32)),
    "mbr_store_node_pending": lambda: lib.mbr_store_node_pending(store_z._handle, ids_z["n1"], bk.addr(px_b), bk.addr(pz_b), bk.addr(pk_b), bk.addr(pe_b)),
    "mbr_store_counters": lambda: lib.mbr_store_counters(store_z._handle, bk.addr(out_cnt)),
    "mbr_store_branches_hist": lambda: lib.mbr_store_branches_hist(store_z._handle, bk.addr(out_hk), bk.addr(out_hc), 4, ctypes.addressof(out_sz)),
    "mbr_store_step_rotations_cache_hit": lambda: lib.mbr_store_step_rotations(store_z._handle, 0, 900, bk.MBR_KIND_T, 0, 1, bk.addr(x0), bk.addr(z0), bk.addr(k0), bk.addr(e1), ctypes.addressof(out_i64)),
    "mbr_store_step_qturns_cache_hit": lambda: lib.mbr_store_step_qturns(store_z._handle, 0, 901, bk.MBR_KIND_SKIP, 0, 1, bk.addr(x0), bk.addr(z0), bk.addr(k0), ctypes.addressof(out_i64)),
    "mbr_batch_step_rotations_warm_4096": lambda: (
        bz_nodes_rot.__setitem__(slice(None), bz_nodes_rot_orig),
        lib.mbr_batch_step_rotations(store_z._handle, B_Z, 0, 910, bk.MBR_KIND_T, 1, bk.addr(x0), bk.addr(z0), 1, bk.addr(x0), bk.addr(z0), bk.addr(k0), bk.addr(e1), bk.addr(fm1), bk.addr(bz_fx), bk.addr(bz_fz), bk.addr(bz_nodes_rot), bk.addr(bz_peaks)),
    )[1],
    "mbr_batch_step_qturns_warm_4096": lambda: (
        bz_nodes_rot.__setitem__(slice(None), bz_nodes_rot_orig),
        lib.mbr_batch_step_qturns(store_z._handle, B_Z, 0, 911, bk.MBR_KIND_SKIP, 1, bk.addr(x0), bk.addr(z0), bk.addr(k0), bk.addr(bz_fx), bk.addr(bz_fz), bk.addr(bz_nodes_rot), bk.addr(bz_peaks)),
    )[1],
    "mbr_batch_step_meas_warm_4096": lambda: (
        bz_nodes_rot.__setitem__(slice(None), bz_nodes_rot_orig),
        lib.mbr_batch_step_meas(store_z._handle, B_Z, 0, 912, bk.MBR_KIND_MEAS, bk.addr(z0), bk.addr(mz8), 0, bk.addr(bz_u), 0, 0, 0, 1, bk.addr(bz_fx), bk.addr(bz_fz), bk.addr(bz_nodes_rot), bk.addr(bz_peaks), bk.addr(bz_lm), bk.addr(bz_tm), bk.addr(bz_pc)),
    )[1],
    "mbr_batch_step_peek_warm_4096": lambda: (
        bz_nodes_rot.__setitem__(slice(None), bz_nodes_rot_orig),
        lib.mbr_batch_step_peek(store_z._handle, B_Z, 913, bk.addr(z0), bk.addr(mz8), 0, bk.addr(bz_fx), bk.addr(bz_fz), bk.addr(bz_nodes_rot), bk.addr(bz_p1)),
    )[1],
    "mbr_batch_step_postselect_warm_4096": lambda: (
        bz_nodes_rot.__setitem__(slice(None), bz_nodes_rot_orig),
        lib.mbr_batch_step_postselect(store_z._handle, B_Z, 0, 914, bk.addr(z0), bk.addr(mz8), 0, bk.addr(bz_des), bk.addr(bz_fx), bk.addr(bz_fz), bk.addr(bz_nodes_rot), bk.addr(bz_peaks), bk.addr(bz_lm), ctypes.addressof(bz_fail)),
    )[1],
}
for zname, zfn in zero_calls.items():
    cnt = count_allocs(zfn)
    report["zero_alloc"][zname] = cnt
    assert cnt == 0, f"{zname} unexpectedly allocated {cnt} times"

st_z.close()
store_z.close()
assert bk.BranchState.live_states() == 0
assert bk.BranchStore.live_stores() == 0
print(json.dumps(report))
"""


def _under_tsan() -> bool:
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__tsan_init")
    except OSError:
        return False


def test_t15_genuine_cpp_oom_all_allocating_entrypoints(tmp_path: Path) -> None:
    """T15: Genuine `std::bad_alloc` -> `-2` -> `MemoryError` on every allocating `mbr_*` entrypoint."""
    assert not _under_tsan(), "TSan owns operator new"
    assert shutil.which("c++") is not None, "c++ compiler required for genuine OOM interposer"

    cpp = tmp_path / "fi_new.cpp"
    so = tmp_path / "libfi_new.so"
    cpp.write_text(_FI_CPP, encoding="utf-8")
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)

    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    ld_preload = f"{so} {existing_preload}" if existing_preload else str(so)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_PRELOAD": ld_preload,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if existing_preload:
        asan_opts = env.get("ASAN_OPTIONS", "")
        if "verify_asan_link_order=0" not in asan_opts:
            env["ASAN_OPTIONS"] = (
                f"{asan_opts}:verify_asan_link_order=0"
                if asan_opts
                else "verify_asan_link_order=0"
            )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", _OOM_CHILD, str(_SRC_DIR), str(so)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"OOM child rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-3000:]}"
    rep = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(rep["allocating"]) == 33
    assert len(rep["zero_alloc"]) == 20
    assert all(v["allocs"] >= 1 and v["tested_ks"] >= 1 for v in rep["allocating"].values())
    assert all(v == 0 for v in rep["zero_alloc"].values())


# =============================================================================================
# Multi-Threaded Stress & `os.fork()` Safety
# =============================================================================================


def test_t15_multithreaded_independent_and_shared_store() -> None:
    """T15: 8-thread concurrent independent states/stores (n=5, 65) + 8-thread RLock-shared store."""
    errors: list[BaseException] = []
    results_indep: dict[int, tuple[bytes, int]] = {}
    res_lock = threading.Lock()

    def indep_worker(tid: int) -> None:
        try:
            n = 5 if (tid % 2 == 0) else 65
            nw = _nw(n)
            rng = np.random.default_rng(777)  # identical seed per parity for determinism check
            st = bk.BranchState.zero(n)
            store = bk.BranchStore(n)
            try:
                nid = 0
                for step in range(8):
                    xi, zi = _rand_mask(rng, n), _rand_mask(rng, n)
                    k = _herm_kappa(xi, zi, False)
                    xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                    st.qturn(xw, zw, k)
                    nid = store.step_rotations(
                        nid, step + 1, bk.MBR_KIND_T, 0,
                        xw.reshape(1, nw), zw.reshape(1, nw),
                        np.array([k], np.uint8), np.array([1], np.uint8),
                    )
                    if step % 3 == 2:
                        nid = store.resolve_node(nid)
                key = st.canonical_key()
                nc = store.node_count
                with res_lock:
                    results_indep[tid] = (key, nc)
            finally:
                st.close()
                store.close()
        except BaseException as exc:
            with res_lock:
                errors.append(exc)

    threads = [threading.Thread(target=indep_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()
    assert not errors, errors
    # Even threads (n=5) must agree bit-identically; odd threads (n=65) must agree bit-identically
    assert len({results_indep[i] for i in range(0, 8, 2)}) == 1
    assert len({results_indep[i] for i in range(1, 8, 2)}) == 1

    # 8-thread RLock-shared BranchStore stress
    shared = bk.BranchStore(5)
    lock = threading.RLock()
    shared_out: list[tuple[int, float, int]] = []
    try:
        def shared_worker() -> None:
            try:
                nid = 0
                for step in range(6):
                    xw = np.array([[1 << (step % 5)]], np.uint64)
                    zw = np.array([[0]], np.uint64)
                    with lock:
                        nid = shared.step_rotations(
                            nid, step + 1, bk.MBR_KIND_T, 0, xw, zw,
                            np.array([0], np.uint8), np.array([1], np.uint8),
                        )
                with lock:
                    p0, c0, c1, _, _, _ = shared.step_meas(
                        nid, 100, bk.MBR_KIND_MEAS,
                        np.array([0], np.uint64), np.array([1], np.uint64), 0,
                    )
                with res_lock:
                    shared_out.append((nid, p0, c0 if c0 >= 0 else c1))
            except BaseException as exc:
                with res_lock:
                    errors.append(exc)

        threads2 = [threading.Thread(target=shared_worker) for _ in range(8)]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=60)
            assert not t.is_alive()
        assert not errors, errors
        assert len(shared_out) == 8 and len(set(shared_out)) == 1
    finally:
        shared.close()

    gc.collect()
    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0


_FORK_CHILD = r"""
import os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from stimside._native import branch_kernels as bk
from stimside._magic.branch_backend import _FORK_POISON_MSG, _SharedBranchStore

def watchdog():
    time.sleep(45)
    print("WATCHDOG", flush=True)
    os._exit(7)

threading.Thread(target=watchdog, daemon=True).start()

def fork_check(work):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        signal.alarm(15)
        try:
            work()
            msg = "OK"
        except BaseException as exc:
            msg = "FAIL " + repr(exc)
        os.write(w, msg.encode())
        os._exit(0)
    os.close(w)
    res = os.read(r, 4096).decode()
    os.close(r)
    _, st = os.waitpid(pid, 0)
    assert st == 0 and res == "OK", f"child status={st} msg={res!r}"

# Scenario 1: Idle BranchState and BranchStore across os.fork()
s = bk.BranchState.zero(5)
s.qturn(np.array([1], np.uint64), np.array([0], np.uint64), 0)
st = bk.BranchStore(5)
n1 = st.step_rotations(
    0, 1, bk.MBR_KIND_T, 0,
    np.array([[1]], np.uint64), np.array([[0]], np.uint64),
    np.array([0], np.uint8), np.array([1], np.uint8),
)
def child_idle():
    n2 = st.resolve_node(n1)
    assert n2 == 2
    s.canon()
    s.close()
    st.close()
    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0
fork_check(child_idle)
assert st.resolve_node(n1) == 2
s.close()
st.close()

# Scenario 2: _SharedBranchStore lock held (idle native) in background thread at fork -> child renews lock
shared_idle = _SharedBranchStore(5)
inside, release = threading.Event(), threading.Event()
def hold_lock():
    with shared_idle.lock:
        inside.set()
        release.wait(15)
t1 = threading.Thread(target=hold_lock)
t1.start()
assert inside.wait(15)
def child_lock_held():
    # Must NOT deadlock! Lock was renewed in child.
    c = shared_idle.counters()
    assert c["store_nodes"] == 1
    shared_idle.clear_inplace()
fork_check(child_lock_held)
release.set()
t1.join(15)
assert shared_idle.counters()["store_nodes"] == 1

# Scenario 3: _SharedBranchStore busy mid-native-call in background thread at fork -> poisoned in child
shared_busy = _SharedBranchStore(5)
inside2, release2 = threading.Event(), threading.Event()
real_lib = shared_busy.native._lib
class GateLib:
    def __init__(self, inner):
        self._inner = inner
    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if name != "mbr_store_step_rotations":
            return fn
        def held(*args):
            inside2.set()
            release2.wait(15)
            return fn(*args)
        return held
shared_busy.native._lib = GateLib(real_lib)
def hold_native():
    with shared_busy.lock:
        shared_busy.step_rotations_unlocked(
            0, 1, bk.MBR_KIND_T, 0,
            np.array([[1]], np.uint64), np.array([[0]], np.uint64),
            np.array([0], np.uint8), np.array([1], np.uint8),
        )
t2 = threading.Thread(target=hold_native)
t2.start()
assert inside2.wait(15)
def child_busy_poisoned():
    for fn in (
        lambda: shared_busy.counters(),
        lambda: shared_busy.clear_inplace(),
        lambda: shared_busy.advance_frame(0, ("U", "H", (0,))),
        lambda: shared_busy.get_or_compile_event((0, "T", 0), lambda: 1),
        lambda: shared_busy.native.node_count,
    ):
        try:
            fn()
        except RuntimeError as exc:
            assert _FORK_POISON_MSG in str(exc), exc
        else:
            raise AssertionError("poisoned store did not raise RuntimeError")
    # Fresh store in child works cleanly!
    fresh = _SharedBranchStore(5)
    assert fresh.counters()["store_nodes"] == 1
    fresh.native.close()
fork_check(child_busy_poisoned)
release2.set()
t2.join(15)
shared_busy.native._lib = real_lib
assert shared_busy.counters()["store_nodes"] == 2

# Scenario 4a: Deterministic gate inside clear_inplace() AFTER native.clear() (during Python cache clear)
shared_clr = _SharedBranchStore(4)
fid_c = shared_clr.advance_frame(0, ("U", "H", (0,)))
shared_clr.get_or_compile_event((fid_c, "T", 0), lambda: ("P", 0))
with shared_clr.lock:
    shared_clr.step_rotations_unlocked(
        0, 1, bk.MBR_KIND_T, 0,
        np.array([[1]], np.uint64), np.array([[0]], np.uint64),
        np.array([0], np.uint8), np.array([1], np.uint8),
    )
# Verify OOM strong exception safety: if native.clear() returns -2 (MemoryError), Python caches & _busy stay intact.
real_clr_lib = shared_clr.native._lib
class _OomClearLib:
    def __init__(self, inner):
        self._inner = inner
    def __getattr__(self, name):
        if name == "mbr_store_clear":
            return lambda *_args: -2
        return getattr(self._inner, name)
shared_clr.native._lib = _OomClearLib(real_clr_lib)
try:
    shared_clr.clear_inplace()
except MemoryError:
    pass
else:
    raise AssertionError("clear_inplace did not propagate MemoryError")
finally:
    shared_clr.native._lib = real_clr_lib
assert not shared_clr._busy and shared_clr._busy_tid == 0
assert len(shared_clr.frame_trie) == 1 and len(shared_clr.event_ids) == 1
assert len(shared_clr.node_peak_cache) == 2

inside_clr, release_clr = threading.Event(), threading.Event()
class _PauseDict(dict):
    def clear(self):
        inside_clr.set()
        release_clr.wait(15)
        super().clear()
shared_clr.frame_trie = _PauseDict(shared_clr.frame_trie)
t_clr = threading.Thread(target=shared_clr.clear_inplace)
t_clr.start()
assert inside_clr.wait(15)
def child_mid_clear_poisoned():
    assert shared_clr._fork_poisoned
    for fn in (
        lambda: shared_clr.counters(),
        lambda: shared_clr.advance_frame(0, ("U", "H", (0,))),
        lambda: shared_clr.get_or_compile_event((0, "T", 0), lambda: 1),
        lambda: shared_clr.step_rotations_unlocked(
            0, 2, bk.MBR_KIND_T, 0,
            np.array([[0]], np.uint64), np.array([[1]], np.uint64),
            np.array([0], np.uint8), np.array([1], np.uint8),
        ),
    ):
        try:
            fn()
        except RuntimeError as exc:
            assert _FORK_POISON_MSG in str(exc), exc
        else:
            raise AssertionError("mid-clear_inplace forked store was not poisoned")
fork_check(child_mid_clear_poisoned)
release_clr.set()
t_clr.join(15)
assert shared_clr.native.node_count == 1
assert shared_clr.node_peak_cache == {0: 1}
assert not shared_clr.frame_trie and not shared_clr.event_ids and not shared_clr.event_data
shared_clr.native.close()

# Scenario 4b: Concurrent os.fork() stress during live populate + clear_inplace() loop (Challenger repro)
prev_interval = sys.getswitchinterval()
sys.setswitchinterval(1e-6)
stress_store = _SharedBranchStore(4, max_components=0)
ccz_x = np.zeros((7, 1), dtype=np.uint64)
ccz_z = np.array([[m] for m in (1, 2, 4, 3, 5, 6, 7)], dtype=np.uint64)
ccz_k = np.zeros(7, dtype=np.uint8)
ccz_e = np.array([1, 1, 1, 7, 7, 7, 1], dtype=np.uint8)
qt_x = np.array([[1], [2], [4]], dtype=np.uint64)
qt_z = np.array([[1], [2], [4]], dtype=np.uint64)
qt_k = np.array([3, 3, 3], dtype=np.uint8)
t_x, t_z = np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64)
t_k, t_e = np.array([0], dtype=np.uint8), np.array([1], np.uint8)
stop_stress = threading.Event()
def _stress_worker():
    while not stop_stress.is_set():
        fid = 0
        for j in range(1500):
            fid = stress_store.advance_frame(fid, ("U", "H", (j & 3,)))
            stress_store.get_or_compile_event((fid, "T", 0), lambda jj=j: ("OLD", jj))
        with stress_store.lock:
            na, _ = stress_store.step_qturns_unlocked(0, 1, bk.MBR_KIND_SKIP, 0, qt_x, qt_z, qt_k)
            nb_, _ = stress_store.step_rotations_unlocked(na, 2, bk.MBR_KIND_CCZ, 0, ccz_x, ccz_z, ccz_k, ccz_e)
            stress_store.step_rotations_unlocked(
                nb_, 3, bk.MBR_KIND_T, 0, np.array([[1]], np.uint64), np.array([[0]], np.uint64), t_k, t_e
            )
        stress_store.clear_inplace()

t_stress = threading.Thread(target=_stress_worker, daemon=True)
t_stress.start()
time.sleep(0.02)
broken_stress = []
try:
    for _ in range(60):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            signal.alarm(15)
            try:
                if stress_store._fork_poisoned:
                    for fn in (
                        lambda: stress_store.counters(),
                        lambda: stress_store.advance_frame(0, ("U", "H", (0,))),
                        lambda: stress_store.get_or_compile_event((0, "T", 0), lambda: 1),
                    ):
                        try:
                            fn()
                        except RuntimeError as exc:
                            if _FORK_POISON_MSG not in str(exc):
                                raise AssertionError(f"unexpected RT: {exc}") from exc
                        else:
                            raise AssertionError("poisoned stress store did not raise")
                    os.write(w, b"POISONED")
                else:
                    nc = stress_store.native.node_count
                    if len(stress_store.node_peak_cache) > nc:
                        os.write(w, f"STALE_PEAK_LEN:nc={nc},cache={len(stress_store.node_peak_cache)}".encode())
                        os._exit(0)
                    for nid_c, pk_c in list(stress_store.node_peak_cache.items()):
                        if nid_c < 0 or nid_c >= nc:
                            os.write(w, f"OOB_PEAK_KEY:nid={nid_c},nc={nc}".encode())
                            os._exit(0)
                        nb_i, _, tpk_i = stress_store.native.node_info(nid_c)
                        if pk_c != min(32767, max(nb_i, tpk_i)):
                            os.write(w, f"WRONG_CACHED_PEAK:nid={nid_c},cached={pk_c},true={max(nb_i,tpk_i)}".encode())
                            os._exit(0)
                    frame_vals = set(stress_store.frame_trie.values())
                    for ek, eid_c in list(stress_store.event_ids.items()):
                        if eid_c not in stress_store.event_data or not (1 <= eid_c < stress_store._next_event_id):
                            os.write(w, f"BAD_EVENT_MAP:ek={ek},eid={eid_c}".encode())
                            os._exit(0)
                        if ek[0] not in frame_vals:
                            os.write(w, f"DESYNC_FRAME_EVENT:fid={ek[0]}".encode())
                            os._exit(0)
                    with stress_store.lock:
                        cur = 0
                        for step in range(1, 4):
                            cur, pk = stress_store.step_rotations_unlocked(
                                cur, 99990 + step, bk.MBR_KIND_T, 0, t_x, t_z, t_k, t_e
                            )
                            nb_i, _, tpk_i = stress_store.native.node_info(cur)
                            if pk != max(nb_i, tpk_i):
                                os.write(w, f"WRONG_PEAK:node={cur},reported={pk},true={max(nb_i,tpk_i)}".encode())
                                os._exit(0)
                    os.write(w, b"OK")
            except RuntimeError as exc:
                os.write(w, b"POISONED" if _FORK_POISON_MSG in str(exc) else f"RT:{exc}".encode())
            except BaseException as exc:
                os.write(w, f"ERR:{exc!r}".encode())
            finally:
                os.close(w)
                os._exit(0)
        os.close(w)
        _, st_rc = os.waitpid(pid, 0)
        msg = os.read(r, 1024).decode()
        os.close(r)
        if st_rc != 0 or msg not in ("OK", "POISONED"):
            broken_stress.append((st_rc, msg))
        time.sleep(0.001)
finally:
    stop_stress.set()
    t_stress.join(timeout=15)
    sys.setswitchinterval(prev_interval)
    stress_store.native.close()
assert not broken_stress, f"concurrent clear_inplace fork failures: {broken_stress}"

shared_idle.native.close()
shared_busy.native.close()
assert bk.BranchState.live_states() == 0
assert bk.BranchStore.live_stores() == 0
print("ALL_FORK_OK")
"""


def test_t15_fork_safety_idle_and_busy_shared_store() -> None:
    """T15: Verify `os.fork()` safety for idle handles, lock-held stores, and busy-thread poisoning."""
    assert hasattr(os, "fork"), "POSIX os.fork required"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-B", "-c", _FORK_CHILD, str(_SRC_DIR)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"fork child rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "ALL_FORK_OK" in proc.stdout


def test_t15_strict_cpp_compiler_warnings() -> None:
    """Verify `magic_kernels.cpp` (including `branch_kernels.hpp`) compiles cleanly with `-Wall -Wextra -Werror`."""
    cxx = shutil.which("c++")
    assert cxx is not None, "c++ compiler not found"
    src = _NATIVE_DIR / "magic_kernels.cpp"
    proc = subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-fno-strict-aliasing",
            "-ffp-contract=off",
            "-g0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"c++ strict check failed:\n{proc.stderr}"
    assert proc.stderr.strip() == ""


# =============================================================================================
# Section D — WP6 Batch-Walk Differential, Spill (>4096, >256 keys), Multi-Word & Concurrency
# =============================================================================================


def _anti_words_py(fx_row: np.ndarray, fz_row: np.ndarray, px: np.ndarray, pz: np.ndarray) -> int:
    acc = int(0)
    for w in range(fx_row.shape[0]):
        acc ^= (int(fx_row[w]) & int(pz[w])) ^ (int(fz_row[w]) & int(px[w]))
    return acc.bit_count() & 1


def _scalar_ref_batch_rotations(
    st: bk.BranchStore,
    eid: int,
    kind: int,
    gen_x: np.ndarray,
    gen_z: np.ndarray,
    base_x: np.ndarray,
    base_z: np.ndarray,
    base_k: np.ndarray,
    ax_e: np.ndarray,
    flip_masks: np.ndarray,
    fx: np.ndarray,
    fz: np.ndarray,
    nodes: np.ndarray,
    peaks: np.ndarray,
    sel_mask: np.ndarray | None = None,
) -> None:
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for s in range(nodes.shape[0]):
        if sel_mask is not None and not bool(sel_mask[s]):
            continue
        nid = int(nodes[s])
        sb = 0
        for g in range(gen_x.shape[0]):
            sb |= _anti_words_py(fx[s], fz[s], gen_x[g], gen_z[g]) << g
        key = (nid, sb)
        if key not in seen:
            fk = np.empty(base_k.shape[0], dtype=np.uint8)
            for i in range(base_k.shape[0]):
                fk[i] = (int(base_k[i]) + (2 if ((sb & int(flip_masks[i])).bit_count() & 1) else 0)) & 3
            cid = st.step_rotations(nid, eid, kind, sb, base_x, base_z, fk, ax_e)
            nb, _, tpk = st.node_info(cid)
            seen[key] = (cid, min(32767, max(nb, tpk)))
        cid, cpk = seen[key]
        nodes[s] = cid
        if peaks[s] < cpk:
            peaks[s] = cpk


def _scalar_ref_batch_qturns(
    st: bk.BranchStore,
    eid: int,
    kind: int,
    base_x: np.ndarray,
    base_z: np.ndarray,
    base_k: np.ndarray,
    fx: np.ndarray,
    fz: np.ndarray,
    nodes: np.ndarray,
    peaks: np.ndarray,
    sel_mask: np.ndarray | None = None,
) -> None:
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for s in range(nodes.shape[0]):
        if sel_mask is not None and not bool(sel_mask[s]):
            continue
        nid = int(nodes[s])
        sb = 0
        for j in range(base_k.shape[0]):
            sb |= _anti_words_py(fx[s], fz[s], base_x[j], base_z[j]) << j
        key = (nid, sb)
        if key not in seen:
            fk = np.array(
                [(int(base_k[j]) + (2 if (sb >> j) & 1 else 0)) & 3 for j in range(base_k.shape[0])],
                dtype=np.uint8,
            )
            cid = st.step_qturns(nid, eid, kind, sb, base_x, base_z, fk)
            nb, _, tpk = st.node_info(cid)
            seen[key] = (cid, min(32767, max(nb, tpk)))
        cid, cpk = seen[key]
        nodes[s] = cid
        if peaks[s] < cpk:
            peaks[s] = cpk


def _scalar_ref_batch_meas(
    st: bk.BranchStore,
    eid: int,
    kind: int,
    mx: np.ndarray,
    mz: np.ndarray,
    mk: int,
    uniforms: np.ndarray,
    fx: np.ndarray,
    fz: np.ndarray,
    nodes: np.ndarray,
    peaks: np.ndarray,
    log_mass: np.ndarray | None,
    update_log_mass: bool,
    sel_mask: np.ndarray | None,
    kick_x: np.ndarray | None,
    kick_z: np.ndarray | None,
    want_one_mask: np.ndarray | None,
    out_true_m: np.ndarray | None,
    out_p_chosen: np.ndarray | None,
) -> None:
    seen: dict[int, tuple[float, int, int, int, np.ndarray, np.ndarray, int, int]] = {}
    for s in range(nodes.shape[0]):
        if sel_mask is not None and not bool(sel_mask[s]):
            continue
        nid = int(nodes[s])
        if nid not in seen:
            p0, c0, c1, mt, ax, az = st.step_meas(nid, eid, kind, mx, mz, mk)
            pk0 = 1 if c0 < 0 else min(32767, max(st.node_info(c0)[0], st.node_info(c0)[2]))
            pk1 = 1 if c1 < 0 else min(32767, max(st.node_info(c1)[0], st.node_info(c1)[2]))
            seen[nid] = (p0, c0, c1, mt, ax, az, pk0, pk1)
        p0, c0, c1, mt, ax, az, pk0, pk1 = seen[nid]
        inner_m = True if c0 < 0 else (False if c1 < 0 else (float(uniforms[s]) >= p0))
        s_bit = bool(_anti_words_py(fx[s], fz[s], mx, mz))
        true_m = inner_m ^ s_bit
        if mt == bk.MBR_MEAS_ABSORBED and inner_m:
            fx[s] ^= ax
            fz[s] ^= az
        nodes[s] = c1 if inner_m else c0
        cpk = pk1 if inner_m else pk0
        if peaks[s] < cpk:
            peaks[s] = cpk
        p_ch = (1.0 - p0) if inner_m else p0
        if out_p_chosen is not None:
            out_p_chosen[s] = p_ch
        if update_log_mass and log_mass is not None:
            log_mass[s] += (-math.inf if p_ch <= 0.0 else (0.0 if p_ch == 1.0 else math.log(p_ch)))
        if kick_x is not None and kick_z is not None:
            want1 = (want_one_mask is not None) and bool(want_one_mask[s])
            if true_m != want1:
                fx[s] ^= kick_x
                fz[s] ^= kick_z
        if out_true_m is not None:
            out_true_m[s] = 1 if true_m else 0


def test_wp6_batch_walk_differential_spill_multiword_and_atomicity() -> None:
    """WP6: Differential parity of all 5 `mbr_batch_step_*` vs scalar `step_*` across B={0,1,17,128,5000}, >256 keys, and n up to 193."""
    rng = np.random.default_rng(20261128)

    # 1. Batch sizes B in {0, 1, 17, 128, 5000} (B=5000 exercises >4096 stack-spill!) + sel_mask + kicks + trace
    for bsz in (0, 1, 17, 128, 5000):
        n, nw = 5, 1
        st_b = bk.BranchStore(n)
        st_r = bk.BranchStore(n)
        try:
            fx_b = rng.integers(0, 1 << n, size=(bsz, nw), dtype=np.uint64)
            fz_b = rng.integers(0, 1 << n, size=(bsz, nw), dtype=np.uint64)
            nodes_b = np.zeros(bsz, dtype=np.int64)
            peaks_b = np.ones(bsz, dtype=np.int16)
            lm_b = np.zeros(bsz, dtype=np.float64)

            fx_r, fz_r = fx_b.copy(), fz_b.copy()
            nodes_r, peaks_r, lm_r = nodes_b.copy(), peaks_b.copy(), lm_b.copy()

            # Step 1: Uniform hot0 qturn (H on q0, q1)
            q_x = np.array([[1], [2]], dtype=np.uint64)
            q_z = np.array([[1], [2]], dtype=np.uint64)
            q_k = np.array([1, 1], dtype=np.uint8)
            st_b.batch_step_qturns(1, bk.MBR_KIND_SKIP, q_x, q_z, q_k, fx_b, fz_b, nodes_b, peaks_b)
            _scalar_ref_batch_qturns(st_r, 1, bk.MBR_KIND_SKIP, q_x, q_z, q_k, fx_r, fz_r, nodes_r, peaks_r)
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(peaks_b, peaks_r)

            # Step 2: Masked 3-generator CCZ rotation (7 axes + flip_masks)
            sel = (rng.random(bsz) < 0.7) if bsz > 0 else np.zeros(0, dtype=np.bool_)
            g_x = np.zeros((3, 1), dtype=np.uint64)
            g_z = np.array([[1], [2], [4]], dtype=np.uint64)
            b_x = np.zeros((7, 1), dtype=np.uint64)
            b_z = np.array([[1], [2], [4], [3], [5], [6], [7]], dtype=np.uint64)
            b_k = np.zeros(7, dtype=np.uint8)
            b_e = np.array([1, 1, 1, 7, 7, 7, 1], dtype=np.uint8)
            fm = np.array([1, 2, 4, 3, 5, 6, 7], dtype=np.uint32)
            st_b.batch_step_rotations(
                2, bk.MBR_KIND_CCZ, g_x, g_z, b_x, b_z, b_k, b_e, fm,
                fx_b, fz_b, nodes_b, peaks_b, sel_mask=sel,
            )
            _scalar_ref_batch_rotations(
                st_r, 2, bk.MBR_KIND_CCZ, g_x, g_z, b_x, b_z, b_k, b_e, fm,
                fx_r, fz_r, nodes_r, peaks_r, sel_mask=sel,
            )
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(peaks_b, peaks_r)

            # Step 3: Anticommuting T rotation on X0 (triggers resolve!)
            tx = np.array([[1]], dtype=np.uint64)
            tz = np.array([[0]], dtype=np.uint64)
            tk = np.array([0], dtype=np.uint8)
            te = np.array([1], dtype=np.uint8)
            tfm = np.array([1], dtype=np.uint32)
            st_b.batch_step_rotations(3, bk.MBR_KIND_T, tx, tz, tx, tz, tk, te, tfm, fx_b, fz_b, nodes_b, peaks_b)
            _scalar_ref_batch_rotations(st_r, 3, bk.MBR_KIND_T, tx, tz, tx, tz, tk, te, tfm, fx_r, fz_r, nodes_r, peaks_r)
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(peaks_b, peaks_r)

            # Step 4: Peek Z0
            p1_b = np.empty(bsz, dtype=np.float64)
            st_b.batch_step_peek(4, np.array([0], np.uint64), np.array([1], np.uint64), 0, fx_b, fz_b, nodes_b, p1_b)
            seen_pk: dict[int, float] = {}
            p1_r = np.empty(bsz, dtype=np.float64)
            for s in range(bsz):
                nid = int(nodes_r[s])
                if nid not in seen_pk:
                    seen_pk[nid] = st_r.step_peek(nid, 4, np.array([0], np.uint64), np.array([1], np.uint64), 0)
                s_bit = bool(_anti_words_py(fx_r[s], fz_r[s], np.array([0], np.uint64), np.array([1], np.uint64)))
                p1_r[s] = min(1.0, max(0.0, seen_pk[nid] if s_bit else (1.0 - seen_pk[nid])))
            assert np.allclose(p1_b, p1_r, atol=0.0, rtol=0.0)

            # Step 5: Measurement with reset kick, want_one_mask, out_true_m, out_p_chosen
            u = rng.random(bsz)
            want1 = (rng.random(bsz) < 0.5) if bsz > 0 else np.zeros(0, dtype=np.bool_)
            tm_b = np.full(bsz, 9, dtype=np.uint8)
            tm_r = tm_b.copy()
            pc_b = np.full(bsz, -2.0, dtype=np.float64)
            pc_r = pc_b.copy()
            st_b.batch_step_meas(
                5, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0, u,
                fx_b, fz_b, nodes_b, peaks_b, log_mass=lm_b, update_log_mass=True,
                sel_mask=sel, kick_x=np.array([1], np.uint64), kick_z=np.array([0], np.uint64),
                want_one_mask=want1, out_true_m=tm_b, out_p_chosen=pc_b,
            )
            _scalar_ref_batch_meas(
                st_r, 5, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0, u,
                fx_r, fz_r, nodes_r, peaks_r, lm_r, True,
                sel, np.array([1], np.uint64), np.array([0], np.uint64), want1, tm_r, pc_r,
            )
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(peaks_b, peaks_r)
            assert np.array_equal(fx_b, fx_r)
            assert np.array_equal(fz_b, fz_r)
            assert np.array_equal(tm_b, tm_r)
            assert np.allclose(pc_b, pc_r, atol=0.0, rtol=0.0)
            assert np.allclose(lm_b, lm_r, atol=1e-15, rtol=0.0)

            # Step 6: URESET (update_log_mass=False leaves log_mass unchanged)
            lm_before = lm_b.copy()
            st_b.batch_step_meas(
                6, bk.MBR_KIND_URESET, np.array([0], np.uint64), np.array([2], np.uint64), 0, u,
                fx_b, fz_b, nodes_b, peaks_b, update_log_mass=False,
                sel_mask=sel, kick_x=np.array([2], np.uint64), kick_z=np.array([0], np.uint64),
                want_one_mask=want1,
            )
            _scalar_ref_batch_meas(
                st_r, 6, bk.MBR_KIND_URESET, np.array([0], np.uint64), np.array([2], np.uint64), 0, u,
                fx_r, fz_r, nodes_r, peaks_r, None, False,
                sel, np.array([2], np.uint64), np.array([0], np.uint64), want1, None, None,
            )
            assert np.array_equal(lm_b, lm_before)
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(fx_b, fx_r) and np.array_equal(fz_b, fz_r)

            # Exact counter parity (including store_nodes and transitions_new/hit)
            cb = st_b.counters()
            cr = st_r.counters()
            for key in ("store_nodes", "transitions_new", "transitions_hit", "resolve_calls", "splits",
                        "meas_definite", "meas_absorbed", "meas_forked", "branches_hist"):
                assert cb[key] == cr[key], f"bsz={bsz} key={key}: {cb[key]} != {cr[key]}"
        finally:
            st_b.close()
            st_r.close()

    # 2. > 256 distinct keys (hash-spill from stack 256-slot table to std::unordered_map)
    n_sp = 6
    st_hb = bk.BranchStore(n_sp)
    st_hr = bk.BranchStore(n_sp)
    try:
        # Create 280 distinct nodes by chaining unique single-qubit T events
        dist_nodes = [0]
        for idx_n in range(280):
            ax_z = np.array([[1 << (idx_n % n_sp)]], dtype=np.uint64)
            ax_x = np.zeros((1, 1), dtype=np.uint64)
            nb = st_hb.step_rotations(dist_nodes[-1], 1000 + idx_n, bk.MBR_KIND_T, 0, ax_x, ax_z, np.array([0], np.uint8), np.array([1], np.uint8))
            nr = st_hr.step_rotations(dist_nodes[-1], 1000 + idx_n, bk.MBR_KIND_T, 0, ax_x, ax_z, np.array([0], np.uint8), np.array([1], np.uint8))
            assert nb == nr
            dist_nodes.append(nb)
        B_sp = 320
        nodes_b = np.array([dist_nodes[i % 281] for i in range(B_sp)], dtype=np.int64)
        nodes_r = nodes_b.copy()
        peaks_b = np.ones(B_sp, dtype=np.int16)
        peaks_r = peaks_b.copy()
        fx_b = rng.integers(0, 1 << n_sp, size=(B_sp, 1), dtype=np.uint64)
        fz_b = rng.integers(0, 1 << n_sp, size=(B_sp, 1), dtype=np.uint64)
        fx_r, fz_r = fx_b.copy(), fz_b.copy()
        lm_b = np.zeros(B_sp, dtype=np.float64)
        lm_r = lm_b.copy()
        u_sp = rng.random(B_sp)

        # Exercises >256 distinct keys in rotations, qturns, meas, peek, and postselect
        gx = np.array([[1]], dtype=np.uint64)
        gz = np.array([[0]], dtype=np.uint64)
        st_hb.batch_step_rotations(5001, bk.MBR_KIND_T, gx, gz, gx, gz, np.array([0], np.uint8), np.array([1], np.uint8), np.array([1], np.uint32), fx_b, fz_b, nodes_b, peaks_b)
        _scalar_ref_batch_rotations(st_hr, 5001, bk.MBR_KIND_T, gx, gz, gx, gz, np.array([0], np.uint8), np.array([1], np.uint8), np.array([1], np.uint32), fx_r, fz_r, nodes_r, peaks_r)
        assert np.array_equal(nodes_b, nodes_r) and np.array_equal(peaks_b, peaks_r)

        st_hb.batch_step_qturns(5002, bk.MBR_KIND_CFAULT, gx, gz, np.array([0], np.uint8), fx_b, fz_b, nodes_b, peaks_b)
        _scalar_ref_batch_qturns(st_hr, 5002, bk.MBR_KIND_CFAULT, gx, gz, np.array([0], np.uint8), fx_r, fz_r, nodes_r, peaks_r)
        assert np.array_equal(nodes_b, nodes_r) and np.array_equal(peaks_b, peaks_r)

        p1_b = np.empty(B_sp, dtype=np.float64)
        st_hb.batch_step_peek(5003, np.array([0], np.uint64), np.array([2], np.uint64), 0, fx_b, fz_b, nodes_b, p1_b)
        assert np.all((p1_b >= 0.0) & (p1_b <= 1.0))

        st_hb.batch_step_meas(5004, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0, u_sp, fx_b, fz_b, nodes_b, peaks_b, log_mass=lm_b)
        _scalar_ref_batch_meas(st_hr, 5004, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0, u_sp, fx_r, fz_r, nodes_r, peaks_r, lm_r, True, None, None, None, None, None, None)
        assert np.array_equal(nodes_b, nodes_r) and np.array_equal(fx_b, fx_r) and np.array_equal(fz_b, fz_r)

        # Postselect atomicity: shot 300 requests impossible Z_5 == 1 (on |0> state with f_x=0)
        fx_ps = np.zeros((B_sp, 1), dtype=np.uint64)
        fz_ps = np.zeros((B_sp, 1), dtype=np.uint64)
        des_ps = np.zeros(B_sp, dtype=np.uint8)
        des_ps[300] = 1
        nodes_ps_before = nodes_b.copy()
        peaks_ps_before = peaks_b.copy()
        lm_ps_before = lm_b.copy()
        fail_s = st_hb.batch_step_postselect(
            5005, np.array([0], np.uint64), np.array([1 << 5], np.uint64), 0,
            des_ps, fx_ps, fz_ps, nodes_b, peaks_b, lm_b,
        )
        assert fail_s == 300
        assert np.array_equal(nodes_b, nodes_ps_before)
        assert np.array_equal(peaks_b, peaks_ps_before)
        assert np.array_equal(lm_b, lm_ps_before)
        assert np.all(fx_ps == 0) and np.all(fz_ps == 0)
        # With valid desired_bits (all 0), postselect succeeds atomically (returns -1)
        des_ps[300] = 0
        fail_ok = st_hb.batch_step_postselect(
            5005, np.array([0], np.uint64), np.array([1 << 5], np.uint64), 0,
            des_ps, fx_ps, fz_ps, nodes_b, peaks_b, lm_b,
        )
        assert fail_ok == -1
    finally:
        st_hb.close()
        st_hr.close()

    # 3. Multi-word batch walk for n in {1, 4, 63, 64, 65, 130, 193} (including cross-word GF(2) absorption)
    for n in (1, 4, 63, 64, 65, 130, 193):
        nw = _nw(n)
        bsz = 24
        st_b = bk.BranchStore(n)
        st_r = bk.BranchStore(n)
        try:
            fx_b = np.zeros((bsz, nw), dtype=np.uint64)
            fz_b = np.zeros((bsz, nw), dtype=np.uint64)
            for s in range(bsz):
                fx_b[s] = _int_to_words(_rand_mask(rng, n), nw)
                fz_b[s] = _int_to_words(_rand_mask(rng, n), nw)
            fx_r, fz_r = fx_b.copy(), fz_b.copy()
            nodes_b = np.zeros(bsz, dtype=np.int64)
            nodes_r = nodes_b.copy()
            peaks_b = np.ones(bsz, dtype=np.int16)
            peaks_r = peaks_b.copy()
            lm_b = np.zeros(bsz, dtype=np.float64)
            lm_r = lm_b.copy()

            hi_bit = 1 << (n - 1)
            lo_bit = 1
            # Put qubit 0 and qubit n-1 into superposition via qturn on Y_0 and Y_{n-1}
            if n >= 2:
                pair_m = hi_bit | lo_bit
                q_x = np.stack((_int_to_words(lo_bit, nw), _int_to_words(hi_bit, nw)), axis=0)
                q_z = np.stack((_int_to_words(lo_bit, nw), _int_to_words(hi_bit, nw)), axis=0)
                q_k = np.array([1, 1], dtype=np.uint8)
            else:
                pair_m = lo_bit
                q_x = _int_to_words(lo_bit, nw).reshape(1, nw)
                q_z = _int_to_words(lo_bit, nw).reshape(1, nw)
                q_k = np.array([1], dtype=np.uint8)
            st_b.batch_step_qturns(10, bk.MBR_KIND_SKIP, q_x, q_z, q_k, fx_b, fz_b, nodes_b, peaks_b)
            _scalar_ref_batch_qturns(st_r, 10, bk.MBR_KIND_SKIP, q_x, q_z, q_k, fx_r, fz_r, nodes_r, peaks_r)
            assert np.array_equal(nodes_b, nodes_r)

            # Apply rotation spanning word 0 and word nw-1
            rz = _int_to_words(pair_m, nw).reshape(1, nw)
            rx = np.zeros((1, nw), dtype=np.uint64)
            st_b.batch_step_rotations(
                11, bk.MBR_KIND_T, rx, rz, rx, rz, np.array([0], np.uint8), np.array([1], np.uint8),
                np.array([1], np.uint32), fx_b, fz_b, nodes_b, peaks_b,
            )
            _scalar_ref_batch_rotations(
                st_r, 11, bk.MBR_KIND_T, rx, rz, rx, rz, np.array([0], np.uint8), np.array([1], np.uint8),
                np.array([1], np.uint32), fx_r, fz_r, nodes_r, peaks_r,
            )
            assert np.array_equal(nodes_b, nodes_r) and np.array_equal(peaks_b, peaks_r)

            # Measure Z_{n-1} (cross-word absorption or forked) with reset kick X_{n-1}
            mx = np.zeros(nw, dtype=np.uint64)
            mz = _int_to_words(hi_bit, nw)
            kx = _int_to_words(hi_bit, nw)
            kz = np.zeros(nw, dtype=np.uint64)
            u_mw = rng.random(bsz)
            tm_b = np.zeros(bsz, dtype=np.uint8)
            tm_r = np.zeros(bsz, dtype=np.uint8)
            st_b.batch_step_meas(
                12, bk.MBR_KIND_MEAS, mx, mz, 0, u_mw, fx_b, fz_b, nodes_b, peaks_b,
                log_mass=lm_b, update_log_mass=True, kick_x=kx, kick_z=kz, out_true_m=tm_b,
            )
            _scalar_ref_batch_meas(
                st_r, 12, bk.MBR_KIND_MEAS, mx, mz, 0, u_mw, fx_r, fz_r, nodes_r, peaks_r,
                lm_r, True, None, kx, kz, None, tm_r, None,
            )
            assert np.array_equal(nodes_b, nodes_r)
            assert np.array_equal(peaks_b, peaks_r)
            assert np.array_equal(fx_b, fx_r)
            assert np.array_equal(fz_b, fz_r)
            assert np.array_equal(tm_b, tm_r)
            assert np.allclose(lm_b, lm_r, atol=1e-15, rtol=0.0)
        finally:
            st_b.close()
            st_r.close()

    gc.collect()
    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0


_WP6_FORK_BATCH_CHILD = r"""
import os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
import numpy as np
from stimside._native import branch_kernels as bk
from stimside._magic.branch_backend import _FORK_POISON_MSG, _SharedBranchStore

shared = _SharedBranchStore(5)
inside, release = threading.Event(), threading.Event()
real_lib = shared.native._lib

class GateBatchLib:
    def __init__(self, inner):
        self._inner = inner
    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if name != "mbr_batch_step_rotations":
            return fn
        def held(*args):
            inside.set()
            release.wait(15)
            return fn(*args)
        return held

shared.native._lib = GateBatchLib(real_lib)
gx = np.array([[1]], np.uint64)
gz = np.array([[0]], np.uint64)
gk = np.array([0], np.uint8)
ge = np.array([1], np.uint8)
fm = np.array([1], np.uint32)
fx = np.zeros((4, 1), np.uint64)
fz = np.zeros((4, 1), np.uint64)
nodes = np.zeros(4, np.int64)
peaks = np.ones(4, np.int16)

t = threading.Thread(
    target=lambda: shared.batch_step_rotations(1, bk.MBR_KIND_T, gx, gz, gx, gz, gk, ge, fm, fx, fz, nodes, peaks)
)
t.start()
assert inside.wait(15)

r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r)
    signal.alarm(15)
    try:
        for fn in (
            lambda: shared.node_count_unlocked(),
            lambda: shared.counters_unlocked(),
            lambda: shared.batch_step_rotations(2, bk.MBR_KIND_T, gx, gz, gx, gz, gk, ge, fm, fx.copy(), fz.copy(), nodes.copy(), peaks.copy()),
            lambda: shared.batch_step_qturns(3, bk.MBR_KIND_SKIP, gx, gz, gk, fx.copy(), fz.copy(), nodes.copy(), peaks.copy()),
            lambda: shared.batch_step_meas(4, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0, np.zeros(4), fx.copy(), fz.copy(), nodes.copy(), peaks.copy(), log_mass=np.zeros(4)),
            lambda: shared.batch_step_peek(5, np.array([0], np.uint64), np.array([1], np.uint64), 0, fx, fz, nodes, np.zeros(4)),
            lambda: shared.batch_step_postselect(6, np.array([0], np.uint64), np.array([1], np.uint64), 0, np.zeros(4, np.uint8), fx.copy(), fz.copy(), nodes.copy(), peaks.copy(), np.zeros(4)),
        ):
            try:
                fn()
            except RuntimeError as exc:
                assert _FORK_POISON_MSG in str(exc), exc
            else:
                raise AssertionError("poisoned batch_step did not raise RuntimeError")
        fresh = _SharedBranchStore(5)
        fnodes = np.zeros(4, np.int64)
        fpeaks = np.ones(4, np.int16)
        fresh.batch_step_rotations(1, bk.MBR_KIND_T, gx, gz, gx, gz, gk, ge, fm, fx, fz, fnodes, fpeaks)
        assert np.all(fnodes == 1) and fresh.node_count_unlocked() == 2
        fresh.native.close()
        os.write(w, b"OK")
    except BaseException as exc:
        os.write(w, f"FAIL:{exc!r}".encode())
    finally:
        os.close(w)
        os._exit(0)
os.close(w)
res = os.read(r, 4096).decode()
os.close(r)
_, st = os.waitpid(pid, 0)
release.set()
t.join(15)
shared.native._lib = real_lib
assert st == 0 and res == "OK", f"st={st} res={res!r}"
assert np.all(nodes == 1)
assert shared.node_count_unlocked() == 2
shared.native.close()
assert bk.BranchStore.live_stores() == 0
print("WP6_FORK_OK")
"""


def test_wp6_shared_store_batch_step_threads_and_fork() -> None:
    """WP6: Multi-threaded + os.fork() stress exercising `_SharedBranchStore.batch_step_*` wrappers."""
    from stimside._magic.branch_backend import _SharedBranchStore

    shared = _SharedBranchStore(6)
    errors: list[BaseException] = []
    out_sigs: list[tuple[bytes, bytes, bytes]] = []
    res_lock = threading.Lock()
    try:
        def worker() -> None:
            try:
                fx = np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.uint64)
                fz = np.array([[0], [0], [1], [1], [2], [2], [3], [3]], dtype=np.uint64)
                nodes = np.zeros(8, dtype=np.int64)
                peaks = np.ones(8, dtype=np.int16)
                lm = np.zeros(8, dtype=np.float64)
                p1 = np.zeros(8, dtype=np.float64)
                tm = np.zeros(8, dtype=np.uint8)
                u = np.linspace(0.05, 0.95, 8, dtype=np.float64)
                qx = np.array([[1]], np.uint64)
                qz = np.array([[1]], np.uint64)
                qk = np.array([1], np.uint8)
                shared.batch_step_qturns(1, bk.MBR_KIND_SKIP, qx, qz, qk, fx, fz, nodes, peaks)
                tx = np.array([[0]], np.uint64)
                tz = np.array([[1]], np.uint64)
                shared.batch_step_rotations(
                    2, bk.MBR_KIND_T, tx, tz, tx, tz, np.array([0], np.uint8), np.array([1], np.uint8),
                    np.array([1], np.uint32), fx, fz, nodes, peaks,
                )
                shared.batch_step_peek(3, np.array([0], np.uint64), np.array([1], np.uint64), 0, fx, fz, nodes, p1)
                shared.batch_step_meas(
                    4, bk.MBR_KIND_MEAS, np.array([0], np.uint64), np.array([1], np.uint64), 0,
                    u, fx, fz, nodes, peaks, log_mass=lm, out_true_m=tm,
                )
                fail = shared.batch_step_postselect(
                    5, np.array([0], np.uint64), np.array([1 << 5], np.uint64), 0,
                    np.zeros(8, np.uint8), fx, fz, nodes, peaks, lm,
                )
                assert fail == -1
                with res_lock:
                    out_sigs.append((nodes.tobytes(), tm.tobytes(), p1.tobytes()))
            except BaseException as exc:
                with res_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()
        assert not errors, errors
        assert len(out_sigs) == 8 and len(set(out_sigs)) == 1
        with shared.lock:
            assert shared.node_count_unlocked() == shared.counters_unlocked()["store_nodes"]
    finally:
        shared.native.close()

    assert hasattr(os, "fork"), "POSIX os.fork required"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-B", "-c", _WP6_FORK_BATCH_CHILD, str(_SRC_DIR)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"wp6 fork child rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "WP6_FORK_OK" in proc.stdout
    gc.collect()
    assert bk.BranchState.live_states() == 0
    assert bk.BranchStore.live_stores() == 0
