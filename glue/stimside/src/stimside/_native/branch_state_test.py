"""T1 tests for the C++ stabilizer-state primitives (`branch_kernels.hpp`) and bindings (`branch_kernels.py`).

Covers:
1. ABI prototype pin against `branch_kernels.hpp`, `mbr_abi_v1`, `branch_abi_ok`, status-code
   exception mapping, symbol isolation, packaging pins, stale-rebuild, and leak checks.
2. >= 500 random states per primitive (`n = 1..8`) verified against dense state vectors (< 1e-12):
   `pauli`, `qturn`, `zproject`, `project`, `canon`, `overlap`, `same_phase`, `dense`.
3. `definite`, `generators`, canonical key invariance & round-trip, and `PhaseQFEComponent`
   round-trips across small (`1..8`) and multi-word (`63, 64, 65, 130`) sizes.
4. Multi-word sizes `n in (63, 64, 65, 130)` explicitly exercising `d = 0, 1, 63, 64, 65, 70, n`
   and targeted regression checks for subtle multi-word / `d > 64` bug traps.
5. Hostile / invalid C ABI inputs (including overflow sizes in a subprocess).
6. Differential verification against the read-only reference prototype `bqfe.py` (skipped cleanly
   when the external reference file is absent).
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stimside._magic.phase_qfe import PhaseQFEComponent
from stimside._native import branch_kernels as bk
from stimside._native import magic_engine as me
from stimside._native import magic_kernels as mk

_NATIVE = Path(bk.__file__).resolve().parent
_PKG_ROOT = _NATIVE.parents[2]
_BQFE_REF_PATH = Path(
    "/usr/local/google/home/mliq/.gemini/jetski/brain/"
    "a4cd4c57-114b-4c18-9db0-bcc178e62419/scratch/magic_ref/bqfe.py"
)
_SCALARS = {
    "size_t": ctypes.c_size_t,
    "uint8_t": ctypes.c_uint8,
    "uint32_t": ctypes.c_uint32,
    "uint64_t": ctypes.c_uint64,
    "int": ctypes.c_int,
    "double": ctypes.c_double,
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
}


# ---------------------------------------------------------------------------------------------
# Shared helpers (dense Pauli/quarter-turn/projector algebra & bit-word conversion)
# ---------------------------------------------------------------------------------------------


def _nw(n: int) -> int:
    return (n + 63) // 64


def _tail_mask_py(n: int) -> int:
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


def _herm_kappa(x_int: int, z_int: int, negative: bool = False) -> int:
    return ((x_int & z_int).bit_count() + (2 if negative else 0)) & 3


def _rand_pauli_int(rng: np.random.Generator, n: int, *, hermitian: bool = True) -> tuple[int, int, int]:
    nw = _nw(n)
    x_int = 0
    z_int = 0
    for w in range(nw):
        xw = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        zw = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        if w == nw - 1:
            tm = _tail_mask_py(n)
            xw &= tm
            zw &= tm
        x_int |= xw << (64 * w)
        z_int |= zw << (64 * w)
    if hermitian:
        k = _herm_kappa(x_int, z_int, bool(rng.integers(0, 2)))
    else:
        k = int(rng.integers(0, 4))
    return x_int, z_int, k


def _dense_apply_pauli(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int) -> np.ndarray:
    """Apply i^kappa X(x) Z(z) (Z acts first) to a little-endian (2^n,) complex128 vector."""
    idx = np.arange(1 << n, dtype=np.uint64)
    # Z(z) acts first on basis |b>, then X(x) maps |b> -> |b ^ x>.
    # So output[b ^ x] = i^kappa * (-1)^(popcount(b & z)) * input[b],
    # i.e. output[c] = i^kappa * (-1)^(popcount((c ^ x) & z)) * input[c ^ x].
    src = idx ^ np.uint64(x_int)
    zb = src & np.uint64(z_int)
    parity = np.zeros(1 << n, dtype=np.uint64)
    tmp = zb.copy()
    while np.any(tmp):
        parity ^= tmp & np.uint64(1)
        tmp >>= np.uint64(1)
    sign = np.where(parity == 0, 1.0, -1.0)
    phase = (1.0, 1.0j, -1.0, -1.0j)[kappa & 3]
    return (phase * sign) * vec[src]


def _dense_apply_qturn(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int) -> np.ndarray:
    """Apply exp(+i*pi*P/4) = (I + iP) / sqrt(2)."""
    p_vec = _dense_apply_pauli(vec, n, x_int, z_int, kappa)
    return (vec + 1.0j * p_vec) * np.sqrt(0.5)


def _dense_apply_project(
    vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int, m: int
) -> np.ndarray:
    """Apply (I + (-1)^m P) / 2."""
    p_vec = _dense_apply_pauli(vec, n, x_int, z_int, kappa)
    return 0.5 * (vec + (1.0 if m == 0 else -1.0) * p_vec)


def _make_random_state(
    rng: np.random.Generator, n: int, steps: int | None = None
) -> tuple[bk.BranchState, complex, np.ndarray]:
    """Build a random stabilizer state on `n <= 8` qubits with parallel exact dense vector."""
    st = bk.BranchState.zero(n)
    w = 1.0 + 0.0j
    vec = np.zeros(1 << n, dtype=np.complex128)
    vec[0] = 1.0
    nw = _nw(n)
    n_steps = int(rng.integers(1, 3 * n + 4)) if steps is None else steps
    for _ in range(n_steps):
        xi, zi, k = _rand_pauli_int(rng, n, hermitian=True)
        xw = _int_to_words(xi, nw)
        zw = _int_to_words(zi, nw)
        if rng.random() < 0.75:
            w *= st.qturn(xw, zw, k)
            vec = _dense_apply_qturn(vec, n, xi, zi, k)
        else:
            w *= st.pauli(xw, zw, k)
            vec = _dense_apply_pauli(vec, n, xi, zi, k)
    return st, w, vec


def _gf2_rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for r in rows:
        x = r
        while x:
            lsb = x & -x
            bit = lsb.bit_length() - 1
            if bit in basis:
                x ^= basis[bit]
            else:
                basis[bit] = x
                break
    return len(basis)


def _bqfe_canonical_key(q: Any) -> bytes:
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


def _load_bqfe_ref() -> Any:
    if not _BQFE_REF_PATH.is_file():
        pytest.skip(f"Reference prototype not present: {_BQFE_REF_PATH}")
    spec = importlib.util.spec_from_file_location("_bqfe_ref", _BQFE_REF_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------
# Group 1: ABI, Prototype Pin, Packaging, Stale-Rebuild & Leak Checks
# ---------------------------------------------------------------------------------------------


def _hpp_prototypes() -> dict[str, tuple[str, list[str]]]:
    src = (_NATIVE / "branch_kernels.hpp").read_text()
    body = src[src.index('extern "C" {'):]
    out: dict[str, tuple[str, list[str]]] = {}
    for m in re.finditer(r"^([\w\*]+)\s+(\w+)\(([^)]*)\)\s*\{", body, re.MULTILINE):
        params = []
        for p in m.group(3).split(","):
            toks = p.replace("const", " ").replace("*", " * ").split()
            if toks and toks != ["void"]:
                params.append("".join(toks[:-1]))
        out[m.group(2)] = (m.group(1), params)
    return out


def test_abi_specs_match_hpp_prototypes_and_symbol_isolation() -> None:
    protos = _hpp_prototypes()
    assert {n for n, _, _ in bk._SPECS} == set(protos)
    assert len(protos) == 46
    assert all(n.startswith("mbr_") for n in protos)
    assert not any(n.startswith("mbr_") for n, _, _ in mk._SPECS)
    assert not any(n.startswith("mbr_") for n, _, _ in me._SPECS)
    assert not any(n.startswith("magic_") or n.startswith("mbe_") for n, _, _ in bk._SPECS)

    lib = bk.get_branch_kernels_lib()
    assert lib.mbr_abi_v1() == 1
    assert bk.branch_abi_ok() is True

    n_ptr = 0
    for name, (ret, params) in protos.items():
        fn = getattr(lib, name)
        assert len(fn.argtypes) == len(params), name
        for i, (ctype, bound) in enumerate(zip(params, fn.argtypes)):
            if ctype.endswith("*"):
                n_ptr += 1
                assert bound is ctypes.c_void_p, f"{name}[{i}] {ctype} bound as {bound}"
            else:
                assert bound is _SCALARS[ctype], f"{name}[{i}] {ctype} bound as {bound}"
        want = None if ret == "void" else ctypes.c_void_p if ret == "void*" else _SCALARS[ret]
        assert fn.restype is want, f"{name} restype"
    assert n_ptr >= 40


def test_status_code_exception_mapping() -> None:
    bk._check_rc(bk.MBR_OK, "op")
    with pytest.raises(ValueError, match="rc=-1"):
        bk._check_rc(bk.MBR_ERR_INVALID_ARG, "op")
    with pytest.raises(MemoryError, match="rc=-2"):
        bk._check_rc(bk.MBR_ERR_OOM, "op")
    with pytest.raises(FloatingPointError, match="rc=-3"):
        bk._check_rc(bk.MBR_ERR_FLOAT_DRIFT, "op")
    with pytest.raises(bk.BranchBudgetError, match="rc=-4"):
        bk._check_rc(bk.MBR_ERR_BUDGET, "op")
    with pytest.raises(RuntimeError, match="-99"):
        bk._check_rc(-99, "op")


def test_packaging_explicit_header_and_no_globs() -> None:
    manifest = (_PKG_ROOT / "MANIFEST.in").read_text()
    pyproject = (_PKG_ROOT / "pyproject.toml").read_text()
    assert "include src/stimside/_native/branch_kernels.hpp" in manifest.splitlines()
    assert '"branch_kernels.hpp"' in pyproject
    assert "*.h" not in manifest and "*.hpp" not in manifest
    assert "*.h" not in pyproject and "*.hpp" not in pyproject
    cpp = (_NATIVE / "magic_kernels.cpp").read_text()
    assert cpp.count('#include "branch_kernels.hpp"') == 1
    assert cpp.rstrip().endswith('#include "branch_kernels.hpp"')


def test_loading_up_to_date_library_never_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_compile(*a: Any, **k: Any) -> None:
        raise AssertionError("unexpected compile")

    monkeypatch.setattr(subprocess, "run", no_compile)
    monkeypatch.setattr(bk, "_LIB_INSTANCE", None)
    lib = bk.get_branch_kernels_lib()
    assert lib.mbr_abi_v1() == 1 and bk.get_branch_kernels_lib() is lib


_STALE_CODE = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "from stimside._native import branch_kernels as bk\n"
    "lib = bk.get_branch_kernels_lib()\n"
    "print('OK', lib.mbr_abi_v1(), bk.branch_abi_ok())\n"
)


def test_stale_library_without_branch_symbols_is_rebuilt_on_first_use(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    keep = shutil.ignore_patterns("__pycache__", "*.so", "*_test.py", "*.build_id")
    shutil.copytree(_NATIVE.parent, site / "stimside", ignore=keep)
    native = site / "stimside" / "_native"
    for other in native.glob("*.cpp"):
        if other.name != "magic_kernels.cpp":
            other.unlink()
    stub = tmp_path / "stale.cpp"
    names = [n for n, _, _ in mk._SPECS] + [n for n, _, _ in me._SPECS]
    body = "".join(f"int {n}() {{ return 0; }}\n" for n in names)
    stub.write_text('extern "C" {\n' + body + "}\n")
    so = native / "libmagic_kernels.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(stub), "-o", str(so)], check=True)
    assert not hasattr(ctypes.CDLL(str(so)), "mbr_abi_v1")
    env = {
        **os.environ,
        "STIMSIDE_PORTABLE_BUILD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
    }
    out = subprocess.run(
        [sys.executable, "-B", "-c", _STALE_CODE, str(site)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0 and out.stdout.split() == ["OK", "1", "True"], out.stderr


def test_live_states_leak_check() -> None:
    gc.collect()
    base = bk.BranchState.live_states()
    assert base == 0
    s1 = bk.BranchState.zero(4)
    s2 = s1.copy()
    assert bk.BranchState.live_states() == 2
    s1.close()
    assert s1.closed
    assert bk.BranchState.live_states() == 1
    with pytest.raises(ValueError, match="closed"):
        s1.canon()
    s2.close()
    assert bk.BranchState.live_states() == 0
    with bk.BranchState.zero(5) as s3:
        assert bk.BranchState.live_states() == 1
        assert s3.n_qubits == 5 and s3.d == 0 and s3.n_words == 1
    assert bk.BranchState.live_states() == 0


# ---------------------------------------------------------------------------------------------
# Group 2: >= 500 Random States per Primitive vs Dense State Vectors (n = 1..8)
# ---------------------------------------------------------------------------------------------


def test_pauli_vs_dense_500_random_states() -> None:
    rng = np.random.default_rng(202601)
    max_err = 0.0
    for trial in range(520):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            xi, zi, k = _rand_pauli_int(rng, n, hermitian=False)
            w *= st.pauli(_int_to_words(xi, 1), _int_to_words(zi, 1), k)
            vec = _dense_apply_pauli(vec, n, xi, zi, k)
            err = float(np.max(np.abs(w * st.dense("little") - vec)))
            if err > max_err:
                max_err = err
    assert max_err < 1e-12, f"pauli max_err={max_err}"


def test_qturn_vs_dense_500_random_states() -> None:
    rng = np.random.default_rng(202602)
    max_err = 0.0
    saw_full_d = False
    for trial in range(520):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            # Explicitly exercise d == n (bug trap #3) on a subset of trials.
            if trial % 16 == 0:
                for q in range(n):
                    if st.d == n:
                        break
                    bit = 1 << q
                    w *= st.qturn(_int_to_words(bit, 1), _int_to_words(bit, 1), 3)
                    vec = _dense_apply_qturn(vec, n, bit, bit, 3)
                if st.d == n:
                    saw_full_d = True
            xi, zi, k = _rand_pauli_int(rng, n, hermitian=True)
            w *= st.qturn(_int_to_words(xi, 1), _int_to_words(zi, 1), k)
            vec = _dense_apply_qturn(vec, n, xi, zi, k)
            err = float(np.max(np.abs(w * st.dense("little") - vec)))
            if err > max_err:
                max_err = err
    assert saw_full_d
    assert max_err < 1e-12, f"qturn max_err={max_err}"


def test_zproject_vs_dense_500_random_states() -> None:
    rng = np.random.default_rng(202603)
    max_err = 0.0
    counts = {"zero": 0, "one": 0, "split": 0}
    for trial in range(520):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            _, zi, _ = _rand_pauli_int(rng, n, hermitian=True)
            sigma = int(rng.integers(0, 2))
            m = int(rng.integers(0, 2))
            before_key = st.canonical_key()
            before_vec = st.dense("little").copy()
            f = st.zproject(_int_to_words(zi, 1), sigma, m)
            proj_vec = _dense_apply_project(vec, n, 0, zi, 2 * sigma, m)
            if f == 0.0:
                counts["zero"] += 1
                assert float(np.max(np.abs(proj_vec))) < 1e-12
                # Impossible outcome must leave handle unchanged!
                assert st.canonical_key() == before_key
                assert np.array_equal(st.dense("little"), before_vec)
            else:
                if abs(abs(f) - 1.0) < 1e-12:
                    counts["one"] += 1
                else:
                    counts["split"] += 1
                err = float(np.max(np.abs((w * f) * st.dense("little") - proj_vec)))
                if err > max_err:
                    max_err = err
    assert all(v > 0 for v in counts.values()), counts
    assert max_err < 1e-12, f"zproject max_err={max_err}"


def test_project_vs_dense_500_random_states() -> None:
    rng = np.random.default_rng(202604)
    max_err = 0.0
    saw_impossible = 0
    saw_x_zero = 0
    for trial in range(520):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            xi, zi, k = _rand_pauli_int(rng, n, hermitian=True)
            if trial % 10 == 0:
                xi = 0
                k = 2 * int(rng.integers(0, 2))
                saw_x_zero += 1
            m = int(rng.integers(0, 2))
            before_key = st.canonical_key()
            before_vec = st.dense("little").copy()
            f = st.project(_int_to_words(xi, 1), _int_to_words(zi, 1), k, m)
            proj_vec = _dense_apply_project(vec, n, xi, zi, k, m)
            if f == 0.0:
                saw_impossible += 1
                assert float(np.max(np.abs(proj_vec))) < 1e-12
                assert st.canonical_key() == before_key
                assert np.array_equal(st.dense("little"), before_vec)
            else:
                err = float(np.max(np.abs((w * f) * st.dense("little") - proj_vec)))
                if err > max_err:
                    max_err = err
    assert saw_impossible > 0 and saw_x_zero > 0
    assert max_err < 1e-12, f"project max_err={max_err}"


def test_canon_vs_dense_and_rcef_invariants_500_random_states() -> None:
    rng = np.random.default_rng(202605)
    max_err = 0.0
    for trial in range(520):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            f = st.canon()
            assert f in (1.0 + 0j, 1.0j, -1.0 + 0j, -1.0j)
            w *= f
            err = float(np.max(np.abs(w * st.dense("little") - vec)))
            if err > max_err:
                max_err = err
            # Verify RCEF & canonical-h invariants directly
            _, d, h_w, A_w, ell, G, piv = st.to_arrays()
            assert list(piv) == sorted(set(piv.tolist()))
            for j in range(d):
                col_int = int(A_w[j, 0])
                assert col_int != 0
                lsb = (col_int & -col_int).bit_length() - 1
                assert int(piv[j]) == lsb
                assert ((int(h_w[0]) >> int(piv[j])) & 1) == 0
                for k_idx in range(d):
                    assert ((int(A_w[k_idx, 0]) >> int(piv[j])) & 1) == (1 if j == k_idx else 0)
            # Idempotence
            key1 = st.canonical_key()
            f2 = st.canon()
            assert f2 == 1.0 + 0j
            assert st.canonical_key() == key1
    assert max_err < 1e-12, f"canon max_err={max_err}"


def test_overlap_and_same_phase_vs_dense_500_random_pairs() -> None:
    rng = np.random.default_rng(202606)
    max_ov_err = 0.0
    max_sp_err = 0.0
    for trial in range(520):
        n = (trial % 8) + 1
        sa, wa, va = _make_random_state(rng, n)
        with sa:
            if trial % 4 == 0:
                # Equal up to phase via re-parameterization (P; P and R(Q); R(-Q))
                sb = sa.copy()
                wb = wa
                vb = va.copy()
                xi, zi, k = _rand_pauli_int(rng, n, hermitian=True)
                xw, zw = _int_to_words(xi, 1), _int_to_words(zi, 1)
                wb *= sb.qturn(xw, zw, k)
                wb *= sb.qturn(xw, zw, (k + 2) & 3)
            elif trial % 4 == 1:
                # Same direction space A, shifted h (exercises disjoint-coset shortcut on both
                # canonical and non-canonical states!)
                sb = sa.copy()
                wb = wa
                vb = va.copy()
                shift = int(rng.integers(1, 1 << n))
                wb *= sb.pauli(_int_to_words(shift, 1), _int_to_words(0, 1), 0)
                vb = _dense_apply_pauli(vb, n, shift, 0, 0)
            else:
                sb, wb, vb = _make_random_state(rng, n)
            with sb:
                ov = sa.overlap(sb)
                # <sa|sb> = conj(1/wa * va) . (1/wb * vb) = wa * conj(wb) * vdot(va, vb)
                # since va = wa |sa>, vb = wb |sb>, so <sa|sb> = vdot(va/wa, vb/wb).
                ref_ov = complex(np.vdot(va / wa, vb / wb))
                err = abs(ov - ref_ov)
                if err > max_ov_err:
                    max_ov_err = err
                eq, ratio = sa.same_phase(sb)
                if eq:
                    assert abs(abs(ref_ov) - 1.0) < 1e-12
                    sp_err = abs(ratio - ref_ov)
                    if sp_err > max_sp_err:
                        max_sp_err = sp_err
                else:
                    assert abs(ref_ov) < 1.0 - 1e-9
                    assert ratio == 0.0j
    assert max_ov_err < 1e-12, f"overlap max_err={max_ov_err}"
    assert max_sp_err < 1e-12, f"same_phase max_err={max_sp_err}"


def test_dense_little_and_big_endian_parity() -> None:
    rng = np.random.default_rng(202607)
    max_err_le = 0.0
    max_err_be = 0.0
    for trial in range(200):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            comp = st.to_phase_qfe(PhaseQFEComponent)
            d_le = st.dense("little")
            d_be = st.dense("big")
            max_err_le = max(max_err_le, float(np.max(np.abs(d_le - comp.to_statevector(endian="little")))))
            max_err_be = max(max_err_be, float(np.max(np.abs(d_be - comp.to_statevector(endian="big")))))
            assert float(np.max(np.abs(w * d_le - vec))) < 1e-12
    assert max_err_le < 1e-15
    assert max_err_be < 1e-15


# ---------------------------------------------------------------------------------------------
# Group 3: definite, generators, canonical key invariance/round-trip, PhaseQFEComponent
# ---------------------------------------------------------------------------------------------


def test_definite_and_generators_500_random_states() -> None:
    rng = np.random.default_rng(202608)
    for trial in range(500):
        n = (trial % 8) + 1
        st, w, vec = _make_random_state(rng, n)
        with st:
            gx, gz, gk = st.generators()
            assert gx.shape == (n, 1) and gz.shape == (n, 1) and gk.shape == (n,)
            symp_rows = []
            for i in range(n):
                xi, zi, ki = int(gx[i, 0]), int(gz[i, 0]), int(gk[i])
                # Hermitian check
                assert ((ki ^ (xi & zi).bit_count()) & 1) == 0
                # Eigenvalue +1 on state and dense vector
                assert st.definite(gx[i], gz[i], ki) == 1
                assert st.definite(gx[i], gz[i], (ki + 2) & 3) == -1
                pvec = _dense_apply_pauli(vec, n, xi, zi, ki)
                assert float(np.max(np.abs(pvec - vec))) < 1e-12
                symp_rows.append(xi | (zi << n))
                # Pairwise commuting
                for j in range(i + 1, n):
                    xj, zj = int(gx[j, 0]), int(gz[j, 0])
                    assert (((xi & zj) ^ (zi & xj)).bit_count() & 1) == 0
            assert _gf2_rank(symp_rows) == n
            # Random Pauli queries
            for _ in range(3):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                ev = st.definite(_int_to_words(xi, 1), _int_to_words(zi, 1), ki)
                exp_val = complex(np.vdot(vec, _dense_apply_pauli(vec, n, xi, zi, ki)))
                if abs(exp_val - 1.0) < 1e-9:
                    assert ev == 1
                elif abs(exp_val + 1.0) < 1e-9:
                    assert ev == -1
                else:
                    assert abs(exp_val) < 1e-9 and ev == 0


def test_canonical_key_invariance_and_round_trip_500_states() -> None:
    rng = np.random.default_rng(202609)
    for trial in range(500):
        n = (trial % 8) + 1
        st, _, _ = _make_random_state(rng, n)
        with st:
            key0 = st.canonical_key()
            # Re-parameterize via P; P and R(Q); R(-Q)
            st2 = st.copy()
            with st2:
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                xw, zw = _int_to_words(xi, 1), _int_to_words(zi, 1)
                st2.pauli(xw, zw, ki)
                st2.pauli(xw, zw, ki)
                xi2, zi2, ki2 = _rand_pauli_int(rng, n, hermitian=True)
                xw2, zw2 = _int_to_words(xi2, 1), _int_to_words(zi2, 1)
                st2.qturn(xw2, zw2, ki2)
                st2.qturn(xw2, zw2, (ki2 + 2) & 3)
                assert st2.canonical_key() == key0
            # Round-trip via from_canonical_key
            with bk.BranchState.from_canonical_key(key0) as restored:
                assert restored.canonical_key() == key0
                eq, ratio = st.same_phase(restored)
                assert eq and abs(abs(ratio) - 1.0) < 1e-15


def test_phase_qfe_component_round_trip_small_and_multiword() -> None:
    rng = np.random.default_rng(202610)
    # Small n = 1..8
    for trial in range(200):
        n = (trial % 8) + 1
        st, _, _ = _make_random_state(rng, n)
        with st:
            st.canon()
            comp = st.to_phase_qfe(PhaseQFEComponent)
            comp.check_invariants(require_canonical_offset=True)
            assert float(np.max(np.abs(st.dense("little") - comp.to_statevector(endian="little")))) < 1e-14
            with bk.BranchState.from_phase_qfe(comp) as st2:
                assert st2.canonical_key() == st.canonical_key()
                assert abs(st.overlap(st2) - 1.0) < 1e-14
    # Multi-word n = 63, 64, 65, 130
    for n in (63, 64, 65, 130):
        nw = _nw(n)
        with bk.BranchState.zero(n) as st:
            for _ in range(min(n, 75)):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                st.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), ki)
            st.canon()
            comp = st.to_phase_qfe(PhaseQFEComponent)
            comp.check_invariants(require_canonical_offset=True)
            with bk.BranchState.from_phase_qfe(comp) as st2:
                assert st2.canonical_key() == st.canonical_key()


# ---------------------------------------------------------------------------------------------
# Group 4: Multi-Word Sizes (63, 64, 65, 130) & Bug-Trap Regression Tests
# ---------------------------------------------------------------------------------------------


def test_multiword_dimensions_and_d_gt_64() -> None:
    rng = np.random.default_rng(202611)
    cases = [
        (63, [0, 1, 63]),
        (64, [0, 1, 63, 64]),
        (65, [0, 1, 63, 64, 65]),
        (130, [0, 1, 63, 64, 65, 70, 130]),
    ]
    for n, d_targets in cases:
        nw = _nw(n)
        tm = np.uint64(_tail_mask_py(n))
        for d_target in d_targets:
            with bk.BranchState.zero(n) as st:
                # Build exact dimension d_target via R(-Y_q) on qubits 0..d_target-1
                for q in range(d_target):
                    bit = 1 << q
                    w_arr = _int_to_words(bit, nw)
                    f = st.qturn(w_arr, w_arr, 3)
                    assert abs(abs(f) - 1.0) < 1e-14
                assert st.d == d_target
                # Entangle across word boundaries while keeping d near/at d_target (incl. d == n!)
                for _ in range(4):
                    xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                    f = st.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), ki)
                    assert abs(abs(f) - 1.0) < 1e-12
                # Verify tail bits zero
                _, d_cur, h_w, A_w, ell, G, piv = st.to_arrays()
                assert (h_w[-1] & ~tm) == np.uint64(0)
                if d_cur > 0:
                    assert np.all((A_w[:, -1] & ~tm) == np.uint64(0))
                # Generators, definite, project
                gx, gz, gk = st.generators()
                assert gx.shape == (n, nw)
                for idx in (0, d_cur // 2, n - 1):
                    assert st.definite(gx[idx], gz[idx], int(gk[idx])) == 1
                    assert st.definite(gx[idx], gz[idx], (int(gk[idx]) + 2) & 3) == -1
                with st.copy() as st_p:
                    f_ok = st_p.project(gx[0], gz[0], int(gk[0]), 0)
                    assert abs(abs(f_ok) - 1.0) < 1e-12
                    key_before = st_p.canonical_key()
                    f_no = st_p.project(gx[0], gz[0], int(gk[0]), 1)
                    assert f_no == 0.0
                    assert st_p.canonical_key() == key_before
                # Overlap with self, canon idempotence, key invariance under P; P and R; R^-1
                assert abs(st.overlap(st) - 1.0) < 1e-12
                key1 = st.canonical_key()
                st.canon()
                assert st.canon() == 1.0 + 0j
                assert st.canonical_key() == key1
                with bk.BranchState.from_canonical_key(key1) as rt:
                    assert rt.canonical_key() == key1
                    assert abs(st.overlap(rt) - 1.0) < 1e-12


def test_bug_traps_definite_triangle_popcount_mod4_and_noncanon_overlap() -> None:
    # Bug trap #1: unreduced gv in definite() on a 3-qubit Gamma triangle (v = (1, 1, 1)).
    # State |s> = 2^{-3/2} sum_{u in F_2^3} (-1)^{u0 u1 + u1 u2 + u0 u2} |u>.
    # X0 X1 X2 maps u -> u + (1,1,1): Q(u + 1) - Q(u) = (0 + 6) + 2*(2)*u = 2 mod 4 -> ev == -1!
    G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    A = np.array([[1], [2], [4]], dtype=np.uint64)
    h = np.array([0], dtype=np.uint64)
    ell = np.array([0, 0, 0], dtype=np.uint8)
    piv = np.array([0, 1, 2], dtype=np.int32)
    with bk.BranchState.from_arrays(3, h, A, ell, G, piv) as tri:
        x_all = np.array([7], dtype=np.uint64)
        z_zero = np.array([0], dtype=np.uint64)
        assert tri.definite(x_all, z_zero, 0) == -1
        vec = tri.dense("little")
        assert np.allclose(_dense_apply_pauli(vec, 3, 7, 0, 0), -vec, atol=1e-14)

    # Bug trap #2: popcount(x & z) mod 4 across word boundary on n=65 (Y_0 Y_64).
    # Y_0 Y_64 = (i X_0 Z_0)(i X_64 Z_64) = i^2 X(1|2^64) Z(1|2^64), kappa=2 is +Y_0 Y_64.
    n = 65
    nw = 2
    x_yy = np.array([1, 1], dtype=np.uint64)
    z_yy = np.array([1, 1], dtype=np.uint64)
    with bk.BranchState.zero(n) as st:
        # Prepare |+i>_0 |+i>_64 where Y_0 = +1, Y_64 = +1, so (+Y_0 Y_64) has eigenvalue +1!
        # R(X)|0> = (|0> + i|1>)/sqrt(2) = |+i>.
        st.qturn(np.array([1, 0], dtype=np.uint64), np.array([0, 0], dtype=np.uint64), 0)
        st.qturn(np.array([0, 1], dtype=np.uint64), np.array([0, 0], dtype=np.uint64), 0)
        assert st.definite(x_yy, z_yy, 2) == 1
        assert st.definite(x_yy, z_yy, 0) == -1

    # Bug trap #5: overlap with same A, different h on non-canonical states (same coset vs disjoint).
    with bk.BranchState.zero(2) as a, bk.BranchState.zero(2) as b, bk.BranchState.zero(2) as c:
        # Prepare |+0> in a and b
        a.qturn(np.array([1], dtype=np.uint64), np.array([1], dtype=np.uint64), 3)
        b.qturn(np.array([1], dtype=np.uint64), np.array([1], dtype=np.uint64), 3)
        c.qturn(np.array([1], dtype=np.uint64), np.array([1], dtype=np.uint64), 3)
        # Apply X_0 to b: non-canonical h=1, still same coset (|+0>) -> overlap MUST be 1!
        b.pauli(np.array([1], dtype=np.uint64), np.array([0], dtype=np.uint64), 0)
        assert abs(a.overlap(b) - 1.0) < 1e-14
        # Apply X_1 to c: disjoint coset (|+1>) -> overlap MUST be 0!
        c.pauli(np.array([2], dtype=np.uint64), np.array([0], dtype=np.uint64), 0)
        assert a.overlap(c) == 0.0j


# ---------------------------------------------------------------------------------------------
# Group 5: Hostile / Invalid C ABI Inputs (returns -1 cleanly, zero crashes)
# ---------------------------------------------------------------------------------------------


def test_hostile_c_abi_inputs_return_invalid_arg() -> None:
    lib = bk.get_branch_kernels_lib()
    live0 = int(lib.mbr_live_states())
    out_h = ctypes.c_void_p()
    out_ri = (ctypes.c_double * 2)()
    out_i32 = ctypes.c_int32()
    out_sz = ctypes.c_size_t()

    # Null pointers & zero/bad dimensions
    assert lib.mbr_state_create_zero(0, ctypes.addressof(out_h)) == bk.MBR_ERR_INVALID_ARG
    assert lib.mbr_state_create_zero(4, 0) == bk.MBR_ERR_INVALID_ARG
    assert lib.mbr_state_copy(0, ctypes.addressof(out_h)) == bk.MBR_ERR_INVALID_ARG
    assert lib.mbr_state_header(0, ctypes.addressof(out_sz), ctypes.addressof(out_sz), ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
    lib.mbr_state_destroy(0)  # null-safe no-op

    # Corrupt from_arrays inputs
    h_ok = np.array([0], dtype=np.uint64)
    A_ok = np.array([[1]], dtype=np.uint64)
    ell_ok = np.array([0], dtype=np.uint8)
    G_ok = np.array([[0]], dtype=np.uint8)
    piv_ok = np.array([0], dtype=np.int32)

    # d > n
    assert lib.mbr_state_from_arrays(
        1, 2, h_ok.ctypes.data, A_ok.ctypes.data, ell_ok.ctypes.data, G_ok.ctypes.data,
        piv_ok.ctypes.data, ctypes.addressof(out_h),
    ) == bk.MBR_ERR_INVALID_ARG
    # Non-zero tail bit in h (n = 3, bit 3 set)
    h_tail = np.array([1 << 3], dtype=np.uint64)
    assert lib.mbr_state_from_arrays(
        3, 0, h_tail.ctypes.data, 0, 0, 0, 0, ctypes.addressof(out_h)
    ) == bk.MBR_ERR_INVALID_ARG
    # Non-zero tail bit in A
    A_tail = np.array([[1 | (1 << 5)]], dtype=np.uint64)
    assert lib.mbr_state_from_arrays(
        3, 1, h_ok.ctypes.data, A_tail.ctypes.data, ell_ok.ctypes.data, G_ok.ctypes.data,
        piv_ok.ctypes.data, ctypes.addressof(out_h),
    ) == bk.MBR_ERR_INVALID_ARG
    # ell >= 4
    ell_bad = np.array([4], dtype=np.uint8)
    assert lib.mbr_state_from_arrays(
        3, 1, h_ok.ctypes.data, A_ok.ctypes.data, ell_bad.ctypes.data, G_ok.ctypes.data,
        piv_ok.ctypes.data, ctypes.addressof(out_h),
    ) == bk.MBR_ERR_INVALID_ARG
    # Non-hollow Gamma
    G_diag = np.array([[1]], dtype=np.uint8)
    assert lib.mbr_state_from_arrays(
        3, 1, h_ok.ctypes.data, A_ok.ctypes.data, ell_ok.ctypes.data, G_diag.ctypes.data,
        piv_ok.ctypes.data, ctypes.addressof(out_h),
    ) == bk.MBR_ERR_INVALID_ARG
    # Asymmetric & non-binary Gamma
    A2 = np.array([[1], [2]], dtype=np.uint64)
    ell2 = np.array([0, 0], dtype=np.uint8)
    piv2 = np.array([0, 1], dtype=np.int32)
    G_asym = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    G_val2 = np.array([[0, 2], [2, 0]], dtype=np.uint8)
    for G_bad in (G_asym, G_val2):
        assert lib.mbr_state_from_arrays(
            3, 2, h_ok.ctypes.data, A2.ctypes.data, ell2.ctypes.data, G_bad.ctypes.data,
            piv2.ctypes.data, ctypes.addressof(out_h),
        ) == bk.MBR_ERR_INVALID_ARG
    # Duplicate / out-of-range / non-principal pivots
    for piv_bad in (
        np.array([0, 0], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([-1, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
    ):
        assert lib.mbr_state_from_arrays(
            3, 2, h_ok.ctypes.data, A2.ctypes.data, ell2.ctypes.data,
            np.zeros((2, 2), dtype=np.uint8).ctypes.data,
            piv_bad.ctypes.data, ctypes.addressof(out_h),
        ) == bk.MBR_ERR_INVALID_ARG

    # Hostile operations on a live state (n = 3)
    with bk.BranchState.zero(3) as st:
        sh = st._handle
        x_tail = np.array([1 << 3], dtype=np.uint64)
        z_zero = np.array([0], dtype=np.uint64)
        x_one = np.array([1], dtype=np.uint64)
        z_one = np.array([1], dtype=np.uint64)
        # Non-zero tail bits
        assert lib.mbr_state_pauli(sh, x_tail.ctypes.data, z_zero.ctypes.data, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_qturn(sh, x_tail.ctypes.data, z_zero.ctypes.data, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_zproject(sh, x_tail.ctypes.data, 0, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_project(sh, x_tail.ctypes.data, z_zero.ctypes.data, 0, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_definite(sh, x_tail.ctypes.data, z_zero.ctypes.data, 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
        # Non-Hermitian Paulis (X_0 Z_0 with kappa=0 is -i Y_0, anti-Hermitian)
        assert lib.mbr_state_qturn(sh, x_one.ctypes.data, z_one.ctypes.data, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_project(sh, x_one.ctypes.data, z_one.ctypes.data, 0, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_definite(sh, x_one.ctypes.data, z_one.ctypes.data, 0, ctypes.addressof(out_i32)) == bk.MBR_ERR_INVALID_ARG
        # Bad sigma / m / endian / dense length / small key buffer
        assert lib.mbr_state_zproject(sh, z_one.ctypes.data, 2, 0, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_zproject(sh, z_one.ctypes.data, 0, 2, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_project(sh, x_one.ctypes.data, z_zero.ctypes.data, 0, 2, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
        psi = np.empty(8, dtype=np.complex128)
        assert lib.mbr_state_to_dense(sh, 2, psi.ctypes.data, 8) == bk.MBR_ERR_INVALID_ARG
        assert lib.mbr_state_to_dense(sh, 0, psi.ctypes.data, 4) == bk.MBR_ERR_INVALID_ARG
        small_buf = np.empty(4, dtype=np.uint8)
        assert lib.mbr_state_canonical_key(sh, small_buf.ctypes.data, 4, ctypes.addressof(out_sz)) == bk.MBR_ERR_INVALID_ARG
        # Mismatched n_qubits in overlap / same_phase
        with bk.BranchState.zero(4) as st4:
            assert lib.mbr_state_overlap(sh, st4._handle, ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG
            assert lib.mbr_state_same_phase(sh, st4._handle, ctypes.addressof(out_i32), ctypes.addressof(out_ri)) == bk.MBR_ERR_INVALID_ARG

    # n = 27 dense must reject
    with bk.BranchState.zero(27) as st27:
        with pytest.raises(ValueError, match="26"):
            st27.dense()
        psi_dummy = np.empty(1, dtype=np.complex128)
        assert lib.mbr_state_to_dense(st27._handle, 0, psi_dummy.ctypes.data, 1 << 27) == bk.MBR_ERR_INVALID_ARG

    # Corrupt canonical keys
    with bk.BranchState.zero(3) as st:
        st.qturn(np.array([1], dtype=np.uint64), np.array([1], dtype=np.uint64), 3)
        st.qturn(np.array([2], dtype=np.uint64), np.array([2], dtype=np.uint64), 3)
        good = bytearray(st.canonical_key())
    bad_keys = [
        b"",
        b"\x00\x00\x00",
        bytes(good[:-1]),
        bytes(good) + b"\x00",
        struct.pack("<II", 0, 0),
        struct.pack("<II", 3, 4) + bytes(good[8:]),
    ]
    # Non-zero h tail bit
    b_htail = bytearray(good)
    b_htail[8] |= 1 << 3
    bad_keys.append(bytes(b_htail))
    # Non-canonical h (bit 0 set on pivot 0)
    b_hcanon = bytearray(good)
    b_hcanon[8] |= 1
    bad_keys.append(bytes(b_hcanon))
    # Bad ell >= 4
    b_ell = bytearray(good)
    b_ell[8 + 3 * 8] = 4
    bad_keys.append(bytes(b_ell))
    # Unused top bits in Gamma byte set (d=2 -> n_tri=1 -> bit 1 is unused)
    b_gpad = bytearray(good)
    b_gpad[-1] |= 0x02
    bad_keys.append(bytes(b_gpad))
    for bad in bad_keys:
        with pytest.raises(ValueError):
            bk.BranchState.from_canonical_key(bad)

    assert int(lib.mbr_live_states()) == live0


_OVERFLOW_SUBPROCESS = r"""
import ctypes, sys
from stimside._native import branch_kernels as bk
lib = bk.get_branch_kernels_lib()
out_h = ctypes.c_void_p()
for huge in (1 << 60, 1 << 62, 1 << 63, (1 << 64) - 1, (1 << 31)):
    rc1 = lib.mbr_state_create_zero(huge, ctypes.addressof(out_h))
    assert rc1 == bk.MBR_ERR_INVALID_ARG, (huge, rc1)
    rc2 = lib.mbr_state_from_arrays(huge, 0, 0, 0, 0, 0, 0, ctypes.addressof(out_h))
    assert rc2 == bk.MBR_ERR_INVALID_ARG, (huge, rc2)
print("OK")
"""


def test_overflow_sizes_rejected_cleanly_in_subprocess() -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(_PKG_ROOT / "src")}
    res = subprocess.run(
        [sys.executable, "-B", "-c", _OVERFLOW_SUBPROCESS],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert res.returncode == 0 and res.stdout.strip() == "OK", res.stderr


def test_python_wrapper_input_validation_and_noncontiguous_memoryview() -> None:
    gc.collect()
    live0 = bk.BranchState.live_states()
    assert live0 == 0

    # 1. Non-contiguous strided & 2D memoryview round-trip in from_canonical_key + empty bytes
    with bk.BranchState.zero(3) as st:
        st.qturn(np.array([1], dtype=np.uint64), np.array([1], dtype=np.uint64), 3)
        st.qturn(np.array([2], dtype=np.uint64), np.array([2], dtype=np.uint64), 3)
        key = st.canonical_key()
    interleaved = bytearray(b"".join(bytes([b, 0xFF]) for b in key))
    strided_mv = memoryview(interleaved)[::2]
    assert not strided_mv.c_contiguous
    with bk.BranchState.from_canonical_key(strided_mv) as rt:
        assert rt.canonical_key() == key
    if len(key) % 2 == 0:
        mv_2d = memoryview(np.frombuffer(key, dtype=np.uint8).reshape(2, len(key) // 2))
        with bk.BranchState.from_canonical_key(mv_2d) as rt2:
            assert rt2.canonical_key() == key
    for empty_key in (b"", memoryview(b"")):
        with pytest.raises(ValueError):
            bk.BranchState.from_canonical_key(empty_key)

    # 2. Out-of-range / negative integer rejection in from_arrays
    with pytest.raises(ValueError):
        bk.BranchState.from_arrays(
            64,
            np.array([0], dtype=np.uint64),
            np.array([[-1]], dtype=np.int64),
            np.array([0], dtype=np.uint8),
            np.array([[0]], dtype=np.uint8),
            np.array([0], dtype=np.int32),
        )
    for bad_ell in (np.array([256], dtype=np.int64), np.array([-256], dtype=np.int64)):
        with pytest.raises(ValueError):
            bk.BranchState.from_arrays(2, [0], [[1]], bad_ell, [[0]], [0])
    for bad_gamma in (
        np.array([[256]], dtype=np.int64),
        np.array([[257]], dtype=np.int64),
        np.array([[256]], dtype=np.uint16),
    ):
        with pytest.raises(ValueError):
            bk.BranchState.from_arrays(2, [0], [[1]], [0], bad_gamma, [0])
    with pytest.raises(ValueError):
        bk.BranchState.from_arrays(2, [0], [[1]], [0], [[0]], np.array([1 << 32], dtype=np.int64))

    # 3. Float and bool array rejection in from_arrays and Pauli-word methods
    for bad_h in (
        np.array([0.5], dtype=np.float64),
        np.array([1.0], dtype=np.float32),
        np.array([True], dtype=np.bool_),
        [True],
        [0.5],
        np.array([True], dtype=object),
    ):
        with pytest.raises((TypeError, ValueError)):
            bk.BranchState.from_arrays(2, bad_h, [[1]], [0], [[0]], [0])

    with bk.BranchState.zero(2) as st:
        z0 = np.array([0], dtype=np.uint64)
        for bad_w in (
            np.array([1.5], dtype=np.float64),
            np.array([True], dtype=np.bool_),
            [True],
            [1.5],
            [-1],
            [1 << 64],
            np.array([1 << 64], dtype=object),
            np.array([-1], dtype=object),
        ):
            with pytest.raises((TypeError, ValueError)):
                st.pauli(bad_w, z0, 0)
            with pytest.raises((TypeError, ValueError)):
                st.qturn(bad_w, z0, 0)
            with pytest.raises((TypeError, ValueError)):
                st.zproject(bad_w, 0, 0)
            with pytest.raises((TypeError, ValueError)):
                st.project(bad_w, z0, 0, 0)
            with pytest.raises((TypeError, ValueError)):
                st.definite(bad_w, z0, 0)

    # 4. Valid Python int sequences >= 2**63 (incl. object dtype & mixed [0, 1 << 63]) succeed cleanly
    top_bit = 1 << 63
    with bk.BranchState.from_arrays(
        64, [0], [[top_bit]], [0], [[0]], [63]
    ) as st64:
        assert st64.definite([top_bit], [0], 0) == 1
        assert st64.definite(np.array([top_bit], dtype=object), [0], 0) == 1
        assert abs(st64.pauli([top_bit], [0], 0) - 1.0) < 1e-15
        assert abs(st64.qturn([top_bit], [0], 0) - (1.0 + 1.0j) * np.sqrt(0.5)) < 1e-15
        assert abs(st64.zproject([0], 0, 0) - 1.0) < 1e-15
        assert abs(st64.project([top_bit], [0], 0, 0) - 1.0) < 1e-15

    with bk.BranchState.from_arrays(
        128, [0, 0], [[0, top_bit]], [0], [[0]], [127]
    ) as st128:
        assert st128.definite([0, top_bit], [0, 0], 0) == 1

    # Out-of-range Python ints in from_arrays must raise ValueError
    for bad_py_u64 in ([1 << 64], [-1], np.array([1 << 64], dtype=object)):
        with pytest.raises(ValueError):
            bk.BranchState.from_arrays(64, bad_py_u64, [[1]], [0], [[0]], [0])

    # 5. Oversized n_qubits must raise ValueError before ctypes.c_size_t wraparound
    for bad_n in (0, -1, (1 << 31), (1 << 64) + 2, 1 << 80):
        with pytest.raises(ValueError):
            bk.BranchState.zero(bad_n)
        with pytest.raises(ValueError):
            bk.BranchState.from_arrays(
                bad_n, [0], np.empty((0, 1), dtype=np.uint64), [], np.empty((0, 0), dtype=np.uint8),
                [],
            )

    # 6. Unaligned native C-contiguous uint64 / int32 views must copy to aligned buffers
    raw = np.zeros(32, dtype=np.uint8)
    u64_unaligned = np.ndarray(shape=(1,), dtype=np.uint64, buffer=raw[1:9])
    u64_unaligned[0] = 1
    assert u64_unaligned.flags.c_contiguous and not u64_unaligned.flags.aligned
    assert u64_unaligned.ctypes.data % 8 == 1
    out_u64 = bk._as_u64_words(u64_unaligned, 1, "h_words")
    assert out_u64.flags.c_contiguous and out_u64.flags.aligned and out_u64.ctypes.data % 8 == 0
    assert int(out_u64[0]) == 1

    i32_unaligned = np.ndarray(shape=(1,), dtype=np.int32, buffer=raw[17:21])
    i32_unaligned[0] = 0
    assert i32_unaligned.flags.c_contiguous and not i32_unaligned.flags.aligned
    assert i32_unaligned.ctypes.data % 4 == 1
    out_i32 = bk._as_contiguous(i32_unaligned, np.int32, (1,), "pivots")
    assert out_i32.flags.c_contiguous and out_i32.flags.aligned and out_i32.ctypes.data % 4 == 0
    assert int(out_i32[0]) == 0

    u64_mat_unaligned = np.ndarray(shape=(1, 1), dtype=np.uint64, buffer=raw[1:9])
    with bk.BranchState.from_arrays(
        2, u64_unaligned, u64_mat_unaligned, [0], [[0]], i32_unaligned
    ) as st_ua:
        assert abs(st_ua.pauli(u64_unaligned, [0], 0) - 1.0) < 1e-15

    # 7. timedelta64[ns] (empty and non-empty) must raise TypeError
    td_nonempty = np.array([1], dtype="timedelta64[ns]")
    td_zero = np.array([0], dtype="timedelta64[ns]")
    td_empty_1d = np.array([], dtype="timedelta64[ns]")
    td_empty_2d = np.empty((0, 1), dtype="timedelta64[ns]")
    with pytest.raises(TypeError):
        bk.BranchState.from_arrays(2, td_nonempty, [[1]], td_zero, [[0]], [0])
    with pytest.raises(TypeError):
        bk.BranchState.from_arrays(
            2, [0], td_empty_2d, td_empty_1d, np.empty((0, 0), dtype=np.uint8), []
        )
    with pytest.raises(TypeError):
        bk.BranchState.from_arrays(
            2, [0], np.empty((0, 1), dtype=np.uint64), [], np.empty((0, 0), dtype=np.uint8),
            td_empty_1d,
        )
    with bk.BranchState.zero(2) as st_td:
        with pytest.raises(TypeError):
            st_td.pauli(td_nonempty, [0], 0)

    assert bk.BranchState.live_states() == 0


# ---------------------------------------------------------------------------------------------
# Group 6: Differential Verification vs Read-Only Reference Prototype `bqfe.py`
# ---------------------------------------------------------------------------------------------


def test_differential_vs_bqfe_small_500_random_states() -> None:
    bqfe = _load_bqfe_ref()
    rng = np.random.default_rng(202612)
    max_amp_err = 0.0
    max_scalar_err = 0.0
    max_ov_err = 0.0

    for trial in range(520):
        n = (trial % 8) + 1
        st = bk.BranchState.zero(n)
        q = bqfe.Q(n)
        with st:
            for _ in range(int(rng.integers(2, 2 * n + 4))):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                xw, zw = _int_to_words(xi, 1), _int_to_words(zi, 1)
                op = int(rng.integers(0, 4))
                if op == 0:
                    f1 = st.pauli(xw, zw, ki)
                    f2 = q.pauli(xi, zi, ki)
                elif op == 1:
                    f1 = st.qturn(xw, zw, ki)
                    f2 = q.qturn(xi, zi, ki)
                elif op == 2:
                    sigma = int(rng.integers(0, 2))
                    m = int(rng.integers(0, 2))
                    q_bak = q.copy()
                    f1 = st.zproject(zw, sigma, m)
                    f2 = q.zproject(zi, sigma, m)
                    if f2 == 0:
                        q = q_bak
                else:
                    m = int(rng.integers(0, 2))
                    q_bak = q.copy()
                    f1 = st.project(xw, zw, ki, m)
                    f2 = q.project((xi, zi, ki), m)
                    if f2 == 0:
                        q = q_bak
                max_scalar_err = max(max_scalar_err, abs(f1 - f2))

            # Compare definite on random Paulis
            for _ in range(3):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                assert st.definite(_int_to_words(xi, 1), _int_to_words(zi, 1), ki) == q.definite((xi, zi, ki))

            # Compare dense (big-endian matches bqfe.Q.dense())
            amp_err = float(np.max(np.abs(st.dense("big").reshape((2,) * n) - q.dense())))
            max_amp_err = max(max_amp_err, amp_err)

            # Compare canon scalar & exact canonical key bytes
            fc1 = st.canon()
            fc2 = q.canon()
            max_scalar_err = max(max_scalar_err, abs(fc1 - fc2))
            assert st.canonical_key() == _bqfe_canonical_key(q)

            # Compare overlap against a second random state
            st2 = bk.BranchState.zero(n)
            q2 = bqfe.Q(n)
            with st2:
                for _ in range(n + 2):
                    xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                    st2.qturn(_int_to_words(xi, 1), _int_to_words(zi, 1), ki)
                    q2.qturn(xi, zi, ki)
                ov1 = st.overlap(st2)
                ov2 = bqfe.overlap(q, q2)
                max_ov_err = max(max_ov_err, abs(ov1 - ov2))

    assert max_amp_err < 1e-12, f"diff max_amp_err={max_amp_err}"
    assert max_scalar_err < 1e-12, f"diff max_scalar_err={max_scalar_err}"
    assert max_ov_err < 1e-12, f"diff max_ov_err={max_ov_err}"


def test_differential_vs_bqfe_multiword_and_d_gt_64() -> None:
    bqfe = _load_bqfe_ref()
    rng = np.random.default_rng(202613)
    max_scalar_err = 0.0
    max_ov_err = 0.0

    for n, d_target in ((63, 63), (64, 64), (65, 65), (130, 70), (130, 130)):
        nw = _nw(n)
        st = bk.BranchState.zero(n)
        q = bqfe.Q(n)
        with st:
            for qb in range(d_target):
                bit = 1 << qb
                w_arr = _int_to_words(bit, nw)
                f1 = st.qturn(w_arr, w_arr, 3)
                f2 = q.qturn(bit, bit, 3)
                max_scalar_err = max(max_scalar_err, abs(f1 - f2))
            assert st.d == d_target and len(q.cols) == d_target
            # Random multi-word qturns, paulis, and projections (including at d == n!)
            for _ in range(10):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                f1 = st.qturn(xw, zw, ki)
                f2 = q.qturn(xi, zi, ki)
                max_scalar_err = max(max_scalar_err, abs(f1 - f2))
            for _ in range(5):
                xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
                assert st.definite(xw, zw, ki) == q.definite((xi, zi, ki))
            # Project onto one non-trivial Pauli (or definite generator)
            gx, gz, gk = st.generators()
            g_xi, g_zi, g_ki = _words_to_int(gx[-1]), _words_to_int(gz[-1]), int(gk[-1])
            assert st.definite(gx[-1], gz[-1], g_ki) == q.definite((g_xi, g_zi, g_ki)) == 1
            xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
            xw, zw = _int_to_words(xi, nw), _int_to_words(zi, nw)
            q_bak = q.copy()
            fp1 = st.project(xw, zw, ki, 0)
            fp2 = q.project((xi, zi, ki), 0)
            if fp2 == 0:
                q = q_bak
            max_scalar_err = max(max_scalar_err, abs(fp1 - fp2))

            # Canonicalize & verify bit-exact canonical key bytes
            fc1 = st.canon()
            fc2 = q.canon()
            max_scalar_err = max(max_scalar_err, abs(fc1 - fc2))
            assert st.canonical_key() == _bqfe_canonical_key(q)

            # Overlap with perturbed copy
            st2 = st.copy()
            q2 = q.copy()
            with st2:
                for _ in range(3):
                    xi, zi, ki = _rand_pauli_int(rng, n, hermitian=True)
                    st2.qturn(_int_to_words(xi, nw), _int_to_words(zi, nw), ki)
                    q2.qturn(xi, zi, ki)
                max_ov_err = max(max_ov_err, abs(st.overlap(st2) - bqfe.overlap(q, q2)))

    assert max_scalar_err < 1e-12, f"multiword diff max_scalar_err={max_scalar_err}"
    assert max_ov_err < 1e-12, f"multiword diff max_ov_err={max_ov_err}"
