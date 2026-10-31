"""Unit and C-ABI parity tests for `stimside.util.magic_kernels` (`libmagic_kernels.so`)."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess
from unittest import mock

import numpy as np
import pytest
import stim

from stimside.util import magic_kernels as mk
from stimside.util import phase_qfe as pq
from stimside.util.magic_compile import (
    compile_direct,
    compile_selector,
    run_direct,
    run_selector,
)

EXPECTED_SYMBOLS = (
    "magic_abi_v1",
    "magic_apply_pauli_batch",
    "magic_apply_quarter_turn",
    "magic_canonicalize_and_merge",
    "magic_branch_pauli_t",
    "magic_branch_ccz",
    "magic_project_diagonal",
    "magic_overlap",
    "magic_quadratic_gauss_sum",
)


def _assert_shot_parity(
    cpp_exp: pq.ShotExpansion, py_exp: pq.ShotExpansion, atol: float = 1e-13
) -> None:
    cpp_exp.check_invariants(require_canonical_offset=cpp_exp._is_canonical_merged)
    py_exp.check_invariants(require_canonical_offset=py_exp._is_canonical_merged)
    assert cpp_exp.d == py_exp.d
    assert cpp_exp.chi == py_exp.chi
    assert abs(cpp_exp.log_mass - py_exp.log_mass) <= atol
    assert np.array_equal(cpp_exp.A_words, py_exp.A_words)
    assert np.array_equal(cpp_exp.a_union, py_exp.a_union)
    assert np.array_equal(cpp_exp.Gamma, py_exp.Gamma)
    assert np.array_equal(cpp_exp.pivots, py_exp.pivots)
    assert np.array_equal(cpp_exp.ell_odd, py_exp.ell_odd)
    assert np.array_equal(cpp_exp.h_words, py_exp.h_words)
    assert np.array_equal(cpp_exp.ell, py_exp.ell)
    assert np.allclose(cpp_exp.weights, py_exp.weights, atol=atol, rtol=0.0)


def test_loader_abi_symbols_caching_rebuild_and_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    so_path = mk._resolve_magic_kernels_so(force_recompile=False)
    assert so_path.is_file() and so_path.name == "libmagic_kernels.so"
    live_stat = (so_path.stat().st_ino, so_path.stat().st_mtime_ns)

    orig_lib = mk._LIB_INSTANCE
    lib1: ctypes.CDLL | None = None
    try:
        mk._LIB_INSTANCE = None
        pq._reset_cpp_kernel_cache()
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("unexpected compile")):
            lib1 = mk.get_magic_kernels_lib(force_recompile=False)
            lib2 = mk.get_magic_kernels_lib(force_recompile=False)
        assert lib1 is lib2
        for sym in EXPECTED_SYMBOLS:
            assert hasattr(lib1, sym), f"Missing symbol {sym}"
        assert int(lib1.magic_abi_v1()) == 1

        exp = pq.ShotExpansion.zero_state(2, use_cpp_kernels=True)
        pq.apply_pauli(exp, [1, 0], [0, 0], 0)
        assert pq._CACHED_MAGIC_LIB is not None and pq._CACHED_MAGIC_LIB is lib1

        tmp_util = tmp_path / "rebuild_pkg" / "src" / "stimside" / "util"
        tmp_util.mkdir(parents=True)
        (tmp_path / "rebuild_pkg" / "setup.py").write_text("# dummy\n", encoding="utf-8")
        stub_cpp = (
            'extern "C" {\nint magic_abi_v1() { return 1; }\n'
            + "".join(f"void {s}() {{}}\n" for s in EXPECTED_SYMBOLS if s != "magic_abi_v1")
            + "}\n"
        )
        (tmp_util / "magic_kernels.cpp").write_text(stub_cpp, encoding="utf-8")
        with monkeypatch.context() as mp:
            mp.setattr(mk, "__file__", str(tmp_util / "magic_kernels.py"))
            mp.setattr(mk, "_find_prebuilt_so_candidate", lambda *a, **kw: None)
            rebuilt = mk.get_magic_kernels_lib(force_recompile=True)
            assert (tmp_util / "libmagic_kernels.so").is_file()
            assert rebuilt is not lib1 and rebuilt._handle != lib1._handle
            assert mk._LIB_INSTANCE is rebuilt
            assert int(rebuilt.magic_abi_v1()) == 1
            assert pq._CACHED_MAGIC_LIB is None
    finally:
        mk._LIB_INSTANCE = lib1 if lib1 is not None else orig_lib
        pq._reset_cpp_kernel_cache()

    assert (so_path.stat().st_ino, so_path.stat().st_mtime_ns) == live_stat

    with monkeypatch.context() as mp:
        empty_dir = tmp_path / "missing_src"
        empty_dir.mkdir(parents=True)
        mp.setattr(mk, "_find_prebuilt_so_candidate", lambda *a, **kw: None)
        mp.setattr(mk, "__file__", str(empty_dir / "magic_kernels.py"))
        with pytest.raises(FileNotFoundError):
            mk._resolve_magic_kernels_so(force_recompile=False)


def test_zero_size_raw_abi_contiguity_guard_and_rejection() -> None:
    with pytest.raises(ValueError):
        pq.ShotExpansion.zero_state(0)

    lib = mk.get_magic_kernels_lib()
    null_d = ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    null_u64 = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint64))
    null_u8 = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
    null_i32 = ctypes.cast(0, ctypes.POINTER(ctypes.c_int32))
    z_u64 = (ctypes.c_uint64 * 1)(0)
    out_chi = ctypes.c_size_t(99)
    out_d = ctypes.c_size_t(99)
    out_norm = ctypes.c_double(99.0)

    lib.magic_apply_pauli_batch(0, 0, 1, null_d, null_u64, null_u8, null_u64, z_u64, z_u64, z_u64, 0)
    lib.magic_canonicalize_and_merge(
        0, 0, 1, null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
        1e-13, ctypes.byref(out_chi), null_d, null_u64, null_u8, ctypes.byref(out_norm),
    )
    assert out_chi.value == 0 and out_norm.value == 0.0

    lib.magic_branch_pauli_t(
        0, 0, 1, null_d, null_u64, null_u8, null_u64, z_u64, null_u8, null_i32, null_u8,
        z_u64, z_u64, 0, 0, 1e-13, ctypes.byref(out_chi), null_d, null_u64, null_u8, ctypes.byref(out_norm),
    )
    assert out_chi.value == 0 and out_norm.value == 0.0

    lib.magic_project_diagonal(
        0, 0, 1, null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
        z_u64, 0, 0, 1e-13, ctypes.byref(out_chi), ctypes.byref(out_d),
        null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8, ctypes.byref(out_norm),
    )
    assert out_chi.value == 0 and out_d.value == 0 and out_norm.value == 0.0

    # Non-Hermitian Pauli (kappa=1 on Z_0) rejected by magic_apply_quarter_turn with rc != 0.
    rc = lib.magic_apply_quarter_turn(
        0, 0, 1, 1, null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
        z_u64, (ctypes.c_uint64 * 1)(1), 1, ctypes.byref(out_chi), ctypes.byref(out_d),
        null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
    )
    assert rc != 0

    # Raw magic_overlap with d_bra=0, d_ket=0 on matching and mismatching h_words.
    h0 = (ctypes.c_uint64 * 2)(5, 9)
    h1 = (ctypes.c_uint64 * 2)(5, 11)
    out_ri = (ctypes.c_double * 2)(0.0, 0.0)
    lib.magic_overlap(2, 0, h0, null_u64, null_u8, null_u8, 0, h0, null_u64, null_u8, null_u8, out_ri)
    assert (out_ri[0], out_ri[1]) == (1.0, 0.0)
    lib.magic_overlap(2, 0, h0, null_u64, null_u8, null_u8, 0, h1, null_u64, null_u8, null_u8, out_ri)
    assert (out_ri[0], out_ri[1]) == (0.0, 0.0)

    # Strided / int64 PhaseQFEComponent arrays handled safely via cpp_overlap and Python fallback.
    base_c = pq.PhaseQFEComponent.from_unpacked(
        [0, 0, 1], np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8), [1, 2], [[0, 1], [1, 0]], [0, 1]
    )
    strided_ell = np.array([1, 99, 2, 99], dtype=np.int64)[::2]
    strided_G = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)[::2, ::2]
    strided_c = pq.PhaseQFEComponent(
        n_qubits=3, d=2, h_words=base_c.h_words.astype(np.int64),
        A_words=base_c.A_words, ell=strided_ell, Gamma=strided_G, pivots=base_c.pivots,
    )
    ov_ref = pq.overlap(base_c, base_c, use_cpp_kernels=False)
    assert abs(pq.overlap(strided_c, base_c, use_cpp_kernels=True) - ov_ref) < 1e-14
    assert abs(pq.overlap(strided_c, base_c, use_cpp_kernels=False) - ov_ref) < 1e-14

    empty_l = np.zeros(0, dtype=np.uint8)
    empty_B = np.zeros((0, 0), dtype=np.uint8)
    for q0 in range(4):
        for scl in (-3, 0, 5):
            vc = pq.quadratic_gauss_sum(empty_l, empty_B, c0=q0, scale_log2=scl, use_cpp_kernels=True)
            vp = pq.quadratic_gauss_sum(empty_l, empty_B, c0=q0, scale_log2=scl, use_cpp_kernels=False)
            assert abs(vc - vp) < 1e-15


@pytest.mark.parametrize("n_qubits", [1, 5, 63, 64, 65, 70, 128, 129, 130])
def test_all_eight_kernels_word_boundaries_in_place_and_log_mass(n_qubits: int) -> None:
    nw = pq.n_words_for_qubits(n_qubits)
    z0 = np.zeros(nw, dtype=np.uint64)
    u_first = pq.unit_bit_words(0, nw)
    u_last = pq.unit_bit_words(n_qubits - 1, nw)
    u_mid = pq.unit_bit_words(n_qubits // 2, nw)

    cpp = pq.ShotExpansion.zero_state(n_qubits, use_cpp_kernels=True)
    py = pq.ShotExpansion.zero_state(n_qubits, use_cpp_kernels=False)
    cpp.log_mass = py.log_mass = -3.75

    cpp_cp = pq.apply_pauli(cpp, u_last, u_first, 1 if n_qubits == 1 else 0, in_place=False)
    py_cp = pq.apply_pauli(py, u_last, u_first, 1 if n_qubits == 1 else 0, in_place=False)
    assert cpp_cp is not cpp
    _assert_shot_parity(cpp_cp, py_cp)

    assert pq.apply_pauli(cpp, u_first, z0, 0, in_place=True) is cpp
    pq.apply_pauli(py, u_first, z0, 0, in_place=True)
    _assert_shot_parity(cpp, py)

    # P2 rank growth, dirty pivot via P1, then H4 rank reduction.
    cpp = pq.apply_quarter_turn(cpp, u_last, z0, 0)
    py = pq.apply_quarter_turn(py, u_last, z0, 0)
    _assert_shot_parity(cpp, py)

    pq.apply_pauli(cpp, u_last, u_first, 1 if n_qubits == 1 else 0, in_place=True)
    pq.apply_pauli(py, u_last, u_first, 1 if n_qubits == 1 else 0, in_place=True)
    cpp_inv = cpp.copy()
    py_inv = pq.apply_quarter_turn(py, u_last, z0, 0, sign=-1, in_place=False)
    assert pq.apply_quarter_turn(cpp_inv, u_last, z0, 0, sign=-1, in_place=True) is cpp_inv
    _assert_shot_parity(cpp_inv, py_inv)

    if n_qubits >= 2:
        kap2 = 1 if (n_qubits // 2) in (0, n_qubits - 1) else 0
        cpp = pq.apply_quarter_turn(cpp, u_first ^ u_last, u_mid, kap2)
        py = pq.apply_quarter_turn(py, u_first ^ u_last, u_mid, kap2)
        _assert_shot_parity(cpp, py)

    cpp = pq.apply_pauli_t(cpp, u_first, u_last, 1 if n_qubits == 1 else 0, dag=False)
    py = pq.apply_pauli_t(py, u_first, u_last, 1 if n_qubits == 1 else 0, dag=False)
    _assert_shot_parity(cpp, py)
    cpp = pq.apply_pauli_t(cpp, z0, u_last, 2, dag=True)
    py = pq.apply_pauli_t(py, z0, u_last, 2, dag=True)
    _assert_shot_parity(cpp, py)

    if n_qubits >= 3:
        q_set = sorted({0, max(1, n_qubits // 2), n_qubits - 1})
        if len(q_set) == 3:
            axes = [(z0, pq.unit_bit_words(q, nw), 0) for q in q_set]
            cpp = pq.apply_ccz(cpp, axes)
            py = pq.apply_ccz(py, axes)
            _assert_shot_parity(cpp, py)

    for b_probe in (u_last, u_mid):
        for sig in (0, 1):
            z_sum = 0.0
            for out_bit in (0, 1):
                p_c, z_c = pq.project_diagonal(cpp, b_probe, sig, out_bit, use_cpp_kernels=True)
                p_p, z_p = pq.project_diagonal(py, b_probe, sig, out_bit, use_cpp_kernels=False)
                assert abs(z_c - z_p) < 1e-13
                _assert_shot_parity(p_c, p_p)
                z_sum += z_c
            assert abs(z_sum - cpp.norm_sq()) < 1e-12

    c_zero = pq.PhaseQFEComponent.zero_state(n_qubits)
    comps = [c for _, c in cpp.to_components()[:3]] + [c_zero]
    for cb in comps:
        for ck in comps:
            ov_c = pq.overlap(cb, ck, use_cpp_kernels=True)
            ov_p = pq.overlap(cb, ck, use_cpp_kernels=False)
            assert abs(ov_c - ov_p) < 1e-13

    rng = np.random.default_rng(1000 + n_qubits)
    for g in (1, 8, 75 if n_qubits == 130 else 17):
        l_arr = rng.integers(0, 4, size=g, dtype=np.uint8)
        u_mat = np.triu(rng.integers(0, 2, size=(g, g), dtype=np.uint8), 1)
        b_mat = np.ascontiguousarray(u_mat ^ u_mat.T)
        gc = pq.quadratic_gauss_sum(l_arr, b_mat, c0=n_qubits & 3, scale_log2=-g, use_cpp_kernels=True)
        gp = pq.quadratic_gauss_sum(l_arr, b_mat, c0=n_qubits & 3, scale_log2=-g, use_cpp_kernels=False)
        assert abs(gc - gp) < 1e-14 * max(1.0, abs(gp))


def test_high_chi_k1_first_seen_order_and_exact_cancellation() -> None:
    rng = np.random.default_rng(2026)
    n_q = 8
    base = pq.ShotExpansion.zero_state(n_q, use_cpp_kernels=False)
    for q in (0, 3, 6):
        base = pq.apply_quarter_turn(base, pq.unit_bit_words(q, 1), np.zeros(1, dtype=np.uint64), 0)

    chi_raw = 512
    h_raw = rng.integers(0, 1 << n_q, size=(chi_raw, 1), dtype=np.uint64)
    even_bits = rng.integers(0, 2, size=(chi_raw, base.d), dtype=np.uint8) << np.uint8(1)
    ell_raw = np.ascontiguousarray((base.ell_odd[None, :] ^ even_bits) & np.uint8(3))
    w_raw = rng.normal(size=chi_raw) + 1j * rng.normal(size=chi_raw)

    unmerged_cpp = pq.ShotExpansion(
        n_qubits=n_q, d=base.d, A_words=base.A_words, a_union=base.a_union,
        Gamma=base.Gamma, pivots=base.pivots, weights=w_raw, h_words=h_raw,
        ell=ell_raw, ell_odd=base.ell_odd, use_cpp_kernels=True, _is_canonical_merged=False,
    )
    unmerged_py = unmerged_cpp.copy()
    unmerged_py.use_cpp_kernels = False

    m_cpp = pq.merge_frames(unmerged_cpp, use_cpp_kernels=True)
    m_py = pq.merge_frames(unmerged_py, use_cpp_kernels=False)
    _assert_shot_parity(m_cpp, m_py)

    cancel_exp = pq.ShotExpansion(
        n_qubits=n_q, d=base.d, A_words=base.A_words, a_union=base.a_union,
        Gamma=base.Gamma, pivots=base.pivots,
        weights=np.concatenate([m_cpp.weights, -m_cpp.weights]),
        h_words=np.vstack([m_cpp.h_words, m_cpp.h_words]),
        ell=np.vstack([m_cpp.ell, m_cpp.ell]),
        ell_odd=base.ell_odd, use_cpp_kernels=True, _is_canonical_merged=False,
    )
    empty_cpp = pq.merge_frames(cancel_exp, use_cpp_kernels=True)
    assert empty_cpp.chi == 0 and empty_cpp.norm_sq() == 0.0
    u0 = pq.unit_bit_words(0, 1)
    assert pq.apply_pauli(empty_cpp, u0, u0, 1).chi == 0
    assert pq.apply_quarter_turn(empty_cpp, u0, np.zeros(1, dtype=np.uint64), 0).chi == 0
    assert pq.apply_pauli_t(empty_cpp, u0, np.zeros(1, dtype=np.uint64), 0).chi == 0


@pytest.mark.parametrize("n_qubits", [5, 70])
def test_measure_expansion_rng_lockstep_cpp_vs_python(n_qubits: int) -> None:
    nw = pq.n_words_for_qubits(n_qubits)
    z0 = np.zeros(nw, dtype=np.uint64)
    qs = [0, 1, n_qubits // 2, n_qubits - 2, n_qubits - 1]
    exp_c = pq.ShotExpansion.zero_state(n_qubits, use_cpp_kernels=True)
    exp_p = pq.ShotExpansion.zero_state(n_qubits, use_cpp_kernels=False)
    for q in qs[:3]:
        uq = pq.unit_bit_words(q, nw)
        exp_c = pq.apply_quarter_turn(exp_c, uq, z0, 0)
        exp_p = pq.apply_quarter_turn(exp_p, uq, z0, 0)
        exp_c = pq.apply_pauli_t(exp_c, z0, uq, 0)
        exp_p = pq.apply_pauli_t(exp_p, z0, uq, 0)

    rng_c = np.random.default_rng(777 + n_qubits)
    rng_p = np.random.default_rng(777 + n_qubits)
    for idx, q in enumerate(qs[:3]):
        uq = pq.unit_bit_words(q, nw)
        a_m, b_m = (uq, z0) if (idx % 2 == 0) else (z0, uq)
        rc = pq.measure_expansion(exp_c, a_m, b_m, 0, rng=rng_c, p_flip=0.2, invert_result=(idx == 1))
        rp = pq.measure_expansion(exp_p, a_m, b_m, 0, rng=rng_p, p_flip=0.2, invert_result=(idx == 1))
        assert (rc.outcome, rc.reported_outcome, rc.readout_flipped) == (
            rp.outcome, rp.reported_outcome, rp.readout_flipped
        )
        assert abs(rc.z0 - rp.z0) < 1e-13 and abs(rc.z1 - rp.z1) < 1e-13
        assert abs(rc.log_branch_mass - rp.log_branch_mass) < 1e-13
        exp_c, exp_p = rc.post_state, rp.post_state  # type: ignore[assignment]
        _assert_shot_parity(exp_c, exp_p, atol=1e-12)
    assert rng_c.random() == rng_p.random()


@pytest.mark.parametrize("n_qubits", [4, 70])
def test_magic_compile_direct_and_selector_cpp_vs_python_parity(n_qubits: int) -> None:
    q0, q1, q2 = 0, n_qubits // 2, n_qubits - 1
    circ = stim.Circuit(f"""
        H {q0} {q1} {q2}
        I[T] {q0} {q2}
        X_ERROR(0.3) {q0} {q2}
        CX {q0} {q1}
        I[CCZ] {q0} {q1} {q2}
        Z_ERROR(0.25) {q1}
        I[T_DAG] {q1} {q2}
        H {q0} {q1} {q2}
        M {q0} {q1} {q2}
    """)
    cd = compile_direct(circ)
    cs = compile_selector(circ, gadgetize_pauli_noise=True)
    noise_trace = {0: 1, 1: 1}

    rd_c = run_direct(cd, pauli_noise_trace=noise_trace, rng=np.random.default_rng(42), use_cpp_kernels=True)
    rd_p = run_direct(cd, pauli_noise_trace=noise_trace, rng=np.random.default_rng(42), use_cpp_kernels=False)
    assert np.array_equal(rd_c.measurements, rd_p.measurements)
    assert abs(rd_c.log_branch_mass - rd_p.log_branch_mass) < 1e-12
    _assert_shot_parity(rd_c.expansion, rd_p.expansion, atol=1e-12)

    rs_c = run_selector(cs, pauli_noise_trace=noise_trace, rng=np.random.default_rng(42), use_cpp_kernels=True)
    rs_p = run_selector(cs, pauli_noise_trace=noise_trace, rng=np.random.default_rng(42), use_cpp_kernels=False)
    assert np.array_equal(rs_c.measurements, rs_p.measurements)
    assert abs(rs_c.log_branch_mass - rs_p.log_branch_mass) < 1e-12
    _assert_shot_parity(rs_c.expansion, rs_p.expansion, atol=1e-12)

    if n_qubits == 4:
        circ_open = stim.Circuit("H 0 1 2\nI[T] 0\nX_ERROR(0.2) 0\nI[CCZ] 0 1 2\nI[T_DAG] 2")
        cs_o = compile_selector(circ_open, gadgetize_pauli_noise=True)
        so_c = run_selector(cs_o, pauli_noise_trace={0: 1}, use_cpp_kernels=True, extract_data=True)
        so_p = run_selector(cs_o, pauli_noise_trace={0: 1}, use_cpp_kernels=False, extract_data=True)
        psi_c = so_c.expansion.to_statevector(ref_tableau=so_c.final_tableau)
        psi_p = so_p.expansion.to_statevector(ref_tableau=so_p.final_tableau)
        assert np.max(np.abs(psi_c - psi_p)) < 1e-12


@pytest.mark.parametrize(
    "case",
    [
        "chi0_d0_self",
        "chi0_d0_comp",
        "chi0_d1",
        "chi2_d0_bell",
        "chi2_orth",
        "nq_5_70",
        "nq_70_5",
        "empty_A",
        "bad_Gamma",
        "h_2d",
    ],
)
def test_overlap_rejects_expansions_qubit_mismatch_and_malformed_shapes(case: str) -> None:
    def _c1(nq: int, h0: int = 0) -> pq.PhaseQFEComponent:
        nw = pq.n_words_for_qubits(nq)
        h, a = np.zeros(nw, dtype=np.uint64), pq.unit_bit_words(0, nw)[None, :]
        h[0] = np.uint64(h0)
        return pq.PhaseQFEComponent(
            nq, 1, h, a, np.zeros(1, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.zeros(1, dtype=np.int32)
        )

    c2, c2_x1 = _c1(2, 0), _c1(2, 2)
    z00, z11 = pq.PhaseQFEComponent.zero_state(2), pq.PhaseQFEComponent.computational_basis_state(2, [1, 1])
    e0, s = pq.ShotExpansion.from_components([], n_qubits=2), 0.5**0.5
    bra, ket = {
        "chi0_d0_self": (e0, e0),
        "chi0_d0_comp": (e0, z00),
        "chi0_d1": (e0, c2),
        "chi2_d0_bell": (
            pq.ShotExpansion.from_components([(s, z00), (s, z11)]),
            pq.ShotExpansion.from_components([(s, z00), (-s, z11)]),
        ),
        "chi2_orth": (
            pq.ShotExpansion.from_components([(s, c2), (s, c2_x1)]),
            pq.ShotExpansion.from_components([(s, c2), (-s, c2_x1)]),
        ),
        "nq_5_70": (_c1(5), _c1(70)),
        "nq_70_5": (_c1(70), _c1(5)),
        "empty_A": (replace(c2, A_words=np.zeros((0, 1), dtype=np.uint64)), c2),
        "bad_Gamma": (replace(c2, Gamma=np.zeros((2, 2), dtype=np.uint8)), c2_x1),
        "h_2d": (replace(c2, h_words=np.zeros((1, 1), dtype=np.uint64)), c2),
    }[case]
    exc = TypeError if case.startswith("chi") else ValueError
    for b, k in ((bra, ket), (ket, bra)):
        with pytest.raises(exc):
            mk.cpp_overlap(b, k)
        with pytest.raises(exc):
            pq.overlap(b, k, use_cpp_kernels=True)
        with pytest.raises(exc):
            pq.overlap(b, k, use_cpp_kernels=False)
