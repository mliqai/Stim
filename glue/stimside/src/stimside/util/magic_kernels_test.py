"""Unit and C-ABI parity tests for `stimside._native.magic_kernels` (`libmagic_kernels.so`)."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from unittest import mock

import numpy as np
import pytest
import stim

from stimside._magic import phase_qfe as pq
from stimside._magic.compile import (
    compile_direct,
    compile_selector,
    run_direct,
    run_selector,
)
from stimside._native import magic_kernels as mk

EXPECTED_SYMBOLS = (
    "magic_abi_v1",
    "magic_apply_pauli_batch",
    "magic_apply_quarter_turn",
    "magic_canonicalize_and_merge",
    "magic_branch_pauli_t",
    "magic_branch_ccz",
    "magic_project_diagonal",
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

        with pytest.raises(TypeError, match="use_cpp_kernels"):
            pq.ShotExpansion.zero(2, use_cpp_kernels=False)  # type: ignore[call-arg]
        exp = pq.ShotExpansion.zero(2)
        pq.apply_pauli(exp, [1, 0], [0, 0], 0)
        assert pq._CACHED_MAGIC_LIB is not None and pq._CACHED_MAGIC_LIB is lib1

        tmp_native = tmp_path / "rebuild_pkg" / "src" / "stimside" / "_native"
        tmp_native.mkdir(parents=True)
        (tmp_path / "rebuild_pkg" / "setup.py").write_text("# dummy\n", encoding="utf-8")
        stub_cpp = (
            'extern "C" {\nint magic_abi_v1() { return 1; }\n'
            + "".join(f"void {s}() {{}}\n" for s in EXPECTED_SYMBOLS if s != "magic_abi_v1")
            + "}\n"
        )
        (tmp_native / "magic_kernels.cpp").write_text(stub_cpp, encoding="utf-8")
        with monkeypatch.context() as mp:
            mp.setattr(mk, "__file__", str(tmp_native / "magic_kernels.py"))
            mp.setattr(mk, "_find_prebuilt_so_candidate", lambda *a, **kw: None)
            rebuilt = mk.get_magic_kernels_lib(force_recompile=True)
            assert (tmp_native / "libmagic_kernels.so").is_file()
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
        pq.ShotExpansion.zero(0)

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


def _dense_pauli_apply(psi: np.ndarray, a_mask: int, b_mask: int, kappa: int) -> np.ndarray:
    idx = np.arange(psi.shape[0], dtype=np.uint64)
    z_par = np.bitwise_count(idx & np.uint64(b_mask)) & 1
    phases = pq._PHASE4[(int(kappa) + (z_par << 1)) & 3]
    out = np.empty_like(psi)
    out[idx ^ np.uint64(a_mask)] = phases * psi
    return out


def _extract_active_statevector(exp: pq.ShotExpansion, qs: list[int]) -> np.ndarray:
    k = len(qs)
    if exp.chi == 0:
        return np.zeros(1 << k, dtype=np.complex128)
    qs_arr = np.asarray(qs, dtype=np.int32)
    loc = {q: j for j, q in enumerate(qs)}
    a_bits = pq.get_bits_at_qubits(exp.A_words, qs_arr) if exp.d > 0 else np.zeros((0, k), dtype=np.uint8)
    h_bits = pq.get_bits_at_qubits(exp.h_words, qs_arr)
    shifts = (np.uint64(1) << np.arange(k, dtype=np.uint64))[None, :]
    a_sub = np.sum(a_bits.astype(np.uint64) * shifts, axis=1, keepdims=True, dtype=np.uint64)
    h_sub = np.sum(h_bits.astype(np.uint64) * shifts, axis=1, keepdims=True, dtype=np.uint64)
    piv_sub = np.array([loc[int(p)] for p in exp.pivots], dtype=np.int32)
    a_u = np.array([np.bitwise_or.reduce(a_sub[:, 0])] if exp.d > 0 else [0], dtype=np.uint64)
    sub = pq.ShotExpansion(
        k, exp.d, a_sub, a_u, exp.Gamma, piv_sub,
        exp.weights, h_sub, exp.ell, exp.ell_odd,
        exp.log_mass, exp.atol, exp._is_canonical_merged,
    )
    return sub.to_statevector()


@pytest.mark.parametrize("n_qubits", [1, 5, 63, 64, 65, 70, 128, 129, 130])
def test_all_eight_kernels_word_boundaries_in_place_and_log_mass(n_qubits: int) -> None:
    nw = pq.n_words_for_qubits(n_qubits)
    z0 = np.zeros(nw, dtype=np.uint64)
    qs = sorted({0, n_qubits // 2, n_qubits - 1})
    loc = {q: j for j, q in enumerate(qs)}
    q_f, q_m, q_l = 0, n_qubits // 2, n_qubits - 1
    u_first = pq.unit_bit_words(q_f, nw)
    u_last = pq.unit_bit_words(q_l, nw)
    u_mid = pq.unit_bit_words(q_m, nw)
    b_f, b_m, b_l = 1 << loc[q_f], 1 << loc[q_m], 1 << loc[q_l]

    cpp = pq.ShotExpansion.zero(n_qubits)
    cpp.log_mass = -3.75
    psi = np.zeros(1 << len(qs), dtype=np.complex128)
    psi[0] = 1.0

    kap1 = 1 if n_qubits == 1 else 0
    cpp_cp = pq.apply_pauli(cpp, u_last, u_first, kap1, in_place=False)
    assert cpp_cp is not cpp and cpp_cp.log_mass == -3.75
    np.testing.assert_allclose(_extract_active_statevector(cpp_cp, qs), _dense_pauli_apply(psi, b_l, b_f, kap1), atol=1e-13)

    assert pq.apply_pauli(cpp, u_first, z0, 0, in_place=True) is cpp
    psi = _dense_pauli_apply(psi, b_f, 0, 0)
    np.testing.assert_allclose(_extract_active_statevector(cpp, qs), psi, atol=1e-13)

    # P2 rank growth, dirty pivot via P1, then H4 rank reduction.
    cpp = pq.apply_quarter_turn(cpp, u_last, z0, 0)
    psi = (psi + 1j * _dense_pauli_apply(psi, b_l, 0, 0)) / np.sqrt(2.0)
    assert cpp.log_mass == -3.75
    np.testing.assert_allclose(_extract_active_statevector(cpp, qs), psi, atol=1e-13)

    pq.apply_pauli(cpp, u_last, u_first, kap1, in_place=True)
    psi = _dense_pauli_apply(psi, b_l, b_f, kap1)
    cpp_inv = cpp.copy()
    assert pq.apply_quarter_turn(cpp_inv, u_last, z0, 0, sign=-1, in_place=True) is cpp_inv
    psi_inv = (psi - 1j * _dense_pauli_apply(psi, b_l, 0, 0)) / np.sqrt(2.0)
    cpp_inv.check_invariants(require_canonical_offset=True)
    np.testing.assert_allclose(_extract_active_statevector(cpp_inv, qs), psi_inv, atol=1e-13)

    if n_qubits >= 2:
        kap2 = 1 if q_m in (q_f, q_l) else 0
        cpp = pq.apply_quarter_turn(cpp, u_first ^ u_last, u_mid, kap2)
        psi = (psi + 1j * _dense_pauli_apply(psi, b_f ^ b_l, b_m, kap2)) / np.sqrt(2.0)

    cpp = pq.apply_pauli_t(cpp, u_first, u_last, kap1, dag=False)
    psi = pq._T_C0 * psi + pq._T_C1 * _dense_pauli_apply(psi, b_f, b_l, kap1)
    cpp = pq.apply_pauli_t(cpp, z0, u_last, 2, dag=True)
    psi = pq._TDAG_C0 * psi + pq._TDAG_C1 * _dense_pauli_apply(psi, 0, b_l, 2)

    if len(qs) == 3:
        cpp = pq.apply_ccz(cpp, [(z0, pq.unit_bit_words(q, nw), 0) for q in qs])
        psi = psi.copy()
        psi[7] *= -1.0
    cpp.check_invariants(require_canonical_offset=True)
    assert cpp.log_mass == -3.75
    np.testing.assert_allclose(_extract_active_statevector(cpp, qs), psi, atol=1e-12)

    idx_k = np.arange(psi.shape[0], dtype=np.uint64)
    for b_probe, b_loc in ((u_last, b_l), (u_mid, b_m)):
        for sig in (0, 1):
            z_sum = 0.0
            for out_bit in (0, 1):
                p_c, z_c = pq.project_diagonal(cpp, b_probe, sig, out_bit)
                p_c.check_invariants(require_canonical_offset=True)
                keep = ((np.bitwise_count(idx_k & np.uint64(b_loc)) ^ sig) & 1) == out_bit
                psi_proj = np.where(keep, psi, 0.0j)
                z_ref = float(np.vdot(psi_proj, psi_proj).real)
                assert abs(z_c - z_ref) < 1e-13
                np.testing.assert_allclose(_extract_active_statevector(p_c, qs), psi_proj, atol=1e-12)
                z_sum += z_c
            assert abs(z_sum - cpp.norm_sq()) < 1e-12


def test_high_chi_k1_first_seen_order_and_exact_cancellation() -> None:
    rng = np.random.default_rng(2026)
    n_q = 8
    base = pq.ShotExpansion.zero(n_q)
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
        ell=ell_raw, ell_odd=base.ell_odd, _is_canonical_merged=False,
    )
    psi_raw = sum(w * c.to_statevector() for w, c in unmerged_cpp.to_components())

    m_cpp = pq.merge_frames(unmerged_cpp)
    assert isinstance(m_cpp, pq.ShotExpansion)
    m_cpp.check_invariants(require_canonical_offset=True)
    np.testing.assert_allclose(m_cpp.to_statevector(), psi_raw, atol=1e-12)
    m_idem = pq.merge_frames(m_cpp)
    assert isinstance(m_idem, pq.ShotExpansion)
    _assert_shot_parity(m_cpp, m_idem)

    cancel_exp = pq.ShotExpansion(
        n_qubits=n_q, d=base.d, A_words=base.A_words, a_union=base.a_union,
        Gamma=base.Gamma, pivots=base.pivots,
        weights=np.concatenate([m_cpp.weights, -m_cpp.weights]),
        h_words=np.vstack([m_cpp.h_words, m_cpp.h_words]),
        ell=np.vstack([m_cpp.ell, m_cpp.ell]),
        ell_odd=base.ell_odd, _is_canonical_merged=False,
    )
    empty_cpp = pq.merge_frames(cancel_exp)
    assert isinstance(empty_cpp, pq.ShotExpansion)
    assert empty_cpp.chi == 0 and empty_cpp.norm_sq() == 0.0
    u0 = pq.unit_bit_words(0, 1)
    assert pq.apply_pauli(empty_cpp, u0, u0, 1).chi == 0
    assert pq.apply_quarter_turn(empty_cpp, u0, np.zeros(1, dtype=np.uint64), 0).chi == 0
    assert pq.apply_pauli_t(empty_cpp, u0, np.zeros(1, dtype=np.uint64), 0).chi == 0


def test_measure_expansion_rng_lockstep_and_embedding_invariance() -> None:
    def _run_meas_seq(n_qubits: int, qs: list[int], seed: int) -> tuple[list[pq.MeasurementResult], float]:
        nw = pq.n_words_for_qubits(n_qubits)
        z0 = np.zeros(nw, dtype=np.uint64)
        exp = pq.ShotExpansion.zero(n_qubits)
        for q in qs[:3]:
            uq = pq.unit_bit_words(q, nw)
            exp = pq.apply_quarter_turn(exp, uq, z0, 0)
            exp = pq.apply_pauli_t(exp, z0, uq, 0)
        rng = np.random.default_rng(seed)
        out: list[pq.MeasurementResult] = []
        for idx, q in enumerate(qs[:3]):
            uq = pq.unit_bit_words(q, nw)
            a_m, b_m = (uq, z0) if (idx % 2 == 0) else (z0, uq)
            r = pq.measure_expansion(exp, a_m, b_m, 0, rng=rng, p_flip=0.2, invert_result=(idx == 1))
            out.append(r)
            assert isinstance(r.post_state, pq.ShotExpansion)
            exp = r.post_state
        return out, float(rng.random())

    res5, tail5 = _run_meas_seq(5, [0, 1, 2, 3, 4], 777)
    res70, tail70 = _run_meas_seq(70, [0, 1, 35, 68, 69], 777)
    assert tail5 == tail70
    for r5, r70 in zip(res5, res70):
        assert (r5.outcome, r5.reported_outcome, r5.readout_flipped) == (
            r70.outcome, r70.reported_outcome, r70.readout_flipped
        )
        assert abs(r5.z0 - r70.z0) < 1e-13 and abs(r5.z1 - r70.z1) < 1e-13
        assert abs(r5.log_branch_mass - r70.log_branch_mass) < 1e-13


def test_magic_compile_direct_vs_selector_and_word_boundary_parity() -> None:
    def _circ(q0: int, q1: int, q2: int) -> stim.Circuit:
        return stim.Circuit(f"""
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

    noise_trace = {0: 1, 1: 1}
    c4, c70 = _circ(0, 2, 3), _circ(0, 35, 69)
    rd4 = run_direct(compile_direct(c4), pauli_noise_trace=noise_trace, rng=np.random.default_rng(42))
    rd70 = run_direct(compile_direct(c70), pauli_noise_trace=noise_trace, rng=np.random.default_rng(42))
    rs4 = run_selector(compile_selector(c4, gadgetize_pauli_noise=True), pauli_noise_trace=noise_trace, rng=np.random.default_rng(42))
    rs70 = run_selector(compile_selector(c70, gadgetize_pauli_noise=True), pauli_noise_trace=noise_trace, rng=np.random.default_rng(42))
    for r_other in (rd70, rs4, rs70):
        assert np.array_equal(rd4.measurements, r_other.measurements)
        assert abs(rd4.log_branch_mass - r_other.log_branch_mass) < 1e-12

    circ_open = stim.Circuit("H 0 1 2\nI[T] 0\nX_ERROR(0.2) 0\nI[CCZ] 0 1 2\nI[T_DAG] 2")
    so_d = run_direct(compile_direct(circ_open), pauli_noise_trace={0: 1})
    so_s = run_selector(compile_selector(circ_open, gadgetize_pauli_noise=True), pauli_noise_trace={0: 1}, extract_data=True)
    psi_d = so_d.expansion.to_statevector(ref_tableau=so_d.final_tableau)
    psi_s = so_s.expansion.to_statevector(ref_tableau=so_s.final_tableau)
    assert abs(abs(np.vdot(psi_d, psi_s)) - 1.0) < 1e-12
