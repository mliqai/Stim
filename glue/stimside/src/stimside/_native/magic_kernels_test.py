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
    "magic_abi_v2",
    "magic_apply_pauli_batch",
    "magic_apply_quarter_turn",
    "magic_canonicalize_and_merge",
    "magic_branch_pauli_t",
    "magic_branch_ccz",
    "magic_project_diagonal",
    "magic_contract_choi",
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
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("unexpected compile")):
            lib1 = mk.get_magic_kernels_lib(force_recompile=False)
            lib2 = mk.get_magic_kernels_lib(force_recompile=False)
        assert lib1 is lib2
        for sym in EXPECTED_SYMBOLS:
            assert hasattr(lib1, sym), f"Missing symbol {sym}"
        assert int(lib1.magic_abi_v1()) == 1
        assert int(lib1.magic_abi_v2()) == 2

        with pytest.raises(TypeError, match="use_cpp_kernels"):
            pq.ShotExpansion.zero(2, use_cpp_kernels=False)  # type: ignore[call-arg]
        exp = pq.ShotExpansion.zero(2)
        pq.apply_pauli(exp, [1, 0], [0, 0], 0)
        assert mk._LIB_INSTANCE is lib1

        tmp_native = tmp_path / "rebuild_pkg" / "src" / "stimside" / "_native"
        tmp_native.mkdir(parents=True)
        (tmp_path / "rebuild_pkg" / "setup.py").write_text("# dummy\n", encoding="utf-8")
        abi_syms = ("magic_abi_v1", "magic_abi_v2")
        stub_cpp = (
            'extern "C" {\nint magic_abi_v1() { return 1; }\nint magic_abi_v2() { return 2; }\n'
            + "".join(
                f"int {s}() {{ return 0; }}\n"
                for s in EXPECTED_SYMBOLS
                if s not in abi_syms
            )
            + "}\n"
        )
        (tmp_native / "magic_kernels.cpp").write_text(stub_cpp, encoding="utf-8")
        with monkeypatch.context() as mp:
            import tempfile

            mp.setattr(tempfile, "tempdir", str(tmp_path))
            mp.setattr(mk, "__file__", str(tmp_native / "magic_kernels.py"))
            rebuilt = mk.get_magic_kernels_lib(force_recompile=True)
            assert (tmp_native / "libmagic_kernels.so").is_file()
            assert rebuilt is not lib1 and rebuilt._handle != lib1._handle
            assert mk._LIB_INSTANCE is rebuilt
            assert int(rebuilt.magic_abi_v1()) == 1
            assert int(rebuilt.magic_abi_v2()) == 2
    finally:
        mk._LIB_INSTANCE = lib1 if lib1 is not None else orig_lib

    assert (so_path.stat().st_ino, so_path.stat().st_mtime_ns) == live_stat

    with monkeypatch.context() as mp:
        empty_dir = tmp_path / "missing_src"
        empty_dir.mkdir(parents=True)
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


def test_magic_contract_choi_word_boundary_and_edge_cases() -> None:
    from stimside._magic.query import QueryBudget, evaluate_queries, prepare_queries

    e0 = pq.ShotExpansion.zero(2)
    e0_choi = pq.ShotExpansion.zero(3)
    empty = pq.ShotExpansion.from_components((), n_qubits=2)
    empty_choi = pq.ShotExpansion.from_components((), n_qubits=3)
    assert pq.contract_choi_expansions(empty_choi, e0, 2, (0,), False).chi == 0
    assert pq.contract_choi_expansions(e0_choi, empty, 2, (0,), False).chi == 0
    m0 = pq.contract_choi_expansions(e0, e0, 2, (), False)
    assert m0.chi == 1 and abs(m0.norm_sq() - 1.0) < 1e-15

    bad = e0_choi.copy()
    bad.weights = np.array([np.nan + 0.0j], dtype=np.complex128)
    with pytest.raises(FloatingPointError):
        pq.contract_choi_expansions(bad, e0, 2, (0,), False)

    def _query_circ(q0: int, q1: int, q2: int) -> stim.Circuit:
        return stim.Circuit(f"""
            H {q0} {q1} {q2}
            I[T] {q0}
            X_ERROR(0.2) {q0}
            CX {q0} {q1}
            I[CCZ] {q0} {q1} {q2}
            Z_ERROR(0.3) {q2}
            I[T_DAG] {q1} {q2}
            H {q0} {q1} {q2}
            M {q0} {q1} {q2}
        """)

    c4, c70 = _query_circ(0, 1, 2), _query_circ(0, 62, 69)
    p4 = prepare_queries(c4)
    p70 = prepare_queries(c70)
    p70_stream = prepare_queries(c70, budget=QueryBudget(max_terms=1, on_budget_exceeded="fallback"))
    for meas_bits in ({0: 0, 1: 1, 2: 0}, {0: 1, 1: 1, 2: 1}):
        real = {"paulis": {0: 1, 1: 0}, "quantum_measurements": meas_bits}
        r4 = evaluate_queries(p4, real, queries=("mass", "prefix_probs"))
        r70 = evaluate_queries(p70, real, queries=("mass", "prefix_probs"))
        r70_s = evaluate_queries(p70_stream, real, queries=("mass", "prefix_probs"))
        assert not r70.diagnostics.used_fallback
        assert r70_s.diagnostics.used_fallback
        assert abs(r4.mass - r70.mass) < 1e-13
        assert abs(r70.mass - r70_s.mass) < 1e-13
        assert np.allclose(r4.prefix_probs, r70.prefix_probs, atol=1e-13)


def test_magic_contract_choi_preserves_idle_ket_qubits_and_validates_q_act() -> None:
    # 1. Idle ket qubits outside q_act must be preserved (including entanglement across 64-bit word boundaries).
    e0_2 = pq.ShotExpansion.zero(2)
    z0_1 = np.zeros(1, dtype=np.uint64)
    u1_1 = pq.unit_bit_words(1, 1)

    ket_0_plus = pq.apply_quarter_turn(pq.ShotExpansion.zero(2), u1_1, z0_1, 0)
    out_empty_q = pq.contract_choi_expansions(e0_2, ket_0_plus, 2, (), False)
    out_empty_q.check_invariants(require_canonical_offset=True)
    np.testing.assert_allclose(out_empty_q.to_statevector(), ket_0_plus.to_statevector(), atol=1e-13)

    for n_op, q_idle in ((2, 1), (64, 63), (70, 69)):
        nw_ket = pq.n_words_for_qubits(n_op)
        nw_choi = pq.n_words_for_qubits(n_op + 1)
        z_ket, z_choi = np.zeros(nw_ket, dtype=np.uint64), np.zeros(nw_choi, dtype=np.uint64)
        # Bell pair (|00> + |11>) / sqrt(2) on (0, q_idle) via sqrt(+i X_0) then CX_{0 -> q_idle} on |0...0>:
        # (|00> + i|11>) / sqrt(2) created directly by P2(X_0 X_{q_idle}):
        bell_ket = pq.apply_quarter_turn(
            pq.ShotExpansion.zero(n_op),
            pq.unit_bit_words(0, nw_ket) ^ pq.unit_bit_words(q_idle, nw_ket),
            z_ket,
            0,
        )
        # Choi state of T_0 on q_act=(0,) has Bell pair (|00> + |11>)/sqrt(2) on (0, n_op), followed by T_0.
        A_bell = pq.unit_bit_words(0, nw_choi)[None, :] ^ pq.unit_bit_words(n_op, nw_choi)[None, :]
        choi_t0 = pq.ShotExpansion(
            n_op + 1, 1, A_bell, A_bell[0].copy(), np.zeros((1, 1), dtype=np.uint8),
            np.array([0], dtype=np.int32), np.array([1.0 + 0.0j], dtype=np.complex128),
            np.zeros((1, nw_choi), dtype=np.uint64), np.zeros((1, 1), dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
        )
        choi_t0 = pq.apply_pauli_t(choi_t0, z_choi, pq.unit_bit_words(0, nw_choi), 0, dag=False)

        got = pq.contract_choi_expansions(choi_t0, bell_ket, n_op, (0,), False)
        got.check_invariants(require_canonical_offset=True)
        want = pq.apply_pauli_t(bell_ket.copy(), z_ket, pq.unit_bit_words(0, nw_ket), 0, dag=False)
        np.testing.assert_allclose(
            _extract_active_statevector(got, [0, q_idle]),
            _extract_active_statevector(want, [0, q_idle]),
            atol=1e-13,
        )

    # 1b. Choi-Choi composition (right_is_choi=True) across multi-word non-contiguous q_act, plus n_qubits mismatch rejection.
    n_cc, q_cc = 70, (3, 62, 69)
    m_cc, nw_cc = len(q_cc), pq.n_words_for_qubits(n_cc + len(q_cc))
    z_cc = np.zeros(nw_cc, dtype=np.uint64)
    u3, u62, u69 = (pq.unit_bit_words(q, nw_cc) for q in q_cc)
    A_cc = np.vstack([pq.unit_bit_words(q_cc[r], nw_cc) ^ pq.unit_bit_words(n_cc + r, nw_cc) for r in range(m_cc)])
    bell_cc = pq.ShotExpansion(
        n_cc + m_cc, m_cc, A_cc, np.bitwise_or.reduce(A_cc, axis=0),
        np.zeros((m_cc, m_cc), dtype=np.uint8), np.asarray(q_cc, dtype=np.int32),
        np.array([1.0 + 0.0j], dtype=np.complex128), np.zeros((1, nw_cc), dtype=np.uint64),
        np.zeros((1, m_cc), dtype=np.uint8), np.zeros(m_cc, dtype=np.uint8),
    )
    choi_r = pq.apply_quarter_turn(
        pq.apply_ccz(bell_cc.copy(), [(z_cc, u3, 0), (z_cc, u62, 0), (z_cc, u69, 0)]), u62, z_cc, 0
    )
    choi_r_pad = pq.ShotExpansion(
        135, choi_r.d, np.pad(choi_r.A_words, ((0, 0), (0, 1))), np.pad(choi_r.a_union, (0, 1)),
        choi_r.Gamma.copy(), choi_r.pivots.copy(), choi_r.weights.copy(),
        np.pad(choi_r.h_words, ((0, 0), (0, 1))), choi_r.ell.copy(), choi_r.ell_odd.copy(),
    )
    choi_l = pq.apply_pauli_t(pq.apply_quarter_turn(bell_cc.copy(), u3, z_cc, 0), z_cc, u69, 0, dag=False)
    with pytest.raises(ValueError, match="n_qubits mismatch"):
        pq.contract_choi_expansions(choi_l, choi_r_pad, n_cc, q_cc, True)
    got_cc = pq.contract_choi_expansions(choi_l, choi_r, n_cc, q_cc, True)
    got_cc.check_invariants(require_canonical_offset=True)
    want_cc = pq.apply_pauli_t(pq.apply_quarter_turn(choi_r.copy(), u3, z_cc, 0), z_cc, u69, 0, dag=False)
    want_cc.weights = want_cc.weights * (2.0 ** -(m_cc >> 1))
    active_cc = [*q_cc, n_cc, n_cc + 1, n_cc + 2]
    np.testing.assert_allclose(
        _extract_active_statevector(got_cc, active_cc),
        _extract_active_statevector(want_cc, active_cc),
        atol=1e-13,
    )

    # 2. Invalid q_act and invalid Choi pivots must raise ValueError cleanly (even on empty expansions).
    empty_2 = pq.ShotExpansion.from_components((), n_qubits=2)
    for bad_q in ((-1,), (2,), (0, 0), (0, 1, 2), (True,)):
        with pytest.raises(ValueError):
            pq.contract_choi_expansions(e0_2, e0_2, 2, bad_q, False)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            pq.contract_choi_expansions(empty_2, e0_2, 2, bad_q, False)  # type: ignore[arg-type]

    # Choi left with a generator pivot on idle qubit 1 (not in q_act=(0,)) must be rejected.
    bad_choi = pq.apply_quarter_turn(pq.ShotExpansion.zero(3), pq.unit_bit_words(1, 1), np.zeros(1, dtype=np.uint64), 0)
    with pytest.raises(ValueError):
        pq.contract_choi_expansions(bad_choi, e0_2, 2, (0,), False)

    # Raw C ABI boundary also rejects out-of-range q_act with status 1 instead of out-of-bounds memory access.
    lib = mk.get_magic_kernels_lib()
    null_d = ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    null_u64 = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint64))
    null_u8 = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
    null_i32 = ctypes.cast(0, ctypes.POINTER(ctypes.c_int32))
    out_chi, out_d = ctypes.c_size_t(99), ctypes.c_size_t(99)
    bad_q_c = (ctypes.c_int32 * 1)(-1)
    rc = int(
        lib.magic_contract_choi(
            0, 0, 1, null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
            0, 0, 1, null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
            2, bad_q_c, 1, 0, 1e-13,
            ctypes.byref(out_chi), ctypes.byref(out_d),
            null_d, null_u64, null_u8, null_u64, null_u8, null_i32, null_u8,
        )
    )
    assert rc == 1 and out_chi.value == 0 and out_d.value == 0

