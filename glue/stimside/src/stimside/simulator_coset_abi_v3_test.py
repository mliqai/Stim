"""Exact differential and C-1/C-2/C-12 regression tests for Coset ABI v3 (Stage A)."""

from __future__ import annotations

import ctypes
import sys
from typing import Any, Literal, cast

import numpy as np
import pytest
import stim

from stimside._coset.mrqf_state import MRQFState
from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside._native import coset_kernels
from stimside._native._kernel_loader import i8_p, i32_p, i64_p, u8_p, u64_p
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator


def _mk_pair(
    text: str, batch: int, seed: int = 123
) -> tuple[CosetsideSimulator, CosetsideSimulator]:
    circ = stim.Circuit(text)
    h1 = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch)
    h2 = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch)
    s_cpp = CosetsideSimulator(circ, compiled_op_handler=h1, batch_size=batch, seed=seed)
    s_py = CosetsideSimulator(circ, compiled_op_handler=h2, batch_size=batch, seed=seed)
    s_py._use_cpp = False
    assert s_cpp._use_cpp and s_cpp._cpp_lib is not None and s_cpp._cpp_eng is not None
    return s_cpp, s_py


def _assert_exact_parity(
    s_cpp: CosetsideSimulator, s_py: CosetsideSimulator, n_q: int, label: str
) -> None:
    s_cpp._sync_cpp_to_py()
    assert set(s_cpp._active_shots) == set(s_py._active_shots), (
        f"{label}: active keys {sorted(s_cpp._active_shots)} != {sorted(s_py._active_shots)}"
    )
    for b in sorted(s_cpp._active_shots):
        a, p = s_cpp._active_shots[b], s_py._active_shots[b]
        assert a.d == p.d, f"{label} shot {b}: d {a.d} != {p.d}"
        assert a.P == p.P, f"{label} shot {b}: P {a.P} != {p.P}"
        assert np.array_equal(a.h_words, p.h_words), f"{label} shot {b}: h_words differ"
        assert np.array_equal(a.a_union, p.a_union), f"{label} shot {b}: a_union differ"
        assert np.array_equal(a.A_words, p.A_words), f"{label} shot {b}: A_words differ"
        assert np.array_equal(a.ell, p.ell), f"{label} shot {b}: ell differ"
        assert np.array_equal(a.Gamma, p.Gamma), f"{label} shot {b}: Gamma differ"
    xs_c, zs_c, _, _, _ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    xs_p, zs_p, _, _, _ = s_py._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    assert np.array_equal(xs_c, xs_p), f"{label}: xs frames differ"
    assert np.array_equal(zs_c, zs_p), f"{label}: zs frames differ"
    qs = list(range(n_q))
    for pauli in ("X", "Y", "Z"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        assert np.array_equal(
            s_cpp.peek_pauli_batch(qs, p_lit), s_py.peek_pauli_batch(qs, p_lit)
        ), f"{label}: peek_{pauli} differs"
    assert np.array_equal(s_cpp._shot_tiebreak_rng, s_py._shot_tiebreak_rng), (
        f"{label}: _shot_tiebreak_rng differs"
    )
    assert s_cpp.np_rng.bit_generator.state == s_py.np_rng.bit_generator.state, (
        f"{label}: np_rng state differs"
    )


def _seed_high_rank(
    s_cpp: CosetsideSimulator,
    s_py: CosetsideSimulator,
    n_q: int,
    d_target: int,
    batch: int,
    rng: np.random.Generator,
) -> None:
    qs = list(range(n_q))
    mask_h = rng.random((n_q, batch)) < 0.85
    mask_h[:d_target, 0] = False
    op_h = stim.CircuitInstruction("H", qs)
    s_cpp.do_conditional_clifford(op_h, mask_h)
    s_py.do_conditional_clifford(op_h, mask_h)
    _assert_exact_parity(s_cpp, s_py, n_q, f"seed_H_n{n_q}_B{batch}")


@pytest.mark.parametrize(
    ("n_q", "d_target", "batch"),
    [(5, 5, 1), (64, 64, 7), (70, 68, 64), (140, 80, 256)],
)
def test_abi_v3_postselect_and_project_pauli_grid_parity(
    n_q: int, d_target: int, batch: int
) -> None:
    rng = np.random.default_rng(1000 + n_q + batch)
    qs_str = " ".join(map(str, range(n_q)))
    m_str = " ".join(map(str, range(min(n_q, 4))))
    c_txt = f"H {qs_str}\nCX 0 1 1 2\nM {m_str}"
    s_cpp, s_py = _mk_pair(c_txt, batch, seed=n_q * 17 + batch)
    _seed_high_rank(s_cpp, s_py, n_q, d_target, batch, rng)
    s_cpp._sync_cpp_to_py()
    assert max(st.d for st in s_cpp._active_shots.values()) >= d_target
    if n_q >= 70:
        assert max(st.d for st in s_cpp._active_shots.values()) > 64

    # Exercise postselect across X/Y/Z on boundary qubits (0 and n_q - 1) with 0 RNG draws.
    rng_before = s_cpp._shot_tiebreak_rng.copy()
    for pauli in ("X", "Y", "Z"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        for q_sel in (0, n_q - 1):
            flip = rng.random(batch) < 0.5
            if batch == 1:
                flip[0] = True
            desired = rng.integers(0, 2, size=batch, dtype=np.uint8)
            s_cpp._postselect_cpp(q_sel, flip, desired, p_lit)
            s_py.postselect_z_batch(q_sel, flip, desired, p_lit)
            _assert_exact_parity(s_cpp, s_py, n_q, f"post_{pauli}_q{q_sel}")
    assert np.array_equal(s_cpp._shot_tiebreak_rng, rng_before)

    # Exercise unmatched X/Y/Z resets with order=None, reversed subset, and both masks set.
    r0 = np.zeros((n_q, batch), dtype=bool)
    r1 = np.zeros((n_q, batch), dtype=bool)
    r0[0, :] = True
    r1[0, ::2] = True  # r1 takes precedence when both r0 and r1 are True
    r1[n_q - 1, ::3] = True
    if n_q > 64:
        r0[64, 1::2] = True
    for pauli, order in (("X", None), ("Y", [n_q - 1, 0, min(64, n_q - 1)]), ("Z", [])):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        s_cpp._project_pauli_cpp(r0, r1, p_lit, order=order)
        s_py.do_unmatched_noisy_reset_z(r0, r1, p_lit, order=order)
        _assert_exact_parity(s_cpp, s_py, n_q, f"proj_{pauli}_ord{order}")


def _seed_shot(sim: CosetsideSimulator, b: int, st: MRQFState) -> None:
    if sim._use_cpp and sim._cpp_lib is not None and sim._cpp_eng is not None:
        m = max(1, st.d)
        p_arr = np.array(st.P if st.d > 0 else [-1], dtype=np.int32)
        sim._cpp_lib.engine_import_shot(
            sim._cpp_eng,
            b,
            st.d,
            np.ascontiguousarray(st.h_words, dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st.a_union, dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st._A_buf[:m], dtype=np.uint64).ctypes.data_as(u64_p),
            np.ascontiguousarray(st._ell_buf[:m], dtype=np.uint8).ctypes.data_as(u8_p),
            np.ascontiguousarray(st._Gamma_buf[:m, :m], dtype=np.uint8).ctypes.data_as(u8_p),
            p_arr.ctypes.data_as(i32_p),
        )
        sim._cpp_dirty = True
    else:
        sim._active_shots[b] = st


def test_abi_v3_postselect_and_reset_eviction_and_impossible_cases() -> None:
    s_cpp, s_py = _mk_pair("H 0 1 2 3\nCX 2 3\nM 0 1 2 3", 8, seed=77)
    for sim in (s_cpp, s_py):
        st0 = MRQFState(4, 1)
        st0.h_words[0] = np.uint64(0b1010)  # d == 0, h != 0 shot
        _seed_shot(sim, 0, st0)
    # a == 0 impossible postselect on active d==0 shot -> evicts h != 0 cleanly
    flip0 = np.zeros(8, dtype=bool)
    flip0[0] = True
    s_cpp._postselect_cpp(0, flip0, np.ones(8, dtype=np.uint8), "Z")
    s_py.postselect_z_batch(0, flip0, np.ones(8, dtype=np.uint8), "Z")
    _assert_exact_parity(s_cpp, s_py, 4, "d0_h_nz_evict")

    # Multi-qubit X/Y reset eviction chain where q=2 heals and flips zs_base[3]
    m_h = np.ones((4, 8), dtype=bool)
    m_h[2, :] = False
    op_h = stim.CircuitInstruction("H", [0, 1, 2, 3])
    op_cx = stim.CircuitInstruction("CX", [2, 3])
    s_cpp.do_conditional_clifford(op_h, m_h)
    s_py.do_conditional_clifford(op_h, m_h)
    s_cpp.do_conditional_clifford(op_cx, np.zeros((1, 8), dtype=bool))
    s_py.do_conditional_clifford(op_cx, np.zeros((1, 8), dtype=bool))
    r0 = np.zeros((4, 8), dtype=bool)
    r1 = np.zeros((4, 8), dtype=bool)
    r0[2, :] = True
    r1[3, :] = True
    for pauli in ("X", "Y", "Z"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        s_cpp._project_pauli_cpp(r0, r1, p_lit)
        s_py.do_unmatched_noisy_reset_z(r0, r1, p_lit)
        _assert_exact_parity(s_cpp, s_py, 4, f"xy_evict_chain_{pauli}")


@pytest.mark.parametrize(
    ("n_q", "d_target", "batch"),
    [(5, 5, 1), (64, 64, 7), (70, 68, 64), (140, 80, 256)],
)
def test_abi_v3_disjoint_and_sequential_clifford_grid_parity(
    n_q: int, d_target: int, batch: int
) -> None:
    rng = np.random.default_rng(5000 + n_q + batch)
    gates_1q = ("H", "S", "C_XYZ")
    gates_2q = ("CX", "CZ", "CY", "XCZ", "ISWAP", "SWAP")
    c_lines = [f"H {' '.join(map(str, range(n_q)))}", "TICK"]
    for g in gates_1q:
        c_lines.append(f"{g} 0 {n_q - 1}")
    for g in gates_2q:
        c_lines.append(f"{g} 0 1 {n_q - 2} {n_q - 1}")
    c_lines.append("CX 0 1 1 2 2 3 3 4")
    c_lines.append("M 0 1 2")
    s_cpp, s_py = _mk_pair("\n".join(c_lines), batch, seed=n_q + batch)
    _seed_high_rank(s_cpp, s_py, n_q, d_target, batch, rng)
    s_cpp._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    s_py._do_bare_instruction(stim.CircuitInstruction("TICK", []))

    # Disjoint gates via _dcc_disjoint_cpp_v3 vs _dcc_disjoint_py
    for idx, op in enumerate(stim.Circuit("\n".join(c_lines[2:-2]))):
        inst = cast(stim.CircuitInstruction, op)
        n_g = len(inst.target_groups())
        m = rng.random((n_g, batch)) < 0.5
        st_c = s_cpp._next_step_meta(inst)
        st_p = s_py._next_step_meta(inst)
        assert st_c.kind == "unitary" and st_p.kind == "unitary"
        s_cpp._dcc_disjoint_cpp_v3(inst, st_c, m, len(s_cpp._compiled_ref.steps), False)
        s_py._dcc_disjoint_py(inst, st_p, m)
        _assert_exact_parity(s_cpp, s_py, n_q, f"disjoint_{idx}_{inst.name}")

    # Overlapping CX via _dcc_sequential_cpp vs _dcc_sequential_py with callback + postselect
    ov_op = stim.CircuitInstruction("CX", [0, 1, 1, 2, 2, 3, 3, 4])
    pw_c = s_cpp._current_packed_words()
    pw_p = s_py._current_packed_words()
    st_c = s_cpp._next_step_meta(ov_op)
    st_p = s_py._next_step_meta(ov_op)
    assert st_c.kind == "unitary" and st_p.kind == "unitary"

    rows = [rng.random(batch) < 0.5 for _ in range(4)]

    def det_gfn(sim: CosetsideSimulator, use_v3: bool) -> Any:
        def gfn(g_idx: int) -> np.ndarray:
            pk = sim.peek_pauli_batch([0, n_q - 1], "Z")
            if g_idx == 1:
                flip = np.ones(batch, dtype=bool)
                des = np.zeros(batch, dtype=np.uint8)
                if use_v3:
                    sim._postselect_cpp(0, flip, des, "Z")
                else:
                    sim.postselect_z_batch(0, flip, des, "Z")
            return rows[g_idx] ^ (pk[1] < 0)

        return gfn

    s_cpp._dcc_sequential_cpp(st_c, pw_c, None, det_gfn(s_cpp, True))
    s_py._dcc_sequential_py(st_p, pw_p, None, det_gfn(s_py, False))
    _assert_exact_parity(s_cpp, s_py, n_q, f"seq_cb_n{n_q}_B{batch}")

    # Verify d > 64 assertion holds on high-rank grid points
    if n_q >= 70:
        s_cpp._sync_cpp_to_py()
        assert max((st.d for st in s_cpp._active_shots.values()), default=0) > 64


def test_abi_v3_mixed_record_and_quantum_groups_parity() -> None:
    rng = np.random.default_rng(777)
    c_txt = "H 0 1 2\nX_ERROR(0.5) 3\nM 3\nCX rec[-1] 0 1 2 2 0\nM 0 1 2"
    s_cpp, s_py = _mk_pair(c_txt, 16, seed=777)
    mask_h = rng.random((3, 16)) < 0.5
    op_h = stim.CircuitInstruction("H", [0, 1, 2])
    s_cpp.do_conditional_clifford(op_h, mask_h)
    s_py.do_conditional_clifford(op_h, mask_h)
    s_cpp._do_bare_instruction(stim.CircuitInstruction("X_ERROR", [3], [0.5]))
    s_py._do_bare_instruction(stim.CircuitInstruction("X_ERROR", [3], [0.5]))
    s_cpp._do_bare_instruction(stim.CircuitInstruction("M", [3]))
    s_py._do_bare_instruction(stim.CircuitInstruction("M", [3]))
    mixed_op = stim.Circuit(c_txt)[3]
    assert isinstance(mixed_op, stim.CircuitInstruction)
    mask = rng.random((3, 16)) < 0.5
    pw_c = s_cpp._current_packed_words()
    pw_p = s_py._current_packed_words()
    st_c = s_cpp._next_step_meta(mixed_op)
    st_p = s_py._next_step_meta(mixed_op)
    assert st_c.kind == "unitary" and st_p.kind == "unitary"
    assert st_c.quantum_op is not None
    _, q_mask_c = s_cpp._dcc_apply_mixed_record(mixed_op, mask, True)
    _, q_mask_p = s_py._dcc_apply_mixed_record(mixed_op, mask, True)
    s_cpp._dcc_sequential_cpp(st_c, pw_c, q_mask_c, None)
    s_py._dcc_sequential_py(st_p, pw_p, q_mask_p, None)
    _assert_exact_parity(s_cpp, s_py, 4, "mixed_rec_seq")


def test_abi_v3_peek_and_evict_v3_and_v2_bounds_checks() -> None:
    s_cpp, s_py = _mk_pair("H 0 1 2 3\nM 0 1 2 3", 8, seed=9)
    s_cpp.do_conditional_clifford(
        stim.CircuitInstruction("H", [0, 1, 2, 3]), np.zeros((4, 8), dtype=bool)
    )
    s_py.do_conditional_clifford(
        stim.CircuitInstruction("H", [0, 1, 2, 3]), np.zeros((4, 8), dtype=bool)
    )
    lib = s_cpp._cpp_lib
    eng = s_cpp._cpp_eng
    assert lib is not None and eng is not None
    assert lib.coset_kernels_abi_v2() == 2
    assert lib.coset_kernels_abi_v3() == 3
    assert lib.engine_num_qubits(eng) == 4
    assert lib.engine_batch_size(eng) == 8
    assert lib.engine_n_words(eng) == 1

    xs, zs, _, _, _ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    xs_c = np.ascontiguousarray(xs, dtype=np.uint8)
    zs_c = np.ascontiguousarray(zs, dtype=np.uint8)
    x2x, x2z, z2x, z2z, xk, zk = s_cpp._current_packed_words()
    tgts = np.array([0, 2, 3], dtype=np.int64)
    out = np.empty((3, 8), dtype=np.int8)
    rc = lib.engine_peek_pauli_batch_v3(
        eng, 3, tgts.ctypes.data_as(i64_p), 1,
        4, 8, xs_c.ctypes.data_as(u8_p), 4, 8, zs_c.ctypes.data_as(u8_p),
        4, 1, x2x.ctypes.data_as(u64_p), x2z.ctypes.data_as(u64_p),
        z2x.ctypes.data_as(u64_p), z2z.ctypes.data_as(u64_p),
        xk.ctypes.data_as(u8_p), zk.ctypes.data_as(u8_p),
        3, 8, out.ctypes.data_as(i8_p),
    )
    assert rc == 0
    assert np.array_equal(out, s_py.peek_pauli_batch([0, 2, 3], "Y"))

    # engine_batch_evict_v3 rejects wrong dims and succeeds on valid dims
    evx = np.zeros((4, 8), dtype=np.uint8)
    evz = np.zeros((4, 8), dtype=np.uint8)
    ax, az = ctypes.c_int(0), ctypes.c_int(0)
    assert lib.engine_batch_evict_v3(
        eng, 99, 1, x2z.ctypes.data_as(u64_p), z2z.ctypes.data_as(u64_p),
        4, 8, xs_c.ctypes.data_as(u8_p), 4, 8, zs_c.ctypes.data_as(u8_p),
        4, 8, evx.ctypes.data_as(u8_p), 4, 8, evz.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0
    assert lib.engine_batch_evict_v3(
        eng, 4, 1, x2z.ctypes.data_as(u64_p), z2z.ctypes.data_as(u64_p),
        4, 8, xs_c.ctypes.data_as(u8_p), 4, 8, zs_c.ctypes.data_as(u8_p),
        4, 8, evx.ctypes.data_as(u8_p), 4, 8, evz.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) == 0


def test_c1_abi_v3_bounds_and_registration() -> None:
    s_cpp, _ = _mk_pair("H 0 1 2 3\nM 0 1 2 3", 8, seed=11)
    lib, eng = s_cpp._cpp_lib, s_cpp._cpp_eng
    assert lib is not None and eng is not None
    xs = np.zeros((4, 8), dtype=np.uint8)
    zs = np.zeros((4, 8), dtype=np.uint8)
    pw = np.zeros((4, 1), dtype=np.uint64)
    kap = np.zeros(4, dtype=np.uint8)
    fm = np.ones(8, dtype=np.uint8)
    ax, az = ctypes.c_int(0), ctypes.c_int(0)

    assert lib.engine_postselect_batch(
        eng, -1, 2, 8, fm.ctypes.data_as(u8_p), 8, fm.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        kap.ctypes.data_as(u8_p), kap.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0
    assert lib.engine_postselect_batch(
        eng, 0, 2, 3, fm.ctypes.data_as(u8_p), 8, fm.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        kap.ctypes.data_as(u8_p), kap.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0
    bad_ord = np.array([0, 99], dtype=np.int64)
    assert lib.engine_project_pauli(
        eng, 0, 4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        2, bad_ord.ctypes.data_as(i64_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
        kap.ctypes.data_as(u8_p), kap.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0

    bad_tg = np.array([0, 99], dtype=np.int64)
    fk = np.zeros((2, 1), dtype=np.uint8)
    fl = np.zeros((2, 1, 1), dtype=np.uint8)
    fw = np.zeros((2, 1, 1), dtype=np.uint64)
    assert lib.engine_register_unitary_step(
        eng, 2, 1, 1, bad_tg.ctypes.data_as(i64_p),
        fk.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
    ) < 0
    assert lib.engine_unitary_step_num_groups(eng, 999) < 0


def test_c1_v2_bounds_and_python_guards() -> None:
    s_cpp, _ = _mk_pair("H 0 1 2 3\nM 0 1 2 3", 8, seed=11)
    lib, eng = s_cpp._cpp_lib, s_cpp._cpp_eng
    assert lib is not None and eng is not None
    xs = np.zeros((4, 8), dtype=np.uint8)
    zs = np.zeros((4, 8), dtype=np.uint8)
    pw = np.zeros((4, 1), dtype=np.uint64)
    kap = np.zeros(4, dtype=np.uint8)
    fm = np.ones(8, dtype=np.uint8)
    ax, az = ctypes.c_int(0), ctypes.c_int(0)
    fk = np.zeros((2, 1), dtype=np.uint8)
    fl = np.zeros((2, 1, 1), dtype=np.uint8)
    fw = np.zeros((2, 1, 1), dtype=np.uint64)
    good_tg = np.array([0, 1], dtype=np.int64)
    sid = lib.engine_register_unitary_step(
        eng, 2, 1, 1, good_tg.ctypes.data_as(i64_p),
        fk.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
    )
    assert sid >= 0 and lib.engine_unitary_step_num_groups(eng, sid) == 2
    assert lib.engine_batch_inject_general_clifford_v3(
        eng, sid, 8, 8, xs.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0
    assert lib.engine_inject_group_v3(
        eng, sid, 99, 8, fm.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        4, 8, xs.ctypes.data_as(u8_p), 4, 8, zs.ctypes.data_as(u8_p),
        ctypes.byref(ax), ctypes.byref(az),
    ) < 0
    bad_p = np.array([99], dtype=np.int32)
    h1 = np.zeros(1, dtype=np.uint64)
    lib.engine_import_shot(
        eng, 0, 1, h1.ctypes.data_as(u64_p), h1.ctypes.data_as(u64_p),
        h1.ctypes.data_as(u64_p), kap.ctypes.data_as(u8_p), kap.ctypes.data_as(u8_p),
        bad_p.ctypes.data_as(i32_p),
    )
    assert lib.engine_get_active_shot_indices(eng, None, None) == 0

    with pytest.raises(ValueError, match="out of range"):
        s_cpp._postselect_cpp(-1, np.ones(8, bool), np.zeros(8, np.uint8))
    with pytest.raises(ValueError, match="flip_mask has shape"):
        s_cpp._postselect_cpp(0, np.ones(3, bool), np.zeros(8, np.uint8))
    with pytest.raises(ValueError, match="desired_bits has shape"):
        s_cpp._postselect_cpp(0, np.ones(8, bool), np.zeros(2, np.uint8))
    with pytest.raises(IndexError):
        s_cpp._project_pauli_cpp(np.ones((4, 8), bool), np.zeros((4, 8), bool), "X", [0, 99])

    s_cpp._init_batch_size = 1
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        s_cpp.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="reset_zero_mask has shape"):
        s_cpp.do_unmatched_noisy_reset_z(np.ones((4, 1), bool), np.zeros((4, 1), bool))
    assert s_cpp.peek_pauli_batch([0, 1], "Z").shape == (2, 8)
    _ = coset_kernels


def test_c2_compile_append_atomicity_and_bad_targets() -> None:
    s_cpp, _ = _mk_pair("H 0 1 2 3\nM 0 1 2 3", 8, seed=13)
    s_cpp.run()
    z2z_before = s_cpp._compiled_ref.ref_chp.z2z.copy()
    steps_before = len(s_cpp._compiled_ref.steps)
    cursor_before = s_cpp._step_cursor

    with pytest.raises(IndexError):
        s_cpp.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1, 0, 99]), np.ones((2, 8), bool)
        )
    assert len(s_cpp._compiled_ref.steps) == steps_before
    assert np.array_equal(s_cpp._compiled_ref.ref_chp.z2z, z2z_before)

    with pytest.raises(IndexError):
        s_cpp.do_conditional_clifford(
            stim.CircuitInstruction("CX", [stim.target_rec(-99), 0]), np.ones((1, 8), bool)
        )
    assert len(s_cpp._compiled_ref.steps) == steps_before
    assert s_cpp._step_cursor == cursor_before


def _raise_on_evict_lib(real_lib: ctypes.CDLL) -> ctypes.CDLL:
    class _RaiseOnEvict:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_batch_evict":
                return fn
            return lambda *args: (fn(*args), (_ for _ in ()).throw(
                KeyboardInterrupt("post-evict interrupt")
            ))

    return cast(ctypes.CDLL, _RaiseOnEvict())


def test_c12_evict_begin_end_and_callback_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Evict wrap on MPP X0*X0 preserves _cpp_dirty and matches Python
    mpp_op = stim.CircuitInstruction(
        "MPP", [stim.target_x(0), stim.target_combiner(), stim.target_x(0)]
    )
    s_cpp, s_py = _mk_pair("MPP X0*X0\nM 0 1", 4, seed=17)
    for sim in (s_cpp, s_py):
        st = MRQFState(2, 1)
        st.h_words[0] = np.uint64(1)
        _seed_shot(sim, 0, st)
    s_cpp.interactive_do(mpp_op)
    s_py.interactive_do(mpp_op)
    assert s_cpp._cpp_dirty
    assert list(s_cpp.active_shots.keys()) == list(s_py.active_shots.keys()) == []
    assert np.array_equal(s_cpp.peek_pauli_batch([0], "Z"), s_py.peek_pauli_batch([0], "Z"))

    # 2. Interrupt right after engine_batch_evict keeps _cpp_dirty=True so re-export matches engine
    s_int, _ = _mk_pair("MPP X0*X0\nM 0 1", 4, seed=18)
    st_i = MRQFState(2, 1)
    st_i.h_words[0] = np.uint64(1)
    _seed_shot(s_int, 0, st_i)
    s_int._sync_cpp_to_py()
    real_lib = s_int._cpp_lib
    assert real_lib is not None
    monkeypatch.setattr(s_int, "_cpp_lib", _raise_on_evict_lib(real_lib))
    with pytest.raises(KeyboardInterrupt, match="post-evict interrupt"):
        s_int.interactive_do(mpp_op)
    assert s_int._cpp_dirty
    monkeypatch.setattr(s_int, "_cpp_lib", real_lib)
    s_int.clear()
    assert list(s_int.active_shots.keys()) == []
    assert not s_int._cpp_dirty
    fresh_i, _ = _mk_pair("MPP X0*X0\nM 0 1", 4, seed=18)
    assert np.array_equal(s_int.peek_pauli_batch([0], "Z"), fresh_i.peek_pauli_batch([0], "Z"))

    # 3. Hostile condition_group_fn callbacks raise RuntimeError / ValueError cleanly
    s_cb, _ = _mk_pair("CX 0 1 1 2\nH 0\nM 0 1 2", 4, seed=19)
    cx = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    h0 = stim.CircuitInstruction("H", [0])
    tick = stim.CircuitInstruction("TICK", [])
    xerr = stim.CircuitInstruction("X_ERROR", [0], [0.1])
    forbidden: list[tuple[str, Any]] = [
        ("clear", lambda: s_cb.clear()),
        ("run", lambda: s_cb.run()),
        ("interactive_do", lambda: s_cb.interactive_do(tick)),
        ("finish_interactive_run", lambda: s_cb.finish_interactive_run()),
        (
            "do_conditional_clifford",
            lambda: s_cb.do_conditional_clifford(h0, np.ones((1, 4), bool)),
        ),
        ("do_conditional_pauli_noise", lambda: s_cb.do_conditional_pauli_noise(xerr, None)),
        (
            "do_unmatched_noisy_reset_z",
            lambda: s_cb.do_unmatched_noisy_reset_z(
                np.zeros((3, 4), bool), np.zeros((3, 4), bool)
            ),
        ),
        ("do_copies_on_fused_step", lambda: s_cb.do_copies_on_fused_step(tick, [], lambda _: None)),
        ("_do_bare_instruction", lambda: s_cb._do_bare_instruction(tick)),
        ("_next_step_meta", lambda: s_cb._next_step_meta(tick)),
    ]
    for fn_name, fn in forbidden:
        s_cb._step_cursor = 0

        def hostile(g_idx: int, action: Any = fn) -> np.ndarray:
            if g_idx == 0:
                action()
            return np.ones(4, dtype=bool)

        pat = rf"^{fn_name} is forbidden inside condition_group_fn"
        with pytest.raises(RuntimeError, match=pat):
            s_cb.do_conditional_clifford(cx, condition_group_fn=hostile)
        assert not s_cb._in_group_callback

    s_cb._step_cursor = 0
    with pytest.raises(ValueError, match="returned shape"):
        s_cb.do_conditional_clifford(cx, condition_group_fn=lambda _g: np.zeros(2, bool))
    with pytest.raises(ValueError, match="returned shape"):
        s_cb.do_conditional_clifford(cx, condition_group_fn=lambda _g: np.zeros((2, 4), bool))
    assert not s_cb._in_group_callback


def test_m11_project_pauli_xy_eviction_chain_updates_zs_base() -> None:
    m_h = np.array([[True] * 8, [False] * 8], dtype=bool)
    r0 = np.ones((2, 8), dtype=bool)
    r1 = np.zeros((2, 8), dtype=bool)
    s_cpp, s_py = _mk_pair("H 0 1\nCX 0 1\nM 0 1", 8, seed=77)
    for sim in (s_cpp, s_py):
        sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), m_h)
        sim.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1]), np.ones((1, 8), dtype=bool)
        )
    s_cpp._project_pauli_cpp(r0, r1, "X", order=[1, 0])
    s_py.do_unmatched_noisy_reset_z(r0, r1, "X", order=[1, 0])
    _assert_exact_parity(s_cpp, s_py, 2, "m11_x_evict_chain")
    assert len(s_cpp._active_shots) == 0

    s_cpp_y, s_py_y = _mk_pair("H 0 1\nS 1\nCX 0 1\nM 0 1", 8, seed=77)
    for sim in (s_cpp_y, s_py_y):
        sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), m_h)
        sim.do_conditional_clifford(
            stim.CircuitInstruction("S", [1]), np.ones((1, 8), dtype=bool)
        )
        sim.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1]), np.ones((1, 8), dtype=bool)
        )
    s_cpp_y._project_pauli_cpp(r0, r1, "Y", order=[1, 0])
    s_py_y.do_unmatched_noisy_reset_z(r0, r1, "Y", order=[1, 0])
    _assert_exact_parity(s_cpp_y, s_py_y, 2, "m11_y_evict_chain")


def test_m12_inject_group_v3_uses_per_group_evict_table() -> None:
    c_txt = "H 0\nTICK\nCX 0 1\nTICK\nCX 0 1 1 0\nM 0 1"
    s_cpp, s_py = _mk_pair(c_txt, 4, seed=5)
    for sim in (s_cpp, s_py):
        sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.ones((1, 4), bool))
        sim._do_bare_instruction(stim.CircuitInstruction("TICK", []))
        sim.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1]), np.zeros((1, 4), bool)
        )
        sim._do_bare_instruction(stim.CircuitInstruction("TICK", []))
        sim.apply_pauli("Z", np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool))
        sim.apply_pauli("X", np.array([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool))
    op_ov = stim.CircuitInstruction("CX", [0, 1, 1, 0])
    pw_c, pw_p = s_cpp._current_packed_words(), s_py._current_packed_words()
    st_c, st_p = s_cpp._next_step_meta(op_ov), s_py._next_step_meta(op_ov)
    assert st_c.kind == "unitary" and st_p.kind == "unitary"

    def gfn(g_idx: int) -> np.ndarray:
        return np.array([g_idx != 0] * 4, dtype=bool)

    s_cpp._dcc_sequential_cpp(st_c, pw_c, None, gfn)
    s_py._dcc_sequential_py(st_p, pw_p, None, gfn)
    _assert_exact_parity(s_cpp, s_py, 2, "m12_group_evict")
    assert len(s_cpp._active_shots) == 0


def test_m14_postselect_xy_nonzero_frames_det_and_stochastic() -> None:
    m_h = np.array([[False] * 4, [True] * 4], dtype=bool)
    z_m = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)
    x_m = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
    flip = np.ones(4, dtype=bool)
    des = np.zeros(4, dtype=np.uint8)
    for pauli, q in (("X", 0), ("Y", 1), ("Y", 0), ("X", 1)):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        s_cpp, s_py = _mk_pair("H 0 1\nS 1\nM 0 1", 4, seed=31)
        for sim in (s_cpp, s_py):
            sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), m_h)
            sim.do_conditional_clifford(stim.CircuitInstruction("S", [1]), np.zeros((1, 4), bool))
            sim.apply_pauli("Z", z_m)
            sim.apply_pauli("X", x_m)
        s_cpp._postselect_cpp(q, flip, des, p_lit)
        s_py.postselect_z_batch(q, flip, des, p_lit)
        _assert_exact_parity(s_cpp, s_py, 2, f"m14_{pauli}_q{q}")


def test_r5_sa_ts1_seq_cpp_callback_public_postselect_and_all_true_g0() -> None:
    s_cpp, s_py = _mk_pair("H 0 1 2\nCX 0 1 1 2\nM 0 1 2", 16, seed=55)
    h_op = stim.CircuitInstruction("H", [0, 1, 2])
    m_h = np.zeros((3, 16), dtype=bool)
    s_cpp.do_conditional_clifford(h_op, m_h)
    s_py.do_conditional_clifford(h_op, m_h)
    cx_op = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    pw_c, pw_p = s_cpp._current_packed_words(), s_py._current_packed_words()
    st_c, st_p = s_cpp._next_step_meta(cx_op), s_py._next_step_meta(cx_op)
    assert st_c.kind == "unitary" and st_p.kind == "unitary"

    def make_gfn(sim: CosetsideSimulator) -> Any:
        def gfn(g_idx: int) -> np.ndarray:
            _ = sim.peek_x(0), sim.peek_y(1), sim.peek_z(2), len(sim.active_shots)
            _ = sim.get_current_noisy_tableau_pauli_state([0, 1], "Z")
            if g_idx == 0:
                flip = np.zeros(16, dtype=bool)
                flip[:8] = True
                sim.postselect_z_batch(0, flip, np.ones(16, dtype=np.uint8), "Z")
                return np.ones(16, dtype=bool)
            return np.array([b % 2 == 0 for b in range(16)], dtype=bool)

        return gfn

    s_cpp._dcc_sequential_cpp(st_c, pw_c, None, make_gfn(s_cpp))
    s_py._dcc_sequential_py(st_p, pw_p, None, make_gfn(s_py))
    _assert_exact_parity(s_cpp, s_py, 3, "r5_sa_ts1_pub_ps")


def test_c2_oob_ref_chp_non_mutation_for_none_and_group_fn() -> None:
    s_cpp, _ = _mk_pair("H 0 1\nM 0 1", 4, seed=41)
    s_cpp.run()
    ref_snap = [w.copy() for w in s_cpp._compiled_ref.ref_chp.snapshot_packed()]
    last_snap = [w.copy() for w in s_cpp._compiled_ref._last_packed_snapshot]
    steps_before = len(s_cpp._compiled_ref.steps)
    cursor_before = s_cpp._step_cursor
    bad_cx = stim.CircuitInstruction("CX", [0, 1, 0, 99])

    for call in (
        lambda: s_cpp.do_conditional_clifford(
            bad_cx, condition_group_fn=lambda _g: np.ones(4, bool)
        ),
        lambda: s_cpp.do_conditional_clifford(bad_cx),
    ):
        with pytest.raises(IndexError):
            call()
        assert len(s_cpp._compiled_ref.steps) == steps_before
        assert s_cpp._step_cursor == cursor_before
        cur_ref = s_cpp._compiled_ref.ref_chp.snapshot_packed()
        cur_last = s_cpp._compiled_ref._last_packed_snapshot
        assert all(np.array_equal(a, b) for a, b in zip(cur_ref, ref_snap))
        assert all(np.array_equal(a, b) for a, b in zip(cur_last, last_snap))


def test_c2_mixed_record_compile_append_and_in_circuit_oob_atomicity() -> None:
    s_cpp, _ = _mk_pair("H 0 1\nM 0 1\nCX 0 1", 4, seed=42)
    s_cpp._do_bare_instruction(stim.CircuitInstruction("H", [0, 1]))
    s_cpp._do_bare_instruction(stim.CircuitInstruction("M", [0, 1]))
    ref_snap = [w.copy() for w in s_cpp._compiled_ref.ref_chp.snapshot_packed()]
    steps_before = len(s_cpp._compiled_ref.steps)
    xs_b, zs_b, _, _, _ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)

    with pytest.raises(IndexError):
        s_cpp.do_conditional_clifford(stim.CircuitInstruction("CX", [0, 1, 0, 99]))
    assert s_cpp._step_cursor == 2 and len(s_cpp._compiled_ref.steps) == steps_before
    assert s_cpp._flip_simulator.num_qubits == 2

    s_cpp.run()
    ref_snap2 = [w.copy() for w in s_cpp._compiled_ref.ref_chp.snapshot_packed()]
    last_snap2 = [w.copy() for w in s_cpp._compiled_ref._last_packed_snapshot]
    steps2 = len(s_cpp._compiled_ref.steps)
    for bad_op in (
        stim.CircuitInstruction("CX", [0, 1, 0, 99]),
        stim.CircuitInstruction("M", [0, 99]),
    ):
        with pytest.raises(IndexError):
            s_cpp._compiled_ref.compile_and_append_instruction(bad_op)
        assert len(s_cpp._compiled_ref.steps) == steps2
        cur_ref = s_cpp._compiled_ref.ref_chp.snapshot_packed()
        cur_last = s_cpp._compiled_ref._last_packed_snapshot
        assert all(np.array_equal(a, b) for a, b in zip(cur_ref, ref_snap2))
        assert all(np.array_equal(a, b) for a, b in zip(cur_last, last_snap2))

    xs_b2, zs_b2, _, _, _ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    bad_mix = stim.CircuitInstruction("CX", [stim.target_rec(-1), 0, stim.target_rec(-2), 99])
    with pytest.raises(IndexError):
        s_cpp.do_conditional_clifford(bad_mix, np.ones((2, 4), bool))
    xs_a, zs_a, _, _, _ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    assert np.array_equal(xs_a, xs_b2) and np.array_equal(zs_a, zs_b2)
    _ = ref_snap, xs_b, zs_b


def test_r5_sa_low2a_no_orphaned_unitary_step_on_bad_mask() -> None:
    s_cpp, _ = _mk_pair("H 0 1\nM 0 1", 4, seed=43)
    s_cpp.run()
    lib, eng = s_cpp._cpp_lib, s_cpp._cpp_eng
    assert lib is not None and eng is not None
    steps_before = len(s_cpp._compiled_ref.steps)
    k = 0
    while lib.engine_unitary_step_num_groups(eng, k) >= 0:
        k += 1
    op_h = stim.CircuitInstruction("H", [0, 1])
    bad_mask = np.ones((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        s_cpp.do_conditional_clifford(op_h, bad_mask)
    assert lib.engine_unitary_step_num_groups(eng, k) == -1
    assert len(s_cpp._unitary_step_ids) == k
    assert len(s_cpp._compiled_ref.steps) == steps_before

    st_c = s_cpp._next_step_meta(op_h)
    assert st_c.kind == "unitary"
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        s_cpp._dcc_disjoint_cpp_v3(op_h, st_c, bad_mask, steps_before, True)
    assert lib.engine_unitary_step_num_groups(eng, k) == -1
    assert len(s_cpp._unitary_step_ids) == k
    assert len(s_cpp._compiled_ref.steps) == steps_before
    assert s_cpp._step_cursor == steps_before

    pw_c = s_cpp._current_packed_words()
    with pytest.raises(ValueError, match="returned shape"):
        s_cpp._dcc_sequential_cpp(st_c, pw_c, None, lambda _g: np.zeros(1, bool))
    assert lib.engine_unitary_step_num_groups(eng, k) == -1
    assert len(s_cpp._unitary_step_ids) == k


def test_d0_evict_parity_unmatched_reset_and_postselect() -> None:
    s_cpp, s_py = _mk_pair("H 0 1\nM 0 1", 4, seed=99)
    s_rz, _ = _mk_pair("H 0 1\nM 0 1", 4, seed=99)
    for sim in (s_cpp, s_py, s_rz):
        st = MRQFState(2, 1)
        st.h_words[0] = np.uint64(1)
        _seed_shot(sim, 0, st)
    r0 = np.zeros((2, 4), dtype=bool)
    r1 = np.zeros((2, 4), dtype=bool)
    r0[1, 1] = True
    s_cpp._project_pauli_cpp(r0, r1, "Z")
    s_py.do_unmatched_noisy_reset_z(r0, r1, "Z")
    s_rz.do_unmatched_noisy_reset_z(r0, r1, "Z")
    _assert_exact_parity(s_cpp, s_py, 2, "d0_evict_proj_vs_py")
    _assert_exact_parity(s_rz, s_py, 2, "d0_evict_rz_vs_py")
    assert len(s_py._active_shots) == 0

    s_ps_c, s_ps_p = _mk_pair("H 0 1\nM 0 1", 4, seed=101)
    for sim in (s_ps_c, s_ps_p):
        st = MRQFState(2, 1)
        st.h_words[0] = np.uint64(1)
        _seed_shot(sim, 0, st)
    flip = np.array([False, True, False, False], dtype=bool)
    des = np.zeros(4, dtype=np.uint8)
    s_ps_c._postselect_cpp(1, flip, des, "Z")
    s_ps_p.postselect_z_batch(1, flip, des, "Z")
    _assert_exact_parity(s_ps_c, s_ps_p, 2, "d0_evict_postselect")
    assert len(s_ps_p._active_shots) == 0

    s_nh_c, s_nh_p = _mk_pair("H 0 1\nM 0 1", 4, seed=102)
    for sim in (s_nh_c, s_nh_p):
        st0 = MRQFState(2, 1)
        st0.h_words[0] = np.uint64(1)
        _seed_shot(sim, 0, st0)
        st1 = MRQFState(2, 1)
        st1.p2_apply_quarter_turn(np.array([1], np.uint64), np.array([1], np.uint64), 1)
        st1.p2_apply_quarter_turn(np.array([2], np.uint64), np.array([2], np.uint64), 1)
        _seed_shot(sim, 1, st1)
    s_nh_c._postselect_cpp(0, flip, des, "X")
    s_nh_p.postselect_z_batch(0, flip, des, "X")
    _assert_exact_parity(s_nh_c, s_nh_p, 2, "d0_evict_ps_nonhealing")
    assert list(s_nh_p._active_shots) == [1]


def test_abi_v2_and_engine_create_overflow_bounds() -> None:
    s_cpp, _ = _mk_pair("H 0 1 2 3\nM 0 1 2 3", 8, seed=47)
    lib, eng = s_cpp._cpp_lib, s_cpp._cpp_eng
    assert lib is not None and eng is not None
    rng = np.zeros(1, dtype=np.uint64)
    assert not lib.engine_create(2147483640, 1, rng.ctypes.data_as(u64_p))
    assert not lib.engine_create(-1, 1, rng.ctypes.data_as(u64_p))

    bad_tg = np.array([-1, 0], dtype=np.int64)
    cm = np.ones((1, 8), dtype=np.uint8)
    xs = np.zeros((4, 8), dtype=np.uint8)
    zs = np.zeros((4, 8), dtype=np.uint8)
    fk = np.zeros((1, 1), dtype=np.uint8)
    fl = np.zeros((1, 1, 2), dtype=np.uint8)
    fw = np.zeros((1, 1, 1), dtype=np.uint64)
    pw = np.zeros((4, 1), dtype=np.uint64)
    for arity, n_fac in ((2, 1), (1, 2)):
        ax, az = ctypes.c_int(1), ctypes.c_int(1)
        lib.engine_batch_inject_general_clifford(
            eng, 0x40000001, arity, n_fac,
            bad_tg.ctypes.data_as(i64_p), cm.ctypes.data_as(u8_p),
            xs.ctypes.data_as(u8_p), zs.ctypes.data_as(u8_p),
            fk.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
            fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
            fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
            fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
            pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
            xs.ctypes.data_as(u8_p), zs.ctypes.data_as(u8_p),
            ctypes.byref(ax), ctypes.byref(az),
        )
        assert (ax.value, az.value) == (0, 0)
    assert lib.engine_register_unitary_step(
        eng, 0x40000001, 2, 1, bad_tg.ctypes.data_as(i64_p),
        fk.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        fl.ctypes.data_as(u8_p), fl.ctypes.data_as(u8_p),
        fw.ctypes.data_as(u64_p), fw.ctypes.data_as(u64_p), fk.ctypes.data_as(u8_p),
        4, 1, pw.ctypes.data_as(u64_p), pw.ctypes.data_as(u64_p),
    ) < 0


def _snap_coset_state(sim: CosetsideSimulator) -> tuple[Any, ...]:
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    return (
        len(sim._compiled_ref.steps),
        sim._step_cursor,
        [w.copy() for w in sim._compiled_ref.ref_chp.snapshot_packed()],
        [w.copy() for w in sim._compiled_ref._last_packed_snapshot],
        xs.copy(),
        zs.copy(),
    )


def _assert_coset_state_same(sim: CosetsideSimulator, before: tuple[Any, ...]) -> None:
    after = _snap_coset_state(sim)
    assert after[:2] == before[:2]
    assert all(np.array_equal(a, b) for a, b in zip(after[2], before[2]))
    assert all(np.array_equal(a, b) for a, b in zip(after[3], before[3]))
    assert np.array_equal(after[4], before[4]) and np.array_equal(after[5], before[5])


def test_c2_nonbool_and_boom_invert_masks_are_atomic() -> None:
    class _BoomInvert:
        def __bool__(self) -> bool:
            return False

        def __invert__(self) -> Any:
            raise RuntimeError("boom invert")

    class _BoomBool:
        def __bool__(self) -> bool:
            raise RuntimeError("boom bool")

    ok_2r = (
        np.array([[0.0, 1.5], [1.0, 1.0]], dtype=np.float64),
        np.array([[None, True], [True, True]], dtype=object),
        np.array([[_BoomInvert(), True], [True, True]], dtype=object),
    )
    bad_2r = (
        np.array([["False", "True"], ["True", "True"]]),
        np.array([[b"0", b"1"], [b"1", b"1"]]),
        np.zeros((2, 2), dtype=[("x", "i4")]),
        np.array([[_BoomBool(), True], [True, True]], dtype=object),
    )
    configs = (
        ("H 0 1 2\nM 0 1 2", stim.CircuitInstruction("CX", [0, 1, 1, 2]), True, ok_2r, bad_2r),
        ("H 0 1 2\nM 0 1 2", stim.CircuitInstruction("CX", [0, 1, 1, 2]), False, ok_2r, bad_2r),
        ("H 0 1 2 3\nM 0 1 2 3", stim.CircuitInstruction("CX", [0, 1, 2, 3]), False, ok_2r, bad_2r),
        (
            "H 0\nM 0", stim.CircuitInstruction("X", [0]), True,
            (np.array([[0, 1]], dtype=np.uint8),), (np.array([["0", "1"]]),),
        ),
        (
            "X 3\nM 3", stim.Circuit("CX rec[-1] 0")[0], True,
            (np.array([[1, 1]], dtype=np.uint8),), (np.array([["1", "1"]]),),
        ),
    )
    for c_txt, op, use_cpp, ok_masks, bad_masks in configs:
        for m in ok_masks:
            sim, ref = _mk_pair(c_txt, 2, seed=51)
            sim._use_cpp, ref._use_cpp = use_cpp, use_cpp
            sim.run()
            ref.run()
            sim.do_conditional_clifford(op, m)
            ref.do_conditional_clifford(op, np.asarray(m, dtype=bool))
            assert _snap_coset_state(sim)[0:2] == _snap_coset_state(ref)[0:2]
            _assert_coset_state_same(sim, _snap_coset_state(ref))
        for m in bad_masks:
            sim, _ = _mk_pair(c_txt, 2, seed=51)
            sim._use_cpp = use_cpp
            sim.run()
            before = _snap_coset_state(sim)
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                sim.do_conditional_clifford(op, m)  # type: ignore[arg-type]
            _assert_coset_state_same(sim, before)

    mix_op = stim.CircuitInstruction("CX", [stim.target_rec(-1), 0, 1, 2, 2, 3])
    for use_cpp in (True, False):
        for m in bad_2r:
            bad_mix = np.concatenate([np.ones((1, 2), dtype=m.dtype), m], axis=0)
            sim, _ = _mk_pair("H 0 1 2 3\nX 3\nM 3", 2, seed=52)
            sim._use_cpp = use_cpp
            sim.run()
            before = _snap_coset_state(sim)
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                sim.do_conditional_clifford(mix_op, bad_mix)  # type: ignore[arg-type]
            _assert_coset_state_same(sim, before)


def test_c2_setprofile_reentrancy_past_end_dcc() -> None:
    sim, _ = _mk_pair("H 0 1 2\nM 0 1 2", 2, seed=61)
    sim.run()
    seen: list[int] = []
    fired = [False]

    def prof_h0(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired[0]
            and event == "return"
            and frame.f_code.co_name == "_check_op_qubit_bounds"
        ):
            fired[0] = True
            sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.ones((1, 2), bool))

    def gfn(g_idx: int) -> np.ndarray:
        if g_idx == 0:
            seen.append(int(sim.peek_pauli_batch([0], "Z")[0, 0]))
        return np.array([g_idx != 0, True], dtype=bool)

    sys.setprofile(prof_h0)
    try:
        sim.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1, 1, 2]), condition_group_fn=gfn
        )
    finally:
        sys.setprofile(None)
    assert fired[0] and seen == [0]

    ref, _ = _mk_pair("H 0 1 2\nM 0 1 2", 2, seed=61)
    ref.run()
    ref.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.ones((1, 2), bool))
    ref.do_conditional_clifford(
        stim.CircuitInstruction("CX", [0, 1, 1, 2]),
        condition_group_fn=lambda g: np.array([g != 0, True], dtype=bool),
    )
    ref._sync_cpp_to_py()
    _assert_exact_parity(sim, ref, 3, "setprofile_nested_dcc")

    sim_b, _ = _mk_pair("H 0 1\nM 0 1", 2, seed=62)
    sim_b.run()
    fired_b = [False]

    def prof_clear(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired_b[0]
            and event == "return"
            and frame.f_code.co_name == "_check_op_qubit_bounds"
        ):
            fired_b[0] = True
            sim_b.clear()

    sys.setprofile(prof_clear)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim_b.do_conditional_clifford(
                stim.CircuitInstruction("CX", [0, 1, 1, 0]), np.zeros((2, 2), bool)
            )
    finally:
        sys.setprofile(None)
    assert fired_b[0] and sim_b._step_cursor == 0
    assert np.array_equal(sim_b.peek_pauli_batch([0, 1], "Z"), np.ones((2, 2), dtype=np.int8))


def test_r3_med1_eval_group_callback_reentrancy_rejected() -> None:
    h0 = stim.CircuitInstruction("H", [0])
    cx = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    ones1 = np.ones((1, 2), bool)

    class _ReenterArr:
        def __init__(self, fn: Any, row: np.ndarray) -> None:
            self._fn, self._row = fn, row

        def __array__(self, dtype: Any = None) -> np.ndarray:
            self._fn()
            return self._row

    sim1, _ = _mk_pair("H 0 1 2\nM 0 1 2", 2, seed=71)
    sim1.run()
    with pytest.raises(RuntimeError, match="forbidden"):
        sim1.do_conditional_clifford(
            cx,
            condition_group_fn=lambda g: (
                _ReenterArr(lambda: sim1.do_conditional_clifford(h0, ones1), np.ones(2, bool))
                if g == 0
                else np.ones(2, bool)
            ),
        )
    assert len(sim1._compiled_ref.steps) == 2 and sim1._step_cursor == 2

    for use_cpp, direct in ((False, False), (True, True)):
        sim_c, _ = _mk_pair("H 0 1 2\nM 0 1 2", 2, seed=72)
        sim_c._use_cpp = use_cpp
        sim_c.run()

        def gfn(g: int, s: CosetsideSimulator = sim_c) -> Any:
            return np.zeros(2, bool) if g == 0 else _ReenterArr(s.clear, np.zeros(2, bool))

        with pytest.raises(RuntimeError, match="forbidden"):
            if direct:
                st_m = sim_c._compiled_ref.compile_instruction(cx)[0]
                sim_c._dcc_sequential_cpp(st_m, sim_c._current_packed_words(), None, gfn)
            else:
                sim_c.do_conditional_clifford(cx, condition_group_fn=gfn)
        assert sim_c._step_cursor == (2 if direct else 3)

    sim4, _ = _mk_pair("H 0 1 2\nM 0 1 2", 2, seed=73)
    sim4.run()
    fired = [False]

    def prof_ret(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_eval_group_condition":
            fired[0] = True
            sim4.do_conditional_clifford(h0, ones1)

    sys.setprofile(prof_ret)
    try:
        with pytest.raises(RuntimeError, match="forbidden"):
            sim4.do_conditional_clifford(cx, condition_group_fn=lambda _g: np.ones(2, bool))
    finally:
        sys.setprofile(None)
    assert fired[0] and len(sim4._compiled_ref.steps) == 2 and sim4._step_cursor == 2


def test_r3_med2_pending_none_reentrancy_rejected() -> None:
    h2 = stim.CircuitInstruction("H", [0, 1])
    x0 = stim.CircuitInstruction("X", [0])
    m_x = np.array([[False, True]], bool)

    sim5, _ = _mk_pair("H 0 1\nX 0\nTICK\nM 0 1", 2, seed=74)
    fired5 = [False]

    def prof_x0(frame: Any, event: str, _arg: Any) -> None:
        if not fired5[0] and event == "return" and frame.f_code.co_name == "_check_op_qubit_bounds":
            fired5[0] = True
            sim5.do_conditional_clifford(x0, m_x)

    sys.setprofile(prof_x0)
    try:
        with pytest.raises(ValueError, match="re-entrant"):
            sim5.do_conditional_clifford(h2, np.zeros((2, 2), bool))
    finally:
        sys.setprofile(None)
    xs5, zs5, _, _, _ = sim5._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    assert fired5[0] and sim5._step_cursor == 2
    assert xs5.tolist() == [[True, False], [False, False]] and not zs5.any()

    c_mix = stim.Circuit("X_ERROR(1) 0\nM 0\nCX rec[-1] 1 0 2\nM 1\nTICK")
    sim8, _ = _mk_pair(str(c_mix), 2, seed=75)
    sim8._do_bare_instruction(c_mix[0])
    sim8._do_bare_instruction(c_mix[1])
    fired8 = [False]

    def prof_m1(frame: Any, event: str, _arg: Any) -> None:
        if not fired8[0] and event == "return" and frame.f_code.co_name == "_check_op_qubit_bounds":
            fired8[0] = True
            sim8._do_bare_instruction(c_mix[3])

    sys.setprofile(prof_m1)
    try:
        with pytest.raises(ValueError, match="re-entrant"):
            sim8.do_conditional_clifford(c_mix[2], np.ones((2, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired8[0] and sim8._step_cursor == 4

    sim_noop, _ = _mk_pair("H 0\nM 0", 2, seed=76)
    sim_noop.run()
    fired_n = [False]

    def prof_noop(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired_n[0]
            and event == "return"
            and frame.f_code.co_name == "_dcc_collect_mixed_record"
        ):
            fired_n[0] = True
            sim_noop.clear()

    sys.setprofile(prof_noop)
    try:
        with pytest.raises(ValueError, match="cleared"):
            sim_noop.do_conditional_clifford(
                stim.CircuitInstruction("CX", [stim.target_rec(-1), 0]), np.ones((1, 2), bool)
            )
    finally:
        sys.setprofile(None)
    assert fired_n[0] and sim_noop._step_cursor == 0


def test_r3_med2_post_1711_int_invert_and_allowed_postselect() -> None:
    h2 = stim.CircuitInstruction("H", [0, 1])
    x0 = stim.CircuitInstruction("X", [0])
    m_x = np.array([[False, True]], bool)

    class _Trig:
        def __init__(self, v: bool, fn: Any, fired: list[bool]) -> None:
            self.v, self.fn, self.fired = v, fn, fired

        def __bool__(self) -> bool:
            if not self.fired[0]:
                self.fired[0] = True
                self.fn()
            return self.v

        def __int__(self) -> int:
            return int(self.__bool__())

    for use_cpp in (True, False):
        sim, _ = _mk_pair("H 0 1\nX 0\nTICK\nM 0 1", 2, seed=77)
        sim._use_cpp = use_cpp
        fired = [False]
        obj_m = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(2, 2):
            obj_m[idx] = _Trig(False, lambda s=sim: s.do_conditional_clifford(x0, m_x), fired)
        pat_m = r"condition_met_mask has shape \(1, 2\), expected \(2, 2\)"
        with pytest.raises(ValueError, match=pat_m):
            sim.do_conditional_clifford(h2, obj_m)  # type: ignore[arg-type]
        xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        assert fired[0] and sim._step_cursor == 0
        assert not xs.any() and not zs.any()

        sim_fit, _ = _mk_pair("X 0 1\nH 0 1\nTICK\nM 0 1", 2, seed=77)
        sim_fit._use_cpp = use_cpp
        x01 = stim.CircuitInstruction("X", [0, 1])
        fired_fit = [False]
        fit_m = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(2, 2):
            fit_m[idx] = _Trig(
                False,
                lambda s=sim_fit, op=x01: s.do_conditional_clifford(op, np.ones((2, 2), bool)),
                fired_fit,
            )
        sim_fit.do_conditional_clifford(h2, fit_m)  # type: ignore[arg-type]
        assert fired_fit[0] and sim_fit._step_cursor == 2

    s12 = stim.CircuitInstruction("S", [1, 2])
    h0 = stim.CircuitInstruction("H", [0])
    _, ref_ps = _mk_pair("H 0\nS 1 2\nM 0 1 2", 2, seed=78)
    ref_ps.do_conditional_clifford(h0, np.ones((1, 2), bool))
    ref_ps.postselect_z_batch(0, np.array([True, False], bool), np.zeros(2, np.uint8), "Z")
    ref_ps.do_conditional_clifford(s12, np.zeros((2, 2), bool))

    sim_ps, _ = _mk_pair("H 0\nS 1 2\nM 0 1 2", 2, seed=78)
    sim_ps.do_conditional_clifford(h0, np.ones((1, 2), bool))
    fired_ps = [False]
    ps_m = np.empty((2, 2), dtype=object)
    for idx in np.ndindex(2, 2):
        ps_m[idx] = _Trig(
            False,
            lambda: sim_ps.postselect_z_batch(
                0, np.array([True, False], bool), np.zeros(2, np.uint8), "Z"
            ),
            fired_ps,
        )
    sim_ps.do_conditional_clifford(s12, ps_m)  # type: ignore[arg-type]
    assert fired_ps[0] and sim_ps._step_cursor == 2
    _assert_exact_parity(sim_ps, ref_ps, 3, "allowed_postselect_bool_trigger")


def test_r3_low1_clear_epoch_and_clear_run_restore() -> None:
    h0 = stim.CircuitInstruction("H", [0])
    sim9, _ = _mk_pair("H 0\nM 0", 2, seed=79)
    sim9.interactive_do(h0)
    sim9.interactive_do(stim.CircuitInstruction("M", [0]))
    fired9 = [False]

    def prof_clr(frame: Any, event: str, _arg: Any) -> None:
        if not fired9[0] and event == "return" and frame.f_code.co_name == "_check_dcc_mask":
            fired9[0] = True
            sim9.clear()

    sys.setprofile(prof_clr)
    try:
        with pytest.raises(ValueError, match="cleared"):
            sim9.do_conditional_clifford(h0, np.ones((1, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired9[0] and sim9._step_cursor == 0 and len(sim9._measurement_columns) == 0

    sim10, _ = _mk_pair("H 0 1\nM 0 1", 2, seed=80)
    sim10.run()
    fired10 = [False]

    def prof_clr_run(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired10[0]
            and event == "return"
            and frame.f_code.co_name == "_check_op_qubit_bounds"
        ):
            fired10[0] = True
            sim10.clear()
            sim10.run()

    sys.setprofile(prof_clr_run)
    try:
        with pytest.raises(ValueError, match="cleared"):
            sim10.do_conditional_clifford(
                stim.CircuitInstruction("CX", [0, 1, 1, 0]), np.zeros((2, 2), bool)
            )
    finally:
        sys.setprofile(None)
    assert fired10[0] and sim10._step_cursor == 2 and len(sim10._compiled_ref.steps) == 2

    sim11, _ = _mk_pair("H 0 1\nM 0 1", 2, seed=81)
    h01 = stim.CircuitInstruction("H", [0, 1])
    fired11 = [False]

    def prof_nsm_clr_reenter(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired11[0]
            and event == "return"
            and frame.f_code.co_name == "_next_step_meta"
        ):
            fired11[0] = True
            sim11.clear()
            sim11.do_conditional_clifford(h01, np.ones((2, 2), bool))

    sys.setprofile(prof_nsm_clr_reenter)
    try:
        with pytest.raises(ValueError, match="cleared"):
            sim11.do_conditional_clifford(h01, np.zeros((2, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired11[0] and sim11._step_cursor == 1


def _mk_sim_r4(
    c_txt: str,
    batch: int = 2,
    seed: int = 11,
    *,
    use_cpp: bool = True,
    ref: CompiledReferenceCircuit | None = None,
) -> CosetsideSimulator:
    c = stim.Circuit(c_txt)
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=batch)
    sim = CosetsideSimulator(
        c, compiled_op_handler=h, batch_size=batch, seed=seed, compiled_ref=ref
    )
    sim._use_cpp = use_cpp
    return sim


def test_r5_sa_r4_med2_seq_eval_group_return_reentry_rejected() -> None:
    """R4 probe_2a & probe_2b: clear() at _eval_group_condition return raises cleanly."""
    c_txt = "H 0 1 2\nM 0 1 2"
    cx = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    sim_py = _mk_sim_r4(c_txt, use_cpp=False)
    sim_py.run()
    calls = [0]

    def prof_g1(frame: Any, event: str, _arg: Any) -> None:
        if event == "return" and frame.f_code.co_name == "_eval_group_condition":
            calls[0] += 1
            if calls[0] == 2:
                sim_py.clear()

    sys.setprofile(prof_g1)
    try:
        with pytest.raises(RuntimeError, match="forbidden inside condition_group_fn"):
            sim_py.do_conditional_clifford(cx, condition_group_fn=lambda _g: np.zeros(2, bool))
    finally:
        sys.setprofile(None)
    assert calls[0] == 2 and sim_py._step_cursor == 3 and not sim_py._in_group_callback

    sim_cpp = _mk_sim_r4(c_txt)
    sim_cpp.run()
    st_meta = sim_cpp._compiled_ref.compile_instruction(cx)[0]
    pw = sim_cpp._current_packed_words()
    fired = [False]

    def prof_g0(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_eval_group_condition":
            fired[0] = True
            sim_cpp.clear()

    sys.setprofile(prof_g0)
    try:
        with pytest.raises(RuntimeError, match="forbidden inside condition_group_fn"):
            sim_cpp._dcc_sequential_cpp(
                st_meta, pw, None, lambda _g: np.zeros(2, bool)  # type: ignore[arg-type]
            )
    finally:
        sys.setprofile(None)
    assert fired[0] and sim_cpp._step_cursor == 2 and not sim_cpp._in_group_callback


def test_r5_sa_r4_med1_validate_and_step_return_reentry() -> None:
    """R4 probe_3a & probe_3b: re-entry at _dcc_validate_and_step return raises cleanly."""
    c_txt = "H 0 1\nX 0\nTICK\nM 0 1"
    h2 = stim.CircuitInstruction("H", [0, 1])
    x0 = stim.CircuitInstruction("X", [0])
    m_x = np.array([[False, True]], bool)

    sim_a = _mk_sim_r4(c_txt)
    fired_a = [False]

    def prof_x0(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired_a[0]
            and event == "return"
            and frame.f_code.co_name == "_dcc_validate_and_step"
        ):
            fired_a[0] = True
            sim_a.do_conditional_clifford(x0, m_x)

    sys.setprofile(prof_x0)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim_a.do_conditional_clifford(h2, np.zeros((2, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired_a[0]

    sim_b = _mk_sim_r4(c_txt)
    fired_b = [False]

    def prof_clr(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired_b[0]
            and event == "return"
            and frame.f_code.co_name == "_dcc_validate_and_step"
        ):
            fired_b[0] = True
            sim_b.clear()

    sys.setprofile(prof_clr)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim_b.do_conditional_clifford(h2, np.zeros((2, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired_b[0] and sim_b._step_cursor == 0


def test_r5_sa_r4_med3_and_low1_witnesses() -> None:
    """R4 r4_p6 (D-34 entry conversion skips __rand__), r4_p7, and r4_p9."""
    c_str = "X_ERROR(1) 0\nM 0\nCX rec[-1] 1 0 2\nH 2\nM 1 2"
    c = stim.Circuit(c_str)
    ref = _mk_sim_r4(c_str, seed=110)
    ref._do_bare_instruction(c[0])
    ref._do_bare_instruction(c[1])
    ref.do_conditional_clifford(c[2], np.array([[False, False], [True, True]], bool))
    ref.do_conditional_clifford(c[3], np.ones((1, 2), bool))
    ref._do_bare_instruction(c[4])

    sim6 = _mk_sim_r4(c_str, seed=110)
    sim6._do_bare_instruction(c[0])
    sim6._do_bare_instruction(c[1])
    rand_fired = [False]

    class _RandTrig:
        def __init__(self, v: bool) -> None:
            self.v = v

        def __bool__(self) -> bool:
            return self.v

        def __rand__(self, other: object) -> bool:
            rand_fired[0] = True
            sim6.do_conditional_clifford(c[3], np.ones((1, 2), bool))
            return bool(other) and self.v

    mask6 = np.empty((2, 2), dtype=object)
    mask6[0, 0], mask6[0, 1] = _RandTrig(False), False
    mask6[1, 0], mask6[1, 1] = True, True
    sim6.do_conditional_clifford(c[2], mask6)  # type: ignore[arg-type]
    assert not rand_fired[0]
    sim6.do_conditional_clifford(c[3], np.ones((1, 2), bool))
    sim6._do_bare_instruction(c[4])
    xs6, zs6, _, _, _ = sim6._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    xsr, zsr, _, _, _ = ref._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    assert np.array_equal(xs6, xsr) and np.array_equal(zs6, zsr)

    c7 = stim.Circuit("H 0 1\nX 0\nM 0 1")
    sim7 = _mk_sim_r4("H 0 1\nX 0\nM 0 1", seed=111)
    fired7 = [False]

    def prof7(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired7[0]
            and event == "return"
            and frame.f_code.co_name == "_dcc_collect_mixed_record"
        ):
            fired7[0] = True
            sim7.do_conditional_clifford(c7[1], np.array([[False, True]], bool))

    sys.setprofile(prof7)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim7.do_conditional_clifford(c7[0], np.zeros((2, 2), bool))
    finally:
        sys.setprofile(None)
    assert fired7[0]

    sim9 = _mk_sim_r4("H 0\nX 0\nM 0", seed=113)
    sim9.interactive_do(stim.CircuitInstruction("H", [0]))
    fired9 = [False]

    def prof9(frame: Any, event: str, _arg: Any) -> None:
        if not fired9[0] and event == "return" and frame.f_code.co_name == "_replays":
            fired9[0] = True
            sim9.clear()

    sys.setprofile(prof9)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim9.do_conditional_clifford(stim.CircuitInstruction("X", [0]), None)
    finally:
        sys.setprofile(None)
    assert fired9[0] and sim9._step_cursor == 0 and sim9._clear_epoch == 1


def test_r5_sa_r4_ts1_kill_m38_identity_group_fn_reentry() -> None:
    """Kills m38_r4_run_unitary_I_skip_cb on I and II with condition_group_fn."""
    for op_name, tgts in (("I", [0, 1]), ("II", [0, 1, 0, 1])):
        sim = _mk_sim_r4("H 0 1\nM 0 1")
        sim.run()
        op = stim.CircuitInstruction(op_name, tgts)

        def gfn(g: int, s: CosetsideSimulator = sim) -> np.ndarray:
            if g == 0:
                s.clear()
            return np.ones(2, dtype=bool)

        with pytest.raises(RuntimeError, match="forbidden inside condition_group_fn"):
            sim.do_conditional_clifford(op, condition_group_fn=gfn)
        assert not sim._in_group_callback
        assert sim._step_cursor == 2 and len(sim._compiled_ref.steps) == 2


def test_r5_sa_r4_ts1_kill_m34_shared_ref_steps_len_growth() -> None:
    """Kills m34_r4_sync_step_skip_steps_len_check via shared CompiledReferenceCircuit."""
    c_txt = "H 0\nM 0"
    shared_ref = CompiledReferenceCircuit(stim.Circuit(c_txt))
    sim1 = _mk_sim_r4(c_txt, ref=shared_ref)
    sim2 = _mk_sim_r4(c_txt, ref=shared_ref)
    sim2.run()
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if (
            not fired[0]
            and event == "return"
            and frame.f_code.co_name == "_dcc_collect_mixed_record"
        ):
            fired[0] = True
            sim2.do_conditional_clifford(
                stim.CircuitInstruction("X", [0]), np.ones((1, 2), dtype=bool)
            )

    sys.setprofile(prof)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim1.do_conditional_clifford(
                stim.CircuitInstruction("H", [0]), np.zeros((1, 2), dtype=bool)
            )
    finally:
        sys.setprofile(None)
    assert fired[0] and len(shared_ref.steps) == 3 and sim1._step_cursor == 0


def test_r5_sa_r4_ts1_kill_m36_validate_clear_at_entry_guard() -> None:
    """Kills m36_r4_validate_drop_clear_at_entry_guard under Option B."""
    h2 = stim.CircuitInstruction("H", [0, 1])
    s0 = stim.CircuitInstruction("S", [0])
    m2, m1 = np.zeros((2, 2), dtype=bool), np.zeros((1, 2), dtype=bool)
    sim = _mk_sim_r4("H 0 1\nS 0\nM 0 1")
    phase = [0]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        back = frame.f_back.f_code.co_name if frame.f_back is not None else ""
        if (
            phase[0] == 0
            and event == "return"
            and frame.f_code.co_name == "_current_packed_words"
            and back == "_dcc_validate_and_step"
        ):
            phase[0] = 1
            sim.do_conditional_clifford(h2, m2)
        elif (
            phase[0] == 1
            and event == "return"
            and frame.f_code.co_name == "_engine_dims"
            and back == "_dcc_validate_and_step"
        ):
            phase[0] = 2
            sim.clear()
            sim.do_conditional_clifford(h2, m2)
            sim.do_conditional_clifford(s0, m1)

    sys.setprofile(prof)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            sim.do_conditional_clifford(h2, m2)
    finally:
        sys.setprofile(None)
    assert phase[0] == 2 and sim._step_cursor == 2


def test_r6_d34_numeric_masks_match_bool_and_desired_bits_low_bit() -> None:
    """A-6 & D-34: numeric/object masks match bool on all paths; desired_bits uses & 1."""
    for use_cpp in (True, False):
        for pauli in ("X", "Y", "Z"):
            p_lit = cast(Literal["X", "Y", "Z"], pauli)
            r0_b = np.array([[True, False, True, False], [False, True, False, False]], bool)
            r1_b = np.array([[False, True, False, True], [True, False, False, False]], bool)
            ref, sim_num = _mk_pair("H 0 1\nCX 0 1\nM 0 1", 4, seed=91)
            ref._use_cpp, sim_num._use_cpp = use_cpp, use_cpp
            h_op = stim.CircuitInstruction("H", [0, 1])
            ref.do_conditional_clifford(h_op, np.zeros((2, 4), bool))
            sim_num.do_conditional_clifford(h_op, np.zeros((2, 4), np.int64))
            r0_f = np.where(r0_b, 0.5, 0.0)
            r1_i = np.where(r1_b, 256, 0)
            ref.do_unmatched_noisy_reset_z(r0_b, r1_b, p_lit)
            sim_num.do_unmatched_noisy_reset_z(r0_f, r1_i, p_lit)  # type: ignore[arg-type]
            ref._sync_cpp_to_py()
            _assert_exact_parity(sim_num, ref, 2, f"reset_{pauli}_cpp{use_cpp}")

        s_ref, s_low = _mk_pair("H 0 1\nM 0 1", 4, seed=92)
        s_ref._use_cpp, s_low._use_cpp = use_cpp, use_cpp
        s_ref.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), np.zeros((2, 4), bool))
        s_low.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), np.zeros((2, 4), bool))
        flip_b = np.array([True, True, False, True], bool)
        flip_num = np.array([0.25, 256.0, 0.0, -3.0], float)
        db_ref = np.array([0, 1, 0, 1], np.uint8)
        db_hi = np.array([2, 3, 4, -1], np.int64)
        s_ref.postselect_z_batch(0, flip_b, db_ref, "Z")
        s_low.postselect_z_batch(0, flip_num, db_hi, "Z")  # type: ignore[arg-type]
        s_ref._sync_cpp_to_py()
        _assert_exact_parity(s_low, s_ref, 2, f"postselect_lowbit_cpp{use_cpp}")


def test_r6_entry_normalization_rejects_bad_operands_atomically() -> None:
    """R1.1: rejected types/shapes/subclasses raise before reading or mutating any state."""
    class _EvilInt(int):
        def __int__(self) -> int:
            raise AssertionError("EvilInt.__int__ must not run")

    class _EvilStr(str):
        def __str__(self) -> str:
            raise AssertionError("EvilStr.__str__ must not run")

    sim, _ = _mk_pair("H 0 1\nM 0 1", 4, seed=93)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1]), np.zeros((2, 4), bool))
    sim.record_direct_measurements(np.array([[1, 0, 2, 0]], dtype=np.int64))
    assert sim.measurement_column(_EvilInt(0)).tolist() == [True, False, True, False]
    assert sim.peek_pauli_batch([0], cast(Any, _EvilStr("Z"))).shape == (1, 4)

    before = _snap_coset_state(sim)
    cols_before = [c.copy() for c in sim._measurement_columns]
    with pytest.raises(TypeError):
        sim.do_unmatched_noisy_reset_z(
            np.array([["True"] * 4, ["False"] * 4]), np.zeros((2, 4), bool), "Z"  # type: ignore
        )
    with pytest.raises(IndexError):
        sim.do_unmatched_noisy_reset_z(
            np.ones((2, 4), bool), np.zeros((2, 4), bool), "X", order=[0, 99]
        )
    with pytest.raises(TypeError):
        sim.postselect_z_batch(0, np.ones(4, bool), np.array([0.5, 0.0, 1.0, 0.0]))  # type: ignore
    with pytest.raises(ValueError):
        sim.apply_pauli("X", np.zeros((2, 4), bool), float("nan"))
    with pytest.raises(ValueError):
        sim.apply_pauli("X", np.zeros((1, 4), bool), 0.0)
    with pytest.raises(ValueError):
        sim.record_direct_measurements(np.array([True, False, True, False]))  # type: ignore
    with pytest.raises(ValueError):
        sim.overwrite_measurement_column(0, np.array([True, False]))
    with pytest.raises(NotImplementedError):
        sim.interactive_do("H 0")  # type: ignore[arg-type]
    assert not sim._used_interactively and sim._unrolled_ops is None
    _assert_coset_state_same(sim, before)
    assert all(np.array_equal(a, b) for a, b in zip(sim._measurement_columns, cols_before))


def _mk_b1_pair(
    text: str, batch: int, seed: int
) -> tuple[CosetsideSimulator, CosetsideSimulator]:
    circ = stim.Circuit(text)
    h1 = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch)
    h2 = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch)
    s_cpp = CosetsideSimulator(
        circ, compiled_op_handler=h1, batch_size=batch, seed=seed,
        record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    s_py = CosetsideSimulator(
        circ, compiled_op_handler=h2, batch_size=batch, seed=seed,
        record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    s_py._use_cpp = False
    return s_cpp, s_py


@pytest.mark.parametrize(
    ("n_q", "d_target", "batch"),
    [(5, 5, 1), (64, 64, 7), (70, 68, 64), (140, 80, 256)],
)
def test_b1_public_postselect_and_reset_grid_parity(
    n_q: int, d_target: int, batch: int
) -> None:
    """Stage B.1: public postselect & X/Y/ordered resets match Python across full §5 grid."""
    rng = np.random.default_rng(7000 + n_q + batch)
    qs_str = " ".join(map(str, range(n_q)))
    c_txt = (
        "I[LEAKAGE_TRANSITION_1: (0.35, U-->2)] 0 1\nTICK\n"
        f"I[LEAKAGE_TRANSITION_1: (0.25, 1-->2)] 0 1\nTICK\nM 0 1\nI {qs_str}"
    )
    s_cpp, s_py = _mk_b1_pair(c_txt, batch, seed=n_q * 19 + batch)
    s_cpp.run()
    s_py.run()
    _seed_high_rank(s_cpp, s_py, n_q, d_target, batch, rng)
    s_cpp._sync_cpp_to_py()
    assert max(st.d for st in s_cpp._active_shots.values()) >= d_target
    if n_q >= 70:
        assert max(st.d for st in s_cpp._active_shots.values()) > 64

    rng_before = s_cpp._shot_tiebreak_rng.copy()
    for pauli in ("X", "Y", "Z"):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        for q_sel in (0, n_q - 1):
            flip = rng.random(batch) < 0.5
            if batch == 1:
                flip[0] = True
            des = rng.integers(0, 2, size=batch, dtype=np.uint8)
            s_cpp.postselect_z_batch(q_sel, flip, des, p_lit)
            s_py.postselect_z_batch(q_sel, flip, des, p_lit)
            assert not np.any(flip) or s_cpp._cpp_dirty
            _assert_exact_parity(s_cpp, s_py, n_q, f"b1_post_{pauli}_q{q_sel}")
    assert np.array_equal(s_cpp._shot_tiebreak_rng, rng_before)

    r0 = np.zeros((n_q, batch), dtype=bool)
    r1 = np.zeros((n_q, batch), dtype=bool)
    r0[0, :] = True
    r1[0, ::2] = True
    r1[n_q - 1, ::3] = True
    if n_q > 64:
        r0[64, 1::2] = True
    ord_sub = [n_q - 1, 0, min(64, n_q - 1)]
    for pauli, order in (
        ("Z", None), ("Z", ord_sub), ("Z", [0, 0]), ("X", None), ("Y", ord_sub), ("X", [])
    ):
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        s_cpp.do_unmatched_noisy_reset_z(r0, r1, p_lit, order=order)
        s_py.do_unmatched_noisy_reset_z(r0, r1, p_lit, order=order)
        _assert_exact_parity(s_cpp, s_py, n_q, f"b1_rst_{pauli}_{order}")

    dm = rng.integers(0, 2, size=(2, batch), dtype=bool)
    s_cpp.record_direct_measurements(dm)
    s_py.record_direct_measurements(dm)
    _assert_exact_parity(s_cpp, s_py, n_q, f"b1_final_n{n_q}_B{batch}")
    assert np.array_equal(
        s_cpp.current_tableau_measurement_record(), s_py.current_tableau_measurement_record()
    )
    assert np.array_equal(
        s_cpp.get_final_measurement_records(), s_py.get_final_measurement_records()
    )
    assert s_cpp.get_leakage_events() == s_py.get_leakage_events()
    assert [a.tolist() for a in s_cpp.get_unleaked_to_leaked_records()] == [
        a.tolist() for a in s_py.get_unleaked_to_leaked_records()
    ]


def test_b1_routing_targets_v3_kernels_and_preserves_sync_and_python_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify B.1 kernel routing, no-op early returns, and sync/Python-fallback modes."""
    s_cpp, s_py = _mk_pair("H 0 1 2\nM 0 1 2", 4, seed=404)
    h_op = stim.CircuitInstruction("H", [0, 1, 2])
    s_cpp.do_conditional_clifford(h_op, np.zeros((3, 4), bool))
    s_py.do_conditional_clifford(h_op, np.zeros((3, 4), bool))

    calls: list[str] = []
    real_lib = s_cpp._cpp_lib
    assert real_lib is not None

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name not in (
                "engine_postselect_batch",
                "engine_project_pauli",
                "engine_batch_unmatched_noisy_reset_z",
            ):
                return fn
            return lambda *a: (calls.append(name), fn(*a))[1]

    monkeypatch.setattr(s_cpp, "_cpp_lib", cast(ctypes.CDLL, _Spy()))
    monkeypatch.setattr(
        s_cpp,
        "_project_and_retire",
        lambda *_a, **_kw: pytest.fail("Python _project_and_retire called in C++ mode"),
    )

    s_cpp._sync_cpp_to_py()
    assert not s_cpp._cpp_dirty
    s_cpp.postselect_z_batch(0, np.zeros(4, bool), np.zeros(4, np.uint8), "X")
    s_cpp.do_unmatched_noisy_reset_z(np.ones((3, 4), bool), np.zeros((3, 4), bool), "Z", order=[])
    assert calls == [] and not s_cpp._cpp_dirty

    flip = np.array([True, False, True, False], bool)
    des = np.array([0, 1, 1, 0], np.uint8)
    s_cpp.postselect_z_batch(0, flip, des, "Z")
    assert calls[-1] == "engine_postselect_batch" and s_cpp._cpp_dirty
    s_cpp._sync_cpp_to_py()
    r0 = np.array([[True, False, True, False], [False, True, False, False], [False] * 4], bool)
    r1 = np.zeros((3, 4), bool)
    s_cpp.do_unmatched_noisy_reset_z(r0, r1, "Z")
    assert calls[-1] == "engine_batch_unmatched_noisy_reset_z" and s_cpp._cpp_dirty
    s_cpp._sync_cpp_to_py()
    s_cpp.do_unmatched_noisy_reset_z(r0, r1, "Z", order=[1, 0])
    assert calls[-1] == "engine_project_pauli" and s_cpp._cpp_dirty
    s_cpp._sync_cpp_to_py()
    s_cpp.do_unmatched_noisy_reset_z(r0, r1, "X")
    assert calls[-1] == "engine_project_pauli" and s_cpp._cpp_dirty
    s_cpp._sync_cpp_to_py()
    s_cpp.do_unmatched_noisy_reset_z(r0, r1, "Y", order=[0])
    assert calls[-1] == "engine_project_pauli" and s_cpp._cpp_dirty

    # Python fallback (_use_cpp=False) and sync_tableside_rng=True keep working via Python path.
    s_py.postselect_z_batch(0, flip, des, "Z")
    s_py.do_unmatched_noisy_reset_z(r0, r1, "Z")
    s_py.do_unmatched_noisy_reset_z(r0, r1, "Z", order=[1, 0])
    s_py.do_unmatched_noisy_reset_z(r0, r1, "X")
    s_py.do_unmatched_noisy_reset_z(r0, r1, "Y", order=[0])
    _assert_exact_parity(s_cpp, s_py, 3, "b1_spy_parity")

    circ = stim.Circuit("H 0 1 2\nM 0 1 2")
    h_sync = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=4)
    s_sync = CosetsideSimulator(
        circ, compiled_op_handler=h_sync, batch_size=4, seed=404, sync_tableside_rng=True
    )
    assert not s_sync._use_cpp and s_sync._cpp_eng is None
    s_sync.do_conditional_clifford(h_op, np.zeros((3, 4), bool))
    s_sync.postselect_z_batch(0, flip, des, "X")
    s_sync.do_unmatched_noisy_reset_z(r0, r1, "Y", order=[1, 0])
    assert s_sync.peek_pauli_batch([0, 1], "Z").shape == (2, 4)


def _b2_run_rep_setup(
    s_cpp: CosetsideSimulator,
    s_py: CosetsideSimulator,
    ops: list[Any],
    n_q: int,
    d_target: int,
    batch: int,
    rng: np.random.Generator,
    rep: int,
) -> int:
    for idx in range(5):
        inst = ops[idx]
        if idx == 1:
            s_cpp._do_bare_instruction(inst)
            s_py._do_bare_instruction(inst)
        elif idx == 4:
            mh = rng.random((n_q, batch)) < 0.85
            mh[:d_target, 0] = False
            s_cpp.do_conditional_clifford(inst, mh)
            s_py.do_conditional_clifford(inst, mh)
        else:
            s_cpp._do(inst)
            s_py._do(inst)
    s_cpp._sync_cpp_to_py()
    max_d = max(st.d for st in s_cpp._active_shots.values())
    for tag, idx in (("cx_ov", 5), ("cz_ov", 6), ("h_rep", 7)):
        m = rng.random((len(ops[idx].target_groups()), batch)) < 0.45
        if rep == 1 and tag == "cz_ov":
            m[:] = True
        s_cpp.do_conditional_clifford(ops[idx], m)
        s_py.do_conditional_clifford(ops[idx], m)
        _assert_exact_parity(s_cpp, s_py, n_q, f"b2_{rep}_{tag}")
    return max_d


def _b2_run_rep(
    s_cpp: CosetsideSimulator,
    s_py: CosetsideSimulator,
    ops: list[Any],
    n_q: int,
    d_target: int,
    batch: int,
    rng: np.random.Generator,
    rep: int,
) -> int:
    max_d = _b2_run_rep_setup(s_cpp, s_py, ops, n_q, d_target, batch, rng, rep)
    for tag, idx in (("I_cb", 8), ("II_cb", 9)):
        rows = [rng.random(batch) < 0.5 for _ in range(len(ops[idx].target_groups()))]

        def make_i(sim: CosetsideSimulator, rs: list[np.ndarray] = rows) -> Any:
            def gfn(g: int) -> np.ndarray:
                pk = sim.peek_pauli_batch([0, n_q - 1], "Z")
                mc = sim.measurement_column(-1)
                _ = len(sim.active_shots)
                if g == 0:
                    fl = np.zeros(batch, dtype=bool)
                    fl[::2] = True
                    sim.postselect_z_batch(0, fl, np.zeros(batch, dtype=np.uint8), "Z")
                return rs[g] ^ (pk[0] < 0) ^ mc

            return gfn

        s_cpp.do_conditional_clifford(ops[idx], condition_group_fn=make_i(s_cpp))
        s_py.do_conditional_clifford(ops[idx], condition_group_fn=make_i(s_py))
        _assert_exact_parity(s_cpp, s_py, n_q, f"b2_{rep}_{tag}")
    m_mix = rng.random((len(ops[10].target_groups()), batch)) < 0.5
    s_cpp.do_conditional_clifford(ops[10], m_mix)
    s_py.do_conditional_clifford(ops[10], m_mix)
    _assert_exact_parity(s_cpp, s_py, n_q, f"b2_{rep}_mix_mask")
    for idx in (11, 13):
        s_cpp._do_bare_instruction(ops[idx])
        s_py._do_bare_instruction(ops[idx])
        if idx == 11:
            rcb = [rng.random(batch) < 0.5 for _ in range(2)]
            s_cpp.do_conditional_clifford(ops[12], condition_group_fn=lambda g, r=rcb: r[g])
            s_py.do_conditional_clifford(ops[12], condition_group_fn=lambda g, r=rcb: r[g])
    ng_cx = len(ops[14].target_groups())
    rcx = [np.ones(batch, dtype=bool)] + [rng.random(batch) < 0.5 for _ in range(ng_cx - 1)]

    def make_cx(sim: CosetsideSimulator) -> Any:
        seen: list[int] = []

        def gfn(g: int) -> np.ndarray:
            seen.append(g)
            pk = sim.peek_pauli_batch([0, 1, n_q - 1], "Y")
            mc = sim.measurement_column(-1)
            _ = len(sim.active_shots)
            if g == 0:
                fl = np.zeros(batch, dtype=bool)
                fl[0] = True
                sim.postselect_z_batch(1, fl, np.ones(batch, dtype=np.uint8), "X")
                return rcx[0]
            return rcx[g] ^ (pk[2] < 0) ^ mc

        cast(Any, gfn).seen = seen
        return gfn

    gc, gp = make_cx(s_cpp), make_cx(s_py)
    s_cpp.do_conditional_clifford(ops[14], condition_group_fn=gc)
    s_py.do_conditional_clifford(ops[14], condition_group_fn=gp)
    assert gc.seen == list(range(ng_cx)) and gp.seen == list(range(ng_cx))
    _assert_exact_parity(s_cpp, s_py, n_q, f"b2_{rep}_cx_cb")
    return max_d


@pytest.mark.parametrize(
    ("n_q", "d_target", "batch"),
    [(5, 5, 1), (64, 64, 7), (70, 68, 64), (140, 80, 256)],
)
def test_b2_public_sequential_and_callback_grid_parity(
    n_q: int, d_target: int, batch: int
) -> None:
    """Stage B.2: public non-disjoint & callback Cliffords match Python across §5 grid."""
    rng = np.random.default_rng(8200 + n_q + batch)
    qs_str = " ".join(map(str, range(n_q)))
    c_txt = "\n".join([
        "I[LEAKAGE_TRANSITION_1: (0.35, U-->2)] 0 1", "TICK", "X_ERROR(0.5) 4", "M 4",
        f"H {qs_str}", "CX 0 1 1 2 2 3 3 4", "CZ 0 1 1 2 2 3",
        f"H 0 0 {n_q - 1} {n_q - 1}", "I 0 1", "II 0 1 2 3",
        "CX rec[-1] 0 1 2 2 1", "TICK", "CX rec[-1] 0 1 2 2 1", "TICK",
        f"CX 0 1 1 0 {n_q - 2} {n_q - 1} {n_q - 1} {n_q - 2}",
    ])
    ops = list(stim.Circuit(c_txt))
    s_cpp, s_py = _mk_b1_pair(c_txt, batch, seed=n_q * 23 + batch)
    max_d = 0
    for rep in range(2):
        max_d = max(max_d, _b2_run_rep(s_cpp, s_py, ops, n_q, d_target, batch, rng, rep))
        dm = rng.integers(0, 2, size=(2, batch), dtype=bool)
        s_cpp.record_direct_measurements(dm)
        s_py.record_direct_measurements(dm)
        s_cpp.finish_interactive_run()
        s_py.finish_interactive_run()
        _assert_exact_parity(s_cpp, s_py, n_q, f"b2_{rep}_final")
        assert np.array_equal(
            s_cpp.get_final_measurement_records(), s_py.get_final_measurement_records()
        )
        assert s_cpp.get_leakage_events() == s_py.get_leakage_events()
        assert [a.tolist() for a in s_cpp.get_unleaked_to_leaked_records()] == [
            a.tolist() for a in s_py.get_unleaked_to_leaked_records()
        ]
        if rep == 0:
            s_cpp.clear()
            s_py.clear()
    if n_q >= 70:
        assert max_d > 64


def test_b2_routing_targets_inject_group_v3_and_preserves_sync_and_python_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify B.2 routing to engine_inject_group_v3, callback invariants, v3 disjoint, sync."""
    c_txt = "H 0 1 2\nTICK\nCX 0 1 1 2\nTICK\nCX 0 1\nTICK\nI 0 1\nM 0 1 2"
    s_cpp, s_py = _mk_pair(c_txt, 4, seed=505)
    h_op = stim.CircuitInstruction("H", [0, 1, 2])
    s_cpp.do_conditional_clifford(h_op, np.zeros((3, 4), bool))
    s_py.do_conditional_clifford(h_op, np.zeros((3, 4), bool))
    s_cpp._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    s_py._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    calls: list[tuple[str, bool, bool]] = []
    real_lib = s_cpp._cpp_lib
    assert real_lib is not None
    tracked = {
        "engine_inject_group_v3", "engine_register_unitary_step",
        "engine_register_unitary_group_evict", "engine_batch_inject_general_clifford_v3",
    }

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name not in tracked:
                return fn
            return lambda *a: (
                calls.append((name, s_cpp._cpp_dirty, s_cpp._in_group_callback)), fn(*a)
            )[1]

    monkeypatch.setattr(s_cpp, "_cpp_lib", cast(ctypes.CDLL, _Spy()))
    monkeypatch.setattr(
        s_cpp, "_dcc_sequential_py",
        lambda *_a, **_kw: pytest.fail("_dcc_sequential_py called in C++ mode"),
    )
    cx_ov = stim.CircuitInstruction("CX", [0, 1, 1, 2])
    seen_g: list[int] = []

    def gfn(g: int) -> np.ndarray:
        seen_g.append(g)
        assert s_cpp._in_group_callback
        _ = s_cpp.peek_pauli_batch([0, 1], "Z")
        return np.ones(4, dtype=bool) if g == 0 else np.array([True, False, True, False], bool)

    s_cpp._sync_cpp_to_py()
    assert not s_cpp._cpp_dirty
    s_cpp.do_conditional_clifford(cx_ov, condition_group_fn=gfn)
    assert seen_g == [0, 1] and not s_cpp._in_group_callback
    assert calls == [
        ("engine_register_unitary_step", False, True),
        ("engine_register_unitary_group_evict", False, True),
        ("engine_inject_group_v3", True, True),
    ]
    s_py.do_conditional_clifford(cx_ov, condition_group_fn=lambda g: (
        np.ones(4, dtype=bool) if g == 0 else np.array([True, False, True, False], bool)
    ))
    _assert_exact_parity(s_cpp, s_py, 3, "b2_spy_cb")
    calls.clear()
    s_cpp._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    s_cpp.do_conditional_clifford(stim.CircuitInstruction("CX", [0, 1]), np.zeros((1, 4), bool))
    assert calls == [
        ("engine_register_unitary_step", False, False),
        ("engine_batch_inject_general_clifford_v3", True, False),
    ]
    calls.clear()
    s_cpp._do_bare_instruction(stim.CircuitInstruction("TICK", []))
    s_cpp.do_conditional_clifford(
        stim.CircuitInstruction("I", [0, 1]), condition_group_fn=lambda _g: np.zeros(4, bool)
    )
    assert calls == [] and not s_cpp._in_group_callback
    circ = stim.Circuit(c_txt)
    h_sync = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=4)
    s_sync = CosetsideSimulator(
        circ, compiled_op_handler=h_sync, batch_size=4, seed=505, sync_tableside_rng=True
    )
    s_sync.run()
    s_sync.do_conditional_clifford(cx_ov, np.zeros((2, 4), bool))
    assert s_sync.peek_pauli_batch([0, 1], "Z").shape == (2, 4)


def test_stage_c_a2_single_compile_past_end_and_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-2: past-end DCC compiles once; hook invalidation forces safe recompile."""
    c_det = "R 0 1 2\nH 0 1 2\nH 0 1 2\nM 0 1 2"
    s_cpp, s_py = _mk_pair(c_det, 4, seed=601)
    s_cpp.run()
    s_py.run()
    calls = [0]
    orig_compile = s_cpp._compiled_ref.compile_instruction

    def count_compile(op: stim.CircuitInstruction) -> Any:
        calls[0] += 1
        return orig_compile(op)

    monkeypatch.setattr(s_cpp._compiled_ref, "compile_instruction", count_compile)
    cx_op = stim.CircuitInstruction("CX", [0, 1])
    m_cx = np.array([[True, False, True, False]], dtype=bool)
    s_cpp.do_conditional_clifford(cx_op, m_cx)
    assert calls[0] == 1
    s_py.do_conditional_clifford(cx_op, m_cx)
    _assert_exact_parity(s_cpp, s_py, 3, "a2_past_end_single_compile")

    # Re-entrant cursor/step advance between _check_dcc_mask and _dcc_validate_and_step
    sim_r, _ = _mk_pair(c_det, 4, seed=602)
    sim_r.run()
    r_calls = [0]
    orig_r = sim_r._compiled_ref.compile_instruction
    monkeypatch.setattr(
        sim_r._compiled_ref,
        "compile_instruction",
        lambda op: (r_calls.__setitem__(0, r_calls[0] + 1), orig_r(op))[1],
    )
    fired = [False]

    def prof_adv(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_check_dcc_mask":
            fired[0] = True
            sim_r.do_conditional_clifford(
                stim.CircuitInstruction("H", [2]), np.ones((1, 4), dtype=bool)
            )

    sys.setprofile(prof_adv)
    try:
        sim_r.do_conditional_clifford(cx_op, m_cx)
    finally:
        sys.setprofile(None)
    assert fired[0] and r_calls[0] == 4
    ref_r, _ = _mk_pair(c_det, 4, seed=602)
    ref_r.run()
    ref_r.do_conditional_clifford(stim.CircuitInstruction("H", [2]), np.ones((1, 4), dtype=bool))
    ref_r.do_conditional_clifford(cx_op, m_cx)
    _assert_exact_parity(sim_r, ref_r, 3, "a2_reentrant_invalidate")

    # Epoch invalidation with identical cursor/steps_len (clear + run between checks)
    sim_c, _ = _mk_pair(c_det, 4, seed=603)
    sim_c.run()
    c_calls = [0]
    orig_c = sim_c._compiled_ref.compile_instruction
    monkeypatch.setattr(
        sim_c._compiled_ref,
        "compile_instruction",
        lambda op: (c_calls.__setitem__(0, c_calls[0] + 1), orig_c(op))[1],
    )
    fired_c = [False]

    def prof_clr(frame: Any, event: str, _arg: Any) -> None:
        if not fired_c[0] and event == "return" and frame.f_code.co_name == "_check_dcc_mask":
            fired_c[0] = True
            sim_c.clear()
            sim_c.run()

    sys.setprofile(prof_clr)
    try:
        with pytest.raises(ValueError, match="was cleared"):
            sim_c.do_conditional_clifford(cx_op, m_cx)
    finally:
        sys.setprofile(None)
    assert fired_c[0] and c_calls[0] == 2 and sim_c._clear_epoch == 1


def test_stage_c_op_bounds_nq_fast_path_and_replay_parity() -> None:
    """Opt 3: I [99] & mismatched ops raise IndexError atomically; valid replays match Python."""
    s_cpp, s_py = _mk_pair("H 0 1\nI 0\nCX 0 1\nM 0 1", 4, seed=604)
    s_cpp.run()
    s_py.run()
    bad_i = stim.CircuitInstruction("I", [99])
    meta_bad_i, _, _ = s_cpp._compiled_ref.compile_instruction(bad_i)
    assert getattr(meta_bad_i, "_op_bounds_nq", None) == -1
    snap_end = _snap_coset_state(s_cpp)
    with pytest.raises(IndexError, match="out of range"):
        s_cpp.do_conditional_clifford(bad_i, np.zeros((1, 4), dtype=bool))
    _assert_coset_state_same(s_cpp, snap_end)

    s_cpp.clear()
    s_py.clear()
    snap_0 = _snap_coset_state(s_cpp)
    # Mismatched op at compiled cursor 0 (`H 0 1` compiled; pass `CX 0 1 0 99`)
    with pytest.raises(IndexError, match="out of range"):
        s_cpp.do_conditional_clifford(
            stim.CircuitInstruction("CX", [0, 1, 0, 99]), np.ones((2, 4), dtype=bool)
        )
    _assert_coset_state_same(s_cpp, snap_0)

    # Advance to compiled `I 0` at cursor 1, then reject mismatched `I 99` atomically
    m_h = np.array([[True, False, True, False], [False, True, True, False]], dtype=bool)
    h_op = stim.CircuitInstruction("H", [0, 1])
    s_cpp.do_conditional_clifford(h_op, m_h)
    s_py.do_conditional_clifford(h_op, m_h)
    snap_1 = _snap_coset_state(s_cpp)
    with pytest.raises(IndexError, match="out of range"):
        s_cpp.do_conditional_clifford(bad_i, np.zeros((1, 4), dtype=bool))
    _assert_coset_state_same(s_cpp, snap_1)

    # Valid replay of `I 0` + `CX 0 1` across clear matches Python path
    i_op = stim.CircuitInstruction("I", [0])
    cx_op = stim.CircuitInstruction("CX", [0, 1])
    m_1 = np.array([[True, True, False, True]], dtype=bool)
    for step_op in (i_op, cx_op):
        s_cpp.do_conditional_clifford(step_op, m_1)
        s_py.do_conditional_clifford(step_op, m_1)
    _assert_exact_parity(s_cpp, s_py, 2, "op_bounds_replay_1")
    s_cpp.clear()
    s_py.clear()
    s_cpp.do_conditional_clifford(h_op, m_h)
    s_py.do_conditional_clifford(h_op, m_h)
    s_cpp.do_conditional_clifford(i_op, m_1)
    s_py.do_conditional_clifford(i_op, m_1)
    s_cpp.do_conditional_clifford(cx_op, m_1)
    s_py.do_conditional_clifford(cx_op, m_1)
    _assert_exact_parity(s_cpp, s_py, 2, "op_bounds_replay_2")


def test_stage_c_dcc_disjoint_cpp_v3_direct_preowned_bool_false() -> None:
    """Opt 1: direct _dcc_disjoint_cpp_v3 (preowned_bool=False) owns/validates masks."""
    s_cpp, s_py = _mk_pair("H 0 1 2\nX 0 1 2\nM 0 1 2", 4, seed=605)
    op_h = stim.CircuitInstruction("H", [0, 1, 2])
    st_c = s_cpp._next_step_meta(op_h)
    st_p = s_py._next_step_meta(op_h)
    assert st_c.kind == "unitary" and st_p.kind == "unitary"

    # Non-contiguous int64 mask converted by truth value and owned (preowned_bool=False)
    raw = np.array([[0, 2, 0, 5, 9], [7, 0, 3, 0, 9], [0, 0, 4, 8, 9]], dtype=np.int64)[:, :4]
    assert not raw.flags.c_contiguous
    s_cpp._dcc_disjoint_cpp_v3(op_h, st_c, raw, 0, True)
    raw.fill(0)
    s_py._dcc_disjoint_py(op_h, st_p, np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], bool))
    _assert_exact_parity(s_cpp, s_py, 3, "direct_v3_preowned_false_int64")

    # Bad dtype (string) and bad shape raise atomically and roll back step cursor
    op_x = stim.CircuitInstruction("X", [0, 1, 2])
    for bad, exc in (
        (np.full((3, 4), "1"), TypeError),
        (np.ones((2, 4), dtype=np.uint8), ValueError),
    ):
        cur_before = s_cpp._step_cursor
        st_x = s_cpp._next_step_meta(op_x)
        assert st_x.kind == "unitary"
        with pytest.raises(exc):
            s_cpp._dcc_disjoint_cpp_v3(op_x, st_x, bad, cur_before, True)
        assert s_cpp._step_cursor == cur_before

