"""T11, multi-branch GF(2) absorption, cap/clear, drift, counter & leak tests for `BranchStore`.

Covers:
1. Explicit multi-branch GF(2) measurement absorption tests:
   - Case A: individual generators `ev == -1` on `S_1`, product `ev == +1` (`|00>, |11>`, measure `X_0`).
   - Case B: individual generators `ev == 0` on `S_1`, product `ev == +1` (`|00>, |Phi+>`, measure `X_0`).
   - Case C: exact `kappa` sign tracking (`|10>, |01>` common `-Z_0 Z_1`) + negative control (`|00>, |01>`
     measuring `X_1` MUST fork).
   - Case D: absorption with commuting pending turns (rejects anticommuting stabilizer, accepts commuting).
   - Case E: `POST` on absorbed inner outcome `1` with `F ^= Q_bar`.
   - Case F: 200+ random multi-branch absorption stress trials vs dense (`< 1e-12`).
2. T11 bounded store-size & non-forking absorption tests:
   - `|+>^n` (`n = 10` and multi-word `n = 70`) over 500 shots.
   - Noiseless rotated surface code (`rotated_memory_z`, `rotated_memory_x`, `d=3, r=3, p=0`) over 200 shots.
   - Noiseless `xmem_t(k)` (`k = 0..4`, `d=3, r=3, p=0`) over 200 shots.
3. Store cap & `clear()` semantics, F5 cold/warm/small-cap determinism, float drift `-3` on `step_meas`
   and `step_peek`, impossible outcome `p <= 1e-13`, all 8 transition kinds & Section 7 counters,
   `branches_hist`, multi-word `n = 65, 130`, and zero handle leaks.
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from stimside._magic.channels import compile_clifford_factors, compile_pauli_product
from stimside._native import branch_kernels as bk
from stimside.util.reference_chp import ReferenceCHPTableau

_OMEGA = complex(math.sqrt(0.5), math.sqrt(0.5))
_OMEGA_P = tuple(_OMEGA**k for k in range(8))


def _nw(n: int) -> int:
    return (n + 63) // 64


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


def _herm_kappa(x_int: int, z_int: int, neg: bool = False) -> int:
    return ((x_int & z_int).bit_count() + (2 if neg else 0)) & 3


def _anti_words(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> int:
    acc = int(np.bitwise_xor.reduce((x1 & z2) ^ (z1 & x2)))
    return acc.bit_count() & 1


def _dense_apply_pauli(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.uint64)
    src = idx ^ np.uint64(x_int)
    zb = src & np.uint64(z_int)
    par = np.bitwise_count(zb) & np.uint64(1)
    sign = np.where(par == 0, 1.0, -1.0)
    phase = (1.0, 1.0j, -1.0, -1.0j)[kappa & 3]
    return (phase * sign) * vec[src]


def _dense_project(vec: np.ndarray, n: int, x_int: int, z_int: int, kappa: int, m: int) -> tuple[float, np.ndarray]:
    pv = _dense_apply_pauli(vec, n, x_int, z_int, kappa)
    proj = 0.5 * (vec + (1.0 if m == 0 else -1.0) * pv)
    p = float(np.real(np.vdot(proj, proj)))
    return p, (proj / math.sqrt(p) if p > 1e-15 else np.zeros_like(vec))


def _phase_aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    ov = np.vdot(b, a)
    if abs(ov) > 1e-15:
        b = b * (ov / abs(ov))
    return float(np.max(np.abs(a - b)))


def _make_basis_state(n: int, bits: int) -> bk.BranchState:
    st = bk.BranchState.zero(n)
    if bits:
        st.pauli(_int_to_words(bits, _nw(n)), np.zeros(_nw(n), dtype=np.uint64), 0)
    st.canon()
    return st


def _make_bell_phi_plus(n: int, q0: int = 0, q1: int = 1) -> bk.BranchState:
    """Build |Phi+> on (q0, q1) and |0> elsewhere."""
    nw = _nw(n)
    st = bk.BranchState.zero(n)
    wq0 = _int_to_words(1 << q0, nw)
    wq1 = _int_to_words(1 << q1, nw)
    z0 = np.zeros(nw, dtype=np.uint64)
    # H on q0 via R(-Y)
    st.qturn(wq0, wq0, 3)
    # CX(q0 -> q1) via R(+Z_q0 X_q1) R(-Z_q0) R(-X_q1)
    st.qturn(wq1, wq0, 0)
    st.qturn(z0, wq0, 2)
    st.qturn(wq1, z0, 2)
    st.canon()
    return st


def _load_xmem_t_builder() -> Any:
    cand = Path(__file__).resolve().parents[3] / "benchmarks" / "_circuits.py"
    if cand.is_file():
        spec = importlib.util.spec_from_file_location("_bench_circuits_wp3", cand)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.xmem_t

    def _inline_xmem_t(k: int, distance: int = 3, rounds: int = 3, p: float = 0.0) -> stim.Circuit:
        base = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,
        ).flattened()
        data = sorted({t.value for i in base if i.name == "MX" for t in i.targets_copy()})
        c, done = stim.Circuit(), False
        for inst in base:
            c.append(inst)
            if inst.name == "TICK" and not done:
                if k:
                    c.append("I", data[:k], tag="T")
                done = True
        return c

    return _inline_xmem_t


def _run_stim_circuit_shots(
    circuit: stim.Circuit,
    n_shots: int,
    seed: int,
    *,
    store: bk.BranchStore | None = None,
    cap_nodes: int | None = None,
) -> tuple[np.ndarray, bk.BranchStore, list[int]]:
    """Run a Stim circuit (with optional `I[T]` tags) on a shared `BranchStore` across `n_shots`."""
    flat = circuit.flattened()
    n = flat.num_qubits
    nw = _nw(n)
    own_store = store is None
    if store is None:
        store = bk.BranchStore(n)

    # Pre-compile moving-frame event table for the circuit (shared across all shots!).
    U = ReferenceCHPTableau(n)
    compiled_ops: list[tuple[str, Any]] = []
    eid = 1
    for inst in flat:
        name = inst.name
        targs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        if name == "I" and inst.tag == "T":
            for q in targs:
                ax = compile_pauli_product(U, [(q, "Z")])
                compiled_ops.append(("T", (eid, ax)))
                eid += 1
        elif name in ("R", "RX", "RY"):
            basis = "Z" if name == "R" else ("X" if name == "RX" else "Y")
            kick = "X" if basis == "Z" else "Z"
            for q in targs:
                ax_m = compile_pauli_product(U, [(q, basis)])
                ax_k = compile_pauli_product(U, [(q, kick)])
                compiled_ops.append(("RESET", (eid, ax_m, ax_k)))
                eid += 1
        elif name in ("M", "MX", "MY"):
            basis = "Z" if name == "M" else ("X" if name == "MX" else "Y")
            for q in targs:
                ax_m = compile_pauli_product(U, [(q, basis)])
                compiled_ops.append(("MEAS", (eid, ax_m)))
                eid += 1
        elif name in ("MR", "MRX", "MRY"):
            basis = "Z" if name == "MR" else ("X" if name == "MRX" else "Y")
            kick = "X" if basis == "Z" else "Z"
            for q in targs:
                ax_m = compile_pauli_product(U, [(q, basis)])
                ax_k = compile_pauli_product(U, [(q, kick)])
                compiled_ops.append(("MR", (eid, ax_m, ax_k)))
                eid += 1
        elif name == "I":
            continue
        else:
            U.do_unitary(name, targs)

    n_meas = sum(1 for kind, _ in compiled_ops if kind in ("MEAS", "MR"))
    outcomes = np.zeros((n_shots, n_meas), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    node_snapshots: list[int] = []

    for s_idx in range(n_shots):
        if cap_nodes is not None and store.node_count > cap_nodes:
            store.clear()
        nid = 0
        fx = np.zeros(nw, dtype=np.uint64)
        fz = np.zeros(nw, dtype=np.uint64)
        m_col = 0
        for kind, payload in compiled_ops:
            if kind == "T":
                op_eid, ax = payload
                s_bit = _anti_words(fx, fz, ax.a_words, ax.b_words)
                k = (int(ax.kappa) + (2 if s_bit else 0)) & 3
                nid = store.step_rotations(
                    nid,
                    op_eid,
                    bk.MBR_KIND_T,
                    s_bit,
                    ax.a_words.reshape(1, nw),
                    ax.b_words.reshape(1, nw),
                    np.array([k], dtype=np.uint8),
                    np.array([1], dtype=np.uint8),
                )
            elif kind == "MEAS":
                op_eid, ax_m = payload
                s_bit = _anti_words(fx, fz, ax_m.a_words, ax_m.b_words)
                p0, c0, c1, mt, abs_x, abs_z = store.step_meas(
                    nid, op_eid, bk.MBR_KIND_MEAS, ax_m.a_words, ax_m.b_words, int(ax_m.kappa)
                )
                u = float(rng.random())
                inner_m = 0 if u < p0 else 1
                if inner_m == 0 and c0 < 0:
                    inner_m = 1
                elif inner_m == 1 and c1 < 0:
                    inner_m = 0
                if mt == bk.MBR_MEAS_ABSORBED and inner_m == 1:
                    fx ^= abs_x
                    fz ^= abs_z
                nid = c0 if inner_m == 0 else c1
                outcomes[s_idx, m_col] = inner_m ^ s_bit
                m_col += 1
            elif kind in ("RESET", "MR"):
                op_eid, ax_m, ax_k = payload
                s_bit = _anti_words(fx, fz, ax_m.a_words, ax_m.b_words)
                p0, c0, c1, mt, abs_x, abs_z = store.step_meas(
                    nid, op_eid, bk.MBR_KIND_MEAS, ax_m.a_words, ax_m.b_words, int(ax_m.kappa)
                )
                u = float(rng.random())
                inner_m = 0 if u < p0 else 1
                if inner_m == 0 and c0 < 0:
                    inner_m = 1
                elif inner_m == 1 and c1 < 0:
                    inner_m = 0
                if mt == bk.MBR_MEAS_ABSORBED and inner_m == 1:
                    fx ^= abs_x
                    fz ^= abs_z
                nid = c0 if inner_m == 0 else c1
                out_m = inner_m ^ s_bit
                if kind == "MR":
                    outcomes[s_idx, m_col] = out_m
                    m_col += 1
                if out_m == 1:
                    fx ^= ax_k.a_words
                    fz ^= ax_k.b_words
        node_snapshots.append(store.node_count)

    assert own_store or store is not None
    return outcomes, store, node_snapshots


# ---------------------------------------------------------------------------------------------
# Group 1: Explicit Multi-Branch GF(2) Measurement Absorption Tests
# ---------------------------------------------------------------------------------------------


def test_absorption_case_a_ev_neg1_product_stabilizer() -> None:
    """Case A: S_0=|00>, S_1=|11> (Z_0, Z_1 have ev=-1 on S_1, Z_0 Z_1 has ev=+1); measure X_0."""
    n = 2
    with bk.BranchStore(n) as store:
        s0 = _make_basis_state(n, 0b00)
        s1 = _make_basis_state(n, 0b11)
        phi = 0.73
        w0 = math.sqrt(0.5)
        w1 = math.sqrt(0.5) * complex(math.cos(phi), math.sin(phi))
        nid = store.node_from_branches([w0, w1], [s0, s1])
        s0.close()
        s1.close()

        psi_in = store.node_dense(nid)
        p0, c0, c1, mt, abs_x, abs_z = store.step_meas(nid, 1, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert mt == bk.MBR_MEAS_ABSORBED
        assert p0 == 0.5 and c0 == c1 and c0 >= 0
        assert int(abs_x[0]) == 0 and int(abs_z[0]) == 0b11  # Q_bar = Z_0 Z_1

        # Verify both m=0 and m=1 (with F ^= Q_bar) match dense < 1e-12.
        p0_d, v0_d = _dense_project(psi_in, n, 1, 0, 0, 0)
        p1_d, v1_d = _dense_project(psi_in, n, 1, 0, 0, 1)
        assert abs(p0_d - 0.5) < 1e-12 and abs(p1_d - 0.5) < 1e-12
        v0_b = store.node_dense(c0)
        v1_b = _dense_apply_pauli(
            v0_b, n, int(abs_x[0]), int(abs_z[0]), _herm_kappa(int(abs_x[0]), int(abs_z[0]))
        )
        assert _phase_aligned_diff(v0_b, v0_d) < 1e-12
        assert _phase_aligned_diff(v1_b, v1_d) < 1e-12


def test_absorption_case_b_ev_zero_product_stabilizer() -> None:
    """Case B: S_0=|00>, S_1=|Phi+> (Z_0, Z_1 have ev=0 on S_1, Z_0 Z_1 has ev=+1); measure X_0."""
    n = 2
    with bk.BranchStore(n) as store:
        s0 = _make_basis_state(n, 0b00)
        s1 = _make_bell_phi_plus(n, 0, 1)
        # Normalize alpha|00> + beta|Phi+> where <00|Phi+> = 1/sqrt(2).
        a, b = 0.6 + 0.2j, -0.3 + 0.5j
        v_raw = a * s0.dense() + b * s1.dense()
        norm = float(np.linalg.norm(v_raw))
        a /= norm
        b /= norm
        nid = store.node_from_branches([a, b], [s0, s1])
        s0.close()
        s1.close()

        psi_in = store.node_dense(nid)
        p0, c0, c1, mt, abs_x, abs_z = store.step_meas(nid, 1, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert mt == bk.MBR_MEAS_ABSORBED
        assert p0 == 0.5 and c0 == c1 and c0 >= 0
        assert int(abs_x[0]) == 0 and int(abs_z[0]) == 0b11  # Q_bar = Z_0 Z_1

        p0_d, v0_d = _dense_project(psi_in, n, 1, 0, 0, 0)
        p1_d, v1_d = _dense_project(psi_in, n, 1, 0, 0, 1)
        assert abs(p0_d - 0.5) < 1e-12 and abs(p1_d - 0.5) < 1e-12
        v0_b = store.node_dense(c0)
        v1_b = _dense_apply_pauli(
            v0_b, n, int(abs_x[0]), int(abs_z[0]), _herm_kappa(int(abs_x[0]), int(abs_z[0]))
        )
        assert _phase_aligned_diff(v0_b, v0_d) < 1e-12
        assert _phase_aligned_diff(v1_b, v1_d) < 1e-12


def test_absorption_case_c_kappa_sign_tracking_and_negative_control() -> None:
    """Case C: common stabilizer -Z_0 Z_1 absorbs; opposite-sign Z_1 on |00>,|01> MUST fork."""
    n = 2
    with bk.BranchStore(n) as store:
        # Positive kappa test: |10> (-Z0, +Z1) and |01> (+Z0, -Z1) share common -Z_0 Z_1 (ev = +1 on both!).
        s10 = _make_basis_state(n, 0b01)
        s01 = _make_basis_state(n, 0b10)
        nid = store.node_from_branches([math.sqrt(0.3), math.sqrt(0.7) * 1j], [s10, s01])
        s10.close()
        s01.close()

        psi_in = store.node_dense(nid)
        p0, c0, c1, mt, abs_x, abs_z = store.step_meas(nid, 1, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert mt == bk.MBR_MEAS_ABSORBED and p0 == 0.5 and c0 == c1
        assert int(abs_x[0]) == 0 and int(abs_z[0]) == 0b11
        p0_d, v0_d = _dense_project(psi_in, n, 1, 0, 0, 0)
        p1_d, v1_d = _dense_project(psi_in, n, 1, 0, 0, 1)
        assert abs(p0_d - 0.5) < 1e-12 and abs(p1_d - 0.5) < 1e-12
        v0_b = store.node_dense(c0)
        v1_b = _dense_apply_pauli(
            v0_b, n, int(abs_x[0]), int(abs_z[0]), _herm_kappa(int(abs_x[0]), int(abs_z[0]))
        )
        assert _phase_aligned_diff(v0_b, v0_d) < 1e-12
        assert _phase_aligned_diff(v1_b, v1_d) < 1e-12

        # Negative control: |00> (+Z0, +Z1) and |01> (+Z0, -Z1): Z_1 has opposite signs!
        # Measuring X_1 gives p0 = |w0 + w1|^2 / 2 != 0.5 -> MUST fork, NEVER absorb!
        s00 = _make_basis_state(n, 0b00)
        s01_b = _make_basis_state(n, 0b10)
        nid_neg = store.node_from_branches([0.6, 0.8], [s00, s01_b])
        s00.close()
        s01_b.close()
        psi_neg = store.node_dense(nid_neg)
        p0_n, c0_n, c1_n, mt_n, _, _ = store.step_meas(nid_neg, 2, bk.MBR_KIND_MEAS, [2], [0], 0)
        assert mt_n == bk.MBR_MEAS_FORKED and c0_n != c1_n
        p0_nd, v0_nd = _dense_project(psi_neg, n, 2, 0, 0, 0)
        p1_nd, v1_nd = _dense_project(psi_neg, n, 2, 0, 0, 1)
        assert abs(p0_n - p0_nd) < 1e-12 and abs(p0_nd - 0.98) < 1e-12
        assert _phase_aligned_diff(store.node_dense(c0_n), v0_nd) < 1e-12
        assert _phase_aligned_diff(store.node_dense(c1_n), v1_nd) < 1e-12


def test_absorption_case_d_pending_turns_and_case_e_post_kick() -> None:
    """Case D: pending turn blocks anticommuting stabilizer, allows commuting one; Case E: POST kick."""
    n = 3
    with bk.BranchStore(n) as store:
        # |000> and |110> share Z_0 Z_1 and Z_2.
        # Add pending turn D(X_0 X_1, e=1) (commutes with Z_0 Z_1 and X_0 and X_1? No, X_0 X_1
        # commutes with X_0 and Z_0 Z_1!).
        # Also test D(X_2, e=1) which commutes with M = X_0, and anticommutes with Z_2 (so Z_2 is blocked,
        # while Z_0 Z_1 commutes with X_2 and anticommutes with M = X_0 -> absorbed!).
        s0 = _make_basis_state(n, 0b000)
        s1 = _make_basis_state(n, 0b011)
        nid_ok = store.node_from_branches(
            [math.sqrt(0.4), math.sqrt(0.6) * 1j],
            [s0, s1],
            pend_x=np.array([[0b100]], dtype=np.uint64),  # X_2
            pend_z=np.array([[0]], dtype=np.uint64),
            pend_kappa=np.array([0], dtype=np.uint8),
            pend_e=np.array([1], dtype=np.uint8),
        )
        psi_ok = store.node_dense(nid_ok)
        # Measure M = X_0 (commutes with X_2, anticommutes with Z_0 Z_1) -> ABSORBED!
        p0, c0, c1, mt, abs_x, abs_z = store.step_meas(nid_ok, 1, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert mt == bk.MBR_MEAS_ABSORBED and p0 == 0.5 and c0 == c1
        assert int(abs_x[0]) == 0 and int(abs_z[0]) == 0b011  # Z_0 Z_1 (NOT Z_2)
        p0_d, v0_d = _dense_project(psi_ok, n, 1, 0, 0, 0)
        p1_d, v1_d = _dense_project(psi_ok, n, 1, 0, 0, 1)
        assert abs(p0_d - 0.5) < 1e-12 and abs(p1_d - 0.5) < 1e-12
        v0_b = store.node_dense(c0)
        v1_b = _dense_apply_pauli(
            v0_b, n, int(abs_x[0]), int(abs_z[0]), _herm_kappa(int(abs_x[0]), int(abs_z[0]))
        )
        assert _phase_aligned_diff(v0_b, v0_d) < 1e-12
        assert _phase_aligned_diff(v1_b, v1_d) < 1e-12

        # Now add pending turn D(X_0 X_2, e=1) which commutes with M = X_2? Wait:
        # Take S_0 = |000>, S_1 = |000> (single branch |000>, gens Z_0, Z_1, Z_2),
        # pending D(X_0, e=1), measure M = X_0 (commutes with X_0!).
        # Only Z_0 anticommutes with M = X_0, AND Z_0 anticommutes with pending X_0!
        # So NO stabilizer commutes with pend AND anticommutes with M -> MUST fork!
        nid_blocked = store.node_from_branches(
            [1.0 + 0.0j],
            [s0],
            pend_x=np.array([[0b001]], dtype=np.uint64),  # X_0
            pend_z=np.array([[0]], dtype=np.uint64),
            pend_kappa=np.array([0], dtype=np.uint8),
            pend_e=np.array([1], dtype=np.uint8),
        )
        p0_bl, c0_bl, c1_bl, mt_bl, _, _ = store.step_meas(nid_blocked, 2, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert mt_bl == bk.MBR_MEAS_FORKED and abs(p0_bl - 0.5) < 1e-12

        # Case E: POST on absorbed inner outcome 1 MUST apply F ^= Q_bar.
        p0_post, cp0, cp1, mt_post, pqx, pqz = store.step_meas(nid_ok, 3, bk.MBR_KIND_POST, [1], [0], 0)
        assert mt_post == bk.MBR_MEAS_ABSORBED and p0_post == 0.5 and cp0 == cp1
        # Postselecting desired inner outcome 1 applies Q_bar to F -> matches dense m=1 projection!
        v_post1 = _dense_apply_pauli(
            store.node_dense(cp1), n, int(pqx[0]), int(pqz[0]), _herm_kappa(int(pqx[0]), int(pqz[0]))
        )
        assert _phase_aligned_diff(v_post1, v1_d) < 1e-12
        s0.close()
        s1.close()


def _conj_pauli_by_qturn_int(
    px: int, pz: int, pk: int, ax: int, az: int, ak: int
) -> tuple[int, int, int]:
    """Compute R(A) P R(A)^dag = i * A * P when A anticommutes with P, else P."""
    anti = ((ax & pz) ^ (az & px)).bit_count() & 1
    if anti == 0:
        return px, pz, pk & 3
    s = (az & px).bit_count() & 1
    return px ^ ax, pz ^ az, (ak + pk + (s << 1) + 1) & 3


def test_absorption_case_f_random_multibranch_stress_vs_dense() -> None:
    """Case F: 220 random Clifford-rotated 2-/3-branch entangled states vs dense (< 1e-12)."""
    rng = np.random.default_rng(20260310)
    n_absorbed = 0
    n_entangled_absorbed = 0
    max_dev = 0.0
    for trial in range(220):
        n = int(rng.integers(2, 6))
        k_br = int(rng.integers(2, 4))
        with bk.BranchStore(n) as store:
            # Build k_br basis states with even parity on qubits (0, 1) so Z_0 Z_1 is common,
            # then conjugate the full node (branches + optional pending Z_2) and M = X_0 by a
            # random sequence of 1Q/2Q Clifford quarter turns so d > 0 entangled states are tested.
            states: list[bk.BranchState] = []
            for _ in range(k_br):
                bits = int(rng.integers(0, 1 << n))
                if ((bits ^ (bits >> 1)) & 1) != 0:
                    bits ^= 1
                states.append(_make_basis_state(n, bits))

            ws = rng.normal(size=k_br) + 1j * rng.normal(size=k_br)
            raw_vec = sum(w * st.dense() for w, st in zip(ws, states))
            norm = float(np.linalg.norm(raw_vec))
            if norm < 1e-6:
                for st in states:
                    st.close()
                continue
            ws = ws / norm

            # Half the trials include a commuting pending turn D(Z_2, e) (commutes with Z_0 Z_1 and X_0).
            if trial % 2 == 1 and n >= 3:
                nid0 = store.node_from_branches(
                    ws,
                    states,
                    pend_x=np.array([[0]], dtype=np.uint64),
                    pend_z=np.array([[4]], dtype=np.uint64),  # Z_2
                    pend_kappa=np.array([0], dtype=np.uint8),
                    pend_e=np.array([int(rng.integers(1, 8))], dtype=np.uint8),
                )
            else:
                nid0 = store.node_from_branches(ws, states)
            for st in states:
                st.close()

            # Sample random 1Q/2Q Clifford quarter turns (H/S/SQRT_X/CX/CZ-style axes).
            qturns: list[tuple[int, int, int]] = []
            for _ in range(n + 3):
                r = rng.random()
                if r < 0.45:
                    q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
                    kind_2q = int(rng.integers(0, 3))
                    if kind_2q == 0:
                        ax, az = (1 << q1), (1 << q0)  # Z_{q0} X_{q1}
                    elif kind_2q == 1:
                        ax, az = 0, (1 << q0) | (1 << q1)  # Z_{q0} Z_{q1}
                    else:
                        ax, az = (1 << q0) | (1 << q1), 0  # X_{q0} X_{q1}
                else:
                    q = int(rng.integers(0, n))
                    kind_1q = int(rng.integers(0, 3))
                    if kind_1q == 0:
                        ax, az = (1 << q), (1 << q)  # Y_q (Hadamard-like)
                    elif kind_1q == 1:
                        ax, az = 0, (1 << q)  # Z_q (S-like)
                    else:
                        ax, az = (1 << q), 0  # X_q (SQRT_X-like)
                ak = _herm_kappa(ax, az, neg=bool(rng.integers(0, 2)))
                qturns.append((ax, az, ak))

            qt_x = np.array([[ax] for ax, _, _ in qturns], dtype=np.uint64)
            qt_z = np.array([[az] for _, az, _ in qturns], dtype=np.uint64)
            qt_k = np.array([ak for _, _, ak in qturns], dtype=np.uint8)
            nid = store.step_qturns(nid0, 1, bk.MBR_KIND_CFAULT, 0, qt_x, qt_z, qt_k)

            # Conjugate the measured Pauli M = X_0 through the same quarter turns: R M R^dag.
            mx, mz, mk = 1, 0, 0
            for ax, az, ak in qturns:
                mx, mz, mk = _conj_pauli_by_qturn_int(mx, mz, mk, ax, az, ak)

            brs = store.node_branches(nid)
            has_entangled = False
            try:
                has_entangled = any(st.d > 0 for _, st in brs)
            finally:
                for _, st in brs:
                    st.close()

            psi_in = store.node_dense(nid)
            p0, c0, c1, mt, abs_x, abs_z = store.step_meas(nid, 2, bk.MBR_KIND_MEAS, [mx], [mz], mk)
            p0_d, v0_d = _dense_project(psi_in, n, mx, mz, mk, 0)
            p1_d, v1_d = _dense_project(psi_in, n, mx, mz, mk, 1)
            max_dev = max(max_dev, abs(p0 - p0_d))
            if mt == bk.MBR_MEAS_ABSORBED:
                n_absorbed += 1
                if has_entangled:
                    n_entangled_absorbed += 1
                assert c0 == c1 and p0 == 0.5
                v0_b = store.node_dense(c0)
                v1_b = _dense_apply_pauli(
                    v0_b, n, int(abs_x[0]), int(abs_z[0]), _herm_kappa(int(abs_x[0]), int(abs_z[0]))
                )
                max_dev = max(max_dev, _phase_aligned_diff(v0_b, v0_d), _phase_aligned_diff(v1_b, v1_d))
            elif mt == bk.MBR_MEAS_FORKED:
                if c0 >= 0:
                    max_dev = max(max_dev, _phase_aligned_diff(store.node_dense(c0), v0_d))
                if c1 >= 0:
                    max_dev = max(max_dev, _phase_aligned_diff(store.node_dense(c1), v1_d))

    assert n_absorbed >= 180
    assert n_entangled_absorbed >= 150
    assert max_dev < 1e-12, f"multi-branch absorption max_dev={max_dev}"


def test_definite_after_step1_resolve_returns_resolved_child() -> None:
    """When step_meas resolves anticommuting pend to a DEFINITE eigenstate, it returns resolved rnid."""
    n = 2
    with bk.BranchStore(n) as store:
        # D(X_0, e=2)|00> = exp(i pi/4 (I - X_0))|00> = e^{i pi/4} R(-X_0)|00>,
        # which is a -1 eigenstate of +Y_0! Measuring +Y_0 anticommutes with pend X_0 ->
        # Step 1 resolves pend, Step 2 finds DEFINITE outcome 1, and MUST return child1 != nid with n_pend == 0.
        nid = store.step_rotations(
            0,
            1,
            bk.MBR_KIND_T,
            0,
            np.array([[1]], dtype=np.uint64),  # X_0
            np.array([[0]], dtype=np.uint64),
            np.array([0], dtype=np.uint8),
            np.array([2], dtype=np.uint8),
        )
        assert store.node_info(nid)[1] == 1  # pending turn attached before measurement
        psi_in = store.node_dense(nid)
        p0, c0, c1, mt, _, _ = store.step_meas(nid, 2, bk.MBR_KIND_MEAS, [1], [1], 1)  # +Y_0
        assert mt == bk.MBR_MEAS_DEFINITE
        assert p0 == 0.0 and c0 == -1 and c1 >= 0 and c1 != nid
        assert store.node_info(c1)[1] == 0  # resolved child has n_pend == 0
        p1_d, v1_d = _dense_project(psi_in, n, 1, 1, 1, 1)
        assert abs(p1_d - 1.0) < 1e-12
        assert _phase_aligned_diff(store.node_dense(c1), v1_d) < 1e-12

        # Multi-branch equivalent: D(X_0, e=2) (sqrt(0.3)|00> + i*sqrt(0.7)|01>) measured on +Y_0.
        s00 = _make_basis_state(n, 0b00)
        s01 = _make_basis_state(n, 0b10)
        nid_mb = store.node_from_branches(
            [math.sqrt(0.3), math.sqrt(0.7) * 1j],
            [s00, s01],
            pend_x=np.array([[1]], dtype=np.uint64),
            pend_z=np.array([[0]], dtype=np.uint64),
            pend_kappa=np.array([0], dtype=np.uint8),
            pend_e=np.array([2], dtype=np.uint8),
        )
        s00.close()
        s01.close()
        assert store.node_info(nid_mb)[1] == 1
        psi_mb = store.node_dense(nid_mb)
        p0_m, c0_m, c1_m, mt_m, _, _ = store.step_meas(nid_mb, 3, bk.MBR_KIND_MEAS, [1], [1], 1)
        assert mt_m == bk.MBR_MEAS_DEFINITE
        assert p0_m == 0.0 and c0_m == -1 and c1_m >= 0 and c1_m != nid_mb
        assert store.node_info(c1_m)[1] == 0
        p1_md, v1_md = _dense_project(psi_mb, n, 1, 1, 1, 1)
        assert abs(p1_md - 1.0) < 1e-12
        assert _phase_aligned_diff(store.node_dense(c1_m), v1_md) < 1e-12


# ---------------------------------------------------------------------------------------------
# Group 2 (T11): Bounded Store Size & Non-Forking Absorption
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [10, 70])
def test_t11_plus_state_measured_in_z_bounded_store_and_uniform(n: int) -> None:
    """T11: |+>^n (n = 10 and multi-word n = 70) measured in Z over 500 shots."""
    c = stim.Circuit()
    c.append("H", range(n))
    c.append("M", range(n))
    outcomes, store, snaps = _run_stim_circuit_shots(c, n_shots=500, seed=20260311 + n)
    try:
        cnt = store.counters()
        assert cnt["meas_absorbed"] == n
        assert cnt["meas_forked"] == 0
        assert cnt["store_nodes"] <= n + 1
        assert snaps[49] == snaps[499]  # strictly constant between 50 and 500 shots!
        # Check outcomes are ~50/50 independent across qubits (within 5 sigma).
        means = outcomes.mean(axis=0)
        assert np.all(np.abs(means - 0.5) < 5.0 * math.sqrt(0.25 / 500))
    finally:
        store.close()


@pytest.mark.parametrize("task", ["surface_code:rotated_memory_z", "surface_code:rotated_memory_x"])
def test_t11_noiseless_rotated_surface_code_does_not_fork(task: str) -> None:
    """T11: Noiseless rotated surface code (d=3, r=3, p=0) has meas_forked == 0 & constant store."""
    c = stim.Circuit.generated(task, distance=3, rounds=3)
    _, store, snaps = _run_stim_circuit_shots(c, n_shots=200, seed=20260312)
    try:
        cnt = store.counters()
        assert cnt["meas_forked"] == 0
        assert cnt["meas_absorbed"] > 0
        assert snaps[49] == snaps[199]  # constant between 50 and 200 shots
    finally:
        store.close()


def test_t11_noiseless_xmem_t_k0_to_k4_bounded_store() -> None:
    """T11: Noiseless xmem_t(k=0..4, d=3, r=3, p=0): k=0 no forks; k=1..4 absorbed & constant store."""
    xmem_t = _load_xmem_t_builder()
    for k in range(5):
        c = xmem_t(k, distance=3, rounds=3, p=0.0)
        _, store, snaps = _run_stim_circuit_shots(c, n_shots=500, seed=20260313 + k)
        try:
            cnt = store.counters()
            assert cnt["meas_absorbed"] > 0
            if k == 0:
                assert cnt["meas_forked"] == 0
                assert snaps[49] == snaps[499]
            else:
                # Strictly bounded and constant once all O(1) non-0.5 X-check branches are visited!
                assert snaps[199] == snaps[499]
            assert cnt["store_nodes"] <= 80
        finally:
            store.close()


# ---------------------------------------------------------------------------------------------
# Group 3: Cap/Clear (F5), Float Drift (-3), Impossible (1e-13), Section 7 Counters & Leaks
# ---------------------------------------------------------------------------------------------


def test_store_cap_clear_and_f5_determinism() -> None:
    """Verify clear() resets trie/sub-caches/bytes, preserves cumulative counters, and obeys F5."""
    xmem_t = _load_xmem_t_builder()
    c = xmem_t(3, distance=3, rounds=3, p=0.0)

    # 1. Warm uncapped run (20 shots).
    out_uncapped, store_uncapped, _ = _run_stim_circuit_shots(c, n_shots=20, seed=777)
    store_uncapped.close()

    # 2. Small-cap run (cap = 20 nodes -> clears between shots).
    out_capped, store_capped, _ = _run_stim_circuit_shots(c, n_shots=20, seed=777, cap_nodes=20)
    try:
        cnt = store_capped.counters()
        assert cnt["store_clears"] >= 1
        assert np.array_equal(out_capped, out_uncapped)

        # Explicit clear() check: resets nodes to 1 & drops bytes_estimate while preserving cumulative counts.
        before_bytes = cnt["store_bytes_estimate"]
        before_clears = cnt["store_clears"]
        before_new = dict(cnt["transitions_new"])
        before_res = cnt["resolve_calls"]
        before_sub_m = cnt["subcache_misses"]

        store_capped.clear()
        after = store_capped.counters()
        assert after["store_nodes"] == 1
        assert after["store_clears"] == before_clears + 1
        assert after["store_bytes_estimate"] < before_bytes
        assert after["transitions_new"] == before_new
        assert after["resolve_calls"] == before_res
        assert after["subcache_misses"] == before_sub_m
        assert after["branches_hist"] == {1: 1}
    finally:
        store_capped.close()

    # 3. Cold store (cleared before every shot via cap_nodes=0) must be bit-identical across all 20 shots!
    out_cold, s_cold, _ = _run_stim_circuit_shots(c, n_shots=20, seed=777, cap_nodes=0)
    try:
        assert s_cold.counters()["store_clears"] == 20
        assert np.array_equal(out_cold, out_uncapped)
    finally:
        s_cold.close()


def test_wp3_hostile_inputs_and_closed_store_guards() -> None:
    """Verify hostile inputs to is_clifford, choose_split, and BranchStore raise cleanly without leaks."""
    with pytest.raises(ValueError):
        bk.is_clifford([0])
    with pytest.raises(ValueError):
        bk.choose_split([])
    with pytest.raises(ValueError):
        bk.BranchStore(0)
    with pytest.raises(ValueError):
        bk.BranchStore(4, max_components=-1)
    with pytest.raises(ValueError):
        bk.BranchStore(2, max_components=1 << 64)
    with pytest.raises(TypeError):
        bk.BranchStore(2, max_components=True)

    with bk.BranchStore(3) as store:
        with pytest.raises(ValueError):
            store.set_max_components(1 << 64)
        with pytest.raises(TypeError):
            store.set_max_components(True)

        # Reject bool and str arrays/lists in weights (_as_complex128_1d).
        with bk.BranchState.zero(3) as st0:
            for bad_w in (
                np.array([True], dtype=bool),
                [True],
                ["1.0"],
                np.array(["1.0"]),
                np.array([True], dtype=object),
            ):
                with pytest.raises(TypeError):
                    store.node_from_branches(bad_w, [st0])

        # Direct C ABI mul_within overflow guards return MBR_ERR_INVALID_ARG (-1) before looping.
        lib = bk.get_branch_kernels_lib()
        huge = (1 << 62) + 7
        dummy_u64 = np.zeros(1, dtype=np.uint64)
        dummy_u8 = np.zeros(1, dtype=np.uint8)
        dummy_f64 = np.zeros(2, dtype=np.float64)
        dummy_ptr = np.zeros(1, dtype=np.uintp)
        out_cid = ctypes.c_int64()
        assert (
            lib.mbr_store_step_qturns(
                store._handle, 0, 1001, bk.MBR_KIND_SKIP, 0, huge,
                bk.addr(dummy_u64), bk.addr(dummy_u64), bk.addr(dummy_u8),
                ctypes.addressof(out_cid),
            )
            == bk.MBR_ERR_INVALID_ARG
        )
        assert (
            lib.mbr_store_step_rotations(
                store._handle, 0, 1002, bk.MBR_KIND_T, 0, huge,
                bk.addr(dummy_u64), bk.addr(dummy_u64), bk.addr(dummy_u8), bk.addr(dummy_u8),
                ctypes.addressof(out_cid),
            )
            == bk.MBR_ERR_INVALID_ARG
        )
        assert (
            lib.mbr_store_node_from_branches(
                store._handle, huge, bk.addr(dummy_f64), bk.addr(dummy_ptr),
                0, 0, 0, 0, 0, ctypes.addressof(out_cid),
            )
            == bk.MBR_ERR_INVALID_ARG
        )
        assert (
            lib.mbr_store_node_from_branches(
                store._handle, 1, bk.addr(dummy_f64), bk.addr(dummy_ptr),
                huge, bk.addr(dummy_u64), bk.addr(dummy_u64), bk.addr(dummy_u8), bk.addr(dummy_u8),
                ctypes.addressof(out_cid),
            )
            == bk.MBR_ERR_INVALID_ARG
        )

        # Out-of-range node_id
        with pytest.raises(ValueError):
            store.node_info(99)
        with pytest.raises(ValueError):
            store.node_info(-1)
        with pytest.raises(ValueError):
            store.resolve_node(42)
        # Non-Hermitian Pauli (X_0 Z_0 with kappa=0 has odd popcount(x&z) ^ kappa)
        with pytest.raises(ValueError):
            store.step_meas(0, 1, bk.MBR_KIND_MEAS, [1], [1], 0)
        with pytest.raises(ValueError):
            store.step_peek(0, 2, [1], [1], 0)
        # Dirty tail bit (bit 3 set for n=3)
        with pytest.raises(ValueError):
            store.step_meas(0, 3, bk.MBR_KIND_MEAS, [8], [0], 0)
        with pytest.raises(ValueError):
            store.step_rotations(
                0, 4, bk.MBR_KIND_T, 0,
                np.array([[0]], dtype=np.uint64), np.array([[8]], dtype=np.uint64),
                np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8),
            )
        # Invalid transition kind
        with pytest.raises(ValueError):
            store.step_rotations(
                0, 5, bk.MBR_KIND_MEAS, 0,
                np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64),
                np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8),
            )
        with pytest.raises(ValueError):
            store.step_qturns(
                0, 6, bk.MBR_KIND_T, 0,
                np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64),
                np.array([0], dtype=np.uint8),
            )
        with pytest.raises(ValueError):
            store.step_meas(0, 7, bk.MBR_KIND_PEEK, [0], [1], 0)

    # Closed store refuses all operations.
    assert store.closed
    with pytest.raises(ValueError, match="closed"):
        _ = store.node_count
    with pytest.raises(ValueError, match="closed"):
        store.clear()

    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


def test_float_drift_err3_and_impossible_outcome_1e13() -> None:
    """Verify -3 FloatingPointError on step_meas & step_peek, and -1 child on p <= 1e-13."""
    n = 2
    with bk.BranchStore(n) as store:
        # Unnormalized 2-branch non-absorbable state (|00> + |01>, measuring X_1):
        # p0 = |2.0 + 2.0|^2 / 2 = 8.0, p1 = 0 -> p0 + p1 = 8.0 != 1.0 -> -3 FloatingPointError!
        s00 = _make_basis_state(n, 0b00)
        s01 = _make_basis_state(n, 0b10)
        nid_bad = store.node_from_branches([2.0, 1.0], [s00, s01])
        before_nodes = store.node_count
        with pytest.raises(FloatingPointError):
            store.step_meas(nid_bad, 1, bk.MBR_KIND_MEAS, [2], [0], 0)
        assert store.node_count == before_nodes
        with pytest.raises(FloatingPointError):
            store.step_peek(nid_bad, 2, [2], [0], 0)
        assert store.node_count == before_nodes

        # Impossible outcome (p1 <= 1e-13 while p0 == 1):
        # (1/sqrt(2))(|00> + |01>) = |0+> -> measuring X_1 gives p0 = 1.0, p1 = 0.0 (forked path
        # since Z_1 has opposite signs on |00> vs |01>!).
        w = math.sqrt(0.5)
        nid_imp = store.node_from_branches([w, w], [s00, s01])
        p0, c0, c1, mt, _, _ = store.step_meas(nid_imp, 3, bk.MBR_KIND_MEAS, [2], [0], 0)
        assert mt == bk.MBR_MEAS_FORKED
        assert p0 == 1.0 and c0 >= 0 and c1 == -1
        assert abs(float(np.linalg.norm(store.node_dense(c0))) - 1.0) < 1e-12

        # Symmetric impossible p0 <= 1e-13: |0-> measuring X_1 gives p0 = 0.0, child0 = -1, child1 >= 0.
        nid_imp0 = store.node_from_branches([w, -w], [s00, s01])
        p0_z, c0_z, c1_z, mt_z, _, _ = store.step_meas(nid_imp0, 4, bk.MBR_KIND_MEAS, [2], [0], 0)
        assert mt_z == bk.MBR_MEAS_FORKED
        assert p0_z == 0.0 and c0_z == -1 and c1_z >= 0
        assert abs(float(np.linalg.norm(store.node_dense(c1_z))) - 1.0) < 1e-12

        s00.close()
        s01.close()


def test_all_8_transition_kinds_section7_counters_multiword_and_leaks() -> None:
    """Exercise all 8 kinds (hits & misses), all Section 7 counters, branches_hist, n=65/130 & leaks."""
    n = 3
    with bk.BranchStore(n) as store:
        # Build |+++> via step_qturns (CFAULT) & step_qturns (SKIP)
        qx_h = np.array([[1], [2], [4]], dtype=np.uint64)
        qz_h = np.array([[1], [2], [4]], dtype=np.uint64)
        qk_h = np.array([3, 3, 3], dtype=np.uint8)
        n1 = store.step_qturns(0, 10, bk.MBR_KIND_CFAULT, 0, qx_h, qz_h, qk_h)
        assert store.step_qturns(0, 10, bk.MBR_KIND_CFAULT, 0, qx_h, qz_h, qk_h) == n1  # hit

        n2 = store.step_qturns(n1, 11, bk.MBR_KIND_SKIP, 0, np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64), np.array([0], dtype=np.uint8))
        assert store.step_qturns(n1, 11, bk.MBR_KIND_SKIP, 0, np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64), np.array([0], dtype=np.uint8)) == n2  # hit

        # T (miss + hit)
        n3 = store.step_rotations(
            n2, 12, bk.MBR_KIND_T, 0,
            np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64),
            np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8),
        )
        assert store.step_rotations(
            n2, 12, bk.MBR_KIND_T, 0,
            np.array([[0]], dtype=np.uint64), np.array([[1]], dtype=np.uint64),
            np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8),
        ) == n3

        # CCZ (miss + hit)
        ccz_z = np.array([[1], [2], [4], [3], [5], [6], [7]], dtype=np.uint64)
        ccz_x = np.zeros((7, 1), dtype=np.uint64)
        ccz_k = np.zeros(7, dtype=np.uint8)
        ccz_e = np.array([1, 1, 1, 7, 7, 7, 1], dtype=np.uint8)
        n4 = store.step_rotations(n3, 13, bk.MBR_KIND_CCZ, 0, ccz_x, ccz_z, ccz_k, ccz_e)
        assert store.step_rotations(n3, 13, bk.MBR_KIND_CCZ, 0, ccz_x, ccz_z, ccz_k, ccz_e) == n4

        # PEEK (miss + hit) - anticommuting X_0 triggers resolve in memory without mutating nodes!
        nodes_before_peek = store.node_count
        p_pk1 = store.step_peek(n4, 14, [1], [0], 0)
        p_pk2 = store.step_peek(n4, 14, [1], [0], 0)
        assert p_pk1 == p_pk2
        assert store.node_count == nodes_before_peek

        # MEAS (miss + hit)
        m1 = store.step_meas(n4, 15, bk.MBR_KIND_MEAS, [1], [0], 0)
        m2 = store.step_meas(n4, 15, bk.MBR_KIND_MEAS, [1], [0], 0)
        assert m1[:4] == m2[:4]

        # POST (miss + hit)
        po1 = store.step_meas(n4, 16, bk.MBR_KIND_POST, [0], [2], 0)
        po2 = store.step_meas(n4, 16, bk.MBR_KIND_POST, [0], [2], 0)
        assert po1[:4] == po2[:4]

        # URESET (miss + hit) on definite Z_0 of root node 0
        ur1 = store.step_meas(0, 17, bk.MBR_KIND_URESET, [0], [1], 0)
        ur2 = store.step_meas(0, 17, bk.MBR_KIND_URESET, [0], [1], 0)
        assert ur1[:4] == ur2[:4] and ur1[3] == bk.MBR_MEAS_DEFINITE

        cnt = store.counters()
        assert cnt["store_nodes"] == store.node_count and cnt["store_nodes"] > 1
        assert cnt["store_bytes_estimate"] > 0
        assert cnt["store_clears"] == 0
        for kind_name in bk.MBR_KIND_NAMES:
            assert cnt["transitions_new"][kind_name] == 1, kind_name
            assert cnt["transitions_hit"][kind_name] == 1, kind_name
        assert cnt["resolve_calls"] >= 1
        assert cnt["splits"] >= 1
        assert cnt["subcache_hits"] >= 1
        assert cnt["subcache_misses"] >= 1
        assert cnt["meas_definite"] >= 1
        assert cnt["meas_absorbed"] + cnt["meas_forked"] >= 1
        assert sum(cnt["branches_hist"].values()) == cnt["store_nodes"]

    # Multi-word store smoke test at n = 65 and n = 130 (rotations, qturns, absorption, clear).
    for n_big in (65, 130):
        nw = _nw(n_big)
        with bk.BranchStore(n_big) as store:
            q_hi = n_big - 1
            wq = _int_to_words(1 << q_hi, nw)
            z0 = np.zeros(nw, dtype=np.uint64)
            # Apply R(-Y) on highest qubit q_hi -> |+> on q_hi
            n_plus = store.step_qturns(
                0, 1, bk.MBR_KIND_CFAULT, 0,
                wq.reshape(1, nw), wq.reshape(1, nw), np.array([3], dtype=np.uint8),
            )
            # Apply T on q_hi (Z_{q_hi})
            n_t = store.step_rotations(
                n_plus, 2, bk.MBR_KIND_T, 0,
                z0.reshape(1, nw), wq.reshape(1, nw),
                np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8),
            )
            # Measure Z_{q_hi} (commutes with T, anticommutes with X_{q_hi} stabilizer -> wait:
            # Z_{q_hi} commutes with pending Z_{q_hi}; stabilizer X_{q_hi} anticommutes with pending Z_{q_hi},
            # so Z_{q_hi} measurement forks with p0 = 0.5!).
            p0, c0, c1, mt, _, _ = store.step_meas(n_t, 3, bk.MBR_KIND_MEAS, z0, wq, 0)
            assert abs(p0 - 0.5) < 1e-12 and c0 >= 0 and c1 >= 0
            # On n_plus (no pending), measuring Z_{q_hi} is ABSORBED with Q_bar = X_{q_hi}!
            p0_a, ca0, ca1, mt_a, ax, az = store.step_meas(n_plus, 4, bk.MBR_KIND_MEAS, z0, wq, 0)
            assert mt_a == bk.MBR_MEAS_ABSORBED and p0_a == 0.5 and ca0 == ca1
            assert np.array_equal(ax, wq) and np.array_equal(az, z0)
            store.clear()
            assert store.node_count == 1 and store.counters()["store_clears"] == 1

    # Strict leak check: zero live stores and zero live states after teardown.
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0
