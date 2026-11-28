"""WP4 tests for the stabilizer-branch engine in MagicsideSimulator (T4–T7, T9–T10, T12–T13, §7)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside
from stimside import MagicsideSimulator
from stimside._magic import branch_backend as bb
from stimside._magic.channels import compile_pauli_product
from stimside._magic.gates import (
    CliffordFaultChannel,
    FaultBranch,
    decompose_clifford_exact,
    eval_decomposition_unitary_le,
)
from stimside._magic.phase_qfe import (
    ImpossiblePostselectionError,
    PhaseQFEComponent,
    apply_local_unitary_c128,
)
from stimside._magic.plan import compile_direct
from stimside._magic.query import QueryBudgetExceededError
from stimside._magic.statevector import _apply_pauli_to_dense_le, reconstruct_statevector
from stimside._magic.trajectory import run_direct
from stimside._native import branch_kernels as bk
from stimside import golden_test
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside import simulator_magic as sm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_aligned_diff(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Minimum L2 norm `||psi_a - e^{i theta} psi_b||` over global phase."""
    ov = np.vdot(psi_b, psi_a)
    if abs(ov) > 1e-15:
        phase = ov / abs(ov)
        return float(np.linalg.norm(psi_a - phase * psi_b))
    return float(np.linalg.norm(psi_a - psi_b))


def _max_z(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled two-proportion z over all bit means and pairwise XORs (matches simulator_magic_test)."""
    na, nb = a.shape[0], b.shape[0]
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    pp = (pa * na + pb * nb) / (na + nb)
    se = np.sqrt(np.maximum(pp * (1.0 - pp) * (1.0 / na + 1.0 / nb), 1e-18))
    z_max = float(np.max(np.abs(pa - pb) / se)) if a.shape[1] else 0.0
    m = a.shape[1]
    if m <= 48:
        for i in range(m):
            xa = a[:, i : i + 1] ^ a[:, i + 1 :]
            xb = b[:, i : i + 1] ^ b[:, i + 1 :]
            if xa.shape[1] == 0:
                continue
            p1, p2 = xa.mean(axis=0), xb.mean(axis=0)
            pm = (p1 * na + p2 * nb) / (na + nb)
            se2 = np.sqrt(np.maximum(pm * (1.0 - pm) * (1.0 / na + 1.0 / nb), 1e-18))
            z_max = max(z_max, float(np.max(np.abs(p1 - p2) / se2)))
    return z_max


def _xmem_t(k: int, distance: int = 3, rounds: int = 3, p: float = 0.002) -> stim.Circuit:
    """Rotated X-memory surface code with `k` T gates on data qubits (matches benchmarks/_circuits.py)."""
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


def _random_t_cx(
    n: int = 8, layers: int = 12, p_t: float = 0.25, seed: int = 7, p_noise: float = 0.002
) -> stim.Circuit:
    """Random T / CX / DEPOLARIZE1 / H layers (matches benchmarks/_circuits.py)."""
    rng = np.random.default_rng(seed)
    c = stim.Circuit()
    c.append("R", range(n))
    c.append("H", range(n))
    for _ in range(layers):
        ts = [q for q in range(n) if rng.random() < p_t]
        if ts:
            c.append("I", ts, tag="T")
        perm = rng.permutation(n)
        c.append("CX", [int(x) for x in perm[: 2 * (n // 2)]])
        c.append("DEPOLARIZE1", range(n), [p_noise])
        hs = [q for q in range(n) if rng.random() < 0.3]
        if hs:
            c.append("H", hs)
    c.append("M", range(n))
    return c


def _corpus_circuit(name: str) -> tuple[stim.Circuit, bool]:
    for cname, _modes, has_leak, text in golden_test.CORPUS:
        if cname == name:
            return stim.Circuit(text), has_leak
    raise KeyError(name)


def _sample_engine(
    circuit: stim.Circuit,
    *,
    engine: str,
    n_shots: int,
    batch_size: int,
    seed: int,
    op_handler: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = MagicsideSimulator(
        circuit,
        op_handler=op_handler,
        batch_size=batch_size,
        seed=seed,
        _engine=engine,
    )
    recs, dets, obs = [], [], []
    n_batches = n_shots // batch_size
    for b in range(n_batches):
        if b > 0:
            sim.clear()
        sim.run()
        recs.append(sim.get_final_measurement_records())
        if circuit.num_detectors:
            dets.append(sim.get_detector_flips())
        if circuit.num_observables:
            obs.append(sim.get_observable_flips())
    r_all = np.vstack(recs)
    d_all = np.vstack(dets) if dets else np.zeros((n_shots, 0), dtype=np.bool_)
    o_all = np.vstack(obs) if obs else np.zeros((n_shots, 0), dtype=np.bool_)
    return r_all, d_all, o_all


class _DenseOracle:
    """Independent dense complex128 statevector oracle (little-endian) for T4 verification."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.psi = np.zeros(1 << n, dtype=np.complex128)
        self.psi[0] = 1.0 + 0.0j
        self.log_mass = 0.0

    def apply_unitary(self, canon: str, grp: tuple[int, ...]) -> None:
        u = eval_decomposition_unitary_le(decompose_clifford_exact(canon))
        self.psi = apply_local_unitary_c128(self.psi, u, grp, self.n, "little")

    def apply_pauli_terms(self, terms: list[tuple[int, str]], negate: bool = False) -> None:
        a = np.zeros(1, dtype=np.uint64)
        b = np.zeros(1, dtype=np.uint64)
        k = 2 if negate else 0
        for q, pch in terms:
            if pch == "X":
                a[0] ^= np.uint64(1 << q)
            elif pch == "Z":
                b[0] ^= np.uint64(1 << q)
            elif pch == "Y":
                a[0] ^= np.uint64(1 << q)
                b[0] ^= np.uint64(1 << q)
                k = (k + 1) & 3
        self.psi = _apply_pauli_to_dense_le(self.psi, a, b, k)

    def _pauli_vec(self, terms: list[tuple[int, str]], negate: bool = False) -> np.ndarray:
        a = np.zeros(1, dtype=np.uint64)
        b = np.zeros(1, dtype=np.uint64)
        k = 2 if negate else 0
        for q, pch in terms:
            if pch == "X":
                a[0] ^= np.uint64(1 << q)
            elif pch == "Z":
                b[0] ^= np.uint64(1 << q)
            elif pch == "Y":
                a[0] ^= np.uint64(1 << q)
                b[0] ^= np.uint64(1 << q)
                k = (k + 1) & 3
        return _apply_pauli_to_dense_le(self.psi, a, b, k)

    def apply_spp(self, terms: list[tuple[int, str]], negate: bool, dag: bool) -> None:
        pv = self._pauli_vec(terms, negate=negate)
        c = 1.0 / math.sqrt(2.0)
        self.psi = np.ascontiguousarray(c * (self.psi + (1j if dag else -1j) * pv))

    def apply_t(self, q: int, dag: bool = False) -> None:
        idx = np.arange(1 << self.n, dtype=np.uint64)
        mask = ((idx >> np.uint64(q)) & np.uint64(1)) != 0
        phase = np.exp((-1j if dag else 1j) * math.pi / 4.0)
        self.psi = self.psi.copy()
        self.psi[mask] *= phase

    def apply_ccz(self, a: int, b: int, c: int) -> None:
        idx = np.arange(1 << self.n, dtype=np.uint64)
        m = (
            (((idx >> np.uint64(a)) & np.uint64(1)) != 0)
            & (((idx >> np.uint64(b)) & np.uint64(1)) != 0)
            & (((idx >> np.uint64(c)) & np.uint64(1)) != 0)
        )
        self.psi = self.psi.copy()
        self.psi[m] *= -1.0

    def peek_p1(self, terms: list[tuple[int, str]]) -> float:
        pv = self._pauli_vec(terms)
        proj1 = 0.5 * (self.psi - pv)
        return float(np.real(np.vdot(proj1, proj1)))

    def project(self, terms: list[tuple[int, str]], outcome: int, *, update_mass: bool = True) -> float:
        pv = self._pauli_vec(terms)
        proj = 0.5 * (self.psi + (1.0 if outcome == 0 else -1.0) * pv)
        p = float(np.real(np.vdot(proj, proj)))
        assert p > 1e-13
        self.psi = np.ascontiguousarray(proj / math.sqrt(p))
        if update_mass:
            self.log_mass += math.log(p)
        return p


# ---------------------------------------------------------------------------
# T4: Batched walk vs single-shot driver & dense oracle (with small cap clears)
# ---------------------------------------------------------------------------

def test_t4_batched_walk_matches_per_shot_and_dense() -> None:
    rng = np.random.default_rng(20261128)
    gates_1q = ("H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y")
    gates_2q = ("CX", "CY", "CZ", "SWAP", "ISWAP", "CXSWAP")
    batch_size = 16
    max_sv_diff = 0.0
    max_mass_diff = 0.0
    max_peek_diff = 0.0

    for n in (3, 4, 5):
        store = bb._SharedBranchStore(n, max_components=0)
        token = bb._StoreOwnerToken()
        store.register_owner(token)
        single_store = bb._SharedBranchStore(n, max_components=0)

        for batch_idx in range(3):
            if store.native.node_count > 50:
                store.clear_inplace()
            batch = bb.BranchBatch(n, batch_size, store)
            singles = [bb.BranchBatch(n, 1, single_store) for _ in range(batch_size)]
            denses = [_DenseOracle(n) for _ in range(batch_size)]

            # Prepare initial superposition.
            for q in range(n):
                batch.apply_unitary_group("H", (q,))
                for s in range(batch_size):
                    singles[s].apply_unitary_group("H", (q,))
                    denses[s].apply_unitary("H", (q,))

            for _step in range(14):
                op_kind = int(rng.integers(0, 10))
                if op_kind == 0:
                    g1 = gates_1q[int(rng.integers(0, len(gates_1q)))]
                    q = int(rng.integers(0, n))
                    mask = rng.random(batch_size) < 0.7 if rng.random() < 0.6 else None
                    batch.apply_unitary_group(g1, (q,), mask)
                    for s in range(batch_size):
                        ms = None if mask is None else np.array([mask[s]], dtype=np.bool_)
                        singles[s].apply_unitary_group(g1, (q,), ms)
                        if mask is None or mask[s]:
                            denses[s].apply_unitary(g1, (q,))
                elif op_kind == 1:
                    g2 = gates_2q[int(rng.integers(0, len(gates_2q)))]
                    q0, q1 = rng.choice(n, size=2, replace=False).tolist()
                    mask = rng.random(batch_size) < 0.75 if rng.random() < 0.6 else None
                    batch.apply_unitary_group(g2, (q0, q1), mask)
                    for s in range(batch_size):
                        ms = None if mask is None else np.array([mask[s]], dtype=np.bool_)
                        singles[s].apply_unitary_group(g2, (q0, q1), ms)
                        if mask is None or mask[s]:
                            denses[s].apply_unitary(g2, (q0, q1))
                elif op_kind == 2:
                    qs = rng.choice(n, size=2, replace=False).tolist()
                    terms = [(qs[0], "XYZ"[int(rng.integers(0, 3))]), (qs[1], "XYZ"[int(rng.integers(0, 3))])]
                    neg = bool(rng.integers(0, 2))
                    dag = bool(rng.integers(0, 2))
                    mask = rng.random(batch_size) < 0.7 if rng.random() < 0.6 else None
                    batch.apply_spp_group(terms, neg, dag, mask)
                    for s in range(batch_size):
                        ms = None if mask is None else np.array([mask[s]], dtype=np.bool_)
                        singles[s].apply_spp_group(terms, neg, dag, ms)
                        if mask is None or mask[s]:
                            denses[s].apply_spp(terms, neg, dag)
                elif op_kind == 3:
                    tag = "T" if rng.random() < 0.5 else "T_DAG"
                    q = int(rng.integers(0, n))
                    mask = rng.random(batch_size) < 0.8 if rng.random() < 0.5 else None
                    batch.apply_magic(tag, (q,), mask)
                    for s in range(batch_size):
                        ms = None if mask is None else np.array([mask[s]], dtype=np.bool_)
                        singles[s].apply_magic(tag, (q,), ms)
                        if mask is None or mask[s]:
                            denses[s].apply_t(q, dag=(tag == "T_DAG"))
                elif op_kind == 4:
                    a, b_q, c_q = rng.choice(n, size=3, replace=False).tolist()
                    mask = rng.random(batch_size) < 0.75 if rng.random() < 0.6 else None
                    batch.apply_magic("CCZ", (a, b_q, c_q), mask)
                    for s in range(batch_size):
                        ms = None if mask is None else np.array([mask[s]], dtype=np.bool_)
                        singles[s].apply_magic("CCZ", (a, b_q, c_q), ms)
                        if mask is None or mask[s]:
                            denses[s].apply_ccz(a, b_q, c_q)
                elif op_kind == 5:
                    xm = rng.random((n, batch_size)) < 0.15
                    zm = rng.random((n, batch_size)) < 0.15
                    batch.apply_pauli_masks_xz(xm, zm)
                    for s in range(batch_size):
                        singles[s].apply_pauli_masks_xz(xm[:, s : s + 1], zm[:, s : s + 1])
                        terms_f = []
                        for q in range(n):
                            if xm[q, s] and zm[q, s]:
                                terms_f.append((q, "Y"))
                            elif xm[q, s]:
                                terms_f.append((q, "X"))
                            elif zm[q, s]:
                                terms_f.append((q, "Z"))
                        if terms_f:
                            denses[s].apply_pauli_terms(terms_f)
                elif op_kind == 6:
                    if rng.random() < 0.5:
                        g_cf = gates_1q[int(rng.integers(0, len(gates_1q)))]
                        t_cf = (int(rng.integers(0, n)),)
                    else:
                        g_cf = gates_2q[int(rng.integers(0, len(gates_2q)))]
                        t_cf = tuple(rng.choice(n, size=2, replace=False).tolist())
                    mask = rng.random(batch_size) < 0.35
                    batch.apply_clifford_fault(g_cf, t_cf, mask)
                    for s in range(batch_size):
                        singles[s].apply_clifford_fault(g_cf, t_cf, np.array([mask[s]], dtype=np.bool_))
                        if mask[s]:
                            denses[s].apply_unitary(g_cf, t_cf)
                elif op_kind == 7:
                    # Peek + conditional postselect on a feasible outcome.
                    q = int(rng.integers(0, n))
                    pch: Any = "XYZ"[int(rng.integers(0, 3))]
                    p1_batch = batch.peek_pauli_born_batch([q], pauli=pch)[0]
                    for s in range(batch_size):
                        p1_s = singles[s].peek_pauli_born_batch([q], pauli=pch)[0, 0]
                        p1_d = denses[s].peek_p1([(q, pch)])
                        max_peek_diff = max(max_peek_diff, abs(p1_batch[s] - p1_s), abs(p1_batch[s] - p1_d))
                    flip_mask = rng.random(batch_size) < 0.5
                    desired = np.zeros(batch_size, dtype=np.uint8)
                    for s in range(batch_size):
                        if not flip_mask[s]:
                            continue
                        if p1_batch[s] <= 1e-10:
                            desired[s] = 0
                        elif p1_batch[s] >= 1.0 - 1e-10:
                            desired[s] = 1
                        else:
                            desired[s] = int(rng.integers(0, 2))
                    batch.postselect_pauli_batch(q, flip_mask, desired, pauli=pch)
                    for s in range(batch_size):
                        if flip_mask[s]:
                            singles[s].postselect_pauli_batch(
                                q, np.array([True]), np.array([desired[s]], dtype=np.uint8), pauli=pch
                            )
                            denses[s].project([(q, pch)], int(desired[s]), update_mass=True)
                elif op_kind == 8:
                    # Measurement or Reset with shared uniform draws.
                    is_rst = bool(rng.integers(0, 2))
                    q = int(rng.integers(0, n))
                    basis: Any = "XYZ"[int(rng.integers(0, 3))]
                    u_fixed = rng.random(batch_size)

                    class _FixedRng:
                        def __init__(self, vals: np.ndarray) -> None:
                            self._v = vals
                        def random(self, sz: int) -> np.ndarray:
                            return self._v[:sz]

                    if is_rst:
                        out_b = batch.measure_or_reset(
                            (), _FixedRng(u_fixed), is_reset=True, reset_qubit=q, reset_basis=basis  # type: ignore[arg-type]
                        )
                        kick_p = "Z" if basis == "X" else "X"
                        for s in range(batch_size):
                            out_s = singles[s].measure_or_reset(
                                (), _FixedRng(u_fixed[s : s + 1]), is_reset=True, reset_qubit=q, reset_basis=basis  # type: ignore[arg-type]
                            )
                            assert bool(out_b[s]) == bool(out_s[0])
                            denses[s].project([(q, basis)], int(out_b[s]), update_mass=True)
                            if out_b[s]:
                                denses[s].apply_pauli_terms([(q, kick_p)])
                    else:
                        q2 = (q + 1) % n
                        terms_m = [(q, basis), (q2, "Z")]
                        out_b = batch.measure_or_reset(terms_m, _FixedRng(u_fixed), is_reset=False)  # type: ignore[arg-type]
                        for s in range(batch_size):
                            out_s = singles[s].measure_or_reset(terms_m, _FixedRng(u_fixed[s : s + 1]), is_reset=False)  # type: ignore[arg-type]
                            assert bool(out_b[s]) == bool(out_s[0])
                            denses[s].project(terms_m, int(out_b[s]), update_mass=True)
                else:
                    # Unmatched noisy reset (leaves log_mass unchanged!).
                    q = int(rng.integers(0, n))
                    pch_u: Any = "XYZ"[int(rng.integers(0, 3))]
                    rz = np.zeros((n, batch_size), dtype=np.bool_)
                    ro = np.zeros((n, batch_size), dtype=np.bool_)
                    act_u = rng.random(batch_size) < 0.5
                    want_1 = rng.random(batch_size) < 0.5
                    rz[q] = act_u & ~want_1
                    ro[q] = act_u & want_1
                    u_fixed = rng.random(batch_size)

                    class _FixedRngU:
                        def __init__(self, vals: np.ndarray) -> None:
                            self._v = vals
                        def random(self, sz: int) -> np.ndarray:
                            return self._v[:sz]

                    batch.unmatched_noisy_reset(rz, ro, _FixedRngU(u_fixed), pauli=pch_u, order=[q, q])  # type: ignore[arg-type]
                    kick_p = "Z" if pch_u == "X" else "X"
                    for s in range(batch_size):
                        singles[s].unmatched_noisy_reset(
                            rz[:, s : s + 1], ro[:, s : s + 1], _FixedRngU(u_fixed[s : s + 1]),  # type: ignore[arg-type]
                            pauli=pch_u, order=[q],
                        )
                        if act_u[s]:
                            p1_d = denses[s].peek_p1([(q, pch_u)])
                            p0_d = 1.0 - p1_d
                            # Match the sampled single-shot outcome via state comparison:
                            if p0_d <= 1e-13:
                                m_d = 1
                            elif p1_d <= 1e-13:
                                m_d = 0
                            else:
                                # Pick the branch matching singles[s].state_vector(0):
                                psi_target = singles[s].state_vector(0)
                                cand = []
                                for m_try in (0, 1):
                                    tmp = copy.deepcopy(denses[s])
                                    tmp.project([(q, pch_u)], m_try, update_mass=False)
                                    if bool(m_try) != bool(want_1[s]):
                                        tmp.apply_pauli_terms([(q, kick_p)])
                                    cand.append((_phase_aligned_diff(tmp.psi, psi_target), m_try, tmp))
                                cand.sort(key=lambda x: x[0])
                                denses[s] = cand[0][2]
                                continue
                            denses[s].project([(q, pch_u)], m_d, update_mass=False)
                            if bool(m_d) != bool(want_1[s]):
                                denses[s].apply_pauli_terms([(q, kick_p)])

            for s in range(batch_size):
                psi_b = batch.state_vector(s, endian="little", normalize=True)
                psi_s = singles[s].state_vector(0, endian="little", normalize=True)
                max_sv_diff = max(
                    max_sv_diff,
                    _phase_aligned_diff(psi_b, psi_s),
                    _phase_aligned_diff(psi_b, denses[s].psi),
                )
                m_b = batch.branch_mass(s)
                m_s = singles[s].branch_mass(0)
                m_d = math.exp(denses[s].log_mass)
                max_mass_diff = max(max_mass_diff, abs(m_b - m_s), abs(m_b - m_d))
                # Verify big-endian and unnormalized scaling.
                psi_un = batch.state_vector(s, endian="little", normalize=False)
                assert abs(float(np.linalg.norm(psi_un)) - math.sqrt(m_b)) < 1e-12
                psi_big = batch.state_vector(s, endian="big", normalize=True)
                psi_big_ref = psi_b.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).ravel()
                assert float(np.linalg.norm(psi_big - psi_big_ref)) < 1e-13

    assert max_peek_diff < 1e-12, f"max_peek_diff={max_peek_diff}"
    assert max_mass_diff < 1e-12, f"max_mass_diff={max_mass_diff}"
    assert max_sv_diff < 1e-12, f"max_sv_diff={max_sv_diff}"


# ---------------------------------------------------------------------------
# T5: Simulator debug trace forced replay against run_direct
# ---------------------------------------------------------------------------

def _replay_and_verify(
    circuit: stim.Circuit,
    *,
    batch_size: int = 12,
    seed: int = 12345,
    fault_channels: list[CliffordFaultChannel] | None = None,
    sweep_bits: np.ndarray | None = None,
    check_statevector: bool = True,
) -> tuple[float, float, float]:
    sim = MagicsideSimulator(
        circuit,
        batch_size=batch_size,
        seed=seed,
        fault_channels=fault_channels,
        sweep_bits=sweep_bits,
        _engine="branches",
    )
    sim._enable_branch_trace(True)
    sim.run()
    trace = sim._get_branch_trace()
    assert trace is not None
    recs = sim.get_final_measurement_records()

    plan = compile_direct(circuit, fault_channels=fault_channels)
    max_step_prob_diff = 0.0
    max_mass_diff = 0.0
    max_sv_diff = 0.0

    for s in range(batch_size):
        tr = trace.for_shot(s)
        sw_row = None if sweep_bits is None else sweep_bits[s]
        res = run_direct(
            plan,
            fault_trace=tr["fault_trace"],
            pauli_noise_trace=tr["pauli_noise_trace"],
            reset_outcomes=tr["reset_outcomes"],
            requested_quantum_outcomes=tr["requested_quantum_outcomes"],
            requested_outcomes=tr["requested_outcomes"],
            sweep_bits=sw_row,
        )
        assert np.array_equal(res.measurements, recs[s]), f"shot {s} measurement mismatch"
        assert len(res.born_masses) == len(tr["step_probs"]), (
            f"shot {s} born_masses len {len(res.born_masses)} != {len(tr['step_probs'])}"
        )
        q_log_mass_ref = 0.0
        for (m_r, z0, z1, z_in), (m_b, q_b, p_b) in zip(res.born_masses, tr["step_probs"]):
            assert m_r == m_b
            p_ref = (z1 if q_b else z0) / z_in
            max_step_prob_diff = max(max_step_prob_diff, abs(p_b - p_ref))
            q_log_mass_ref += math.log(p_ref)

        # Quantum branch mass matches product of run_direct quantum Born probs.
        max_mass_diff = max(max_mass_diff, abs(sim.get_branch_mass(s) - math.exp(q_log_mass_ref)))

        if check_statevector and circuit.num_qubits <= 8:
            psi_b = sim.state_vector(s, endian="little", normalize=True)
            psi_r = reconstruct_statevector(res)
            n_r = float(np.linalg.norm(psi_r))
            psi_r_norm = psi_r / n_r if n_r > 0.0 else psi_r
            max_sv_diff = max(max_sv_diff, _phase_aligned_diff(psi_b, psi_r_norm))

    assert max_step_prob_diff < 1e-10, f"max_step_prob_diff={max_step_prob_diff}"
    assert max_mass_diff < 1e-10, f"max_mass_diff={max_mass_diff}"
    assert max_sv_diff < 1e-10, f"max_sv_diff={max_sv_diff}"
    return max_step_prob_diff, max_mass_diff, max_sv_diff


def test_t5_replay_against_run_direct() -> None:
    loss_free_corpus = (
        "c01_clifford_noise_feedback",
        "c02_mpp_mxx_heralded_corr",
        "c02b_spp_clifford",
        "c06_magic_t_ccz_clifford",
        "c07_magic_inline_faults",
        "c10_magic_multiword_68q",
    )
    max_p = max_m = max_sv = 0.0
    for cname in loss_free_corpus:
        circ, has_leak = _corpus_circuit(cname)
        assert not has_leak
        dp, dm, dsv = _replay_and_verify(
            circ, batch_size=12, seed=777, check_statevector=(circ.num_qubits <= 8)
        )
        max_p, max_m, max_sv = max(max_p, dp), max(max_m, dm), max(max_sv, dsv)

    # Dedicated stress circuit: zero-prob noise slots, noisy readout + feedback, sweep feedback,
    # MPAD (noiseless + noisy), inverted targets, MXX 0 0, MR/MRX/MRY, programmatic fault channels.
    stress_circ = stim.Circuit(
        """
        R 0 1 2 3
        RX 4
        RY 5
        H 0 1 2 3
        X_ERROR(0.0) 0 1
        DEPOLARIZE1(0.0) 2 3
        DEPOLARIZE2(0.0) 0 1
        CORRELATED_ERROR(0.0) X0 Z1
        ELSE_CORRELATED_ERROR(0.25) Y1 Z2
        I[T] 0 1
        I[CCZ] 0 1 2
        I[T_DAG] 2
        SPP X0*Y1 !Z2*X3
        SPP_DAG Z1*Z3
        HERALDED_ERASE(0.15) 0 1
        HERALDED_PAULI_CHANNEL_1(0.05, 0.05, 0.05, 0.05) 2 3
        MPAD 0 1
        MPAD(0.1) 0 1
        M(0.08) !0 1
        CX rec[-1] 2
        CZ rec[-2] 3
        CY sweep[0] 4
        CX sweep[1] 5
        MXX(0.05) 0 1 1 2
        MYY 2 3
        MZZ !3 4
        MPP(0.05) X0*Z1 !Y2*Z3 X4*X4
        MR(0.04) !0
        MRX 4
        MRY !5
        I[T] 0 3
        H 0 1 2 3
        M 0 1 2 3
        """
    )
    prog_faults = [
        CliffordFaultChannel(
            0,
            (FaultBranch(0.25, "H", (0,)), FaultBranch(0.2, "S", (1,))),
        ),
        CliffordFaultChannel(
            10,
            (FaultBranch(0.3, "CX", (0, 2)),),
        ),
        CliffordFaultChannel(
            len(stress_circ),
            (FaultBranch(0.25, "CZ", (1, 3)),),
        ),
    ]
    sw = np.random.default_rng(99).integers(0, 2, size=(12, 2)).astype(np.bool_)
    dp, dm, dsv = _replay_and_verify(
        stress_circ, batch_size=12, seed=2026, fault_channels=prog_faults, sweep_bits=sw
    )
    max_p, max_m, max_sv = max(max_p, dp), max(max_m, dm), max(max_sv, dsv)

    # Empty circuit with programmatic fault at step=0.
    empty_circ = stim.Circuit()
    empty_fault = [
        CliffordFaultChannel(
            0,
            (FaultBranch(0.5, "H", (0,)),),
        )
    ]
    dp, dm, dsv = _replay_and_verify(empty_circ, batch_size=8, seed=11, fault_channels=empty_fault)
    max_p, max_m, max_sv = max(max_p, dp), max(max_m, dm), max(max_sv, dsv)

    assert max_p < 1e-12
    assert max_m < 1e-12
    assert max_sv < 1e-12


# ---------------------------------------------------------------------------
# T6: Statistical agreement against term engine (5-sigma)
# ---------------------------------------------------------------------------

def test_t6_statistics_vs_term_engine() -> None:
    n_shots = 4096
    batch_size = 1024
    cases: list[tuple[str, stim.Circuit]] = [
        ("c06", _corpus_circuit("c06_magic_t_ccz_clifford")[0]),
        ("c07", _corpus_circuit("c07_magic_inline_faults")[0]),
        ("c10", _corpus_circuit("c10_magic_multiword_68q")[0]),
    ]
    for k in range(7):
        cases.append((f"xmem_t({k})", _xmem_t(k)))
    cases.append(("random_t_cx(8)", _random_t_cx(8)))

    worst_z = 0.0
    for idx, (label, circ) in enumerate(cases):
        r_t, d_t, o_t = _sample_engine(circ, engine="terms", n_shots=n_shots, batch_size=batch_size, seed=1000 + idx)
        r_b, d_b, o_b = _sample_engine(circ, engine="branches", n_shots=n_shots, batch_size=batch_size, seed=2000 + idx)
        z_r = _max_z(r_t, r_b)
        z_d = _max_z(d_t, d_b) if d_t.shape[1] else 0.0
        z_o = _max_z(o_t, o_b) if o_t.shape[1] else 0.0
        z_case = max(z_r, z_d, z_o)
        worst_z = max(worst_z, z_case)
        assert z_case < 5.0, f"T6 failed on {label}: z={z_case:.3f}"
    assert worst_z < 5.0


# ---------------------------------------------------------------------------
# T7: Clifford circuits against Stim's sampler
# ---------------------------------------------------------------------------

def test_t7_clifford_circuits_vs_stim() -> None:
    n_shots = 4096
    batch_size = 1024
    circuits = [
        ("c01", _corpus_circuit("c01_clifford_noise_feedback")[0]),
        ("c02", _corpus_circuit("c02_mpp_mxx_heralded_corr")[0]),
        ("c02b", _corpus_circuit("c02b_spp_clifford")[0]),
        (
            "rot_sc_z_d3_r3",
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=3,
                rounds=3,
                after_clifford_depolarization=0.005,
            ),
        ),
    ]
    for idx, (label, circ) in enumerate(circuits):
        r_b, d_b, o_b = _sample_engine(
            circ, engine="branches", n_shots=n_shots, batch_size=batch_size, seed=3000 + idx
        )
        r_s = circ.compile_sampler(seed=4000 + idx).sample(n_shots)
        d_s, o_s = circ.compile_detector_sampler(seed=5000 + idx).sample(
            n_shots, separate_observables=True
        )
        z_r = _max_z(r_b, r_s)
        z_d = _max_z(d_b, d_s) if d_s.shape[1] else 0.0
        z_o = _max_z(o_b, o_s) if o_s.shape[1] else 0.0
        assert max(z_r, z_d, z_o) < 5.0, f"T7 failed on {label}: z={(z_r, z_d, z_o)}"


# ---------------------------------------------------------------------------
# T9: Golden mode & public API invariants
# ---------------------------------------------------------------------------

def test_t9_golden_mode_inventory_invariants() -> None:
    assert len(golden_test.CORPUS) == 18
    assert len(stimside.__all__) == 38
    digests = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))
    branch_circuits = [name for name, modes, _, _ in golden_test.CORPUS if "magic_branch" in modes]
    ds_circuits = [name for name, modes, _, _ in golden_test.CORPUS if "magic_ds" in modes]
    assert branch_circuits == ds_circuits
    assert len(branch_circuits) == 14
    for name in branch_circuits:
        assert "magic_branch" in digests[name]
        assert len(digests[name]["magic_branch"]) == 32


# ---------------------------------------------------------------------------
# T10: Determinism (warm vs cold vs 50-node cap) & switches / env var
# ---------------------------------------------------------------------------

def test_t10_determinism_warm_cold_small_cap_and_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    for label, circ, has_leak in (
        ("c06", _corpus_circuit("c06_magic_t_ccz_clifford")[0], False),
        ("xmem_t(3)", _xmem_t(3), False),
        ("c08", _corpus_circuit("c08_magic_with_leakage")[0], True),
    ):
        def _run_4_batches(cap: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[float]], dict[str, Any]]:
            monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", cap)
            op_h = LeakageUint8Magic() if has_leak else None
            sim = MagicsideSimulator(circ, op_handler=op_h, batch_size=32, seed=9001, _engine="branches")
            recs, dets, obs, masses = [], [], [], []
            for b in range(4):
                if b > 0:
                    sim.clear()
                sim.run()
                recs.append(sim.get_final_measurement_records())
                dets.append(sim.get_detector_flips())
                obs.append(sim.get_observable_flips())
                masses.append([sim.get_branch_mass(s) for s in range(32)])
            return recs, dets, obs, masses, sim._branch_counters()

        r_warm, d_warm, o_warm, m_warm, c_warm = _run_4_batches(300_000)
        r_cap50, d_cap50, o_cap50, m_cap50, c_cap50 = _run_4_batches(50)
        r_cold, d_cold, o_cold, m_cold, c_cold = _run_4_batches(0)

        assert c_warm["store_clears"] == 0, label
        assert c_cold["store_clears"] == 3, label
        if c_warm["store_nodes"] > 50:
            assert c_cap50["store_clears"] >= 1, label

        for b in range(4):
            assert np.array_equal(r_warm[b], r_cap50[b]), (label, b)
            assert np.array_equal(r_warm[b], r_cold[b]), (label, b)
            assert np.array_equal(d_warm[b], d_cap50[b]), (label, b)
            assert np.array_equal(d_warm[b], d_cold[b]), (label, b)
            assert np.array_equal(o_warm[b], o_cap50[b]), (label, b)
            assert np.array_equal(o_warm[b], o_cold[b]), (label, b)
            for s in range(32):
                assert abs(m_warm[b][s] - m_cap50[b][s]) < 1e-14
                assert abs(m_warm[b][s] - m_cold[b][s]) < 1e-14

    # Verify monkeypatching cap on branch_backend and branch_kernels also works.
    monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", 300_000)
    monkeypatch.setattr(bb, "_BRANCH_STORE_MAX_NODES", 1)
    assert bb.effective_store_max_nodes(sm._BRANCH_STORE_MAX_NODES) == 1
    monkeypatch.setattr(bb, "_BRANCH_STORE_MAX_NODES", 300_000)
    monkeypatch.setattr(bk, "_BRANCH_STORE_MAX_NODES", 2)
    assert bb.effective_store_max_nodes(sm._BRANCH_STORE_MAX_NODES) == 2
    monkeypatch.setattr(bk, "_BRANCH_STORE_MAX_NODES", 300_000)

    # Verify _BRANCH_ENGINE monkeypatch & _engine override & invalid _engine ValueError.
    c06, _ = _corpus_circuit("c06_magic_t_ccz_clifford")
    monkeypatch.setattr(sm, "_BRANCH_ENGINE", True)
    sim_br = MagicsideSimulator(c06, batch_size=4, seed=7)
    assert sim_br._branch_batch is not None
    sim_tm = MagicsideSimulator(c06, batch_size=4, seed=7, _engine="terms")
    assert sim_tm._branch_batch is None
    with pytest.raises(ValueError, match="_engine"):
        MagicsideSimulator(c06, _engine="bogus")
    with pytest.raises(ValueError, match="_engine"):
        MagicsideSimulator(c06, _engine=123)  # type: ignore[arg-type]

    # Verify _compute_noiseless_magic_ref is invariant when _BRANCH_ENGINE=True.
    c08, _ = _corpus_circuit("c08_magic_with_leakage")
    sim_c08_br = MagicsideSimulator(c08, op_handler=LeakageUint8Magic(), batch_size=2, seed=5, _engine="branches")
    sim_c08_tm = MagicsideSimulator(c08, op_handler=LeakageUint8Magic(), batch_size=2, seed=5, _engine="terms")
    assert np.array_equal(sim_c08_br.get_reference_sample(), sim_c08_tm.get_reference_sample())

    # Subprocess checks for STIMSIDE_MAGICSIDE_ENGINE env var (branches, terms, invalid) + MagicsideSampler.
    src_dir = str(Path(__file__).resolve().parents[1])
    base_env = dict(os.environ)
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    base_env["PYTHONPATH"] = src_dir

    for env_val, expect_branch in (("branches", True), ("terms", False), ("", False)):
        env = dict(base_env, STIMSIDE_MAGICSIDE_ENGINE=env_val)
        code = (
            "import sinter, stim; from stimside import simulator_magic as sm, MagicsideSampler; "
            f"assert sm._BRANCH_ENGINE is {expect_branch}; "
            "c = stim.Circuit('R 0\\nH 0\\nI[T] 0\\nH 0\\nX_ERROR(0.05) 0\\nM 0\\nDETECTOR rec[-1]\\nOBSERVABLE_INCLUDE(0) rec[-1]\\n'); "
            "s = sm.MagicsideSimulator(c, batch_size=4, seed=1); "
            f"assert (s._branch_batch is not None) is {expect_branch}; "
            "s.run(); "
            "comp = MagicsideSampler(batch_size=4, seed=1).compiled_sampler_for_task(sinter.Task(circuit=c)); "
            "stats = comp.sample(4); assert stats.shots == 4"
        )
        proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"env={env_val!r} stderr={proc.stderr}"

    env_bad = dict(base_env, STIMSIDE_MAGICSIDE_ENGINE="not_a_mode")
    proc_bad = subprocess.run(
        [sys.executable, "-B", "-c", "import stimside.simulator_magic"],
        env=env_bad,
        capture_output=True,
        text=True,
    )
    assert proc_bad.returncode != 0
    assert "ValueError" in proc_bad.stderr


# ---------------------------------------------------------------------------
# T12: Section 2.9 guards, budget, postselection, and state_vector bounds
# ---------------------------------------------------------------------------

def test_t12_section29_guards_budget_postselect_and_statevector_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circ = stim.Circuit("R 0 1\nH 0 1\nI[T] 0 1\nCX 0 1\nH 0 1\nM 0 1\n")

    # Construction ValueError guards.
    with pytest.raises(ValueError, match="QUERY"):
        MagicsideSimulator(circ, backend="QUERY", _engine="branches")
    with pytest.raises(ValueError, match="SELECTOR"):
        MagicsideSimulator(circ, fault_mode="SELECTOR", _engine="branches")
    with pytest.raises(ValueError, match="fault_realizations"):
        MagicsideSimulator(circ, fault_realizations={"faults": {}}, _engine="branches")
    with pytest.raises(ValueError, match="fault_realizations"):
        MagicsideSimulator(circ, fault_realizations={}, _engine="branches")
    with pytest.raises(ValueError, match="sync_tableside_rng"):
        MagicsideSimulator(circ, sync_tableside_rng=True, _engine="branches")

    # clear() also enforces guards if _BRANCH_ENGINE flips after construction.
    sim_sync = MagicsideSimulator(circ, sync_tableside_rng=True)
    monkeypatch.setattr(sm, "_BRANCH_ENGINE", True)
    with pytest.raises(ValueError, match="sync_tableside_rng"):
        sim_sync.clear()
    monkeypatch.setattr(sm, "_BRANCH_ENGINE", False)

    # Method NotImplementedError guards.
    sim = MagicsideSimulator(circ, batch_size=4, seed=1, _engine="branches")
    with pytest.raises(NotImplementedError):
        sim.interactive_do(stim.Circuit("H 0"))
    with pytest.raises(NotImplementedError):
        sim.finish_interactive_run()
    sim.run()
    with pytest.raises(NotImplementedError):
        sim.get_qfe_expansion(0)
    with pytest.raises(NotImplementedError):
        sim.get_state_expansions()

    # Term-mode private branch getters raise RuntimeError.
    sim_t = MagicsideSimulator(circ, batch_size=2, seed=1, _engine="terms")
    with pytest.raises(RuntimeError):
        sim_t._get_branch_state(0)
    with pytest.raises(RuntimeError):
        sim_t._branch_counters()
    with pytest.raises(RuntimeError):
        sim_t._branch_peak_per_shot()

    # Budget exceeded -> QueryBudgetExceededError (and fallback/unlimited succeeds).
    deep_magic = _random_t_cx(n=8, layers=10, p_t=0.4, seed=13, p_noise=0.0)
    sim_bud = MagicsideSimulator(
        deep_magic, batch_size=4, seed=1, max_components=2, on_budget_exceeded="raise", _engine="branches"
    )
    with pytest.raises(QueryBudgetExceededError):
        sim_bud.run()

    sim_fb = MagicsideSimulator(
        deep_magic, batch_size=4, seed=1, max_components=2, on_budget_exceeded="fallback", _engine="branches"
    )
    sim_fb.run()
    assert sim_fb.get_final_measurement_records().shape == (4, 8)

    # ImpossiblePostselectionError on postselect_z_batch without mutating state.
    sim_ps = MagicsideSimulator(stim.Circuit("R 0 1\nX 1\n"), batch_size=3, seed=1, _engine="branches")
    sim_ps.run()
    node_before = sim_ps._branch_batch.node.copy()  # type: ignore[union-attr]
    lm_before = sim_ps._branch_batch.log_mass.copy()  # type: ignore[union-attr]
    with pytest.raises(ImpossiblePostselectionError):
        sim_ps.postselect_z_batch(
            0,
            np.array([True, True, False], dtype=np.bool_),
            np.array([0, 1, 0], dtype=np.uint8),  # shot 1 wants Z_0 = 1 on |0>, impossible!
        )
    assert np.array_equal(sim_ps._branch_batch.node, node_before)  # type: ignore[union-attr]
    assert np.array_equal(sim_ps._branch_batch.log_mass, lm_before)  # type: ignore[union-attr]

    # state_vector bounds: n == 0, n > 26, endian validation, shot index bounds.
    sim_0 = MagicsideSimulator(stim.Circuit(), batch_size=2, seed=1, _engine="branches")
    sim_0.run()
    sv0 = sim_0.state_vector(0)
    assert sv0.shape == (1,) and abs(sv0[0] - 1.0) < 1e-14
    assert np.array_equal(sim_0.state_vector(-1), sv0)
    with pytest.raises(IndexError):
        sim_0.state_vector(2)
    with pytest.raises(ValueError, match="endian"):
        sim_0.state_vector(0, endian="middle")  # type: ignore[arg-type]

    sim_27 = MagicsideSimulator(stim.Circuit("H 26\nM 26\n"), batch_size=1, seed=1, _engine="branches")
    sim_27.run()
    with pytest.raises(ValueError, match="26"):
        sim_27.state_vector(0)


# ---------------------------------------------------------------------------
# T13: copy.copy, shared store cap replacement, sole-owner clear, threads
# ---------------------------------------------------------------------------

def test_t13_copy_clear_shared_store_and_multithreaded_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circ = _xmem_t(3)
    sim1 = MagicsideSimulator(circ, batch_size=16, seed=42, _engine="branches")
    sim1.run()
    sim2 = copy.copy(sim1)
    assert sim2._branch_store is sim1._branch_store
    assert sim2._branch_batch is not sim1._branch_batch
    assert sim1._branch_store is not None and sim1._branch_store.owner_count() == 2

    # Save sim1 post-run branch state & state_vector before sim2 triggers a cap overflow!
    c06, _ = _corpus_circuit("c06_magic_t_ccz_clifford")
    s_a = MagicsideSimulator(c06, batch_size=16, seed=77, _engine="branches")
    s_a.run()
    sv_a_before = [s_a.state_vector(i) for i in range(4)]
    st_a_before = s_a._get_branch_state(0)
    s_b = copy.copy(s_a)
    assert s_b._branch_store is s_a._branch_store
    shared_old = s_a._branch_store
    assert shared_old is not None and shared_old.native.node_count > 5

    # Lower cap to 1 so s_b.clear() sees shared store above cap -> must replace s_b's store,
    # keeping s_a's store intact so s_a's post-run node IDs NEVER dangle!
    monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", 1)
    s_b.clear()
    assert s_b._branch_store is not shared_old
    assert s_a._branch_store is shared_old
    assert s_b._branch_counters()["store_clears"] >= 1
    for i in range(4):
        assert float(np.linalg.norm(s_a.state_vector(i) - sv_a_before[i])) < 1e-14
    st_a_after = s_a._get_branch_state(0)
    assert st_a_after.node_id == st_a_before.node_id

    # Now s_a is sole owner of shared_old -> s_a.clear() clears IN-PLACE!
    assert shared_old.owner_count() == 1
    s_a.clear()
    assert s_a._branch_store is shared_old
    assert shared_old.native.node_count == 1
    assert s_a._branch_counters()["store_clears"] >= 1
    monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", 300_000)

    # Multi-threaded concurrent run() on copies sharing one _SharedBranchStore matches single-thread.
    base_sim = MagicsideSimulator(circ, batch_size=32, seed=555, _engine="branches")
    copies = [copy.copy(base_sim) for _ in range(4)]
    for i, sc in enumerate(copies):
        sc.seed = 600 + i
        sc.clear()

    def _run_one(sc: MagicsideSimulator) -> np.ndarray:
        sc.run()
        return sc.get_final_measurement_records()

    with ThreadPoolExecutor(max_workers=4) as ex:
        mt_recs = list(ex.map(_run_one, copies))

    for i in range(4):
        ref = MagicsideSimulator(circ, batch_size=32, seed=600 + i, _engine="branches")
        ref.run()
        assert np.array_equal(mt_recs[i], ref.get_final_measurement_records())

    # Repeated clear() across 5 batches with LeakageUint8Magic.
    c08, _ = _corpus_circuit("c08_magic_with_leakage")
    sim_l = MagicsideSimulator(c08, op_handler=LeakageUint8Magic(), batch_size=16, seed=88, _engine="branches")
    for b in range(5):
        if b > 0:
            sim_l.clear()
        sim_l.run()
    assert sim_l._batches_completed == 5
    cnt_l = sim_l._branch_counters()
    assert sum(cnt_l["branches_hist"].values()) == 5 * 16


# ---------------------------------------------------------------------------
# Section 7 counters, _get_branch_state, multi-word n = 70 / n = 130, leak check
# ---------------------------------------------------------------------------

def test_section7_counters_branch_state_multiword_and_no_leaks() -> None:
    c08, _ = _corpus_circuit("c08_magic_with_leakage")
    sim = MagicsideSimulator(c08, op_handler=LeakageUint8Magic(), batch_size=32, seed=314, _engine="branches")
    for b in range(2):
        if b > 0:
            sim.clear()
        sim.run()

    cnt = sim._branch_counters()
    expected_keys = {
        "store_nodes",
        "store_bytes_estimate",
        "store_clears",
        "transitions_new",
        "transitions_hit",
        "resolve_calls",
        "splits",
        "subcache_hits",
        "subcache_misses",
        "meas_definite",
        "meas_absorbed",
        "meas_forked",
        "branches_hist",
        "faults_realized",
    }
    assert set(cnt.keys()) == expected_keys
    assert set(cnt["transitions_new"].keys()) == set(bk.MBR_KIND_NAMES)
    assert set(cnt["transitions_hit"].keys()) == set(bk.MBR_KIND_NAMES)
    assert set(cnt["faults_realized"].keys()) == {"pauli_faults", "skipped_gates", "skipped_cczs"}
    assert cnt["store_nodes"] >= 1
    assert cnt["store_bytes_estimate"] > 0
    assert sum(cnt["branches_hist"].values()) == 2 * 32
    assert cnt["faults_realized"]["skipped_gates"] > 0
    assert cnt["faults_realized"]["skipped_cczs"] > 0

    peaks = sim._branch_peak_per_shot()
    assert peaks.dtype == np.int16 and peaks.shape == (32,)
    assert int(peaks.min()) >= 1

    # Inspect _get_branch_state snapshot and confirm zero BranchState leaks.
    snap = sim._get_branch_state(0)
    assert isinstance(snap, bb.BranchStateSnapshot)
    assert snap.n_branches == len(snap.branches)
    assert snap.peak == int(peaks[0])
    for w, comp in snap.branches:
        assert isinstance(w, complex)
        assert isinstance(comp, PhaseQFEComponent)
    assert bk.BranchState.live_states() == 0

    # Multi-word n = 70 and n = 130 straddling 64-bit word boundaries (words 0, 1, 2).
    for nq, qs in ((70, (0, 63, 64, 69)), (130, (0, 63, 64, 129))):
        q0, q1, q2, q3 = qs
        mw_circ = stim.Circuit(
            f"""
            R {q0} {q1} {q2} {q3}
            H {q0} {q1} {q2} {q3}
            CX {q0} {q2} {q1} {q3}
            SPP X{q0}*Z{q2} !Y{q1}*X{q3}
            I[T] {q0} {q3}
            I[CLIFFORD_FAULT: (0.25, CX)] {q1} {q2}
            I[CCZ] {q0} {q1} {q3}
            DEPOLARIZE2(0.05) {q0} {q2} {q1} {q3}
            MPP X{q0}*X{q2} Z{q1}*Z{q3}
            MR {q0} {q2}
            H {q1} {q3}
            M {q0} {q1} {q2} {q3}
            """
        )
        _replay_and_verify(mw_circ, batch_size=8, seed=nq, check_statevector=False)

    # Verify POST on an ABSORBED state with inner_m == 1 (via X frame flip + desired=0)
    # and all-False skip masks on Clifford, SPP, and CCZ.
    sim_abs = MagicsideSimulator(stim.Circuit("R 0 1 2\nH 0 1 2\n"), batch_size=2, seed=1, _engine="branches")
    sim_abs.run()
    assert sim_abs._branch_batch is not None
    # Flip shot 1's X frame so Z_0 anticommutes with F_1 (s_bit=1); desired=0 -> inner_m=1 on ABSORBED!
    sim_abs.apply_pauli("X", np.array([[False, True], [False, False], [False, False]], dtype=np.bool_))
    sim_abs.postselect_z_batch(
        0,
        np.array([True, True], dtype=np.bool_),
        np.array([0, 0], dtype=np.uint8),
        pauli="Z",
    )
    for s in (0, 1):
        assert abs(sim_abs.get_branch_mass(s) - 0.5) < 1e-13
        # Qubit 0 must be in |0> (p1 == 0.0).
        assert sim_abs.peek_pauli_born_batch([0], "Z")[0, s] < 1e-13
    # All-False skip masks on 2Q unitary, SPP, and CCZ leave physical state unchanged.
    sv_pre = [sim_abs.state_vector(s) for s in (0, 1)]
    all_false = np.zeros((1, 2), dtype=np.bool_)
    sim_abs.do_conditional_clifford(stim.CircuitInstruction("CX", [0, 1]), all_false)
    sim_abs.do_conditional_clifford(
        stim.CircuitInstruction("SPP", [stim.target_x(0), stim.target_combiner(), stim.target_z(1)]),
        all_false,
    )
    sim_abs.do_conditional_magic(stim.CircuitInstruction("I", [0, 1, 2], tag="CCZ"), [0, 1, 2], all_false)
    for s in (0, 1):
        assert _phase_aligned_diff(sim_abs.state_vector(s), sv_pre[s]) < 1e-13
    c_abs = sim_abs._branch_counters()
    assert c_abs["meas_absorbed"] >= 1
    assert c_abs["transitions_new"]["POST"] >= 1
    assert c_abs["transitions_new"]["PEEK"] >= 1
    assert c_abs["transitions_new"]["SKIP"] >= 2

    # Release all local simulator references and assert zero native handle leaks.
    del sim, snap, sim_abs
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


def test_mpp_repeated_qubit_neg_sign_replay() -> None:
    """Regression test: MPP repeated-qubit neg=True stays separate from `!` inv in replay & cache."""
    cases = [
        "H 0\nI[T] 0\nMPP X0*Z0*X0",
        "H 0\nI[T] 0\nMPP(0.25) X0*Z0*X0",
        "H 0\nI[T] 0\nMPP X0*Z0*X0*Z0",
        "H 0\nI[T] 0\nMPP !X0*Z0*X0",
        "H 0\nI[T] 0\nMPP(0.25) !X0*Z0*X0*Z0",
        "H 0\nI[T] 0\nMPP Z0 X0*Z0*X0",
        "H 0\nI[T] 0\nMPP X0*Z0*X0 Z0",
    ]
    for idx, c_str in enumerate(cases):
        circ = stim.Circuit(c_str)
        ms, mm, msv = _replay_and_verify(circ, batch_size=12, seed=42 + idx, check_statevector=True)
        assert ms < 1e-12, f"{c_str!r}: max_step_prob_diff={ms}"
        assert mm < 1e-12, f"{c_str!r}: max_mass_diff={mm}"
        assert msv < 1e-12, f"{c_str!r}: max_sv_diff={msv}"

    # Verify event key distinguishes +Z0 vs -Z0 (X0*Z0*X0) on the same frame_id.
    sim_pair = MagicsideSimulator(
        stim.Circuit("H 0\nI[T] 0\nMPP Z0 X0*Z0*X0"), batch_size=16, seed=99, _engine="branches"
    )
    sim_pair.run()
    recs = sim_pair.get_final_measurement_records()
    assert np.all(recs[:, 0] ^ recs[:, 1])

    del sim_pair
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


def test_branch_engine_toggle_true_to_false_clear_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: toggling _BRANCH_ENGINE True -> False before clear() resets all _branch_* state."""
    monkeypatch.setattr(sm, "_BRANCH_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nM 0"), batch_size=4, seed=7)
    sim.run()
    assert sim._branch_batch is not None
    assert sim._branch_store is not None
    assert sim._branch_counters()["store_nodes"] >= 1

    monkeypatch.setattr(sm, "_BRANCH_ENGINE", False)
    sim.clear()
    gc.collect()

    branch_attrs = (
        "_branch_batch",
        "_branch_store",
        "_branch_store_token",
        "_branch_retired_counters",
        "_branch_shot_hist",
        "_branch_faults_realized",
    )
    for attr in branch_attrs:
        assert getattr(sim, attr) is None, f"{attr} not reset to None"
    assert bk.BranchStore.live_stores() == 0

    for method in ("_get_branch_state", "_branch_peak_per_shot", "_branch_counters"):
        with pytest.raises(RuntimeError):
            getattr(sim, method)()

    sim.run()
    sim_copy = copy.copy(sim)
    for attr in branch_attrs:
        assert getattr(sim, attr) is None
        assert getattr(sim_copy, attr) is None

    sim.clear()
    sim_copy.clear()
    sim.run()
    sim_copy.run()
    assert sim.get_final_measurement_records().shape == (4, 1)
    assert sim_copy.get_final_measurement_records().shape == (4, 1)
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


def test_branch_engine_toggle_false_to_true_copy_and_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: _BRANCH_ENGINE False -> True -> copy.copy() -> False never aliases _live_store."""
    monkeypatch.setattr(sm, "_BRANCH_ENGINE", False)
    circ = stim.Circuit("H 0 1\nI[T] 0\nCX 0 1\nI[T] 1\nM 0 1")
    s1 = MagicsideSimulator(circ, batch_size=8, seed=111)
    s1.run()

    monkeypatch.setattr(sm, "_BRANCH_ENGINE", True)
    s1.clear()
    assert "_live_store" not in s1.__dict__
    s1.run()

    s2 = copy.copy(s1)
    assert "_live_store" not in s2.__dict__

    monkeypatch.setattr(sm, "_BRANCH_ENGINE", False)
    s1.seed, s2.seed = 111, 222
    s1.clear()
    s2.clear()
    assert s1._live_exps is not s2._live_exps

    s1.run()
    sv_before = [s1.state_vector(i).copy() for i in range(8)]
    s2.run()
    sv_after = [s1.state_vector(i).copy() for i in range(8)]
    assert max(float(np.linalg.norm(a - b)) for a, b in zip(sv_before, sv_after)) == 0.0

    del s1, s2
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0
