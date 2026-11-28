# Copyright 2026 Google LLC
"""WP7 T8 loss end-to-end tests for the stabilizer-branch engine in MagicsideSimulator."""

from __future__ import annotations

import gc
import json
import math
from typing import Any, Callable, Literal
import warnings

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import CosetsideSimulator, MagicsideSimulator
from stimside._magic.gates import decompose_clifford_exact, eval_decomposition_unitary_le
from stimside._magic.phase_qfe import apply_local_unitary_c128
from stimside._magic.statevector import _apply_pauli_to_dense_le
from stimside._native import branch_kernels as bk
from stimside import golden_test
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
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
    """Pooled two-proportion z over all bit means and pairwise XORs (m <= 48)."""
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


def _corpus_circuit(name: str) -> tuple[stim.Circuit, bool]:
    for cname, _modes, has_leak, text in golden_test.CORPUS:
        if cname == name:
            return stim.Circuit(text), has_leak
    raise KeyError(name)


def _sample_magic(
    circuit: stim.Circuit,
    *,
    engine: Literal["terms", "branches"],
    n_shots: int,
    batch_size: int,
    seed: int,
    op_handler_factory: Callable[[], LeakageUint8Magic],
    suppress_warnings: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recs, dets, obs = [], [], []
    n_batches = n_shots // batch_size
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore", UserWarning)
        sim = MagicsideSimulator(
            circuit,
            op_handler=op_handler_factory(),
            batch_size=batch_size,
            seed=seed,
            _engine=engine,
        )
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


def _sample_coset(
    circuit: stim.Circuit,
    *,
    n_shots: int,
    batch_size: int,
    seed: int,
    unconditional_condition_on_U: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    handler = LeakageUint8Coset(
        unconditional_condition_on_U=unconditional_condition_on_U
    ).compile_op_handler(circuit=circuit, batch_size=batch_size)
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch_size,
        seed=seed,
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


def _build_leaky_rotated_surface_code(
    distance: int = 3, rounds: int = 2, p_pauli: float = 0.005, p_leak: float = 0.02
) -> stim.Circuit:
    """Rotated Z-memory surface code with U-->2 loss, 2-->U seepage, and leakage projection."""
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_pauli,
        before_measure_flip_probability=p_pauli,
        after_reset_flip_probability=p_pauli,
    ).flattened()
    mr_qubits = {t.value for inst in base if inst.name == "MR" for t in inst.targets_copy()}
    data_qubits = sorted(set(range(base.num_qubits)) - mr_qubits)
    out = stim.Circuit()
    for inst in base:
        name, targets, args = inst.name, inst.targets_copy(), inst.gate_args_copy()
        if name == "CX":
            out.append("CX", targets)
            out.append("I_ERROR", [t.value for t in targets], [0.0], tag=f"LEAKAGE_TRANSITION_1: ({p_leak}, U-->2)")
        elif name == "MR":
            out.append("M", targets, args, tag="LEAKAGE_PROJECTION_Z: (1.0, 2)")
            out.append("R", targets)
            out.append("I", targets, [], tag="LEAKAGE_TRANSITION_1: (1.0, 2-->0)")
            out.append("I", data_qubits, [], tag="LEAKAGE_TRANSITION_1: (0.4, 2-->U)")
        elif name == "M":
            out.append("M", targets, args, tag="LEAKAGE_PROJECTION_Z: (1.0, 2)")
        else:
            out.append(inst)
    return out


class _DenseLeakageOracle:
    """Independent dense complex128 statevector oracle (little-endian) for T8 verification."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.psi = np.zeros(1 << n, dtype=np.complex128)
        self.psi[0] = 1.0 + 0.0j
        self.log_mass = 0.0

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

    def apply_unitary(self, canon: str, grp: tuple[int, ...]) -> None:
        u = eval_decomposition_unitary_le(decompose_clifford_exact(canon))
        self.psi = apply_local_unitary_c128(self.psi, u, grp, self.n, "little")

    def apply_pauli(self, q: int, pauli: str) -> None:
        self.psi = self._pauli_vec([(q, pauli)])

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

    def peek_p1(self, q: int, pauli: str) -> float:
        pv = self._pauli_vec([(q, pauli)])
        proj1 = 0.5 * (self.psi - pv)
        return float(np.real(np.vdot(proj1, proj1)))

    def project(self, q: int, pauli: str, outcome: int, *, update_mass: bool = True) -> float:
        pv = self._pauli_vec([(q, pauli)])
        proj = 0.5 * (self.psi + (1.0 if outcome == 0 else -1.0) * pv)
        p = float(np.real(np.vdot(proj, proj)))
        assert p > 1e-13
        self.psi = np.ascontiguousarray(proj / math.sqrt(p))
        if update_mass:
            self.log_mass += math.log(p)
        return p

    def computational_prob(self, bits: np.ndarray) -> float:
        idx = 0
        for q, bit in enumerate(bits):
            if int(bit):
                idx |= 1 << q
        amp = self.psi[idx]
        return float(amp.real * amp.real + amp.imag * amp.imag)


def _assert_no_native_leaks() -> None:
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


# ---------------------------------------------------------------------------
# Test 1: Clifford + leakage circuits vs CosetsideSimulator
# ---------------------------------------------------------------------------


def test_t8_clifford_leakage_vs_cosetside() -> None:
    n_shots, batch_size = 4096, 1024

    circ_uc_false = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2
        CX 0 1 2 3
        I_ERROR[LEAKAGE_TRANSITION_1: (0.3, U-->2) (0.2, 1-->3)] 0 1 2
        CX 0 2 1 3
        H 0 3
        DEPOLARIZE1(0.05) 0 1 2 3
        I_ERROR[LEAKAGE_TRANSITION_1: (0.35, 2-->U) (0.25, 3-->0)] 0 1 2
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2) (0.85, 3): 0 1 2] 0 0 0
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.88, 2) (0.95, 3)] 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        DETECTOR rec[-3] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-4]
        """
    )

    cases: list[tuple[str, stim.Circuit, bool]] = [
        ("c03", _corpus_circuit("c03_leakage_transitions_swap_meas")[0], True),
        ("c04", _corpus_circuit("c04_leakage_conditioning_and_repeated_targets")[0], True),
        ("c09_70q", _corpus_circuit("c09_multiword_70q_leakage")[0], True),
        ("c12", _corpus_circuit("c12_overlapping_lt2_pauli_xor")[0], True),
        ("c16_20q", _corpus_circuit("c16_sparse_multibatch")[0], True),
        ("leaky_sc_d3_r2", _build_leaky_rotated_surface_code(distance=3, rounds=2), True),
        ("uc_false_4q", circ_uc_false, False),
    ]

    for idx, (label, circ, uc) in enumerate(cases):
        r_b, d_b, o_b = _sample_magic(
            circ,
            engine="branches",
            n_shots=n_shots,
            batch_size=batch_size,
            seed=7100 + idx,
            op_handler_factory=lambda uc=uc: LeakageUint8Magic(unconditional_condition_on_U=uc),
        )
        r_c, d_c, o_c = _sample_coset(
            circ,
            n_shots=n_shots,
            batch_size=batch_size,
            seed=8100 + idx,
            unconditional_condition_on_U=uc,
        )
        z_r = _max_z(r_b, r_c)
        z_d = _max_z(d_b, d_c) if d_b.shape[1] else 0.0
        z_o = _max_z(o_b, o_c) if o_b.shape[1] else 0.0
        assert max(z_r, z_d, z_o) < 5.0, f"T8 Cosetside mismatch on {label}: z={(z_r, z_d, z_o)}"

    _assert_no_native_leaks()


# ---------------------------------------------------------------------------
# Test 2: Leaky magic circuits vs term engine (with WP1)
# ---------------------------------------------------------------------------


def test_t8_leaky_magic_vs_term_engine() -> None:
    n_shots, batch_size = 4096, 1024

    custom_6q = stim.Circuit(
        """
        R 0 1 2 3 4 5
        H 0 1 2 3 4 5
        I[T] 0 2
        I[T_DAG] 1 4
        CX 0 1 2 3 4 5
        I_ERROR[LEAKAGE_TRANSITION_1: (0.25, U-->2)] 0 1 2 3 4 5
        I[CCZ] 0 1 2 2 3 4 4 5 0
        HERALDED_ERASE(0.08) 0 !3
        HERALDED_PAULI_CHANNEL_1(0.02, 0.02, 0.02, 0.02) 1 !4
        I[T] 3
        I[T_DAG] 5
        CX 1 2 3 4
        I_ERROR[LEAKAGE_TRANSITION_1: (0.3, 2-->0) (0.25, 2-->1) (0.25, 2-->U)] 0 1 2 3 4 5
        MR[LEAKAGE_PROJECTION_Z: (0.9, 2)] 4 5
        H 0 1 2 3
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.95, 2)] 0 1 2 3
        DETECTOR rec[-1] rec[-5]
        DETECTOR rec[-2] rec[-6]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )

    multiword_68q = stim.Circuit(
        """
        R 0 63 64 67
        H 0 63 64 67
        I[T] 0 64
        I_ERROR[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 63 64 67
        I[CCZ] 0 63 67
        I[T] 64
        CX 0 64
        I_ERROR[LEAKAGE_TRANSITION_1: (0.25, 2-->U)] 0 67
        H 0 63 64 67
        M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 63 64 67
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )

    uc_false_4q = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        I[T] 0 1
        I_ERROR[LEAKAGE_TRANSITION_1: (0.35, U-->2)] 0 1 2
        I[CCZ] 0 1 2
        I[T_DAG] 2 3
        CX 1 3
        H 0 1 2 3
        M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 1 2 3
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4]
        """
    )

    fast_leak_4q = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        I[T] 0 1
        H 0 1
        S 0 1
        I_ERROR[LEAKAGE_TRANSITION_1: (0.6, 1-->2)] 0 1
        I[CCZ] 0 1 2
        CX 1 3
        I[T_DAG] 2 3
        H 2 3
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.92, 1) (0.95, 2)] 0 1 2 3
        DETECTOR rec[-1] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-4]
        """
    )

    cases: list[tuple[str, stim.Circuit, Callable[[], LeakageUint8Magic]]] = [
        ("c08", _corpus_circuit("c08_magic_with_leakage")[0], lambda: LeakageUint8Magic()),
        ("c13_spp", _corpus_circuit("c13_leaked_untagged_spp")[0], lambda: LeakageUint8Magic()),
        ("custom_6q_overlap_ccz", custom_6q, lambda: LeakageUint8Magic()),
        ("multiword_68q", multiword_68q, lambda: LeakageUint8Magic()),
        ("uc_false_4q", uc_false_4q, lambda: LeakageUint8Magic(unconditional_condition_on_U=False)),
        ("fast_leak_1to2_4q", fast_leak_4q, lambda: LeakageUint8Magic(fast_leakage=True)),
    ]

    for idx, (label, circ, make_h) in enumerate(cases):
        r_b, d_b, o_b = _sample_magic(
            circ,
            engine="branches",
            n_shots=n_shots,
            batch_size=batch_size,
            seed=7200 + idx,
            op_handler_factory=make_h,
        )
        r_t, d_t, o_t = _sample_magic(
            circ,
            engine="terms",
            n_shots=n_shots,
            batch_size=batch_size,
            seed=8200 + idx,
            op_handler_factory=make_h,
        )
        z_r = _max_z(r_b, r_t)
        z_d = _max_z(d_b, d_t) if d_b.shape[1] else 0.0
        z_o = _max_z(o_b, o_t) if o_b.shape[1] else 0.0
        assert max(z_r, z_d, z_o) < 5.0, f"T8 term-engine mismatch on {label}: z={(z_r, z_d, z_o)}"

    _assert_no_native_leaks()


# ---------------------------------------------------------------------------
# Test 3: Dense state-vector oracle on random leaky circuits + T/CCZ matrix
# ---------------------------------------------------------------------------


def _build_random_leaky_circuit(
    n: int, rng: np.random.Generator
) -> tuple[stim.Circuit, list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Build a random non-Clifford circuit with U-->2 leak + deterministic MPAD flags."""
    gates_1q = ("H", "S", "S_DAG", "SQRT_X", "SQRT_Y", "X", "Y", "Z")
    gates_2q = ("CX", "CZ", "CY", "SWAP")

    circ = stim.Circuit()
    circ.append("R", list(range(n)))
    pre_ops: list[tuple[Any, ...]] = []
    post_ops: list[tuple[Any, ...]] = []

    # Pre-leak entangled non-Clifford state.
    for q in range(n):
        circ.append("H", [q])
        pre_ops.append(("1q", "H", q))
    for q in range(n - 1):
        g2 = gates_2q[int(rng.integers(0, 2))]  # CX or CZ
        circ.append(g2, [q, q + 1])
        pre_ops.append(("2q", g2, q, q + 1))
    t0, t1 = rng.choice(n, size=2, replace=False).tolist()
    circ.append("I", [int(t0)], tag="T")
    pre_ops.append(("T", int(t0), False))
    circ.append("I", [int(t1)], tag="T_DAG")
    pre_ops.append(("T", int(t1), True))

    # U-->2 leak + exact non-destructive MPAD readout of per-shot leak mask L_s.
    circ.append("I_ERROR", list(range(n)), [0.0], tag="LEAKAGE_TRANSITION_1: (0.35, U-->2)")
    for q in range(n):
        circ.append("MPAD", [0], tag=f"LEAKAGE_MEASUREMENT: (1.0, 2) : {q}")

    # Ensure post-leak layers always exercise T/T_DAG, overlapping CCZ, 1q/2q Cliffords, and SPP.
    # Start with a guaranteed overlapping CCZ pair + guaranteed T/T_DAG on distinct qubits.
    p_ord = rng.permutation(n).tolist()
    triples_init = [(p_ord[0], p_ord[1], p_ord[2]), (p_ord[1], p_ord[2], p_ord[3])]
    flat_ccz: list[int] = []
    for a, b, c in triples_init:
        flat_ccz.extend([a, b, c])
    circ.append("I", flat_ccz, tag="CCZ")
    post_ops.append(("CCZ", triples_init))

    t_q = int(p_ord[0])
    td_q = int(p_ord[1])
    circ.append("I", [t_q], tag="T")
    post_ops.append(("T", t_q, False))
    circ.append("I", [td_q], tag="T_DAG")
    post_ops.append(("T", td_q, True))

    for _ in range(5):
        kind = int(rng.integers(0, 5))
        if kind == 0:
            g1 = gates_1q[int(rng.integers(0, len(gates_1q)))]
            q = int(rng.integers(0, n))
            circ.append(g1, [q])
            post_ops.append(("1q", g1, q))
        elif kind == 1:
            g2 = gates_2q[int(rng.integers(0, len(gates_2q)))]
            q0, q1 = rng.choice(n, size=2, replace=False).tolist()
            circ.append(g2, [int(q0), int(q1)])
            post_ops.append(("2q", g2, int(q0), int(q1)))
        elif kind == 2:
            q0, q1 = rng.choice(n, size=2, replace=False).tolist()
            p0 = "XYZ"[int(rng.integers(0, 3))]
            p1 = "XYZ"[int(rng.integers(0, 3))]
            neg = bool(rng.integers(0, 2))
            dag = bool(rng.integers(0, 2))
            pauli_targets = [
                stim.target_pauli(int(q0), p0, invert=neg),
                stim.target_combiner(),
                stim.target_pauli(int(q1), p1),
            ]
            circ.append("SPP_DAG" if dag else "SPP", pauli_targets)
            post_ops.append(("SPP", [(int(q0), p0), (int(q1), p1)], neg, dag))
        elif kind == 3:
            q = int(rng.integers(0, n))
            dag = bool(rng.integers(0, 2))
            circ.append("I", [q], tag="T_DAG" if dag else "T")
            post_ops.append(("T", q, dag))
        else:
            perm = rng.permutation(n).tolist()
            triples = [(perm[0], perm[1], perm[2])]
            if n >= 5 and rng.random() < 0.7:
                triples.append((perm[2], perm[3], perm[4]))
            flat = [q for tr in triples for q in tr]
            circ.append("I", flat, tag="CCZ")
            post_ops.append(("CCZ", triples))

    return circ, pre_ops, post_ops


def _eval_dense_leaky(
    n: int,
    pre_ops: list[tuple[Any, ...]],
    post_ops: list[tuple[Any, ...]],
    leaked: set[int],
    *,
    uc: bool,
    wrong_ccz_mode: bool = False,
    wrong_skip_t: bool = False,
) -> _DenseLeakageOracle:
    oracle = _DenseLeakageOracle(n)
    for op in pre_ops:
        if op[0] == "1q":
            oracle.apply_unitary(op[1], (op[2],))
        elif op[0] == "2q":
            oracle.apply_unitary(op[1], (op[2], op[3]))
        elif op[0] == "T":
            oracle.apply_t(op[1], dag=op[2])
    for op in post_ops:
        tag = op[0]
        if tag == "1q":
            if not (uc and op[2] in leaked):
                oracle.apply_unitary(op[1], (op[2],))
        elif tag == "2q":
            if not (uc and (op[2] in leaked or op[3] in leaked)):
                oracle.apply_unitary(op[1], (op[2], op[3]))
        elif tag == "SPP":
            terms, neg, dag = op[1], op[2], op[3]
            if not (uc and any(q in leaked for q, _ in terms)):
                oracle.apply_spp(terms, neg, dag)
        elif tag == "T":
            # Correct rule: I[T]/I[T_DAG] ALWAYS act regardless of uc or leaked status.
            if not (wrong_skip_t and op[1] in leaked):
                oracle.apply_t(op[1], dag=op[2])
        elif tag == "CCZ":
            for a, b, c in op[1]:
                hit = (a in leaked) or (b in leaked) or (c in leaked)
                if wrong_ccz_mode:
                    # Flip the CCZ skip rule for leaked triples as a negative control.
                    do_apply = hit if uc else (not hit)
                else:
                    do_apply = not (uc and hit)
                if do_apply:
                    oracle.apply_ccz(a, b, c)
    return oracle


def _ccz_repeat2_oracle(leaked: set[int]) -> np.ndarray:
    """Exact state vector for the 6q overlapping-CCZ REPEAT 2 circuit under uc=True."""
    oracle = _DenseLeakageOracle(6)
    for q in range(6):
        oracle.apply_unitary("H", (q,))
    for q in (0, 3):
        oracle.apply_t(q, dag=False)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        oracle.apply_unitary("CZ", (a, b))
    for _ in range(2):
        for q in range(6):
            oracle.apply_t(q, dag=False)
        for triple in ((0, 1, 2), (2, 3, 4), (4, 5, 0)):
            if not (leaked & set(triple)):
                oracle.apply_ccz(*triple)
        for q in (0, 2, 4):
            if q not in leaked:
                oracle.apply_unitary("H", (q,))
    return oracle.psi


def _run_t8_dense_part_a() -> None:
    rng = np.random.default_rng(20260703)
    batch_size = 16
    neg_ccz_disagreements = 0
    neg_t_disagreements = 0

    for uc in (True, False):
        for n in (4, 5, 6):
            for trial in range(2):
                circ, pre_ops, post_ops = _build_random_leaky_circuit(n, rng)
                seed = 7300 + (100 if uc else 0) + 10 * n + trial

                sim = MagicsideSimulator(
                    circ,
                    op_handler=LeakageUint8Magic(unconditional_condition_on_U=uc),
                    batch_size=batch_size,
                    seed=seed,
                    _engine="branches",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    sim.run()
                flags = sim.get_final_measurement_records()
                assert flags.shape == (batch_size, n)

                peeks = {
                    p: sim.peek_pauli_born_batch(list(range(n)), p) for p in ("X", "Y", "Z")
                }

                oracles: list[_DenseLeakageOracle] = []
                for s in range(batch_size):
                    leaked = {q for q in range(n) if flags[s, q]}
                    oracle = _eval_dense_leaky(n, pre_ops, post_ops, leaked, uc=uc)
                    oracles.append(oracle)

                    sv_diff = _phase_aligned_diff(sim.state_vector(s), oracle.psi)
                    assert sv_diff < 1e-12, (uc, n, trial, s, sv_diff)

                    for p in ("X", "Y", "Z"):
                        for q in range(n):
                            p_diff = abs(float(peeks[p][q, s]) - oracle.peek_p1(q, p))
                            assert p_diff < 1e-12, (uc, n, trial, s, p, q, p_diff)

                    if leaked:
                        wrong_ccz = _eval_dense_leaky(
                            n, pre_ops, post_ops, leaked, uc=uc, wrong_ccz_mode=True
                        )
                        if _phase_aligned_diff(sim.state_vector(s), wrong_ccz.psi) > 0.1:
                            neg_ccz_disagreements += 1
                        wrong_t = _eval_dense_leaky(
                            n, pre_ops, post_ops, leaked, uc=uc, wrong_skip_t=True
                        )
                        if _phase_aligned_diff(sim.state_vector(s), wrong_t.psi) > 0.1:
                            neg_t_disagreements += 1

                # Terminal computational-basis measurement: verify exact Born mass and |m> collapse.
                circ_m = circ.copy()
                circ_m.append("M", list(range(n)))
                sim_m = MagicsideSimulator(
                    circ_m,
                    op_handler=LeakageUint8Magic(unconditional_condition_on_U=uc),
                    batch_size=batch_size,
                    seed=seed,
                    _engine="branches",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    sim_m.run()
                recs_m = sim_m.get_final_measurement_records()
                np.testing.assert_array_equal(recs_m[:, :n], flags)
                for s in range(batch_size):
                    bits = recs_m[s, n:]
                    want_mass = oracles[s].computational_prob(bits)
                    got_mass = sim_m.get_branch_mass(s)
                    assert abs(got_mass - want_mass) < 1e-12, (uc, n, trial, s, got_mass, want_mass)

                    ket = np.zeros(1 << n, dtype=np.complex128)
                    idx = sum((1 << q) for q in range(n) if int(bits[q]))
                    ket[idx] = 1.0 + 0.0j
                    post_diff = _phase_aligned_diff(sim_m.state_vector(s), ket)
                    assert post_diff < 1e-12, (uc, n, trial, s, post_diff)

    assert neg_ccz_disagreements > 10
    assert neg_t_disagreements > 10


def _run_t8_dense_part_b() -> None:
    batch_size = 16

    # (i) I[T]^4 and I[T_DAG]^4 on leaked qubit emit 4 UserWarnings each & flip MPP X0 == 1
    # under BOTH uc=True and uc=False.
    for tag in ("T", "T_DAG"):
        circ_t4 = stim.Circuit(
            f"R 0\nH 0\nI_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
            f"I[{tag}] 0\nI[{tag}] 0\nI[{tag}] 0\nI[{tag}] 0\nMPP X0\n"
        )
        for uc in (True, False):
            sim = MagicsideSimulator(
                circ_t4,
                op_handler=LeakageUint8Magic(unconditional_condition_on_U=uc),
                batch_size=batch_size,
                seed=7401,
                _engine="branches",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sim.run()
            u_warns = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            assert len(u_warns) == 4, (tag, uc, u_warns)
            assert all(f"I[{tag}]" in m for m in u_warns)
            assert np.all(sim.get_final_measurement_records()[:, 0])

    # (ii) I[CCZ] 0 1 2 with each leg {0, 1, 2} leaked individually.
    for leg in (0, 1, 2):
        # Put |+> on an unleaked probe qubit so H+M always acts on the probe; prepare |1> on the other two.
        probe = 0 if leg == 2 else 2
        ones = [q for q in (0, 1, 2) if q != probe]
        circ_leg = stim.Circuit(
            f"R 0 1 2\n"
            f"X {ones[0]} {ones[1]}\nH {probe}\n"
            f"I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] {leg}\n"
            f"I[CCZ] 0 1 2\n"
            f"MPP X{probe}\nH {probe}\nM {probe}\n"
        )
        # uc=True: 0 warnings, CCZ skipped whole (MPP X == 0, M == 0), skipped_cczs == B.
        sim_t = MagicsideSimulator(
            circ_leg,
            op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
            batch_size=batch_size,
            seed=7410 + leg,
            _engine="branches",
        )
        with warnings.catch_warnings(record=True) as caught_t:
            warnings.simplefilter("always")
            sim_t.run()
        u_warns_t = [str(w.message) for w in caught_t if issubclass(w.category, UserWarning)]
        assert u_warns_t == [], (leg, u_warns_t)
        recs_t = sim_t.get_final_measurement_records()
        assert not np.any(recs_t[:, 0]) and not np.any(recs_t[:, 1])
        assert sim_t._branch_counters()["faults_realized"]["skipped_cczs"] == batch_size

        # uc=False: 1 UserWarning naming (0, 1, 2), CCZ acts (MPP X == 1, M == 1), skipped_cczs == 0.
        sim_f = MagicsideSimulator(
            circ_leg,
            op_handler=LeakageUint8Magic(unconditional_condition_on_U=False),
            batch_size=batch_size,
            seed=7420 + leg,
            _engine="branches",
        )
        with warnings.catch_warnings(record=True) as caught_f:
            warnings.simplefilter("always")
            sim_f.run()
        u_warns_f = [str(w.message) for w in caught_f if issubclass(w.category, UserWarning)]
        assert u_warns_f == ["Magic gate I[CCZ] applied to leaked group (0, 1, 2)"], (leg, u_warns_f)
        recs_f = sim_f.get_final_measurement_records()
        assert np.all(recs_f[:, 0]) and np.all(recs_f[:, 1])
        assert sim_f._branch_counters()["faults_realized"]["skipped_cczs"] == 0

    # (iii) Mixed-shot p_leak=0.5 CCZ: 0 warnings under uc=True, M == ~leaked, skipped_cczs == sum(leaked),
    # and skipping shots have identical state_vector (< 1e-12) to the circuit with I[CCZ] omitted.
    pre_mix_clean = (
        "R 0 1 2 3\nX 0 1\nH 2 3\nI[T] 3\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0\n"
    )
    with_ccz = stim.Circuit(pre_mix_clean + "I[CCZ] 0 1 2\nH 2\nM 2\n")
    without_ccz = stim.Circuit(pre_mix_clean + "H 2\nM 2\n")
    sim_w = MagicsideSimulator(
        with_ccz,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=32,
        seed=7430,
        _engine="branches",
    )
    sim_wo = MagicsideSimulator(
        without_ccz,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=32,
        seed=7430,
        _engine="branches",
    )
    with warnings.catch_warnings(record=True) as caught_m:
        warnings.simplefilter("always")
        sim_w.run()
        sim_wo.run()
    u_warns_m = [str(w.message) for w in caught_m if issubclass(w.category, UserWarning)]
    assert u_warns_m == []
    recs_w = sim_w.get_final_measurement_records()
    leaked_mask = recs_w[:, 0]
    assert 0 < int(leaked_mask.sum()) < 32
    np.testing.assert_array_equal(recs_w[:, 1], ~leaked_mask)
    assert sim_w._branch_counters()["faults_realized"]["skipped_cczs"] == int(leaked_mask.sum())
    # Also verify pre-measurement state vector identity on skipping shots.
    sim_w_sv = MagicsideSimulator(
        stim.Circuit(pre_mix_clean + "I[CCZ] 0 1 2\n"),
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=32,
        seed=7430,
        _engine="branches",
    )
    sim_wo_sv = MagicsideSimulator(
        stim.Circuit(pre_mix_clean),
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=32,
        seed=7430,
        _engine="branches",
    )
    sim_w_sv.run()
    sim_wo_sv.run()
    flags_sv = sim_w_sv.get_final_measurement_records()[:, 0]
    for s in range(32):
        diff = _phase_aligned_diff(sim_w_sv.state_vector(s), sim_wo_sv.state_vector(s))
        if flags_sv[s]:
            assert diff < 1e-12, (s, diff)
        else:
            assert diff > 0.5, (s, diff)

    # (iv) Overlapping 6q I[CCZ] 0 1 2 2 3 4 4 5 0 inside REPEAT 2.
    flags_str = "\n".join(f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : {q}] 0" for q in range(6))
    circ_ov = stim.Circuit(
        "R 0 1 2 3 4 5\nH 0 1 2 3 4 5\nI[T] 0 3\nCZ 0 1 2 3 4 5\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 1 2 3 4 5\n"
        f"{flags_str}\n"
        "REPEAT 2 {\n"
        "    I[T] 0 1 2 3 4 5\n"
        "    I[CCZ] 0 1 2 2 3 4 4 5 0\n"
        "    H 0 2 4\n"
        "}\n"
    )
    sim_ov = MagicsideSimulator(
        circ_ov,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=32,
        seed=7440,
        _engine="branches",
    )
    with warnings.catch_warnings(record=True) as caught_ov:
        warnings.simplefilter("always")
        sim_ov.run()
    ccz_warns = [
        str(w.message)
        for w in caught_ov
        if issubclass(w.category, UserWarning) and "CCZ" in str(w.message)
    ]
    assert ccz_warns == []
    flags_ov = sim_ov.get_final_measurement_records()
    expected_skipped_cczs = 0
    for s in range(32):
        leaked_s = {q for q in range(6) if flags_ov[s, q]}
        want = _ccz_repeat2_oracle(leaked_s)
        diff = _phase_aligned_diff(sim_ov.state_vector(s), want)
        assert diff < 1e-12, (s, leaked_s, diff)
        for triple in ((0, 1, 2), (2, 3, 4), (4, 5, 0)):
            if leaked_s & set(triple):
                expected_skipped_cczs += 2
    assert expected_skipped_cczs > 0
    assert sim_ov._branch_counters()["faults_realized"]["skipped_cczs"] == expected_skipped_cczs


def test_t8_dense_oracle_random_leaky_circuits_and_skip_semantics() -> None:
    _run_t8_dense_part_a()
    _run_t8_dense_part_b()
    _assert_no_native_leaks()


# ---------------------------------------------------------------------------
# Test 4: Leakage counters, faults_realized, and PEEK/POST/URESET contracts
# ---------------------------------------------------------------------------


def _run_t8_counters_part_a() -> None:
    batch_size = 16
    circ = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        I[T] 0
        I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0
        MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0
        CX 0 1
        SPP Z0*X2
        I[CCZ] 0 1 2 1 2 3
        X_ERROR(1.0) 3
        I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 2-->0)] 0
        M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1 2 3
        """
    )
    sim = MagicsideSimulator(
        circ,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=batch_size,
        seed=7501,
        _engine="branches",
    )

    # Batch 1
    sim.run()
    n_leaked_1 = int(sim.get_final_measurement_records()[:, 0].sum())
    assert 0 < n_leaked_1 < batch_size

    c1 = sim._branch_counters()
    fr1 = c1["faults_realized"]
    assert fr1["skipped_gates"] == 2 * n_leaked_1
    assert fr1["skipped_cczs"] == n_leaked_1
    assert fr1["pauli_faults"] >= batch_size

    required_kinds = ("T", "CCZ", "SKIP", "MEAS", "PEEK", "POST", "URESET")
    for k in required_kinds:
        assert c1["transitions_new"][k] > 0, f"expected transitions_new[{k}] > 0, got {c1['transitions_new']}"

    peak1 = sim._branch_peak_per_shot()
    assert peak1.dtype == np.int16 and peak1.shape == (batch_size,)
    assert int(peak1.min()) >= 1

    # Batch 2 (warm store)
    sim.clear()
    sim.run()
    n_leaked_2 = int(sim.get_final_measurement_records()[:, 0].sum())
    assert 0 < n_leaked_2 < batch_size

    c2 = sim._branch_counters()
    fr2 = c2["faults_realized"]
    assert fr2["skipped_gates"] == 2 * (n_leaked_1 + n_leaked_2)
    assert fr2["skipped_cczs"] == n_leaked_1 + n_leaked_2
    assert fr2["pauli_faults"] >= 2 * batch_size

    for k in required_kinds:
        assert c2["transitions_hit"][k] > 0, f"expected transitions_hit[{k}] > 0, got {c2['transitions_hit']}"

    assert sum(c2["branches_hist"].values()) == 2 * batch_size
    peak2 = sim._branch_peak_per_shot()
    assert peak2.dtype == np.int16 and peak2.shape == (batch_size,)


def _run_t8_counters_part_b() -> None:
    # Non-Clifford state R_Y(pi/4)|0>: P(Z=-1) = sin^2(pi/8) = (2 - sqrt(2)) / 4.
    circ = stim.Circuit("R 0\nH 0\nI[T] 0\nH 0\nS 0\n")
    p1_want = (2.0 - math.sqrt(2.0)) / 4.0
    p0_want = 1.0 - p1_want
    batch_size = 16

    # URESET leaves branch mass invariant (< 1e-14) while resetting shots to desired |0> / |1>.
    sim_u = MagicsideSimulator(circ, batch_size=batch_size, seed=7510, _engine="branches")
    sim_u.run()
    mass_before_u = [sim_u.get_branch_mass(s) for s in range(batch_size)]
    p1_before = sim_u.peek_pauli_born_batch([0], "Z")[0]
    assert np.allclose(p1_before, p1_want, atol=1e-12)

    reset_zero = np.zeros((1, batch_size), dtype=np.bool_)
    reset_one = np.zeros((1, batch_size), dtype=np.bool_)
    reset_zero[0, : batch_size // 2] = True
    reset_one[0, batch_size // 2 :] = True
    sim_u.do_unmatched_noisy_reset_z(reset_zero, reset_one, pauli="Z")

    for s in range(batch_size):
        assert abs(sim_u.get_branch_mass(s) - mass_before_u[s]) < 1e-14
    p1_after_u = sim_u.peek_pauli_born_batch([0], "Z")[0]
    assert np.allclose(p1_after_u[: batch_size // 2], 0.0, atol=1e-12)
    assert np.allclose(p1_after_u[batch_size // 2 :], 1.0, atol=1e-12)

    # POST multiplies branch mass by exact Born probability (< 1e-12).
    sim_p = MagicsideSimulator(circ, batch_size=batch_size, seed=7510, _engine="branches")
    sim_p.run()
    mass_before_p = [sim_p.get_branch_mass(s) for s in range(batch_size)]

    flip_mask = np.ones(batch_size, dtype=np.bool_)
    desired = np.array([s & 1 for s in range(batch_size)], dtype=np.uint8)
    sim_p.postselect_z_batch(0, flip_mask, desired, pauli="Z")

    for s in range(batch_size):
        want_m = mass_before_p[s] * (p1_want if desired[s] else p0_want)
        assert abs(sim_p.get_branch_mass(s) - want_m) < 1e-12
    p1_after_p = sim_p.peek_pauli_born_batch([0], "Z")[0]
    assert np.allclose(p1_after_p, desired.astype(np.float64), atol=1e-12)


def test_t8_leakage_counters_faults_and_peek_post_ureset() -> None:
    _run_t8_counters_part_a()
    _run_t8_counters_part_b()
    _assert_no_native_leaks()


# ---------------------------------------------------------------------------
# Test 5: F5 determinism on leaky circuits (warm vs cap=50 vs cold)
# ---------------------------------------------------------------------------


def _run_f5_4_batches(
    circ: stim.Circuit,
    *,
    uc: bool,
    cap: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[list[list[int]]],
    list[Any],
    list[np.ndarray],
    list[list[float]],
    dict[str, Any],
]:
    monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", cap)
    sim = MagicsideSimulator(
        circ,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=uc),
        batch_size=32,
        seed=202607,
        record_unleaked_to_leaked=True,
        record_leakage_events=True,
        _engine="branches",
    )
    recs, dets, obs, u2l_list, lev_list, peaks, masses = [], [], [], [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for b in range(4):
            if b > 0:
                sim.clear()
            u2l = sim.run()
            recs.append(sim.get_final_measurement_records())
            dets.append(sim.get_detector_flips())
            obs.append(sim.get_observable_flips())
            u2l_list.append([a.tolist() for a in u2l] if u2l is not None else [])
            lev_list.append(sim.get_leakage_events())
            peaks.append(sim._branch_peak_per_shot().copy())
            masses.append([sim.get_branch_mass(s) for s in range(32)])
    return recs, dets, obs, u2l_list, lev_list, peaks, masses, sim._branch_counters()


def test_t8_f5_determinism_leaky_circuits_warm_cold_cap50(monkeypatch: pytest.MonkeyPatch) -> None:
    rich_5q = stim.Circuit(
        """
        R 0 1 2 3 4
        H 0 1 2 3 4
        I[T] 0 2
        I[T_DAG] 1 3
        I_ERROR[LEAKAGE_TRANSITION_1: (0.3, 1-->2) (0.2, U-->2)] 0 1 2
        CX 0 1 2 3
        SPP Z0*X4
        I[CCZ] 0 1 2 2 3 4
        HERALDED_ERASE(0.1) 1 !4
        I[T] 4
        I[T_DAG] 0
        I_ERROR[LEAKAGE_TRANSITION_1: (0.35, 2-->0) (0.3, 2-->U)] 0 1 2
        MPAD[LEAKAGE_MEASUREMENT: (0.1, U) (0.9, 2) : 0 1 2] 0 0 0
        H 0 1 2 3 4
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.9, 1) (0.95, 2)] 0 1 2 3 4
        DETECTOR rec[-1] rec[-6]
        DETECTOR rec[-2] rec[-7]
        OBSERVABLE_INCLUDE(0) rec[-3] rec[-4] rec[-5]
        """
    )

    cases = (
        ("c08", _corpus_circuit("c08_magic_with_leakage")[0], True),
        ("rich_5q_uc_true", rich_5q, True),
        ("rich_5q_uc_false", rich_5q, False),
    )

    for label, circ, uc in cases:
        warm = _run_f5_4_batches(circ, uc=uc, cap=300_000, monkeypatch=monkeypatch)
        cap50 = _run_f5_4_batches(circ, uc=uc, cap=50, monkeypatch=monkeypatch)
        cold = _run_f5_4_batches(circ, uc=uc, cap=0, monkeypatch=monkeypatch)

        c_warm, c_cap50, c_cold = warm[7], cap50[7], cold[7]
        assert c_warm["store_nodes"] > 50, (label, c_warm["store_nodes"])
        assert c_warm["store_clears"] == 0, label
        assert c_cap50["store_clears"] >= 1, (label, c_cap50["store_clears"])
        assert c_cold["store_clears"] == 3, (label, c_cold["store_clears"])

        assert c_warm["faults_realized"] == c_cap50["faults_realized"] == c_cold["faults_realized"], label

        for b in range(4):
            for other_label, other in (("cap50", cap50), ("cold", cold)):
                assert np.array_equal(warm[0][b], other[0][b]), (label, other_label, "recs", b)
                assert np.array_equal(warm[1][b], other[1][b]), (label, other_label, "dets", b)
                assert np.array_equal(warm[2][b], other[2][b]), (label, other_label, "obs", b)
                assert warm[3][b] == other[3][b], (label, other_label, "u2l", b)
                assert warm[4][b] == other[4][b], (label, other_label, "lev", b)
                assert np.array_equal(warm[5][b], other[5][b]), (label, other_label, "peak", b)
                for s in range(32):
                    assert abs(warm[6][b][s] - other[6][b][s]) < 1e-14, (label, other_label, "mass", b, s)

    _assert_no_native_leaks()


# ---------------------------------------------------------------------------
# Test 6: Golden c08 & leakage corpus invariants
# ---------------------------------------------------------------------------


def test_t8_golden_c08_and_corpus_invariants() -> None:
    assert len(golden_test.CORPUS) == 18
    digests = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))

    by_name = {name: (modes, has_leak, text) for name, modes, has_leak, text in golden_test.CORPUS}
    c08_modes, c08_has_leak, c08_text = by_name["c08_magic_with_leakage"]
    assert c08_has_leak is True
    assert "magic_branch" in c08_modes

    expected_leak_branch = (
        "c03_leakage_transitions_swap_meas",
        "c04_leakage_conditioning_and_repeated_targets",
        "c08_magic_with_leakage",
        "c09_multiword_70q_leakage",
        "c11_high_prec_gate_arg",
        "c12_overlapping_lt2_pauli_xor",
        "c13_leaked_untagged_spp",
        "c16_sparse_multibatch",
    )
    actual_leak_branch = tuple(
        name for name, modes, has_leak, _ in golden_test.CORPUS if has_leak and "magic_branch" in modes
    )
    assert actual_leak_branch == expected_leak_branch

    hex_chars = frozenset("0123456789abcdef")
    for name in expected_leak_branch:
        d = digests[name]["magic_branch"]
        assert isinstance(d, str) and len(d) == 32 and set(d) <= hex_chars, (name, d)

    live_c08 = golden_test._run_case(stim.Circuit(c08_text), "magic_branch", True)
    assert live_c08 == digests["c08_magic_with_leakage"]["magic_branch"]

    _assert_no_native_leaks()
