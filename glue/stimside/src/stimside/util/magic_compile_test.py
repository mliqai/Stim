"""Verification suite for `stimside.util.magic_compile` (Milestone M1: V1.2b, V1.3, V1.4).

Covers:
1. Exact 8th-root-of-unity (`omega^v`) Clifford decompositions across all 55 1Q/2Q Stim gates.
2. `V1.3`: `I[T] -> fault -> I[T_DAG]` boundary discipline (no spurious cancellation),
   zero rank-branching under active Clifford faults, and `Eq. eq:defect` dimension bound.
3. `V1.2b`: 100 randomized 1-5Q `stim.Circuit` trajectories vs an independent dense
   `complex128` NumPy trajectory simulator (`< 1e-10`, zero global phase fudge).
4. `V1.4`: `DIRECT` vs `SELECTOR` exact complex statevector & branch mass equivalence across
   all 40 1Q/2Q `C_s(R(P))` axes, all 9 2Q controlled Paulis via 1-gate `CCZ`, all Clifford
   phases `v in {0..7}`, binary & multi-branch categorical channels, `gadgetize_pauli_noise`,
   and 3,000-shot Monte Carlo distribution checks.
5. Multi-word (`N in {63, 64, 65, 70, 128, 130}`) compiler & selector extraction remap
   invariance across 64-bit word boundaries, plus malformed tag/channel negative tests.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest
import stim

from stimside.util.magic_compile import (
    CONTROLLED_PAULI_2Q_AXES,
    CliffordFaultChannel,
    CompiledOpenBlockRecord,
    CompiledPauliAxis,
    DECOMP_1Q,
    DECOMP_2Q,
    DirectFaultSlotRecord,
    FaultBranch,
    LeakageBarrierRecord,
    LeftResetXEvent,
    LeftSppEvent,
    LeftUnitaryEvent,
    MagicPhaseTerm,
    MagicTermRecord,
    OMEGA_POWERS,
    RightRotEvent,
    compile_coordinates,
    compile_direct,
    compile_open_block,
    compile_pauli_product,
    compile_selector,
    decompose_clifford_exact,
    eval_decomposition_unitary_le,
    extract_circuit_faults,
    extract_data_expansion,
    gf2_symplectic_defect_rank,
    parse_clifford_fault_tag,
    reconstruct_statevector,
    run_direct,
    run_selector,
    sample_selector_activations,
)
from stimside.util.phase_qfe import (
    PhaseQFEComponent,
    ShotExpansion,
    apply_pauli,
    apply_quarter_turn,
    get_bit_in_words,
    merge_frames,
    n_words_for_qubits,
    overlap,
    tail_mask,
    unit_bit_words,
)

# ---------------------------------------------------------------------------
# Independent First-Principles Dense `complex128` Oracles (Zero Shared Code)
# ---------------------------------------------------------------------------

OMEGA = np.exp(1j * math.pi / 4.0)
OMEGA8 = np.array([np.exp(1j * k * math.pi / 4.0) for k in range(8)], dtype=np.complex128)
T_MAT = np.array([[1.0, 0.0], [0.0, OMEGA]], dtype=np.complex128)
T_DAG_MAT = np.array([[1.0, 0.0], [0.0, np.conj(OMEGA)]], dtype=np.complex128)
CCZ_MAT = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
_EXACT_MAGS = np.array([0.0, 0.5, math.sqrt(0.5), 1.0], dtype=np.float64)


def exact_stim_gate_local_c128(gate_name: str) -> np.ndarray:
    """Independent exact `complex128` little-endian unitary for any 1Q/2Q Stim Clifford gate.

    Stim's `gate_data(name).unitary_matrix` uses little-endian basis ordering with exact
    physical global phase, stored in `complex64`. Every entry of any 1Q or 2Q Clifford
    unitary lies in `{0, 1/2, 1/sqrt(2), 1} * exp(i * k * pi / 4)`, allowing exact
    algebraic promotion to `complex128` (`< 1e-15`) verified against `complex64` (`< 5e-8`).
    """
    gd_name = "I" if gate_name == "I_" else gate_name
    u32 = stim.gate_data(gd_name).unitary_matrix.astype(np.complex128)
    u64 = np.zeros_like(u32, dtype=np.complex128)
    for r in range(u32.shape[0]):
        for c in range(u32.shape[1]):
            z = u32[r, c]
            m_idx = int(np.argmin(np.abs(_EXACT_MAGS - abs(z))))
            assert abs(_EXACT_MAGS[m_idx] - abs(z)) < 5e-7, (gate_name, z)
            if m_idx > 0:
                ang = int(round(math.atan2(z.imag, z.real) / (math.pi / 4.0))) % 8
                u64[r, c] = _EXACT_MAGS[m_idx] * OMEGA8[ang]
                assert abs(u64[r, c] - z) < 5e-7, (gate_name, z, u64[r, c])
    return u64


def embed_unitary_le(U_local: np.ndarray, targets: Sequence[int], n: int) -> np.ndarray:
    """Embed a `k`-qubit little-endian unitary `U_local` onto `targets` in an `n`-qubit space."""
    k = len(targets)
    assert U_local.shape == (1 << k, 1 << k)
    assert len(set(targets)) == k and all(0 <= q < n for q in targets)
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    target_mask = sum(1 << q for q in targets)
    for x in range(dim):
        loc_in = sum(((x >> targets[r]) & 1) << r for r in range(k))
        base = x & ~target_mask
        for loc_out in range(1 << k):
            val = U_local[loc_out, loc_in]
            if val != 0.0:
                y = base | sum(((loc_out >> r) & 1) << targets[r] for r in range(k))
                out[y, x] += val
    return out


def dense_pauli_le(
    n: int, a_bits: Sequence[int], b_bits: Sequence[int], kappa: int
) -> np.ndarray:
    """Exact `P(a, b, kappa) = i^kappa X(a) Z(b)` on `n` qubits (little-endian)."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    a_mask = sum((int(a_bits[q]) & 1) << q for q in range(n))
    b_mask = sum((int(b_bits[q]) & 1) << q for q in range(n))
    pref = (1j) ** (int(kappa) % 4)
    for x in range(dim):
        z_par = (x & b_mask).bit_count() & 1
        out[x ^ a_mask, x] = -pref if z_par else pref
    return out


def dense_pauli_terms_le(
    n: int, terms: Sequence[tuple[int, str]], *, negate: bool = False
) -> np.ndarray:
    """Exact product `(-1)^negate prod_{(q, P)} P_q` on `n` qubits (little-endian)."""
    dim = 1 << n
    mat = (-1.0 + 0.0j) * np.eye(dim, dtype=np.complex128) if negate else np.eye(dim, dtype=np.complex128)
    for q, pch in terms:
        if pch == "I":
            continue
        mat = mat @ embed_unitary_le(exact_stim_gate_local_c128(pch), [q], n)
    return mat


def dense_quarter_turn_le(
    n: int, a_bits: Sequence[int], b_bits: Sequence[int], kappa: int
) -> np.ndarray:
    """Exact `R(P) = (I + i P) / sqrt(2)`."""
    P = dense_pauli_le(n, a_bits, b_bits, kappa)
    return (np.eye(1 << n, dtype=np.complex128) + 1j * P) / math.sqrt(2.0)


def _split_combiner_groups_oracle(op: stim.CircuitInstruction) -> list[tuple[list[tuple[int, str]], bool]]:
    groups: list[tuple[list[tuple[int, str]], bool]] = []
    cur: list[tuple[int, str]] = []
    inv = False
    expect_pauli = True
    for t in op.targets_copy():
        if t.is_combiner:
            expect_pauli = True
            continue
        if not expect_pauli and cur:
            groups.append((cur, inv))
            cur = []
            inv = False
        pch = "X" if t.is_x_target else ("Y" if t.is_y_target else "Z")
        cur.append((int(t.qubit_value), pch))
        if t.is_inverted_result_target:
            inv = not inv
        expect_pauli = False
    if cur:
        groups.append((cur, inv))
    return groups


def dense_simulate_trajectory_le(
    circuit: stim.Circuit,
    *,
    num_qubits: int | None = None,
    fault_channels: Sequence[CliffordFaultChannel] = (),
    fault_choices: Sequence[int] = (),
    pauli_noise_choices: Sequence[int] = (),
    reset_choices: Sequence[int] = (),
    quantum_outcomes: dict[int, int] | None = None,
    readout_flips: dict[int, bool] | None = None,
    sweep_bits: Sequence[int | bool] = (),
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[int | None, float, float, float]]]:
    """Independent first-principles dense `complex128` trajectory simulator (zero phase fudge)."""
    flat = circuit.flattened()
    n = int(num_qubits) if num_qubits is not None else max(1, flat.num_qubits)
    for ch in fault_channels:
        for br in ch.branches:
            for q in br.targets:
                n = max(n, q + 1)

    dim = 1 << n
    eye = np.eye(dim, dtype=np.complex128)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j
    branch_mass = 1.0
    meas_rec: list[bool] = []
    born_list: list[tuple[int | None, float, float, float]] = []

    q_out = quantum_outcomes or {}
    r_flips = readout_flips or {}
    by_step: dict[int, list[CliffordFaultChannel]] = {}
    for ch in fault_channels:
        by_step.setdefault(ch.step_index, []).append(ch)

    fault_idx = 0
    noise_idx = 0
    reset_idx = 0
    corr_occurred = False

    def run_prog_channels(step: int) -> None:
        nonlocal psi, fault_idx
        for ch in by_step.get(step, ()):
            c_val = int(fault_choices[fault_idx])
            fault_idx += 1
            if c_val > 0:
                br = ch.branches[c_val - 1]
                psi = embed_unitary_le(exact_stim_gate_local_c128(br.gate_name), br.targets, n) @ psi

    def project_and_renorm(obs_mat: np.ndarray, m_bit: int, m_idx: int | None) -> None:
        nonlocal psi, branch_mass
        z_in = float(np.vdot(psi, psi).real)
        proj0 = 0.5 * (eye + obs_mat) @ psi
        proj1 = 0.5 * (eye - obs_mat) @ psi
        z0 = float(np.vdot(proj0, proj0).real)
        z1 = float(np.vdot(proj1, proj1).real)
        born_list.append((m_idx, z0, z1, z_in))
        chosen_vec = proj1 if m_bit == 1 else proj0
        zm = z1 if m_bit == 1 else z0
        assert zm > 1e-14, f"Zero dense probability mass zm={zm} for m_bit={m_bit}"
        branch_mass *= zm / z_in
        psi = chosen_vec / math.sqrt(zm / z_in)

    pauli_2q = [p0 + p1 for p0 in "IXYZ" for p1 in "IXYZ" if not (p0 == "I" and p1 == "I")]

    ops = list(flat)

    def _exec_single_op(op: stim.CircuitInstruction) -> None:
        nonlocal psi, fault_idx, noise_idx, reset_idx, corr_occurred
        name = op.name
        tag = op.tag
        targs = op.targets_copy()

        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
            return
        if tag in ("T", "T_DAG"):
            umat = T_MAT if tag == "T" else T_DAG_MAT
            for t in targs:
                psi = embed_unitary_le(umat, [t.qubit_value], n) @ psi
            return
        if tag == "CCZ":
            qs = [t.qubit_value for t in targs]
            for off in range(0, len(qs), 3):
                psi = embed_unitary_le(CCZ_MAT, qs[off : off + 3], n) @ psi
            return
        if tag.startswith("CLIFFORD_FAULT"):
            parsed = parse_clifford_fault_tag(tag)
            arity = 1 if parsed[0][1] in DECOMP_1Q else 2
            qs = [t.qubit_value for t in targs]
            for off in range(0, len(qs), arity):
                grp = qs[off : off + arity]
                c_val = int(fault_choices[fault_idx])
                fault_idx += 1
                if c_val > 0:
                    gname = parsed[c_val - 1][1]
                    psi = embed_unitary_le(exact_stim_gate_local_c128(gname), grp, n) @ psi
            return

        if name in ("SPP", "SPP_DAG"):
            dag = name == "SPP_DAG"
            for terms, neg in _split_combiner_groups_oracle(op):
                P_mat = dense_pauli_terms_le(n, terms, negate=neg)
                fac = OMEGA8[7] * (eye + 1j * P_mat) / math.sqrt(2.0) if dag else OMEGA8[1] * (eye - 1j * P_mat) / math.sqrt(2.0)
                psi = fac @ psi
            return

        if name in DECOMP_1Q:
            U_loc = exact_stim_gate_local_c128(name)
            for t in targs:
                psi = embed_unitary_le(U_loc, [t.qubit_value], n) @ psi
            return

        if name in DECOMP_2Q:
            for off in range(0, len(targs), 2):
                t0, t1 = targs[off], targs[off + 1]
                if t0.is_qubit_target and t1.is_qubit_target:
                    psi = embed_unitary_le(
                        exact_stim_gate_local_c128(name), [t0.qubit_value, t1.qubit_value], n
                    ) @ psi
                else:
                    # Classical feedback gate (e.g. CX rec[-1] q)
                    fb_map = {
                        "CX": ("Z", "X"), "CNOT": ("Z", "X"), "ZCX": ("Z", "X"),
                        "CY": ("Z", "Y"), "ZCY": ("Z", "Y"),
                        "CZ": ("Z", "Z"), "ZCZ": ("Z", "Z"),
                        "XCZ": ("X", "Z"), "YCZ": ("Y", "Z"),
                    }
                    p_if_0, p_if_1 = fb_map[name]
                    if not t0.is_qubit_target:
                        ctrl_t, q_t, pch = t0, t1.qubit_value, p_if_1
                    else:
                        ctrl_t, q_t, pch = t1, t0.qubit_value, p_if_0
                    if ctrl_t.is_measurement_record_target:
                        cond = bool(meas_rec[len(meas_rec) + ctrl_t.value])
                    else:
                        cond = bool(sweep_bits[ctrl_t.value]) if ctrl_t.value < len(sweep_bits) else False
                    if cond:
                        psi = embed_unitary_le(exact_stim_gate_local_c128(pch), [q_t], n) @ psi
            return

        if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            pch = name[0]
            for t in targs:
                c_n = int(pauli_noise_choices[noise_idx])
                noise_idx += 1
                if c_n > 0:
                    psi = embed_unitary_le(exact_stim_gate_local_c128(pch), [t.qubit_value], n) @ psi
            return

        if name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
            for t in targs:
                c_n = int(pauli_noise_choices[noise_idx])
                noise_idx += 1
                if c_n > 0:
                    pch = ("X", "Y", "Z")[c_n - 1]
                    psi = embed_unitary_le(exact_stim_gate_local_c128(pch), [t.qubit_value], n) @ psi
            return

        if name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
            for off in range(0, len(targs), 2):
                q0, q1 = targs[off].qubit_value, targs[off + 1].qubit_value
                c_n = int(pauli_noise_choices[noise_idx])
                noise_idx += 1
                if c_n > 0:
                    p0, p1 = pauli_2q[c_n - 1]
                    psi = dense_pauli_terms_le(n, [(q0, p0), (q1, p1)]) @ psi
            return

        if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            order = ("I", "X", "Y", "Z")
            for t in targs:
                c_h = int(pauli_noise_choices[noise_idx])
                noise_idx += 1
                meas_rec.append(c_h > 0)
                if c_h > 0:
                    pch = order[c_h - 1]
                    if pch != "I":
                        psi = embed_unitary_le(exact_stim_gate_local_c128(pch), [t.qubit_value], n) @ psi
            return

        if name in ("CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"):
            if name != "ELSE_CORRELATED_ERROR":
                corr_occurred = False
            c_c = int(pauli_noise_choices[noise_idx])
            noise_idx += 1
            if not corr_occurred and c_c > 0:
                corr_occurred = True
                xz_map: dict[int, tuple[int, int]] = {}
                for grp_t, _ in _split_combiner_groups_oracle(op):
                    for q_t, p_t in grp_t:
                        xv, zv = xz_map.get(q_t, (0, 0))
                        if p_t in ("X", "Y"):
                            xv ^= 1
                        if p_t in ("Z", "Y"):
                            zv ^= 1
                        xz_map[q_t] = (xv, zv)
                red_terms = [
                    (q_t, "IZXY"[(xv << 1) | zv])
                    for q_t, (xv, zv) in xz_map.items()
                    if (xv or zv)
                ]
                if red_terms:
                    psi = dense_pauli_terms_le(n, red_terms) @ psi
            return

        if name in ("M", "MZ", "MX", "MY"):
            pch = "Z" if name in ("M", "MZ") else name[1]
            for t in targs:
                m_idx = len(meas_rec)
                m_q = int(q_out[m_idx])
                obs = dense_pauli_terms_le(n, [(t.qubit_value, pch)])
                project_and_renorm(obs, m_q, m_idx)
                inv = bool(t.is_inverted_result_target)
                flip = bool(r_flips.get(m_idx, False))
                meas_rec.append(bool(m_q ^ int(inv) ^ int(flip)))
            return

        if name in ("MXX", "MYY", "MZZ"):
            pch = name[1]
            for off in range(0, len(targs), 2):
                t0, t1 = targs[off], targs[off + 1]
                m_idx = len(meas_rec)
                m_q = int(q_out[m_idx])
                obs = dense_pauli_terms_le(n, [(t0.qubit_value, pch), (t1.qubit_value, pch)])
                project_and_renorm(obs, m_q, m_idx)
                inv = bool(t0.is_inverted_result_target) ^ bool(t1.is_inverted_result_target)
                flip = bool(r_flips.get(m_idx, False))
                meas_rec.append(bool(m_q ^ int(inv) ^ int(flip)))
            return

        if name == "MPP":
            for terms, inv in _split_combiner_groups_oracle(op):
                m_idx = len(meas_rec)
                m_q = int(q_out[m_idx])
                obs = dense_pauli_terms_le(n, terms)
                project_and_renorm(obs, m_q, m_idx)
                flip = bool(r_flips.get(m_idx, False))
                meas_rec.append(bool(m_q ^ int(inv) ^ int(flip)))
            return

        if name in ("R", "RZ", "RX", "RY", "MR", "MRZ", "MRX", "MRY"):
            is_mr = name.startswith("MR")
            tail = name[2:] if is_mr else name[1:]
            basis = "Z" if tail in ("", "Z") else tail
            U_b = (
                np.eye(2, dtype=np.complex128)
                if basis == "Z"
                else exact_stim_gate_local_c128("H" if basis == "X" else "H_YZ")
            )
            X_loc = exact_stim_gate_local_c128("X")
            for t in targs:
                q = t.qubit_value
                m_idx = len(meas_rec) if is_mr else None
                m_q = int(reset_choices[reset_idx])
                reset_idx += 1
                U_emb = embed_unitary_le(U_b, [q], n)
                # Exact ReferenceCHPTableau Kraus convention: U_B * X_q^m * Pi_{Z_q, m} * U_B
                psi = U_emb @ psi
                obs_z = dense_pauli_terms_le(n, [(q, "Z")])
                project_and_renorm(obs_z, m_q, m_idx)
                if m_q == 1:
                    psi = embed_unitary_le(X_loc, [q], n) @ psi
                psi = U_emb @ psi
                if is_mr:
                    assert m_idx is not None
                    inv = bool(t.is_inverted_result_target)
                    flip = bool(r_flips.get(m_idx, False))
                    meas_rec.append(bool(m_q ^ int(inv) ^ int(flip)))
            return

        if name == "MPAD":
            for t in targs:
                m_idx = len(meas_rec)
                pad_b = bool(t.value) ^ bool(t.is_inverted_result_target)
                flip = bool(r_flips.get(m_idx, False))
                meas_rec.append(pad_b ^ flip)
            return

        raise AssertionError(f"Unhandled instruction in dense oracle: {op!r}")

    for step_i, op in enumerate(ops):
        run_prog_channels(step_i)
        _exec_single_op(op)
    run_prog_channels(len(ops))

    return psi, np.array(meas_rec, dtype=np.bool_), branch_mass, born_list


# ---------------------------------------------------------------------------
# Group A: Exact Clifford Decompositions & V1.3 Non-Cancellation / Defect Bound
# ---------------------------------------------------------------------------


def _reconstruct_unitary_via_qfe_primitives(
    gate_or_tab: str | stim.Tableau, targets: Sequence[int], n: int, *, inverse: bool = False
) -> tuple[int, np.ndarray]:
    """Reconstruct `U_qfe` column-by-column by applying `omega^v`, `P1`, and `P2` to `|x>`."""
    decomp = decompose_clifford_exact(gate_or_tab, targets, inverse=inverse)
    dim = 1 << n
    n_w = n_words_for_qubits(n)
    U_qfe = np.zeros((dim, dim), dtype=np.complex128)

    def spec_to_words(spec: str) -> tuple[np.ndarray, np.ndarray, int]:
        neg = spec[0] == "-"
        chars = spec[1:]
        a_w = np.zeros(n_w, dtype=np.uint64)
        b_w = np.zeros(n_w, dtype=np.uint64)
        y_cnt = 0
        for idx_l, ch in enumerate(chars):
            q_g = targets[idx_l]
            if ch in ("X", "Y"):
                a_w |= unit_bit_words(q_g, n_w)
            if ch in ("Z", "Y"):
                b_w |= unit_bit_words(q_g, n_w)
            if ch == "Y":
                y_cnt += 1
        kap = (y_cnt + (2 if neg else 0)) & 3
        return a_w, b_w, kap

    p0_a, p0_b, p0_k = spec_to_words(decomp.p0_spec)
    r_axes = [spec_to_words(r_sp) for r_sp in decomp.r_specs]

    for x in range(dim):
        comp = PhaseQFEComponent.zero(n)
        comp.h_words[0] = np.uint64(x)
        comp.check_invariants()
        w = OMEGA_POWERS[decomp.omega_power]
        g_p, comp = apply_pauli(comp, p0_a, p0_b, p0_k)
        w *= g_p
        for ra, rb, rk in r_axes:
            g_r, comp = apply_quarter_turn(comp, ra, rb, rk, sign=1)
            w *= g_r
            comp.check_invariants()
        U_qfe[:, x] = w * comp.to_statevector(endian="little")
    return decomp.omega_power, U_qfe


def test_decompose_clifford_exact_all_stim_gates() -> None:
    """Verify all 55 1Q & 2Q Stim Clifford gates retain exact `omega^v` (`< 1e-14`)."""
    gates_1q = list(DECOMP_1Q.keys())
    gates_2q = list(DECOMP_2Q.keys())
    assert len(gates_1q) + len(gates_2q) == 55

    for g1 in gates_1q:
        U_oracle = exact_stim_gate_local_c128(g1)
        v_pow, U_qfe = _reconstruct_unitary_via_qfe_primitives(g1, (0,), 1)
        assert v_pow in range(8)
        assert np.max(np.abs(U_qfe - U_oracle)) < 1e-14, g1
        _, U_qfe_inv = _reconstruct_unitary_via_qfe_primitives(g1, (0,), 1, inverse=True)
        assert np.max(np.abs(U_qfe_inv - U_oracle.conj().T)) < 1e-14, g1

    for g2 in gates_2q:
        U_loc = exact_stim_gate_local_c128(g2)
        for targs in ((0, 1), (1, 0), (2, 0)):
            U_oracle = embed_unitary_le(U_loc, targs, 3)
            v_pow, U_qfe = _reconstruct_unitary_via_qfe_primitives(g2, targs, 3)
            assert v_pow in range(8)
            assert np.max(np.abs(U_qfe - U_oracle)) < 1e-14, (g2, targs)
        _, U_inv = _reconstruct_unitary_via_qfe_primitives(g2, (1, 0), 3, inverse=True)
        assert np.max(np.abs(U_inv - embed_unitary_le(U_loc.conj().T, (1, 0), 3))) < 1e-14, g2

    # Also verify general stim.Tableau fallback on a 3Q random Clifford tableau
    tab3 = stim.Tableau.from_circuit(stim.Circuit("H 0\nS 1\nCX 0 2\nISWAP 1 2\nC_XYZ 0"))
    _, U_tab_qfe = _reconstruct_unitary_via_qfe_primitives(tab3, (0, 1, 2), 3)
    U_tab_ref = tab3.to_unitary_matrix(endian="little").astype(np.complex128)
    assert np.max(np.abs(U_tab_qfe - U_tab_ref)) < 1e-7  # Tableau.to_unitary_matrix is complex64


def test_v1_3_regression_t_fault_tdag_no_spurious_cancellation() -> None:
    """`V1.3`: `I[T] 0 -> candidate fault -> I[T_DAG] 0` never cancels across fault/noise/meas barriers."""
    expected_inactive = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    expected_h_active = np.array([1.0 + 0.0j, np.exp(-1j * math.pi / 4.0)], dtype=np.complex128) / math.sqrt(2.0)

    # 1. Tagged I[CLIFFORD_FAULT: (0.5, H)] 0 and prob=0.0 slot barrier
    for prob in (0.5, 0.0):
        c_tag = stim.Circuit(f"I[T] 0\nI[CLIFFORD_FAULT: ({prob}, H)] 0\nI[T_DAG] 0")
        dp = compile_direct(c_tag, c2_budget=64)
        sp = compile_selector(c_tag, c2_budget=64)
        # Verify C2 kept two separate CompiledOpenBlockRecords around DirectFaultSlotRecord in DIRECT
        open_blocks = [r for r in dp.records if isinstance(r, CompiledOpenBlockRecord)]
        fault_slots = [r for r in dp.records if isinstance(r, DirectFaultSlotRecord)]
        assert len(open_blocks) == 2 and len(fault_slots) == 1

        for c_choice, exp_ket in ((0, expected_inactive), (1, expected_h_active)):
            rd = run_direct(dp, fault_trace=[c_choice])
            rs = run_selector(sp, fault_trace=[c_choice], extract_data=True)
            assert np.max(np.abs(reconstruct_statevector(rd) - exp_ket)) < 1e-14
            assert np.max(np.abs(reconstruct_statevector(rs) - exp_ket)) < 1e-14

    # 2. Programmatic CliffordFaultChannel inserted at step_index=1 (between I[T] 0 and I[T_DAG] 0)
    c_prog = stim.Circuit("I[T] 0\nI[T_DAG] 0")
    ch_h = CliffordFaultChannel(step_index=1, branches=(FaultBranch(0.5, "H", (0,)),))
    dp_p = compile_direct(c_prog, fault_channels=[ch_h], c2_budget=64)
    sp_p = compile_selector(c_prog, fault_channels=[ch_h], c2_budget=64)
    for c_choice, exp_ket in ((0, expected_inactive), (1, expected_h_active)):
        psi_d_oracle, _, _, _ = dense_simulate_trajectory_le(
            c_prog, num_qubits=1, fault_channels=[ch_h], fault_choices=[c_choice]
        )
        assert np.max(np.abs(psi_d_oracle - exp_ket)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(run_direct(dp_p, fault_trace=[c_choice])) - exp_ket)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(run_selector(sp_p, fault_trace=[c_choice], extract_data=True)) - exp_ket)) < 1e-14

    # 3. X_ERROR barrier: active X fault between T and T_DAG on |0> gives exp(-i*pi/4)|1>, NOT +|1>!
    c_x = stim.Circuit("I[T] 0\nX_ERROR(0.5) 0\nI[T_DAG] 0")
    dp_x = compile_direct(c_x, c2_budget=64)
    sp_x = compile_selector(c_x, gadgetize_pauli_noise=True, c2_budget=64)
    exp_x_active = np.array([0.0 + 0.0j, np.exp(-1j * math.pi / 4.0)], dtype=np.complex128)
    assert np.max(np.abs(reconstruct_statevector(run_direct(dp_x, pauli_noise_trace=[1])) - exp_x_active)) < 1e-14
    assert np.max(np.abs(reconstruct_statevector(run_selector(sp_x, pauli_noise_trace=[1], extract_data=True)) - exp_x_active)) < 1e-14

    # 4. MX, RX, and CX rec[-1] 0 barriers between I[T] 0 and I[T_DAG] 0
    for mid_op, kwargs, dense_check in (
        ("MX 0", {"requested_quantum_outcomes": {0: 1}}, True),
        ("RX 0", {"reset_outcomes": [0]}, True),
        ("M 1\nCX rec[-1] 0", {"requested_quantum_outcomes": {0: 1}}, True),
    ):
        circ_b = stim.Circuit(f"X 1\nI[T] 0\n{mid_op}\nI[T_DAG] 0")
        dp_b = compile_direct(circ_b, c2_budget=64)
        sp_b = compile_selector(circ_b, c2_budget=64)
        rd_b = run_direct(dp_b, **kwargs)
        rs_b = run_selector(sp_b, extract_data=True, **kwargs)
        psi_d, _, _, _ = dense_simulate_trajectory_le(
            circ_b,
            reset_choices=rd_b.reset_choices,
            quantum_outcomes=rd_b.quantum_outcomes,
            readout_flips=rd_b.readout_flips,
        )
        assert dense_check and np.max(np.abs(reconstruct_statevector(rd_b) - psi_d)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(rs_b) - psi_d)) < 1e-14


def test_v1_3_zero_rank_branching_and_eq_defect_bound() -> None:
    """`V1.3`: Zero rank-branching on active Clifford faults (`chi_after <= chi_before`) & `Eq. eq:defect`."""
    # (a) Prepare a chi=16 magic state on n=4 and apply 20 sequential active 1Q/2Q Clifford faults
    prep = stim.Circuit("H 0 1 2 3\nI[T] 0 1 2 3")
    fault_seq = [
        ("H", (0,)), ("S", (1,)), ("S_DAG", (2,)), ("C_XYZ", (3,)),
        ("CX", (0, 1)), ("CZ", (1, 2)), ("XCY", (2, 3)), ("SWAP", (0, 3)),
        ("ISWAP", (1, 3)), ("H_XY", (0,)), ("H_YZ", (2,)), ("SQRT_X", (1,)),
        ("SQRT_Y_DAG", (3,)), ("YCZ", (0, 2)), ("XCX", (1, 3)), ("C_ZYX", (2,)),
        ("SQRT_XX", (0, 1)), ("SQRT_ZZ_DAG", (2, 3)), ("CXSWAP", (0, 2)), ("ISWAP_DAG", (1, 2)),
    ]
    dp_prep = compile_direct(prep, c2_budget=1)
    res_prep = run_direct(dp_prep)
    exp = res_prep.expansion
    assert exp.chi == 16 and exp.d == 0

    ref_chp = res_prep.final_tableau
    for gname, qs in fault_seq:
        chi_before = exp.chi
        decomp = decompose_clifford_exact(gname, qs)
        p0_neg = decomp.p0_spec[0] == "-"
        p0_terms = [(qs[i], decomp.p0_spec[1 + i]) for i in range(len(qs)) if decomp.p0_spec[1 + i] != "I"]
        if p0_terms or p0_neg:
            exp = apply_pauli(exp, compile_pauli_product(ref_chp, p0_terms, negate=p0_neg))
        for r_sp in decomp.r_specs:
            r_neg = r_sp[0] == "-"
            r_terms = [(qs[i], r_sp[1 + i]) for i in range(len(qs)) if r_sp[1 + i] != "I"]
            exp = apply_quarter_turn(exp, compile_pauli_product(ref_chp, r_terms, negate=r_neg), sign=1)
        if decomp.v != 0:
            exp.weights = np.ascontiguousarray(exp.weights * OMEGA_POWERS[decomp.v])
        assert exp.chi <= chi_before
        exp.check_invariants()

    # (b) Inactive fault slots leave expansion completely untouched
    c_inact = stim.Circuit(
        "H 0 1 2\nI[T] 0 1\nI[CLIFFORD_FAULT: (0.4, H)] 0\n"
        "I[CLIFFORD_FAULT: (0.3, SWAP)] 1 2\nI[CCZ] 0 1 2"
    )
    c_clean = stim.Circuit("H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2")
    r_inact = run_direct(compile_direct(c_inact, c2_budget=1), fault_trace=[0, 0])
    r_clean = run_direct(compile_direct(c_clean, c2_budget=1))
    assert r_inact.expansion.chi == r_clean.expansion.chi
    assert r_inact.expansion.d == r_clean.expansion.d == 0
    assert np.allclose(r_inact.expansion.weights, r_clean.expansion.weights, atol=1e-15)

    # (c) Fault-free + Pauli-only + T/CCZ on N=8 keeps d == 0 at every step while chi > 1
    c_pauli_magic = stim.Circuit(
        "H 0 1 2 3 4 5 6 7\nI[T] 0 2 5\nX_ERROR(0.9) 1 4\n"
        "CX 0 3 2 6 5 7\nI[CCZ] 0 3 6\nZ_ERROR(0.9) 2 7\nI[T_DAG] 3"
    )
    r_pm = run_direct(compile_direct(c_pauli_magic, c2_budget=1), pauli_noise_trace=[1, 1, 1, 1])
    assert r_pm.expansion.d == 0 and r_pm.expansion.chi > 1

    # (d) Eq. eq:defect: d <= min(N, sum_j gf2_symplectic_defect_rank(F_j)) on N=6 and N=70
    for N_test, q_map in ((6, (0, 1, 2, 3, 4, 5)), (70, (0, 15, 63, 64, 65, 69))):
        q0, q1, q2, q3, q4, q5 = q_map
        fault_defs = [
            ("S", (q0,)),        # defect rank = 1 (transvection)
            ("H", (q1,)),        # defect rank = 1 (H - I has rank 1 over F_2!)
            ("SWAP", (q2, q3)),  # defect rank = 2
            ("ISWAP", (q4, q5)), # defect rank >= 2
        ]
        assert gf2_symplectic_defect_rank("X") == 0
        assert gf2_symplectic_defect_rank("S") == 1
        assert gf2_symplectic_defect_rank("H") == 1
        assert gf2_symplectic_defect_rank("SWAP") == 2

        for k_active in (1, 2, 3, 4):
            lines = [
                f"H {q0} {q1} {q2} {q3} {q4} {q5}",
                f"I[T] {q0} {q2}",
                f"I[CLIFFORD_FAULT: (0.5, {fault_defs[0][0]})] {fault_defs[0][1][0]}",
                f"CX {q0} {q3} {q1} {q4}",
                f"I[CLIFFORD_FAULT: (0.5, {fault_defs[1][0]})] {fault_defs[1][1][0]}",
                f"I[CCZ] {q0} {q2} {q4}",
                f"I[CLIFFORD_FAULT: (0.5, {fault_defs[2][0]})] {fault_defs[2][1][0]} {fault_defs[2][1][1]}",
                f"X_ERROR(0.5) {q3}",
                f"I[CLIFFORD_FAULT: (0.5, {fault_defs[3][0]})] {fault_defs[3][1][0]} {fault_defs[3][1][1]}",
                f"I[T_DAG] {q5}",
            ]
            trace = [1 if j < k_active else 0 for j in range(4)]
            bound = min(
                N_test,
                sum(gf2_symplectic_defect_rank(fault_defs[j][0]) for j in range(k_active)),
            )
            dp_def = compile_direct(stim.Circuit("\n".join(lines)), c2_budget=8)
            rd_def = run_direct(dp_def, fault_trace=trace, pauli_noise_trace=[1])
            assert rd_def.expansion.d <= bound, (N_test, k_active, rd_def.expansion.d, bound)
            rd_def.expansion.check_invariants()


# ---------------------------------------------------------------------------
# Group B: V1.2b Full Stim Trajectories & V1.4 DIRECT vs SELECTOR Equivalence
# ---------------------------------------------------------------------------

_ALL_1Q_GATES = tuple(g for g in DECOMP_1Q if g != "I_")
_ALL_2Q_GATES = tuple(DECOMP_2Q.keys())
_MEAS_1Q_GATES = ("M", "MZ", "MX", "MY")
_MEAS_2Q_GATES = ("MXX", "MYY", "MZZ")
_RESET_GATES = ("R", "RZ", "RX", "RY")
_MR_GATES = ("MR", "MRZ", "MRX", "MRY")
_FB_GATES = ("CX", "CY", "CZ", "XCZ", "YCZ")


def _build_random_v12b_circuit(
    rng: np.random.Generator, n: int, num_ops: int = 36
) -> tuple[stim.Circuit, list[bool]]:
    """Generate a rich `n`-qubit `stim.Circuit` mixing all M1 supported instruction families."""
    lines: list[str] = []
    meas_count = 0
    sweep_bits = [bool(rng.integers(0, 2)) for _ in range(3)]

    for _ in range(num_ops):
        bucket = int(rng.integers(0, 100))
        if bucket < 22:
            g1 = _ALL_1Q_GATES[int(rng.integers(0, len(_ALL_1Q_GATES)))]
            k = int(rng.integers(1, min(n, 3) + 1))
            qs = rng.choice(n, size=k, replace=False)
            lines.append(f"{g1} " + " ".join(str(int(q)) for q in qs))
        elif bucket < 38 and n >= 2:
            g2 = _ALL_2Q_GATES[int(rng.integers(0, len(_ALL_2Q_GATES)))]
            qs = rng.choice(n, size=2, replace=False)
            lines.append(f"{g2} {int(qs[0])} {int(qs[1])}")
        elif bucket < 43:
            spp_name = "SPP" if int(rng.integers(0, 2)) == 0 else "SPP_DAG"
            k = int(rng.integers(1, min(n, 3) + 1))
            qs = rng.choice(n, size=k, replace=False)
            inv_prefix = "!" if int(rng.integers(0, 2)) == 1 else ""
            terms = [
                f"{inv_prefix if idx == 0 else ''}{'XYZ'[int(rng.integers(0, 3))]}{int(q)}"
                for idx, q in enumerate(qs)
            ]
            lines.append(f"{spp_name} " + "*".join(terms))
        elif bucket < 56:
            tag = "T" if int(rng.integers(0, 2)) == 0 else "T_DAG"
            k = int(rng.integers(1, min(n, 2) + 1))
            qs = rng.choice(n, size=k, replace=False)
            lines.append(f"I[{tag}] " + " ".join(str(int(q)) for q in qs))
        elif bucket < 63 and n >= 3:
            qs = rng.choice(n, size=3, replace=False)
            lines.append(f"I[CCZ] {int(qs[0])} {int(qs[1])} {int(qs[2])}")
        elif bucket < 74:
            # Pauli noise family
            sub = int(rng.integers(0, 6))
            if sub == 0:
                err = ("X_ERROR", "Y_ERROR", "Z_ERROR")[int(rng.integers(0, 3))]
                q = int(rng.integers(0, n))
                lines.append(f"{err}(0.25) {q}")
            elif sub == 1:
                q = int(rng.integers(0, n))
                lines.append(f"DEPOLARIZE1(0.3) {q}")
            elif sub == 2:
                q = int(rng.integers(0, n))
                lines.append(f"PAULI_CHANNEL_1(0.1, 0.1, 0.1) {q}")
            elif sub == 3 and n >= 2:
                qs = rng.choice(n, size=2, replace=False)
                lines.append(f"DEPOLARIZE2(0.35) {int(qs[0])} {int(qs[1])}")
            elif sub == 4:
                qs = rng.choice(n, size=min(n, 2), replace=False)
                t0 = "*".join(f"{'XYZ'[int(rng.integers(0, 3))]}{int(q)}" for q in qs)
                t1 = "*".join(f"{'XYZ'[int(rng.integers(0, 3))]}{int(q)}" for q in qs)
                lines.append(f"CORRELATED_ERROR(0.3) {t0}")
                lines.append(f"ELSE_CORRELATED_ERROR(0.4) {t1}")
            else:
                q = int(rng.integers(0, n))
                lines.append(f"HERALDED_PAULI_CHANNEL_1(0.05, 0.1, 0.1, 0.1) {q}")
                meas_count += 1
        elif bucket < 83:
            # Clifford fault slot (1Q or 2Q, single or multi-branch)
            if n >= 2 and int(rng.integers(0, 2)) == 1:
                qs = rng.choice(n, size=2, replace=False)
                g_a = _ALL_2Q_GATES[int(rng.integers(0, len(_ALL_2Q_GATES)))]
                g_b = _ALL_2Q_GATES[int(rng.integers(0, len(_ALL_2Q_GATES)))]
                lines.append(
                    f"I[CLIFFORD_FAULT: (0.25, {g_a}) (0.2, {g_b})] {int(qs[0])} {int(qs[1])}"
                )
            else:
                q = int(rng.integers(0, n))
                g_a = _ALL_1Q_GATES[int(rng.integers(0, len(_ALL_1Q_GATES)))]
                g_b = _ALL_1Q_GATES[int(rng.integers(0, len(_ALL_1Q_GATES)))]
                lines.append(f"I[CLIFFORD_FAULT: (0.25, {g_a}) (0.2, {g_b})] {q}")
        elif bucket < 91:
            # Measurements: 1Q, 2Q, MPP, MPAD
            sub = int(rng.integers(0, 4))
            p_arg = "(0.15)" if int(rng.integers(0, 3)) == 0 else ""
            inv = "!" if int(rng.integers(0, 3)) == 0 else ""
            if sub == 0 or n < 2:
                gm = _MEAS_1Q_GATES[int(rng.integers(0, len(_MEAS_1Q_GATES)))]
                q = int(rng.integers(0, n))
                lines.append(f"{gm}{p_arg} {inv}{q}")
                meas_count += 1
            elif sub == 1:
                gm2 = _MEAS_2Q_GATES[int(rng.integers(0, len(_MEAS_2Q_GATES)))]
                qs = rng.choice(n, size=2, replace=False)
                lines.append(f"{gm2}{p_arg} {inv}{int(qs[0])} {int(qs[1])}")
                meas_count += 1
            elif sub == 2:
                k = int(rng.integers(1, min(n, 3) + 1))
                qs = rng.choice(n, size=k, replace=False)
                terms = [
                    f"{inv if idx == 0 else ''}{'XYZ'[int(rng.integers(0, 3))]}{int(q)}"
                    for idx, q in enumerate(qs)
                ]
                lines.append(f"MPP{p_arg} " + "*".join(terms))
                meas_count += 1
            else:
                pad_v = int(rng.integers(0, 2))
                lines.append(f"MPAD {pad_v}")
                meas_count += 1
        elif bucket < 96:
            # Resets and Measure-Resets
            q = int(rng.integers(0, n))
            if int(rng.integers(0, 2)) == 0:
                gr = _RESET_GATES[int(rng.integers(0, len(_RESET_GATES)))]
                lines.append(f"{gr} {q}")
            else:
                gmr = _MR_GATES[int(rng.integers(0, len(_MR_GATES)))]
                inv = "!" if int(rng.integers(0, 2)) == 1 else ""
                lines.append(f"{gmr}(0.2) {inv}{q}")
                meas_count += 1
        else:
            # Classical feedback from rec[-k] or sweep[k]
            q = int(rng.integers(0, n))
            fb = _FB_GATES[int(rng.integers(0, len(_FB_GATES)))]
            if meas_count > 0 and int(rng.integers(0, 3)) > 0:
                k_back = int(rng.integers(1, min(meas_count, 4) + 1))
                if fb in ("XCZ", "YCZ"):
                    lines.append(f"{fb} {q} rec[{-k_back}]")
                else:
                    lines.append(f"{fb} rec[{-k_back}] {q}")
            else:
                sw_i = int(rng.integers(0, len(sweep_bits)))
                if fb in ("XCZ", "YCZ"):
                    lines.append(f"{fb} {q} sweep[{sw_i}]")
                else:
                    lines.append(f"{fb} sweep[{sw_i}] {q}")

    return stim.Circuit("\n".join(lines)), sweep_bits


def test_v1_2b_random_stim_circuit_trajectories() -> None:
    """`V1.2b`: 100 random 1-5Q `stim.Circuit` trajectories vs dense oracle (`< 1e-10`, ZERO phase fudge)."""
    rng = np.random.default_rng(20261029)
    budgets = (1, 4, 64)

    for traj_idx in range(100):
        n = (traj_idx % 5) + 1
        circ, sweep = _build_random_v12b_circuit(rng, n, num_ops=36)
        budget = budgets[traj_idx % len(budgets)]

        dp = compile_direct(circ, c2_budget=budget)
        rd = run_direct(dp, rng=rng, sweep_bits=sweep)

        psi_qfe = reconstruct_statevector(rd)
        psi_dense, meas_dense, mass_dense, born_dense = dense_simulate_trajectory_le(
            circ,
            num_qubits=dp.num_qubits,
            fault_choices=rd.fault_choices,
            pauli_noise_choices=rd.pauli_noise_choices,
            reset_choices=rd.reset_choices,
            quantum_outcomes=rd.quantum_outcomes,
            readout_flips=rd.readout_flips,
            sweep_bits=sweep,
        )

        # Direct complex statevector residual — NO global phase alignment!
        assert np.max(np.abs(psi_qfe - psi_dense)) < 1e-10, (traj_idx, n)
        assert np.array_equal(rd.measurements, meas_dense), (traj_idx, n)
        assert abs(rd.branch_mass - mass_dense) < 1e-10, (traj_idx, rd.branch_mass, mass_dense)

        # Check every step's Born mass conservation and agreement with dense projector masses
        assert len(rd.born_masses) == len(born_dense)
        for (m_i, z0_q, z1_q, zin_q), (m_d, z0_d, z1_d, zin_d) in zip(
            rd.born_masses, born_dense
        ):
            assert m_i == m_d
            assert abs((z0_q + z1_q) - zin_q) < 1e-10
            assert abs(z0_q - z0_d) < 1e-10
            assert abs(z1_q - z1_d) < 1e-10

        rd.expansion.check_invariants()

        # On every 10th trajectory, verify deterministic replay across all C2 budgets (1, 4, 64)
        if traj_idx % 10 == 0:
            for alt_b in budgets:
                dp_alt = compile_direct(circ, c2_budget=alt_b)
                rd_alt = run_direct(
                    dp_alt,
                    fault_trace=rd.fault_choices,
                    pauli_noise_trace=rd.pauli_noise_choices,
                    reset_outcomes=rd.reset_choices,
                    requested_quantum_outcomes=rd.quantum_outcomes,
                    requested_outcomes={i: int(b) for i, b in enumerate(rd.measurements)},
                    sweep_bits=sweep,
                )
                assert np.max(np.abs(reconstruct_statevector(rd_alt) - psi_dense)) < 1e-10


def test_v1_4_selector_all_40_controlled_quarter_turns() -> None:
    """`V1.4`: Verify ALL 40 1Q (8) and 2Q (32) Hermitian Paulis via `C_s(R(P)) = G_{-1}(P) G_{+1}(Z_s P)`."""
    # 1. All 8 1-qubit Hermitian Paulis (+/- I, +/- X, +/- Y, +/- Z) on magic input T H |0>
    psi_in_1q = T_MAT @ (exact_stim_gate_local_c128("H") @ np.array([1.0, 0.0], dtype=np.complex128))
    exp_in_2q_aug = ShotExpansion.zero(2)  # qubit 0 = data, qubit 1 = selector s
    # Prepare T_0 H_0 |0>_0 |0>_1 directly in QFE
    c_prep_1q = stim.Circuit("H 0\nI[T] 0")
    # Also test end-to-end via MagicPhaseTerm synthesis on (n_data=1, s=1) for all 8 axes!
    for sign in (+1, -1):
        for pch in ("I", "X", "Y", "Z"):
            a_b = [1 if pch in ("X", "Y") else 0]
            b_b = [1 if pch in ("Z", "Y") else 0]
            kap = ((1 if pch == "Y" else 0) + (2 if sign < 0 else 0)) & 3
            U_rp = dense_quarter_turn_le(1, a_b, b_b, kap)

            # Build (2-qubit: q0=data, q1=s) axes P and Z_s P
            a_p = np.array([a_b[0]], dtype=np.uint64)
            b_p = np.array([b_b[0]], dtype=np.uint64)
            ax_P = CompiledPauliAxis(a_words=a_p, b_words=b_p, kappa=kap)
            ax_ZsP = CompiledPauliAxis(
                a_words=a_p.copy(),
                b_words=np.array([b_b[0] | 2], dtype=np.uint64),
                kappa=kap,
            )
            recs = compile_open_block(
                [
                    MagicTermRecord(MagicPhaseTerm(axes=(ax_ZsP,), kind="T", epsilon=+1)),
                    MagicTermRecord(MagicPhaseTerm(axes=(ax_P,), kind="T", epsilon=-1)),
                ],
                n_qubits=2,
                budget=16,
            )
            assert len(recs) == 1 and isinstance(recs[0], CompiledOpenBlockRecord)

            for c_s in (0, 1):
                # Start from (T H |0>)_0 tensor |c_s>_1
                comp0 = PhaseQFEComponent.zero(2)
                comp0.h_words[0] = np.uint64(c_s << 1)
                comp1 = PhaseQFEComponent.zero(2)
                comp1.h_words[0] = np.uint64(1 | (c_s << 1))
                init_exp = merge_frames(
                    [(psi_in_1q[0], comp0), (psi_in_1q[1], comp1)]
                )
                from stimside.util.phase_qfe import apply_compiled_block
                from stimside.util.reference_chp import ReferenceCHPTableau

                out_aug = apply_compiled_block(init_exp, recs[0].block)
                _, data_exp = extract_data_expansion(
                    ReferenceCHPTableau(2), out_aug, 1, [c_s]
                )
                psi_sel = data_exp.to_statevector(endian="little")
                psi_expected = (U_rp @ psi_in_1q) if c_s == 1 else psi_in_1q
                assert np.max(np.abs(psi_sel - psi_expected)) < 1e-12, (sign, pch, c_s)

    # 2. All 32 2-qubit Hermitian Paulis (+/- {I, X, Y, Z}^2) on entangled magic state
    #    psi_in_2q = (I_0 tensor T_1) CZ_{0,1} (T_0 tensor S_1) H^{\otimes 2} |00>
    c_in2 = stim.Circuit("H 0 1\nI[T] 0\nS 1\nCZ 0 1\nI[T] 1\nCX 1 0")
    psi_in_2q = reconstruct_statevector(run_direct(compile_direct(c_in2)))
    from stimside.util.phase_qfe import apply_compiled_block
    from stimside.util.reference_chp import ReferenceCHPTableau

    count_2q = 0
    for sign in (+1, -1):
        for p0 in ("I", "X", "Y", "Z"):
            for p1 in ("I", "X", "Y", "Z"):
                count_2q += 1
                a_bits = [1 if p0 in ("X", "Y") else 0, 1 if p1 in ("X", "Y") else 0]
                b_bits = [1 if p0 in ("Z", "Y") else 0, 1 if p1 in ("Z", "Y") else 0]
                y_cnt = (1 if p0 == "Y" else 0) + (1 if p1 == "Y" else 0)
                kap = (y_cnt + (2 if sign < 0 else 0)) & 3
                U_rp2 = dense_quarter_turn_le(2, a_bits, b_bits, kap)

                a_mask = a_bits[0] | (a_bits[1] << 1)
                b_mask = b_bits[0] | (b_bits[1] << 1)
                ax_P = CompiledPauliAxis(
                    a_words=np.array([a_mask], dtype=np.uint64),
                    b_words=np.array([b_mask], dtype=np.uint64),
                    kappa=kap,
                )
                ax_ZsP = CompiledPauliAxis(
                    a_words=np.array([a_mask], dtype=np.uint64),
                    b_words=np.array([b_mask | 4], dtype=np.uint64),  # s = qubit 2
                    kappa=kap,
                )
                recs = compile_open_block(
                    [
                        MagicTermRecord(MagicPhaseTerm(axes=(ax_ZsP,), kind="T", epsilon=+1)),
                        MagicTermRecord(MagicPhaseTerm(axes=(ax_P,), kind="T", epsilon=-1)),
                    ],
                    n_qubits=3,
                    budget=16,
                )
                for c_s in (0, 1):
                    terms_init = []
                    for x_idx in range(4):
                        comp = PhaseQFEComponent.zero(3)
                        comp.h_words[0] = np.uint64(x_idx | (c_s << 2))
                        terms_init.append((psi_in_2q[x_idx], comp))
                    init_exp = merge_frames(terms_init)
                    out_aug = apply_compiled_block(init_exp, recs[0].block)
                    _, data_exp = extract_data_expansion(
                        ReferenceCHPTableau(3), out_aug, 2, [c_s]
                    )
                    psi_sel = data_exp.to_statevector(endian="little")
                    psi_expected = (U_rp2 @ psi_in_2q) if c_s == 1 else psi_in_2q
                    assert np.max(np.abs(psi_sel - psi_expected)) < 1e-12, (sign, p0, p1, c_s)

    assert count_2q == 32

    # 3. Also test full compile_selector -> run_selector -> extract_data_expansion across all
    #    30 non-trivial 2Q Hermitian axes conjugated via basis Cliffords around SQRT_ZZ / SQRT_ZZ_DAG!
    basis_prep = {"X": "H", "Y": "H_YZ", "Z": "I"}
    for p0 in ("I", "X", "Y", "Z"):
        for p1 in ("I", "X", "Y", "Z"):
            if p0 == "I" and p1 == "I":
                continue
            for dag_suffix in ("", "_DAG"):
                if p0 == "I":
                    fault_line = f"{basis_prep[p1]} 1\nI[CLIFFORD_FAULT: (0.5, SQRT_Z{dag_suffix})] 1\n{basis_prep[p1]} 1"
                elif p1 == "I":
                    fault_line = f"{basis_prep[p0]} 0\nI[CLIFFORD_FAULT: (0.5, SQRT_Z{dag_suffix})] 0\n{basis_prep[p0]} 0"
                else:
                    fault_line = (
                        f"{basis_prep[p0]} 0\n{basis_prep[p1]} 1\n"
                        f"I[CLIFFORD_FAULT: (0.5, SQRT_ZZ{dag_suffix})] 0 1\n"
                        f"{basis_prep[p0]} 0\n{basis_prep[p1]} 1"
                    )
                circ_full = stim.Circuit(
                    f"H 0 1\nI[T] 0\nCX 0 1\nI[T_DAG] 1\n{fault_line}\nH 0\nI[T] 0"
                )
                dp = compile_direct(circ_full, c2_budget=16)
                sp = compile_selector(circ_full, c2_budget=16)
                for c_s in (0, 1):
                    psi_d = reconstruct_statevector(run_direct(dp, fault_trace=[c_s]))
                    rs = run_selector(sp, fault_trace=[c_s])
                    dtab, dexp = extract_data_expansion(sp, rs.expansion, rs.selector_bits)
                    psi_s = reconstruct_statevector(dtab, dexp)
                    assert np.max(np.abs(psi_s - psi_d)) < 1e-12


def test_v1_4_selector_all_9_controlled_paulis_and_clifford_phases() -> None:
    """`V1.4`: Verify 1-gate `CCZ` shortcut on all 9 2Q controlled Paulis + all Clifford phases `v in {0..7}`."""
    # (a) All 9 2Q controlled Paulis + 4 Stim aliases on both (0, 1) and (1, 0) in a 3Q magic circuit
    cp_gates = ("CZ", "CX", "CY", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "CNOT", "ZCX", "ZCY", "ZCZ")
    assert set(CONTROLLED_PAULI_2Q_AXES.keys()) == set(cp_gates)

    for g_cp in cp_gates:
        for q0, q1 in ((0, 1), (1, 0)):
            # Pure fault circuit to assert raw_records emits EXACTLY 1 CCZ MagicTermRecord (not 6 Ts!)
            c_only = stim.Circuit(f"I[CLIFFORD_FAULT: (0.5, {g_cp})] {q0} {q1}")
            sp_only = compile_selector(c_only, c2_budget=1)
            magic_recs = [r for r in sp_only.raw_records if isinstance(r, MagicTermRecord)]
            assert len(magic_recs) == 1 and magic_recs[0].term.kind == "CCZ", (g_cp, magic_recs)

            # Full entangled 3Q non-stabilizer circuit around the fault
            circ = stim.Circuit(
                f"H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2\nS 1\nCX 2 0\n"
                f"I[CLIFFORD_FAULT: (0.5, {g_cp})] {q0} {q1}\n"
                f"H 0 1\nI[T_DAG] 0\nCZ 1 2"
            )
            dp = compile_direct(circ, c2_budget=32)
            sp = compile_selector(circ, c2_budget=32)
            for c_s in (0, 1):
                rd = run_direct(dp, fault_trace=[c_s])
                rs = run_selector(sp, fault_trace=[c_s])
                # Test BOTH signatures of extract_data_expansion!
                dtab1, dexp1 = extract_data_expansion(sp, rs.expansion, rs.selector_bits)
                dtab2, dexp2 = extract_data_expansion(sp.final_tableau, rs.expansion, 3, rs.selector_bits)
                psi_d = reconstruct_statevector(rd)
                psi_s1 = reconstruct_statevector(dtab1, dexp1)
                psi_s2 = reconstruct_statevector(dtab2, dexp2)
                psi_dense, _, _, _ = dense_simulate_trajectory_le(circ, num_qubits=3, fault_choices=[c_s])
                assert np.max(np.abs(psi_s1 - psi_d)) < 1e-12, (g_cp, q0, q1, c_s)
                assert np.max(np.abs(psi_s2 - psi_d)) < 1e-12, (g_cp, q0, q1, c_s)
                assert np.max(np.abs(psi_s1 - psi_dense)) < 1e-12, (g_cp, q0, q1, c_s)

    # (b) Rich 1Q & 2Q Clifford faults (SWAP, ISWAP, ISWAP_DAG, CXSWAP, SWAPCX, C_XYZ, C_ZYX, H*, SQRT_*)
    special_gates_1q = ("I", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H", "H_XY", "H_YZ", "C_XYZ", "C_ZYX")
    special_gates_2q = ("SWAP", "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG", "CXSWAP", "SWAPCX", "CZSWAP")

    for g1 in special_gates_1q:
        circ = stim.Circuit(
            f"H 0 1 2\nI[T] 0\nI[CCZ] 0 1 2\nI[CLIFFORD_FAULT: (0.5, {g1})] 0\nH 0\nI[T_DAG] 0"
        )
        dp = compile_direct(circ, c2_budget=32)
        sp = compile_selector(circ, c2_budget=32)
        for c_s in (0, 1):
            psi_d = reconstruct_statevector(run_direct(dp, fault_trace=[c_s]))
            psi_s = reconstruct_statevector(run_selector(sp, fault_trace=[c_s], extract_data=True))
            psi_dense, _, _, _ = dense_simulate_trajectory_le(circ, num_qubits=3, fault_choices=[c_s])
            assert np.max(np.abs(psi_s - psi_d)) < 1e-12, (g1, c_s)
            assert np.max(np.abs(psi_s - psi_dense)) < 1e-12, (g1, c_s)

    for g2 in special_gates_2q:
        circ = stim.Circuit(
            f"H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2\nI[CLIFFORD_FAULT: (0.5, {g2})] 1 0\nH 1\nI[T] 0"
        )
        dp = compile_direct(circ, c2_budget=32)
        sp = compile_selector(circ, c2_budget=32)
        for c_s in (0, 1):
            psi_d = reconstruct_statevector(run_direct(dp, fault_trace=[c_s]))
            psi_s = reconstruct_statevector(run_selector(sp, fault_trace=[c_s], extract_data=True))
            psi_dense, _, _, _ = dense_simulate_trajectory_le(circ, num_qubits=3, fault_choices=[c_s])
            assert np.max(np.abs(psi_s - psi_d)) < 1e-12, (g2, c_s)
            assert np.max(np.abs(psi_s - psi_dense)) < 1e-12, (g2, c_s)

    # (c) Hit ALL 8 selector phase powers v_eff in {0..7} via (H S)^k = omega^k I sequences on selector!
    # Note: (H S)^3 = omega^1 I! Applying k copies of [H, S] as co-activated faults or testing all v in {0..7}:
    # Even simpler: check that every v in {0..7} in Z8_DIAG selector phase matches omega^(v * c_s) exactly!
    for k_pow in range(8):
        # Build k_pow sequential triplets of (H, S, H, S, H, S) on qubit 0: each triplet = omega^1 * I!
        fault_ops = "\n".join(
            ["I[CLIFFORD_FAULT: (0.5, H)] 0\nI[CLIFFORD_FAULT: (0.5, S)] 0" for _ in range(3 * k_pow)]
        )
        circ_phase = stim.Circuit(f"H 0 1\nI[T] 0\nCX 0 1\n{fault_ops}\nI[T_DAG] 1")
        dp_ph = compile_direct(circ_phase, c2_budget=32)
        sp_ph = compile_selector(circ_phase, c2_budget=32)
        trace_all_on = [1] * (6 * k_pow)
        psi_d = reconstruct_statevector(run_direct(dp_ph, fault_trace=trace_all_on))
        psi_s = reconstruct_statevector(run_selector(sp_ph, fault_trace=trace_all_on, extract_data=True))
        psi_0 = reconstruct_statevector(run_direct(dp_ph, fault_trace=[0] * (6 * k_pow)))
        # Since ((H S)^3)^k_pow = omega^k_pow * I, psi_d and psi_s MUST equal omega^k_pow * psi_0!
        assert np.max(np.abs(psi_d - OMEGA8[k_pow] * psi_0)) < 1e-12, k_pow
        assert np.max(np.abs(psi_s - OMEGA8[k_pow] * psi_0)) < 1e-12, k_pow


def test_v1_4_direct_vs_selector_binary_and_categorical_circuits() -> None:
    """`V1.4`: 40 random 3-4Q circuits with binary & multi-branch categorical faults + `gadgetize_pauli_noise`."""
    from stimside.util.phase_qfe import project_diagonal

    rng = np.random.default_rng(20261030)
    cat_1q_templates = (
        "(0.2, H) (0.2, S) (0.15, SQRT_X_DAG)",
        "(0.25, C_XYZ) (0.2, H_YZ) (0.15, Y)",
        "(0.3, S_DAG) (0.25, SQRT_Y)",
    )
    cat_2q_templates = (
        "(0.2, CX) (0.15, SWAP) (0.15, ISWAP)",
        "(0.2, CZ) (0.15, SQRT_XX) (0.15, XCY)",
        "(0.25, ISWAP_DAG) (0.2, SQRT_ZZ_DAG)",
    )

    for idx in range(40):
        n = 3 if (idx % 2 == 0) else 4
        gad_noise = bool(idx % 2 == 1)
        lines: list[str] = [f"H {' '.join(str(q) for q in range(n))}"]
        for step in range(14):
            kind = int(rng.integers(0, 7))
            if kind == 0:
                q = int(rng.integers(0, n))
                tag = "T" if int(rng.integers(0, 2)) == 0 else "T_DAG"
                lines.append(f"I[{tag}] {q}")
            elif kind == 1:
                qs = rng.choice(n, size=3, replace=False)
                lines.append(f"I[CCZ] {int(qs[0])} {int(qs[1])} {int(qs[2])}")
            elif kind == 2:
                g1 = _ALL_1Q_GATES[int(rng.integers(0, len(_ALL_1Q_GATES)))]
                lines.append(f"{g1} {int(rng.integers(0, n))}")
            elif kind == 3:
                g2 = _ALL_2Q_GATES[int(rng.integers(0, len(_ALL_2Q_GATES)))]
                qs = rng.choice(n, size=2, replace=False)
                lines.append(f"{g2} {int(qs[0])} {int(qs[1])}")
            elif kind == 4:
                q = int(rng.integers(0, n))
                tmpl = cat_1q_templates[int(rng.integers(0, len(cat_1q_templates)))]
                lines.append(f"I[CLIFFORD_FAULT: {tmpl}] {q}")
            elif kind == 5:
                qs = rng.choice(n, size=2, replace=False)
                tmpl = cat_2q_templates[int(rng.integers(0, len(cat_2q_templates)))]
                lines.append(f"I[CLIFFORD_FAULT: {tmpl}] {int(qs[0])} {int(qs[1])}")
            else:
                if step % 2 == 0:
                    q = int(rng.integers(0, n))
                    lines.append(f"DEPOLARIZE1(0.25) {q}")
                else:
                    gm = ("M", "MX", "MY")[int(rng.integers(0, 3))]
                    q = int(rng.integers(0, n))
                    lines.append(f"{gm} {q}")

        # Ensure at least one categorical fault channel exists in every circuit
        lines.insert(2, f"I[CLIFFORD_FAULT: {cat_1q_templates[0]}] 0")
        lines.insert(5, f"I[CLIFFORD_FAULT: {cat_2q_templates[0]}] 0 1")
        circ = stim.Circuit("\n".join(lines))

        dp = compile_direct(circ, c2_budget=32)
        sp = compile_selector(circ, gadgetize_pauli_noise=gad_noise, c2_budget=32)

        L = len(dp.fault_channels)
        assert L >= 2
        # Test 3 distinct realizations per circuit: all-inactive, single-active, and random multi-active
        traces = [
            [0] * L,
            [min(1, len(dp.fault_channels[0].branches))] + [0] * (L - 1),
            [int(rng.integers(0, len(ch.branches) + 1)) for ch in dp.fault_channels],
        ]

        for f_trace in traces:
            rd = run_direct(dp, rng=rng, fault_trace=f_trace)
            rs_full = run_selector(
                sp,
                rng=rng,
                fault_trace=f_trace,
                pauli_noise_trace=rd.pauli_noise_choices,
                reset_outcomes=rd.reset_choices,
                requested_quantum_outcomes=rd.quantum_outcomes,
                requested_outcomes={i: int(b) for i, b in enumerate(rd.measurements)},
            )
            assert rs_full.selector_bits is not None

            # Verify augmented selector state has < 1e-15 mass on every WRONG selector bit 1 - c_k!
            n_w_aug = n_words_for_qubits(sp.num_total_qubits)
            for s_off, bit_val in enumerate(rs_full.selector_bits.tolist()):
                s_q = sp.num_data_qubits + s_off
                # Check Z_{s_q} in augmented expansion (since Z_s commutes with everything!)
                _, z_wrong = project_diagonal(
                    rs_full.expansion,
                    unit_bit_words(s_q, n_w_aug),
                    0,
                    1 - int(bit_val),
                )
                assert abs(z_wrong) < 1e-15, (idx, s_off, z_wrong)

            dtab, dexp = extract_data_expansion(sp, rs_full.expansion, rs_full.selector_bits)
            psi_s = reconstruct_statevector(dtab, dexp)
            psi_d = reconstruct_statevector(rd)
            psi_dense, meas_dense, mass_dense, _ = dense_simulate_trajectory_le(
                circ,
                num_qubits=n,
                fault_choices=rd.fault_choices,
                pauli_noise_choices=rd.pauli_noise_choices,
                reset_choices=rd.reset_choices,
                quantum_outcomes=rd.quantum_outcomes,
                readout_flips=rd.readout_flips,
            )

            # Strict ZERO-PHASE-FUDGE assertions (catches any F_j^2 double-application!)
            assert np.max(np.abs(psi_s - psi_d)) < 1e-10, (idx, gad_noise, f_trace)
            assert np.max(np.abs(psi_s - psi_dense)) < 1e-10, (idx, gad_noise, f_trace)
            assert abs(rs_full.branch_mass - rd.branch_mass) < 1e-10
            assert abs(rs_full.branch_mass - mass_dense) < 1e-10
            assert np.array_equal(rs_full.measurements, meas_dense)


def test_v1_4_direct_vs_selector_sampled_statistics() -> None:
    """`V1.4`: `S = 3,000` Monte Carlo shots from `DIRECT` & `SELECTOR` vs exact 8-outcome distribution."""
    circ = stim.Circuit(
        "H 0 1 2\n"
        "I[T] 0 1\n"
        "I[CLIFFORD_FAULT: (0.20, H) (0.15, S)] 0\n"
        "I[CCZ] 0 1 2\n"
        "I[CLIFFORD_FAULT: (0.18, CX) (0.12, SWAP)] 1 2\n"
        "DEPOLARIZE1(0.09) 1\n"
        "H 0 2\n"
        "I[T_DAG] 2\n"
        "M 0 1 2"
    )
    # Compute exact 8-outcome probability distribution by enumerating all 3 * 3 * 4 = 36 fault/noise branches!
    p_slot0 = (0.65, 0.20, 0.15)
    g_slot0 = (None, "H", "S")
    p_slot1 = (0.70, 0.18, 0.12)
    g_slot1 = (None, "CX", "SWAP")
    p_dep1 = (0.91, 0.03, 0.03, 0.03)
    g_dep1 = (None, "X", "Y", "Z")

    p_exact = np.zeros(8, dtype=np.float64)
    H3 = embed_unitary_le(exact_stim_gate_local_c128("H"), [0], 3) @ (
        embed_unitary_le(exact_stim_gate_local_c128("H"), [1], 3)
        @ embed_unitary_le(exact_stim_gate_local_c128("H"), [2], 3)
    )
    T01 = embed_unitary_le(T_MAT, [1], 3) @ embed_unitary_le(T_MAT, [0], 3)
    U_ccz = embed_unitary_le(CCZ_MAT, [0, 1, 2], 3)
    U_tail = (
        embed_unitary_le(T_DAG_MAT, [2], 3)
        @ embed_unitary_le(exact_stim_gate_local_c128("H"), [2], 3)
        @ embed_unitary_le(exact_stim_gate_local_c128("H"), [0], 3)
    )
    psi_base = T01 @ (H3 @ np.array([1.0] + [0.0] * 7, dtype=np.complex128))

    for w0, g0 in zip(p_slot0, g_slot0):
        psi0 = (embed_unitary_le(exact_stim_gate_local_c128(g0), [0], 3) @ psi_base) if g0 else psi_base
        psi_mid = U_ccz @ psi0
        for w1, g1 in zip(p_slot1, g_slot1):
            psi1 = (embed_unitary_le(exact_stim_gate_local_c128(g1), [1, 2], 3) @ psi_mid) if g1 else psi_mid
            for w2, g2 in zip(p_dep1, g_dep1):
                psi2 = (embed_unitary_le(exact_stim_gate_local_c128(g2), [1], 3) @ psi1) if g2 else psi1
                psi_f = U_tail @ psi2
                p_exact += (w0 * w1 * w2) * (np.abs(psi_f) ** 2)

    assert abs(float(np.sum(p_exact)) - 1.0) < 1e-14

    S = 3000
    rng_d = np.random.default_rng(20261101)
    rng_s = np.random.default_rng(20261102)
    dp = compile_direct(circ, c2_budget=32)
    sp = compile_selector(circ, gadgetize_pauli_noise=True, c2_budget=32)

    counts_d = np.zeros(8, dtype=np.int64)
    counts_s = np.zeros(8, dtype=np.int64)
    for _ in range(S):
        rd = run_direct(dp, rng=rng_d)
        idx_d = int(rd.measurements[0]) | (int(rd.measurements[1]) << 1) | (int(rd.measurements[2]) << 2)
        counts_d[idx_d] += 1

        rs = run_selector(sp, rng=rng_s)
        assert rs.selector_bits is not None
        # Verify one-hot mutual exclusion on every categorical selector group!
        for grp in sp.fault_selector_groups:
            active_in_grp = sum(int(rs.selector_bits[q_s - sp.num_data_qubits]) for q_s in grp)
            assert active_in_grp <= 1
        for _, _, grp_n in sp.noise_selector_groups:
            active_in_grp = sum(int(rs.selector_bits[q_s - sp.num_data_qubits]) for q_s in grp_n)
            assert active_in_grp <= 1
        idx_s = int(rs.measurements[0]) | (int(rs.measurements[1]) << 1) | (int(rs.measurements[2]) << 2)
        counts_s[idx_s] += 1

    freq_d = counts_d / float(S)
    freq_s = counts_s / float(S)
    tvd_d = 0.5 * float(np.sum(np.abs(freq_d - p_exact)))
    tvd_s = 0.5 * float(np.sum(np.abs(freq_s - p_exact)))
    assert tvd_d < 0.03, (tvd_d, freq_d, p_exact)
    assert tvd_s < 0.03, (tvd_s, freq_s, p_exact)

    for k_out in range(8):
        p_k = float(p_exact[k_out])
        tol_k = 4.5 * math.sqrt(max(p_k * (1.0 - p_k), 1e-4) / float(S))
        assert abs(float(freq_d[k_out]) - p_k) <= tol_k
        assert abs(float(freq_s[k_out]) - p_k) <= tol_k


# ---------------------------------------------------------------------------
# Group C: Multi-Word Compiler Remap Invariance & Malformed Tag Negative Tests
# ---------------------------------------------------------------------------


def _reconstruct_5q_active_subspace_via_o1(
    N: int,
    q_map: tuple[int, int, int, int, int],
    data_exp: ShotExpansion,
    data_frame_events: Sequence[object],
) -> np.ndarray:
    """Reconstruct the 32 active-support amplitudes on `q_map` via `N`-qubit `O1` `overlap` (no `2^N` alloc!)."""
    n_w = n_words_for_qubits(N)
    tm = int(tail_mask(N))
    assert (int(data_exp.a_union[-1]) & ~tm) == 0
    if data_exp.d > 0:
        assert np.all((data_exp.A_words[:, -1] & np.uint64(~tm & 0xFFFFFFFFFFFFFFFF)) == 0)
    if data_exp.chi > 0:
        assert np.all((data_exp.h_words[:, -1] & np.uint64(~tm & 0xFFFFFFFFFFFFFFFF)) == 0)

    active_set = set(q_map)
    inv_map = {q_map[j]: j for j in range(5)}
    for s_spec in range(N):
        if s_spec not in active_set:
            assert get_bit_in_words(data_exp.a_union, s_spec) == 0
            for c_i in range(data_exp.chi):
                assert get_bit_in_words(data_exp.h_words[c_i], s_spec) == 0

    comps = data_exp.to_components()
    ket5 = np.zeros(32, dtype=np.complex128)
    for y in range(32):
        bra = PhaseQFEComponent.zero(N)
        hw = np.zeros(n_w, dtype=np.uint64)
        for j in range(5):
            if (y >> j) & 1:
                hw |= unit_bit_words(q_map[j], n_w)
        bra.h_words = hw
        amp = 0.0 + 0.0j
        for w_c, d_comp in comps:
            amp += w_c * overlap(bra, d_comp, use_cpp_kernels=False)
        ket5[y] = amp

    # Apply the 5-qubit restriction of data_frame_events using independent dense 5Q matrices
    right_evs = [ev for ev in data_frame_events if isinstance(ev, RightRotEvent)]
    left_evs = [ev for ev in data_frame_events if not isinstance(ev, RightRotEvent)]
    eye5 = np.eye(32, dtype=np.complex128)

    for r_ev in reversed(right_evs):
        # Verify rot_axis has zero X-support on spectators
        for s_spec in range(N):
            if s_spec not in active_set:
                assert get_bit_in_words(r_ev.rot_axis.a_words, s_spec) == 0
        a5 = [get_bit_in_words(r_ev.rot_axis.a_words, q_map[j]) for j in range(5)]
        b5 = [get_bit_in_words(r_ev.rot_axis.b_words, q_map[j]) for j in range(5)]
        P5 = dense_pauli_le(5, a5, b5, r_ev.rot_axis.kappa)
        ket5 = ((eye5 - 1j * P5) / math.sqrt(2.0)) @ ket5

    for l_ev in left_evs:
        if isinstance(l_ev, LeftUnitaryEvent):
            loc_t = [inv_map[q] for q in l_ev.targets]
            ket5 = embed_unitary_le(exact_stim_gate_local_c128(l_ev.gate_name), loc_t, 5) @ ket5
        elif isinstance(l_ev, LeftResetXEvent):
            ket5 = embed_unitary_le(exact_stim_gate_local_c128("X"), [inv_map[l_ev.qubit]], 5) @ ket5
        elif isinstance(l_ev, LeftSppEvent):
            loc_terms = [(inv_map[q], pch) for q, pch in l_ev.terms]
            P5 = dense_pauli_terms_le(5, loc_terms, negate=l_ev.negate)
            fac = OMEGA8[7] * (eye5 + 1j * P5) / math.sqrt(2.0) if l_ev.dag else OMEGA8[1] * (eye5 - 1j * P5) / math.sqrt(2.0)
            ket5 = fac @ ket5
        else:
            raise AssertionError(f"Unexpected frame event {l_ev!r}")

    return ket5


@pytest.mark.parametrize("N_big", [63, 64, 65, 70, 128, 130])
def test_multiword_compiler_direct_and_selector_remap_invariance(N_big: int) -> None:
    """Verify `DIRECT` and `SELECTOR` across word boundaries (`63 -> 66`, `128 -> 131`) under remap."""
    if N_big == 63:
        mono = (0, 1, 31, 61, 62)
    elif N_big == 64:
        mono = (0, 1, 31, 62, 63)
    elif N_big == 65:
        mono = (0, 31, 62, 63, 64)
    elif N_big == 70:
        mono = (0, 62, 63, 64, 69)
    elif N_big == 128:
        mono = (0, 63, 64, 126, 127)
    else:
        mono = (0, 63, 64, 127, 129)
    rev = (mono[4], mono[2], mono[3], mono[1], mono[0])
    assert len(set(mono)) == 5 and len(set(rev)) == 5

    def build_mapped_circuit(qs: Sequence[int], force_num_qubits: int | None) -> stim.Circuit:
        q0, q1, q2, q3, q4 = qs
        lines = []
        if force_num_qubits is not None:
            lines.append(f"QUBIT_COORDS(0) {force_num_qubits - 1}")
        lines.extend([
            f"H {q0} {q1} {q2} {q3} {q4}",
            f"I[T] {q0} {q2}",
            f"CX {q0} {q3} {q1} {q4}",
            f"I[CLIFFORD_FAULT: (0.4, H) (0.3, S)] {q1}",
            f"I[CCZ] {q0} {q2} {q4}",
            f"X_ERROR(0.5) {q3}",
            f"I[CLIFFORD_FAULT: (0.5, ISWAP)] {q2} {q3}",
            f"SPP X{q0}*Z{q4}",
            f"MPP Y{q1}*X{q2}",
            f"MRX {q3}",
            f"CX rec[-2] {q4}",
            f"I[T_DAG] {q4} {q0}",
            f"M {q0}",
        ])
        return stim.Circuit("\n".join(lines))

    circ_5q = build_mapped_circuit((0, 1, 2, 3, 4), None)
    # Fixed realization across all embeddings: fault_choices=[1, 1] (K_sel = 2 + 1 = 3!),
    # so N_big=63 -> N_aug=66 (crosses 64!) and N_big=128 -> N_aug=131 (crosses 128!)
    f_trace = [1, 1]
    n_trace = [1]
    r_trace = [1]
    q_outcomes = {0: 1, 1: 1, 2: 0}
    c_outcomes = {0: 1, 1: 1, 2: 0}

    psi_5q_dense, meas_5q, mass_5q, _ = dense_simulate_trajectory_le(
        circ_5q,
        num_qubits=5,
        fault_choices=f_trace,
        pauli_noise_choices=n_trace,
        reset_choices=r_trace,
        quantum_outcomes=q_outcomes,
        readout_flips={0: False, 1: False, 2: False},
    )

    for q_map in (mono, rev):
        circ_big = build_mapped_circuit(q_map, N_big)
        dp = compile_direct(circ_big, c2_budget=32)
        sp = compile_selector(circ_big, c2_budget=32)
        assert dp.num_qubits == N_big
        assert sp.num_data_qubits == N_big and sp.num_selector_qubits == 3
        assert sp.num_total_qubits == N_big + 3

        rd = run_direct(
            dp,
            fault_trace=f_trace,
            pauli_noise_trace=n_trace,
            reset_outcomes=r_trace,
            requested_quantum_outcomes=q_outcomes,
            requested_outcomes=c_outcomes,
        )
        rs = run_selector(
            sp,
            fault_trace=f_trace,
            pauli_noise_trace=n_trace,
            reset_outcomes=r_trace,
            requested_quantum_outcomes=q_outcomes,
            requested_outcomes=c_outcomes,
        )
        assert rs.selector_bits is not None
        # Check augmented expansion upper tail bits on N_big + 3 as well!
        tm_aug = int(tail_mask(sp.num_total_qubits))
        assert (int(rs.expansion.a_union[-1]) & ~tm_aug) == 0

        dtab, dexp = extract_data_expansion(sp, rs.expansion, rs.selector_bits)
        psi_act_dir = _reconstruct_5q_active_subspace_via_o1(
            N_big, q_map, rd.expansion, dp.frame_events
        )
        psi_act_sel = _reconstruct_5q_active_subspace_via_o1(
            N_big, q_map, dexp, sp.data_frame_events
        )
        del dtab

        assert np.max(np.abs(psi_act_dir - psi_5q_dense)) < 1e-10, (N_big, q_map)
        assert np.max(np.abs(psi_act_sel - psi_5q_dense)) < 1e-10, (N_big, q_map)
        assert abs(float(np.sum(np.abs(psi_act_dir) ** 2)) - 1.0) < 1e-10
        assert abs(rd.branch_mass - mass_5q) < 1e-10
        assert abs(rs.branch_mass - rd.branch_mass) < 1e-10
        assert np.array_equal(rd.measurements, meas_5q)
        assert np.array_equal(rs.measurements, meas_5q)


@pytest.mark.parametrize(
    "bad_circuit_text",
    [
        # (1) Bad CCZ arity (len % 3 != 0)
        "I[CCZ] 0 1",
        "I[CCZ] 0 1 2 3",
        # (2) Duplicate qubit inside CCZ triple
        "I[CCZ] 0 1 0",
        "I[CCZ] 2 2 2",
        "I[CCZ] 0 1 2 4 3 4",
        # (3) Magic tag on op.name != "I"
        "X[T] 0",
        "H[T_DAG] 0",
        "CZ[CCZ] 0 1",
        "M[T] 0",
        # (5) Bad CLIFFORD_FAULT probabilities
        "I[CLIFFORD_FAULT: (-0.1, H)] 0",
        "I[CLIFFORD_FAULT: (0.7, H) (0.5, S)] 0",
        "I[CLIFFORD_FAULT: (1.2, X)] 0",
        "I[CLIFFORD_FAULT: (nan, H)] 0",
        # (6) Unknown / non-Clifford / measurement gate inside CLIFFORD_FAULT
        "I[CLIFFORD_FAULT: (0.1, UNKNOWN_GATE)] 0",
        "I[CLIFFORD_FAULT: (0.1, M)] 0",
        "I[CLIFFORD_FAULT: (0.1, T)] 0",
        # (7) Bad CLIFFORD_FAULT arity or duplicate targets
        "I[CLIFFORD_FAULT: (0.1, CX)] 0",
        "I[CLIFFORD_FAULT: (0.1, CZ)] 0 1 2",
        "I[CLIFFORD_FAULT: (0.1, CX)] 1 1",
        "I[CLIFFORD_FAULT: (0.1, H) (0.1, CX)] 0 1",
        # (8) CLIFFORD_FAULT on non-I instruction or malformed tag syntax
        "H[CLIFFORD_FAULT: (0.1, X)] 0",
        "I[CLIFFORD_FAULT: malformed] 0",
        "I[CLIFFORD_FAULT: ] 0",
        "I[CLIFFORD_FAULT: (0.1, H) garbage (0.1, S)] 0",
    ],
)
def test_malformed_magic_and_clifford_fault_tags_raise(bad_circuit_text: str) -> None:
    """Verify every malformed magic/fault tag raises `ValueError` in both compilers."""
    circ = stim.Circuit(bad_circuit_text)
    with pytest.raises(ValueError):
        compile_direct(circ)
    with pytest.raises(ValueError):
        compile_selector(circ)
    if "CLIFFORD_FAULT" in bad_circuit_text:
        with pytest.raises(ValueError):
            extract_circuit_faults(circ)


def test_malformed_gate_args_and_programmatic_channels_raise() -> None:
    """Verify non-empty `gate_args` on `I[T]`/`I[CLIFFORD_FAULT]` and bad `CliffordFaultChannel` raise `ValueError`."""
    from stimside.util.magic_compile import (
        _validate_and_emit_magic_op,
        parse_clifford_fault_instruction,
    )
    from stimside.util.reference_chp import ReferenceCHPTableau

    class _MockOpWithArgs:
        def __init__(self, name: str, tag: str, args: Sequence[float]) -> None:
            self.name = name
            self.tag = tag
            self._args = list(args)

        def gate_args_copy(self) -> list[float]:
            return list(self._args)

        def targets_copy(self) -> list[stim.GateTarget]:
            return [stim.GateTarget(0)]

    # (4) Non-empty gate_args on tagged carrier instructions (both Stim I_ERROR and op.name == "I" guards)
    for bad_tag in ("T", "T_DAG", "CLIFFORD_FAULT: (0.1, H)"):
        c_args = stim.Circuit(f"I_ERROR[{bad_tag}](0.1) 0")
        with pytest.raises(ValueError):
            compile_direct(c_args)
        with pytest.raises(ValueError):
            compile_selector(c_args)

        mock_op = _MockOpWithArgs("I", bad_tag, [0.1])
        if bad_tag.startswith("CLIFFORD_FAULT"):
            with pytest.raises(ValueError):
                parse_clifford_fault_instruction(mock_op, 0)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValueError):
                _validate_and_emit_magic_op(mock_op, ReferenceCHPTableau(1), [])  # type: ignore[arg-type]

    # Programmatic FaultBranch / CliffordFaultChannel validation & out-of-range step_index
    with pytest.raises(ValueError):
        FaultBranch(-0.05, "H", (0,))
    with pytest.raises(ValueError):
        FaultBranch(float("nan"), "H", (0,))
    with pytest.raises(ValueError):
        FaultBranch(0.2, "CX", (0,))
    with pytest.raises(ValueError):
        FaultBranch(0.2, "CX", (1, 1))
    with pytest.raises(ValueError):
        CliffordFaultChannel(step_index=0, branches=(FaultBranch(0.6, "H", (0,)), FaultBranch(0.6, "S", (0,))))
    with pytest.raises(ValueError):
        CliffordFaultChannel(step_index=-1, branches=(FaultBranch(0.2, "H", (0,)),))

    c_small = stim.Circuit("H 0\nI[T] 0")
    ch_oob = CliffordFaultChannel(step_index=5, branches=(FaultBranch(0.2, "H", (0,)),))
    with pytest.raises(ValueError):
        compile_coordinates(c_small, fault_channels=[ch_oob])
    with pytest.raises(ValueError):
        compile_direct(c_small, fault_channels=[ch_oob])
    with pytest.raises(ValueError):
        compile_selector(c_small, fault_channels=[ch_oob])

    # Invalid atol values raise ValueError immediately in run_direct and run_selector
    dp_ok = compile_direct(c_small)
    sp_ok = compile_selector(c_small)
    for bad_atol in (0.0, -1e-10, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            run_direct(dp_ok, atol=bad_atol)
        with pytest.raises(ValueError):
            run_selector(sp_ok, atol=bad_atol)

    # PROJECT.md contract: compile_open_block(records, *, budget=64) without explicit n_qubits
    from stimside.util.phase_qfe import apply_compiled_block

    c_3q = stim.Circuit("H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2")
    coord_3q = compile_coordinates(c_3q)
    for recs_3q in (
        compile_open_block(coord_3q.raw_records, budget=64),
        compile_open_block(coord_3q, budget=64),
    ):
        assert len(recs_3q) == 1 and isinstance(recs_3q[0], CompiledOpenBlockRecord)
        assert recs_3q[0].block.n_qubits == 3
        out_3q = apply_compiled_block(ShotExpansion.zero(3), recs_3q[0].block)
        assert isinstance(out_3q, ShotExpansion) and abs(out_3q.norm_sq() - 1.0) < 1e-12

    # Manual CompiledPauliAxis without n_qubits hint also works when word count matches
    manual_ax = CompiledPauliAxis(
        a_words=np.array([1], dtype=np.uint64),
        b_words=np.array([0], dtype=np.uint64),
        kappa=0,
    )
    recs_man = compile_open_block(
        [MagicTermRecord(MagicPhaseTerm(axes=(manual_ax,), kind="T", epsilon=1))],
        budget=64,
    )
    assert len(recs_man) == 1 and isinstance(recs_man[0], CompiledOpenBlockRecord)
    out_man = apply_compiled_block(ShotExpansion.zero(3), recs_man[0].block)
    assert isinstance(out_man, ShotExpansion) and abs(out_man.norm_sq() - 1.0) < 1e-12

    # Explicit n_qubits mismatch still raises ValueError
    with pytest.raises(ValueError):
        apply_compiled_block(ShotExpansion.zero(2), recs_3q[0].block)


# ---------------------------------------------------------------------------
# Group D: M1 Iteration 2 Gate Regression Suite (Findings 1-6 & Bugs 1-5)
# ---------------------------------------------------------------------------


def test_m1_r2_step_index_before_op_and_mixed_chronological_fault_alignment() -> None:
    """Verify before-op `step_index` in `0..len(ops)` and mixed inline/external chronological slot alignment."""
    # 1. Empty circuit accepts step_index=0 and rejects step_index=1
    c_empty = stim.Circuit()
    ch_empty = CliffordFaultChannel(step_index=0, branches=(FaultBranch(1.0, "X", (0,)),))
    dp_e = compile_direct(c_empty, fault_channels=[ch_empty])
    sp_e = compile_selector(c_empty, fault_channels=[ch_empty])
    ket_1 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    assert np.max(np.abs(reconstruct_statevector(run_direct(dp_e, fault_trace=[1])) - ket_1)) < 1e-14
    assert np.max(
        np.abs(reconstruct_statevector(run_selector(sp_e, fault_trace=[1], extract_data=True)) - ket_1)
    ) < 1e-14
    with pytest.raises(ValueError):
        compile_direct(c_empty, fault_channels=[CliffordFaultChannel(1, (FaultBranch(0.5, "X", (0,)),))])

    # 2. All 3 gap positions 0, 1, 2 on 2-op circuit `I[T] 0; I[T_DAG] 0`
    c_tt = stim.Circuit("I[T] 0\nI[T_DAG] 0")
    omega_conj = np.exp(-1j * math.pi / 4.0)
    expected_by_gap = {
        0: np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),           # T_DAG T H |0> = |+>
        1: np.array([1.0, omega_conj], dtype=np.complex128) / math.sqrt(2.0),    # T_DAG H T |0> = |T_DAG>
        2: np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),           # H T_DAG T |0> = |+>
    }
    for gap_idx, exp_ket in expected_by_gap.items():
        ch = CliffordFaultChannel(step_index=gap_idx, branches=(FaultBranch(0.5, "H", (0,)),))
        dp = compile_direct(c_tt, fault_channels=[ch])
        sp = compile_selector(c_tt, fault_channels=[ch])
        psi_ora, _, _, _ = dense_simulate_trajectory_le(
            c_tt, num_qubits=1, fault_channels=[ch], fault_choices=[1]
        )
        assert np.max(np.abs(psi_ora - exp_ket)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(run_direct(dp, fault_trace=[1])) - exp_ket)) < 1e-14
        assert np.max(
            np.abs(reconstruct_statevector(run_selector(sp, fault_trace=[1], extract_data=True)) - exp_ket)
        ) < 1e-14

    # 3. Out-of-order external channels + inline CLIFFORD_FAULT (Challenger Bug 2a & 2b + Reviewer Finding 5)
    c_mix = stim.Circuit("H 0\nI[T] 0\nI[CLIFFORD_FAULT: (0.25, H)] 0\nI[T_DAG] 0\nM 1")
    ch_late = CliffordFaultChannel(
        step_index=5,
        branches=(FaultBranch(1.0, "X", (1,)),),
        slot_id=20,
    )
    ch_early = CliffordFaultChannel(
        step_index=1,
        branches=(FaultBranch(0.0, "X", (0,)), FaultBranch(1.0, "S", (0,))),
        slot_id=0,  # Explicitly tests slot_id=0 at slot_index=0 vs non-colliding auto slot_id on inline slot!
    )
    # Pass out-of-order [ch_late, ch_early]; chronological order is: ch_early (step 1), inline (step 2), ch_late (step 5)
    dp_m = compile_direct(c_mix, fault_channels=[ch_late, ch_early])
    sp_m = compile_selector(c_mix, fault_channels=[ch_late, ch_early])
    assert [c.step_index for c in dp_m.fault_channels] == [1, 2, 5]
    assert [c.step_index for c in sp_m.fault_channels] == [1, 2, 5]
    assert [c.slot_id for c in dp_m.fault_channels] == [0, 1, 20]
    assert [len(g) for g in sp_m.fault_selector_groups] == [2, 1, 1]

    for trace in ([2, 1, 1], {0: 2, 1: 1, 20: 1}, {0: 2, 20: 1}):
        rd_m = run_direct(dp_m, fault_trace=trace, requested_quantum_outcomes={0: 0})
        rs_m = run_selector(
            sp_m, fault_trace=trace, requested_quantum_outcomes={0: 0}, extract_data=True
        )
        psi_ora_m, meas_ora_m, _, _ = dense_simulate_trajectory_le(
            c_mix,
            num_qubits=2,
            fault_channels=[ch_early, ch_late],
            fault_choices=rd_m.fault_choices,
            quantum_outcomes=rd_m.quantum_outcomes,
        )
        assert rd_m.fault_choices == rs_m.fault_choices
        assert rd_m.measurements.tolist() == [False]
        assert np.array_equal(rd_m.measurements, rs_m.measurements)
        assert np.array_equal(rd_m.measurements, meas_ora_m)
        assert np.max(np.abs(reconstruct_statevector(rd_m) - psi_ora_m)) < 1e-13
        assert np.max(np.abs(reconstruct_statevector(rs_m) - psi_ora_m)) < 1e-13

    # Reviewer Finding 5: slot_id=10 at slot_index=0 and slot_id=0 at slot_index=1 with fault_trace={0: 1}
    dp_col = compile_direct(
        c_tt,
        fault_channels=[
            CliffordFaultChannel(0, (FaultBranch(0.5, "X", (0,)),), slot_id=10),
            CliffordFaultChannel(2, (FaultBranch(0.5, "Z", (0,)),), slot_id=0),
        ],
    )
    sp_col = compile_selector(
        c_tt,
        fault_channels=[
            CliffordFaultChannel(0, (FaultBranch(0.5, "X", (0,)),), slot_id=10),
            CliffordFaultChannel(2, (FaultBranch(0.5, "Z", (0,)),), slot_id=0),
        ],
    )
    assert run_direct(dp_col, fault_trace={0: 1}).fault_choices == (0, 1)
    assert run_selector(sp_col, fault_trace={0: 1}).fault_choices == (0, 1)

    # Unknown Mapping fault_trace key (e.g. {99: 1}), bool fields, empty branches=(), duplicate slot_id raise ValueError
    with pytest.raises(ValueError):
        run_direct(dp_col, fault_trace={99: 1})
    with pytest.raises(ValueError):
        run_selector(sp_col, fault_trace={99: 1})
    with pytest.raises(ValueError):
        sample_selector_activations(sp_col, fault_trace={99: 1})
    with pytest.raises(ValueError):
        CliffordFaultChannel(step_index=0, branches=())
    with pytest.raises(ValueError):
        CliffordFaultChannel(step_index=True, branches=(FaultBranch(0.5, "X", (0,)),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CliffordFaultChannel(step_index=0, branches=(FaultBranch(0.5, "X", (0,)),), slot_id=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FaultBranch(0.5, "X", (True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compile_direct(
            c_tt,
            fault_channels=[
                CliffordFaultChannel(0, (FaultBranch(0.1, "X", (0,)),), slot_id=7),
                CliffordFaultChannel(2, (FaultBranch(0.1, "Z", (0,)),), slot_id=7),
            ],
        )


def test_m1_r2_leakage_tag_barriers_and_unknown_tag_validation() -> None:
    """Verify leakage-tagged instructions act as C2 barriers without mutating ref_chp and unknown tags raise ValueError."""
    leakage_ops = (
        "I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0",
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.5, 1-->2)](0.0) 0",
        "II[LEAKAGE_TRANSITION_2: (1.0, 1_0-->2_3)] 0 1",
        "I[LEAKAGE_TRANSITION_Z: (0.2, 1-->2)] 0",
        "II[LEAKAGE_CONTROLLED_ERROR: (0.2, 2-->Z)] 0 1",
        "X[CONDITIONED_ON_SELF: 2] 0",
        "X[CONDITIONED_ON_OTHER: 2 : 1] 0",
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1",
        "X_ERROR[CONDITIONED_ON_SELF: 2](0.5) 0",
        "SWAP[LEAKAGE_SWAP] 0 1",
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0",
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0",
        "DETECTOR[LEAKAGE_DETECTOR: 0] rec[-1]",
    )
    for mid_leak in leakage_ops:
        circ = stim.Circuit(f"M 1\nI[T] 0\n{mid_leak}\nI[T_DAG] 0")
        coord = compile_coordinates(circ)
        dp = compile_direct(circ, c2_budget=64)
        sp_f = compile_selector(circ, gadgetize_pauli_noise=False, c2_budget=64)
        sp_t = compile_selector(circ, gadgetize_pauli_noise=True, c2_budget=64)

        assert sum(isinstance(r, LeakageBarrierRecord) for r in dp.records) == 1
        assert sum(isinstance(r, CompiledOpenBlockRecord) for r in dp.records) == 2
        assert sum(isinstance(r, CompiledOpenBlockRecord) for r in sp_f.records) == 2
        assert sum(isinstance(r, CompiledOpenBlockRecord) for r in sp_t.records) == 2
        # Gadgetized Pauli selector count must NOT allocate selectors for leakage-tagged X_ERROR!
        assert sp_t.num_selector_qubits == 0
        # Conditional X[CONDITIONED_ON_SELF: 2] 0 must NOT unconditionally flip Z0 -> -Z0 in ref_chp!
        assert int(coord.final_tableau.z_kap[0]) == 0
        # Standalone execution without LeakageUint8Magic must raise ValueError cleanly
        with pytest.raises(ValueError):
            run_direct(dp)
        with pytest.raises(ValueError):
            run_selector(sp_t)

    for bad_tag_circ in (
        "I[TDAG] 0",
        "I[TOFFOLI] 0 1 2",
        "H[UNKNOWN_TAG] 0",
        "X_ERROR[BOGUS](0.1) 0",
        "M[CONDITIONED_ON_SELF: 2] 0",
        "CX[LEAKAGE_SWAP] 0 1",
        "DETECTOR[LEAKAGE_DETECTOR]",
        "SWAP[ LEAKAGE_SWAP] 0 1",
    ):
        c_bad = stim.Circuit(bad_tag_circ)
        with pytest.raises(ValueError):
            compile_coordinates(c_bad)
        with pytest.raises(ValueError):
            compile_direct(c_bad)
        with pytest.raises(ValueError):
            compile_selector(c_bad, gadgetize_pauli_noise=True)


def test_m1_r2_sample_selector_activations_public_api() -> None:
    """Verify public `sample_selector_activations` one-hot invariants, trace overrides, and seed reproducibility."""
    import stimside.util.magic_compile as mc_mod

    assert "sample_selector_activations" in mc_mod.__all__
    assert "LeakageBarrierRecord" in mc_mod.__all__

    circ = stim.Circuit(
        "H 0 1\n"
        "I[CLIFFORD_FAULT: (0.3, H) (0.4, S)] 0\n"
        "DEPOLARIZE1(0.6) 1\n"
        "CORRELATED_ERROR(0.5) X0 Z1\n"
        "ELSE_CORRELATED_ERROR(0.8) Y0\n"
        "I[T] 0 1"
    )
    sp = compile_selector(circ, gadgetize_pauli_noise=True)
    # 2 fault selectors + 3 DEPOLARIZE1 + 1 CORR + 1 ELSE_CORR = 7 selector qubits
    assert sp.num_selector_qubits == 7

    bits_a = np.asarray(sample_selector_activations(sp, rng=12345), dtype=np.bool_)
    bits_b = np.asarray(sample_selector_activations(sp, rng=12345), dtype=np.bool_)
    assert bits_a.shape == (7,) and bits_a.dtype == np.bool_
    assert np.array_equal(bits_a, bits_b)

    # Explicit trace override: fault branch 2 (index 1), DEPOLARIZE1 Y (index 3), CORR=1 (index 5), ELSE_CORR=1 blocked!
    bits_det = np.asarray(
        sample_selector_activations(sp, fault_trace=[2], pauli_noise_trace=[2, 1, 1]),
        dtype=np.bool_,
    )
    assert bits_det.tolist() == [False, True, False, True, False, True, False]

    # Stochastic 256-shot mutual exclusion across all categorical groups and CORR/ELSE_CORR chain
    rng = np.random.default_rng(90210)
    for _ in range(256):
        b = np.asarray(sample_selector_activations(sp, rng=rng), dtype=np.bool_)
        assert int(b[0]) + int(b[1]) <= 1
        assert int(b[2]) + int(b[3]) + int(b[4]) <= 1
        assert int(b[5]) + int(b[6]) <= 1


def test_m1_r2_gadgetize_pauli_noise_correlated_and_heralded_interleaving() -> None:
    """Verify `compile_selector(gadgetize_pauli_noise=True)` on interleaved Pauli, `CORRELATED_ERROR`, and `HERALDED_*`."""
    circ = stim.Circuit(
        "H 0 1 2\n"
        "X_ERROR(0.25) 0\n"                             # noise ch 0 (gadgetized CAT, 1 sel)
        "HERALDED_ERASE(0.40) 1\n"                      # noise ch 1 (HeraldedNoiseRecord, m0)
        "CORRELATED_ERROR(0.35) X0 Z2\n"                # noise ch 2 (gadgetized CORR, 1 sel)
        "ELSE_CORRELATED_ERROR(0.60) Y1 X2\n"           # noise ch 3 (gadgetized ELSE_CORR, 1 sel)
        "HERALDED_PAULI_CHANNEL_1(0.05, 0.15, 0.10, 0.10) 2\n"  # noise ch 4 (HeraldedNoiseRecord, m1)
        "Z_ERROR(0.30) 0\n"                             # noise ch 5 (gadgetized CAT, 1 sel)
        "I[CCZ] 0 1 2\n"
        "H 0 1 2\n"
        "M 0 1 2"
    )
    dp = compile_direct(circ)
    sp_f = compile_selector(circ, gadgetize_pauli_noise=False)
    sp_t = compile_selector(circ, gadgetize_pauli_noise=True)
    assert sp_t.num_selector_qubits == 4

    # 1. Exhaustive check over key interleaved pauli_noise_trace realizations (including ELSE blocking [1, 1])
    traces = [
        [0, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 2, 1],
        [0, 4, 1, 1, 0, 1],  # CORR=1 blocks ELSE_CORR=1 even with HERALDED channels interleaved!
        [1, 1, 0, 1, 4, 0],  # CORR=0 allows ELSE_CORR=1!
    ]
    for p_tr in traces:
        rd = run_direct(dp, pauli_noise_trace=p_tr, requested_quantum_outcomes={2: 0, 3: 1, 4: 0})
        rs_f = run_selector(
            sp_f, pauli_noise_trace=p_tr, requested_quantum_outcomes={2: 0, 3: 1, 4: 0}, extract_data=True
        )
        rs_t = run_selector(
            sp_t, pauli_noise_trace=p_tr, requested_quantum_outcomes={2: 0, 3: 1, 4: 0}, extract_data=True
        )
        assert rs_t.peak_dim == 0
        assert rd.pauli_noise_choices == rs_f.pauli_noise_choices == rs_t.pauli_noise_choices
        assert np.array_equal(rd.measurements, rs_t.measurements)
        psi_ora, meas_ora, _, _ = dense_simulate_trajectory_le(
            circ,
            pauli_noise_choices=rd.pauli_noise_choices,
            quantum_outcomes=rd.quantum_outcomes,
        )
        assert np.array_equal(rs_t.measurements, meas_ora)
        assert np.max(np.abs(reconstruct_statevector(rs_t) - psi_ora)) < 1e-13
        assert np.max(np.abs(reconstruct_statevector(rd) - psi_ora)) < 1e-13

    # 2. Stochastic sampling check vs exact 32-outcome analytical distribution (verifying pauli_noise_choices len == 6)
    H3 = (
        embed_unitary_le(exact_stim_gate_local_c128("H"), [2], 3)
        @ embed_unitary_le(exact_stim_gate_local_c128("H"), [1], 3)
        @ embed_unitary_le(exact_stim_gate_local_c128("H"), [0], 3)
    )
    U_post = H3 @ embed_unitary_le(CCZ_MAT, [0, 1, 2], 3)
    psi_h3 = H3 @ np.array([1.0] + [0.0] * 7, dtype=np.complex128)
    p_exact_32 = np.zeros(32, dtype=np.float64)
    for w_x0, t_x0 in ((0.75, []), (0.25, [(0, "X")])):
        s0 = (dense_pauli_terms_le(3, t_x0) @ psi_h3) if t_x0 else psi_h3
        for w_h1, m0_b, t_h1 in (
            (0.60, 0, []),
            (0.10, 1, []),
            (0.10, 1, [(1, "X")]),
            (0.10, 1, [(1, "Y")]),
            (0.10, 1, [(1, "Z")]),
        ):
            s1 = (dense_pauli_terms_le(3, t_h1) @ s0) if t_h1 else s0
            for w_corr, t_corr in (
                (0.35, [(0, "X"), (2, "Z")]),
                (0.65 * 0.60, [(1, "Y"), (2, "X")]),
                (0.65 * 0.40, []),
            ):
                s2 = (dense_pauli_terms_le(3, t_corr) @ s1) if t_corr else s1
                for w_h2, m1_b, t_h2 in (
                    (0.60, 0, []),
                    (0.05, 1, []),
                    (0.15, 1, [(2, "X")]),
                    (0.10, 1, [(2, "Y")]),
                    (0.10, 1, [(2, "Z")]),
                ):
                    s3 = (dense_pauli_terms_le(3, t_h2) @ s2) if t_h2 else s2
                    for w_z0, t_z0 in ((0.70, []), (0.30, [(0, "Z")])):
                        s4 = (dense_pauli_terms_le(3, t_z0) @ s3) if t_z0 else s3
                        probs_q = np.abs(U_post @ s4) ** 2
                        w_tot = w_x0 * w_h1 * w_corr * w_h2 * w_z0
                        base_idx = m0_b | (m1_b << 1)
                        for q_bits in range(8):
                            p_exact_32[base_idx | (q_bits << 2)] += w_tot * float(probs_q[q_bits])
    assert abs(float(np.sum(p_exact_32)) - 1.0) < 1e-14

    rng_d = np.random.default_rng(4401)
    rng_s = np.random.default_rng(4402)
    counts_d = np.zeros(32, dtype=np.int64)
    counts_s = np.zeros(32, dtype=np.int64)
    S = 4000
    for _ in range(S):
        rd_s = run_direct(dp, rng=rng_d)
        rs_s = run_selector(sp_t, rng=rng_s)
        assert len(rd_s.pauli_noise_choices) == 6
        assert len(rs_s.pauli_noise_choices) == 6
        # Verify ELSE_CORR never co-fires with CORR
        assert not (rs_s.pauli_noise_choices[2] == 1 and rs_s.pauli_noise_choices[3] == 1)
        idx_d = sum(int(rd_s.measurements[k]) << k for k in range(5))
        idx_s = sum(int(rs_s.measurements[k]) << k for k in range(5))
        counts_d[idx_d] += 1
        counts_s[idx_s] += 1
    tvd_d = 0.5 * float(np.sum(np.abs(counts_d / S - p_exact_32)))
    tvd_s = 0.5 * float(np.sum(np.abs(counts_s / S - p_exact_32)))
    assert tvd_d < 0.055
    assert tvd_s < 0.055


def test_m1_r2_duplicate_qubit_correlated_error_pauli_reduction() -> None:
    """Verify duplicate-qubit `CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` reduction across all compiler modes."""
    cases = (
        # X0 Z0 == Y0 (flips |0> -> i|1> and |+> -> -i|->)
        ("H 0\nI[T] 0\nCORRELATED_ERROR(1.0) X0 Z0\nI[T_DAG] 0", [1]),
        ("H 0\nI[T] 0\nE(1.0) Z0 X0\nI[T_DAG] 0", [1]),
        ("H 0\nI[T] 0\nCORRELATED_ERROR(1.0) X0*Z0\nI[T_DAG] 0", [1]),
        # X0 X0 == I and X0 Y0 Z0 == I, AND firing X0 X0 MUST still block following ELSE_CORRELATED_ERROR!
        ("H 0 1\nI[T] 0 1\nCORRELATED_ERROR(1.0) X0 X0\nELSE_CORRELATED_ERROR(1.0) X1\nI[T_DAG] 0 1", [1, 1]),
        ("H 0 1\nI[T] 0 1\nCORRELATED_ERROR(0.0) X0 X0\nELSE_CORRELATED_ERROR(1.0) X0 Y0 Z0 X1\nI[T_DAG] 0 1", [0, 1]),
    )
    for circ_str, p_trace in cases:
        circ = stim.Circuit(circ_str)
        dp = compile_direct(circ)
        sp_f = compile_selector(circ, gadgetize_pauli_noise=False)
        sp_t = compile_selector(circ, gadgetize_pauli_noise=True)

        rd = run_direct(dp, pauli_noise_trace=p_trace)
        rs_f = run_selector(sp_f, pauli_noise_trace=p_trace, extract_data=True)
        rs_t = run_selector(sp_t, pauli_noise_trace=p_trace, extract_data=True)
        psi_ora, _, _, _ = dense_simulate_trajectory_le(circ, pauli_noise_choices=rd.pauli_noise_choices)

        assert np.max(np.abs(reconstruct_statevector(rd) - psi_ora)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(rs_f) - psi_ora)) < 1e-14
        assert np.max(np.abs(reconstruct_statevector(rs_t) - psi_ora)) < 1e-14

    # Confirm CORRELATED_ERROR(1.0) X0 X0 + ELSE_CORRELATED_ERROR(1.0) X1 matches Stim C++ [False, False] on |00>
    c_block = stim.Circuit("CORRELATED_ERROR(1.0) X0 X0\nELSE_CORRELATED_ERROR(1.0) X1\nM 0 1")
    assert run_direct(compile_direct(c_block), rng=1).measurements.tolist() == [False, False]
    assert run_selector(compile_selector(c_block, gadgetize_pauli_noise=True), rng=1).measurements.tolist() == [
        False,
        False,
    ]


def test_m1_r2_mpad_flip_prob_and_postselection_with_heralded_noise() -> None:
    """Verify `MPAD(p)` flip probability, `requested_outcomes` postselection, and `HERALDED_*` `!q` Stim parity."""
    # 1. Deterministic MPAD(1.0) 0 1 -> [True, False]
    c_mp1 = stim.Circuit("MPAD(1.0) 0 1")
    for res in (
        run_direct(compile_direct(c_mp1), rng=0),
        run_selector(compile_selector(c_mp1), rng=0),
    ):
        assert res.measurements.tolist() == [True, False]
        assert abs(res.branch_mass - 1.0) < 1e-15

    # 2. Impossible postselection on MPAD 0 with requested_outcomes={0: 1} -> branch_mass == 0.0, log_branch_mass == -inf
    c_mp0 = stim.Circuit("H 0\nI[T] 0\nMPAD 0\nH 0\nM 0")
    dp0 = compile_direct(c_mp0)
    sp0 = compile_selector(c_mp0)
    for r_imp in (
        run_direct(dp0, requested_outcomes={0: 1}, rng=7),
        run_selector(sp0, requested_outcomes={0: 1}, rng=7, extract_data=True),
    ):
        assert bool(r_imp.measurements[0]) is True
        assert r_imp.branch_mass == 0.0
        assert r_imp.log_branch_mass == -math.inf
        assert np.max(np.abs(reconstruct_statevector(r_imp))) == 0.0

    # 3. Stochastic MPAD(0.3) 0 1 sampling rates & exact postselected branch masses (0.3 * 0.3 = 0.09)
    c_mp_p = stim.Circuit("MPAD(0.3) 0 1")
    dp_p = compile_direct(c_mp_p)
    sp_p = compile_selector(c_mp_p)
    r_post_d = run_direct(dp_p, requested_outcomes={0: 1, 1: 0})
    r_post_s = run_selector(sp_p, requested_outcomes={0: 1, 1: 0}, extract_data=True)
    assert abs(r_post_d.branch_mass - 0.09) < 1e-14
    assert abs(r_post_s.branch_mass - 0.09) < 1e-14
    assert abs(r_post_d.log_branch_mass - math.log(0.09)) < 1e-14

    rng = np.random.default_rng(5503)
    samples = np.array([run_direct(dp_p, rng=rng).measurements for _ in range(2000)], dtype=np.float64)
    means = samples.mean(axis=0)
    assert abs(means[0] - 0.30) < 0.04 and abs(means[1] - 0.70) < 0.04

    # 4. HeraldedNoiseRecord postselection via requested_outcomes (both 0, 1, and impossible 0.0 / 1.0)
    c_her = stim.Circuit("H 0\nI[T] 0\nHERALDED_ERASE(0.4) 0\nH 0\nM 0")
    dp_h = compile_direct(c_her)
    sp_h = compile_selector(c_her, gadgetize_pauli_noise=True)
    for runner, pl in ((run_direct, dp_h), (run_selector, sp_h)):
        r_no = runner(pl, requested_outcomes={0: 0, 1: 0})
        # On m0=0 (no erase, prob 0.6), state is H T H |0>, so P(M0=0 | m0=0) = cos^2(pi/8)
        expected_m00 = 0.6 * (math.cos(math.pi / 8.0) ** 2)
        assert r_no.measurements.tolist() == [False, False]
        assert r_no.pauli_noise_choices == (0,)
        assert abs(r_no.branch_mass - expected_m00) < 1e-13

        # On m0=1 (erase occurred, prob 0.4), conditional Pauli c_h in {1:I, 2:X, 3:Y, 4:Z} gives
        # single-shot mass 0.4 * cos^2(pi/8) for {I, X} and 0.4 * sin^2(pi/8) for {Y, Z}, whose average is 0.2!
        r_yes = runner(pl, requested_outcomes={0: 1, 1: 0}, rng=42)
        assert r_yes.measurements.tolist() == [True, False]
        c_h_sampled = r_yes.pauli_noise_choices[0]
        assert c_h_sampled in (1, 2, 3, 4)
        exp_cond = 0.4 * (
            (math.cos(math.pi / 8.0) ** 2)
            if c_h_sampled in (1, 2)
            else (math.sin(math.pi / 8.0) ** 2)
        )
        assert abs(r_yes.branch_mass - exp_cond) < 1e-13

        rng_marg = np.random.default_rng(4242)
        marg_masses = [
            runner(pl, requested_outcomes={0: 1, 1: 0}, rng=rng_marg).branch_mass
            for _ in range(1200)
        ]
        assert abs(float(np.mean(marg_masses)) - 0.2) < 0.015

    # Impossible HERALDED_ERASE(0.0) with requested_outcomes={0: 1} -> branch_mass == 0.0
    c_h0 = stim.Circuit("HERALDED_ERASE(0.0) 0")
    r_h0 = run_direct(compile_direct(c_h0), requested_outcomes={0: 1})
    assert r_h0.branch_mass == 0.0 and r_h0.log_branch_mass == -math.inf

    # 5. Mandatory Stim 1.16 parity check: `!q` on HERALDED_ERASE / HERALDED_PAULI_CHANNEL_1 is ignored (NOT inverted!)
    for h_bang_str in (
        "HERALDED_ERASE(1.0) !0",
        "HERALDED_PAULI_CHANNEL_1(0.25, 0.25, 0.25, 0.25) !0",
    ):
        c_bang = stim.Circuit(h_bang_str)
        stim_ref = c_bang.compile_sampler(seed=0).sample(1)[0].tolist()
        assert stim_ref == [True]
        assert run_direct(compile_direct(c_bang), rng=0).measurements.tolist() == stim_ref
        assert run_selector(compile_selector(c_bang, gadgetize_pauli_noise=True), rng=0).measurements.tolist() == stim_ref


def test_m1_r2_c2_budget_1_sequential_and_n63_to_66_word_boundary_crossing() -> None:
    """Verify `c2_budget=1` sequential fallback (`c2_max_rank`/`c2_max_poly_terms`) and `n_data=63 -> n_total=66`."""
    # 1. c2_budget=1 sequential fallback + c2_max_rank=3 / c2_max_poly_terms=8 block splitting
    c_seq = stim.Circuit(
        "H 0 1 2 3\n"
        "I[T] 0 1 2 3\n"
        "I[CCZ] 0 1 2\n"
        "I[CCZ] 1 2 3\n"
        "I[CLIFFORD_FAULT: (0.4, CXSWAP)] 0 1\n"
        "SPP !X0*Z2\n"
        "I[T_DAG] 0 2\n"
        "MRX 1\n"
        "CX rec[-1] 3\n"
        "MPP Z0*Z3"
    )
    dp_64 = compile_direct(c_seq, c2_budget=64)
    dp_1 = compile_direct(c_seq, c2_budget=1, c2_max_rank=3, c2_max_poly_terms=8)
    sp_1 = compile_selector(c_seq, c2_budget=1, c2_max_rank=3, c2_max_poly_terms=8)
    assert any(
        isinstance(r, CompiledOpenBlockRecord) and r.block.mode == "sequential"
        for r in dp_1.records
    )
    for f_c in (0, 1):
        rd_ref = run_direct(dp_64, fault_trace=[f_c], rng=99)
        rd_seq = run_direct(
            dp_1,
            fault_trace=[f_c],
            reset_outcomes=rd_ref.reset_choices,
            requested_quantum_outcomes=rd_ref.quantum_outcomes,
            requested_outcomes={i: int(b) for i, b in enumerate(rd_ref.measurements)},
        )
        rs_seq = run_selector(
            sp_1,
            fault_trace=[f_c],
            reset_outcomes=rd_ref.reset_choices,
            requested_quantum_outcomes=rd_ref.quantum_outcomes,
            requested_outcomes={i: int(b) for i, b in enumerate(rd_ref.measurements)},
            extract_data=True,
        )
        psi_ref = reconstruct_statevector(rd_ref)
        assert np.max(np.abs(reconstruct_statevector(rd_seq) - psi_ref)) < 1e-13
        assert np.max(np.abs(reconstruct_statevector(rs_seq) - psi_ref)) < 1e-13

    # 2. Single-word n_data=63 (1 word) crossing to multi-word n_total=66 (2 words) in SELECTOR
    q_map_63 = (0, 15, 31, 47, 62)
    c_63 = stim.Circuit(
        "QUBIT_COORDS(0) 62\n"
        "H 0 15 31 47 62\n"
        "I[T] 0 31 62\n"
        "I[CLIFFORD_FAULT: (0.3, CZ) (0.3, CXSWAP)] 0 62\n"
        "I[CCZ] 0 31 62\n"
        "X_ERROR(0.4) 47\n"
        "SPP !X0*Z62\n"
        "I[T_DAG] 15 62\n"
        "MPP !Z0*Z47\n"
        "CX rec[-1] 62"
    )
    dp_63 = compile_direct(c_63, c2_budget=1)
    sp_66 = compile_selector(c_63, gadgetize_pauli_noise=True, c2_budget=1)
    assert dp_63.num_qubits == 63 and dp_63.final_tableau.n_words == 1
    assert sp_66.num_data_qubits == 63 and sp_66.num_selector_qubits == 3
    assert sp_66.num_total_qubits == 66 and sp_66.final_tableau.n_words == 2

    for f_choice in (0, 1, 2):
        for n_choice in (0, 1):
            rd_63 = run_direct(
                dp_63,
                fault_trace=[f_choice],
                pauli_noise_trace=[n_choice],
                requested_quantum_outcomes={0: 1},
            )
            rs_66 = run_selector(
                sp_66,
                fault_trace=[f_choice],
                pauli_noise_trace=[n_choice],
                requested_quantum_outcomes={0: 1},
            )
            assert rs_66.expansion.n_words == 2
            assert rs_66.selector_bits is not None
            dtab_63, dexp_63 = extract_data_expansion(sp_66, rs_66.expansion, rs_66.selector_bits)
            assert dtab_63.n_words == 1 and dexp_63.n_words == 1
            # Upper bit 63 in single uint64 word must be strictly zero
            assert int(dexp_63.a_union[0] >> np.uint64(63)) == 0
            if dexp_63.chi > 0:
                assert np.all((dexp_63.h_words[:, 0] >> np.uint64(63)) == 0)

            ket_dir = _reconstruct_5q_active_subspace_via_o1(
                63, q_map_63, rd_63.expansion, dp_63.frame_events
            )
            ket_sel = _reconstruct_5q_active_subspace_via_o1(
                63, q_map_63, dexp_63, sp_66.data_frame_events
            )
            assert np.max(np.abs(ket_dir - ket_sel)) < 1e-12


@pytest.mark.parametrize(
    ("circ_str", "req", "expected_mass", "expected_q0", "expected_flip0"),
    [
        ("M(0.25) 0", {0: 0}, 0.75, 0, False),
        ("M(0.25) 0", {0: 1}, 0.25, 0, True),
        ("M(0.25) !0", {0: 0}, 0.25, 0, True),
        ("M(0.25) !0", {0: 1}, 0.75, 0, False),
        ("MR(0.25) 0\nM 0", {0: 0, 1: 0}, 0.75, 0, False),
        ("MR(0.25) 0\nM 0", {0: 1, 1: 0}, 0.25, 0, True),
        ("H 0\nMX(0.25) 0", {0: 1}, 0.25, 0, True),
        ("MPP(0.25) Z0*Z1", {0: 1}, 0.25, 0, True),
    ],
)
def test_m1_r3_noisy_readout_and_conflicting_postselection(
    circ_str: str,
    req: dict[int, int],
    expected_mass: float,
    expected_q0: int,
    expected_flip0: bool,
) -> None:
    """Finding 1: Exact Bayesian Born mass on noisy `M(p)`/`MR(p)`/`MPP(p)` and conflicting `(q, rep)`."""
    circ = stim.Circuit(circ_str)
    dp = compile_direct(circ)
    sp = compile_selector(circ, gadgetize_pauli_noise=True)
    for res in (
        run_direct(dp, requested_outcomes=req),
        run_selector(sp, requested_outcomes=req, extract_data=True),
    ):
        assert abs(res.branch_mass - expected_mass) < 1e-13
        assert res.quantum_outcomes[0] == expected_q0
        assert res.readout_flips[0] is expected_flip0
        assert [int(res.measurements[k]) for k in req] == [req[k] for k in req]


def test_m1_r3_superposition_posterior_and_postselection_conflicts() -> None:
    """Finding 1: Superposition Bayesian collapse (`P(Q|C)`) and conflicting `(q, rep)` postselection."""
    from stimside.util.phase_qfe import ImpossiblePostselectionError

    # Superposition Bayesian posterior on H 0 -> M(0.25) 0 -> I[T] 0 -> H 0 (P(Q=0|C=0)=0.75, P(Q=1|C=0)=0.25)
    dp_post = compile_direct(stim.Circuit("H 0\nM(0.25) 0\nI[T] 0\nH 0"))
    rng = np.random.default_rng(31415)
    q_samples = []
    for _ in range(800):
        r_p = run_direct(dp_post, requested_outcomes={0: 0}, rng=rng)
        assert abs(r_p.branch_mass - 0.5) < 1e-14
        q_samples.append(r_p.quantum_outcomes[0])
    assert abs(float(np.mean(q_samples)) - 0.25) < 0.045

    # Joint quantum + classical postselection on H 0 -> M(0.25) 0 -> M 0 AND noiseless conflict
    dp_two = compile_direct(stim.Circuit("H 0\nM(0.25) 0\nM 0"))
    assert abs(
        run_direct(dp_two, requested_quantum_outcomes={0: 0}, requested_outcomes={0: 0, 1: 0}).branch_mass - 0.375
    ) < 1e-14
    assert abs(
        run_direct(dp_two, requested_quantum_outcomes={0: 1}, requested_outcomes={0: 0, 1: 1}).branch_mass - 0.125
    ) < 1e-14

    # Impossible quantum eigenvalue still raises ImpossiblePostselectionError even with noisy classical postselection
    with pytest.raises(ImpossiblePostselectionError):
        run_direct(compile_direct(stim.Circuit("M(0.25) 0")), requested_quantum_outcomes={0: 1}, requested_outcomes={0: 1})

    for c_conf_str in ("H 0\nM 0", "H 0\nMR 0"):
        dp_c = compile_direct(stim.Circuit(c_conf_str))
        sp_c = compile_selector(stim.Circuit(c_conf_str))
        for res_c in (
            run_direct(dp_c, requested_quantum_outcomes={0: 0}, requested_outcomes={0: 1}),
            run_selector(sp_c, requested_quantum_outcomes={0: 0}, requested_outcomes={0: 1}),
        ):
            assert res_c.branch_mass == 0.0 and res_c.log_branch_mass == -math.inf


@pytest.mark.parametrize(
    "bad_circ_str",
    [
        "X[CONDITIONED_ON_OTHER: 2 : 0 1] 0\nI 1",
        "X[CONDITIONED_ON_OTHER: 2 : 0 1] 0 1 0",
        "X[CONDITIONED_ON_OTHER: 2 : 0]\nI 0",
        "I[ ] 0",
        "H 0\nI[T] 0\nI[ ] 0\nI[T_DAG] 0",
        "I[ T] 0",
        "I[T ] 0",
        "I[ CCZ] 0 1 2",
    ],
)
def test_m1_r3_conditioned_on_other_arity_and_whitespace_tags(bad_circ_str: str) -> None:
    """Findings 2 & 3: `CONDITIONED_ON_OTHER` target-arity mismatches and whitespace tags raise `ValueError`."""
    c_bad = stim.Circuit(bad_circ_str)
    for fn in (
        compile_coordinates,
        compile_direct,
        lambda c: compile_selector(c, gadgetize_pauli_noise=True),
    ):
        with pytest.raises(ValueError):
            fn(c_bad)

    # Valid fused multiple (4 targets == 2 * len([0, 1])) must still compile cleanly
    c_ok = stim.Circuit("X[CONDITIONED_ON_OTHER: 2 : 0 1] 0 1 0 1")
    assert sum(isinstance(r, LeakageBarrierRecord) for r in compile_direct(c_ok).records) == 1


def test_m1_r3_selector_activations_heralded_and_strict_trace_validation() -> None:
    """Finding 4: `sample_selector_activations` validates heralded channels; traces/outcomes reject bad keys/values."""
    circ = stim.Circuit("X_ERROR(0.2) 0\nY_ERROR(0.2) 0\nHERALDED_ERASE(0.3) 0\nMR 0\nM 0")
    dp = compile_direct(circ)
    sp = compile_selector(circ, gadgetize_pauli_noise=True)

    for bad_p_trace in ({2: 99}, {2: -1}, [0, 0], {True: 0}, {-1: 0}, {99: 0}, {0: True}):
        with pytest.raises(ValueError):
            sample_selector_activations(sp, pauli_noise_trace=bad_p_trace)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            run_selector(sp, pauli_noise_trace=bad_p_trace)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            run_direct(dp, pauli_noise_trace=bad_p_trace)  # type: ignore[arg-type]

    for kw_name, bad_map in (
        ("requested_outcomes", {True: 1}),
        ("requested_outcomes", {-1: 0}),
        ("requested_outcomes", {99: 0}),
        ("requested_outcomes", {0: 2}),
        ("requested_quantum_outcomes", {True: 0}),
        ("requested_quantum_outcomes", {99: 1}),
        ("requested_quantum_outcomes", {1: -1}),
        ("reset_outcomes", {True: 0}),
        ("reset_outcomes", {99: 0}),
        ("reset_outcomes", {0: 2}),
    ):
        with pytest.raises(ValueError):
            run_direct(dp, **{kw_name: bad_map})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            run_selector(sp, **{kw_name: bad_map})  # type: ignore[arg-type]
