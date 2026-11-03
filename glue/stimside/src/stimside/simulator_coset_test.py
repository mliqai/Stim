from concurrent.futures import ThreadPoolExecutor
import dataclasses

import numpy as np
import pytest
import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit, MeasStepMeta
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator


def test_shared_compiled_reference_circuit_no_per_batch_scratch_or_undeclared_attrs() -> None:
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0
        CZ[CONDITIONED_ON_PAIR: (U, U)] 0 2 1 3
        ISWAP[CONDITIONED_ON_PAIR: (U, U)] 0 3 1 2
        M 0
        R 1 2 3
        X_ERROR(0.3) 1 2 3
        M 1 2 3
        """
    )

    def run_once(
        batch_size: int, seed: int, compiled_ref: CompiledReferenceCircuit | None
    ) -> np.ndarray:
        handler = LeakageUint8Coset(unconditional_condition_on_U=True).compile_op_handler(
            circuit=circuit, batch_size=batch_size
        )
        sim = CosetsideSimulator(
            circuit=circuit,
            compiled_op_handler=handler,
            batch_size=batch_size,
            seed=seed,
            compiled_ref=compiled_ref,
        )
        sim.run()
        return sim.get_final_measurement_records()

    expected_a = run_once(5, 101, None)
    expected_b = run_once(17, 202, None)

    shared_ref = CompiledReferenceCircuit(circuit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs_a = [pool.submit(run_once, 5, 101, shared_ref) for _ in range(8)]
        futs_b = [pool.submit(run_once, 17, 202, shared_ref) for _ in range(8)]
        for fut in futs_a:
            np.testing.assert_array_equal(fut.result(), expected_a)
        for fut in futs_b:
            np.testing.assert_array_equal(fut.result(), expected_b)

    for step in shared_ref.steps:
        assert not hasattr(step, "_cpp_m_mat")
        declared = {f.name for f in dataclasses.fields(step)}
        assert set(vars(step)) <= declared, set(vars(step)) - declared
        if isinstance(step, MeasStepMeta):
            for sub in step.substeps:
                sub_declared = {f.name for f in dataclasses.fields(sub.meta)}
                assert set(vars(sub.meta)) <= sub_declared, (
                    set(vars(sub.meta)) - sub_declared
                )


def test_correlated_error_occurred_initialized_and_reset_on_clear() -> None:
    circuit = stim.Circuit("ELSE_CORRELATED_ERROR(1.0) X0\nM 0\nCORRELATED_ERROR(1.0) Z0")
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=8)
    sim = CosetsideSimulator(circuit=circuit, compiled_op_handler=handler, batch_size=8, seed=5)
    assert sim._correlated_error_occurred.shape == (8,)
    assert not np.any(sim._correlated_error_occurred)

    sim.run()
    assert np.all(sim.get_final_measurement_records()[:, 0])
    assert np.all(sim._correlated_error_occurred)

    sim.clear()
    assert not np.any(sim._correlated_error_occurred)
    sim.run()
    assert np.all(sim.get_final_measurement_records()[:, 0])


@pytest.mark.parametrize("sync_rng", [False, True])
@pytest.mark.parametrize(
    "prep_ops,reset_op,meas_op,expected_meas",
    [
        ("X 0", "R 0", "M 0", [0]),
        ("X 0", "MR 0", "M 0", [1, 0]),
        ("H 0\nZ 0", "RX 0", "MX 0", [0]),
        ("H 0\nZ 0", "MRX 0", "MX 0", [1, 0]),
        ("H 0\nS 0\nZ 0", "RY 0", "MY 0", [0]),
        ("H 0\nS 0\nZ 0", "MRY 0", "MY 0", [1, 0]),
    ],
)
def test_coset_resets_when_m_ref_is_one(
    sync_rng: bool,
    prep_ops: str,
    reset_op: str,
    meas_op: str,
    expected_meas: list[int],
):
    """Regression test for N1 & N6: CosetsideSimulator resets and measure-resets when m_ref == 1."""
    circuit = stim.Circuit(f"{prep_ops}\n{reset_op}\n{meas_op}")
    batch_size = 16
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=batch_size
    )
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch_size,
        seed=123,
        sync_tableside_rng=sync_rng,
    )
    sim.run()
    records = sim.get_final_measurement_records()
    assert records.shape == (batch_size, len(expected_meas))
    for idx, exp_val in enumerate(expected_meas):
        assert np.all(records[:, idx] == bool(exp_val))


def test_coset_resets_when_m_ref_is_one_with_active_mrqf():
    """Verify resets when m_ref == 1 and shots have active MRQF states (d >= 1)."""
    circuit = stim.Circuit(
        """
        I[LEAKAGE_TRANSITION_1: (0.5, 0-->2)] 0
        H 1
        CZ 0 1
        X 0
        MR 0
        M 0
        """
    )
    batch_size = 64
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=batch_size
    )
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch_size,
        seed=42,
    )
    sim.run()
    records = sim.get_final_measurement_records()
    assert np.all(records[:, 1] == False)


def test_conditional_pauli_noise_duplicate_targets():
    """Regression test for Bug 4: conditional X_ERROR(1.0) 0 0 cancels out on unleaked shots."""
    circuit = stim.Circuit(
        """
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 1
        X_ERROR[CONDITIONED_ON_OTHER: U : 1 1](1.0) 0 0
        M 0
        """
    )
    handler = LeakageUint8Coset().compile_op_handler(
        circuit=circuit, batch_size=64
    )
    sim = CosetsideSimulator(
        circuit, compiled_op_handler=handler, batch_size=64, seed=99
    )
    sim.run()
    recs = sim.get_final_measurement_records()[:, 0]
    assert np.all(recs == False), (
        "X_ERROR(1.0) 0 0 should apply X^2 = I on unleaked shots"
    )


def test_coset_in_place_circuit_mutation_re_parses_tags():
    c = stim.Circuit("I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0 1")
    h1 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=4)
    s1 = CosetsideSimulator(c, compiled_op_handler=h1, batch_size=4, seed=5)
    s1.run()
    assert np.all(h1.state[0] == 2) and np.all(h1.state[1] == 0)

    # Stim fuses the two identical tagged I instructions, keeping len(c) == 2.
    c.pop()
    c.append("I", [1], tag="LEAKAGE_TRANSITION_1: (1.0, U-->2)")
    c.append("M", [0, 1])
    assert len(c) == 2
    h2 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=4)
    s2 = CosetsideSimulator(c, compiled_op_handler=h2, batch_size=4, seed=5)
    s2.run()
    assert np.all(h2.state[0] == 2) and np.all(h2.state[1] == 2)


def test_vectorized_cpp_cocycle_arrays_match_dense_and_per_group_oracle():
    from stimside._coset.reference_compile import (
        CompiledReferenceCircuit,
        UnitaryStepMeta,
        get_inverse_clifford_mrqf_factors,
    )
    from stimside.util.bitwords import pauli_multiply
    from stimside.util.reference_chp import pauli_image

    i2 = np.eye(2, dtype=np.complex128)
    px = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    py = 1j * (px @ pz)
    pauli_mats = {(0, 0): i2, (1, 0): px, (0, 1): pz, (1, 1): py}

    def _local_pauli(lx: tuple[int, ...], lz: tuple[int, ...], sgn: int) -> np.ndarray:
        m = pauli_mats[(lx[0], lz[0])]
        for bx, bz in zip(lx[1:], lz[1:]):
            m = np.kron(m, pauli_mats[(bx, bz)])
        return sgn * m

    for g_name, gd in stim.gate_data().items():
        if not gd.is_unitary or not (gd.is_single_qubit_gate or gd.is_two_qubit_gate) or g_name in ("SPP", "SPP_DAG"):
            continue
        k = 2 if gd.is_two_qubit_gate else 1
        dim = 1 << k
        u_mat = stim.Tableau.from_named_gate(g_name).to_unitary_matrix(endian="big")
        e_mat = np.eye(dim, dtype=np.complex128)
        for fac in get_inverse_clifford_mrqf_factors(g_name):
            if fac[0] == "PCP":
                _, lx1, lz1, s1, lx2, lz2, s2 = fac
                p1, p2 = _local_pauli(lx1, lz1, s1), _local_pauli(lx2, lz2, s2)
                f_mat = (np.eye(dim) + p1 + p2 - p1 @ p2) / 2.0
            else:
                kind, lx, lz, sgn = fac
                p = _local_pauli(lx, lz, sgn)
                f_mat = p if kind == "P1" else (np.eye(dim) + 1j * p) / np.sqrt(2.0)
            e_mat = f_mat @ e_mat
        prod = e_mat @ u_mat
        assert abs(abs(prod[0, 0]) - 1.0) < 1e-6
        np.testing.assert_allclose(prod, prod[0, 0] * np.eye(dim), atol=1e-6)

    gates_1q = ("H", "S", "S_DAG", "SQRT_X", "SQRT_Y_DAG", "H_XY", "H_YZ", "C_XYZ", "C_ZYX")
    gates_2q = ("CZ", "CX", "CY", "XCZ", "YCX", "SWAP", "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_ZZ_DAG", "CXSWAP")
    for n_q in (8, 140):
        c = stim.Circuit()
        c.append("H", list(range(0, n_q, 2)))
        c.append("CX", [q for i in range(0, n_q - 1, 2) for q in (i, i + 1)])
        for g in gates_1q:
            c.append(g, list(range(n_q)))
        for g in gates_2q:
            c.append(g, [q for i in range(0, n_q - 1, 2) for q in (i, i + 1)])
        # Overlapping target groups exercise per-group _group_packed_words gathering.
        c.append("CX", [0, 1, 1, 2, 2, n_q - 1])
        c.append("ISWAP", [0, 1, 1, n_q - 1])
        comp = CompiledReferenceCircuit(c)
        for st in comp.steps:
            if not isinstance(st, UnitaryStepMeta) or st.gate_name in ("I", "II", "X", "Y", "Z"):
                continue
            num_g, _, n_fac, _, f_arrs = st.get_cpp_cocycle_arrays(comp.n_words)
            raw_factors = get_inverse_clifford_mrqf_factors(st.gate_name)
            assert n_fac == len(raw_factors)
            for g_i in range(num_g):
                grp = st.target_groups[g_i]
                pw = st._group_packed_words[g_i] if st._group_packed_words is not None else st._get_packed_words(comp.n_words)

                def _oracle_tuple(lx: tuple[int, ...], lz: tuple[int, ...], sgn: int):
                    terms = []
                    for r_i, (bx, bz) in enumerate(zip(lx, lz)):
                        if bx or bz:
                            p_s = {(1, 0): "X", (0, 1): "Z"}.get((bx, bz), "Y")
                            terms.append(pauli_image(p_s, grp[r_i], *pw))
                    aw, bw, kap = terms[0] if len(terms) == 1 else pauli_multiply(*terms[0], *terms[1])
                    return aw, bw, ((kap + 2) & 3) if sgn == -1 else kap

                for f_i, fac in enumerate(raw_factors):
                    if fac[0] == "PCP":
                        _, lx1, lz1, s1, lx2, lz2, s2 = fac
                        aw1, bw1, k1 = _oracle_tuple(lx1, lz1, s1)
                        aw2, bw2, k2 = _oracle_tuple(lx2, lz2, s2)
                        assert int(f_arrs[0][g_i, f_i]) == 2
                        np.testing.assert_array_equal(f_arrs[3][g_i, f_i], aw1)
                        np.testing.assert_array_equal(f_arrs[4][g_i, f_i], bw1)
                        assert int(f_arrs[5][g_i, f_i]) == k1
                        np.testing.assert_array_equal(f_arrs[8][g_i, f_i], aw2)
                        np.testing.assert_array_equal(f_arrs[9][g_i, f_i], bw2)
                        assert int(f_arrs[10][g_i, f_i]) == k2
                    else:
                        kind, lx, lz, sgn = fac
                        aw, bw, kap = _oracle_tuple(lx, lz, sgn)
                        assert int(f_arrs[0][g_i, f_i]) == (0 if kind == "P1" else 1)
                        np.testing.assert_array_equal(f_arrs[3][g_i, f_i], aw)
                        np.testing.assert_array_equal(f_arrs[4][g_i, f_i], bw)
                        assert int(f_arrs[5][g_i, f_i]) == kap


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("bad_q", [-1, 3, 10_000_000])
def test_peek_out_of_bounds_raises_value_error(sync: bool, bad_q: int):
    c = stim.Circuit("R 0 1 2\nH 0\nM 0 1 2")
    coh = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=4)
    sim = CosetsideSimulator(c, compiled_op_handler=coh, batch_size=4, seed=1, sync_tableside_rng=sync)
    with pytest.raises(ValueError, match="peek target out of range"):
        sim.peek_pauli_batch([bad_q], "Z")
    with pytest.raises(ValueError, match="peek target out of range"):
        sim.peek_z([bad_q])


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("n_q", [1, 70])
def test_unmatched_noisy_reset_z_eigenstate_with_active_frame(sync: bool, n_q: int) -> None:
    q = 0 if n_q == 1 else 66
    c = stim.Circuit(f"H {q}\nCX {q} {n_q - 1}" if n_q > 1 else "H 0")
    batch = 64
    coh = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=batch)
    sim = CosetsideSimulator(c, compiled_op_handler=coh, batch_size=batch, seed=3, sync_tableside_rng=sync)
    sim.run()
    rng = np.random.default_rng(0)
    xmask = np.zeros((sim.num_qubits, batch), dtype=bool)
    xmask[q] = rng.random(batch) < 0.5
    sim.apply_pauli("Z", xmask)
    sim.apply_pauli("X", xmask)
    r1 = np.zeros((sim.num_qubits, batch), dtype=bool)
    r1[q] = rng.random(batch) < 0.5
    r0 = np.zeros_like(r1)
    r0[q] = ~r1[q]
    sim.do_unmatched_noisy_reset_z(r0, r1)
    np.testing.assert_array_equal(sim.peek_pauli_batch([q], "Z")[0], np.where(r1[q], -1, 1))


def test_cpp_vs_python_highrank_splitmix_differential() -> None:
    n_q, batch, layers = 140, 3, 16
    rng = np.random.default_rng(4242)
    ops: list[tuple[str, stim.CircuitInstruction]] = [
        ("bare", stim.CircuitInstruction("H", list(range(0, n_q, 2))))
    ]
    for layer in range(layers):
        perm = [int(x) for x in rng.permutation(n_q)]
        g2 = ("CZ", "CX", "ISWAP", "SQRT_XX", "CY", "XCZ")[layer % 6]
        ops.append(("cond", stim.CircuitInstruction(g2, perm[: 2 * (n_q // 2)])))
        g1 = ("S", "SQRT_X", "H", "C_XYZ")[layer % 4]
        ops.append(("cond", stim.CircuitInstruction(g1, perm[: n_q // 2])))
    ops.append(("bare", stim.CircuitInstruction("CX", [q for i in range(0, n_q - 1, 2) for q in (i, i + 1)])))

    circ = stim.Circuit()
    for _, op in ops:
        circ.append(op)
        circ.append("TICK")
    circ.append("R", [n_q, n_q + 1])
    circ.append("CX", [0, n_q, 2, n_q + 1])
    circ.append("TICK")
    circ.append("M", [n_q, n_q + 1])

    sims = [
        CosetsideSimulator(
            circ,
            compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch),
            batch_size=batch,
            seed=77,
        )
        for _ in range(2)
    ]
    sims[1]._use_cpp = False
    tick = stim.CircuitInstruction("TICK", [])
    max_d = 0
    for kind, op in ops:
        masks = rng.random((len(op.target_groups()), batch)) < 0.5
        for s in sims:
            s.do_conditional_clifford(op, None if kind == "bare" else masks.copy())
            s._do_bare_instruction(tick)
        if sims[0]._cpp_dirty:
            sims[0]._sync_cpp_to_py()
            sims[0]._cpp_dirty = True
        assert sims[0]._active_shots.keys() == sims[1]._active_shots.keys()
        for b, st0 in sims[0]._active_shots.items():
            st1 = sims[1]._active_shots[b]
            max_d = max(max_d, st0.d)
            np.testing.assert_array_equal(st0.h_words, st1.h_words)
            np.testing.assert_array_equal(st0.A_words, st1.A_words)
            np.testing.assert_array_equal(st0.ell, st1.ell)
            np.testing.assert_array_equal(st0.Gamma, st1.Gamma)
            assert list(st0.P) == list(st1.P)

    assert max_d > 64
    qs = list(range(n_q))
    for p in ("X", "Y", "Z"):
        np.testing.assert_array_equal(sims[0].peek_pauli_batch(qs, p), sims[1].peek_pauli_batch(qs, p))

    r0 = np.zeros((sim_nq := sims[0].num_qubits, batch), dtype=bool)
    r1 = np.zeros((sim_nq, batch), dtype=bool)
    r0[0] = [True, False, True]
    r1[2] = [False, True, True]
    for s in sims:
        s.do_unmatched_noisy_reset_z(r0, r1)
        s._do_bare_instruction(stim.CircuitInstruction("R", [n_q, n_q + 1]))
        s._do_bare_instruction(tick)
        s.do_conditional_clifford(stim.CircuitInstruction("CX", [0, n_q, 2, n_q + 1]), None)
        s._do_bare_instruction(tick)
        s._do_bare_instruction(stim.CircuitInstruction("M", [n_q, n_q + 1]))
    np.testing.assert_array_equal(
        np.column_stack(sims[0]._measurement_columns),
        np.column_stack(sims[1]._measurement_columns),
    )


def test_coset_duplicate_target_leakage_projection_z() -> None:
    circ = stim.Circuit(
        "R 0 1\nH 0 1\n"
        "I[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 0 1\n"
        "H 0\n"
        "MR[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 0\n"
    )
    coh = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=128)
    sim = CosetsideSimulator(circ, compiled_op_handler=coh, batch_size=128, seed=19, sync_tableside_rng=False)
    sim.run()
    recs = sim.get_final_measurement_records()
    unleaked_q0 = coh.state[0] < 2
    assert np.any(unleaked_q0) and np.any(~unleaked_q0)
    np.testing.assert_array_equal(recs[unleaked_q0, 0], recs[unleaked_q0, 1])
    assert not np.any(recs[unleaked_q0, 4])

    with pytest.raises(NotImplementedError, match="Duplicate targets"):
        CosetsideSimulator(
            circ,
            compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=8),
            batch_size=8,
            seed=19,
            sync_tableside_rng=True,
        ).run()

    circ_unleaked = stim.Circuit(
        "R 0\nH 0\nM[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 0\n"
        "H 0\nMR[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 0\n"
    )
    sim_sync = CosetsideSimulator(
        circ_unleaked,
        compiled_op_handler=LeakageUint8Coset().compile_op_handler(circuit=circ_unleaked, batch_size=32),
        batch_size=32,
        seed=19,
        sync_tableside_rng=True,
    )
    sim_sync.run()
    recs_sync = sim_sync.get_final_measurement_records()
    np.testing.assert_array_equal(recs_sync[:, 0], recs_sync[:, 1])
    assert not np.any(recs_sync[:, 3])


