"""Tier 5 white-box adversarial regression tests for magicside (M1-M6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sinter
import stim

from stimside import (
    BaseDecoder,
    EvaluateQueries,
    LeakageUint8Magic,
    MagicsideSampler,
    MagicsideSimulator,
    PrepareQueries,
    QueryBudget,
    QueryBudgetExceededError,
)
from stimside.util import magic_compile as mc
from stimside.util import phase_qfe as pq


def _axis_1q(q: int, pauli: str, n_qubits: int, neg: bool = False) -> mc.CompiledPauliAxis:
    nw = pq.n_words_for_qubits(n_qubits)
    zero = np.zeros(nw, dtype=np.uint64)
    bit = pq.unit_bit_words(q, nw)
    a_w = bit if pauli in ("X", "Y") else zero
    b_w = bit if pauli in ("Z", "Y") else zero
    kap = ((1 if pauli == "Y" else 0) + (2 if neg else 0)) & 3
    return mc.CompiledPauliAxis(a_w, b_w, kap)


@pytest.mark.parametrize("budget", [1, 64])
def test_adv_compile_open_block_colliding_and_negated_ccz_axes(budget: int) -> None:
    """Generalized CCZ terms with repeated/opposite commuting axes must match N2 apply_ccz."""
    base = pq.ShotExpansion.zero(2)
    z = np.zeros(1, dtype=np.uint64)
    for q in (0, 1):
        base = pq.apply_quarter_turn(base, pq.unit_bit_words(q, 1), z, 0)
    z0_p, z0_m = _axis_1q(0, "Z", 2, False), _axis_1q(0, "Z", 2, True)
    z1_p, z1_m = _axis_1q(1, "Z", 2, False), _axis_1q(1, "Z", 2, True)
    for axes in (
        (z0_p, z0_p, z1_p),
        (z0_p, z0_m, z1_p),
        (z0_m, z0_m, z1_m),
        (z0_p, z0_p, z0_p),
        (z0_m, z0_m, z0_m),
    ):
        want = pq.apply_ccz(base, [ax.as_tuple() for ax in axes]).to_statevector()
        recs = mc.compile_open_block(
            [mc.MagicTermRecord(mc.MagicPhaseTerm(axes=axes, kind="CCZ"))],
            n_qubits=2,
            budget=budget,
        )
        got_exp = base.copy()
        for rec in recs:
            assert isinstance(rec, mc.CompiledOpenBlockRecord)
            got_exp = pq.apply_compiled_block(got_exp, rec.block)
        assert np.linalg.norm(got_exp.to_statevector() - want) < 1e-12


def test_adv_repeat_blocks_with_inline_and_external_clifford_faults() -> None:
    """Verify REPEAT unrolling with inline and programmatic CliffordFaultChannel step offsets."""
    circ = stim.Circuit(
        "H 0 1\nREPEAT 3 {\n  I[T] 0\n  I[CLIFFORD_FAULT: (0.3, S) (0.2, H)] 0\n"
        "  CX 0 1\n  M 1\n  CZ rec[-1] 0\n}\nI[T_DAG] 1\n"
    )
    unrolled = stim.Circuit()
    for inst in circ.flattened():
        unrolled.append(inst)
    ext = [mc.CliffordFaultChannel(2, (mc.FaultBranch(0.25, "SQRT_X", (1,)),))]
    dp_rep = mc.compile_direct(circ, fault_channels=ext)
    sp_rep = mc.compile_selector(circ, fault_channels=ext)
    dp_flat = mc.compile_direct(unrolled, fault_channels=ext)
    assert len(dp_rep.fault_channels) == 4 and len(sp_rep.fault_channels) == 4
    for f_tr in ((0, 0, 0, 0), (1, 1, 2, 1)):
        rd = mc.run_direct(dp_rep, fault_trace=f_tr, rng=7)
        rf = mc.run_direct(dp_flat, fault_trace=f_tr, rng=7)
        assert np.array_equal(rd.measurements, rf.measurements)
        sv_d = mc.reconstruct_statevector(rd, plan=dp_rep)
        assert np.linalg.norm(sv_d - mc.reconstruct_statevector(rf, plan=dp_flat)) < 1e-13
        c_pin = {i: int(b) for i, b in enumerate(rd.measurements)}
        rs = mc.run_selector(
            sp_rep,
            fault_trace=f_tr,
            requested_outcomes=c_pin,
            requested_quantum_outcomes=rd.quantum_outcomes,
            extract_data=True,
        )
        assert np.linalg.norm(mc.reconstruct_statevector(rs, plan=sp_rep) - sv_d) < 1e-12


def test_adv_pauli_channel_2_exact_branch_order_and_runner_budgets() -> None:
    """Verify exact 15-branch PAULI_CHANNEL_2 indexing and M1 runner max_components guards."""
    probs = tuple(0.005 * (i + 1) for i in range(15))
    circ = stim.Circuit(f"H 0 1\nI[T] 0 1\nPAULI_CHANNEL_2({', '.join(map(str, probs))}) 1 0")
    dp = mc.compile_direct(circ)
    sp_n = mc.compile_selector(circ, gadgetize_pauli_noise=False)
    sp_g = mc.compile_selector(circ, gadgetize_pauli_noise=True)
    stim_order = (
        "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
        "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
    )
    for br_idx, pair in enumerate(stim_order, start=1):
        ref_plan = mc.compile_direct(stim.Circuit(f"H 0 1\nI[T] 0 1\n{pair[0]} 1\n{pair[1]} 0"))
        want = mc.reconstruct_statevector(mc.run_direct(ref_plan), plan=ref_plan)
        for runner, pl in ((mc.run_direct, dp), (mc.run_selector, sp_n), (mc.run_selector, sp_g)):
            kw = {"extract_data": True} if pl is not dp else {}
            got = mc.reconstruct_statevector(runner(pl, pauli_noise_trace=[br_idx], **kw), plan=pl)
            assert np.linalg.norm(got - want) < 1e-13

    circ_b = stim.Circuit("H 0 1 2\nI[T] 0 1 2\nH 0 1 2\nI[T] 0 1 2")
    for c2_b in (1, 64):
        dp_b = mc.compile_direct(circ_b, c2_budget=c2_b)
        sp_b = mc.compile_selector(circ_b, c2_budget=c2_b)
        assert mc.run_direct(dp_b, max_components=8).peak_components == 8
        assert mc.run_selector(sp_b, max_components=8, extract_data=True).peak_components == 8
        with pytest.raises(RuntimeError, match="max_components"):
            mc.run_direct(dp_b, max_components=7)
        with pytest.raises(RuntimeError, match="max_components"):
            mc.run_selector(sp_b, max_components=7)


def test_adv_big_endian_statevector_and_coherent_selector_projection() -> None:
    """Verify big-endian ket conversion and coherent superposition extract_data_expansion."""
    def swap_endian(psi: np.ndarray, n: int) -> np.ndarray:
        return np.transpose(psi.reshape((2,) * n), tuple(range(n - 1, -1, -1))).reshape(1 << n)

    c0 = pq.PhaseQFEComponent.computational_basis_state(3, [1, 0, 0])
    assert np.array_equal(c0.to_statevector(endian="big"), swap_endian(c0.to_statevector(), 3))
    dp = mc.compile_direct(stim.Circuit("H 0 2\nI[T] 0\nCX 0 1\nS 2\nI[CCZ] 0 1 2"))
    rd = mc.run_direct(dp)
    sv_le = rd.expansion.to_statevector(ref_tableau=rd.final_tableau, endian="little")
    sv_be = rd.expansion.to_statevector(ref_tableau=rd.final_tableau, endian="big")
    assert np.linalg.norm(sv_be - swap_endian(sv_le, 3)) < 1e-14

    sp = mc.compile_selector(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.4, S)] 0\nI[T] 0"))
    exp_sup = pq.apply_quarter_turn(
        pq.ShotExpansion.zero(sp.num_total_qubits),
        pq.unit_bit_words(1, pq.n_words_for_qubits(sp.num_total_qubits)),
        np.zeros(pq.n_words_for_qubits(sp.num_total_qubits), dtype=np.uint64),
        0,
    )
    for rec in sp.records:
        assert isinstance(rec, mc.CompiledOpenBlockRecord)
        exp_sup = pq.apply_compiled_block(exp_sup, rec.block)
    dp_f = mc.compile_direct(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.4, S)] 0\nI[T] 0"))
    for b_sel, ph in ((0, 1.0 + 0.0j), (1, 1.0j)):
        t_b, e_b = mc.extract_data_expansion(sp, exp_sup, [b_sel])
        want = (ph / np.sqrt(2.0)) * mc.reconstruct_statevector(mc.run_direct(dp_f, fault_trace=[b_sel]))
        assert np.linalg.norm(mc.reconstruct_statevector(t_b, e_b, plan=sp) - want) < 1e-13


def test_adv_cpp_heap_fallbacks_high_nwords_and_high_dim() -> None:
    """Exercise C++ kernel heap paths for n_words > 16 (N=1030) and d > 256 (d=260)."""
    for n_q, qs in ((1030, [0, 63, 64, 512, 1024, 1029]), (264, list(range(260)))):
        nw = pq.n_words_for_qubits(n_q)
        zw = np.zeros(nw, dtype=np.uint64)
        ec = pq.ShotExpansion.zero(n_q)
        for q in qs:
            uq = pq.unit_bit_words(q, nw)
            ec = pq.apply_quarter_turn(ec, uq, zw, 0)
        assert ec.d == len(qs) and ec.chi == 1
        ec.check_invariants(require_canonical_offset=True)

        a_mix = pq.unit_bit_words(qs[0], nw) ^ pq.unit_bit_words(qs[-1], nw)
        b_mix = pq.unit_bit_words(qs[1], nw) ^ pq.unit_bit_words(qs[-2], nw)
        ec = pq.apply_pauli_t(ec, a_mix, b_mix, 0)
        assert ec.chi == 2 and abs(ec.norm_sq() - 1.0) < 1e-12

        uncomp = pq.apply_pauli_t(ec, a_mix, b_mix, 0, dag=True)
        for q in reversed(qs):
            uq = pq.unit_bit_words(q, nw)
            uncomp = pq.apply_quarter_turn(uncomp, uq, zw, 0, sign=-1)
        assert uncomp.d == 0 and uncomp.chi == 1
        assert not pq.words_any_nonzero(uncomp.h_words[0])
        assert abs(uncomp.weights[0] - 1.0) < 1e-12

        mc_r = pq.measure_expansion(ec, zw, a_mix, 0, rng=np.random.default_rng(5))
        sc = mc_r.post_state
        assert isinstance(sc, pq.ShotExpansion)
        assert abs(mc_r.z0 - 0.5) < 1e-12 and abs(mc_r.z1 - 0.5) < 1e-12
        assert sc.d == len(qs) - 1 and abs(sc.norm_sq() - 1.0) < 1e-12
        sc.check_invariants(require_canonical_offset=True)
        sc.to_components()[0][1].check_invariants(require_canonical_offset=True)


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_adv_m6_tandem_sync_and_sub_1e15_fallback_seam(fault_mode: str) -> None:
    """Verify exact 0.0 log-mass cancellation and <=1e-15 pinned recovery across fallback."""
    magic_blk = "H 0 1 2 3\nI[CCZ] 0 1 2 1 2 3\nI[T] 0 1 2 3\nI[CLIFFORD_FAULT: (0.3, Z)] 0\nH 0 1 2 3"
    budget_fb = QueryBudget(max_terms=2, on_budget_exceeded="fallback")

    for pad_bit in (False, True):
        pad_tok = "1" if pad_bit else "0"
        for fp in (1e-16, 1e-15, 1e-14, 0.3, 1.0 - 1e-14, 1.0 - 1e-15, 1.0 - 1e-16):
            circ = stim.Circuit(f"MPAD({fp:.17g}) {pad_tok}\n{magic_blk}\nMPAD({fp:.17g}) {pad_tok}")
            stored_fp = float(circ[0].gate_args_copy()[0])
            plan = PrepareQueries(circ, fault_mode=fault_mode, budget=budget_fb)

            seen_bits: set[int] = set()
            n_seeds = 16 if fp == 0.3 else 3
            for s in range(n_seeds):
                res = EvaluateQueries(
                    plan,
                    {"faults": {0: s % 2}},
                    queries=("mass", "prefix_probs", "sample_measurements"),
                    conditional=False,
                    rng=900 + s,
                )
                assert res.diagnostics.used_fallback is True
                assert res.mass == 1.0
                assert res.sampled_measurements is not None
                seen_bits.update(int(b) for b in res.sampled_measurements)
            if fp == 0.3:
                assert seen_bits == {0, 1}

            for req_bit in (0, 1):
                want_p = stored_fp if req_bit != int(pad_bit) else (1.0 - stored_fp)
                res_pin = EvaluateQueries(
                    plan,
                    {"faults": {0: 1}, "measurements": {0: req_bit}},
                    queries=("mass", "prefix_probs", "sample_measurements"),
                    conditional=False,
                    rng=42,
                )
                assert res_pin.diagnostics.used_fallback is True
                tol = 1e-14 * want_p if want_p <= 1e-15 else 1e-15
                assert (res_pin.mass or 0.0) > 0.0 and abs((res_pin.mass or 0.0) - want_p) <= tol
                want_pp0 = stored_fp if pad_bit else (1.0 - stored_fp)
                assert res_pin.prefix_probs is not None
                assert abs(res_pin.prefix_probs[0][0] - want_pp0) <= 1e-15

    for ph in (1e-16, 1e-15, 1e-14, 0.3):
        circ_h = stim.Circuit(f"HERALDED_ERASE({ph:.17g}) 0\n{magic_blk}\nM({ph:.17g}) 0")
        stored_ph = float(circ_h[0].gate_args_copy()[0])
        plan_h = PrepareQueries(circ_h, fault_mode=fault_mode, budget=budget_fb)
        res_h = EvaluateQueries(
            plan_h,
            {"faults": {0: 1}, "measurements": {0: 1}},
            queries=("mass", "prefix_probs"),
            conditional=False,
        )
        assert res_h.diagnostics.used_fallback is True
        assert abs((res_h.mass or 0.0) - stored_ph) <= 1e-14 * max(1e-30, stored_ph)
        assert res_h.prefix_probs is not None
        assert abs(res_h.prefix_probs[0][1] - stored_ph) <= 1e-15


def test_adv_max_components_budget_across_stream_query_leakage_and_sampler() -> None:
    """Verify max_components raise vs fallback across all STREAM, QUERY, leakage, and sampler paths."""
    circ = stim.Circuit(
        "H 0 1 2 3\nI[CCZ] 0 1 2 1 2 3\nI[T] 0 1 2 3\nH 0 1 2 3\n"
        "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    with pytest.raises(QueryBudgetExceededError):
        MagicsideSimulator(circ, batch_size=2, max_components=2, on_budget_exceeded="raise").run()

    sim_int = MagicsideSimulator(circ, batch_size=2, max_components=2, on_budget_exceeded="raise")
    with pytest.raises(QueryBudgetExceededError):
        sim_int.interactive_do(circ)
        sim_int.finish_interactive_run()

    for fm, fr in (("SELECTOR", None), ("DIRECT", [{}])):
        with pytest.raises(QueryBudgetExceededError):
            MagicsideSimulator(
                circ,
                batch_size=1,
                fault_mode=fm,
                fault_realizations=fr,
                max_components=2,
                on_budget_exceeded="raise",
            ).run()

    with pytest.raises(QueryBudgetExceededError):
        MagicsideSimulator(
            circ,
            batch_size=1,
            backend="QUERY",
            query_budget=QueryBudget(max_terms=64, on_budget_exceeded="raise"),
            max_components=2,
        ).run()

    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            sim_fb = MagicsideSimulator(
                circ,
                batch_size=4,
                seed=17,
                fault_mode=fm,
                backend=be,
                max_components=2,
                on_budget_exceeded="fallback",
            )
            sim_fb.run()
            assert sim_fb.get_final_measurement_records().shape == (4, 1)

    with pytest.raises(QueryBudgetExceededError):
        MagicsideSimulator(
            circ, op_handler=LeakageUint8Magic(), batch_size=2, max_components=2, on_budget_exceeded="raise"
        ).run()

    task = sinter.Task(circuit=circ, detector_error_model=stim.DetectorErrorModel("error(0.1) D0 L0"))
    for be in ("STREAM", "QUERY"):
        compiled = MagicsideSampler(
            dem_decoder=BaseDecoder(decoder="pymatching"),
            batch_size=4,
            backend=be,
            max_components=2,
            on_budget_exceeded="raise",
        ).compiled_sampler_for_task(task)
        with pytest.raises(QueryBudgetExceededError):
            compiled.sample(4)


def test_adv_leakage_entangled_multitarget_collapse_and_strided_sweep_m2d() -> None:
    """Verify sequential conditional collapse in multi-target leakage ops and strided sweep_bits handling."""
    circ_ent = stim.Circuit(
        "H 0\nI[T] 0\nH 0\nCX 0 1\n"
        "I[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0 1\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0 1] 0 0"
    )
    p_leak = math.sin(math.pi / 8.0) ** 2
    for sync_rng in (False, True):
        handler = LeakageUint8Magic().compile_op_handler(circuit=circ_ent, batch_size=2048)
        sim = MagicsideSimulator(
            circ_ent, compiled_op_handler=handler, batch_size=2048, seed=314, sync_tableside_rng=sync_rng
        )
        sim.run()
        recs = sim.get_final_measurement_records()
        np.testing.assert_array_equal(recs[:, 0], recs[:, 1])
        assert abs(float(np.mean(recs[:, 0])) - p_leak) < 4.5 * math.sqrt(p_leak * (1.0 - p_leak) / 2048)

    circ_sw = stim.Circuit(
        "CX sweep[0] 0 sweep[2] 1 sweep[1] 3\nH 2\nI[CCZ] 0 1 2\nH 2\nM 2 3\n"
        "DETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]"
    )
    logical_sw = np.array([[(s >> 0) & 1, (s >> 1) & 1, (s >> 2) & 1] for s in range(8)], dtype=np.bool_)
    rev_buf = np.ascontiguousarray(logical_sw[::-1, ::-1])
    neg_stride_sw = rev_buf[::-1, ::-1]
    assert neg_stride_sw.strides[0] < 0 and neg_stride_sw.strides[1] < 0
    want_raw = np.column_stack([logical_sw[:, 0] & logical_sw[:, 2], logical_sw[:, 1]])

    for fm in ("DIRECT", "SELECTOR"):
        for be in ("STREAM", "QUERY"):
            for mode in ("auto", "stim", "zero"):
                sim_sw = MagicsideSimulator(
                    circ_sw, batch_size=8, fault_mode=fm, backend=be, sweep_bits=neg_stride_sw, m2d_reference_mode=mode
                )
                sim_sw.run()
                np.testing.assert_array_equal(sim_sw.get_final_measurement_records(), want_raw)
                dets = sim_sw.get_detector_flips()
                if mode == "auto":
                    assert not np.any(dets)
                elif mode == "zero":
                    np.testing.assert_array_equal(dets, want_raw)
                else:
                    np.testing.assert_array_equal(dets, np.column_stack([want_raw[:, 0], np.zeros(8, dtype=np.bool_)]))
