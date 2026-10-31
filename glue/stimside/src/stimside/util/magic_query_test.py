"""Unit and verification tests (`V1.5`) for `stimside.util.magic_query` (Milestone M3)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.util.magic_compile import OMEGA, CliffordFaultChannel, FaultBranch, reconstruct_statevector, run_direct
from stimside.util.magic_query import EvaluateQueries, PreparedQueryPlan, PrepareQueries, QueryBudget, QueryBudgetExceededError, QueryDiagnostics, QueryResult, evaluate_queries, prepare_queries, run_query_batch, run_query_shot
from stimside.util.phase_qfe import ShotExpansion, apply_pauli_t
from stimside.util.reference_chp import ReferenceCHPTableau


def _dense_unitary_psi(circ: stim.Circuit) -> np.ndarray:
    plan = PrepareQueries(circ)
    res = run_direct(plan.compiled_plan, rng=0)  # type: ignore[arg-type]
    return reconstruct_statevector(res, plan=plan.compiled_plan)


def _partial_trace_overlap_dense(psi_unnorm: np.ndarray, n_qubits: int, out_qubits: list[int], target_ket: np.ndarray) -> float:
    k = len(out_qubits)
    traced = [q for q in range(n_qubits) if q not in out_qubits]
    perm = [n_qubits - 1 - q for q in out_qubits] + [n_qubits - 1 - q for q in traced]
    mat = np.transpose(psi_unnorm.reshape((2,) * n_qubits), perm).reshape(1 << k, 1 << len(traced))
    if k > 0:
        mat = np.transpose(mat.reshape((2,) * k + (1 << len(traced),)), [k - 1 - j for j in range(k)] + [k]).reshape(1 << k, 1 << len(traced))
    proj = (target_ket / np.linalg.norm(target_ket)).conj() @ mat
    return float(np.vdot(proj, proj).real)


def test_exports_aliases_and_validation_errors() -> None:
    """Verify public aliases, exception hierarchy, and strict input validation."""
    assert prepare_queries is PrepareQueries and evaluate_queries is EvaluateQueries
    assert issubclass(QueryBudgetExceededError, RuntimeError)
    for bad_kw in ({"max_terms": 0}, {"max_terms": True}, {"max_dim": -1}, {"max_contraction_ops": 0}, {"max_cache_bytes": -1}, {"on_budget_exceeded": "invalid"}):
        with pytest.raises(ValueError):
            QueryBudget(**bad_kw)  # type: ignore[arg-type]
    assert QueryBudget(on_budget_exceeded="FALLBACK_STREAM").on_budget_exceeded == "fallback"

    c = stim.Circuit("H 0 1\nI[T] 0\nM 0")
    fc = [CliffordFaultChannel(step_index=1, branches=(FaultBranch(0.2, "S", (0,)),))]
    with pytest.raises(TypeError):
        PrepareQueries("H 0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_cpp_kernels"):
        PrepareQueries(c, use_cpp_kernels=False)  # type: ignore[call-arg]
    for bad_prep in ({"fault_mode": "BOGUS"}, {"atol": 0.0}, {"output_qubits": [0, 0]}, {"fault_mode": "SELECTOR", "fault_channels": fc, "output_qubits": [-1]}, {"fault_mode": "SELECTOR", "fault_channels": fc, "output_qubits": [2]}, {"target_spec": stim.Circuit("H 0\nM 0"), "output_qubits": [0]}):
        with pytest.raises(ValueError):
            PrepareQueries(c, **bad_prep)  # type: ignore[arg-type]
    plan = PrepareQueries(c, fault_channels=fc)
    for bad_eval in ({"queries": "mass"}, {"queries": ("unknown_metric",)}, {"queries": ("fidelity",)}, {"realization": {"bogus": {}}}, {"realization": {"faults": {99: 0}}}, {"realization": {"faults": {0: 5}}}, {"realization": {"measurements": {5: 0}}}, {"acceptance_spec": {5: 0}}):
        with pytest.raises(ValueError):
            EvaluateQueries(plan, **bad_eval)  # type: ignore[arg-type]

    masses = [EvaluateQueries(PrepareQueries(c, fault_mode=fm, fault_channels=fc), {"measurements": {0: 0}}).mass for fm in ("direct", "DIRECT", "Direct", "selector", "SELECTOR", "Selector")]
    assert all(m is not None and abs(m - masses[0]) < 1e-13 for m in masses)  # type: ignore[operator]


@pytest.mark.parametrize("conditional", [False, True])
def test_empty_zero_meas_and_impossible_branch_edges(conditional: bool) -> None:
    """Verify empty circuit, 0-measurement unitary, and impossible zero-mass postselection."""
    for txt in ("", "H 0\nI[T] 0\nCX 0 1"):
        plan = PrepareQueries(stim.Circuit(txt))
        res = EvaluateQueries(plan, queries=("mass", "acceptance", "prefix_probs", "sample_measurements"), conditional=conditional, rng=7)
        assert isinstance(plan, PreparedQueryPlan) and isinstance(res, QueryResult) and isinstance(res.diagnostics, QueryDiagnostics)
        assert res.mass is not None and abs(res.mass - 1.0) < 1e-14 and res.acceptance_mass is not None and abs(res.acceptance_mass - 1.0) < 1e-14
        assert res.prefix_probs == [] and res.sampled_measurements is not None and res.sampled_measurements.shape == (0,)

    p_imp = PrepareQueries(stim.Circuit("I[T] 0\nM 0"), target_spec=np.array([1.0 + 0.0j, 0.0j]), output_qubits=[0], acceptance_spec={0: 1})
    r_imp = EvaluateQueries(p_imp, {"measurements": {0: 1}}, queries=("mass", "acceptance", "fidelity"), conditional=conditional)
    assert r_imp.mass == 0.0 and r_imp.acceptance_mass == 0.0 and r_imp.fidelity_mass == 0.0


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_choi_open_slot_noise_resets_and_born_partition(fault_mode: str) -> None:
    """Verify open fault non-cancellation, DIRECT==SELECTOR, Pauli/herald/reset/MPAD, and sum(Z_h)==1."""
    c_open = stim.Circuit("I[T] 0\nI[CLIFFORD_FAULT: (0.5, H)] 0\nI[T_DAG] 0\nM 0")
    p_open = PrepareQueries(c_open, fault_mode=fault_mode)
    assert abs((EvaluateQueries(p_open, {"faults": {0: 0}, "measurements": {0: 0}}).mass or 0.0) - 1.0) < 1e-12
    for m_b in (0, 1):
        assert abs((EvaluateQueries(p_open, {"faults": {0: 1}, "measurements": {0: m_b}}).mass or 0.0) - 0.5) < 1e-12

    c_mix = stim.Circuit("H 0 1 2\nI[T] 0 1\nDEPOLARIZE1(0.15) 0\nI[CCZ] 0 1 2\nHERALDED_ERASE(0.2) 1\nMR 0\nR 2\nMPAD(0.25) 1\nH 1 2\nM(0.1) !1")
    fc = [CliffordFaultChannel(step_index=2, branches=(FaultBranch(0.2, "S", (0,)), FaultBranch(0.2, "CZ", (1, 2))))]
    plan = PrepareQueries(c_mix, fault_mode=fault_mode, fault_channels=fc)
    plan_dir = PrepareQueries(c_mix, fault_mode="DIRECT", fault_channels=fc)
    for f_b in (0, 1, 2):
        tot = 0.0
        for bits in itertools.product((0, 1), repeat=4):
            real = {"faults": {0: f_b}, "pauli_noise": {0: "X"}, "measurements": {j: bits[j] for j in range(4)}}
            r_mode, r_ref = EvaluateQueries(plan, real, queries=("mass", "prefix_probs")), EvaluateQueries(plan_dir, real, queries=("mass", "prefix_probs"))
            assert r_mode.mass is not None and r_ref.mass is not None and abs(r_mode.mass - r_ref.mass) < 1e-12 and np.allclose(r_mode.prefix_probs, r_ref.prefix_probs, atol=1e-12)  # type: ignore[arg-type]
            tot += r_mode.mass
        assert abs(tot - 1.0) < 1e-11


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_segment_tree_cache_hits_and_partial_invalidation(fault_mode: str) -> None:
    """Verify cold -> hit -> partial invalidation diagnostics and exact fallback parity."""
    circ = stim.Circuit("H 0 1 2 3\nI[CCZ] 0 1 2\nI[T] 0 1 2 3\nCX 0 3 1 2\nI[CCZ] 1 2 3\nI[T_DAG] 0 2\nCX 2 0 3 1\nI[CCZ] 0 2 3\nI[T] 1 3\nH 0 1 2 3\nM 0 1")
    fcs = [CliffordFaultChannel(step_index=st, branches=(FaultBranch(0.2, g, tg),), slot_id=sid) for sid, st, g, tg in ((0, 2, "S", (0,)), (1, 4, "H", (1,)), (2, 6, "CZ", (2, 3)), (3, 8, "S_DAG", (3,)))]
    plan = PrepareQueries(circ, fault_mode=fault_mode, fault_channels=fcs)
    plan_fb = PrepareQueries(circ, fault_mode=fault_mode, fault_channels=fcs, budget=QueryBudget(max_terms=1, on_budget_exceeded="fallback"))
    r0 = {"faults": {0: 0, 1: 0, 2: 0, 3: 0}, "measurements": {0: 0, 1: 1}}
    c0, ref0 = EvaluateQueries(plan, r0, queries=("mass",)), EvaluateQueries(plan_fb, r0, queries=("mass",))
    tot_nodes = c0.diagnostics.node_evaluations
    assert tot_nodes >= 2 and c0.diagnostics.invalidated_nodes == 0 and ref0.diagnostics.used_fallback and abs((c0.mass or 0.0) - (ref0.mass or 0.0)) < 1e-13
    c1 = EvaluateQueries(plan, r0, queries=("mass",))
    assert c1.diagnostics.cache_hits >= 1 and c1.diagnostics.node_evaluations == 0 and c1.diagnostics.invalidated_nodes == 0 and abs((c1.mass or 0.0) - (c0.mass or 0.0)) < 1e-15
    r_right = {"faults": {0: 0, 1: 0, 2: 0, 3: 1}, "measurements": {0: 0, 1: 1}}
    c2, ref2 = EvaluateQueries(plan, r_right, queries=("mass",)), EvaluateQueries(plan_fb, r_right, queries=("mass",))
    assert c2.diagnostics.cache_hits >= 1 and 0 < c2.diagnostics.invalidated_nodes < tot_nodes and 0 < c2.diagnostics.node_evaluations < tot_nodes
    assert abs((c2.mass or 0.0) - (ref2.mass or 0.0)) < 1e-13

    r_left = {"faults": {0: 1, 1: 0, 2: 0, 3: 1}, "measurements": {0: 0, 1: 1}}
    c3 = EvaluateQueries(plan, r_left, queries=("mass",))
    assert c3.diagnostics.cache_hits >= 1 and 0 < c3.diagnostics.invalidated_nodes < tot_nodes


def test_causal_prefix_probs_feedback_and_sampling() -> None:
    """Verify causal prefix_probs invariance under future acceptance_spec, rec/sweep feedback, and sampling."""
    circ = stim.Circuit("H 0\nI[T] 0\nH 0\nM 0\nCZ sweep[0] 0\nCX rec[-1] 1\nH 1 2\nI[CCZ] 0 1 2\nCY rec[-1] 2\nH 1 2\nM 1 2")
    p_none, p_dict, p_fn = PrepareQueries(circ, acceptance_spec=None), PrepareQueries(circ, acceptance_spec={2: 1}), PrepareQueries(circ, acceptance_spec=lambda m: bool(m[2]))

    exact_probs = np.zeros(8, dtype=np.float64)
    for idx in range(8):
        bits = {j: (idx >> j) & 1 for j in range(3)}
        real = {"measurements": bits, "sweep_bits": [1]}
        rn = EvaluateQueries(p_none, real, queries=("mass", "prefix_probs"))
        rd = EvaluateQueries(p_dict, real, queries=("acceptance", "prefix_probs"), conditional=False)
        rf = EvaluateQueries(p_fn, real, queries=("acceptance", "prefix_probs"), conditional=False)
        assert rn.mass is not None and rn.prefix_probs is not None and rd.prefix_probs is not None and rf.prefix_probs is not None
        assert all(abs(rn.prefix_probs[j][0] - r.prefix_probs[j][0]) < 1e-15 for j in (0, 1) for r in (rd, rf))
        assert abs(math.prod(rn.prefix_probs[j][bits[j]] for j in range(3)) - rn.mass) < 1e-12
        assert abs((rd.acceptance_mass or 0.0) - (rn.mass if bits[2] == 1 else 0.0)) < 1e-13
        exact_probs[idx] = rn.mass

    assert abs(float(np.sum(exact_probs)) - 1.0) < 1e-12
    rng, counts = np.random.default_rng(12345), np.zeros(8, dtype=np.int64)
    for _ in range(2048):
        s_res = EvaluateQueries(p_none, {"sweep_bits": [1]}, queries=("sample_measurements",), rng=rng)
        assert s_res.sampled_measurements is not None and s_res.sampled_measurements.dtype == np.bool_
        sm = s_res.sampled_measurements
        counts[int(sm[0]) | (int(sm[1]) << 1) | (int(sm[2]) << 2)] += 1
    assert 0.5 * float(np.sum(np.abs(counts / 2048.0 - exact_probs))) < 0.04


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_partial_trace_fidelity_all_formats_multiword_and_mc_ratio(fault_mode: str) -> None:
    """Verify all 3 target_spec formats vs dense partial trace, N=66 & N=130 multi-word, and MC ratio."""
    circ = stim.Circuit("H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2\nCX 0 2\nH 1 2\nM 2")
    fc = [CliffordFaultChannel(step_index=2, branches=(FaultBranch(0.25, "S", (0,)),))]
    tgt_circ, ref_t = stim.Circuit("H 0 1\nCZ 0 1\nI[T] 0"), ReferenceCHPTableau(2)
    tgt_ket = _dense_unitary_psi(tgt_circ)
    ref_t.do_unitary("H", [0, 1])
    ref_t.do_unitary("CZ", [0, 1])
    exp_t = apply_pauli_t(ShotExpansion.zero(2), ([1, 0], [0, 1], 0), sign=1)
    assert isinstance(exp_t, ShotExpansion)

    psi_m0 = _dense_unitary_psi(stim.Circuit("H 0 1 2\nI[T] 0 1\nI[CCZ] 0 1 2\nCX 0 2\nH 1 2")).copy()
    psi_m0[4:] = 0.0
    fid_dense_m0 = _partial_trace_overlap_dense(psi_m0, 3, [0, 1], tgt_ket)
    for t_spec in (tgt_circ, tgt_ket, (ref_t, exp_t)):
        plan = PrepareQueries(circ, fault_mode=fault_mode, fault_channels=fc, target_spec=t_spec, output_qubits=[0, 1], acceptance_spec={0: 0})
        r_u = EvaluateQueries(plan, {"faults": {0: 0}, "measurements": {0: 0}}, queries=("mass", "acceptance", "fidelity"), conditional=False)
        assert r_u.fidelity_mass is not None and abs(r_u.fidelity_mass - fid_dense_m0) < 1e-12

    psi_4q, t_2q = _dense_unitary_psi(stim.Circuit("H 0 1 2 3\nI[T] 0\nI[CCZ] 0 1 2\nCX 2 3")), _dense_unitary_psi(stim.Circuit("H 0 1\nI[T] 0"))
    fid_ref_4q = _partial_trace_overlap_dense(psi_4q, 4, [0, 1], t_2q)
    for _, qs in ((65, [0, 63, 64, 62]), (130, [0, 64, 129, 128])):
        q0, q1, q2, q3 = qs
        p_big_u = PrepareQueries(stim.Circuit(f"H {q0} {q1} {q2} {q3}\nI[T] {q0}\nI[CCZ] {q0} {q1} {q2}\nCX {q2} {q3}"), fault_mode=fault_mode, target_spec=t_2q, output_qubits=[q0, q1])
        ru_big = EvaluateQueries(p_big_u, queries=("fidelity",), conditional=False)
        assert ru_big.fidelity_mass is not None and abs(ru_big.fidelity_mass - fid_ref_4q) < 1e-12
        c_big = stim.Circuit(f"H {q0} {q1} {q2} {q3}\nI[T] {q0}\nI[CCZ] {q0} {q1} {q2}\nCX {q2} {q3}\nI[CLIFFORD_FAULT: (0.2, Z)] {q1}\nMPP X{q0}*Z{q1}*X{q2}")
        p_mpp = PrepareQueries(c_big, fault_mode=fault_mode, target_spec=_dense_unitary_psi(stim.Circuit("H 0 1 2\nI[T] 0\nI[CCZ] 0 1 2")), output_qubits=[q0, q1, q2])
        m0, m1 = EvaluateQueries(p_mpp, {"faults": {0: 0}, "measurements": {0: 0}}, queries=("mass",), conditional=False), EvaluateQueries(p_mpp, {"faults": {0: 0}, "measurements": {0: 1}}, queries=("mass",), conditional=False)
        assert abs((m0.mass or 0.0) + (m1.mass or 0.0) - 1.0) < 1e-12

    c_mc = stim.Circuit("H 0 1\nI[T] 0 1\nZ_ERROR(0.15) 0\nCX 0 1\nH 1\nM 1")
    t_mc = np.array([1.0, OMEGA * 1j], dtype=np.complex128) / math.sqrt(2.0)
    p_mc = PrepareQueries(c_mc, target_spec=t_mc, output_qubits=[0], acceptance_spec={0: 0})
    r_ex = EvaluateQueries(p_mc, None, queries=("acceptance", "fidelity"), conditional=False)
    f_ex = (r_ex.fidelity_mass or 0.0) / (r_ex.acceptance_mass or 1.0)
    rng_mc = np.random.default_rng(99)
    ab = [EvaluateQueries(p_mc, None, queries=("acceptance", "fidelity", "sample_measurements"), conditional=True, rng=rng_mc) for _ in range(1500)]
    assert abs((sum(r.fidelity_mass or 0.0 for r in ab) / sum(r.acceptance_mass or 0.0 for r in ab)) - f_ex) < 0.035


def test_budget_raise_fallback_and_m4_batch_hooks() -> None:
    """Verify QueryBudget('raise' vs 'fallback') and run_query_shot / run_query_batch parity."""
    circ = stim.Circuit("H 0 1 2 3\nI[T] 0 1 2 3\nI[CLIFFORD_FAULT: (0.2, H)] 0\nI[CLIFFORD_FAULT: (0.2, H)] 1\nCX 0 1 1 2 2 3\nI[CCZ] 0 1 2\nH 0 1 2 3\nM 0 1")
    tgt, real = np.array([1.0 + 0.0j, 0.0j]), {"faults": {0: 1, 1: 1}, "measurements": {0: 0, 1: 1}}

    for b_raise in (QueryBudget(max_terms=2, on_budget_exceeded="raise"), QueryBudget(max_dim=1, on_budget_exceeded="raise"), QueryBudget(max_contraction_ops=1, on_budget_exceeded="raise")):
        with pytest.raises(QueryBudgetExceededError):
            EvaluateQueries(PrepareQueries(circ, budget=b_raise), real, queries=("mass",))

    p_ref = PrepareQueries(circ, target_spec=tgt, output_qubits=[2], acceptance_spec={0: 0, 1: 1})
    r_ref = EvaluateQueries(p_ref, real, queries=("mass", "acceptance", "fidelity", "prefix_probs"), conditional=False)
    p_fb = PrepareQueries(circ, target_spec=tgt, output_qubits=[2], acceptance_spec={0: 0, 1: 1}, budget=QueryBudget(max_terms=2, on_budget_exceeded="fallback"))
    r_fb = EvaluateQueries(p_fb, real, queries=("mass", "acceptance", "fidelity", "prefix_probs"), conditional=False)
    assert r_fb.diagnostics.used_fallback is True and abs((r_fb.mass or 0.0) - (r_ref.mass or 0.0)) < 1e-12 and abs((r_fb.fidelity_mass or 0.0) - (r_ref.fidelity_mass or 0.0)) < 1e-12
    assert np.allclose(r_fb.prefix_probs, r_ref.prefix_probs, atol=1e-12)  # type: ignore[arg-type]

    p_sel, p_dir = PrepareQueries(circ, fault_mode="SELECTOR"), PrepareQueries(circ, fault_mode="DIRECT")
    t_sel, t_dir = run_query_shot(p_sel, realization=real, extract_data=True), run_query_shot(p_dir, realization=real)
    psi_sel, psi_dir = reconstruct_statevector(t_sel.final_tableau, t_sel.expansion, plan=p_sel.compiled_plan), reconstruct_statevector(t_dir.final_tableau, t_dir.expansion, plan=p_dir.compiled_plan)
    assert abs(t_sel.branch_mass - t_dir.branch_mass) < 1e-12 and abs(abs(complex(np.vdot(psi_sel, psi_dir))) - 1.0) < 1e-12

    recs, trajs = run_query_batch(p_sel, 8, rng=42, extract_data=True)
    assert recs.shape == (8, 2) and len(trajs) == 8


def test_runtime_atol_enforcement_and_no_hidden_stream_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify runtime atol drift raises FloatingPointError and non-fallback paths never call run_direct/run_selector."""
    import stimside.util.magic_compile as mc
    import stimside.util.magic_query as mq

    orig_pd, orig_acb = mq.project_diagonal, mq.apply_compiled_block
    for bad_z in (0.1, float("nan")):
        monkeypatch.setattr(mq, "project_diagonal", lambda t, b, s=0, o=0, _z=bad_z, **kw: (orig_pd(t, b, s, o, **kw)[0], _z))
        with pytest.raises(FloatingPointError):
            EvaluateQueries(PrepareQueries(stim.Circuit("H 0\nM 0"), atol=1e-10), {"measurements": {0: 0}})
    monkeypatch.setattr(mq, "project_diagonal", orig_pd)
    monkeypatch.setattr(mq, "apply_compiled_block", lambda e, b, **kw: (lambda o: (setattr(o, "weights", o.weights * 0.5), o)[1])(orig_acb(e, b, **kw)))
    with pytest.raises(FloatingPointError):
        EvaluateQueries(PrepareQueries(stim.Circuit("H 0\nI[T] 0\nI[CLIFFORD_FAULT: (0.2, H)] 0\nI[T] 0\nM 0"), atol=1e-10), {"faults": {0: 1}, "measurements": {0: 0}})
    monkeypatch.setattr(mq, "apply_compiled_block", orig_acb)
    assert not hasattr(mq, "run_direct") and not hasattr(mq, "run_selector")
    for fn in ("run_direct", "run_selector"):
        monkeypatch.setattr(mc, fn, lambda *a, **kw: (_ for _ in ()).throw(AssertionError("FORBIDDEN")))
    c100 = stim.Circuit("H 0 50 99\nI[T] 0 50 99\nCX 0 50 50 99")
    r100 = EvaluateQueries(PrepareQueries(c100, target_spec=c100), queries=("fidelity",), conditional=False)
    assert r100.fidelity_mass is not None and abs(r100.fidelity_mass - 1.0) < 1e-12 and not r100.diagnostics.used_fallback


def test_initial_dynamic_vacuum_guard_and_deep_underflow_scaling() -> None:
    """Verify op_off == 0 ket preservation and >= 60-measurement frexp trajectory scaling."""
    r_my = EvaluateQueries(PrepareQueries(stim.Circuit("MY 2\nI[T_DAG] 1")), {}, queries=("mass", "prefix_probs"))
    assert abs((r_my.mass or 0.0) - 1.0) < 1e-12 and r_my.prefix_probs is not None and abs(r_my.prefix_probs[0][0] - 0.5) < 1e-12
    p_xe, p_hm = PrepareQueries(stim.Circuit("X_ERROR(1.0) 0\nI[T] 0\nM 0")), PrepareQueries(stim.Circuit("H 0\nM 0\nH 0\nI[T] 0\nH 0\nM 0"))
    assert abs((EvaluateQueries(p_xe, {"measurements": {0: 1}}).mass or 0.0) - 1.0) < 1e-12 and bool(np.all(run_query_batch(p_xe, 4, rng=0)[0]))
    assert abs(sum((EvaluateQueries(p_hm, {"measurements": {0: a, 1: b}}).mass or 0.0) for a in (0, 1) for b in (0, 1)) - 1.0) < 1e-12
    p_deep = PrepareQueries(stim.Circuit("REPEAT 60 {\nH 0\nM 0\n}\nR 0\nX 0\nM 0"), output_qubits=[0], acceptance_spec={60: 1}, target_spec=stim.Circuit("X 0"))
    r_deep = EvaluateQueries(p_deep, None, queries=("acceptance", "fidelity", "prefix_probs", "sample_measurements"), conditional=True, rng=42)
    assert r_deep.accepted is True and abs((r_deep.acceptance_mass or 0.0) - 1.0) < 1e-12 and abs((r_deep.fidelity_mass or 0.0) - 1.0) < 1e-12
    assert r_deep.prefix_probs is not None and np.allclose(r_deep.prefix_probs[55], (0.5, 0.5)) and np.allclose(r_deep.prefix_probs[60], (0.0, 1.0))
    assert run_query_shot(PrepareQueries(stim.Circuit("I[T] 0\nREPEAT 60 {\nH 0\nMR 0\n}")), rng=7).expansion.chi >= 1


def test_mixed_fallback_sampling_unbiasedness_and_noise_semantics() -> None:
    """Verify mixed Kraus fallback, unbiased mid-circuit sampling fallback, ELSE_CORRELATED_ERROR, and R/M(p) mass parity."""
    fb = QueryBudget(max_terms=1, on_budget_exceeded="fallback")
    for c_txt, rl in (("H 0 1 2\nI[T] 0 1 2\nZ_ERROR(0.3) 0\nI[CCZ] 0 1 2\nH 0 1 2\nM 0", None), ("H 0\nI[T] 0\nR 0\nI[T] 0\nH 0\nM 0", {"measurements": {0: 0}}), ("H 0 1 2\nI[T] 0 1 2\nI[CCZ] 0 1 2\nM(0.25) 0\nM 0", {"measurements": {0: 1, 1: 0}}), ("H 1 2 3\nI[CCZ] 1 2 3\nI[T] 0\nM 0\nM 1", {"measurements": {0: 1, 1: 0}})):
        c = stim.Circuit(c_txt)
        r_ex = EvaluateQueries(PrepareQueries(c, acceptance_spec={0: 0}), rl, queries=("mass", "acceptance", "prefix_probs"), conditional=False)
        r_fb = EvaluateQueries(PrepareQueries(c, acceptance_spec={0: 0}, budget=fb), rl, queries=("mass", "acceptance", "prefix_probs"), conditional=False)
        assert r_fb.diagnostics.used_fallback and abs((r_fb.mass or 0.0) - (r_ex.mass or 0.0)) < 1e-12 and np.allclose(r_fb.prefix_probs, r_ex.prefix_probs, atol=1e-12)  # type: ignore[arg-type]
    gad = "".join(f"H {q}\nI[T] {q}\nH {q}\nI[T] {q}\nCX rec[-1] {q}\nI[T_DAG] {q}\nH {q}\nI[T_DAG] {q}\n" for q in range(1, 7))
    recs, _ = run_query_batch(PrepareQueries(stim.Circuit("H 0\nM 0\n" + gad + "M 1"), budget=QueryBudget(max_terms=32, on_budget_exceeded="fallback")), 400, rng=2026)
    assert abs(float(np.mean(recs[:, 0] == 0)) - 0.5) < 0.08
    assert run_query_shot(PrepareQueries(stim.Circuit("H 2 3 4\nI[CCZ] 2 3 4\nX_ERROR(1.0) 0\nX_ERROR(1.0) 1\nM 0 1"), budget=fb), realization={"paulis": {0: 1}}, rng=7).measurements.tolist() == [True, True]
    c_corr = stim.Circuit("H 0\nI[T] 0\nCORRELATED_ERROR(0.2) X0\nELSE_CORRELATED_ERROR(0.3) Z0\nH 0\nM 0")
    assert abs((EvaluateQueries(PrepareQueries(c_corr), {"paulis": {0: 1, 1: 1}, "measurements": {0: 0}}).mass or 0.0) - 0.8535533905932737) < 1e-12
    for c_txt in ("H 0\nI[T] 0\nR 0\nH 0\nM 0", "H 0\nI[T] 0\nH 0\nM(0.2) 0"):
        p_s = PrepareQueries(stim.Circuit(c_txt))
        assert abs(run_query_shot(p_s, rng=np.random.default_rng(123)).branch_mass - run_direct(p_s.compiled_plan, rng=np.random.default_rng(123)).branch_mass) < 1e-12  # type: ignore[arg-type]


def test_mixed_acceptance_fidelity_budget_and_invalidation_diagnostics() -> None:
    """Verify is_mixed under rng/MR(p), fidelity QueryBudget enforcement, bool fault/sweep validation, and invalidated_nodes."""
    assert EvaluateQueries(PrepareQueries(stim.Circuit("H 0\nM 0"), acceptance_spec={0: 0}), {"faults": {}}, queries=("acceptance",), rng=42).accepted is True
    assert EvaluateQueries(PrepareQueries(stim.Circuit("MR(0.2) 0"), acceptance_spec={0: 0}), {"resets": {0: 0}}, queries=("acceptance",)).accepted is True
    p_1b = PrepareQueries(stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.2, Z)] 0\nH 0\nM 0"))
    assert abs((EvaluateQueries(p_1b, {"faults": {0: True}, "measurements": {0: 1}}).mass or 0.0) - 1.0) < 1e-12 and abs((EvaluateQueries(p_1b, {"faults": [True], "measurements": {0: 1}}).mass or 0.0) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        EvaluateQueries(p_1b, {"sweep": {-1: 1}})
    c_fid = stim.Circuit("H 0 1 2 3\nI[CCZ] 0 1 2 1 2 3\nI[T] 0 1 2 3\nH 0 1 2 3")
    with pytest.raises(QueryBudgetExceededError):
        EvaluateQueries(PrepareQueries(c_fid, target_spec=c_fid, budget=QueryBudget(max_terms=16, on_budget_exceeded="raise")), queries=("fidelity",))
    p_d1 = PrepareQueries(stim.Circuit("I[CLIFFORD_FAULT: (0.2, H)] 0\nI[T] 0\nI[CLIFFORD_FAULT: (0.2, H)] 1\nI[T] 1\nM 0"), budget=QueryBudget(max_terms=2, on_budget_exceeded="raise"))
    EvaluateQueries(p_d1, {"faults": {0: 0, 1: 0}, "measurements": {0: 0}})
    with pytest.raises(QueryBudgetExceededError):
        EvaluateQueries(p_d1, {"faults": {0: 1, 1: 1}, "measurements": {0: 0}})
    assert EvaluateQueries(p_d1, {"faults": {0: 0, 1: 0}, "measurements": {0: 0}}).diagnostics.invalidated_nodes == 0
    p_det = PrepareQueries(stim.Circuit("I[T] 0\nM 0\nCX rec[-1] 1\nH 1\nI[T] 1\nH 1\nM 1"))
    run_query_shot(p_det, rng=1)
    assert run_query_shot(p_det, rng=2).diagnostics.invalidated_nodes == 0
    p_mrg = PrepareQueries(stim.Circuit("H 0 1\nI[T] 0\nI[CLIFFORD_FAULT: (0.2, S)] 0\nCX 0 1\nI[T] 1\nM 0 1"))
    EvaluateQueries(p_mrg, {"faults": {0: 0}}, queries=("mass",))
    assert EvaluateQueries(p_mrg, {"faults": {0: 1}}, queries=("mass",)).diagnostics.invalidated_nodes > 0


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_custom_slot_fallback_zero_mass_prefix_and_lru_invalidation(fault_mode: str) -> None:
    """Verify custom slot_id fallback, zero-prob reset fallback, dead-end prefix retention, conditional zero-mass prefix_probs, and A->B->A->A invalidation."""
    fb = QueryBudget(max_terms=2, on_budget_exceeded="fallback")
    c_cs = stim.Circuit("H 0 1 2\nI[T] 0 1 2\nI[CCZ] 0 1 2\nH 0 1 2\nM 0 1")
    fc_cs = [CliffordFaultChannel(step_index=1, branches=(FaultBranch(0.5, "Z", (0,)),), slot_id=42), CliffordFaultChannel(step_index=2, branches=(FaultBranch(0.4, "X", (1,)),), slot_id=7)]
    r_tr = EvaluateQueries(PrepareQueries(c_cs, fault_mode=fault_mode, fault_channels=fc_cs), {"faults": {42: 1, 7: 1}, "measurements": {0: 0, 1: 1}}, queries=("mass", "prefix_probs"))
    r_fb = EvaluateQueries(PrepareQueries(c_cs, fault_mode=fault_mode, fault_channels=fc_cs, budget=fb), {"faults": {42: 1, 7: 1}, "measurements": {0: 0, 1: 1}}, queries=("mass", "prefix_probs"))
    assert r_fb.diagnostics.used_fallback and abs((r_fb.mass or 0.0) - (r_tr.mass or 0.0)) < 1e-12 and np.allclose(r_fb.prefix_probs, r_tr.prefix_probs, atol=1e-12)  # type: ignore[arg-type]
    assert abs(run_query_shot(PrepareQueries(c_cs, fault_mode=fault_mode, fault_channels=fc_cs, budget=fb), rng=np.random.default_rng(11)).branch_mass - run_query_shot(PrepareQueries(c_cs, fault_mode=fault_mode, fault_channels=fc_cs), rng=np.random.default_rng(11)).branch_mass) < 1e-12
    for c_rz, rl_rz in (("H 1 2 3\nI[CCZ] 1 2 3\nR 0\nM 0", {"resets": {0: 1}, "measurements": {0: 0}}), ("H 0 1 2 3\nI[CCZ] 1 2 3\nRX 0\nMX 0", {"resets": {0: 1}, "measurements": {0: 0}}), ("H 1 2 3\nI[CCZ] 1 2 3\nMR 0", {"resets": {0: 1}, "measurements": {0: 0}}), ("H 1 2 3\nI[CCZ] 1 2 3\nMR(0.2) 0", {"resets": {0: 1}, "measurements": {0: 0}})):
        rz, rz_tr = EvaluateQueries(PrepareQueries(stim.Circuit(c_rz), fault_mode=fault_mode, budget=fb), rl_rz, queries=("mass", "acceptance", "prefix_probs"), conditional=False), EvaluateQueries(PrepareQueries(stim.Circuit(c_rz), fault_mode=fault_mode), rl_rz, queries=("mass", "acceptance", "prefix_probs"), conditional=False)
        assert rz.diagnostics.used_fallback and rz.mass == 0.0 and rz_tr.mass == 0.0 and rz.acceptance_mass == 0.0 and np.allclose(rz.prefix_probs, rz_tr.prefix_probs, atol=1e-12)  # type: ignore[arg-type]
        assert run_query_shot(PrepareQueries(stim.Circuit(c_rz), fault_mode=fault_mode, budget=fb), realization=rl_rz, rng=5).branch_mass == 0.0
    for c_z, m_z in (("X 0\nM 0\nH 1 2 3\nI[CCZ] 1 2 3\nM 4", {1: 1}), ("X_ERROR(1.0) 0\nM 0\nH 1 2 3\nI[CCZ] 1 2 3\nM 4", {1: 1}), ("HERALDED_ERASE(1.0) 0\nH 1 2 3\nI[CCZ] 1 2 3\nM 4", {1: 1}), ("H 0\nM(0.2) 0\nCX rec[-1] 0\nM 0\nH 1 2 3\nI[CCZ] 1 2 3\nM 4", {2: 1})):
        rt_z = EvaluateQueries(PrepareQueries(stim.Circuit(c_z), fault_mode=fault_mode, acceptance_spec={0: 1}), {"measurements": m_z}, queries=("mass", "acceptance", "prefix_probs", "sample_measurements"), rng=np.random.default_rng(19))
        rf_z = EvaluateQueries(PrepareQueries(stim.Circuit(c_z), fault_mode=fault_mode, acceptance_spec={0: 1}, budget=fb), {"measurements": m_z}, queries=("mass", "acceptance", "prefix_probs", "sample_measurements"), rng=np.random.default_rng(19))
        assert rt_z.mass == 0.0 and rf_z.mass == 0.0 and rf_z.diagnostics.used_fallback and rt_z.sampled_measurements is not None and rf_z.sampled_measurements is not None
        assert np.array_equal(rt_z.sampled_measurements, rf_z.sampled_measurements) and rt_z.accepted == bool(rt_z.sampled_measurements[0]) and rf_z.accepted == rt_z.accepted and np.allclose(rf_z.prefix_probs, rt_z.prefix_probs, atol=1e-12)  # type: ignore[arg-type]
    assert EvaluateQueries(PrepareQueries(stim.Circuit("HERALDED_ERASE(0.5) 0\nM 1"), fault_mode=fault_mode, acceptance_spec={0: 1}), {"paulis": {0: 1}, "measurements": {1: 1}}, queries=("acceptance",)).accepted is True
    assert EvaluateQueries(PrepareQueries(stim.Circuit("M 1\nX_ERROR(0.25) 1\nM 0"), fault_mode=fault_mode, acceptance_spec={0: 0}), {"measurements": {0: 0, 1: 1}}, queries=("acceptance",)).accepted is True
    assert run_query_shot(PrepareQueries(stim.Circuit("H 1 2 3\nI[CCZ] 1 2 3\nHERALDED_ERASE(1.0) 0\nX_ERROR(0.2) 0"), fault_mode=fault_mode, budget=fb), realization={"measurements": {0: 0}, "paulis": {1: 1}}, rng=3, extract_data=True).expansion.chi == 0
    p_lru = PrepareQueries(stim.Circuit("H 0 1\nI[CLIFFORD_FAULT: (0.2, Z)] 0\nI[T] 0\nCX 0 1\nI[CLIFFORD_FAULT: (0.2, I)] 1\nI[T] 1\nM 0"), fault_mode=fault_mode)
    inv = [EvaluateQueries(p_lru, {"faults": {0: b}, "measurements": {0: 0}}).diagnostics.invalidated_nodes for b in (0, 1, 0, 0)]
    assert inv[0] == 0 and inv[1] > 0 and inv[2] == inv[1] and inv[3] == 0


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_deep_log_mass_fallback_and_sweep_mapping_normalization(fault_mode: str) -> None:
    """Verify Challenge 1 (M=1050..2500 log-space extra_cl_log/m_log & -inf/NaN guards) and Challenge 2 (_norm_sweep & batch)."""
    fb = QueryBudget(max_terms=2, on_budget_exceeded="fallback")
    s1050 = run_query_shot(PrepareQueries(stim.Circuit("REPEAT 1050 {\nMPAD(0.5) 0\n}\nH 0 1 2\nI[CCZ] 0 1 2\nM 0"), fault_mode=fault_mode, budget=fb), rng=np.random.default_rng(7), extract_data=True)
    assert s1050.diagnostics.used_fallback and abs(s1050.branch_mass - 0.5) < 1e-12
    for m_cnt in (1050, 1200, 2500):
        p_m = PrepareQueries(stim.Circuit(f"REPEAT {m_cnt} {{\nMPAD(0.5) 0\n}}\nH 0 1 2\nI[CCZ] 0 1 2"), fault_mode=fault_mode, budget=fb)
        assert abs(run_query_shot(p_m, rng=7).branch_mass - 1.0) < 1e-12 and abs((EvaluateQueries(p_m, None, queries=("mass", "sample_measurements"), rng=7).mass or 0.0) - 1.0) < 1e-12
    c_t4 = stim.Circuit("REPEAT 400 {\nH 0\nCX 0 1\nI[T] 0\nH 0 1\nM(0.3) 0 1\nR 0 1\nH 0\nI[T] 0\nH 0\nM(0.3) 0\nR 0\n}\nH 0 1 2\nI[CCZ] 0 1 2")
    r_t4 = EvaluateQueries(PrepareQueries(c_t4, fault_mode=fault_mode, budget=fb), None, queries=("acceptance", "sample_measurements"), conditional=True, rng=np.random.default_rng(414))
    r_he = EvaluateQueries(PrepareQueries(stim.Circuit("REPEAT 1200 {\nHERALDED_ERASE(0.5) 0\n}\nH 0 1 2\nI[CCZ] 0 1 2"), fault_mode=fault_mode, budget=fb), {"paulis": {0: 0}, "measurements": {j: 0 for j in range(1, 1200)}}, queries=("acceptance", "sample_measurements"), conditional=True, rng=5)
    assert r_t4.diagnostics.used_fallback and r_t4.acceptance_mass == 1.0 and r_he.diagnostics.used_fallback and r_he.acceptance_mass == 1.0
    assert abs(run_query_shot(PrepareQueries(stim.Circuit("MPAD(1e-17) 1\nM(0.3) 0\nH 0 1 2\nI[CCZ] 0 1 2"), fault_mode=fault_mode, budget=fb), realization={"measurements": {1: 1}}, rng=9).branch_mass - 0.3) < 1e-12
    for c_imp, m_imp in (("MPAD(0.0) 0\nH 0 1 2\nI[CCZ] 0 1 2", {0: 1}), ("REPEAT 1050 {\nMPAD(0.5) 0\n}\nH 0 1 2\nI[CCZ] 0 1 2\nM(0.0) 3", {1050: 1})):
        p_i = PrepareQueries(stim.Circuit(c_imp), fault_mode=fault_mode, budget=fb)
        assert EvaluateQueries(p_i, {"measurements": m_imp}, queries=("mass", "sample_measurements"), rng=3).mass == 0.0 and run_query_shot(p_i, realization={"measurements": m_imp}, rng=3).branch_mass == 0.0
    p_sw = PrepareQueries(stim.Circuit("CX sweep[0] 0\nCX sweep[2] 1\nI[CLIFFORD_FAULT: (0.0, X)] 1\nM 0 1"), fault_mode=fault_mode)
    assert [run_query_shot(p_sw, sweep_bits=sb, rng=0).measurements.tolist() for sb in ({0: 1}, {1: 0}, {2: 1}, {0: 1, 2: 1})] == [[True, False], [False, False], [False, True], [True, True]]
    assert run_query_shot(PrepareQueries(stim.Circuit("CX sweep[0] 0\nCX sweep[2] 1\nH 2 3 4\nI[CCZ] 2 3 4\nM 0 1"), fault_mode=fault_mode, budget=fb), realization={"sweep_bits": {0: 1, 10**9: 1}}, sweep_bits={2: 1, 10**12: 1}, rng=0).measurements.tolist() == [False, True]
    assert [run_query_batch(p_sw, 2, rng=0, **kw)[0].tolist() for kw in ({"fault_realizations": {"faults": {0: 1}}, "sweep_bits": {0: 1}}, {"sweep_bits": [1, 0, 1]}, {"sweep_bits": [{0: 1}, {2: 1}]})] == [[[True, True], [True, True]], [[True, True], [True, True]], [[True, False], [False, True]]]
    for fn, bad_sw in itertools.product((lambda s: run_query_shot(p_sw, sweep_bits=s, rng=0), lambda s: EvaluateQueries(p_sw, {"sweep_bits": s})), ({0: 999}, {-1: 1}, {True: 1}, "01", b"\x00\x01", bytearray([0, 1]), memoryview(b"\x00"), {0, 1}, np.array(1), np.zeros((2, 2), dtype=np.bool_))):
        with pytest.raises(ValueError):
            fn(bad_sw)
    for bad_b in ({"sweep_bits": "ab"}, {"sweep_bits": [{0: 1}]}, {"sweep_bits": np.zeros((3, 3), dtype=np.bool_)}, {"fault_realizations": [None]}, {"fault_realizations": [None, {"faults": {99: 1}}]}, {"rng": True}):
        with pytest.raises(ValueError):
            run_query_batch(p_sw, 2, **bad_b)  # type: ignore[arg-type]


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_fallback_zero_mass_unbiasedness_pinning_and_sub15_exactness(fault_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unbiased zero-mass fallback MC, Markov stopping-time independence, sub-1e-15 exactness, D3/D4, and 1x exit."""
    import stimside.util.magic_query as mq
    fb1, fb2, hdr, n_mc = QueryBudget(max_terms=1, on_budget_exceeded="fallback"), QueryBudget(max_terms=2, on_budget_exceeded="fallback"), "H 2 3 4\nI[CCZ] 2 3 4\n", 400
    cases = ((f"{hdr}M 0\nX_ERROR(0.3) 1\nM 1", fb1, {"measurements": {1: 1}}, 0.30, True), (f"{hdr}M 0\nI[CLIFFORD_FAULT: (0.3, X)] 1\nM 1", fb1, {"measurements": {1: 1}}, 0.30, False),
             ("H 0 1 2\nI[CCZ] 0 1 2\nHERALDED_PAULI_CHANNEL_1(0.0, 0.1, 0.2, 0.3) 3\nM 3", fb1, {"measurements": {0: 1, 1: 0}}, 0.30, True),
             (f"Z_ERROR(0.1) 2\n{hdr}MPAD(0.5) 0\nX_ERROR(0.3) 1\nM 1", fb2, {"measurements": {1: 1}}, 0.30, True),
             (f"{hdr}H 0\nR 0\nX_ERROR(0.3) 1\nR 1", fb1, {"resets": {1: 1}}, 0.15, True), (f"{hdr}H 0\nM 0\nCX rec[-1] 1\nH 5\nM 5\nCX rec[-1] 1\nR 1", fb1, {"resets": {0: 1}}, 0.125, True))
    for c_str, b_cfg, rl, ex_m, has_eq in cases:
        p = PrepareQueries(stim.Circuit(c_str), fault_mode=fault_mode, budget=b_cfg)
        shs, (_, b_shs) = [run_query_shot(p, realization=rl, rng=10000 + s) for s in range(n_mc)], run_query_batch(p, n_mc, fault_realizations=rl, rng=np.random.default_rng(777))
        assert all(t.diagnostics.used_fallback for t in shs) and abs(float(np.mean([t.branch_mass for t in shs])) - ex_m) < 0.05 and abs(float(np.mean([t.branch_mass for t in b_shs])) - ex_m) < 0.05
        for s in (range(50) if has_eq else ()):
            r_no = EvaluateQueries(p, rl, queries=("mass", "sample_measurements"), rng=10000 + s)
            r_pp = EvaluateQueries(p, rl, queries=("mass", "prefix_probs", "sample_measurements"), rng=10000 + s)
            assert r_no.diagnostics.used_fallback and r_pp.diagnostics.used_fallback and r_no.mass == shs[s].branch_mass == r_pp.mass
            assert np.array_equal(r_no.sampled_measurements, shs[s].measurements) and np.array_equal(r_pp.sampled_measurements, shs[s].measurements)
    p_ce0, p_prb, g_mk = PrepareQueries(stim.Circuit(cases[0][0]), fault_mode=fault_mode, budget=fb1), PrepareQueries(stim.Circuit("X_ERROR(0.3) 0\nM 0"), fault_mode=fault_mode), np.random.default_rng(2026)
    prb = [int(run_query_shot(p_prb, rng=g_mk).measurements[0]) for _ in range(600) if run_query_shot(p_ce0, realization={"measurements": {1: 1}}, rng=g_mk).branch_mass == 0.0]
    assert len(prb) > 350 and abs(float(np.mean(prb)) - 0.30) < 0.055
    p_s15 = PrepareQueries(stim.Circuit(f"{hdr}X_ERROR(0.35) 1\nM(1e-17) 1\nH 5\nM 5"), fault_mode=fault_mode, budget=fb1)
    s15 = [EvaluateQueries(p_s15, {"measurements": {0: 1}}, queries=("mass", "sample_measurements"), rng=30000 + s) for s in range(n_mc)]
    s15_c = [r for r in s15 if 0.0 < (r.mass or 0.0) < 1e-10]
    assert abs(float(np.mean([r.mass or 0.0 for r in s15])) - 0.175) < 0.05 and len(s15_c) > 200 and all(abs((r.mass or 0.0) - 0.5e-17) < 1e-29 for r in s15_c)
    assert abs(float(np.mean([int(r.sampled_measurements[1]) for r in s15_c if r.sampled_measurements is not None])) - 0.50) < 0.07
    tgt3 = stim.Circuit("H 0 1 2\nI[CCZ] 0 1 2")
    for c_u, m_u, ex_u in ((f"{hdr}M(1e-17) 0", {0: 1}, 1e-17), (f"{hdr}M(1e-300) 0", {0: 1}, 1e-300), (f"{hdr}MPAD(1e-17) 1", {0: 0}, 1e-17),
                           (f"{hdr}HERALDED_ERASE(1e-17) 0", {0: 1}, 1e-17), (f"{hdr}HERALDED_ERASE({1.0 - 1e-16!r}) 0", {0: 0}, 1.0 - (1.0 - 1e-16)),
                           (f"{hdr}HERALDED_PAULI_CHANNEL_1(0.0, 1e-17, 2e-17, 3e-17) 0", {0: 1}, 6e-17)):
        pu = PrepareQueries(stim.Circuit(c_u), fault_mode=fault_mode, budget=fb1, acceptance_spec=m_u, output_qubits=[2, 3, 4], target_spec=tgt3)
        ru, su = EvaluateQueries(pu, {"measurements": m_u}, queries=("mass", "acceptance", "fidelity", "sample_measurements"), conditional=True, rng=5), run_query_shot(pu, realization={"measurements": m_u}, rng=5, extract_data=True)
        assert ru.diagnostics.used_fallback and math.isclose(ru.mass or 0.0, ex_u, rel_tol=1e-12, abs_tol=0.0) and ru.acceptance_mass == 1.0 and abs((ru.fidelity_mass or 0.0) - 1.0) < 1e-11
        assert math.isclose(su.branch_mass, ex_u, rel_tol=1e-12, abs_tol=0.0) and su.expansion.chi > 0 and abs(su.expansion.norm_sq() - 1.0) < 1e-12
    p_d3, orig_ev, calls = PrepareQueries(stim.Circuit(f"{hdr}X_ERROR(0.3) 0\nM 0\nM !5\nMPAD 1\nHERALDED_ERASE(1.0) 1\nH 1\nM 1\nM 0"), fault_mode=fault_mode, budget=fb1), mq._eval_via_tree, []
    monkeypatch.setattr(mq, "_eval_via_tree", lambda *a, **kw: calls.append(bool(kw.get("stream_only", False))) or orig_ev(*a, **kw))
    for sd in range(30):
        calls.clear()
        rn, sd_sh = EvaluateQueries(p_d3, {"measurements": {0: 1}}, queries=("mass", "sample_measurements"), rng=sd), run_query_shot(p_d3, realization={"measurements": {0: 1}}, rng=sd)
        rp = EvaluateQueries(p_d3, {"measurements": {0: 1}}, queries=("mass", "prefix_probs", "sample_measurements"), rng=sd)
        assert np.array_equal(rn.sampled_measurements, rp.sampled_measurements) and np.array_equal(sd_sh.measurements, rp.sampled_measurements)
        assert rn.mass != 0.0 or (calls == [False, False, False, True] and sd_sh.expansion.chi == 0 and sd_sh.measurements.tolist() == [True, False, False, False, False, False])


@pytest.mark.parametrize("fault_mode", ["DIRECT", "SELECTOR"])
def test_correlated_chain_interleaving_and_mpad_ulp_fallback_parity(fault_mode: str) -> None:
    """Verify interleaved/orphan ELSE_CORRELATED_ERROR chains and MPAD sub-ULP fallback parity."""
    fb1, hdr, zr = QueryBudget(max_terms=1, on_budget_exceeded="fallback"), "H 3 4 5\nI[CCZ] 3 4 5\n", (lambda: type("_Z", (np.random.Generator,), {"random": lambda s, *a, **k: 0.0})(np.random.PCG64(0)))
    c_a = stim.Circuit(f"{hdr}X_ERROR(1) 0\nELSE_CORRELATED_ERROR(1) X1\nZ_ERROR(1) 0\nELSE_CORRELATED_ERROR(1) X1\nCORRELATED_ERROR(0) X0\nX_ERROR(1) 0\nHERALDED_ERASE(1) 6\nELSE_CORRELATED_ERROR(1) X2\nM 0 1 2")
    c_p = stim.Circuit(f"{hdr}CORRELATED_ERROR(0.4) X0\nX_ERROR(0.7) 1\nHERALDED_ERASE(0.6) 2\nELSE_CORRELATED_ERROR(0.8) X2\nM 0 1 2")
    for b_cfg in (QueryBudget(), fb1):
        pt, pp = PrepareQueries(c_a, fault_mode=fault_mode, budget=b_cfg), PrepareQueries(c_p, fault_mode=fault_mode, budget=b_cfg)
        r_n, s_0 = EvaluateQueries(pt, queries=("mass", "sample_measurements"), rng=0), run_query_shot(pt, rng=0)
        r_p, s_p = EvaluateQueries(pt, queries=("mass", "prefix_probs", "sample_measurements"), rng=0), run_query_shot(pt, realization={"paulis": {0: 1}}, rng=0)
        assert [m.tolist() for m in (s_0.measurements, s_p.measurements, r_n.sampled_measurements, r_p.sampled_measurements) if m is not None] == [[True, False, True, True]] * 4
        assert math.isclose(EvaluateQueries(pp, {"paulis": {0: 0, 1: 1, 2: 2}, "measurements": {0: 1, 1: 0, 2: 1, 3: 0}}).mass or 0.0, 0.8, rel_tol=1e-12)
        assert EvaluateQueries(pp, {"paulis": {0: 1, 1: 0, 2: 2, 3: 1}, "measurements": {0: 1, 1: 1, 2: 0, 3: 0}}).mass == 0.0
    for p in (5e-324, 1e-323, 1.03e-15, 9.5 * 2**-53, 1.06e-15):
        p_b, ex_b = PrepareQueries(stim.Circuit(f"MPAD({p!r}) 1\n{hdr}MPAD({p!r}) 1\nX 0\nH 1\nM 0 1"), fault_mode=fault_mode, budget=fb1, acceptance_spec={1: 0}), 0.5 * (p if 1.0 - (1.0 - p) <= 1e-15 else 1.0 - (1.0 - p))
        u_e, u_s = EvaluateQueries(p_b, None, queries=("mass", "sample_measurements"), rng=zr()), run_query_shot(p_b, rng=zr())
        assert u_s.diagnostics.used_fallback and math.isclose(u_s.branch_mass, 0.5, rel_tol=1e-12) and u_e.mass == u_s.branch_mass and u_s.measurements[:3].tolist() == [False, False, True]
        if p > 5e-324:
            b_n, b_s = EvaluateQueries(p_b, {"measurements": {1: 0}}, queries=("mass", "acceptance", "sample_measurements"), rng=0), run_query_shot(p_b, realization={"measurements": {1: 0}}, rng=0)
            assert b_n.diagnostics.used_fallback and b_n.acceptance_mass == 1.0 and b_s.expansion.chi > 0 and math.isclose(b_n.mass or 0.0, ex_b, rel_tol=1e-12, abs_tol=0.0)
            assert b_n.mass == EvaluateQueries(p_b, {"measurements": {1: 0}}, queries=("mass", "acceptance", "prefix_probs", "sample_measurements"), rng=0).mass == b_s.branch_mass
