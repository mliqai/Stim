"""Section 10 `Q1`-`Q2` Global Scalar-Network Query Engine (`magic_query`, Milestone M3)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic import compile as _mc
from stimside._magic.compile import CliffordFaultChannel, CompiledCliffordFactorList, CompiledOpenBlockRecord, CompiledPauliAxis, CorrelatedNoiseRecord, DirectFaultSlotRecord, DirectPlan, FeedbackPauliRecord, HeraldedNoiseRecord, LeakageBarrierRecord, MagicTermRecord, MeasurementRecord, MPadRecord, PauliNoiseRecord, ResetRecord, SelectorPlan, TrajectoryRunResult, _coerce_strict_int, _compile_pauli_measurement_c15, _normalize_choice_map, _sample_categorical_branch, clone_reference_chp, compile_direct, compile_open_block, compile_selector, extract_data_expansion
from stimside._magic.phase_qfe import DEFAULT_ATOL, K1_CANCEL_REL_TOL, CompiledPauliBlock, CompiledPauliBlockStep, ImpossiblePostselectionError, PhaseQFEComponent, ShotExpansion, _check_unitary_norm, apply_compiled_block, apply_pauli, apply_quarter_turn, eliminate_zero_amplitude, make_principal, merge_frames, project_diagonal, restrict_amplitude, stim_tableau_to_reference_chp, validate_atol
from stimside.util.bitwords import (
    OMEGA_POWERS,
    SQRT2 as _SQRT2,
    first_set_bit_in_words,
    n_words_for_qubits,
    tail_mask,
    unit_bit_words,
)
from stimside.util.reference_chp import ReferenceCHPTableau

__all__ = ["QueryBudgetExceededError", "QueryBudget", "QueryDiagnostics", "QueryResult", "PreparedQueryPlan", "PrepareQueries", "EvaluateQueries", "prepare_queries", "evaluate_queries", "QueryShotTrajectory", "run_query_shot", "run_query_batch"]

_EMPTY_DEPS = ((), (), ())
_PAULI_1Q_STR, _PAULI_4_STR = {"I": 0, "X": 1, "Y": 2, "Z": 3}, {"I": 0, "X": 2, "Y": 3, "Z": 4}
_PAULI_2Q_STR = {a + b: i for i, (a, b) in enumerate((p, q) for p in "IXYZ" for q in "IXYZ")}
_VALID_QUERIES = frozenset({"mass", "acceptance", "fidelity", "prefix_probs", "sample_measurements"})
_VALID_REAL_KEYS = frozenset({"faults", "measurements", "quantum_measurements", "paulis", "pauli_noise", "resets", "sweep", "sweep_bits"})

class QueryBudgetExceededError(RuntimeError):
    """Raised when `Q1`/`Q2` contraction exceeds `QueryBudget` with `on_budget_exceeded == 'raise'`."""

@dataclass(frozen=True)
class QueryBudget:
    """Resource budget governing `Q1`-`Q2` segment-tree contraction and caching."""
    max_terms: int = 4096
    max_dim: int | None = None
    max_contraction_ops: int = 1_000_000
    max_cache_bytes: int = 256 * 1024 * 1024
    on_budget_exceeded: str = "raise"

    def __post_init__(self) -> None:
        bounds = (("max_terms", 1), ("max_contraction_ops", 1), ("max_cache_bytes", 0)) + ((("max_dim", 0),) if self.max_dim is not None else ())
        for fn, lo in bounds:
            iv = _coerce_strict_int(getattr(self, fn), fn)
            if iv < lo:
                raise ValueError(f"{fn} must be >= {lo}, got {iv}")
            object.__setattr__(self, fn, iv)
        mode = self.on_budget_exceeded.strip().lower() if isinstance(self.on_budget_exceeded, str) else ""
        if mode not in ("raise", "error", "fallback", "fallback_stream"):
            raise ValueError(f"on_budget_exceeded must be 'raise' or 'fallback', got {self.on_budget_exceeded!r}")
        object.__setattr__(self, "on_budget_exceeded", "raise" if mode in ("raise", "error") else "fallback")

@dataclass
class QueryDiagnostics:
    """Telemetry and cache statistics emitted by `EvaluateQueries`."""
    cache_hits: int = 0
    cache_misses: int = 0
    node_evaluations: int = 0
    invalidated_nodes: int = 0
    peak_terms: int = 0
    peak_dim: int = 0
    contraction_ops: int = 0
    used_fallback: bool = False

@dataclass
class QueryResult:
    """Output bundle returned by `EvaluateQueries`."""
    mass: float | None = None
    acceptance_mass: float | None = None
    fidelity_mass: float | None = None
    prefix_probs: list[tuple[float, float]] | None = None
    sampled_measurements: NDArray[np.bool_] | None = None
    accepted: bool | None = None
    diagnostics: QueryDiagnostics = field(default_factory=QueryDiagnostics)

@dataclass
class QueryShotTrajectory:
    """Single-shot trajectory output for `MagicsideSimulator(..., backend='QUERY')`."""
    measurements: NDArray[np.bool_]
    branch_mass: float
    expansion: ShotExpansion
    final_tableau: ReferenceCHPTableau
    selector_bits: NDArray[np.bool_] | None
    diagnostics: QueryDiagnostics

@dataclass
class PreparedQueryPlan:
    """Compiled `Q1`-`Q2` query plan holding the hybrid Choi/Ket segment tree and LRU cache."""
    circuit: stim.Circuit
    fault_mode: str
    logical_num_qubits: int
    num_data_qubits: int
    num_total_qubits: int
    num_measurements: int
    output_qubits: tuple[int, ...]
    budget: QueryBudget
    atol: float
    compiled_plan: DirectPlan | SelectorPlan
    target_spec: Any
    acceptance_spec: Any
    _state: dict[str, Any] = field(default_factory=dict, repr=False)

_FP_ZERO, _ln = (0.0, 0), (lambda x: math.log(x) if x > 0.0 else -math.inf)
def _ldexp(raw: float, exp2: int) -> float:
    if not math.isfinite(raw) or (raw > 0.0 and math.frexp(raw)[1] + exp2 > 1024):
        raise FloatingPointError(f"Non-finite or overflowing scalar norm raw={raw!r}, exp2={exp2}")
    return 0.0 if (raw <= 0.0 or math.frexp(raw)[1] + exp2 < -1074) else float(math.ldexp(*math.frexp(raw)[:1], math.frexp(raw)[1] + exp2))
def _fp(x: float, e: int = 0) -> tuple[float, int]:
    if not math.isfinite(x):
        raise FloatingPointError(f"Non-finite branch weight {x!r}")
    return _FP_ZERO if x <= 0.0 else (math.frexp(x)[0], e + math.frexp(x)[1])
def _fp_mul(a: tuple[float, int], x: float, e: int = 0) -> tuple[float, int]:
    return _FP_ZERO if (a[0] <= 0.0 or x <= 0.0) else _fp(a[0] * x, a[1] + e)
def _fp_add(a: tuple[float, int], b: tuple[float, int]) -> tuple[float, int]:
    hi, lo = (a, b) if a[1] >= b[1] else (b, a)
    return (b if a[0] <= 0.0 else a) if (a[0] <= 0.0 or b[0] <= 0.0) else (hi if (hi[1] - lo[1] > 60) else _fp(hi[0] + math.ldexp(lo[0], lo[1] - hi[1]), hi[1]))
def _fp_div(num: tuple[float, int], den: tuple[float, int]) -> float:
    return 0.0 if (num[0] <= 0.0 or den[0] <= 0.0) else _ldexp(num[0] / den[0], num[1] - den[1])

class _ScaledExpansion:
    def __init__(self, exp: ShotExpansion, log2_scale: int = 0) -> None:
        self.exp, self.log2_scale = exp, log2_scale
    def copy(self) -> _ScaledExpansion:
        return _ScaledExpansion(self.exp.copy(), self.log2_scale)
    def norm_sq(self) -> float:
        return _ldexp(self.exp.norm_sq() if self.exp.chi > 0 else 0.0, 2 * self.log2_scale)

class _SegTreeNode:
    def __init__(self, nid: int, l: int, r: int, left: _SegTreeNode | None = None, right: _SegTreeNode | None = None, leaf: Any = None, deps: Any = _EMPTY_DEPS) -> None:
        self.node_id, self.l, self.r, self.left, self.right, self.leaf, self.deps, self.last_key = nid, l, r, left, right, leaf, deps, None

class _EvalCtx:
    def __init__(self, budget: QueryBudget, atol: float, m_act: int, faults: dict[int, int], meas: dict[int, int], sweep: dict[int, int], *, enforce_budget: bool = True, prev_diag: QueryDiagnostics | None = None) -> None:
        self.budget, self.atol, self.m_act, self.enforce_budget = budget, atol, m_act, enforce_budget
        self.diagnostics = QueryDiagnostics(prev_diag.cache_hits, prev_diag.cache_misses, prev_diag.node_evaluations, prev_diag.invalidated_nodes, prev_diag.peak_terms, prev_diag.peak_dim, prev_diag.contraction_ops, True) if prev_diag is not None else QueryDiagnostics()
        self.faults, self.meas, self.sweep, self.visited = faults, meas, sweep, set()
        self.p_map, self.r_map, self.m_map, self.q_map, self.extra_cl_log = {}, {}, {}, {}, 0.0
    def bump_op(self) -> None:
        self.diagnostics.contraction_ops += 1
        if self.enforce_budget and self.diagnostics.contraction_ops > self.budget.max_contraction_ops:
            raise QueryBudgetExceededError(f"Contraction ops {self.diagnostics.contraction_ops} > {self.budget.max_contraction_ops}")
    def check_exp(self, exp: ShotExpansion, *, is_choi: bool, enforce_terms: bool = True) -> None:
        d_eff = max(0, exp.d - (self.m_act if is_choi else 0))
        self.diagnostics.peak_terms, self.diagnostics.peak_dim = max(self.diagnostics.peak_terms, exp.chi), max(self.diagnostics.peak_dim, d_eff)
        if self.enforce_budget and enforce_terms and exp.chi > self.budget.max_terms:
            raise QueryBudgetExceededError(f"Expansion terms {exp.chi} > max_terms={self.budget.max_terms}")
        if self.enforce_budget and self.budget.max_dim is not None and d_eff > self.budget.max_dim:
            raise QueryBudgetExceededError(f"Generator dimension {d_eff} > max_dim={self.budget.max_dim}")

def _empty_se(n_q: int, atol: float) -> _ScaledExpansion:
    return _ScaledExpansion(ShotExpansion.from_components((), n_qubits=max(1, n_q), atol=atol), 0)

def _pad1(w: NDArray[np.uint64], nw: int) -> NDArray[np.uint64]:
    return w if w.shape[0] == nw else np.ascontiguousarray(np.pad(w, (0, nw - w.shape[0])))

def _ax_tup(ax: CompiledPauliAxis, nw: int) -> tuple[NDArray[np.uint64], NDArray[np.uint64], int]:
    return (_pad1(ax.a_words, nw), _pad1(ax.b_words, nw), int(ax.kappa) & 3)

def _pad_block(blk: CompiledPauliBlock, n_q: int) -> CompiledPauliBlock:
    if blk.n_qubits == n_q:
        return blk
    nw = n_words_for_qubits(n_q)
    if blk.mode == "pauli_sum" and not blk.is_identity:
        assert blk.a_words is not None and blk.b_words is not None
        pw = nw - blk.a_words.shape[1]
        aw, bw = (np.ascontiguousarray(np.pad(blk.a_words, ((0, 0), (0, pw)))), np.ascontiguousarray(np.pad(blk.b_words, ((0, 0), (0, pw))))) if pw else (blk.a_words, blk.b_words)
        return CompiledPauliBlock(n_q, "pauli_sum", coeffs=blk.coeffs, a_words=aw, b_words=bw, kappas=blk.kappas, global_scalar=blk.global_scalar, is_unitary=blk.is_unitary, is_identity=blk.is_identity)
    steps = tuple(_pad_block(st, n_q) if isinstance(st, CompiledPauliBlock) else CompiledPauliBlockStep(st.kind, tuple((_pad1(a, nw), _pad1(b, nw), kp) for a, b, kp in st.paulis), st.sign, st.weights) for st in blk.steps)
    return CompiledPauliBlock(n_q, "sequential", steps=steps, global_scalar=blk.global_scalar, is_unitary=blk.is_unitary, is_identity=blk.is_identity)

def _scatter_pairs(src: NDArray[np.uint64], pairs: Sequence[tuple[int, int]], n_dst: int) -> NDArray[np.uint64]:
    dst = np.zeros((src.shape[0], n_words_for_qubits(max(1, n_dst))), dtype=np.uint64)
    for s_b, d_b in (pairs if src.shape[0] > 0 else ()):
        if (s_b >> 6) < src.shape[1]:
            dst[:, d_b >> 6] ^= ((src[:, s_b >> 6] >> np.uint64(s_b & 63)) & np.uint64(1)) << np.uint64(d_b & 63)
    return dst

def _normalize_frexp(exp: ShotExpansion, log2_scale: int) -> _ScaledExpansion:
    if exp.chi == 0:
        return _ScaledExpansion(exp, 0)
    if not np.all(np.isfinite(exp.weights)):
        raise FloatingPointError("Non-finite expansion weights encountered")
    w_max = float(np.max(np.abs(exp.weights)))
    if w_max == 0.0:
        return _empty_se(exp.n_qubits, exp.atol)
    shift = math.frexp(w_max)[1] - 1
    if shift:
        exp.weights = np.ascontiguousarray(exp.weights * math.ldexp(1.0, -shift))
    return _ScaledExpansion(exp, log2_scale + shift)

def _tensor_pairs(L: ShotExpansion, p_L: Sequence[tuple[int, int]], R: ShotExpansion, p_R: Sequence[tuple[int, int]], n_tot: int, *, atol: float) -> ShotExpansion:
    chi_L, chi_R, d_L, d_R, d_tot = L.chi, R.chi, L.d, R.d, L.d + R.d
    h_tot = np.repeat(_scatter_pairs(L.h_words, p_L, n_tot), chi_R, axis=0) ^ np.tile(_scatter_pairs(R.h_words, p_R, n_tot), (chi_L, 1))
    A_tot = np.vstack([_scatter_pairs(L.A_words, p_L, n_tot), _scatter_pairs(R.A_words, p_R, n_tot)])
    G_tot = np.zeros((d_tot, d_tot), dtype=np.uint8)
    G_tot[:d_L, :d_L], G_tot[d_L:, d_L:] = L.Gamma, R.Gamma
    ell_L = np.repeat(L.ell, chi_R, axis=0) if d_L else np.zeros((chi_L * chi_R, 0), dtype=np.uint8)
    ell_R = np.tile(R.ell, (chi_L, 1)) if d_R else np.zeros((chi_L * chi_R, 0), dtype=np.uint8)
    m_L, m_R = dict(p_L), dict(p_R)
    piv_tot = np.array([m_L[int(p)] for p in L.pivots] + [m_R[int(p)] for p in R.pivots], dtype=np.int32)
    assert L.ell_odd is not None and R.ell_odd is not None
    a_un = np.bitwise_or.reduce(A_tot, axis=0) if d_tot else np.zeros(n_words_for_qubits(max(1, n_tot)), dtype=np.uint64)
    return ShotExpansion(max(1, n_tot), d_tot, A_tot, a_un, G_tot, piv_tot, np.kron(L.weights, R.weights), h_tot, np.hstack([ell_L, ell_R]) if d_tot else ell_L, np.concatenate([L.ell_odd, R.ell_odd]), atol=atol, _is_canonical_merged=False)

def _contract_choi(left_se: _ScaledExpansion, right_se: _ScaledExpansion, n_op: int, q_act: tuple[int, ...], right_is_choi: bool, ctx: _EvalCtx, *, enforce_terms: bool = True) -> _ScaledExpansion:
    ctx.bump_op()
    m, n_keep = len(q_act), ((n_op + len(q_act)) if right_is_choi else n_op)
    z_left = left_se.norm_sq()
    _check_unitary_norm(1.0, z_left, ctx.atol, "_contract_choi(left)")
    if right_is_choi:
        _check_unitary_norm(1.0, right_se.norm_sq(), ctx.atol, "_contract_choi(right)")
    if right_se.exp.chi == 0:
        return _empty_se(n_keep, ctx.atol)
    z_rhs_rel, prod_chi = right_se.exp.norm_sq(), left_se.exp.chi * right_se.exp.chi
    if ctx.enforce_budget and enforce_terms and prod_chi > max(256, 16 * ctx.budget.max_terms):
        ctx.diagnostics.peak_terms = max(ctx.diagnostics.peak_terms, prod_chi)
        raise QueryBudgetExceededError(f"Contraction product {prod_chi} exceeds max_terms={ctx.budget.max_terms}")
    p_L = [(q, q) for q in q_act] + [(n_op + r, n_keep + r) for r in range(m)]
    p_R = [(q_act[r], n_keep + m + r) for r in range(m)] + ([(n_op + r, n_op + r) for r in range(m)] if right_is_choi else [])
    work = _tensor_pairs(left_se.exp, p_L, right_se.exp, p_R, n_keep + 2 * m, atol=ctx.atol)
    for j in range(m):
        qL, qR = n_keep + j, n_keep + m + j
        wL, sL, wR, sR = qL >> 6, np.uint64(qL & 63), qR >> 6, np.uint64(qR & 63)
        restrict_amplitude(work, (((work.A_words[:, wL] >> sL) ^ (work.A_words[:, wR] >> sR)) & np.uint64(1)).astype(np.uint8), (((work.h_words[:, wL] >> sL) ^ (work.h_words[:, wR] >> sR)) & np.uint64(1)).astype(np.uint8))
        if work.chi == 0:
            _check_unitary_norm(z_left * z_rhs_rel, 0.0, ctx.atol, "_contract_choi")
            return _empty_se(n_keep, ctx.atol)
        for w, s in ((wL, sL), (wR, sR)):
            if work.d > 0:
                work.A_words[:, w] &= ~(np.uint64(1) << s)
            work.h_words[:, w] &= ~(np.uint64(1) << s)
        work._refresh_a_union()
        for r0 in range(work.d):
            if int(work.pivots[r0]) in (qL, qR):
                p_new = first_set_bit_in_words(work.A_words[r0])
                make_principal(work, r0, p_new) if p_new >= 0 else eliminate_zero_amplitude(work, r0)
                break
    nw_keep, tm_keep = n_words_for_qubits(n_keep), tail_mask(n_keep)
    work.n_qubits, work.A_words, work.h_words = n_keep, np.ascontiguousarray(work.A_words[:, :nw_keep]), np.ascontiguousarray(work.h_words[:, :nw_keep])
    if work.d > 0:
        work.A_words[:, -1] &= tm_keep
    work.h_words[:, -1] &= tm_keep
    work._refresh_a_union()
    work._is_canonical_merged = False
    merged = merge_frames(work)
    if (m & 1) and merged.chi > 0:
        merged.weights = np.ascontiguousarray(merged.weights * _SQRT2)
    out_se = _normalize_frexp(merged, left_se.log2_scale + right_se.log2_scale + (m >> 1))
    _check_unitary_norm(z_left * z_rhs_rel, _ldexp(out_se.exp.norm_sq() if out_se.exp.chi > 0 else 0.0, 2 * (out_se.log2_scale - right_se.log2_scale)), ctx.atol, "_contract_choi")
    ctx.check_exp(out_se.exp, is_choi=right_is_choi, enforce_terms=enforce_terms)
    return out_se

def _compile_target_spec(target_spec: Any, output_qubits: tuple[int, ...], atol: float) -> tuple[ReferenceCHPTableau, ShotExpansion] | str | None:
    if target_spec is None:
        return None
    k = len(output_qubits)
    if isinstance(target_spec, stim.Tableau):
        if len(target_spec) > k:
            raise ValueError(f"target_spec Tableau length {len(target_spec)} > {k}")
        target_spec = target_spec.to_circuit(method="elimination")
    if isinstance(target_spec, stim.Circuit):
        if target_spec.num_qubits > k:
            raise ValueError(f"target_spec circuit qubits {target_spec.num_qubits} > {k}")
        padded = target_spec.copy()
        if k > 0 and padded.num_qubits < k:
            padded.append("I", [k - 1])
        t_plan = compile_direct(padded, c2_budget=64, c2_max_rank=8)
        if t_plan.fault_channels or any(not isinstance(r, (CompiledOpenBlockRecord, MagicTermRecord)) for r in t_plan.records):
            raise ValueError("target_spec circuit must contain only unitary Clifford and I[T]/I[T_DAG]/I[CCZ] gates")
        if k == 0:
            return "empty"
        exp_c = ShotExpansion.zero(k, atol=atol)
        for rec in t_plan.records:
            for sb in ((rec,) if isinstance(rec, CompiledOpenBlockRecord) else compile_open_block((rec,), n_qubits=k, budget=64, c2_max_rank=8)):
                exp_c = apply_compiled_block(exp_c, sb.block, check_unitary=True, atol=atol) if isinstance(sb, CompiledOpenBlockRecord) else exp_c
        target_spec = (t_plan.final_tableau, exp_c)
    elif isinstance(target_spec, (PhaseQFEComponent, ShotExpansion)):
        target_spec = (ReferenceCHPTableau(target_spec.n_qubits), target_spec)
    elif isinstance(target_spec, ReferenceCHPTableau):
        target_spec = (target_spec, ShotExpansion.zero(max(1, k), atol=atol))
    is_pair = isinstance(target_spec, tuple) and len(target_spec) == 2 and isinstance(target_spec[0], (stim.Tableau, ReferenceCHPTableau))
    if isinstance(target_spec, (np.ndarray, list, tuple)) and not is_pair:
        if k > 20:
            raise ValueError(f"Dense statevector target_spec exceeds 20 qubits (k={k})")
        arr = np.asarray(target_spec, dtype=np.complex128)
        nrm2 = float(np.vdot(arr, arr).real) if (arr.ndim == 1 and arr.shape == (1 << k,)) else -1.0
        if not math.isfinite(nrm2) or nrm2 <= 1e-15:
            raise ValueError("Invalid dense target_spec shape or norm")
        if k == 0:
            return "empty"
        nz = np.flatnonzero(np.abs(arr) > 1e-15 * math.sqrt(nrm2))
        target_spec, is_pair = (ReferenceCHPTableau(k), ShotExpansion(k, 0, np.zeros((0, 1), dtype=np.uint64), np.zeros(1, dtype=np.uint64), np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.int32), np.ascontiguousarray(arr[nz] / math.sqrt(nrm2), dtype=np.complex128), np.ascontiguousarray(nz.astype(np.uint64)[:, None]), np.zeros((int(nz.shape[0]), 0), dtype=np.uint8), np.zeros(0, dtype=np.uint8), atol=atol, _is_canonical_merged=True)), True
    if not is_pair:
        raise TypeError(f"Unsupported target_spec type: {type(target_spec)!r}")
    raw_ref_t, exp_t = target_spec
    ref_t = stim_tableau_to_reference_chp(raw_ref_t) if isinstance(raw_ref_t, stim.Tableau) else raw_ref_t
    exp_t = ShotExpansion.from_component(exp_t, atol=atol) if isinstance(exp_t, PhaseQFEComponent) else exp_t
    if not isinstance(exp_t, ShotExpansion):
        raise TypeError(f"Expected ShotExpansion in target_spec tuple, got {type(exp_t)!r}")
    if ref_t.num_qubits != k or (k > 0 and exp_t.n_qubits != k):
        raise ValueError(f"target_spec QFE tuple qubit mismatch vs len(output_qubits)={k}")
    if k == 0:
        return "empty"
    merged_t = merge_frames(exp_t.copy())
    nrm = merged_t.norm_sq()
    if not math.isfinite(nrm) or nrm <= 1e-15 or merged_t.chi == 0:
        raise ValueError("target_spec ShotExpansion has zero or non-finite norm")
    merged_t.weights = np.ascontiguousarray(merged_t.weights / math.sqrt(nrm))
    return (clone_reference_chp(ref_t), merged_t)

def _normalize_acceptance_spec(acc_spec: Any, num_meas: int) -> tuple[str, Any]:
    if acc_spec is None or callable(acc_spec):
        return ("all", None) if acc_spec is None else ("callable", acc_spec)
    if not isinstance(acc_spec, Mapping):
        raise TypeError(f"Invalid acceptance_spec: {type(acc_spec)!r}")
    norm_map, imp = {}, False
    for raw_k, raw_v in acc_spec.items():
        idx = _coerce_strict_int(raw_k, "acceptance_spec key")
        idx, bit = (idx + num_meas if idx < 0 else idx), _coerce_strict_int(raw_v, f"acceptance_spec[{raw_k}]", allow_bool=True)
        if idx < 0 or idx >= num_meas or bit not in (0, 1):
            raise ValueError(f"Invalid acceptance_spec entry {raw_k!r}: {raw_v!r} for M={num_meas}")
        imp, norm_map[idx] = (imp or (idx in norm_map and norm_map[idx] != bit)), bit
    return ("impossible", None) if imp else ("map", tuple(sorted(norm_map.items())))

def _check_acc(norm_acc: tuple[str, Any], mb: NDArray[np.bool_]) -> bool:
    return True if norm_acc[0] == "all" else (False if norm_acc[0] == "impossible" else (all(int(mb[i]) == b for i, b in norm_acc[1]) if norm_acc[0] == "map" else bool(norm_acc[1](mb))))

def _bell_steps_for(ref_data: ReferenceCHPTableau, rt: ReferenceCHPTableau, output_qubits: tuple[int, ...], bcache: dict[Any, Any]) -> tuple[Any, ...]:
    n, k = ref_data.num_qubits, len(output_qubits)
    key = (n, k, output_qubits, *(getattr(ref_data, a)[:n].tobytes() for a in ("x2x", "x2z", "z2x", "z2z", "x_kap", "z_kap")), *(getattr(rt, a)[:k].tobytes() for a in ("x2x", "x2z", "z2x", "z2z", "x_kap", "z_kap")))
    if key not in bcache:
        p_R, nw_j, j_ref = [(i, n + i) for i in range(k)], n_words_for_qubits(max(1, n + k)), ReferenceCHPTableau(n + k)
        for attr in ("x2x", "x2z", "z2x", "z2z"):
            dst, src = getattr(j_ref, attr), getattr(ref_data, attr)[:n]
            dst[:n].fill(0)
            dst[:n, :src.shape[1]], dst[n:] = src, _scatter_pairs(getattr(rt, attr)[:k], p_R, n + k)
        j_ref.x_kap[:n], j_ref.z_kap[:n], j_ref.x_kap[n:], j_ref.z_kap[n:] = ref_data.x_kap[:n], ref_data.z_kap[:n], (-rt.x_kap[:k]) & np.uint8(3), (-rt.z_kap[:k]) & np.uint8(3)
        bcache.pop(next(iter(bcache))) if len(bcache) >= 16 else None
        bcache[key] = tuple((mr.rot_axis, (unit_bit_words(first_set_bit_in_words(mr.obs_axis.a_words), nw_j), 0) if mr.rot_axis is not None else (mr.obs_axis.b_words, (mr.obs_axis.kappa >> 1) & 1)) for j, q in enumerate(output_qubits) for pch in ("X", "Z") for mr in (_compile_pauli_measurement_c15(j_ref, [(q, pch), (n + j, pch)], invert_result=False, flip_prob=0.0, meas_index=0),))
    return bcache[key]

def _compute_partial_trace_fidelity_mass(ref_data: ReferenceCHPTableau, exp_data: ShotExpansion, log2_scale: int, comp_target: tuple[ReferenceCHPTableau, ShotExpansion] | str, output_qubits: tuple[int, ...], ctx: _EvalCtx, bcache: dict[Any, Any]) -> tuple[float, int]:
    k, atol = len(output_qubits), ctx.atol
    if exp_data.chi == 0 or k == 0 or comp_target == "empty":
        return _FP_ZERO if exp_data.chi == 0 else _fp(exp_data.norm_sq(), 2 * log2_scale)
    assert isinstance(comp_target, tuple)
    ctx.bump_op()
    n, (rt, et), prod_chi = ref_data.num_qubits, comp_target, exp_data.chi * comp_target[1].chi
    _check_unitary_norm(1.0, et.norm_sq(), atol, "_compute_partial_trace_fidelity_mass(target)")
    if ctx.enforce_budget and prod_chi > ctx.budget.max_terms:
        ctx.diagnostics.peak_terms = max(ctx.diagnostics.peak_terms, prod_chi)
        raise QueryBudgetExceededError(f"Fidelity tensor terms {prod_chi} > max_terms={ctx.budget.max_terms}")
    conj_exp = et.copy()
    conj_exp.weights = np.ascontiguousarray(conj_exp.weights.conj())
    conj_exp.ell = np.ascontiguousarray((-conj_exp.ell) & np.uint8(3)) if (conj_exp.d > 0 and conj_exp.chi > 0) else conj_exp.ell
    j_se = _normalize_frexp(_tensor_pairs(exp_data, [(i, i) for i in range(n)], conj_exp, [(i, n + i) for i in range(k)], n + k, atol=atol), log2_scale)
    ctx.check_exp(j_se.exp, is_choi=False)
    for rot_ax, (bd, sg) in _bell_steps_for(ref_data, rt, output_qubits, bcache):
        if j_se.exp.chi == 0:
            return _FP_ZERO
        ctx.bump_op()
        w_exp = apply_quarter_turn(j_se.exp, rot_ax.a_words, rot_ax.b_words, rot_ax.kappa, sign=1, check_unitary=True, atol=atol) if rot_ax is not None else j_se.exp
        ctx.check_exp(w_exp, is_choi=False)
        z_pre, (p_exp, z0_r) = w_exp.norm_sq(), project_diagonal(w_exp, bd, sg, 0)
        z0_f, scale_z = float(z0_r), max(1.0, abs(z_pre))
        if not math.isfinite(z0_f) or z0_f < -1e-12 * scale_z or z0_f > z_pre + atol * scale_z:
            raise FloatingPointError(f"Bell projection norm drift: z0={z0_f}, z_pre={z_pre}, atol={atol}")
        if z0_f <= K1_CANCEL_REL_TOL * z_pre or p_exp.chi == 0:
            return _FP_ZERO
        j_se = _normalize_frexp(p_exp, j_se.log2_scale)
        ctx.check_exp(j_se.exp, is_choi=False)
    return _fp(j_se.exp.norm_sq(), 2 * j_se.log2_scale + k)

def PrepareQueries(circuit: stim.Circuit, *, fault_mode: str = "DIRECT", fault_channels: Sequence[CliffordFaultChannel] | None = None, target_spec: Any = None, output_qubits: Sequence[int] | None = None, acceptance_spec: Mapping[int, int | bool] | Callable[[NDArray[np.bool_]], bool] | None = None, budget: QueryBudget | None = None, atol: float = DEFAULT_ATOL) -> PreparedQueryPlan:
    """`Q1`: Compile `circuit` into a dependency-tracked QFE Choi/Ket segment-tree plan."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"Expected stim.Circuit, got {type(circuit)!r}")
    if not isinstance(fault_mode, str) or fault_mode.strip().upper() not in ("DIRECT", "SELECTOR"):
        raise ValueError(f"fault_mode must be 'DIRECT' or 'SELECTOR', got {fault_mode!r}")
    mode_norm, tol, eff_budget = fault_mode.strip().upper(), validate_atol(atol), (QueryBudget() if budget is None else budget)
    if not isinstance(eff_budget, QueryBudget):
        raise TypeError(f"Expected QueryBudget, got {type(eff_budget)!r}")
    compiled: DirectPlan | SelectorPlan = compile_direct(circuit, fault_channels=fault_channels, c2_budget=64, c2_max_rank=6) if mode_norm == "DIRECT" else compile_selector(circuit, fault_channels=fault_channels, gadgetize_pauli_noise=False, c2_budget=64, c2_max_rank=6)
    if any(isinstance(rec, LeakageBarrierRecord) for rec in compiled.records):
        raise ValueError("Unsupported leakage tag in PrepareQueries")
    n_data, n_op = (compiled.num_qubits if isinstance(compiled, DirectPlan) else compiled.num_data_qubits), max(1, compiled.num_qubits if isinstance(compiled, DirectPlan) else compiled.num_total_qubits)
    nw_op, logical_nq = n_words_for_qubits(n_op), max([int(circuit.num_qubits)] + [q + 1 for ch in compiled.fault_channels for br in ch.branches for q in br.targets])
    out_qs = tuple(range(logical_nq)) if output_qubits is None else tuple(_coerce_strict_int(q, "output_qubit") for q in output_qubits)
    if any(q < 0 or q >= logical_nq for q in out_qs) or len(set(out_qs)) != len(out_qs):
        raise ValueError(f"Invalid output_qubits={list(out_qs)} for num_data_qubits={logical_nq}")
    comp_target = _compile_target_spec(target_spec, out_qs, tol)
    _normalize_acceptance_spec(acceptance_spec, compiled.num_measurements)
    slot_id_to_idx = {i: i for i in range(len(compiled.fault_channels))} | {int(ch.slot_id): i for i, ch in enumerate(compiled.fault_channels) if ch.slot_id is not None}
    axes_of = lambda r: r.axes if isinstance(r, CompiledOpenBlockRecord) else (r.term.axes if isinstance(r, MagicTermRecord) else (
        [ax for bf in r.branch_factors for ax in ((bf.p0,) if bf.p0 is not None else ()) + bf.r_axes] if isinstance(r, DirectFaultSlotRecord) else (
            [ax for ax in r.branch_axes if ax is not None] if isinstance(r, (PauliNoiseRecord, HeraldedNoiseRecord)) else ((r.axis,) if isinstance(r, (CorrelatedNoiseRecord, FeedbackPauliRecord)) else ([ax for ax in (r.obs_axis, getattr(r, "reset_kick_axis", None), r.rot_axis) if ax is not None] if isinstance(r, (MeasurementRecord, ResetRecord)) else ())))))
    x_union = np.zeros(nw_op, dtype=np.uint64)
    for rec in compiled.records:
        for ax in axes_of(rec):
            x_union[: ax.a_words.shape[0]] |= ax.a_words
    for sq in (sq for s_grp in (compiled.fault_selector_groups if isinstance(compiled, SelectorPlan) else ()) for sq in s_grp):
        x_union[sq >> 6] |= np.uint64(1) << np.uint64(sq & 63)
    q_act = tuple(q for q in range(n_op) if ((int(x_union[q >> 6]) >> (q & 63)) & 1)) or (0,)
    n_choi, sel_ins = n_op + len(q_act), {}
    if isinstance(compiled, SelectorPlan):
        z0 = np.zeros(nw_op, dtype=np.uint64)
        for s_idx, s_grp in enumerate(compiled.fault_selector_groups):
            fpos = next((i for i, rec in enumerate(compiled.records) if any((q >> 6) < ax.a_words.shape[0] and ((int((ax.a_words | ax.b_words)[q >> 6]) >> (q & 63)) & 1) for ax in axes_of(rec) for q in s_grp)), len(compiled.records))
            bfacs = tuple(CompiledCliffordFactorList(0, CompiledPauliAxis(unit_bit_words(sq, nw_op), z0, 0), ()) for sq in s_grp)
            sel_ins.setdefault(fpos, []).append(DirectFaultSlotRecord(s_idx, compiled.fault_channels[s_idx], bfacs, np.empty(0, dtype=np.float64)))
    ops, dyn, noise_meta, rc, cs = [], [], {}, 0, ()
    for r_i, rec in enumerate(compiled.records):
        ops.extend((sr, None, ((sr.slot_index,), (), ())) for sr in sel_ins.get(r_i, ()))
        if isinstance(rec, CompiledOpenBlockRecord):
            ops.append((_pad_block(rec.block, n_op), _pad_block(rec.block, n_choi), _EMPTY_DEPS))
        elif isinstance(rec, MagicTermRecord):
            ops.extend((_pad_block(sb.block, n_op), _pad_block(sb.block, n_choi), _EMPTY_DEPS) for sb in compile_open_block((rec,), n_qubits=n_op, budget=64, c2_max_rank=6) if isinstance(sb, CompiledOpenBlockRecord))
        elif isinstance(rec, DirectFaultSlotRecord):
            ops.append((rec, None, ((rec.slot_index,), (), ())))
        elif isinstance(rec, FeedbackPauliRecord):
            di = rec.current_meas_count + rec.source_index if rec.source_kind == "rec" else rec.source_index
            ops.append((rec, di, ((), (di,), ()) if rec.source_kind == "rec" else ((), (), (di,))))
        elif isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord, CorrelatedNoiseRecord)):
            ni = rec.noise_index
            is_corr, prior = isinstance(rec, CorrelatedNoiseRecord), (cs if getattr(rec, "is_else", False) else ())
            noise_meta[ni], cs = ("CORRELATED_ERROR" if is_corr else rec.kind), ((*prior, ni) if is_corr else cs)
            dyn.append((len(ops), rec, (ni, prior) if is_corr else ni))
        elif isinstance(rec, (MeasurementRecord, ResetRecord, MPadRecord)):
            dyn.append((len(ops), rec, rc if isinstance(rec, ResetRecord) else None))
            rc += int(isinstance(rec, ResetRecord))
    ops.extend((sr, None, ((sr.slot_index,), (), ())) for sr in sel_ins.get(len(compiled.records), ()))
    otup, nodes = tuple(ops), []

    def _build(l: int, r: int) -> _SegTreeNode:
        nid = len(nodes)
        nodes.append(None)  # type: ignore[arg-type]
        if r - l == 1:
            u = _SegTreeNode(nid, l, r, leaf=otup[l], deps=otup[l][2])
        else:
            lc, rc_n = _build(l, (l + r) >> 1), _build((l + r) >> 1, r)
            u = _SegTreeNode(nid, l, r, left=lc, right=rc_n, deps=tuple((tuple(sorted(set(a).union(b))) if (a and b) else (a or b)) for a, b in zip(lc.deps, rc_n.deps)))
        nodes[nid] = u
        return u

    root = _build(0, len(otup)) if otup else None
    bcache: dict[Any, Any] = {}
    _bell_steps_for(compiled.data_final_tableau if isinstance(compiled, SelectorPlan) else compiled.final_tableau, comp_target[0], out_qs, bcache) if (isinstance(comp_target, tuple) and out_qs) else None
    st = {"target": comp_target, "bcache": bcache, "slots": slot_id_to_idx, "noise": noise_meta, "q_act": q_act, "ops": otup, "dyn": tuple(dyn), "root": root, "nodes": tuple(nodes), "lru": OrderedDict(), "bytes": 0}
    return PreparedQueryPlan(circuit, mode_norm, logical_nq, n_data, n_op, compiled.num_measurements, out_qs, eff_budget, tol, compiled, target_spec, acceptance_spec, st)

def _apply_leaf(se: _ScaledExpansion, leaf: tuple[Any, Any, Any], ctx: _EvalCtx, *, is_choi: bool, enforce_terms: bool = True) -> _ScaledExpansion:
    ctx.bump_op()
    if se.exp.chi == 0:
        return se
    z_pre, exp, nw, (rec, aux, _) = se.exp.norm_sq(), se.exp.copy(), se.exp.n_words, leaf
    if isinstance(rec, CompiledPauliBlock):
        exp = apply_compiled_block(exp, aux if is_choi else rec, check_unitary=True, atol=ctx.atol)
    elif isinstance(rec, DirectFaultSlotRecord) and ctx.faults.get(rec.slot_index, 0) > 0:
        fac = rec.branch_factors[ctx.faults[rec.slot_index] - 1]
        if fac.p0 is not None:
            exp = apply_pauli(exp, *_ax_tup(fac.p0, nw))
        for ra in fac.r_axes:
            exp = apply_quarter_turn(exp, *_ax_tup(ra, nw), sign=1, check_unitary=True, atol=ctx.atol)
            ctx.check_exp(exp, is_choi=is_choi, enforce_terms=enforce_terms)
        if fac.v:
            exp.weights = np.ascontiguousarray(exp.weights * OMEGA_POWERS[fac.v])
    elif isinstance(rec, FeedbackPauliRecord) and (ctx.meas.get(aux, 0) if rec.source_kind == "rec" else ctx.sweep.get(aux, 0)):
        exp = apply_pauli(exp, *_ax_tup(rec.axis, nw))
    _check_unitary_norm(z_pre, exp.norm_sq() if exp.chi > 0 else 0.0, ctx.atol, "_apply_leaf")
    out_se = _normalize_frexp(exp, se.log2_scale)
    ctx.check_exp(out_se.exp, is_choi=is_choi, enforce_terms=enforce_terms)
    return out_se

def _apply_op_to_ket(plan: PreparedQueryPlan, ctx: _EvalCtx, op: Any, ket: _ScaledExpansion) -> _ScaledExpansion:
    for item in ((op,) if isinstance(op, _ScaledExpansion) else op):
        if ket.exp.chi == 0:
            break
        ket = _contract_choi(item, ket, plan.num_total_qubits, plan._state["q_act"], False, ctx) if isinstance(item, _ScaledExpansion) else _apply_leaf(ket, item, ctx, is_choi=False)
    return ket

def _eval_node(plan: PreparedQueryPlan, ctx: _EvalCtx, node: _SegTreeNode, *, is_choi: bool) -> Any:
    df, dm, ds = node.deps
    dkey = (tuple(ctx.faults.get(i, 0) for i in df), tuple(ctx.meas.get(i, -1) for i in dm), tuple(ctx.sweep.get(i, 0) for i in ds))
    ctx.visited.add(node.node_id)
    ckey, st, cap = (node.node_id, "CHOI" if is_choi else "KET", dkey), plan._state, plan.budget.max_cache_bytes
    if cap > 0 and ckey in st["lru"]:
        hit = st["lru"][ckey]
        st["lru"].move_to_end(ckey)
        ctx.diagnostics.cache_hits += 1
        if isinstance(hit, _ScaledExpansion):
            ctx.check_exp(hit.exp, is_choi=is_choi)
            return hit.copy()
        return hit
    ctx.diagnostics.cache_misses += 1
    q_act, n_op = st["q_act"], plan.num_total_qubits
    if is_choi:
        allow_defer, choi_cap = ctx.budget.max_terms > 256, min(ctx.budget.max_terms, 256)
        if node.leaf is not None:
            ctx.diagnostics.node_evaluations += 1
            m, nw = len(q_act), n_words_for_qubits(n_op + len(q_act))
            A = np.zeros((m, nw), dtype=np.uint64)
            for r, q in enumerate(q_act):
                A[r, q >> 6] ^= np.uint64(1) << np.uint64(q & 63)
                A[r, (n_op + r) >> 6] ^= np.uint64(1) << np.uint64((n_op + r) & 63)
            bell = _ScaledExpansion(ShotExpansion(n_op + m, m, A, np.bitwise_or.reduce(A, axis=0), np.zeros((m, m), dtype=np.uint8), np.array(q_act, dtype=np.int32), np.array([1.0 + 0.0j], dtype=np.complex128), np.zeros((1, nw), dtype=np.uint64), np.zeros((1, m), dtype=np.uint8), np.zeros(m, dtype=np.uint8), atol=ctx.atol), 0)
            se = _apply_leaf(bell, node.leaf, ctx, is_choi=True, enforce_terms=not allow_defer)
            val: Any = (node.leaf,) if (allow_defer and se.exp.chi > choi_cap) else se
        else:
            assert node.left is not None and node.right is not None
            v_l, v_r = _eval_node(plan, ctx, node.left, is_choi=True), _eval_node(plan, ctx, node.right, is_choi=True)
            ctx.diagnostics.node_evaluations += 1
            if isinstance(v_l, _ScaledExpansion) and isinstance(v_r, _ScaledExpansion) and not (allow_defer and v_l.exp.chi * v_r.exp.chi > 8 * choi_cap):
                comp = _contract_choi(v_r, v_l, n_op, q_act, True, ctx, enforce_terms=not allow_defer)
                val = (v_l, v_r) if (allow_defer and comp.exp.chi > choi_cap) else comp
            else:
                val = (v_l if isinstance(v_l, tuple) else (v_l,)) + (v_r if isinstance(v_r, tuple) else (v_r,))
    elif node.leaf is not None:
        ctx.diagnostics.node_evaluations += 1
        val = _apply_leaf(_ScaledExpansion(ShotExpansion.zero(n_op, atol=ctx.atol), 0), node.leaf, ctx, is_choi=False)
    else:
        assert node.left is not None and node.right is not None
        ket_l, op_r = _eval_node(plan, ctx, node.left, is_choi=False), _eval_node(plan, ctx, node.right, is_choi=True)
        ctx.diagnostics.node_evaluations += 1
        val = _apply_op_to_ket(plan, ctx, op_r, ket_l)
    sz = lambda v: (int(v.exp.weights.nbytes + v.exp.h_words.nbytes + v.exp.ell.nbytes + v.exp.A_words.nbytes + v.exp.Gamma.nbytes + 64) if isinstance(v, _ScaledExpansion) else sum(sz(x) if isinstance(x, _ScaledExpansion) else 64 for x in v))
    nb = sz(val)
    if 0 < nb <= cap:
        old = st["lru"].pop(ckey, None)
        st["bytes"], st["lru"][ckey] = st["bytes"] + nb - (sz(old) if old is not None else 0), (val.copy() if isinstance(val, _ScaledExpansion) else val)
        while st["bytes"] > cap and st["lru"]:
            st["bytes"] -= sz(st["lru"].popitem(last=False)[1])
    return val.copy() if isinstance(val, _ScaledExpansion) else val

def _apply_node_range(plan: PreparedQueryPlan, ctx: _EvalCtx, u: _SegTreeNode | None, ql: int, qr: int, ket: _ScaledExpansion, *, is_vacuum: bool = False) -> _ScaledExpansion:
    if u is None or ql >= qr or ql >= u.r or qr <= u.l:
        return ket
    if ql <= u.l and u.r <= qr:
        return _eval_node(plan, ctx, u, is_choi=False) if (is_vacuum and u.l == 0 and ql == 0) else (_apply_op_to_ket(plan, ctx, _eval_node(plan, ctx, u, is_choi=True), ket) if ket.exp.chi > 0 else ket)
    return _apply_node_range(plan, ctx, u.right, ql, qr, _apply_node_range(plan, ctx, u.left, ql, qr, ket, is_vacuum=is_vacuum), is_vacuum=False)

def _selector_bits(plan: SelectorPlan, faults: Mapping[int, int]) -> NDArray[np.bool_]:
    sel = np.zeros(plan.num_selector_qubits, dtype=np.bool_)
    for s_idx, s_grp in enumerate(plan.fault_selector_groups):
        if faults.get(s_idx, 0) > 0:
            sel[s_grp[faults[s_idx] - 1] - plan.num_data_qubits] = True
    return sel

def _norm_sweep(raw_s: Any) -> dict[int, int]:
    if raw_s is not None and not isinstance(raw_s, Mapping) and (isinstance(raw_s, (str, bytes, bytearray, memoryview)) or (raw_s.ndim != 1 if isinstance(raw_s, np.ndarray) else not isinstance(raw_s, Sequence))):
        raise ValueError(f"Invalid sweep_bits container: {type(raw_s)!r}")
    out = {} if raw_s is None else ({_coerce_strict_int(k, "sweep_index"): _coerce_strict_int(v, "sweep_bit", allow_bool=True) for k, v in raw_s.items()} if isinstance(raw_s, Mapping) else {i: _coerce_strict_int(v, "sweep_bit", allow_bool=True) for i, v in enumerate(raw_s)})
    if any(k < 0 or v not in (0, 1) for k, v in out.items()):
        raise ValueError("Sweep keys must be >= 0 and sweep bits must be 0 or 1")
    return out

def _unpack_realization(plan: PreparedQueryPlan, realization: Mapping[Any, Any] | None) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    if not isinstance(plan, PreparedQueryPlan) or (realization is not None and not isinstance(realization, Mapping)):
        raise TypeError(f"Expected PreparedQueryPlan and Mapping|None realization, got {type(plan)!r}, {type(realization)!r}")
    if not realization:
        return {}, {}, {}, {}, {}, {}
    if any(isinstance(k, str) for k in realization):
        if set(realization) - _VALID_REAL_KEYS:
            raise ValueError(f"Unknown realization keys: {sorted(map(str, set(realization) - _VALID_REAL_KEYS))}")
        raw_f, raw_m, raw_qm = realization.get("faults"), realization.get("measurements"), realization.get("quantum_measurements")
        raw_p, raw_r = realization.get("paulis", realization.get("pauli_noise")), realization.get("resets")
        raw_s = realization.get("sweep_bits", realization.get("sweep"))
    else:
        raw_f, raw_m, raw_qm, raw_p, raw_r, raw_s = realization, None, None, None, None, None
    cp, n_meta, max_c = plan.compiled_plan, plan._state["noise"], plan.compiled_plan.noise_channel_max_choices
    if isinstance(raw_p, Mapping):
        conv_p: dict[Any, Any] = {}
        for k, v in raw_p.items():
            ik = _coerce_strict_int(k, "paulis key")
            if ik < 0 or ik >= len(max_c):
                raise ValueError(f"Pauli slot {ik} out of range")
            mc, kind = max_c[ik], n_meta.get(ik, "")
            if isinstance(v, str):
                s, tbl = v.strip().upper(), (({"I": 0, kind[:1]: 1} if kind in ("X_ERROR", "Y_ERROR", "Z_ERROR") else {"I": 0}) if mc == 1 else ({3: _PAULI_1Q_STR, 4: _PAULI_4_STR, 15: _PAULI_2Q_STR}.get(mc, {})))
                if s not in tbl:
                    raise ValueError(f"Invalid Pauli label {v!r} at slot {ik}")
                conv_p[ik] = tbl[s]
            else:
                conv_p[ik] = int(bool(v)) if (mc == 1 and isinstance(v, (bool, np.bool_))) else v
        raw_p = conv_p
    f_max = [len(ch.branches) for ch in cp.fault_channels]
    raw_f = ({k: (int(bool(v)) if (isinstance(v, (bool, np.bool_)) and not isinstance(k, (bool, np.bool_)) and isinstance(k, (int, np.integer)) and int(k) in plan._state["slots"] and f_max[plan._state["slots"][int(k)]] == 1) else v) for k, v in raw_f.items()} if isinstance(raw_f, Mapping) else ([(int(bool(v)) if (isinstance(v, (bool, np.bool_)) and i < len(f_max) and f_max[i] == 1) else v) for i, v in enumerate(raw_f)] if ((isinstance(raw_f, Sequence) and not isinstance(raw_f, (str, bytes))) or (isinstance(raw_f, np.ndarray) and raw_f.ndim == 1)) else raw_f))
    faults = _normalize_choice_map(raw_f, f_max, "faults", key_to_slot=plan._state["slots"], allow_none_zero=True) or {}
    meas = _normalize_choice_map(raw_m, (1,) * plan.num_measurements, "measurements", allow_bool_values=True) or {}
    qmeas = _normalize_choice_map(raw_qm, (1,) * plan.num_measurements, "quantum_measurements", allow_bool_values=True) or {}
    paulis, resets = (_normalize_choice_map(raw_p, max_c, "paulis") or {}), (_normalize_choice_map(raw_r, (1,) * cp.num_resets, "resets", allow_bool_values=True) or {})
    return faults, meas, qmeas, paulis, resets, _norm_sweep(raw_s)

def _eval_via_tree(plan: PreparedQueryPlan, faults_in: dict[int, int], meas_in: dict[int, int], qmeas_in: dict[int, int], paulis_in: dict[int, int], resets_in: dict[int, int], sweep_in: dict[int, int],
                   q_set: frozenset[str], norm_acc: tuple[str, Any], comp_target: Any, rng_obj: np.random.Generator | None, *, sample_mc_faults: bool, is_mixed: bool,
                   stream_only: bool = False, prev_diag: QueryDiagnostics | None = None, out_ctx: list[_EvalCtx] | None = None, pin: Any = None
                   ) -> tuple[tuple[float, int], tuple[float, int], tuple[float, int], bool, list[tuple[float, float]], NDArray[np.bool_], _ScaledExpansion, dict[int, int], QueryDiagnostics]:
    use_sampling = ("sample_measurements" in q_set) or sample_mc_faults
    r_gen = (rng_obj if rng_obj is not None else np.random.default_rng()) if use_sampling else None
    faults = {i: (faults_in[i] if i in faults_in else _sample_categorical_branch([b.prob for b in ch.branches], r_gen)) for i, ch in enumerate(plan.compiled_plan.fault_channels)} if sample_mc_faults else dict(faults_in)  # type: ignore[arg-type]
    m_act, otup, dyn, nodes, root, bcache = len(plan._state["q_act"]), plan._state["ops"], plan._state["dyn"], plan._state["nodes"], plan._state["root"], plan._state["bcache"]
    ctx = _EvalCtx(plan.budget, plan.atol, m_act, faults, dict(meas_in), dict(sweep_in), enforce_budget=not stream_only, prev_diag=prev_diag)
    if out_ctx is not None:
        out_ctx.append(ctx)
    init_ket, step_w = _ScaledExpansion(ShotExpansion.zero(plan.num_total_qubits, atol=plan.atol), 0), [[_FP_ZERO] * plan.num_measurements, [_FP_ZERO] * plan.num_measurements]
    totals, last_ket, last_m, m_vals = [_FP_ZERO, _FP_ZERO, _FP_ZERO], _empty_se(plan.num_total_qubits, plan.atol), dict(meas_in), {k: {v} for k, v in meas_in.items()}
    run_seg = lambda kt, cl, cr, iv: kt if cl >= cr else ((lambda out: [out := _apply_leaf(out, otup[ki], ctx, is_choi=False) for ki in range(cl, cr) if out.exp.chi > 0] and out or out)(kt) if stream_only else _apply_node_range(plan, ctx, root, cl, cr, kt, is_vacuum=iv))
    stack: list[tuple[int, int, _ScaledExpansion, tuple[float, int], dict[int, int], dict[int, int], dict[int, int]]] = [(0, 0, init_ket, (0.5, 1), dict(paulis_in), dict(resets_in), {})]
    while stack:
        di, cur_l, ket, pi_w, p_map, r_map, m_map = stack.pop()
        if pi_w[0] <= 0.0 or ket.exp.chi == 0:
            continue
        ctx.meas = m_map
        for mk, mv in (() if stream_only else m_map.items()):
            m_vals.setdefault(mk, set()).add(mv)
        if di >= len(dyn):
            ket_e = run_seg(ket, cur_l, len(otup), di == 0)
            last_ket, last_m, z_b = ket_e, m_map, _fp_mul(pi_w, ket_e.exp.norm_sq() if ket_e.exp.chi > 0 else 0.0, 2 * ket_e.log2_scale)
            if z_b[0] > 0.0:
                totals[0] = _fp_add(totals[0], z_b)
                if _check_acc(norm_acc, np.array([bool(m_map.get(j, 0)) for j in range(plan.num_measurements)], dtype=np.bool_)):
                    totals[1] = _fp_add(totals[1], z_b)
                    if "fidelity" in q_set and comp_target is not None:
                        d_state = extract_data_expansion(plan.compiled_plan, ket_e.exp, _selector_bits(plan.compiled_plan, faults)) if isinstance(plan.compiled_plan, SelectorPlan) else (clone_reference_chp(plan.compiled_plan.final_tableau), ket_e.exp.copy())
                        totals[2] = _fp_add(totals[2], _fp_mul(pi_w, *_compute_partial_trace_fidelity_mass(*d_state, ket_e.log2_scale, comp_target, plan.output_qubits, ctx, bcache)))
            continue
        op_off, rec, aux = dyn[di]
        ket = run_seg(ket, cur_l, op_off, di == 0)
        if ket.exp.chi == 0:
            continue
        z_in_fp, nxt = _fp_mul(pi_w, ket.exp.norm_sq(), 2 * ket.log2_scale), []
        if isinstance(rec, (PauliNoiseRecord, CorrelatedNoiseRecord, HeraldedNoiseRecord, MPadRecord)):
            mi = rec.meas_index if isinstance(rec, (HeraldedNoiseRecord, MPadRecord)) else None
            if isinstance(rec, PauliNoiseRecord):
                ni, prior, dist = aux, False, ([(p_map[aux], None, 1.0)] if aux in p_map else [(0, None, max(0.0, 1.0 - sum(rec.probs))), *((i + 1, None, p) for i, p in enumerate(rec.probs))])
            elif isinstance(rec, CorrelatedNoiseRecord):
                ni, cs = aux
                prior = any(p_map.get(j, 0) > 0 for j in cs)
                dist = [(0, None, 1.0)] if prior else ([(p_map[ni], None, 1.0)] if ni in p_map else [(0, None, 1.0 - rec.prob), (1, None, rec.prob)])
            elif isinstance(rec, HeraldedNoiseRecord):
                ni, prior, dist = aux, False, ([(p_map[aux], int(p_map[aux] > 0), 1.0)] if aux in p_map else [(0, 0, max(0.0, 1.0 - sum(rec.probs))), *((i + 1, 1, p) for i, p in enumerate(rec.probs))])
            else:
                ni, prior, dist = None, False, [(None, 0, rec.flip_prob if rec.pad_bit else 1.0 - rec.flip_prob), (None, 1, 1.0 - rec.flip_prob if rec.pad_bit else rec.flip_prob)]
            for _, mb, pr in (dist if mi is not None else ()):
                step_w[int(bool(mb))][mi] = _fp_add(step_w[int(bool(mb))][mi], _fp_mul(z_in_fp, pr))  # type: ignore[index]
            surv = [(cb, mb, pr) for cb, mb, pr in dist if pr > 0.0 and (mi is None or mi not in meas_in or meas_in[mi] == mb)]
            if use_sampling and surv:
                assert r_gen is not None
                w_sum = sum(pr for _, _, pr in surv)
                ps = next((s for s in surv if ((mi in pin[0] and s[1] == pin[0][mi]) if isinstance(rec, MPadRecord) else (ni in pin[3] and s[0] == pin[3][ni]))), None) if pin else None
                if ps is None and isinstance(rec, MPadRecord) and mi not in meas_in:
                    mb = int(bool(rec.pad_bit) ^ (rec.flip_prob > 0.0 and float(r_gen.random()) < rec.flip_prob))
                    cb, ctx.extra_cl_log = None, ctx.extra_cl_log + _ln((1.0 - rec.flip_prob) if bool(mb) == bool(rec.pad_bit) else (((1.0 - (1.0 - rec.flip_prob)) or rec.flip_prob) if rec.pad_bit else rec.flip_prob))
                elif ps is None and isinstance(rec, CorrelatedNoiseRecord) and ni not in p_map and not prior:
                    cb, mb = int(float(r_gen.random()) < rec.prob), None
                elif ps is None and isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord)) and ni not in p_map and (mi is None or mi not in meas_in):
                    cb = _sample_categorical_branch(rec.probs, r_gen)
                    mb = int(cb > 0) if mi is not None else None
                else:
                    cb, mb, _ = ps if ps is not None else surv[_sample_categorical_branch([pr / w_sum for _, _, pr in surv[:-1]], r_gen) - 1 if len(surv) > 1 else 0]
                if ni is not None and cb is not None and ni not in paulis_in:
                    ctx.p_map[ni] = cb
                if mi is not None and mb is not None and mi not in meas_in:
                    ctx.m_map[mi] = mb
                surv = [(cb, mb, w_sum if (mi is not None and mi in meas_in) else 1.0)]
            for cb, mb, pw in surv:
                ax = (rec.branch_axes[cb - 1] if isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord)) else rec.axis) if (cb is not None and cb > 0 and not isinstance(rec, MPadRecord)) else None
                if ax is not None:
                    ctx.bump_op()
                    sk = _normalize_frexp(apply_pauli(ket.exp.copy(), *_ax_tup(ax, ket.exp.n_words)), ket.log2_scale)
                    ctx.check_exp(sk.exp, is_choi=False)
                else:
                    sk = ket
                nxt.append((sk, _fp_mul(pi_w, pw), ({**p_map, ni: cb} if (ni is not None and cb is not None) else p_map), r_map, ({**m_map, mi: mb} if (mi is not None and mb is not None) else m_map)))
        else:
            assert isinstance(rec, (MeasurementRecord, ResetRecord))
            ctx.bump_op()
            nw, z_in_r, work = ket.exp.n_words, ket.exp.norm_sq(), ket.exp.copy()
            if rec.rot_axis is not None:
                work = apply_quarter_turn(work, *_ax_tup(rec.rot_axis, nw), sign=1, check_unitary=True, atol=ctx.atol)
                ctx.check_exp(work, is_choi=False)
                b_diag, sig = unit_bit_words(first_set_bit_in_words(rec.obs_axis.a_words), nw), 0
            else:
                _, b_diag, kp = _ax_tup(rec.obs_axis, nw)
                sig = (kp >> 1) & 1
            (e0, z0_r), (e1, z1_r) = project_diagonal(work, b_diag, sig, 0), project_diagonal(work, b_diag, sig, 1)
            z0_f, z1_f = float(z0_r), float(z1_r)
            if not (math.isfinite(z0_f) and math.isfinite(z1_f) and math.isfinite(z_in_r)):
                raise FloatingPointError(f"Non-finite projected Born masses: z0={z0_f!r}, z1={z1_f!r}, z_in={z_in_r!r}")
            scale_z = max(1.0, z_in_r)
            if z0_f < -1e-12 * scale_z or z1_f < -1e-12 * scale_z:
                raise FloatingPointError(f"Significantly negative projected Born mass: z0={z0_f!r}, z1={z1_f!r}")
            z0_c, z1_c = max(0.0, z0_f), max(0.0, z1_f)
            if abs((z0_c + z1_c) - z_in_r) > ctx.atol * scale_z:
                raise FloatingPointError(f"Born mass conservation violated: z0={z0_c}, z1={z1_c}, z_in={z_in_r}, atol={ctx.atol}")
            brs = []
            for qb, eb, zb_c in ((0, e0, z0_c), (1, e1, z1_c)):
                if z_in_r > 0.0 and zb_c > K1_CANCEL_REL_TOL * z_in_r and eb.chi > 0:
                    eb.weights = np.ascontiguousarray(eb.weights * (1.0 / math.sqrt(zb_c)))
                    eb = apply_pauli(eb, *_ax_tup(rec.reset_kick_axis, nw)) if (isinstance(rec, ResetRecord) and bool(qb) != rec.reset_flip_ref) else eb
                    sk = _normalize_frexp(eb, 0)
                    ctx.check_exp(sk.exp, is_choi=False)
                    brs.append((qb, sk, zb_c / (z0_c + z1_c)))
                else:
                    brs.append((qb, _empty_se(ket.exp.n_qubits, ctx.atol), 0.0))
            (_, s0, pb0), (_, s1, pb1) = brs
            mi, fp, iota = rec.meas_index, (rec.flip_prob if rec.meas_index is not None else 0.0), int(rec.invert_result)
            fl_opts = ((0, 1.0),) if mi is None else ((0, 1.0 - fp), (1, fp))
            for qb, pb in (((0, pb0), (1, pb1)) if mi is not None else ()):
                for fl, pfl in (fl_opts if pb > 0.0 else ()):
                    step_w[qb ^ iota ^ fl][mi] = _fp_add(step_w[qb ^ iota ^ fl][mi], _fp_mul(z_in_fp, pb * pfl))  # type: ignore[index]
            q_cands = [(qb, sk, pb) for qb, sk, pb in ((0, s0, pb0), (1, s1, pb1)) if pb > 0.0 and (not isinstance(rec, ResetRecord) or aux not in r_map or r_map[aux] == qb) and (mi is None or mi not in qmeas_in or qmeas_in[mi] == qb)]
            cands = [(qb, sk, pb, (None if mi is None else (qb ^ iota ^ fl)), pfl) for qb, sk, pb in q_cands for fl, pfl in fl_opts if pfl > 0.0 and (mi is None or mi not in meas_in or meas_in[mi] == (qb ^ iota ^ fl))]
            if use_sampling and cands:
                assert r_gen is not None
                pc = next((c for c in cands if ((aux in pin[2] and c[0] == pin[2][aux]) if mi is None else (mi in pin[1] and c[0] == pin[1][mi] and (mi in meas_in or c[3] == pin[0].get(mi))))), None) if pin else None
                if mi is None or mi not in meas_in:
                    qb = pc[0] if pc else (q_cands[0][0] if len(q_cands) == 1 else int(float(r_gen.random()) < pb1))
                    fl = (qb ^ iota ^ int(pc[3])) if (pc and mi is not None) else (int(float(r_gen.random()) < fp) if (mi is not None and fp > 0.0) else 0)
                    sk, pb, pfl, rp = *((s1, pb1) if qb else (s0, pb0)), ((fp if fl else (1.0 - fp)) if mi is not None else 1.0), (None if mi is None else (qb ^ iota ^ fl))
                    if mi is not None:
                        ctx.m_map[mi], ctx.q_map[mi], ctx.extra_cl_log = rp, qb, ctx.extra_cl_log + _ln(pfl)
                    cands = [(qb, sk, pb, rp, pb)] if pb > 0.0 else []
                else:
                    w_sum = sum(pb * pfl for _, _, pb, _, pfl in cands)
                    qb, sk, pb, rp, pfl = pc if pc is not None else cands[int(float(r_gen.random()) < (cands[1][2] * cands[1][4]) / w_sum) if (len(cands) > 1 and w_sum > 0.0) else 0]
                    ctx.q_map[mi], ctx.extra_cl_log, cands = qb, ctx.extra_cl_log + (_ln((pb * pfl) / w_sum) if w_sum > 0.0 else -math.inf), [(qb, sk, pb, rp, w_sum)]
                if isinstance(rec, ResetRecord) and aux not in resets_in and cands:
                    ctx.r_map[aux] = cands[0][0]
            else:
                cands = [(qb, sk, pb, rp, pb * pfl) for qb, sk, pb, rp, pfl in cands]
            for qb, sk, _, rp, pw in cands:
                nxt.append((sk, _fp_mul(z_in_fp, pw), p_map, ({**r_map, aux: qb} if isinstance(rec, ResetRecord) else r_map), ({**m_map, mi: rp} if (mi is not None and rp is not None) else m_map)))
        for br in reversed(nxt):
            stack.append((di + 1, op_off, *br))
    for u in (() if stream_only else nodes):
        if not (any(not m_vals.get(i) for i in u.deps[1]) and u.node_id not in ctx.visited):
            ck = (tuple(faults.get(i, 0) for i in u.deps[0]), tuple(tuple(sorted(m_vals.get(i, ()))) for i in u.deps[1]), tuple(sweep_in.get(i, 0) for i in u.deps[2]))
            ctx.diagnostics.invalidated_nodes += int(u.last_key is not None and u.last_key != ck)
            u.last_key = ck if (u.node_id in ctx.visited or u.last_key is not None) else None
    m_arr = np.array([bool((last_m if is_mixed else (last_m | m_map)).get(j, 0)) for j in range(plan.num_measurements)], dtype=np.bool_)
    acc = (True if (norm_acc[0] == "all" and totals[0][0] > 0.0) else (totals[1][0] > 0.0)) if (is_mixed and len(meas_in) < plan.num_measurements) else _check_acc(norm_acc, m_arr)
    p_probs = [((_fp_div(w0, _fp_add(w0, w1)), _fp_div(w1, _fp_add(w0, w1))) if _fp_add(w0, w1)[0] > 0.0 else (0.0, 0.0)) for w0, w1 in zip(step_w[0], step_w[1])]
    return totals[0], totals[1], totals[2], acc, p_probs, m_arr, last_ket, faults, ctx.diagnostics

def _run_stream_trajectory(plan: PreparedQueryPlan, faults: dict[int, int], meas: dict[int, int], qmeas: dict[int, int], paulis: dict[int, int], resets: dict[int, int], sweep: dict[int, int],
                           rng_obj: np.random.Generator | None, *, sample_mc_faults: bool, use_sampling: bool, extract_data: bool, pctx: _EvalCtx | None = None,
                           skip: bool = False) -> tuple[TrajectoryRunResult | None, tuple[float, int], NDArray[np.bool_], dict[int, int], Any, bool, QueryDiagnostics]:
    f_in, fb_dg = (pctx.faults if (pctx is not None and sample_mc_faults) else faults), (pctx.diagnostics if pctx else QueryDiagnostics(used_fallback=True))
    mp, qp, rp, pp = (dict(pctx.m_map), dict(pctx.q_map), dict(pctx.r_map), dict(pctx.p_map)) if pctx else ({}, {}, {}, {})
    if skip or resets:
        return None, _FP_ZERO, np.empty(0, dtype=np.bool_), dict(f_in), (mp, qp, rp, pp), True, fb_dg
    r_gen, eff_m, eff_q, p_seed = (rng_obj if rng_obj is not None else np.random.default_rng(0)), {**meas, **mp}, {**qmeas, **qp}, {**paulis, **pp}
    f_tr = [(f_in[i] if i in f_in else _sample_categorical_branch([b.prob for b in ch.branches], r_gen)) if sample_mc_faults else f_in.get(i, 0) for i, ch in enumerate(plan.compiled_plan.fault_channels)]
    m_log, p_tr = ((-pctx.extra_cl_log if math.isfinite(pctx.extra_cl_log) else -math.inf) if pctx else 0.0), (None if (use_sampling and not p_seed) else {})
    for _, rec, aux in (plan._state["dyn"] if p_tr is not None else ()):
        if isinstance(rec, PauliNoiseRecord):
            p_tr[aux] = p_seed[aux] if aux in p_seed else (_sample_categorical_branch(rec.probs, r_gen) if use_sampling else 0)  # type: ignore[index]
        elif isinstance(rec, CorrelatedNoiseRecord):
            p_tr[aux[0]] = 0 if any(p_tr.get(j, 0) > 0 for j in aux[1]) else (p_seed[aux[0]] if aux[0] in p_seed else (int(float(r_gen.random()) < rec.prob) if use_sampling else 0))  # type: ignore[union-attr,index]
        elif isinstance(rec, HeraldedNoiseRecord):
            ph, req_b, hp = min(1.0, max(0.0, float(sum(rec.probs)))), bool(eff_m.get(rec.meas_index, 0)), [(i + 1, p) for i, p in enumerate(rec.probs) if p > 0.0]
            m_log += _ln(ph if req_b else max(0.0, 1.0 - ph)) if (aux not in paulis and rec.meas_index in meas) else 0.0
            c_req = hp[_sample_categorical_branch([p / ph for _, p in hp[:-1]], r_gen) - 1 if len(hp) > 1 else 0][0] if (req_b and ph > 0.0 and hp) else int(not req_b and 1.0 - ph <= 0.0)
            p_tr[aux] = p_seed[aux] if aux in p_seed else (c_req if rec.meas_index in eff_m else (_sample_categorical_branch(rec.probs, r_gen) if use_sampling else 0))  # type: ignore[index]
    sw_m = max((k for k in plan._state["root"].deps[2] if sweep.get(k, 0)), default=-1) if (sweep and plan._state["root"]) else -1
    kw: dict[str, Any] = dict(rng=r_gen, fault_trace=f_tr, pauli_noise_trace=p_tr, reset_outcomes=rp or None, requested_outcomes=eff_m or None,
                              requested_quantum_outcomes=eff_q or None, sweep_bits=([sweep.get(i, 0) for i in range(sw_m + 1)] if sw_m >= 0 else None), atol=plan.atol)
    try:
        tr = _mc.run_selector(plan.compiled_plan, extract_data=extract_data, **kw) if isinstance(plan.compiled_plan, SelectorPlan) else _mc.run_direct(plan.compiled_plan, **kw)
    except ImpossiblePostselectionError:
        return None, _FP_ZERO, np.empty(0, dtype=np.bool_), dict(enumerate(f_tr)), (mp, qp, rp, dict(p_tr) if p_tr is not None else pp), True, fb_dg
    pd = pctx.diagnostics if pctx is not None else None
    dg = (QueryDiagnostics(pd.cache_hits, pd.cache_misses, pd.node_evaluations, pd.invalidated_nodes, max(pd.peak_terms, tr.peak_components), max(pd.peak_dim, tr.peak_dim), pd.contraction_ops, True)
          if pd is not None else QueryDiagnostics(peak_terms=tr.peak_components, peak_dim=tr.peak_dim, used_fallback=True))
    lg2, bm_it, clip = ((tr.log_branch_mass + m_log) / math.log(2.0) if (math.isfinite(m_log) and math.isfinite(tr.log_branch_mass)) else -math.inf), iter(tr.born_masses), False
    for _, rec, aux in (() if math.isfinite(lg2) else plan._state["dyn"]):
        mi, ni = getattr(rec, "meas_index", None), (aux[0] if isinstance(rec, CorrelatedNoiseRecord) else (aux if isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord)) else None))
        cb, ph = (int(tr.pauli_noise_choices[ni]) if ni is not None else 0), (min(1.0, max(0.0, float(sum(rec.probs)))) if isinstance(rec, HeraldedNoiseRecord) else 0.0)
        req_b, p1 = bool(eff_m.get(mi, 0)), ((1.0 - rec.flip_prob if rec.pad_bit else rec.flip_prob) if isinstance(rec, MPadRecord) else 1.0)
        p_pad = (rec.flip_prob if req_b != bool(rec.pad_bit) else 1.0 - rec.flip_prob) if isinstance(rec, MPadRecord) else 1.0
        pr = (ph if req_b else max(0.0, 1.0 - ph)) if isinstance(rec, HeraldedNoiseRecord) else p_pad
        dead = mi in eff_m and (((cb > 0) != req_b or (p_tr is None and pr <= 1e-15)) if isinstance(rec, HeraldedNoiseRecord) else (p1 if req_b else 1.0 - p1) <= 1e-15)
        if (dead if isinstance(rec, (MPadRecord, HeraldedNoiseRecord)) else (ni is None and next(bm_it)[3] <= 0.0)):
            clip = ((ni not in p_seed) and pr > 0.0) if isinstance(rec, (MPadRecord, HeraldedNoiseRecord)) else (mi is not None and mi in eff_m and 0.0 < rec.flip_prob < 1.0)
            break
        if ni is not None:
            pp[ni] = cb
        if isinstance(rec, ResetRecord):
            rp[aux] = int(tr.reset_choices[aux])
        if mi is not None:
            mp[mi] = int(tr.measurements[mi])
            if isinstance(rec, (MeasurementRecord, ResetRecord)):
                qp[mi] = int(tr.quantum_outcomes[mi])
    if math.isfinite(lg2):
        return tr, _fp(2.0 ** (lg2 - math.floor(lg2)), math.floor(lg2)), tr.measurements, {}, None, False, dg
    m_out = np.array([bool((meas | mp).get(j, 0)) for j in range(plan.num_measurements)], dtype=np.bool_)
    return tr, _FP_ZERO, m_out, {i: int(c) for i, c in enumerate(tr.fault_choices)}, (mp, qp, rp, pp), clip, dg

def EvaluateQueries(plan: PreparedQueryPlan, realization: Mapping[Any, Any] | None = None, *, queries: Sequence[str] = ("mass",), conditional: bool = True, rng: np.random.Generator | int | None = None, acceptance_spec: Mapping[int, int | bool] | Callable[[NDArray[np.bool_]], bool] | None = None, target_spec: Any = None) -> QueryResult:
    """`Q2`: Evaluate global scalar-network queries against `plan` for a given `realization`."""
    if not isinstance(plan, PreparedQueryPlan):
        raise TypeError(f"Expected PreparedQueryPlan, got {type(plan)!r}")
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence) or not queries or any(not isinstance(q, str) or q not in _VALID_QUERIES for q in queries):
        raise ValueError(f"Invalid or unknown query metrics: {queries!r}")
    q_set, cond, comp_target = frozenset(queries), bool(conditional), (_compile_target_spec(target_spec, plan.output_qubits, plan.atol) if target_spec is not None else plan._state["target"])
    if "fidelity" in q_set and comp_target is None:
        raise ValueError("target_spec must be provided when querying 'fidelity'")
    norm_acc, (faults, meas, qmeas, paulis, resets, sweep) = _normalize_acceptance_spec(acceptance_spec if acceptance_spec is not None else plan.acceptance_spec, plan.num_measurements), _unpack_realization(plan, realization)
    sample_mc_faults, use_sampling = ((realization is None) and (rng is not None)), (("sample_measurements" in q_set) or ((realization is None) and (rng is not None)))
    rng_obj = (rng if isinstance(rng, np.random.Generator) else np.random.default_rng(_coerce_strict_int(rng, "rng seed"))) if rng is not None else (np.random.default_rng() if use_sampling else None)
    is_mixed = (not use_sampling) and any(
        (isinstance(r, PauliNoiseRecord) and ax not in paulis and sum(r.probs) > 0.0) or (isinstance(r, ResetRecord) and ax not in resets and (r.meas_index is None or r.meas_index not in qmeas))
        or (isinstance(r, CorrelatedNoiseRecord) and ax[0] not in paulis and r.prob > 0.0 and not any(paulis.get(j, 0) > 0 for j in ax[1]))
        or (isinstance(r, HeraldedNoiseRecord) and ax not in paulis and sum(r.probs) > 0.0 and meas.get(r.meas_index, 1) == 1)
        or (isinstance(r, (MeasurementRecord, ResetRecord, MPadRecord)) and r.meas_index is not None and ((r.meas_index not in meas and r.flip_prob not in (0.0, 1.0)) or (not isinstance(r, MPadRecord) and r.meas_index not in qmeas and not (isinstance(r, ResetRecord) and ax in resets) and (r.meas_index not in meas or r.flip_prob not in (0.0, 1.0))))) for _, r, ax in plan._state["dyn"])
    pack = lambda z_fp, a_fp, b_fp, ac, pp, ma, dg: QueryResult(
        _ldexp(*z_fp) if "mass" in q_set else None, ((_fp_div(a_fp, z_fp) if cond else _ldexp(*a_fp)) if "acceptance" in q_set else None),
        ((_fp_div(b_fp, z_fp) if cond else _ldexp(*b_fp)) if "fidelity" in q_set else None), pp if "prefix_probs" in q_set else None,
        ma.copy() if ("sample_measurements" in q_set and ma is not None) else None, bool(ac) if "acceptance" in q_set else None, dg)
    c_box: list[_EvalCtx] = []
    try:
        z_fp, a_fp, b_fp, acc, p_probs, m_arr, _, _, diag = _eval_via_tree(plan, faults, meas, qmeas, paulis, resets, sweep, q_set, norm_acc, comp_target, rng_obj, sample_mc_faults=sample_mc_faults, is_mixed=is_mixed, out_ctx=c_box)
        return pack(z_fp, a_fp, b_fp, acc, p_probs, None if is_mixed else m_arr, diag)
    except QueryBudgetExceededError:
        if plan.budget.on_budget_exceeded == "raise":
            raise
        tr, z_fp, m_out, f_pin, pin, clip, diag = _run_stream_trajectory(plan, faults, meas, qmeas, paulis, resets, sweep, rng_obj, sample_mc_faults=sample_mc_faults, use_sampling=use_sampling, extract_data=True, pctx=c_box[0] if c_box else None, skip=is_mixed)
        if tr is None or (z_fp[0] <= 0.0 and (clip or "prefix_probs" in q_set)):
            z_fp, a_fp, b_fp, acc, p_probs, m_arr, _, _, diag = _eval_via_tree(plan, f_pin, meas, qmeas, paulis, resets, sweep, q_set, norm_acc, comp_target, rng_obj, sample_mc_faults=sample_mc_faults, is_mixed=is_mixed, stream_only=True, prev_diag=diag, pin=pin)
            return pack(z_fp, a_fp, b_fp, acc, p_probs, None if is_mixed else m_arr, diag)
        acc, f_ctx = _check_acc(norm_acc, m_out), _EvalCtx(plan.budget, plan.atol, len(plan._state["q_act"]), faults, meas, sweep, enforce_budget=False, prev_diag=diag)
        b_fp = _fp_mul(z_fp, *_compute_partial_trace_fidelity_mass(tr.final_tableau, tr.expansion, 0, comp_target, plan.output_qubits, f_ctx, plan._state["bcache"])) if ("fidelity" in q_set and acc and z_fp[0] > 0.0 and comp_target is not None) else _FP_ZERO
        p_probs_fb: list[tuple[float, float]] | None = [(0.0, 0.0)] * plan.num_measurements if "prefix_probs" in q_set else None
        bm = {m_i: (z0, z1, zin) for m_i, z0, z1, zin in tr.born_masses if m_i is not None} if p_probs_fb is not None else {}
        for _, rec, aux in (plan._state["dyn"] if p_probs_fb is not None else ()):
            if isinstance(rec, (MeasurementRecord, ResetRecord)) and rec.meas_index is not None and sum(bm.get(rec.meas_index, (0.0, 0.0, 0.0))[:2]) > 0.0 and bm[rec.meas_index][2] > 0.0:
                q0, fp = (bm[rec.meas_index][1 if rec.invert_result else 0]) / sum(bm[rec.meas_index][:2]), rec.flip_prob
                p_probs_fb[rec.meas_index] = (max(0.0, min(1.0, (1.0 - fp) * q0 + fp * (1.0 - q0))), max(0.0, min(1.0, (1.0 - fp) * (1.0 - q0) + fp * q0)))
            elif isinstance(rec, (MPadRecord, HeraldedNoiseRecord)):
                ph_r = min(1.0, max(0.0, sum(rec.probs))) if isinstance(rec, HeraldedNoiseRecord) else 0.0
                p_probs_fb[rec.meas_index] = (((rec.flip_prob, 1.0 - rec.flip_prob) if rec.pad_bit else (1.0 - rec.flip_prob, rec.flip_prob)) if isinstance(rec, MPadRecord)
                                              else ((1.0 - float(paulis[aux] > 0), float(paulis[aux] > 0)) if aux in paulis else (max(0.0, 1.0 - ph_r), ph_r)))
        return pack(z_fp, z_fp if acc else _FP_ZERO, b_fp, acc, p_probs_fb, m_out, f_ctx.diagnostics)

prepare_queries, evaluate_queries = PrepareQueries, EvaluateQueries

def run_query_shot(plan: PreparedQueryPlan, *, realization: Mapping[Any, Any] | None = None, sweep_bits: Mapping[int, int | bool] | Sequence[int | bool] | NDArray[np.bool_] | None = None, rng: np.random.Generator | int | None = None, extract_data: bool = False) -> QueryShotTrajectory:
    """Execute one causal shot via `plan`'s cached segment tree (or `STREAM` fallback)."""
    faults, meas, qmeas, paulis, resets, r_sweep = _unpack_realization(plan, realization)
    sweep, cp = (_norm_sweep(sweep_bits) if sweep_bits is not None else r_sweep), plan.compiled_plan
    rng_obj, c_box = (rng if isinstance(rng, np.random.Generator) else np.random.default_rng(None if rng is None else _coerce_strict_int(rng, "rng seed"))), []
    def _pack_shot(zf: tuple[float, int], ma: NDArray[np.bool_], kf: _ScaledExpansion, ef: dict[int, int], dg: QueryDiagnostics) -> QueryShotTrajectory:
        sb = _selector_bits(cp, ef) if isinstance(cp, SelectorPlan) else None
        ft, oe = (extract_data_expansion(cp, kf.exp, sb) if (extract_data and isinstance(cp, SelectorPlan) and sb is not None) else (clone_reference_chp(cp.final_tableau), kf.exp.copy()))
        if oe.chi > 0 and oe.norm_sq() > 0.0:
            oe.weights = np.ascontiguousarray(oe.weights / math.sqrt(oe.norm_sq()))
        return QueryShotTrajectory(ma, _ldexp(*zf), oe, ft, sb, dg)
    try:
        z_fp, _, _, _, _, m_arr, ket_f, eff_f, diag = _eval_via_tree(plan, faults, meas, qmeas, paulis, resets, sweep, frozenset({"mass", "sample_measurements"}), ("all", None), None, rng_obj, sample_mc_faults=True, is_mixed=False, out_ctx=c_box)
        return _pack_shot(z_fp, m_arr, ket_f, eff_f, diag)
    except QueryBudgetExceededError:
        if plan.budget.on_budget_exceeded == "raise":
            raise
        tr, z_fp, m_out, f_pin, pin, clip, diag = _run_stream_trajectory(plan, faults, meas, qmeas, paulis, resets, sweep, rng_obj, sample_mc_faults=True, use_sampling=True, extract_data=extract_data, pctx=c_box[0] if c_box else None)
        if tr is None or (z_fp[0] <= 0.0 and clip):
            z_fp, _, _, _, _, m_arr, ket_f, eff_f, diag = _eval_via_tree(plan, f_pin, meas, qmeas, paulis, resets, sweep, frozenset({"mass", "sample_measurements"}), ("all", None), None, rng_obj, sample_mc_faults=True, is_mixed=False, stream_only=True, prev_diag=diag, pin=pin)
            return _pack_shot(z_fp, m_arr, ket_f, eff_f, diag)
        return QueryShotTrajectory(m_out, _ldexp(*z_fp), tr.expansion, tr.final_tableau, tr.selector_bits, diag) if z_fp[0] > 0.0 else _pack_shot(_FP_ZERO, m_out, _empty_se(plan.num_total_qubits, plan.atol), f_pin, diag)

def run_query_batch(plan: PreparedQueryPlan, batch_size: int, *, rng: np.random.Generator | int | None = None, fault_realizations: Sequence[Mapping[Any, Any] | None] | Mapping[Any, Any] | None = None, sweep_bits: Any = None, extract_data: bool = False, retain_states: bool = True) -> tuple[NDArray[np.bool_], list[QueryShotTrajectory]]:
    """Execute `batch_size` shots reusing `plan`'s cached segment tree across the batch."""
    if not isinstance(plan, PreparedQueryPlan):
        raise TypeError(f"Expected PreparedQueryPlan, got {type(plan)!r}")
    b_sz, is_seq = _coerce_strict_int(batch_size, "batch_size"), (lambda x: isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray, memoryview)))
    f_bcast = fault_realizations is None or isinstance(fault_realizations, Mapping)
    f_seq = [fault_realizations] * b_sz if f_bcast else (fault_realizations if (is_seq(fault_realizations) and len(fault_realizations) == b_sz) else None)
    sw_1d = isinstance(sweep_bits, Mapping) or (isinstance(sweep_bits, np.ndarray) and sweep_bits.ndim == 1) or (is_seq(sweep_bits) and (len(sweep_bits) == 0 or isinstance(sweep_bits[0], (int, np.integer, bool, np.bool_))))
    s_raw = [None] * b_sz if sweep_bits is None else ([_norm_sweep(sweep_bits)] * b_sz if sw_1d else (sweep_bits if ((isinstance(sweep_bits, np.ndarray) and sweep_bits.ndim == 2 and sweep_bits.shape[0] == b_sz) or (is_seq(sweep_bits) and len(sweep_bits) == b_sz)) else None))
    if b_sz < 1 or f_seq is None or s_raw is None:
        raise ValueError(f"Invalid batch_size={b_sz}, fault_realizations, or sweep_bits for run_query_batch")
    s_seq, _ = (s_raw if (sweep_bits is None or sw_1d) else [None if sr is None else _norm_sweep(sr) for sr in s_raw]), [_unpack_realization(plan, r) for r in ((fault_realizations,) if f_bcast else f_seq)]
    rng_obj, records, trajectories = (rng if isinstance(rng, np.random.Generator) else np.random.default_rng(None if rng is None else _coerce_strict_int(rng, "rng seed"))), np.zeros((b_sz, plan.num_measurements), dtype=np.bool_), []
    for s in range(b_sz):
        traj = run_query_shot(plan, realization=f_seq[s], sweep_bits=s_seq[s], rng=rng_obj, extract_data=extract_data)
        records[s] = traj.measurements
        trajectories.append(traj) if retain_states else None
    return records, trajectories
