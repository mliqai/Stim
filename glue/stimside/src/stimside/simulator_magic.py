"""MagicsideSimulator: non-Clifford Phase-QFE circuit simulator for stimside."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic.channels import (
    FEEDBACK_2Q_GATES, PAULI_2Q_ORDER, CompiledCliffordFactorList, CompiledOpenBlockRecord,
    CorrelatedNoiseRecord, FrameEvent, HeraldedNoiseRecord, LeftSppEvent, LeftUnitaryEvent,
    MagicTermRecord, MeasurementRecord, PauliNoiseRecord, ResetRecord, apply_clifford_factors,
    compile_clifford_factors, compile_pauli_measurement_c15, compile_pauli_product, compile_reset_c15,
    emit_magic_terms, infer_total_data_qubits, normalize_choice_map, normalize_pauli_slot_map,
    sample_categorical_branch, sample_heralded_pauli_branch,
    split_pauli_product_targets,
)
from stimside._magic.gates import (
    CliffordFaultChannel, CompiledPauliAxis, canonical_clifford_gate_name_and_arity,
    canonical_gate_data, coerce_strict_int, extract_circuit_faults,
    merge_and_validate_fault_channels, parse_clifford_fault_instruction,
)
from stimside._magic.plan import DirectPlan, SelectorPlan, compile_direct, compile_open_block, compile_selector
from stimside._magic.statevector import extract_data_expansion, reconstruct_statevector
from stimside._magic.trajectory import prepare_run_maps, run_direct, run_selector
from stimside._magic.phase_qfe import (
    DEFAULT_ATOL, ImpossiblePostselectionError, K1_CANCEL_REL_TOL, ShotExpansion, apply_compiled_block, apply_pauli,
    apply_quarter_turn, measure_expansion, measure_expansions_batch, project_diagonal,
    validate_atol,
)
from stimside._magic.query import (
    PrepareQueries, PreparedQueryPlan, QueryBudget, QueryBudgetExceededError, run_query_batch,
    _normalize_realization,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler, _TrivialCompiledOpHandler
from stimside.op_handlers.seam import SyncRngs
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic, _strip_circuit_for_noiseless_ref
from stimside.op_handlers.leakage_handlers.tag_registry import MAGIC_TAGS as _MAGIC_TAGS, validate_magic_op
from stimside.util.bitwords import (
    first_set_bit_in_words,
    unit_bit_words,
    word_dot_parity,
    words_any_nonzero,
)
from stimside.util.circuit import (
    apply_coords_op,
    check_replay,
    format_detector_flips,
    pack_shot_flips,
    reduce_hermitian_pauli_product,
    stack_measurement_columns,
    unroll_circuit,
)
from stimside.util.leakage_events import LeakageEvent, LeakageEventsRecorderMixin
from stimside.util.reference_chp import ReferenceCHPTableau
from stimside.util.seeding import derived_seed
from stimside.util.sync_rng import (
    apply_sync_meas_flips,
    build_sync_rngs,
    conditioned_targets_per_shot,
    draw_tab_rng_bit,
    exec_sync_correlated_error,
    exec_sync_heralded_noise,
    exec_sync_pauli_batch,
)
from stimside.util.unleaked_to_leaked import UnleakedToLeakedRecordsMixin

FAULT_MODES: dict[str, Literal["DIRECT", "SELECTOR"]] = {"DIRECT": "DIRECT", "SELECTOR": "SELECTOR"}
BACKENDS: dict[str, Literal["STREAM", "QUERY"]] = {"STREAM": "STREAM", "QUERY": "QUERY"}
M2D_MODES: dict[str, Literal["auto", "magicside", "stim", "zero"]] = {
    "auto": "auto", "magicside": "magicside", "magic": "magicside", "noiseless_magic": "magicside",
    "stim": "stim", "clifford": "stim", "zero": "zero", "skip": "zero",
}


def resolve_budget(
    query_budget: QueryBudget | None, max_components: int | None, on_budget_exceeded: str | None
) -> tuple[QueryBudget | None, int | None, Literal["raise", "fallback"]]:
    if query_budget is not None and not isinstance(query_budget, QueryBudget):
        raise TypeError(f"Expected QueryBudget or None, got {type(query_budget).__name__}")
    mc = None if max_components is None else coerce_strict_int(max_components, "max_components")
    if mc is not None and mc < 1:
        raise ValueError(f"max_components must be >= 1, got {mc}")
    if query_budget is None and mc is None and on_budget_exceeded is None:
        return None, None, "raise"
    base = QueryBudget() if query_budget is None else query_budget
    mt = (mc if query_budget is None else min(base.max_terms, mc)) if mc is not None else base.max_terms
    obe = base.on_budget_exceeded if on_budget_exceeded is None else on_budget_exceeded
    eff_qb = dataclasses.replace(base, max_terms=mt, on_budget_exceeded=obe)
    return eff_qb, mc, eff_qb.on_budget_exceeded  # type: ignore[return-value]


def _normalize_shot_realization(
    raw: Any, fault_max: Sequence[int], fault_id_to_slot: Mapping[int, int], num_measurements: int
) -> dict[str, Any]:
    sec = _normalize_realization(
        raw, fault_max, fault_id_to_slot, num_measurements, strict_faults=True
    )
    if "sweep_bits" in sec:
        sw = sec["sweep_bits"]
        w = (max(coerce_strict_int(k, "sweep_bits key") for k in sw) + 1) if isinstance(sw, Mapping) and sw else (len(sw) if not isinstance(sw, (str, bytes, Mapping)) and hasattr(sw, "__len__") else 0)
        sw_m = normalize_choice_map(sw, (1,) * max(0, w), "sweep_bits", allow_bool_values=True) or {}
        sec["sweep_bits"] = np.array([bool(sw_m.get(i, 0)) for i in range(max(sw_m) + 1)], dtype=np.bool_) if sw_m else np.zeros(0, dtype=np.bool_)
    return sec


def _normalize_stream_paulis(plan: DirectPlan | SelectorPlan, raw_p: Any) -> Any:
    if not isinstance(raw_p, Mapping):
        return raw_p
    kinds = {
        rec.noise_index: getattr(rec, "kind", "")
        for rec in plan.records
        if isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord, CorrelatedNoiseRecord))
    }
    return normalize_pauli_slot_map(raw_p, plan.noise_channel_max_choices, kinds)


def _log0(x: Any) -> Any:
    """Natural log with log(0) = -inf (no warning): branch masses are stored as logs."""
    with np.errstate(divide="ignore"):
        return np.log(x)


def parse_enum(val: Any, table: Mapping[str, Any], name: str, *, lower: bool = False) -> Any:
    key = (val.strip().lower() if lower else val.strip().upper()) if isinstance(val, str) else ""
    if key not in table:
        raise ValueError(f"Unsupported {name}: {val!r}")
    return table[key]


def _normalize_sweep_matrix(sweep_bits: Any, batch_size: int) -> NDArray[np.bool_] | None:
    if sweep_bits is None:
        return None
    if isinstance(sweep_bits, Mapping):
        w = (max(int(k) for k in sweep_bits) + 1) if sweep_bits else 0
        row = np.zeros(w, dtype=np.bool_)
        for k, v in sweep_bits.items():
            row[int(k)] = bool(v)
        return np.broadcast_to(row, (batch_size, w)).copy()
    arr = np.asarray(sweep_bits, dtype=np.bool_)
    if arr.ndim == 1:
        return np.broadcast_to(arr, (batch_size, arr.shape[0])).copy()
    if arr.ndim == 2 and arr.shape[0] == batch_size:
        return arr.copy()
    raise ValueError(f"Invalid sweep_bits shape {arr.shape} for batch_size={batch_size}")


class MagicsideSimulator(UnleakedToLeakedRecordsMixin, LeakageEventsRecorderMixin):
    """Non-Clifford Phase-QFE simulator compatible with CosetsideSimulator APIs."""

    def __init__(
        self, circuit: stim.Circuit, compiled_op_handler: CompiledOpHandler[Any] | None = None, *,
        op_handler: OpHandler[Any] | None = None, batch_size: int = 1, fault_mode: str = "DIRECT",
        backend: str = "STREAM", seed: int | None = None, sweep_bits: Any = None,
        fault_channels: Sequence[CliffordFaultChannel] | None = None, fault_realizations: Any = None,
        query_budget: QueryBudget | None = None, max_components: int | None = None,
        on_budget_exceeded: str | None = None, m2d_reference_mode: str = "auto",
        reference_sample: Sequence[int | bool] | NDArray[np.bool_] | None = None,
        record_unleaked_to_leaked: bool = False, record_leakage_events: bool = False,
        sync_tableside_rng: bool | None = False, atol: float = DEFAULT_ATOL,
    ) -> None:
        if not isinstance(circuit, stim.Circuit):
            raise TypeError(f"Expected stim.Circuit, got {type(circuit)!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.circuit, self.batch_size = circuit, int(batch_size)
        self.fault_mode: Literal["DIRECT", "SELECTOR"] = parse_enum(fault_mode, FAULT_MODES, "fault_mode")
        self.backend: Literal["STREAM", "QUERY"] = parse_enum(backend, BACKENDS, "backend")
        self.atol = validate_atol(atol)
        self.m2d_reference_mode: Literal["auto", "magicside", "stim", "zero"] = parse_enum(m2d_reference_mode, M2D_MODES, "m2d_reference_mode", lower=True)
        self.record_unleaked_to_leaked = bool(record_unleaked_to_leaked)
        self.record_leakage_events = bool(record_leakage_events)
        self.sync_tableside_rng = bool(sync_tableside_rng)
        if seed is not None:
            derived_seed(seed, 0)
        self.seed = seed
        self.query_budget, self.max_components, self.on_budget_exceeded = resolve_budget(
            query_budget, max_components, on_budget_exceeded
        )
        self._explicit_ref_sample: NDArray[np.bool_] | None = None
        if reference_sample is not None:
            ref_arr = np.asarray(reference_sample, dtype=np.bool_)
            if ref_arr.ndim != 1 or ref_arr.shape[0] != circuit.num_measurements:
                raise ValueError(f"reference_sample length {ref_arr.shape} != {circuit.num_measurements}")
            self._explicit_ref_sample = ref_arr.copy()
        self.num_qubits = int(circuit.num_qubits)
        self._unrolled_ops: list[stim.CircuitInstruction] = unroll_circuit(circuit)
        self._eff_m2d_mode: Literal["magicside", "stim", "zero"] = ("magicside" if any(op.tag in _MAGIC_TAGS for op in self._unrolled_ops) else "stim") if self.m2d_reference_mode == "auto" else self.m2d_reference_mode
        flat_ops, inline_faults = extract_circuit_faults(circuit)
        self._user_fault_channels = tuple(fault_channels) if fault_channels is not None else None
        self._prog_faults, self._inline_faults, self._all_fault_channels = merge_and_validate_fault_channels(len(flat_ops), inline_faults, fault_channels)
        self._logical_nq = max([self.num_qubits] + [q + 1 for ch in self._all_fault_channels for b in ch.branches for q in b.targets])
        self._eff_nq = infer_total_data_qubits(circuit, self._all_fault_channels)
        self._tab_scratch_q = self.num_qubits
        self._sweep_mat = _normalize_sweep_matrix(sweep_bits, self.batch_size)
        if compiled_op_handler is None and op_handler is not None:
            compiled_op_handler = op_handler.compile_op_handler(circuit=circuit, batch_size=self.batch_size)
        self._compiled_op_handler: CompiledOpHandler[Any] = _TrivialCompiledOpHandler() if compiled_op_handler is None else compiled_op_handler
        self._has_trivial_handler = isinstance(self._compiled_op_handler, _TrivialCompiledOpHandler)
        self.compiled_op_handler = self._compiled_op_handler
        if (self.backend == "QUERY" or self.fault_mode == "SELECTOR") and (not self._has_trivial_handler or self.sync_tableside_rng):
            raise ValueError(f"backend={self.backend!r}, fault_mode={self.fault_mode!r} is incompatible with non-trivial op_handler or sync_tableside_rng=True")
        self._norm_reals: list[dict[str, Any]] | None = None
        self._has_non_fault_reals = False
        if fault_realizations is not None:
            if not isinstance(fault_realizations, Mapping) and (isinstance(fault_realizations, (str, bytes)) or not isinstance(fault_realizations, Sequence) or len(fault_realizations) != self.batch_size):
                raise ValueError("fault_realizations must be a Mapping or Sequence of length batch_size")
            f_max = [len(ch.branches) for ch in self._all_fault_channels]
            f_id_map = {int(ch.slot_id): i for i, ch in enumerate(self._all_fault_channels)}  # type: ignore[arg-type]
            raw_seq = [fault_realizations] if isinstance(fault_realizations, Mapping) else fault_realizations
            parsed = [_normalize_shot_realization(r, f_max, f_id_map, int(circuit.num_measurements)) for r in raw_seq]
            self._norm_reals = parsed * self.batch_size if isinstance(fault_realizations, Mapping) else parsed
            if not any(parsed):  # all-empty realizations must behave exactly like None
                self._norm_reals = None
            self._has_non_fault_reals = any(bool(set(nr) - {"faults"}) for nr in parsed)
            if self._has_non_fault_reals and (not self._has_trivial_handler or self.sync_tableside_rng):
                raise ValueError("Non-fault keys in fault_realizations require trivial op_handler and sync_tableside_rng=False")
            if any("sweep_bits" in nr for nr in parsed):
                rows = [nr["sweep_bits"] if "sweep_bits" in nr else (self._sweep_mat[s] if self._sweep_mat is not None else ()) for s, nr in enumerate(self._norm_reals)]
                self._sweep_mat = np.zeros((self.batch_size, max(len(r) for r in rows)), dtype=np.bool_)
                for s, r in enumerate(rows):
                    self._sweep_mat[s, : len(r)] = r
        self._batches_completed = 0
        self._compiled_stream_plan: DirectPlan | SelectorPlan | None = None
        self._compiled_query_plan: PreparedQueryPlan | None = None
        self._ref_sample_cache: dict[bytes, NDArray[np.bool_]] = {}
        self._m2d_skip_conv: Any = None
        self._m2d_stim_conv: Any = None
        self._measurement_columns: list[NDArray[np.bool_]] = []
        self._unleaked_to_leaked_events: list[list[int]] = []
        self._leakage_events: list[list[LeakageEvent]] = []
        self.clear()

    def _init_rngs(self) -> None:
        if not self.sync_tableside_rng:
            self._shot_np_rngs: list[np.random.Generator] | None = None
            self._tab_rngs: list[stim.TableauSimulator] | None = None
            self._tab_rngs_x: list[stim.TableauSimulator] | None = None
            self.np_rng = np.random.default_rng(seed=None if self.seed is None else derived_seed(self.seed, self._batches_completed))
            return
        _, self._shot_np_rngs, self._tab_rngs, self._tab_rngs_x = build_sync_rngs(
            self.seed, self._batches_completed, self.batch_size, self.num_qubits
        )
        self.np_rng = self._shot_np_rngs[0]

    def _draw_tab_rng_bit(self, b_idx: int) -> bool:
        assert self._tab_rngs is not None and self._tab_rngs_x is not None
        return draw_tab_rng_bit(self._tab_rngs[b_idx], self._tab_rngs_x[b_idx], self._tab_scratch_q)

    def _sample_pauli_noise_via_tab_rng(self, op_name: str, targets_per_shot: Sequence[Sequence[int]], gate_args: Sequence[float]) -> None:
        assert self._tab_rngs is not None and self._tab_rngs_x is not None
        xm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        zm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        exec_sync_pauli_batch(self._tab_rngs, self._tab_rngs_x, op_name, targets_per_shot, gate_args, xm, zm)
        self._apply_pauli_mask("X", xm)
        self._apply_pauli_mask("Z", zm)

    sample_pauli_noise_sync = _sample_pauli_noise_via_tab_rng

    def sync_rngs(self) -> SyncRngs | None:
        """The per-shot sync RNGs, or None unless ``sync_tableside_rng=True``."""
        if self._tab_rngs is None:
            return None
        assert self._tab_rngs_x is not None and self._shot_np_rngs is not None
        return SyncRngs(self._tab_rngs, self._tab_rngs_x, self._shot_np_rngs)

    def measurement_column(self, index: int) -> NDArray[np.bool_]:
        """Measurement record column ``index`` (one bit per shot). Do not modify it in place."""
        return self._measurement_columns[index]

    def overwrite_measurement_column(self, index: int, values: NDArray[np.bool_]) -> None:
        """Overwrite measurement record column ``index`` in place."""
        self._measurement_columns[index][:] = values

    def clear(self) -> None:
        self._measurement_columns.clear()
        self._cached_m2d: tuple[NDArray[np.bool_], NDArray[np.bool_]] | None = None
        self.qubit_coords: dict[int, list[float]] = {}
        self.qubit_tags: dict[int, str] = {}
        self.coords_shifts: list[float] = []
        self._init_rngs()
        self._compiled_op_handler.clear()
        if self.record_unleaked_to_leaked:
            self._unleaked_to_leaked_events = [[] for _ in range(self.batch_size)]
        if self.record_leakage_events:
            self._leakage_events = [[] for _ in range(self.batch_size)]
        self._circuit_time = 0
        self._used_interactively = False
        self._finished_running_circuit = False
        self._live_ref_tab = ReferenceCHPTableau(self._eff_nq)
        self._live_frame_events: list[FrameEvent] = []
        self._live_exps: list[ShotExpansion] = [ShotExpansion.zero(self._eff_nq, atol=self.atol) for _ in range(self.batch_size)]
        self._shot_tabs: list[ReferenceCHPTableau] | None = None
        self._active_plan: DirectPlan | SelectorPlan | None = None
        self._shot_sel_bits: list[NDArray[np.bool_] | None] | None = None
        self._log_branch_masses = np.zeros(self.batch_size, dtype=np.float64)
        self._pending_magic: list[MagicTermRecord] = []
        self._corr_occurred = np.zeros(self.batch_size, dtype=np.bool_)

    @property
    def num_measurements(self) -> int:
        return int(self.circuit.num_measurements)

    @property
    def num_detectors(self) -> int:
        return int(self.circuit.num_detectors)

    @property
    def num_observables(self) -> int:
        return int(self.circuit.num_observables)

    def num_qubits_in_new_circuit(self) -> int:
        return self.num_qubits

    def get_current_circuit_time(self) -> int:
        return self._circuit_time

    def run(self) -> list[np.ndarray] | None:
        if self._used_interactively or self._circuit_time != 0 or self._finished_running_circuit:
            raise ValueError("MagicsideSimulator is not clean; call .clear() before .run().")
        can_use_compiled = self._has_trivial_handler and not self.sync_tableside_rng
        if self.backend == "QUERY":
            self._run_compiled_query()
        elif can_use_compiled and (self.fault_mode == "SELECTOR" or self._norm_reals is not None):
            self._run_compiled_stream()
        else:
            self._do(self.circuit)
            if len(self._unrolled_ops) == 0 and 0 in self._prog_faults:
                for slot_idx, ch in self._prog_faults[0]:
                    self._fire_fault_channel_live(slot_idx, ch)
            self._flush_magic()
        self._finished_running_circuit = True
        self._batches_completed += 1
        return self.get_unleaked_to_leaked_records() if self.record_unleaked_to_leaked else None

    def _finish_compiled_batch(
        self, cp: DirectPlan | SelectorPlan, recs: NDArray[np.bool_], tabs: list[ReferenceCHPTableau],
        exps: list[ShotExpansion], sel_bits: list[NDArray[np.bool_] | None],
    ) -> None:
        self._measurement_columns = [recs[:, j].copy() for j in range(recs.shape[1])]
        self._active_plan, self._shot_sel_bits = cp, sel_bits
        self._live_frame_events = list(cp.frame_events)
        self._shot_tabs = tabs
        self._live_ref_tab = tabs[0].copy()
        self._live_exps = exps
        for op in self._unrolled_ops:
            apply_coords_op(op, self.qubit_coords, self.qubit_tags, self.coords_shifts)
        self._circuit_time = len(self._unrolled_ops)

    def _ensure_query_plan(self) -> PreparedQueryPlan:
        if self._compiled_query_plan is None:
            self._compiled_query_plan = PrepareQueries(
                self.circuit, fault_mode=self.fault_mode, fault_channels=self._user_fault_channels,
                budget=self.query_budget, atol=self.atol,
            )
        return self._compiled_query_plan

    def _run_compiled_query(self) -> None:
        qplan = self._ensure_query_plan()
        q_reals = (
            [{k: ({int(self._all_fault_channels[i].slot_id): v for i, v in val.items()} if k == "faults" else val) for k, val in nr.items()} for nr in self._norm_reals]  # type: ignore[arg-type]
            if self._norm_reals is not None else None
        )
        recs, trajs = run_query_batch(
            qplan, self.batch_size, rng=self.np_rng, fault_realizations=q_reals,
            sweep_bits=self._sweep_mat, extract_data=False,
        )
        self._log_branch_masses = _log0(np.array([t.branch_mass for t in trajs], dtype=np.float64))
        self._finish_compiled_batch(
            qplan.compiled_plan, recs, [t.final_tableau for t in trajs],  # type: ignore[misc]
            [t.expansion for t in trajs], [t.selector_bits for t in trajs],  # type: ignore[misc]
        )

    def _presample_stream_maps(
        self, plan: DirectPlan | SelectorPlan, f_map: dict[int, int] | None, n_map: dict[int, int] | None, c_map: Mapping[int, int]
    ) -> tuple[dict[int, int] | None, dict[int, int] | None, float]:
        rng = self.np_rng
        f_arg = None if f_map is None else {
            int(ch.slot_id): (f_map[i] if i in f_map else sample_categorical_branch([b.prob for b in ch.branches], rng))  # type: ignore[arg-type]
            for i, ch in enumerate(plan.fault_channels)
        }
        if n_map is None:
            return f_arg, None, 1.0
        n_arg, corr_occ, mass_scale = dict(n_map), False, 1.0
        for rec in plan.records:
            if not isinstance(rec, (PauliNoiseRecord, HeraldedNoiseRecord, CorrelatedNoiseRecord)):
                continue
            idx = rec.noise_index
            if isinstance(rec, PauliNoiseRecord):
                if idx not in n_arg:
                    n_arg[idx] = sample_categorical_branch(rec.probs, rng)
            elif isinstance(rec, HeraldedNoiseRecord):
                if idx not in n_arg:
                    n_arg[idx], p_step = sample_heralded_pauli_branch(
                        rec.probs, c_map.get(rec.meas_index), rng, presampled=True
                    )
                    mass_scale *= p_step
            else:
                if not rec.is_else:
                    corr_occ = False
                c_c = 0 if corr_occ else (n_arg[idx] if idx in n_arg else int(float(rng.random()) < rec.prob))
                corr_occ = corr_occ or (c_c > 0)
                n_arg[idx] = c_c
        return f_arg, n_arg, mass_scale

    def _run_compiled_stream(self) -> None:
        if self._compiled_stream_plan is None:
            fc = self._user_fault_channels
            self._compiled_stream_plan = compile_selector(self.circuit, fault_channels=fc) if self.fault_mode == "SELECTOR" else compile_direct(self.circuit, fault_channels=fc)
        plan = self._compiled_stream_plan
        mc_cap = self.max_components if self.on_budget_exceeded == "raise" else None
        recs = np.zeros((self.batch_size, plan.num_measurements), dtype=np.bool_)
        exps, tabs, sel_bits = [], [], []
        for s in range(self.batch_size):
            nr = self._norm_reals[s] if self._norm_reals is not None else {}
            f_raw = {int(plan.fault_channels[i].slot_id): v for i, v in nr["faults"].items()} if nr.get("faults") is not None else None  # type: ignore[arg-type]
            f_map, n_map, r_map, c_map, q_map = prepare_run_maps(
                plan, fault_trace=f_raw, pauli_noise_trace=_normalize_stream_paulis(plan, nr.get("paulis")),
                reset_outcomes=nr.get("resets"), requested_outcomes=nr.get("measurements"),
                requested_quantum_outcomes=nr.get("quantum_measurements"),
            )
            f_arg, n_arg, mass_scale = self._presample_stream_maps(plan, f_map, n_map, c_map)
            sw_row: Any = self._sweep_mat[s] if self._sweep_mat is not None else nr.get("sweep_bits")
            kw: dict[str, Any] = dict(
                rng=self.np_rng, fault_trace=f_arg, pauli_noise_trace=n_arg, reset_outcomes=r_map or None,
                requested_outcomes=c_map or None, requested_quantum_outcomes=q_map or None, sweep_bits=sw_row,
                max_components=mc_cap, atol=self.atol,
            )
            try:
                res = run_selector(plan, extract_data=False, **kw) if isinstance(plan, SelectorPlan) else run_direct(plan, **kw)
            except ImpossiblePostselectionError:
                recs[s] = [bool(c_map.get(j, 0)) for j in range(plan.num_measurements)]
                self._log_branch_masses[s] = -math.inf
                exps.append(ShotExpansion.from_components((), n_qubits=plan.final_tableau.num_qubits, merge=False, atol=self.atol))
                tabs.append(plan.final_tableau.copy())
                sel_bits.append(np.zeros(plan.num_selector_qubits, dtype=np.bool_) if isinstance(plan, SelectorPlan) else None)
            except RuntimeError as exc:
                if mc_cap is not None and "max_components" in str(exc):
                    raise QueryBudgetExceededError(str(exc)) from exc
                raise
            else:
                recs[s] = res.measurements
                self._log_branch_masses[s] = res.log_branch_mass + _log0(mass_scale)
                exps.append(res.expansion)
                tabs.append(res.final_tableau)
                sel_bits.append(res.selector_bits)
        self._finish_compiled_batch(plan, recs, tabs, exps, sel_bits)

    def _check_interactive_mode(self) -> None:
        if self.backend == "QUERY" or self.fault_mode == "SELECTOR":
            raise ValueError(f"Interactive execution requires backend='STREAM' and fault_mode='DIRECT' (got backend={self.backend!r}, fault_mode={self.fault_mode!r})")
        if self._has_non_fault_reals:
            raise ValueError("Interactive execution only supports fault-slot keys in fault_realizations")

    def interactive_do(self, this: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock, /) -> None:
        self._check_interactive_mode()
        if self._finished_running_circuit:
            raise RuntimeError("Cannot call interactive_do after finish_interactive_run().")
        self._used_interactively = True
        self._do(this)

    def finish_interactive_run(self) -> None:
        self._check_interactive_mode()
        if not self._finished_running_circuit:
            if self._circuit_time == 0 and len(self._unrolled_ops) == 0 and 0 in self._prog_faults:
                for slot_idx, ch in self._prog_faults[0]:
                    self._fire_fault_channel_live(slot_idx, ch)
            self._flush_magic()
            self._finished_running_circuit = True
            self._batches_completed += 1

    def _flush_magic(self) -> None:
        if not self._pending_magic:
            return
        recs = compile_open_block(self._pending_magic, n_qubits=self._eff_nq)
        self._pending_magic.clear()
        mc_cap = self.max_components if self.on_budget_exceeded == "raise" else None
        for rec in recs:
            assert isinstance(rec, CompiledOpenBlockRecord)
            for s in range(self.batch_size):
                if self._live_exps[s].chi > 0:
                    self._live_exps[s] = apply_compiled_block(self._live_exps[s], rec.block, check_unitary=True, atol=self.atol)
                    if mc_cap is not None and self._live_exps[s].chi > mc_cap:
                        raise QueryBudgetExceededError(f"Expansion exceeded max_components={mc_cap} (got chi={self._live_exps[s].chi})")

    def _apply_pauli_mask(self, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_]) -> None:
        affected = np.flatnonzero(np.any(mask[: self._eff_nq], axis=1))
        if len(affected) == 0:
            return
        self._flush_magic()
        for q_np in affected:
            ax = compile_pauli_product(self._live_ref_tab, [(int(q_np), pauli)])
            for b_np in np.flatnonzero(mask[int(q_np)]):
                b = int(b_np)
                if self._live_exps[b].chi > 0:
                    self._live_exps[b] = apply_pauli(self._live_exps[b], ax)

    def apply_pauli(self, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_], p: float = 1.0) -> None:
        if p <= 0.0:
            return
        eff = mask
        if p < 1.0:
            cnt = int(np.count_nonzero(mask))
            if cnt == 0:
                return
            eff = np.zeros(mask.shape, dtype=np.bool_)
            eff[mask] = self.np_rng.random(cnt) < p
        self._apply_pauli_mask(pauli, eff)

    def _apply_factor_list(self, s: int, fac: CompiledCliffordFactorList, *, inverse: bool = False) -> None:
        self._live_exps[s] = apply_clifford_factors(
            self._live_exps[s], fac, inverse=inverse, check_unitary=True, atol=self.atol
        )

    def _forced_fault_branch(self, s: int, slot_idx: int) -> int | None:
        if self._norm_reals is None or slot_idx < 0:
            return None
        f_map = self._norm_reals[s].get("faults")
        return f_map.get(slot_idx) if f_map is not None else None

    def _fire_fault_channel_live(self, slot_idx: int, ch: CliffordFaultChannel) -> None:
        self._flush_magic()
        probs = [b.prob for b in ch.branches]
        facs = [compile_clifford_factors(self._live_ref_tab, b.gate_name, b.targets) for b in ch.branches]
        for s in range(self.batch_size):
            forced = self._forced_fault_branch(s, slot_idx)
            rng_s = self._shot_np_rngs[s] if self._shot_np_rngs is not None else self.np_rng
            c = sample_categorical_branch(probs, rng_s) if forced is None else forced
            if c > 0:
                self._apply_factor_list(s, facs[c - 1])

    def _next_step_meta(self, op: stim.CircuitInstruction) -> None:
        pass

    def do_copies_on_fused_step(
        self,
        op: stim.CircuitInstruction,
        copies: list[stim.CircuitInstruction],
        do_copy: Callable[[stim.CircuitInstruction], None],
    ) -> None:
        for sub_op in copies:
            do_copy(sub_op)

    def record_direct_measurements(self, outcomes_per_target: NDArray[np.bool_]) -> None:
        self._cached_m2d = None
        for row in np.asarray(outcomes_per_target, dtype=np.bool_):
            self._measurement_columns.append(row.copy())

    def _do(self, obj: stim.Circuit | stim.CircuitInstruction | stim.CircuitRepeatBlock) -> None:
        for op in unroll_circuit(obj):
            if self._used_interactively:
                check_replay(
                    op,
                    self._unrolled_ops,
                    self._circuit_time,
                    "MagicsideSimulator",
                    num_qubits=self.num_qubits,
                    allow_tag=lambda tag: tag in _MAGIC_TAGS or tag.startswith("CLIFFORD_FAULT"),
                )
            self._do_instruction(op)

    def _do_instruction(self, op: stim.CircuitInstruction) -> None:
        t, n = self._circuit_time, len(self._unrolled_ops)
        if (t < n or (t == 0 and n == 0)) and t in self._prog_faults:
            for slot_idx, ch in self._prog_faults[t]:
                self._fire_fault_channel_live(slot_idx, ch)
        if op.tag.startswith("CLIFFORD_FAULT"):
            for slot_idx, ch in (self._inline_faults.get(t, ()) if t < n else ((-1, c) for c in parse_clifford_fault_instruction(op, t))):
                self._fire_fault_channel_live(slot_idx, ch)
        elif apply_coords_op(op, self.qubit_coords, self.qubit_tags, self.coords_shifts):
            pass
        else:
            self._compiled_op_handler.handle_op(op=op, sss=self)
        self._circuit_time += 1
        if t + 1 == n and n in self._prog_faults:
            for slot_idx, ch in self._prog_faults[n]:
                self._fire_fault_channel_live(slot_idx, ch)

    def _emit_magic_op(self, op: stim.CircuitInstruction, qs: Sequence[int] | None = None) -> None:
        emit_magic_terms(
            op.tag, validate_magic_op(op) if qs is None else qs, self._live_ref_tab, self._pending_magic
        )

    emit_magic_op = _emit_magic_op

    def _do_bare_instruction(self, op: stim.Circuit | stim.CircuitInstruction) -> None:
        if isinstance(op, stim.Circuit):
            for sub in op:
                self._do_bare_instruction(sub)  # type: ignore[arg-type]
            return
        name, tag = op.name, op.tag
        if tag in _MAGIC_TAGS:
            self._emit_magic_op(op)
            return
        if tag != "":
            raise ValueError(f"Unrecognized instruction tag {tag!r} on {op!r}")
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "BARRIER"):
            return
        if name == "MPAD":
            raw = op.targets_copy()
            for gt in raw:
                self._measurement_columns.append(np.full(self.batch_size, bool(gt.value) ^ bool(gt.is_inverted_result_target), dtype=np.bool_))
            args = op.gate_args_copy()
            if raw and args and args[0] > 0.0:
                self._apply_meas_flip_noise(float(args[0]), len(raw))
            return
        if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
            self._do_heralded_noise(op)
            return
        cname = canonical_gate_data(name).name
        gd = stim.gate_data(cname)
        if gd.is_noisy_gate and not gd.produces_measurements and not gd.is_reset:
            self.do_conditional_pauli_noise(op, condition_met_mask=None)
        elif gd.is_unitary:
            self.do_conditional_clifford(op, condition_met_mask=None)
        elif gd.produces_measurements or gd.is_reset:
            self._do_live_meas_or_reset(op, cname, gd)
        else:
            raise ValueError(f"Unsupported instruction {op!r}")

    def _do_heralded_noise(self, op: stim.CircuitInstruction) -> None:
        raw = op.targets_copy()
        if not raw:
            return
        self._flush_magic()
        qs = [int(t.qubit_value) for t in raw]
        xm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        zm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        if self._tab_rngs is not None:
            assert self._tab_rngs_x is not None
            _, _, cols = exec_sync_heralded_noise(
                self._tab_rngs, self._tab_rngs_x, op, set(qs), len(qs), xm, zm
            )
            self._measurement_columns.extend(cols)
        elif op.name == "HERALDED_ERASE":
            pe = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
            for q in qs:
                hk = self.np_rng.random(self.batch_size) < pe
                pi = self.np_rng.integers(0, 4, size=self.batch_size)
                xm[q] ^= hk & ((pi == 1) | (pi == 2))
                zm[q] ^= hk & ((pi == 2) | (pi == 3))
                self._measurement_columns.append(hk)
        else:
            args = op.gate_args_copy()
            p0, px, py, pz = (float(args[i]) if i < len(args) else 0.0 for i in range(4))
            ci, cx, cy, cz = p0, p0 + px, p0 + px + py, p0 + px + py + pz
            for q in qs:
                r = self.np_rng.random(self.batch_size)
                xm[q] ^= (r >= ci) & (r < cy)
                zm[q] ^= (r >= cx) & (r < cz)
                self._measurement_columns.append(r < cz)
        self._apply_pauli_mask("X", xm)
        self._apply_pauli_mask("Z", zm)

    def _apply_meas_flip_noise(self, p_err: float, num_m: int) -> None:
        if p_err <= 0.0 or num_m <= 0:
            return
        if self._tab_rngs is not None:
            assert self._tab_rngs_x is not None
            apply_sync_meas_flips(self._tab_rngs, self._tab_rngs_x, self._measurement_columns, p_err, num_m)
        else:
            flips = self.np_rng.random((num_m, self.batch_size)) < p_err
            for k in range(num_m):
                self._measurement_columns[-num_m + k] ^= flips[k]

    def _obs_p1(self, exp: ShotExpansion, obs: CompiledPauliAxis) -> float:
        if exp.chi == 0:
            return 0.0
        if not words_any_nonzero(obs.a_words) and exp.d == 0 and exp.chi == 1:
            return float(((obs.kappa >> 1) ^ word_dot_parity(obs.b_words, exp.h_words[0])) & 1)
        z_in = exp.norm_sq()
        if z_in <= 0.0:
            return 0.0
        if words_any_nonzero(obs.a_words):
            ep = unit_bit_words(first_set_bit_in_words(obs.a_words), exp.n_words)
            work = apply_quarter_turn(exp, obs.a_words, np.ascontiguousarray(obs.b_words ^ ep), (obs.kappa + 1) & 3, sign=1, check_unitary=True, atol=self.atol)
            bd, sg = ep, 0
        else:
            work, bd, sg = exp, obs.b_words, (obs.kappa >> 1) & 1
        _, z1_r = project_diagonal(work, bd, sg, 1)
        return min(1.0, max(0.0, float(z1_r) / z_in))

    def _sample_sync_born_bit(self, b: int, p1: float) -> int:
        return (
            int(self._draw_tab_rng_bit(b))
            if abs(p1 - 0.5) <= 1e-12
            else int(self._shot_np_rngs[b].random() < p1)  # type: ignore[index]
        )

    def _project_substep_batch(self, rec: MeasurementRecord | ResetRecord) -> NDArray[np.bool_]:
        m_shots = np.zeros(self.batch_size, dtype=np.bool_)
        obs = rec.obs_axis
        idx: list[int] = []
        for b in range(self.batch_size):
            exp = self._live_exps[b]
            if exp.chi == 0:
                self._log_branch_masses[b] = -math.inf
                continue
            if rec.rot_axis is None and exp.d == 0 and exp.chi == 1:
                bit = ((obs.kappa >> 1) ^ word_dot_parity(obs.b_words, exp.h_words[0])) & 1
                m_shots[b] = bool(bit)
                if isinstance(rec, ResetRecord) and (bool(bit) != rec.reset_flip_ref):
                    self._live_exps[b] = apply_pauli(exp, rec.reset_kick_axis)
                continue
            idx.append(b)
        if not idx:
            return m_shots
        sampler = (
            None
            if self._tab_rngs is None
            else lambda i, p1: (
                0 if p1 <= K1_CANCEL_REL_TOL
                else 1 if p1 >= 1.0 - K1_CANCEL_REL_TOL
                else self._sample_sync_born_bit(idx[i], p1)
            )
        )
        out, z0, z1, posts = measure_expansions_batch(
            [self._live_exps[b] for b in idx], obs, rng=self.np_rng, sample_outcome=sampler,
            keep_rotated_frame=(rec.rot_axis is not None), atol=self.atol,
        )
        self._log_branch_masses[idx] += _log0(np.where(out, z1, z0) / (z0 + z1))
        m_shots[idx] = out
        for b, post, outcome in zip(idx, posts, out.tolist()):
            if isinstance(rec, ResetRecord) and (outcome != rec.reset_flip_ref):
                post = apply_pauli(post, rec.reset_kick_axis)
            self._live_exps[b] = post
        return m_shots

    def _do_live_meas_or_reset(self, op: stim.CircuitInstruction, cname: str, gd: stim.GateData) -> None:
        self._flush_magic()
        args, raw = op.gate_args_copy(), op.targets_copy()
        p_err = float(args[0]) if (gd.produces_measurements and args) else 0.0
        if not (math.isfinite(p_err) and 0.0 <= p_err <= 1.0):
            raise ValueError(f"Invalid measurement flip probability {p_err!r}")
        specs: list[tuple[set[int], Any, bool]] = []
        if cname == "MPP":
            for grp, inv in split_pauli_product_targets(op):
                red, neg = reduce_hermitian_pauli_product(self._live_ref_tab.num_qubits, grp)
                specs.append(({q for q, _ in red}, red, inv ^ neg))
        elif cname in ("MXX", "MYY", "MZZ"):
            pch = cname[-1]
            for a, b_t in zip(raw[::2], raw[1::2]):
                q0, q1 = int(a.qubit_value), int(b_t.qubit_value)
                specs.append(({q0, q1}, [] if q0 == q1 else [(q0, pch), (q1, pch)], bool(a.is_inverted_result_target) ^ bool(b_t.is_inverted_result_target)))
        else:
            basis: Literal["X", "Y", "Z"] = "X" if "X" in cname else ("Y" if "Y" in cname else "Z")
            for gt in raw:
                q, inv = int(gt.qubit_value), bool(gt.is_inverted_result_target)
                specs.append(({q}, (q, basis) if gd.is_reset else [(q, basis)], inv))
        seen: set[int] = set()
        seg_len = 0
        for supp, payload, inv in specs:
            if gd.produces_measurements and cname in ("MPP", "MXX", "MYY", "MZZ") and seg_len > 0 and (not supp or bool(seen & supp)):
                # Flush the previous segment's readout flips before projecting this product (Stim's draw order).
                if p_err > 0.0:
                    self._apply_meas_flip_noise(p_err, seg_len)
                seg_len = 0
                seen.clear()
            rec: MeasurementRecord | ResetRecord = (
                compile_reset_c15(self._live_ref_tab, payload[0], payload[1], meas_index=0 if gd.produces_measurements else None, invert_result=inv, flip_prob=0.0, frame_events=self._live_frame_events)
                if gd.is_reset
                else compile_pauli_measurement_c15(self._live_ref_tab, payload, invert_result=inv, flip_prob=0.0, meas_index=0, frame_events=self._live_frame_events)
            )
            m_shots = self._project_substep_batch(rec)
            if gd.produces_measurements:
                self._measurement_columns.append(~m_shots if inv else m_shots.copy())
                if cname == "MPP" and not supp:
                    if p_err > 0.0:
                        self._apply_meas_flip_noise(p_err, 1)
                else:
                    seg_len += 1
                    seen |= supp
        if seg_len > 0 and p_err > 0.0:
            self._apply_meas_flip_noise(p_err, seg_len)

    def _read_classical_cond(self, tgt: stim.GateTarget) -> NDArray[np.bool_]:
        if tgt.is_measurement_record_target:
            idx = len(self._measurement_columns) + tgt.value
            if idx < 0 or idx >= len(self._measurement_columns):
                raise ValueError(f"Feedback rec[{tgt.value}] out of range")
            return self._measurement_columns[idx]
        if tgt.is_sweep_bit_target:
            idx = tgt.value
            return self._sweep_mat[:, idx] if (self._sweep_mat is not None and 0 <= idx < self._sweep_mat.shape[1]) else np.zeros(self.batch_size, dtype=np.bool_)
        raise ValueError(f"Unsupported classical target {tgt!r}")

    def do_conditional_clifford(self, op: stim.CircuitInstruction, condition_met_mask: NDArray[np.bool_] | None = None, *, condition_group_fn: Callable[[int], NDArray[np.bool_]] | None = None) -> None:
        name = canonical_gate_data(op.name).name
        if name in ("I", "II"):
            if condition_group_fn is not None:
                arity = 1 if name == "I" else 2
                for g_idx in range(sum(t.qubit_value is not None for t in op.targets_copy()) // arity):
                    condition_group_fn(g_idx)
            return
        mask_at = lambda i: condition_group_fn(i) if condition_group_fn is not None else (condition_met_mask[i] if condition_met_mask is not None else None)
        if name in ("SPP", "SPP_DAG"):
            dag = name == "SPP_DAG"
            for g_idx, (grp, inv) in enumerate(split_pauli_product_targets(op)):
                self._apply_conditional_ref_step(
                    mask_at(g_idx),
                    lambda g=grp, iv=inv: (
                        self._live_ref_tab.do_spp(g, negate=iv, dag=dag),
                        self._live_frame_events.append(LeftSppEvent(tuple(g), iv, dag)),
                    ),
                    lambda g=grp, iv=inv: CompiledCliffordFactorList(
                        7 if dag else 1, None, (compile_pauli_product(self._live_ref_tab, g, negate=iv ^ (not dag)),)
                    ),
                )
            return
        canon, arity = canonical_clifford_gate_name_and_arity(name)
        raw = op.targets_copy()
        if arity == 2 and any(t.qubit_value is None for t in raw):
            fb_axes = FEEDBACK_2Q_GATES.get(canon)
            for p_idx, (a, b_t) in enumerate(zip(raw[::2], raw[1::2])):
                mask_g = mask_at(p_idx)
                if a.qubit_value is not None and b_t.qubit_value is not None:
                    self._apply_unitary_group(canon, (int(a.qubit_value), int(b_t.qubit_value)), mask_g)
                    continue
                if fb_axes is None or (a.qubit_value is None and b_t.qubit_value is None):
                    raise ValueError(f"Unsupported feedback pair {(a, b_t)} in {op!r}")
                cl_t, q_t, pch = (a, int(b_t.qubit_value), fb_axes[1]) if (a.qubit_value is None and fb_axes[0] == "Z") else ((b_t, int(a.qubit_value), fb_axes[0]) if (b_t.qubit_value is None and fb_axes[1] == "Z") else (None, -1, ""))
                if cl_t is None:
                    raise ValueError(f"Invalid feedback control basis in {op!r}")
                self._flush_magic()
                cond = self._read_classical_cond(cl_t)
                if mask_g is not None:
                    cond = cond & mask_g
                if np.any(cond):
                    ax = compile_pauli_product(self._live_ref_tab, [(q_t, pch)])
                    for s in np.flatnonzero(cond):
                        si = int(s)
                        if self._live_exps[si].chi > 0:
                            self._live_exps[si] = apply_pauli(self._live_exps[si], ax)
            return
        qs = [int(t.qubit_value) for t in raw]
        for g_idx, off in enumerate(range(0, len(qs), arity)):
            self._apply_unitary_group(canon, tuple(qs[off : off + arity]), mask_at(g_idx))

    def _apply_unitary_group(self, canon: str, grp: tuple[int, ...], mask_g: NDArray[np.bool_] | None) -> None:
        self._apply_conditional_ref_step(
            mask_g,
            lambda: (self._live_ref_tab.do_unitary(canon, grp), self._live_frame_events.append(LeftUnitaryEvent(canon, grp))),
            lambda: compile_clifford_factors(self._live_ref_tab, canon, grp),
        )

    def _apply_conditional_ref_step(
        self,
        mask_g: NDArray[np.bool_] | None,
        advance_ref: Callable[[], object],
        compile_fac: Callable[[], CompiledCliffordFactorList],
    ) -> None:
        if mask_g is None or np.all(mask_g):
            advance_ref()
            return
        n_act = int(np.count_nonzero(mask_g))
        if n_act == 0:
            return
        self._flush_magic()
        fac = compile_fac()
        if 2 * n_act <= self.batch_size:
            for s in np.flatnonzero(mask_g):
                self._apply_factor_list(int(s), fac, inverse=False)
        else:
            advance_ref()
            for s in np.flatnonzero(~mask_g):
                self._apply_factor_list(int(s), fac, inverse=True)

    def do_conditional_pauli_noise(self, op: stim.CircuitInstruction, condition_met_mask: NDArray[np.bool_] | None = None) -> None:
        cname = canonical_gate_data(op.name).name
        if cname in ("I_ERROR", "II_ERROR", "I", "II"):
            return
        args = [float(a) for a in op.gate_args_copy()]
        if cname in ("CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"):
            p_c = args[0] if args else 0.0
            raw_t = op.targets_copy()
            xm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
            zm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
            if self._tab_rngs is not None:
                assert self._tab_rngs_x is not None
                exec_sync_correlated_error(
                    self._tab_rngs, self._tab_rngs_x, cname, raw_t, p_c, condition_met_mask, xm, zm
                )
            else:
                if p_c <= 0.0:
                    if cname != "ELSE_CORRELATED_ERROR":
                        self._corr_occurred.fill(False)
                    return
                u = self.np_rng.random(self.batch_size)
                if cname != "ELSE_CORRELATED_ERROR":
                    occ = u < p_c
                    self._corr_occurred[:] = occ
                else:
                    occ = (~self._corr_occurred) & (u < p_c)
                    self._corr_occurred |= occ
                if np.any(occ):
                    for t_idx, gt in enumerate(raw_t):
                        if gt.qubit_value is None:
                            continue
                        hit = occ if condition_met_mask is None else (occ & condition_met_mask[t_idx])
                        if gt.is_x_target or gt.is_y_target:
                            xm[int(gt.qubit_value)] ^= hit
                        if gt.is_z_target or gt.is_y_target:
                            zm[int(gt.qubit_value)] ^= hit
            self._apply_pauli_mask("X", xm)
            self._apply_pauli_mask("Z", zm)
            return
        raw = [int(t.qubit_value) for t in op.targets_copy()]
        if not raw or not any(p > 0.0 for p in args):
            return
        arity = 2 if cname in ("DEPOLARIZE2", "PAULI_CHANNEL_2") else 1
        if self._tab_rngs is not None:
            self._sample_pauli_noise_via_tab_rng(
                cname, conditioned_targets_per_shot(raw, arity, self.batch_size, condition_met_mask), args
            )
            return
        groups = [raw[i : i + arity] for i in range(0, len(raw), arity)]
        xm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        zm = np.zeros((self._eff_nq, self.batch_size), dtype=np.bool_)
        u = self.np_rng.random((len(groups), self.batch_size))
        if condition_met_mask is not None:
            u = np.where(condition_met_mask, u, 2.0)
        if cname in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            terms: Sequence[tuple[str, float]] = [(cname[0], args[0])]
        elif cname in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
            probs1 = (args[0] / 3.0,) * 3 if cname == "DEPOLARIZE1" else (args[0], args[1], args[2])
            terms = list(zip(("X", "Y", "Z"), probs1))
        else:
            probs2 = (args[0] / 15.0,) * 15 if cname == "DEPOLARIZE2" else tuple(args[:15])
            terms = list(zip(PAULI_2Q_ORDER, probs2))
        acc = 0.0
        for label, pr in terms:
            lo, acc = acc, acc + pr
            hit = (u >= lo) & (u < acc)
            if np.any(hit):
                for g_i, grp in enumerate(groups):
                    for k, pch in enumerate(label):
                        if pch in ("X", "Y"):
                            xm[grp[k]] ^= hit[g_i]
                        if pch in ("Z", "Y"):
                            zm[grp[k]] ^= hit[g_i]
        self._apply_pauli_mask("X", xm)
        self._apply_pauli_mask("Z", zm)

    def peek_pauli_born_batch(self, targets: Sequence[int] | NDArray[np.int_], pauli: Literal["X", "Y", "Z"] = "Z") -> NDArray[np.float64]:
        self._flush_magic()
        t_list = [int(q) for q in targets]
        out = np.zeros((len(t_list), self.batch_size), dtype=np.float64)
        for row_i, q in enumerate(t_list):
            shared_obs = None if self._shot_tabs is not None else compile_pauli_product(self._live_ref_tab, [(q, pauli)])
            for b in range(self.batch_size):
                obs = shared_obs if shared_obs is not None else compile_pauli_product(self._shot_tabs[b], [(q, pauli)])  # type: ignore[index]
                out[row_i, b] = self._obs_p1(self._live_exps[b], obs)
        return out

    def peek_pauli_batch(self, targets: Sequence[int] | NDArray[np.int_], pauli: Literal["X", "Y", "Z"] = "Z") -> NDArray[np.int8]:
        p1 = self.peek_pauli_born_batch(targets, pauli)
        res = np.zeros(p1.shape, dtype=np.int8)
        res[p1 <= 1e-12] = 1
        res[p1 >= 1.0 - 1e-12] = -1
        return res

    def postselect_z_batch(self, q: int, flip_mask: NDArray[np.bool_], desired_bits: NDArray[np.uint8], pauli: Literal["X", "Y", "Z"] = "Z") -> None:
        shots = np.flatnonzero(flip_mask)
        if len(shots) == 0:
            return
        self._flush_magic()
        obs = compile_pauli_product(self._live_ref_tab, [(int(q), pauli)])
        for b_np in shots:
            b = int(b_np)
            if self._live_exps[b].chi == 0:
                continue
            m_res = measure_expansion(self._live_exps[b], obs, requested_bit=int(desired_bits[b]) & 1, renormalize=True, keep_rotated_frame=False, atol=self.atol)
            self._live_exps[b] = m_res.post_state
            zin = m_res.z0 + m_res.z1
            if zin > 0.0:
                self._log_branch_masses[b] += _log0((m_res.z1 if m_res.outcome else m_res.z0) / zin)

    def do_unmatched_noisy_reset_z(self, reset_zero_mask: NDArray[np.bool_], reset_one_mask: NDArray[np.bool_], pauli: Literal["X", "Y", "Z"] = "Z", order: Sequence[int] | None = None) -> None:
        comb = reset_zero_mask | reset_one_mask
        if not np.any(comb):
            return
        self._flush_magic()
        qs = list(dict.fromkeys(order)) if order is not None else [int(x) for x in np.flatnonzero(np.any(comb, axis=1))]
        kick_p: Literal["X", "Z"] = "Z" if pauli == "X" else "X"
        for q in qs:
            shots = np.flatnonzero(comb[q])
            if len(shots) == 0:
                continue
            obs = compile_pauli_product(self._live_ref_tab, [(int(q), pauli)])
            kick = compile_pauli_product(self._live_ref_tab, [(int(q), kick_p)])
            for b_np in shots:
                b = int(b_np)
                pb = self._obs_p1(self._live_exps[b], obs)
                if pb <= 1e-12 or pb >= 1.0 - 1e-12:
                    m_b = int(pb >= 1.0 - 1e-12)
                else:
                    m_b = self._sample_sync_born_bit(b, pb) if self._tab_rngs is not None else int(float(self.np_rng.random()) < pb)
                    m_res = measure_expansion(self._live_exps[b], obs, requested_bit=m_b, renormalize=True, keep_rotated_frame=False, atol=self.atol)
                    self._live_exps[b] = m_res.post_state
                if m_b != int(reset_one_mask[q, b]) and self._live_exps[b].chi > 0:
                    self._live_exps[b] = apply_pauli(self._live_exps[b], kick)

    def current_tableau_measurement_record(self) -> NDArray[np.bool_]:
        if not self._measurement_columns:
            return np.zeros((0,), dtype=np.bool_)
        mat = np.column_stack(self._measurement_columns)
        return mat[0].copy() if self.batch_size == 1 else mat

    def get_final_measurement_records(self) -> NDArray[np.bool_]:
        if not self._finished_running_circuit:
            raise RuntimeError("Run not finished; call run() or finish_interactive_run() first.")
        return stack_measurement_columns(self._measurement_columns, self.batch_size)

    def _compute_noiseless_magic_ref(self, sweep_row: NDArray[np.bool_] | None) -> NDArray[np.bool_]:
        key = b"" if sweep_row is None else np.asarray(sweep_row, dtype=np.bool_).tobytes()
        if key not in self._ref_sample_cache:
            stripped, needs_leak = _strip_circuit_for_noiseless_ref(self.circuit)
            if not needs_leak:
                ref = run_direct(compile_direct(stripped), rng=np.random.default_rng(0), sweep_bits=sweep_row, atol=self.atol).measurements
            else:
                sub = MagicsideSimulator(stripped, op_handler=LeakageUint8Magic(), batch_size=1, seed=0, sweep_bits=sweep_row, m2d_reference_mode="zero", atol=self.atol)
                sub.run()
                ref = sub.get_final_measurement_records()[0]
            self._ref_sample_cache[key] = ref.copy()
        return self._ref_sample_cache[key].copy()

    def get_reference_sample(self, sweep_bits: Any = None) -> NDArray[np.bool_]:
        if self._explicit_ref_sample is not None:
            return self._explicit_ref_sample.copy()
        if self._eff_m2d_mode == "zero":
            return np.zeros(self.num_measurements, dtype=np.bool_)
        if self._eff_m2d_mode == "stim":
            return np.asarray(self.circuit.reference_sample(), dtype=np.bool_)
        return self._compute_noiseless_magic_ref(None if sweep_bits is None else _normalize_sweep_matrix(sweep_bits, 1)[0])  # type: ignore[index]

    def _convert_m2d(self, sweep_bits: Any) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        if sweep_bits is None and self._cached_m2d is not None:
            return self._cached_m2d
        recs = self.get_final_measurement_records()
        if recs.shape[1] != self.num_measurements:
            raise ValueError(f"Measurement count {recs.shape[1]} != circuit.num_measurements={self.num_measurements}")
        sw = _normalize_sweep_matrix(sweep_bits, self.batch_size) if sweep_bits is not None else self._sweep_mat
        nsw = self.circuit.num_sweep_bits
        sw = (sw[:, :nsw] if sw.shape[1] >= nsw else np.pad(sw, ((0, 0), (0, nsw - sw.shape[1])))) if (sw is not None and sw.shape[1] > 0 and nsw > 0) else None
        if self._explicit_ref_sample is None and self._eff_m2d_mode == "stim":
            if self._m2d_stim_conv is None:
                self._m2d_stim_conv = self.circuit.compile_m2d_converter()
            kw: dict[str, Any] = {"separate_observables": True}
            if sw is not None:
                kw["sweep_bits"] = sw
            res: tuple[NDArray[np.bool_], NDArray[np.bool_]] = self._m2d_stim_conv.convert(measurements=recs, **kw)
        else:
            if self._m2d_skip_conv is None:
                self._m2d_skip_conv = self.circuit.compile_m2d_converter(skip_reference_sample=True)
            if self._explicit_ref_sample is not None:
                xor_recs = np.ascontiguousarray(recs ^ self._explicit_ref_sample[None, :])
            elif self._eff_m2d_mode == "zero":
                xor_recs = np.ascontiguousarray(recs)
            elif sw is None:
                xor_recs = np.ascontiguousarray(recs ^ self._compute_noiseless_magic_ref(None)[None, :])
            else:
                xor_recs = np.ascontiguousarray(recs ^ np.stack([self._compute_noiseless_magic_ref(sw[s]) for s in range(self.batch_size)], axis=0))
            res = self._m2d_skip_conv.convert(measurements=xor_recs, separate_observables=True)
        if sweep_bits is None:
            self._cached_m2d = res
        return res

    def get_detector_flips(
        self, append_observables: bool = False, *, sweep_bits: Any = None, bit_packed: bool = False
    ) -> NDArray[np.bool_] | NDArray[np.uint8]:
        dets, obs = self._convert_m2d(sweep_bits)
        return format_detector_flips(
            dets, obs, append_observables=append_observables, bit_packed=bit_packed
        )

    def get_observable_flips(
        self, *, sweep_bits: Any = None, bit_packed: bool = False
    ) -> NDArray[np.bool_] | NDArray[np.uint8]:
        return pack_shot_flips(self._convert_m2d(sweep_bits)[1], bit_packed=bit_packed)

    def _validate_shot_idx(self, shot: int) -> int:
        s = int(shot) + (self.batch_size if int(shot) < 0 else 0)
        if s < 0 or s >= self.batch_size:
            raise IndexError(f"shot {shot} out of range for batch_size={self.batch_size}")
        return s

    def _resolve_shot_state(self, s: int, trace_selectors: bool) -> tuple[ReferenceCHPTableau, ShotExpansion, Sequence[FrameEvent]]:
        tab = self._shot_tabs[s] if self._shot_tabs is not None else self._live_ref_tab
        exp, cp = self._live_exps[s], self._active_plan
        if isinstance(cp, SelectorPlan) and self._shot_sel_bits is not None:
            sb = self._shot_sel_bits[s]
            if trace_selectors and sb is not None:
                tab, exp = extract_data_expansion(cp, exp, sb)
                return tab, exp, cp.data_frame_events
            return tab, exp, cp.frame_events
        return tab, exp, self._live_frame_events

    def get_qfe_expansion(self, shot: int = 0, *, trace_selectors: bool = True) -> tuple[stim.Tableau, ShotExpansion]:
        self._flush_magic()
        tab, exp, _ = self._resolve_shot_state(self._validate_shot_idx(shot), bool(trace_selectors))
        return tab.to_stim_tableau(), exp.copy()

    def get_state_expansions(self, *, trace_selectors: bool = True) -> list[tuple[stim.Tableau, ShotExpansion]]:
        self._flush_magic()
        return [self.get_qfe_expansion(s, trace_selectors=trace_selectors) for s in range(self.batch_size)]

    def get_branch_mass(self, shot: int = 0) -> float:
        self._flush_magic()
        return math.exp(float(self._log_branch_masses[self._validate_shot_idx(shot)]))

    def state_vector(
        self, shot: int = 0, *, endian: Literal["little", "big"] = "little", trace_selectors: bool = True, normalize: bool = True,
    ) -> NDArray[np.complex128]:
        self._flush_magic()
        s = self._validate_shot_idx(shot)
        if endian not in ("little", "big"):
            raise ValueError(f"endian must be 'little' or 'big', got {endian!r}")
        scale = 1.0 if normalize else math.exp(0.5 * float(self._log_branch_masses[s]))
        if self._logical_nq == 0:
            return np.array([complex(scale)], dtype=np.complex128)
        tab, exp, fevs = self._resolve_shot_state(s, bool(trace_selectors))
        nq = exp.n_qubits
        if nq > 26:
            raise ValueError(f"state_vector requires num_qubits <= 26, got {nq}")
        psi = reconstruct_statevector(tab, exp, frame_events=fevs)
        if not normalize:
            psi = np.ascontiguousarray(psi * scale)
        if endian == "big" and nq > 1:
            psi = np.ascontiguousarray(psi.reshape((2,) * nq).transpose(tuple(range(nq - 1, -1, -1))).ravel())
        return psi

    get_statevector = state_vector
