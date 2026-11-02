"""`DIRECT` and `SELECTOR` trajectory runners and selector activation sampling (`stimside._magic.trajectory`)."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from stimside._magic.channels import (
    CompiledOpenBlockRecord,
    CompiledRecord,
    CorrelatedNoiseRecord,
    DirectFaultSlotRecord,
    FeedbackPauliRecord,
    FrameEvent,
    HeraldedNoiseRecord,
    LeakageBarrierRecord,
    MPadRecord,
    MeasurementRecord,
    PauliNoiseRecord,
    ResetRecord,
    _normalize_choice_map,
    _sample_categorical_branch,
    _sample_heralded_pauli_branch,
    apply_clifford_factors,
)
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.phase_qfe import (
    DEFAULT_ATOL,
    ImpossiblePostselectionError,
    K1_CANCEL_REL_TOL,
    ShotExpansion,
    apply_compiled_block,
    apply_pauli,
    born_split_expansion,
    measure_expansion,
    validate_atol,
)
from stimside._magic.plan import DirectPlan, SelectorPlan, TrajectoryRunResult
from stimside._magic.statevector import extract_data_expansion
from stimside.util.bitwords import n_words_for_qubits, unit_bit_words
from stimside.util.reference_chp import ReferenceCHPTableau


def _prepare_run_maps(
    plan: DirectPlan | SelectorPlan,
    *,
    fault_trace: object = None,
    pauli_noise_trace: object = None,
    reset_outcomes: object = None,
    requested_outcomes: object = None,
    requested_quantum_outcomes: object = None,
) -> tuple[
    dict[int, int] | None,
    dict[int, int] | None,
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    fault_max = [len(ch.branches) for ch in plan.fault_channels]
    fault_id_to_slot = {
        (ch.slot_id if ch.slot_id is not None else idx): idx for idx, ch in enumerate(plan.fault_channels)
    }
    meas_bounds = (1,) * plan.num_measurements
    f_map = _normalize_choice_map(
        fault_trace, fault_max, "fault_trace", key_to_slot=fault_id_to_slot, allow_none_zero=True
    )
    n_map = _normalize_choice_map(pauli_noise_trace, plan.noise_channel_max_choices, "pauli_noise_trace")
    r_map = _normalize_choice_map(reset_outcomes, (1,) * plan.num_resets, "reset_outcomes", allow_bool_values=True)
    c_map = _normalize_choice_map(
        requested_outcomes, meas_bounds, "requested_outcomes", allow_bool_values=True, mapping_only=True
    )
    q_map = _normalize_choice_map(
        requested_quantum_outcomes, meas_bounds, "requested_quantum_outcomes",
        allow_bool_values=True, mapping_only=True,
    )
    return f_map, n_map, r_map or {}, c_map or {}, q_map or {}


def _measure_with_readout_and_postselection(
    exp: ShotExpansion,
    obs_axis: CompiledPauliAxis,
    *,
    flip_prob: float,
    invert_result: bool,
    rng: np.random.Generator,
    req_q: int | None,
    fixed_classical: int | None,
    zero_mass: bool,
    atol: float,
) -> tuple[ShotExpansion, int, bool, bool, tuple[float, float, float], bool]:
    """Unified exact Bayesian readout + postselection collapse for `M*`, `MPP`, `R*`, and `MR*`."""
    iota = 1 if invert_result else 0
    p = float(flip_prob)
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"Invalid measurement flip probability {flip_prob!r}")

    if zero_mass:
        exp.log_mass = -math.inf
        q_out = (
            req_q if req_q is not None
            else ((fixed_classical ^ iota ^ (1 if p == 1.0 else 0)) if fixed_classical is not None else 0)
        )
        rep_bit = bool(fixed_classical) if fixed_classical is not None else bool(q_out ^ iota)
        return exp, q_out, rep_bit, bool(int(rep_bit) ^ q_out ^ iota), (0.0, 0.0, 0.0), True

    if fixed_classical is None:
        m_res = measure_expansion(
            exp, obs_axis, rng=rng, requested_bit=req_q, p_flip=p, invert_result=invert_result,
            renormalize=True, keep_rotated_frame=True, atol=atol,
        )
        return (
            m_res.post_state, m_res.outcome, bool(m_res.reported_outcome), bool(m_res.readout_flipped),
            (m_res.z0, m_res.z1, m_res.input_norm_sq), False,
        )

    tol = validate_atol(atol)
    c = int(fixed_classical) & 1
    z_in, in_log_mass = exp.norm_sq(), exp.log_mass
    if not (math.isfinite(z_in) and z_in >= 0.0):
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={z_in!r}")
    if exp.chi == 0 or z_in <= 0.0:
        if req_q is not None:
            raise ImpossiblePostselectionError("Cannot postselect quantum outcome on zero-norm state")
        exp.log_mass = -math.inf
        return exp, 0, bool(c), bool(c ^ iota), (0.0, 0.0, 0.0), True

    exp0, z0, exp1, z1, _, _, _ = born_split_expansion(
        exp, obs_axis.a_words, obs_axis.b_words, obs_axis.kappa, atol=tol, z_in=z_in
    )
    scale_z = max(1.0, z_in)

    zero_floor = K1_CANCEL_REL_TOL * z_in
    if req_q is not None:
        req_z, req_exp = (z0, exp0) if req_q == 0 else (z1, exp1)
        if req_z <= zero_floor or req_exp.chi == 0:
            raise ImpossiblePostselectionError(
                f"Requested quantum outcome {req_q} has zero Born probability (z_{req_q}={req_z:.3e})"
            )

    m0 = 0.0 if ((req_q is not None and req_q != 0) or z0 <= zero_floor or exp0.chi == 0) else z0 * ((1.0 - p) if c == iota else p)
    m1 = 0.0 if ((req_q is not None and req_q != 1) or z1 <= zero_floor or exp1.chi == 0) else z1 * ((1.0 - p) if c == (1 ^ iota) else p)
    m_step = m0 + m1
    if m_step <= 1e-15 * scale_z:
        q_dead = req_q if req_q is not None else (c ^ iota ^ (1 if p == 1.0 else 0))
        exp.log_mass = -math.inf
        return exp, q_dead, bool(c), bool(c ^ q_dead ^ iota), (0.0, 0.0, 0.0), True

    q_out = 0 if m1 <= 0.0 else (1 if m0 <= 0.0 else int(float(rng.random()) < min(1.0, max(0.0, m1 / m_step))))
    chosen_z, chosen_exp = (z0, exp0) if q_out == 0 else (z1, exp1)
    chosen_exp.weights = np.ascontiguousarray(chosen_exp.weights * (1.0 / math.sqrt(chosen_z)))
    chosen_exp.log_mass = in_log_mass + math.log(m_step)
    return chosen_exp, q_out, bool(c), bool(c ^ q_out ^ iota), (z0, z1, z_in), False


def _execute_compiled_records(
    exp: ShotExpansion,
    records: Sequence[CompiledRecord],
    num_measurements: int,
    final_tableau: ReferenceCHPTableau,
    *,
    rng: np.random.Generator,
    fault_map: Mapping[int, int] | None,
    noise_map: Mapping[int, int] | None,
    reset_map: Mapping[int, int],
    req_c_map: Mapping[int, int],
    req_q_map: Mapping[int, int],
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None,
    max_components: int | None,
    atol: float,
    preset_fault_choices: tuple[int, ...] | None = None,
    preset_noise_map: Mapping[int, int] | None = None,
    total_noise_channels: int | None = None,
    selector_bits: NDArray[np.bool_] | None = None,
    frame_events: tuple[FrameEvent, ...] | None = None,
) -> TrajectoryRunResult:
    atol = validate_atol(atol)
    meas_record = np.zeros(num_measurements, dtype=np.bool_)
    chosen_faults: list[int] = []
    chosen_noise: list[int] = []
    recorded_noise_map: dict[int, int] = {}
    chosen_resets: list[int] = []
    quantum_outcomes: dict[int, int] = {}
    readout_flips: dict[int, bool] = {}
    born_masses: list[tuple[int | None, float, float, float]] = []
    reset_cursor = 0
    correlated_occurred, zero_mass = False, False
    peak_chi, peak_d = max(1, exp.chi), exp.d

    for rec in records:
        if isinstance(rec, CompiledOpenBlockRecord):
            if not zero_mass:
                exp = apply_compiled_block(exp, rec.block, atol=atol)
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)
                if max_components is not None and exp.chi > max_components:
                    raise RuntimeError(f"Trajectory exceeded max_components={max_components} (got chi={exp.chi})")

        elif isinstance(rec, DirectFaultSlotRecord):
            c_idx = fault_map.get(rec.slot_index, 0) if fault_map is not None else _sample_categorical_branch(rec.probs, rng)
            chosen_faults.append(c_idx)
            if c_idx > 0 and not zero_mass:
                exp = apply_clifford_factors(exp, rec.branch_factors[c_idx - 1], atol=atol)
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, PauliNoiseRecord):
            ch_idx = rec.noise_index
            c_n = noise_map.get(ch_idx, 0) if noise_map is not None else _sample_categorical_branch(rec.probs, rng)
            chosen_noise.append(c_n)
            recorded_noise_map[ch_idx] = c_n
            if c_n > 0 and not zero_mass:
                exp = apply_pauli(exp, rec.branch_axes[c_n - 1])

        elif isinstance(rec, HeraldedNoiseRecord):
            ch_idx = rec.noise_index
            m_idx = rec.meas_index
            req_h = req_c_map.get(m_idx)

            if noise_map is not None:
                c_h = noise_map.get(ch_idx, 0)
                true_h = rep_h = c_h > 0
                if req_h is not None and rep_h != bool(req_h):
                    zero_mass, rep_h, exp.log_mass = True, bool(req_h), -math.inf
            else:
                c_h, p_step = _sample_heralded_pauli_branch(rec.probs, req_h, rng)
                true_h = c_h > 0
                rep_h = bool(req_h) if req_h is not None else true_h
                if req_h is not None:
                    if p_step <= 1e-15:
                        zero_mass, exp.log_mass = True, -math.inf
                    else:
                        exp.log_mass += math.log(p_step)

            chosen_noise.append(c_h)
            recorded_noise_map[ch_idx] = c_h
            meas_record[m_idx] = rep_h
            quantum_outcomes[m_idx] = int(true_h)
            readout_flips[m_idx] = False
            if c_h > 0 and rec.branch_axes[c_h - 1] is not None and not zero_mass:
                exp = apply_pauli(exp, rec.branch_axes[c_h - 1])

        elif isinstance(rec, CorrelatedNoiseRecord):
            ch_idx = rec.noise_index
            if not rec.is_else:
                correlated_occurred = False
            if correlated_occurred:
                c_corr = 0
            else:
                c_corr = (
                    noise_map.get(ch_idx, 0) if noise_map is not None
                    else int(float(rng.random()) < rec.prob)
                )
                if c_corr > 0:
                    correlated_occurred = True
                    if not zero_mass:
                        exp = apply_pauli(exp, rec.axis)
            chosen_noise.append(c_corr)
            recorded_noise_map[ch_idx] = c_corr

        elif isinstance(rec, MeasurementRecord):
            m_idx = rec.meas_index
            exp, q_out, rep_bit, flipped, born_tup, zero_mass = _measure_with_readout_and_postselection(
                exp, rec.obs_axis, flip_prob=rec.flip_prob, invert_result=rec.invert_result, rng=rng,
                req_q=req_q_map.get(m_idx), fixed_classical=req_c_map.get(m_idx),
                zero_mass=zero_mass, atol=atol,
            )
            meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_bit, q_out, flipped
            born_masses.append((m_idx, *born_tup))
            if not zero_mass:
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, ResetRecord):
            req_r, m_idx = reset_map.get(reset_cursor), rec.meas_index
            req_mq = req_q_map.get(m_idx) if m_idx is not None else None
            if req_r is not None and req_mq is not None and req_r != req_mq and not zero_mass:
                raise ImpossiblePostselectionError(
                    f"Conflicting reset_outcomes[{reset_cursor}]={req_r} and "
                    f"requested_quantum_outcomes[{m_idx}]={req_mq}"
                )
            exp, q_out, rep_bit, flipped, born_tup, zero_mass = _measure_with_readout_and_postselection(
                exp, rec.obs_axis, flip_prob=0.0 if m_idx is None else rec.flip_prob,
                invert_result=rec.invert_result, rng=rng,
                req_q=req_r if req_r is not None else req_mq,
                fixed_classical=req_c_map.get(m_idx) if m_idx is not None else None,
                zero_mass=zero_mass, atol=atol,
            )
            chosen_resets.append(q_out)
            reset_cursor += 1
            if not zero_mass and (bool(q_out) != rec.reset_flip_ref):
                exp = apply_pauli(exp, rec.reset_kick_axis)
            if m_idx is not None:
                meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_bit, q_out, flipped
            born_masses.append((m_idx, *born_tup))
            if not zero_mass:
                peak_chi, peak_d = max(peak_chi, exp.chi), max(peak_d, exp.d)

        elif isinstance(rec, MPadRecord):
            m_idx, pad_b, p_flip = rec.meas_index, bool(rec.pad_bit), float(rec.flip_prob)
            p1_pad = (1.0 - p_flip) if pad_b else p_flip
            req_pad = req_c_map.get(m_idx)
            if req_pad is not None:
                rep_pad = bool(req_pad)
                flipped_pad = rep_pad ^ pad_b
                p_req = p1_pad if rep_pad else (1.0 - p1_pad)
                if p_req <= 1e-15:
                    zero_mass, exp.log_mass = True, -math.inf
                else:
                    exp.log_mass += math.log(p_req)
            else:
                flipped_pad = bool(float(rng.random()) < p_flip) if p_flip > 0.0 else False
                rep_pad = pad_b ^ flipped_pad
            meas_record[m_idx], quantum_outcomes[m_idx], readout_flips[m_idx] = rep_pad, int(pad_b), flipped_pad

        elif isinstance(rec, FeedbackPauliRecord):
            if rec.source_kind == "rec":
                idx = rec.current_meas_count + rec.source_index
                if idx < 0 or idx >= rec.current_meas_count:
                    raise ValueError(
                        f"Feedback rec[{rec.source_index}] out of bounds at meas_count={rec.current_meas_count}"
                    )
                cond = bool(meas_record[idx])
            else:
                cond = bool(sweep_bits[rec.source_index]) if (sweep_bits is not None and 0 <= rec.source_index < len(sweep_bits)) else False
            if cond and not zero_mass:
                exp = apply_pauli(exp, rec.axis)

        elif isinstance(rec, LeakageBarrierRecord):
            raise ValueError(
                f"Cannot execute leakage-tagged instruction {rec.instruction!r} in standalone "
                "trajectory runner; execute via MagicsideSimulator with LeakageUint8Magic."
            )
        else:
            raise ValueError(f"Unexpected record during trajectory run: {rec!r}")

    if zero_mass:
        exp = ShotExpansion.from_components(
            [], n_qubits=exp.n_qubits, merge=False, atol=atol
        )
        exp.log_mass = -math.inf

    if preset_noise_map is not None:
        merged_noise = {**preset_noise_map, **recorded_noise_map}
        n_ch = total_noise_channels if total_noise_channels is not None else (max(merged_noise) + 1 if merged_noise else 0)
        final_noise_choices = tuple(int(merged_noise.get(i, 0)) for i in range(n_ch))
    else:
        final_noise_choices = tuple(chosen_noise)

    return TrajectoryRunResult(
        expansion=exp,
        final_tableau=final_tableau,
        measurements=meas_record,
        branch_mass=exp.branch_mass(),
        log_branch_mass=exp.log_branch_mass(),
        fault_choices=preset_fault_choices if preset_fault_choices is not None else tuple(chosen_faults),
        pauli_noise_choices=final_noise_choices,
        reset_choices=tuple(chosen_resets),
        quantum_outcomes=quantum_outcomes,
        readout_flips=readout_flips,
        born_masses=tuple(born_masses),
        selector_bits=selector_bits,
        peak_components=peak_chi,
        peak_dim=peak_d,
        frame_events=frame_events,
    )


def run_direct(
    plan: DirectPlan,
    *,
    rng: np.random.Generator | int | None = None,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
    reset_outcomes: Sequence[int] | Mapping[int, int] | None = None,
    requested_outcomes: Mapping[int, int | bool] | None = None,
    requested_quantum_outcomes: Mapping[int, int | bool] | None = None,
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None = None,
    max_components: int | None = None,
    atol: float = DEFAULT_ATOL,
) -> TrajectoryRunResult:
    """`A2`: Execute a `DirectPlan` trajectory."""
    f_map, n_map, r_map, c_map, q_map = _prepare_run_maps(
        plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace,
        reset_outcomes=reset_outcomes, requested_outcomes=requested_outcomes,
        requested_quantum_outcomes=requested_quantum_outcomes,
    )
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    exp = ShotExpansion.zero(plan.num_qubits, atol=atol)
    return _execute_compiled_records(
        exp, plan.records, plan.num_measurements, plan.final_tableau,
        rng=rng_obj, fault_map=f_map, noise_map=n_map, reset_map=r_map,
        req_c_map=c_map, req_q_map=q_map, sweep_bits=sweep_bits,
        max_components=max_components, atol=atol,
        frame_events=plan.frame_events,
    )


def fault_selector_bits(plan: SelectorPlan, faults: Mapping[int, int]) -> NDArray[np.bool_]:
    """Build one-hot fault selector bit vector `bool[num_selector_qubits]` from slot choices."""
    sel = np.zeros(plan.num_selector_qubits, dtype=np.bool_)
    for s_idx, s_grp in enumerate(plan.fault_selector_groups):
        c_val = faults.get(s_idx, 0)
        if c_val > 0:
            sel[s_grp[c_val - 1] - plan.num_data_qubits] = True
    return sel


def _sample_selector_activations_from_maps(
    plan: SelectorPlan,
    rng_obj: np.random.Generator,
    fault_map: Mapping[int, int] | None,
    noise_map: Mapping[int, int] | None,
) -> tuple[NDArray[np.bool_], tuple[int, ...], dict[int, int]]:
    fault_choices = tuple(
        fault_map.get(slot_idx, 0) if fault_map is not None
        else _sample_categorical_branch([b.prob for b in ch.branches], rng_obj)
        for slot_idx, ch in enumerate(plan.fault_channels)
    )
    sel_bits = fault_selector_bits(plan, dict(enumerate(fault_choices)))

    noise_choice_map: dict[int, int] = {}
    if plan.gadgetize_pauli_noise and plan.noise_selector_groups:
        corr_active = False
        for (g_kind, probs_g, s_grp), ch_idx in zip(plan.noise_selector_groups, plan.noise_selector_channel_indices):
            if g_kind == "CAT":
                c_n = noise_map.get(ch_idx, 0) if noise_map is not None else _sample_categorical_branch(probs_g, rng_obj)
                noise_choice_map[ch_idx] = c_n
                if c_n > 0:
                    sel_bits[s_grp[c_n - 1] - plan.num_data_qubits] = True
            else:
                if g_kind == "CORR":
                    corr_active = False
                if corr_active:
                    c_c = 0
                else:
                    c_c = (
                        noise_map.get(ch_idx, 0) if noise_map is not None
                        else int(float(rng_obj.random()) < probs_g[0])
                    )
                    if c_c > 0:
                        corr_active = True
                        sel_bits[s_grp[0] - plan.num_data_qubits] = True
                noise_choice_map[ch_idx] = c_c

    return sel_bits, tuple(fault_choices), noise_choice_map


def sample_selector_activations(
    plan: SelectorPlan,
    rng: np.random.Generator | int | None = None,
    *,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
) -> NDArray[np.bool_]:
    """Sample joint one-hot `selector_bits` (`bool[num_selector_qubits]`) for `plan` at `t=0`."""
    if not isinstance(plan, SelectorPlan):
        raise TypeError(f"Expected SelectorPlan, got {type(plan)!r}")
    f_map, n_map, _, _, _ = _prepare_run_maps(plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace)
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return _sample_selector_activations_from_maps(plan, rng_obj, f_map, n_map)[0]


def run_selector(
    plan: SelectorPlan,
    *,
    rng: np.random.Generator | int | None = None,
    fault_trace: Sequence[int] | Mapping[int, int] | None = None,
    pauli_noise_trace: Sequence[int] | Mapping[int, int] | None = None,
    reset_outcomes: Sequence[int] | Mapping[int, int] | None = None,
    requested_outcomes: Mapping[int, int | bool] | None = None,
    requested_quantum_outcomes: Mapping[int, int | bool] | None = None,
    sweep_bits: Sequence[int | bool] | NDArray[np.bool_] | None = None,
    max_components: int | None = None,
    atol: float = DEFAULT_ATOL,
    extract_data: bool = False,
) -> TrajectoryRunResult:
    """`B2`: Execute `SelectorPlan` via joint one-hot selector `X_s` activations at `t=0`."""
    f_map, n_map, r_map, c_map, q_map = _prepare_run_maps(
        plan, fault_trace=fault_trace, pauli_noise_trace=pauli_noise_trace,
        reset_outcomes=reset_outcomes, requested_outcomes=requested_outcomes,
        requested_quantum_outcomes=requested_quantum_outcomes,
    )
    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    exp = ShotExpansion.zero(plan.num_total_qubits, atol=atol)
    sel_bits, fault_choices, gad_noise_map = _sample_selector_activations_from_maps(plan, rng_obj, f_map, n_map)
    n_w_tot = n_words_for_qubits(plan.num_total_qubits)
    zero_w = np.zeros(n_w_tot, dtype=np.uint64)
    for j in np.flatnonzero(sel_bits):
        s_active = plan.num_data_qubits + int(j)
        exp = apply_pauli(exp, CompiledPauliAxis(unit_bit_words(s_active, n_w_tot), zero_w, 0))

    res = _execute_compiled_records(
        exp, plan.records, plan.num_measurements, plan.final_tableau,
        rng=rng_obj, fault_map=None, noise_map=n_map, reset_map=r_map,
        req_c_map=c_map, req_q_map=q_map, sweep_bits=sweep_bits,
        max_components=max_components, atol=atol,
        preset_fault_choices=fault_choices,
        preset_noise_map=gad_noise_map if plan.gadgetize_pauli_noise else None,
        total_noise_channels=plan.total_noise_channels if plan.gadgetize_pauli_noise else None,
        selector_bits=sel_bits,
        frame_events=plan.data_frame_events if extract_data else plan.frame_events,
    )
    if extract_data:
        res.final_tableau, res.expansion = extract_data_expansion(
            plan, res.expansion, sel_bits
        )
    return res
