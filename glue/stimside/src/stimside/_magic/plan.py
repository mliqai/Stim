"""Circuit lowering passes, `C2` commuting phase-block compiler, and `DIRECT`/`SELECTOR` plans (`stimside._magic.plan`)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside._magic.channels import (
    CompiledOpenBlockRecord,
    CompiledRecord,
    CorrelatedNoiseRecord,
    DirectFaultSlotRecord,
    FEEDBACK_2Q_GATES,
    FeedbackPauliRecord,
    FrameEvent,
    HeraldedNoiseRecord,
    LeakageBarrierRecord,
    LeftSppEvent,
    LeftUnitaryEvent,
    MPadRecord,
    MagicPhaseTerm,
    MagicTermRecord,
    PAULI_2Q_ORDER,
    PauliNoiseRecord,
    RightRotEvent,
    compile_clifford_factors,
    compile_pauli_measurement_c15,
    compile_pauli_product,
    compile_reset_c15,
    emit_magic_terms,
    infer_total_data_qubits,
    reduce_pauli_product_targets,
    restrict_tableau_to_data,
    split_pauli_product_targets,
)
from stimside._magic.gates import (
    CONTROLLED_PAULI_2Q_AXES,
    CliffordFaultChannel,
    CompiledPauliAxis,
    DECOMP_1Q,
    DECOMP_2Q,
    FaultBranch,
    canonical_gate_data,
    decompose_clifford_exact,
    extract_circuit_faults,
    merge_and_validate_fault_channels,
    plain_qubits,
    spec_terms,
)
from stimside._magic.phase_qfe import (
    OMEGA_ARR as _OMEGA_ARR,
    OMEGA_POWERS,
    CompiledPauliBlock,
    CompiledPauliBlockStep,
    ShotExpansion,
)
from stimside.op_handlers.leakage_handlers.tag_registry import (
    check_tag_qubits,
    parse_leakage_tag_magic,
    validate_magic_op,
)
from stimside.util.bitwords import (
    n_words_for_qubits,
    pauli_multiply,
    tail_mask,
    word_dot_parity,
)
from stimside.util.reference_chp import ReferenceCHPTableau

_ANNOTATION_GATES: frozenset[str] = frozenset(
    {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
)
_CAT_NOISE_META: dict[str, tuple[int, int]] = {
    "X_ERROR": (1, 1), "Y_ERROR": (1, 1), "Z_ERROR": (1, 1),
    "DEPOLARIZE1": (1, 3), "PAULI_CHANNEL_1": (1, 3),
    "DEPOLARIZE2": (2, 15), "PAULI_CHANNEL_2": (2, 15),
}


def _instruction_num_measurements(op: stim.CircuitInstruction, canon_name: str) -> int:
    if not canonical_gate_data(canon_name).produces_measurements:
        return 0
    if canon_name == "MPP":
        return len(split_pauli_product_targets(op))
    n_t = len(op.targets_copy())
    return (n_t // 2) if canon_name in ("MXX", "MYY", "MZZ") else n_t


def _validate_and_emit_non_magic_tagged_op(
    op: stim.CircuitInstruction,
    num_qubits: int,
    next_meas_idx: int,
    out_records: list[CompiledRecord],
) -> int:
    """Validate a registered leakage tag (emitting `LeakageBarrierRecord`) or reject unknown tags."""
    tag = op.tag
    if not (tag.startswith("LEAKAGE") or tag.startswith("CONDITIONED_ON")):
        raise ValueError(f"Unrecognized instruction tag {tag!r} on {op!r}")
    params, _ = parse_leakage_tag_magic(op, syntax_only=True)
    tag_prefix = tag.split(":", 1)[0].strip()
    if params is not None:
        check_tag_qubits({op: params}, num_qubits)

    canon_name = canonical_gate_data(op.name).name
    n_meas = _instruction_num_measurements(op, canon_name)
    out_records.append(LeakageBarrierRecord(op, tag_prefix, n_meas))
    return next_meas_idx + n_meas


def _slice_axis_to_qubits(ax: CompiledPauliAxis, n_qubits: int) -> CompiledPauliAxis:
    n_w, tm = n_words_for_qubits(n_qubits), tail_mask(n_qubits)
    a_d = np.ascontiguousarray(ax.a_words[:n_w].copy(), dtype=np.uint64)
    b_d = np.ascontiguousarray(ax.b_words[:n_w].copy(), dtype=np.uint64)
    a_d[-1] &= tm
    b_d[-1] &= tm
    return CompiledPauliAxis(a_words=a_d, b_words=b_d, kappa=ax.kappa)


def _count_gadget_pauli_selectors(flat_ops: Sequence[stim.CircuitInstruction]) -> int:
    total = 0
    for op in flat_ops:
        if op.tag != "":
            continue
        cname = canonical_gate_data(op.name).name
        if cname in _CAT_NOISE_META:
            arity, k_br = _CAT_NOISE_META[cname]
            total += k_br * (len(op.targets_copy()) // arity)
        elif cname in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
            total += 1
    return total


def _emit_selector_pauli_branch(
    s_qubit: int,
    pauli_terms: Sequence[tuple[int, str]],
    n_data: int,
    ref_sel: ReferenceCHPTableau,
    frame_events: list[FrameEvent],
    gauge_steps: list[tuple[int, CompiledPauliAxis]],
) -> None:
    """Synthesize `C_s(P_data)` on `ref_sel` and record data-qubit gauge step."""
    for q_t, pch in pauli_terms:
        if pch != "I":
            gauge_steps.append((s_qubit, _slice_axis_to_qubits(compile_pauli_product(ref_sel, [(q_t, pch)]), n_data)))
            c_gate = "C" + pch
            ref_sel.do_unitary(c_gate, [s_qubit, q_t])
            frame_events.append(LeftUnitaryEvent(c_gate, (s_qubit, q_t)))


def _emit_selector_clifford_branch(
    s_qubit: int,
    branch: FaultBranch,
    n_data: int,
    ref_sel: ReferenceCHPTableau,
    raw_records: list[CompiledRecord],
    frame_events: list[FrameEvent],
    gauge_steps: list[tuple[int, CompiledPauliAxis]],
) -> None:
    """Synthesize `C_s(F)` (`B1`), using 1-gate `CCZ` shortcut for all 9 2Q controlled Paulis."""
    g, qs = branch.gate_name, branch.targets
    if g in CONTROLLED_PAULI_2Q_AXES:
        ax0_ch, ax1_ch = CONTROLLED_PAULI_2Q_AXES[g]
        axes = (
            compile_pauli_product(ref_sel, [(s_qubit, "Z")]),
            compile_pauli_product(ref_sel, [(qs[0], ax0_ch)]),
            compile_pauli_product(ref_sel, [(qs[1], ax1_ch)]),
        )
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=axes, kind="CCZ")))
        return

    dec = decompose_clifford_exact(g, qs)
    p0_neg, p0_terms = spec_terms(dec.p0_spec, qs)
    v_eff = (dec.v + (4 if p0_neg else 0)) & 7
    _emit_selector_pauli_branch(s_qubit, p0_terms, n_data, ref_sel, frame_events, gauge_steps)

    for r_sp in dec.r_specs:
        neg_r, r_terms = spec_terms(r_sp, qs)
        bar_P = compile_pauli_product(ref_sel, r_terms, negate=neg_r)
        bar_ZsP = compile_pauli_product(ref_sel, [(s_qubit, "Z")] + r_terms, negate=neg_r)
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_ZsP,), kind="T", epsilon=1)))
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_P,), kind="T", epsilon=-1)))

    if v_eff != 0:
        bar_Zs = compile_pauli_product(ref_sel, [(s_qubit, "Z")])
        kind: Literal["T", "Z8_DIAG"] = "T" if v_eff in (1, 7) else "Z8_DIAG"
        eps = (1 if v_eff == 1 else -1) if kind == "T" else v_eff
        raw_records.append(MagicTermRecord(MagicPhaseTerm(axes=(bar_Zs,), kind=kind, epsilon=eps)))


@dataclass(frozen=True)
class _CircuitPassResult:
    n_data: int
    n_sel: int
    n_total: int
    num_measurements: int
    raw_records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    fault_sel_groups: tuple[tuple[int, ...], ...]
    noise_sel_groups: tuple[tuple[str, tuple[float, ...], tuple[int, ...]], ...]
    noise_sel_ch_indices: tuple[int, ...]
    total_noise_channels: int
    noise_max_choices: tuple[int, ...]
    num_resets: int
    gauge_steps: tuple[tuple[int, CompiledPauliAxis], ...]
    data_frame_events: tuple[FrameEvent, ...] = ()
    data_final_tableau: ReferenceCHPTableau | None = None


def _compile_circuit_pass(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    selector_mode: bool = False,
    gadgetize_pauli_noise: bool = False,
) -> _CircuitPassResult:
    """Shared single-pass lowering for `compile_coordinates`, `compile_direct`, and `compile_selector`."""
    flat_ops, inline_faults = extract_circuit_faults(circuit)
    prog_by_step, inline_by_step, all_channels = merge_and_validate_fault_channels(
        len(flat_ops), inline_faults, fault_channels
    )
    n_data = infer_total_data_qubits(circuit, all_channels)
    gad_noise = selector_mode and gadgetize_pauli_noise
    n_fault_sel = sum(len(ch.branches) for ch in all_channels) if selector_mode else 0
    n_sel = n_fault_sel + (_count_gadget_pauli_selectors(flat_ops) if gad_noise else 0)
    n_total = n_data + n_sel

    ref_chp = ReferenceCHPTableau(n_total)
    raw_records: list[CompiledRecord] = []
    frame_events: list[FrameEvent] = []
    gauge_steps: list[tuple[int, CompiledPauliAxis]] = []
    fault_sel_groups: list[tuple[int, ...]] = []
    noise_sel_groups: list[tuple[str, tuple[float, ...], tuple[int, ...]]] = []
    noise_sel_ch_indices: list[int] = []
    noise_max_choices: list[int] = []
    next_sel_q = n_data
    next_meas_idx = 0
    next_noise_idx = 0
    num_resets = 0

    def _emit_fault_slots(entries: Sequence[tuple[int, CliffordFaultChannel]]) -> None:
        nonlocal next_sel_q
        for slot_idx, ch in entries:
            if selector_mode:
                s_grp = tuple(range(next_sel_q, next_sel_q + len(ch.branches)))
                next_sel_q += len(ch.branches)
                fault_sel_groups.append(s_grp)
                for s_q, br in zip(s_grp, ch.branches):
                    _emit_selector_clifford_branch(
                        s_q, br, n_data, ref_chp, raw_records, frame_events, gauge_steps
                    )
            else:
                b_factors = tuple(compile_clifford_factors(ref_chp, b.gate_name, b.targets) for b in ch.branches)
                raw_records.append(
                    DirectFaultSlotRecord(slot_idx, b_factors, np.array([b.prob for b in ch.branches], dtype=np.float64))
                )

    def _emit_cat_pauli_noise(
        kind: str, probs_tup: tuple[float, ...], branch_term_groups: Sequence[Sequence[tuple[int, str]]]
    ) -> None:
        nonlocal next_sel_q, next_noise_idx
        k_br = len(probs_tup)
        if gad_noise:
            s_grp = tuple(range(next_sel_q, next_sel_q + k_br))
            next_sel_q += k_br
            noise_sel_groups.append(("CAT", probs_tup, s_grp))
            noise_sel_ch_indices.append(next_noise_idx)
            for s_q, b_terms in zip(s_grp, branch_term_groups):
                _emit_selector_pauli_branch(s_q, b_terms, n_data, ref_chp, frame_events, gauge_steps)
        else:
            b_axes = tuple(compile_pauli_product(ref_chp, b_terms) for b_terms in branch_term_groups)
            raw_records.append(PauliNoiseRecord(kind, probs_tup, b_axes, next_noise_idx))
        noise_max_choices.append(k_br)
        next_noise_idx += 1

    def _emit_measurements(groups: Sequence[tuple[Sequence[tuple[int, str]], bool]], p_flip: float) -> None:
        nonlocal next_meas_idx
        for grp_terms, grp_inv in groups:
            m_rec = compile_pauli_measurement_c15(
                ref_chp, grp_terms, invert_result=grp_inv, flip_prob=p_flip,
                meas_index=next_meas_idx, frame_events=frame_events,
            )
            raw_records.append(m_rec)
            if selector_mode and m_rec.rot_axis is not None:
                gauge_steps.append((-1, _slice_axis_to_qubits(m_rec.rot_axis, n_data)))
            next_meas_idx += 1

    for step_idx, op in enumerate(flat_ops):
        if step_idx in prog_by_step:
            _emit_fault_slots(prog_by_step[step_idx])

        if op.tag != "":
            if op.tag in ("T", "T_DAG", "CCZ"):
                emit_magic_terms(op.tag, validate_magic_op(op), ref_chp, raw_records)
            else:
                next_meas_idx = _validate_and_emit_non_magic_tagged_op(
                    op, n_data, next_meas_idx, raw_records
                )
        else:
            name = op.name
            canon_name = canonical_gate_data(name).name
            args = op.gate_args_copy()
            targs = op.targets_copy()

            if canon_name in _ANNOTATION_GATES or canon_name in ("I_ERROR", "II_ERROR"):
                pass
            elif canon_name in ("SPP", "SPP_DAG"):
                dag = canon_name == "SPP_DAG"
                for grp_terms, grp_inv in split_pauli_product_targets(op):
                    ref_chp.do_spp(grp_terms, negate=grp_inv, dag=dag)
                    frame_events.append(LeftSppEvent(tuple(grp_terms), grp_inv, dag))
            elif canon_name in FEEDBACK_2Q_GATES and any(
                t.is_measurement_record_target or t.is_sweep_bit_target for t in targs
            ):
                ax0, ax1 = FEEDBACK_2Q_GATES[canon_name]
                for off in range(0, len(targs), 2):
                    t0, t1 = targs[off], targs[off + 1]
                    q0_plain = t0.is_qubit_target and not t0.is_inverted_result_target
                    q1_plain = t1.is_qubit_target and not t1.is_inverted_result_target
                    if q0_plain and q1_plain:
                        q_pair = (int(t0.qubit_value), int(t1.qubit_value))
                        ref_chp.do_unitary(canon_name, list(q_pair))
                        frame_events.append(LeftUnitaryEvent(canon_name, q_pair))
                        continue
                    if (t0.is_measurement_record_target or t0.is_sweep_bit_target) and q1_plain and ax0 == "Z":
                        t_src, q_dst, p_dst = t0, int(t1.qubit_value), ax1
                    elif (t1.is_measurement_record_target or t1.is_sweep_bit_target) and q0_plain and ax1 == "Z":
                        t_src, q_dst, p_dst = t1, int(t0.qubit_value), ax0
                    else:
                        raise ValueError(f"Unsupported classical feedback target pair {(t0, t1)} on {name!r}")
                    src_kind: Literal["rec", "sweep"] = "rec" if t_src.is_measurement_record_target else "sweep"
                    raw_records.append(
                        FeedbackPauliRecord(
                            src_kind, int(t_src.value), next_meas_idx,
                            compile_pauli_product(ref_chp, [(q_dst, p_dst)]),
                        )
                    )
            elif canon_name in DECOMP_1Q or canon_name in DECOMP_2Q:
                arity = 1 if canon_name in DECOMP_1Q else 2
                qs_u = plain_qubits(targs, name)
                if canon_name not in ("I", "II"):
                    for off in range(0, len(qs_u), arity):
                        grp_u = tuple(qs_u[off : off + arity])
                        ref_chp.do_unitary(canon_name, list(grp_u))
                        frame_events.append(LeftUnitaryEvent(canon_name, grp_u))
            elif canon_name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                probs_1, pch = (float(args[0]),), canon_name[0]
                for t in targs:
                    _emit_cat_pauli_noise(canon_name, probs_1, ([(int(t.qubit_value), pch)],))
            elif canon_name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
                probs_3 = (
                    (float(args[0]) / 3.0,) * 3 if canon_name == "DEPOLARIZE1"
                    else (float(args[0]), float(args[1]), float(args[2]))
                )
                for t in targs:
                    q = int(t.qubit_value)
                    _emit_cat_pauli_noise(canon_name, probs_3, tuple([(q, pch)] for pch in "XYZ"))
            elif canon_name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
                probs_15 = (
                    (float(args[0]) / 15.0,) * 15 if canon_name == "DEPOLARIZE2"
                    else tuple(float(x) for x in args[:15])
                )
                for off in range(0, len(targs), 2):
                    q0, q1 = int(targs[off].qubit_value), int(targs[off + 1].qubit_value)
                    _emit_cat_pauli_noise(
                        canon_name, probs_15, tuple([(q0, pair[0]), (q1, pair[1])] for pair in PAULI_2Q_ORDER)
                    )
            elif canon_name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
                probs_h = (
                    (float(args[0]) * 0.25,) * 4 if canon_name == "HERALDED_ERASE"
                    else (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
                for t in targs:
                    q = int(t.qubit_value)
                    b_axes: tuple[CompiledPauliAxis | None, ...] = (
                        None,
                        compile_pauli_product(ref_chp, [(q, "X")]),
                        compile_pauli_product(ref_chp, [(q, "Y")]),
                        compile_pauli_product(ref_chp, [(q, "Z")]),
                    )
                    raw_records.append(HeraldedNoiseRecord(canon_name, next_meas_idx, probs_h, b_axes, next_noise_idx))
                    noise_max_choices.append(4)
                    next_meas_idx += 1
                    next_noise_idx += 1
            elif canon_name in ("E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"):
                p, is_else = float(args[0]), (canon_name == "ELSE_CORRELATED_ERROR")
                simp_terms = reduce_pauli_product_targets(
                    [tm for g_terms, _ in split_pauli_product_targets(op) for tm in g_terms]
                )
                if gad_noise:
                    s_q = next_sel_q
                    next_sel_q += 1
                    noise_sel_groups.append(("ELSE_CORR" if is_else else "CORR", (p,), (s_q,)))
                    noise_sel_ch_indices.append(next_noise_idx)
                    _emit_selector_pauli_branch(s_q, simp_terms, n_data, ref_chp, frame_events, gauge_steps)
                else:
                    raw_records.append(
                        CorrelatedNoiseRecord(is_else, p, compile_pauli_product(ref_chp, simp_terms), next_noise_idx)
                    )
                noise_max_choices.append(1)
                next_noise_idx += 1
            elif canon_name in ("M", "MZ", "MX", "MY", "MXX", "MYY", "MZZ", "MPP"):
                p_flip = float(args[0]) if args else 0.0
                if canon_name == "MPP":
                    groups_m: Sequence[tuple[Sequence[tuple[int, str]], bool]] = split_pauli_product_targets(op)
                elif canon_name in ("MXX", "MYY", "MZZ"):
                    pch = canon_name[1]
                    groups_m = [
                        (
                            [(int(targs[off].qubit_value), pch), (int(targs[off + 1].qubit_value), pch)],
                            bool(targs[off].is_inverted_result_target) ^ bool(targs[off + 1].is_inverted_result_target),
                        )
                        for off in range(0, len(targs), 2)
                    ]
                else:
                    pch = "Z" if canon_name in ("M", "MZ") else canon_name[1]
                    groups_m = [([(int(t.qubit_value), pch)], bool(t.is_inverted_result_target)) for t in targs]
                _emit_measurements(groups_m, p_flip)
            elif canon_name == "MPAD":
                p_flip = float(args[0]) if args else 0.0
                if not math.isfinite(p_flip) or p_flip < 0.0 or p_flip > 1.0:
                    raise ValueError(f"Invalid MPAD flip probability {p_flip!r}")
                for t in targs:
                    raw_records.append(MPadRecord(next_meas_idx, bool(t.value) ^ bool(t.is_inverted_result_target), p_flip))
                    next_meas_idx += 1
            elif canon_name in ("R", "RZ", "RX", "RY", "MR", "MRZ", "MRX", "MRY"):
                is_mr = canon_name.startswith("MR")
                p_flip = float(args[0]) if (is_mr and args) else 0.0
                basis_ch: Literal["Z", "X", "Y"] = (
                    "X" if canon_name.endswith("X") else ("Y" if canon_name.endswith("Y") else "Z")
                )
                for t in targs:
                    m_idx = next_meas_idx if is_mr else None
                    if is_mr:
                        next_meas_idx += 1
                    r_rec = compile_reset_c15(
                        ref_chp, int(t.qubit_value), basis_ch, meas_index=m_idx,
                        invert_result=bool(t.is_inverted_result_target) if is_mr else False,
                        flip_prob=p_flip, frame_events=frame_events,
                    )
                    raw_records.append(r_rec)
                    num_resets += 1
                    if selector_mode and r_rec.rot_axis is not None:
                        gauge_steps.append((-1, _slice_axis_to_qubits(r_rec.rot_axis, n_data)))
            else:
                raise ValueError(f"Unsupported Stim instruction {op!r}")

        if step_idx in inline_by_step:
            _emit_fault_slots(inline_by_step[step_idx])

    if len(flat_ops) in prog_by_step:
        _emit_fault_slots(prog_by_step[len(flat_ops)])

    frame_tup = tuple(frame_events)
    data_events: tuple[FrameEvent, ...] = ()
    data_tab: ReferenceCHPTableau | None = None
    if selector_mode:
        data_events = tuple(
            RightRotEvent(_slice_axis_to_qubits(ev.rot_axis, n_data)) if isinstance(ev, RightRotEvent) else ev
            for ev in frame_tup
            if not (isinstance(ev, LeftUnitaryEvent) and any(q >= n_data for q in ev.targets))
        )
        data_tab = restrict_tableau_to_data(ref_chp, n_data)
    return _CircuitPassResult(
        n_data, n_sel, n_total, next_meas_idx, tuple(raw_records), ref_chp, frame_tup,
        all_channels, tuple(fault_sel_groups), tuple(noise_sel_groups), tuple(noise_sel_ch_indices),
        next_noise_idx, tuple(noise_max_choices), num_resets, tuple(gauge_steps), data_events, data_tab,
    )


def compile_coordinates(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
) -> DirectPlan:
    """`C1` + `C1.5`: Compile circuit into moving-reference records and `ReferenceCHPTableau`."""
    res = _compile_circuit_pass(circuit, fault_channels=fault_channels, selector_mode=False)
    return DirectPlan(
        res.n_data, res.num_measurements, res.raw_records, res.final_tableau,
        res.frame_events, res.fault_channels, res.raw_records, res.noise_max_choices, res.num_resets,
    )


# ---------------------------------------------------------------------------
# C2: Commuting Diagonal Phase Open-Block Compiler (`compile_open_block`)
# ---------------------------------------------------------------------------

_DEG1_SEQ: dict[int, tuple[int, tuple[tuple[Literal["P1", "P2", "N1"], int], ...]]] = {
    1: (0, (("N1", 1),)),
    2: (1, (("P2", -1),)),
    3: (0, (("P1", 1), ("N1", -1))),
    4: (0, (("P1", 1),)),
    5: (0, (("P1", 1), ("N1", 1))),
    6: (7, (("P2", 1),)),
    7: (0, (("N1", -1),)),
}


def _axes_commute(ax1: CompiledPauliAxis, ax2: CompiledPauliAxis) -> bool:
    return word_dot_parity(ax1.a_words, ax2.b_words) == word_dot_parity(ax2.a_words, ax1.b_words)


def _canonical_unsigned_axis(ax: CompiledPauliAxis) -> tuple[CompiledPauliAxis, bool]:
    ab_par = word_dot_parity(ax.a_words, ax.b_words)
    diff = (ax.kappa - ab_par) & 3
    if diff == 0:
        return ax, False
    if diff == 2:
        return CompiledPauliAxis(ax.a_words, ax.b_words, ab_par), True
    raise ValueError(f"Non-Hermitian CompiledPauliAxis: {ax}")


def _monomial_to_sequential_steps(
    axes: Sequence[CompiledPauliAxis],
    mask: int,
    coeff_mod8: int,
) -> tuple[int, list[CompiledPauliBlockStep]]:
    """Convert one simplified Z_8 monomial `(mask, coeff_mod8)` into exact sequential steps."""
    bits = [u for u in range(len(axes)) if (mask >> u) & 1]
    deg, c = len(bits), (coeff_mod8 & 7)
    if c == 0 or deg == 0:
        return 0, []
    if deg == 1:
        ax = axes[bits[0]].as_tuple()
        dv, spec = _DEG1_SEQ[c]
        return dv, [CompiledPauliBlockStep(kind=kind, paulis=(ax,), sign=sgn) for kind, sgn in spec]
    if deg == 2 and c == 4:
        p0, p1 = axes[bits[0]], axes[bits[1]]
        p01 = pauli_multiply(p0.a_words, p0.b_words, p0.kappa, p1.a_words, p1.b_words, p1.kappa)
        return 1, [
            CompiledPauliBlockStep(kind="P2", paulis=(p1.as_tuple(),), sign=-1),
            CompiledPauliBlockStep(kind="P2", paulis=(p0.as_tuple(),), sign=-1),
            CompiledPauliBlockStep(kind="P2", paulis=(p01,), sign=1),
        ]
    if deg == 3 and c == 4:
        return 0, [CompiledPauliBlockStep(kind="N2", paulis=tuple(axes[u].as_tuple() for u in bits))]
    raise ValueError(f"Unsupported sequential monomial degree={deg}, coeff_mod8={c}")


def _walsh_expand_subblock(
    n_qubits: int,
    axes: Sequence[CompiledPauliAxis],
    poly_items: Sequence[tuple[int, int]],
    global_v_mod8: int,
) -> CompiledPauliBlock:
    """Compute exact 2^r Walsh-Hadamard Pauli sum for `r` commuting axes."""
    r = len(axes)
    n_w = n_words_for_qubits(n_qubits)
    if r == 0:
        return CompiledPauliBlock(
            n_qubits=n_qubits, mode="pauli_sum", global_scalar=OMEGA_POWERS[global_v_mod8 & 7],
            is_unitary=True, is_identity=True,
        )
    dim = 1 << r
    s_grid = np.arange(dim, dtype=np.uint64)
    exps = np.full(dim, global_v_mod8 & 7, dtype=np.uint8)
    for m_mask, c_val in poly_items:
        m_u64 = np.uint64(m_mask)
        active = (s_grid & m_u64) == m_u64
        exps[active] = (exps[active] + np.uint8(c_val & 7)) & np.uint8(7)

    vals = _OMEGA_ARR[exps].copy()
    step = 1
    while step < dim:
        vals_2d = vals.reshape(-1, 2 * step)
        u, v = vals_2d[:, :step].copy(), vals_2d[:, step:].copy()
        vals_2d[:, :step] = u + v
        vals_2d[:, step:] = u - v
        step <<= 1
    vals *= math.ldexp(1.0, -r)

    nz_masks = np.flatnonzero(np.abs(vals) > 1e-15)
    prod_a = np.zeros((dim, n_w), dtype=np.uint64)
    prod_b = np.zeros((dim, n_w), dtype=np.uint64)
    prod_k = np.zeros(dim, dtype=np.uint8)
    for bit_idx, ax in enumerate(axes):
        bit = 1 << bit_idx
        for prev in range(bit):
            a_p, b_p, k_p = pauli_multiply(
                prod_a[prev], prod_b[prev], int(prod_k[prev]), ax.a_words, ax.b_words, ax.kappa
            )
            prod_a[prev | bit], prod_b[prev | bit], prod_k[prev | bit] = a_p, b_p, np.uint8(k_p)

    return CompiledPauliBlock(
        n_qubits=n_qubits, mode="pauli_sum",
        coeffs=np.ascontiguousarray(vals[nz_masks], dtype=np.complex128),
        a_words=np.ascontiguousarray(prod_a[nz_masks], dtype=np.uint64),
        b_words=np.ascontiguousarray(prod_b[nz_masks], dtype=np.uint64),
        kappas=np.ascontiguousarray(prod_k[nz_masks], dtype=np.uint8),
        is_unitary=True, is_identity=False,
    )


def compile_open_block(
    records: Sequence[CompiledRecord] | DirectPlan,
    *,
    n_qubits: int | None = None,
    budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> tuple[CompiledRecord, ...]:
    """`C2`: Fuse consecutive pairwise-commuting `MagicTermRecord`s, never crossing barriers."""
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    if isinstance(records, DirectPlan):
        if n_qubits is None:
            n_qubits = records.num_qubits
        rec_seq: Sequence[CompiledRecord] = records.raw_records
    else:
        rec_seq = records
    if n_qubits is None:
        raise ValueError("compile_open_block requires explicit n_qubits when records is not DirectPlan")
    n_q = int(n_qubits)

    out_records: list[CompiledRecord] = []
    cur_axes: list[CompiledPauliAxis] = []
    cur_axis_map: dict[tuple[bytes, bytes], int] = {}
    cur_poly: dict[int, int] = {}
    cur_v_mod8: int = 0
    has_terms: bool = False

    def _reset_cur() -> None:
        nonlocal cur_v_mod8, has_terms
        cur_axes.clear()
        cur_axis_map.clear()
        cur_poly.clear()
        has_terms = False
        cur_v_mod8 = 0

    def _flush() -> None:
        if not has_terms and cur_v_mod8 == 0 and not cur_poly:
            return
        surviving = {m: (c & 7) for m, c in cur_poly.items() if (c & 7) != 0 and m != 0}
        v0 = cur_v_mod8 & 7
        if not surviving and v0 == 0:
            _reset_cur()
            return

        used_mask = 0
        for m in surviving:
            used_mask |= m
        old_to_new: dict[int, int] = {}
        pruned_axes: list[CompiledPauliAxis] = []
        for old_u, ax in enumerate(cur_axes):
            if (used_mask >> old_u) & 1:
                old_to_new[old_u] = len(pruned_axes)
                pruned_axes.append(ax)

        sorted_items = sorted(
            (sum(1 << new_u for old_u, new_u in old_to_new.items() if (m >> old_u) & 1), c)
            for m, c in surviving.items()
        )
        axes_tup = tuple(pruned_axes)
        r = len(axes_tup)
        block_obj: CompiledPauliBlock | None = None
        if r == 0:
            block_obj = CompiledPauliBlock(
                n_qubits=n_q, mode="pauli_sum", global_scalar=OMEGA_POWERS[v0],
                is_unitary=True, is_identity=True,
            )
        elif budget >= 2 and r <= c2_max_rank and (1 << r) <= max(4096, budget * 16):
            cand = _walsh_expand_subblock(n_q, axes_tup, sorted_items, v0)
            assert cand.coeffs is not None
            if len(cand.coeffs) <= budget:
                block_obj = cand

        if block_obj is None:
            v_seq = v0
            seq_steps: list[CompiledPauliBlockStep | CompiledPauliBlock] = []
            for m_item, c_item in sorted_items:
                dv, st_list = _monomial_to_sequential_steps(axes_tup, m_item, c_item)
                v_seq = (v_seq + dv) & 7
                seq_steps.extend(st_list)
            block_obj = CompiledPauliBlock(
                n_qubits=n_q, mode="sequential", steps=tuple(seq_steps),
                global_scalar=OMEGA_POWERS[v_seq], is_unitary=True,
                is_identity=(len(seq_steps) == 0),
            )

        out_records.append(CompiledOpenBlockRecord(block_obj, axes_tup))
        _reset_cur()

    for rec in rec_seq:
        if not isinstance(rec, MagicTermRecord):
            _flush()
            out_records.append(rec)
            continue

        term = rec.term
        norm_infos = [_canonical_unsigned_axis(ax) for ax in term.axes]
        for i in range(len(norm_infos)):
            for j in range(i + 1, len(norm_infos)):
                if not _axes_commute(norm_infos[i][0], norm_infos[j][0]):
                    raise ValueError("Non-commuting axes inside single MagicPhaseTerm")

        new_unique_keys: set[tuple[bytes, bytes]] = set()
        commutes_with_all = True
        for c_ax, _ in norm_infos:
            k_bytes = (c_ax.a_words.tobytes(), c_ax.b_words.tobytes())
            if k_bytes not in cur_axis_map:
                new_unique_keys.add(k_bytes)
                if any(not _axes_commute(c_ax, ex_ax) for ex_ax in cur_axes):
                    commutes_with_all = False
                    break

        if (
            not commutes_with_all
            or (len(cur_axes) + len(new_unique_keys)) > max(3, c2_max_rank)
            or (len(cur_poly) + 8) > max(8, c2_max_poly_terms)
        ):
            _flush()

        u_indices: list[int] = []
        neg_flags: list[bool] = []
        for c_ax, neg_f in norm_infos:
            k_bytes = (c_ax.a_words.tobytes(), c_ax.b_words.tobytes())
            u_idx = cur_axis_map.get(k_bytes)
            if u_idx is None:
                u_idx = len(cur_axes)
                cur_axes.append(c_ax)
                cur_axis_map[k_bytes] = u_idx
            u_indices.append(u_idx)
            neg_flags.append(neg_f)

        has_terms = True
        if term.kind in ("T", "Z8_DIAG"):
            m0, c_val = 1 << u_indices[0], int(term.epsilon) & 7
            if neg_flags[0]:
                cur_v_mod8 = (cur_v_mod8 + c_val) & 7
                cur_poly[m0] = (cur_poly.get(m0, 0) - c_val) & 7
            else:
                cur_poly[m0] = (cur_poly.get(m0, 0) + c_val) & 7
        elif term.kind == "CCZ":
            for sub in range(8):
                if any(not ((sub >> j) & 1) and not neg_flags[j] for j in range(3)):
                    continue
                m_acc = 0
                for j in range(3):
                    if (sub >> j) & 1:
                        m_acc |= 1 << u_indices[j]
                if m_acc == 0:
                    cur_v_mod8 = (cur_v_mod8 + 4) & 7
                else:
                    cur_poly[m_acc] = (cur_poly.get(m_acc, 0) + 4) & 7
        else:
            raise ValueError(f"Unknown MagicPhaseTerm kind: {term.kind!r}")

    _flush()
    return tuple(out_records)


# ---------------------------------------------------------------------------
# Plans & Trajectory Results (`DIRECT` and `SELECTOR`)
# ---------------------------------------------------------------------------


@dataclass
class DirectPlan:
    num_qubits: int
    num_measurements: int
    records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    raw_records: tuple[CompiledRecord, ...] = ()
    noise_channel_max_choices: tuple[int, ...] = ()
    num_resets: int = 0


@dataclass
class SelectorPlan:
    num_data_qubits: int
    num_selector_qubits: int
    num_total_qubits: int
    num_measurements: int
    records: tuple[CompiledRecord, ...]
    final_tableau: ReferenceCHPTableau
    data_final_tableau: ReferenceCHPTableau
    frame_events: tuple[FrameEvent, ...]
    data_frame_events: tuple[FrameEvent, ...]
    fault_channels: tuple[CliffordFaultChannel, ...]
    fault_selector_groups: tuple[tuple[int, ...], ...]
    noise_selector_groups: tuple[tuple[str, tuple[float, ...], tuple[int, ...]], ...] = ()
    gadgetize_pauli_noise: bool = False
    raw_records: tuple[CompiledRecord, ...] = ()
    noise_selector_channel_indices: tuple[int, ...] = ()
    total_noise_channels: int = 0
    noise_channel_max_choices: tuple[int, ...] = ()
    num_resets: int = 0
    _gauge_steps: tuple[tuple[int, CompiledPauliAxis], ...] = ()


@dataclass
class TrajectoryRunResult:
    expansion: ShotExpansion
    final_tableau: ReferenceCHPTableau
    measurements: NDArray[np.bool_]
    branch_mass: float
    log_branch_mass: float
    fault_choices: tuple[int, ...]
    pauli_noise_choices: tuple[int, ...] = ()
    reset_choices: tuple[int, ...] = ()
    quantum_outcomes: dict[int, int] = field(default_factory=dict)
    readout_flips: dict[int, bool] = field(default_factory=dict)
    born_masses: tuple[tuple[int | None, float, float, float], ...] = ()
    selector_bits: NDArray[np.bool_] | None = None
    peak_components: int = 1
    peak_dim: int = 0
    frame_events: tuple[FrameEvent, ...] | None = None


def compile_direct(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    c2_budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> DirectPlan:
    """`A1`: Compile `circuit` for zero-branching `DIRECT` trajectory execution."""
    coord = compile_coordinates(circuit, fault_channels=fault_channels)
    records = compile_open_block(
        coord, budget=c2_budget, c2_max_rank=c2_max_rank, c2_max_poly_terms=c2_max_poly_terms,
    )
    return replace(coord, records=records)


def compile_selector(
    circuit: stim.Circuit,
    fault_channels: Sequence[CliffordFaultChannel] | None = None,
    *,
    gadgetize_pauli_noise: bool = False,
    c2_budget: int = 64,
    c2_max_rank: int = 12,
    c2_max_poly_terms: int = 64,
) -> SelectorPlan:
    """`B1`: Compile `circuit` into private-selector-ancilla template (`SELECTOR` mode)."""
    res = _compile_circuit_pass(
        circuit, fault_channels=fault_channels, selector_mode=True, gadgetize_pauli_noise=gadgetize_pauli_noise
    )
    assert res.data_final_tableau is not None
    records = compile_open_block(
        res.raw_records, n_qubits=res.n_total,
        budget=c2_budget, c2_max_rank=c2_max_rank, c2_max_poly_terms=c2_max_poly_terms,
    )
    return SelectorPlan(
        res.n_data, res.n_sel, res.n_total, res.num_measurements, records,
        res.final_tableau, res.data_final_tableau, res.frame_events, res.data_frame_events,
        res.fault_channels, res.fault_sel_groups, res.noise_sel_groups, gadgetize_pauli_noise,
        res.raw_records, res.noise_sel_ch_indices, res.total_noise_channels,
        res.noise_max_choices, res.num_resets, res.gauge_steps,
    )
