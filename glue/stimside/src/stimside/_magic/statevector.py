"""Matrix-free data-qubit extraction and dense `complex128` statevector reconstruction (`stimside._magic.statevector`)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from stimside._magic.channels import (
    FrameEvent,
    LeftResetXEvent,
    LeftSppEvent,
    LeftUnitaryEvent,
    RightRotEvent,
    _restrict_tableau_to_data,
)
from stimside._magic.gates import (
    CompiledPauliAxis,
    decompose_clifford_exact,
    eval_decomposition_unitary_le,
)
from stimside._magic.phase_qfe import (
    I_POWERS_ARR as _PHASE4_ARR,
    INV_SQRT2 as _INV_SQRT2,
    OMEGA_POWERS,
    ShotExpansion,
    _apply_local_unitary_c128,
    apply_pauli,
    apply_reference_tableau_to_statevector,
    merge_frames,
    project_diagonal,
)
from stimside._magic.plan import DirectPlan, SelectorPlan, TrajectoryRunResult
from stimside.util.bitwords import (
    first_set_bit_in_words,
    get_bit_in_words,
    n_words_for_qubits,
    pauli_multiply,
    tail_mask,
    unit_bit_words,
    word_dot_parity,
    words_any_nonzero,
)
from stimside.util.reference_chp import ReferenceCHPTableau


def _slice_expansion_to_data_qubits(
    expansion: ShotExpansion,
    n_data: int,
    sel_arr: NDArray[np.bool_],
) -> ShotExpansion:
    """Project/verify selector bits and slice `ShotExpansion` to `0 .. n_data - 1`."""
    if not isinstance(expansion, ShotExpansion):
        raise TypeError(f"Expected ShotExpansion, got {type(expansion)!r}")
    n_total, n_sel = expansion.n_qubits, len(sel_arr)
    if n_data + n_sel != n_total:
        raise ValueError(f"n_data ({n_data}) + len(selector_bits) ({n_sel}) != expansion.n_qubits ({n_total})")
    work, n_w_tot = expansion, n_words_for_qubits(n_total)
    for j in range(n_sel):
        s_q, want_bit = n_data + j, int(sel_arr[j])
        has_x = work.d > 0 and any(get_bit_in_words(work.A_words[r], s_q) != 0 for r in range(work.d))
        mismatch_h = work.chi > 0 and any(get_bit_in_words(work.h_words[c], s_q) != want_bit for c in range(work.chi))
        if has_x or mismatch_h:
            proj_exp, _ = project_diagonal(
                work, unit_bit_words(s_q, n_w_tot), 0, want_bit
            )
            proj_exp.log_mass = work.log_mass
            work = proj_exp

    n_w_data, tm_data = n_words_for_qubits(n_data), tail_mask(n_data)
    A_d = np.ascontiguousarray(work.A_words[:, :n_w_data].copy(), dtype=np.uint64)
    if work.d > 0:
        A_d[:, -1] &= tm_data
    h_d = np.ascontiguousarray(work.h_words[:, :n_w_data].copy(), dtype=np.uint64)
    if work.chi > 0:
        h_d[:, -1] &= tm_data
    data_exp = work.copy()
    data_exp.n_qubits, data_exp.A_words, data_exp.h_words = n_data, A_d, h_d
    data_exp._refresh_a_union()
    return merge_frames(data_exp)


def extract_data_expansion(
    plan_or_tableau: SelectorPlan | ReferenceCHPTableau,
    expansion: ShotExpansion,
    n_data_or_bits: int | Sequence[int | bool] | NDArray[Any],
    selector_bits: Sequence[int | bool] | NDArray[Any] | None = None,
) -> tuple[ReferenceCHPTableau, ShotExpansion]:
    """Matrix-free `O(n_words * chi)` extraction of `(data_tableau, data_expansion)`.

    Passing `SelectorPlan` applies selector gauge phases (`< 1e-15` global-phase exact when
    paired with `reconstruct_statevector(..., plan=...)` or `frame_events=`); passing a bare
    `ReferenceCHPTableau` restricts to data qubits up to global phase (`|<psi|phi>| = 1`).
    """
    plan_meta: SelectorPlan | None = None
    if isinstance(plan_or_tableau, SelectorPlan):
        plan_meta = plan_or_tableau
        raw_bits = n_data_or_bits if selector_bits is None else selector_bits
        n_data = plan_meta.num_data_qubits
    elif isinstance(plan_or_tableau, ReferenceCHPTableau):
        if selector_bits is None or not isinstance(n_data_or_bits, (int, np.integer)):
            raise ValueError(
                "When passing ReferenceCHPTableau, signature is "
                "extract_data_expansion(ref_tableau, expansion, n_data_qubits, selector_bits)"
            )
        n_data, raw_bits = int(n_data_or_bits), selector_bits
    else:
        raise TypeError(f"Expected SelectorPlan or ReferenceCHPTableau, got {type(plan_or_tableau)!r}")

    sel_arr = np.asarray(raw_bits, dtype=np.bool_).ravel()
    data_exp = _slice_expansion_to_data_qubits(expansion, n_data, sel_arr)

    if plan_meta is not None:
        n_w_data = n_words_for_qubits(n_data)
        w_a, w_b, w_kap = np.zeros(n_w_data, dtype=np.uint64), np.zeros(n_w_data, dtype=np.uint64), 0
        for s_q, ax_d in plan_meta._gauge_steps:
            if s_q >= 0:
                if bool(sel_arr[s_q - n_data]):
                    w_kap = (ax_d.kappa + w_kap + (word_dot_parity(ax_d.b_words, w_a) << 1)) & 3
                    w_a ^= ax_d.a_words
                    w_b ^= ax_d.b_words
            elif get_bit_in_words(w_a, first_set_bit_in_words(ax_d.a_words)) == 1:
                f2 = word_dot_parity(w_a, ax_d.b_words) ^ word_dot_parity(w_b, ax_d.a_words)
                fac_kap = (ax_d.kappa + (3 if f2 else 1)) & 3
                w_kap = (w_kap + fac_kap + (word_dot_parity(w_b, ax_d.a_words) << 1)) & 3
                w_a ^= ax_d.a_words
                w_b ^= ax_d.b_words

        odd_ph = ((w_kap - word_dot_parity(w_a, w_b)) & 1) != 0
        kap_herm = ((w_kap - 1) & 3) if odd_ph else w_kap
        if words_any_nonzero(w_a) or words_any_nonzero(w_b) or kap_herm != 0:
            data_exp = apply_pauli(data_exp, CompiledPauliAxis(w_a, w_b, kap_herm))
            data_exp = merge_frames(data_exp)
        if odd_ph and data_exp.chi > 0:
            data_exp.weights = np.ascontiguousarray(data_exp.weights * 1.0j)
        return plan_meta.data_final_tableau.copy(), data_exp

    assert isinstance(plan_or_tableau, ReferenceCHPTableau)
    return _restrict_tableau_to_data(plan_or_tableau, n_data, sel_arr), data_exp


_UNITARY_1Q_2Q_CACHE: dict[str, NDArray[np.complex128]] = {}


def _cached_gate_unitary_le(gate_name: str) -> NDArray[np.complex128]:
    u = _UNITARY_1Q_2Q_CACHE.get(gate_name)
    if u is None:
        u = eval_decomposition_unitary_le(decompose_clifford_exact(gate_name))
        _UNITARY_1Q_2Q_CACHE[gate_name] = u
    return u


def _apply_pauli_to_dense_le(
    psi: NDArray[np.complex128],
    a_words: NDArray[np.uint64],
    b_words: NDArray[np.uint64],
    kappa: int,
) -> NDArray[np.complex128]:
    x_idx = np.arange(psi.shape[0], dtype=np.uint64) ^ np.uint64(int(a_words[0]))
    par = np.bitwise_count(x_idx & np.uint64(int(b_words[0]))) & np.uint64(1)
    p_idx = (np.uint64(kappa & 3) + (par << np.uint64(1))) & np.uint64(3)
    return np.ascontiguousarray(_PHASE4_ARR[p_idx] * psi[x_idx.astype(np.intp)])


def reconstruct_statevector(
    first: ShotExpansion | ReferenceCHPTableau | TrajectoryRunResult,
    second: ShotExpansion | ReferenceCHPTableau | None = None,
    *,
    frame_events: Sequence[FrameEvent] | None = None,
    ref_tableau: ReferenceCHPTableau | None = None,
    plan: DirectPlan | SelectorPlan | None = None,
) -> NDArray[np.complex128]:
    """Reconstruct `complex128` little-endian physical statevector.

    Preserves exact global phase (`< 1e-15`) when given a `TrajectoryRunResult` or explicit
    `plan=` / `frame_events=`; bare `(ReferenceCHPTableau, ShotExpansion)` without `plan=`
    or `frame_events=` reconstructs up to global phase (`|<psi|phi>| = 1`).
    """
    if isinstance(first, TrajectoryRunResult):
        expansion = first.expansion
        if ref_tableau is None:
            ref_tableau = first.final_tableau
        if frame_events is None:
            frame_events = first.frame_events
    elif isinstance(first, ReferenceCHPTableau):
        if not isinstance(second, ShotExpansion):
            raise TypeError("Expected ShotExpansion as second argument when first is ReferenceCHPTableau")
        ref_tableau, expansion = first, second
    elif isinstance(first, ShotExpansion):
        expansion = first
        if isinstance(second, ReferenceCHPTableau) and ref_tableau is None:
            ref_tableau = second
    else:
        raise TypeError(f"Unsupported argument type for reconstruct_statevector: {type(first)!r}")

    n = expansion.n_qubits
    if n > 26:
        raise ValueError(f"reconstruct_statevector requires n_qubits <= 26, got {n}")
    events = frame_events
    if events is None and plan is not None:
        if isinstance(plan, DirectPlan):
            events = plan.frame_events
        elif isinstance(plan, SelectorPlan):
            events = plan.data_frame_events if n == plan.num_data_qubits else plan.frame_events

    if events is None:
        psi_ref = expansion.to_statevector(endian="little")
        return psi_ref if ref_tableau is None else apply_reference_tableau_to_statevector(psi_ref, ref_tableau, endian="little")

    psi = expansion.to_statevector(endian="little")
    for r_ev in reversed([ev for ev in events if isinstance(ev, RightRotEvent)]):
        p_psi = _apply_pauli_to_dense_le(psi, r_ev.rot_axis.a_words, r_ev.rot_axis.b_words, r_ev.rot_axis.kappa)
        psi = np.ascontiguousarray((psi - 1.0j * p_psi) * _INV_SQRT2)

    n_w = n_words_for_qubits(n)
    zero_w = np.zeros(n_w, dtype=np.uint64)
    for l_ev in (ev for ev in events if not isinstance(ev, RightRotEvent)):
        if isinstance(l_ev, LeftUnitaryEvent):
            psi = _apply_local_unitary_c128(psi, _cached_gate_unitary_le(l_ev.gate_name), l_ev.targets, n, "little")
        elif isinstance(l_ev, LeftResetXEvent):
            psi = _apply_local_unitary_c128(psi, _cached_gate_unitary_le("X"), (l_ev.qubit,), n, "little")
        elif isinstance(l_ev, LeftSppEvent):
            a_w, b_w, kap = zero_w.copy(), zero_w.copy(), (2 if l_ev.negate else 0)
            for q_s, pch_s in l_ev.terms:
                if pch_s != "I":
                    qa = unit_bit_words(q_s, n_w) if pch_s in ("X", "Y") else zero_w
                    qb = unit_bit_words(q_s, n_w) if pch_s in ("Z", "Y") else zero_w
                    a_w, b_w, kap = pauli_multiply(a_w, b_w, kap, qa, qb, 1 if pch_s == "Y" else 0)
            p_psi = _apply_pauli_to_dense_le(psi, a_w, b_w, kap)
            phase, sign_j = (OMEGA_POWERS[7], 1.0j) if l_ev.dag else (OMEGA_POWERS[1], -1.0j)
            psi = np.ascontiguousarray(phase * _INV_SQRT2 * (psi + sign_j * p_psi))
        else:
            raise ValueError(f"Unexpected Left event {l_ev!r}")

    return psi
