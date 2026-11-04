"""Shared transition sampling and output-effect helpers for Tableau and Coset leakage handlers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.op_handlers.leakage_handlers.leakage_parameters import (
    LeakageConditioningParams,
    LeakageMeasurementParams,
    LeakageSwapParams,
    LeakageTransition1Params,
    LeakageTransition2Params,
)

if TYPE_CHECKING:
    from stimside.simulator_tableau import TablesideSimulator


@dataclasses.dataclass
class TransitionEffectMasks:
    """Boolean effect masks accumulated across a transition instruction."""

    set_zero: NDArray[np.bool_]
    set_one: NDArray[np.bool_]
    depolarize: NDArray[np.bool_]
    pauli_x: NDArray[np.bool_] | None = None
    pauli_y: NDArray[np.bool_] | None = None
    pauli_z: NDArray[np.bool_] | None = None
    rand_x: NDArray[np.bool_] | None = None
    rand_phase: NDArray[np.bool_] | None = None

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, two_qubit: bool) -> TransitionEffectMasks:
        mk = lambda: np.zeros(shape, dtype=np.bool_)
        if not two_qubit:
            return cls(set_zero=mk(), set_one=mk(), depolarize=mk())
        return cls(
            set_zero=mk(),
            set_one=mk(),
            depolarize=mk(),
            pauli_x=mk(),
            pauli_y=mk(),
            pauli_z=mk(),
            rand_x=mk(),
            rand_phase=mk(),
        )


# out_key -> (reset_zero_always, reset_zero_if_leaked, set_one, depolarize, pauli_attr, rand_v)
_OUTPUT_EFFECT_TABLE: dict[str, tuple[bool, bool, bool, bool, str | None, bool]] = {
    "0": (True, False, False, False, None, False),
    "1": (False, False, True, False, None, False),
    "U": (False, True, False, True, None, False),
    "D": (False, True, False, True, None, False),
    "V": (True, False, False, False, None, True),
    "X": (False, True, False, False, "pauli_x", False),
    "Y": (False, True, False, False, "pauli_y", False),
    "Z": (False, True, False, False, "pauli_z", False),
}


def apply_output_effect(
    state: NDArray[np.uint8],
    masks: TransitionEffectMasks,
    idx: Any,
    in_st: int | str,
    out_st: int | str,
) -> None:
    """Update `state[idx]` and accumulate quantum side-effects in `masks` for `in_st -> out_st`.

    `idx` may be a scalar qubit `q` (Tableau), a `(q, b)` pair (Coset sync), or a
    disjoint `(q_arr, b_arr)` index tuple (Coset batched).
    """
    out_key = str(out_st)
    if out_key == str(in_st):
        # "No transition": leave the state as is, so an earlier matching key's transition
        # in the same instruction is kept (keys apply in order; OD3-2).
        return

    entry = _OUTPUT_EFFECT_TABLE.get(out_key)
    if entry is None:
        out_int = int(out_key)
        state[idx] = 0 if out_int < 2 else out_int
        return

    rz_always, rz_if_leaked, set_one, depol, pauli_attr, rand_v = entry
    in_is_leaked = in_st not in (0, 1, "U")
    if rz_always or (rz_if_leaked and in_is_leaked):
        masks.set_zero[idx] = True
    if set_one:
        masks.set_one[idx] = True
    if depol:
        masks.depolarize[idx] = True
    if pauli_attr is not None:
        arr = getattr(masks, pauli_attr)
        assert arr is not None
        arr[idx] ^= True
    if rand_v:
        assert masks.rand_x is not None and masks.rand_phase is not None
        masks.rand_x[idx] = True
        masks.rand_phase[idx] = True
    state[idx] = 0


def emit_tableau_transition_effects(
    tss: TablesideSimulator, masks: TransitionEffectMasks
) -> None:
    """Emit accumulated transition side-effects to `tss` in canonical Stim instruction order."""
    idx_z0 = np.where(masks.set_zero)[0]
    if len(idx_z0) > 0:
        tss._do_bare_instruction(stim.CircuitInstruction("R", idx_z0))
    idx_z1 = np.where(masks.set_one)[0]
    if len(idx_z1) > 0:
        tss._do_bare_instruction(stim.CircuitInstruction("R", idx_z1))
        tss._do_bare_instruction(stim.CircuitInstruction("X", idx_z1))
    idx_dep = np.where(masks.depolarize)[0]
    if len(idx_dep) > 0:
        tss._do_bare_instruction(stim.CircuitInstruction("DEPOLARIZE1", idx_dep, [0.75]))
    for gate, arr, args in (
        ("X", masks.pauli_x, ()),
        ("Y", masks.pauli_y, ()),
        ("Z", masks.pauli_z, ()),
        ("X_ERROR", masks.rand_x, [0.5]),
        ("Z_ERROR", masks.rand_phase, [0.5]),
    ):
        if arr is not None:
            idx = np.where(arr)[0]
            if len(idx) > 0:
                tss._do_bare_instruction(stim.CircuitInstruction(gate, idx, args))


def sample_tableau_transitions(
    params: LeakageTransition1Params | LeakageTransition2Params,
    input_state: Any,
    samples_to_take: int,
    np_rng: np.random.Generator,
    *,
    sync_rng: bool,
) -> tuple[slice | NDArray[np.intp], Sequence[Any]] | None:
    """Sample transition outputs for Tableau, keeping sparse (<0.05) and sync RNG draws exact."""
    transitions = params.args_by_input_state[input_state]
    total_p = sum(p for _, p in transitions)
    if not sync_rng and total_p <= 0.0:
        return None
    if not sync_rng and total_p < 0.05 and len(transitions) >= 1:
        n_hits = int(np_rng.binomial(samples_to_take, total_p))
        if n_hits == 0:
            return None
        chosen = np_rng.choice(samples_to_take, size=n_hits, replace=False)
        if len(transitions) == 1:
            return chosen, [transitions[0][0]] * n_hits
        cond_probs = [p / total_p for _, p in transitions]
        br_idx = np_rng.choice(len(transitions), size=n_hits, p=cond_probs)
        return chosen, [transitions[int(b)][0] for b in br_idx]
    return slice(None), params.sample_transitions_from_state(
        input_state=input_state, num_samples=samples_to_take, np_rng=np_rng
    ).astype(str)


def sample_batched_group_transitions(
    transitions: Sequence[tuple[Any, float]],
    total_p: float,
    np_rng: np.random.Generator,
    mask: NDArray[np.bool_] | None = None,
    *,
    num_rows: int = 0,
    batch_size: int = 0,
) -> list[tuple[Any, NDArray[np.intp], NDArray[np.intp]]]:
    """Sample non-empty `(out_state, row_idx, shot_idx)` groups preserving sparse RNG order."""
    if total_p <= 0.0:
        return []
    if mask is None:
        m_total = num_rows * batch_size
        n_hits = m_total if total_p >= 1.0 - 1e-12 else int(np_rng.binomial(m_total, total_p))
        if n_hits == 0:
            return []
        if n_hits == m_total:
            row_idx = np.repeat(np.arange(num_rows, dtype=np.intp), batch_size)
            b_idx = np.tile(np.arange(batch_size, dtype=np.intp), num_rows)
        else:
            flat_idx = np_rng.choice(m_total, size=n_hits, replace=False)
            row_idx, b_idx = flat_idx // batch_size, flat_idx % batch_size
    else:
        m_total = int(np.count_nonzero(mask))
        if m_total == 0:
            return []
        n_hits = m_total if total_p >= 1.0 - 1e-12 else int(np_rng.binomial(m_total, total_p))
        if n_hits == 0:
            return []
        elig_rows, elig_bs = np.where(mask)
        if n_hits == m_total:
            row_idx, b_idx = elig_rows, elig_bs
        else:
            chosen = np_rng.choice(m_total, size=n_hits, replace=False)
            row_idx, b_idx = elig_rows[chosen], elig_bs[chosen]
    if len(transitions) == 1:
        return [(transitions[0][0], row_idx, b_idx)]
    cond_probs = np.array([p / total_p for _, p in transitions], dtype=np.float64)
    cond_probs /= cond_probs.sum()
    br_idx = np_rng.choice(len(transitions), size=n_hits, p=cond_probs)
    return [
        (out_val, row_idx[sel], b_idx[sel])
        for b_i, (out_val, _) in enumerate(transitions)
        if np.any(sel := (br_idx == b_i))
    ]


def eval_allowed_state_mask(
    target_states: NDArray[Any],
    is_classical: NDArray[np.bool_],
    allowed_group: Sequence[int | str],
    full_list: Any = None,
) -> NDArray[np.bool_]:
    """Evaluate allowed-state membership mask for 1Q or 2Q Tableau/Coset targets."""
    mask = np.zeros(target_states.shape, dtype=np.bool_)
    for st in allowed_group:
        if st == "U":
            mask |= is_classical | (target_states < 2)
        else:
            st_i = int(st)
            mask |= (is_classical & (st_i < 2)) | ((~is_classical) & (target_states == st_i))
    return mask


def dispatch_tagged_leakage_op(
    handler: Any,
    op: stim.CircuitInstruction,
    sss: Any,
    *,
    state2d: NDArray[np.uint8] | None = None,
) -> None:
    """Dispatch a parsed leakage op on ``handler`` and record u2l / leakage events on ``sss``."""
    params = handler.ops_to_params[op]
    is_transition = isinstance(
        params, (LeakageTransition1Params, LeakageTransition2Params, LeakageSwapParams)
    )
    rec_ev = is_transition and getattr(sss, "record_leakage_events", False)
    if (sss.record_unleaked_to_leaked or rec_ev) and is_transition:
        if state2d is None:
            handler._ensure_batch_size(sss.batch_size)
            state2d = handler._state
        target_indices = np.unique(
            [
                t.qubit_value if t.qubit_value is not None else t.value
                for t in op.targets_copy()
                if (t.qubit_value if t.qubit_value is not None else t.value) >= 0
            ]
        )
        was_unleaked = state2d[target_indices, :] < 2
        old_states = state2d[target_indices, :].copy() if rec_ev else None
    else:
        target_indices = None
        was_unleaked = None
        old_states = None

    match params:
        case LeakageConditioningParams():
            handler.leakage_conditioning(op=op, tss=sss, params=params)
        case LeakageTransition1Params():
            handler.leakage_transition_1(op=op, tss=sss, params=params)
        case LeakageTransition2Params():
            handler.leakage_transition_2(op=op, tss=sss, params=params)
        case LeakageMeasurementParams():
            handler.leakage_measurement(op=op, tss=sss, params=params)
        case LeakageSwapParams():
            handler.leakage_swap(op=op, tss=sss)
        case _:
            raise ValueError(f"Unrecognised LEAKAGE params: {params}")

    if state2d is None:
        state2d = getattr(handler, "_state", None)
    if (
        sss.record_unleaked_to_leaked
        and was_unleaked is not None
        and target_indices is not None
        and state2d is not None
    ):
        counts = np.count_nonzero(was_unleaked & (state2d[target_indices, :] >= 2), axis=0)
        sss._record_unleaked_to_leaked_counts(counts, sss._circuit_time)
    if old_states is not None and target_indices is not None and state2d is not None:
        new_states = state2d[target_indices, :]
        for i_q, b in zip(*np.nonzero(old_states != new_states)):
            sss._record_leakage_event(
                sss._circuit_time,
                target_indices[i_q],
                old_states[i_q, b],
                new_states[i_q, b],
                shot_idx=int(b),
            )


def exec_leakage_swap(
    op: stim.CircuitInstruction, tss: Any, state: NDArray[np.uint8]
) -> None:
    """Execute bare SWAP followed by pairwise leakage-state swaps on `state`."""
    tss._do_bare_instruction(
        stim.CircuitInstruction(op.name, op.targets_copy(), op.gate_args_copy())
    )
    targets = [t.qubit_value for t in op.targets_copy()]
    for q0, q1 in zip(targets[::2], targets[1::2]):
        state[[q0, q1]] = state[[q1, q0]]


def meas_basis(op_name: str) -> Any:
    """Return `'X'`, `'Y'`, or `'Z'` basis for a single-qubit measurement/reset gate."""
    return "X" if op_name in ("MX", "MRX") else ("Y" if op_name in ("MY", "MRY") else "Z")


def unpack_leakage_measurement_targets(
    op: stim.CircuitInstruction, params: Any
) -> tuple[bool, list[stim.GateTarget], list[int], list[bool]]:
    """Unpack `(is_mpad, raw_op_targets, targets, inv_flags)` and validate MPAD arity."""
    raw = op.targets_copy()
    is_mpad = params.targets is not None
    if is_mpad:
        targets = list(params.targets)
        if len(raw) != len(targets):
            raise ValueError(
                "The number of targets in the MPAD operation with a LEAKAGE_MEASUREMENT tag "
                "does not equal to the number of targets specified in the tag."
            )
        inv_flags = [bool(t.value) ^ bool(t.is_inverted_result_target) for t in raw]
    else:
        targets = [t.qubit_value if t.qubit_value is not None else t.value for t in raw]
        inv_flags = [bool(t.is_inverted_result_target) for t in raw]
    return is_mpad, raw, targets, inv_flags
