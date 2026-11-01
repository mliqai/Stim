"""Shared transition sampling and output-effect helpers for Tableau and Coset leakage handlers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from stimside.op_handlers.leakage_handlers.leakage_parameters import (
        LeakageTransition1Params,
        LeakageTransition2Params,
    )
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
        state[idx] = 0 if in_st in (0, 1, "U") else int(in_st)
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


def sample_coset_batched_groups(
    transitions: Sequence[tuple[Any, float]],
    mask: NDArray[np.bool_],
    total_p: float,
    np_rng: np.random.Generator,
) -> list[tuple[Any, NDArray[np.intp], NDArray[np.intp]]]:
    """Sample `(out_state, row_indices, shot_indices)` groups for Coset's non-sync batched path."""
    samples_to_take = int(np.count_nonzero(mask))
    if samples_to_take == 0:
        return []
    k_trans = (
        int(np_rng.binomial(samples_to_take, total_p))
        if total_p < 1.0 - 1e-12
        else samples_to_take
    )
    if k_trans == 0:
        return []
    r_all, b_all = np.where(mask)
    if k_trans < samples_to_take:
        chosen = np_rng.choice(samples_to_take, size=k_trans, replace=False)
        r_trans, b_trans = r_all[chosen], b_all[chosen]
    else:
        r_trans, b_trans = r_all, b_all
    if len(transitions) == 1:
        return [(transitions[0][0], r_trans, b_trans)]
    cond_probs = np.array([p / total_p for _, p in transitions], dtype=np.float64)
    cond_probs /= cond_probs.sum()
    t_idx = np_rng.choice(len(transitions), size=k_trans, p=cond_probs)
    return [
        (transitions[t_i][0], r_trans[t_idx == t_i], b_trans[t_idx == t_i])
        for t_i in range(len(transitions))
    ]
