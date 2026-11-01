"""Shared TableauSimulator-backed RNG lockstep primitives for sync_tableside_rng."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]

from stimside.util.seeding import derived_seed


def build_sync_rngs(
    seed: int | None, batches_completed: int, batch_size: int, num_qubits: int
) -> tuple[int, list[np.random.Generator], list[stim.TableauSimulator], list[stim.TableauSimulator]]:
    """Initialize per-shot numpy RNGs and paired Z/X TableauSimulators matching Tableside."""
    base = int(np.random.default_rng().integers(0, 2**63)) if seed is None else seed
    n0 = 0 if seed is None else batches_completed * batch_size
    shot_seeds = [derived_seed(base, n0 + b) for b in range(batch_size)]
    base_seed = base if seed is None else shot_seeds[0]
    shot_np_rngs = [np.random.default_rng(seed=s) for s in shot_seeds]
    tab_rngs, tab_rngs_x = [], []
    all_q = list(range(num_qubits))
    for s in shot_seeds:
        tz, tx = stim.TableauSimulator(seed=s), stim.TableauSimulator(seed=s)
        tz.set_num_qubits(num_qubits + 1)
        tx.set_num_qubits(num_qubits + 1)
        if all_q:
            tx.h(*all_q)
        tab_rngs.append(tz)
        tab_rngs_x.append(tx)
    return base_seed, shot_np_rngs, tab_rngs, tab_rngs_x


def draw_tab_rng_bit(tz: stim.TableauSimulator, tx: stim.TableauSimulator, scratch_q: int) -> bool:
    """Draw one random measurement bit from paired Z/X TableauSimulators on ``scratch_q``."""
    tz.h(scratch_q)
    bit = bool(tz.measure(scratch_q))
    if bit:
        tz.x(scratch_q)
    tx.h(scratch_q)
    if tx.measure(scratch_q):
        tx.x(scratch_q)
    return bit


def exec_tab_rng_op(
    tz: stim.TableauSimulator,
    tx: stim.TableauSimulator,
    b: int,
    inst: stim.CircuitInstruction,
    qs: Iterable[int],
    xm: NDArray[np.bool_] | None,
    zm: NDArray[np.bool_] | None,
    n_rec: int = 0,
) -> tuple[bool, bool, list[bool]]:
    """Execute ``inst`` on shot ``b``'s Z/X TableauSimulators and extract kicks and measurements."""
    tz.do(inst)
    tx.do(inst)
    any_x = any_z = False
    for q in qs:
        if xm is not None and tz.peek_z(q) == -1:
            xm[q, b] ^= True
            tz.x(q)
            any_x = True
        if zm is not None and tx.peek_x(q) == -1:
            zm[q, b] ^= True
            tx.z(q)
            any_z = True
    rec = [bool(v) for v in tz.current_measurement_record()[-n_rec:]] if n_rec > 0 else []
    return any_x, any_z, rec


def exec_sync_pauli_batch(
    tab_rngs: Sequence[stim.TableauSimulator],
    tab_rngs_x: Sequence[stim.TableauSimulator],
    op_name: str,
    targets_per_shot: Sequence[Sequence[int]],
    gate_args: Sequence[float],
    xm: NDArray[np.bool_],
    zm: NDArray[np.bool_],
) -> tuple[bool, bool]:
    any_x = any_z = False
    for b, t_b in enumerate(targets_per_shot):
        if t_b:
            ax, az, _ = exec_tab_rng_op(
                tab_rngs[b], tab_rngs_x[b], b, stim.CircuitInstruction(op_name, t_b, gate_args), set(t_b), xm, zm
            )
            any_x |= ax
            any_z |= az
    return any_x, any_z


def exec_sync_correlated_error(
    tab_rngs: Sequence[stim.TableauSimulator],
    tab_rngs_x: Sequence[stim.TableauSimulator],
    op_name: str,
    raw_targets: Sequence[stim.GateTarget],
    p_err: float,
    condition_met_mask: NDArray[np.bool_] | None,
    xm: NDArray[np.bool_],
    zm: NDArray[np.bool_],
) -> tuple[bool, bool]:
    any_x = any_z = False
    for b in range(len(tab_rngs)):
        t_b = raw_targets if condition_met_mask is None else [raw_targets[i] for i in range(len(raw_targets)) if condition_met_mask[i, b]]
        qs = {gt.qubit_value for gt in t_b if gt.qubit_value is not None}
        ax, az, _ = exec_tab_rng_op(
            tab_rngs[b], tab_rngs_x[b], b, stim.CircuitInstruction(op_name, t_b, [p_err]), qs, xm, zm
        )
        any_x |= ax
        any_z |= az
    return any_x, any_z


def exec_sync_heralded_noise(
    tab_rngs: Sequence[stim.TableauSimulator],
    tab_rngs_x: Sequence[stim.TableauSimulator],
    op: stim.CircuitInstruction,
    qs: set[int],
    num_m: int,
    xm: NDArray[np.bool_],
    zm: NDArray[np.bool_],
) -> tuple[bool, bool, list[NDArray[np.bool_]]]:
    cols = [np.zeros(len(tab_rngs), dtype=np.bool_) for _ in range(num_m)]
    any_x = any_z = False
    for b in range(len(tab_rngs)):
        ax, az, rec = exec_tab_rng_op(tab_rngs[b], tab_rngs_x[b], b, op, qs, xm, zm, num_m)
        any_x |= ax
        any_z |= az
        for k, v in enumerate(rec):
            cols[k][b] = v
    return any_x, any_z, cols


def conditioned_targets_per_shot(
    targets: Sequence[int], arity: int, batch_size: int, condition_met_mask: NDArray[np.bool_] | None
) -> list[list[int]]:
    if condition_met_mask is None:
        t_list = list(targets)
        return [t_list for _ in range(batch_size)]
    groups = [targets[i : i + arity] for i in range(0, len(targets), arity)]
    return [[q for g_i, grp in enumerate(groups) if condition_met_mask[g_i, b] for q in grp] for b in range(batch_size)]


def apply_sync_meas_flips(
    tab_rngs: Sequence[stim.TableauSimulator],
    tab_rngs_x: Sequence[stim.TableauSimulator],
    measurement_columns: Sequence[NDArray[np.bool_]],
    p_err: float,
    num_m: int,
) -> None:
    mp_op = stim.CircuitInstruction("MPAD", [0] * num_m, [p_err])
    for b in range(len(tab_rngs)):
        _, _, flips = exec_tab_rng_op(tab_rngs[b], tab_rngs_x[b], b, mp_op, (), None, None, num_m)
        for k, fl in enumerate(flips):
            if fl:
                measurement_columns[-num_m + k][b] ^= True
