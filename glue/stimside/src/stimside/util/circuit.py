from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]


def qubit_targets(raw_targets: list[stim.GateTarget]) -> np.ndarray:
    """Qubit indices of the qubit-valued targets (record / sweep / combiner targets skipped)."""
    return np.fromiter(
        (t.qubit_value for t in raw_targets if t.qubit_value is not None), dtype=np.intp
    )


def unroll_circuit(
    circuit: stim.Circuit | stim.CircuitRepeatBlock | stim.CircuitInstruction,
) -> list[stim.CircuitInstruction]:
    """Unroll REPEAT blocks into a flat list of CircuitInstructions without
    stripping SHIFT_COORDS or fusing adjacent gates across loop boundaries.
    """
    if isinstance(circuit, stim.CircuitInstruction):
        return [circuit]
    if isinstance(circuit, stim.CircuitRepeatBlock):
        return unroll_circuit(circuit.body_copy()) * circuit.repeat_count
    ops: list[stim.CircuitInstruction] = []
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            ops.extend(unroll_circuit(item.body_copy()) * item.repeat_count)
        else:
            ops.append(item)
    return ops


def replays(this: stim.CircuitInstruction, op: stim.CircuitInstruction) -> bool:
    """Whether interactive_do may do `this` where the construction circuit has `op`.

    Only an untagged noisy gate's arguments (noise strengths) may differ: what the
    simulators precompute from the construction circuit (reference sample or tape,
    known states, detector/observable converter, op handler tag map) is noiseless
    or keyed by tagged ops, and noise is applied live from `this`.
    """
    return this == op or (
        this.tag == op.tag == ""
        and this.name == op.name
        and this.targets_copy() == op.targets_copy()
        and stim.gate_data(this.name).is_noisy_gate
    )


def check_replay(
    this: stim.CircuitInstruction,
    unrolled_ops: list[stim.CircuitInstruction],
    t: int,
    sim_name: str,
    *,
    allow_extension: bool = True,
    num_qubits: int | None = None,
    allow_tag: Any = None,
) -> None:
    """Validate an ``interactive_do`` instruction against ``unrolled_ops`` at step ``t``."""
    n = len(unrolled_ops)
    if t < n or not allow_extension:
        if t >= n or not replays(this, unrolled_ops[t]):
            where = (
                f"which has `{unrolled_ops[t]}` there"
                if t < n
                else f"which ends after {n} operations"
            )
            raise ValueError(
                f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` doesn't match "
                f"the circuit the {sim_name} was constructed with, {where}. Pass the circuit "
                "itself or its parts, not `circuit.flattened()` or `circuit + more`."
            )
        return
    if this.name in ("DETECTOR", "OBSERVABLE_INCLUDE") or (
        this.tag and not (allow_tag is not None and allow_tag(this.tag))
    ):
        raise ValueError(
            f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` is past the end "
            f"of the circuit the {sim_name} was constructed with ({n} operations)."
        )
    if num_qubits is not None and this.name not in ("MPAD", "QUBIT_COORDS"):
        for tgt in this.targets_copy():
            q = tgt.qubit_value
            if q is not None and q >= num_qubits:
                raise ValueError(
                    f"interactive_do: operation {t} (REPEAT blocks unrolled) `{this}` targets "
                    f"qubit {q}, which the {sim_name} wasn't constructed for "
                    f"(num_qubits={num_qubits})."
                )


def apply_coords_op(
    op: stim.CircuitInstruction,
    qubit_coords: dict[int, list[float]],
    qubit_tags: dict[int, str],
    coords_shifts: list[float],
) -> bool:
    """Update coordinate/tag state in place for `QUBIT_COORDS` or `SHIFT_COORDS`."""
    name = op.name
    if name == "QUBIT_COORDS":
        args = op.gate_args_copy()
        n_shifts = len(coords_shifts)
        coords = [c + (coords_shifts[i] if i < n_shifts else 0.0) for i, c in enumerate(args)]
        tag = op.tag
        for gt in op.targets_copy():
            q = gt.qubit_value
            qubit_coords[q] = list(coords)
            qubit_tags[q] = tag
        return True
    if name == "SHIFT_COORDS":
        for i, s in enumerate(op.gate_args_copy()):
            if i >= len(coords_shifts):
                coords_shifts.append(s)
            else:
                coords_shifts[i] += s
        return True
    return False


def stack_measurement_columns(columns: list[NDArray[np.bool_]], batch_size: int) -> NDArray[np.bool_]:
    return np.zeros((batch_size, 0), dtype=np.bool_) if not columns else np.column_stack(columns)


def pack_shot_flips(
    flips: NDArray[np.bool_], *, bit_packed: bool = False
) -> NDArray[np.bool_] | NDArray[np.uint8]:
    if bit_packed:
        return np.packbits(flips, axis=-1, bitorder="little")
    return flips.copy()


def format_detector_flips(
    dets: NDArray[np.bool_],
    obs: NDArray[np.bool_] | None,
    *,
    append_observables: bool = False,
    bit_packed: bool = False,
) -> NDArray[np.bool_] | NDArray[np.uint8]:
    if append_observables and obs is not None:
        out = np.hstack([dets, obs])
        return np.packbits(out, axis=-1, bitorder="little") if bit_packed else out
    return pack_shot_flips(dets, bit_packed=bit_packed)


class ShotMajorM2DMixin:
    """Shared shot-major m2d conversion and ``get_detector_flips`` / ``get_observable_flips``."""

    _detector_flips: NDArray[np.bool_] | None
    _observable_flips: NDArray[np.bool_] | None
    _compiled_m2d_converter: Any

    def get_final_measurement_records(self) -> NDArray[np.bool_]:
        raise NotImplementedError

    def _convert_measurements_to_detector_flips(self) -> None:
        self._detector_flips, self._observable_flips = self._compiled_m2d_converter.convert(
            measurements=self.get_final_measurement_records(), separate_observables=True
        )

    def get_detector_flips(
        self, append_observables: bool = False, *, bit_packed: bool = False
    ) -> NDArray[np.bool_] | NDArray[np.uint8]:
        if self._detector_flips is None:
            self._convert_measurements_to_detector_flips()
        assert self._detector_flips is not None
        return format_detector_flips(
            self._detector_flips, self._observable_flips,
            append_observables=append_observables, bit_packed=bit_packed,
        )

    def get_observable_flips(self, *, bit_packed: bool = False) -> NDArray[np.bool_] | NDArray[np.uint8]:
        if self._observable_flips is None:
            self._convert_measurements_to_detector_flips()
        assert self._observable_flips is not None
        return pack_shot_flips(self._observable_flips, bit_packed=bit_packed)
