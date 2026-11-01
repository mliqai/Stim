from __future__ import annotations

import stim  # type: ignore[import-untyped]


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
