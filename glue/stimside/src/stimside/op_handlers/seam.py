"""Typed seam between the batched leakage handlers and the Cosetside/Magicside simulators."""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
import stim  # type: ignore[import-untyped]


@dataclasses.dataclass(frozen=True)
class SyncRngs:
    """Per-shot RNG streams of a ``sync_tableside_rng=True`` simulator (index = shot)."""

    tab_rngs: list[stim.TableauSimulator]
    tab_rngs_x: list[stim.TableauSimulator]
    shot_np_rngs: list[np.random.Generator]


class BatchedLeakageSimulator(Protocol):
    """Everything ``CompiledLeakageUint8Coset``/``Magic`` may use on the simulator they drive.

    Implemented by ``CosetsideSimulator`` and ``MagicsideSimulator``.
    """

    batch_size: int
    seed: int | None
    np_rng: np.random.Generator
    record_unleaked_to_leaked: bool
    record_leakage_events: bool

    # Dispatch & cursor.
    def _do_bare_instruction(self, op: stim.CircuitInstruction) -> None: ...

    def _next_step_meta(self, op: stim.CircuitInstruction) -> object: ...

    def do_copies_on_fused_step(
        self,
        op: stim.CircuitInstruction,
        copies: list[stim.CircuitInstruction],
        do_copy: Callable[[stim.CircuitInstruction], None],
    ) -> None: ...

    # Conditioned & unmatched ops.
    def do_conditional_clifford(
        self,
        op: stim.CircuitInstruction,
        condition_met_mask: NDArray[np.bool_] | None = None,
        *,
        condition_group_fn: Callable[[int], NDArray[np.bool_]] | None = None,
    ) -> None: ...

    def do_conditional_pauli_noise(
        self,
        op: stim.CircuitInstruction,
        condition_met_mask: NDArray[np.bool_] | None = None,
    ) -> None: ...

    def do_unmatched_noisy_reset_z(
        self,
        reset_zero_mask: NDArray[np.bool_],
        reset_one_mask: NDArray[np.bool_],
        pauli: Literal["X", "Y", "Z"] = "Z",
        order: Sequence[int] | None = None,
    ) -> None: ...

    def apply_pauli(
        self, pauli: Literal["X", "Y", "Z"], mask: NDArray[np.bool_], p: float = 1.0
    ) -> None: ...

    # State query, projection & direct readout.
    def peek_pauli_batch(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> NDArray[np.int8]: ...

    def peek_pauli_born_batch(
        self,
        targets: Sequence[int] | NDArray[np.int_],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> NDArray[np.float64]: ...

    def postselect_z_batch(
        self,
        q: int,
        flip_mask: NDArray[np.bool_],
        desired_bits: NDArray[np.uint8],
        pauli: Literal["X", "Y", "Z"] = "Z",
    ) -> None: ...

    def record_direct_measurements(self, outcomes_per_target: NDArray[np.bool_]) -> None: ...

    # Sync-mode RNGs (None unless sync_tableside_rng=True).
    def sync_rngs(self) -> SyncRngs | None: ...

    def sample_pauli_noise_sync(
        self,
        op_name: str,
        targets_per_shot: Sequence[Sequence[int]],
        gate_args: Sequence[float],
    ) -> None: ...

    # Measurement record (index as in a list, e.g. -1 = latest).
    def measurement_column(self, index: int) -> NDArray[np.bool_]: ...

    def overwrite_measurement_column(self, index: int, values: NDArray[np.bool_]) -> None: ...

    # Leakage records (shared with TablesideSimulator via the util mixins).
    def get_current_circuit_time(self) -> int: ...

    def record_u2l_counts(self, counts: np.ndarray, op_idx: int) -> None: ...

    def record_leakage_event(
        self, op_idx: int, qubit: int, old_state: int, new_state: int, shot_idx: int = 0
    ) -> None: ...
