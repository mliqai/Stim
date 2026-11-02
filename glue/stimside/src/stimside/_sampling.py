"""Shared batch-sampling loop and Sinter sampler base classes across stimside engines."""

from __future__ import annotations

import collections
import os
import time
from typing import Any

import numpy as np
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    LeakageDecoderLike,
    as_leakage_decoder,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler


def next_compiled_seed(seed: int | None, idx: int, salt: int) -> int | None:
    """Return `seed` on `(idx=0, salt=0)` or a deterministic per-compile/per-worker 30-bit seed."""
    if seed is None:
        return None
    if idx == 0 and salt == 0:
        return int(seed)
    ss = np.random.SeedSequence([int(seed), int(idx), int(salt)])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x3FFFFFFF)


def sample_batches(
    simulator: Any,
    compiled_dem_decoder: CompiledLeakageDecoder,
    suggested_shots: int,
    *,
    needs_records: bool = False,
    needs_leakage_events: bool = False,
    track_leakage_count: bool = False,
) -> sinter.AnonTaskStats:
    """Run ``simulator`` in batches until ``suggested_shots`` are decoded into ``sinter.AnonTaskStats``."""
    start_time = time.process_time()
    shots_taken = 0
    num_errors = 0
    leak_total = 0

    while shots_taken < suggested_shots:
        simulator.clear()
        u2l_list = simulator.run()

        det_packed = simulator.get_detector_flips(bit_packed=True)
        obs_packed = simulator.get_observable_flips(bit_packed=True)

        decode_kwargs: dict[str, Any] = {
            "bit_packed_detection_event_data": det_packed,
            "records": (
                simulator.get_final_measurement_records() if needs_records else None
            ),
        }
        if needs_leakage_events:
            decode_kwargs["leakage_events"] = simulator.get_leakage_events()
        pred_packed = compiled_dem_decoder.decode_shots_bit_packed(**decode_kwargs)

        num_errors += int(
            np.count_nonzero(np.any(pred_packed != obs_packed, axis=-1))
        )
        shots_taken += simulator.batch_size
        if track_leakage_count and u2l_list is not None:
            leak_total += sum(len(arr) for arr in u2l_list)

    seconds = time.process_time() - start_time
    if track_leakage_count:
        counts = (
            collections.Counter({"leakage_count": leak_total})
            if leak_total
            else collections.Counter()
        )
        return sinter.AnonTaskStats(
            shots=shots_taken,
            errors=num_errors,
            discards=0,
            seconds=seconds,
            custom_counts=counts,
        )
    return sinter.AnonTaskStats(
        shots=shots_taken, errors=num_errors, discards=0, seconds=seconds
    )


class SamplerBase(sinter.Sampler):
    """Common Sinter sampler configuration for Flipside, Tableside, and Cosetside."""

    def __init__(
        self,
        op_handler: OpHandler[Any],
        dem_decoder: LeakageDecoderLike,
        batch_size: int,
        seed: int | None = None,
    ) -> None:
        self.op_handler = op_handler
        self.dem_decoder = as_leakage_decoder(dem_decoder, type(self).__name__)
        self.batch_size = batch_size
        self.seed = seed
        self._init_pid = os.getpid()
        self._compile_count = 0

    def _next_compiled_seed(self) -> int | None:
        idx = self._compile_count
        self._compile_count = idx + 1
        return next_compiled_seed(self.seed, idx, (os.getpid() - self._init_pid) & 0xFFFFFFFF)

    def _require_circuit(self, task: sinter.Task) -> stim.Circuit:
        if task.circuit is None:
            raise ValueError(
                f"{type(self).__name__} requires a circuit in the task to compile a sampler."
            )
        return task.circuit


class CompiledSamplerBase(sinter.CompiledSampler):
    """Common Sinter compiled-sampler runner delegating ``sample()`` to ``sample_batches``."""

    simulator: Any
    compiled_dem_decoder: CompiledLeakageDecoder
    needs_records: bool
    needs_leakage_events: bool
    _track_leakage_count: bool = False

    def _init_common(
        self,
        circuit: stim.Circuit | None,
        compiled_op_handler: CompiledOpHandler[Any] | None,
        batch_size: int,
        compiled_dem_decoder: CompiledLeakageDecoder,
        needs_records: bool,
        needs_leakage_events: bool,
    ) -> None:
        self.circuit = circuit
        self.batch_size = int(batch_size)
        self.compiled_dem_decoder = compiled_dem_decoder
        self.compiled_op_handler = compiled_op_handler
        self.needs_records = bool(needs_records)
        self.needs_leakage_events = bool(needs_leakage_events)

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        return sample_batches(
            self.simulator,
            self.compiled_dem_decoder,
            suggested_shots,
            needs_records=self.needs_records,
            needs_leakage_events=self.needs_leakage_events,
            track_leakage_count=self._track_leakage_count,
        )
