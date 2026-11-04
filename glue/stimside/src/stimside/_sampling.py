"""Shared batch-sampling loop and Sinter sampler base classes across stimside engines."""

from __future__ import annotations

import collections
import os
import time
from typing import Any

import numpy as np
import sinter  # type: ignore[import-untyped]

from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    LeakageDecoderLike,
    as_leakage_decoder,
)


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
    postselection_mask: np.ndarray | None = None,
    postselected_observables_mask: np.ndarray | None = None,
) -> sinter.AnonTaskStats:
    """Run ``simulator`` in batches until ``suggested_shots`` are decoded into ``sinter.AnonTaskStats``."""
    start_time = time.process_time()
    shots_taken = 0
    num_errors = 0
    num_discards = 0
    leak_total = 0

    while shots_taken < suggested_shots:
        simulator.clear()
        u2l_list = simulator.run()

        det_packed = simulator.get_detector_flips(bit_packed=True)
        obs_packed = simulator.get_observable_flips(bit_packed=True)
        shots_taken += simulator.batch_size
        if track_leakage_count and u2l_list is not None:
            leak_total += sum(len(arr) for arr in u2l_list)

        records = simulator.get_final_measurement_records() if needs_records else None
        leakage_events = simulator.get_leakage_events() if needs_leakage_events else None

        if postselection_mask is not None:
            if postselection_mask.dtype != np.uint8 or postselection_mask.shape != (det_packed.shape[1],):
                raise ValueError(
                    f"Expected uint8 postselection_mask of shape {(det_packed.shape[1],)}, "
                    f"got dtype={postselection_mask.dtype}, shape={postselection_mask.shape}."
                )
            det_disc = np.any(det_packed & postselection_mask, axis=-1)
            n_det_disc = int(np.count_nonzero(det_disc))
            if n_det_disc:
                num_discards += n_det_disc
                if n_det_disc == det_packed.shape[0]:
                    continue
                keep = ~det_disc
                det_packed = det_packed[keep]
                obs_packed = obs_packed[keep]
                if records is not None:
                    records = records[keep]
                if leakage_events is not None:
                    leakage_events = [ev for ev, k in zip(leakage_events, keep) if k]

        decode_kwargs: dict[str, Any] = {
            "bit_packed_detection_event_data": det_packed,
            "records": records,
        }
        if needs_leakage_events:
            decode_kwargs["leakage_events"] = leakage_events
        pred_packed = compiled_dem_decoder.decode_shots_bit_packed(**decode_kwargs)
        if (
            pred_packed.ndim == 2
            and pred_packed.shape[0] == obs_packed.shape[0]
            and pred_packed.shape[1] == obs_packed.shape[1] + 1
        ):
            # sinter's protocol: an extra last byte that is nonzero marks a decoder discard.
            dec_disc = pred_packed[:, -1] != 0
            num_discards += int(np.count_nonzero(dec_disc))
            pred_packed = pred_packed[~dec_disc, :-1]
            obs_packed = obs_packed[~dec_disc]
        if pred_packed.shape != obs_packed.shape:
            raise ValueError(
                f"Decoder returned shape {pred_packed.shape}, expected {obs_packed.shape}."
            )

        diff = pred_packed ^ obs_packed
        if postselected_observables_mask is not None:
            if (
                postselected_observables_mask.dtype != np.uint8
                or postselected_observables_mask.shape != (obs_packed.shape[1],)
            ):
                raise ValueError(
                    f"Expected uint8 postselected_observables_mask of shape {(obs_packed.shape[1],)}, "
                    f"got dtype={postselected_observables_mask.dtype}, shape={postselected_observables_mask.shape}."
                )
            obs_disc = np.any(diff & postselected_observables_mask, axis=-1)
            n_obs_disc = int(np.count_nonzero(obs_disc))
            if n_obs_disc:
                num_discards += n_obs_disc
                diff = diff[~obs_disc]

        num_errors += int(np.count_nonzero(np.any(diff, axis=-1)))

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
            discards=num_discards,
            seconds=seconds,
            custom_counts=counts,
        )
    return sinter.AnonTaskStats(
        shots=shots_taken, errors=num_errors, discards=num_discards, seconds=seconds
    )


from stimside.op_handlers.abstract_op_handler import OpHandler

_DEFAULT_BATCH: Any = object()


class SamplerBase(sinter.Sampler):
    """Common Sinter sampler configuration for Flipside, Tableside, and Cosetside."""

    _compiled_cls: type[Any]
    _default_batch_size: int = 256

    def __init__(
        self,
        op_handler: OpHandler[Any],
        dem_decoder: LeakageDecoderLike,
        batch_size: int = _DEFAULT_BATCH,
        seed: int | None = None,
    ) -> None:
        if batch_size is _DEFAULT_BATCH:
            batch_size = self._default_batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.op_handler = op_handler
        self.dem_decoder = as_leakage_decoder(dem_decoder, type(self).__name__)
        self.batch_size = int(batch_size)
        self.seed = seed
        self._init_pid = os.getpid()
        self._compile_count = 0

    def _next_compiled_seed(self) -> int | None:
        idx = self._compile_count
        self._compile_count = idx + 1
        return next_compiled_seed(self.seed, idx, (os.getpid() - self._init_pid) & 0xFFFFFFFF)

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        if task.circuit is None:
            raise ValueError(
                f"{type(self).__name__} requires a circuit in the task to compile a sampler."
            )
        circuit = task.circuit
        return self._compiled_cls(
            circuit=circuit,
            compiled_op_handler=self.op_handler.compile_op_handler(
                circuit=circuit, batch_size=self.batch_size
            ),
            batch_size=self.batch_size,
            compiled_dem_decoder=self.dem_decoder.compile_for_task(task),
            needs_records=self.dem_decoder.needs_records,
            needs_leakage_events=self.dem_decoder.needs_leakage_events,
            seed=self._next_compiled_seed(),
            postselection_mask=task.postselection_mask,
            postselected_observables_mask=task.postselected_observables_mask,
        )


class CompiledSamplerBase(sinter.CompiledSampler):
    """Common Sinter compiled-sampler runner delegating ``sample()`` to ``sample_batches``."""

    simulator: Any
    compiled_dem_decoder: CompiledLeakageDecoder
    needs_records: bool
    needs_leakage_events: bool
    postselection_mask: np.ndarray | None = None
    postselected_observables_mask: np.ndarray | None = None
    _track_leakage_count: bool = False

    def __init__(
        self,
        simulator: Any,
        compiled_dem_decoder: CompiledLeakageDecoder,
        *,
        needs_records: bool,
        needs_leakage_events: bool = False,
        postselection_mask: np.ndarray | None = None,
        postselected_observables_mask: np.ndarray | None = None,
    ) -> None:
        self.simulator = simulator
        self.batch_size = int(simulator.batch_size)
        self.compiled_dem_decoder = compiled_dem_decoder
        self.needs_records = bool(needs_records)
        self.needs_leakage_events = bool(needs_leakage_events)
        self.postselection_mask = postselection_mask
        self.postselected_observables_mask = postselected_observables_mask

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        return sample_batches(
            self.simulator,
            self.compiled_dem_decoder,
            suggested_shots,
            needs_records=self.needs_records,
            needs_leakage_events=self.needs_leakage_events,
            track_leakage_count=self._track_leakage_count,
            postselection_mask=self.postselection_mask,
            postselected_observables_mask=self.postselected_observables_mask,
        )
