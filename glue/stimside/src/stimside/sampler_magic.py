"""MagicsideSampler: sinter.Sampler integration for MagicsideSimulator."""

from __future__ import annotations

import collections
import os
import time
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside.dem_generators.leakage_decoder import (
    BaseDecoder, CompiledLeakageDecoder, LeakageDecoder, LeakageDecoderLike, as_leakage_decoder,
)
from stimside.op_handlers.abstract_op_handler import OpHandler, _TrivialOpHandler
from stimside.sampler_tableau import _CompiledSeedMixin
from stimside.simulator_magic import MagicsideSimulator, _BACKENDS, _FAULT_MODES, _M2D_MODES, _parse_enum, _resolve_budget
from stimside.util.magic_compile import CliffordFaultChannel
from stimside.util.magic_query import QueryBudget
from stimside.util.phase_qfe import DEFAULT_ATOL, validate_atol
from stimside.util.seeding import derived_seed


def _coerce_magic_decoder(dem_decoder: LeakageDecoderLike | sinter.Decoder | str | None) -> LeakageDecoder | None:
    if dem_decoder is None:
        return None
    return BaseDecoder(decoder=dem_decoder) if isinstance(dem_decoder, (str, sinter.Decoder)) else as_leakage_decoder(dem_decoder, "MagicsideSampler")


class MagicsideSampler(_CompiledSeedMixin, sinter.Sampler):
    """Sinter sampler running MagicsideSimulator with an arbitrary OpHandler and LeakageDecoder."""

    def __init__(
        self, op_handler: OpHandler[Any] | None = None,
        dem_decoder: LeakageDecoderLike | sinter.Decoder | str | None = None,
        batch_size: int = 256, seed: int | None = None, *,
        fault_mode: str = "DIRECT", backend: str = "STREAM",
        use_cpp_kernels: bool = True, m2d_reference_mode: str = "auto",
        reference_sample: Sequence[int | bool] | NDArray[np.bool_] | None = None,
        fault_channels: Sequence[CliffordFaultChannel] | None = None,
        query_budget: QueryBudget | None = None, max_components: int | None = None,
        on_budget_exceeded: str | None = None, atol: float = DEFAULT_ATOL,
    ) -> None:
        if op_handler is not None and not isinstance(op_handler, OpHandler):
            raise TypeError(f"Expected OpHandler or None, got {type(op_handler).__name__}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.op_handler, self.dem_decoder, self.batch_size = op_handler, _coerce_magic_decoder(dem_decoder), int(batch_size)
        self.fault_mode: Literal["DIRECT", "SELECTOR"] = _parse_enum(fault_mode, _FAULT_MODES, "fault_mode")
        self.backend: Literal["STREAM", "QUERY"] = _parse_enum(backend, _BACKENDS, "backend")
        if (self.backend == "QUERY" or self.fault_mode == "SELECTOR") and op_handler is not None and not isinstance(op_handler, _TrivialOpHandler):
            raise ValueError(f"backend={self.backend!r}, fault_mode={self.fault_mode!r} is incompatible with non-trivial op_handler")
        self.use_cpp_kernels = bool(use_cpp_kernels)
        self.m2d_reference_mode: Literal["auto", "magicside", "stim", "zero"] = _parse_enum(m2d_reference_mode, _M2D_MODES, "m2d_reference_mode", lower=True)
        self.reference_sample = None if reference_sample is None else np.asarray(reference_sample, dtype=np.bool_).copy()
        self.fault_channels = tuple(fault_channels) if fault_channels is not None else None
        self.atol = validate_atol(atol)
        if seed is not None:
            derived_seed(seed, 0)
        self.seed = seed
        self.query_budget, self.max_components, self.on_budget_exceeded = _resolve_budget(
            query_budget, max_components, on_budget_exceeded
        )
        self._init_pid, self._compile_count = os.getpid(), 0
        self._spawn_count, self._worker_index = 0, None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        w_idx = self._spawn_count
        self._spawn_count = w_idx + 1
        state["_worker_index"] = w_idx
        return state

    def _next_compiled_seed(self) -> int | None:
        if self._worker_index is None:
            return super()._next_compiled_seed()
        if self.seed is None:
            return None
        idx = self._compile_count
        self._compile_count = idx + 1
        if idx == 0 and self._worker_index == 0:
            return int(self.seed)
        ss = np.random.SeedSequence([int(self.seed), int(idx), int(self._worker_index)])
        return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x3FFFFFFF)

    def compiled_sampler_for_task(self, task: sinter.Task) -> CompiledMagicsideSampler:
        if task.circuit is None:
            raise ValueError("MagicsideSampler requires a circuit in the task.")
        eff_dec = self.dem_decoder or BaseDecoder(
            decoder=task.decoder if getattr(task, "decoder", None) in sinter.BUILT_IN_DECODERS else "pymatching"
        )
        compiled_handler = None if self.op_handler is None else self.op_handler.compile_op_handler(circuit=task.circuit, batch_size=self.batch_size)
        return CompiledMagicsideSampler(
            circuit=task.circuit, compiled_op_handler=compiled_handler,
            compiled_decoder=eff_dec.compile_for_task(task),
            needs_records=eff_dec.needs_records, needs_leakage_events=eff_dec.needs_leakage_events,
            batch_size=self.batch_size, fault_mode=self.fault_mode, backend=self.backend,
            use_cpp_kernels=self.use_cpp_kernels, seed=self._next_compiled_seed(),
            m2d_reference_mode=self.m2d_reference_mode, reference_sample=self.reference_sample,
            fault_channels=self.fault_channels, query_budget=self.query_budget,
            max_components=self.max_components, on_budget_exceeded=self.on_budget_exceeded, atol=self.atol,
        )


class CompiledMagicsideSampler(sinter.CompiledSampler):
    """Task-compiled MagicsideSampler executing batches in MagicsideSimulator and decoding via Sinter."""

    def __init__(
        self, *, circuit: stim.Circuit, compiled_op_handler: Any,
        compiled_decoder: CompiledLeakageDecoder, needs_records: bool,
        needs_leakage_events: bool = False, batch_size: int = 256,
        fault_mode: str = "DIRECT", backend: str = "STREAM",
        use_cpp_kernels: bool = True, seed: int | None = None,
        m2d_reference_mode: str = "auto",
        reference_sample: Sequence[int | bool] | NDArray[np.bool_] | None = None,
        fault_channels: Sequence[CliffordFaultChannel] | None = None,
        query_budget: QueryBudget | None = None, max_components: int | None = None,
        on_budget_exceeded: str | None = None, atol: float = DEFAULT_ATOL,
    ) -> None:
        self.batch_size, self.compiled_decoder = int(batch_size), compiled_decoder
        self.needs_records, self.needs_leakage_events = bool(needs_records), bool(needs_leakage_events)
        self.simulator = MagicsideSimulator(
            circuit=circuit, compiled_op_handler=compiled_op_handler, batch_size=self.batch_size,
            fault_mode=fault_mode, backend=backend, use_cpp_kernels=use_cpp_kernels, seed=seed,
            m2d_reference_mode=m2d_reference_mode, reference_sample=reference_sample,
            fault_channels=fault_channels, query_budget=query_budget, max_components=max_components,
            on_budget_exceeded=on_budget_exceeded, record_unleaked_to_leaked=True,
            record_leakage_events=self.needs_leakage_events, atol=atol,
        )

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        t0, stats = time.monotonic(), sinter.AnonTaskStats()
        while stats.shots < suggested_shots:
            self.simulator.clear()
            u2l_list = self.simulator.run()
            assert u2l_list is not None
            records = self.simulator.get_final_measurement_records() if self.needs_records else None
            leakage_events = self.simulator.get_leakage_events() if self.needs_leakage_events else None
            packed_dets = np.packbits(self.simulator.get_detector_flips(), axis=1, bitorder="little")
            packed_obs = np.packbits(self.simulator.get_observable_flips(), axis=1, bitorder="little")
            packed_preds = self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=packed_dets, records=records, leakage_events=leakage_events,
            )
            num_errors = int(np.count_nonzero(np.any(packed_preds != packed_obs, axis=1)))
            leak_cnt = sum(len(arr) for arr in u2l_list)
            counts = collections.Counter({"leakage_count": leak_cnt}) if leak_cnt else collections.Counter()
            stats += sinter.AnonTaskStats(
                shots=self.batch_size, errors=num_errors, discards=0, seconds=0.0, custom_counts=counts,
            )
        return sinter.AnonTaskStats(
            shots=stats.shots, errors=stats.errors, discards=stats.discards,
            seconds=time.monotonic() - t0, custom_counts=stats.custom_counts,
        )
