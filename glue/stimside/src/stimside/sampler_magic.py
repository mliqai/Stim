"""MagicsideSampler: sinter.Sampler integration for MagicsideSimulator."""

from __future__ import annotations

import os
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside._magic.gates import CliffordFaultChannel
from stimside._magic.phase_qfe import DEFAULT_ATOL, validate_atol
from stimside._magic.query import QueryBudget
from stimside._sampling import CompiledSamplerBase, next_compiled_seed
from stimside.dem_generators.leakage_decoder import (
    BaseDecoder,
    CompiledLeakageDecoder,
    LeakageDecoder,
    LeakageDecoderLike,
    as_leakage_decoder,
)
from stimside.op_handlers.abstract_op_handler import OpHandler, TrivialOpHandler
from stimside.simulator_magic import (
    MagicsideSimulator,
    _BACKENDS,
    _FAULT_MODES,
    _M2D_MODES,
    _parse_enum,
    _resolve_budget,
)
from stimside.util.seeding import derived_seed


def _coerce_magic_decoder(
    dem_decoder: LeakageDecoderLike | sinter.Decoder | str | None,
) -> LeakageDecoder | None:
    if dem_decoder is None:
        return None
    return (
        BaseDecoder(decoder=dem_decoder)
        if isinstance(dem_decoder, (str, sinter.Decoder))
        else as_leakage_decoder(dem_decoder, "MagicsideSampler")
    )


class MagicsideSampler(sinter.Sampler):
    """Sinter sampler running MagicsideSimulator with an arbitrary OpHandler and LeakageDecoder."""

    def __init__(
        self,
        op_handler: OpHandler[Any] | None = None,
        dem_decoder: LeakageDecoderLike | sinter.Decoder | str | None = None,
        batch_size: int = 256,
        seed: int | None = None,
        *,
        fault_mode: str = "DIRECT",
        backend: str = "STREAM",
        m2d_reference_mode: str = "auto",
        reference_sample: Sequence[int | bool] | NDArray[np.bool_] | None = None,
        fault_channels: Sequence[CliffordFaultChannel] | None = None,
        query_budget: QueryBudget | None = None,
        max_components: int | None = None,
        on_budget_exceeded: str | None = None,
        atol: float = DEFAULT_ATOL,
    ) -> None:
        if op_handler is not None and not isinstance(op_handler, OpHandler):
            raise TypeError(f"Expected OpHandler or None, got {type(op_handler).__name__}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.op_handler, self.dem_decoder, self.batch_size = (
            op_handler,
            _coerce_magic_decoder(dem_decoder),
            int(batch_size),
        )
        fm: Literal["DIRECT", "SELECTOR"] = _parse_enum(fault_mode, _FAULT_MODES, "fault_mode")
        be: Literal["STREAM", "QUERY"] = _parse_enum(backend, _BACKENDS, "backend")
        if (be == "QUERY" or fm == "SELECTOR") and op_handler is not None and not isinstance(op_handler, TrivialOpHandler):
            raise ValueError(f"backend={be!r}, fault_mode={fm!r} is incompatible with non-trivial op_handler")
        if seed is not None:
            derived_seed(seed, 0)
        self.seed = seed
        qb, mc, obe = _resolve_budget(query_budget, max_components, on_budget_exceeded)
        self._sim_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "fault_mode": fm,
            "backend": be,
            "m2d_reference_mode": _parse_enum(m2d_reference_mode, _M2D_MODES, "m2d_reference_mode", lower=True),
            "reference_sample": None if reference_sample is None else np.asarray(reference_sample, dtype=np.bool_).copy(),
            "fault_channels": tuple(fault_channels) if fault_channels is not None else None,
            "query_budget": qb,
            "max_components": mc,
            "on_budget_exceeded": obe,
            "atol": validate_atol(atol),
        }
        self._init_pid, self._compile_count = os.getpid(), 0
        self._spawn_count, self._worker_index = 0, None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        w_idx = self._spawn_count
        self._spawn_count = w_idx + 1
        state["_worker_index"] = w_idx
        return state

    def _next_compiled_seed(self) -> int | None:
        idx = self._compile_count
        self._compile_count = idx + 1
        salt = int(self._worker_index) if self._worker_index is not None else ((os.getpid() - self._init_pid) & 0xFFFFFFFF)
        return next_compiled_seed(self.seed, idx, salt)

    def compiled_sampler_for_task(self, task: sinter.Task) -> CompiledMagicsideSampler:
        if task.circuit is None:
            raise ValueError("MagicsideSampler requires a circuit in the task.")
        eff_dec = self.dem_decoder or BaseDecoder(
            decoder=task.decoder if getattr(task, "decoder", None) in sinter.BUILT_IN_DECODERS else "pymatching"
        )
        compiled_handler = (
            None
            if self.op_handler is None
            else self.op_handler.compile_op_handler(circuit=task.circuit, batch_size=self.batch_size)
        )
        return CompiledMagicsideSampler(
            circuit=task.circuit,
            compiled_op_handler=compiled_handler,
            compiled_decoder=eff_dec.compile_for_task(task),
            needs_records=eff_dec.needs_records,
            needs_leakage_events=eff_dec.needs_leakage_events,
            seed=self._next_compiled_seed(),
            postselection_mask=task.postselection_mask,
            postselected_observables_mask=task.postselected_observables_mask,
            **self._sim_kwargs,
        )


class CompiledMagicsideSampler(CompiledSamplerBase):
    """Task-compiled MagicsideSampler executing batches in MagicsideSimulator and decoding via Sinter."""

    _track_leakage_count = True

    def __init__(
        self,
        *,
        circuit: stim.Circuit,
        compiled_op_handler: Any,
        compiled_decoder: CompiledLeakageDecoder,
        needs_records: bool,
        needs_leakage_events: bool = False,
        batch_size: int = 256,
        seed: int | None = None,
        postselection_mask: NDArray[np.uint8] | None = None,
        postselected_observables_mask: NDArray[np.uint8] | None = None,
        **sim_kwargs: Any,
    ) -> None:
        self._init_common(
            circuit,
            compiled_op_handler,
            batch_size,
            compiled_decoder,
            needs_records,
            needs_leakage_events,
            postselection_mask,
            postselected_observables_mask,
        )
        self.simulator = MagicsideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=self.batch_size,
            seed=seed,
            record_unleaked_to_leaked=True,
            record_leakage_events=self.needs_leakage_events,
            **sim_kwargs,
        )
