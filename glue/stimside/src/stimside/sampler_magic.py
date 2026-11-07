"""MagicsideSampler: sinter.Sampler integration for MagicsideSimulator."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside._magic.gates import CliffordFaultChannel
from stimside._magic.phase_qfe import DEFAULT_ATOL, validate_atol
from stimside._magic.query import QueryBudget
from stimside._sampling import CompiledSamplerBase, SamplerBase
from stimside.dem_generators.leakage_decoder import (
    BaseDecoder,
    CompiledLeakageDecoder,
    LeakageDecoder,
    LeakageDecoderLike,
)
from stimside.op_handlers.abstract_op_handler import OpHandler, TrivialOpHandler
from stimside.simulator_magic import (
    BACKENDS,
    FAULT_MODES,
    M2D_MODES,
    MagicsideSimulator,
    parse_enum,
    resolve_budget,
)
from stimside.util.seeding import derived_seed


class _TaskDecoder(LeakageDecoder):
    """``dem_decoder=None``: use ``task.decoder`` if sinter has it built in, else pymatching."""

    name = "task_decoder"

    def compile_for_task(self, task: sinter.Task) -> CompiledLeakageDecoder:
        decoder = getattr(task, "decoder", None)
        if decoder not in sinter.BUILT_IN_DECODERS:
            warnings.warn(
                f"MagicsideSampler(dem_decoder=None): task.decoder={decoder!r} is not a sinter "
                "built-in decoder, so 'pymatching' is used instead. Pass dem_decoder to choose.",
                UserWarning,
                stacklevel=3,
            )
            decoder = "pymatching"
        return BaseDecoder(decoder=decoder).compile_for_task(task)


class MagicsideSampler(SamplerBase):
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
        if dem_decoder is None:
            dem_decoder = _TaskDecoder()
        elif isinstance(dem_decoder, (str, sinter.Decoder)):
            dem_decoder = BaseDecoder(decoder=dem_decoder)
        super().__init__(op_handler, dem_decoder, batch_size, seed)
        fm: Literal["DIRECT", "SELECTOR"] = parse_enum(fault_mode, FAULT_MODES, "fault_mode")
        be: Literal["STREAM", "QUERY"] = parse_enum(backend, BACKENDS, "backend")
        trivial = op_handler is None or isinstance(op_handler, TrivialOpHandler)
        if (be == "QUERY" or fm == "SELECTOR") and not trivial:
            raise ValueError(
                f"backend={be!r}, fault_mode={fm!r} is incompatible with non-trivial op_handler"
            )
        if seed is not None:
            derived_seed(seed, 0)
        qb, mc, obe = resolve_budget(query_budget, max_components, on_budget_exceeded)
        self._sim_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "fault_mode": fm,
            "backend": be,
            "m2d_reference_mode": parse_enum(m2d_reference_mode, M2D_MODES, "m2d_reference_mode", lower=True),
            "reference_sample": None if reference_sample is None else np.asarray(reference_sample, dtype=np.bool_).copy(),
            "fault_channels": tuple(fault_channels) if fault_channels is not None else None,
            "query_budget": qb,
            "max_components": mc,
            "on_budget_exceeded": obe,
            "atol": validate_atol(atol),
        }

    def _make_compiled(
        self,
        task: sinter.Task,
        compiled_handler: Any,
        compiled_decoder: CompiledLeakageDecoder,
        seed: int | None,
    ) -> CompiledMagicsideSampler:
        return CompiledMagicsideSampler(
            circuit=task.circuit,
            compiled_op_handler=compiled_handler,
            compiled_dem_decoder=compiled_decoder,
            needs_records=self.dem_decoder.needs_records,
            needs_leakage_events=self.dem_decoder.needs_leakage_events,
            seed=seed,
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
        compiled_dem_decoder: CompiledLeakageDecoder | None = None,
        compiled_decoder: CompiledLeakageDecoder | None = None,
        needs_records: bool,
        needs_leakage_events: bool = False,
        batch_size: int = 256,
        seed: int | None = None,
        postselection_mask: NDArray[np.uint8] | None = None,
        postselected_observables_mask: NDArray[np.uint8] | None = None,
        **sim_kwargs: Any,
    ) -> None:
        if (compiled_dem_decoder is None) == (compiled_decoder is None):
            raise TypeError("Pass exactly one of compiled_dem_decoder or compiled_decoder.")
        eff_compiled_decoder = compiled_dem_decoder if compiled_dem_decoder is not None else compiled_decoder
        assert eff_compiled_decoder is not None
        sim = MagicsideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=int(batch_size),
            seed=seed,
            record_unleaked_to_leaked=True,
            record_leakage_events=bool(needs_leakage_events),
            **sim_kwargs,
        )
        super().__init__(
            sim,
            eff_compiled_decoder,
            needs_records=needs_records,
            needs_leakage_events=needs_leakage_events,
            postselection_mask=postselection_mask,
            postselected_observables_mask=postselected_observables_mask,
        )
