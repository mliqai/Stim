from __future__ import annotations

import sinter
import stim  # type: ignore[import-untyped]

from stimside._sampling import CompiledSamplerBase, SamplerBase, _DEFAULT_BATCH
from stimside.dem_generators import BranchAndBoundDecoder, MarginalDecoder
from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    LeakageDecoder,
    LeakageDecoderLike,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.simulator_flip import FlipsideSimulator


def _reject_loss_oracle(needs_leakage_events: bool) -> None:
    if needs_leakage_events:
        raise ValueError(
            "FlipsideSampler cannot run a dem_decoder that needs the shots' leakage events "
            "(e.g. MarginalLeakageDemGenerator(loss_oracle=True)): FlipsideSimulator does "
            "not record them. Use TablesideSampler or CosetsideSampler."
        )


def _reject_skip_semantics_decoder(dem_decoder: LeakageDecoder) -> None:
    if isinstance(dem_decoder, (MarginalDecoder, BranchAndBoundDecoder)):
        raise ValueError(
            f"FlipsideSampler cannot run {type(dem_decoder).__name__}: its DEMs model leaked "
            "qubits as frozen, with untagged gates skipped by default, but FlipsideSimulator "
            "depolarizes leaked qubits and applies the gates. Use TablesideSampler or "
            "CosetsideSampler, or BaseDecoder with FlipsideSampler."
        )


class CompiledFlipsideSampler(CompiledSamplerBase):
    def __init__(
        self,
        circuit: stim.Circuit,
        compiled_op_handler: CompiledOpHandler[FlipsideSimulator],
        batch_size: int,
        compiled_dem_decoder: CompiledLeakageDecoder,
        *,
        needs_records: bool,
        needs_leakage_events: bool = False,
        seed: int | None = None,
        postselection_mask: object = None,
        postselected_observables_mask: object = None,
    ) -> None:
        _reject_loss_oracle(needs_leakage_events)
        sim = FlipsideSimulator(
            circuit=circuit,
            batch_size=batch_size,
            compiled_op_handler=compiled_op_handler,
            seed=seed,
        )
        super().__init__(
            sim,
            compiled_dem_decoder,
            needs_records=needs_records,
            needs_leakage_events=needs_leakage_events,
            postselection_mask=postselection_mask,  # type: ignore[arg-type]
            postselected_observables_mask=postselected_observables_mask,  # type: ignore[arg-type]
        )


class FlipsideSampler(SamplerBase):
    _compiled_cls = CompiledFlipsideSampler
    _default_batch_size = 2**10

    def __init__(
        self,
        op_handler: OpHandler[FlipsideSimulator],
        dem_decoder: LeakageDecoderLike,
        batch_size: int = _DEFAULT_BATCH,
        seed: int | None = None,
    ) -> None:
        super().__init__(op_handler, dem_decoder, batch_size, seed)
        _reject_loss_oracle(self.dem_decoder.needs_leakage_events)

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        # After compiling, so that the decoder's own task errors come first (e.g. MarginalDecoder's
        # NotImplementedError for SWAP[LEAKAGE_SWAP]).
        compiled = super().compiled_sampler_for_task(task)
        _reject_skip_semantics_decoder(self.dem_decoder)
        return compiled
