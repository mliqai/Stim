from __future__ import annotations

import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside._sampling import CompiledSamplerBase, SamplerBase
from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    LeakageDecoderLike,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.simulator_coset import CosetsideSimulator


class CosetsideSampler(SamplerBase):
    """Sinter sampler for CosetsideSimulator with multi-shot CPU batching."""

    def __init__(
        self,
        op_handler: OpHandler[CosetsideSimulator],
        dem_decoder: LeakageDecoderLike,
        batch_size: int = 256,
        seed: int | None = None,
    ) -> None:
        super().__init__(op_handler, dem_decoder, batch_size, seed)

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        circuit = self._require_circuit(task)
        return CompiledCosetsideSampler(
            circuit=circuit,
            compiled_op_handler=self.op_handler.compile_op_handler(
                circuit=circuit, batch_size=self.batch_size
            ),
            batch_size=self.batch_size,
            compiled_dem_decoder=self.dem_decoder.compile_for_task(task),
            needs_records=self.dem_decoder.needs_records,
            needs_leakage_events=self.dem_decoder.needs_leakage_events,
            compiled_ref=CompiledReferenceCircuit(circuit),
            seed=self._next_compiled_seed(),
            postselection_mask=task.postselection_mask,
            postselected_observables_mask=task.postselected_observables_mask,
        )


class CompiledCosetsideSampler(CompiledSamplerBase):
    """Compiled sinter sampler executing B independent trajectories per batch in CosetsideSimulator."""

    def __init__(
        self,
        circuit: stim.Circuit,
        compiled_op_handler: CompiledOpHandler[CosetsideSimulator],
        batch_size: int,
        compiled_dem_decoder: CompiledLeakageDecoder,
        *,
        needs_records: bool,
        needs_leakage_events: bool,
        compiled_ref: CompiledReferenceCircuit | None = None,
        seed: int | None = None,
        postselection_mask: object = None,
        postselected_observables_mask: object = None,
    ) -> None:
        self._init_common(
            circuit,
            compiled_op_handler,
            batch_size,
            compiled_dem_decoder,
            needs_records,
            needs_leakage_events,
            postselection_mask,  # type: ignore[arg-type]
            postselected_observables_mask,  # type: ignore[arg-type]
        )
        self.simulator = CosetsideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=batch_size,
            compiled_ref=compiled_ref,
            seed=seed,
            record_leakage_events=needs_leakage_events,
        )
