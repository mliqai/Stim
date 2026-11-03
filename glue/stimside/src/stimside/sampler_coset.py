from __future__ import annotations

import stim  # type: ignore[import-untyped]

from stimside._sampling import CompiledSamplerBase, SamplerBase
from stimside.dem_generators.leakage_decoder import CompiledLeakageDecoder
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.simulator_coset import CosetsideSimulator


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
        seed: int | None = None,
        postselection_mask: object = None,
        postselected_observables_mask: object = None,
    ) -> None:
        sim = CosetsideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=batch_size,
            seed=seed,
            record_leakage_events=needs_leakage_events,
        )
        super().__init__(
            sim,
            compiled_dem_decoder,
            needs_records=needs_records,
            needs_leakage_events=needs_leakage_events,
            postselection_mask=postselection_mask,  # type: ignore[arg-type]
            postselected_observables_mask=postselected_observables_mask,  # type: ignore[arg-type]
        )


class CosetsideSampler(SamplerBase):
    """Sinter sampler for CosetsideSimulator with multi-shot CPU batching."""

    _compiled_cls = CompiledCosetsideSampler
    _default_batch_size = 256
