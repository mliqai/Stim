from __future__ import annotations

import stim  # type: ignore[import-untyped]

from stimside._sampling import CompiledSamplerBase, SamplerBase
from stimside.dem_generators.leakage_decoder import CompiledLeakageDecoder
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.simulator_tableau import TablesideSimulator


class CompiledTablesideSampler(CompiledSamplerBase):
    def __init__(
        self,
        circuit: stim.Circuit,
        compiled_op_handler: CompiledOpHandler[TablesideSimulator],
        batch_size: int,
        compiled_dem_decoder: CompiledLeakageDecoder,
        *,
        needs_records: bool,
        needs_leakage_events: bool,
        seed: int | None = None,
        postselection_mask: object = None,
        postselected_observables_mask: object = None,
    ) -> None:
        self.tab_simulator = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=batch_size,
            seed=seed,
            record_leakage_events=needs_leakage_events,
        )
        super().__init__(
            self.tab_simulator,
            compiled_dem_decoder,
            needs_records=needs_records,
            needs_leakage_events=needs_leakage_events,
            postselection_mask=postselection_mask,  # type: ignore[arg-type]
            postselected_observables_mask=postselected_observables_mask,  # type: ignore[arg-type]
        )


class TablesideSampler(SamplerBase):
    _compiled_cls = CompiledTablesideSampler
    _default_batch_size = 1
