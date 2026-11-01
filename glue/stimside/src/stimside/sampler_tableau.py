from __future__ import annotations

import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside._sampling import CompiledSamplerBase, SamplerBase
from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    LeakageDecoderLike,
)
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.simulator_tableau import TablesideSimulator


class TablesideSampler(SamplerBase):
    def __init__(
        self,
        op_handler: OpHandler[TablesideSimulator],
        dem_decoder: LeakageDecoderLike,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(op_handler, dem_decoder, batch_size, seed)

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        circuit = self._require_circuit(task)
        return CompiledTablesideSampler(
            circuit=circuit,
            compiled_op_handler=self.op_handler.compile_op_handler(
                circuit=circuit, batch_size=self.batch_size
            ),
            batch_size=self.batch_size,
            compiled_dem_decoder=self.dem_decoder.compile_for_task(task),
            needs_records=self.dem_decoder.needs_records,
            needs_leakage_events=self.dem_decoder.needs_leakage_events,
            seed=self._next_compiled_seed(),
        )


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
    ) -> None:
        self._init_common(
            circuit,
            compiled_op_handler,
            batch_size,
            compiled_dem_decoder,
            needs_records,
            needs_leakage_events,
        )
        self.tab_simulator = TablesideSimulator(
            circuit=circuit,
            compiled_op_handler=compiled_op_handler,
            batch_size=batch_size,
            seed=seed,
            record_leakage_events=needs_leakage_events,
        )
        self.simulator = self.tab_simulator
