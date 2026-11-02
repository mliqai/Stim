import sinter
import stim # type: ignore[import-untyped]

from stimside.dem_generators.dem_decoding import decode_with_generated_dems
from stimside.dem_generators.leakage_decoder import (
    CompiledLeakageDecoder,
    BaseDecoder,
    LeakageDecoder,
    MarginalDecoder,
)
from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
from stimside.sampler_tableau import TablesideSampler


class _CallableCompiled(CompiledLeakageDecoder):
    def __init__(self, dem_gen, circuit):
        self.dem_gen = dem_gen
        self.circuit = circuit

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data, records=None, leakage_events=None
    ):
        return decode_with_generated_dems(
            "pymatching",
            self.dem_gen(self.circuit, records),
            bit_packed_detection_event_data,
        )


class _CallableDecoder(LeakageDecoder):
    """A custom LeakageDecoder decoding with the DEM(s) of dem_gen(circuit, records)."""

    def __init__(self, dem_gen):
        self.dem_gen = dem_gen

    @property
    def name(self):
        return "callable"

    @property
    def needs_records(self):
        return True

    def compile_for_task(self, task):
        return _CallableCompiled(self.dem_gen, task.circuit)


def test_sampler():

    op_handler = TrivialOpHandler()

    sampler = TablesideSampler(op_handler=op_handler, dem_decoder=BaseDecoder())

    circuit = stim.Circuit(
        """
        R 0 1
        H 0 1
        M 0 1
    """
    )

    task = sinter.Task(circuit=circuit, decoder=sinter.BUILT_IN_DECODERS["pymatching"])

    compiled_sampler = sampler.compiled_sampler_for_task(task=task)
    assert compiled_sampler is not None


def test_sampler_sample_and_dem_gen():
    import pytest

    op_handler = TrivialOpHandler()
    circuit = stim.Circuit(
        """
        R 0 1
        X_ERROR(0.1) 0
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    )

    called_with_records = []

    def custom_dem_gen(circ: stim.Circuit, records):
        called_with_records.append(records.shape)
        return circ.detector_error_model()

    sampler = TablesideSampler(
        op_handler=op_handler, batch_size=4, dem_decoder=_CallableDecoder(custom_dem_gen)
    )
    task = sinter.Task(circuit=circuit, decoder="pymatching")
    compiled = sampler.compiled_sampler_for_task(task=task)
    stats = compiled.sample(suggested_shots=4)
    assert stats.shots == 4
    assert called_with_records == [(4, 2)]

    with pytest.raises(ValueError, match="MarginalDecoder requires a decoder"):
        MarginalDecoder(decoder=None)


def test_batched_sampler_without_records_counts_leak_errors():
    # The leaked qubit skips the X, so in every shot L0 differs from the circuit's
    # reference and a vacuous decoder (which doesn't need the records) errs.
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8

    circuit = stim.Circuit(
        "R 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nX 0\nM 0\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )
    sampler = TablesideSampler(
        op_handler=LeakageUint8(),
        batch_size=10,
        dem_decoder=BaseDecoder(decoder="vacuous"),
        seed=0,
    )
    stats = sampler.compiled_sampler_for_task(sinter.Task(circuit=circuit)).sample(
        suggested_shots=20
    )
    assert stats.shots == 20
    assert stats.errors == 20


def test_6_tableside_batch_size_greater_than_one_and_sampler():
    from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
        LeakageUint8 as LeakageUint8Tableau,
    )
    from stimside.simulator_tableau import TablesideSimulator

    circuit = stim.Circuit(
        """
        R 0 1
        H 0
        CX 0 1
        I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0
        M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1) (0.8, 2)] 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        """
    )
    handler = LeakageUint8Tableau().compile_op_handler(circuit=circuit, batch_size=6)
    sim = TablesideSimulator(
        circuit,
        compiled_op_handler=handler,
        seed=77,
        batch_size=6,
        record_unleaked_to_leaked=True,
    )
    t_ev = sim.run()
    assert sim.get_final_measurement_records().shape == (6, 2)
    assert sim.get_detector_flips().shape == (6, 1)
    assert sim.get_observable_flips().shape == (6, 1)
    assert len(t_ev) == 6

    sampler = TablesideSampler(
        op_handler=LeakageUint8Tableau(),
        seed=77,
        batch_size=4,
        dem_decoder=BaseDecoder(decoder="vacuous"),
    )
    compiled = sampler.compiled_sampler_for_task(sinter.Task(circuit=circuit))
    stats = compiled.sample(8)
    assert stats.shots == 8


def test_bit_packed_detector_and_observable_getters_across_engines() -> None:
    import numpy as np
    from stimside.simulator_coset import CosetsideSimulator
    from stimside.simulator_magic import MagicsideSimulator
    from stimside.simulator_tableau import TablesideSimulator

    lines = ["R " + " ".join(map(str, range(10))), "X_ERROR(0.3) " + " ".join(map(str, range(10))), "M " + " ".join(map(str, range(10)))]
    for i in range(10):
        lines.append(f"DETECTOR rec[-{10 - i}]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]")
    lines.append("OBSERVABLE_INCLUDE(1) rec[-2] rec[-4]")
    circ = stim.Circuit("\n".join(lines))

    th = TrivialOpHandler()
    sims = [
        TablesideSimulator(circ, compiled_op_handler=th.compile_op_handler(circuit=circ, batch_size=7), batch_size=7, seed=11),
        CosetsideSimulator(circ, compiled_op_handler=th.compile_op_handler(circuit=circ, batch_size=7), batch_size=7, seed=11),
        MagicsideSimulator(circ, op_handler=th, batch_size=7, seed=11),
    ]
    for sim in sims:
        sim.run()
        dets = sim.get_detector_flips()
        obs = sim.get_observable_flips()
        both = sim.get_detector_flips(append_observables=True)
        dets_p = sim.get_detector_flips(bit_packed=True)
        obs_p = sim.get_observable_flips(bit_packed=True)
        both_p = sim.get_detector_flips(append_observables=True, bit_packed=True)
        assert dets_p.shape == (7, 2) and dets_p.dtype == np.uint8
        assert obs_p.shape == (7, 1) and obs_p.dtype == np.uint8
        assert both_p.shape == (7, 2) and both_p.dtype == np.uint8
        np.testing.assert_array_equal(dets_p, np.packbits(dets, axis=-1, bitorder="little"))
        np.testing.assert_array_equal(obs_p, np.packbits(obs, axis=-1, bitorder="little"))
        np.testing.assert_array_equal(both_p, np.packbits(both, axis=-1, bitorder="little"))

