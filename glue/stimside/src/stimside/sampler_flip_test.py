import pytest
import sinter
import stim  # type: ignore[import-untyped]

from stimside.dem_generators import (
    BranchAndBoundDecoder,
    CallableDecoder,
    MarginalDecoder,
    MarginalLeakageDemGenerator,
    bnb_decoder,
)
from stimside.dem_generators.leakage_decoder import BaseDecoder
from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import (
    LeakageUint8 as LeakageUint8Flip,
)
from stimside.sampler_flip import FlipsideSampler
from stimside.sampler_tableau_test import _CallableDecoder


def test_sampler():

    op_handler = TrivialOpHandler()

    sampler = FlipsideSampler(op_handler=op_handler, dem_decoder=BaseDecoder())

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

    sampler_with_seed = FlipsideSampler(
        op_handler=op_handler, dem_decoder=BaseDecoder(), seed=42
    )
    task_str_decoder = sinter.Task(
        circuit=stim.Circuit(
            """
            R 0
            X_ERROR(0.1) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """
        ),
        decoder="pymatching",
    )
    compiled_with_seed = sampler_with_seed.compiled_sampler_for_task(
        task=task_str_decoder
    )
    assert compiled_with_seed.simulator.seed == 42
    stats = compiled_with_seed.sample(suggested_shots=128)
    assert stats.shots >= 128
    assert isinstance(stats.errors, int)


def test_flipside_sampler_with_dem_gen():
    circuit = stim.Circuit(
        "R 0\nX_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    flips = stim.DetectorErrorModel("error(0.1) D0 L0")
    keeps = stim.DetectorErrorModel("error(0.1) D0\nlogical_observable L0")

    def gen(circ, records):
        return [flips if row[0] else keeps for row in records]

    sampler = FlipsideSampler(
        op_handler=TrivialOpHandler(), batch_size=64, dem_decoder=_CallableDecoder(gen)
    )
    compiled = sampler.compiled_sampler_for_task(
        sinter.Task(circuit=circuit, decoder="pymatching")
    )
    stats = compiled.sample(suggested_shots=256)
    assert stats.shots >= 256
    assert stats.errors == 0


# A heralded leak: its DEM is empty, but FlipsideSimulator fires D0 half the time.
_LEAK_TASK = sinter.Task(
    circuit=stim.Circuit(
        "R 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0\nDETECTOR rec[-1]\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
)


def test_flipside_sampler_rejects_marginal_decoder():
    sampler = FlipsideSampler(LeakageUint8Flip(), MarginalDecoder(MarginalLeakageDemGenerator()))
    with pytest.raises(ValueError, match="skip"):
        sampler.compiled_sampler_for_task(_LEAK_TASK)


def test_flipside_sampler_rejects_branch_and_bound_decoder(monkeypatch):
    # Compiling it checks for the PyMatching fork, which only its decoding needs.
    monkeypatch.setattr(bnb_decoder, "_has_fork_matching", lambda: True)
    sampler = FlipsideSampler(LeakageUint8Flip(), BranchAndBoundDecoder())
    with pytest.raises(ValueError, match="skip"):
        sampler.compiled_sampler_for_task(_LEAK_TASK)


@pytest.mark.parametrize(
    "decoder", [BaseDecoder(), CallableDecoder(lambda circuit, dets, records=None: dets[:, :1] * 0)]
)
def test_flipside_sampler_accepts_base_and_callable_decoders(decoder):
    assert FlipsideSampler(LeakageUint8Flip(), decoder).compiled_sampler_for_task(_LEAK_TASK)
