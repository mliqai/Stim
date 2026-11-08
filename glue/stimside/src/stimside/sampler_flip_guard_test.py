"""FlipsideSampler rejects skip-semantics decoders hidden in a CallableDecoder (WP-22 item 7)."""

from __future__ import annotations

import contextlib
import signal

import pytest
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside.dem_generators import (
    CallableDecoder,
    MarginalDecoder,
    MarginalLeakageDemGenerator,
)
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8Flip
from stimside.sampler_flip import FlipsideSampler, _reject_skip_semantics_decoder

_LEAK_TASK = sinter.Task(
    circuit=stim.Circuit(
        "R 0\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nM 0\nDETECTOR rec[-1]\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
)


class _CallableMarginalDecoder(MarginalDecoder):
    """A MarginalDecoder with `__call__`: the only kind a CallableDecoder can actually run."""

    def __call__(self, circuit, dets, records=None):
        raise AssertionError("never sampled")


@pytest.mark.parametrize(
    "wrap",
    [
        lambda d: CallableDecoder(d),
        lambda d: CallableDecoder(CallableDecoder(d)),
        lambda d: CallableDecoder(_CallableMarginalDecoder(d.dem_gen), needs_records=False),
    ],
    ids=["wrapped", "double_wrapped", "callable_subclass"],
)
def test_flipside_sampler_rejects_marginal_decoder_inside_a_callable_decoder(wrap) -> None:
    decoder = wrap(MarginalDecoder(MarginalLeakageDemGenerator()))
    sampler = FlipsideSampler(LeakageUint8Flip(), decoder)
    with pytest.raises(ValueError, match="skip"):
        sampler.compiled_sampler_for_task(_LEAK_TASK)


@contextlib.contextmanager
def _fail_instead_of_hanging(seconds: int = 3):
    def on_alarm(signum, frame):
        raise AssertionError(f"still running after {seconds} s: infinite loop")

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _noop(circuit, dets, records=None):
    return dets


def _cyclic_decoders() -> dict[str, CallableDecoder]:
    self_cycle = CallableDecoder(_noop)
    self_cycle.fn = self_cycle
    a, b = CallableDecoder(_noop), CallableDecoder(_noop)
    a.fn, b.fn = b, a
    tail = CallableDecoder(CallableDecoder(_noop))  # tail -> w -> tail
    tail.fn.fn = tail
    return {"self_cycle": self_cycle, "two_cycle": a, "cycle_behind_a_wrapper": tail}


@pytest.mark.parametrize("name", ["self_cycle", "two_cycle", "cycle_behind_a_wrapper"])
def test_cyclic_callable_decoder_does_not_hang_the_skip_semantics_guard(name: str) -> None:
    # A cycle holds no MarginalDecoder, so (as at the baseline) nothing is rejected: no hang.
    decoder = _cyclic_decoders()[name]
    with _fail_instead_of_hanging():
        _reject_skip_semantics_decoder(decoder)
        sampler = FlipsideSampler(LeakageUint8Flip(), decoder)
        assert sampler.compiled_sampler_for_task(_LEAK_TASK) is not None


def test_acyclic_callable_decoder_chain_ending_in_a_noop_is_still_accepted() -> None:
    with _fail_instead_of_hanging():
        _reject_skip_semantics_decoder(CallableDecoder(CallableDecoder(_noop)))
