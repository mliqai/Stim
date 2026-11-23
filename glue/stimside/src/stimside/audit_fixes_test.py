"""Regression tests for five defects that an audit found in the opt-in `fast_leakage` work.

Each defect has a test that fails on the tree before the fix and passes after it. Two more tests
pin behaviour that the fixes must keep: a noise instruction without targets and with probability
zero draws nothing, and `compile_op_handler` still records whether the option applies.

* Cosetside fast path, memory: two kernels read the leakage-state array of the handler, the same
  array that the public `handler.state` returns. Python code that ran during the call (a profile
  hook, a finalizer, a signal handler) could resize it in place, and the kernel then read past its
  end. Kernels now read a private copy whose shape is checked after copying.
* Cosetside fast path, randomness: a noise instruction without targets was skipped. stim draws a
  random number for such an instruction, and the general path hands it to stim, so the seeded
  outputs that followed it differed between the two paths.
* Constructor order: the option's fields were inserted before the existing parameters
  `u_measurement_ops` and `magic_ops` of `CompiledLeakageUint8Magic`, so a positional call bound
  them wrongly. `fast_leakage` is keyword-only now, and the two private fields are not parameters.
* Atomic rejection: with the option on, `interactive_do` rejected an instruction past the compiled
  circuit that may read a leaked qubit's stored state only when it reached it, after earlier
  instructions of the same call had run; Cosetside then refused every call until `clear()`. The
  check is now part of the validation that runs before anything changes.
* Magicside interrupt: an exception that arrived at the end of the entry of the native-section
  guard left the engine marked busy for good, so even `clear()` raised.
"""

from __future__ import annotations

import ctypes
import gc
import inspect
import sys
import warnings
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside.simulator_magic as sm
from stimside import (
    CompiledLeakageUint8Coset,
    CompiledLeakageUint8Magic,
    CompiledLeakageUint8Tableau,
    CosetsideSimulator,
    LeakageUint8Coset,
    LeakageUint8Magic,
    MagicsideSimulator,
)
from stimside._magic import live_store
from stimside._native import magic_engine

_QS = " ".join(str(q) for q in range(8))
_STATE_KERNELS = {"engine_cond_pauli_noise_v4": 6, "engine_batch_inject_unitary_v4": 5}


def _coset_records(text: str, *, fast_glue: bool, seed: int = 7) -> np.ndarray:
    circuit = stim.Circuit(text)
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=4)
    sim = CosetsideSimulator(circuit, compiled_op_handler=handler, seed=seed, batch_size=4)
    sim._use_fast_glue = fast_glue
    sim.run()
    return np.asarray(sim.get_final_measurement_records(), dtype=np.uint8)


class _StateSpy:
    """Stands in for the kernel library; checks which leakage-state buffer each kernel gets."""

    def __init__(self, lib: Any, handler: Any) -> None:
        self._lib = lib
        self._handler = handler
        self.calls = 0
        self.calls_given_handler_array = 0

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if name not in _STATE_KERNELS:
            return fn

        def spied(*args: Any) -> Any:
            address = ctypes.cast(args[_STATE_KERNELS[name]], ctypes.c_void_p).value
            if address:
                self.calls += 1
                # Both buffers are alive here, so equal addresses mean the same buffer.
                if address == self._handler._state.ctypes.data:
                    self.calls_given_handler_array += 1
            return fn(*args)

        return spied


_STATE_CIRCUIT = f"""
R {_QS}
H {_QS}
I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0 5
TICK
X_ERROR(0.5) {_QS}
TICK
CZ 0 1 2 3 4 5 6 7
TICK
DEPOLARIZE2(0.3) 0 1 2 3 4 5 6 7
M {_QS}
"""


def _state_sim() -> tuple[CosetsideSimulator, Any, _StateSpy]:
    circuit = stim.Circuit(_STATE_CIRCUIT)
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=64)
    sim = CosetsideSimulator(circuit, compiled_op_handler=handler, seed=3, batch_size=64)
    if sim._cpp_lib is None or not sim._use_cpp:
        pytest.skip("Cosetside C++ kernel library is unavailable")
    spy = _StateSpy(sim._cpp_lib, handler)
    sim._cpp_lib = spy  # type: ignore[assignment]
    return sim, handler, spy


def test_kernels_read_a_private_copy_of_the_leakage_state() -> None:
    """Neither state-reading kernel is ever handed the handler's own array."""
    sim, _, spy = _state_sim()
    sim.run()
    assert spy.calls, "no state-reading kernel ran: the test does not reach the fast path"
    assert spy.calls_given_handler_array == 0


@pytest.mark.parametrize(("where", "caller", "op_name", "raises"), [
    # Resized before the copy is made: the shape check on the copy stops the call.
    ("_match_cursor_noop", "_try_fast_leakage_op", "X_ERROR", True),
    ("_require_cpp", "_fast_cond_unitary_v4", None, True),
    # Resized after the copy is made: the kernel reads the whole private copy.
    ("_run_noise_v4", None, None, False),
])
def test_resizing_handler_state_during_a_call_is_safe(
    where: str, caller: str | None, op_name: str | None, raises: bool
) -> None:
    """A profile hook resizes `handler.state` in place once; no kernel reads the old extent."""
    sim, handler, spy = _state_sim()
    fired: list[str] = []

    def hook(frame: Any, event: str, arg: Any) -> None:
        if event != "call" or fired or frame.f_code.co_name != where:
            return
        back = frame.f_back
        if caller is not None and (back is None or back.f_code.co_name != caller):
            return
        if op_name is not None and frame.f_locals["op"].name != op_name:
            return
        fired.append(where)
        handler.state.resize((1, 1), refcheck=False)

    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        sim.run()
        outcome = "ran"
    except RuntimeError as exc:
        outcome = str(exc)
    finally:
        sys.setprofile(previous)
    assert fired == [where]
    if raises:
        assert "changed shape" in outcome
    else:
        assert outcome == "ran"
    assert spy.calls_given_handler_array == 0


@pytest.mark.parametrize("empty", [
    "DEPOLARIZE1(0.2)", "X_ERROR(0.1)", "DEPOLARIZE2(0.1)", "Z_ERROR[CONDITIONED_ON_SELF: U](0.2)",
])
def test_noise_instruction_without_targets_draws_like_the_general_path(empty: str) -> None:
    """The fast path hands a noise instruction without targets to stim, as the general path does."""
    text = f"R {_QS}\nH 0 1\n{empty}\nX_ERROR(0.5) {_QS}\nM {_QS}"
    general = _coset_records(text, fast_glue=False)
    np.testing.assert_array_equal(_coset_records(text, fast_glue=True), general)
    # stim's draw for the empty instruction is visible: the same circuit without it differs
    without = _coset_records(f"R {_QS}\nH 0 1\nX_ERROR(0.5) {_QS}\nM {_QS}", fast_glue=False)
    assert not np.array_equal(general, without)


def test_noise_instruction_without_targets_and_zero_probability_draws_nothing() -> None:
    with_empty = f"R {_QS}\nH 0 1\nX_ERROR(0)\nX_ERROR(0.5) {_QS}\nM {_QS}"
    without = f"R {_QS}\nH 0 1\nX_ERROR(0.5) {_QS}\nM {_QS}"
    for fast_glue in (True, False):
        np.testing.assert_array_equal(
            _coset_records(with_empty, fast_glue=fast_glue),
            _coset_records(without, fast_glue=fast_glue),
        )


def test_compiled_handlers_keep_their_positional_parameter_order() -> None:
    expected = {
        CompiledLeakageUint8Coset: ["num_qubits", "batch_size", "unconditional_condition_on_U",
                                    "ops_to_params"],
        CompiledLeakageUint8Tableau: ["num_qubits", "batch_size", "unconditional_condition_on_U",
                                      "ops_to_params"],
        CompiledLeakageUint8Magic: ["num_qubits", "batch_size", "unconditional_condition_on_U",
                                    "ops_to_params", "u_measurement_ops", "magic_ops"],
    }
    for cls, positional in expected.items():
        params = list(inspect.signature(cls.__init__).parameters.values())[1:]
        pos = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
        kw_only = [p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]
        assert pos == positional, cls.__name__
        assert kw_only == ["fast_leakage"], cls.__name__


def test_positional_compiled_magic_handler_matches_the_compiled_one() -> None:
    """Six positional arguments, in the order they had before the option, build the same handler."""
    circuit = stim.Circuit(
        "H 0\nI[T] 0\nMPAD[LEAKAGE_MEASUREMENT: (1.0, U) : 0] 0\nH 0\n"
        "I[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0"
    )
    batch = 2048
    ref = LeakageUint8Magic().compile_op_handler(circuit=circuit, batch_size=batch)
    hand = CompiledLeakageUint8Magic(
        circuit.num_qubits, batch, True, ref.ops_to_params, set(ref.u_measurement_ops),
        dict(ref.magic_ops),
    )
    assert hand.u_measurement_ops == ref.u_measurement_ops
    assert hand.magic_ops == ref.magic_ops
    assert hand.fast_leakage is False

    def records(h: Any) -> np.ndarray:
        sim = MagicsideSimulator(circuit, compiled_op_handler=h, batch_size=batch, seed=4)
        sim.run()
        return np.asarray(sim.get_final_measurement_records(), dtype=np.uint8)

    np.testing.assert_array_equal(records(hand), records(ref))


def test_compile_op_handler_still_records_the_qualification() -> None:
    circuit = stim.Circuit(
        "R 0 1\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nX 0\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 1"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = LeakageUint8Coset(fast_leakage=True).compile_op_handler(circuit=circuit, batch_size=4)
    assert h._fast_1q is True
    assert h._compiled_len == len(circuit)


def test_rejected_late_reader_leaves_cosetside_as_it_was() -> None:
    base = stim.Circuit("R 0 1 2\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\nX 0\nTICK")
    h = LeakageUint8Coset(fast_leakage=True).compile_op_handler(circuit=base, batch_size=4)
    sim = CosetsideSimulator(base, compiled_op_handler=h, batch_size=4, seed=3)
    sim.interactive_do(base)
    time_before = sim.get_current_circuit_time()
    peek_before = np.asarray(sim.peek_pauli_batch([2], "Z")).copy()
    with pytest.raises(ValueError, match="fast_leakage"):
        sim.interactive_do(stim.Circuit("X 2\nM 1"))
    assert sim.get_current_circuit_time() == time_before
    np.testing.assert_array_equal(np.asarray(sim.peek_pauli_batch([2], "Z")), peek_before)
    sim.interactive_do(stim.Circuit("H 1"))  # still usable without clear()
    sim.finish_interactive_run()


def _is_guard_entry(frame: Any) -> bool:
    """True for the `__enter__` that a `with` statement on the native-section guard calls."""
    code = frame.f_code
    if code.co_name != "__enter__":
        return False
    if code.co_filename == magic_engine.__file__:  # a guard class written in magic_engine.py
        return True
    gen = getattr(frame.f_locals.get("self"), "gen", None)  # contextlib's wrapper of _section()
    gen_code = getattr(gen, "gi_code", None)
    return (
        gen_code is not None
        and gen_code.co_filename == magic_engine.__file__
        and gen_code.co_name == "_section"
    )


@pytest.mark.skipif(
    not live_store.engine_numerics_available(), reason="magic engine library is unavailable"
)
def test_magicside_engine_usable_after_an_interrupt_at_the_section_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised as the guard's entry returns leaves the engine usable afterwards."""
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    circuit = stim.Circuit("H 0 1\nI[T] 0\nCX 0 1\nI[T] 1\nM 0 1")
    sim = MagicsideSimulator(circuit, batch_size=4, seed=2)
    items = list(circuit)
    for item in items[:-1]:
        sim.interactive_do(item)
    fired: list[str] = []

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event != "call" or fired or not _is_guard_entry(frame):
            return None

        def local(fr: Any, ev: str, a: Any) -> Any:
            if ev == "return" and not fired:
                fired.append(fr.f_code.co_name)
                raise KeyboardInterrupt("raised by a trace hook as the guard's entry returns")
            return local

        return local

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            sim.interactive_do(items[-1])
    finally:
        sys.settrace(previous)
    assert fired, "the hook did not fire: no native section was entered"
    # The guard's generator stays suspended until the exception is gone; collect any cycle.
    gc.collect()
    sim.clear()  # raised "re-entrant call ..." before the fix
    sim.run()
    assert np.asarray(sim.get_final_measurement_records()).shape == (4, 2)
