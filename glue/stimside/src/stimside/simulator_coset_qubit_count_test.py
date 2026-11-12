"""`CosetsideSimulator` refuses a `compiled_ref` or handler of another qubit count (R3-F1).

The engine, its kernels and the Python helpers are sized by `circuit.num_qubits`. A
`compiled_ref` (packed words and tables of `n_ref` qubits) or an op handler (scratch masks of
`n_handler` rows) built for a different circuit was read past its end, by the kernels (ASan
heap-buffer-overflow / heap-use-after-free in `engine_peek_pauli_batch`,
`engine_batch_unmatched_noisy_reset_z`, `engine_batch_mixed_substep`,
`engine_batch_inject_general_clifford`), with MORE qubits than the circuit as well as fewer.
`__init__` now raises ValueError first, before the engine exists. The tests check that

* each mismatch (reference / handler, fewer / more qubits) raises ValueError naming both counts,
  before any engine is created (so there is nothing to read past);
* what is legitimate still works: one `compiled_ref` shared by simulators of the same circuit
  (with any batch size), a handler compiled for the same circuit, a handler with no
  `num_qubits` (not checked), with and without the kernels (`sync_tableside_rng=True`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_B = 8


def _circuit(n: int) -> stim.Circuit:
    """`n` qubits: H on 0 and a measurement of every qubit."""
    return stim.Circuit("H 0\nM " + " ".join(map(str, range(n))))


def _handler(circuit: stim.Circuit, batch: int = _B) -> Any:
    return LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=batch)


@pytest.fixture
def no_engine(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Fails the test if a kernel library is loaded (so an engine created) during construction."""
    loaded: list[int] = []

    def forbidden() -> Any:
        loaded.append(1)
        raise AssertionError("an engine was created for a simulator that must be rejected")

    monkeypatch.setattr("stimside.simulator_coset.get_coset_kernels_lib", forbidden)
    return loaded


@pytest.mark.parametrize("ref_qubits, circuit_qubits", [(1, 6), (6, 1), (5, 6), (7, 6), (200, 1)])
def test_a_compiled_ref_for_another_qubit_count_is_rejected(
    ref_qubits: int, circuit_qubits: int, no_engine: list[int]
) -> None:
    """Fewer qubits: kernels read past the packed words. More: they read past the engine's N."""
    circuit = _circuit(circuit_qubits)
    ref = CompiledReferenceCircuit(_circuit(ref_qubits))
    with pytest.raises(ValueError) as err:
        CosetsideSimulator(
            circuit, compiled_op_handler=_handler(circuit), batch_size=_B, compiled_ref=ref
        )
    msg = str(err.value)
    assert "compiled_ref" in msg and str(ref_qubits) in msg and str(circuit_qubits) in msg
    assert no_engine == []


@pytest.mark.parametrize("handler_qubits, circuit_qubits", [(2, 6), (6, 2), (300, 2), (1, 6)])
def test_a_handler_for_another_qubit_count_is_rejected(
    handler_qubits: int, circuit_qubits: int, no_engine: list[int]
) -> None:
    """A smaller handler hands the reset kernel (1, B) masks for N rows; the guards on the masks
    reject that at run time, this rejects it at construction, for a bigger one too."""
    handler = _handler(_circuit(handler_qubits))
    with pytest.raises(ValueError) as err:
        CosetsideSimulator(_circuit(circuit_qubits), compiled_op_handler=handler, batch_size=_B)
    msg = str(err.value)
    assert "compiled_op_handler" in msg
    assert str(handler_qubits) in msg and str(circuit_qubits) in msg
    assert no_engine == []


def test_a_compiled_ref_of_no_qubits_is_rejected_like_any_other_count(
    no_engine: list[int],
) -> None:
    """Zero is a count like the others (no special case): an empty reference has no packed words
    at all, so every kernel read of it is out of bounds."""
    circuit = _circuit(6)
    empty = CompiledReferenceCircuit(stim.Circuit())
    assert empty.num_qubits == 0
    with pytest.raises(ValueError, match="compiled_ref was compiled for 0 qubits"):
        CosetsideSimulator(
            circuit, compiled_op_handler=_handler(circuit), batch_size=_B, compiled_ref=empty
        )
    assert no_engine == []


def test_the_explorer_witness_circuits_are_rejected_at_construction(no_engine: list[int]) -> None:
    """The two public-API witnesses (a `compiled_ref` compiled from another circuit, and a
    handler compiled for a smaller one) that ran `run()` into the kernels under ASan."""
    real = stim.Circuit(
        "I[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\nM 0 1 2 3 4 5"
    )
    other = stim.Circuit("I 0\nCZ 0 1 2 3 4 5\nM 0 1 2 3 4 5\nM 6")  # 7 qubits
    with pytest.raises(ValueError, match="compiled_ref"):
        CosetsideSimulator(
            real, compiled_op_handler=_handler(real, 64), batch_size=64,
            compiled_ref=CompiledReferenceCircuit(other),
        )
    ops = (
        "I[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0\n"
    )
    small, big = stim.Circuit(ops), stim.Circuit(ops + "M 1 2 3 4 5\n")
    assert (small.num_qubits, big.num_qubits) == (1, 6)
    with pytest.raises(ValueError, match="compiled_op_handler"):
        CosetsideSimulator(big, compiled_op_handler=_handler(small, 64), batch_size=64)
    assert no_engine == []


def test_a_mismatch_is_rejected_without_the_kernels_too() -> None:
    """The check does not depend on the engine: `sync_tableside_rng=True` runs in Python."""
    circuit = _circuit(6)
    with pytest.raises(ValueError, match="compiled_ref"):
        CosetsideSimulator(
            circuit, compiled_op_handler=_handler(circuit), batch_size=_B,
            sync_tableside_rng=True, compiled_ref=CompiledReferenceCircuit(_circuit(2)),
        )
    with pytest.raises(ValueError, match="compiled_op_handler"):
        CosetsideSimulator(
            circuit, compiled_op_handler=_handler(_circuit(2)), batch_size=_B,
            sync_tableside_rng=True,
        )


def test_the_compiled_ref_is_checked_before_the_handler() -> None:
    circuit = _circuit(6)
    with pytest.raises(ValueError, match="compiled_ref"):
        CosetsideSimulator(
            circuit, compiled_op_handler=_handler(_circuit(2)), batch_size=_B,
            compiled_ref=CompiledReferenceCircuit(_circuit(3)),
        )


@pytest.mark.parametrize("sync", [None, True], ids=["kernels", "python"])
def test_a_compiled_ref_shared_by_simulators_of_the_same_circuit_still_runs(
    sync: bool | None,
) -> None:
    """Reuse for the same circuit (any batch size) equals a simulator that compiles its own."""
    circuit = stim.Circuit(
        "R 0 1 2 3\nH 0 1\nI[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 0\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 2 1 3\nM 0 1 2 3"
    )
    shared = CompiledReferenceCircuit(circuit)
    for batch in (1, 5, 17):
        results = []
        for ref in (None, shared, shared):
            sim = CosetsideSimulator(
                circuit, compiled_op_handler=_handler(circuit, batch), batch_size=batch,
                seed=11, sync_tableside_rng=sync, compiled_ref=ref,
            )
            sim.run()
            results.append(sim.get_final_measurement_records())
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[0], results[2])


def test_a_handler_with_no_or_non_integer_num_qubits_is_not_checked() -> None:
    """`CompiledOpHandler` has no `num_qubits` contract: only an integer one is compared."""
    circuit = _circuit(4)
    plain = TrivialOpHandler().compile_op_handler(circuit=circuit, batch_size=_B)
    assert not hasattr(plain, "num_qubits")
    CosetsideSimulator(circuit, compiled_op_handler=plain, batch_size=_B).run()
    for value in ("six", None, 3.5, [4]):
        plain.num_qubits = value  # type: ignore[attr-defined]
        CosetsideSimulator(circuit, compiled_op_handler=plain, batch_size=_B)


def test_a_numpy_integer_num_qubits_is_compared_like_an_int() -> None:
    circuit = _circuit(4)
    plain = TrivialOpHandler().compile_op_handler(circuit=circuit, batch_size=_B)
    plain.num_qubits = np.int64(4)  # type: ignore[attr-defined]
    CosetsideSimulator(circuit, compiled_op_handler=plain, batch_size=_B)
    plain.num_qubits = np.int64(5)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="compiled_op_handler was compiled for 5 qubits"):
        CosetsideSimulator(circuit, compiled_op_handler=plain, batch_size=_B)


def test_a_bool_num_qubits_is_compared_as_the_integer_it_is() -> None:
    """`bool` is an `int` subclass: `True` is the count 1, not a value to skip."""
    plain = TrivialOpHandler().compile_op_handler(circuit=_circuit(4), batch_size=_B)
    plain.num_qubits = True  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="compiled_op_handler was compiled for True qubits"):
        CosetsideSimulator(_circuit(4), compiled_op_handler=plain, batch_size=_B)


def test_the_compiled_ref_is_checked_whatever_the_handler_is() -> None:
    """The reference check does not depend on the handler (here: none at all)."""
    circuit = _circuit(6)
    with pytest.raises(ValueError, match="compiled_ref was compiled for 2 qubits"):
        CosetsideSimulator(
            circuit, compiled_op_handler=None,  # type: ignore[arg-type]
            batch_size=_B, compiled_ref=CompiledReferenceCircuit(_circuit(2)),
        )


def test_the_qubit_count_check_does_not_hide_a_bad_batch_size() -> None:
    """The existing batch_size ValueError still comes first, and with its own message."""
    circuit = _circuit(4)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        CosetsideSimulator(circuit, compiled_op_handler=_handler(_circuit(2)), batch_size=0)
