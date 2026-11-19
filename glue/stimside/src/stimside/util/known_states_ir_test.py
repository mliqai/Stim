"""Tests for IR-backed known-state and known-measurement analysis."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.util import circuit_ir, known_states
from stimside.util.circuit import unroll_circuit


def test_known_states_duplicate_mr_targets_backward_order() -> None:
    """Repeated qubit in MR must propagate the pre-reset outcome backwards."""
    circ = stim.Circuit("R 0 1\nH 0\nCX 0 1\nMR 0 0\n")
    states = known_states.compute_known_states_uint8(circ)
    # Before MR 0 0, qubit 0 is entangled so its state must stay unknown (0), not +Z (5).
    assert int(states[3, 0]) == 0
    assert int(states[4, 0]) == 5


def test_known_states_ir_and_unrolled_ops_parity() -> None:
    """Explicit ir= and unrolled_ops= match circuit analysis and share one IR."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "X 2\n"
        "M 2\n"
        "MPP Z0*Z1\n"
        "M 0\n"
        "RX 0\n"
        "H 0\n"
        "MX 0\n"
        "RY 1\n"
        "MRY !1\n"
    )
    expected = known_states.compute_known_states_uint8(circ)
    uops = unroll_circuit(circ)
    ir_uc_false = circuit_ir.build_ir(circ, uc=False)
    ir_uc_true = circuit_ir.build_ir(circ, uc=True)
    assert np.array_equal(
        known_states.compute_known_states_uint8(circ, ir=ir_uc_false), expected
    )
    assert np.array_equal(
        known_states.compute_known_states_uint8(circ, ir=ir_uc_true), expected
    )
    assert np.array_equal(
        known_states.compute_known_states_uint8(circ, unrolled_ops=uops), expected
    )
    with pytest.raises(ValueError, match="Cannot pass both unrolled_ops and ir"):
        known_states.compute_known_states_uint8(circ, unrolled_ops=uops, ir=ir_uc_false)


def test_known_states_single_ir_and_no_lazy_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_known_states_uint8 builds at most one IR and never forces lazy flat alignment."""
    circ = stim.Circuit("R 0\nREPEAT 3 {\n  H 0\n}\nM 0\n")
    build_calls = 0
    orig_build = circuit_ir.build_ir

    def spy_build(*args: Any, **kwargs: Any) -> circuit_ir.CircuitIR:
        nonlocal build_calls
        build_calls += 1
        return orig_build(*args, **kwargs)

    def boom_align(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("known_states must not trigger lazy flat alignment")

    monkeypatch.setattr(known_states, "build_ir", spy_build)
    monkeypatch.setattr(circuit_ir, "_align_flat_ops", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_where", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_flat_ops", boom_align)

    want = known_states.compute_known_states_uint8(circ)
    assert build_calls == 1

    ir = orig_build(circ)
    build_calls = 0
    got_ir = known_states.compute_known_states_uint8(circ, ir=ir)
    assert build_calls == 0
    assert np.array_equal(got_ir, want)

    got_uops = known_states.compute_known_states_uint8(
        circ, unrolled_ops=unroll_circuit(circ)
    )
    assert np.array_equal(got_uops, want)
    assert known_states.compute_known_measurements(circ) == [None]
    assert len(known_states.compute_known_states(circ)) == 6


def test_known_states_empty_unrolled_ops_and_trailing_qubits() -> None:
    """Empty unrolled_ops preserves circuit.num_qubits column count."""
    circ = stim.Circuit("R 0 1 2\nM 0\n")
    st = known_states.compute_known_states_uint8(circ, unrolled_ops=[])
    assert st.shape == (1, 3)
    assert st.tolist() == [[5, 5, 5]]


def test_known_states_backprop_conflict_and_duplicate_bwd_unitary() -> None:
    """Backward pass handles duplicate unitary targets and detects impossible conflicts."""
    circ = stim.Circuit("R 0 1\nCX 0 1\nS 0 0\nZ 0\nM 0\n")
    st = known_states.compute_known_states_uint8(circ)
    assert st.shape == (6, 2)

    non_eig = stim.Circuit("R 0 1\nCX 0 1\nH 0\nS 0 0\nH 0\nM 0\n")
    non_eig_st = known_states.compute_known_states_uint8(non_eig)
    assert np.array_equal(
        non_eig_st,
        np.array(
            [[5, 5], [5, 5], [5, 0], [1, 0], [2, 0], [6, 0], [6, 0]],
            dtype=np.uint8,
        ),
    )

    cxyz = stim.Circuit("R 0 1\nCX 0 1\nC_XYZ 0 0\nH_YZ 0\nM 0\n")
    cxyz_st = known_states.compute_known_states_uint8(cxyz)
    assert np.array_equal(
        cxyz_st,
        np.array(
            [[5, 5], [5, 5], [5, 0], [3, 0], [5, 0], [5, 0]],
            dtype=np.uint8,
        ),
    )

    bad_ops = [
        stim.Circuit("R 0")[0],
        stim.Circuit("H 0")[0],
        stim.Circuit("M 0")[0],
    ]
    with pytest.raises(ValueError, match="Backprop known states disagreed"):
        known_states.compute_known_states_uint8(stim.Circuit("R 0\nM 0\n"), unrolled_ops=bad_ops)


def test_known_states_noisy_measure_reset_and_arity_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noisy MR* strips args in reference measurements and arity cache respects monkeypatching."""
    assert known_states.compute_known_measurements(stim.Circuit("R 0\nMR(0.1) 0\n")) == [False]
    real_gate_data = stim.gate_data
    known_states._IS_1Q_GATE_CACHE.clear()
    h_circ = stim.Circuit("R 0\nH 0\n")
    assert known_states.compute_known_states_uint8(h_circ).tolist() == [[5], [5], [1]]
    assert known_states._IS_1Q_GATE_CACHE["H"] is True

    assert known_states._is_single_qubit_gate("H") is True
    assert known_states._is_single_qubit_gate("H") is True
    assert known_states._is_single_qubit_gate("CX") is False
    assert known_states._is_single_qubit_gate("CX") is False
    assert known_states._IS_1Q_GATE_CACHE["H"] is True
    assert known_states._IS_1Q_GATE_CACHE["CX"] is False

    def fake_gate_data(name: str) -> Any:
        gd = real_gate_data(name)
        return SimpleNamespace(
            name=gd.name,
            produces_measurements=gd.produces_measurements,
            is_reset=gd.is_reset,
            is_unitary=gd.is_unitary,
            is_noisy_gate=gd.is_noisy_gate,
            is_single_qubit_gate=False if gd.name == "H" else gd.is_single_qubit_gate,
            is_two_qubit_gate=gd.is_two_qubit_gate,
        )

    with monkeypatch.context() as ctx:
        ctx.setattr(stim, "gate_data", fake_gate_data)
        assert known_states.compute_known_states_uint8(h_circ).tolist() == [[5], [5], [0]]
    assert known_states.compute_known_states_uint8(h_circ).tolist() == [[5], [5], [1]]

    monkeypatch.setattr(stim, "gate_data", lambda _n: SimpleNamespace(is_single_qubit_gate=False))
    assert known_states._is_single_qubit_gate("H") is False
    monkeypatch.setattr(stim, "gate_data", lambda _n: SimpleNamespace(is_single_qubit_gate=True))
    assert known_states._is_single_qubit_gate("CX") is True


def test_without_noise_preserving_heralds_and_states() -> None:
    """Heralds become MPAD preserving inversion while pure noise and meas args are stripped."""
    circ = stim.Circuit(
        "R 0 1\n"
        "X_ERROR(0.25) 0\n"
        "HERALDED_ERASE(0.15) 0 !1\n"
        "REPEAT 2 {\n"
        "  X_ERROR(0.25) 1\n"
        "  HERALDED_PAULI_CHANNEL_1(0.1, 0.02, 0.02, 0.02) !0 1\n"
        "}\n"
        "MPAD(0.1) 0\n"
        "M(0.2) 0 1\n"
    )
    stripped = known_states._without_noise_preserving_heralds(circ)
    expected_stripped = stim.Circuit(
        "R 0 1\n"
        "MPAD 0 1\n"
        "REPEAT 2 {\n"
        "  MPAD 1 0\n"
        "}\n"
        "MPAD 0\n"
        "M 0 1\n"
    )
    assert stripped == expected_stripped
    assert known_states.compute_known_measurements(circ) == [
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    states = known_states.compute_known_states_uint8(circ)
    assert states.shape == (10, 2)
    assert states[-1].tolist() == [5, 5]


def test_convert_paulis_to_arrays_ndarray_parity() -> None:
    """convert_paulis_to_arrays handles uint8 arrays identically to PauliString lists."""
    paulis = [
        stim.PauliString("_"),
        stim.PauliString("+X"),
        stim.PauliString("-X"),
        stim.PauliString("+Y"),
        stim.PauliString("-Y"),
        stim.PauliString("+Z"),
        stim.PauliString("-Z"),
    ]
    codes = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
    expected = known_states.convert_paulis_to_arrays(paulis)
    actual = known_states.convert_paulis_to_arrays(codes)
    assert len(actual) == 6
    for got_arr, want_arr in zip(actual, expected):
        assert got_arr.dtype == np.bool_
        assert got_arr.shape == (7,)
        assert np.array_equal(got_arr, want_arr)

    empty = known_states.convert_paulis_to_arrays(np.empty(0, dtype=np.uint8))
    assert all(arr.dtype == np.bool_ and arr.shape == (0,) for arr in empty)

    states = known_states.compute_known_states_uint8(stim.Circuit("R 0 1\nX 1\nH 0\nM 0 1\n"))
    row_arrays = known_states.convert_paulis_to_arrays(states[2])
    assert row_arrays[0].tolist() == [False, False]
    assert row_arrays[4].tolist() == [True, False]
    assert row_arrays[5].tolist() == [False, True]


def test_known_states_unrolled_ops_measurement_shift() -> None:
    """Backward pass aligns measurement offsets when unrolled_ops drops earlier ops."""
    c1 = stim.Circuit("R 0 1\nX 0\nM 0\nCX 0 1\nCX 0 1\nM 1\n")
    ops1 = [
        stim.CircuitInstruction("CX", [0, 1]),
        stim.CircuitInstruction("CX", [0, 1]),
        stim.CircuitInstruction("M", [1]),
    ]
    got1 = known_states.compute_known_states_uint8(c1, unrolled_ops=ops1)
    assert got1.tolist() == [[5, 5], [0, 0], [0, 5], [0, 6]]
    assert known_states.compute_known_states_uint8(c1, unrolled_ops=list(c1)[3:]).tolist() == [
        [5, 5],
        [0, 5],
        [0, 6],
    ]

    c2 = stim.Circuit("R 0 1\nX 0\nM 0\nH 1\nM 1\n")
    got2 = known_states.compute_known_states_uint8(c2, unrolled_ops=list(c2)[4:])
    assert got2.tolist() == [[5, 5], [5, 6]]

    c3 = stim.Circuit("R 0 1\nM 0\nX 1\nM 1\n")
    with pytest.raises(ValueError, match="Backprop known states disagreed"):
        known_states.compute_known_states_uint8(c3, unrolled_ops=list(c3)[1:2])


def test_compute_known_states_returns_independent_pauli_copies() -> None:
    """Mutating a returned PauliString must not alias other rows or future calls."""
    res = known_states.compute_known_states(stim.Circuit("R 0\n"))
    assert res[0][0] is not res[1][0]
    try:
        res[0][0][0] = "X"
        assert res[1][0] == stim.PauliString("+Z")
        assert known_states.compute_known_states(stim.Circuit("R 0\n")) == [
            [stim.PauliString("+Z")],
            [stim.PauliString("+Z")],
        ]
    finally:
        res[0][0][0] = "Z"


def test_known_states_multi_shift_and_empty_unrolled_exclusion() -> None:
    """Reject empty unrolled_ops with ir= and keep alignment for shift >= 2."""
    c_ir = stim.Circuit("R 0\nH 0\n")
    with pytest.raises(ValueError, match="Cannot pass both unrolled_ops and ir"):
        known_states.compute_known_states_uint8(
            c_ir, unrolled_ops=[], ir=circuit_ir.build_ir(c_ir)
        )
    c_sh2 = stim.Circuit("R 0 1\nX 0\nM 0 0\nCX 0 1\nCX 0 1\nM 1\n")
    ops_sh2 = [stim.CircuitInstruction("CX", [0, 1])] * 2 + list(c_sh2)[4:]
    got_sh2 = known_states.compute_known_states_uint8(c_sh2, unrolled_ops=ops_sh2)
    assert np.array_equal(got_sh2, [[5, 5], [0, 0], [0, 5], [0, 6]])


def test_known_states_multi_target_dup_unitary_and_reset_clearing() -> None:
    """Handle >=3-target non-adjacent 1q duplicates and multi-target reset backprop."""
    got_fwd = known_states.compute_known_states_uint8(stim.Circuit("R 0 1\nC_XYZ 0 1 0\n"))
    assert np.array_equal(got_fwd, [[5, 5], [5, 5], [3, 1]])
    c_bwd = stim.Circuit("R 0 1\nH 0\nCX 0 1\nCX 0 1\nC_XYZ 0 1 0\nM 0\n")
    got_bwd = known_states.compute_known_states_uint8(c_bwd)
    assert np.array_equal(got_bwd, [[5, 5], [5, 5], [1, 5], [1, 0], [5, 0], [5, 0]])
    got_rst = known_states.compute_known_states_uint8(stim.Circuit("R 0 1\nX 0 1\nR 0 1\n"))
    assert np.array_equal(got_rst, [[5, 5], [5, 5], [6, 6], [5, 5]])
