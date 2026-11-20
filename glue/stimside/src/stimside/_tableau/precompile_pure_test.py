"""Behavioral purity and contract tests for ``precompile_tableside_circuit``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import LeakageUint8Tableau, TablesideSimulator
from stimside._tableau.runner import (
    PrecompiledTablesideCircuit,
    PrecompiledTablesideData,
    get_precompiled_circuit,
    precompile_tableside_circuit,
)


def _parsed_ops_for(circuit: stim.Circuit) -> dict[Any, Any]:
    return dict(
        LeakageUint8Tableau().compile_op_handler(circuit=circuit, batch_size=1).ops_to_params
    )


def _rich_circuit() -> stim.Circuit:
    return stim.Circuit(
        """
        QUBIT_COORDS(1, 2) 0
        QUBIT_COORDS(3, 4) 1
        R 0 1 2
        REPEAT 2 {
            H 0
            CX 0 1
            I[LEAKAGE_TRANSITION_1: (0.2, U-->2)] 1 0
            H[CONDITIONED_ON_OTHER: U : 0] 1
            PAULI_CHANNEL_1[CONDITIONED_ON_SELF: U](0.01, 0.02, 0.03) 0
            II[LEAKAGE_TRANSITION_2: (0.05, U_U-->2_U)] 0 1
            SWAP[LEAKAGE_SWAP] 1 2
            I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1
            M[LEAKAGE_PROJECTION_Z: (1.0, 2)] !0 1
            SHIFT_COORDS(0, 1)
            TICK
            DETECTOR(0, 0) rec[-1]
        }
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def _assert_data_byte_equal(
    a: PrecompiledTablesideData,
    b: PrecompiledTablesideData,
) -> None:
    assert a is not b
    assert a.num_qubits == b.num_qubits
    assert a.num_words == b.num_words
    assert a.unconditional_condition_on_U == b.unconditional_condition_on_U
    assert a.parsed_ops == b.parsed_ops
    assert a.final_qubit_coords == b.final_qubit_coords
    assert a.final_qubit_tags == b.final_qubit_tags
    assert a.final_coords_shifts == b.final_coords_shifts
    assert a.ref_insts == b.ref_insts
    assert a.orig_ops == b.orig_ops
    assert a.full_bare_insts == b.full_bare_insts
    assert a.quantum_bare_insts == b.quantum_bare_insts
    assert a.step_to_full_idx == b.step_to_full_idx
    assert a.step_to_quant_idx == b.step_to_quant_idx
    assert a.step_has_bare_op == b.step_has_bare_op
    assert a.requires_tableau == b.requires_tableau
    assert a.num_steps == b.num_steps
    assert a.collapses_z == b.collapses_z
    assert a.unrolled_num_ops == b.unrolled_num_ops
    assert a.step_to_unrolled_idx == b.step_to_unrolled_idx
    assert len(a.steps_py) == len(b.steps_py)
    for sa, sb in zip(a.steps_py, b.steps_py):
        assert sa[:4] == sb[:4]
        assert np.array_equal(sa[4], sb[4]) and sa[4].dtype == sb[4].dtype
        assert sa[5] == sb[5]
    assert np.array_equal(a.raw_t_np, b.raw_t_np) and a.raw_t_np.dtype == b.raw_t_np.dtype
    assert np.array_equal(a.qubit_t_np, b.qubit_t_np) and a.qubit_t_np.dtype == b.qubit_t_np.dtype
    assert np.array_equal(a.masks_np, b.masks_np) and a.masks_np.dtype == b.masks_np.dtype
    assert (a.c_steps is None) == (b.c_steps is None)
    if a.c_steps is not None:
        assert bytes(a.c_steps) == bytes(b.c_steps)
    assert (a.c_branches is None) == (b.c_branches is None)
    if a.c_branches is not None:
        assert bytes(a.c_branches) == bytes(b.c_branches)


@pytest.mark.parametrize("uc", [True, False])
def test_purity_determinism_and_zero_input_mutation(uc: bool) -> None:
    circuit = _rich_circuit()
    circuit_before = circuit.copy()
    text_before = str(circuit)
    supplied = _parsed_ops_for(circuit)
    supplied_before = dict(supplied)

    first = precompile_tableside_circuit(circuit, uc, supplied)
    # Interleave an unrelated precompile call to confirm no hidden global state leaks.
    _ = precompile_tableside_circuit(stim.Circuit("H 0\nM 0\n"), not uc, None)
    second = precompile_tableside_circuit(circuit, uc, supplied)

    _assert_data_byte_equal(first, second)
    assert circuit == circuit_before
    assert str(circuit) == text_before
    assert supplied == supplied_before

    wrapper = PrecompiledTablesideCircuit(circuit, uc, supplied)
    assert circuit == circuit_before
    assert supplied == supplied_before
    assert wrapper.num_steps == first.num_steps


def test_precompiled_tableside_data_is_frozen() -> None:
    data = precompile_tableside_circuit(stim.Circuit("H 0\nM 0\n"))
    assert isinstance(data, PrecompiledTablesideData)
    for attr, val in (
        ("num_qubits", 99),
        ("num_steps", 99),
        ("requires_tableau", True),
        ("collapses_z", True),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(data, attr, val)


@pytest.mark.parametrize("uc", [True, False])
@pytest.mark.parametrize("use_supplied", [False, True])
def test_wrapper_matches_pure_function_across_options(
    uc: bool,
    use_supplied: bool,
) -> None:
    circuit = _rich_circuit()
    supplied = _parsed_ops_for(circuit) if use_supplied else None
    data = precompile_tableside_circuit(circuit, uc, supplied)
    pre = PrecompiledTablesideCircuit(circuit, uc, supplied)

    public_fields = (
        "num_qubits",
        "num_words",
        "unconditional_condition_on_U",
        "parsed_ops",
        "final_qubit_coords",
        "final_qubit_tags",
        "final_coords_shifts",
        "ref_insts",
        "orig_ops",
        "full_bare_insts",
        "quantum_bare_insts",
        "step_to_full_idx",
        "step_to_quant_idx",
        "step_has_bare_op",
        "requires_tableau",
        "num_steps",
        "collapses_z",
        "unrolled_num_ops",
        "step_to_unrolled_idx",
    )
    for field in public_fields:
        assert getattr(pre, field) == getattr(data, field), f"mismatch on {field}"
    assert len(pre.steps_py) == len(data.steps_py)
    for sp, sd in zip(pre.steps_py, data.steps_py):
        assert sp[:4] == sd[:4]
        assert np.array_equal(sp[4], sd[4])
        assert sp[5] == sd[5]


def test_parsed_ops_supplied_empty_custom_and_nonstandard_fallback() -> None:
    circuit = stim.Circuit(
        """
        R 0
        I[LEAKAGE_TRANSITION_1: (0.1, U-->2)] 0
        M 0
        """
    )
    lt1_op = list(circuit)[1]
    parsed = _parsed_ops_for(circuit)

    # 1. Supplied standard parsed_ops matches None byte-for-byte.
    d_none = precompile_tableside_circuit(circuit, True, None)
    d_sup = precompile_tableside_circuit(circuit, True, parsed)
    _assert_data_byte_equal(d_none, d_sup)

    # 2. Empty dict {} takes the supplied fast path and treats tagged ops as unclaimed.
    d_empty = precompile_tableside_circuit(circuit, True, {})
    assert d_empty.parsed_ops == {}
    assert d_empty.c_steps[1].kind == 1
    assert d_empty.c_branches is None

    # 3. Custom standard param overrides tag parsing and appears in packed C buffers.
    custom_circuit = stim.Circuit("I[LEAKAGE_TRANSITION_1: (0.42, U-->2)] 0\n")
    custom_param = next(iter(_parsed_ops_for(custom_circuit).values()))
    d_custom = precompile_tableside_circuit(circuit, True, {lt1_op: custom_param})
    assert d_custom.parsed_ops == {lt1_op: custom_param}
    assert d_custom.c_steps[1].kind == 3
    assert d_custom.c_steps[1].p_total_u == pytest.approx(0.42)
    assert d_custom.c_branches is not None
    assert d_custom.c_branches[0].raw_prob == pytest.approx(0.42)

    # 4. Non-standard value falls back to parsing the tag from the circuit.
    bad_map: dict[Any, Any] = {lt1_op: object()}
    d_fallback = precompile_tableside_circuit(circuit, True, bad_map)
    _assert_data_byte_equal(d_none, d_fallback)


def test_has_repeat_fast_path_and_no_cross_boundary_fusion() -> None:
    rep_circuit = stim.Circuit(
        """
        REPEAT 3 {
            H 0
        }
        H 0
        """
    )
    flat_circuit = rep_circuit.flattened()
    assert len(list(flat_circuit)) == 1

    d_rep = precompile_tableside_circuit(rep_circuit)
    d_flat = precompile_tableside_circuit(flat_circuit)
    assert d_rep.unrolled_num_ops == 4
    assert d_rep.num_steps == 4
    assert d_rep.step_to_unrolled_idx == [0, 1, 2, 3]
    assert d_flat.unrolled_num_ops == 1
    assert d_flat.num_steps == 1
    assert d_flat.step_to_unrolled_idx == [0]

    nested = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        REPEAT 2 {
            REPEAT 2 {
                H 0
                TICK
            }
            SHIFT_COORDS(1, 0)
        }
        QUBIT_COORDS(0, 1) 1
        M 0 1
        DETECTOR(0, 0) rec[-1]
        """
    )
    d_nested = precompile_tableside_circuit(nested)
    assert d_nested.final_qubit_coords == {0: [0.0, 0.0], 1: [2.0, 1.0]}
    assert d_nested.final_coords_shifts == [2.0, 0.0]
    assert d_nested.num_steps == 5
    assert d_nested.step_to_unrolled_idx == [1, 3, 6, 8, 12]


def test_template_cache_and_lt1_meas_copy_on_write_isolation() -> None:
    circuit = stim.Circuit(
        """
        R 0 1
        I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1
        M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 1 0
        I[LEAKAGE_TRANSITION_1: (0.1, 1-->2)] 0 1
        H 0
        M 0 1
        """
    )
    data = precompile_tableside_circuit(circuit)
    assert data.num_steps == 6
    # Step 1 is fused with step 2 -> kind=7; identical op at step 3 stays unfused -> kind=6.
    assert data.c_steps[1].kind == 7
    assert data.steps_py[1][0] == 7
    assert data.c_steps[2].kind == 5
    assert data.c_steps[3].kind == 6
    assert data.steps_py[3][0] == 6
    assert data.c_steps[1].unrolled_idx == 1
    assert data.c_steps[3].unrolled_idx == 3
    assert data.c_steps[1].unrolled_idx != data.c_steps[3].unrolled_idx


@pytest.mark.parametrize(
    "bad_circuit_text",
    [
        "I[LEAKAGE_TRANSITION_Z: (0.1, U-->2)] 0\n",
        "MPAD[LEAKAGE_MEASUREMENT: (0.5, U) : 0] 0\n",
        "M[LEAKAGE_PROJECTION_Z: (1.0, 2)](0.05) 0\n",
    ],
)
def test_raising_circuits_preserve_exception_and_inputs(bad_circuit_text: str) -> None:
    circuit = stim.Circuit(bad_circuit_text)
    snapshot = circuit.copy()
    with pytest.raises(ValueError) as exc_pure:
        precompile_tableside_circuit(circuit)
    with pytest.raises(ValueError) as exc_wrap:
        PrecompiledTablesideCircuit(circuit)
    assert str(exc_pure.value) == str(exc_wrap.value)
    assert circuit == snapshot


def test_empty_annotation_only_and_multiword_masks() -> None:
    ann_only = stim.Circuit(
        """
        QUBIT_COORDS(1, 2) 0
        SHIFT_COORDS(0, 3)
        TICK
        """
    )
    d_ann = precompile_tableside_circuit(ann_only)
    assert d_ann.num_steps == 0
    assert d_ann.c_steps is None
    assert d_ann.c_branches is None
    assert len(d_ann.raw_t_np) == 0
    assert len(d_ann.qubit_t_np) == 0
    assert len(d_ann.masks_np) == d_ann.num_words
    assert d_ann.step_to_full_idx == [3]
    assert d_ann.step_to_quant_idx == [0]

    wide = stim.Circuit("H 0 70\nM 0 70\n")
    d_wide = precompile_tableside_circuit(wide)
    assert d_wide.num_qubits == 71
    assert d_wide.num_words == 2
    m_off = d_wide.c_steps[0].mask_offset
    assert int(d_wide.masks_np[m_off]) == (1 << 0)
    assert int(d_wide.masks_np[m_off + 1]) == (1 << (70 - 64))


def test_integration_coh_equality_cache_eviction_and_simulator_run() -> None:
    circuit = _rich_circuit()
    coh = LeakageUint8Tableau().compile_op_handler(circuit=circuit, batch_size=1)
    pre1 = get_precompiled_circuit(circuit, True, coh.ops_to_params)
    pre2 = get_precompiled_circuit(circuit, True, coh.ops_to_params)
    assert pre1 is pre2
    assert coh.ops_to_params == pre1.parsed_ops

    # Fill the 4-entry precompile cache with distinct circuits so the first entry is evicted.
    others = [stim.Circuit(f"H {i}\nM {i}\n") for i in range(5)]
    for other in others:
        _ = get_precompiled_circuit(other, True, None)
    pre3 = get_precompiled_circuit(circuit, True, coh.ops_to_params)
    assert pre3 is not pre1

    # Both C++ and sync-RNG TablesideSimulator paths execute cleanly and reproducibly.
    for sync_rng in (False, True):
        s1 = TablesideSimulator(
            circuit,
            compiled_op_handler=LeakageUint8Tableau().compile_op_handler(
                circuit=circuit, batch_size=1
            ),
            seed=12345,
            sync_tableside_rng=sync_rng,
        )
        s2 = TablesideSimulator(
            circuit,
            compiled_op_handler=LeakageUint8Tableau().compile_op_handler(
                circuit=circuit, batch_size=1
            ),
            seed=12345,
            sync_tableside_rng=sync_rng,
        )
        s1.run()
        s2.run()
        m1 = s1.get_final_measurement_records()
        m2 = s2.get_final_measurement_records()
        assert np.array_equal(m1, m2)
        assert m1.shape == (1, circuit.num_measurements)


@pytest.mark.parametrize(
    ("inverted", "plain"),
    [
        ("SPP !X0 !X1", "SPP X0 X1"),
        ("SPP_DAG !Y0 !Z1 !X2", "SPP_DAG Y0 Z1 X2"),
        ("E(0.25) !X0 !Y1", "E(0.25) X0 Y1"),
        ("ELSE_CORRELATED_ERROR(0.25) !X0 !Y1", "ELSE_CORRELATED_ERROR(0.25) X0 Y1"),
    ],
)
def test_inverted_pauli_targets_precompile_like_plain_pauli_targets(
    inverted: str, plain: str
) -> None:
    """An inverted Pauli target is still a Pauli target, never one half of a plain qubit pair."""
    kinds_inverted = [s[0] for s in PrecompiledTablesideCircuit(stim.Circuit(inverted)).steps_py]
    kinds_plain = [s[0] for s in PrecompiledTablesideCircuit(stim.Circuit(plain)).steps_py]
    assert kinds_inverted == kinds_plain == [8]


def test_pair_condition_on_an_odd_target_count_takes_the_general_path() -> None:
    """The pair filter reads the targets two at a time, so an odd count never gets it."""
    circuit = stim.Circuit("CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\nH[x] 0 1 2")
    params = _parsed_ops_for(circuit)
    params[circuit[1]] = params[circuit[0]]  # hand-built: the tag parser never pairs up an H
    kinds = [s[0] for s in PrecompiledTablesideCircuit(circuit, True, params).steps_py]
    assert kinds == [2, 8]


def _final_records(text: str, *, sync_rng: bool, seed: int) -> list[int]:
    circuit = stim.Circuit(text)
    sim = TablesideSimulator(
        circuit,
        compiled_op_handler=LeakageUint8Tableau().compile_op_handler(
            circuit=circuit, batch_size=1
        ),
        seed=seed,
        sync_tableside_rng=sync_rng,
    )
    sim.run()
    return [int(bit) for bit in sim.get_final_measurement_records()[0]]


@pytest.mark.parametrize("sync_rng", [False, True])
def test_spp_with_inverted_pauli_targets_skips_only_the_leaked_qubit(sync_rng: bool) -> None:
    """q0 is leaked, so `SPP !X0 !X1` leaves `SQRT_X_DAG 1`: MY reads 0, and 1 without the `!`."""
    leak_q0 = "R 0 1 2\nI[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0\n"
    cases = [
        ("SPP !X0 !X1\nMY 1", [0]),
        ("SPP X0 X1\nMY 1", [1]),
        ("SPP !X0 !X1 !X2\nMY 1 2", [0, 0]),
    ]
    for body, want in cases:
        for seed in range(8):
            assert _final_records(leak_q0 + body, sync_rng=sync_rng, seed=seed) == want
