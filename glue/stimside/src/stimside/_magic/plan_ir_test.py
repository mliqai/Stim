"""Behavioral tests for CircuitIR-backed Magicside fault extraction and compilation."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import simulator_magic
from stimside._magic import gates as magic_gates
from stimside._magic import plan as magic_plan
from stimside._magic.channels import (
    FeedbackPauliRecord,
    LeftUnitaryEvent,
    RightRotEvent,
)
from stimside._magic.gates import (
    CliffordFaultChannel,
    FaultBranch,
    extract_circuit_faults,
)
from stimside._magic.plan import (
    _compile_circuit_pass,
    compile_coordinates,
    compile_direct,
    compile_selector,
)
from stimside._magic.query import EvaluateQueries, PrepareQueries
from stimside._magic.statevector import extract_data_expansion, reconstruct_statevector
from stimside._magic.trajectory import run_direct, run_selector
from stimside.simulator_magic import MagicsideSimulator
from stimside.util import circuit_ir


def _install_build_ir_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Count ``build_ir`` calls across all Magic consumers."""
    calls = [0]
    orig = circuit_ir.build_ir

    def spy(*args: Any, **kwargs: Any) -> circuit_ir.CircuitIR:
        calls[0] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(circuit_ir, "build_ir", spy)
    monkeypatch.setattr(magic_gates, "build_ir", spy)
    monkeypatch.setattr(magic_plan, "build_ir", spy)
    monkeypatch.setattr(simulator_magic, "build_ir", spy)
    return calls


def _sample_repeat_magic_circuit() -> stim.Circuit:
    return stim.Circuit(
        "QUBIT_COORDS(0, 0) 0\n"
        "R 0 1 2\n"
        "REPEAT 2 {\n"
        "  H 0\n"
        "  REPEAT 2 {\n"
        "    I[CLIFFORD_FAULT: (0.1, S) (0.05, H)] 0\n"
        "    CX 0 1\n"
        "  }\n"
        "  II[CLIFFORD_FAULT: (0.1, CZ)] 0 1\n"
        "  I[T] 0\n"
        "  HERALDED_ERASE(0.04) 1\n"
        "  HERALDED_PAULI_CHANNEL_1(0.01, 0.01, 0.01, 0.01) 2\n"
        "  TICK\n"
        "  SHIFT_COORDS(0, 1)\n"
        "  M 1\n"
        "  DETECTOR(0, 0) rec[-1]\n"
        "}\n"
        "M 0\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )


def test_magic_compile_entry_points_build_at_most_one_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each compile entry point builds at most one IR and zero when ``ir=`` is supplied."""
    circ = _sample_repeat_magic_circuit()
    prebuilt = circuit_ir.build_ir(circ)
    calls = _install_build_ir_spy(monkeypatch)

    ops_a, flts_a = extract_circuit_faults(circ)
    assert calls[0] == 1
    calls[0] = 0
    ops_b, flts_b = extract_circuit_faults(circ, ir=prebuilt)
    assert calls[0] == 0
    assert [str(op) for op in ops_a] == [str(op) for op in ops_b]
    assert flts_a == flts_b

    for fn in (compile_coordinates, compile_direct, compile_selector):
        calls[0] = 0
        p1 = fn(circ)
        assert calls[0] == 1
        calls[0] = 0
        p2 = fn(circ, ir=prebuilt)
        assert calls[0] == 0
        assert p1.num_measurements == p2.num_measurements
        assert p1.fault_channels == p2.fault_channels

    for fmode in ("DIRECT", "SELECTOR"):
        calls[0] = 0
        qp = PrepareQueries(circ, fault_mode=fmode)
        assert calls[0] == 1
        assert qp.num_measurements == circ.num_measurements

    assert "ir" not in inspect.signature(PrepareQueries).parameters


def test_magicside_simulator_builds_single_ir_across_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MagicsideSimulator builds one IR at init and reuses it across STREAM/QUERY/SELECTOR."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "REPEAT 2 {\n"
        "  H 0\n"
        "  I[CLIFFORD_FAULT: (0.1, S)] 0\n"
        "  CX 0 1\n"
        "  I[T] 0\n"
        "  M 1\n"
        "}\n"
        "M 0 2\n"
    )
    calls = _install_build_ir_spy(monkeypatch)
    unroll_calls = [0]
    orig_unroll = simulator_magic.unroll_circuit

    def spy_unroll(*args: Any, **kwargs: Any) -> list[stim.CircuitInstruction]:
        unroll_calls[0] += 1
        return orig_unroll(*args, **kwargs)

    monkeypatch.setattr(simulator_magic, "unroll_circuit", spy_unroll)

    configs: list[dict[str, Any]] = [
        {"backend": "STREAM", "fault_mode": "DIRECT"},
        {"backend": "STREAM", "fault_mode": "SELECTOR"},
        {"backend": "QUERY", "fault_mode": "DIRECT"},
        {"backend": "QUERY", "fault_mode": "SELECTOR"},
        {"backend": "STREAM", "fault_mode": "DIRECT", "fault_realizations": {"faults": {0: 0}}},
    ]
    for cfg in configs:
        calls[0] = 0
        unroll_calls[0] = 0
        sim = MagicsideSimulator(circ, batch_size=4, seed=11, **cfg)
        sim.run()
        assert calls[0] == 1, cfg
        assert unroll_calls[0] == 0, cfg
        assert sim.get_final_measurement_records().shape == (4, circ.num_measurements)


def test_multibranch_selector_z8_ccz_t_statevector_parity() -> None:
    """DIRECT, SELECTOR, and PrepareQueries agree on multi-branch 1Q/2Q Clifford faults."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "H 0 2\n"
        "I[CLIFFORD_FAULT: (0.1, H) (0.1, S_DAG) (0.05, X)] 0 1\n"
        "II[CLIFFORD_FAULT: (0.1, CX) (0.05, SWAP) (0.05, CXSWAP)] 0 1\n"
        "I[T] 0\n"
        "I[T_DAG] 1\n"
        "I[CCZ] 0 1 2\n"
    )
    dplan = compile_direct(circ)
    splan = compile_selector(circ)
    assert len(splan.fault_selector_groups) == 3
    assert [len(g) for g in splan.fault_selector_groups] == [3, 3, 3]

    for trace in ({0: 0, 1: 0, 2: 0}, {0: 2, 1: 1, 2: 1}, {0: 3, 1: 2, 2: 3}):
        rd = run_direct(dplan, fault_trace=trace, rng=np.random.default_rng(1))
        rs = run_selector(splan, fault_trace=trace, rng=np.random.default_rng(1), extract_data=True)
        sv_d = reconstruct_statevector(rd.expansion, rd.final_tableau)
        sv_s = reconstruct_statevector(rs.expansion, rs.final_tableau)
        overlap = abs(np.vdot(sv_d, sv_s))
        assert np.isclose(overlap, 1.0, atol=1e-9), trace

    qd = PrepareQueries(circ, fault_mode="DIRECT")
    qs = PrepareQueries(circ, fault_mode="SELECTOR")
    real = {"faults": {0: 2, 1: 1, 2: 3}}
    res_d = EvaluateQueries(qd, real, queries=("mass",))
    res_s = EvaluateQueries(qs, real, queries=("mass",))
    assert np.isclose(res_d.mass, res_s.mass, atol=1e-9)


def test_gauge_steps_data_frame_and_multiword_selector_slicing() -> None:
    """Gauge steps, data-restricted frames, and multi-word selector tail masks stay exact."""
    circ_66 = stim.Circuit(
        "R 0 65\n"
        "H 0 65\n"
        "I[CLIFFORD_FAULT: (0.15, H) (0.1, S)] 65\n"
        "II[CLIFFORD_FAULT: (0.1, CX) (0.05, SWAP)] 0 65\n"
        "DEPOLARIZE1(0.03) 65\n"
        "E(0.05) X0*Z65\n"
        "M 0 65\n"
        "H 65\n"
        "R 65\n"
    )
    res_plain = _compile_circuit_pass(circ_66, selector_mode=True, gadgetize_pauli_noise=False)
    res_gad = _compile_circuit_pass(circ_66, selector_mode=True, gadgetize_pauli_noise=True)

    assert res_plain.n_data == 66
    assert res_plain.n_sel == 4
    assert res_gad.n_sel == 8
    assert len(res_gad.noise_sel_groups) == 2
    assert len(res_plain.noise_sel_groups) == 0

    for res in (res_plain, res_gad):
        assert res.data_final_tableau is not None
        assert res.data_final_tableau.num_qubits == 66
        assert any(s >= 66 for s, _ in res.gauge_steps)
        assert any(s == -1 for s, _ in res.gauge_steps)
        for _, ax in res.gauge_steps:
            assert ax.a_words.shape == (2,)
            assert ax.b_words.shape == (2,)
            assert int(ax.a_words[1] >> np.uint64(2)) == 0
            assert int(ax.b_words[1] >> np.uint64(2)) == 0
        for ev in res.data_frame_events:
            if isinstance(ev, LeftUnitaryEvent):
                assert all(q < 66 for q in ev.targets)
            if isinstance(ev, RightRotEvent):
                assert ev.rot_axis.a_words.shape == (2,)
                assert int(ev.rot_axis.a_words[1] >> np.uint64(2)) == 0
                assert int(ev.rot_axis.b_words[1] >> np.uint64(2)) == 0

    splan = compile_selector(circ_66, gadgetize_pauli_noise=True)
    raw = run_selector(
        splan,
        fault_trace={0: 1, 1: 2},
        pauli_noise_trace={0: 2, 1: 1},
        rng=np.random.default_rng(7),
        extract_data=False,
    )
    assert raw.selector_bits is not None
    data_tab, data_exp = extract_data_expansion(splan, raw.expansion, raw.selector_bits)
    assert data_tab.num_qubits == 66
    assert data_exp.n_qubits == 66
    assert np.isclose(data_exp.norm_sq(), 1.0, atol=1e-9)


def test_feedback_all_five_gates_orders_and_invalid_rejection() -> None:
    """All five FEEDBACK_2Q_GATES x rec/sweep x both orders compile and match live runs."""
    circ = stim.Circuit(
        "R 0 1 2 3\n"
        "X 0\n"
        "M 0\n"
        "CX rec[-1] 1 0 2 sweep[0] 3\n"
        "CY rec[-1] 2 sweep[1] 1 2 3\n"
        "CZ rec[-1] 1 2 rec[-1] sweep[0] 3 0 sweep[1] 0 1\n"
        "XCZ 1 rec[-1] 2 sweep[0] 0 3\n"
        "YCZ 1 rec[-1] 2 sweep[1] 1 3\n"
        "M 0 1 2 3\n"
    )
    res = _compile_circuit_pass(circ, selector_mode=False)
    fb_recs = [r for r in res.raw_records if isinstance(r, FeedbackPauliRecord)]
    assert len(fb_recs) == 12
    assert sum(1 for r in fb_recs if r.source_kind == "rec") == 6
    assert sum(1 for r in fb_recs if r.source_kind == "sweep") == 6

    dplan = compile_direct(circ)
    splan = compile_selector(circ)
    for sw in ([False, False], [True, False], [False, True], [True, True]):
        rd = run_direct(dplan, sweep_bits=sw, rng=np.random.default_rng(3))
        rs = run_selector(splan, sweep_bits=sw, rng=np.random.default_rng(3))
        sim = MagicsideSimulator(circ, batch_size=1, seed=3, sweep_bits=sw)
        sim.run()
        np.testing.assert_array_equal(rd.measurements, rs.measurements)
        np.testing.assert_array_equal(rd.measurements, sim.get_final_measurement_records()[0])

    for bad_text in (
        "R 0 1\nM 0\nCX 1 rec[-1]\nM 1\n",
        "R 0 1\nM 0\nXCZ sweep[0] 1\nM 1\n",
    ):
        with pytest.raises(ValueError, match="Unsupported classical feedback target pair"):
            _compile_circuit_pass(stim.Circuit(bad_text))


def test_nested_repeat_inline_and_programmed_fault_slot_ordering() -> None:
    """Nested REPEAT inline faults align with unrolled steps and fire after programmed slots."""
    circ = _sample_repeat_magic_circuit()
    flat_ops, inline_faults = extract_circuit_faults(circ)
    ir = circuit_ir.build_ir(circ)
    assert len(flat_ops) == len(ir.instrs)
    assert [ch.step_index for ch in inline_faults] == [3, 5, 7, 16, 18, 20]
    for ch in inline_faults:
        assert flat_ops[ch.step_index].name == "I"
        assert flat_ops[ch.step_index].tag == ""

    ext_same = CliffordFaultChannel(step_index=3, branches=(FaultBranch(0.2, "X", (0,)),))
    ext_end = CliffordFaultChannel(
        step_index=len(flat_ops), branches=(FaultBranch(0.2, "Z", (0,)),)
    )
    res = _compile_circuit_pass(circ, fault_channels=[ext_same, ext_end], selector_mode=False)
    assert res.fault_channels[0].step_index == 3
    assert res.fault_channels[0].branches == ext_same.branches
    assert res.fault_channels[1].step_index == 3
    assert res.fault_channels[1].branches == inline_faults[0].branches
    assert res.fault_channels[-1].step_index == len(flat_ops)
    assert res.fault_channels[-1].branches == ext_end.branches


def test_magicside_simulator_post_init_circuit_mutation_fallback() -> None:
    """Mutating ``sim.circuit`` after init bypasses cached IR/ops across all modes."""
    base = stim.Circuit("R 0\nM 0\n")
    for cfg in (
        {"backend": "STREAM", "fault_mode": "DIRECT"},
        {"backend": "STREAM", "fault_mode": "SELECTOR"},
        {"backend": "QUERY", "fault_mode": "DIRECT"},
        {"backend": "QUERY", "fault_mode": "SELECTOR"},
    ):
        c = base.copy()
        sim = MagicsideSimulator(c, batch_size=3, seed=5, **cfg)
        c.clear()
        c.append("R", [0])
        c.append("X", [0])
        c.append("M", [0])
        sim.run()
        recs = sim.get_final_measurement_records()
        assert recs.shape == (3, 1), cfg
        assert np.all(recs), cfg

    # Verify copy independence works across modes after IR caching.
    for cfg in (
        {"backend": "STREAM", "fault_mode": "DIRECT"},
        {"backend": "STREAM", "fault_mode": "SELECTOR"},
        {"backend": "QUERY", "fault_mode": "DIRECT"},
    ):
        sim = MagicsideSimulator(
            stim.Circuit("R 0\nH 0\nI[T] 0\nM 0\n"), batch_size=4, seed=9, **cfg
        )
        twin = copy.copy(sim)
        sim.run()
        twin.run()
        np.testing.assert_array_equal(
            sim.get_final_measurement_records(), twin.get_final_measurement_records()
        )
