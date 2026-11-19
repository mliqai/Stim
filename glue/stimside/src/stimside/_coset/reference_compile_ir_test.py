"""Tests for CircuitIR-backed Cosetside reference compilation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside._coset import reference_compile
from stimside._coset.reference_compile import (
    CompiledReferenceCircuit,
    MeasStepMeta,
    NoopStepMeta,
    UnitaryStepMeta,
)
from stimside.util import circuit_ir
from stimside.util.circuit import unroll_circuit
from stimside.util.reference_chp import DetMeasMeta


def test_compile_reference_single_ir_and_no_lazy_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build at most one CircuitIR per circuit and never trigger lazy alignment."""
    circ = stim.Circuit(
        "QUBIT_COORDS(0, 0) 0\n"
        "R 0 1\n"
        "REPEAT 3 {\n"
        "  H 0\n"
        "  CX 0 1\n"
        "  SHIFT_COORDS(1, 0)\n"
        "}\n"
        "REPEAT 2 {\n"
        "  REPEAT 2 {\n"
        "    X 0\n"
        "  }\n"
        "}\n"
        "REPEAT 4 {\n"
        "}\n"
        "M 0 1\n"
        "DETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\n"
        "TICK\n"
    )
    build_calls = 0
    orig_build = circuit_ir.build_ir

    def spy_build(*args: Any, **kwargs: Any) -> circuit_ir.CircuitIR:
        nonlocal build_calls
        build_calls += 1
        return orig_build(*args, **kwargs)

    def boom_align(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("reference compilation must not trigger lazy alignment")

    def boom_classify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("CompiledReferenceCircuit.__init__ must reuse ir.instrs categories")

    monkeypatch.setattr(reference_compile, "build_ir", spy_build)
    monkeypatch.setattr(reference_compile, "classify_category", boom_classify)
    monkeypatch.setattr(circuit_ir, "_align_flat_ops", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_where", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_flat_ops", boom_align)

    want = CompiledReferenceCircuit(circ)
    assert build_calls == 1
    assert len(want.steps) == 15  # R + 3*(H,CX) + 4*X + M + DETECTOR + OBS + TICK
    assert sum(1 for st in want.steps if st.op.name == "H") == 3
    assert sum(1 for st in want.steps if st.op.name == "X") == 4

    ir_uc_false = orig_build(circ, uc=False)
    ir_uc_true = orig_build(circ, uc=True)
    ir_oplist = orig_build(unroll_circuit(circ))
    monkeypatch.setattr(circuit_ir, "unroll_circuit", boom_align)

    want_snap = want.ref_chp.snapshot_packed()
    for supplied_ir in (ir_uc_false, ir_uc_true, ir_oplist):
        build_calls = 0
        got = CompiledReferenceCircuit(circ, ir=supplied_ir)
        assert build_calls == 0
        assert [str(st.op) for st in got.steps] == [str(st.op) for st in want.steps]
        assert [st.kind for st in got.steps] == [st.kind for st in want.steps]
        got_snap = got.ref_chp.snapshot_packed()
        for idx in range(len(want_snap)):
            assert np.array_equal(want_snap[idx], got_snap[idx])


def test_compile_reference_ir_validation_empty_and_coord_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject mismatched ir shapes before tableau setup, allow empty circuit, skip coords."""
    empty_ref = CompiledReferenceCircuit(stim.Circuit())
    assert empty_ref.num_qubits == 0
    assert empty_ref.n_words == 1
    assert empty_ref.steps == []
    assert empty_ref._initial_packed_words[0].shape == (0, 1)

    circ = stim.Circuit("R 0 1\nM 0\n")
    bad_nq_small = circuit_ir.build_ir(stim.Circuit("R 0\nM 0\n"))
    bad_nq_big = circuit_ir.build_ir(stim.Circuit("R 0 1 2\nM 0\n"))
    bad_nm_small = circuit_ir.build_ir(stim.Circuit("R 0 1\n"))
    bad_nm_big = circuit_ir.build_ir(stim.Circuit("R 0 1\nM 0 1\n"))

    called_chp = False

    def spy_chp(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called_chp
        called_chp = True
        raise AssertionError("ReferenceCHPTableau should not run on invalid ir")

    with monkeypatch.context() as ctx:
        ctx.setattr(reference_compile, "ReferenceCHPTableau", spy_chp)
        for bad_ir in (bad_nq_small, bad_nq_big, bad_nm_small, bad_nm_big):
            with pytest.raises(ValueError, match="Supplied CircuitIR does not match"):
                CompiledReferenceCircuit(circ, ir=bad_ir)
            assert not called_chp

    coord_only = stim.Circuit(
        "QUBIT_COORDS(1, 2) 0\n"
        "REPEAT 2 {\n"
        "  SHIFT_COORDS(0, 1)\n"
        "  QUBIT_COORDS(2, 3) 1\n"
        "}\n"
    )
    ref_coord = CompiledReferenceCircuit(coord_only)
    assert ref_coord.steps == []
    assert ref_coord.num_qubits == 2

    q_op = stim.CircuitInstruction("QUBIT_COORDS", [0], [1.0, 2.0])
    s_op = stim.CircuitInstruction("SHIFT_COORDS", [], [1.0])
    for coord_op in (q_op, s_op):
        st, _, _ = ref_coord.compile_instruction(coord_op)
        assert isinstance(st, NoopStepMeta)
        assert st.op == coord_op


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_compile_reference_spp_and_spp_dag_contract(gate_name: str) -> None:
    """SPP and SPP_DAG raise exact NotImplementedError across all entry points."""
    expected_msg = (
        f"CosetsideSimulator does not support {gate_name} (Pauli product rotations). "
        "Rewrite them with single- and two-qubit Clifford gates, or use "
        "TablesideSimulator or FlipsideSimulator."
    )
    c_direct = stim.Circuit(f"R 0 1\n{gate_name} X0*Z1\nM 0\n")
    with pytest.raises(NotImplementedError) as exc_direct:
        CompiledReferenceCircuit(c_direct)
    assert str(exc_direct.value) == expected_msg

    c_repeat = stim.Circuit(f"R 0 1\nREPEAT 2 {{\n  H 0\n  {gate_name} X0\n}}\n")
    with pytest.raises(NotImplementedError) as exc_repeat:
        CompiledReferenceCircuit(c_repeat)
    assert str(exc_repeat.value) == expected_msg

    base = CompiledReferenceCircuit(stim.Circuit("R 0 1\nH 0\n"))
    steps_before = list(base.steps)
    snap_before = base._last_packed_snapshot
    chp_before = [w.copy() for w in base.ref_chp.snapshot_packed()]
    pauli_op = stim.CircuitInstruction(
        gate_name, [stim.target_x(0), stim.target_combiner(), stim.target_z(1)]
    )
    empty_op = stim.CircuitInstruction(gate_name, [])
    for op in (pauli_op, empty_op):
        with pytest.raises(NotImplementedError) as exc_inst:
            base.compile_instruction(op)
        assert str(exc_inst.value) == expected_msg
        with pytest.raises(NotImplementedError) as exc_app:
            base.compile_and_append_instruction(op)
        assert str(exc_app.value) == expected_msg
        assert base.steps == steps_before
        assert base._last_packed_snapshot is snap_before
        cur_snap = base.ref_chp.snapshot_packed()
        for idx in range(len(chp_before)):
            assert np.array_equal(chp_before[idx], cur_snap[idx])


def test_compile_reference_category_dispatch_and_meas_reset_flags() -> None:
    """Category dispatch separates annotations, heralds, MPAD, noise, resets, and measures."""
    circ = stim.Circuit(
        "R 0 1\n"
        "X 0\n"
        "M !0\n"
        "TICK\n"
        "MZ 0\n"
        "MRZ 0\n"
        "MZ 0\n"
        "RZ 0\n"
        "RX 1\n"
        "MX 1\n"
        "MRX !1\n"
        "MX 1\n"
        "RY 0\n"
        "MY !0\n"
        "MRY 0\n"
        "MY 0\n"
        "MXX 0 1\n"
        "MYY !0 1\n"
        "MZZ 0 !1\n"
        "MPP X0*!Z1\n"
        "MPAD 0 1\n"
        "HERALDED_ERASE(0.1) 0 !1\n"
        "HERALDED_PAULI_CHANNEL_1(0.01, 0.01, 0.01, 0.01) !0 1\n"
        "X_ERROR(0.1) 0\n"
        "DEPOLARIZE2(0.1) 0 1\n"
        "E(0.1) X0 Z1\n"
        "ELSE_CORRELATED_ERROR(0.1) Z0\n"
        "DETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    ref = CompiledReferenceCircuit(circ)
    by_name: dict[str, list[Any]] = {}
    for st in ref.steps:
        by_name.setdefault(st.op.name, []).append(st)

    for noop_name in (
        "MPAD", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "X_ERROR",
        "DEPOLARIZE2", "E", "ELSE_CORRELATED_ERROR", "TICK",
        "DETECTOR", "OBSERVABLE_INCLUDE",
    ):
        assert all(isinstance(st, NoopStepMeta) for st in by_name[noop_name])
        assert all(not isinstance(st, MeasStepMeta) for st in by_name[noop_name])

    for rst_name in ("R", "RX", "RY"):
        for st in by_name[rst_name]:
            assert isinstance(st, MeasStepMeta)
            assert st.produces_measurements is False
            assert st.is_reset is True
            assert all(sub.meta.m_ref == 0 for sub in st.substeps)

    m_steps = by_name["M"]
    for name in ("M", "MX", "MY", "MXX", "MYY", "MZZ", "MPP"):
        st = by_name[name][0]
        assert isinstance(st, MeasStepMeta)
        assert st.produces_measurements is True and st.is_reset is False
    for name in ("MR", "MRX", "MRY"):
        st = by_name[name][0]
        assert isinstance(st, MeasStepMeta)
        assert st.produces_measurements is True and st.is_reset is True
    assert m_steps[0].substeps[0].invert_result is True and m_steps[0].substeps[0].meta.m_ref == 1
    assert m_steps[1].substeps[0].meta.m_ref == 1  # MZ canonicalizes to M, did not reset |1>
    assert by_name["MR"][0].substeps[0].meta.m_ref == 1
    assert by_name["MR"][0].substeps[0].meta.reset_flip_ref is True
    assert m_steps[2].substeps[0].meta.m_ref == 0  # MRZ reset |1> -> |0>
    assert by_name["MX"][0].substeps[0].meta.m_ref == 0
    assert by_name["MX"][0].substeps[0].meta.is_random is False
    assert [op.name for op in by_name["MX"][0].substeps[0].pre_ops] == ["H"]
    assert by_name["MRX"][0].substeps[0].invert_result is True
    assert by_name["MY"][0].substeps[0].meta.m_ref == 0
    assert by_name["MY"][0].substeps[0].meta.is_random is False
    assert [op.name for op in by_name["MY"][0].substeps[0].pre_ops] == ["H_YZ"]
    assert by_name["MYY"][0].substeps[0].invert_result is True
    assert by_name["MZZ"][0].substeps[0].invert_result is True


def test_compile_reference_snapshot_reuse_and_inv_dirty_resets() -> None:
    """Reuse packed snapshots for no-op unitaries/det-meas and clear _inv_dirty on steps."""
    ref_det = CompiledReferenceCircuit(stim.Circuit("R 0\nM 0\nRX 0\nM 0\nTICK\nM 0\n"))
    assert ref_det.steps[1]._packed_words is ref_det.steps[0]._packed_words
    assert ref_det.steps[2]._packed_words is not ref_det.steps[1]._packed_words
    assert ref_det.steps[3]._packed_words is not ref_det.steps[2]._packed_words
    assert ref_det.steps[5]._packed_words is ref_det.steps[3]._packed_words
    assert ref_det.ref_chp._inv_dirty is False

    ref_seq = CompiledReferenceCircuit(stim.Circuit("R 0 1 2\nCX 0 1 1 2\nI 0\nH 0 0\nI 1\n"))
    assert ref_seq.ref_chp._inv_dirty is False
    assert ref_seq.steps[2]._packed_words is ref_seq.steps[1]._packed_words
    assert ref_seq.steps[3].are_groups_disjoint is False  # type: ignore[union-attr]
    assert ref_seq.steps[4]._packed_words is ref_seq.steps[3]._packed_words

    exact_64 = CompiledReferenceCircuit(stim.Circuit("H 63"))
    assert exact_64.num_qubits == 64 and exact_64.n_words == 1

    wide = CompiledReferenceCircuit(stim.Circuit("R 0 64\nH 64\nMPP !X64*X64\n"))
    assert wide.num_qubits == 65 and wide.n_words == 2
    assert wide.steps[2].substeps[0].meta.beta_Z_words.shape == (2,)  # type: ignore[union-attr]


def test_compile_reference_pair_and_multi_term_mpp_outcomes() -> None:
    """Verify MXX/MYY/MZZ, multi-term MPP, basis pre/post ops, and MRX/MRY resets."""
    circ = stim.Circuit(
        "R 0 1 2 3\n"
        "H 0\n"
        "CX 0 1\n"
        "MXX !0 !1\n"
        "TICK\n"
        "MXX 0 1 2 3\n"
        "MYY !0 1\n"
        "MZZ 0 !1\n"
        "RX 2\n"
        "Z 2\n"
        "MRX 2\n"
        "MX 2\n"
        "RY 0 1\n"
        "MPP !Y0*Y1 !X0*!Z1 X0*Z1*Y2\n"
    )
    ref = CompiledReferenceCircuit(circ)
    by_name: dict[str, list[MeasStepMeta]] = {}
    for st in ref.steps:
        if isinstance(st, MeasStepMeta):
            by_name.setdefault(st.gate_name, []).append(st)

    # Bell state |00>+|11> has XX=+1 (0), YY=-1 (1^inv), ZZ=+1 (0^inv); !0 !1 -> False.
    assert by_name["MXX"][0].substeps[0].meta.m_ref == 0
    assert by_name["MXX"][0].substeps[0].invert_result is False
    assert len(by_name["MXX"][1].substeps) == 2
    assert [op.name for op in by_name["MXX"][0].substeps[0].pre_ops] == ["H", "CX"]
    assert [op.name for op in by_name["MXX"][0].substeps[0].post_ops] == ["CX", "H"]
    assert by_name["MYY"][0].substeps[0].meta.m_ref == 1
    assert by_name["MYY"][0].substeps[0].invert_result is True
    assert [op.name for op in by_name["MYY"][0].substeps[0].pre_ops] == ["H_YZ", "CX"]
    assert [op.name for op in by_name["MYY"][0].substeps[0].post_ops] == ["CX", "H_YZ"]
    assert by_name["MZZ"][0].substeps[0].meta.m_ref == 0
    assert by_name["MZZ"][0].substeps[0].invert_result is True
    assert [op.name for op in by_name["MZZ"][0].substeps[0].pre_ops] == ["CX"]
    assert [op.name for op in by_name["MZZ"][0].substeps[0].post_ops] == ["CX"]

    # Z flips |+> to |-> (m_ref=1), MRX resets back to |+> so follow-up MX reads 0.
    assert by_name["MRX"][0].substeps[0].meta.m_ref == 1
    assert by_name["MX"][0].substeps[0].meta.is_random is False
    assert by_name["MX"][0].substeps[0].meta.m_ref == 0

    mpp_subs = by_name["MPP"][0].substeps
    assert len(mpp_subs) == 3
    assert mpp_subs[0].invert_result is True
    assert mpp_subs[0].meta.is_random is False and mpp_subs[0].meta.m_ref == 0
    assert [op.name for op in mpp_subs[0].pre_ops] == ["H_YZ", "H_YZ", "CX"]
    assert [op.name for op in mpp_subs[0].post_ops] == ["CX", "H_YZ", "H_YZ"]
    assert mpp_subs[1].invert_result is False
    assert [op.name for op in mpp_subs[2].pre_ops] == ["H", "H_YZ", "CX", "CX"]
    assert [str(op) for op in mpp_subs[2].post_ops] == ["CX 2 0", "CX 1 0", "H_YZ 2", "H 0"]


def test_compile_reference_unitary_mixed_overlapping_and_bounds() -> None:
    """Exercise unitary disjoint/overlapping, mixed rec/sweep, exact bounds, and snapshots."""
    circ = stim.Circuit(
        "R 0 1 2 3\n"
        "I 0\n"
        "II 0 1\n"
        "H 0 1\n"
        "CX 0 1 1 2\n"
        "M 0 1\n"
        "CX rec[-1] 2\n"
        "XCZ 0 rec[-1]\n"
        "TICK\n"
        "CX rec[-1] 0 1 2\n"
        "XCZ 0 rec[-1] 1 2\n"
        "TICK\n"
        "CX sweep[0] 2\n"
        "TICK\n"
        "CX sweep[0] 0 2 3\n"
        "TICK\n"
        "X_ERROR(0.1) 0\n"
        "TICK\n"
        "X_ERROR(0.1) 3\n"
        "MPAD 0 1\n"
        "DETECTOR rec[-1]\n"
    )
    ref = CompiledReferenceCircuit(circ)
    assert ref.ref_chp._inv_dirty is False
    unitaries = [st for st in ref.steps if isinstance(st, UnitaryStepMeta)]
    assert unitaries[0].gate_name == "I" and unitaries[0]._op_bounds_nq == 4
    assert unitaries[0]._is_mixed is False
    assert unitaries[0]._packed_words is ref.steps[0]._packed_words  # I leaves _inv_dirty=False
    assert unitaries[1].gate_name == "II"
    assert unitaries[2].are_groups_disjoint is True and unitaries[2]._group_packed_words is None
    assert unitaries[2]._packed_words is not unitaries[0]._packed_words  # H sets _inv_dirty=True
    assert unitaries[3].are_groups_disjoint is False
    assert unitaries[3]._group_packed_words is not None
    assert len(unitaries[3]._group_packed_words) == 2
    assert unitaries[3]._packed_words is unitaries[3]._group_packed_words[-1]

    # Pure classical feedback compiles as noop; mixed emits quantum_op on quantum pairs only.
    fb_steps = [st for st in ref.steps if st.op.name in ("CX", "XCZ")]
    assert isinstance(fb_steps[1], NoopStepMeta)  # CX rec[-1] 2
    assert isinstance(fb_steps[2], NoopStepMeta)  # XCZ 0 rec[-1]
    assert isinstance(fb_steps[5], NoopStepMeta)  # CX sweep[0] 2
    for mixed, pair in ((fb_steps[3], [1, 2]), (fb_steps[4], [1, 2]), (fb_steps[6], [2, 3])):
        assert isinstance(mixed, UnitaryStepMeta)
        assert mixed._is_mixed is True
        assert mixed.target_groups == [tuple(pair)]
        assert mixed.quantum_op == stim.CircuitInstruction(mixed.gate_name, pair)

    # Propagated snapshot after H/CX differs from initial snapshot on trailing NoopStepMeta.
    assert isinstance(ref.steps[-1], NoopStepMeta) and ref.steps[-1]._packed_words is not None
    assert ref._initial_packed_words is not None
    assert not np.array_equal(ref.steps[-1]._packed_words[0], ref._initial_packed_words[0])

    # Exact _op_bounds_nq boundaries: 0 and num_qubits-1 in-bounds; num_qubits out-of-bounds.
    noops = {str(st.op): st for st in ref.steps if isinstance(st, NoopStepMeta)}
    assert noops["X_ERROR(0.1) 0"]._op_bounds_nq == 4
    assert noops["X_ERROR(0.1) 3"]._op_bounds_nq == 4
    assert noops["MPAD 0 1"]._op_bounds_nq == 4
    assert noops["DETECTOR rec[-1]"]._op_bounds_nq == 4

    for oob_op in (
        stim.CircuitInstruction("I", [4]),
        stim.CircuitInstruction("I", [9]),
        stim.CircuitInstruction("X_ERROR", [4], [0.1]),
        stim.CircuitInstruction("X_ERROR", [9], [0.1]),
    ):
        oob_st, _, _ = ref.compile_instruction(oob_op)
        assert isinstance(oob_st, (UnitaryStepMeta, NoopStepMeta))
        assert oob_st._op_bounds_nq == -1


def test_compile_reference_incremental_wrappers_and_snapshots() -> None:
    """Verify snapshot caching and compile_instruction/append_compiled_step state isolation."""
    ref = CompiledReferenceCircuit(stim.Circuit("R 0 1\n"))
    snap0 = ref._get_packed_snapshot()
    ref.ref_chp.do_unitary("H", [0])
    snap1 = ref._get_packed_snapshot()
    assert not np.array_equal(snap0[0], snap1[0])
    assert ref.ref_chp._inv_dirty is False
    assert ref._get_packed_snapshot() is snap1

    ref2 = CompiledReferenceCircuit(stim.Circuit("R 0 1\n"))
    snap_before = ref2._last_packed_snapshot
    chp_before = [w.copy() for w in ref2.ref_chp.snapshot_packed()]
    st_h, chp_h, snap_h = ref2.compile_instruction(stim.CircuitInstruction("H", [0]))
    assert isinstance(st_h, UnitaryStepMeta)
    assert len(ref2.steps) == 1 and ref2._last_packed_snapshot is snap_before
    for idx in range(len(chp_before)):
        assert np.array_equal(ref2.ref_chp.snapshot_packed()[idx], chp_before[idx])
    assert not np.array_equal(chp_h.z2x[0], chp_before[2][0])

    committed = ref2.append_compiled_step(st_h, chp_h, snap_h)
    assert committed is st_h
    assert ref2.ref_chp is chp_h
    assert ref2._last_packed_snapshot is snap_h
    assert ref2.steps[-1] is st_h and len(ref2.steps) == 2

    for op in (
        stim.CircuitInstruction("M", [0]),
        stim.CircuitInstruction("MR", [0]),
        stim.CircuitInstruction("R", [1]),
        stim.CircuitInstruction("X", [0]),
        stim.CircuitInstruction("MPAD", [0]),
        stim.CircuitInstruction("HERALDED_ERASE", [0], [0.1]),
        stim.CircuitInstruction("X_ERROR", [0], [0.1]),
        stim.CircuitInstruction("TICK", []),
    ):
        app = ref2.compile_and_append_instruction(op)
        assert ref2.steps[-1] is app and app.op == op


def test_compile_reference_mpp_reduction_and_anti_hermitian() -> None:
    """MPP reduces repeated-qubit products to identity/Pauli and rejects anti-Hermitian terms."""
    circ = stim.Circuit(
        "R 0 1\n"
        "MPP !X0*X0\n"
        "TICK\n"
        "MPP X0*Z0*X0*Z0\n"
        "TICK\n"
        "MPP !X0*Z0*X0*Z0\n"
        "TICK\n"
        "MPP X0*Z0*X0*Z1\n"
    )
    ref = CompiledReferenceCircuit(circ)
    mpp_steps = [st for st in ref.steps if isinstance(st, MeasStepMeta) and st.gate_name == "MPP"]
    assert len(mpp_steps) == 4
    for step, expected_inv in (
        (mpp_steps[0], True),
        (mpp_steps[1], True),
        (mpp_steps[2], False),
    ):
        sub = step.substeps[0]
        assert sub.q == -1 and sub.invert_result is expected_inv
        assert isinstance(sub.meta, DetMeasMeta)
        assert sub.meta.is_random is False
    assert mpp_steps[3].substeps[0].q == 0 and mpp_steps[3].substeps[0].invert_result is True

    anti = stim.Circuit("R 0\nMPP X0*Y0\n")
    with pytest.raises(ValueError, match="anti-Hermitian"):
        CompiledReferenceCircuit(anti)

    oob_mpp = stim.Circuit("MPP X5*Z6")[0]
    with pytest.raises(IndexError):
        CompiledReferenceCircuit(stim.Circuit("R 0 1")).compile_instruction(oob_mpp)


def test_compile_reference_unitary_cocycle_cache_invalidation() -> None:
    """Populating per-group packed words invalidates cached cocycle arrays."""
    ref = CompiledReferenceCircuit(stim.Circuit("R 0 1\nH 0 1\n"))
    u = ref.steps[1]
    assert isinstance(u, UnitaryStepMeta)
    assert u._group_packed_words is None
    cached = u.get_cpp_cocycle_arrays(ref.n_words)
    assert u._cpp_cocycle_arrays is cached
    assert ref.steps[0]._packed_words is not None
    group_pws = u.ensure_group_packed_words(ref.steps[0]._packed_words)
    assert len(group_pws) == 2
    assert u._group_packed_words is group_pws
    assert u._cpp_cocycle_arrays is None
    rebuilt = u.get_cpp_cocycle_arrays(ref.n_words)
    assert rebuilt is not cached
    assert u._cpp_cocycle_arrays is rebuilt
