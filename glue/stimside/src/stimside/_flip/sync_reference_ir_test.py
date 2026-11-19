"""Tests for IR-backed synchronized reference walking and FlipsideSimulator IR sharing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import simulator_flip
from stimside._flip import sync_reference
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, TrivialOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8Flip
from stimside.simulator_flip import FlipsideSimulator
from stimside.util import circuit as circuit_util
from stimside.util import circuit_ir, known_states
from stimside.util.circuit import unroll_circuit


class _CustomHookHandler(CompiledOpHandler[FlipsideSimulator]):
    """Handler exercising mpad_z_collapse_targets and op_z_collapse_targets edge cases."""

    def __init__(self, mpad_targets: list[int] | None, op_targets: Any) -> None:
        self._mpad_targets = mpad_targets
        self._op_targets = op_targets

    def handle_op(self, op: stim.CircuitInstruction, sss: FlipsideSimulator) -> None:
        sss.do_on_flip_simulator(op)

    def clear(self) -> None:
        pass

    def mpad_z_collapse_targets(self, op: stim.CircuitInstruction) -> list[int] | None:
        return self._mpad_targets if op.name == "MPAD" else None

    def op_z_collapse_targets(
        self, op: stim.CircuitInstruction
    ) -> tuple[list[int], list[int]] | None:
        return self._op_targets(op)


def test_sync_reference_single_ir_and_no_flat_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_sync_reference builds at most one IR and never triggers lazy alignment."""
    circ = stim.Circuit(
        "R 0 1\n"
        "REPEAT 3 {\n"
        "  H 0\n"
        "  CX 0 1\n"
        "  M 1\n"
        "  DETECTOR rec[-1]\n"
        "}\n"
    )
    build_calls = 0
    unroll_calls = 0
    orig_build = circuit_ir.build_ir
    orig_unroll = circuit_util.unroll_circuit

    def spy_build(*args: Any, **kwargs: Any) -> circuit_ir.CircuitIR:
        nonlocal build_calls
        build_calls += 1
        return orig_build(*args, **kwargs)

    def spy_unroll(c: stim.Circuit) -> list[stim.CircuitInstruction]:
        nonlocal unroll_calls
        unroll_calls += 1
        return orig_unroll(c)

    def boom_align(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("sync_reference must not trigger lazy flat alignment")

    monkeypatch.setattr(sync_reference, "build_ir", spy_build)
    monkeypatch.setattr(circuit_ir, "unroll_circuit", spy_unroll)
    monkeypatch.setattr(circuit_ir, "_align_flat_ops", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_where", boom_align)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_flat_ops", boom_align)

    uops = unroll_circuit(circ)
    want_ref, want_mpad, want_op = sync_reference.compute_sync_reference(
        circ.num_qubits, uops, None
    )
    assert build_calls == 1 and unroll_calls == 0
    assert want_ref.dtype == np.bool_ and want_ref.shape == (3,)

    for label, src, exp_unroll in (
        ("tuple", tuple(uops), 0),
        ("circuit", circ, 1),
        ("gen", (op for op in uops), 0),
    ):
        build_calls = 0
        unroll_calls = 0
        got = sync_reference.compute_sync_reference(circ.num_qubits, src, None)
        assert build_calls == 1, label
        assert unroll_calls == exp_unroll, label
        assert np.array_equal(got[0], want_ref) and got[1] == want_mpad and got[2] == want_op

    for uc in (False, True):
        ir = orig_build(circ, uc=uc)
        build_calls = 0
        unroll_calls = 0
        got_ref, got_mpad, got_op = sync_reference.compute_sync_reference(
            circ.num_qubits, compiled_op_handler=None, ir=ir
        )
        assert build_calls == 0 and unroll_calls == 0
        assert np.array_equal(got_ref, want_ref)
        assert got_mpad == want_mpad and got_op == want_op


def test_sync_reference_input_validation_empty_and_copy_independence() -> None:
    """Reject missing or conflicting inputs, skip annotations/noise, and isolate outputs."""
    circ = stim.Circuit("R 0\nM 0\n")
    ir = circuit_ir.build_ir(circ)
    uops = unroll_circuit(circ)
    with pytest.raises(ValueError, match="Cannot pass both unrolled_ops and ir"):
        sync_reference.compute_sync_reference(1, uops, None, ir=ir)
    with pytest.raises(ValueError, match="Cannot pass both unrolled_ops and ir"):
        sync_reference.compute_sync_reference(1, [], None, ir=ir)
    with pytest.raises(ValueError, match="Either unrolled_ops or ir must be provided"):
        sync_reference.compute_sync_reference(1, None, None)
    with pytest.raises(TypeError, match="CircuitRepeatBlock"):
        sync_reference.compute_sync_reference(1, [stim.Circuit("REPEAT 2 {\n H 0\n}")[0]], None)

    empty_ref, empty_mpad, empty_op = sync_reference.compute_sync_reference(3, [], None)
    assert empty_ref.dtype == np.bool_ and empty_ref.shape == (0,)
    assert empty_mpad == {} and empty_op == {}

    skip_circ = stim.Circuit(
        "QUBIT_COORDS(1, 2) 0\nSHIFT_COORDS(3)\nTICK\nI 0\nII 0 1\n"
        "X_ERROR(1.0) 0\nY_ERROR(1.0) 0\nZ_ERROR(1.0) 1\n"
        "DEPOLARIZE1(0.75) 0\nDEPOLARIZE2(0.75) 0 1\n"
        "PAULI_CHANNEL_1(0.5, 0.25, 0.25) 0\n"
        "PAULI_CHANNEL_2(0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1\n"
        "I_ERROR(0.5) 0\nII_ERROR(0.5) 0 1\n"
        "E(1.0) X0 Z1\nELSE_CORRELATED_ERROR(1.0) Z0\n"
        "M 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    s_ref, s_mpad, s_op = sync_reference.compute_sync_reference(2, skip_circ, None)
    assert s_ref.dtype == np.bool_ and s_ref.tolist() == [False, False]
    assert s_mpad == {} and s_op == {}

    c_copy = stim.Circuit("R 0 1\nH 0 1\nMPAD 0\nI 1\n")
    ir_copy = circuit_ir.build_ir(c_copy)
    hook = _CustomHookHandler([0], lambda op: ([1], []) if op.name == "I" else None)
    r1, m1, o1 = sync_reference.compute_sync_reference(2, None, hook, ir=ir_copy)
    r1[:] = True
    m1[0][0][:] = True
    m1[0][1][:] = False
    o1[3][0][1] = True
    o1[3][1].append(99)
    r2, m2, o2 = sync_reference.compute_sync_reference(2, None, hook, ir=ir_copy)
    assert r2.tolist() == [False]
    assert m2[0][0].tolist() == [False] and m2[0][1].tolist() == [True]
    assert o2 == {3: ({1: False}, [1])}


def test_sync_reference_single_qubit_meas_resets_and_heralds() -> None:
    """Verify R/RX/RY/RZ, M/MX/MY/MZ, MR*, inverted targets, order, and heralds."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "X 0\n"
        "HERALDED_ERASE(1.0) 0 !1 2\n"
        "HERALDED_PAULI_CHANNEL_1(0.5, 0.1, 0.1, 0.1) !0 1\n"
        "M 0 !0 1\n"
        "RZ 0\n"
        "MZ !0\n"
        "H 0\n"
        "MX 0 !0\n"
        "MX(0.2) 0\n"
        "H_YZ 1\n"
        "MY 1 !1\n"
        "MY(0.2) 1\n"
        "RX 0\n"
        "Z 0\n"
        "MRX(0.2) !0\n"
        "MX 0\n"
        "RY 1\n"
        "X 1\n"
        "MRY(0.2) !1\n"
        "MY 1\n"
        "X 2\n"
        "MR(0.2) !2\n"
        "MRZ 2\n"
        "M 2\n"
    )
    ir = circuit_ir.build_ir(circ)
    ref_u, mpad_u, op_u = sync_reference.compute_sync_reference(
        circ.num_qubits, unroll_circuit(circ), None
    )
    ref_ir, mpad_ir, op_ir = sync_reference.compute_sync_reference(
        circ.num_qubits, None, None, ir=ir
    )
    assert np.array_equal(ref_u, ref_ir)
    assert mpad_u == mpad_ir == {} and op_u == op_ir == {}
    want = [
        False, False, False, False, False,  # 2 heralds (3 + 2 targets)
        True, False, False,  # M 0 !0 1
        True,  # MZ !0 after RZ 0
        False, True, False,  # MX 0 !0; MX(0.2) 0 (post-MX stays +X)
        False, True, False,  # MY 1 !1; MY(0.2) 1 (post-MY stays +Y)
        False, False,  # MRX !0 on -X -> 0, resets to +X -> MX 0 == 0
        False, False,  # MRY !1 on -Y -> 0, resets to +Y -> MY 1 == 0
        False, False, False,  # MR !2 on |1> -> 0, MRZ 2 -> 0, M 2 -> 0
    ]
    assert ref_u.tolist() == want

    # Unsorted RX/RY target order preserves RNG draw order across entangled qubits (seed=0).
    rx_ord = stim.Circuit("R 0 1 2\nH 0\nCX 0 2\nRX 1 0\nMX 2\n")
    ry_ord = stim.Circuit("R 0 1 2\nH 0\nCX 0 2\nRY 1 0\nMY 2\n")
    assert sync_reference.compute_sync_reference(3, rx_ord, None)[0].tolist() == [True]
    assert sync_reference.compute_sync_reference(3, ry_ord, None)[0].tolist() == [False]
    # Non-deterministic H measurements pin seed=0 reference RNG sequence.
    assert sync_reference.compute_sync_reference(2, stim.Circuit("H 0 1\nM 0 1\n"), None)[
        0
    ].tolist() == [False, True]


def test_sync_reference_product_measurements_and_post_state() -> None:
    """Verify MXX/MYY/MZZ, multi-term MPP, repeated-qubit terms, inversions, and uncompute."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "X 0\n"
        "MZZ !0 !1 0 1\n"
        "RX 0 1\n"
        "Z 0\n"
        "MXX !0 !1 0 !1\n"
        "RY 0 1\n"
        "X 1\n"
        "MYY !0 !1 0 !1\n"
        "RX 0\n"
        "RY 1\n"
        "R 2\n"
        "Z 0\n"
        "MPP X0*Y1*Z2 !X0*Y1*Z2 !X0*!Y1*Z2 X0 !X0*X0*Z2 Y0*Y0*Z2\n"
        "MX 0\n"
        "MY 1\n"
        "M 2\n"
    )
    ref, _, _ = sync_reference.compute_sync_reference(3, circ, None)
    want = [
        True, True,  # MZZ !0 !1 (inverts cancel: 1^0^T^T=1), MZZ 0 1 (1)
        True, False,  # MXX !0 !1 (q0=-X, q1=+X -> 1), MXX 0 !1 (1^T=0)
        True, False,  # MYY !0 !1 (q0=+Y, q1=-Y -> 1), MYY 0 !1 (1^T=0)
        True, False, True,  # MPP X0*Y1*Z2 (-1), !X0*Y1*Z2 (0), !X0*!Y1*Z2 (1)
        True, True,  # MPP X0 (-X -> 1), !X0*X0*Z2 (+Z2 with ! -> 1)
        False,  # MPP Y0*Y0*Z2 (+Z2 -> 0)
        True, False, False,  # post-MPP basis uncomputation: MX 0=1, MY 1=0, M 2=0
    ]
    assert ref.tolist() == want

    # Non-deterministic MXX/MYY/MZZ and MPP with seed=0 and post-state consistency.
    nondet = stim.Circuit("H 0 1\nMXX 0 1\nMYY 0 1\nMZZ 0 1\nMPP X0*Y1 !X0*X0*Z1\n")
    nref, _, _ = sync_reference.compute_sync_reference(2, nondet, None)
    assert nref.tolist() == [False, False, True, True, True]


def test_sync_reference_classical_control_and_fb_cache() -> None:
    """Exercise all 5 controlled Paulis, rec/sweep/mixed targets, tag rec[, and fb_cache cap."""
    # Omit initial R so ref_ts has only 2 measurement bits while ref_bits has 7 (rec[-6] is invalid
    # in ref_ts, and ref_bits[0]=True exposes any sweep[0] treated as rec[0]).
    circ = stim.Circuit(
        "CX 0 1\n"  # non-feedback CX cached BEFORE feedback CX (tests full op key, not name)
        "X 0\n"
        "M 0 1\n"  # rec[-7]=True, rec[-6]=False -> ref_bits[0]=True
        "MPAD 1 0 1 0\n"  # ref_bits=[T, F, T, F, T, F]
        "HERALDED_ERASE(0.5) 0\n"  # adds False -> ref_bits[-1]=False, [-5]=True, [-7]=True
        "H 4 5 7\n"  # |+X> on q4, q5, q7 so Z flip is observable via MX
        "CX rec[-1] 6 rec[-5] 2 sweep[0] 6 0 1\n"  # noop(F q6), X q2(T), sweep q6, CX 0 1
        "CY rec[-1] 6 rec[-5] 3 sweep[0] 6\n"  # noop(F q6), Y q3(T), sweep q6
        "CZ rec[-1] 7 rec[-5] 4 7 rec[-1] 5 rec[-5] sweep[0] 7\n"  # Z q4(T), Z q5(T); q7 stays +X
        "XCZ 6 rec[-1] 2 rec[-5] 6 sweep[0] sweep[0] 6\n"  # X q2(T -> 0); sweep right & left noop
        "YCZ 6 rec[-1] 3 rec[-5] 6 sweep[0]\n"  # Y q3(T -> 0); sweep noop
        "XCZ sweep[0] 6\n"  # unfused sweep-only op (without rec[ in op) must not reach ref_ts.do
        "YCZ 6 sweep[0]\n"  # unfused sweep-only q-sweep op
        "REPEAT 2 {\n"
        "  CX[rec[0] 0 1\n"  # tag with rec[ hits feedback branch and fb_cache twice
        "  CX rec[-5] 2\n"  # feedback CX hits fb_cache twice while reading ref_bits
        "}\n"
        "M 0 1 2 3 6\n"
        "MX 4 5 7\n"
    )
    ref, _, _ = sync_reference.compute_sync_reference(circ.num_qubits, circ, None)
    assert ref[:7].tolist() == [True, False, True, False, True, False, False]
    assert ref[-8:].tolist() == [True, True, False, False, False, True, True, False]

    with pytest.raises(IndexError):
        sync_reference.compute_sync_reference(1, stim.Circuit("CX rec[-1] 0\n"), None)
    with pytest.raises(IndexError):
        sync_reference.compute_sync_reference(1, stim.Circuit("CZ 0 rec[-1]\n"), None)
    with pytest.raises(IndexError):
        sync_reference.compute_sync_reference(1, stim.Circuit("M 0\nCX rec[-5] 0\n"), None)

    # Exceed the 4096-entry fb_cache bound with 4105 genuinely distinct tagged CX instructions.
    many_cx = stim.Circuit("\n".join(f"CX[t{i}] 0 1" for i in range(4105)))
    ref_many, _, _ = sync_reference.compute_sync_reference(2, many_cx, None)
    assert ref_many.shape == (0,)


def test_sync_reference_mpad_and_op_z_collapse_hooks() -> None:
    """Exercise MPAD split offsets, ! targets, empty/unordered/dup hooks, and pre-skip op_z."""
    circ = stim.Circuit(
        "R 0 1 2\n"
        "H 0\n"
        "X 1\n"
        "M 2\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 0 1] 0 1 0 1 1 0\n"
        "I[LEAKAGE_TRANSITION_1: (1.0, 0-->1)] 0 0\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 0-->1)] 1\n"
        "M 0 1\n"
    )
    handler = LeakageUint8Flip().compile_op_handler(circuit=circ, batch_size=4)
    ref, mpad_ref_z, op_ref_z = sync_reference.compute_sync_reference(
        circ.num_qubits, circ, handler
    )
    assert ref.tolist() == [False, False, True, False, True, True, False, False, True]
    # Three fused MPAD copies start at measurement offsets 1, 3, and 5 (after M 2 at offset 0).
    assert sorted(mpad_ref_z) == [1, 3, 5]
    rz0, col0 = mpad_ref_z[1]
    rz1, col1 = mpad_ref_z[3]
    rz2, col2 = mpad_ref_z[5]
    assert rz0.dtype == np.bool_ and col0.dtype == np.bool_
    assert rz0.tolist() == [False, True] and col0.tolist() == [True, False]
    assert rz1.tolist() == [False, True] and col1.tolist() == [False, False]
    assert rz2.tolist() == [False, True] and col2.tolist() == [False, False]
    # I[...] is at unrolled index 5; I_ERROR[...] (Category.NOISE) is at unrolled index 6.
    assert op_ref_z[5] == ({0: False}, [])
    assert op_ref_z[6] == ({1: True}, [])

    # Real handler rejects non-multiple fused MPAD targets via check_tag_qubits.
    bad_fused = stim.Circuit("R 0 1 2\nMPAD[LEAKAGE_MEASUREMENT: (0, 0) (1, 1) : 0 1] 0 0 0\n")
    with pytest.raises(ValueError):
        LeakageUint8Flip().compile_op_handler(circuit=bad_fused, batch_size=2)

    # Custom hook: non-multiple MPAD falls back via split_fused_instruction; noisy MPAD args work.
    bad_mpad = stim.Circuit("R 0 1 2\nH 2\nX 0\nMPAD(0.1) 1 1 0\nM 2\n")
    bad_ref, bad_mpad_z, _ = sync_reference.compute_sync_reference(
        bad_mpad.num_qubits, bad_mpad, _CustomHookHandler([2, 0], lambda _op: None)
    )
    assert bad_ref.tolist() == [True, True, False, False]
    assert list(bad_mpad_z) == [0]
    assert bad_mpad_z[0][0].tolist() == [False, True]
    assert bad_mpad_z[0][1].tolist() == [True, False]


def test_sync_reference_mpad_custom_hooks() -> None:
    """Exercise custom/empty/duplicate MPAD hooks, pre-skip op_z hooks, and TrivialOpHandler."""
    c_custom = stim.Circuit("R 0 1 2 3 4\nH 1 2\nX 3\nTICK\nX_ERROR(1.0) 0\nMPAD 1\nI 0\n")
    saw_noise_or_tick: list[str] = []

    def op_hook(op: stim.CircuitInstruction) -> tuple[list[int], list[int]] | None:
        if op.name in ("TICK", "X_ERROR"):
            saw_noise_or_tick.append(op.name)
            return ([], [3])
        if op.name == "I":
            return ([1, 1, 0], [0, 2, 2, 3, 4])
        return None

    _, c_mpad_empty, c_op = sync_reference.compute_sync_reference(
        5, c_custom, _CustomHookHandler([], op_hook)
    )
    assert saw_noise_or_tick == ["TICK", "X_ERROR"]
    assert c_mpad_empty[0][0].shape == (0,) and c_mpad_empty[0][1].shape == (0,)
    assert c_op == {
        3: ({3: True}, []),
        4: ({3: True}, []),
        6: ({0: False, 1: False, 2: None, 3: True, 4: False}, [1]),
    }

    # Duplicate targets in mpad_z_collapse_targets collapse q1 once.
    _, dup_mpad, _ = sync_reference.compute_sync_reference(
        2, stim.Circuit("R 0 1\nH 1\nMPAD 0\n"), _CustomHookHandler([1, 1], lambda _op: None)
    )
    assert dup_mpad[0][0].tolist() == [False, False]
    assert dup_mpad[0][1].tolist() == [True, False]

    # TrivialOpHandler lacks collapse hooks -> treated as None without error.
    t_ref, t_mpad, t_op = sync_reference.compute_sync_reference(
        1, stim.Circuit("R 0\nMPAD 1\nM 0\n"), TrivialOpHandler()
    )
    assert t_ref.tolist() == [True, False] and t_mpad == {} and t_op == {}


def test_sync_reference_other_category_and_unused_qubits() -> None:
    """Skip Category.OTHER safely and size TableauSimulator to caller num_qubits."""
    circ = stim.Circuit("H 4\nR 0\nMPP X4\n")
    ir = circuit_ir.build_ir(circ)
    other_instr = circuit_ir.IRInstr(
        unrolled_index=1,
        op=ir.instrs[1].op,
        tag="",
        category=circuit_ir.Category.OTHER,
        untagged_filter=circuit_ir.UntaggedFilter.NONE,
        num_measurements=0,
        first_measurement=0,
    )
    synth_ir = circuit_ir.CircuitIR(
        instrs=(ir.instrs[0], other_instr, ir.instrs[2]),
        num_qubits=1,
        num_measurements=1,
        has_repeat=False,
        uc=False,
        from_op_list=True,
    )
    hook = _CustomHookHandler(None, lambda op: ([4], []) if op.name == "MPP" else None)
    ref, mpad, opz = sync_reference.compute_sync_reference(5, None, hook, ir=synth_ir)
    assert ref.tolist() == [False] and mpad == {} and opz == {2: ({4: False}, [4])}


def test_flipside_simulator_batch_size_validation_and_u2l() -> None:
    """FlipsideSimulator validates batch_size and initializes unleaked-to-leaked records."""
    circ = stim.Circuit("R 0\nI[LEAKAGE_TRANSITION_1: (1.0, 0-->2)] 0\nM 0\n")
    coh = LeakageUint8Flip().compile_op_handler(circuit=circ, batch_size=2)
    bad_sizes: tuple[Any, ...] = (0, -1, True, 1.5)
    for bad_bs in bad_sizes:
        with pytest.raises(ValueError, match="batch_size"):
            FlipsideSimulator(circ, compiled_op_handler=coh, batch_size=bad_bs)

    sim = FlipsideSimulator(
        circ,
        compiled_op_handler=coh,
        batch_size=np.int64(2),  # type: ignore[arg-type]
        record_unleaked_to_leaked=True,
        seed=5,
    )
    assert sim.num_qubits == 1
    u2l = sim.run()
    assert u2l is not None and len(u2l) == 2 and all(arr.tolist() == [1] for arr in u2l)


def _install_simulator_ir_spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install build_ir/compute_sync_reference counters and poison secondary IR/alignment paths."""
    counts = {"build": 0, "sync": 0}
    orig_build = circuit_ir.build_ir
    orig_sync = sync_reference.compute_sync_reference

    def spy_build(*args: Any, **kwargs: Any) -> circuit_ir.CircuitIR:
        counts["build"] += 1
        return orig_build(*args, **kwargs)

    def spy_sync(*args: Any, **kwargs: Any) -> Any:
        counts["sync"] += 1
        return orig_sync(*args, **kwargs)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Unexpected second IR build or lazy flat alignment")

    monkeypatch.setattr(simulator_flip, "build_ir", spy_build)
    monkeypatch.setattr(simulator_flip, "compute_sync_reference", spy_sync)
    monkeypatch.setattr(known_states, "build_ir", boom)
    monkeypatch.setattr(sync_reference, "build_ir", boom)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_where", boom)
    monkeypatch.setattr(circuit_ir._FlatAlignment, "get_flat_ops", boom)
    return counts


def test_flipside_simulator_shares_single_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FlipsideSimulator builds one IR, shares it with KS/sync, and caches ref walk."""
    circ = stim.Circuit(
        "R 0 1\n"
        "REPEAT 2 {\n}\n"
        "REPEAT 2 {\n"
        "  REPEAT 2 {\n    X 0\n  }\n"
        "  H 1\n"
        "  CX 1 0\n"
        "}\n"
        "I[LEAKAGE_TRANSITION_1: (0.5, 0-->2)] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.0, 0) (1.0, 1) (1.0, 2) : 0] 0\n"
        "M 0 1\n"
        "DETECTOR rec[-1] rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    counts = _install_simulator_ir_spies(monkeypatch)
    for compute_ks in (True, False):
        counts["build"] = 0
        counts["sync"] = 0
        h1 = LeakageUint8Flip().compile_op_handler(circuit=circ, batch_size=4)
        sim = FlipsideSimulator(
            circ,
            compiled_op_handler=h1,
            batch_size=4,
            compute_known_states=compute_ks,
            disable_stabilizer_randomization=compute_ks,
            seed=7,
        )
        assert counts["build"] == (1 if compute_ks else 0) and sim.num_qubits == circ.num_qubits
        assert sim._unrolled_ops == unroll_circuit(circ) and sim.run() is None
        _ = sim.ref_measurements
        dets, obs, meas = (
            sim.get_detector_flips(),
            sim.get_observable_flips(),
            sim.get_final_measurement_records(),
        )
        assert counts["build"] == 1 and counts["sync"] == 1

        h2 = LeakageUint8Flip().compile_op_handler(circuit=circ, batch_size=4)
        sim2 = FlipsideSimulator(
            circ,
            compiled_op_handler=h2,
            batch_size=4,
            compute_known_states=compute_ks,
            disable_stabilizer_randomization=compute_ks,
            seed=7,
        )
        sim2.interactive_do(circ)
        assert np.array_equal(sim2.get_detector_flips(), dets)
        assert np.array_equal(sim2.get_observable_flips(), obs)
        assert np.array_equal(sim2.get_final_measurement_records(), meas)
        sim.clear()
        sim.interactive_do(circ)
        assert sim.get_final_measurement_records().shape == meas.shape
        assert counts["build"] == 2 and counts["sync"] == 2


def test_c2_modules_do_not_import_unroll_circuit() -> None:
    """Confirm unroll_circuit is absent from the migrated sync_reference walker."""
    assert not hasattr(sync_reference, "unroll_circuit")


def test_flipside_constructor_without_known_states_builds_no_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without known states the IR waits for the first reference walk, then is built once."""
    counts = _install_simulator_ir_spies(monkeypatch)
    circ = stim.Circuit("R 0 1\nREPEAT 3 {\n  H 0\n  CX 0 1\n  M 0 1\n}\nDETECTOR rec[-1] rec[-2]")
    h = TrivialOpHandler().compile_op_handler(circuit=circ, batch_size=4)
    sim = FlipsideSimulator(
        circ, compiled_op_handler=h, batch_size=4, compute_known_states=False, seed=7
    )
    assert counts["build"] == 0 and sim._unrolled_ops == unroll_circuit(circ)
    assert sim.num_qubits == circ.num_qubits
    sim.run()
    _ = sim.ref_measurements
    _ = sim.get_detector_flips()
    assert counts["build"] == 1 and counts["sync"] == 1
