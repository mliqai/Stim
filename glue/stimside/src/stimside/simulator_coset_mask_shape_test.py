"""Mask-shape validation of `CosetsideSimulator` conditional Cliffords and unmatched resets.

Covers `do_conditional_clifford` and `do_unmatched_noisy_reset_z`.

The C++ kernels `engine_batch_inject_general_clifford` and `engine_batch_unmatched_noisy_reset_z`
read `num_groups * B` resp. `num_qubits * B` mask bytes from the buffer they are handed, so a mask
of any other shape was an out-of-bounds heap read. The methods now raise ValueError first.
The tests check that

* every wrong shape raises ValueError on each path that consumes the mask, and that the kernel is
  NOT called (a spy that raises AssertionError stands in for it, so the tests never trigger the
  undefined behaviour themselves);
* a rejected call changes no simulator state at all, whether it replays a compiled step or runs
  past the end of the reference (frames, shots, RNG streams, step cursor, the compiled
  reference's steps and packed words), so a retry with a corrected mask acts exactly once;
* every mask is converted once, to a plain `np.ndarray` (`np.asarray`): the array whose shape is
  checked is the one the row reads and the kernel use, also for a list, an object whose
  `__array__` answers differently each time, or an ndarray subclass that lies about its `shape`
  or `__getitem__` (its hooks never run: it is read as the buffer it is);
* correctly shaped masks reach the kernel exactly as before (digests pinned on e4b7049).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit, UnitaryStepMeta
from stimside._native import coset_kernels
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_DCC_KERNEL = "engine_batch_inject_general_clifford_v3"
_GROUP_KERNEL = "engine_inject_group_v3"
_RESET_KERNEL = "engine_batch_unmatched_noisy_reset_z"
_PROJECT_KERNEL = "engine_project_pauli"
_POSTSELECT_KERNEL = "engine_postselect_batch"
_ALL_MASK_KERNELS = (
    _DCC_KERNEL,
    _GROUP_KERNEL,
    _RESET_KERNEL,
    _PROJECT_KERNEL,
    _POSTSELECT_KERNEL,
)
_N = 6  # qubits of every circuit below
_H4 = stim.CircuitInstruction("H", [0, 1, 2, 3])  # 4 disjoint groups
_CX_DISJOINT = stim.CircuitInstruction("CX", [0, 1, 2, 3])  # 2 disjoint groups
_CX_OVERLAP = stim.CircuitInstruction("CX", [0, 1, 1, 2, 2, 3])  # 3 overlapping groups
_X4 = stim.CircuitInstruction("X", [0, 1, 2, 3])  # Pauli: bypasses the cocycle machinery
_TAIL = "\nTICK\nM 0 1 2 3 4 5"


def _coset_cpp_missing() -> bool:
    native_dir = Path(coset_kernels.__file__).resolve().parent
    has_so = (native_dir / "libcoset_kernels.so").exists()
    can_compile = (native_dir / "coset_kernels.cpp").exists() and os.access(native_dir, os.W_OK)
    return not (has_so or can_compile)


_SKIP_NO_CPP = pytest.mark.skipif(
    _coset_cpp_missing(), reason="coset C++ kernel library is unavailable"
)


class _KernelSpy:
    """Delegates to the real library; forbidden kernels raise AssertionError, others are counted."""

    def __init__(self, lib: ctypes.CDLL, forbid: tuple[str, ...] = ()) -> None:
        self._lib, self._forbid = lib, forbid
        self.calls = {k: 0 for k in _ALL_MASK_KERNELS}

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if name not in self.calls:
            return fn

        def spied(*args: Any) -> Any:
            if name in self._forbid:
                raise AssertionError(f"{name} was called")
            self.calls[name] += 1
            return fn(*args)

        return spied


def _sim(
    text: str, batch: int, *, use_cpp: bool = True, sync: bool | None = None, forbid: bool = True
) -> tuple[CosetsideSimulator, _KernelSpy | None]:
    circuit = stim.Circuit(text)
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=batch)
    sim = CosetsideSimulator(
        circuit, compiled_op_handler=handler, batch_size=batch, seed=5, sync_tableside_rng=sync
    )
    if not use_cpp:
        sim._use_cpp = False
    if sim._cpp_lib is None:
        return sim, None
    spy = _KernelSpy(sim._cpp_lib, _ALL_MASK_KERNELS if forbid else ())
    sim._cpp_lib = cast(ctypes.CDLL, spy)
    return sim, spy


def _bad_shapes(rows: int, batch: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "too_few_rows": (rows - 1, batch),
        "one_row": (1, batch),
        "too_few_cols": (rows, batch - 1),
        "too_many_cols": (rows, batch + 1),
        "too_many_rows": (rows + 1, batch),
        "one_dim_batch": (batch,),
        "one_dim_flat": (rows * batch,),
        "zero_dim": (),
        "zero_rows": (0, batch),
        "zero_cols": (rows, 0),
        "three_dim": (1, rows, batch),
    }
    return {k: v for k, v in shapes.items() if v != (rows, batch) and min(v, default=1) >= 0}


def _fingerprint(sim: CosetsideSimulator) -> str:
    """Hash of what a rejected call must leave alone: frames, shots, RNG streams, flags, cursor,
    and the compiled reference (number of steps, its live tableau, the packed words at the cursor).
    """
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    shots = sorted((b, st.d, st.h_words.tolist()) for b, st in sim._active_shots.items())
    rng = sim.np_rng.bit_generator.state["state"]
    flags = (sim._cpp_dirty, len(sim._measurement_columns), sim._step_cursor)
    parts = (xs.tobytes(), zs.tobytes(), repr(shots), sim._shot_tiebreak_rng.tobytes(), repr(rng))
    ref = sim._compiled_ref
    words = [w.tobytes() for w in (*ref.ref_chp.snapshot_packed(), *sim._current_packed_words())]
    blob = b"|".join(p if isinstance(p, bytes) else p.encode() for p in (*parts, *words))
    blob += repr((flags, len(ref.steps))).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _bad_mask(shape: tuple[int, ...], fill: bool, dtype: Any = np.bool_) -> np.ndarray:
    return np.full(shape, fill, dtype=dtype)


# (circuit text, op, number of target groups of the step, use_cpp, id)
_DCC_CASES = [
    ("H 0 1 2 3" + _TAIL, _H4, 4, True, "disjoint_kernel"),
    ("H 0 1 2 3" + _TAIL, _H4, 4, False, "disjoint_python"),
    ("CX 0 1 2 3" + _TAIL, _CX_DISJOINT, 2, True, "cx_disjoint_kernel"),
    ("CX 0 1 1 2 2 3" + _TAIL, _CX_OVERLAP, 3, True, "overlapping_python"),
    ("X 0 1 2 3" + _TAIL, _X4, 4, True, "pauli_x"),
]


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("fill", [False, True], ids=["some_failed", "all_met"])
@pytest.mark.parametrize("case", _DCC_CASES, ids=[c[4] for c in _DCC_CASES])
def test_conditional_clifford_rejects_every_wrong_mask_shape_before_the_kernel(
    case: tuple, fill: bool, batch: int
) -> None:
    text, op, groups, use_cpp, _ = case
    for name, shape in _bad_shapes(groups, batch).items():
        sim, spy = _sim(text, batch, use_cpp=use_cpp)
        before = _fingerprint(sim)
        with pytest.raises(ValueError, match=r"condition_met_mask has shape"):
            sim.do_conditional_clifford(op, _bad_mask(shape, fill))
        assert spy is None or sum(spy.calls.values()) == 0, name
        assert _fingerprint(sim) == before, f"state changed by the rejected {name} call"


@_SKIP_NO_CPP
def test_the_asan_repro_a_one_row_mask_for_a_four_group_step_raises() -> None:
    """run2's mini.py: `H 0` with a (1, B) mask against the 4-group step of `H 0 1 2 3`.

    On e4b7049 the engine kernel read rows 1..3 of the 1-row buffer (heap-buffer-overflow)."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, 64)
    with pytest.raises(ValueError, match=r"\(1, 64\), expected \(4, 64\)"):
        sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.zeros((1, 64), bool))
    assert spy is not None and spy.calls[_DCC_KERNEL] == 0


@_SKIP_NO_CPP
def test_a_mask_that_fits_the_op_but_not_its_compiled_step_raises() -> None:
    """After `clear()` rewinds the step cursor a caller can replay an op against the wrong step;
    the mask then has one row per op group but fewer than the step's groups."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, 8)
    op = stim.CircuitInstruction("H", [0, 1])  # 2 groups, the step has 4
    with pytest.raises(ValueError, match=r"\(2, 8\), expected \(4, 8\)"):
        sim.do_conditional_clifford(op, np.zeros((2, 8), bool))
    assert spy is not None and spy.calls[_DCC_KERNEL] == 0


@_SKIP_NO_CPP
@pytest.mark.parametrize("batch", [1, 64])
def test_a_step_with_no_target_groups_takes_a_zero_row_mask(batch: int) -> None:
    """Past the circuit's end `do_conditional_clifford` compiles a new step: `H` with no targets."""
    empty = stim.CircuitInstruction("H", [])
    sim, spy = _sim("H 0\nTICK\nM 0", batch)
    sim.run()
    sim.do_conditional_clifford(empty, np.zeros((0, batch), bool))
    assert spy is not None and spy.calls[_DCC_KERNEL] == 0
    sim, _ = _sim("H 0\nTICK\nM 0", batch)
    sim.run()
    with pytest.raises(ValueError, match=r"expected \(0, "):
        sim.do_conditional_clifford(empty, np.zeros((1, batch), bool))


@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("case", _DCC_CASES, ids=[c[4] for c in _DCC_CASES])
def test_a_rejected_conditional_clifford_can_be_retried_with_a_corrected_mask(
    case: tuple, batch: int
) -> None:
    text, op, groups, use_cpp, _ = case
    good = np.random.default_rng(3).random((groups, batch)) < 0.5
    ref, _ = _sim(text, batch, use_cpp=use_cpp, forbid=False)
    ref.do_conditional_clifford(op, good)
    sim, spy = _sim(text, batch, use_cpp=use_cpp)
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(op, good[:-1])
    if spy is not None:
        spy._forbid = ()
    sim.do_conditional_clifford(op, good)  # same step as the rejected call, not the next one
    assert sim._step_cursor == ref._step_cursor == 1
    assert _fingerprint(sim) == _fingerprint(ref)


_MIXED = "X_ERROR(1) 0\nM 0\nCX rec[-1] 1 2 3\nTICK\nM 1 2 3"  # rec-controlled + quantum group


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("shape", [(1, 8), (3, 8), (2, 7), (16,)], ids=str)
def test_a_wrong_mask_for_a_mixed_record_controlled_op_is_rejected_before_any_flip(
    shape: tuple[int, ...], use_cpp: bool
) -> None:
    """The rec-controlled group flips frames in a loop over the op's groups; that must not run
    on a mask that is rejected (on either path: only the kernel path reads the mask later)."""
    mixed = stim.Circuit(_MIXED)
    sim, spy = _sim(_MIXED, 8, use_cpp=use_cpp)
    sim.interactive_do(mixed[0])
    sim.interactive_do(mixed[1])
    before = _fingerprint(sim)
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(mixed[2], np.ones(shape, bool))
    assert _fingerprint(sim) == before
    assert spy is None or sum(spy.calls.values()) == 0
    if spy is not None:
        spy._forbid = ()
    sim.do_conditional_clifford(mixed[2], np.ones((2, 8), bool))  # retry replays the CX step
    assert sim._step_cursor == 3


_MIXED_OP = stim.Circuit("CX rec[-1] 1 2 3")[0]  # a record-controlled group and a quantum one
_BEYOND = "H 0 1 2 3" + _TAIL  # its end is where the ops below run (rec[-1] exists there)
# (op, rows of a valid mask, use_cpp, id): ops that are NOT in the circuit, so each call compiles a
# new reference step
_PAST_END_CASES = [
    (_H4, 4, True, "disjoint_kernel"), (_H4, 4, False, "disjoint_python"),
    (_CX_DISJOINT, 2, True, "cx_disjoint"), (_CX_OVERLAP, 3, True, "overlapping"),
    (_X4, 4, True, "pauli_x"), (_MIXED_OP, 2, True, "mixed_kernel"),
    (_MIXED_OP, 2, False, "mixed_python"),
]


@pytest.mark.parametrize("batch", [1, 7, 65])
@pytest.mark.parametrize("case", _PAST_END_CASES, ids=[c[3] for c in _PAST_END_CASES])
def test_a_rejected_call_past_the_end_of_the_reference_changes_nothing(
    case: tuple, batch: int
) -> None:
    """Past the circuit's end `do_conditional_clifford` compiles and appends a reference step,
    which cannot be undone: the mask must be rejected BEFORE that (frames, shots, RNG streams,
    cursor, number of steps and the packed words are all in the fingerprint)."""
    op, groups, use_cpp, _ = case
    for name, shape in _bad_shapes(groups, batch).items():
        sim, spy = _sim(_BEYOND, batch, use_cpp=use_cpp)
        sim.run()
        before = _fingerprint(sim)
        steps = len(sim._compiled_ref.steps)
        with pytest.raises(ValueError, match="condition_met_mask"):
            sim.do_conditional_clifford(op, _bad_mask(shape, True))
        assert len(sim._compiled_ref.steps) == steps == sim._step_cursor, name
        assert _fingerprint(sim) == before, f"state changed by the rejected {name} call"
        assert spy is None or sum(spy.calls.values()) == 0, name


@pytest.mark.parametrize("batch", [1, 7, 65])
@pytest.mark.parametrize("case", _PAST_END_CASES, ids=[c[3] for c in _PAST_END_CASES])
def test_a_rejected_call_past_the_end_can_be_retried_and_acts_once(case: tuple, batch: int) -> None:
    """Reject + retry must equal the clean call: before the fix the rejected call left its step
    on the reference, so the retry ran the op on the reference twice (X then M read 0, not 1)."""
    op, groups, use_cpp, _ = case
    good = np.random.default_rng(4).random((groups, batch)) < 0.5
    ref, _ = _sim(_BEYOND, batch, use_cpp=use_cpp, forbid=False)
    ref.run()
    ref.do_conditional_clifford(op, good)
    sim, spy = _sim(_BEYOND, batch, use_cpp=use_cpp)
    sim.run()
    with pytest.raises(ValueError, match="condition_met_mask"):
        sim.do_conditional_clifford(op, good[:-1])
    if spy is not None:
        spy._forbid = ()
    sim.do_conditional_clifford(op, good)
    assert _fingerprint(sim) == _fingerprint(ref)
    qubits: list[int] = list(range(_N))
    for pauli in ("X", "Y", "Z"):
        peek = (sim.peek_pauli_batch(qubits, pauli), ref.peek_pauli_batch(qubits, pauli))
        assert np.array_equal(*peek), pauli


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
def test_x_then_measure_reads_one_after_a_rejected_and_retried_x(use_cpp: bool) -> None:
    """The witness: X on |0>, then M. The rejected X used to be applied to the reference twice."""
    results = []
    for rejected_first in (False, True):
        circuit = stim.Circuit("R 0\nTICK")
        handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=4)
        sim = CosetsideSimulator(circuit, compiled_op_handler=handler, batch_size=4, seed=5)
        sim._use_cpp = use_cpp
        for inst in circuit:
            sim.interactive_do(inst)
        x = stim.CircuitInstruction("X", [0])
        if rejected_first:
            with pytest.raises(ValueError, match="condition_met_mask has shape"):
                sim.do_conditional_clifford(x, np.ones((2, 4), bool))
        sim.do_conditional_clifford(x, np.ones((1, 4), bool))
        sim._circuit_time += 1
        sim.interactive_do(stim.CircuitInstruction("M", [0]))
        results.append(np.array(sim._measurement_columns).astype(int).tolist())
    assert results[0] == [[1, 1, 1, 1]] == results[1]


def test_a_mask_for_a_mismatched_second_check_is_rejected_before_the_record_loop_flips() -> None:
    """The mixed op's record-controlled group is applied by a loop over the op, before the kernel
    reads a mask restricted to its record-free groups: the cursor's step `H 1 2` needs two rows,
    the op `CX rec[-1] 1 rec[-1] 2` has none, so the second shape check rejects it. It runs
    first: no frame may be flipped by a call that raises."""
    sim, _ = _sim("M 0\nH 1 2\nM 1 2\nTICK\nM 3 4 5", 64)
    sim.record_direct_measurements(np.ones((1, 64), bool))
    sim.do_conditional_clifford(stim.CircuitInstruction("M", [0]), None)  # cursor: step 1
    before = _fingerprint(sim)
    op = stim.Circuit("CX rec[-1] 1 rec[-1] 2")[0]
    expected = r"record-free groups has shape \(0, 64\), expected \(2, 64\)"
    with pytest.raises(ValueError, match=expected):
        sim.do_conditional_clifford(op, np.ones((2, 64), bool))
    assert _fingerprint(sim) == before
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    assert not xs.any() and not zs.any() and sim._step_cursor == 1


@pytest.mark.parametrize("op_text", ["H 0 1", "H 0", "S 0", "S 1 0", "CX 0 1"], ids=str)
def test_an_op_that_does_not_replay_the_step_is_reported_as_that_not_as_a_mask_error(
    op_text: str
) -> None:
    """interactive_do's mismatch error precedes the mask check, as it did before the guards. The
    step at the cursor is `S 0 1`: an op that does not replay it (another name, or the name with
    other targets, or neither) is rejected as that, whatever the mask's shape, and nothing moves;
    the op that does replay it is then judged by its mask."""
    sim, _ = _sim("H 0 1\nS 0 1\nTICK\nM 0 1 2 3 4 5", 4, forbid=False)
    sim.interactive_do(stim.CircuitInstruction("H", [0, 1]))
    before = _fingerprint(sim)
    with pytest.raises(ValueError, match="doesn't match"):
        sim.do_conditional_clifford(stim.Circuit(op_text)[0], np.zeros((5, 4), bool))
    assert _fingerprint(sim) == before
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(stim.Circuit("S 0 1")[0], np.zeros((5, 4), bool))
    assert _fingerprint(sim) == before
    sim.do_conditional_clifford(stim.Circuit("S 0 1")[0], np.zeros((2, 4), bool))  # replays
    assert sim._step_cursor == 2


@pytest.mark.parametrize("op_text", [
    "H 0 1 2", "I 0 1", "X 0", "CX 0 1 2 3", "II 0 1", "SWAP 0 1 2 3 4 5", "CX rec[-1] 1 2 3",
    "CX rec[-1] 1 rec[-2] 2", "XCZ 1 rec[-1] 2 3", "CZ sweep[0] 1 2 3", "SQRT_XX 0 1",
    "M 0 1", "R 0", "TICK", "DEPOLARIZE1(0.1) 0 1", "MPP X0*X1", "SPP X0*X1", "H",
])
def test_the_predicted_rows_of_a_past_the_end_step_are_those_it_compiles_to(op_text: str) -> None:
    """`compile_instruction` compiles without mutating the reference and matches
    `compile_and_append_instruction` on target groups and exceptions."""
    op = stim.Circuit(op_text)[0]
    ref = CompiledReferenceCircuit(stim.Circuit("R 0 1 2 3 4 5"))
    ref2 = CompiledReferenceCircuit(stim.Circuit("R 0 1 2 3 4 5"))
    steps_before = len(ref.steps)
    snap_before = [w.copy() for w in ref.ref_chp.snapshot_packed()]
    try:
        step, _, _ = ref.compile_instruction(op)
    except NotImplementedError:
        assert len(ref.steps) == steps_before
        assert all(np.array_equal(a, b) for a, b in zip(ref.ref_chp.snapshot_packed(), snap_before))
        with np.testing.assert_raises(NotImplementedError):
            ref2.compile_and_append_instruction(op)
        return
    assert len(ref.steps) == steps_before
    assert all(np.array_equal(a, b) for a, b in zip(ref.ref_chp.snapshot_packed(), snap_before))
    step2 = ref2.compile_and_append_instruction(op)
    pred = len(step.target_groups) if isinstance(step, UnitaryStepMeta) else None
    expect = len(step2.target_groups) if isinstance(step2, UnitaryStepMeta) else None
    assert pred == expect


class _OnceThenOther:
    """Not an ndarray: `__array__` gives `first` once, then `later` (same shape, other values).

    Two conversions of this object disagree, so a method that checked one and handed the other to
    the kernel would act on a mask it never validated (with a smaller `later` it would be an
    out-of-bounds read; the same shape keeps this test safe)."""

    def __init__(self, first: np.ndarray, later: np.ndarray) -> None:
        self.first, self.later, self.calls = first, later, 0

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        self.calls += 1
        return self.first if self.calls == 1 else self.later


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
def test_a_condition_met_mask_is_converted_once_and_the_checked_array_is_the_one_used(
    use_cpp: bool
) -> None:
    good = np.random.default_rng(6).random((4, 7)) < 0.5
    ref, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    ref.do_conditional_clifford(_H4, good)
    sim, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    hostile = _OnceThenOther(good, np.ones((4, 7), bool))
    sim.do_conditional_clifford(_H4, hostile)  # type: ignore[arg-type]
    assert hostile.calls == 1
    assert _fingerprint(sim) == _fingerprint(ref)


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("pauli", ["Z", "X", "Y"])
def test_reset_masks_are_converted_once_and_the_checked_arrays_are_the_ones_used(
    use_cpp: bool, pauli: str
) -> None:
    rng = np.random.default_rng(7)
    r0, r1 = rng.random((_N, 7)) < 0.3, rng.random((_N, 7)) < 0.3
    ref, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    ref.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)  # type: ignore[arg-type]
    sim, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    h0 = _OnceThenOther(r0, np.zeros((_N, 7), bool))
    h1 = _OnceThenOther(r1, np.zeros((_N, 7), bool))
    sim.do_unmatched_noisy_reset_z(h0, h1, pauli=pauli)  # type: ignore[arg-type]
    assert (h0.calls, h1.calls) == (1, 1)
    assert _fingerprint(sim) == _fingerprint(ref)


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
def test_a_well_formed_ndarray_subclass_mask_acts_like_its_plain_array_and_hooks_never_run(
    use_cpp: bool
) -> None:
    """A subclass (`np.ma`, `np.matrix`, a user class) is read as the plain buffer it wraps: the
    method's one conversion drops the subclass, so its `__getitem__` is never called."""
    seen: list[Any] = []

    class _Rows(np.ndarray):
        def __getitem__(self, item: Any) -> Any:
            seen.append(item)
            return super().__getitem__(item)

    plain = np.random.default_rng(9).random((3, 7)) < 0.5
    ref, _ = _sim("CX 0 1 1 2 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    ref.do_conditional_clifford(_CX_OVERLAP, plain)
    sim, _ = _sim("CX 0 1 1 2 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    sim.do_conditional_clifford(_CX_OVERLAP, plain.view(_Rows))
    assert seen == []
    assert _fingerprint(sim) == _fingerprint(ref)


def _liar(real_shape: tuple[int, ...], lie: tuple[int, ...], fill: bool) -> np.ndarray:
    """An ndarray subclass whose `.shape` reports `lie` over a buffer of `real_shape`."""

    class _Liar(np.ndarray):
        @property
        def shape(self) -> tuple[int, ...]:  # type: ignore[override]
            return lie

        @shape.setter
        def shape(self, value: Any) -> None:
            np.ndarray.shape.__set__(self, value)  # type: ignore[attr-defined]

    return np.full(real_shape, fill, bool).view(_Liar)


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("fill", [False, True], ids=["some_failed", "all_met"])
def test_a_subclass_that_lies_about_its_shape_cannot_get_a_short_buffer_to_the_dcc_kernel(
    use_cpp: bool, fill: bool
) -> None:
    """ASan (round-2 code, `np.asanyarray`): a (1, 8)-byte buffer reporting shape (4, 8) passed
    the shape check and `engine_batch_inject_general_clifford` read 32 bytes of it."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, 8, use_cpp=use_cpp)
    liar = _liar((1, 8), (4, 8), fill)
    before = _fingerprint(sim)
    with pytest.raises(ValueError, match=r"mask has shape \(1, 8\), expected \(4, 8\)"):
        sim.do_conditional_clifford(_H4, liar)
    assert spy is None or sum(spy.calls.values()) == 0
    assert _fingerprint(sim) == before


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("which", ["zero", "one"])
@pytest.mark.parametrize("pauli", ["Z", "X"])
def test_a_subclass_that_lies_about_its_shape_cannot_get_a_short_buffer_to_the_reset_kernel(
    use_cpp: bool, which: str, pauli: str
) -> None:
    """ASan (round-2 code): a (1, 8) buffer reporting (6, 8) as `reset_zero_mask` /
    `reset_one_mask` was read as 6 x 8 bytes by `engine_batch_unmatched_noisy_reset_z`."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, 8, use_cpp=use_cpp)
    liar, good = _liar((1, 8), (_N, 8), True), np.zeros((_N, 8), bool)
    r0, r1 = (liar, good) if which == "zero" else (good, liar)
    before = _fingerprint(sim)
    with pytest.raises(ValueError, match=rf"reset_{which}_mask has shape \(1, 8\)"):
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)  # type: ignore[arg-type]
    assert spy is None or sum(spy.calls.values()) == 0
    assert _fingerprint(sim) == before


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
def test_a_subclass_with_a_lying_getitem_cannot_change_the_mask_a_mixed_op_hands_on(
    use_cpp: bool
) -> None:
    """ASan: for a record-controlled op the mask is sliced to its record-free rows
    (`mask[quant_indices, :]`) after the shape check; a subclass whose `__getitem__` answered with
    a smaller array sent the kernel a short buffer. The slice must come from the plain array."""
    calls: list[Any] = []

    class _GetItemLiar(np.ndarray):
        def __getitem__(self, item: Any) -> Any:
            calls.append(item)
            return np.zeros((1, 8), bool)  # right shape for the kernel, wrong content

    plain = np.zeros((2, 8), bool)
    plain[:, ::2] = True
    mixed = stim.Circuit(_MIXED)
    ref, _ = _sim(_MIXED, 8, use_cpp=use_cpp, forbid=False)
    sim, _ = _sim(_MIXED, 8, use_cpp=use_cpp, forbid=False)
    for s in (ref, sim):
        s.interactive_do(mixed[0])
        s.interactive_do(mixed[1])
    ref.do_conditional_clifford(mixed[2], plain)
    sim.do_conditional_clifford(mixed[2], plain.view(_GetItemLiar))
    assert calls == []
    assert _fingerprint(sim) == _fingerprint(ref)


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("shape", [(1, 8), (3, 8), (2, 7), (16,)], ids=str)
def test_a_wrong_mask_for_an_op_whose_groups_all_read_the_record_is_rejected_atomically(
    shape: tuple[int, ...], use_cpp: bool
) -> None:
    """`CX rec[-1] 2 rec[-2] 3` has no record-free group, so its reference step is a no-op with no
    rows to compare; the record loop still reads one mask row per group. A short mask must be
    rejected before that loop flips frame bits and before the step cursor moves."""
    text = "X_ERROR(1) 1\nM 0 1\nCX rec[-1] 2 rec[-2] 3\nM 2 3"
    circuit = stim.Circuit(text)
    sim, _ = _sim(text, 8, use_cpp=use_cpp)
    sim.interactive_do(circuit[0])
    sim.interactive_do(circuit[1])
    before = _fingerprint(sim)
    cursor = sim._step_cursor
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(circuit[2], np.ones(shape, bool))
    assert _fingerprint(sim) == before
    assert sim._step_cursor == cursor
    sim.do_conditional_clifford(circuit[2], np.ones((2, 8), bool))  # a corrected retry acts once
    assert sim._step_cursor == cursor + 1


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
def test_a_list_mask_is_checked_like_the_array_it_converts_to(use_cpp: bool) -> None:
    """A nested list has no `.shape`: it is converted, checked, and the kernel gets the result."""
    good = np.random.default_rng(8).random((4, 7)) < 0.5
    ref, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp, forbid=False)
    ref.do_conditional_clifford(_H4, good)
    sim, _ = _sim("H 0 1 2 3" + _TAIL, 7, use_cpp=use_cpp)
    for bad in (good[:-1].tolist(), good.T.tolist(), good[0].tolist(), [[True] * 7] * 3):
        with pytest.raises(ValueError, match="condition_met_mask has shape"):
            sim.do_conditional_clifford(_H4, bad)  # type: ignore[arg-type]
    sim._cpp_lib = ref._cpp_lib if use_cpp else None
    sim.do_conditional_clifford(_H4, good.tolist())  # type: ignore[arg-type]
    assert _fingerprint(sim) == _fingerprint(ref)
    r0, r1 = np.zeros((_N, 7), bool), np.zeros((_N, 7), bool)
    r1[2] = True
    ref.do_unmatched_noisy_reset_z(r0, r1)
    sim.do_unmatched_noisy_reset_z(r0.tolist(), r1.tolist())  # type: ignore[arg-type]
    assert _fingerprint(sim) == _fingerprint(ref)
    with pytest.raises(ValueError, match="reset_one_mask has shape"):
        sim.do_unmatched_noisy_reset_z(r0.tolist(), r1[:-1].tolist())  # type: ignore[arg-type]


def test_condition_group_fn_ignores_the_mask_of_a_step_and_leaves_state_to_the_function() -> None:
    """With `condition_group_fn` the mask is not read for a plain step, so its shape is not
    checked: the call equals the one given no mask at all (here a one-row mask for 4 groups)."""
    calls = []

    def group_fn(g: int) -> np.ndarray:
        calls.append(g)
        return np.arange(7) % 2 == g % 2

    fps = []
    for mask in (None, np.zeros((1, 7), bool), np.ones((9, 3), bool)):
        sim, _ = _sim("H 0 1 2 3" + _TAIL, 7, forbid=False)
        masked = cast(Any, mask)
        sim.do_conditional_clifford(_H4, masked, condition_group_fn=group_fn)
        fps.append(_fingerprint(sim))
    assert fps[0] == fps[1] == fps[2] and calls == [0, 1, 2, 3] * 3


def test_condition_group_fn_still_needs_a_mask_that_fits_a_record_controlled_loop() -> None:
    """A mixed op's record-controlled groups read `mask[g]` whatever `condition_group_fn` is."""
    mixed = stim.Circuit(_MIXED)
    sim, _ = _sim(_MIXED, 8)
    sim.interactive_do(mixed[0])
    sim.interactive_do(mixed[1])
    before = _fingerprint(sim)
    with pytest.raises(ValueError, match="condition_met_mask has shape"):
        sim.do_conditional_clifford(
            mixed[2], np.ones((1, 8), bool), condition_group_fn=lambda g: np.ones(8, bool)
        )
    assert _fingerprint(sim) == before


@pytest.mark.parametrize("in_circuit", [True, False], ids=["replayed_step", "past_the_end"])
@pytest.mark.parametrize("op_text", ["M 0", "R 0", "TICK", "MPAD 0", "X_ERROR(0.1) 0"])
def test_a_mask_for_a_step_that_is_not_unitary_is_not_read_so_not_checked(
    op_text: str, in_circuit: bool
) -> None:
    """Measurements, resets and noise reach `do_conditional_clifford` too and return before the
    mask is used (as before the guards): a mask of any shape must not be rejected for them."""
    op = stim.Circuit(op_text)[0]
    sim, spy = _sim(op_text + "\nTICK\nM 0 1 2 3 4 5" if in_circuit else "H 0\nM 1", 4)
    if not in_circuit:
        sim.run()
    cursor = sim._step_cursor
    sim.do_conditional_clifford(op, np.zeros((9, 3), bool))
    assert sim._step_cursor == cursor + 1
    assert spy is None or sum(spy.calls.values()) == 0


@pytest.mark.parametrize("annotation", ["DETECTOR rec[-1]", "OBSERVABLE_INCLUDE(0) rec[-1]"])
def test_an_annotation_with_a_record_target_is_not_a_mixed_gate_and_its_mask_is_not_checked(
    annotation: str
) -> None:
    """Only a TWO-QUBIT gate with a record target is record-controlled (`CX rec[-1] 1`): other
    ops with record targets must not get the mixed-op row check (the mask is not read)."""
    text = f"M 0\n{annotation}\nM 1 2 3"
    circuit = stim.Circuit(text)
    sim, spy = _sim(text, 4)
    sim._do_bare_instruction(circuit[0])
    sim.do_conditional_clifford(circuit[1], np.zeros((9, 3), bool))
    assert spy is None or sum(spy.calls.values()) == 0


def test_the_check_names_the_step_and_both_shapes() -> None:
    sim, _ = _sim("CX 0 1 2 3" + _TAIL, 5)
    with pytest.raises(ValueError) as err:
        sim.do_conditional_clifford(_CX_DISJOINT, np.zeros((2, 4), bool))
    msg = str(err.value)
    assert "condition_met_mask has shape (2, 4), expected (2, 5)" in msg
    assert "CX" in msg and "batch_size" in msg


_RESET_CASES = [("Z", True, None), ("Z", False, None), ("X", True, None), ("Y", True, None),
                ("Z", True, True)]  # (pauli, use_cpp, sync_tableside_rng)


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("which", ["zero", "one", "both"])
@pytest.mark.parametrize("fill", [False, True], ids=["all_clear", "some_set"])
@pytest.mark.parametrize("case", _RESET_CASES, ids=lambda c: f"{c[0]}-cpp{c[1]}-sync{c[2]}")
def test_unmatched_reset_rejects_every_wrong_mask_shape_before_the_kernel(
    case: tuple, fill: bool, which: str, batch: int
) -> None:
    pauli, use_cpp, sync = case
    for name, shape in _bad_shapes(_N, batch).items():
        sim, spy = _sim("H 0 1 2 3" + _TAIL, batch, use_cpp=use_cpp, sync=sync)
        good = np.zeros((_N, batch), bool)
        good[1, ::2] = fill
        bad = _bad_mask(shape, fill)
        r0, r1 = (bad, good) if which == "zero" else (good, bad)
        if which == "both":
            r0 = bad
        before = _fingerprint(sim)
        with pytest.raises(ValueError, match=r"reset_(zero|one)_mask has shape"):
            sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)
        assert spy is None or sum(spy.calls.values()) == 0, name
        assert _fingerprint(sim) == before, f"state changed by the rejected {name} call"


@_SKIP_NO_CPP
def test_the_unmatched_reset_repro_a_short_mask_raises_instead_of_reading_past_it() -> None:
    sim, spy = _sim("H 0 1 2 3" + _TAIL, 64)
    r1 = np.zeros((1, 64), bool)
    r1[0] = True  # the kernel would read rows 1..5 past a 64-byte buffer
    with pytest.raises(ValueError, match=r"\(1, 64\), expected \(6, 64\)"):
        sim.do_unmatched_noisy_reset_z(np.zeros((_N, 64), bool), r1)
    assert spy is not None and spy.calls[_RESET_KERNEL] == 0


def test_a_wrong_shaped_reset_mask_is_rejected_even_when_nothing_is_set() -> None:
    """The check precedes the all-False early return: shape errors do not depend on the values."""
    sim, _ = _sim("H 0 1 2 3" + _TAIL, 8)
    with pytest.raises(ValueError, match="reset_zero_mask has shape"):
        sim.do_unmatched_noisy_reset_z(np.zeros((1, 8), bool), np.zeros((_N, 8), bool))


def _scenario(sim: CosetsideSimulator, batch: int) -> str:
    """Valid calls only (what the engine handlers pass): masks of exactly the expected shape."""
    rng = np.random.default_rng(99)
    tick = stim.CircuitInstruction("TICK", [])
    masks = rng.random((4, batch)) < 0.5
    sim.do_conditional_clifford(_H4, masks.copy())
    sim._do_bare_instruction(tick)
    sim.do_conditional_clifford(_CX_DISJOINT, rng.random((2, batch)) < 0.5)
    sim._do_bare_instruction(tick)
    for pauli in ("Z", "X", "Y", "Z"):
        r1 = np.zeros((_N, batch), bool)
        r1[1] = rng.random(batch) < 0.5
        r1[2, ::2] = True
        r0 = np.zeros_like(r1)
        r0[3] = rng.random(batch) < 0.5
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)  # type: ignore[arg-type]
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None)
    sim._do_bare_instruction(tick)
    sim.do_conditional_clifford(_H4, np.ones((4, batch), bool))
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    shots = sorted((b, st.d, st.h_words.tolist()) for b, st in sim.active_shots.items())
    peeks = np.stack([sim.peek_pauli_batch(list(range(_N)), p) for p in ("X", "Y", "Z")])
    parts = (xs.tobytes(), zs.tobytes(), repr(shots).encode(), peeks.tobytes())
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


# (batch, use_cpp) -> digest of `_scenario`, taken on the unguarded e4b7049 code
_VALID_DIGESTS: dict[tuple[int, bool], str] = {
    (1, True): "7130c3b1b94e19bc", (1, False): "7130c3b1b94e19bc",
    (8, True): "e21448070ee6afea", (8, False): "e21448070ee6afea",
    (65, True): "ea71fe21d9e3df53", (65, False): "ea71fe21d9e3df53",
}


@pytest.mark.parametrize(
    "use_cpp", [pytest.param(True, marks=_SKIP_NO_CPP), False], ids=["kernel", "python"]
)
@pytest.mark.parametrize("batch", [1, 8, 65])
def test_correctly_shaped_masks_give_the_results_of_the_unguarded_code(
    batch: int, use_cpp: bool
) -> None:
    text = "H 0 1 2 3\nTICK\nCX 0 1 2 3\nTICK\nH 0\nTICK\nH 0 1 2 3"  # steps in call order
    sim, spy = _sim(text + _TAIL, batch, use_cpp=use_cpp, forbid=False)
    digest = _scenario(sim, batch)
    if use_cpp:
        assert (
            spy is not None
            and spy.calls[_DCC_KERNEL] >= 1
            and spy.calls[_RESET_KERNEL] >= 1
            and spy.calls[_PROJECT_KERNEL] >= 1
        )
    assert digest == _VALID_DIGESTS[(batch, use_cpp)]


def _mixed_state(sim: CosetsideSimulator) -> tuple[bytes, bytes, list[Any], bytes]:
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    shots = sorted((b, st.d, st.h_words.tolist()) for b, st in sim._active_shots.items())
    peeks = np.stack([sim.peek_pauli_batch(list(range(_N)), p) for p in ("X", "Y", "Z")])
    return xs.tobytes(), zs.tobytes(), shots, peeks.tobytes()


@pytest.mark.parametrize("use_cpp", [True, False], ids=["kernel", "python"])
@pytest.mark.parametrize("fail_row", [0, 1, 2], ids=["row0", "row1", "row2"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["tail_q", "split_q"])
@pytest.mark.parametrize("gate", ["CX", "CY"])
def test_mixed_op_with_multiple_record_free_groups_uses_every_quantum_mask_row(
    gate: str, interleaved: bool, fail_row: int, use_cpp: bool
) -> None:
    """R3-F10: a mixed 2Q op with >=2 record-free groups must slice all `quant_indices` rows."""
    op_str = f"{gate} 0 1 rec[-1] 2 3 4" if interleaved else f"{gate} rec[-1] 1 2 3 4 5"
    split_ops = (
        f"{gate} 0 1\nTICK\n{gate} rec[-1] 2\nTICK\n{gate} 3 4"
        if interleaved
        else f"{gate} rec[-1] 1\nTICK\n{gate} 2 3\nTICK\n{gate} 4 5"
    )
    prep = "H 0 3" if interleaved else "H 2 4"
    head = f"{prep}\nX_ERROR(0.5) 0 1 2 3 4 5\nM 0\n"
    mask = np.ones((3, 8), bool)
    mask[fail_row, :4] = False

    sim, spy = _sim(f"{head}{op_str}\nTICK\nM 1 2 3 4 5", 8, use_cpp=use_cpp, forbid=False)
    ref, _ = _sim(f"{head}{split_ops}\nTICK\nM 1 2 3 4 5", 8, use_cpp=use_cpp, forbid=False)
    all_t, _ = _sim(f"{head}{op_str}\nTICK\nM 1 2 3 4 5", 8, use_cpp=use_cpp, forbid=False)
    for s in (sim, ref, all_t):
        for idx in range(3):
            s.interactive_do(s.circuit[idx])
    sim.do_conditional_clifford(sim.circuit[3], mask)
    all_t.do_conditional_clifford(all_t.circuit[3], np.ones((3, 8), bool))
    for idx, row in ((3, mask[0:1]), (5, mask[1:2]), (7, mask[2:3])):
        if idx > 3:
            ref.do_conditional_clifford(ref.circuit[idx - 1], None)
        ref.do_conditional_clifford(ref.circuit[idx], row)
    assert _mixed_state(sim) == _mixed_state(ref)
    assert _mixed_state(sim) != _mixed_state(all_t)
    if use_cpp and spy is not None:
        rec_row = 1 if interleaved else 0
        assert spy.calls[_DCC_KERNEL] == (0 if fail_row == rec_row else 1)
