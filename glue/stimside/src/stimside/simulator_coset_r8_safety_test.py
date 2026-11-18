"""Run 8 Stage B1 safety & correctness tests (R7-F1, D-36, R7-F4, R7-F3/F3b, R7-F5, R7-F6)."""

from __future__ import annotations

import abc
import ctypes
import gc
import inspect
import sys
import time
import weakref
from typing import Any, cast

import numpy as np
import pytest
import stim

import stimside.simulator_coset as sc
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator, _FailedCosetsideSimulator


def _mk_sim(
    circ_text: str,
    *,
    batch_size: int = 4,
    seed: int = 7,
    use_cpp: bool = True,
    sync: bool = False,
    record_u2l: bool = False,
    record_events: bool = False,
) -> CosetsideSimulator:
    c = stim.Circuit(circ_text)
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=batch_size)
    sim = CosetsideSimulator(
        c,
        compiled_op_handler=h,
        batch_size=batch_size,
        seed=seed,
        sync_tableside_rng=sync,
        record_unleaked_to_leaked=record_u2l,
        record_leakage_events=record_events,
    )
    if not use_cpp and not sync:
        sim._use_cpp = False
    return sim


def _stateful_calls(sim: CosetsideSimulator, *, include_clear: bool) -> list[Any]:
    n_q, bs = sim.num_qubits, sim.batch_size
    calls: list[Any] = [
        lambda: sim.active_shots,
        lambda: sim.unleaked_to_leaked_records,
        lambda: sim.run(),
        lambda: sim.interactive_do(stim.CircuitInstruction("TICK", [])),
        lambda: sim.finish_interactive_run(),
        lambda: sim.sync_rngs(),
        lambda: sim.measurement_column(0),
        lambda: sim.overwrite_measurement_column(0, np.zeros(bs, dtype=bool)),
        lambda: sim.get_current_circuit_time(),
        lambda: sim.do_copies_on_fused_step(
            stim.CircuitInstruction("TICK", []), [], lambda _: None
        ),
        lambda: sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None),
        lambda: sim.apply_pauli("X", np.zeros((n_q, bs), dtype=bool)),
        lambda: sim.do_conditional_pauli_noise(stim.CircuitInstruction("X_ERROR", [0], [0.1])),
        lambda: sim.do_unmatched_noisy_reset_z(
            np.zeros((n_q, bs), dtype=bool), np.zeros((n_q, bs), dtype=bool)
        ),
        lambda: sim.postselect_z_batch(
            0, np.zeros(bs, dtype=bool), np.zeros(bs, dtype=np.uint8)
        ),
        lambda: sim.record_direct_measurements(np.zeros((1, bs), dtype=bool)),
        lambda: sim.sample_pauli_noise_sync("X_ERROR", [[0]] * bs, [0.1]),
        lambda: sim.peek_pauli_batch([0], "Z"),
        lambda: sim.peek_pauli_born_batch([0], "Z"),
        lambda: sim.peek_z(0),
        lambda: sim.peek_x(0),
        lambda: sim.peek_y(0),
        lambda: sim.get_current_noisy_tableau_pauli_state([0], "Z"),
        lambda: sim.current_tableau_measurement_record(),
        lambda: sim.get_final_measurement_records(),
        lambda: sim.get_detector_flips(),
        lambda: sim.get_observable_flips(),
        lambda: sim.get_unleaked_to_leaked_records(),
        lambda: sim.get_unleaked_to_leaked_op_indices(),
        lambda: sim.get_leakage_events(),
        lambda: sim.record_u2l_counts(np.zeros(bs, dtype=np.int64), 0),
        lambda: sim.record_leakage_event(0, 0, 0, 2, 0),
        lambda: sim._record_unleaked_to_leaked_counts(np.zeros(bs, dtype=np.int64), 0),
        lambda: sim._record_leakage_event(0, 0, 0, 2, 0),
        lambda: sim._do_bare_instruction(stim.CircuitInstruction("TICK", [])),
        lambda: sim._next_step_meta(stim.CircuitInstruction("TICK", [])),
    ]
    if include_clear:
        calls.insert(0, lambda: sim.clear())
    return calls


# ---------------------------------------------------------------------------
# B1-1 (R7-F1): Shifted tape after interrupted first interactive_do
# ---------------------------------------------------------------------------


def test_r7_f1_interrupted_first_interactive_do_refuses_run_and_clears_cleanly() -> None:
    sim = _mk_sim("X 0\nM 0\nX 0\nM 0", batch_size=3, seed=7)
    fired = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code.co_name == "_dcc_build_state" and not fired[0]:
            fired[0] = True
            raise KeyboardInterrupt("mid-first-op")

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt, match="mid-first-op"):
            sim.interactive_do(sim.circuit[0])
    finally:
        sys.settrace(None)

    assert fired[0]
    assert sim._used_interactively is True
    assert sim._step_cursor == 1
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.run()
    sim.clear()
    sim.run()
    assert sim.get_final_measurement_records().astype(int).tolist() == [[1, 0]] * 3


def test_r7_f1_past_end_tape_growth_interrupt_latches_and_clears() -> None:
    sim = _mk_sim("X 0\nM 0", batch_size=2, seed=3)
    sim.interactive_do(sim.circuit)
    fired = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> Any:
        if frame.f_code.co_name == "append_compiled_step":
            if event == "call":
                return tracer
            if event == "return" and not fired[0]:
                fired[0] = True
                raise KeyboardInterrupt("after-tape-grow")
        return None

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt, match="after-tape-grow"):
            sim.interactive_do(stim.CircuitInstruction("X", [0]))
    finally:
        sys.settrace(None)

    assert fired[0]
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.peek_z(0)
    sim.clear()
    sim.run()
    assert sim.get_final_measurement_records().astype(int).tolist() == [[1], [1]]


# ---------------------------------------------------------------------------
# B1-2 (D-36 + R7-F4): Class-swap aborted/failed latch + user subclasses
# ---------------------------------------------------------------------------


def test_d36_aborted_call_blocks_all_stateful_and_seam_members_until_clear() -> None:
    sim = _mk_sim(
        "H 0\nM 0\nDETECTOR rec[-1]",
        batch_size=3,
        seed=5,
        record_u2l=True,
        record_events=True,
    )
    fired = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code.co_name == "_dcc_build_state" and not fired[0]:
            fired[0] = True
            raise KeyboardInterrupt("aborted-step")

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            sim.interactive_do(sim.circuit[0])
    finally:
        sys.settrace(None)

    assert sim.num_measurements == 1
    assert sim.num_detectors == 1
    assert sim.num_observables == 0
    assert sim.num_qubits_in_new_circuit() == 1
    for fn in _stateful_calls(sim, include_clear=False):
        with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
            fn()
    sim.clear()
    sim.run()
    assert sim.get_final_measurement_records().shape == (3, 1)


def test_d36_interrupted_clear_stays_latched_until_retry_succeeds() -> None:
    sim = _mk_sim("H 0\nM 0", batch_size=4, seed=9)
    sim.interactive_do(sim.circuit[0])
    real_lib = sim._cpp_lib
    assert real_lib is not None
    fails_left = [2]

    class _InterruptClear:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_clear":
                return fn

            def wrapped(*args: Any) -> Any:
                res = fn(*args)
                if fails_left[0] > 0:
                    fails_left[0] -= 1
                    raise KeyboardInterrupt(f"clear-abort-{fails_left[0]}")
                return res

            return wrapped

    sim._cpp_lib = cast(ctypes.CDLL, _InterruptClear())
    # First interrupted clear on healthy sim latches.
    with pytest.raises(KeyboardInterrupt, match="clear-abort-1"):
        sim.clear()
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.peek_z(0)
    # Second interrupted clear on already-aborted sim stays latched.
    with pytest.raises(KeyboardInterrupt, match="clear-abort-0"):
        sim.clear()
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        _ = sim.active_shots
    # Third clear succeeds and restores normal usability.
    sim.clear()
    assert type(sim) is CosetsideSimulator
    sim.run()
    assert sim.get_final_measurement_records().shape == (4, 1)


def test_d36_failed_engine_takes_permanent_precedence_over_aborted() -> None:
    sim = _mk_sim("H 0\nM 0", batch_size=2, seed=4)
    sim._record_aborted_call(KeyboardInterrupt("first-abort"))
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.peek_z(0)
    real_lib = sim._cpp_lib
    assert real_lib is not None

    class _FailedLib:
        def __getattr__(self, name: str) -> Any:
            if name == "engine_failed":
                return lambda _ptr: 1
            return getattr(real_lib, name)

    sim._cpp_lib = cast(ctypes.CDLL, _FailedLib())
    with pytest.raises(RuntimeError, match=r"MemoryError.*new simulator"):
        sim.clear()
    assert type(sim) is _FailedCosetsideSimulator
    # Recording another abort must not downgrade permanent failure.
    sim._record_aborted_call(KeyboardInterrupt("later"))
    assert type(sim) is _FailedCosetsideSimulator
    with pytest.raises(RuntimeError, match=r"MemoryError.*new simulator"):
        sim.peek_z(0)
    for fn in _stateful_calls(sim, include_clear=True):
        with pytest.raises(RuntimeError, match=r"MemoryError.*new simulator"):
            fn()


def test_r7_f4_plain_slotted_and_custom_setattr_subclasses() -> None:
    class PlainSub(CosetsideSimulator):
        flavour = "plain"

        def helper(self) -> int:
            return 42

        def get_current_circuit_time(self) -> int:
            return 100 + super().get_current_circuit_time()

    class SlotSub(CosetsideSimulator):
        __slots__ = ("extra",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.extra = "slot-val"

    class ChildSlotSub(SlotSub):
        __slots__ = ("more",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.more = 99

    class GuardedSetattrSub(CosetsideSimulator):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "__class__":
                raise PermissionError("blocked")
            super().__setattr__(name, value)

    c = stim.Circuit("H 0\nM 0")
    for cls in (PlainSub, SlotSub, ChildSlotSub, GuardedSetattrSub):
        h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
        sim = cls(c, compiled_op_handler=h, batch_size=2, seed=5)
        # Test aborted latch + recovery
        sim._record_aborted_call(KeyboardInterrupt("sub-abort"))
        assert isinstance(sim, cls)
        assert isinstance(sim, CosetsideSimulator)
        if cls is PlainSub:
            assert sim.helper() == 42 and sim.flavour == "plain"
        if issubclass(cls, SlotSub):
            assert sim.extra == "slot-val"
        if cls is ChildSlotSub:
            assert sim.more == 99
        for fn in _stateful_calls(sim, include_clear=False):
            with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
                fn()
        sim.clear()
        assert type(sim) is cls
        assert sim.get_current_circuit_time() == (100 if cls is PlainSub else 0)
        # Test permanent failed engine
        sim._record_engine_failed("MemoryError: oom")
        assert isinstance(sim, cls)
        assert isinstance(sim, CosetsideSimulator)
        for fn in _stateful_calls(sim, include_clear=True):
            with pytest.raises(RuntimeError, match=r"MemoryError: oom.*new simulator"):
                fn()


def test_d36_direct_mutators_latch_on_mid_call_interrupt() -> None:
    # 1) do_conditional_pauli_noise
    sim1 = _mk_sim("H 0\nX_ERROR(0.5) 0\nM 0", batch_size=4, seed=1)
    sim1.do_conditional_clifford(sim1.circuit[0], None)
    fired = [False]

    def tr_noise(frame: Any, event: str, _arg: Any) -> None:
        if (
            event == "call"
            and frame.f_code.co_name == "_do_cond_uncorrelated_noise"
            and not fired[0]
        ):
            fired[0] = True
            raise KeyboardInterrupt("noise-abort")

    sys.settrace(tr_noise)
    try:
        with pytest.raises(KeyboardInterrupt, match="noise-abort"):
            sim1.do_conditional_pauli_noise(sim1.circuit[1], None)
    finally:
        sys.settrace(None)
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim1.peek_z(0)
    sim1.clear()
    sim1.run()

    # 2) apply_pauli
    sim2 = _mk_sim("H 0\nM 0", batch_size=4, seed=2)
    fired2 = [False]

    def tr_pauli(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code.co_name == "broadcast_pauli_errors" and not fired2[0]:
            fired2[0] = True
            raise KeyboardInterrupt("pauli-abort")

    sys.settrace(tr_pauli)
    try:
        with pytest.raises(KeyboardInterrupt, match="pauli-abort"):
            sim2.apply_pauli("X", np.ones((1, 4), dtype=bool), 0.5)
    finally:
        sys.settrace(None)
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        _ = sim2.active_shots
    sim2.clear()
    sim2.run()


# ---------------------------------------------------------------------------
# B1-3 (R7-F3 + R7-F3b): Atomic multi-op interactive_do & noise allowlist
# ---------------------------------------------------------------------------


def test_r7_f3_interactive_do_multi_instruction_and_repeat_atomic() -> None:
    sim = _mk_sim("H 0 1\nCX 0 1\nM 0 1", batch_size=4, seed=5)
    # Valid first op followed by mismatched second op -> atomic rejection at t=0.
    with pytest.raises(ValueError, match="interactive_do"):
        sim.interactive_do(stim.Circuit("H 0 1\nS 0"))
    assert sim._circuit_time == 0
    assert sim._step_cursor == 0
    assert sim._used_interactively is False
    assert sim._unrolled_ops is None
    assert np.all(sim.peek_pauli_batch([0, 1], "Z") == 1)

    # Simulator still clean for run()!
    sim.run()
    assert sim.get_final_measurement_records().shape == (4, 2)
    sim.clear()

    # Repeat block mismatching on 2nd iteration -> atomic rejection.
    sim_rep = _mk_sim("H 0\nM 0", batch_size=4, seed=5)
    bad_rep = stim.Circuit("REPEAT 2 {\nH 0\n}")[0]
    with pytest.raises(ValueError, match="interactive_do"):
        sim_rep.interactive_do(bad_rep)
    assert sim_rep._circuit_time == 0 and sim_rep._step_cursor == 0
    assert sim_rep._used_interactively is False

    # Past-end large repeat block with bad op (DETECTOR) rejected atomically in O(body).
    sim_rep.interactive_do(sim_rep.circuit)
    t_before, c_before = sim_rep._circuit_time, sim_rep._step_cursor
    past_bad = stim.Circuit("X 0\nREPEAT 1000000 {\nDETECTOR rec[-1]\n}")
    with pytest.raises(ValueError, match="past the end"):
        sim_rep.interactive_do(past_bad)
    assert sim_rep._circuit_time == t_before and sim_rep._step_cursor == c_before

    # Past-end SPP rejected atomically before applying preceding X 0.
    with pytest.raises(NotImplementedError):
        sim_rep.interactive_do(stim.Circuit("X 0\nSPP X0"))
    assert sim_rep._circuit_time == t_before and sim_rep._step_cursor == c_before

    # Past-end empty repeat block with huge repeat_count finishes immediately.
    empty_rep = stim.Circuit("REPEAT 1000000 {\n}")[0]
    sim_rep.interactive_do(empty_rep)
    assert sim_rep._circuit_time == t_before


def test_r7_f3b_do_conditional_pauli_noise_rejects_non_noise_atomically() -> None:
    bad_ops = [
        stim.CircuitInstruction("H", [0]),
        stim.CircuitInstruction("CX", [0, 1]),
        stim.CircuitInstruction("M", [0]),
        stim.CircuitInstruction("MR", [0]),
        stim.CircuitInstruction("R", [0]),
        stim.CircuitInstruction("TICK", []),
        stim.CircuitInstruction("MPAD", [0]),
        stim.CircuitInstruction("HERALDED_ERASE", [0], [0.1]),
        stim.CircuitInstruction("HERALDED_PAULI_CHANNEL_1", [0], [0.01, 0.01, 0.01, 0.01]),
    ]
    for op in bad_ops:
        sim = _mk_sim("H 0 1\nM 0 1", batch_size=4, seed=3)
        sim.do_conditional_clifford(sim.circuit[0], None)
        c0 = sim._step_cursor
        rng0 = sim.np_rng.bit_generator.state
        with pytest.raises(ValueError, match="Pauli noise channel or identity"):
            sim.do_conditional_pauli_noise(op, None)
        assert sim._step_cursor == c0
        assert sim.np_rng.bit_generator.state == rng0
        # Still usable (not latched).
        assert sim.peek_z(0) == 0

    good_ops = [
        ("I 0", 1),
        ("II 0 1", 1),
        ("I_ERROR(0.1) 0", 1),
        ("II_ERROR(0.1) 0 1", 1),
        ("X_ERROR(0.2) 0", 1),
        ("Y_ERROR(0.2) 0", 1),
        ("Z_ERROR(0.2) 0", 1),
        ("DEPOLARIZE1(0.2) 0", 1),
        ("DEPOLARIZE2(0.2) 0 1", 1),
        ("PAULI_CHANNEL_1(0.05,0.05,0.05) 0", 1),
        ("PAULI_CHANNEL_2(" + ",".join(["0.01"] * 15) + ") 0 1", 1),
        ("E(0.2) X0", 1),
        ("CORRELATED_ERROR(0.2) Z0", 1),
        ("ELSE_CORRELATED_ERROR(0.2) Y0", 1),
    ]
    for text, rows in good_ops:
        op = stim.Circuit(text)[0]
        sim = _mk_sim(f"H 0 1\n{text}\nM 0 1", batch_size=4, seed=3)
        sim.do_conditional_clifford(sim.circuit[0], None)
        sim.do_conditional_pauli_noise(op, np.ones((rows, 4), dtype=bool))
        assert sim._step_cursor == 2


# ---------------------------------------------------------------------------
# B1-4 (R7-F5): _EngineOwner safe explicit __del__() mid-kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["export", "clear", "dcc", "post", "reset", "peek"])
def test_r7_f5_explicit_del_mid_kernel_keeps_owner_alive_and_destroys_once(target: str) -> None:
    n_q, bs = 16, 2
    c = stim.Circuit(f"H {' '.join(map(str, range(n_q)))}")
    sim = _mk_sim(str(c), batch_size=bs, seed=3)
    hop = stim.CircuitInstruction("H", list(range(n_q)))
    m = np.ones((n_q, bs), dtype=bool)
    m[0, 0] = False
    sim.do_conditional_clifford(hop, m)
    owner = sim._cpp_owner
    assert owner is not None and owner.ptr is not None
    orig_destroy = owner._destroy
    destroyed: list[int] = []
    owner._destroy = lambda p: (destroyed.append(p), orig_destroy(p))[1]
    ptr = owner.ptr
    del owner  # only simulator holds owner now

    state: dict[str, Any] = {"on": True, "fired": False, "destroyed_mid_call": None}

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if state["on"] and event == "call" and frame.f_code.co_name == "guarded":
            state["on"] = False
            state["fired"] = True
            sim.__del__()
            state["destroyed_mid_call"] = list(destroyed)

    calls = {
        "export": lambda: sim.active_shots,
        "clear": lambda: sim.clear(),
        "dcc": lambda: sim.do_conditional_clifford(hop, np.zeros((n_q, bs), dtype=bool)),
        "post": lambda: sim.postselect_z_batch(0, np.ones(bs, bool), np.zeros(bs, np.uint8), "X"),
        "reset": lambda: sim.do_unmatched_noisy_reset_z(
            np.ones((n_q, bs), bool), np.zeros((n_q, bs), bool)
        ),
        "peek": lambda: sim.peek_pauli_batch([0, 1], "Z"),
    }
    sys.setprofile(prof)
    try:
        calls[target]()
    except RuntimeError:
        pass
    finally:
        state["on"] = False
        sys.setprofile(None)
    assert state["fired"]
    assert state["destroyed_mid_call"] == []
    assert sim._cpp_owner is None and sim._cpp_eng is None
    gc.collect()
    assert destroyed == [ptr]
    # Second explicit __del__() must not double-free.
    sim.__del__()
    assert destroyed == [ptr]


# ---------------------------------------------------------------------------
# B1-5 (R7-F6): Export stale flag, active_shots/clear re-entry, run() circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_clear", [True, False])
def test_r7_f6_abort_export_always_marks_dirty(with_clear: bool) -> None:
    n_q, bs = 16, 2
    h_all = stim.CircuitInstruction("H", list(range(n_q)))
    m_one = np.ones((n_q, bs), dtype=bool)
    m_one[0, 0] = False
    sim = _mk_sim(f"QUBIT_COORDS(0) {n_q - 1}", batch_size=bs, seed=3)
    sim.do_conditional_clifford(h_all, m_one)
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "call" and frame.f_code.co_name == "_install_export":
            fired[0] = True
            if with_clear:
                sim.clear()
                sim.do_conditional_clifford(h_all, m_one)
            else:
                m_two = np.ones((n_q, bs), dtype=bool)
                m_two[1, 0] = False
                sim.do_conditional_clifford(h_all, m_two)
            _ = sim.active_shots

    sys.setprofile(prof)
    try:
        with pytest.raises(RuntimeError, match="changed during active_shots export"):
            _ = sim.active_shots
    finally:
        sys.setprofile(None)
    assert fired[0]
    assert sim._cpp_dirty is True
    assert 0 in sim.active_shots


def test_r7_f6_active_shots_copy_reentry_raises() -> None:
    sim = _mk_sim("H 0 1\nM 0 1", batch_size=2, seed=3)
    sim.do_conditional_clifford(
        stim.CircuitInstruction("H", [0, 1]), np.array([[False, True], [True, True]])
    )
    _ = sim.active_shots  # clean cache
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "call" and frame.f_code.co_name == "copy":
            fired[0] = True
            sim.clear()

    sys.setprofile(prof)
    try:
        with pytest.raises(RuntimeError, match="changed during active_shots read"):
            _ = sim.active_shots
    finally:
        sys.setprofile(None)
    assert fired[0]
    assert sim.active_shots == {}


def test_r7_f6_run_detects_in_place_and_reassigned_circuit_changes() -> None:
    sim = _mk_sim("X 0\nM 0", batch_size=3, seed=7)
    sim.run()
    assert sim.get_final_measurement_records().astype(int).tolist() == [[1]] * 3
    ops_first = sim._run_ops

    # Unchanged circuit reuses cached _run_ops.
    sim.clear()
    sim.run()
    assert sim._run_ops is ops_first

    # In-place mutation of sim.circuit rebuilds _run_ops.
    sim.clear()
    sim.circuit.append("X", [0])
    sim.circuit.append("M", [0])
    sim.run()
    assert sim._run_ops is not ops_first
    assert sim.get_final_measurement_records().astype(int).tolist() == [[1, 0]] * 3

    # Reassignment of sim.circuit rebuilds _run_ops.
    sim.clear()
    sim.circuit = stim.Circuit("X 0\nM 0")
    sim.run()
    assert sim.get_final_measurement_records().astype(int).tolist() == [[1]] * 3


# ---------------------------------------------------------------------------
# Stage B Round 2 regressions & mutant killers (M1-M4, TS1-TS2, L1-L2)
# ---------------------------------------------------------------------------


def test_r8_r2_clear_aborted_latch_interrupt_at_every_line_recovers() -> None:
    for target_hit in range(1, 8):
        sim = _mk_sim("H 0\nM 0", batch_size=3, seed=7)
        sim._record_aborted_call(KeyboardInterrupt("initial-abort"))
        hits = [0]

        def tracer(
            frame: Any, event: str, _arg: Any, th: int = target_hit, cnt: list[int] = hits
        ) -> Any:
            if frame.f_code.co_name == "_clear_aborted_latch":
                if event == "call":
                    return tracer
                if event == "line":
                    cnt[0] += 1
                    if cnt[0] == th:
                        raise KeyboardInterrupt(f"line-hit-{th}")
            return tracer

        sys.settrace(tracer)
        caught = False
        try:
            sim.clear()
        except KeyboardInterrupt:
            caught = True
        finally:
            sys.settrace(None)
        if not caught:
            assert type(sim) is CosetsideSimulator
            break
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            sim.run()
        sim.clear()
        assert type(sim) is CosetsideSimulator
        assert sim._orig_class is None and sim._aborted_call_msg is None
        sim.run()
        assert sim.get_final_measurement_records().shape == (3, 1)

    sim_fb = _mk_sim("H 0\nM 0", batch_size=2, seed=1)
    object.__setattr__(sim_fb, "_aborted_call_msg", None)
    with pytest.raises(RuntimeError, match=r"call clear\(\)"):
        sc._raise_aborted_call(sim_fb)
    object.__setattr__(sim_fb, "_engine_failed_msg", None)
    with pytest.raises(RuntimeError, match="create a new simulator"):
        sc._raise_engine_failed(sim_fb)


def test_r8_r2_record_engine_failed_interrupt_completes_swap_on_retry() -> None:
    for target_hit in range(1, 10):
        sim = _mk_sim("H 0\nM 0", batch_size=2, seed=3)
        hits = [0]

        def tracer(
            frame: Any, event: str, _arg: Any, th: int = target_hit, cnt: list[int] = hits
        ) -> Any:
            if frame.f_code.co_name == "_record_engine_failed":
                if event == "call":
                    return tracer
                if event == "line":
                    cnt[0] += 1
                    if cnt[0] == th:
                        raise KeyboardInterrupt(f"fail-hit-{th}")
            return tracer

        sys.settrace(tracer)
        caught = False
        try:
            sim._record_engine_failed("MemoryError: simulated")
        except KeyboardInterrupt:
            caught = True
        finally:
            sys.settrace(None)
        if not caught:
            break
        if sim._engine_failed_msg is not None:
            with pytest.raises(RuntimeError, match="MemoryError: simulated"):
                sim.clear()
            assert type(sim) is _FailedCosetsideSimulator
            with pytest.raises(RuntimeError, match="MemoryError: simulated"):
                sim.get_current_circuit_time()
            with pytest.raises(RuntimeError, match="MemoryError: simulated"):
                sim.apply_pauli("X", np.ones((1, 2), dtype=bool))


def test_r8_r2_subclasses_init_subclass_kwarg_registry_and_custom_meta() -> None:
    tracked_seen: set[str] = set()

    class KwargInitSub(CosetsideSimulator):
        def __init_subclass__(cls, *, tag: str) -> None:
            super().__init_subclass__()
            cls._tag = tag

    class ConcreteKwargSub(KwargInitSub, tag="v1"):
        __slots__ = ("slot_v",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.slot_v = "ok"

    class TrackedBase(CosetsideSimulator):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            tracked_seen.add(cls.__name__)

    class TrackedSub(TrackedBase):
        pass

    class AbstractSim(CosetsideSimulator, metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def metric(self) -> int: ...

    class ConcreteABCSim(AbstractSim):
        def metric(self) -> int:
            return 99

    class CustomMeta(type):
        def __new__(
            mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any], *,
            version: int | None = None,
        ) -> type:
            if version is None:
                raise TypeError(f"version= required for {name}")
            return super().__new__(mcls, name, bases, ns)

    class MetaSim(CosetsideSimulator, metaclass=CustomMeta, version=1):
        pass

    class ABCKwargMeta(abc.ABCMeta):
        def __new__(
            mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any], *,
            v: int | None = None,
        ) -> type:
            if v is None:
                raise TypeError("v= required")
            return super().__new__(mcls, name, bases, ns)

    class ABCMetaSim(CosetsideSimulator, metaclass=ABCKwargMeta, v=2):
        pass

    c = stim.Circuit("H 0\nM 0")
    before_seen = set(tracked_seen)
    for cls in (ConcreteKwargSub, TrackedSub, ConcreteABCSim, MetaSim, ABCMetaSim):
        h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
        sim = cls(c, compiled_op_handler=h, batch_size=2, seed=1)
        sim._record_aborted_call(KeyboardInterrupt("sub-ki"))
        assert isinstance(sim, cls)
        with pytest.raises(RuntimeError, match="KeyboardInterrupt"):
            sim.get_current_circuit_time()
        sim.clear()
        assert type(sim) is cls
        sim._record_engine_failed("MemoryError")
        assert isinstance(sim, cls)
        with pytest.raises(RuntimeError, match="MemoryError"):
            sim.clear()
    assert tracked_seen == before_seen


def test_r8_r2_latch_class_oom_fallback_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlainSub(CosetsideSimulator):
        pass

    c = stim.Circuit("H 0\nM 0")
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
    sim = PlainSub(c, compiled_op_handler=h, batch_size=2, seed=2)
    monkeypatch.setattr(sc, "_get_aborted_class", lambda _c: (_ for _ in ()).throw(MemoryError()))
    sim._record_aborted_call(KeyboardInterrupt("oom-fallback"))
    assert type(sim) is sc._AbortedCosetsideSimulator
    sim.clear()
    assert type(sim) is PlainSub
    monkeypatch.setattr(sc, "_get_failed_class", lambda _c: (_ for _ in ()).throw(MemoryError()))
    sim._record_engine_failed("MemoryError")
    assert type(sim) is _FailedCosetsideSimulator


def test_r8_r2_ephemeral_subclasses_garbage_collected() -> None:
    gc.collect()
    before_a, before_f = len(sc._ABORTED_CLASSES), len(sc._FAILED_CLASSES)
    refs: list[weakref.ReferenceType[type[Any]]] = []
    for i in range(20):
        def _make(idx: int = i) -> None:
            if idx % 3 == 0:
                class Ephem(CosetsideSimulator):
                    pass
            elif idx % 3 == 1:
                class Ephem(CosetsideSimulator):  # type: ignore[no-redef]
                    __slots__ = ("slot_x",)
            else:
                class Ephem(CosetsideSimulator, metaclass=abc.ABCMeta):  # type: ignore[no-redef]
                    pass

            refs.append(weakref.ref(Ephem))
            sc._get_aborted_class(Ephem)
            sc._get_failed_class(Ephem)

        _make()
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    assert alive == 0
    assert len(sc._ABORTED_CLASSES) == before_a
    assert len(sc._FAILED_CLASSES) == before_f


def test_r8_r2_mpp_anti_hermitian_and_rec_oob_atomic() -> None:
    for mpp_text in ("MPP X0*Y0", "MPP X0*Z0", "MPP X0*Y0*Z0"):
        sim = _mk_sim("H 0", batch_size=2, seed=1)
        with pytest.raises(ValueError, match="anti-Hermitian"):
            sim.interactive_do(stim.Circuit(f"H 0\n{mpp_text}"))
        assert type(sim) is CosetsideSimulator
        assert sim._mutation_epoch == 0 and sim._step_cursor == 0
        assert sim._circuit_time == 0 and sim._used_interactively is False

    # Valid repeated-qubit Hermitian MPP succeeds.
    sim_ok = _mk_sim("H 0", batch_size=2, seed=1)
    sim_ok.interactive_do(stim.Circuit("H 0\nMPP X0*X0"))
    assert len(sim_ok._measurement_columns) == 1

    # Feedback rec[-k] OOB rejects atomically under standard handler.
    for fb in ("CX rec[-999] 0", "CZ rec[-1] 0"):
        sim_fb = _mk_sim("H 0", batch_size=2, seed=1)
        with pytest.raises(IndexError):
            sim_fb.interactive_do(stim.Circuit(f"H 0\n{fb}"))
        assert type(sim_fb) is CosetsideSimulator
        assert sim_fb._mutation_epoch == 0 and sim_fb._step_cursor == 0
        assert sim_fb._circuit_time == 0 and sim_fb._used_interactively is False

    # Valid rec[-k] after in-batch measurements (including REPEAT) succeeds.
    sim_v = _mk_sim("H 0", batch_size=2, seed=1)
    sim_v.interactive_do(stim.Circuit("H 0\nREPEAT 5 {\nM 0\n}\nCX rec[-5] 0"))
    assert len(sim_v._measurement_columns) == 5

    # Custom CompiledOpHandler injecting direct measurements is NOT falsely rejected.
    class InjectMeasHandler(CompiledOpHandler[Any]):
        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            if op.name == "H":
                sss.record_direct_measurements(np.zeros((1, sss.batch_size), dtype=bool))
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            pass

    c_h = stim.Circuit("H 0")
    sim_c = CosetsideSimulator(c_h, compiled_op_handler=InjectMeasHandler(), batch_size=2, seed=1)
    sim_c.interactive_do(stim.Circuit("H 0\nCX rec[-1] 0"))
    assert len(sim_c._measurement_columns) == 1


def test_r8_r2_nested_empty_repeat_o1_validation_and_do() -> None:
    inner = stim.Circuit()
    inner.append(stim.CircuitRepeatBlock(1, stim.Circuit()))
    outer = stim.CircuitRepeatBlock(2_000_000, inner)
    bad_mismatch = stim.Circuit()
    bad_mismatch.append(outer)
    bad_mismatch.append("SPP", [stim.target_x(0)])
    sim = _mk_sim("H 0", batch_size=2, seed=1)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="interactive_do"):
        sim.interactive_do(bad_mismatch)
    assert time.perf_counter() - t0 < 0.5
    assert sim._circuit_time == 0 and sim._used_interactively is False

    bad_spp = stim.Circuit()
    bad_spp.append("H", [0])
    bad_spp.append(outer)
    bad_spp.append("SPP", [stim.target_x(0)])
    t0b = time.perf_counter()
    with pytest.raises(NotImplementedError):
        sim.interactive_do(bad_spp)
    assert time.perf_counter() - t0b < 0.5
    assert sim._circuit_time == 0 and sim._used_interactively is False

    # Executing valid nested empty repeat via interactive_do also finishes in O(1).
    t1 = time.perf_counter()
    sim.interactive_do(outer)
    assert time.perf_counter() - t1 < 0.5

    # Deeper nested empty repeat followed by mismatch also finishes in O(1).
    deep_inner = stim.Circuit()
    deep_inner.append(outer)
    bad_deep = stim.Circuit()
    bad_deep.append(stim.CircuitRepeatBlock(2_000_000, deep_inner))
    bad_deep.append("Z", [0])
    t2 = time.perf_counter()
    with pytest.raises(ValueError, match="interactive_do"):
        sim.interactive_do(bad_deep)
    assert time.perf_counter() - t2 < 0.5


def test_r8_r2_interactive_do_restore_with_fail_once_handler() -> None:
    class FailOnceHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.calls = 0

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("fail once before touching sss")
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            pass

    c = stim.Circuit("H 0")
    s1 = CosetsideSimulator(c, compiled_op_handler=FailOnceHandler(), batch_size=2, seed=1)
    with pytest.raises(ValueError, match="fail once"):
        s1.interactive_do(c[0])
    assert type(s1) is CosetsideSimulator
    assert s1._used_interactively is False and s1._unrolled_ops is None
    s1.run()  # kills I01 & I08

    s2 = CosetsideSimulator(c, compiled_op_handler=FailOnceHandler(), batch_size=2, seed=1)
    with pytest.raises(ValueError, match="fail once"):
        s2.interactive_do(c[0])
    c_ext = stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
    s2.circuit = c_ext
    s2.interactive_do(c_ext)  # kills I07 (fails if stale _unrolled_ops kept)

    # Interrupt right after self._unrolled_ops = unrolled inside try: restores cleanly.
    s3 = _mk_sim("H 0\nM 0", batch_size=2, seed=1)
    fired = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> Any:
        if frame.f_code.co_name == "interactive_do":
            if event == "call":
                return tracer
            if event == "line" and not fired[0] and s3._unrolled_ops is not None:
                fired[0] = True
                raise KeyboardInterrupt("between-unrolled-and-do")
        return tracer

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt, match="between-unrolled-and-do"):
            s3.interactive_do(s3.circuit)
    finally:
        sys.settrace(None)
    assert fired[0]
    assert s3._used_interactively is False and s3._unrolled_ops is None
    s3.run()


def test_r8_r2_verify_and_commit_export_late_reentry() -> None:
    n_q, bs = 8, 2
    hop = stim.CircuitInstruction("H", list(range(n_q)))
    m_one = np.ones((n_q, bs), dtype=bool)
    m_one[0, 0] = False
    m_two = np.ones((n_q, bs), dtype=bool)
    m_two[1, 0] = False

    # Mutate + reseed via clear() inside np.array_equal -> kills M18 alone and M15+M18 combo
    # (under M18 alone, stale rng_buf overwrites the fresh post-clear _shot_tiebreak_rng).
    sim = _mk_sim(f"QUBIT_COORDS(0) {n_q - 1}", batch_size=bs, seed=3)
    sim.do_conditional_clifford(hop, m_one)
    rng_before = sim._shot_tiebreak_rng.copy()
    rng_after_clear: list[Any] = []
    fired = [False]
    eq_calls = [0]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if event == "return" and frame.f_code.co_name == "array_equal":
            eq_calls[0] += 1
            if eq_calls[0] == 2 and not fired[0]:
                fired[0] = True
                sim.seed = 99
                sim.clear()
                rng_after_clear.append(sim._shot_tiebreak_rng.copy())

    sys.setprofile(prof)
    try:
        with pytest.raises(RuntimeError, match="changed during active_shots export"):
            sim._sync_cpp_to_py()
    finally:
        sys.setprofile(None)
    assert fired[0] and sim._cpp_dirty is True
    assert not np.array_equal(rng_before, rng_after_clear[0])
    assert np.array_equal(sim._shot_tiebreak_rng, rng_after_clear[0])

    # Mutate directly on `self._cpp_dirty = False` (after the `if` passed) -> kills M15 alone!
    sim2 = _mk_sim(f"QUBIT_COORDS(0) {n_q - 1}", batch_size=bs, seed=4)
    sim2.do_conditional_clifford(hop, m_one)
    src_lines, start_ln = inspect.getsourcelines(sim2._verify_and_commit_export)
    dirty_lines = [
        start_ln + idx
        for idx, line in enumerate(src_lines)
        if line.strip() == "self._cpp_dirty = False"
    ]
    assert len(dirty_lines) == 1
    target_ln = dirty_lines[0]
    fired2 = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> Any:
        if frame.f_code.co_name == "_verify_and_commit_export":
            if event == "call":
                return tracer
            if event == "line" and not fired2[0] and frame.f_lineno == target_ln:
                fired2[0] = True
                sim2.do_conditional_clifford(hop, m_two)
                return None
        return tracer

    sys.settrace(tracer)
    try:
        with pytest.raises(RuntimeError, match="changed during active_shots export"):
            sim2._sync_cpp_to_py()
    finally:
        sys.settrace(None)
    assert fired2[0] and sim2._cpp_dirty is True


def test_r8_r2_next_step_meta_tape_growth_interrupt_latches() -> None:
    sim = _mk_sim("X 0", batch_size=2, seed=1)
    sim._next_step_meta(sim.circuit[0])
    fired = [False]

    def tracer(frame: Any, event: str, _arg: Any) -> Any:
        if frame.f_code.co_name == "append_compiled_step":
            if event == "call":
                return tracer
            if event == "return" and not fired[0]:
                fired[0] = True
                raise KeyboardInterrupt("after-append-before-cursor")
        return None

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt, match="after-append-before-cursor"):
            sim._next_step_meta(stim.CircuitInstruction("H", [0]))
    finally:
        sys.settrace(None)
    assert fired[0]
    assert type(sim).__name__ == "_AbortedCosetsideSimulator"
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.get_current_circuit_time()
    sim.clear()
    assert type(sim) is CosetsideSimulator


def test_r8_r2_validate_repeat_fast_forward_op_index() -> None:
    sim = _mk_sim("H 0", batch_size=2, seed=1)
    bad = stim.Circuit("H 0\nREPEAT 1000 {\nX 0\n}\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match=r"operation 1001 \(REPEAT blocks unrolled\)"):
        sim.interactive_do(bad)
    assert sim._circuit_time == 0 and sim._used_interactively is False


@pytest.mark.parametrize("target", ["dcc", "reset", "reset_z_no_order", "postselect"])
def test_r8_r2_reentrant_clear_during_ctypes_prep_keeps_dirty(target: str) -> None:
    n_q, bs = 8, 2
    hop = stim.CircuitInstruction("H", list(range(n_q)))
    sim = _mk_sim(f"H {' '.join(map(str, range(n_q)))}", batch_size=bs, seed=3)
    if target == "reset_z_no_order":
        sim._do_bare_instruction(hop)
    elif target != "postselect":
        m = np.ones((n_q, bs), dtype=bool)
        m[0, 0] = False
        sim.do_conditional_clifford(hop, m)
    _ = sim.active_shots
    fired = [False]

    def prof(frame: Any, event: str, arg: Any) -> None:
        if fired[0]:
            return
        if target == "reset_z_no_order":
            if (
                event == "call"
                and frame.f_code.co_name == "_cpp_out_args"
                and frame.f_back is not None
                and frame.f_back.f_code.co_name == "_reset_z_cpp"
            ):
                fired[0] = True
                sim.clear()
        elif (
            event == "c_call"
            and getattr(arg, "__name__", "") == "byref"
            and frame.f_code.co_name
            in ("_dcc_disjoint_cpp_v3", "_project_pauli_cpp_exec", "_postselect_cpp_exec")
        ):
            fired[0] = True
            sim.clear()

    sys.setprofile(prof)
    try:
        if target == "dcc":
            sim.do_conditional_clifford(hop, np.zeros((n_q, bs), dtype=bool))
        elif target == "reset":
            sim.do_unmatched_noisy_reset_z(
                np.ones((n_q, bs), dtype=bool),
                np.zeros((n_q, bs), dtype=bool),
                pauli="Z",
                order=list(range(n_q)),
            )
        elif target == "reset_z_no_order":
            sim.do_unmatched_noisy_reset_z(
                np.ones((n_q, bs), dtype=bool),
                np.zeros((n_q, bs), dtype=bool),
                pauli="Z",
                order=None,
            )
        else:
            sim.postselect_z_batch(0, np.ones(bs, dtype=bool), np.zeros(bs, dtype=np.uint8), "X")
    finally:
        sys.setprofile(None)
    assert fired[0]
    assert sim._cpp_dirty is True
    assert len(sim.active_shots) > 0


def test_r8_delta_abc_register_does_not_leak_to_latch_class() -> None:
    class AbcSim(CosetsideSimulator, metaclass=abc.ABCMeta):
        pass

    aborted_cls = sc._get_aborted_class(AbcSim)
    failed_cls = sc._get_failed_class(AbcSim)
    AbcSim.register(int)
    assert issubclass(int, AbcSim) is True
    assert issubclass(int, aborted_cls) is False
    assert issubclass(int, failed_cls) is False
    assert isinstance(1, aborted_cls) is False
    assert isinstance(1, failed_cls) is False


def test_r8_delta_latch_class_strong_cache_survives_gc() -> None:
    class KeepAliveSub(CosetsideSimulator):
        pass

    wr_ab = weakref.ref(sc._get_aborted_class(KeepAliveSub))
    wr_fl = weakref.ref(sc._get_failed_class(KeepAliveSub))
    gc.collect()
    assert wr_ab() is not None
    assert wr_fl() is not None
    assert sc._get_aborted_class(KeepAliveSub) is wr_ab()
    assert sc._get_failed_class(KeepAliveSub) is wr_fl()


def test_r8_delta_slotted_subclass_prewarmed_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    class PlainSlotSub(CosetsideSimulator):
        __slots__ = ("slot_v",)

    c = stim.Circuit("H 0\nM 0")
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
    sim = PlainSlotSub(c, compiled_op_handler=h, batch_size=2, seed=1)
    ab_cls = PlainSlotSub.__dict__["_stimside_aborted_cls"]
    fl_cls = PlainSlotSub.__dict__["_stimside_failed_cls"]
    assert sc._ABORTED_CLASSES[PlainSlotSub]() is ab_cls
    assert sc._FAILED_CLASSES[PlainSlotSub]() is fl_cls

    # Confirm cache hits require zero class creation by failing if abc.ABCMeta.__new__ runs.
    def _forbid_new(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("latch class should already be pre-warmed at __init__")

    monkeypatch.setattr(sc.abc.ABCMeta, "__new__", _forbid_new)
    sim._record_aborted_call(KeyboardInterrupt("prewarm-ab"))
    assert type(sim) is ab_cls
    sim.clear()
    assert type(sim) is PlainSlotSub
    sim._record_engine_failed("MemoryError: prewarm-fl")
    assert type(sim) is fl_cls


class _StrictMroMeta(type):
    def mro(cls) -> list[type]:
        if any(b.__name__ == "_SuppressInitSubclass" for b in cls.__bases__):
            raise TypeError("Sealed MRO refuses _SuppressInitSubclass")
        return super().mro()


class _SealedSlotSim(CosetsideSimulator, metaclass=_StrictMroMeta):
    __slots__ = ("tag",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tag = 777


def test_r8_delta_sealed_slot_sim_dict_latch_and_recovery() -> None:
    c = stim.Circuit("X 0\nM 0\nX 0\nM 0")
    ref = _mk_sim("X 0\nM 0\nX 0\nM 0", batch_size=3, seed=1)
    ref.run()
    expected = ref.get_final_measurement_records().tolist()

    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=3)
    sim = _SealedSlotSim(
        c, compiled_op_handler=h, batch_size=3, seed=1, record_unleaked_to_leaked=True
    )
    def custom_peek(_q: int) -> int:
        return 12345

    sim.__dict__["peek_z"] = custom_peek

    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_next_step_meta":
            fired[0] = True
            raise KeyboardInterrupt("sealed-abort")

    sys.setprofile(prof)
    try:
        with pytest.raises(KeyboardInterrupt, match="sealed-abort"):
            sim.run()
    finally:
        sys.setprofile(None)

    assert type(sim) is _SealedSlotSim and sim.tag == 777
    assert sim._latched_dict_saved is not None
    for call in (
        lambda: sim.run(),
        lambda: sim.get_current_circuit_time(),
        lambda: sim.active_shots,
        lambda: sim.unleaked_to_leaked_records,
        lambda: sim._do_bare_instruction(stim.Circuit("TICK")[0]),
        lambda: sim._next_step_meta(stim.Circuit("TICK")[0]),
        lambda: sim.peek_z(0),
    ):
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            call()

    sim.clear()
    assert sim._latched_dict_saved is None
    assert sim.__dict__["peek_z"] is custom_peek
    del sim.__dict__["peek_z"]
    sim.run()
    assert sim.get_final_measurement_records().tolist() == expected
    assert isinstance(sim.unleaked_to_leaked_records, list)
    assert len(sim.active_shots) == 0


def test_r8_delta_sealed_slot_sim_clear_interrupt_recovers() -> None:
    c = stim.Circuit("X 0\nM 0\nX 0\nM 0")
    for target_hit in range(1, 16):
        h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
        sim = _SealedSlotSim(
            c, compiled_op_handler=h, batch_size=2, seed=1, record_unleaked_to_leaked=True
        )
        sim._record_aborted_call(KeyboardInterrupt("init-sealed"))
        hits = [0]

        def tracer(
            frame: Any, event: str, _arg: Any, th: int = target_hit, cnt: list[int] = hits
        ) -> Any:
            if frame.f_code.co_name in ("_clear_aborted_latch", "_restore_dict_latch"):
                if event == "call":
                    return tracer
                if event == "line":
                    cnt[0] += 1
                    if cnt[0] == th:
                        raise KeyboardInterrupt(f"sealed-clear-{th}")
            return tracer

        sys.settrace(tracer)
        caught = False
        try:
            sim.clear()
        except KeyboardInterrupt:
            caught = True
        finally:
            sys.settrace(None)
        if not caught:
            assert sim._latched_dict_saved is None
            break
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            sim.run()
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            _ = sim.active_shots
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            _ = sim.unleaked_to_leaked_records
        sim.clear()
        sim.run()
        assert sim.get_final_measurement_records().tolist() == [[True, False], [True, False]]


def test_r8_delta_sealed_slot_sim_failed_precedence_and_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = stim.Circuit("X 0\nM 0\nX 0\nM 0")
    h2 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
    sim_f = _SealedSlotSim(
        c, compiled_op_handler=h2, batch_size=2, seed=1, record_unleaked_to_leaked=True
    )
    sim_f._record_aborted_call(KeyboardInterrupt("ab-first"))
    sim_f._record_engine_failed("MemoryError: sealed-oom")
    sim_f._record_aborted_call(KeyboardInterrupt("late-ab-ignored"))
    for fn in (
        lambda s: s.clear(),
        lambda s: s.run(),
        lambda s: s.active_shots,
        lambda s: s.unleaked_to_leaked_records,
    ):
        with pytest.raises(RuntimeError, match="native engine failed"):
            fn(sim_f)
    wr = weakref.ref(sim_f)
    del sim_f
    gc.collect()
    assert wr() is None

    # Upgrading from dict-latched aborted to class-swapped failed removes stale dict entries.
    class SlotUpgrade(CosetsideSimulator):
        __slots__ = ("extra",)

    h3 = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
    sim_u = SlotUpgrade(c, compiled_op_handler=h3, batch_size=2, seed=1)
    orig_get_ab = sc._get_aborted_class
    monkeypatch.setattr(sc, "_get_aborted_class", lambda _c: (_ for _ in ()).throw(MemoryError()))
    sim_u._record_aborted_call(KeyboardInterrupt("ab-dict"))
    assert sim_u._latched_dict_saved is not None
    monkeypatch.setattr(sc, "_get_aborted_class", orig_get_ab)
    sim_u._record_engine_failed("MemoryError: upgrade")
    assert sim_u._latched_dict_saved is None
    with pytest.raises(RuntimeError, match="native engine failed"):
        sim_u.run()


def test_r8_delta_bad_str_and_format_error_do_not_mask_or_unlatch() -> None:
    class BadStrError(Exception):
        def __str__(self) -> str:
            raise ValueError("broken __str__!")

    class OomStrError(Exception):
        def __str__(self) -> str:
            raise MemoryError("oom during str(exc)")

    c = stim.Circuit("H 0 1\nM 0 1")
    sim = _mk_sim("H 0 1\nM 0 1", batch_size=2, seed=1)

    def cb(g_idx: int) -> Any:
        if g_idx == 1:
            raise BadStrError("orig-group1-err")
        return np.zeros(2, dtype=bool)

    with pytest.raises(BadStrError):
        sim.do_conditional_clifford(c[0], None, condition_group_fn=cb)
    assert type(sim) is sc._AbortedCosetsideSimulator
    assert sim._aborted_call_msg == sc._ABORT_FB
    with pytest.raises(RuntimeError, match=r"clear\(\)"):
        sim.run()
    sim.clear()
    assert type(sim) is CosetsideSimulator

    sim2 = _mk_sim("H 0\nM 0", batch_size=2, seed=1)
    c2 = stim.Circuit("H 0\nM 0")
    with pytest.raises(BadStrError):
        sim2.do_copies_on_fused_step(
            c2[0], [c2[0]], lambda _op: (_ for _ in ()).throw(BadStrError("fused"))
        )
    assert type(sim2) is sc._AbortedCosetsideSimulator
    with pytest.raises(RuntimeError, match=r"clear\(\)"):
        sim2.get_current_circuit_time()

    sim3 = _mk_sim("H 0\nM 0", batch_size=2, seed=1)
    assert sim3._record_engine_failed(BadStrError("boom")) == sc._FAILED_FB
    assert type(sim3) is _FailedCosetsideSimulator

    sim4 = _mk_sim("H 0 1\nM 0 1", batch_size=2, seed=1)

    def cb_oom(g_idx: int) -> Any:
        if g_idx == 1:
            raise OomStrError("E" * 2000)
        return np.zeros(2, dtype=bool)

    with pytest.raises(OomStrError):
        sim4.do_conditional_clifford(c[0], None, condition_group_fn=cb_oom)
    assert type(sim4) is sc._AbortedCosetsideSimulator
    assert sim4._aborted_call_msg == sc._ABORT_FB
    assert sc._fmt_latch_msg(OomStrError(), sc._FAILED_FB) == sc._FAILED_FB


def test_r8_delta_bogus_cached_latch_attr_ignored() -> None:
    class BadCacheInt(CosetsideSimulator):
        _stimside_aborted_cls = int

    class BadCacheSelf(CosetsideSimulator):
        pass

    BadCacheSelf._stimside_aborted_cls = BadCacheSelf  # type: ignore[attr-defined]
    c = stim.Circuit("H 0\nM 0")
    for cls in (BadCacheInt, BadCacheSelf):
        h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=2)
        sim = cls(c, compiled_op_handler=h, batch_size=2, seed=1)
        sim._record_aborted_call(KeyboardInterrupt("bad-cache"))
        assert isinstance(sim, cls) and type(sim) is not cls
        with pytest.raises(RuntimeError, match=r"clear\(\)"):
            sim.get_current_circuit_time()
        sim.clear()
        assert type(sim) is cls
