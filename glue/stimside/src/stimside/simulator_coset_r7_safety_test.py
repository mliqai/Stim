"""Run 7 Stage A'' regression tests for CosetsideSimulator (A''-1, A''-3 D-36, A''-4, A''-5)."""

from __future__ import annotations

import ctypes
import gc
import sys
from typing import Any, cast

import numpy as np
import pytest
import stim

from stimside._coset.reference_compile import CompiledReferenceCircuit
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator


def _mk_sim(
    text: str,
    batch_size: int = 4,
    seed: int = 7,
    *,
    use_cpp: bool = True,
    sync: bool = False,
    record_u2l: bool = False,
    record_events: bool = False,
) -> CosetsideSimulator:
    circ = stim.Circuit(text)
    h = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=batch_size)
    sim = CosetsideSimulator(
        circ,
        compiled_op_handler=h,
        batch_size=batch_size,
        seed=seed,
        sync_tableside_rng=sync,
        record_unleaked_to_leaked=record_u2l,
        record_leakage_events=record_events,
    )
    sim._use_cpp = use_cpp
    return sim


class _ExportReentryProxy:
    """Injects a callback before or inside ``engine_export_shot_v3`` (once)."""

    def __init__(self, lib: ctypes.CDLL, when: str, action: Any) -> None:
        self._lib = lib
        self._when = when
        self._action = action
        self.fired = False
        self.last_rc: int | None = None

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if name != "engine_export_shot_v3":
            return fn

        def wrapped(*args: Any) -> int:
            if not self.fired and self._when == "before":
                self.fired = True
                self._action()
            rc = int(fn(*args))
            self.last_rc = rc
            if not self.fired and self._when == "inside":
                self.fired = True
                self._action()
            return rc

        return wrapped


def _setup_export_reentry_sim(n_q: int = 24, batch: int = 2) -> CosetsideSimulator:
    sim = _mk_sim(f"QUBIT_COORDS(0) {n_q - 1}", batch_size=batch, seed=3)
    h_all = stim.CircuitInstruction("H", list(range(n_q)))
    m_one = np.ones((n_q, batch), dtype=bool)
    m_one[0, 0] = False
    sim.do_conditional_clifford(h_all, m_one)
    sim.do_conditional_clifford(stim.CircuitInstruction("S", [5]), np.zeros((1, batch), dtype=bool))
    assert sim._cpp_dirty
    return sim


@pytest.mark.parametrize("kind", ["grow", "shrink", "clear", "same_rank"])
def test_a1_export_proxy_reentry_between_query_and_export_raises(kind: str) -> None:
    n_q, batch = 24, 2
    sim = _setup_export_reentry_sim(n_q, batch)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    h_all = stim.CircuitInstruction("H", list(range(n_q)))

    def act() -> None:
        if kind == "grow":
            sim.do_conditional_clifford(h_all, np.zeros((n_q, batch), dtype=bool))
        elif kind == "shrink":
            sim.postselect_z_batch(0, np.array([True, False]), np.zeros(batch, np.uint8), "X")
        elif kind == "clear":
            sim.clear()
        else:
            sim.do_conditional_clifford(
                stim.CircuitInstruction("X", [0]), np.zeros((1, batch), dtype=bool)
            )

    proxy = _ExportReentryProxy(real_lib, "before", act)
    sim._cpp_lib = cast(ctypes.CDLL, proxy)
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert proxy.fired
    assert sim._cpp_dirty is True
    sim._cpp_lib = real_lib
    shots = sim.active_shots
    if kind == "grow":
        assert shots[0].d >= n_q - 1
    elif kind == "shrink":
        assert shots[0].d == 1
    elif kind == "clear":
        assert 0 not in shots


def test_a1_export_proxy_reentry_triggers_rc_minus_4() -> None:
    n_q, batch = 24, 2
    sim = _setup_export_reentry_sim(n_q, batch)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    h_all = stim.CircuitInstruction("H", list(range(n_q)))

    class _InsideGrowProxy:
        def __init__(self) -> None:
            self.fired = False
            self.rcs: list[int] = []

        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_export_shot_v3":
                return fn

            def wrapped(*args: Any) -> int:
                if not self.fired:
                    self.fired = True
                    sim.do_conditional_clifford(h_all, np.zeros((n_q, batch), dtype=bool))
                rc = int(fn(*args))
                self.rcs.append(rc)
                return rc

            return wrapped

    proxy = _InsideGrowProxy()
    sim._cpp_lib = cast(ctypes.CDLL, proxy)
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert proxy.fired and proxy.rcs == [-4]
    assert sim._cpp_dirty
    sim._cpp_lib = real_lib
    assert sim.active_shots[0].d >= n_q - 1


def _assert_all_stateful_members_failed(sim: CosetsideSimulator, pat: str) -> None:
    assert sim.num_measurements == sim.circuit.num_measurements
    assert sim.num_detectors == sim.circuit.num_detectors
    assert sim.num_observables == sim.circuit.num_observables
    assert sim.num_qubits_in_new_circuit() == sim.num_qubits

    calls: list[Any] = [
        lambda: sim.clear(),
        lambda: sim.active_shots,
        lambda: sim.sync_rngs(),
        lambda: sim.measurement_column(0),
        lambda: sim.get_current_circuit_time(),
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
        lambda: sim.unleaked_to_leaked_records,
        lambda: sim.get_leakage_events(),
        lambda: sim.run(),
        lambda: sim.interactive_do(stim.CircuitInstruction("TICK", [])),
        lambda: sim.finish_interactive_run(),
        lambda: sim.do_copies_on_fused_step(
            stim.CircuitInstruction("TICK", []), [], lambda _: None
        ),
        lambda: sim.overwrite_measurement_column(0, np.zeros(sim.batch_size, dtype=bool)),
        lambda: sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None),
        lambda: sim.apply_pauli("X", np.zeros((sim.num_qubits, sim.batch_size), dtype=bool)),
        lambda: sim.do_conditional_pauli_noise(
            stim.CircuitInstruction("X_ERROR", [0], [0.1]), None
        ),
        lambda: sim.do_unmatched_noisy_reset_z(
            np.zeros((sim.num_qubits, sim.batch_size), dtype=bool),
            np.zeros((sim.num_qubits, sim.batch_size), dtype=bool),
        ),
        lambda: sim.postselect_z_batch(
            0, np.zeros(sim.batch_size, dtype=bool), np.zeros(sim.batch_size, dtype=np.uint8)
        ),
        lambda: sim.record_direct_measurements(np.zeros((1, sim.batch_size), dtype=bool)),
        lambda: sim.sample_pauli_noise_sync("X_ERROR", [[0]] * sim.batch_size, [0.1]),
        lambda: sim.record_u2l_counts(np.zeros(sim.batch_size, dtype=np.int64), 0),
        lambda: sim.record_leakage_event(0, 0, 0, 2, 0),
        lambda: sim._record_unleaked_to_leaked_counts(np.zeros(sim.batch_size, dtype=np.int64), 0),
        lambda: sim._record_leakage_event(0, 0, 0, 2, 0),
        lambda: sim._do_bare_instruction(stim.CircuitInstruction("TICK", [])),
        lambda: sim._next_step_meta(stim.CircuitInstruction("TICK", [])),
    ]
    for fn in calls:
        with pytest.raises(RuntimeError, match=pat):
            fn()


@pytest.mark.parametrize("mode", ["cpp", "py"])
def test_d36_atomic_rejections_and_getter_errors_keep_usable(mode: str) -> None:
    sim = _mk_sim("H 0 1\nM 0 1\nDETECTOR rec[-1]", batch_size=4, seed=10, use_cpp=(mode == "cpp"))
    with pytest.raises(ValueError):
        sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.ones((3, 4), dtype=bool))
    with pytest.raises(ValueError):
        sim.do_conditional_pauli_noise(
            stim.CircuitInstruction("X_ERROR", [0], [0.2]), np.ones((2, 4), dtype=bool)
        )
    with pytest.raises(ValueError):
        sim.postselect_z_batch(0, np.ones(3, dtype=bool), np.zeros(4, dtype=np.uint8))
    with pytest.raises(IndexError):
        sim.do_unmatched_noisy_reset_z(
            np.ones((2, 4), dtype=bool), np.zeros((2, 4), dtype=bool), "Z", order=[99]
        )
    with pytest.raises(ValueError):
        sim.apply_pauli("X", np.zeros((2, 4), dtype=bool), float("nan"))
    with pytest.raises(ValueError):
        sim.peek_pauli_batch([99], "Z")
    with pytest.raises(IndexError):
        sim.measurement_column(99)
    with pytest.raises(RuntimeError, match="not been fully run"):
        sim.get_final_measurement_records()
    with pytest.raises(RuntimeError, match="not been fully run"):
        sim.get_detector_flips()
    with pytest.raises(ValueError):
        sim.get_leakage_events()
    with pytest.raises(ValueError):
        sim.get_unleaked_to_leaked_records()

    sim.run()
    assert sim.get_final_measurement_records().shape == (4, 2)


def test_d36_clear_ordering_and_failed_engine_refusal() -> None:
    sim = _mk_sim("H 0\nM 0", batch_size=4, seed=11)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), np.zeros((1, 4), dtype=bool))
    _ = sim.active_shots
    assert not sim._cpp_dirty
    real_lib = sim._cpp_lib
    assert real_lib is not None

    class _FailAfterClear:
        def __init__(self) -> None:
            self.armed = True

        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_clear":
                return fn

            def wrapped(*args: Any) -> None:
                fn(*args)
                if self.armed:
                    self.armed = False
                    raise KeyboardInterrupt("post-engine-clear")

            return wrapped

    sim._cpp_lib = cast(ctypes.CDLL, _FailAfterClear())
    with pytest.raises(KeyboardInterrupt, match="post-engine-clear"):
        sim.clear()
    assert sim._cpp_dirty
    sim._cpp_lib = real_lib
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        _ = sim.active_shots
    sim.clear()
    assert not sim._cpp_dirty and sim.active_shots == {}

    class _FailedEngineLib:
        def __init__(self, code: int) -> None:
            self._code = code

        def __getattr__(self, name: str) -> Any:
            if name == "engine_failed":
                return lambda _ptr: self._code
            return getattr(real_lib, name)

    for code, pat in ((1, r"MemoryError.*new simulator"), (2, r"C\+\+ exception.*new simulator")):
        s_bad = _mk_sim("H 0\nM 0\nDETECTOR rec[-1]", batch_size=2, seed=12, record_events=True)
        s_bad._cpp_lib = cast(ctypes.CDLL, _FailedEngineLib(code))
        with pytest.raises(RuntimeError, match=pat):
            s_bad.clear()
        _assert_all_stateful_members_failed(s_bad, pat)


def test_r6_f2_do_conditional_pauli_noise_atomic_and_truth_value_masks() -> None:
    ops: list[tuple[str, stim.CircuitInstruction, int]] = [
        ("X_ERROR", stim.CircuitInstruction("X_ERROR", [0, 1, 2, 3], [0.5]), 4),
        ("Z_ERROR", stim.CircuitInstruction("Z_ERROR", [0, 1, 2, 3], [0.5]), 4),
        ("DEPOLARIZE1", stim.CircuitInstruction("DEPOLARIZE1", [0, 1, 2, 3], [0.6]), 4),
        (
            "PAULI_CHANNEL_1",
            stim.CircuitInstruction("PAULI_CHANNEL_1", [0, 1, 2, 3], [0.2, 0.2, 0.2]),
            4,
        ),
        ("DEPOLARIZE2", stim.CircuitInstruction("DEPOLARIZE2", [0, 1, 2, 3], [0.6]), 2),
        (
            "PAULI_CHANNEL_2",
            stim.CircuitInstruction("PAULI_CHANNEL_2", [0, 1, 2, 3], [0.04] * 15),
            2,
        ),
        (
            "E",
            stim.CircuitInstruction("E", [stim.target_x(0), stim.target_z(1)], [0.5]),
            2,
        ),
    ]
    batch = 16
    rng = np.random.default_rng(1)
    for sync in (False, True):
        for name, op, rows in ops:
            circ = stim.Circuit("H 0 1 2 3\n" + str(op) + "\nM 0 1 2 3")
            bool_mask = rng.random((rows, batch)) < 0.5
            sim_ref = _mk_sim(str(circ), batch_size=batch, seed=9, sync=sync)
            sim_ref._do_bare_instruction(circ[0])
            sim_ref.do_conditional_pauli_noise(op, bool_mask)
            xs_ref, zs_ref, *_ = sim_ref._flip_simulator.to_numpy(output_xs=True, output_zs=True)

            alt_masks: list[np.ndarray] = [
                bool_mask.astype(np.uint8),
                bool_mask.astype(np.int64) * 2,
                np.where(bool_mask, -1.5, 0.0),
                np.where(bool_mask, "x", None).astype(object),
            ]
            for alt in alt_masks:
                sim = _mk_sim(str(circ), batch_size=batch, seed=9, sync=sync)
                sim._do_bare_instruction(circ[0])
                sim.do_conditional_pauli_noise(op, alt)  # type: ignore[arg-type]
                xs, zs, *_ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
                assert np.array_equal(xs, xs_ref) and np.array_equal(zs, zs_ref), (name, sync)

            # Atomic rejections before cursor or RNG changes
            sim_rej = _mk_sim(str(circ), batch_size=batch, seed=9, sync=sync)
            sim_rej._do_bare_instruction(circ[0])
            rng_state_before = sim_rej.np_rng.bit_generator.state
            cursor_before = sim_rej._step_cursor
            for bad in (
                np.ones((rows + 1, batch), dtype=bool),
                np.ones((rows, 1), dtype=bool),
                np.ones(batch, dtype=bool),
            ):
                with pytest.raises(ValueError):
                    sim_rej.do_conditional_pauli_noise(op, bad)
            with pytest.raises(TypeError):
                sim_rej.do_conditional_pauli_noise(
                    op, np.full((rows, batch), "a")  # type: ignore[arg-type]
                )
            assert sim_rej._step_cursor == cursor_before
            assert sim_rej.np_rng.bit_generator.state == rng_state_before
            sim_rej.do_conditional_pauli_noise(op, bool_mask)
            xs_r, zs_r, *_ = sim_rej._flip_simulator.to_numpy(output_xs=True, output_zs=True)
            assert np.array_equal(xs_r, xs_ref) and np.array_equal(zs_r, zs_ref)


def test_r6_f7_rejected_first_interactive_do_allows_run() -> None:
    sim = _mk_sim("H 0\nM 0", batch_size=4, seed=5)
    with pytest.raises(ValueError):
        sim.interactive_do(stim.CircuitInstruction("Z", [0]))
    assert not sim._used_interactively and sim._unrolled_ops is None
    with pytest.raises(NotImplementedError):
        sim.interactive_do("H 0")  # type: ignore[arg-type]
    assert not sim._used_interactively and sim._unrolled_ops is None
    sim.run()
    assert sim.get_final_measurement_records().shape == (4, 1)


def test_ordered_z_reset_cpp_matches_python_and_leaves_unlisted_unreset() -> None:
    batch, seed = 16, 3

    def run_case(use_cpp: bool, order: list[int] | None) -> tuple[CosetsideSimulator, np.ndarray]:
        c = stim.Circuit("H 0 1\nCX 0 2 1 3\nTICK\nM 0 1 2 3")
        sim = _mk_sim(str(c), batch_size=batch, seed=seed, use_cpp=use_cpp)
        sim._do_bare_instruction(c[0])
        sim._do_bare_instruction(c[1])
        r0 = np.zeros((4, batch), dtype=bool)
        r0[[0, 1]] = True
        sim.do_unmatched_noisy_reset_z(r0, np.zeros((4, batch), dtype=bool), "Z", order=order)
        return sim, sim.peek_pauli_batch([0, 1, 2, 3], "Z")

    for order in (None, [0, 1], [1, 0], [1], []):
        s_cpp, z_cpp = run_case(True, order)
        s_py, z_py = run_case(False, order)
        assert np.array_equal(z_cpp, z_py)
        xs_c, zs_c, *_ = s_cpp._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        xs_p, zs_p, *_ = s_py._flip_simulator.to_numpy(output_xs=True, output_zs=True)
        assert np.array_equal(xs_c, xs_p) and np.array_equal(zs_c, zs_p)
        assert set(s_cpp.active_shots.keys()) == set(s_py.active_shots.keys())

    _, z_none = run_case(True, None)
    _, z_01 = run_case(True, [0, 1])
    _, z_10 = run_case(True, [1, 0])
    assert np.array_equal(z_none, z_01)
    assert np.array_equal(z_01[2], z_10[3]) and np.array_equal(z_01[3], z_10[2])

    _, z_only1 = run_case(True, [1])
    assert np.all(z_only1[1] == 1) and np.all(np.abs(z_only1[3]) == 1)
    assert np.all(z_only1[0] == 0) and np.all(z_only1[2] == 0)

    _, z_empty = run_case(True, [])
    assert np.all(z_empty == 0)


def test_r6_sc_c1_low1_past_end_cache_key_captured_before_compile_reentry() -> None:
    """R6-SC-C1-LOW-1: re-entering during `_past_end_step_meta` must not reuse a stale cache."""
    c_txt = "H 0\nTICK\nM 0 1"
    h1 = stim.CircuitInstruction("H", [1])
    cx = stim.CircuitInstruction("CX", [0, 1])

    ref = _mk_sim(c_txt, batch_size=4, seed=13)
    ref.run()
    ref.do_conditional_clifford(h1, np.ones((1, 4), bool))
    ref.do_conditional_clifford(cx, np.zeros((1, 4), bool))

    sim = _mk_sim(c_txt, batch_size=4, seed=13)
    sim.run()
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == "_past_end_step_meta":
            fired[0] = True
            sim.do_conditional_clifford(h1, np.ones((1, 4), bool))

    sys.setprofile(prof)
    try:
        sim.do_conditional_clifford(cx, np.zeros((1, 4), bool))
    finally:
        sys.setprofile(None)
    assert fired[0]
    for pauli in ("X", "Y", "Z"):
        assert np.array_equal(
            sim.peek_pauli_batch([0, 1], pauli), ref.peek_pauli_batch([0, 1], pauli)
        )
    assert len(sim._compiled_ref.steps) == len(ref._compiled_ref.steps) == 5


@pytest.mark.parametrize("use_cpp", [True, False])
def test_r6_sa_ts1_ma4_postselect_bad_desired_bits_rejected_when_flip_mask_all_false(
    use_cpp: bool,
) -> None:
    """R6-SA-TS-1 (`ma4`): `desired_bits` is validated even when `flip_mask` is all False."""
    sim = _mk_sim("H 0\nM 0", batch_size=4, seed=15, use_cpp=use_cpp)
    all_false = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="desired_bits"):
        sim.postselect_z_batch(0, all_false, np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="desired_bits"):
        sim.postselect_z_batch(0, all_false, np.zeros((2, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="desired_bits"):
        bad_db = cast(Any, np.zeros(4, dtype=np.float64))
        sim.postselect_z_batch(0, all_false, bad_db)


def test_r7_a1_high1_clear_init_rngs_reentry_gc_and_setprofile() -> None:
    """R7-A1-HIGH-1: re-entry into C++ RNG draws during clear()->_init_rngs() stays memory-safe."""
    n_q, batch = 2, 20000
    h_op = stim.CircuitInstruction("H", [0, 1])
    m_h = np.ones((2, batch), dtype=bool)
    r0 = np.zeros((n_q, batch), dtype=bool)
    r0[0, :] = True
    r1 = np.zeros((n_q, batch), dtype=bool)

    for target_k in (2, 5):
        sim_gc = _mk_sim("H 0 1\nM 0 1", batch_size=batch, seed=1)
        gc_state = {"armed": True, "fired": False, "n": 0}

        def gc_cb(phase: str, _info: Any, _tk: int = target_k, _s: Any = sim_gc,
                  _st: dict[str, Any] = gc_state) -> None:
            if _st["armed"] and not _st["fired"] and phase == "start":
                _st["n"] += 1
                if _st["n"] == _tk:
                    _st["fired"] = True
                    _s.do_conditional_clifford(h_op, m_h)
                    _s.do_unmatched_noisy_reset_z(r0, r1, "Z")

        old_thresh = gc.get_threshold()
        gc.collect()
        gc.callbacks.append(gc_cb)
        gc.set_threshold(1, 1, 1)
        gc_err: RuntimeError | None = None
        try:
            sim_gc.clear()
        except RuntimeError as exc:
            gc_err = exc
        finally:
            gc_state["armed"] = False
            gc.callbacks.remove(gc_cb)
            gc.set_threshold(*old_thresh)
        assert gc_state["fired"], f"gc_cb did not fire at target_k={target_k}"
        assert gc_err is not None and "changed during clear" in str(gc_err), (target_k, gc_err)

    sim_prof = _mk_sim("H 0 1\nM 0 1", batch_size=batch, seed=1)
    prof_fired = [False]
    prof_armed = [True]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if prof_armed[0] and not prof_fired[0] and event == "call":
            fb = frame.f_back
            if (
                fb is not None
                and fb.f_code.co_name == "_init_rngs"
                and frame.f_code.co_name == "data_as"
            ):
                prof_fired[0] = True
                sim_prof.do_conditional_clifford(h_op, m_h)
                sim_prof.do_unmatched_noisy_reset_z(r0, r1, "Z")
                sim_prof.clear()

    sys.setprofile(prof)
    try:
        with pytest.raises(RuntimeError, match="changed during clear"):
            sim_prof.clear()
    finally:
        prof_armed[0] = False
        sys.setprofile(None)
    assert prof_fired[0]
    sim_prof.clear()
    sim_prof.run()
    assert sim_prof.get_final_measurement_records().shape == (batch, 2)


def test_r7_a1_high1_sync_rngs_reentry_abort_preserves_cleared_tiebreak_rng() -> None:
    sim = _setup_export_reentry_sim(8, 4)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    fired = [False]

    class _GetRngClearProxy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_get_rng_states":
                return fn

            def wrapped(*args: Any) -> int:
                rc = int(fn(*args))
                if not fired[0]:
                    fired[0] = True
                    sim.clear()
                return rc

            return wrapped

    sim._cpp_lib = cast(ctypes.CDLL, _GetRngClearProxy())
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert fired[0]
    sim._cpp_lib = real_lib
    ref = _mk_sim("QUBIT_COORDS(0) 7", batch_size=4, seed=3)
    ref.clear()
    assert np.array_equal(sim._shot_tiebreak_rng, ref._shot_tiebreak_rng)


@pytest.mark.parametrize("sync", [False, True])
def test_r7_a1_med1_do_conditional_pauli_noise_oob_qubit_atomic(sync: bool) -> None:
    """R7-A1-MED-1: out-of-range qubits in do_conditional_pauli_noise raise atomically."""
    batch = 4
    cases: list[tuple[stim.CircuitInstruction, np.ndarray[Any, Any] | None]] = [
        (stim.CircuitInstruction("X_ERROR", [999], [0.5]), None),
        (
            stim.CircuitInstruction("X_ERROR", [999], [0.5]),
            np.array([[True, False, True, True]]),
        ),
        (stim.CircuitInstruction("Y_ERROR", [999], [0.5]), np.ones((1, batch), bool)),
        (stim.CircuitInstruction("Z_ERROR", [999], [0.5]), np.ones((1, batch), bool)),
        (stim.CircuitInstruction("DEPOLARIZE1", [999], [0.5]), None),
        (stim.CircuitInstruction("DEPOLARIZE2", [0, 999], [0.5]), np.ones((1, batch), bool)),
        (stim.CircuitInstruction("E", [stim.target_x(999)], [0.5]), np.ones((1, batch), bool)),
        (stim.CircuitInstruction("ELSE_CORRELATED_ERROR", [stim.target_z(999)], [0.5]), None),
        (stim.CircuitInstruction("I_ERROR", [999], [0.5]), None),
        (stim.CircuitInstruction("II_ERROR", [0, 999], [0.5]), np.ones((1, batch), bool)),
    ]
    for op, mask in cases:
        sim = _mk_sim("H 0 1\nM 0 1", batch_size=batch, seed=1, sync=sync)
        sim.run()
        c0 = sim._step_cursor
        st0 = len(sim._compiled_ref.steps)
        mut0 = sim._mutation_epoch
        rng0 = sim.np_rng.bit_generator.state
        corr0 = sim._correlated_error_occurred.copy()
        q0 = sim._flip_simulator.to_numpy(output_xs=True)[0].shape[0]
        with pytest.raises(IndexError, match="out of range"):
            sim.do_conditional_pauli_noise(op, mask)
        assert sim._step_cursor == c0
        assert len(sim._compiled_ref.steps) == st0
        assert sim._mutation_epoch == mut0
        assert sim.np_rng.bit_generator.state == rng0
        assert np.array_equal(sim._correlated_error_occurred, corr0)
        assert sim._flip_simulator.to_numpy(output_xs=True)[0].shape[0] == q0
        sim.do_conditional_pauli_noise(stim.CircuitInstruction("X_ERROR", [0], [0.5]), None)

    for op, mask in (
        (stim.CircuitInstruction("X_ERROR", [0, 999], [0.2]), None),
        (
            stim.CircuitInstruction("X_ERROR", [0, 999], [0.2]),
            np.array([[True, False, True, True], [False, True, False, True]]),
        ),
    ):
        sim_mid = _mk_sim("H 0 1\nX_ERROR(0.2) 0 1\nM 0 1", batch_size=batch, seed=1, sync=sync)
        sim_mid._do_bare_instruction(stim.CircuitInstruction("H", [0, 1]))
        assert sim_mid._step_cursor == 1
        c0 = sim_mid._step_cursor
        st0 = len(sim_mid._compiled_ref.steps)
        mut0 = sim_mid._mutation_epoch
        rng0 = sim_mid.np_rng.bit_generator.state
        with pytest.raises(IndexError, match="out of range"):
            sim_mid.do_conditional_pauli_noise(op, mask)
        assert sim_mid._step_cursor == c0
        assert len(sim_mid._compiled_ref.steps) == st0
        assert sim_mid._mutation_epoch == mut0
        assert sim_mid.np_rng.bit_generator.state == rng0
        sim_mid.do_conditional_pauli_noise(stim.CircuitInstruction("X_ERROR", [0, 1], [0.2]), None)
        assert sim_mid._step_cursor == c0 + 1


def test_r7_a2_sync_get_rng_nonzero_rc_aborts_export() -> None:
    """R7-A2-TS-2: non-zero rc from engine_get_rng_states aborts export without clearing dirty."""
    sim = _setup_export_reentry_sim(8, 4)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    rng_before = sim._shot_tiebreak_rng.copy()
    called = [False]

    class _BadGetRngRcProxy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_get_rng_states":
                return fn

            def wrapped(*_args: Any) -> int:
                called[0] = True
                return -3

            return wrapped

    sim._cpp_lib = cast(ctypes.CDLL, _BadGetRngRcProxy())
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert called[0]
    assert sim._cpp_dirty
    assert np.array_equal(sim._shot_tiebreak_rng, rng_before)
    sim._cpp_lib = real_lib
    assert 0 in sim.active_shots


@pytest.mark.parametrize("sync", [False, True])
def test_r8_b3_do_conditional_pauli_noise_boundary_q_eq_num_qubits(sync: bool) -> None:
    """T2 (B02): q == num_qubits is rejected atomically and does not grow the frame simulator."""
    batch = 4
    m1 = np.ones((1, batch), dtype=bool)
    m2 = np.array([[True, False, True, True], [False, True, False, True]])
    cases: list[tuple[stim.CircuitInstruction, np.ndarray[Any, Any] | None]] = [
        (stim.CircuitInstruction("X_ERROR", [2], [0.5]), None),
        (stim.CircuitInstruction("X_ERROR", [2], [0.5]), m1),
        (stim.CircuitInstruction("DEPOLARIZE1", [2], [0.5]), None),
        (stim.CircuitInstruction("DEPOLARIZE1", [2], [0.5]), m1),
        (stim.CircuitInstruction("I_ERROR", [2], [0.5]), None),
        (stim.CircuitInstruction("I_ERROR", [2], [0.5]), m1),
        (stim.CircuitInstruction("DEPOLARIZE2", [0, 2], [0.5]), None),
        (stim.CircuitInstruction("DEPOLARIZE2", [0, 2], [0.5]), m1),
        (stim.CircuitInstruction("II_ERROR", [0, 2], [0.5]), None),
        (stim.CircuitInstruction("II_ERROR", [0, 2], [0.5]), m1),
        (stim.CircuitInstruction("E", [stim.target_x(2)], [0.5]), None),
        (stim.CircuitInstruction("E", [stim.target_x(2)], [0.5]), m1),
    ]
    for op, mask in cases:
        sim = _mk_sim("H 0 1\nM 0 1", batch_size=batch, seed=1, sync=sync)
        sim.run()
        c0, st0, mut0 = sim._step_cursor, len(sim._compiled_ref.steps), sim._mutation_epoch
        rng0 = sim.np_rng.bit_generator.state
        corr0 = sim._correlated_error_occurred.copy()
        with pytest.raises(IndexError, match="out of range"):
            sim.do_conditional_pauli_noise(op, mask)
        assert (sim._step_cursor, len(sim._compiled_ref.steps), sim._mutation_epoch) == (
            c0, st0, mut0
        )
        assert sim.np_rng.bit_generator.state == rng0
        assert np.array_equal(sim._correlated_error_occurred, corr0)
        assert sim._flip_simulator.to_numpy(output_xs=True)[0].shape[0] == 2

    for mask in (None, m2):
        sim_mid = _mk_sim("H 0 1\nX_ERROR(0.2) 0 1\nM 0 1", batch_size=batch, seed=1, sync=sync)
        sim_mid._do_bare_instruction(stim.CircuitInstruction("H", [0, 1]))
        c0, st0, mut0 = (
            sim_mid._step_cursor, len(sim_mid._compiled_ref.steps), sim_mid._mutation_epoch
        )
        with pytest.raises(IndexError, match="out of range"):
            sim_mid.do_conditional_pauli_noise(
                stim.CircuitInstruction("X_ERROR", [0, 2], [0.2]), mask
            )
        assert (
            sim_mid._step_cursor,
            len(sim_mid._compiled_ref.steps),
            sim_mid._mutation_epoch,
        ) == (c0, st0, mut0)
        assert sim_mid._flip_simulator.to_numpy(output_xs=True)[0].shape[0] == 2


def test_r8_b3_ensure_run_ops_checks_op_equality_with_shared_compiled_ref() -> None:
    """T3 (E03): _ensure_run_ops keeps circuit's op when shared compiled_ref differs."""
    ref = CompiledReferenceCircuit(stim.Circuit("X_ERROR(0.0) 0\nM 0"))
    circ = stim.Circuit("X_ERROR(1.0) 0\nM 0")
    h = LeakageUint8Coset().compile_op_handler(circuit=circ, batch_size=4)
    sim = CosetsideSimulator(circ, compiled_op_handler=h, batch_size=4, seed=1, compiled_ref=ref)
    sim.run()
    assert np.all(sim.get_final_measurement_records())


@pytest.mark.parametrize(
    ("circ_text", "hook_fn_name"),
    [
        ("X_ERROR(1.0) 0\nM 0", "do_conditional_pauli_noise"),
        ("QUBIT_COORDS(0, 0) 0\nX_ERROR(1.0) 0\nM 0", "apply_coords_op"),
    ],
)
def test_r8_b3_interactive_do_interrupted_at_t0_keeps_used_and_blocks_run(
    circ_text: str, hook_fn_name: str
) -> None:
    """T4a (I05): interrupting interactive_do at t==0 after mutation keeps _used_interactively."""
    sim = _mk_sim(circ_text, batch_size=4, seed=2)
    fired = [False]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "return" and frame.f_code.co_name == hook_fn_name:
            fired[0] = True
            raise KeyboardInterrupt(f"interrupt after {hook_fn_name}")

    sys.setprofile(prof)
    try:
        with pytest.raises(KeyboardInterrupt, match=hook_fn_name):
            sim.interactive_do(sim.circuit)
    finally:
        sys.setprofile(None)
    assert fired[0]
    assert sim._circuit_time == 0
    assert sim._used_interactively is True
    with pytest.raises(RuntimeError, match=r"KeyboardInterrupt.*clear\(\)"):
        sim.run()
    sim.clear()
    sim.run()
    assert np.all(sim.get_final_measurement_records())


def test_r8_b3_noise_reentry_during_active_shots_export_detected() -> None:
    """T4b (P12): re-entering do_conditional_pauli_noise during active_shots export raises."""
    sim = _setup_export_reentry_sim(8, 2)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    noise_op = stim.CircuitInstruction("X_ERROR", [0], [0.5])
    proxy = _ExportReentryProxy(
        real_lib, "before", lambda: sim.do_conditional_pauli_noise(noise_op, None)
    )
    sim._cpp_lib = cast(ctypes.CDLL, proxy)
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert proxy.fired
    assert sim._cpp_dirty is True
    sim._cpp_lib = real_lib
    assert 0 in sim.active_shots


@pytest.mark.parametrize("with_clear", [True, False])
def test_r8_b3_reentrant_clear_mutate_active_shots_hook_keeps_consistent(
    with_clear: bool,
) -> None:
    """T1 (S15, S17): re-entrant clear()+mutate+active_shots inside export stays consistent."""
    n_q, batch = 24, 2
    sim = _setup_export_reentry_sim(n_q, batch)
    h_all = stim.CircuitInstruction("H", list(range(n_q)))
    m_one = np.ones((n_q, batch), dtype=bool)
    m_one[0, 0] = False
    fired = [False]
    inner_len = [-1]

    def prof(frame: Any, event: str, _arg: Any) -> None:
        if not fired[0] and event == "call" and frame.f_code.co_name == "_install_export":
            fired[0] = True
            if with_clear:
                sim.clear()
            sim.do_conditional_clifford(h_all, m_one)
            inner_len[0] = len(sim.active_shots)

    sys.setprofile(prof)
    try:
        with pytest.raises(RuntimeError, match="changed during active_shots export"):
            _ = sim.active_shots
    finally:
        sys.setprofile(None)
    assert fired[0] and inner_len[0] == (1 if with_clear else 2)
    assert sim._active_shots == {}
    assert sim._cpp_dirty is True
    assert 0 in sim.active_shots


@pytest.mark.parametrize("mode", ["neg4", "d_plus_1"])
def test_r8_b3_export_shot_v3_bad_rc_without_epoch_change_aborts(mode: str) -> None:
    """T10 (S04, S05): rc != d from engine_export_shot_v3 aborts even when epochs are unchanged."""
    sim = _setup_export_reentry_sim(8, 2)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    called = [False]

    class _BadExportRcProxy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(real_lib, name)
            if name != "engine_export_shot_v3":
                return fn

            def wrapped(*args: Any) -> int:
                called[0] = True
                real_rc = int(fn(*args))
                return -4 if mode == "neg4" else real_rc + 1

            return wrapped

    sim._cpp_lib = cast(ctypes.CDLL, _BadExportRcProxy())
    with pytest.raises(RuntimeError, match="changed during active_shots export"):
        _ = sim.active_shots
    assert called[0]
    assert sim._active_shots == {}
    assert sim._cpp_dirty is True
    sim._cpp_lib = real_lib
    assert 0 in sim.active_shots


def test_r8_b3_init_rngs_nonzero_rc_raises_runtime_error() -> None:
    """T10 (R01): non-zero rc from engine_set_rng_states (with engine_failed==0) raises."""
    sim = _mk_sim("H 0\nM 0", batch_size=2, seed=4)
    real_lib = sim._cpp_lib
    assert real_lib is not None
    called = [False]

    class _BadSetRngProxy:
        def __getattr__(self, name: str) -> Any:
            if name == "engine_set_rng_states":
                def wrapped(*_args: Any) -> int:
                    called[0] = True
                    return -1
                return wrapped
            if name == "engine_failed":
                return lambda _ptr: 0
            return getattr(real_lib, name)

    sim._cpp_lib = cast(ctypes.CDLL, _BadSetRngProxy())
    with pytest.raises(RuntimeError, match="engine_set_rng_states failed with code -1"):
        sim.clear()
    assert called[0]
    sim._cpp_lib = real_lib
    sim.clear()
    sim.run()
    assert sim.get_final_measurement_records().shape == (2, 1)
