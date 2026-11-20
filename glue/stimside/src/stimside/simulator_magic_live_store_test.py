"""The private engine switch of `simulator_magic`: results must not depend on it.

With `_LIVE_STORE_ENGINE` on, the live per-shot expansions live in the C++ MagicBatchEngine behind
the `_live_exps` proxy. Every golden case is recomputed in that mode and compared with the blessed
digests, and the engine commands are checked to really run (a switch that silently fell back to
the Python list would pass the digest comparison vacuously).
"""

from __future__ import annotations

import collections
import copy
import json
import math
import threading
import types
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside
import stimside._native.magic_engine as me
import stimside.simulator_magic as sm
from stimside import golden_test
from stimside._magic.gates import CliffordFaultChannel, FaultBranch
from stimside._magic.live_store import LiveExpansions
from stimside._magic.phase_qfe import ShotExpansion
from stimside._magic.query import QueryBudget, QueryBudgetExceededError
from stimside.op_handlers.abstract_op_handler import CompiledOpHandler, OpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside.simulator_magic import MagicsideSimulator
from stimside.util.circuit import unroll_circuit

_PAULI_CIRCUIT = "R 0 1 2\nH 0\nI[T] 0\nCX 0 1\nX_ERROR(0.3) 0 1\nM 0 1 2\nDETECTOR rec[-1]\n"


class _CountingStore(LiveExpansions):
    """A `LiveExpansions` that counts how often the engine Pauli command ran."""

    calls = 0

    def apply_pauli_mask(self, axis, mask) -> None:  # type: ignore[no-untyped-def]
        type(self).calls += 1
        super().apply_pauli_mask(axis, mask)


def _engine_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    monkeypatch.setattr(sm, "LiveExpansions", _CountingStore)
    _CountingStore.calls = 0


def test_every_golden_digest_is_unchanged_with_the_engine_switch_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine_on(monkeypatch)
    actual = golden_test._compute_all_digests()
    expected = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
    assert sum("magic_ds" in modes for modes in expected.values()) >= 10
    assert _CountingStore.calls >= 20  # heralded noise / feedback really went through the engine


def _noisy_run(engine: bool, monkeypatch: pytest.MonkeyPatch, seed: int = 5):
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=16, seed=seed)
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
    sim.finish_interactive_run()
    return sim, [sim.get_qfe_expansion(b)[1] for b in range(16)]


def test_a_noisy_magic_run_has_identical_expansions_and_rng_state_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref, ref_exps = _noisy_run(False, monkeypatch)
    eng, eng_exps = _noisy_run(True, monkeypatch)
    assert isinstance(eng._live_exps, LiveExpansions) and isinstance(ref._live_exps, list)
    assert [c.tobytes() for c in eng._measurement_columns] == [
        c.tobytes() for c in ref._measurement_columns
    ]
    assert eng._log_branch_masses.tobytes() == ref._log_branch_masses.tobytes()
    assert eng.np_rng.bit_generator.state == ref.np_rng.bit_generator.state
    for a, b in zip(ref_exps, eng_exps):
        assert a.weights.view(np.uint64).tobytes() == b.weights.view(np.uint64).tobytes()
        assert a.h_words.tobytes() == b.h_words.tobytes() and a.ell.tobytes() == b.ell.tobytes()
        assert a.A_words.tobytes() == b.A_words.tobytes() and a.d == b.d and a.chi == b.chi


def test_clear_reuses_one_engine_and_resets_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=1)
    assert isinstance(sim._live_exps, LiveExpansions)
    engine = sim._live_exps.engine
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
    sim.finish_interactive_run()
    sim.clear()
    assert isinstance(sim._live_exps, LiveExpansions)
    assert sim._live_exps.engine is engine  # reset in place, not recreated
    zero = ShotExpansion.zero(sim._eff_nq, atol=sim.atol)
    assert all(
        e.weights.tobytes() == zero.weights.tobytes() and e.chi == 1 and e.d == 0
        for e in sim._live_exps
    )


@pytest.mark.skipif(
    sm._LIVE_STORE_ENGINE is not True,
    reason="pins the ON default: forcing the switch OFF (to run the suite on the reference "
    "list) skips this default pin, not a regression",
)
def test_the_switch_is_private_and_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import stimside
    from stimside._magic.live_store import engine_numerics_available

    assert sm._LIVE_STORE_ENGINE is True
    assert not any("live" in n.lower() or "engine" in n.lower() for n in stimside.__all__)
    circ = stim.Circuit("H 0\nI[T] 0\nH 0\nM 0\n")
    sim = MagicsideSimulator(circ, batch_size=2, seed=1)
    if engine_numerics_available():
        assert isinstance(sim._live_exps, LiveExpansions)
    else:
        assert isinstance(sim._live_exps, list)
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", False)
    ref = MagicsideSimulator(circ, batch_size=2, seed=1)
    assert isinstance(ref._live_exps, list)


def test_with_the_switch_on_the_realization_style_chi_zero_assignment_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The white-box flow of simulator_magic_realization_test.py, on the engine-backed store.
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nH 0\nM 0"), batch_size=2, seed=1)
    sim.interactive_do(stim.Circuit("H 0\nI[T] 0\nH 0"))
    sim._flush_magic()
    assert isinstance(sim._live_exps, LiveExpansions)
    sim._live_exps[0] = ShotExpansion.from_components((), n_qubits=sim._eff_nq, merge=False)
    sim.interactive_do(stim.Circuit("M 0"))
    sim.finish_interactive_run()
    assert sim.get_branch_mass(0) == 0.0
    assert sim.get_branch_mass(1) > 0.0


_FEEDBACK_CIRCUIT = stim.Circuit(
    "R 0 1\nH 0\nI[T] 0\nH 0\nH 1\nI[T] 1\nH 1\nM 1\nCX rec[-1] 0\n"
)


def _feedback_run(
    engine: bool, monkeypatch: pytest.MonkeyPatch, seen: list[np.ndarray]
) -> tuple[MagicsideSimulator, np.ndarray, np.ndarray]:
    """Run to the feedback gate, then apply it through a per-group mask (`mask_g`)."""
    real = MagicsideSimulator._apply_axis_to_shots

    def spy(self: MagicsideSimulator, ax: Any, mask: np.ndarray) -> None:
        seen.append(np.array(mask, copy=True))
        real(self, ax, mask)

    with monkeypatch.context() as patch:  # each run wraps the real method, never an older spy
        patch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        patch.setattr(MagicsideSimulator, "_apply_axis_to_shots", spy)
        sim = MagicsideSimulator(_FEEDBACK_CIRCUIT, batch_size=8, seed=3)
        sim.interactive_do(_FEEDBACK_CIRCUIT[:-1])
        measured = sim._measurement_columns[-1].copy()
        mask_g = np.array([[True, False] * 4])  # one group (one CX pair), conditioned per shot
        seen.clear()
        sim.do_conditional_clifford(_FEEDBACK_CIRCUIT[-1], mask_g)
        sim._flush_magic()
    return sim, measured, mask_g[0]


def test_the_mask_g_conditioned_feedback_pauli_reaches_the_engine_with_the_anded_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_seen: list[np.ndarray] = []
    eng_seen: list[np.ndarray] = []
    ref, measured, mask_g = _feedback_run(False, monkeypatch, ref_seen)
    eng, measured_eng, _ = _feedback_run(True, monkeypatch, eng_seen)
    assert measured.tobytes() == measured_eng.tobytes()
    want = measured & mask_g
    # The path is only exercised if mask_g really removes shots the measurement alone selects.
    assert (measured & ~mask_g).any() and want.any() and not want.all()
    assert len(ref_seen) == len(eng_seen) == 1
    assert ref_seen[0].tolist() == eng_seen[0].tolist() == want.tolist()
    assert isinstance(eng._live_exps, LiveExpansions) and isinstance(ref._live_exps, list)
    assert all(_same_bytes(ref._live_exps[b], eng._live_exps[b]) for b in range(8))


def _same_bytes(x: ShotExpansion, y: ShotExpansion) -> bool:
    return (
        x.weights.view(np.uint64).tobytes() == y.weights.view(np.uint64).tobytes()
        and x.h_words.tobytes() == y.h_words.tobytes() and x.ell.tobytes() == y.ell.tobytes()
        and x.log_mass.hex() == y.log_mass.hex() and x.chi == y.chi and x.d == y.d
    )


def test_copy_copy_with_the_switch_on_gives_the_copy_its_own_engine_and_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=2)
    sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT)[:4])
    sim._flush_magic()
    twin = copy.copy(sim)
    assert isinstance(sim._live_exps, LiveExpansions)
    assert isinstance(twin._live_exps, LiveExpansions)
    assert twin._live_exps is twin._live_store and twin._live_store is not sim._live_store
    assert twin._live_exps.engine is not sim._live_exps.engine
    before = [sim._live_exps[b] for b in range(4)]
    assert any(e.chi > 1 or e.h_words.any() for e in before)  # not just the zero state
    assert all(_same_bytes(a, twin._live_exps[b]) for b, a in enumerate(before))
    twin._live_exps.close()  # the copy lets go of its engine ...
    twin.clear()  # ... and is cleared, which resets its own engine only
    assert all(_same_bytes(a, sim._live_exps[b]) for b, a in enumerate(before))
    assert sim._live_exps.engine is not twin._live_exps.engine


def test_copy_copy_with_the_switch_off_gives_independent_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", False)
    circ = stim.Circuit(_PAULI_CIRCUIT)
    sim = MagicsideSimulator(circ, batch_size=2, seed=2)
    sim.interactive_do(circ[:4])
    twin = copy.copy(sim)
    assert type(twin) is type(sim)
    assert twin.__dict__.keys() == sim.__dict__.keys()
    assert isinstance(sim._live_exps, list) and isinstance(twin._live_exps, list)
    assert twin._live_exps is not sim._live_exps
    assert all(a is not b and _same_bytes(a, b) for a, b in zip(sim._live_exps, twin._live_exps))
    for attr in ("_pending_magic", "_measurement_columns", "_log_branch_masses", "_corr_occurred"):
        assert getattr(twin, attr) is not getattr(sim, attr)
    assert twin._live_ref_tab is not sim._live_ref_tab
    assert twin.np_rng is not sim.np_rng
    assert twin._compiled_op_handler is not sim._compiled_op_handler
    twin.interactive_do(circ[4:])
    twin.finish_interactive_run()
    twin_recs = twin.get_final_measurement_records().copy()
    twin.clear()
    sim.interactive_do(circ[4:])
    sim.finish_interactive_run()
    np.testing.assert_array_equal(sim.get_final_measurement_records(), twin_recs)


def test_copy_copy_after_a_compiled_run_replaced_the_store_by_a_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=2, seed=2)
    sim._live_exps = [sim._live_exps[b] for b in range(2)]  # what the compiled paths leave
    twin = copy.copy(sim)
    assert twin._live_exps is not sim._live_exps
    assert isinstance(twin._live_exps, list)
    assert all(a is not b and _same_bytes(a, b) for a, b in zip(sim._live_exps, twin._live_exps))
    assert twin._live_store is not sim._live_store  # never share the native store
    assert isinstance(twin._live_store, LiveExpansions)


def test_a_library_that_cannot_load_makes_the_switch_fall_back_to_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(force_recompile: bool = False) -> Any:
        raise OSError("libmagic_kernels.so: cannot open shared object file")

    me.numerics_match_numpy.cache_clear()
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    monkeypatch.setattr(me, "get_magic_engine_lib", broken)
    try:
        sim = MagicsideSimulator(stim.Circuit(_PAULI_CIRCUIT), batch_size=4, seed=5)
        assert isinstance(sim._live_exps, list)  # silent fallback, not an OSError
        sim.interactive_do(stim.Circuit(_PAULI_CIRCUIT))
        sim.finish_interactive_run()
        sim.clear()
        assert isinstance(sim._live_exps, list)
    finally:
        monkeypatch.undo()
        me.numerics_match_numpy.cache_clear()
    assert me.numerics_match_numpy()


@pytest.mark.parametrize("engine", [False, True])
def test_copy_copy_with_pending_magic_and_read_on_copy_preserves_original(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("H 0\nI[T] 0\nH 0\nM 0")
    batch = 512
    ref = MagicsideSimulator(circ, batch_size=batch, seed=11)
    ref.run()
    sim = MagicsideSimulator(circ, batch_size=batch, seed=11)
    sim.interactive_do(circ[:2])  # H 0; I[T] 0 (pending magic)
    twin = copy.copy(sim)
    twin.state_vector(0)  # reading flushes pending magic on twin only
    twin.interactive_do(circ[2:])
    twin.finish_interactive_run()
    sim.interactive_do(circ[2:])
    sim.finish_interactive_run()
    want = ref.get_final_measurement_records()
    np.testing.assert_array_equal(sim.get_final_measurement_records(), want)
    np.testing.assert_array_equal(twin.get_final_measurement_records(), want)
    frac = float(np.asarray(want).mean())
    assert abs(frac - math.sin(math.pi / 8) ** 2) < 0.08


def _boundary_random_circuit(seed: int) -> stim.Circuit:
    rng = np.random.default_rng(seed)
    n = 4
    lines = ["R 0 1 2 3", "H 0 1 2 3"]
    n_meas = 0
    for _ in range(10):
        kind = str(
            rng.choice(["t", "ccz", "u1", "noise", "leak1", "cond1", "cond2", "swap", "meas", "fb"])
        )
        if kind == "t":
            q = int(rng.integers(0, n))
            lines.append(f"I[{rng.choice(['T', 'T_DAG'])}] {q}")
        elif kind == "ccz":
            qs = " ".join(str(int(q)) for q in rng.choice(n, size=3, replace=False))
            lines.append(f"I[CCZ] {qs}")
        elif kind == "u1":
            lines.append(f"{rng.choice(['H', 'S', 'SQRT_X'])} {int(rng.integers(0, n))}")
        elif kind == "noise":
            q = int(rng.integers(0, n))
            lines.append(f"{rng.choice(['X_ERROR(0.25)', 'DEPOLARIZE1(0.2)'])} {q}")
        elif kind == "leak1":
            q = int(rng.integers(0, n))
            lines.append(f"I[LEAKAGE_TRANSITION_1: (0.3, U-->2) (0.4, 2-->U)] {q}")
        elif kind == "cond1":
            q = int(rng.integers(0, n))
            lines.append(f"H[CONDITIONED_ON_SELF: {rng.choice(['U', '0', '1'])}] {q}")
        elif kind == "cond2":
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            lines.append(f"CX[CONDITIONED_ON_PAIR: (U, U)] {a} {b}")
        elif kind == "swap":
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            lines.append(f"SWAP[LEAKAGE_SWAP] {a} {b}")
        elif kind == "meas":
            q = int(rng.integers(0, n))
            n_meas += 1
            if rng.random() < 0.5:
                lines.append(f"M[LEAKAGE_PROJECTION_Z: (1.0, 2) (0.85, 1)] {q}")
            else:
                lines.append(f"MR {q}")
        elif kind == "fb" and n_meas > 0:
            lines.append(f"CX rec[-1] {int(rng.integers(0, n))}")
    lines.extend(["M 0 1 2 3", "DETECTOR rec[-1] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-3]"])
    return stim.Circuit("\n".join(lines))


def _assert_matches_uncopied(sim: MagicsideSimulator, ref: MagicsideSimulator) -> None:
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(sim.get_detector_flips(), ref.get_detector_flips())
    np.testing.assert_array_equal(sim.get_observable_flips(), ref.get_observable_flips())
    assert [a.tolist() for a in sim.get_unleaked_to_leaked_records()] == [
        a.tolist() for a in ref.get_unleaked_to_leaked_records()
    ]
    assert sim.get_leakage_events() == ref.get_leakage_events()
    for b in range(sim.batch_size):
        assert math.isclose(sim.get_branch_mass(b), ref.get_branch_mass(b), rel_tol=1e-12)
        np.testing.assert_allclose(sim.state_vector(b), ref.state_vector(b), atol=1e-11)


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("sync", [False, True])
def test_copy_copy_at_every_instruction_boundary_equals_uncopied_twin(
    monkeypatch: pytest.MonkeyPatch, engine: bool, sync: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = _boundary_random_circuit(seed=2026 if not sync else 2027)
    ops = unroll_circuit(circ)
    kw: dict[str, Any] = {
        "op_handler": LeakageUint8Magic(),
        "batch_size": 4,
        "seed": 17,
        "sync_tableside_rng": sync,
        "record_unleaked_to_leaked": True,
        "record_leakage_events": True,
    }
    uncopied = MagicsideSimulator(circ, **kw)
    for op in ops:
        uncopied.interactive_do(op)
    uncopied.finish_interactive_run()

    for k in range(len(ops) + 1):
        sim = MagicsideSimulator(circ, **kw)
        for op in ops[:k]:
            sim.interactive_do(op)
        twin = copy.copy(sim)
        assert twin.compiled_op_handler is twin._compiled_op_handler
        assert twin.compiled_op_handler is not sim.compiled_op_handler
        if sync:
            assert twin._shot_np_rngs is not None and sim._shot_np_rngs is not None
            assert twin.np_rng is twin._shot_np_rngs[0]
            assert twin._shot_np_rngs[0] is not sim._shot_np_rngs[0]
            assert twin._tab_rngs is not None and sim._tab_rngs is not None
            assert twin._tab_rngs[0] is not sim._tab_rngs[0]
        twin.peek_pauli_born_batch([0], "Z")
        twin.state_vector(0)
        twin.current_tableau_measurement_record()
        for op in ops[k:]:
            sim.interactive_do(op)
            twin.interactive_do(op)
        sim.finish_interactive_run()
        twin.finish_interactive_run()
        _assert_matches_uncopied(sim, uncopied)
        _assert_matches_uncopied(twin, uncopied)


@pytest.mark.parametrize("engine", [False, True])
def test_copy_copy_custom_compiled_op_handler_and_aux_fields_are_independent(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    class _CountingHandler(CompiledOpHandler[Any]):
        def __init__(self, op: stim.CircuitInstruction | None = None) -> None:
            self.seen: list[str] = []
            self.buf = np.zeros(2, dtype=np.int64)
            self.unpicklable = op

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            self.seen.append(op.name)
            self.buf[0] += 1
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            self.seen.clear()

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("QUBIT_COORDS(1, 2) 0\nSHIFT_COORDS(1, 0)\nH 0\nM 0")
    h1 = _CountingHandler()
    sim = MagicsideSimulator(circ, compiled_op_handler=h1, batch_size=2, seed=5)
    sim.interactive_do(circ[:3])
    sim._cached_m2d = (np.zeros((2, 1), dtype=np.bool_), np.zeros((2, 0), dtype=np.bool_))
    twin = copy.copy(sim)
    assert twin.compiled_op_handler is not sim.compiled_op_handler
    assert twin._cached_m2d is not sim._cached_m2d
    assert twin.qubit_coords[0] is not sim.qubit_coords[0]
    assert twin.coords_shifts is not sim.coords_shifts
    twin.interactive_do(circ[3:])
    twin.finish_interactive_run()
    assert len(h1.seen) == 1 and int(h1.buf[0]) == 1

    # Verify deep cloning when a custom handler holds stim.CircuitInstruction.
    h2 = _CountingHandler(circ[2])
    sim2 = MagicsideSimulator(circ, compiled_op_handler=h2, batch_size=2, seed=5)
    twin2 = copy.copy(sim2)
    h2_copy = twin2.compiled_op_handler
    assert isinstance(h2_copy, _CountingHandler)
    assert h2_copy is not h2 and h2_copy.unpicklable == circ[2]
    twin2.interactive_do(circ)
    twin2.finish_interactive_run()
    assert h2.seen == [] and int(h2.buf[0]) == 0
    assert h2_copy.seen == ["H", "M"] and int(h2_copy.buf[0]) == 2

    h3 = _CountingHandler(circ[2])
    sim3 = MagicsideSimulator(circ, compiled_op_handler=h3, batch_size=2, seed=5)
    sim3.interactive_do(circ[:2])
    twin3 = copy.copy(sim3)
    h3_copy = twin3.compiled_op_handler
    assert isinstance(h3_copy, _CountingHandler) and h3_copy is not h3
    twin3.interactive_do(circ[2:])
    twin3.finish_interactive_run()
    assert h3.seen == [] and int(h3.buf[0]) == 0
    assert h3_copy.seen == ["H", "M"] and int(h3_copy.buf[0]) == 2
    sim3.interactive_do(circ[2:])
    sim3.finish_interactive_run()
    assert h3.seen == ["H", "M"] and int(h3.buf[0]) == 2


class _NonMarkovianStimHandler(CompiledOpHandler[Any]):
    """Stateful custom handler holding immutable Stim types in nested containers."""

    def __init__(self, circuit: stim.Circuit, wrapper: OpHandler[Any] | None = None) -> None:
        ops = list(circuit)
        first_inst = next(op for op in ops if isinstance(op, stim.CircuitInstruction))
        repeat_blk = next(op for op in ops if isinstance(op, stim.CircuitRepeatBlock))
        gt = first_inst.targets_copy()[0]
        dem_inst = stim.DetectorErrorModel("error(0.1) D0")[0]
        dem_rep = stim.DetectorErrorModel("repeat 2 {\n error(0.1) D0\n}")[0]
        self.parsed_ops: dict[stim.CircuitInstruction, float] = {
            op: (op.gate_args_copy()[0] if op.gate_args_copy() else 0.0)
            for op in ops
            if isinstance(op, stim.CircuitInstruction)
        }
        self.nested: dict[str, Any] = {
            "targets": tuple(first_inst.targets_copy()),
            "repeat": repeat_blk,
            "target_set": {gt},
            "frozen_ops": frozenset([first_inst]),
            "queue": collections.deque([first_inst, repeat_blk]),
            "dem": (dem_inst, dem_rep, dem_inst.targets_copy()[0]),
            "meta": (stim.gate_data("H"), stim.Flow("X -> Z")),
            "obj_arr": np.array([first_inst], dtype=object),
        }
        self.wrapper = wrapper
        self.history: list[str] = []
        self.counters = np.zeros(4, dtype=np.int64)

    def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
        self.history.append(op.name)
        self.counters[0] += 1
        sss._do_bare_instruction(op)
        if op.name == "I" and (int(self.counters[0]) % 2 == 0):
            sss.apply_pauli("X", np.ones((1, sss.batch_size), dtype=bool))

    def clear(self) -> None:
        self.history.clear()
        self.counters[:] = 0


class _WrappingNonMarkovianOpHandler(OpHandler[Any]):
    def __init__(self, circuit: stim.Circuit) -> None:
        self.cached_ops = list(circuit)
        self.extra_tags = [op.targets_copy()[0] for op in circuit if op.name == "R"]

    def compile_op_handler(
        self, *, circuit: stim.Circuit, batch_size: int
    ) -> CompiledOpHandler[Any]:
        return _NonMarkovianStimHandler(circuit, wrapper=self)


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("via", ["compiled", "op_handler"])
def test_copy_copy_nonmarkovian_stim_handler_is_independent(
    monkeypatch: pytest.MonkeyPatch, engine: bool, via: str
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("R 0\nREPEAT 2 {\n H 0\n}\nI 0\nM 0\nDETECTOR rec[-1]")
    if via == "compiled":
        sim = MagicsideSimulator(
            circ, compiled_op_handler=_NonMarkovianStimHandler(circ), batch_size=3, seed=11
        )
        ref = MagicsideSimulator(
            circ, compiled_op_handler=_NonMarkovianStimHandler(circ), batch_size=3, seed=11
        )
    else:
        sim = MagicsideSimulator(
            circ, op_handler=_WrappingNonMarkovianOpHandler(circ), batch_size=3, seed=11
        )
        ref = MagicsideSimulator(
            circ, op_handler=_WrappingNonMarkovianOpHandler(circ), batch_size=3, seed=11
        )
    sim.interactive_do(circ[:1])
    ref.interactive_do(circ[:1])
    twin = copy.copy(sim)
    sim_h = sim.compiled_op_handler
    twin_h = twin.compiled_op_handler
    assert isinstance(sim_h, _NonMarkovianStimHandler)
    assert isinstance(twin_h, _NonMarkovianStimHandler)
    assert twin_h is not sim_h
    assert twin_h.history is not sim_h.history and twin_h.counters is not sim_h.counters
    assert twin_h.nested["queue"] is not sim_h.nested["queue"]
    assert twin_h.nested["target_set"] is not sim_h.nested["target_set"]
    if via == "op_handler":
        assert twin_h.wrapper is not sim_h.wrapper
    twin.interactive_do(circ[1:3])
    sim.interactive_do(circ[1:])
    sim.finish_interactive_run()
    ref.interactive_do(circ[1:])
    ref.finish_interactive_run()
    twin.interactive_do(circ[3:])
    twin.finish_interactive_run()
    assert bool(np.all(ref.get_final_measurement_records()))
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(sim.get_detector_flips(), ref.get_detector_flips())
    np.testing.assert_array_equal(
        twin.get_final_measurement_records(), ref.get_final_measurement_records()
    )


class _CyclicSlotNode:
    __slots__ = ("__priv", "op", "peer")

    def __init__(self, op: stim.CircuitInstruction) -> None:
        self.__priv = [op]
        self.peer: _CyclicSlotNode | None = None
        self.op = op

    def priv(self) -> list[stim.CircuitInstruction]:
        return self.__priv


class _SlottedNonMarkovianHandler(CompiledOpHandler[Any]):
    __slots__ = ("gt", "history", "node", "repeat_blk")

    def __init__(self, circuit: stim.Circuit) -> None:
        ops = list(circuit)
        first_inst = next(op for op in ops if isinstance(op, stim.CircuitInstruction))
        repeat_blk = next(op for op in ops if isinstance(op, stim.CircuitRepeatBlock))
        n1 = _CyclicSlotNode(first_inst)
        n2 = _CyclicSlotNode(ops[2])
        n1.peer, n2.peer = n2, n1
        self.node = n1
        self.history: list[str] = []
        self.repeat_blk = repeat_blk
        self.gt = first_inst.targets_copy()[0]

    def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
        self.history.append(op.name)
        sss._do_bare_instruction(op)
        if op.name == "I" and (len(self.history) % 2 == 0):
            sss.apply_pauli("X", np.ones((1, sss.batch_size), dtype=bool))

    def clear(self) -> None:
        self.history.clear()


@pytest.mark.parametrize("engine", [False, True])
def test_copy_copy_slotted_cyclic_handler_is_independent(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("R 0\nREPEAT 2 {\n H 0\n}\nI 0\nM 0\nDETECTOR rec[-1]")
    sim = MagicsideSimulator(
        circ, compiled_op_handler=_SlottedNonMarkovianHandler(circ), batch_size=2, seed=7
    )
    ref = MagicsideSimulator(
        circ, compiled_op_handler=_SlottedNonMarkovianHandler(circ), batch_size=2, seed=7
    )
    sim.interactive_do(circ[:1])
    ref.interactive_do(circ[:1])
    twin = copy.copy(sim)
    sim_h = sim.compiled_op_handler
    twin_h = twin.compiled_op_handler
    assert isinstance(sim_h, _SlottedNonMarkovianHandler)
    assert isinstance(twin_h, _SlottedNonMarkovianHandler)
    assert twin_h is not sim_h and twin_h.node is not sim_h.node
    assert twin_h.node.peer is not None and twin_h.node.peer.peer is twin_h.node
    assert twin_h.node.priv() is not sim_h.node.priv()
    twin.interactive_do(circ[1:3])
    sim.interactive_do(circ[1:])
    sim.finish_interactive_run()
    ref.interactive_do(circ[1:])
    ref.finish_interactive_run()
    twin.interactive_do(circ[3:])
    twin.finish_interactive_run()
    assert bool(np.all(ref.get_final_measurement_records()))
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(sim.get_detector_flips(), ref.get_detector_flips())


@pytest.mark.parametrize("engine", [False, True])
def test_copy_copy_handler_custom_copy_and_deepcopy_hooks(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    class _CustomCopyHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.tab = stim.TableauSimulator()
            self.log: list[str] = []

        def __copy__(self) -> _CustomCopyHandler:
            out = _CustomCopyHandler()
            out.tab = self.tab.copy()
            out.log = list(self.log)
            return out

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            self.log.append(op.name)
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            self.log.clear()

    class _CustomDeepcopyHandler(CompiledOpHandler[Any]):
        def __init__(self, op: stim.CircuitInstruction) -> None:
            self.op = op
            self.log: list[str] = []
            self.deepcopied = False

        def __deepcopy__(self, memo: dict[int, Any]) -> _CustomDeepcopyHandler:
            out = _CustomDeepcopyHandler(copy.deepcopy(self.op, memo))
            out.log = list(self.log)
            out.deepcopied = True
            return out

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            self.log.append(op.name)
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            self.log.clear()

    class _UnpicklableWithoutCopyHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.tab = stim.TableauSimulator()

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            return

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("R 0\nI 0\nM 0")
    sim_cc = MagicsideSimulator(
        circ, compiled_op_handler=_CustomCopyHandler(), batch_size=2, seed=7
    )
    twin_cc = copy.copy(sim_cc)
    assert isinstance(twin_cc.compiled_op_handler, _CustomCopyHandler)
    assert isinstance(sim_cc.compiled_op_handler, _CustomCopyHandler)
    assert twin_cc.compiled_op_handler.tab is not sim_cc.compiled_op_handler.tab
    twin_cc.interactive_do(circ)
    assert sim_cc.compiled_op_handler.log == []

    sim_dc = MagicsideSimulator(
        circ, compiled_op_handler=_CustomDeepcopyHandler(circ[0]), batch_size=2, seed=7
    )
    twin_dc = copy.copy(sim_dc)
    assert isinstance(sim_dc.compiled_op_handler, _CustomDeepcopyHandler)
    assert isinstance(twin_dc.compiled_op_handler, _CustomDeepcopyHandler)
    assert twin_dc.compiled_op_handler.deepcopied
    assert twin_dc.compiled_op_handler.op is sim_dc.compiled_op_handler.op
    assert twin_dc.compiled_op_handler.op == circ[0]

    sim_bad = MagicsideSimulator(
        circ, compiled_op_handler=_UnpicklableWithoutCopyHandler(), batch_size=2, seed=7
    )
    with pytest.raises(TypeError):
        copy.copy(sim_bad)


@pytest.mark.parametrize("engine", [False, True])
def test_run_after_first_instruction_exception_raises_until_clear(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    class _BoomOnFirstHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.should_fail = True

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            if self.should_fail:
                self.should_fail = False
                sss.do_conditional_clifford(stim.CircuitInstruction("X", [0]))
                raise RuntimeError("boom on instruction 0")
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            return

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("H 0\nM 0")
    sim = MagicsideSimulator(circ, compiled_op_handler=_BoomOnFirstHandler(), batch_size=4, seed=3)
    with pytest.raises(RuntimeError, match="boom on instruction 0"):
        sim.run()
    assert sim.get_current_circuit_time() == 0
    with pytest.raises(ValueError, match="not clean.*clear"):
        sim.run()
    with pytest.raises(ValueError, match="not clean.*clear"):
        sim.interactive_do(circ)
    with pytest.raises(ValueError, match="not clean.*clear"):
        sim.finish_interactive_run()
    sim.clear()
    sim.run()
    ref = MagicsideSimulator(circ, batch_size=4, seed=3)
    ref.run()
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("cls_name", ["CompiledLeakageUint8Coset", "CompiledLeakageUint8Magic"])
def test_copy_copy_leakage_subclasses(monkeypatch: Any, engine: bool, cls_name: str) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("R 0\nI 0\nI[LEAKAGE_TRANSITION_1: (0.0, U-->2)] 0\nM 0\nDETECTOR rec[-1]")
    ops_map = LeakageUint8Magic().compile_op_handler(circuit=circ, batch_size=3).ops_to_params
    class _Sub(getattr(stimside, cls_name)):  # type: ignore[misc]
        def __init__(self, hook: bool = False) -> None:
            super().__init__(1, 3, True, dict(ops_map))
            self.history: list[str] = []
            self.counters, self.hook = np.zeros(2, dtype=np.int64), hook
        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            self.history.append(op.name)
            self.counters[0] += 1
            super().handle_op(op, sss)
            if op.name == "I" and not op.tag and int(self.counters[0]) % 2 == 0:
                sss.apply_pauli("X", np.ones((1, sss.batch_size), dtype=bool))
    class _SubCopy(_Sub):
        def __copy__(self) -> _SubCopy:
            return _SubCopy(hook=True)
    sim, ref = (MagicsideSimulator(circ, compiled_op_handler=_Sub(), batch_size=3) for _ in (0, 1))
    sim.interactive_do(circ[0])
    twin = copy.copy(sim)
    twin.interactive_do(circ[1])
    sim.interactive_do(circ[1:])
    sim.finish_interactive_run()
    ref.run()
    sh, th = sim.compiled_op_handler, twin.compiled_op_handler
    assert isinstance(sh, _Sub) and isinstance(th, _Sub) and th is not sh
    assert th.history is not sh.history and th.counters is not sh.counters
    for m in ("get_final_measurement_records", "get_detector_flips"):
        assert getattr(sim, m)().tolist() == getattr(ref, m)().tolist() == [[True]] * 3
    tc = copy.copy(MagicsideSimulator(circ, compiled_op_handler=_SubCopy(), batch_size=3))
    tc.run()
    assert getattr(tc.compiled_op_handler, "hook", False)


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("kind", ["tableau", "lock", "generator"])
def test_copy_undeepcopyable_handler_raises_clear_type_error(
    monkeypatch: pytest.MonkeyPatch, engine: bool, kind: str
) -> None:
    class _UncopyableUserHandler(CompiledOpHandler[Any]):
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            return

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    payload: Any
    if kind == "tableau":
        payload = stim.TableauSimulator()
    elif kind == "lock":
        payload = threading.Lock()
    else:
        payload = (i for i in range(2))
    circ = stim.Circuit("H 0\nM 0")
    sim = MagicsideSimulator(
        circ, compiled_op_handler=_UncopyableUserHandler(payload), batch_size=2, seed=5
    )
    with pytest.raises(TypeError) as exc_info:
        copy.copy(sim)
    msg = str(exc_info.value)
    assert "_UncopyableUserHandler" in msg
    assert "__copy__" in msg
    assert "__deepcopy__" in msg
    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize("engine", [False, True])
def test_copy_remaps_handler_back_references_in_both_engine_modes(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    class _BackRefHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.sim: Any = None
            self.cb: Any = None
            self.nested: list[Any] = []
            self.seen = 0

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            if self.sim is None:
                self.sim = sss
                self.cb = sss.apply_pauli
                self.nested = [sss]
            self.seen += 1
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            self.seen = 0

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("R 0 1\nH 0\nI[T] 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1]")
    sim = MagicsideSimulator(
        circ, compiled_op_handler=_BackRefHandler(), batch_size=8, seed=11
    )
    sim.interactive_do(circ[:3])
    twin = copy.copy(sim)
    sh, th = sim.compiled_op_handler, twin.compiled_op_handler
    assert isinstance(sh, _BackRefHandler) and isinstance(th, _BackRefHandler)
    assert th is not sh
    assert th.sim is twin and th.sim is not sim
    assert th.cb.__self__ is twin
    assert th.nested[0] is twin
    assert sh.sim is sim

    sim.interactive_do(circ[3:])
    sim.finish_interactive_run()
    twin.interactive_do(circ[3:])
    twin.finish_interactive_run()

    ref = MagicsideSimulator(
        circ, compiled_op_handler=_BackRefHandler(), batch_size=8, seed=11
    )
    ref.run()
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(
        twin.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(sim.get_detector_flips(), ref.get_detector_flips())
    np.testing.assert_array_equal(twin.get_detector_flips(), ref.get_detector_flips())


_STRUCTURAL_IMMUTABLE_TYPES = (
    int,
    float,
    complex,
    str,
    bytes,
    bool,
    type(None),
    frozenset,
    types.FunctionType,
    type,
    np.generic,
    stim.CircuitInstruction,
    stim.GateTarget,
    stim.CircuitRepeatBlock,
)

_ALLOWED_SHARED_TOP_LEVEL = frozenset({
    "circuit",
    "_unrolled_ops",
    "_prog_faults",
    "_inline_faults",
    "_all_fault_channels",
    "_user_fault_channels",
    "_norm_reals",
    "_sweep_mat",
    "_explicit_ref_sample",
    "_compiled_stream_plan",
    "_compiled_query_plan",
    "_active_plan",
    "_m2d_skip_conv",
    "_m2d_stim_conv",
    "query_budget",
})

_ALLOWED_SHARED_CONTAINER_ITEMS = frozenset({
    "_live_frame_events",
    "_pending_magic",
    "_ref_sample_cache",
    "handler.ops_to_params",
})

_ALLOWED_MISSING_REF_TAB_KEYS = frozenset({"_sid", "_memo_ref", "_in_memo"})


def _is_structural_immutable(val: Any) -> bool:
    if isinstance(val, _STRUCTURAL_IMMUTABLE_TYPES):
        return True
    if isinstance(val, tuple):
        return all(_is_structural_immutable(item) for item in val)
    return False


def _collect_shared_members(
    a_dict: dict[str, Any], b_dict: dict[str, Any], prefix: str = ""
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for k, v in a_dict.items():
        name = prefix + k
        if k not in b_dict:
            findings.append((name, "missing"))
            continue
        tv = b_dict[k]
        if tv is v:
            if not _is_structural_immutable(v):
                findings.append((name, "same_object"))
            continue
        if isinstance(v, np.ndarray) and isinstance(tv, np.ndarray) and np.shares_memory(v, tv):
            findings.append((name, "shares_memory"))
        if isinstance(v, (list, tuple)) and isinstance(tv, (list, tuple)):
            if any(x is y and not _is_structural_immutable(x) for x, y in zip(v, tv)):
                findings.append((name, "item_same_object"))
        elif (
            isinstance(v, dict)
            and isinstance(tv, dict)
            and any(
                kk in tv and v[kk] is tv[kk] and not _is_structural_immutable(v[kk])
                for kk in v
            )
        ):
            findings.append((name, "value_same_object"))
    for k in b_dict:
        if k not in a_dict:
            findings.append((prefix + k, "only_in_twin"))
    return findings


def _build_copy_member_states(engine: bool) -> list[tuple[str, MagicsideSimulator]]:
    plain = stim.Circuit(
        "QUBIT_COORDS(0, 1) 0\nR 0 1 2 3\nH 0 1 2 3\nI[T] 0\nCX 0 1\nM 1\n"
        "CX rec[-1] 2\nI[T] 2\nDEPOLARIZE1(0.2) 0 1\nI[CCZ] 0 2 3\n"
        "SHIFT_COORDS(1)\nM 0 1 2 3\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
    )
    leak = stim.Circuit(
        "QUBIT_COORDS(0, 1) 0\nR 0 1 2 3\nH 0 1 2 3\nI[T] 0\n"
        "I[LEAKAGE_TRANSITION_1: (0.5, U-->2) (0.4, 2-->U)] 1\n"
        "CX[CONDITIONED_ON_PAIR: (U, U)] 0 1\nM[LEAKAGE_PROJECTION_Z: (1.0, 2)] 1\n"
        "CX rec[-1] 2\nI[T] 2\nE(0.3) X1\nELSE_CORRELATED_ERROR(0.3) Z2\nI[CCZ] 0 2 3\n"
        "SHIFT_COORDS(1)\nM 0 1 2 3\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
    )
    out: list[tuple[str, MagicsideSimulator]] = []
    out.append(("fresh", MagicsideSimulator(plain, batch_size=5, seed=1)))
    s = MagicsideSimulator(plain, batch_size=5, seed=1)
    s.interactive_do(plain[:9])
    out.append(("mid_pending_t", s))
    s = MagicsideSimulator(
        leak, op_handler=LeakageUint8Magic(), batch_size=5, seed=1,
        sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    s.interactive_do(leak[:9])
    out.append(("leak_sync_mid", s))
    s = MagicsideSimulator(
        leak, op_handler=LeakageUint8Magic(), batch_size=5, seed=1,
        record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    s.run()
    s.get_detector_flips()
    out.append(("leak_finished", s))
    s = MagicsideSimulator(plain, batch_size=5, seed=1, m2d_reference_mode="stim")
    s.run()
    s.get_detector_flips()
    out.append(("stim_m2d_finished", s))
    ch = CliffordFaultChannel(step_index=2, branches=(FaultBranch(0.2, "X", (0,)),))
    s = MagicsideSimulator(
        plain, batch_size=5, seed=1, fault_channels=(ch,), query_budget=QueryBudget()
    )
    s.interactive_do(plain[:5])
    out.append(("fault_channel_mid", s))
    for label, kw in (
        ("query", {"backend": "QUERY"}),
        ("selector", {"fault_mode": "SELECTOR"}),
        ("query_selector", {"backend": "QUERY", "fault_mode": "SELECTOR"}),
        ("fault_reals", {"fault_realizations": {"sweep_bits": [True]}}),
    ):
        s = MagicsideSimulator(plain, batch_size=5, seed=1, **kw)
        s.run()
        s.get_detector_flips()
        out.append((f"compiled_{label}", s))
    s = MagicsideSimulator(
        plain, batch_size=5, seed=1,
        reference_sample=np.zeros(plain.num_measurements, dtype=bool),
    )
    s.run()
    s.get_detector_flips()
    out.append(("explicit_ref", s))
    assert isinstance(engine, bool)
    return out


@pytest.mark.parametrize("engine", [False, True])
def test_copy_has_no_unexpected_shared_mutable_members(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    for label, sim in _build_copy_member_states(engine):
        twin = copy.copy(sim)
        assert twin.__dict__.keys() == sim.__dict__.keys(), label
        if sm._SIM_MEMOS.get(sim) is not None:
            assert sm._SIM_MEMOS.get(twin) is not sm._SIM_MEMOS.get(sim), label
        findings = _collect_shared_members(sim.__dict__, twin.__dict__)
        h, th = sim.compiled_op_handler, twin.compiled_op_handler
        if h is not th and hasattr(h, "__dict__"):
            findings.extend(_collect_shared_members(h.__dict__, th.__dict__, prefix="handler."))
        tab, ttab = sim._live_ref_tab, twin._live_ref_tab
        assert tab is not ttab, label
        tab_findings = _collect_shared_members(tab.__dict__, ttab.__dict__, prefix="_live_ref_tab.")
        unexpected: list[tuple[str, str]] = []
        for name, kind in findings:
            if kind == "same_object" and name in _ALLOWED_SHARED_TOP_LEVEL:
                continue
            if kind in ("item_same_object", "value_same_object") and (
                name in _ALLOWED_SHARED_CONTAINER_ITEMS
            ):
                continue
            unexpected.append((name, kind))
        for name, kind in tab_findings:
            short = name.removeprefix("_live_ref_tab.")
            if kind == "missing" and short in _ALLOWED_MISSING_REF_TAB_KEYS:
                continue
            unexpected.append((name, kind))
        assert unexpected == [], f"engine={engine} state={label} unexpected={unexpected}"


@pytest.mark.parametrize("engine", [False, True])
def test_copy_isolates_tab_rngs_x(monkeypatch: pytest.MonkeyPatch, engine: bool) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit(
        "R 0 1 2\nH 0 1 2\nZ_ERROR(0.5) 0 1 2\nDEPOLARIZE1(0.5) 0 1 2\nMX 0 1 2\n"
    )
    sim = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(), batch_size=16, seed=7, sync_tableside_rng=True
    )
    sim.interactive_do(circ[:2])
    twin = copy.copy(sim)
    assert sim._tab_rngs_x is not None and twin._tab_rngs_x is not None
    assert len(sim._tab_rngs_x) == len(twin._tab_rngs_x) == 16
    for s_rng, t_rng in zip(sim._tab_rngs_x, twin._tab_rngs_x):
        assert t_rng is not s_rng
    sim.interactive_do(circ[2:])
    sim.finish_interactive_run()
    twin.interactive_do(circ[2:])
    twin.finish_interactive_run()
    ref = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(), batch_size=16, seed=7, sync_tableside_rng=True
    )
    ref.run()
    np.testing.assert_array_equal(
        sim.get_final_measurement_records(), ref.get_final_measurement_records()
    )
    np.testing.assert_array_equal(
        twin.get_final_measurement_records(), ref.get_final_measurement_records()
    )


@pytest.mark.parametrize("engine", [False, True])
def test_copy_isolates_qubit_tags(monkeypatch: pytest.MonkeyPatch, engine: bool) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("QUBIT_COORDS[a](0, 0) 0\nH 0\nQUBIT_COORDS[b](1, 1) 1\nM 0 1\n")
    sim = MagicsideSimulator(circ, batch_size=4, seed=3)
    sim.interactive_do(circ[:2])
    twin = copy.copy(sim)
    sim.interactive_do(circ[2:3])
    assert twin.qubit_tags == {0: "a"}
    assert sim.qubit_tags == {0: "a", 1: "b"}


@pytest.mark.parametrize("engine", [False, True])
def test_copy_isolates_measurement_columns_arrays(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("X 0\nM 0\nM 0\n")
    sim = MagicsideSimulator(circ, batch_size=4, seed=3)
    sim.interactive_do(circ[:2])
    twin = copy.copy(sim)
    sim.overwrite_measurement_column(0, np.zeros(4, dtype=bool))
    assert twin.measurement_column(0).all()
    assert not sim.measurement_column(0).any()
    assert not np.shares_memory(twin.measurement_column(0), sim.measurement_column(0))


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("mode", ["query", "selector", "direct_reals", "live"])
def test_failed_run_blocks_retries_and_copies_across_all_execution_paths(
    monkeypatch: pytest.MonkeyPatch, engine: bool, mode: str
) -> None:
    class _FailFirstLiveHandler(CompiledOpHandler[Any]):
        def __init__(self) -> None:
            self.fail = True

        def handle_op(self, op: stim.CircuitInstruction, sss: Any) -> None:
            if self.fail:
                self.fail = False
                raise RuntimeError("boom on live op 0")
            sss._do_bare_instruction(op)

        def clear(self) -> None:
            return

    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("H 0 1\nI[T] 0 1\nM 0 1")
    expected_exc: type[Exception] = QueryBudgetExceededError
    if mode == "query":
        sim = MagicsideSimulator(
            circ, batch_size=2, seed=3, backend="QUERY",
            query_budget=QueryBudget(max_terms=1, on_budget_exceeded="raise"),
        )
    elif mode == "selector":
        sim = MagicsideSimulator(
            circ, batch_size=2, seed=3, fault_mode="SELECTOR",
            max_components=1, on_budget_exceeded="raise",
        )
    elif mode == "direct_reals":
        sim = MagicsideSimulator(
            circ, batch_size=2, seed=3, fault_realizations={"sweep_bits": [True]},
            max_components=1, on_budget_exceeded="raise",
        )
    else:
        expected_exc = RuntimeError
        sim = MagicsideSimulator(
            stim.Circuit("H 0\nM 0"), compiled_op_handler=_FailFirstLiveHandler(),
            batch_size=2, seed=3,
        )

    with pytest.raises(expected_exc):
        sim.run()
    with pytest.raises(ValueError, match="not clean.*clear"):
        sim.run()
    twin = copy.copy(sim)
    with pytest.raises(ValueError, match="not clean.*clear"):
        twin.run()
    if mode == "live":
        with pytest.raises(ValueError, match="not clean.*clear"):
            twin.interactive_do(stim.Circuit("H 0\nM 0"))
        with pytest.raises(ValueError, match="not clean.*clear"):
            twin.finish_interactive_run()

    twin.clear()
    if mode == "query":
        with pytest.raises(QueryBudgetExceededError):
            twin.run()
    else:
        if mode in ("selector", "direct_reals"):
            twin.max_components = 1024
        twin.run()
        assert twin.get_final_measurement_records().shape[0] == 2
    with pytest.raises(ValueError, match="not clean.*clear"):
        sim.run()
