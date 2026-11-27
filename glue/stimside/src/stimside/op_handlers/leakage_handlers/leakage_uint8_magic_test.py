"""Unit tests for LeakageUint8Magic and CompiledLeakageUint8Magic."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import stim

import stimside.simulator_magic as sim_magic
from stimside._magic.query import QueryBudgetExceededError
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import CompiledLeakageUint8Magic, LeakageUint8Magic
from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import LeakageUint8
from stimside.simulator_coset import CosetsideSimulator
from stimside.simulator_magic import MagicsideSimulator
from stimside.simulator_tableau import TablesideSimulator


def test_u_basis_mpad_leakage_measurement_preserves_magic_coherence() -> None:
    circ = stim.Circuit(
        "H 0\n"
        "I[T] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, U) (0.0, 2) : 0] 0\n"
        "I[T_DAG] 0\n"
        "H 0\n"
        "M 0"
    )
    handler = LeakageUint8Magic().compile_op_handler(circuit=circ, batch_size=32)
    assert isinstance(handler, CompiledLeakageUint8Magic)
    sim = MagicsideSimulator(circ, compiled_op_handler=handler, batch_size=32, seed=13)
    sim.run()
    recs = sim.get_final_measurement_records()
    assert np.all(recs[:, 0])
    assert not np.any(recs[:, 1])


def test_skewed_born_collapse_on_state_selective_transition() -> None:
    # R_Y(pi/4)|0>: P(|1>) = sin^2(pi/8) = (2 - sqrt(2))/4 ~= 0.1464466
    circ = stim.Circuit(
        "H 0\nI[T] 0\nH 0\nS 0\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (1.0, 2)] 0"
    )
    sim = MagicsideSimulator(circ, op_handler=LeakageUint8Magic(), batch_size=2048, seed=42)
    sim.run()
    recs = sim.get_final_measurement_records()
    np.testing.assert_array_equal(recs[:, 0], recs[:, 1])
    p_want = (2.0 - math.sqrt(2.0)) / 4.0
    assert abs(float(np.mean(recs[:, 0])) - p_want) < 0.03


def test_sync_tableside_rng_clifford_leakage_lockstep_with_cosetside() -> None:
    circ = stim.Circuit(
        "H 0 1\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.5, 1-->2)] 0\n"
        "TICK\n"
        "CZ[CONDITIONED_ON_PAIR: (U, U)] 0 1\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.8, 2) : 0] 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (1.0, 1) (0.5, 2)] 0 1"
    )
    sc = CosetsideSimulator(
        circ, compiled_op_handler=LeakageUint8Coset(False).compile_op_handler(circuit=circ, batch_size=16),
        batch_size=16, seed=77, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sm = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(False), batch_size=16, seed=77,
        sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    assert [a.tolist() for a in sm.run()] == [a.tolist() for a in sc.run()]  # type: ignore[union-attr]
    np.testing.assert_array_equal(sm.get_final_measurement_records(), sc.get_final_measurement_records())
    assert sm.get_leakage_events() == sc.get_leakage_events()


def test_t_on_leaked_targets_warns_and_acts_while_ccz_is_skipped() -> None:
    circ = stim.Circuit(
        "H 0 4\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 0 4\n"
        "I[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\n"
        "I[T_DAG] 4\nI[T_DAG] 4\nI[T_DAG] 4\nI[T_DAG] 4\n"
        "MPP X0 X4\n"
        "X 1 2\nH 3\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 1\n"
        "I[CCZ] 1 2 3\n"
        "H 3\nM 3"
    )
    sim = MagicsideSimulator(
        circ, op_handler=LeakageUint8Magic(unconditional_condition_on_U=True), batch_size=8, seed=17
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run()
    u_warns = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(u_warns) == 8 and not any("CCZ" in m for m in u_warns)
    recs = sim.get_final_measurement_records()
    assert np.all(recs[:, :2])  # T^4 = T_DAG^4 = Z acted on the stored |+> of leaked qubits 0 and 4
    assert not np.any(recs[:, 2])  # the CCZ with leaked qubit 1 was skipped

    for bad in ("X[T] 0", "I[T](0.1) 0", "I[CCZ] 0 1", "I[CCZ] 0 0 1"):
        with pytest.raises(ValueError):
            LeakageUint8Magic().compile_op_handler(circuit=stim.Circuit(bad), batch_size=1)


def test_sync_rng_nonclifford_born_stats_mpad_entangled_and_speed() -> None:
    p_sin2 = (2.0 - math.sqrt(2.0)) / 4.0
    sigma = math.sqrt(p_sin2 * (1.0 - p_sin2) / 1024.0)
    circ_mpad = stim.Circuit(
        "H 2\nI[T] 2\nH 2\nS 2\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.0, 0) (1.0, 1) : 2] 0 0\n"
        "M 0 2"
    )
    circ_cx = stim.Circuit(
        "H 0 1\nI[T] 1\nH 1\nS 1\nCX 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.05, 0) (0.95, 1)] 0 1\n"
        "M 0 1"
    )
    for sync in (False, True):
        sim_m = MagicsideSimulator(
            circ_mpad, op_handler=LeakageUint8Magic(), batch_size=1024, seed=42, sync_tableside_rng=sync
        )
        sim_m.run()
        rec_m = sim_m.get_final_measurement_records()
        assert not np.any(rec_m[:, 2])
        np.testing.assert_array_equal(rec_m[:, 0], rec_m[:, 1])
        np.testing.assert_array_equal(rec_m[:, 0], rec_m[:, 3])
        assert abs(float(np.mean(rec_m[:, 0])) - p_sin2) < 4.5 * sigma

        sim_cx = MagicsideSimulator(
            circ_cx, op_handler=LeakageUint8Magic(), batch_size=1024, seed=84, sync_tableside_rng=sync
        )
        sim_cx.run()
        rec_cx = sim_cx.get_final_measurement_records()
        assert abs(float(np.mean(rec_cx[:, 2] ^ rec_cx[:, 3])) - p_sin2) < 4.5 * sigma

    circ_ent = stim.Circuit(
        "H 0 1 2\nI[CCZ] 0 1 2\nH 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.0, 0) (0.9999, 1) (0.5, 2)] 0 1"
    )
    sim_e = MagicsideSimulator(
        circ_ent, op_handler=LeakageUint8Magic(), batch_size=512, seed=91, sync_tableside_rng=True
    )
    sim_e.run()
    recs_e = sim_e.get_final_measurement_records()
    p00 = float(np.mean((~recs_e[:, 0]) & (~recs_e[:, 1])))
    p01 = float(np.mean((~recs_e[:, 0]) & recs_e[:, 1]))
    p10 = float(np.mean(recs_e[:, 0] & (~recs_e[:, 1])))
    p11 = float(np.mean(recs_e[:, 0] & recs_e[:, 1]))
    # 0.10 is ~4.5 sigma at 512 shots.
    assert abs(p00 - 0.50) < 0.10 and p01 < 0.01 and abs(p10 - 0.25) < 0.10 and abs(p11 - 0.25) < 0.10


def test_sync_rng_three_way_clifford_lockstep() -> None:
    circ = stim.Circuit(
        "H 0 1 2\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.35, 1-->2)] 0 2\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.9, 2) : 0] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.3, 0) (0.3, 1) : 1] 0\n"
        "MPAD[LEAKAGE_MEASUREMENT: (0.2, 0) (0.8, 1) (0.9, 2) : 2] 0 0\n"
        "M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 1\n"
        "M[LEAKAGE_PROJECTION_Z: (0.1, 0) (0.85, 1) (0.95, 2)] 0 1 2"
    )
    st = TablesideSimulator(
        circ,
        compiled_op_handler=LeakageUint8(False).compile_op_handler(circuit=circ, batch_size=64),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sc = CosetsideSimulator(
        circ,
        compiled_op_handler=LeakageUint8Coset(False).compile_op_handler(circuit=circ, batch_size=64),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    sm = MagicsideSimulator(
        circ,
        op_handler=LeakageUint8Magic(False),
        batch_size=64, seed=123, sync_tableside_rng=True, record_unleaked_to_leaked=True, record_leakage_events=True,
    )
    u2l_t, u2l_c, u2l_m = st.run(), sc.run(), sm.run()
    assert [a.tolist() for a in u2l_m] == [a.tolist() for a in u2l_c] == [a.tolist() for a in u2l_t]  # type: ignore[union-attr]
    np.testing.assert_array_equal(sm.get_final_measurement_records(), sc.get_final_measurement_records())
    np.testing.assert_array_equal(sm.get_final_measurement_records(), st.get_final_measurement_records())
    assert sm.get_leakage_events() == sc.get_leakage_events() == st.get_leakage_events()


_CCZ_N = 6
_CCZ_FLAGS = "\n".join(f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : {q}] 0" for q in range(_CCZ_N))
_CCZ_CIRCUIT = (
    "H 0 1 2 3 4 5\nI[T] 0 3\nCZ 0 1 2 3 4 5\n"
    "I_ERROR[LEAKAGE_TRANSITION_1: (0.3, U-->2)] 0 1 2 3 4 5\n"
    f"{_CCZ_FLAGS}\n"
    "REPEAT 2 {\n    I[T] 0 1 2 3 4 5\n    I[CCZ] 0 1 2 2 3 4 4 5 0\n    H 0 2 4\n}"
)


def _ccz_skip_oracle(leaked: set[int]) -> np.ndarray:
    """State vector (qubit 0 = lowest bit) of `_CCZ_CIRCUIT` in a shot whose leaked qubits are `leaked`.

    After the leak: H skips a leaked qubit, a CCZ triple is skipped if any of its qubits is leaked,
    and T acts on the stored state of every qubit.
    """
    n, idx = _CCZ_N, np.arange(1 << _CCZ_N)
    h_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    def had(psi: np.ndarray, q: int) -> np.ndarray:
        t = np.moveaxis(psi.reshape((2,) * n), n - 1 - q, 0)
        return np.moveaxis(np.tensordot(h_mat, t, axes=1), 0, n - 1 - q).reshape(-1)

    def phase(psi: np.ndarray, qs: tuple[int, ...], ph: complex) -> np.ndarray:
        out = psi.copy()
        out[np.logical_and.reduce([((idx >> q) & 1).astype(bool) for q in qs])] *= ph
        return out

    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    for q in range(n):
        psi = had(psi, q)
    for q in (0, 3):
        psi = phase(psi, (q,), np.exp(0.25j * np.pi))
    for pair in ((0, 1), (2, 3), (4, 5)):
        psi = phase(psi, pair, -1)
    for _ in range(2):
        for q in range(n):
            psi = phase(psi, (q,), np.exp(0.25j * np.pi))
        for triple in ((0, 1, 2), (2, 3, 4), (4, 5, 0)):
            if not leaked & set(triple):
                psi = phase(psi, triple, -1)
        for q in (0, 2, 4):
            if q not in leaked:
                psi = had(psi, q)
    return psi


def _run_quiet(circ: stim.Circuit, **kw: object) -> MagicsideSimulator:
    sim = MagicsideSimulator(circ, op_handler=LeakageUint8Magic(), **kw)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim.run()
    return sim


@pytest.mark.parametrize("sync", [False, True])
def test_ccz_skips_each_triple_with_a_leaked_qubit_per_shot(monkeypatch: pytest.MonkeyPatch, sync: bool) -> None:
    circ, per_engine = stim.Circuit(_CCZ_CIRCUIT), {}
    for engine in (True, False):
        monkeypatch.setattr(sim_magic, "_LIVE_STORE_ENGINE", engine)
        sim = _run_quiet(circ, batch_size=48, seed=31, sync_tableside_rng=sync)
        flags = sim.get_final_measurement_records()
        states = np.stack([sim.state_vector(s) for s in range(48)])
        for s in range(48):
            want = _ccz_skip_oracle({q for q in range(_CCZ_N) if flags[s, q]})
            assert abs(np.vdot(want, states[s])) > 1 - 1e-9, (engine, s)
        per_engine[engine] = (flags, states)
    flags = per_engine[True][0]
    assert len({tuple(r) for r in flags.tolist()}) > 8  # many different leak patterns in the batch
    np.testing.assert_array_equal(per_engine[True][0], per_engine[False][0])
    np.testing.assert_array_equal(per_engine[True][1], per_engine[False][1])


def test_ccz_skip_leaves_skipping_shots_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    pre = (
        "H 0 1 2 3\nI[T] 0 1 2 3\nCZ 0 1 2 3\nH 1 2\nI[T] 1 2\n"
        "I_ERROR[LEAKAGE_TRANSITION_1: (0.4, U-->2)] 0 1 2\n"
        + "\n".join(f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : {q}] 0" for q in range(3))
    )
    for engine in (True, False):
        monkeypatch.setattr(sim_magic, "_LIVE_STORE_ENGINE", engine)
        with_ccz = _run_quiet(stim.Circuit(pre + "\nI[CCZ] 0 1 2"), batch_size=64, seed=3)
        without = _run_quiet(stim.Circuit(pre), batch_size=64, seed=3)
        flags = with_ccz.get_final_measurement_records()
        np.testing.assert_array_equal(flags, without.get_final_measurement_records())
        skips = flags.any(axis=1)
        assert 0 < int(skips.sum()) < 64
        for s in range(64):
            same = sim_magic._state_key(with_ccz.get_qfe_expansion(s)[1]) == sim_magic._state_key(without.get_qfe_expansion(s)[1])
            assert same == bool(skips[s]), (engine, s)


def test_ccz_on_a_leaked_qubit_follows_unconditional_condition_on_u() -> None:
    def run(text: str, uc: bool, batch: int = 8) -> tuple[np.ndarray, list[str]]:
        sim = MagicsideSimulator(
            stim.Circuit(text), op_handler=LeakageUint8Magic(unconditional_condition_on_U=uc), batch_size=batch, seed=5
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.run()
        return sim.get_final_measurement_records(), [str(w.message) for w in caught]

    # |1>|1>|+>: the CCZ flips qubit 3 from |+> to |->, so `H 3; M 3` reads 1 iff the CCZ acted.
    always = "X 1 2\nH 3\nI_ERROR[LEAKAGE_TRANSITION_1: (1.0, U-->2)] 1\nI[CCZ] 1 2 3\nH 3\nM 3"
    recs, warns = run(always, uc=True)
    assert not recs.any() and warns == []
    recs, warns = run(always, uc=False)
    assert recs.all() and warns == ["Magic gate I[CCZ] applied to leaked group (1, 2, 3)"]

    half = (
        "X 1 2\nH 3\nI_ERROR[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 1\nI[CCZ] 1 2 3\nH 3\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 1] 0\nM 3"
    )
    recs, warns = run(half, uc=True, batch=64)
    assert warns == [] and 0 < int(recs[:, 0].sum()) < 64
    np.testing.assert_array_equal(recs[:, 1], ~recs[:, 0])


def test_do_conditional_magic_restores_skipping_shots_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        "H 0 1 2 3\nI[T] 0 1 2 3\nH 0 1 2 3\nI_ERROR[LEAKAGE_TRANSITION_1: (0.5, U-->2)] 3\n"
        "MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 3] 0\nI[T] 0 1 2"
    )
    for engine in (True, False):
        monkeypatch.setattr(sim_magic, "_LIVE_STORE_ENGINE", engine)
        sim = MagicsideSimulator(stim.Circuit(text), op_handler=LeakageUint8Magic(), batch_size=16, seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sim.interactive_do(stim.Circuit(text))
            leaked = sim.measurement_column(0).copy()
            assert 0 < int(leaked.sum()) < 16
            keys = [sim_magic._state_key(sim.get_qfe_expansion(s)[1]) for s in range(16)]
            sim.max_components, sim.on_budget_exceeded = 1, "raise"  # the CCZ block must now fail
            with pytest.raises(QueryBudgetExceededError):
                sim.interactive_do(stim.Circuit("I[CCZ] 1 2 3\nH 1"))
        for s in np.flatnonzero(leaked):
            assert sim_magic._state_key(sim._live_exps[int(s)]) == keys[int(s)], (engine, int(s))


def test_do_conditional_magic_applies_t_per_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    op = stim.CircuitInstruction("I", [0, 1], tag="T")
    mask = np.zeros((2, 8), dtype=np.bool_)
    mask[0, ::2] = True  # qubit 0: even shots only; qubit 1: no shot
    for engine in (True, False):
        monkeypatch.setattr(sim_magic, "_LIVE_STORE_ENGINE", engine)
        sim = MagicsideSimulator(stim.Circuit("H 0 1\nTICK\nH 0 1\nM 0 1"), batch_size=8, seed=2)
        sim.interactive_do(stim.Circuit("H 0 1\nTICK"))
        for _ in range(4):  # T^4 = Z
            sim.do_conditional_magic(op, [0, 1], mask)
        sim.interactive_do(stim.Circuit("H 0 1\nM 0 1"))
        sim.finish_interactive_run()
        recs = sim.get_final_measurement_records()
        np.testing.assert_array_equal(recs[:, 0], mask[0])
        assert not recs[:, 1].any()
