"""Tests for E1 reference-tableau memoization in `stimside.simulator_magic`."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import stim

import stimside.simulator_magic as sm
from stimside import golden_test
from stimside._magic.phase_qfe import ImpossiblePostselectionError
from stimside.op_handlers.abstract_op_handler import TrivialOpHandler
from stimside.op_handlers.leakage_handlers.leakage_uint8_magic import LeakageUint8Magic
from stimside.simulator_magic import MagicsideSimulator

_CASES_3 = ("c04", "c08", "c13")


def _corpus_entry(prefix: str) -> tuple[str, bool, str]:
    for name, _, has_leak, text in golden_test.CORPUS:
        if name.startswith(prefix):
            return name, has_leak, text
    raise KeyError(f"Missing golden corpus entry {prefix!r}")


def _build_corpus_sim(
    prefix: str, *, mode: str = "magic_ds", batch: int = 8, seed: int = golden_test.SEED
) -> MagicsideSimulator:
    _, has_leak, text = _corpus_entry(prefix)
    circ = stim.Circuit(text)
    oh_cls = LeakageUint8Magic() if has_leak else TrivialOpHandler()
    oh = oh_cls.compile_op_handler(circuit=circ, batch_size=batch)
    sync = mode == "magic_sync"
    return MagicsideSimulator(
        circ, compiled_op_handler=oh, batch_size=batch, seed=seed, sync_tableside_rng=sync
    )


def _iter_arrays(val: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if isinstance(val, np.ndarray):
        out.append(val)
    elif isinstance(val, (tuple, list)):
        for item in val:
            out.extend(_iter_arrays(item))
    elif dataclasses.is_dataclass(val) and not isinstance(val, type):
        for f in dataclasses.fields(val):
            out.extend(_iter_arrays(getattr(val, f.name)))
    return out


def _assert_full_state_equal(a: MagicsideSimulator, b: MagicsideSimulator) -> None:
    assert [c.tobytes() for c in a._measurement_columns] == [
        c.tobytes() for c in b._measurement_columns
    ]
    assert a._log_branch_masses.tobytes() == b._log_branch_masses.tobytes()
    assert sm._tab_equal(a._live_ref_tab, b._live_ref_tab.snapshot_packed())
    assert repr(a._live_frame_events) == repr(b._live_frame_events)
    for (ta, ea), (tb, eb) in zip(a.get_state_expansions(), b.get_state_expansions()):
        assert repr(ta) == repr(tb)
        assert ea.weights.view(np.uint64).tobytes() == eb.weights.view(np.uint64).tobytes()
        assert ea.h_words.tobytes() == eb.h_words.tobytes()
        assert ea.ell.tobytes() == eb.ell.tobytes()
        assert ea.A_words.tobytes() == eb.A_words.tobytes()
        assert ea.log_mass.hex() == eb.log_mass.hex()
    if a._eff_nq <= 10:
        for s in range(a.batch_size):
            assert np.array_equal(a.state_vector(s), b.state_vector(s))


@pytest.mark.parametrize("case", _CASES_3)
@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("mode", ["magic_ds", "magic_sync"])
def test_e1_hits_and_misses_on_corpus_cases(
    monkeypatch: pytest.MonkeyPatch, case: str, engine: bool, mode: str
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", False)
    sim = _build_corpus_sim(case, mode=mode, batch=8)
    memo = sim._ref_memo
    prev_hits = -1
    for idx in range(6):
        if idx > 0:
            sim.clear()
        sim.run()
        assert memo.misses > 0
        assert memo.hits > prev_hits
        prev_hits = memo.hits
    assert memo.hits > memo.misses
    assert len(memo.entries) == memo.misses
    assert memo.total_bytes > 0 and len(memo._trie) > 0


@pytest.mark.parametrize("case", _CASES_3)
@pytest.mark.parametrize("engine", [False, True])
def test_e1_lockstep_full_state_equality_with_verify(
    monkeypatch: pytest.MonkeyPatch, case: str, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = _build_corpus_sim(case, mode="magic_ds", batch=8)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = _build_corpus_sim(case, mode="magic_ds", batch=8)
    for k in range(5):
        if k > 0:
            ref.clear()
            sim.clear()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        ref.run()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        sim.run()
        _assert_full_state_equal(ref, sim)
    assert sim._ref_memo.hits > 0


@pytest.mark.parametrize("engine", [False, True])
def test_e1_repeat_cross_index_hits_and_lockstep_verify(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    circ = stim.Circuit(
        "R 0 1\nH 0 1\n"
        "REPEAT 3 {\n"
        "  I[T] 0 1\n  X_ERROR(0.1) 0 1\n  Z_ERROR(0.1) 0 1\n  I[T_DAG] 0\n"
        "}\nCX 0 1\nM 0 1\n"
    )
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(circ, batch_size=6, seed=7)
    ref.run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = MagicsideSimulator(circ, batch_size=6, seed=7)
    sim.run()
    _assert_full_state_equal(ref, sim)
    assert sim._ref_memo.hits > 0


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize(
    ("memo_on", "verify_on"), [(False, False), (True, False), (True, True)]
)
def test_e1_goldens_all_engine_memo_verify_modes(
    monkeypatch: pytest.MonkeyPatch, engine: bool, memo_on: bool, verify_on: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", memo_on)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", verify_on)
    expected = json.loads(golden_test.DIGESTS_PATH.read_text(encoding="utf-8"))
    for name, modes, has_leak, text in golden_test.CORPUS:
        circ = stim.Circuit(text)
        mb = name == "c16_sparse_multibatch"
        for m in modes:
            if not m.startswith("magic_"):
                continue
            got = golden_test._run_case(
                circ, m, has_leak, batch=70 if mb else golden_test.BATCH, multibatch_packed=mb
            )
            assert got == expected[name][m], (name, m, engine, memo_on, verify_on)
    if engine and memo_on and verify_on:
        golden_test.test_golden_digests()


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"fault_mode": "SELECTOR"},
        {"backend": "QUERY"},
        {"fault_realizations": {"faults": {0: 1}}},
    ],
)
def test_e1_compiled_engines_bypass_memo(
    monkeypatch: pytest.MonkeyPatch, engine: bool, kwargs: dict[str, Any]
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    circ = stim.Circuit("H 0 1\nI[T] 0\nI[CLIFFORD_FAULT: (0.3, X)] 0\nCX 0 1\nM 0 1")
    sim = MagicsideSimulator(circ, batch_size=8, seed=3, **kwargs)
    sim.run()
    memo = sim._ref_memo
    assert memo.hits == 0 and memo.misses == 0 and len(memo.entries) == 0
    assert not isinstance(sim._live_ref_tab, sm._TrackedRefTableau)


def _shared_sim_runs(texts: list[str]) -> None:
    base = stim.Circuit("R 0 1\nH 0 1\nM 0 1")
    sim = MagicsideSimulator(base, batch_size=4, seed=1)
    for t in texts:
        sim.clear()
        for op in stim.Circuit(t):
            sim._do_instruction(op)


def _direct_meas_pair(k1: dict[str, Any], k2: dict[str, Any]) -> None:
    sim = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=2, seed=1)
    sim._compile_meas_ref([(0, "Z")], invert_result=False, **k1)
    sim.clear()
    sim._compile_meas_ref([(0, "Z")], invert_result=False, **k2)


def _direct_reset_pair(k1: dict[str, Any], k2: dict[str, Any]) -> None:
    sim = MagicsideSimulator(stim.Circuit("H 0\nR 0"), batch_size=2, seed=1)
    sim._compile_reset_ref(0, "Z", meas_index=0, invert_result=False, **k1)
    sim.clear()
    sim._compile_reset_ref(0, "Z", meas_index=0, invert_result=False, **k2)


def _tab_sid(sim: MagicsideSimulator) -> int | None:
    assert isinstance(sim._live_ref_tab, sm._TrackedRefTableau)
    return sim._live_ref_tab._sid


def _apply_h_then_s(sim: MagicsideSimulator, mask: np.ndarray) -> None:
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None)
    sim.do_conditional_clifford(stim.CircuitInstruction("S", [0]), mask)


def _run_fac_pair(
    fn1: Callable[[MagicsideSimulator, np.ndarray], Any],
    fn2: Callable[[MagicsideSimulator, np.ndarray], Any],
) -> None:
    sim = MagicsideSimulator(stim.Circuit("H 0 1\nS 0"), batch_size=4, seed=1)
    mask = np.array([[True, False, False, False]])
    fn1(sim, mask)
    sim.clear()
    fn2(sim, mask)


def _mut_op_kind() -> None:
    sim = MagicsideSimulator(stim.Circuit("I[T] 0"), batch_size=4, seed=1)
    sim.interactive_do(stim.Circuit("I[T] 0"))
    m = np.array([[True, False, False, False]])
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), m)
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None)


def _mut_axis_neg() -> None:
    sim = MagicsideSimulator(stim.Circuit("H 0\nSPP X0\nSPP !X0"), batch_size=4, seed=1)
    m = np.array([[True, False, False, False]])
    sim.do_conditional_clifford(stim.CircuitInstruction("SPP", [stim.target_x(0)]), m)
    sim.clear()
    sim.do_conditional_clifford(
        stim.CircuitInstruction("SPP", [stim.target_inv(stim.target_x(0))]), m
    )


def _mut_direct(text: str) -> None:
    sim = MagicsideSimulator(stim.Circuit(text), batch_size=4, seed=1)
    sim.run()


_KEY_MUTANTS: list[tuple[str, Callable[[Any], Any], Callable[[], None]]] = [
    ("drop_op_kind_u_fac", lambda k: ("*",) + k[1:], _mut_op_kind),
    (
        "u_drop_sid",
        lambda k: ("u", 0, *k[2:]) if k[0] == "u" else k,
        lambda: _mut_direct("S 0\nH 0\nS 0\nM 0"),
    ),
    (
        "u_drop_canon",
        lambda k: ("u", k[1], k[3]) if k[0] == "u" else k,
        lambda: _shared_sim_runs(["H 0\nM 0", "S 0\nM 0"]),
    ),
    (
        "u_drop_grp",
        lambda k: ("u", k[1], k[2]) if k[0] == "u" else k,
        lambda: _shared_sim_runs(["X 0\nM 0 1", "X 1\nM 0 1"]),
    ),
    (
        "spp_drop_sid",
        lambda k: ("spp", 0, *k[2:]) if k[0] == "spp" else k,
        lambda: _shared_sim_runs(["SPP X0\nM 0", "H 0\nSPP X0\nM 0"]),
    ),
    (
        "spp_drop_terms",
        lambda k: ("spp", k[1], k[3], k[4]) if k[0] == "spp" else k,
        lambda: _shared_sim_runs(["SPP X0\nM 0", "SPP Z0\nM 0"]),
    ),
    (
        "spp_drop_negate",
        lambda k: ("spp", k[1], k[2], k[4]) if k[0] == "spp" else k,
        lambda: _shared_sim_runs(["SPP X0\nM 0", "SPP !X0\nM 0"]),
    ),
    (
        "spp_drop_dag",
        lambda k: ("spp", k[1], k[2], k[3]) if k[0] == "spp" else k,
        lambda: _shared_sim_runs(["SPP X0\nM 0", "SPP_DAG X0\nM 0"]),
    ),
    (
        "axis_drop_sid",
        lambda k: ("axis", 0, *k[2:]) if k[0] == "axis" else k,
        lambda: _shared_sim_runs(["X_ERROR(1.0) 0\nM 0", "H 0\nX_ERROR(1.0) 0\nH 0\nM 0"]),
    ),
    (
        "axis_drop_terms",
        lambda k: ("axis", k[1], k[3]) if k[0] == "axis" else k,
        lambda: _shared_sim_runs(["X_ERROR(1.0) 0\nZ_ERROR(1.0) 0\nM 0"]),
    ),
    (
        "axis_drop_negate",
        lambda k: ("axis", k[1], k[2]) if k[0] == "axis" else k,
        _mut_axis_neg,
    ),
    (
        "fac_drop_sid",
        lambda k: ("fac", 0, *k[2:]) if k[0] == "fac" else k,
        lambda: _run_fac_pair(
            lambda s, m: s.do_conditional_clifford(stim.CircuitInstruction("S", [0]), m),
            _apply_h_then_s,
        ),
    ),
    (
        "fac_drop_gate",
        lambda k: ("fac", k[1], k[3]) if k[0] == "fac" else k,
        lambda: _run_fac_pair(
            lambda s, m: s.do_conditional_clifford(stim.CircuitInstruction("H", [0]), m),
            lambda s, m: s.do_conditional_clifford(stim.CircuitInstruction("S", [0]), m),
        ),
    ),
    (
        "fac_drop_targets",
        lambda k: ("fac", k[1], k[2]) if k[0] == "fac" else k,
        lambda: _run_fac_pair(
            lambda s, m: s.do_conditional_clifford(stim.CircuitInstruction("H", [0]), m),
            lambda s, m: s.do_conditional_clifford(stim.CircuitInstruction("H", [1]), m),
        ),
    ),
    (
        "meas_drop_sid",
        lambda k: ("meas", 0, *k[2:]) if k[0] == "meas" else k,
        lambda: _shared_sim_runs(["M 0", "H 0\nM 0"]),
    ),
    (
        "meas_drop_terms",
        lambda k: ("meas", k[1], *k[3:]) if k[0] == "meas" else k,
        lambda: _shared_sim_runs(["H 0 1\nM 0", "H 0 1\nM 1"]),
    ),
    (
        "meas_drop_invert",
        lambda k: ("meas", k[1], k[2], *k[4:]) if k[0] == "meas" else k,
        lambda: _shared_sim_runs(["M 0", "M !0"]),
    ),
    (
        "meas_drop_flip_prob",
        lambda k: ("meas", k[1], k[2], k[3], k[5]) if k[0] == "meas" else k,
        lambda: _direct_meas_pair(
            {"flip_prob": 0.0, "meas_index": 0}, {"flip_prob": 0.1, "meas_index": 0}
        ),
    ),
    (
        "meas_drop_meas_index",
        lambda k: ("meas", k[1], k[2], k[3], k[4]) if k[0] == "meas" else k,
        lambda: _direct_meas_pair(
            {"flip_prob": 0.0, "meas_index": 0}, {"flip_prob": 0.0, "meas_index": 1}
        ),
    ),
    (
        "reset_drop_sid",
        lambda k: ("reset", 0, *k[2:]) if k[0] == "reset" else k,
        lambda: _shared_sim_runs(["R 0", "X 0\nR 0"]),
    ),
    (
        "reset_drop_q",
        lambda k: ("reset", k[1], *k[3:]) if k[0] == "reset" else k,
        lambda: _shared_sim_runs(["X 0\nR 0", "X 0\nR 1"]),
    ),
    (
        "reset_drop_basis",
        lambda k: ("reset", k[1], k[2], *k[4:]) if k[0] == "reset" else k,
        lambda: _shared_sim_runs(["R 0", "RX 0"]),
    ),
    (
        "reset_drop_meas_index",
        lambda k: ("reset", k[1], k[2], k[3], *k[5:]) if k[0] == "reset" else k,
        lambda: _shared_sim_runs(["R 0", "MR 0"]),
    ),
    (
        "reset_drop_invert",
        lambda k: ("reset", k[1], k[2], k[3], k[4], k[6]) if k[0] == "reset" else k,
        lambda: _shared_sim_runs(["MR 0", "MR !0"]),
    ),
    (
        "reset_drop_flip_prob",
        lambda k: ("reset", k[1], k[2], k[3], k[4], k[5]) if k[0] == "reset" else k,
        lambda: _direct_reset_pair({"flip_prob": 0.0}, {"flip_prob": 0.1}),
    ),
    (
        "magic_drop_sid",
        lambda k: ("magic", 0, *k[2:]) if k[0] == "magic" else k,
        lambda: _shared_sim_runs(["I[T] 0\nM 0", "H 0\nI[T] 0\nM 0"]),
    ),
    (
        "magic_drop_tag",
        lambda k: ("magic", k[1], k[3]) if k[0] == "magic" else k,
        lambda: _shared_sim_runs(["H 0\nI[T] 0\nM 0", "H 0\nI[T_DAG] 0\nM 0"]),
    ),
    (
        "magic_drop_qs",
        lambda k: ("magic", k[1], k[2]) if k[0] == "magic" else k,
        lambda: _shared_sim_runs(["H 0 1\nI[T] 0", "H 0 1\nI[T] 1"]),
    ),
]

_CHILD_MUTANTS: list[tuple[str, Callable[[Any], Any], Callable[[], None]]] = [
    ("u_no_advance", lambda s: "u", lambda: _mut_direct("X 0\nX 0\nM 0")),
    ("spp_no_advance", lambda s: "spp", lambda: _mut_direct("SPP X0\nSPP X0\nM 0")),
    ("meas_no_advance", lambda s: "meas", lambda: _mut_direct("H 0\nM 0\nM 0")),
    (
        "reset_no_advance",
        lambda s: "reset",
        lambda: _shared_sim_runs(["H 0\nM 0", "H 0\nR 0\nM 0"]),
    ),
    (
        "u_child_drop_canon",
        lambda s: ("u", s[2]) if s[0] == "u" else s,
        lambda: _shared_sim_runs(["H 0\nM 0", "S 0\nM 0"]),
    ),
    (
        "meas_child_drop_terms",
        lambda s: ("meas",) if s[0] == "meas" else s,
        lambda: _shared_sim_runs(["H 0 1\nM 0\nM 1", "H 0 1\nM 1\nM 1"]),
    ),
    (
        "reset_child_drop_q",
        lambda s: ("reset", s[2]) if s[0] == "reset" else s,
        lambda: _shared_sim_runs(["H 0 1\nR 0\nM 0 1", "H 0 1\nR 1\nM 0 1"]),
    ),
    (
        "reset_child_drop_basis",
        lambda s: ("reset", s[1]) if s[0] == "reset" else s,
        lambda: _shared_sim_runs(["H 0\nR 0\nM 0", "H 0\nRX 0\nM 0"]),
    ),
]


@pytest.mark.parametrize(
    ("name", "xform", "runner"), _KEY_MUTANTS, ids=[m[0] for m in _KEY_MUTANTS]
)
def test_e1_key_drop_mutants_killed_under_verify(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    xform: Callable[[Any], Any],
    runner: Callable[[], None],
) -> None:
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    orig_get, orig_put = sm._RefMemo.get, sm._RefMemo.put
    monkeypatch.setattr(sm._RefMemo, "get", lambda self, k: orig_get(self, xform(k)))
    monkeypatch.setattr(sm._RefMemo, "put", lambda self, k, v, nb: orig_put(self, xform(k), v, nb))
    with pytest.raises((AssertionError, TypeError)):
        runner()


@pytest.mark.parametrize(
    ("name", "step_xform", "runner"), _CHILD_MUTANTS, ids=[m[0] for m in _CHILD_MUTANTS]
)
def test_e1_sid_advance_mutants_killed_under_verify(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    step_xform: Callable[[Any], Any],
    runner: Callable[[], None],
) -> None:
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    orig_child = sm._RefMemo.child

    def patched_child(self: sm._RefMemo, sid: int, step: Any) -> int:
        if isinstance(step, tuple):
            mapped = step_xform(step)
            if isinstance(mapped, str) and step[0] == mapped:
                return sid
            return orig_child(self, sid, mapped)
        return orig_child(self, sid, step)

    monkeypatch.setattr(sm._RefMemo, "child", patched_child)
    with pytest.raises(AssertionError):
        runner()


@pytest.mark.parametrize("engine", [False, True])
def test_e1_error_paths_fault_channel_and_impossible_postselection(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)

    c_fc = stim.Circuit("R 0 1\nH 0 1\nI[T] 0\nI[CLIFFORD_FAULT: (0.5, H)] 0\nM 0 1")
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sf = MagicsideSimulator(c_fc, batch_size=16, seed=1, atol=1e-300)
    with pytest.raises(FloatingPointError):
        sf.run()
    sf.clear()
    sf.interactive_do(c_fc[:3])
    sf.finish_interactive_run()

    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    sf_ref = MagicsideSimulator(c_fc, batch_size=16, seed=1, atol=1e-300)
    sf_ref.clear()
    sf_ref.interactive_do(c_fc[:3])
    sf_ref.finish_interactive_run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    _assert_full_state_equal(sf_ref, sf)

    sp = MagicsideSimulator(stim.Circuit("R 0\nI[T] 0\nM 0"), batch_size=2, seed=1)
    sp.interactive_do(stim.Circuit("R 0"))
    with pytest.raises(ImpossiblePostselectionError):
        sp.postselect_z_batch(0, np.array([True, True]), np.array([1, 1], np.uint8))
    sp.clear()
    sp.run()

    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    sp_ref = MagicsideSimulator(stim.Circuit("R 0\nI[T] 0\nM 0"), batch_size=2, seed=1)
    sp_ref.run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    _assert_full_state_equal(sp_ref, sp)

    with pytest.raises(ValueError):
        sp.do_conditional_clifford(
            stim.CircuitInstruction(
                "SPP", [stim.target_x(0), stim.target_combiner(), stim.target_z(0)]
            )
        )
    assert _tab_sid(sp) is None
    sp.clear()
    sp.run()
    _assert_full_state_equal(sp_ref, sp)


@pytest.mark.parametrize("engine", [False, True])
def test_e1_mid_mutator_failure_invalidates_sid_without_poisoning(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    circ = stim.Circuit("H 0 1\nCX 0 1\nM 0 1")
    sim = MagicsideSimulator(circ, batch_size=4, seed=2)
    sim.interactive_do(stim.Circuit("H 0 1"))
    entries_before = len(sim._ref_memo.entries)

    def fail_u(self: sm._TrackedRefTableau, gate_name: str, targets: Any) -> None:
        self.x_kap[0] ^= 2
        raise RuntimeError("injected mid-unitary failure")

    with monkeypatch.context() as mp:
        mp.setattr(sm._TrackedRefTableau, "do_unitary", fail_u)
        with pytest.raises(RuntimeError, match="injected mid-unitary failure"):
            sim.do_conditional_clifford(stim.CircuitInstruction("H", [0]), None)
    assert _tab_sid(sim) is None
    assert len(sim._ref_memo.entries) == entries_before
    sim.interactive_do(stim.Circuit("CX 0 1\nM 0 1"))
    assert sim._ref_memo.hits == 0

    sim.clear()
    sim.interactive_do(stim.Circuit("H 0 1\nCX 0 1"))
    entries_meas_before = len(sim._ref_memo.entries)

    def fail_rot(self: sm._TrackedRefTableau, ra: Any, rb: Any, fk: int) -> None:
        self.z_kap[0] ^= 2
        raise RuntimeError("injected mid-meas failure")

    with monkeypatch.context() as mp:
        mp.setattr(sm._TrackedRefTableau, "apply_right_rot", fail_rot)
        with pytest.raises(RuntimeError, match="injected mid-meas failure"):
            sim.interactive_do(stim.Circuit("M 0 1"))
    assert _tab_sid(sim) is None
    assert len(sim._ref_memo.entries) == entries_meas_before

    sim.clear()
    sim.run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(circ, batch_size=4, seed=2)
    ref.run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    _assert_full_state_equal(ref, sim)


def test_e1_bounded_caps_trie_and_oversized_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    memo = sm._RefMemo(max_entries=5, max_bytes=2048)
    for i in range(20):
        memo.put(("k", i), np.zeros((10, 1), dtype=np.uint64), 200)
    assert len(memo.entries) == 5 and 0 <= memo.total_bytes <= 2048
    assert list(memo.entries.keys()) == [("k", i) for i in range(15, 20)]

    memo.put(("huge", 0), np.zeros(400, dtype=np.uint64), 3200)
    assert ("huge", 0) not in memo.entries and memo.total_bytes <= 2048

    memo.max_trie = 4
    sids = [memo.child(0, ("u", "H", (i,))) for i in range(10)]
    assert len(memo._trie) <= 4 and len(set(sids)) == 10
    re_sid = memo.child(0, ("u", "H", (0,)))
    assert re_sid not in sids

    memo0 = sm._RefMemo(max_entries=0, max_bytes=1024)
    memo0.max_trie = 0
    assert memo0.child(0, ("u", "H", (0,))) == 1 and len(memo0._trie) == 0
    memo0.put(("k", 0), np.zeros(2, dtype=np.uint64), 16)
    assert len(memo0.entries) == 0

    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_MAX_ENTRIES", 5)
    monkeypatch.setattr(sm, "_REF_MEMO_MAX_BYTES", 2048)
    sim = _build_corpus_sim("c04", mode="magic_ds", batch=4)
    sim.run()
    assert len(sim._ref_memo.entries) <= 5 and 0 <= sim._ref_memo.total_bytes <= 2048


@pytest.mark.parametrize("engine", [False, True])
def test_e1_copy_isolation_and_shallow_dict_parity(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nM 0"), batch_size=4, seed=1)
    sim.interactive_do(stim.Circuit("H 0\nI[T] 0"))
    twin = copy.copy(sim)
    if not engine:
        assert twin.__dict__.keys() == sim.__dict__.keys()
        assert all(twin.__dict__[k] is v for k, v in sim.__dict__.items())
    else:
        assert twin._live_ref_tab is not sim._live_ref_tab
    assert twin._ref_memo is not sim._ref_memo
    assert twin._ref_memo.hits == 0 and twin._ref_memo.misses == 0
    twin.interactive_do(stim.Circuit("M 0"))
    twin.finish_interactive_run()
    assert twin._ref_memo.hits == 0 and twin._ref_memo.misses == 0
    if not engine:
        assert _tab_sid(sim) is None
    sim.interactive_do(stim.Circuit("M 0"))
    sim.finish_interactive_run()
    twin.clear()
    twin.run()
    assert twin._ref_memo.misses > 0
    sim.clear()
    sim.run()


@pytest.mark.parametrize("engine", [False, True])
def test_e1_cached_buffers_readonly_and_restored_tab_writable(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    seen_kinds: set[str] = set()
    for case in ("c02b", *_CASES_3):
        sim = _build_corpus_sim(case, mode="magic_ds", batch=4)
        for k in range(3):
            if k > 0:
                sim.clear()
            sim.run()
        seen_kinds.update(key[0] for key in sim._ref_memo.entries)
        arrays = [arr for val, _ in sim._ref_memo.entries.values() for arr in _iter_arrays(val)]
        assert len(arrays) > 0
        for arr in arrays:
            assert arr.flags.writeable is False
            if arr.size > 0:
                with pytest.raises(ValueError):
                    arr.flat[0] = 0
        assert sim._live_ref_tab.x2x.flags.writeable is True
    assert {"u", "spp", "axis", "fac", "meas", "reset", "magic"} <= seen_kinds

    sim_ext = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=2, seed=1)
    sim_ext.clear()
    sim_ext._live_ref_tab.do_unitary("H", [0])
    assert _tab_sid(sim_ext) is None
    p1 = sim_ext.peek_pauli_born_batch([0], "Z")
    assert np.allclose(p1, 0.5)
    assert sim_ext._ref_memo.hits == 0


def _assert_interactive_parity(
    monkeypatch: pytest.MonkeyPatch,
    base_circ: stim.Circuit,
    bodies: list[str],
    *,
    engine: bool,
    seed: int = 1,
    batch_size: int = 4,
) -> MagicsideSimulator:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(base_circ, batch_size=batch_size, seed=seed)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = MagicsideSimulator(base_circ, batch_size=batch_size, seed=seed)
    prefix = str(base_circ)
    for idx, body in enumerate(bodies):
        if idx > 0:
            ref.clear()
            sim.clear()
        seq = stim.Circuit(f"{prefix}\n{body}")
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        ref.interactive_do(seq)
        ref.finish_interactive_run()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        sim.interactive_do(seq)
        sim.finish_interactive_run()
        _assert_full_state_equal(ref, sim)
    return sim


def _assert_circuit_parity_repeats(
    monkeypatch: pytest.MonkeyPatch,
    circ: stim.Circuit,
    *,
    engine: bool,
    runs: int = 3,
    batch_size: int = 4,
    seed: int = 1,
) -> MagicsideSimulator:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(circ, batch_size=batch_size, seed=seed)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = MagicsideSimulator(circ, batch_size=batch_size, seed=seed)
    for idx in range(runs):
        if idx > 0:
            ref.clear()
            sim.clear()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        ref.run()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        sim.run()
        _assert_full_state_equal(ref, sim)
    return sim


@pytest.mark.parametrize("engine", [False, True])
def test_e1_unmonkeypatched_key_slots_and_child_subfields(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    sim_magic = _assert_circuit_parity_repeats(
        monkeypatch,
        stim.Circuit("H 0 1\nI[T] 0\nTICK\nI[T] 1\nM 0 1"),
        engine=engine,
        runs=2,
    )
    assert sim_magic._ref_memo.hits > 0

    sim_r = _assert_interactive_parity(
        monkeypatch,
        stim.Circuit("I 0"),
        ["R 0\nMX 0", "RX 0\nMX 0", "RY 0\nMY 0", "RX 0\nMX 0"],
        engine=engine,
    )
    assert sim_r._ref_memo.hits > 0

    sim_mr = _assert_interactive_parity(
        monkeypatch,
        stim.Circuit("I 0"),
        ["R 0\nM 0", "MR 0\nM 0", "MR !0\nM 0", "MR !0\nM 0"],
        engine=engine,
    )
    assert sim_mr._ref_memo.hits > 0

    _assert_interactive_parity(
        monkeypatch,
        stim.Circuit("I 0"),
        ["M 0", "M !0", "H 0\nM 0", "H 0\nM !0"],
        engine=engine,
    )
    sim_spp = _assert_interactive_parity(
        monkeypatch,
        stim.Circuit("I 0"),
        [
            "SPP X0\nS 0\nH 0\nM 0",
            "SPP Z0\nS 0\nH 0\nM 0",
            "SPP !X0\nS 0\nH 0\nM 0",
            "SPP_DAG X0\nS 0\nH 0\nM 0",
            "SPP_DAG X0\nS 0\nH 0\nM 0",
        ],
        engine=engine,
    )
    assert sim_spp._ref_memo.hits > 0

    sim_cx = _assert_interactive_parity(
        monkeypatch,
        stim.Circuit("I 0 1 2"),
        ["H 0\nCX 0 1\nM 0 1 2", "H 0\nCX 0 2\nM 0 1 2", "H 0\nCX 0 2\nM 0 1 2"],
        engine=engine,
    )
    assert sim_cx._ref_memo.hits > 0


@pytest.mark.parametrize("engine", [False, True])
def test_e1_unmonkeypatched_axis_neg_and_direct_meas_reset_slots(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    mask = np.array([[True, False, False, False]])
    op_pos = stim.CircuitInstruction("SPP", [stim.target_x(0)])
    op_neg = stim.CircuitInstruction("SPP", [stim.target_inv(stim.target_x(0))])
    base = stim.Circuit("I 0")
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(base, batch_size=4, seed=1)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = MagicsideSimulator(base, batch_size=4, seed=1)
    for idx in range(2):
        if idx > 0:
            ref.clear()
            sim.clear()
        for target_sim, flag in ((ref, False), (sim, True)):
            monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", flag)
            target_sim.do_conditional_clifford(op_neg, mask)
            target_sim.do_conditional_clifford(op_pos, mask)
            target_sim.interactive_do(stim.Circuit("I 0\nM 0"))
            target_sim.finish_interactive_run()
        _assert_full_state_equal(ref, sim)
    assert sim._ref_memo.hits > 0

    sim_d = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=2, seed=1)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref_d = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=2, seed=1)
    for idx, (fp, mi) in enumerate([(0.0, 0), (0.2, 0), (0.0, 5), (0.2, 0)]):
        if idx > 0:
            sim_d.clear()
            ref_d.clear()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        r_ref = ref_d._compile_meas_ref(
            [(0, "Z")], invert_result=False, flip_prob=fp, meas_index=mi
        )
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        r_sim = sim_d._compile_meas_ref(
            [(0, "Z")], invert_result=False, flip_prob=fp, meas_index=mi
        )
        assert r_sim.flip_prob == fp and r_sim.meas_index == mi
        assert sm._memo_val_equal(r_sim, r_ref)
    assert sim_d._ref_memo.hits > 0

    reset_cases: list[tuple[int | None, float]] = [
        (None, 0.0), (0, 0.0), (5, 0.0), (0, 0.3), (0, 0.3)
    ]
    for mi_opt, fp in reset_cases:
        sim_d.clear()
        ref_d.clear()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        z_ref = ref_d._compile_reset_ref(
            0, "Z", meas_index=mi_opt, invert_result=False, flip_prob=fp
        )
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        z_sim = sim_d._compile_reset_ref(
            0, "Z", meas_index=mi_opt, invert_result=False, flip_prob=fp
        )
        assert z_sim.meas_index == mi_opt and z_sim.flip_prob == fp
        assert sm._memo_val_equal(z_sim, z_ref)


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize(
    "circ_text",
    [
        "R 0\nSPP X0\nM 0",
        "H 0\nR 0",
        "H 0\nR 0\nM 0",
        "X 0\nR 0\nM 0",
        "H 0\nRX 0\nM 0",
    ],
)
def test_e1_spp_and_nontrivial_reset_hit_paths(
    monkeypatch: pytest.MonkeyPatch, engine: bool, circ_text: str
) -> None:
    sim = _assert_circuit_parity_repeats(
        monkeypatch, stim.Circuit(circ_text), engine=engine, runs=3
    )
    assert sim._ref_memo.hits > 0


def test_e1_byte_cap_eviction_before_entry_cap() -> None:
    memo = sm._RefMemo(max_entries=100, max_bytes=500)
    memo.put(("k", 0), np.zeros(10, dtype=np.uint64), 200)
    memo.put(("k", 1), np.zeros(10, dtype=np.uint64), 200)
    assert len(memo.entries) == 2 and memo.total_bytes == 400
    assert memo.get(("k", 0)) is not None
    memo.put(("k", 2), np.zeros(10, dtype=np.uint64), 200)
    assert memo.total_bytes == 400 and memo.total_bytes <= 500
    assert list(memo.entries.keys()) == [("k", 0), ("k", 2)]
    memo.put(("k", 3), np.zeros(10, dtype=np.uint64), 200)
    memo.put(("k", 4), np.zeros(10, dtype=np.uint64), 200)
    assert memo.total_bytes == 400 and memo.total_bytes <= 500
    assert list(memo.entries.keys()) == [("k", 3), ("k", 4)]


@pytest.mark.parametrize("engine", [False, True])
def test_e1_sim_byte_cap_eviction_parity(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    monkeypatch.setattr(sm, "_REF_MEMO_MAX_ENTRIES", 100)
    monkeypatch.setattr(sm, "_REF_MEMO_MAX_BYTES", 500)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = _build_corpus_sim("c04", mode="magic_ds", batch=4)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = _build_corpus_sim("c04", mode="magic_ds", batch=4)
    for idx in range(2):
        if idx > 0:
            ref.clear()
            sim.clear()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
        ref.run()
        monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
        sim.run()
        _assert_full_state_equal(ref, sim)
    assert len(sim._ref_memo.entries) < 100
    assert sim._ref_memo.misses > len(sim._ref_memo.entries)
    assert 0 < sim._ref_memo.total_bytes <= 500


@pytest.mark.parametrize("engine", [False, True])
def test_e1_mid_reset_failure_invalidates_sid_without_poisoning(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    base = stim.Circuit("I 0")
    sim = MagicsideSimulator(base, batch_size=4, seed=5)
    sim.interactive_do(stim.Circuit("I 0\nH 0\nM 0"))
    sim.finish_interactive_run()
    sim.clear()
    sim.interactive_do(stim.Circuit("I 0\nH 0"))
    entries_before = len(sim._ref_memo.entries)

    def fail_mz(
        self: sm._TrackedRefTableau,
        q: int,
        *,
        is_reset: bool = False,
        m_ref_override: int = 0,
    ) -> Any:
        self.z_kap[q] ^= 2
        raise RuntimeError("injected mid-reset failure")

    with monkeypatch.context() as mp:
        mp.setattr(sm._TrackedRefTableau, "measure_z", fail_mz)
        with pytest.raises(RuntimeError, match="injected mid-reset failure"):
            sim.interactive_do(stim.Circuit("RX 0"))
    assert _tab_sid(sim) is None
    assert len(sim._ref_memo.entries) == entries_before
    hits_after_err = sim._ref_memo.hits
    sim.interactive_do(stim.Circuit("M 0"))
    sim.finish_interactive_run()
    assert sim._ref_memo.hits == hits_after_err

    sim.clear()
    sim.interactive_do(stim.Circuit("I 0\nH 0\nRX 0\nM 0"))
    sim.finish_interactive_run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref = MagicsideSimulator(base, batch_size=4, seed=5)
    for body in ("H 0\nM 0", "TICK", "H 0\nRX 0\nM 0"):
        if ref._batches_completed > 0:
            ref.clear()
        ref.interactive_do(stim.Circuit(f"I 0\n{body}"))
        ref.finish_interactive_run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    _assert_full_state_equal(ref, sim)


@pytest.mark.parametrize("engine", [False, True])
def test_e1_external_do_spp_and_measure_z_invalidate_sid(
    monkeypatch: pytest.MonkeyPatch, engine: bool
) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    monkeypatch.setattr(sm, "_REF_MEMO_VERIFY", True)
    c_spp = stim.Circuit("H 0\nM 0")
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref_spp = MagicsideSimulator(c_spp, batch_size=2, seed=1)
    ref_spp.run()
    ref_spp.clear()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim_spp = MagicsideSimulator(c_spp, batch_size=2, seed=1)
    sim_spp.run()
    sim_spp.clear()
    assert _tab_sid(sim_spp) is not None
    hits_before_spp = sim_spp._ref_memo.hits
    sim_spp._live_ref_tab.do_spp([(0, "X")])
    ref_spp._live_ref_tab.do_spp([(0, "X")])
    assert _tab_sid(sim_spp) is None
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref_spp.run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim_spp.run()
    assert sim_spp._ref_memo.hits == hits_before_spp
    _assert_full_state_equal(ref_spp, sim_spp)

    c_mz = stim.Circuit("X 0\nM 0")
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref_mz = MagicsideSimulator(c_mz, batch_size=2, seed=1)
    ref_mz.run()
    ref_mz.clear()
    ref_mz.interactive_do(stim.Circuit("X 0"))
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim_mz = MagicsideSimulator(c_mz, batch_size=2, seed=1)
    sim_mz.run()
    sim_mz.clear()
    sim_mz.interactive_do(stim.Circuit("X 0"))
    assert _tab_sid(sim_mz) is not None
    sim_mz._live_ref_tab.measure_z(0, is_reset=True)
    ref_mz._live_ref_tab.measure_z(0, is_reset=True)
    assert _tab_sid(sim_mz) is None
    hits_before_mz = sim_mz._ref_memo.hits
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", False)
    ref_mz.interactive_do(stim.Circuit("M 0"))
    ref_mz.finish_interactive_run()
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim_mz.interactive_do(stim.Circuit("M 0"))
    sim_mz.finish_interactive_run()
    assert sim_mz._ref_memo.hits == hits_before_mz
    _assert_full_state_equal(ref_mz, sim_mz)
