# Copyright 2026 Google LLC
"""WP8 Section 7 diagnostic counter and benchmark CLI tests for the stabilizer-branch engine."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside import LeakageUint8Magic, MagicsideSimulator, QueryBudgetExceededError
from stimside._native import branch_kernels as bk
from stimside import simulator_magic as sm


EXPECTED_14_KEYS: frozenset[str] = frozenset(
    {
        "store_nodes",
        "store_bytes_estimate",
        "store_clears",
        "transitions_new",
        "transitions_hit",
        "resolve_calls",
        "splits",
        "subcache_hits",
        "subcache_misses",
        "meas_definite",
        "meas_absorbed",
        "meas_forked",
        "branches_hist",
        "faults_realized",
    }
)
EXPECTED_8_KINDS: frozenset[str] = frozenset(
    {"T", "CCZ", "SKIP", "CFAULT", "MEAS", "PEEK", "POST", "URESET"}
)
EXPECTED_3_FAULTS: frozenset[str] = frozenset(
    {"pauli_faults", "skipped_gates", "skipped_cczs"}
)


def _assert_exact_int(val: Any, label: str) -> None:
    assert type(val) is int, f"{label} expected exact int, got {type(val).__name__}: {val!r}"
    assert val >= 0, f"{label} expected >= 0, got {val}"


def _assert_section7_structure(cnt: dict[str, Any]) -> None:
    assert set(cnt.keys()) == EXPECTED_14_KEYS
    for key in (
        "store_nodes",
        "store_bytes_estimate",
        "store_clears",
        "resolve_calls",
        "splits",
        "subcache_hits",
        "subcache_misses",
        "meas_definite",
        "meas_absorbed",
        "meas_forked",
    ):
        _assert_exact_int(cnt[key], key)
    assert cnt["store_nodes"] >= 1
    assert cnt["store_bytes_estimate"] > 0

    for dkey in ("transitions_new", "transitions_hit"):
        dval = cnt[dkey]
        assert isinstance(dval, dict)
        assert set(dval.keys()) == EXPECTED_8_KINDS
        for k, v in dval.items():
            _assert_exact_int(v, f"{dkey}[{k}]")

    fr = cnt["faults_realized"]
    assert isinstance(fr, dict)
    assert set(fr.keys()) == EXPECTED_3_FAULTS
    for k, v in fr.items():
        _assert_exact_int(v, f"faults_realized[{k}]")

    bh = cnt["branches_hist"]
    assert isinstance(bh, dict)
    for k, v in bh.items():
        _assert_exact_int(k, "branches_hist key")
        assert k >= 1
        _assert_exact_int(v, f"branches_hist[{k}]")


def _pkg_and_benchmarks() -> tuple[Path, Path | None]:
    src_dir = Path(__file__).resolve().parents[1]
    pkg_root = Path(__file__).resolve().parents[2]
    bdir = pkg_root / "benchmarks"
    if (bdir / "bench_magic_suite.py").is_file() and (bdir / "compare_magic_engines.py").is_file():
        return src_dir, bdir
    return src_dir, None


def _run_bench_subprocess(src_dir: Path, script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(src_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-B", str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_section7_pre_run_terms_guards_copy_isolation_and_zero_qubit() -> None:
    circ = stim.Circuit("H 0\nI[T] 0\nH 0\nM 0\n")
    batch_size = 8

    # 1. Terms mode rejects branch counter getters.
    sim_terms = MagicsideSimulator(circ, batch_size=batch_size, seed=11, _engine="terms")
    with pytest.raises(RuntimeError, match="_branch_counters"):
        sim_terms._branch_counters()
    with pytest.raises(RuntimeError, match="_branch_peak_per_shot"):
        sim_terms._branch_peak_per_shot()

    # 2. Pre-run branch mode structure and initial values.
    sim = MagicsideSimulator(circ, batch_size=batch_size, seed=11, _engine="branches")
    c0 = sim._branch_counters()
    _assert_section7_structure(c0)
    assert c0["store_nodes"] == 1
    assert c0["store_bytes_estimate"] > 0
    assert c0["store_clears"] == 0
    assert c0["resolve_calls"] == 0
    assert c0["splits"] == 0
    assert c0["subcache_hits"] == 0
    assert c0["subcache_misses"] == 0
    assert c0["meas_definite"] == 0
    assert c0["meas_absorbed"] == 0
    assert c0["meas_forked"] == 0
    assert c0["branches_hist"] == {}
    assert all(v == 0 for v in c0["transitions_new"].values())
    assert all(v == 0 for v in c0["transitions_hit"].values())
    assert all(v == 0 for v in c0["faults_realized"].values())

    pk0 = sim._branch_peak_per_shot()
    assert isinstance(pk0, np.ndarray)
    assert pk0.dtype == np.int16
    assert pk0.shape == (batch_size,)
    assert np.all(pk0 == 1)

    # 3. Defensive copy isolation on returned dicts and peak array.
    c0["store_nodes"] = 999
    c0["transitions_new"]["T"] = 777
    c0["transitions_hit"]["MEAS"] = 666
    c0["faults_realized"]["pauli_faults"] = 555
    c0["branches_hist"][99] = 444
    pk0[:] = 42

    c0_again = sim._branch_counters()
    assert c0_again["store_nodes"] == 1
    assert c0_again["transitions_new"]["T"] == 0
    assert c0_again["transitions_hit"]["MEAS"] == 0
    assert c0_again["faults_realized"]["pauli_faults"] == 0
    assert c0_again["branches_hist"] == {}
    assert np.all(sim._branch_peak_per_shot() == 1)

    # 4. Zero-qubit circuit (MPAD only).
    sim_zero = MagicsideSimulator(stim.Circuit("MPAD 0 1\n"), batch_size=batch_size, seed=12, _engine="branches")
    sim_zero.run()
    cz = sim_zero._branch_counters()
    _assert_section7_structure(cz)
    assert cz["store_nodes"] == 1
    assert cz["branches_hist"] == {1: batch_size}
    assert cz["faults_realized"] == {"pauli_faults": 0, "skipped_gates": 0, "skipped_cczs": 0}
    pkz = sim_zero._branch_peak_per_shot()
    assert pkz.dtype == np.int16 and pkz.shape == (batch_size,) and np.all(pkz == 1)

    # 5. Failed run() does not fold partial histogram/faults; clear() recovers cleanly.
    sim_cap = MagicsideSimulator(
        stim.Circuit("H 0\nX_ERROR(0.5) 0\nI[T] 0\nH 0\nM 0\n"),
        batch_size=batch_size,
        seed=13,
        max_components=1,
        on_budget_exceeded="raise",
        _engine="branches",
    )
    with pytest.raises(QueryBudgetExceededError):
        sim_cap.run()
    cf = sim_cap._branch_counters()
    _assert_section7_structure(cf)
    assert cf["branches_hist"] == {}
    assert cf["faults_realized"] == {"pauli_faults": 0, "skipped_gates": 0, "skipped_cczs": 0}
    sim_cap.max_components = 4
    sim_cap.clear()
    assert np.all(sim_cap._branch_peak_per_shot() == 1)
    sim_cap.run()
    cr = sim_cap._branch_counters()
    assert sum(cr["branches_hist"].values()) == batch_size


def test_section7_all_8_transition_kinds_new_hit_and_meas_invariants() -> None:
    batch_size = 32
    circ = stim.Circuit(
        """
        R 0 1 2 3 4
        H 0 1 2 3
        I[T] 0
        I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 1-->2)] 0
        MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) : 0] 0
        CX 0 1
        SPP Z0*X2
        I[CLIFFORD_FAULT: (0.25, CX)] 1 2
        I[CCZ] 0 1 2 1 2 3
        X_ERROR(0.5) 3
        I_ERROR[LEAKAGE_TRANSITION_1: (1.0, 2-->0)] 0
        M 4
        H 4
        M 4
        H 0 1 2 3
        M[LEAKAGE_PROJECTION_Z: (1.0, 2)] 0 1 2 3
        """
    )
    sim = MagicsideSimulator(
        circ,
        op_handler=LeakageUint8Magic(unconditional_condition_on_U=True),
        batch_size=batch_size,
        seed=8101,
        _engine="branches",
    )

    # Batch 1
    sim.run()
    c1 = sim._branch_counters()
    _assert_section7_structure(c1)
    for k in EXPECTED_8_KINDS:
        assert c1["transitions_new"][k] > 0, f"expected transitions_new[{k}] > 0, got {c1['transitions_new']}"
    assert c1["resolve_calls"] > 0
    assert c1["splits"] > 0
    assert c1["subcache_hits"] > 0
    assert c1["subcache_misses"] > 0
    assert c1["meas_definite"] > 0
    assert c1["meas_absorbed"] > 0
    assert c1["meas_forked"] > 0
    assert (
        c1["meas_definite"] + c1["meas_absorbed"] + c1["meas_forked"]
        == c1["transitions_new"]["MEAS"] + c1["transitions_new"]["POST"] + c1["transitions_new"]["URESET"]
    )

    # Batch 2 (warm store)
    sim.clear()
    sim.run()
    c2 = sim._branch_counters()
    _assert_section7_structure(c2)
    for k in EXPECTED_8_KINDS:
        assert c2["transitions_hit"][k] > 0, f"expected transitions_hit[{k}] > 0, got {c2['transitions_hit']}"
    assert (
        c2["meas_definite"] + c2["meas_absorbed"] + c2["meas_forked"]
        == c2["transitions_new"]["MEAS"] + c2["transitions_new"]["POST"] + c2["transitions_new"]["URESET"]
    )
    assert sum(c2["branches_hist"].values()) == 2 * batch_size
    pk2 = sim._branch_peak_per_shot()
    assert pk2.dtype == np.int16 and pk2.shape == (batch_size,)
    assert int(pk2.max()) >= 2
    for val in np.unique(pk2).tolist():
        assert int(val) in c2["branches_hist"]

    # Dedicated PEEK check: anticommuting PEEK resolves without adding nodes or meas_* counts;
    # repeat PEEK hits cache without resolving again.
    sim_pk = MagicsideSimulator(stim.Circuit("R 0\nH 0\nI[T] 0\n"), batch_size=8, seed=8102, _engine="branches")
    sim_pk.run()
    cb = sim_pk._branch_counters()
    p1_first = sim_pk.peek_pauli_born_batch([0], "X")[0]
    expected_p1 = (2.0 - math.sqrt(2.0)) / 4.0
    assert np.allclose(p1_first, expected_p1, atol=1e-12)
    ca1 = sim_pk._branch_counters()
    assert ca1["store_nodes"] == cb["store_nodes"]
    assert ca1["meas_definite"] == cb["meas_definite"]
    assert ca1["meas_absorbed"] == cb["meas_absorbed"]
    assert ca1["meas_forked"] == cb["meas_forked"]
    assert ca1["transitions_new"]["PEEK"] == cb["transitions_new"]["PEEK"] + 1
    assert ca1["transitions_hit"]["PEEK"] == cb["transitions_hit"]["PEEK"]
    assert ca1["resolve_calls"] == cb["resolve_calls"] + 1

    p1_second = sim_pk.peek_pauli_born_batch([0], "X")[0]
    assert np.allclose(p1_second, expected_p1, atol=1e-12)
    ca2 = sim_pk._branch_counters()
    assert ca2["store_nodes"] == ca1["store_nodes"]
    assert ca2["meas_definite"] == ca1["meas_definite"]
    assert ca2["meas_absorbed"] == ca1["meas_absorbed"]
    assert ca2["meas_forked"] == ca1["meas_forked"]
    assert ca2["transitions_new"]["PEEK"] == ca1["transitions_new"]["PEEK"]
    assert ca2["transitions_hit"]["PEEK"] == ca1["transitions_hit"]["PEEK"] + 1
    assert ca2["resolve_calls"] == ca1["resolve_calls"]


def test_section7_clifford_absorption_vs_nonclifford_resolve() -> None:
    batch_size = 16
    sim = MagicsideSimulator(stim.Circuit("H 0\nCX 0 1\nM 0 1\n"), batch_size=batch_size, seed=8201, _engine="branches")
    sim.run()
    c1 = sim._branch_counters()
    _assert_section7_structure(c1)
    assert c1["resolve_calls"] == 0
    assert c1["splits"] == 0
    assert c1["meas_forked"] == 0
    assert c1["meas_absorbed"] == 1
    assert c1["meas_definite"] == 1
    assert c1["transitions_new"]["MEAS"] == 2
    assert c1["branches_hist"] == {1: batch_size}

    sim.clear()
    sim.run()
    c2 = sim._branch_counters()
    assert c2["transitions_new"]["MEAS"] == 2
    assert c2["transitions_hit"]["MEAS"] == 2
    assert c2["branches_hist"] == {1: 2 * batch_size}

    # Zero-shot sel_mask checks: no phantom store transitions or skipped counts.
    sim_noop = MagicsideSimulator(stim.Circuit("R 0 1 2\nH 0 1 2\n"), batch_size=batch_size, seed=8202, _engine="branches")
    sim_noop.run()
    c_before = sim_noop._branch_counters()
    all_true = np.ones((1, batch_size), dtype=np.bool_)
    all_false_1d = np.zeros(batch_size, dtype=np.bool_)
    all_false_2d = np.zeros((1, batch_size), dtype=np.bool_)
    all_false_q = np.zeros((3, batch_size), dtype=np.bool_)
    desired_zeros = np.zeros(batch_size, dtype=np.uint8)

    sim_noop.do_conditional_clifford(stim.CircuitInstruction("CX", [0, 1]), all_true)
    sim_noop.do_conditional_magic(stim.CircuitInstruction("I", [0, 1, 2], tag="CCZ"), [0, 1, 2], all_false_2d)
    sim_noop.postselect_z_batch(0, all_false_1d, desired_zeros, pauli="Z")
    sim_noop.do_unmatched_noisy_reset_z(all_false_q, all_false_q, pauli="Z")

    c_after = sim_noop._branch_counters()
    assert c_after["store_nodes"] == c_before["store_nodes"]
    assert c_after["transitions_new"] == c_before["transitions_new"]
    assert c_after["transitions_hit"] == c_before["transitions_hit"]
    assert c_after["resolve_calls"] == c_before["resolve_calls"]
    assert c_after["meas_definite"] == c_before["meas_definite"]
    assert c_after["meas_absorbed"] == c_before["meas_absorbed"]
    assert c_after["meas_forked"] == c_before["meas_forked"]
    assert sim_noop._branch_batch is not None
    assert sim_noop._branch_batch.faults_realized["skipped_gates"] == 0


def test_section7_cap_clear_inplace_and_copy_shared_store_continuity(monkeypatch: pytest.MonkeyPatch) -> None:
    circ = stim.Circuit(
        """
        R 0 1 2
        H 0 1 2
        I[T] 0 1
        DEPOLARIZE1(0.25) 0 1 2
        CX 0 1 1 2
        I[CCZ] 0 1 2
        H 0 1
        M 0 1
        """
    )
    sim1 = MagicsideSimulator(circ, batch_size=16, seed=8301, _engine="branches")
    sim1.run()
    sim1.clear()
    sim1.run()

    c_warm = sim1._branch_counters()
    _assert_section7_structure(c_warm)
    assert c_warm["store_nodes"] > 5
    assert c_warm["store_clears"] == 0
    bytes_warm = c_warm["store_bytes_estimate"]
    sv1_before = sim1.state_vector(0)

    sim2 = copy.copy(sim1)
    old_store = sim1._branch_store
    assert old_store is not None
    assert sim2._branch_store is old_store
    assert old_store.owner_count() == 2

    cumulative_keys = (
        "transitions_new",
        "transitions_hit",
        "resolve_calls",
        "splits",
        "subcache_hits",
        "subcache_misses",
        "meas_definite",
        "meas_absorbed",
        "meas_forked",
        "branches_hist",
        "faults_realized",
    )

    monkeypatch.setattr(sm, "_BRANCH_STORE_MAX_NODES", 1)

    # Shared owner clear(): sim2 retires old_store snapshot and gets fresh store; sim1 unaffected.
    sim2.clear()
    assert sim2._branch_store is not old_store
    assert sim1._branch_store is old_store
    assert old_store.owner_count() == 1

    c2 = sim2._branch_counters()
    _assert_section7_structure(c2)
    assert c2["store_nodes"] == 1
    assert 0 < c2["store_bytes_estimate"] < bytes_warm
    assert c2["store_clears"] == 1
    for key in cumulative_keys:
        assert c2[key] == c_warm[key], f"sim2 mismatch on {key}: {c2[key]} vs {c_warm[key]}"

    c1_still = sim1._branch_counters()
    assert c1_still == c_warm
    assert np.allclose(sim1.state_vector(0), sv1_before, atol=1e-15)

    # Sole owner clear(): sim1 clears old_store in-place preserving store identity and counters.
    sim1.clear()
    assert sim1._branch_store is old_store
    c1_cleared = sim1._branch_counters()
    _assert_section7_structure(c1_cleared)
    assert c1_cleared["store_nodes"] == 1
    assert 0 < c1_cleared["store_bytes_estimate"] < bytes_warm
    assert c1_cleared["store_clears"] == 1
    for key in cumulative_keys:
        assert c1_cleared[key] == c_warm[key], f"sim1 mismatch on {key}: {c1_cleared[key]} vs {c_warm[key]}"

    del sim1, sim2, old_store
    gc.collect()
    assert bk.BranchStore.live_stores() == 0
    assert bk.BranchState.live_states() == 0


def test_section7_faults_realized_all_noise_types_sparse_vs_traced() -> None:
    batch_size = 32
    circ = stim.Circuit(
        """
        R 0 1 2 3
        H 0 1 2 3
        X_ERROR(0.35) 0 1
        Y_ERROR(0.25) 2
        Z_ERROR(0.3) 3
        X_ERROR(1.0) 0 0
        DEPOLARIZE1(0.4) 0 1 2 3
        PAULI_CHANNEL_1(1.0, 0.0, 0.0) 1
        PAULI_CHANNEL_1(0.0, 0.0, 0.0) 2
        PAULI_CHANNEL_1(0.2, 0.2, 0.2) 3
        DEPOLARIZE2(0.4) 0 1 2 3
        PAULI_CHANNEL_2(0.1, 0.05, 0.05, 0.05, 0.1, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05) 0 2
        CORRELATED_ERROR(0.4) X0 Z1
        ELSE_CORRELATED_ERROR(0.5) Y1 X2
        CORRELATED_ERROR(1.0) X0 Z0
        HERALDED_ERASE(0.3) 1 3
        HERALDED_PAULI_CHANNEL_1(0.1, 0.1, 0.1, 0.1) 0 2
        M 0
        CX rec[-1] 1
        I[CLIFFORD_FAULT: (0.3, S)] 2
        I[T] 1
        H 1 2 3
        M 1 2 3
        """
    )

    sim_sparse = MagicsideSimulator(circ, batch_size=batch_size, seed=8401, _engine="branches")
    sim_traced = MagicsideSimulator(circ, batch_size=batch_size, seed=8401, _engine="branches")
    sim_traced._enable_branch_trace(True)

    sim_sparse.run()
    sim_traced.run()
    assert np.array_equal(sim_sparse.get_final_measurement_records(), sim_traced.get_final_measurement_records())
    cs = sim_sparse._branch_counters()
    ct = sim_traced._branch_counters()
    assert cs["faults_realized"] == ct["faults_realized"]
    assert cs["faults_realized"]["pauli_faults"] > 0

    # Exact PAULI_CHANNEL_2 1-body (IX -> 1 per shot) vs 2-body (XX -> 2 per shot) accounting.
    c_ix = stim.Circuit("PAULI_CHANNEL_2(1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0) 0 1\nM 0 1\n")
    sim_ix = MagicsideSimulator(c_ix, batch_size=batch_size, seed=8402, _engine="branches")
    sim_ix.run()
    assert sim_ix._branch_counters()["faults_realized"]["pauli_faults"] == batch_size

    c_xx = stim.Circuit("PAULI_CHANNEL_2(0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0) 0 1\nM 0 1\n")
    sim_xx = MagicsideSimulator(c_xx, batch_size=batch_size, seed=8403, _engine="branches")
    sim_xx.run()
    assert sim_xx._branch_counters()["faults_realized"]["pauli_faults"] == 2 * batch_size

    # Classical feedback-only circuit: zero faults realized.
    sim_fb = MagicsideSimulator(
        stim.Circuit("H 0\nM 0\nCX rec[-1] 1\nM 1\n"),
        batch_size=batch_size,
        seed=8404,
        _engine="branches",
    )
    sim_fb.run()
    assert sim_fb._branch_counters()["faults_realized"] == {
        "pauli_faults": 0,
        "skipped_gates": 0,
        "skipped_cczs": 0,
    }

    # CLIFFORD_FAULT-only circuit: pauli_faults == 0 (CFAULT transitions still counted).
    sim_cf = MagicsideSimulator(
        stim.Circuit("H 0\nI[CLIFFORD_FAULT: (0.5, S)] 0\nH 0\nM 0\n"),
        batch_size=batch_size,
        seed=8405,
        _engine="branches",
    )
    sim_cf.run()
    ccf = sim_cf._branch_counters()
    assert ccf["faults_realized"] == {"pauli_faults": 0, "skipped_gates": 0, "skipped_cczs": 0}
    assert ccf["transitions_new"]["CFAULT"] > 0


def test_benchmark_bench_magic_suite_jsonl_sc_and_rand(tmp_path: Path) -> None:
    src_dir, bdir = _pkg_and_benchmarks()
    if bdir is None:
        pytest.skip("benchmarks/ directory not present (wheel-only install)")
    script = bdir / "bench_magic_suite.py"

    # 1. sc suite: 2 distances * 3 engines = 6 "ok" records conforming to stimside-paper-results/1.
    sc_jsonl = tmp_path / "nested" / "sc.jsonl"
    res_sc = _run_bench_subprocess(src_dir, script, ["sc", "--reps", "1", "--jsonl", str(sc_jsonl)])
    assert res_sc.returncode == 0, f"stdout={res_sc.stdout}\nstderr={res_sc.stderr}"
    assert "Coset" in res_sc.stdout and "Magic" in res_sc.stdout and "branch" in res_sc.stdout
    assert sc_jsonl.is_file()

    sc_lines = [line for line in sc_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sc_lines) == 6
    sc_recs = [json.loads(line) for line in sc_lines]

    engine_names = [r["engine"]["name"] for r in sc_recs]
    assert engine_names.count("cosetside") == 2
    assert engine_names.count("magicside_terms") == 2
    assert engine_names.count("magicside_branches") == 2

    for rec in sc_recs:
        assert rec["schema"] == "stimside-paper-results/1"
        assert rec["figure"] == "bench_magic_suite"
        assert rec["status"] == "ok"
        assert "error" not in rec

        eng = rec["engine"]
        assert set(eng.keys()) >= {"name", "version", "package_version", "options"}
        assert eng["version"].startswith("stimside@")
        assert isinstance(eng["options"], dict)

        wl = rec["workload"]
        assert set(wl.keys()) >= {"id", "file", "sha256", "qubits", "params"}
        assert len(wl["sha256"]) == 64
        assert wl["qubits"] > 0

        mach = rec["machine"]
        assert set(mach.keys()) >= {
            "env_file",
            "cores",
            "loadavg_before",
            "loadavg_after",
            "cpu_model",
            "cpu_count",
            "platform",
            "python",
            "numpy",
            "stim",
        }
        assert len(mach["loadavg_before"]) == 3
        assert len(mach["loadavg_after"]) == 3

        run = rec["run"]
        assert set(run.keys()) >= {
            "seed",
            "batch_size",
            "runs_per_rep",
            "reps",
            "shots",
            "seconds_total",
            "seconds_construct",
            "warmup_seconds",
            "best_rep_seconds",
            "rep_seconds",
            "rss_mb",
        }
        assert run["seed"] == 1
        assert run["reps"] == 1
        assert run["shots"] > 0
        assert run["seconds_total"] > 0.0
        assert run["rss_mb"] > 0.0

        met = rec["metrics"]
        assert set(met.keys()) >= {
            "shots_per_s",
            "steady_shots_per_s",
            "accepted",
            "first_det",
            "metrics_shots",
            "store_states",
            "peak_branches",
            "faults_by_type",
            "branch_counters",
        }
        assert met["shots_per_s"] > 0.0
        assert met["steady_shots_per_s"] > 0.0
        assert isinstance(met["accepted"], int) and 0 <= met["accepted"] <= met["metrics_shots"]
        assert isinstance(met["first_det"], list) and sum(met["first_det"]) == met["metrics_shots"]

        if eng["name"] == "magicside_branches":
            assert isinstance(met["store_states"], int) and met["store_states"] >= 1
            assert isinstance(met["peak_branches"], int) and met["peak_branches"] >= 1
            assert set(met["faults_by_type"].keys()) == {"P", "L", "S2", "S3"}
            assert met["faults_by_type"]["L"] is None
            bc = met["branch_counters"]
            assert isinstance(bc, dict)
            assert set(bc.keys()) == EXPECTED_14_KEYS
            assert set(bc["transitions_new"].keys()) == EXPECTED_8_KINDS
            assert set(bc["transitions_hit"].keys()) == EXPECTED_8_KINDS
            assert set(bc["faults_realized"].keys()) == EXPECTED_3_FAULTS
        else:
            assert met["store_states"] is None
            assert met["peak_branches"] is None
            assert met["faults_by_type"] is None
            assert met["branch_counters"] is None

    # 2. rand suite via --json-lines alias: live ok + branch failed cleanly on max_components=4096.
    rand_jsonl = tmp_path / "rand.jsonl"
    res_rand = _run_bench_subprocess(src_dir, script, ["rand", "--reps", "1", "--json-lines", str(rand_jsonl)])
    assert res_rand.returncode == 0, f"stdout={res_rand.stdout}\nstderr={res_rand.stderr}"
    assert "FAILED (QueryBudgetExceededError)" in res_rand.stdout

    rand_recs = [json.loads(line) for line in rand_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rand_recs) == 2
    by_engine = {r["engine"]["name"]: r for r in rand_recs}
    assert set(by_engine.keys()) == {"magicside_terms", "magicside_branches"}
    assert by_engine["magicside_terms"]["status"] == "ok"
    assert by_engine["magicside_terms"]["metrics"]["steady_shots_per_s"] > 0.0

    rb = by_engine["magicside_branches"]
    assert rb["schema"] == "stimside-paper-results/1"
    assert rb["status"] == "failed"
    assert "QueryBudgetExceededError" in rb["error"]
    assert rb["run"]["shots"] == 0
    assert rb["metrics"]["shots_per_s"] is None
    assert rb["metrics"]["steady_shots_per_s"] is None
    assert rb["metrics"]["peak_branches"] == 4096
    assert isinstance(rb["metrics"]["store_states"], int) and rb["metrics"]["store_states"] >= 1
    assert set(rb["metrics"]["branch_counters"].keys()) == EXPECTED_14_KEYS

    # 3. Hostile CLI arguments exit with rc == 2.
    res_bad_reps = _run_bench_subprocess(src_dir, script, ["sc", "--reps", "0"])
    assert res_bad_reps.returncode == 2
    res_dir_jsonl = _run_bench_subprocess(src_dir, script, ["sc", "--reps", "1", "--jsonl", str(tmp_path)])
    assert res_dir_jsonl.returncode == 2


def test_benchmark_compare_magic_engines_cli_synthetic(tmp_path: Path) -> None:
    src_dir, bdir = _pkg_and_benchmarks()
    if bdir is None:
        pytest.skip("benchmarks/ directory not present (wheel-only install)")
    script = bdir / "compare_magic_engines.py"

    synth_path = tmp_path / "synth.stim"
    synth_path.write_text(
        "\n".join(
            [
                "R 0 1 2 3",
                "H 0 1 2",
                "I[T] 0 1",
                "I[CCZ] 0 1 2",
                "CX 0 3 1 3",
                "DEPOLARIZE1(0.05) 0 1 2 3",
                "H 0 1 2",
                "M 0 1 2 3",
                "DETECTOR rec[-1]",
                "DETECTOR rec[-2]",
                "DETECTOR rec[-3] rec[-4]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    json_out = tmp_path / "out" / "cmp.json"
    res = _run_bench_subprocess(
        src_dir,
        script,
        [
            "--circuit",
            str(synth_path),
            "--shots",
            "256",
            "--batch-size",
            "64",
            "--seed",
            "7",
            "--json-out",
            str(json_out),
            "--check",
        ],
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "verdict=PASS" in res.stdout
    assert json_out.is_file()

    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(report.keys()) == {
        "circuit",
        "num_qubits",
        "num_measurements",
        "num_detectors",
        "leakage",
        "shots",
        "batch_size",
        "terms",
        "branches",
        "speedup",
        "stats",
    }
    assert report["num_qubits"] == 4
    assert report["num_measurements"] == 4
    assert report["num_detectors"] == 3
    assert report["leakage"] is False
    assert report["shots"] == 256
    assert report["batch_size"] == 64
    assert report["stats"]["passed"] is True
    assert report["terms"]["shots_per_s"] > 0.0
    assert report["branches"]["shots_per_s"] > 0.0
    cnt = report["branches"]["counters"]
    assert set(cnt.keys()) == EXPECTED_14_KEYS
    assert set(cnt["transitions_new"].keys()) == EXPECTED_8_KINDS
    assert set(cnt["transitions_hit"].keys()) == EXPECTED_8_KINDS
    assert set(cnt["faults_realized"].keys()) == EXPECTED_3_FAULTS
