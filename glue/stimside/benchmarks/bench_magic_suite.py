# Copyright 2026 Google LLC
"""Magicside throughput benchmarks (warm, best-of-N), run as a script; never collected by pytest.

    taskset -c 2 python benchmarks/bench_magic_suite.py [sc] [xmem] [rand] [--reps N] [--jsonl PATH]

See README.md for the expected baseline numbers and how to compare runs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _circuits import build_leakage_surface_code, random_t_cx, t_count, xmem_t

import stimside
from stimside import (
    CosetsideSimulator,
    LeakageUint8Coset,
    LeakageUint8Magic,
    MagicsideSimulator,
    QueryBudgetExceededError,
)

BATCH = 64
BRANCH_MAX_COMPONENTS = 4096
ENGINES = (
    ("live", {}),
    ("compiled", {"fault_mode": "SELECTOR"}),
    ("query", {"backend": "QUERY"}),
    ("branch", {"_engine": "branches", "max_components": BRANCH_MAX_COMPONENTS}),
)


def _git_commit_sha():
    try:
        res = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = res.stdout.strip()
        if res.returncode == 0 and len(sha) == 40:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(ru) / (1024.0 * 1024.0)
    return float(ru) / 1024.0


def _pinned_cores():
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            return []
    return []


def _machine_base():
    return {
        "env_file": None,
        "cores": _pinned_cores(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "stim": stim.__version__,
    }


def _normalize_counters(raw):
    out = {}
    for k, v in raw.items():
        if k == "branches_hist":
            out[k] = {str(bk): int(bv) for bk, bv in sorted(v.items(), key=lambda kv: int(kv[0]))}
        elif isinstance(v, dict):
            out[k] = {str(sk): int(sv) for sk, sv in v.items()}
        else:
            out[k] = int(v)
    return out


def _detector_summary(dets, n_det):
    n_shots = int(dets.shape[0])
    if n_det == 0:
        return n_shots, [n_shots]
    any_det = dets.any(axis=1)
    first = np.where(any_det, dets.argmax(axis=1), n_det)
    hist = np.bincount(first, minlength=n_det + 1)
    return int((~any_det).sum()), [int(x) for x in hist.tolist()]


def _format_entry_rate(rec, *, with_unit=True):
    if rec["status"] == "ok":
        sps = rec["metrics"]["steady_shots_per_s"]
        return f"{sps:7.1f} shots/s" if with_unit else f"{sps:7.1f}"
    err_name = rec.get("error", "Error").split(":", 1)[0]
    tag = "FAILED" if rec["status"] == "failed" else "UNSUPPORTED"
    return f"{tag} ({err_name})"


def best_throughput(make_sim, runs, reps):
    """Shots/s of `clear(); run(); get_detector_flips()`: best of `reps` after one warm-up run."""
    sim = make_sim()
    sim.run()
    sim.get_detector_flips()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(runs):
            sim.clear()
            sim.run()
            sim.get_detector_flips()
        best = min(best, time.perf_counter() - t0)
    return BATCH * runs / best


def _measure_entry(
    make_sim,
    runs,
    reps,
    *,
    circuit,
    engine_name,
    engine_options,
    workload_id,
    workload_file,
    workload_params,
    machine_base,
    git_sha,
    is_branch=False,
):
    load_before = [round(x, 3) for x in os.getloadavg()]
    t_entry0 = time.perf_counter()
    sim = None
    seconds_construct = None
    warmup_seconds = None
    rep_seconds = []
    best = None
    last_dets = None
    status = "ok"
    error = None
    total_shots = 0
    shots_per_s = None
    steady_shots_per_s = None
    accepted = None
    first_det = None
    metrics_shots = 0

    try:
        sim = make_sim()
        seconds_construct = time.perf_counter() - t_entry0
        t_w0 = time.perf_counter()
        sim.run()
        last_dets = sim.get_detector_flips()
        warmup_seconds = time.perf_counter() - t_w0
        best_val = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(runs):
                sim.clear()
                sim.run()
                last_dets = sim.get_detector_flips()
            dt = time.perf_counter() - t0
            rep_seconds.append(dt)
            if dt < best_val:
                best_val = dt
        best = best_val
    except (QueryBudgetExceededError, MemoryError) as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    except NotImplementedError as exc:
        status = "unsupported"
        error = f"{type(exc).__name__}: {exc}"

    seconds_total = time.perf_counter() - t_entry0
    load_after = [round(x, 3) for x in os.getloadavg()]

    if status == "ok" and last_dets is not None and best is not None:
        accepted, first_det = _detector_summary(last_dets, int(circuit.num_detectors))
        metrics_shots = int(last_dets.shape[0])
        total_shots = BATCH * (1 + runs * reps)
        shots_per_s = float(total_shots / seconds_total) if seconds_total > 0.0 else None
        steady_shots_per_s = float(BATCH * runs / best) if best > 0.0 else None

    store_states = None
    peak_branches = None
    faults_by_type = None
    branch_counters = None
    if is_branch and sim is not None:
        cnt = _normalize_counters(sim._branch_counters())
        pk_arr = sim._branch_peak_per_shot()
        hist_peaks = [int(k) for k in cnt["branches_hist"]]
        arr_peak = int(pk_arr.max()) if pk_arr.size else 0
        peak_branches = max([0, arr_peak, *hist_peaks])
        store_states = int(cnt["store_nodes"])
        fr = cnt["faults_realized"]
        faults_by_type = {
            "P": int(fr["pauli_faults"]),
            "L": None,
            "S2": int(fr["skipped_gates"]),
            "S3": int(fr["skipped_cczs"]),
        }
        branch_counters = cnt

    machine = {
        "env_file": machine_base["env_file"],
        "cores": list(machine_base["cores"]),
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "cpu_model": machine_base["cpu_model"],
        "cpu_count": machine_base["cpu_count"],
        "platform": machine_base["platform"],
        "python": machine_base["python"],
        "numpy": machine_base["numpy"],
        "stim": machine_base["stim"],
    }
    rec = {
        "schema": "stimside-paper-results/1",
        "figure": "bench_magic_suite",
        "engine": {
            "name": engine_name,
            "version": f"stimside@{git_sha}",
            "package_version": stimside.__version__,
            "options": dict(engine_options),
        },
        "workload": {
            "id": workload_id,
            "file": workload_file,
            "sha256": hashlib.sha256(str(circuit).encode("utf-8")).hexdigest(),
            "qubits": int(circuit.num_qubits),
            "params": dict(workload_params),
        },
        "machine": machine,
        "run": {
            "seed": 1,
            "batch_size": BATCH,
            "runs_per_rep": int(runs),
            "reps": int(reps),
            "shots": int(total_shots),
            "seconds_total": float(seconds_total),
            "seconds_construct": None if seconds_construct is None else float(seconds_construct),
            "warmup_seconds": None if warmup_seconds is None else float(warmup_seconds),
            "best_rep_seconds": None if best is None else float(best),
            "rep_seconds": [float(x) for x in rep_seconds],
            "rss_mb": _rss_mb(),
        },
        "metrics": {
            "shots_per_s": shots_per_s,
            "steady_shots_per_s": steady_shots_per_s,
            "accepted": accepted,
            "first_det": first_det,
            "metrics_shots": int(metrics_shots),
            "store_states": store_states,
            "peak_branches": peak_branches,
            "faults_by_type": faults_by_type,
            "branch_counters": branch_counters,
        },
        "status": status,
    }
    if status != "ok":
        rec["error"] = error
    return rec


def leaky_factory(circuit, sim_cls, handler_cls, **kwargs):
    def _make():
        handler = handler_cls().compile_op_handler(circuit=circuit, batch_size=BATCH)
        return sim_cls(circuit, compiled_op_handler=handler, batch_size=BATCH, seed=1, **kwargs)

    return _make


def magic_factory(circuit, **kwargs):
    return lambda: MagicsideSimulator(circuit, batch_size=BATCH, seed=1, **kwargs)


def leaky_sc(reps, records=None, machine_base=None, git_sha=None):
    if machine_base is None:
        machine_base = _machine_base()
    if git_sha is None:
        git_sha = _git_commit_sha()
    print("== leaky rotated SC (Clifford + leakage), B=64: Coset vs Magic vs branch ==")
    for d in (3, 5):
        c = build_leakage_surface_code(d, d)
        wparams = {
            "suite": "sc",
            "distance": d,
            "rounds": d,
            "p_pauli": 0.002,
            "p_leak": 0.004,
            "measurements": int(c.num_measurements),
            "detectors": int(c.num_detectors),
        }
        r_coset = _measure_entry(
            leaky_factory(c, CosetsideSimulator, LeakageUint8Coset),
            10,
            reps,
            circuit=c,
            engine_name="cosetside",
            engine_options={"batch_size": BATCH, "suite_label": "Coset", "leakage_handler": "LeakageUint8Coset"},
            workload_id=f"sc_d{d}",
            workload_file="benchmarks/_circuits.py:build_leakage_surface_code",
            workload_params=wparams,
            machine_base=machine_base,
            git_sha=git_sha,
            is_branch=False,
        )
        r_magic = _measure_entry(
            leaky_factory(c, MagicsideSimulator, LeakageUint8Magic),
            2,
            reps,
            circuit=c,
            engine_name="magicside_terms",
            engine_options={"batch_size": BATCH, "suite_label": "Magic", "leakage_handler": "LeakageUint8Magic"},
            workload_id=f"sc_d{d}",
            workload_file="benchmarks/_circuits.py:build_leakage_surface_code",
            workload_params=wparams,
            machine_base=machine_base,
            git_sha=git_sha,
            is_branch=False,
        )
        r_branch = _measure_entry(
            leaky_factory(
                c,
                MagicsideSimulator,
                LeakageUint8Magic,
                _engine="branches",
                max_components=BRANCH_MAX_COMPONENTS,
            ),
            2,
            reps,
            circuit=c,
            engine_name="magicside_branches",
            engine_options={
                "batch_size": BATCH,
                "suite_label": "branch",
                "leakage_handler": "LeakageUint8Magic",
                "_engine": "branches",
                "max_components": BRANCH_MAX_COMPONENTS,
            },
            workload_id=f"sc_d{d}",
            workload_file="benchmarks/_circuits.py:build_leakage_surface_code",
            workload_params=wparams,
            machine_base=machine_base,
            git_sha=git_sha,
            is_branch=True,
        )
        if records is not None:
            records.extend((r_coset, r_magic, r_branch))
        coset = r_coset["metrics"]["steady_shots_per_s"]
        magic = r_magic["metrics"]["steady_shots_per_s"]
        branch_str = _format_entry_rate(r_branch, with_unit=True)
        print(
            f"d={d}: Coset {coset:8.1f}  Magic {magic:7.1f} shots/s  ratio {coset / magic:5.1f}x"
            f"   branch {branch_str}"
        )


def xmem_k_t(reps, records=None, machine_base=None, git_sha=None):
    if machine_base is None:
        machine_base = _machine_base()
    if git_sha is None:
        git_sha = _git_commit_sha()
    print("== X-memory SC d=3, 3 rounds, + k T gates on data qubits (no leakage), B=64 ==")
    for k in (0, 2, 4, 6):
        c = xmem_t(k)
        wparams = {
            "suite": "xmem",
            "distance": 3,
            "rounds": 3,
            "k_t": k,
            "t_count": int(t_count(c)),
            "p": 0.002,
            "measurements": int(c.num_measurements),
            "detectors": int(c.num_detectors),
        }
        row = []
        for label, kwargs in ENGINES:
            is_branch = label == "branch"
            rec = _measure_entry(
                magic_factory(c, **kwargs),
                2,
                reps,
                circuit=c,
                engine_name="magicside_branches" if is_branch else "magicside_terms",
                engine_options={"batch_size": BATCH, "suite_label": label, **kwargs},
                workload_id=f"xmem_k{k}",
                workload_file="benchmarks/_circuits.py:xmem_t",
                workload_params=wparams,
                machine_base=machine_base,
                git_sha=git_sha,
                is_branch=is_branch,
            )
            if records is not None:
                records.append(rec)
            row.append(f"{label} {_format_entry_rate(rec, with_unit=False)}")
        print(f"k={k}: " + "   ".join(row) + " shots/s")


def random_t_cx_n12(reps, records=None, machine_base=None, git_sha=None):
    if machine_base is None:
        machine_base = _machine_base()
    if git_sha is None:
        git_sha = _git_commit_sha()
    print(f"== random T+CX n=12 (live vs branch, max_components={BRANCH_MAX_COMPONENTS}), B=64 ==")
    c = random_t_cx(12)
    nt = int(t_count(c))
    wparams = {
        "suite": "rand",
        "n_qubits": 12,
        "layers": 12,
        "p_t": 0.25,
        "circuit_seed": 7,
        "p_noise": 0.002,
        "t_count": nt,
        "measurements": int(c.num_measurements),
        "detectors": int(c.num_detectors),
    }
    r_live = _measure_entry(
        magic_factory(c),
        3,
        reps,
        circuit=c,
        engine_name="magicside_terms",
        engine_options={"batch_size": BATCH, "suite_label": "live"},
        workload_id="rand_n12",
        workload_file="benchmarks/_circuits.py:random_t_cx",
        workload_params=wparams,
        machine_base=machine_base,
        git_sha=git_sha,
        is_branch=False,
    )
    branch_kwargs = {"_engine": "branches", "max_components": BRANCH_MAX_COMPONENTS}
    r_branch = _measure_entry(
        magic_factory(c, **branch_kwargs),
        3,
        reps,
        circuit=c,
        engine_name="magicside_branches",
        engine_options={"batch_size": BATCH, "suite_label": "branch", **branch_kwargs},
        workload_id="rand_n12",
        workload_file="benchmarks/_circuits.py:random_t_cx",
        workload_params=wparams,
        machine_base=machine_base,
        git_sha=git_sha,
        is_branch=True,
    )
    if records is not None:
        records.extend((r_live, r_branch))
    live_str = _format_entry_rate(r_live, with_unit=True)
    branch_str = _format_entry_rate(r_branch, with_unit=True)
    print(f"n=12 ({nt} T): live {live_str}   branch {branch_str}")


SUITES = {"sc": leaky_sc, "xmem": xmem_k_t, "rand": random_t_cx_n12}


def _write_jsonl_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".tmp_bench_magic_",
            suffix=".jsonl",
            delete=False,
        ) as tf:
            tmp_path = Path(tf.name)
            for rec in records:
                tf.write(json.dumps(rec, allow_nan=False) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suites", nargs="*", choices=list(SUITES), help="default: all")
    ap.add_argument("--reps", type=int, default=3, help="best-of N timing repetitions (default 3)")
    ap.add_argument(
        "--jsonl",
        "--json-lines",
        dest="jsonl",
        type=Path,
        default=None,
        help="Optional path to write stimside-paper-results/1 JSON-lines output.",
    )
    args = ap.parse_args(argv)
    if args.reps < 1:
        ap.error("--reps must be >= 1")
    if args.jsonl is not None and args.jsonl.exists() and args.jsonl.is_dir():
        ap.error(f"--jsonl path is a directory: {args.jsonl}")

    warnings.simplefilter("ignore")
    machine_base = _machine_base()
    git_sha = _git_commit_sha()
    records = [] if args.jsonl is not None else None
    load = " ".join(f"{x:.1f}" for x in os.getloadavg())
    print(f"stim {stim.__version__}  loadavg(1/5/15 min) {load}")
    for name in args.suites or list(SUITES):
        SUITES[name](args.reps, records=records, machine_base=machine_base, git_sha=git_sha)
    print("loadavg after: " + " ".join(f"{x:.1f}" for x in os.getloadavg()))
    if args.jsonl is not None and records is not None:
        _write_jsonl_atomic(args.jsonl, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
