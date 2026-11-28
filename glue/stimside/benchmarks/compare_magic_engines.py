# Copyright 2026 Google LLC
"""Compare MagicsideSimulator `_engine="terms"` vs `_engine="branches"` on any `.stim` circuit.

Run as a standalone script (never collected by pytest):

    PYTHONPATH=src python benchmarks/compare_magic_engines.py --circuit /path/to/circuit.stim \\
        --shots 102400 --batch-size 4096 --seed 1 --check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Literal
import warnings

import numpy as np
import stim  # type: ignore[import-untyped]

from stimside import LeakageUint8Magic, MagicsideSimulator


def _circuit_has_leakage(circuit: stim.Circuit) -> bool:
    """Return True if any instruction (recursively across REPEAT blocks) carries a leakage tag."""
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            if _circuit_has_leakage(item.body_copy()):
                return True
        elif item.tag.startswith(("LEAKAGE", "CONDITIONED_ON")):
            return True
    return False


def _gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = Gamma(a, x) / Gamma(a) (pure stdlib)."""
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError(f"a must be a finite number > 0, got {a}")
    if math.isnan(x):
        return math.nan
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    eps = 1e-15
    fpmin = 1e-300
    gln = math.lgamma(a)
    if x < a + 1.0:
        # Series expansion for P(a, x), then Q = 1 - P.
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * eps:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return min(1.0, max(0.0, 1.0 - p))
    # Modified Lentz's algorithm for continued fraction of Q(a, x).
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return min(1.0, max(0.0, q))


def _chi2_sf(chi2: float, dof: int) -> float:
    """Survival function P(Chi2_dof >= chi2) without SciPy."""
    if dof <= 0:
        return 1.0
    if math.isnan(chi2):
        return math.nan
    if chi2 <= 0.0:
        return 1.0
    return _gamma_q(0.5 * dof, 0.5 * chi2)


def compare_histograms(
    first_det_a: Sequence[int] | np.ndarray,
    accepted_a: int,
    first_det_b: Sequence[int] | np.ndarray,
    accepted_b: int,
    *,
    max_z: float = 3.0,
    min_p: float = 0.01,
) -> dict[str, Any]:
    """Compare two samples on acceptance rate (two-proportion z) and first-detection histogram (chi2)."""
    if not (math.isfinite(max_z) and max_z > 0.0):
        raise ValueError(f"max_z must be a finite number > 0, got {max_z}")
    if not (math.isfinite(min_p) and 0.0 <= min_p < 1.0):
        raise ValueError(f"min_p must be a finite number in [0, 1), got {min_p}")
    ha = np.asarray(first_det_a, dtype=np.int64)
    hb = np.asarray(first_det_b, dtype=np.int64)
    if ha.ndim != 1 or hb.ndim != 1 or ha.shape != hb.shape:
        raise ValueError(f"Histogram shapes must match 1-D arrays, got {ha.shape} vs {hb.shape}")
    if np.any(ha < 0) or np.any(hb < 0):
        raise ValueError("Histogram counts must be non-negative")
    na = int(ha.sum())
    nb = int(hb.sum())
    if na <= 0 or nb <= 0:
        raise ValueError(f"Both samples must have positive shot counts, got na={na}, nb={nb}")
    if not (0 <= accepted_a <= na) or not (0 <= accepted_b <= nb):
        raise ValueError(
            f"Accepted counts must satisfy 0 <= accepted <= total shots, "
            f"got accepted_a={accepted_a} (na={na}), accepted_b={accepted_b} (nb={nb})"
        )

    pa = accepted_a / na
    pb = accepted_b / nb
    pp = (accepted_a + accepted_b) / (na + nb)
    se = math.sqrt(max(0.0, pp * (1.0 - pp) * (1.0 / na + 1.0 / nb)))
    z = 0.0 if se == 0.0 else (pa - pb) / se

    nz = (ha + hb) > 0
    table = np.vstack([ha[nz], hb[nz]]).astype(np.float64)
    pooled_bins = 0
    if table.shape[1] <= 1:
        chi2, dof, p_value = 0.0, 0, 1.0
        bins_used = int(table.shape[1])
    else:
        row_totals = np.array([[float(na)], [float(nb)]], dtype=np.float64)
        exp = table.sum(axis=0, keepdims=True) * row_totals / float(na + nb)
        keep = (exp >= 5.0).all(axis=0)
        pooled_bins = int((~keep).sum())
        if pooled_bins > 0:
            table = np.hstack([table[:, keep], table[:, ~keep].sum(axis=1, keepdims=True)])
        bins_used = int(table.shape[1])
        if bins_used <= 1:
            chi2, dof, p_value = 0.0, 0, 1.0
        else:
            exp = table.sum(axis=0, keepdims=True) * row_totals / float(na + nb)
            chi2 = float(np.sum((table - exp) ** 2 / exp))
            dof = bins_used - 1
            p_value = _chi2_sf(chi2, dof)

    passed = bool(abs(z) < max_z and p_value > min_p)
    return {
        "z_acceptance": float(z),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "bins_used": int(bins_used),
        "pooled_bins": int(pooled_bins),
        "max_z": float(max_z),
        "min_p": float(min_p),
        "passed": passed,
    }


def _normalize_counters(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly copy of `_branch_counters()` with sorted `branches_hist` keys."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "branches_hist":
            out[k] = {str(bk): int(bv) for bk, bv in sorted(v.items(), key=lambda kv: int(kv[0]))}
        elif isinstance(v, dict):
            out[k] = {str(sk): int(sv) for sk, sv in v.items()}
        else:
            out[k] = int(v)
    return out


def _run_engine(
    circuit: stim.Circuit,
    *,
    engine: Literal["terms", "branches"],
    shots: int,
    batch_size: int,
    seed: int,
    use_leakage: bool,
) -> dict[str, Any]:
    eff_batch = min(batch_size, shots)
    n_det = circuit.num_detectors
    hist = np.zeros(n_det + 1, dtype=np.int64)
    accepted = 0

    handler = (
        LeakageUint8Magic().compile_op_handler(circuit=circuit, batch_size=eff_batch)
        if use_leakage
        else None
    )
    sim = MagicsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=eff_batch,
        seed=seed,
        _engine=engine,
    )

    n_batches = math.ceil(shots / eff_batch)
    total_simulated = n_batches * eff_batch
    remaining = shots

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = time.perf_counter()
        for b in range(n_batches):
            if b > 0:
                sim.clear()
            sim.run()
            dets = sim.get_detector_flips()
            take = min(eff_batch, remaining)
            if take < eff_batch:
                dets = dets[:take]
            remaining -= take
            if n_det == 0:
                hist[0] += take
                accepted += take
            else:
                any_det = dets.any(axis=1)
                first = np.where(any_det, dets.argmax(axis=1), n_det)
                hist += np.bincount(first, minlength=n_det + 1)
                accepted += int((~any_det).sum())
        elapsed = time.perf_counter() - t0

    res: dict[str, Any] = {
        "seed": int(seed),
        "simulated_shots": int(total_simulated),
        "accepted": int(accepted),
        "acceptance_rate": float(accepted / shots),
        "first_det": [int(x) for x in hist.tolist()],
        "wall_s": float(elapsed),
        "shots_per_s": float(total_simulated / elapsed) if elapsed > 0.0 else float("inf"),
    }
    if engine == "branches":
        res["counters"] = _normalize_counters(sim._branch_counters())
    return res


def _format_report(report: dict[str, Any], load_before: tuple[float, float, float], load_after: tuple[float, float, float], leakage_mode: str) -> str:
    terms = report["terms"]
    branches = report["branches"]
    stats = report["stats"]
    cnt = branches["counters"]
    verdict = "PASS" if stats["passed"] else "FAIL"
    lines = [
        f"circuit: {report['circuit']}",
        (
            f"qubits={report['num_qubits']}  measurements={report['num_measurements']}  "
            f"detectors={report['num_detectors']}  leakage={'on' if report['leakage'] else 'off'} (mode={leakage_mode})"
        ),
        (
            f"shots={report['shots']}  batch_size={report['batch_size']}  "
            f"seeds=(terms:{terms['seed']}, branches:{branches['seed']})  stim={stim.__version__}"
        ),
        (
            f"loadavg before: {load_before[0]:.2f} {load_before[1]:.2f} {load_before[2]:.2f}  "
            f"after: {load_after[0]:.2f} {load_after[1]:.2f} {load_after[2]:.2f}"
        ),
        (
            f"terms   : accepted={terms['accepted']}/{report['shots']} ({terms['acceptance_rate']:.6f})  "
            f"wall={terms['wall_s']:.3f}s  {terms['shots_per_s']:9.1f} shots/s"
        ),
        (
            f"branches: accepted={branches['accepted']}/{report['shots']} ({branches['acceptance_rate']:.6f})  "
            f"wall={branches['wall_s']:.3f}s  {branches['shots_per_s']:9.1f} shots/s  "
            f"speedup={report['speedup']:.2f}x"
        ),
        (
            f"stats   : z={stats['z_acceptance']:+.4f} (|z|<{stats['max_z']})  "
            f"chi2={stats['chi2']:.4f}  dof={stats['dof']}  p={stats['p_value']:.6f} (p>{stats['min_p']})  "
            f"bins={stats['bins_used']} (pooled={stats['pooled_bins']})  verdict={verdict}"
        ),
        (
            f"counters: store_nodes={cnt['store_nodes']}  store_bytes={cnt['store_bytes_estimate']}  "
            f"store_clears={cnt['store_clears']}  resolve_calls={cnt['resolve_calls']}  "
            f"splits={cnt['splits']}  subcache={cnt['subcache_hits']}/{cnt['subcache_misses']}"
        ),
        (
            f"meas    : definite={cnt['meas_definite']}  absorbed={cnt['meas_absorbed']}  "
            f"forked={cnt['meas_forked']}"
        ),
        f"t_new   : {cnt['transitions_new']}",
        f"t_hit   : {cnt['transitions_hit']}",
        f"faults  : {cnt['faults_realized']}",
        f"branches_hist: {cnt['branches_hist']}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compare MagicsideSimulator terms vs branches engines on a .stim circuit."
    )
    ap.add_argument("--circuit", required=True, type=Path, help="Path to .stim circuit file.")
    ap.add_argument("--shots", type=int, default=102400, help="Total shots per engine (default: 102400).")
    ap.add_argument("--batch-size", type=int, default=4096, help="Batch size per run() (default: 4096).")
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed; terms uses seed, branches uses seed+1 (default: 1).",
    )
    ap.add_argument(
        "--leakage",
        choices=("auto", "on", "off"),
        default="auto",
        help="Leakage handler mode (default: auto).",
    )
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON report.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if statistical thresholds (|z| < max_z and p > min_p) are not met.",
    )
    ap.add_argument("--max-z", type=float, default=3.0, help="Maximum allowed |z| for acceptance rate (default: 3.0).")
    ap.add_argument("--min-p", type=float, default=0.01, help="Minimum allowed chi-square p-value (default: 0.01).")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error("--shots must be >= 1")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    if not math.isfinite(args.max_z) or args.max_z <= 0.0:
        ap.error("--max-z must be a finite number > 0")
    if not math.isfinite(args.min_p) or not (0.0 <= args.min_p < 1.0):
        ap.error("--min-p must be a finite number in [0, 1)")
    try:
        is_file = args.circuit.is_file()
    except OSError as exc:
        ap.error(f"failed to load --circuit {args.circuit}: {exc}")
    if not is_file:
        ap.error(f"--circuit file not found: {args.circuit}")

    try:
        circuit = stim.Circuit.from_file(str(args.circuit))
    except (OSError, ValueError) as exc:
        ap.error(f"failed to load --circuit {args.circuit}: {exc}")
    if args.leakage == "auto":
        use_leakage = _circuit_has_leakage(circuit)
    else:
        use_leakage = args.leakage == "on"

    eff_batch = min(args.batch_size, args.shots)
    load_before = os.getloadavg()

    terms_res = _run_engine(
        circuit,
        engine="terms",
        shots=args.shots,
        batch_size=eff_batch,
        seed=args.seed,
        use_leakage=use_leakage,
    )
    branches_res = _run_engine(
        circuit,
        engine="branches",
        shots=args.shots,
        batch_size=eff_batch,
        seed=args.seed + 1,
        use_leakage=use_leakage,
    )

    load_after = os.getloadavg()
    stats = compare_histograms(
        terms_res["first_det"],
        terms_res["accepted"],
        branches_res["first_det"],
        branches_res["accepted"],
        max_z=args.max_z,
        min_p=args.min_p,
    )
    speedup = (
        branches_res["shots_per_s"] / terms_res["shots_per_s"]
        if terms_res["shots_per_s"] > 0.0
        else float("inf")
    )
    report: dict[str, Any] = {
        "circuit": str(args.circuit),
        "num_qubits": int(circuit.num_qubits),
        "num_measurements": int(circuit.num_measurements),
        "num_detectors": int(circuit.num_detectors),
        "leakage": bool(use_leakage),
        "shots": int(args.shots),
        "batch_size": int(eff_batch),
        "terms": terms_res,
        "branches": branches_res,
        "speedup": float(speedup),
        "stats": stats,
    }

    print(_format_report(report, load_before, load_after, args.leakage))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.check and not stats["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
