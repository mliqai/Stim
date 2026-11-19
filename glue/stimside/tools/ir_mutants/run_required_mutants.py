#!/usr/bin/env python3
"""Scripted required-mutant runner for stimside IR and HEAD contracts.

Executes the required mutants listed in ``required_mutants.tsv`` against isolated
scratch copies of the ``stimside`` package.

Mandatory Baseline-First Hygiene Rule:
1. Pre-check ``PATTERN-COUNT == 1`` for every selected row against the unmutated
   baseline snapshot; abort with nonzero exit (``rc=2``) if any row fails.
2. Run ALL detectors on the unmutated baseline snapshot FIRST:
   ``INV`` (``util/circuit_ir_test.py``), ``ORACLE`` (``ir_oracle_test.py``),
   and every named owner-test group (``GOLD``, ``TAB``, ``MAGIC``, ``DEM``,
   ``UTIL``, ``FLIP``, ``COSET``), plus optional ``E2E_ORACLE`` if
   ``--include-e2e-circuits`` is set. Abort with nonzero exit (``rc=3``) if
   any baseline detector fails or crashes.
3. Never modify the live package tree. Every mutant runs in an isolated scratch
   copy with ``PYTHONDONTWRITEBYTECODE=1``, ``-p no:cacheprovider``, and
   ``--basetemp`` inside ``--work``.
4. Transparently distinguish ``KILLED(test)`` (pytest assertion failure,
   ``rc=1``), ``KILLED(crash)`` (import/syntax/internal crash, non-1 exit, or
   uncaught non-assertion exception in a failed test), and ``ERROR`` (pytest
   fixture/setup error: ``rc=1`` with ``n_error > 0`` and ``n_failed == 0``) in
   console, TSV, Markdown, and JSON. Exit nonzero on unexpected survivors,
   timeouts, setup ``ERROR``s, or killed ``expected=equivalent`` controls.
5. Clean up all temporary mutant worktrees in ``try/finally`` unless
   ``--keep-work`` is requested.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PKG = HERE.parents[1]
DEFAULT_TSV = HERE / "required_mutants.tsv"

GROUPS: dict[str, tuple[str, ...]] = {
    "GOLD": ("golden_test.py",),
    "TAB": ("simulator_tableau_test.py", "simulator_tableau_routing_test.py"),
    "MAGIC": ("_magic/compile_test.py",),
    "DEM": (
        "dem_generators/dem_generator_marginal_test.py",
        "dem_generators/marginal_builder_reference_test.py",
        "dem_generators/dem_generator_oracle_test.py",
    ),
    "UTIL": ("util",),
    "FLIP": (
        "simulator_flip_test.py",
        "op_handlers/leakage_handlers/leakage_uint8_flip_test.py",
    ),
    "COSET": ("simulator_coset_test.py",),
}

_ASSERT_EXCS = frozenset({"AssertionError", "Failed"})
_TB_FILE_EXC_RE = re.compile(r"^\S+\.py:\d+:\s+([A-Za-z_][A-Za-z0-9_.]*)(?::|\s|$)")
_TB_E_EXC_RE = re.compile(r"^E   ([A-Za-z_][A-Za-z0-9_.]*)(?::|\s*$)")
_ERROR_RE = re.compile(r"^ERROR\s+(\S+)")
_CORES_RE = re.compile(r"^[0-9,-]+$")

_E2E_DUMP_CODE = """
import json, sys
from pathlib import Path
import stim
from stimside import ir_oracle_test as M

circ_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
circuits = [
    ("e2e_" + p.stem, stim.Circuit.from_file(str(p)))
    for p in sorted(circ_dir.glob("*.stim"))
]
raw = M.compute_oracle_digests(circuits=circuits)
flat = {
    f"{cn}|{wn}": [sha, nbytes]
    for cn, ws in sorted(raw.items())
    for wn, (sha, nbytes, _msg) in sorted(ws.items())
}
out_path.write_text(json.dumps(flat, sort_keys=True), encoding="utf-8")
print(f"e2e_entries={len(flat)} circuits={len(circuits)}")
"""


@dataclasses.dataclass(frozen=True)
class MutantSpec:
    """Single row parsed from ``required_mutants.tsv``."""

    mid: str
    suite: str
    layer: str
    expected: str
    rel_file: str
    old: str
    new: str
    groups: tuple[str, ...]
    note: str


@dataclasses.dataclass
class DetectorResult:
    """Outcome of one detector run (pytest group or E2E oracle)."""

    name: str
    status: str  # PASS | KILLED(test) | KILLED(crash) | ERROR | TIMEOUT
    rc: int
    elapsed_s: float
    n_failed: int
    n_error: int
    failed_tests: list[str]
    exc_kinds: list[str]
    detail: str


@dataclasses.dataclass
class MutantResult:
    """Aggregated outcome for one mutant across all executed detectors."""

    spec: MutantSpec
    verdict: str
    elapsed_s: float
    detectors: dict[str, DetectorResult]


def load_mutants(tsv_path: Path, only: set[str]) -> list[MutantSpec]:
    """Load and validate mutant definitions from ``tsv_path``."""
    rows: list[MutantSpec] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(tsv_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(
                f"{tsv_path}:{lineno}: expected 9 tab-separated columns, got {len(fields)}"
            )
        mid, suite, layer, expected, rel_file, old_s, new_s, groups_s, note = fields
        if mid in seen_ids:
            raise ValueError(f"{tsv_path}:{lineno}: duplicate mutant id {mid!r}")
        seen_ids.add(mid)
        if layer not in ("head", "ir"):
            raise ValueError(f"{tsv_path}:{lineno}: invalid layer {layer!r}")
        if expected not in ("kill", "equivalent"):
            raise ValueError(f"{tsv_path}:{lineno}: invalid expected {expected!r}")
        groups = tuple(g for g in groups_s.split() if g)
        bad_groups = [g for g in groups if g not in GROUPS]
        if bad_groups:
            raise ValueError(f"{tsv_path}:{lineno}: unknown groups {bad_groups!r}")
        if only and mid not in only:
            continue
        rows.append(
            MutantSpec(
                mid=mid,
                suite=suite,
                layer=layer,
                expected=expected,
                rel_file=rel_file,
                old=old_s.replace("\\n", "\n"),
                new=new_s.replace("\\n", "\n"),
                groups=groups,
                note=note,
            )
        )
    if only:
        missing = sorted(only - {r.mid for r in rows})
        if missing:
            raise ValueError(f"unknown mutant id(s) in --only: {missing}")
    return rows


def check_pattern_counts(
    base_tree: Path, mutants: Sequence[MutantSpec]
) -> list[tuple[str, str]]:
    """Verify every mutant matches ``PATTERN-COUNT == 1`` in ``base_tree``."""
    errors: list[tuple[str, str]] = []
    for m in mutants:
        target = base_tree / "src" / "stimside" / m.rel_file
        if not target.is_file():
            errors.append((m.mid, f"missing file src/stimside/{m.rel_file}"))
            continue
        src = target.read_text(encoding="utf-8")
        cnt = src.count(m.old)
        if cnt != 1:
            errors.append((m.mid, f"PATTERN-COUNT={cnt} in {m.rel_file}"))
        elif m.old == m.new:
            errors.append((m.mid, "NO-OP mutation (old == new)"))
    return errors


def _build_cmd(base_cmd: list[str], cores: str) -> list[str]:
    if not cores:
        return base_cmd
    return ["taskset", "-c", cores, *base_cmd]


def _make_env(tree_root: Path, home_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(tree_root / "src")
    if home_dir is not None:
        env["HOME"] = str(home_dir)
    return env


def _record_exc(exc_raw: str, exc_kinds: list[str]) -> str:
    exc = exc_raw.strip().rstrip(":").rsplit(".", 1)[-1]
    if exc == "assert":
        exc = "AssertionError"
    if exc and exc[0].isupper() and exc != "File" and exc not in exc_kinds:
        exc_kinds.append(exc)
    return exc


def _classify_pytest_output(
    name: str, rc: int, output: str, elapsed_s: float
) -> DetectorResult:
    """Classify a pytest run into PASS / KILLED(test) / KILLED(crash) / ERROR."""
    if rc == 0:
        return DetectorResult(name, "PASS", 0, elapsed_s, 0, 0, [], [], "PASS")

    failed_tests: list[str] = []
    exc_kinds: list[str] = []
    n_error = 0
    for line in output.splitlines():
        if line.startswith("FAILED "):
            rest = line[len("FAILED ") :].strip()
            node, sep, tail = rest.partition(" - ")
            failed_tests.append(node.replace("src/stimside/", ""))
            if sep and tail:
                tok = tail.split(":", 1)[0].split(None, 1)[0]
                if tok:
                    _record_exc(tok, exc_kinds)
        elif _ERROR_RE.match(line):
            n_error += 1
            _, sep, tail = line.partition(" - ")
            if sep and tail:
                tok = tail.split(":", 1)[0].split(None, 1)[0]
                if tok:
                    _record_exc(tok, exc_kinds)
        else:
            m_tb = _TB_FILE_EXC_RE.match(line)
            if m_tb:
                _record_exc(m_tb.group(1), exc_kinds)
            else:
                m_e = _TB_E_EXC_RE.match(line)
                if m_e:
                    _record_exc(m_e.group(1), exc_kinds)

    has_assert_fail = any(e in _ASSERT_EXCS for e in exc_kinds)
    if rc != 1:
        if not exc_kinds:
            for tok in (
                "SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError"
            ):
                if tok in output:
                    exc_kinds.append(tok)
        excs = ",".join(exc_kinds) if exc_kinds else f"rc={rc}"
        detail = f"rc={rc} errors={n_error} exc={excs}"
        return DetectorResult(
            name, "KILLED(crash)", rc, elapsed_s, len(failed_tests), n_error,
            failed_tests[:5], exc_kinds, detail,
        )

    if n_error > 0 and not failed_tests:
        status = "ERROR"
    elif failed_tests and has_assert_fail:
        status = "KILLED(test)"
    else:
        status = "KILLED(crash)"
    excs = ",".join(exc_kinds) if exc_kinds else f"rc={rc}"
    detail = f"rc={rc} failed={len(failed_tests)} errors={n_error} exc={excs}"
    return DetectorResult(
        name, status, rc, elapsed_s, len(failed_tests), n_error,
        failed_tests[:5], exc_kinds, detail,
    )


def _run_pytest_detector(
    name: str,
    rel_targets: Sequence[str],
    tree_root: Path,
    basetemp: Path,
    python_exe: str,
    cores: str,
    timeout: int,
    extra_args: Sequence[str] = (),
    home_dir: Path | None = None,
) -> DetectorResult:
    files = [str(tree_root / "src" / "stimside" / rel) for rel in rel_targets]
    cmd = _build_cmd(
        [
            python_exe, "-m", "pytest", "-p", "no:cacheprovider",
            "-q", "-rfE", "--tb=line", "-c", "pyproject.toml", "--rootdir", ".",
            "--maxfail=5", f"--basetemp={basetemp}", *extra_args, *files,
        ],
        cores,
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(tree_root),
            env=_make_env(tree_root, home_dir=home_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return DetectorResult(
            name, "TIMEOUT", 124, time.perf_counter() - t0, 0, 0, [], ["Timeout"], "TIMEOUT"
        )
    finally:
        shutil.rmtree(basetemp, ignore_errors=True)
    dt = time.perf_counter() - t0
    return _classify_pytest_output(name, proc.returncode, proc.stdout + "\n" + proc.stderr, dt)


def _run_e2e_oracle(
    tree_root: Path,
    e2e_dir: Path,
    out_json: Path,
    python_exe: str,
    cores: str,
    timeout: int,
    baseline: dict[str, list[Any]] | None,
) -> tuple[DetectorResult, dict[str, list[Any]] | None]:
    cmd = _build_cmd([python_exe, "-B", "-c", _E2E_DUMP_CODE, str(e2e_dir), str(out_json)], cores)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(tree_root),
            env=_make_env(tree_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            DetectorResult(
                "E2E_ORACLE", "TIMEOUT", 124, time.perf_counter() - t0, 0, 0,
                [], ["Timeout"], "TIMEOUT",
            ),
            None,
        )
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_json.is_file():
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        tail = msg[-1][:120] if msg else f"rc={proc.returncode}"
        return (
            DetectorResult(
                "E2E_ORACLE", "KILLED(crash)", proc.returncode, dt, 0, 1,
                [], ["Crash"], f"rc={proc.returncode}: {tail}",
            ),
            None,
        )
    data: dict[str, list[Any]] = json.loads(out_json.read_text(encoding="utf-8"))
    out_json.unlink(missing_ok=True)
    if baseline is None:
        return (
            DetectorResult(
                "E2E_ORACLE", "PASS", 0, dt, 0, 0, [], [], f"entries={len(data)}"
            ),
            data,
        )
    diffs = [k for k in baseline if data.get(k) != baseline[k]]
    extra = [k for k in data if k not in baseline]
    total = len(diffs) + len(extra)
    if total == 0:
        return DetectorResult("E2E_ORACLE", "PASS", 0, dt, 0, 0, [], [], "0 diffs"), data
    return (
        DetectorResult(
            "E2E_ORACLE", "KILLED(test)", 1, dt, total, 0,
            diffs[:5], ["DigestDiff"], f"diffs={total}",
        ),
        data,
    )


def _snapshot_pkg(pkg_root: Path, dst_root: Path) -> None:
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "build", "dist", "*.egg-info",
        ".benchmarks", "*.pyc",
    )
    shutil.copytree(pkg_root / "src", dst_root / "src", ignore=ignore)
    shutil.copy2(pkg_root / "pyproject.toml", dst_root / "pyproject.toml")


def _run_baseline(
    base_tree: Path,
    work_dir: Path,
    empty_home: Path,
    python_exe: str,
    cores: str,
    jobs: int,
    timeout: int,
    e2e_dir: Path | None,
) -> tuple[dict[str, DetectorResult], dict[str, list[Any]] | None]:
    tasks: list[tuple[str, tuple[str, ...], tuple[str, ...], Path | None]] = [
        ("INV", ("util/circuit_ir_test.py",), (), empty_home),
        ("ORACLE", ("ir_oracle_test.py",), (), empty_home),
        ("GOLD", GROUPS["GOLD"], (), None),
        ("TAB", GROUPS["TAB"], (), None),
        ("MAGIC", GROUPS["MAGIC"], (), None),
        ("DEM", GROUPS["DEM"], (), None),
        ("UTIL", GROUPS["UTIL"], ("--ignore=src/stimside/util/circuit_ir_test.py",), None),
        ("FLIP", GROUPS["FLIP"], (), None),
        ("COSET", GROUPS["COSET"], (), None),
    ]
    results: dict[str, DetectorResult] = {}

    def _run_one(
        item: tuple[str, tuple[str, ...], tuple[str, ...], Path | None]
    ) -> DetectorResult:
        name, rels, extra, home = item
        return _run_pytest_detector(
            name, rels, base_tree, work_dir / f"bt_base_{name}",
            python_exe, cores, timeout, extra_args=extra, home_dir=home,
        )

    with cf.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(tasks)))) as ex:
        for res in ex.map(_run_one, tasks):
            results[res.name] = res

    e2e_base: dict[str, list[Any]] | None = None
    if e2e_dir is not None:
        e2e_res, e2e_base = _run_e2e_oracle(
            base_tree, e2e_dir, work_dir / "e2e_base.json",
            python_exe, cores, timeout, None,
        )
        results[e2e_res.name] = e2e_res
    return results, e2e_base


def _aggregate_verdict(spec: MutantSpec, detectors: dict[str, DetectorResult]) -> str:
    statuses = [d.status for d in detectors.values()]
    if "ERROR" in statuses:
        return "ERROR"
    if "KILLED(test)" in statuses:
        return "KILLED(test)"
    if "KILLED(crash)" in statuses:
        return "KILLED(crash)"
    if "TIMEOUT" in statuses:
        return "TIMEOUT"
    if spec.expected == "equivalent":
        return "SURVIVED(justified_equiv)"
    return "SURVIVED"


def _run_one_mutant(
    spec: MutantSpec,
    base_tree: Path,
    work_dir: Path,
    empty_home: Path,
    python_exe: str,
    cores: str,
    timeout: int,
    run_owner_tests: bool,
    e2e_dir: Path | None,
    e2e_base: dict[str, list[Any]] | None,
    keep_work: bool,
) -> MutantResult:
    t0 = time.perf_counter()
    mroot = work_dir / f"m_{spec.mid}"
    shutil.rmtree(mroot, ignore_errors=True)
    detectors: dict[str, DetectorResult] = {}
    try:
        shutil.copytree(base_tree, mroot)
        target = mroot / "src" / "stimside" / spec.rel_file
        src = target.read_text(encoding="utf-8")
        if src.count(spec.old) != 1:
            raise RuntimeError(
                f"{spec.mid}: PATTERN-COUNT={src.count(spec.old)} in {spec.rel_file}"
            )
        target.write_text(src.replace(spec.old, spec.new, 1), encoding="utf-8")

        detectors["INV"] = _run_pytest_detector(
            "INV", ("util/circuit_ir_test.py",), mroot,
            work_dir / f"bt_{spec.mid}_INV", python_exe, cores, timeout,
            home_dir=empty_home,
        )
        detectors["ORACLE"] = _run_pytest_detector(
            "ORACLE", ("ir_oracle_test.py",), mroot,
            work_dir / f"bt_{spec.mid}_ORACLE", python_exe, cores, timeout,
            home_dir=empty_home,
        )
        if run_owner_tests:
            run_groups: list[str] = ["GOLD"] + list(spec.groups)
            for grp in run_groups:
                extra = (
                    ("--ignore=src/stimside/util/circuit_ir_test.py",)
                    if grp == "UTIL"
                    else ()
                )
                detectors[grp] = _run_pytest_detector(
                    grp, GROUPS[grp], mroot, work_dir / f"bt_{spec.mid}_{grp}",
                    python_exe, cores, timeout, extra_args=extra,
                )
        if e2e_dir is not None and e2e_base is not None:
            e2e_res, _ = _run_e2e_oracle(
                mroot, e2e_dir, work_dir / f"e2e_{spec.mid}.json",
                python_exe, cores, timeout, e2e_base,
            )
            detectors["E2E_ORACLE"] = e2e_res
    finally:
        if not keep_work:
            shutil.rmtree(mroot, ignore_errors=True)
    verdict = _aggregate_verdict(spec, detectors)
    elapsed = time.perf_counter() - t0
    print(f"[run] {spec.mid:38s} {verdict:25s} {elapsed:6.2f}s\n", end="", flush=True)
    return MutantResult(spec, verdict, elapsed, detectors)


def _format_outputs(
    results: Sequence[MutantResult],
    baseline: dict[str, DetectorResult],
) -> tuple[str, str, dict[str, Any]]:
    tsv_cols = [
        "id", "suite", "layer", "expected", "verdict", "inv", "oracle",
        "owner_tests", "e2e_oracle", "kill_test_detectors", "kill_crash_detectors",
        "exc_kinds", "elapsed_s", "note",
    ]
    tsv_lines = ["\t".join(tsv_cols)]
    md_lines = [
        (
            "| mutant | suite | layer | expected | verdict | INV | ORACLE | owner groups "
            "| killed(test) | killed(crash) | exc | wall(s) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---:|",
    ]
    json_rows: list[dict[str, Any]] = []
    for r in results:
        d = r.detectors
        owner_names = [k for k in d if k not in ("INV", "ORACLE", "E2E_ORACLE")]
        owner_str = ";".join(
            f"{k}:{d[k].status}" for k in owner_names if d[k].status != "PASS"
        ) or ("PASS" if owner_names else "-")
        k_test = [k for k, v in d.items() if v.status == "KILLED(test)"]
        k_crash = [k for k, v in d.items() if v.status == "KILLED(crash)"]
        excs = sorted({e for v in d.values() for e in v.exc_kinds})
        inv_s = d["INV"].status
        ora_s = d["ORACLE"].status
        e2e_s = d["E2E_ORACLE"].status if "E2E_ORACLE" in d else "-"
        tsv_lines.append(
            "\t".join([
                r.spec.mid, r.spec.suite, r.spec.layer, r.spec.expected, r.verdict,
                inv_s, ora_s, owner_str, e2e_s,
                ",".join(k_test), ",".join(k_crash), ",".join(excs),
                f"{r.elapsed_s:.2f}", r.spec.note,
            ])
        )
        md_lines.append(
            f"| `{r.spec.mid}` | {r.spec.suite} | {r.spec.layer} | {r.spec.expected} "
            f"| **{r.verdict}** | {inv_s} | {ora_s} | {owner_str} "
            f"| {','.join(k_test) or '-'} | {','.join(k_crash) or '-'} "
            f"| {','.join(excs) or '-'} | {r.elapsed_s:.2f} |"
        )
        json_rows.append(
            {
                "id": r.spec.mid,
                "suite": r.spec.suite,
                "layer": r.spec.layer,
                "expected": r.spec.expected,
                "verdict": r.verdict,
                "elapsed_s": round(r.elapsed_s, 3),
                "note": r.spec.note,
                "detectors": {
                    k: dataclasses.asdict(v) for k, v in r.detectors.items()
                },
            }
        )
    payload = {
        "baseline": {k: dataclasses.asdict(v) for k, v in baseline.items()},
        "mutants": json_rows,
    }
    return "\n".join(tsv_lines) + "\n", "\n".join(md_lines) + "\n", payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WP-27 Step 0 required-mutant runner (E6).")
    ap.add_argument("--pkg", type=Path, default=DEFAULT_PKG, help="stimside package root")
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="mutants TSV file")
    ap.add_argument(
        "--work",
        type=Path,
        default=Path(f"/tmp/stimside_scratch/ir_mutants_{os.getpid()}"),
        help="scratch directory for isolated snapshots",
    )
    ap.add_argument("--jobs", type=int, default=4, help="parallel mutant workers")
    ap.add_argument("--cores", default="", help="CPU set for taskset -c (e.g. 0-7)")
    ap.add_argument("--out", type=Path, default=None, help="output prefix (.tsv/.md/.json)")
    ap.add_argument("--only", default="", help="comma-separated subset of mutant IDs")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter for pytest")
    ap.add_argument("--timeout", type=int, default=900, help="per-detector timeout (sec)")
    ap.add_argument("--no-owner-tests", action="store_true", help="skip GOLD/owner groups")
    ap.add_argument(
        "--include-e2e-circuits",
        nargs="?",
        const="AUTO",
        default=None,
        help="optional path to external e2e_tests/circuits dir (default: auto)",
    )
    ap.add_argument("--keep-work", action="store_true", help="keep temp worktrees on exit")
    return ap.parse_args(argv)


def _resolve_e2e_dir(pkg_root: Path, opt: str | None) -> Path | None:
    if opt is None:
        return None
    if opt == "AUTO":
        cand = pkg_root.resolve().parents[2] / "e2e_tests" / "circuits"
    else:
        cand = Path(opt).resolve()
    if not cand.is_dir():
        raise ValueError(f"--include-e2e-circuits dir not found: {cand}")
    stims = sorted(cand.glob("*.stim"))
    if not stims:
        raise ValueError(f"--include-e2e-circuits has no *.stim files: {cand}")
    return cand


def _cleanup_work_dir(work_dir: Path, base_tree: Path, empty_home: Path) -> None:
    shutil.rmtree(base_tree, ignore_errors=True)
    shutil.rmtree(empty_home, ignore_errors=True)
    for child in list(work_dir.glob("m_*")) + list(work_dir.glob("bt_*")):
        shutil.rmtree(child, ignore_errors=True)
    for jf in list(work_dir.glob("e2e_*.json")):
        jf.unlink(missing_ok=True)


def _summarize_and_write(
    results: Sequence[MutantResult],
    baseline: dict[str, DetectorResult],
    wall: float,
    out_prefix: Path | None,
) -> int:
    tsv_text, md_text, payload = _format_outputs(results, baseline)
    n_k_test = sum(
        1
        for r in results
        if r.spec.expected == "kill" and r.verdict == "KILLED(test)"
    )
    n_k_crash = sum(
        1
        for r in results
        if r.spec.expected == "kill" and r.verdict == "KILLED(crash)"
    )
    n_eq_surv = sum(1 for r in results if r.verdict == "SURVIVED(justified_equiv)")
    n_bad_surv = sum(1 for r in results if r.verdict == "SURVIVED")
    n_timeout = sum(1 for r in results if r.verdict == "TIMEOUT")
    n_setup_err = sum(1 for r in results if r.verdict == "ERROR")
    n_non_equiv = sum(1 for r in results if r.spec.expected == "kill")
    n_killed = sum(
        1 for r in results
        if r.spec.expected == "kill" and r.verdict in ("KILLED(test)", "KILLED(crash)")
    )
    n_eq_killed = sum(
        1 for r in results
        if r.spec.expected == "equivalent" and r.verdict in ("KILLED(test)", "KILLED(crash)")
    )
    summary = (
        f"SUMMARY: total={len(results)} non_equiv_killed={n_killed}/{n_non_equiv} "
        f"(KILLED(test)={n_k_test} KILLED(crash)={n_k_crash}) "
        f"justified_equiv_survived={n_eq_surv} unexpected_survived={n_bad_surv} "
        f"timeout={n_timeout} setup_error={n_setup_err} "
        f"equiv_killed={n_eq_killed} wall={wall:.2f}s"
    )
    payload["summary"] = summary
    if out_prefix is not None:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        out_prefix.with_suffix(".tsv").write_text(tsv_text, encoding="utf-8")
        out_prefix.with_suffix(".md").write_text(
            md_text + "\n" + summary + "\n", encoding="utf-8"
        )
        out_prefix.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(md_text)
    print(summary)
    ok = (
        n_bad_surv == 0
        and n_timeout == 0
        and n_setup_err == 0
        and n_eq_killed == 0
        and n_killed == n_non_equiv
    )
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point enforcing E6 baseline-first and PATTERN-COUNT==1 hygiene."""
    args = _parse_args(argv)
    pkg_root = args.pkg.resolve()
    if args.cores:
        if not _CORES_RE.fullmatch(args.cores):
            print(f"ERROR: invalid --cores {args.cores!r}", file=sys.stderr)
            return 2
        if shutil.which("taskset") is None:
            print("ERROR: --cores requested but taskset not found", file=sys.stderr)
            return 2

    only = {tok.strip() for tok in args.only.split(",") if tok.strip()}
    try:
        mutants = load_mutants(args.tsv.resolve(), only)
        e2e_dir = _resolve_e2e_dir(pkg_root, args.include_e2e_circuits)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not mutants:
        print("ERROR: no mutants selected", file=sys.stderr)
        return 2

    work_dir = args.work.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    base_tree = work_dir / "baseline_tree"
    empty_home = work_dir / "empty_home"
    empty_home.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    try:
        _snapshot_pkg(pkg_root, base_tree)
        pat_errors = check_pattern_counts(base_tree, mutants)
        if pat_errors:
            print("E6 ABORT: PATTERN-COUNT != 1 violation(s):", file=sys.stderr)
            for mid, reason in pat_errors:
                print(f"  {mid}: {reason}", file=sys.stderr)
            return 2
        print(f"[E6] pattern check OK: {len(mutants)}/{len(mutants)} rows have PATTERN-COUNT == 1")

        baseline, e2e_base = _run_baseline(
            base_tree, work_dir, empty_home, args.python,
            args.cores, args.jobs, args.timeout, e2e_dir,
        )
        base_bad = [d for d in baseline.values() if d.status != "PASS"]
        for name, d in baseline.items():
            print(f"[E6] baseline {name:10s}: {d.status:13s} ({d.elapsed_s:.2f}s) {d.detail}")
        if base_bad:
            print("E6 ABORT: baseline detector failure(s)/crash(es):", file=sys.stderr)
            for d in base_bad:
                print(f"  {d.name}: {d.status} ({d.detail})", file=sys.stderr)
            return 3

        def _exec(m: MutantSpec) -> MutantResult:
            return _run_one_mutant(
                m, base_tree, work_dir, empty_home, args.python, args.cores,
                args.timeout, not args.no_owner_tests, e2e_dir, e2e_base, args.keep_work,
            )

        with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            results = list(ex.map(_exec, mutants))
    finally:
        if not args.keep_work:
            _cleanup_work_dir(work_dir, base_tree, empty_home)

    return _summarize_and_write(results, baseline, time.perf_counter() - t_total, args.out)


if __name__ == "__main__":
    sys.exit(main())
