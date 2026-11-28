# Magicside benchmarks

Throughput and cross-engine comparison benchmarks for the Magicside simulator (default term engine
and stabilizer-branch engine) and its Coset comparison. They live outside `src/` on purpose:
`layering_test` rejects unclassified modules under `src/stimside`, pytest only collects `src`
(`testpaths`), and the sdist/wheel ship `src/` only. Nothing here runs in CI (except lightweight
CLI/schema smoke checks in `simulator_magic_branch_counters_test.py` when run from a source checkout).

```bash
cd Stim/glue/stimside
PYTHONPATH=src taskset -c 2 python benchmarks/bench_magic_suite.py            # all suites
PYTHONPATH=src taskset -c 2 python benchmarks/bench_magic_suite.py xmem --reps 5
PYTHONPATH=src taskset -c 2 python benchmarks/bench_magic_suite.py --reps 3 --jsonl /path/to/results.jsonl
PYTHONPATH=src taskset -c 2 python benchmarks/compare_magic_engines.py \
    --circuit /path/to/circuit.stim --shots 102400 --batch-size 4096 --seed 101 --check
```

### Scripts & Suites

- **`bench_magic_suite.py`**: Runs three synthetic suites built in `_circuits.py` at batch size `B=64`,
  `seed=1`, one warm-up batch, then best of `--reps` (default 3) of `clear(); run(); get_detector_flips()`
  (numbers are steady-state shots/s):
  - `sc`: leaky rotated surface code `d=3, 5` (`Coset`, `Magic` term engine, and `branch` engine).
  - `xmem`: X-memory `d=3`, 3 rounds, `+ k = 0, 2, 4, 6` T gates (`live` = default term engine,
    `compiled` = `fault_mode="SELECTOR"`, `query` = `backend="QUERY"`, and `branch` =
    `_engine="branches", max_components=4096`).
  - `rand`: random `T+CX` on `n=12` qubits (43 non-cancelling T gates; `live` vs `branch` with
    `max_components=4096`). On unstructured random T circuits without check cancellation, branches
    grow exponentially and `branch` cleanly reports `FAILED (QueryBudgetExceededError)` rather than
    hanging unbounded.
  - Optional `--jsonl PATH` (alias `--json-lines`) writes one JSON object per `(workload, engine)`
    entry (24 lines for all 3 suites) conforming to the `stimside-paper-results/1` schema, keeping
    `status="failed"` / `"unsupported"` entries alongside `"ok"` entries and embedding Section 7
    `branch_counters` on branch entries. Note that `run.rss_mb` in `--jsonl` is process-wide
    `resource.getrusage(RUSAGE_SELF).ru_maxrss` (cumulative high-water mark across entries in the
    same Python process), not isolated per-entry RSS; use fresh subprocesses (as in P1–P3 below)
    for isolated memory measurements.
- **`compare_magic_engines.py`**: Compares `_engine="terms"` vs `_engine="branches"` on any external
  `.stim` circuit (`--circuit <path>`) on acceptance rate (two-proportion `z`-test, `|z| < 3.0`),
  first-detection-event histogram (chi-square goodness-of-fit, `p > 0.01`), throughput (`shots/s`),
  and Section 7 branch counters (with optional `--json-out` and `--check`). External cultivation
  circuits derived from `github.com/kaavyas99/MSC_foldedH` (arXiv:2509.05212) stay strictly outside
  the `Stim` repository because the upstream repository carries no license file.

## Measuring correctly (the machine is noisy)

- Pin to a dedicated core (`taskset -c 2`), fix single-threaded BLAS (`OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`), and record `os.getloadavg()`
  before and after every run: absolute numbers roughly halve at load 75 on 64 cores. Compare
  **ratios** (Coset/Magic, branches/terms) and **before/after on the same load**: interleave fresh
  subprocesses of the old and new engines/trees across at least 3 rounds and report medians.
- Throughput is a performance reference only: no test asserts on it.

## Baseline (Plan v8 §0.2, portable build, engine ON by default, B=64, pinned-core CPU shots/s)

| Benchmark | Expected |
|---|---|
| leaky SC d=3, Coset vs Magic | 4,275–4,367 vs 1,675 shots/s (~2.6x; OFF 675, ON/OFF ~2.4–2.5x) |
| leaky SC d=5, Coset vs Magic | 1,647–1,670 vs 397 shots/s (~4.2x; OFF 128, ON/OFF ~3.1x) |
| X-memory d=3 + k=6 T, live | 4,019 shots/s (OFF 905, ON/OFF ~4.3–4.4x) |
| X-memory d=3 + k T, compiled / QUERY | ~89–99 / ~72 shots/s (W3 ref; no live engine) |
| random T+CX n=12 (43 T), live | 180–189 shots/s portable (~206 native; OFF 133, ~1.35–1.4x) |

Measured on `11b42b5` / Run 8 Stage A with `STIMSIDE_PORTABLE_BUILD` (the default); setting
`STIMSIDE_NATIVE_BUILD=1` gives ~10% higher throughput on `random T+CX n=12` (~206 shots/s) and
matches within noise on SC and X-memory. Compiled and `QUERY` modes do not use the live engine
and keep their W3 reference figures.

## Stabilizer-branch engine official benchmarks (WP8, P1–P4)

### Environment, Commit Attribution & Protocol

- **Hardware**: `AMD EPYC 7B13` (`x86_64`, 64 logical CPUs / 32 physical cores, 1 socket, `128 MiB` L3 cache, `117.9 GiB` RAM)
- **OS & Compiler**: `Linux-7.1.6-1rodete1-amd64-x86_64-with-glibc2.43`, `g++ (Debian 15.2.0-17) 15.2.0`
- **Software**: `Python 3.13.18`, `numpy 2.4.4`, `stim 1.16.dev1776288508`, `stimside 0.1.0`
- **Pinning & Isolation**: `taskset -c 2`, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`, fresh subprocess per engine/round, native `.so` pre-warmed prior to timing, `cpu/wall = 1.000` on all P1–P3 runs, 3 interleaved rounds per benchmark (`os.getloadavg()` recorded before & after every run; 1-min load ranged `1.03`–`1.81` across 64 logical CPUs).
- **Commit Attribution**:
  - Measured on the WP8 working tree atop WP7 `98cbe0733bb1dbea79a0fc4b673f6521791adbdc`; production engine code under `src/stimside/` is unchanged from WP6 `50fedd1` and WP7 `98cbe07` and is identical to the WP8 commit (WP8 modifies only `benchmarks/bench_magic_suite.py`, `benchmarks/README.md`, and `src/stimside/simulator_magic_branch_counters_test.py`).
  - P4 compares WP8's default engine against pre-branch baseline `3ad1d19c07fafafd9c4cd57ca4f959fc75ed1dfb` in an isolated temporary worktree using `3ad1d19`'s identical 17-metric `bench_magic_suite.py --reps 3` script.
- **External W2m Circuits** (kept outside `Stim`; magic stage of fault-distance-3 cultivation, 20 qubits, 14 measurements, 14 detectors, derived from `github.com/kaavyas99/MSC_foldedH`, arXiv:2509.05212):
  - `w2m_p1e3.stim` (Pauli `1e-3`, no loss): SHA-256 `fc50ef7d83c19dec56408f049f952bdc173474891bb79d02830e2fa57af48c72`
  - `w2m_loss1e3.stim` (Pauli `1e-3` + atom loss `1e-3`): SHA-256 `afcd206edf9fcaa383d03c9ec209e43e4ca21c16241077b7e590bc89e2bb550d`

### Table 1 — Official P1–P4 Summary (3 interleaved rounds, `taskset -c 2`)

| Target | Workload & Config | Baseline / Reference Median (min–max) | WP8 Branch / Default Median (min–max) | Ratio / Peak RSS | 1-min Loadavg | Verdict |
|---|---|---|---|---|---|---|
| **P1** (warm `>= 3.0×`) | `w2m_p1e3.stim`, `B=4096`, 250 batches (`1,024,000` shots from fresh start) | `terms`: **10,144.1** sps (10,026.1–10,151.5) | `branches`: **69,859.9** sps (69,266.4–70,840.5; after batch 0: 70,514.6 sps) | **6.887×** (`VmHWM` 182.0–183.8 MiB) | 1.16–1.38 | **PASS** |
| **P2** (cold `>= 1.0×`) | `w2m_p1e3.stim`, `B=4096`, 12 batches (`49,152` shots), `_BRANCH_STORE_MAX_NODES=0` (`store_clears=11`) | `terms`: **9,861.9** sps (9,640.9–10,001.1) | `branches_cold`: **40,997.9** sps (39,882.3–41,131.0) | **4.157×** (`VmHWM` 67.48–67.62 MiB) | 1.20–1.28 | **PASS** |
| **P3** (memory `<= 1536 MiB`) | `w2m_loss1e3.stim`, `LeakageUint8Magic()`, `B=4096`, `cap=300,000`, 120 batches (`491,520` shots, `store_clears=1`) | Limit: `1,536.0 MiB` (`1.500 GiB`) | `VmHWM` (`== ru_maxrss`): **506.41 MiB** (503.11–515.32 MiB; `peak_RssAnon` 476.68–488.60 MiB; **20,213.6** sps) | **0.495 GiB** med / **0.503 GiB** max | 1.13–1.20 | **PASS** |
| **P4** (no regression) | Pre-branch `3ad1d19` vs WP8 default engine on identical 17-metric `bench_magic_suite.py --reps 3` | `3ad1d19` (17 metrics, wall 43.34–43.49 s) | WP8 default (17 metrics, wall 43.17–44.08 s) | **0.998×** median (range 0.962×–1.022×) | 1.03–1.33 | **PASS** |

### Table 2 — `bench_magic_suite.py` (`B=64`, `--reps 3`, `taskset -c 2`, 3-round medians, steady shots/s)

| Suite / Workload | Coset / `live` | Magic / `compiled` | `query` | `branch` (`max_components=4096`) | Branch `peak_branches` / `store_nodes` |
|---|---|---|---|---|---|
| `sc_d3` (leaky SC `d=3`) | 7,790.9 | 2,934.1 | — | **2,454.6** | `1` / `1,209` |
| `sc_d5` (leaky SC `d=5`) | 3,242.4 | 794.5 | — | **493.0** | `1` / `12,376` |
| `xmem_k0` (`d=3, k=0 T`) | 12,986.7 | 122.2 | 86.5 | **8,441.1** | `1` / `18` |
| `xmem_k2` (`d=3, k=2 T`) | 11,722.5 | 115.1 | 83.8 | **8,170.6** | `2` / `40` |
| `xmem_k4` (`d=3, k=4 T`) | 9,296.2 | 113.2 | 82.1 | **8,190.7** | `4` / `75` |
| `xmem_k6` (`d=3, k=6 T`) | 8,926.2 | 105.3 | 82.9 | **8,129.4** | `8` / `191` |
| `rand_n12` (12q, 43 T) | 190.7 | — | — | **FAILED (`QueryBudgetExceededError`)** | `4096` / `164` |

**Honest operating-point notes on Table 2**:
- `bench_magic_suite.py` runs at small batch size `B=64` (preserving the Plan v8 §0.2 baseline). At
  `B=64`, the branch engine's per-instruction Python/ctypes dispatch is amortized over only 64 shots,
  so `branch` trails `live`'s fused C++ blocks on Clifford-heavy `sc` and `xmem k=0..6` (while
  remaining ~67–98× faster than `compiled`/`query` and staying nearly flat from `k=0` to `k=6`). At
  the production batch size `B=4096`, dispatch amortizes away and `branches` outperforms `terms` by
  **6.89×** on W2m Pauli `1e-3` (Table 1) and **3.71×** on W2m with loss `1e-3` (Table 3).
- On `rand_n12` (43 non-cancelling random T gates with no check structure), exact stabilizer-branch
  count grows exponentially and hits `max_components=4096`, recorded cleanly with `status="failed"`
  in `--jsonl` (`23 ok + 1 failed` across the 24-record full suite).

### Table 3 — W2m Statistical & Throughput Verification (`compare_magic_engines.py`, `102,400` shots/engine, `B=4096`)

Recorded in WP7 (`work/wp7_w2m_compare.txt`, measured at `50fedd1` — identical engine code to `98cbe07` / WP8):

| Circuit | Leakage | `terms` Accepted (`rate`) | `branches` Accepted (`rate`) | `z` (`|z|<3.0`) | `chi2` (`dof`) | `p` (`p>0.01`) | `terms` sps | `branches` sps | Speedup | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `w2m_p1e3.stim` (`seed=101`) | off | 81,934 / 102,400 (0.800137) | 81,845 / 102,400 (0.799268) | `+0.4914` | `9.2534` (`14`) | `0.814486` | 10,045.6 | 59,522.5 | **5.93×** | **PASS** |
| `w2m_loss1e3.stim` (`seed=201`) | on (`LeakageUint8Magic`) | 74,192 / 102,400 (0.724531) | 74,241 / 102,400 (0.725010) | `-0.2424` | `13.5741` (`14`) | `0.481900` | 5,088.9 | 18,874.5 | **3.71×** | **PASS** |

### Reproducible Commands

```bash
cd Stim/glue/stimside
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Full bench_magic_suite with stimside-paper-results/1 JSONL output (24 records: 23 ok + 1 failed):
PYTHONPATH=src taskset -c 2 python -B benchmarks/bench_magic_suite.py --reps 3 --jsonl /path/to/wp8_bench_suite.jsonl

# Statistical + throughput comparison on external W2m circuits:
PYTHONPATH=src taskset -c 2 python -B benchmarks/compare_magic_engines.py \
    --circuit /path/to/w2m_p1e3.stim --shots 102400 --batch-size 4096 --seed 101 --check
PYTHONPATH=src taskset -c 2 python -B benchmarks/compare_magic_engines.py \
    --circuit /path/to/w2m_loss1e3.stim --shots 102400 --batch-size 4096 --seed 201 --check
```

## Related checks that are NOT here

The chi envelope (12 commuting T on 12 qubits -> chi = 4096; 8 qubits with k = 8 -> chi = 256) is
already asserted by
`simulator_magic_test.py::test_magic_chi_envelope_commuting_t_and_scaling_pattern`
(0.25 s, runs in the default suite).
