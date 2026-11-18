# Magicside benchmarks

Throughput benchmarks for the Magicside simulator and its Coset comparison. They live outside
`src/` on purpose: `layering_test` rejects unclassified modules under `src/stimside`, pytest only
collects `src` (`testpaths`), and the sdist/wheel ship `src/` only. Nothing here runs in CI.

```bash
cd Stim/glue/stimside
PYTHONPATH=src taskset -c 2-5 python benchmarks/bench_magic_suite.py            # all suites
PYTHONPATH=src taskset -c 2-5 python benchmarks/bench_magic_suite.py xmem --reps 5
```

Suites: `sc` (leaky rotated surface code d=3/d=5, Coset vs Magic), `xmem` (X-memory d=3 plus
k = 0/2/4/6 T gates; `live` = default engine, `compiled` = `fault_mode="SELECTOR"`, `query` =
`backend="QUERY"`), `rand` (random T+CX, n=12, 43 T). All use batch size 64, seed 1, a warm-up run,
then best of `--reps` (default 3) of `clear(); run(); get_detector_flips()`; numbers are shots/s.
Circuits are built in `_circuits.py`.

## Measuring correctly (the machine is noisy)

- Pin to a few cores (`taskset`), and read `uptime`/the printed loadavg: the absolute numbers below
  roughly halve at load 75 on 64 cores. Compare **ratios** (Coset/Magic) and **before/after on the
  same load**: interleave runs of the old and new tree, at least 3 rounds, take the best.
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

## Related checks that are NOT here

The chi envelope (12 commuting T on 12 qubits -> chi = 4096; 8 qubits with k = 8 -> chi = 256) is
already asserted by
`simulator_magic_test.py::test_magic_chi_envelope_commuting_t_and_scaling_pattern`
(0.25 s, runs in the default suite).
