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

## Baseline (plan v2 §3, quiet-ish machine, W3 tree)

| Benchmark | Expected |
|---|---|
| leaky SC d=3, Coset vs Magic | 4,518 vs 538 shots/s (8.4x) |
| leaky SC d=5, Coset vs Magic | 1,720 vs 109 shots/s (15.8x) |
| X-memory d=3 + k T, live, k = 0 / 2 / 4 / 6 | 1,141 / 909 / 591 / 468 shots/s |
| X-memory d=3 + k T, compiled | 89-99 shots/s |
| X-memory d=3 + k T, QUERY | ~72 shots/s |
| random T+CX n=12 (43 T), live | 44 shots/s |

First measurement of this harness (WP-24 batch B1, tree with the header move, no behavioural
change) is in the WP-24 B1 handoff; it matched the ratios above and was lower in absolute terms
because of machine load.

## Related checks that are NOT here

The chi envelope (12 commuting T on 12 qubits -> chi = 4096; 8 qubits with k = 8 -> chi = 256) is
already asserted by
`simulator_magic_test.py::test_magic_chi_envelope_commuting_t_and_scaling_pattern`
(0.25 s, runs in the default suite).
