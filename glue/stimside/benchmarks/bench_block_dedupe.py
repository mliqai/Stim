# Copyright 2026 Google LLC
"""Block-dedupe A/B benchmark: interleaved dedupe ON vs OFF, one process (not run by pytest).

    taskset -c 2-5 python benchmarks/bench_block_dedupe.py [xk4 xk6 r8 r12 ...] [--rounds N]

OFF replaces the private seam `stimside.simulator_magic._group_shots_by_state` by identity grouping
(per-shot block application). Each round runs ON then OFF; the best wall and CPU time over
the rounds is reported (CPU time = `time.process_time`, less sensitive to machine load).
Circuit names: `xkK` = X-memory d=3 + K T gates, `rN` = random T+CX on N qubits. B=64, seed=1.
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _circuits import random_t_cx, xmem_t

import stimside.simulator_magic as sm
from stimside import MagicsideSimulator

BATCH = 64
DEDUPE_ON = sm._group_shots_by_state


def dedupe_off(exps):
    return [[s] for s, e in enumerate(exps) if e.chi > 0]


def make_sim(name):
    c = xmem_t(int(name[2:])) if name.startswith("xk") else random_t_cx(int(name[1:]))
    return MagicsideSimulator(c, batch_size=BATCH, seed=1)


def timed_run(sim, grouping):
    sm._group_shots_by_state = grouping
    w0, c0 = time.perf_counter(), time.process_time()
    sim.clear()
    sim.run()
    sim.get_detector_flips()
    return time.perf_counter() - w0, time.process_time() - c0


def ab(name, rounds):
    sims = {"on": make_sim(name), "off": make_sim(name)}
    groupings = {"on": DEDUPE_ON, "off": dedupe_off}
    for tag, sim in sims.items():  # warm-up (also loads the C++ kernels)
        timed_run(sim, groupings[tag])
    best = {t: [float("inf"), float("inf")] for t in sims}
    for _ in range(rounds):
        for tag, sim in sims.items():
            w, c = timed_run(sim, groupings[tag])
            best[tag] = [min(best[tag][0], w), min(best[tag][1], c)]
    sm._group_shots_by_state = DEDUPE_ON
    on, off = best["on"], best["off"]
    print(
        f"{name:5s} OFF wall {1e3 * off[0]:8.1f} ms cpu {1e3 * off[1]:8.1f} ms | "
        f"ON wall {1e3 * on[0]:8.1f} ms cpu {1e3 * on[1]:8.1f} ms | "
        f"speedup wall {off[0] / on[0]:.2f}x cpu {off[1] / on[1]:.2f}x",
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("circuits", nargs="*", default=["xk4", "xk6", "r8", "r12"])
    ap.add_argument("--rounds", type=int, default=5, help="interleaved ON/OFF rounds (default 5)")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    print(f"loadavg(1/5/15 min) {' '.join(f'{x:.1f}' for x in os.getloadavg())}")
    for name in args.circuits:
        ab(name, args.rounds)
    print("loadavg after: " + " ".join(f"{x:.1f}" for x in os.getloadavg()))


if __name__ == "__main__":
    main()
