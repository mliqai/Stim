# Copyright 2026 Google LLC
"""Magicside throughput benchmarks (warm, best-of-N), run as a script; never collected by pytest.

    taskset -c 2-5 python benchmarks/bench_magic_suite.py [sc] [xmem] [rand] [--reps N]

See README.md for the expected baseline numbers and how to compare runs.
"""

import argparse
import os
import sys
import time
import warnings

import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _circuits import build_leakage_surface_code, random_t_cx, t_count, xmem_t

from stimside import (
    CosetsideSimulator,
    LeakageUint8Coset,
    LeakageUint8Magic,
    MagicsideSimulator,
)

BATCH = 64
ENGINES = (("live", {}), ("compiled", {"fault_mode": "SELECTOR"}), ("query", {"backend": "QUERY"}))


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


def leaky_factory(circuit, sim_cls, handler_cls):
    handler = handler_cls().compile_op_handler(circuit=circuit, batch_size=BATCH)
    return lambda: sim_cls(circuit, compiled_op_handler=handler, batch_size=BATCH, seed=1)


def magic_factory(circuit, **kwargs):
    return lambda: MagicsideSimulator(circuit, batch_size=BATCH, seed=1, **kwargs)


def leaky_sc(reps):
    print("== leaky rotated SC (Clifford + leakage), B=64: Coset vs Magic ==")
    for d in (3, 5):
        c = build_leakage_surface_code(d, d)
        coset = best_throughput(leaky_factory(c, CosetsideSimulator, LeakageUint8Coset), 10, reps)
        magic = best_throughput(leaky_factory(c, MagicsideSimulator, LeakageUint8Magic), 2, reps)
        print(f"d={d}: Coset {coset:8.1f}  Magic {magic:7.1f} shots/s  ratio {coset / magic:5.1f}x")


def xmem_k_t(reps):
    print("== X-memory SC d=3, 3 rounds, + k T gates on data qubits (no leakage), B=64 ==")
    for k in (0, 2, 4, 6):
        c = xmem_t(k)
        row = []
        for label, kwargs in ENGINES:
            row.append(f"{label} {best_throughput(magic_factory(c, **kwargs), 2, reps):7.1f}")
        print(f"k={k}: " + "   ".join(row) + " shots/s")


def random_t_cx_n12(reps):
    print("== random T+CX n=12 (live), B=64 ==")
    c = random_t_cx(12)
    print(f"n=12 ({t_count(c)} T): live {best_throughput(magic_factory(c), 3, reps):7.1f} shots/s")


SUITES = {"sc": leaky_sc, "xmem": xmem_k_t, "rand": random_t_cx_n12}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suites", nargs="*", choices=list(SUITES), help="default: all")
    ap.add_argument("--reps", type=int, default=3, help="best-of N timing repetitions (default 3)")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    load = " ".join(f"{x:.1f}" for x in os.getloadavg())
    print(f"stim {stim.__version__}  loadavg(1/5/15 min) {load}")
    for name in args.suites or list(SUITES):
        SUITES[name](args.reps)
    print("loadavg after: " + " ".join(f"{x:.1f}" for x in os.getloadavg()))


if __name__ == "__main__":
    main()
