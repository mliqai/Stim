"""Seed derivation helpers shared across stimside simulators and samplers."""

from __future__ import annotations

import operator
import os

import numpy as np

_M64 = (1 << 64) - 1
_M63 = (1 << 63) - 1


class CompiledSeedMixin:
    """`seed` for the first compile in the creating process; a derived per-compile/per-process seed after."""

    seed: int | None

    def _next_compiled_seed(self) -> int | None:
        if self.seed is None:
            return None
        idx = getattr(self, "_compile_count", 0)
        init_pid = getattr(self, "_init_pid", os.getpid())
        cur_pid = os.getpid()
        self._compile_count = idx + 1
        if idx == 0 and cur_pid == init_pid:
            return int(self.seed)
        pid_delta = (cur_pid - init_pid) & 0xFFFFFFFF
        ss = np.random.SeedSequence([int(self.seed), int(idx), int(pid_delta)])
        return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x3FFFFFFF)


def _mix64(z: int) -> int:
    """The splitmix64 output function, a bijection of 64-bit integers."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def derived_seed(seed: int, n: int) -> int:
    """Seed in [0, 2**63) of random stream `n` (a shot or a batch) of a run seeded with `seed`.

    A splitmix64 hash of (seed, n), for every n including 0. Unlike `seed + n`, runs with different
    seeds don't share streams (shifted by a few shots or batches) and consecutive streams don't get
    related seeds. Tableside and Cosetside derive the same seeds, keeping sync_tableside_rng lockstep.
    `seed` must be an integer (else TypeError) in [0, 2**64), the range stim accepts (else ValueError).
    """
    seed = operator.index(seed)
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be None or an integer in [0, 2**64), got {seed}")
    return _mix64((_mix64(seed) + int(n) * 0x9E3779B97F4A7C15) & _M64) & _M63
