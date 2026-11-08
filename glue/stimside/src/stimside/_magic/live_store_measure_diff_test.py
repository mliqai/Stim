"""Bit-exact differential tests of `mbe_measure_split` / `mbe_measure_finish` (WP-25 stage S5a).

Three paths run the same measurement substeps on the same states and the same random stream:

* the ENGINE: `MagicsideSimulator._project_substep_batch` with the shots in a `LiveExpansions`
  (C++ split, Python checks and draw, C++ finish);
* the REFERENCE: the same method on a plain `list` of `ShotExpansion`s (today's code, the
  `measure_expansions_batch` kernel path; the golden digests pin it);
* the ORACLE: today's PER-SHOT functions, `measure_expansion` and `apply_pauli`, driven by a loop
  written here (no engine or batch code involved; non-sync runs only).

Everything is compared as bytes (weights through a uint64 view, so -0.0 != 0.0): the measured
bits, the branch masses, every shot's eight arrays, `log_mass`, the canonical flag, the numpy
random generator state (an extra or missing draw shows), and in sync mode each per-shot stream,
each stim tableau stream and the sequence of `(shot, p1)` sampler calls. An exception must have
the same type and text on both paths and leave the same partial state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import stim  # type: ignore[import-untyped]

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import MeasurementRecord, ResetRecord
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_test import (
    _CELLS,
    _first_difference,
    _nonzero_axis,
    random_state,
)
from stimside.simulator_magic import MagicsideSimulator
from stimside.util.bitwords import word_dot_parity

_ATOL = 1e-10
_KINDS = ("plain", "d", "raw", "signed0", "zero", "basis", "fresh")


@dataclass
class Coverage:
    cases: int = 0  # (shot, substep) pairs run on all paths
    active: int = 0
    trivial: int = 0
    dead: int = 0
    rotating: int = 0
    kept_frame: int = 0
    resets: int = 0
    kicked: int = 0
    d_positive: int = 0
    non_canonical: int = 0
    chi_gt1: int = 0
    signed_zero: int = 0
    oracle_cases: int = 0
    raised: dict[str, int] = field(default_factory=dict)


def _state(rng: np.random.Generator, n: int, chi: int, kind: str) -> pq.ShotExpansion:
    if kind == "fresh":
        return pq.ShotExpansion.zero(n)
    if kind == "basis":  # chi = 1, d = 0, h != 0: a trivial shot whose bit is not always 0
        a, b, kap = _nonzero_axis(rng, n)
        return pq.apply_pauli(pq.ShotExpansion.zero(n), a, b, kap)
    return random_state(rng, n, chi, kind)


def _axis_words(rng: np.random.Generator, n: int, kind: str) -> tuple[Any, Any, int]:
    nw = pq.n_words_for_qubits(n)
    if kind == "gen":
        return _nonzero_axis(rng, n, density=float(rng.choice([0.05, 0.3, 0.8])))
    b = np.zeros(nw, np.uint64)
    if kind == "unit":
        q = int(rng.integers(0, n))
        b[q >> 6] = np.uint64(1) << np.uint64(q & 63)
    else:  # "z": a random nonzero Z-type product
        while not b.any():
            b = pq.pack_bit_vector((rng.random(n) < 0.3).astype(np.uint8), n)
    return np.zeros(nw, np.uint64), b, 2 * int(rng.integers(0, 2))


def _record(
    rng: np.random.Generator, n: int, reset: bool, flip_ref: bool, kind: str, violate: bool
) -> MeasurementRecord | ResetRecord:
    ax = CompiledPauliAxis(*_axis_words(rng, n, kind))
    rotates = bool(ax.a_words.any())
    rot = ax if rotates != violate else None  # `violate`: rot_axis disagrees with the words
    if not reset:
        return MeasurementRecord(0, 0.0, False, ax, rot)
    kick = CompiledPauliAxis(*_nonzero_axis(rng, n))
    return ResetRecord(None, 0.0, False, ax, rot, flip_ref, kick)


def _sim(n: int, batch: int, seed: int, sync: bool, atol: float = _ATOL) -> MagicsideSimulator:
    circ = stim.Circuit(f"R {n - 1}\nI 0")
    sim = MagicsideSimulator(circ, batch_size=batch, seed=seed, sync_tableside_rng=sync, atol=atol)
    assert sim._eff_nq == n
    return sim


def _calls(sim: MagicsideSimulator) -> list[tuple[int, str]]:
    """Record every `(shot, p1)` the sync sampler is asked for (the callback stays Python)."""
    seen: list[tuple[int, str]] = []
    real = sim._sample_sync_born_bit

    def spy(b: int, p1: float) -> int:
        seen.append((b, float(p1).hex()))
        return real(b, p1)

    sim._sample_sync_born_bit = spy  # type: ignore[method-assign]
    return seen


def _pair(
    states: list[pq.ShotExpansion], n: int, seed: int, sync: bool, atol: float = _ATOL
) -> tuple[MagicsideSimulator, MagicsideSimulator, list[Any], list[Any]]:
    ref, eng = (_sim(n, len(states), seed, sync, atol) for _ in range(2))
    ref._live_exps = [s.copy() for s in states]
    store = LiveExpansions(len(states), n, atol)
    for b, s in enumerate(states):
        store[b] = s
    eng._live_exps = store
    assert not isinstance(ref._live_exps, LiveExpansions)  # whatever the default store is
    assert isinstance(eng._live_exps, LiveExpansions)
    return ref, eng, _calls(ref), _calls(eng)


def _step(sim: MagicsideSimulator, rec: Any) -> tuple[Any, Exception | None]:
    try:
        with np.errstate(all="ignore"):
            return sim._project_substep_batch(rec), None
    except (FloatingPointError, ValueError, ZeroDivisionError, MemoryError, RuntimeError) as exc:
        return None, exc  # the type and text are compared by the caller


def _rng_states(sim: MagicsideSimulator) -> list[Any]:
    out = [sim.np_rng.bit_generator.state]
    out += [r.bit_generator.state for r in sim._shot_np_rngs or []]
    return out


def _drain_tab_streams(sim: MagicsideSimulator) -> list[list[bool]]:
    return [[sim._draw_tab_rng_bit(b) for _ in range(6)] for b in range(len(sim._tab_rngs or []))]


def _assert_same(
    ref: MagicsideSimulator, eng: MagicsideSimulator, label: str, ref_calls: list[Any],
    eng_calls: list[Any],
) -> None:
    assert ref._log_branch_masses.tobytes() == eng._log_branch_masses.tobytes(), label
    for b in range(ref.batch_size):
        diff = _first_difference(ref._live_exps[b], eng._live_exps[b])
        assert diff is None, f"{label} shot {b}: {diff}"
    assert _rng_states(ref) == _rng_states(eng), f"{label}: random state"
    assert ref_calls == eng_calls, f"{label}: sampler calls"
    if ref._tab_rngs is not None:
        assert _drain_tab_streams(ref) == _drain_tab_streams(eng), f"{label}: tableau streams"


def _classify(exps: list[pq.ShotExpansion], rec: Any, cov: Coverage, res: Any) -> None:
    for b, e in enumerate(exps):
        cov.cases += 1
        cov.dead += e.chi == 0
        triv = e.chi > 0 and rec.rot_axis is None and e.d == 0 and e.chi == 1
        cov.trivial += triv
        act = e.chi > 0 and not triv
        cov.active += act
        cov.d_positive += act and e.d > 0
        cov.non_canonical += act and not e._is_canonical_merged
        cov.chi_gt1 += act and e.chi > 1
        cov.signed_zero += act and bool(np.any(e.weights.view(np.uint64) == np.uint64(1 << 63)))
        cov.rotating += act and rec.obs_axis.a_words.any()
        cov.kept_frame += act and rec.rot_axis is not None
    cov.resets += isinstance(rec, ResetRecord)


def _oracle_step(
    exps: list[pq.ShotExpansion], rec: Any, rng: np.random.Generator, atol: float
) -> tuple[Any, list[pq.ShotExpansion], list[float]]:
    """Today's per-shot functions in the order `_project_substep_batch` uses them: the trivial
    shots first (`apply_pauli` kicks), then `measure_expansion` shot by shot, then the kicks."""
    obs, m = rec.obs_axis, np.zeros(len(exps), np.bool_)
    out, masses = [e.copy() for e in exps], [0.0] * len(exps)
    idx: list[int] = []
    for b, e in enumerate(exps):
        if e.chi == 0:
            masses[b] = -math.inf
        elif rec.rot_axis is None and e.d == 0 and e.chi == 1:
            bit = ((obs.kappa >> 1) ^ word_dot_parity(obs.b_words, e.h_words[0])) & 1
            m[b] = bool(bit)
            if isinstance(rec, ResetRecord) and bool(bit) != rec.reset_flip_ref:
                out[b] = pq.apply_pauli(e, rec.reset_kick_axis)
        else:
            idx.append(b)
    for b in idx:
        r = pq.measure_expansion(
            exps[b], obs, rng=rng, renormalize=True, keep_rotated_frame=rec.rot_axis is not None,
            atol=atol,
        )
        post = r.post_state
        if isinstance(rec, ResetRecord) and r.outcome != int(rec.reset_flip_ref):
            post = pq.apply_pauli(post, rec.reset_kick_axis)
        out[b], m[b] = post, bool(r.outcome)
    return m, out, masses


def _cell_case(
    n: int, chi: int, seed: int, sync: bool, cov: Coverage, steps: int = 3
) -> None:
    rng = np.random.default_rng([seed, n, chi, 5, int(sync)])
    batch = int(rng.choice([1, 2, 3, 7, 12]))
    states = [_state(rng, n, chi, _KINDS[int(rng.integers(0, len(_KINDS)))]) for _ in range(batch)]
    ref, eng, rc, ec = _pair(states, n, int(rng.integers(0, 2**30)), sync)
    for step in range(steps):
        rec = _record(
            rng, n, bool(rng.random() < 0.45), bool(rng.integers(0, 2)),
            str(rng.choice(["gen", "gen", "z", "unit"])), bool(rng.random() < 0.08),
        )
        before = [ref._live_exps[b].copy() for b in range(batch)]
        orc_rng = copy.deepcopy(ref.np_rng)
        want, wexc = _step(ref, rec)
        got, gexc = _step(eng, rec)
        label = f"n={n} chi={chi} seed={seed} sync={sync} step={step}"
        assert (type(gexc), str(gexc)) == (type(wexc), str(wexc)), (label, gexc, wexc)
        if wexc is not None:
            cov.raised[type(wexc).__name__] = cov.raised.get(type(wexc).__name__, 0) + 1
            _assert_same(ref, eng, label + " (after an error)", rc, ec)
            return
        assert want.tobytes() == got.tobytes(), label
        _assert_same(ref, eng, label, rc, ec)
        _classify(before, rec, cov, want)
        if not sync:  # the per-shot oracle, on the states before the step
            m, post, _masses = _oracle_step(before, rec, orc_rng, _ATOL)
            assert m.tobytes() == want.tobytes(), label + " (oracle bits)"
            for b in range(batch):
                diff = _first_difference(post[b], eng._live_exps[b])
                assert diff is None, f"{label} oracle shot {b}: {diff}"
            assert orc_rng.bit_generator.state == eng.np_rng.bit_generator.state, label
            cov.oracle_cases += batch


def _run_all(sync: bool, seeds: range) -> Coverage:
    cov = Coverage()
    for n, chi in _CELLS:
        for seed in seeds:
            _cell_case(n, chi, seed, sync, cov)
    return cov


def _check_coverage(cov: Coverage, minimum: int) -> None:
    assert cov.cases >= minimum, cov
    assert cov.active >= minimum // 2 and cov.trivial >= 100 and cov.dead >= 50, cov
    assert cov.rotating >= 200 and cov.kept_frame >= 100 and cov.resets >= 100, cov
    assert cov.d_positive >= 150 and cov.chi_gt1 >= 150 and cov.non_canonical >= 20, cov


def test_engine_measurements_match_the_reference_and_the_per_shot_oracle_on_many_cases() -> None:
    cov = _run_all(False, range(30))
    _check_coverage(cov, 2000)
    assert cov.oracle_cases >= 2000, cov


def test_engine_measurements_match_the_reference_in_sync_mode_on_many_cases() -> None:
    """Sync mode: no `rng.random(n)`; the per-shot sampler callback is called with the same
    `(shot, p1)` in the same order, and every per-shot and tableau stream ends in the same state."""
    cov = _run_all(True, range(30))
    _check_coverage(cov, 2000)
