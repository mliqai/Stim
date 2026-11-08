"""Bit-exact differential tests of `mbe_peek`, `mbe_postselect` and `mbe_unmatched_apply` (S5b).

The REFERENCE is `MagicsideSimulator` on a plain list of `ShotExpansion`s (today's per-shot code:
`_obs_p1`, `measure_expansion`, `apply_pauli`); the ENGINE is the same simulator method with the
shots in a `LiveExpansions`. Each case builds two simulators from the same states, drives the three
public methods (`peek_pauli_born_batch`, `postselect_z_batch`, `do_unmatched_noisy_reset_z`) on
both with a random circuit-free observable (`_live_ref_tab` scrambled by random Cliffords so the
pulled-back axes are general `X(a) Z(b)` products) and compares, as bytes: the peeked
probabilities, every shot's eight arrays, `log_mass`, the canonical flag, the branch masses, the
numpy generator state (an extra or missing draw shows), and in sync mode each per-shot stream,
each stim tableau stream and the `(shot, p1)` sampler calls. An exception must have the same
type and text on both paths and leave the same partial state.

Cases: n in (5, 64, 70, 130) x chi in (1, 4, 64) states of every kind (plain, d > 0, raw
non-canonical, signed zeros, dead, basis, fresh), masks with dead / pinned / drawn shots, and
failures (impossible postselection at a shot in the middle of the mask, non-finite or zero-norm
weights, an invalid atol).
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions
from stimside._magic.live_store_measure_diff_test import _assert_same, _calls, _pair
from stimside._magic.live_store_test import _CELLS, random_state
from stimside.simulator_magic import MagicsideSimulator

_ATOL = 1e-10
_KINDS = ("plain", "d", "raw", "signed0", "zero", "basis", "fresh")
_GATES_1Q = ("H", "S", "X", "Z", "Y", "S_DAG", "SQRT_X", "SQRT_Y")


@pytest.fixture(autouse=True)
def _collect_cycles_after_each_test() -> Iterator[None]:
    """Simulators and their spies form reference cycles that hold a native engine until the
    cyclic gc runs; free them here so the live-engine counts other tests assert on stay exact."""
    yield
    gc.collect()



@dataclass
class Cov:
    """Shots run through each command on both paths, and what the states and masks contained."""

    peek: int = 0  # (shot, target) peeks compared
    postselect: int = 0  # shots of a postselection mask
    unmatched: int = 0  # (qubit, shot) entries of an unmatched-reset mask
    kicks: int = 0
    pinned: int = 0  # peeks that snap to 0 / 1
    drawn: int = 0  # peeks strictly inside (a draw happens in an unmatched reset)
    d_positive: int = 0
    dead: int = 0
    raised: dict[str, int] = field(default_factory=dict)


def _state(rng: np.random.Generator, n: int, chi: int, kind: str) -> pq.ShotExpansion:
    if kind == "fresh":
        return pq.ShotExpansion.zero(n)
    if kind == "basis":
        a = np.zeros(pq.n_words_for_qubits(n), np.uint64)
        b = np.zeros_like(a)
        q = int(rng.integers(0, n))
        a[q >> 6] = np.uint64(1) << np.uint64(q & 63)
        return pq.apply_pauli(pq.ShotExpansion.zero(n), a, b, 0)
    return random_state(rng, n, chi, kind)


def _scramble(sim: MagicsideSimulator, rng: np.random.Generator, n: int, steps: int) -> None:
    """The same random Cliffords on the reference tableau of both simulators (not the shots)."""
    for _ in range(steps):
        g = str(rng.choice(["CX", "CZ", "H", "S", "SWAP", "SQRT_X"]))
        qs = rng.permutation(n)[: 2 if g in ("CX", "CZ", "SWAP") else 1].tolist()
        sim._live_ref_tab.do_unitary(g, qs)


def _make_pair(
    states: list[pq.ShotExpansion], n: int, seed: int, sync: bool, scramble_seed: int
) -> tuple[MagicsideSimulator, MagicsideSimulator, list[Any], list[Any]]:
    ref, eng, rc, ec = _pair(states, n, seed, sync, _ATOL)
    for sim in (ref, eng):
        _scramble(sim, np.random.default_rng(scramble_seed), n, 6 + n // 8)
    return ref, eng, rc, ec


def _run(sim: MagicsideSimulator, fn: str, *args: Any) -> Exception | None:
    try:
        with np.errstate(all="ignore"):
            getattr(sim, fn)(*args)
    except (FloatingPointError, ValueError, ZeroDivisionError, MemoryError, RuntimeError) as exc:
        return exc
    return None


def _same_exception(a: Exception | None, b: Exception | None, label: str) -> None:
    assert (type(a), str(a)) == (type(b), str(b)), (label, a, b)


def _masks(
    rng: np.random.Generator, n: int, batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(flip_mask, desired_bits, reset_zero_mask, reset_one_mask) with rows over all `n` qubits."""
    flip = rng.random(batch) < 0.6
    desired = rng.integers(0, 2, batch).astype(np.uint8)
    rz = rng.random((n, batch)) < 0.25
    ro = rng.random((n, batch)) < 0.25
    rz[rng.random(n) < 0.5] = False
    ro[rng.random(n) < 0.5] = False
    return flip, desired, rz, ro


def _cell(n: int, chi: int, seed: int, sync: bool, cov: Cov) -> None:
    rng = np.random.default_rng([seed, n, chi, 11, int(sync)])
    batch = int(rng.choice([1, 2, 3, 7, 12]))
    states = [_state(rng, n, chi, _KINDS[int(rng.integers(0, len(_KINDS)))]) for _ in range(batch)]
    ref, eng, rc, ec = _make_pair(states, n, int(rng.integers(0, 2**30)), sync, seed)
    assert not isinstance(ref._live_exps, LiveExpansions)
    assert isinstance(eng._live_exps, LiveExpansions)
    label = f"n={n} chi={chi} seed={seed} sync={sync}"
    pauli = str(rng.choice(["X", "Y", "Z"]))
    targets = rng.permutation(n)[: int(rng.integers(1, min(n, 4) + 1))].tolist()
    # --- peek: values first, then the state (nothing may change)
    want = _peek(ref, targets, pauli)
    got = _peek(eng, targets, pauli)
    _same_exception(want[1], got[1], label + " peek")
    if want[1] is None:
        assert want[0].tobytes() == got[0].tobytes(), label + " peek values"
        cov.peek += want[0].size
        pin = (want[0] <= 1e-12) | (want[0] >= 1.0 - 1e-12)
        cov.pinned += int(pin.sum())
        cov.drawn += int((~pin).sum())
    _assert_same(ref, eng, label + " after peek", rc, ec)
    flip, desired, rz, ro = _masks(rng, n, batch)
    q = int(rng.integers(0, n))
    # --- postselect, then unmatched reset, on the very same states
    for fn, args in (
        ("postselect_z_batch", (q, flip, desired, pauli)),
        ("do_unmatched_noisy_reset_z", (rz, ro, pauli, rng.permutation(n).tolist() * 1)),
    ):
        _count(fn, args, cov)
        a, b = _run(ref, fn, *args), _run(eng, fn, *args)
        _same_exception(a, b, f"{label} {fn}")
        if a is not None:
            cov.raised[type(a).__name__] = cov.raised.get(type(a).__name__, 0) + 1
        _assert_same(ref, eng, f"{label} after {fn}", rc, ec)
        cov.d_positive += sum(ref._live_exps[b].d > 0 for b in range(batch))
        cov.dead += sum(ref._live_exps[b].chi == 0 for b in range(batch))


def _peek(sim: MagicsideSimulator, targets: list[int], pauli: str) -> tuple[Any, Exception | None]:
    try:
        with np.errstate(all="ignore"):
            return sim.peek_pauli_born_batch(targets, pauli), None  # type: ignore[arg-type]
    except (FloatingPointError, ValueError, ZeroDivisionError, MemoryError, RuntimeError) as exc:
        return None, exc


def _count(fn: str, args: tuple[Any, ...], cov: Cov) -> None:
    if fn == "postselect_z_batch":
        cov.postselect += int(np.count_nonzero(args[1]))
    else:
        cov.unmatched += int(np.count_nonzero(args[0] | args[1]))
        cov.kicks += int(np.count_nonzero(args[1]))


def _run_all(sync: bool, seeds: range) -> Cov:
    cov = Cov()
    for n, chi in _CELLS:
        for seed in seeds:
            _cell(n, chi, seed, sync, cov)
    return cov


def _check(cov: Cov, minimum: int) -> None:
    """At least `minimum` cases of each command (a case = one shot / one mask entry)."""
    assert cov.peek >= minimum and cov.postselect >= minimum, cov
    assert cov.unmatched >= minimum and cov.kicks >= minimum // 2, cov
    assert cov.pinned >= 200 and cov.drawn >= 200, cov
    assert cov.d_positive >= 200 and cov.dead >= 50, cov


def test_engine_peek_postselect_and_unmatched_reset_match_the_reference_on_many_cases() -> None:
    cov = _run_all(False, range(90))
    _check(cov, 2000)
    assert sum(cov.raised.values()) >= 20, cov  # failure paths really are exercised


def test_engine_peek_postselect_and_unmatched_reset_match_in_sync_mode_on_many_cases() -> None:
    """Sync mode: the unmatched reset draws per shot from the shot's own streams."""
    cov = _run_all(True, range(60))
    _check(cov, 2000)


def test_the_engine_paths_really_run_on_an_engine_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: the differential cases above must reach the C++ commands, not the fallback."""
    n = 5
    states = [pq.ShotExpansion.zero(n) for _ in range(3)]
    _, eng, _, _ = _make_pair(states, n, 1, False, 3)
    seen: list[str] = []
    store = eng._live_exps
    assert isinstance(store, LiveExpansions)
    for name in ("peek", "postselect", "unmatched_apply"):
        real = getattr(store, name)
        monkeypatch.setattr(
            store, name, lambda *a, _r=real, _n=name, **k: (seen.append(_n), _r(*a, **k))[1]
        )
    eng.peek_pauli_born_batch([0, 1], "Z")
    eng.postselect_z_batch(0, np.array([1, 0, 1], bool), np.array([0, 0, 0], np.uint8))
    eng.do_unmatched_noisy_reset_z(
        np.ones((n, 3), bool), np.zeros((n, 3), bool), "Z", list(range(n))
    )
    assert {"peek", "postselect", "unmatched_apply"} <= set(seen), seen


def _ones_circ(n: int) -> stim.Circuit:
    return stim.Circuit(f"R {n - 1}\nI 0")


def test_sync_calls_of_the_sampler_agree_between_the_paths() -> None:
    """The `(shot, p1)` pairs the sync sampler is asked for are the same, in the same order."""
    n = 5
    rng = np.random.default_rng(8)
    states = [random_state(rng, n, 4, "d") for _ in range(6)]
    ref, eng, rc, ec = _make_pair(states, n, 5, True, 4)
    rz, ro = np.ones((n, 6), bool), np.zeros((n, 6), bool)
    for sim in (ref, eng):
        sim.do_unmatched_noisy_reset_z(rz, ro, "X", list(range(n)))
    assert rc == ec and len(rc) > 0
    _assert_same(ref, eng, "sync sampler calls", rc, ec)


def test_calls_helper_is_the_one_of_the_s5a_tests() -> None:
    sim = _pair([pq.ShotExpansion.zero(5)], 5, 1, True, _ATOL)[0]
    assert _calls(sim) == []
