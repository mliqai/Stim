"""Sensitivity tests added after the WP-25 S5a gate (iteration 1): engine measurement vs reference.

The differential suites of S5a missed four behaviours of `mbe_measure_split` / `mbe_measure_finish`
that a mutant could change without any test failing. Each test here pins one, by running the same
substep on the REFERENCE (a plain list of `ShotExpansion`s: `measure_expansions_batch`) and on the
ENGINE, and comparing bytes (state, masses, random streams, exception type and text):

* M09: a non-canonical shot is merged BEFORE the rotation (the merge sums duplicate rows and the
  finalize threshold then drops a ~1e-13 row; unmerged rows survive it);
* F04: the un-rotation of the finish is unitarity-checked (a window where the split's checks pass
  and the un-rotation's pairwise check fails: the reference raises `FloatingPointError`);
* R06: the rotation check of the split accepts a deviation of exactly `atol * max(1, z)`;
* R02: an invalid observable is not an error when no shot is active (the engine validates after
  classifying, as the reference does).

Nothing is hardcoded: the expected result is whatever the reference path does.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import MeasurementRecord
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store_measure_diff_test import _assert_same, _pair, _step
from stimside._magic.live_store_test import _nonzero_axis, random_state

_ATOL = 1e-10


def _same_step(
    states: list[pq.ShotExpansion], n: int, rec: Any, seed: int, atol: float = _ATOL
) -> Exception | None:
    """Run `rec` on the reference and on the engine; assert identical results, exceptions and
    state; return the exception (None when the substep ran)."""
    ref, eng, rc, ec = _pair([s.copy() for s in states], n, seed, False, atol)
    want, wexc = _step(ref, rec)
    got, gexc = _step(eng, rec)
    assert (type(gexc), str(gexc)) == (type(wexc), str(wexc))
    if wexc is None:
        assert want.tobytes() == got.tobytes()
    _assert_same(ref, eng, "gate", rc, ec)
    return wexc


def _kept_frame(ax: CompiledPauliAxis) -> MeasurementRecord:
    return MeasurementRecord(0, 0.0, False, ax, ax)


def _raw_state(n: int, weights: Any, h: Any, atol: float = _ATOL) -> pq.ShotExpansion:
    """A hand-built d = 0 expansion that is not canonical (what the merge-free ops leave)."""
    nw = pq.n_words_for_qubits(n)
    w = np.asarray(weights, np.complex128)
    return pq.ShotExpansion(
        n, 0, np.zeros((0, nw), np.uint64), np.zeros(nw, np.uint64), np.zeros((0, 0), np.uint8),
        np.zeros(0, np.int32), w, np.asarray(h, np.uint64).reshape(len(w), nw),
        np.zeros((len(w), 0), np.uint8), None, 0.0, atol, False,
    )


def _words(a: int, b: int, kappa: int = 0) -> CompiledPauliAxis:
    return CompiledPauliAxis(np.array([a], np.uint64), np.array([b], np.uint64), kappa)


# ---------------------------------------------------------------------------------------------
# M09
# ---------------------------------------------------------------------------------------------


def test_m09_a_noncanonical_shot_is_merged_before_the_rotation() -> None:
    """Rows 0 and 1 are the same component (h = 2); merged they weigh 1.2, and the ~1e-13 row 2
    is then below the finalize threshold `1e-13 * min(1, w_max)` and dropped. Rotated unmerged it
    would stay. The rotated, kept-frame state must be the reference's, bit for bit."""
    w = [0.6, 0.6, 1e-13, 0.10355339059327379 - 0.10355339059327379j]
    st = _raw_state(4, w, [2, 2, 7, 15])
    ref, eng, rc, ec = _pair([st.copy()], 4, 1, False)
    rec = _kept_frame(_words(8, 0))
    assert _step(ref, rec)[1] is None and _step(eng, rec)[1] is None
    _assert_same(ref, eng, "m09 witness", rc, ec)
    assert ref._live_exps[0].chi < st.chi  # the merge really dropped rows (the case is live)


def _duplicated(rng: np.random.Generator, n: int, mode: int) -> pq.ShotExpansion:
    """A non-canonical shot whose duplicate (h, ell) rows are NOT adjacent, with random,
    near-cancelling (mode 1) or scaled-down (mode 2) weights."""
    base = random_state(rng, n, int(rng.integers(3, 14)), str(rng.choice(["plain", "d", "raw"])))
    src = rng.integers(0, base.chi, size=int(rng.integers(1, 4)))
    w = np.concatenate([base.weights, base.weights[src]])
    h = np.vstack([base.h_words, base.h_words[src]])
    ell = np.vstack([base.ell, base.ell[src]]) if base.d > 0 else np.zeros((len(w), 0), np.uint8)
    perm = rng.permutation(len(w))
    w, h, ell = w[perm], h[perm], ell[perm]
    r = rng.standard_normal(len(w)) + 1j * rng.standard_normal(len(w))
    if mode == 1:
        for j, s in enumerate(src):
            eps = 10.0 ** rng.uniform(-16, -11)
            copy_at = int(np.where(perm == base.chi + j)[0][0])
            r[copy_at] = -r[int(np.where(perm == s)[0][0])] * (1 + eps)
    if mode == 2:
        r *= 10.0 ** rng.uniform(-6, 0)
    r = r / np.sqrt(np.sum(np.abs(r) ** 2))
    return dataclasses.replace(
        base, weights=r.astype(np.complex128), h_words=h, ell=ell, _is_canonical_merged=False
    )


def test_m09_duplicate_rows_that_are_not_adjacent_rotate_like_the_reference() -> None:
    cases = d_positive = 0
    for seed in range(360):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([5, 9, 70]))
        st = _duplicated(rng, n, seed % 3)
        d_positive += st.d > 0
        for _ in range(3):
            a, b, kap = _nonzero_axis(rng, n, density=float(rng.choice([0.1, 0.4, 0.8])))
            if not a.any():
                continue
            _same_step([st, st], n, _kept_frame(CompiledPauliAxis(a, b, kap)), seed)
            cases += 1
    assert cases > 600 and d_positive > 60  # the loop really exercised d > 0 geometry


# ---------------------------------------------------------------------------------------------
# F04
# ---------------------------------------------------------------------------------------------

# 21 components of a d = 0, n = 4 shot, mostly 1e-15 .. 1e-20 rows (found by a search over random
# shots, then shrunk; kept literal). A Pauli X on qubit 0 measured with the rotated frame NOT kept
# (`rot_axis=None` with a != 0: no compiled record has it, so the finish un-rotates) at
# atol <= 1e-19 gives, for outcome 1, a split that passes its own checks and an un-rotation whose
# pairwise norm check fails.
_F04_W = [
    -0.002082644851215789 - 0.0015428571472494413j, 0.005524819132659693 - 0.006588594086367923j,
    7.041700146494039e-17 - 1.4273280414765948e-15j, -0.005052208261172612 + 0.004974916419311562j,
    -0.010734718199712985 + 0.0065720053276777735j, 0.026802734360258325 - 0.014174436397462969j,
    -0.009812316804035335 - 0.005261123909525349j, -4.21839659699175e-17 - 2.2364284015977683e-17j,
    -0.018527687767889992 - 0.007126979892856666j, 2.8058775327813435e-15 + 1.5302305618042871e-15j,
    3.615204988586126e-16 + 3.068977038340046e-14j, 8.972490127168209e-16 - 2.9500987226019218e-15j,
    1.2770019778587937e-18 - 2.163774946973581e-16j,
    -0.0016168103007765538 - 0.0019139525330355495j,
    -0.010760875405563365 + 0.013135203136581326j, -0.009391461066176176 - 0.005316192142556048j,
    5.2526937210695295e-14 + 5.183228849638512e-14j, -0.012688999034152009 - 0.005455895384134481j,
    -1.3061913166040636e-16 + 4.539704587342867e-16j,
    -2.4369557106066258e-15 - 1.7122696340009063e-14j, 0.009675215397912393 + 0.008728647673930811j,
]
_F04_H = [0, 0, 5, 2, 12, 3, 10, 4, 6, 15, 4, 10, 3, 1, 6, 8, 9, 0, 12, 2, 5]


@pytest.mark.parametrize("atol", [1e-19, 1e-300])
def test_f04_the_unrotation_of_a_finish_is_unitarity_checked(atol: float) -> None:
    st = _raw_state(4, _F04_W, _F04_H, atol)
    rec = MeasurementRecord(0, 0.0, False, _words(1, 0), None)
    raised = 0
    for seed in range(24):  # the random draw picks the outcome; about half pick outcome 1
        wexc = _same_step([st], 4, rec, seed, atol)
        if isinstance(wexc, FloatingPointError):
            assert "unitary norm conservation" in str(wexc)
            raised += 1
    assert raised >= 2  # the window is really reached: the reference raises, so the engine must


# ---------------------------------------------------------------------------------------------
# R06
# ---------------------------------------------------------------------------------------------


def test_r06_the_rotation_check_accepts_a_deviation_of_exactly_atol_times_the_norm() -> None:
    """The split rejects the rotation when `|post - z| > atol * max(1, z)`: with `atol` set to the
    deviation itself it must pass, one ulp lower it must fail (`<=`, not `<`), as in the
    reference. `post` is the sequential sum of the rotated weights, as the kernel computes it."""
    exact_ok = below_raised = cases = 0
    for seed in range(300):
        rng = np.random.default_rng([seed, 606])
        n = int(rng.integers(3, 9))
        st = random_state(rng, n, int(rng.choice([2, 4, 8, 16])), str(rng.choice(["plain", "d"])))
        st.weights = st.weights * complex(10.0 ** rng.uniform(-2, 2), rng.uniform(-1, 1))
        a, b, kap = _nonzero_axis(rng, n, density=0.5)
        if not a.any():
            continue
        z = float(st.norm_sq())
        unit = b ^ pq.unit_bit_words(pq.first_set_bit_in_words(a), pq.n_words_for_qubits(n))
        rot = pq.apply_quarter_turn(
            st, a, unit, (kap + 1) & 3, sign=1, check_unitary=False, atol=1.0
        )
        post = 0.0
        for x in rot.weights:
            post += float(x.real) * float(x.real) + float(x.imag) * float(x.imag)
        dlt = abs(post - z) / max(1.0, z)
        if not 0.0 < dlt < 1.0:
            continue
        rec = _kept_frame(CompiledPauliAxis(a, b, kap))
        exact_ok += _same_step([st, st], n, rec, 1, dlt) is None
        below_raised += _same_step([st, st], n, rec, 1, float(np.nextafter(dlt, 0))) is not None
        cases += 1
    assert cases > 50 and exact_ok > 10 and below_raised > 10


# ---------------------------------------------------------------------------------------------
# R02
# ---------------------------------------------------------------------------------------------


def _bad_observable(bad: str) -> CompiledPauliAxis:
    if bad == "non_hermitian":  # Z with kappa = 1: parity 0, odd kappa
        return _words(0, 1, 1)
    return _words(0, 1 << 60)  # padding bits above qubit 4 (n = 5)


@pytest.mark.parametrize("kind", ["trivial", "dead", "dead+trivial"])
@pytest.mark.parametrize("bad", ["non_hermitian", "padding_bits"])
def test_r02_an_invalid_observable_is_not_an_error_when_no_shot_is_active(
    kind: str, bad: str
) -> None:
    n = 5
    dead = pq.ShotExpansion.from_components((), n_qubits=n, merge=False)
    zero = pq.ShotExpansion.zero(n)
    states = {"trivial": [zero, zero], "dead": [dead, dead], "dead+trivial": [dead, zero]}[kind]
    rec = MeasurementRecord(0, 0.0, False, _bad_observable(bad), None)
    assert _same_step(states, n, rec, 1) is None  # the reference accepts it (nothing is active)


@pytest.mark.parametrize("bad", ["non_hermitian", "padding_bits"])
def test_r02_the_same_invalid_observable_is_refused_when_a_shot_is_active(bad: str) -> None:
    n = 5
    st = random_state(np.random.default_rng(3), n, 4, "d")
    rec = MeasurementRecord(0, 0.0, False, _bad_observable(bad), None)
    assert _same_step([st, st], n, rec, 1) is not None  # both refuse, with the same error
