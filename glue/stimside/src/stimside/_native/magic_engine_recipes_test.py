"""The C++ numpy recipes of the block command, against the installed numpy, bit for bit.

`mbe_cmul_form` exposes the three multiply forms the block command uses (A: `w * c` and in-place
`w *= c` for two or more elements; B: `np.complex128(c) * w`; plain: in-place `w *= c` on ONE
element). The expected values here are numpy's own results, never engine output, on data with
zeros of both signs. `numpy_recipes_drift_test.py` pins the numpy side; these pin that the C++
side copies it.
"""

from __future__ import annotations

import numpy as np
import pytest

import stimside.simulator_magic as sm
from stimside._native import magic_engine as me
from stimside._native.magic_kernels import w_addr


def _data(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    w[::4], w[1::7] = complex(-0.0, 0.0), complex(0.0, -0.0)
    w[2::9] = complex(0.0, 0.0)
    return w


_SCALARS = [
    0.5 + 0.25j, -0.7071067811865476 + 0.7071067811865476j, 1j, -1 + 0j, 0.25 + 0j, 1 + 0j,
    complex(0.0, -0.0), complex(-0.0, 0.0), 3.3 - 1.1j, 0.8535533905932737 + 0.35355339059327373j,
]


def _run(form: int, w: np.ndarray, c: complex) -> np.ndarray:
    lib = me.get_magic_engine_lib()
    mine = w.copy()
    lib.mbe_cmul_form(len(w), w_addr(mine), c.real, c.imag, form)
    return mine


@pytest.mark.parametrize("n", [2, 3, 9, 130])
@pytest.mark.parametrize("c", _SCALARS)
def test_form_0_is_numpy_array_times_python_complex_and_in_place_for_two_or_more(
    n: int, c: complex
) -> None:
    w = _data(1, n)
    want = w * c
    inplace = w.copy()
    inplace *= c
    got = _run(0, w, c)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
    assert np.array_equal(got.view(np.uint64), inplace.view(np.uint64))


@pytest.mark.parametrize("n", [1, 2, 9, 130])
@pytest.mark.parametrize("c", _SCALARS)
def test_form_1_is_numpy_complex128_scalar_times_array(n: int, c: complex) -> None:
    w = _data(2, n)
    want = np.complex128(c) * w
    assert np.array_equal(_run(1, w, c).view(np.uint64), want.view(np.uint64))


@pytest.mark.parametrize("c", _SCALARS)
def test_form_2_is_the_unfused_in_place_multiply_of_a_single_element(c: complex) -> None:
    for seed in range(40):
        w = _data(seed, 1)
        want = w.copy()
        want *= c
        assert np.array_equal(_run(2, w, c).view(np.uint64), want.view(np.uint64)), (seed, c)


def test_the_forms_really_differ_on_generic_data() -> None:
    """Not a vacuous pin: A, B and the plain form give different bits somewhere."""
    c = 0.8535533905932737 + 0.35355339059327373j
    w = _data(5, 200)
    a, b, p = _run(0, w, c), _run(1, w, c), _run(2, w, c)
    assert not np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert not np.array_equal(a.view(np.uint64), p.view(np.uint64))


def test_the_startup_guard_passes_on_this_numpy() -> None:
    me.numerics_match_numpy.cache_clear()
    assert me.numerics_match_numpy() is True


def _sequential_norm_sq(w: np.ndarray) -> np.float64:
    """A recipe that differs from numpy's pairwise sum: plain left-to-right addition."""
    acc = 0.0
    for x in w.real * w.real + w.imag * w.imag:
        acc += float(x)
    return np.float64(acc)


# Each entry replaces one numpy expression of the guard by a plausible different recipe, as
# another numpy build might compute it: a sequential sum; form B where A is expected and the
# other way round; the fused product where numpy computes a single-element `*=` unfused.
_FAKE_RECIPES = {
    "norm_sq": ("_numpy_norm_sq", _sequential_norm_sq),
    "form_a": ("_numpy_form_a", lambda w, c: np.complex128(c) * w),
    "form_b": ("_numpy_form_b", lambda w, c: w * c),
    "in_place_one": ("_numpy_in_place", lambda one, c: one * c),
}


@pytest.mark.parametrize("which", sorted(_FAKE_RECIPES))
def test_a_numpy_recipe_that_differs_from_the_engines_flips_the_guard_to_false(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard compares the engine with each numpy expression (norm_sq, form A, form B and
    the single-element in-place multiply): a numpy build that rounded any ONE of them
    differently must make it False, so the engine is not used (a guard that checked only
    form A, as before, stays True here for form B and the in-place quirk)."""
    name, fake = _FAKE_RECIPES[which]
    me.numerics_match_numpy.cache_clear()
    monkeypatch.setattr(me, name, fake)
    try:
        assert me.numerics_match_numpy() is False, which
    finally:
        monkeypatch.undo()
        me.numerics_match_numpy.cache_clear()
    assert me.numerics_match_numpy() is True  # restored: the real recipes still match


@pytest.mark.parametrize("which", ["form_b", "in_place_one"])
def test_the_simulator_keeps_the_list_store_when_the_numerics_guard_is_false(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a drifting form B / in-place recipe makes `_fresh_live_exps` return the
    plain list although the engine switch is on."""
    import stim  # type: ignore[import-untyped]

    name, fake = _FAKE_RECIPES[which]
    circ = stim.Circuit("R 0 1\nH 0\nI[T] 0\nM 0 1")
    me.numerics_match_numpy.cache_clear()
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    try:
        sim = sm.MagicsideSimulator(circ, batch_size=4, seed=1)
        assert not isinstance(sim._live_exps, list)  # sanity: the engine really is on
        monkeypatch.setattr(me, name, fake)
        me.numerics_match_numpy.cache_clear()
        sim2 = sm.MagicsideSimulator(circ, batch_size=4, seed=1)
        assert isinstance(sim2._live_exps, list), which
        sim2.run()
    finally:
        monkeypatch.undo()
        me.numerics_match_numpy.cache_clear()
