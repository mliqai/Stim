"""Pins the numpy arithmetic recipes the engine copies, against the installed numpy.

The engine reproduces numpy bit for bit, so it depends on how *this* numpy build rounds. The
three recipes (found by experiment, see the S4b-1 handoff):

* form A, `array * python_complex` (the omega multiply of `apply_clifford_factors`):
  re = fma(a, cr, -(b*ci)), im = fma(a, ci, b*cr);
* form B, `np.complex128 * array` (`coeffs[m] * weights` of `_branch_pauli_sum_shot`, S4b-2):
  re = fma(a, cr, -(b*ci)), im = fma(b, cr, a*ci);
* in-place `w *= c` on a *single* element is plain unfused (a*cr - b*ci, a*ci + b*cr), while for
  two or more elements it is form A.

If numpy changes any of them these tests fail first, with the recipe named, instead of the
engine silently differing from the reference. The expected values come from exact rational
arithmetic (`fractions`), never from the engine.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest


def _fma(a: float, b: float, c: float) -> float:
    """Correctly rounded a*b + c, signed zeros as IEEE-754 fusedMultiplyAdd."""
    if not all(np.isfinite(x) for x in (a, b, c)):
        return a * b + c
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        return a * b + c if (a * b == 0 and c == 0) else 0.0
    return float(exact)


def _form_a(a: float, b: float, cr: float, ci: float) -> tuple[float, float]:
    return _fma(a, cr, -(b * ci)), _fma(a, ci, b * cr)


def _form_b(a: float, b: float, cr: float, ci: float) -> tuple[float, float]:
    return _fma(a, cr, -(b * ci)), _fma(b, cr, a * ci)


def _plain(a: float, b: float, cr: float, ci: float) -> tuple[float, float]:
    return a * cr - b * ci, a * ci + b * cr


def _bits(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


def _same(got: complex, want: tuple[float, float]) -> bool:
    return _bits(got.real) == _bits(want[0]) and _bits(got.imag) == _bits(want[1])


_SPECIAL = [0.0, -0.0, 1.0, -1.0, 0.5, -0.7071067811865476, 3.0]


def _cases(seed: int, count: int) -> list[tuple[float, float, complex]]:
    rng = np.random.default_rng(seed)
    out = [
        (a, b, complex(cr, ci))
        for a in _SPECIAL[:5] for b in _SPECIAL[:5] for cr in _SPECIAL for ci in _SPECIAL
    ]
    out += [(float(x), float(y), complex(*rng.standard_normal(2))) for x, y in
            rng.standard_normal((count, 2))]
    return out


@pytest.mark.parametrize("n", [2, 3, 9, 130])
def test_array_times_python_complex_is_the_fma_form_a(n: int) -> None:
    for a, b, c in _cases(1, 300):
        w = np.array([complex(a, b)] * n)
        out = w * c
        assert all(_same(out[i], _form_a(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)


@pytest.mark.parametrize("n", [1, 2, 9, 130])
def test_numpy_scalar_times_array_is_the_fma_form_b(n: int) -> None:
    for a, b, c in _cases(2, 300):
        w = np.array([complex(a, b)] * n)
        out = np.complex128(c) * w
        assert all(_same(out[i], _form_b(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)


def test_a_one_element_array_times_python_complex_is_still_form_a() -> None:
    for a, b, c in _cases(3, 300):
        out = np.array([complex(a, b)]) * c
        assert _same(out[0], _form_a(a, b, c.real, c.imag)), (a, b, c)


@pytest.mark.parametrize("n", [2, 3, 9, 130])
def test_in_place_multiply_of_two_or_more_elements_is_form_a(n: int) -> None:
    for a, b, c in _cases(4, 200):
        w = np.array([complex(a, b)] * n)
        w *= c
        assert all(_same(w[i], _form_a(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)


def test_in_place_multiply_of_a_single_element_is_plain_unfused() -> None:
    """The quirk the block path must copy: `out.weights *= g_sc` with chi == 1."""
    rng = np.random.default_rng(5)
    fused_differs = 0
    for _ in range(400):
        a, b = (float(x) for x in rng.standard_normal(2))
        c = complex(*rng.standard_normal(2))
        w = np.array([complex(a, b)])
        w *= c
        assert _same(w[0], _plain(a, b, c.real, c.imag)), (a, b, c)
        fused_differs += not _same(w[0], _form_a(a, b, c.real, c.imag))
    assert fused_differs > 50  # the two forms really differ on generic data: not a vacuous pin


def test_array_times_phase_table_is_exact_for_every_form() -> None:
    """`weights * PHASE4[...]` (array x array of 0, +-1, +-i) has one result whatever the form."""
    phases = np.array([1 + 0j, 1j, -1 + 0j, -1j])
    for a, b, _ in _cases(6, 200):
        for q in range(4):
            out = np.array([complex(a, b)] * 3) * phases[np.array([q] * 3)]
            p = phases[q]
            for form in (_form_a, _form_b, _plain):
                assert _same(out[0], form(a, b, p.real, p.imag)), (a, b, q, form.__name__)


@pytest.mark.parametrize("n", [1, 2, 9, 130])
def test_python_complex_times_array_is_form_b_and_not_the_form_a_of_array_times_it(
    n: int,
) -> None:
    """Operand order matters: `c * w` (python complex on the left) is form B, `w * c` is form A."""
    differs = 0
    for a, b, c in _cases(7, 300):
        w = np.array([complex(a, b)] * n)
        left, right = c * w, w * c
        assert all(_same(left[i], _form_b(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)
        assert all(_same(right[i], _form_a(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)
        differs += not _same(left[0], _form_a(a, b, c.real, c.imag))
    assert differs > 50  # the two forms really differ on this data: not a vacuous pin


@pytest.mark.parametrize("n", [1, 2, 9, 130])
def test_array_times_numpy_scalar_is_form_a_and_numpy_scalar_times_array_is_form_b(
    n: int,
) -> None:
    for a, b, c in _cases(8, 200):
        w, s = np.array([complex(a, b)] * n), np.complex128(c)
        right, left = w * s, s * w
        assert all(_same(right[i], _form_a(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)
        assert all(_same(left[i], _form_b(a, b, c.real, c.imag)) for i in range(n)), (a, b, c)
