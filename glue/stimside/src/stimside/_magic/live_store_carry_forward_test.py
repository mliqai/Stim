"""Carry-forward fixes from the S4b-1 gate (S4b-2 worker, PART 1 items b and c).

(b) `well_formed_lists` is total and conservative: any factor list that is not exactly an int
    `v` and declared Pauli-axis objects with `uint64[(n_words,)]` words goes to the per-shot
    reference function WITHOUT raising, so the engine path and the reference path give the same
    exception type, text and state for every one of them.
(c) shot_list validation (wide integers, entries < -1, entries == n_lists), `failed_shot` reset,
    `_as` layout, int64 words, and the Pauli rows being private copies.

The oracle for every differential is today's per-shot `apply_clifford_factors`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import CompiledCliffordFactorList as CL
from stimside._magic.channels import apply_clifford_factors
from stimside._magic.gates import CompiledPauliAxis as AX
from stimside._magic.live_store import LiveExpansions, factor_table, well_formed_lists
from stimside._magic.live_store_test import (
    _first_difference,
    _nonzero_axis,
    random_state,
)
from stimside._native import magic_engine as me

_ATOL = 1e-10
_EXPECTED_ERRORS = (
    ValueError, TypeError, AttributeError, OverflowError, FloatingPointError, RuntimeError,
    IndexError, KeyError,
)


class _IndexOnly:
    """Has `__index__` but is not an int: `v & 7` is a TypeError for the reference."""

    def __index__(self) -> int:
        return 3


class _NoKappa:
    """An axis-like object with word arrays but no `kappa`."""

    def __init__(self, ax: AX) -> None:
        self.a_words, self.b_words = ax.a_words, ax.b_words


def _ax(rng: np.random.Generator, n: int) -> AX:
    return AX(*_nonzero_axis(rng, n))


def _reference(states, lists, which, inverse):
    out = [s.copy() for s in states]
    for s, w in enumerate(which):
        if w < 0:
            continue
        try:
            out[s] = apply_clifford_factors(
                out[s], lists[int(w)], inverse=inverse, check_unitary=True, atol=_ATOL
            )
        except _EXPECTED_ERRORS as exc:  # the comparison is on the exception itself
            return (type(exc).__name__, str(exc)), out
    return None, out


def _engine(states, lists, which, inverse, n):
    store = LiveExpansions(len(states), n, _ATOL)
    for i, s in enumerate(states):
        store[i] = s.copy()
    try:
        store.inject_factor_lists(lists, which, inverse)
        res = None
    except _EXPECTED_ERRORS as exc:
        res = (type(exc).__name__, str(exc))
    out = [store[i] for i in range(len(states))]
    store.close()
    return res, out


def _hostile_cases(rng: np.random.Generator, n: int) -> dict[str, list[CL]]:
    ax = _ax(rng, n)
    return {
        "tuple_p0": [CL(1, ax.as_tuple(), ())],  # type: ignore[arg-type]
        "tuple_r_axis": [CL(1, None, (ax.as_tuple(),))],  # type: ignore[arg-type]
        "bool_p0": [CL(1, False, ())],  # type: ignore[arg-type]
        "dict_axis": [CL(1, None, ({"a_words": 1},))],  # type: ignore[arg-type]
        "r_axes_none": [CL(1, None, None)],  # type: ignore[arg-type]
        "index_only_v": [CL(_IndexOnly(), None, (ax,))],  # type: ignore[arg-type]
        "bool_v": [CL(True, None, (ax,))],
        "float_v": [CL(3.0, None, (ax,))],  # type: ignore[arg-type]
        "string_v": [CL("3", None, (ax,))],  # type: ignore[arg-type]
        "missing_kappa": [CL(1, None, (_NoKappa(ax),))],  # type: ignore[arg-type]
        "int64_words": [CL(1, None, (AX(ax.a_words.astype(np.int64), ax.b_words, ax.kappa),))],
        "list_words": [
            CL(1, None, (AX(list(ax.a_words), ax.b_words, ax.kappa),))  # type: ignore[arg-type]
        ],
        "none_axis": [CL(1, None, (None,))],  # type: ignore[arg-type]
    }


@pytest.mark.parametrize("n", [5, 70, 130])
@pytest.mark.parametrize("inverse", [False, True])
def test_every_hostile_factor_list_class_behaves_exactly_as_the_reference(
    n: int, inverse: bool
) -> None:
    rng = np.random.default_rng([n, 21])
    nw = pq.n_words_for_qubits(n)
    for name, lists in _hostile_cases(rng, n).items():
        assert well_formed_lists(lists, nw) is False, name  # total: False, never an exception
        for pattern in ([0, 0, -1], [-1, 0, 0], [-1, -1, -1]):
            states = [random_state(rng, n, 3, "plain") for _ in range(3)]
            which = np.array(pattern, dtype=np.int32)
            ref_exc, ref_out = _reference(states, lists, which, inverse)
            eng_exc, eng_out = _engine(states, lists, which, inverse, n)
            assert eng_exc == ref_exc, (name, pattern, eng_exc, ref_exc)
            for i, (x, y) in enumerate(zip(ref_out, eng_out, strict=True)):
                assert _first_difference(x, y) is None, (name, pattern, i)


def test_a_tuple_axis_is_applied_by_the_engine_path_as_the_reference_applies_it() -> None:
    """The reference ACCEPTS `(a, b, kappa)` tuples: the state changes, no exception."""
    n, rng = 70, np.random.default_rng(3)
    ax = _ax(rng, n)
    lists = [CL(1, ax.as_tuple(), (ax.as_tuple(),))]  # type: ignore[arg-type]
    states = [random_state(rng, n, 3, "plain") for _ in range(2)]
    which = np.array([0, 0], dtype=np.int32)
    ref_exc, ref_out = _reference(states, lists, which, False)
    eng_exc, eng_out = _engine(states, lists, which, False, n)
    assert ref_exc is None and eng_exc is None
    assert any(_first_difference(a, b) is not None for a, b in zip(states, ref_out, strict=True))
    for x, y in zip(ref_out, eng_out, strict=True):
        assert _first_difference(x, y) is None


def test_well_formed_lists_accepts_what_the_compiler_produces_and_only_that() -> None:
    n, rng = 70, np.random.default_rng(4)
    nw = pq.n_words_for_qubits(n)
    ax = _ax(rng, n)
    good = [CL(3, ax, (ax, ax)), CL(0, None, ()), CL(np.int64(5), None, (ax,))]
    assert well_formed_lists(good, nw)
    assert well_formed_lists([], nw)
    assert not well_formed_lists([CL(1, ax, ())], nw + 1)  # wrong word count
    assert not well_formed_lists([CL(np.bool_(True), None, ())], nw)  # type: ignore[arg-type]


def _store(states: list[pq.ShotExpansion], n: int) -> LiveExpansions:
    store = LiveExpansions(len(states), n, _ATOL)
    for i, s in enumerate(states):
        store[i] = s
    return store


@pytest.mark.parametrize("big", [2**31, 2**32, 2**32 + 1, 2**40, 2**63 - 1])
def test_a_wide_shot_list_value_is_refused_not_wrapped_into_a_valid_list_index(big: int) -> None:
    """int64 2**32 used to wrap to list 0 and silently change the state."""
    n, rng = 5, np.random.default_rng(5)
    states = [random_state(rng, n, 3, "plain") for _ in range(3)]
    store = _store(states, n)
    fac = CL(1, None, (_ax(rng, n),))
    with pytest.raises(RuntimeError, match="failed before running"):
        store.inject_factor_lists([fac], np.array([0, big, -1], dtype=np.int64), False)
    for i in range(3):
        assert _first_difference(states[i], store[i]) is None
    store.close()


def test_a_shot_list_value_below_minus_one_skips_the_shot_as_the_reference_does() -> None:
    n, rng = 70, np.random.default_rng(6)
    states = [random_state(rng, n, 3, "plain") for _ in range(4)]
    fac = CL(3, _ax(rng, n), (_ax(rng, n),))
    which = np.array([0, -2, -(2**40), 0], dtype=np.int64)
    ref_exc, ref_out = _reference(states, [fac], which, False)
    eng_exc, eng_out = _engine(states, [fac], which, False, n)
    assert ref_exc is None and eng_exc is None
    assert _first_difference(states[1], eng_out[1]) is None  # skipped, not applied
    assert _first_difference(states[0], eng_out[0]) is not None  # applied
    for x, y in zip(ref_out, eng_out, strict=True):
        assert _first_difference(x, y) is None


def test_a_shot_list_value_equal_to_n_lists_is_refused_with_state_untouched() -> None:
    """The boundary: list indices are 0..n_lists-1 (an off-by-one let `n_lists` through)."""
    n, rng = 5, np.random.default_rng(7)
    states = [random_state(rng, n, 3, "plain") for _ in range(3)]
    fac = CL(1, None, (_ax(rng, n),))
    for bad in (1, 2):  # n_lists == 1 and == 2
        store = _store(states, n)
        lists = [fac] * bad
        with pytest.raises(RuntimeError, match="failed before running"):
            store.inject_factor_lists(lists, np.array([0, bad, 0], dtype=np.int32), False)
        for i in range(3):
            assert _first_difference(states[i], store[i]) is None
        store.close()
    store = _store(states, n)  # the largest legal index is accepted
    store.inject_factor_lists([fac, fac], np.array([1, -1, 1], dtype=np.int32), False)
    store.close()


def test_failed_shot_is_reset_at_the_start_of_every_call() -> None:
    n, rng = 5, np.random.default_rng(8)
    states = [random_state(rng, n, 3, "plain") for _ in range(3)]
    store = _store(states, n)
    a, b, _ = _nonzero_axis(rng, n)
    bad_axis = AX(a, b, (int(pq.word_dot_parity(a, b)) + 1) & 3)  # not Hermitian
    with pytest.raises(ValueError, match="Hermitian"):
        store.inject_factor_lists([CL(0, None, (bad_axis,))], np.array([-1, 0, 0], np.int32), False)
    assert store.failed_shot == 1
    store.inject_factor_lists([CL(0, None, ())], np.array([0, 0, 0], np.int32), False)
    assert store.failed_shot is None  # a successful call forgets the old failure
    with pytest.raises(ValueError, match="Hermitian"):
        store.inject_factor_lists([CL(0, None, (bad_axis,))], np.array([-1, 0, 0], np.int32), False)
    assert store.failed_shot == 1
    store.inject_factor_lists([CL(0, None, ())], np.array([-1, -1, -1], np.int32), False)
    assert store.failed_shot is None
    store.close()


def test_the_private_copy_helper_always_yields_a_c_contiguous_aligned_array() -> None:
    """`order="K"` kept a Fortran layout, which the C++ would read as flat memory."""
    base = np.asfortranarray(np.arange(12, dtype=np.uint64).reshape(3, 4))
    assert not base.flags.c_contiguous
    out = me._as(base, np.uint64, (3, 4), "x")
    assert out.flags.c_contiguous and out.flags.aligned and out is not base
    assert out.tolist() == base.tolist()
    t = np.arange(24, dtype=np.int64).reshape(4, 6)[:, ::2].T  # strided and transposed
    out2 = me._as(t, np.int32, t.shape, "t")
    assert out2.flags.c_contiguous and out2.tolist() == t.tolist()


def test_int64_words_take_the_reference_path_and_raise_what_it_raises() -> None:
    n, rng = 70, np.random.default_rng(9)
    ax = _ax(rng, n)
    bad = CL(1, None, (AX(ax.a_words.astype(np.int64), ax.b_words.astype(np.int64), ax.kappa),))
    assert not well_formed_lists([bad], 2)
    with pytest.raises(ValueError, match="Hermitian|padding|shape|Expected"):
        factor_table([bad], 2)  # the table builder refuses what the encoder cannot hold
    states = [random_state(rng, n, 3, "plain") for _ in range(2)]
    which = np.array([0, 0], dtype=np.int32)
    ref_exc, ref_out = _reference(states, [bad], which, False)
    eng_exc, eng_out = _engine(states, [bad], which, False, n)
    assert eng_exc == ref_exc
    for x, y in zip(ref_out, eng_out, strict=True):
        assert _first_difference(x, y) is None


def test_the_native_call_gets_private_copies_of_the_pauli_words_not_the_callers_arrays() -> None:
    """R15 at the assertion level: the address handed to the native call is never the address
    of an array the caller (or a racing thread) can still write."""
    n, rng = 70, np.random.default_rng(10)
    store = _store([random_state(rng, n, 3, "plain") for _ in range(2)], n)
    a, b, kap = _nonzero_axis(rng, n)
    mask = np.array([True, True])
    seen: list[tuple[str, tuple[Any, ...]]] = []
    eng = store.engine
    inner = eng._lib

    class Spy:
        def __getattr__(self, name: str) -> Any:
            fn = getattr(inner, name)
            if name != "mbe_apply_pauli_mask":
                return fn

            def call(*args: Any) -> Any:
                seen.append((name, args))
                return fn(*args)

            return call

    eng._lib = Spy()  # type: ignore[assignment]
    eng.apply_pauli_mask(a, b, kap, mask)
    assert len(seen) == 1
    args = seen[0][1]
    caller_addrs = {me.addr(a), me.addr(b), me.addr(mask.view(np.uint8))}
    assert not caller_addrs & set(args[1:3] + args[4:5]), "native call got a caller-owned buffer"
    store.close()


def test_the_raw_abi_also_skips_shot_list_entries_below_minus_one() -> None:
    """The wrapper clips negatives to -1, so only the raw call can show the native rule: any
    negative entry means "skip the shot" (as the reference's `which >= 0`), not an error."""
    n, rng = 70, np.random.default_rng(11)
    states = [random_state(rng, n, 3, "plain") for _ in range(3)]
    store = _store(states, n)
    eng = store.engine
    fac = CL(3, _ax(rng, n), (_ax(rng, n),))
    t = factor_table([fac], eng.n_words)
    keep = [np.ascontiguousarray(x) for x in (*t[:5], t.axes_a, t.axes_b, t.axes_kappa)]
    shots = np.array([0, -2, 0], dtype=np.int32)
    err, errd = np.zeros(4, np.int64), np.zeros(2)
    rc = eng._lib.mbe_inject_cliffords(
        eng._h(), 1, int(t.axes_kappa.shape[0]), *(me.addr(x) for x in keep), me.addr(shots), 0,
        me.addr(err), me.addr(errd),
    )
    assert rc == -1, rc  # ran to completion: -2 was a skip
    assert _first_difference(states[1], store[1]) is None
    assert _first_difference(states[0], store[0]) is not None
    store.close()


@pytest.mark.parametrize("chi", [2**63, 2**63 + 2**62, 2**64 - 1])
def test_a_raw_import_whose_size_products_wrap_to_zero_is_refused_not_stored(chi: int) -> None:
    """`chi * 2`, `chi * n_words` and `chi * d` all wrap to 0 for chi = 2**63: without the
    explicit size check the import would "succeed" and store an impossible chi."""
    n, rng = 70, np.random.default_rng(12)
    states = [random_state(rng, n, 3, "plain")]
    store = _store(states, n)
    eng = store.engine
    buf = np.zeros(8, np.uint64)
    p = me.addr(buf)
    rc = eng._lib.mbe_import_shot(eng._h(), 0, chi, 0, 1, 0.0, 1e-10, *[p] * 8)
    assert rc == 1, rc  # invalid arguments
    assert _first_difference(states[0], store[0]) is None  # nothing was changed
    store.close()
