"""Bit-exact differential tests of the engine-backed shot store against the per-shot functions.

Oracle: today's `phase_qfe.apply_pauli` applied shot by shot (the loop `_apply_pauli_mask`
runs). Subject: `LiveExpansions` (C++ `MagicBatchEngine` behind a list-like proxy). Everything is
compared as bytes (weights through a uint64 view, so -0.0 != 0.0).
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
import stimside._native.magic_engine as me
from stimside._magic.live_store import (
    LiveExpansions,
    arrays_to_expansion,
    engine_numerics_available,
    expansion_to_arrays,
)

_FIELDS = ("A_words", "a_union", "Gamma", "pivots", "h_words", "ell", "ell_odd")


def _first_difference(x: pq.ShotExpansion, y: pq.ShotExpansion) -> str | None:
    """Name of the first field in which two expansions differ at the byte level, else None."""
    if (x.n_qubits, x.d, x.chi) != (y.n_qubits, y.d, y.chi):
        return "n_qubits/d/chi"
    if bool(x._is_canonical_merged) != bool(y._is_canonical_merged):
        return "canonical flag"
    if float(x.atol).hex() != float(y.atol).hex():
        return "atol"
    if float(x.log_mass).hex() != float(y.log_mass).hex():
        return "log_mass"
    if x.weights.dtype != y.weights.dtype or x.weights.shape != y.weights.shape:
        return "weights layout"
    if not np.array_equal(x.weights.view(np.uint64), y.weights.view(np.uint64)):
        return "weights"
    for name in _FIELDS:
        a, b = getattr(x, name), getattr(y, name)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return name
    return None


def _axis(rng: np.random.Generator, n: int, density: float = 0.3, hermitian: bool = True):
    """Random Pauli words (a, b, kappa); Hermitian (kappa parity == <a,b>) unless told not."""
    nw = pq.n_words_for_qubits(n)
    a = pq.pack_bit_vector((rng.random(n) < density).astype(np.uint8), n)
    b = pq.pack_bit_vector((rng.random(n) < density).astype(np.uint8), n)
    par = int(pq.word_dot_parity(a, b))
    kap = (par + 2 * int(rng.integers(0, 2))) & 3 if hermitian else int(rng.integers(0, 4))
    assert a.shape == (nw,)
    return a, b, kap


def _nonzero_axis(rng: np.random.Generator, n: int, **kw):
    while True:
        a, b, kap = _axis(rng, n, **kw)
        if a.any() or b.any():
            return a, b, kap


@dataclass
class Coverage:
    states: int = 0
    d_positive: int = 0
    non_canonical: int = 0
    chi_zero: int = 0
    signed_zero: int = 0
    chi_max: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)


def random_state(rng: np.random.Generator, n: int, chi: int, kind: str) -> pq.ShotExpansion:
    """A legal expansion reachable with public ops. kind: plain | d | raw | zero | signed0."""
    if kind == "zero":
        return pq.ShotExpansion.from_components((), n_qubits=n, merge=False)
    e = pq.ShotExpansion.zero(n)
    if kind == "d":  # quarter turns grow the geometry d
        for _ in range(int(rng.integers(1, 5))):
            a, b, kap = _nonzero_axis(rng, n)
            e = pq.apply_quarter_turn(e, a, b, kap)
    merge = kind != "raw"  # raw: merge=False leaves duplicate rows and canon=False
    for _ in range(60):
        if e.chi >= chi:
            break
        a, b, kap = _nonzero_axis(rng, n)
        e = pq.apply_pauli_t(e, a, b, kap, dag=bool(rng.integers(0, 2)), merge=merge)
    e.log_mass = float(rng.standard_normal())
    if kind == "signed0":  # exact zeros of both signs in the weights
        w = e.weights.copy()
        w[::2] = complex(-0.0, 0.0)
        w[1::3] = complex(0.0, -0.0)
        e.weights = w
    return e


_CELLS = [(n, chi) for n in (5, 64, 70, 130) for chi in (1, 4, 64)]
_KINDS = ("plain", "d", "raw", "signed0", "zero")


def _run_cell(n: int, chi: int, seed: int, cov: Coverage) -> int:
    """Random command programs on one (n, chi) cell; returns the number of states compared."""
    rng = np.random.default_rng([seed, n, chi])
    batch = len(_KINDS) * 18  # 90 shots per cell, 18 of each kind
    kinds = [_KINDS[i % len(_KINDS)] for i in rng.permutation(batch)]  # any kind at any position
    states = [random_state(rng, n, chi, k) for k in kinds]
    oracle = [s.copy() for s in states]
    store = LiveExpansions(batch, n, 1e-10)
    for b, s in enumerate(states):
        store[b] = s
    compared = 0
    for step in range(6):
        a, b, kap = _axis(rng, n, density=float(rng.choice([0.05, 0.3, 0.9])),
                          hermitian=bool(step % 2))
        mask = rng.random(batch) < float(rng.choice([0.0, 0.2, 0.5, 1.0]))
        for hit in np.flatnonzero(mask).tolist():
            if oracle[hit].chi > 0:
                oracle[hit] = pq.apply_pauli(oracle[hit], a, b, kap)
        store.apply_pauli_mask(pq_axis(a, b, kap), mask)
        for i in range(batch):
            diff = _first_difference(oracle[i], store[i])
            assert diff is None, f"n={n} chi={chi} step={step} shot={i} differs in {diff}"
            compared += 1
    for s in oracle:
        cov.states += 1
        cov.d_positive += s.d > 0
        cov.non_canonical += not s._is_canonical_merged
        cov.chi_zero += s.chi == 0
        cov.chi_max = max(cov.chi_max, s.chi)
        cov.signed_zero += bool(np.any(np.signbit(s.weights.real) | np.signbit(s.weights.imag)))
    store.close()
    return compared


def pq_axis(a, b, kap):
    from stimside._magic.gates import CompiledPauliAxis

    return CompiledPauliAxis(a, b, kap)


def test_apply_pauli_mask_is_bit_identical_to_the_per_shot_loop_on_over_a_thousand_states() -> None:
    cov, compared = Coverage(), 0
    for n, chi in _CELLS:
        compared += _run_cell(n, chi, 20260107, cov)
    assert cov.states >= 1000 and compared >= 6000
    # The generator must really reach the interesting classes (else the comparison is vacuous).
    assert cov.d_positive >= 150 and cov.non_canonical >= 100 and cov.chi_zero >= 100
    assert cov.signed_zero >= 200 and cov.chi_max >= 64


@pytest.mark.parametrize("n", [5, 70])
def test_each_shot_index_is_updated_when_it_alone_is_masked(n: int) -> None:
    rng = np.random.default_rng(n + 100)
    batch = 9
    states = [random_state(rng, n, 4, "d") for _ in range(batch)]
    axis = _nonzero_axis(rng, n)
    for only in range(batch):
        store = LiveExpansions(batch, n, 1e-10)
        for s, st in enumerate(states):
            store[s] = st
        mask = np.zeros(batch, dtype=bool)
        mask[only] = True
        store.apply_pauli_mask(pq_axis(*axis), mask)
        for s in range(batch):
            want = pq.apply_pauli(states[s], *axis) if s == only else states[s]
            assert _first_difference(store[s], want) is None, (only, s)
        store.close()


@pytest.mark.parametrize("n", [5, 64, 70, 130])
def test_import_export_round_trip_is_bit_exact(n: int) -> None:
    rng = np.random.default_rng(n)
    store = LiveExpansions(len(_KINDS) * 4, n, 3e-11)
    originals = [random_state(rng, n, 16, _KINDS[i % len(_KINDS)]) for i in range(store.batch_size)]
    for b, s in enumerate(originals):
        store[b] = s
    for b, s in enumerate(originals):
        assert _first_difference(store[b], s) is None
        assert store[b] is not store[b]  # a fresh copy on every read
    store.close()


def test_exports_and_imports_never_alias_engine_state() -> None:
    n = 70
    rng = np.random.default_rng(3)
    src = random_state(rng, n, 8, "d")
    store = LiveExpansions(3, n, 1e-10)
    store[1] = src
    snapshot = src.copy()
    src.weights[:] = 0.0  # mutating the object that was imported must not reach the engine
    src.h_words[:] ^= np.uint64(1)
    assert _first_difference(store[1], snapshot) is None
    out = store[1]
    out.weights[:] = 7.0  # mutating an export must not reach the engine either
    out.A_words[:] = np.uint64(5)
    assert _first_difference(store[1], snapshot) is None
    assert _first_difference(store[0], pq.ShotExpansion.zero(n)) is None  # neighbours untouched
    assert _first_difference(store[2], pq.ShotExpansion.zero(n)) is None


def test_reset_restores_every_shot_to_zero_and_keeps_the_engine_usable() -> None:
    n = 130
    rng = np.random.default_rng(5)
    store = LiveExpansions(8, n, 1e-10)
    for b in range(8):
        store[b] = random_state(rng, n, 32, "d")
    store.reset()
    zero = pq.ShotExpansion.zero(n, atol=1e-10)
    assert all(_first_difference(store[b], zero) is None for b in range(8))
    a, b_, kap = _nonzero_axis(rng, n)
    store.apply_pauli_mask(pq_axis(a, b_, kap), np.ones(8, dtype=bool))
    assert _first_difference(store[3], pq.apply_pauli(zero, a, b_, kap)) is None


def _dirty_same_size_state(n: int, atol: float, flavour: str) -> pq.ShotExpansion:
    """chi=1, d=0 states that differ from `zero` only in fields a size-keyed reset would skip."""
    nw = pq.n_words_for_qubits(n)
    h = np.zeros((1, nw), np.uint64)
    h[0, -1] = np.uint64(1) << np.uint64((n - 1) & 63)  # the highest legal qubit, last word
    h[0, 0] |= np.uint64(1)
    a_union = h[0].copy() if flavour == "a_union" else np.zeros(nw, np.uint64)
    return pq.ShotExpansion(
        n, 0, np.zeros((0, nw), np.uint64), a_union, np.zeros((0, 0), np.uint8),
        np.zeros(0, np.int32), np.array([0.3 - 0.4j], np.complex128), h,
        np.zeros((1, 0), np.uint8), np.zeros(0, np.uint8), -1.75, atol,
        flavour != "non_canonical",
    )


@pytest.mark.parametrize("n", [5, 70])
@pytest.mark.parametrize("flavour", ["h_and_log_mass", "non_canonical", "a_union"])
def test_reset_zeroes_same_size_states_bytewise_including_flag_log_mass_and_atol(
    n: int, flavour: str
) -> None:
    # chi=1 / d=0 shots keep their vector sizes across a reset, so only an unconditional
    # re-zeroing clears h, the canonical flag, log_mass, a_union and the shot's atol.
    store = LiveExpansions(4, n, 1e-10)
    dirty = _dirty_same_size_state(n, 3e-7, flavour)
    for b in range(4):
        store[b] = dirty
    assert store[0].h_words.any() and store[0].log_mass == -1.75
    assert store[0].atol == 3e-7
    assert bool(store[0]._is_canonical_merged) == (flavour != "non_canonical")
    store.reset()
    zero = pq.ShotExpansion.zero(n, atol=1e-10)
    for b in range(4):
        assert _first_difference(store[b], zero) is None, (b, _first_difference(store[b], zero))


@pytest.mark.parametrize("n", [5, 70])
def test_reset_zeroes_a_mixed_batch_of_chi_zero_non_canonical_and_grown_shots(n: int) -> None:
    rng = np.random.default_rng(n)
    store = LiveExpansions(6, n, 1e-10)
    store[0] = pq.ShotExpansion.from_components((), n_qubits=n, merge=False)  # chi == 0
    store[1] = random_state(rng, n, 8, "raw")  # chi > 1, canon False
    store[2] = random_state(rng, n, 4, "d")  # d > 0
    store[3] = _dirty_same_size_state(n, 1e-6, "non_canonical")
    assert store[0].chi == 0 and not store[1]._is_canonical_merged and store[2].d > 0
    store.reset()
    zero = pq.ShotExpansion.zero(n, atol=1e-10)
    assert all(_first_difference(store[b], zero) is None for b in range(6))
    a, b_, kap = _nonzero_axis(rng, n)  # and the reset shots really are live again
    store.apply_pauli_mask(pq_axis(a, b_, kap), np.ones(6, dtype=bool))
    assert all(_first_difference(store[b], pq.apply_pauli(zero, a, b_, kap)) is None
               for b in range(6))


def test_setitem_takes_integer_indices_only_like_getitem() -> None:
    n = 5
    store = LiveExpansions(3, n, 1e-10)
    marked = pq.ShotExpansion.zero(n, atol=1e-10)
    marked.log_mass = -2.0
    bad_indices: list[Any] = [1.5, 1.0, "1", None, (1,)]
    for bad in bad_indices:
        with pytest.raises(TypeError, match="integer"):
            store[bad] = marked
        with pytest.raises(TypeError, match="integer"):
            store[bad]
    with pytest.raises(TypeError, match="integer"):
        store[0:2] = marked  # type: ignore[index]
    assert all(store[b].log_mass == 0.0 for b in range(3))  # nothing was written anywhere
    numpy_index: Any = np.int64(1)
    store[numpy_index] = marked
    store[-3] = marked
    assert [store[b].log_mass for b in range(3)] == [-2.0, -2.0, 0.0]
    with pytest.raises(IndexError):
        store[3] = marked
    with pytest.raises(IndexError):
        store[-4] = marked
    store.close()


@pytest.mark.parametrize("atol", [float("nan"), float("inf"), 0.0, -1.0, True, None])
def test_the_store_rejects_a_bad_atol_and_bad_sizes_like_the_reference(atol: Any) -> None:
    with pytest.raises(ValueError, match="atol"):
        LiveExpansions(2, 5, atol)
    with pytest.raises(ValueError, match="atol"):
        pq.validate_atol(atol)  # the same inputs the reference rejects
    for bad in (1.5, True, None):
        with pytest.raises(TypeError):
            LiveExpansions(bad, 5, 1e-10)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            LiveExpansions(2, bad, 1e-10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LiveExpansions(-1, 5, 1e-10)
    with pytest.raises(ValueError):
        LiveExpansions(2, 0, 1e-10)


def test_copy_is_an_independent_store_with_the_same_shots() -> None:
    n = 70
    rng = np.random.default_rng(8)
    store = LiveExpansions(5, n, 1e-10)
    states = [random_state(rng, n, 8, k) for k in ("d", "raw", "plain", "zero", "signed0")]
    for b, s in enumerate(states):
        store[b] = s
    twin = store.copy()
    assert twin.engine is not store.engine
    assert all(_first_difference(twin[b], states[b]) is None for b in range(5))
    twin.close()  # freeing the copy leaves the original alone ...
    assert all(_first_difference(store[b], states[b]) is None for b in range(5))
    twin = store.copy()
    a, b_, kap = _nonzero_axis(rng, n)
    twin.apply_pauli_mask(pq_axis(a, b_, kap), np.ones(5, dtype=bool))  # ... as does mutating it
    assert all(_first_difference(store[b], states[b]) is None for b in range(5))
    fresh = LiveExpansions(2, n, 1e-10)  # no engine yet: the copy is the zero state, still lazy
    assert fresh.copy()._engine is None
    store.close()
    twin.close()


def test_a_store_shared_by_threads_serialises_its_engine_creation() -> None:
    store = LiveExpansions(3, 5, 1e-10)
    start, seen = threading.Barrier(8), []

    def grab() -> None:
        start.wait(30)
        seen.append(store.engine)

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(seen) == 8 and all(e is seen[0] for e in seen)  # one engine, not eight
    assert me.get_magic_engine_lib().mbe_live_engines() >= 1
    store.close()


def test_the_numerics_check_is_false_not_an_exception_when_the_library_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(force_recompile: bool = False) -> Any:
        raise OSError("libmagic_kernels.so: cannot open shared object file")

    me.numerics_match_numpy.cache_clear()
    monkeypatch.setattr(me, "get_magic_engine_lib", broken)
    try:
        assert engine_numerics_available() is False
        monkeypatch.setattr(me, "get_magic_engine_lib", lambda **_: (_ for _ in ()).throw(
            RuntimeError("rebuild failed")))
        me.numerics_match_numpy.cache_clear()
        assert engine_numerics_available() is False
    finally:
        monkeypatch.undo()
        me.numerics_match_numpy.cache_clear()  # later tests must see the real answer
    assert engine_numerics_available()


def test_the_copy_module_refuses_a_store_and_points_at_its_copy_method() -> None:
    store = LiveExpansions(1, 5, 1e-10)
    for clone in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="cannot be pickled or copied"):
            clone(store)
    store.close()


def test_chi_zero_shot_is_skipped_and_realization_style_assignment_works() -> None:
    n = 5
    store = LiveExpansions(2, n, 1e-10)
    empty = pq.ShotExpansion.from_components((), n_qubits=n, merge=False)
    store[0] = empty  # what simulator_magic_realization_test does through `_live_exps[0] = ...`
    assert store[0].chi == 0 and store[0].branch_mass() == 0.0
    a, b, kap = _nonzero_axis(np.random.default_rng(1), n)
    store.apply_pauli_mask(pq_axis(a, b, kap), np.ones(2, dtype=bool))
    assert _first_difference(store[0], empty) is None
    assert store[1].chi == 1


def test_non_finite_weights_raise_after_the_earlier_shots_were_updated() -> None:
    n = 64
    rng = np.random.default_rng(11)
    states = [random_state(rng, n, 4, "plain") for _ in range(5)]
    bad = states[2].copy()
    bad.weights = bad.weights.copy()
    bad.weights[0] = complex(np.nan, 0.0)
    states[2] = bad
    store = LiveExpansions(5, n, 1e-10)
    for b, s in enumerate(states):
        store[b] = s
    a, b_, kap = _nonzero_axis(rng, n)
    mask = np.array([1, 1, 1, 1, 1], dtype=bool)
    with pytest.raises(FloatingPointError, match="Non-finite weights"):
        pq.apply_pauli(bad, a, b_, kap)  # the oracle raises on the same shot
    with pytest.raises(FloatingPointError, match="Non-finite weights"):
        store.apply_pauli_mask(pq_axis(a, b_, kap), mask)
    for i in (0, 1):  # shots before the failing one were updated, like the per-shot loop
        assert _first_difference(store[i], pq.apply_pauli(states[i], a, b_, kap)) is None
    for i in (2, 3, 4):  # the failing shot and later ones are untouched
        assert _first_difference(store[i], states[i]) is None


def test_proxy_indexing_iteration_and_type_checks() -> None:
    n = 5
    store = LiveExpansions(3, n, 1e-10)
    rng = np.random.default_rng(2)
    exps = [random_state(rng, n, 4, "plain") for _ in range(3)]
    for b, e in enumerate(exps):
        store[b] = e
    assert len(store) == 3
    assert all(_first_difference(a, b) is None for a, b in zip(store, exps))
    assert _first_difference(store[-1], exps[2]) is None
    store[-1] = exps[0]
    assert _first_difference(store[2], exps[0]) is None
    with pytest.raises(IndexError):
        store[3]
    assert [_first_difference(a, b) for a, b in zip(store[0:2], exps)] == [None, None]
    with pytest.raises(TypeError):
        store["x"]  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="ShotExpansion"):
        store[0] = "x"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="n_qubits"):
        store[0] = pq.ShotExpansion.zero(6)


def test_conversion_helpers_round_trip_and_pickling_is_refused() -> None:
    import pickle

    e = random_state(np.random.default_rng(9), 70, 8, "d")
    assert _first_difference(arrays_to_expansion(70, expansion_to_arrays(e)), e) is None
    store = LiveExpansions(1, 5, 1e-10)
    store[0] = pq.ShotExpansion.zero(5)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(store)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(store.engine)
    store.close()


def test_the_engine_numerics_self_check_passes_on_this_machine() -> None:
    assert engine_numerics_available()
