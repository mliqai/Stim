"""Hostile-caller tests of the block entry points (`mbe_apply_block`, `mbe_partition`).

The C++ runs with the GIL released, so (like `mbe_inject_cliffords`) it must never trust a
caller buffer after validating it, and nothing may unwind through `extern "C"`:
* deterministic: change every caller array at the moment the native call starts; the result must
  still be the reference's (the wrapper copies, the C++ copies once more and reads once);
* raw ABI (bypassing the wrapper): malformed shapes, out-of-range step starts, bad kinds and
  signs, non-Hermitian steps, padding bits, null pointers, huge sizes: an error code, the state
  untouched, the process alive;
* a racing thread rewriting the raw buffers during the call: no crash, no sanitizer report.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import MagicPhaseTerm, MagicTermRecord
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, block_table
from stimside._magic.live_store_test import _first_difference, random_state
from stimside._magic.plan import compile_open_block
from stimside._native import magic_engine as me
from stimside._native.magic_engine_test import _SpyLib

_N, _B, _ATOL = 70, 6, 1e-10


def _z_axis(rng: np.random.Generator, n: int) -> CompiledPauliAxis:
    bits = (rng.random(n) < 0.3).astype(np.uint8)
    bits[int(rng.integers(0, n))] = 1
    b = pq.pack_bit_vector(bits, n)
    return CompiledPauliAxis(np.zeros_like(b), b, 0)


def _block(rng: np.random.Generator, sequential: bool) -> pq.CompiledPauliBlock:
    recs = [MagicTermRecord(MagicPhaseTerm((_z_axis(rng, _N),), "T", 1))]
    rec = compile_open_block(recs, n_qubits=_N, budget=1 if sequential else 64)[0]
    blk = rec.block  # type: ignore[attr-defined]
    assert (blk.mode == "sequential") == sequential
    return blk


def _states(rng: np.random.Generator) -> list[pq.ShotExpansion]:
    kinds = ("plain", "d", "plain", "raw", "plain", "zero")
    return [random_state(rng, _N, 4, k) for k in kinds]


def _store(states: list[pq.ShotExpansion]) -> LiveExpansions:
    store = LiveExpansions(len(states), _N, _ATOL)
    for b, s in enumerate(states):
        store[b] = s
    return store


def _expect(states: list[pq.ShotExpansion], blk: pq.CompiledPauliBlock) -> list[pq.ShotExpansion]:
    out = [s.copy() for s in states]
    for i, s in enumerate(out):
        if s.chi > 0:
            out[i] = pq.apply_compiled_block(s, blk, check_unitary=True, atol=_ATOL)
    return out


@pytest.mark.parametrize("sequential", [False, True])
def test_the_block_command_reads_a_private_copy_of_every_caller_array(sequential: bool) -> None:
    rng = np.random.default_rng(1)
    states, blk = _states(rng), _block(rng, sequential)
    want = _expect(states, blk)
    table = block_table(blk, _N)
    assert table is not None
    store = _store(states)

    def scribble() -> None:  # what a racing thread could do to the caller's buffers
        for arr in (table.step_kind, table.step_sign, table.step_start, table.axes_kappa):
            arr[:] = 7
        table.axes_a[:] = ~np.uint64(0)
        table.axes_b[:] = ~np.uint64(0)
        table.axes_w_ri[:] = np.nan

    eng = store.engine
    hook = lambda n, _a: scribble() if n == "mbe_apply_block" else None
    eng._lib = _SpyLib(eng._lib, hook)  # type: ignore[assignment]
    res = eng.apply_block(table, False, None)
    assert res.failure is None
    assert np.isnan(table.axes_w_ri).all()  # the scribble really ran
    for i, w in enumerate(want):
        assert _first_difference(w, store[i]) is None, f"shot {i}"
    store.close()


def _raw_args(store: LiveExpansions, **over: Any) -> tuple[Any, ...]:
    """Valid raw arguments for one pauli_sum row (a Z-type axis on qubit 0), with overrides."""
    nw = store.engine.n_words
    a, b = np.zeros((1, nw), np.uint64), np.zeros((1, nw), np.uint64)
    b[0, 0] = 1
    z32 = np.zeros(0, np.int32)
    keep = {
        "mode": 0, "n_steps": 0, "n_axes": 1, "g_re": 1.0, "g_im": 0.0, "kind": z32, "sign": z32,
        "start": z32, "a": a, "b": b, "kappa": np.zeros(1, np.uint8),
        "w": np.array([[1.0, 0.0]]), "dedupe": 1, "mc": 2**63 - 1,
    }
    keep.update(over)
    keep["_keep"] = [keep[k] for k in ("kind", "sign", "start", "a", "b", "kappa", "w")]
    return keep


def _raw_call(store: LiveExpansions, k: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray]:
    err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
    ad = me.addr

    def p(x: Any) -> int:
        return ad(x) if x.size else 0

    rc = store.engine._lib.mbe_apply_block(
        store.engine._h(), k["mode"], k["n_steps"], k["n_axes"], k["g_re"], k["g_im"],
        p(k["kind"]), p(k["sign"]), p(k["start"]), p(k["a"]), p(k["b"]), p(k["kappa"]), p(k["w"]),
        k["dedupe"], k["mc"], ad(err), ad(errd), ad(info),
    )
    return int(rc), err, info


def _same_state(store: LiveExpansions, before: list[pq.ShotExpansion]) -> bool:
    return all(_first_difference(before[i], store[i]) is None for i in range(len(before)))


def _bad_rows() -> dict[str, dict[str, Any]]:
    one = np.ones(1, np.int32)
    return {
        "bad_mode": {"mode": 2},
        "pauli_sum_with_steps": {
            "n_steps": 1, "kind": one, "sign": one, "start": np.zeros(1, np.int32)
        },
        "step_kind_out_of_range": {"mode": 1, "n_steps": 1, "kind": np.array([9], np.int32),
                                   "sign": one, "start": np.zeros(1, np.int32)},
        "step_start_past_axes": {"mode": 1, "n_steps": 1, "kind": np.array([0], np.int32),
                                 "sign": one, "start": np.array([1], np.int32)},
        "step_start_negative": {"mode": 1, "n_steps": 1, "kind": np.array([0], np.int32),
                                "sign": one, "start": np.array([-1], np.int32)},
        "ccz_rows_run_past_axes": {"mode": 1, "n_steps": 1, "kind": np.array([3], np.int32),
                                   "sign": one, "start": np.zeros(1, np.int32)},
        "bad_sign": {"mode": 1, "n_steps": 1, "kind": np.array([1], np.int32),
                     "sign": np.array([5], np.int32), "start": np.zeros(1, np.int32)},
        "padding_bits_in_b": {"b": np.full((1, 2), ~np.uint64(0), np.uint64)},
        "n_axes_huge": {"n_axes": 2**62},
        "n_axes_wraps_size": {"n_axes": 2**63 + 2**61},
        "n_steps_huge": {"mode": 1, "n_steps": 2**61},
    }


@pytest.mark.parametrize("name", sorted(_bad_rows()))
def test_a_malformed_raw_block_is_refused_before_anything_changes(name: str) -> None:
    rng = np.random.default_rng(2)
    states = _states(rng)
    store = _store(states)
    rc, _err, _info = _raw_call(store, _raw_args(store, **_bad_rows()[name]))
    assert rc in (-2, -3), (name, rc)  # refused (or out of memory): never run, never a crash
    assert _same_state(store, states), name
    store.close()


def test_a_non_hermitian_quarter_turn_step_is_refused_by_the_raw_abi() -> None:
    rng = np.random.default_rng(3)
    states = _states(rng)
    store = _store(states)
    k = _raw_args(store, mode=1, n_steps=1, kind=np.array([1], np.int32),
                  sign=np.array([1], np.int32), start=np.array([0], np.int32))
    k["a"][0, 0] = 1  # X on qubit 0 and Z on qubit 0 with kappa 0: <a,b> = 1, kappa parity 0
    rc, _e, _i = _raw_call(store, k)
    assert rc == -2 and _same_state(store, states)
    store.close()


def test_null_arguments_are_refused_by_the_raw_abi() -> None:
    store = _store(_states(np.random.default_rng(4)))
    lib, h = store.engine._lib, store.engine._h()
    sizes = [ctypes.c_size_t(0)] * 2
    nulls = (None,) * 7
    z = ctypes.c_void_p(0)
    assert lib.mbe_apply_block(h, 0, 0, 0, 1.0, 0.0, *nulls, 1, 2**63 - 1, z, z, z) == -2
    assert lib.mbe_apply_block(None, 0, 0, 0, 1.0, 0.0, *nulls, 1, 2**63 - 1, z, z, z) == -2
    assert lib.mbe_partition(h, 1, 0) == -2
    assert lib.mbe_partition(None, 1, 0) == -2
    assert lib.mbe_live_shots(None) == -2
    assert sizes  # silence the unused name
    store.close()


def test_a_closed_engine_refuses_the_block_commands() -> None:
    rng = np.random.default_rng(5)
    store = _store(_states(rng))
    table = block_table(_block(rng, False), _N)
    assert table is not None
    eng = store.engine
    eng.close()
    for call in (lambda: eng.apply_block(table, True, None), lambda: eng.partition(True),
                 eng.live_shots):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_a_racing_thread_rewriting_the_raw_buffers_cannot_crash_the_block_command() -> None:
    """Raw ABI, no wrapper copies: the C++ validates then reads a private copy, so flipping the
    caller's step table between 'valid' and 'out of range' never reads outside it."""
    rng = np.random.default_rng(6)
    states = _states(rng)
    store = _store(states)
    k = _raw_args(store, mode=1, n_steps=1, kind=np.array([0], np.int32),
                  sign=np.array([1], np.int32), start=np.array([0], np.int32))
    stop = threading.Event()

    def flip() -> None:
        while not stop.is_set():
            k["start"][0] = 2**30
            k["start"][0] = 0
            k["kind"][0] = 9
            k["kind"][0] = 0

    t = threading.Thread(target=flip)
    t.start()
    try:
        for _ in range(300):
            rc, _e, _i = _raw_call(store, k)
            assert rc in (-1, -2)
    finally:
        stop.set()
        t.join()
    store.close()


def test_partition_equals_an_independent_pairwise_comparison_on_hostile_batches() -> None:
    """Identical shots, shots differing in exactly one byte (weights last bit, log_mass, canon,
    one h word) and chi == 0 shots: the engine's groups equal a pairwise byte comparison."""
    rng = np.random.default_rng(7)
    base = [random_state(rng, _N, 4, "d") for _ in range(2)]
    shots = [base[0].copy(), base[1].copy(), base[0].copy(), base[0].copy(), base[1].copy(),
             random_state(rng, _N, 4, "zero")]
    w = shots[2].weights.copy()
    w.view(np.uint64)[0] ^= np.uint64(1)  # last-bit difference in a weight
    shots[2].weights = w
    shots[3].log_mass = float(np.nextafter(shots[3].log_mass, 1e9))
    shots[4]._is_canonical_merged = not shots[4]._is_canonical_merged
    store = _store(shots)
    rep = store.engine.partition(True).tolist()
    want = [-1 if s.chi == 0 else next(j for j in range(i + 1)
            if shots[j].chi > 0 and _first_difference(shots[j], s) is None)
            for i, s in enumerate(shots)]
    assert rep == want, (rep, want)
    assert store.engine.partition(False).tolist() == [
        -1 if s.chi == 0 else i for i, s in enumerate(shots)
    ]
    assert store.engine.live_shots() == sum(s.chi > 0 for s in shots)
    store.close()


def test_a_difference_in_an_unsampled_row_keeps_shots_apart_in_the_engine_partition() -> None:
    """The cheap prefilter samples ~9 of the chi weights and the last h / ell row; two shots that
    differ only in an unsampled row must still be separate groups (the full comparison decides)."""
    rng = np.random.default_rng(8)
    a = random_state(rng, _N, 64, "plain")
    assert a.chi >= 24
    b, c = a.copy(), a.copy()
    row = 1  # chi // 8 >= 3, so rows 1, 2 are neither strided samples nor the last row
    w = b.weights.copy()
    w.view(np.uint64)[2 * row] ^= np.uint64(1)
    b.weights = w
    h = c.h_words.copy()
    h[row, 0] ^= np.uint64(1)
    c.h_words = h
    store = _store([a, b, a.copy(), c, b.copy()])
    assert store.engine.partition(True).tolist() == [0, 1, 0, 3, 1]
    store.close()
