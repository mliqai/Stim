"""Behaviour pins found missing by mutation testing of the engine wrapper and the live store.

Each test names the mutation that survived the first S4a tests (N07, N19, ...) and fails on it:
shape checks by element count, the padding check for registers of 64k+1 qubits, strided slice
reads of the store, `copy()` dropping the tolerance, a Pauli batch whose first shot is non-finite,
a full-rank shot (d == n_qubits), and the store-level lock that serialises engine creation and
`close()` (a constructor slowed down by a gate makes the missing lock visible).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

import stimside._magic.live_store as ls
import stimside._magic.phase_qfe as pq
from stimside._magic.live_store import LiveExpansions, expansion_to_arrays
from stimside._native import magic_engine as me

_WAIT_S = 30


def test_import_rejects_an_array_of_the_right_size_but_the_wrong_shape() -> None:
    eng = me.MagicBatchEngine(1, 70, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(70))
    with pytest.raises(ValueError, match="shape"):
        eng.import_shot(0, s._replace(h_words=s.h_words.reshape(2, 1)))  # (1, 2) -> (2, 1)
    eng.close()


@pytest.mark.parametrize("n", [1, 65, 129])
def test_a_padding_bit_is_rejected_when_the_last_word_has_one_valid_bit(n: int) -> None:
    nw = (n + 63) // 64
    bad = np.zeros(nw, np.uint64)
    bad[-1] = 2  # bit 1 of the last word: only bit 0 is a qubit, so this is padding
    eng = me.MagicBatchEngine(1, n, 1e-10)
    with pytest.raises(ValueError, match="padding"):
        eng.apply_pauli_mask(bad, np.zeros(nw, np.uint64), 0, np.ones(1, bool))
    eng.close()


def test_an_atol_given_as_a_size_one_array_is_rejected_like_validate_atol() -> None:
    with pytest.raises(ValueError):
        pq.validate_atol(np.array([1e-10]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="atol"):
        me.MagicBatchEngine(1, 3, np.array([1e-10]))  # type: ignore[arg-type]


def test_slice_reads_of_the_store_follow_list_semantics() -> None:
    store = LiveExpansions(5, 5, 1e-10)
    for b in range(5):
        z = pq.ShotExpansion.zero(5)
        z.log_mass = float(b)
        store[b] = z
    reference = [float(b) for b in range(5)]
    for sl in (slice(None, None, 2), slice(-2, None), slice(None, None, -1), slice(1, 4)):
        assert [e.log_mass for e in store[sl]] == reference[sl], sl
    store.close()


def test_copy_keeps_the_tolerance_and_a_reset_of_the_copy_uses_it() -> None:
    store = LiveExpansions(2, 5, 3e-10)
    twin = store.copy()
    assert twin.atol == 3e-10
    twin.reset()
    assert twin[0].atol == 3e-10 and twin[1].atol == 3e-10
    store.close()
    twin.close()


def test_a_shot_of_full_rank_d_equal_n_qubits_can_be_stored() -> None:
    n = 3
    shot = pq.ShotExpansion(
        n, 3, np.array([[1], [2], [4]], np.uint64), np.array([7], np.uint64),
        np.eye(3, dtype=np.uint8), np.arange(3, dtype=np.int32), np.array([1 + 0j]),
        np.zeros((1, 1), np.uint64), np.zeros((1, 3), np.uint8), np.zeros(3, np.uint8),
        0.0, 1e-10, True,
    )
    store = LiveExpansions(2, n, 1e-10)
    store[0] = shot
    assert store[0].d == 3
    store.close()


def test_a_non_finite_weight_in_the_first_shot_of_a_batch_is_reported() -> None:
    eng = me.MagicBatchEngine(3, 3, 1e-10)
    s = expansion_to_arrays(pq.ShotExpansion.zero(3))
    for b in range(3):
        eng.import_shot(b, s)
    eng.import_shot(0, s._replace(weights=np.array([complex("nan")])))
    with pytest.raises(FloatingPointError):
        eng.apply_pauli_mask(np.array([1], np.uint64), np.zeros(1, np.uint64), 0, np.ones(3, bool))
    eng.close()


def test_the_largest_size_t_batch_that_fits_is_a_memory_error_not_a_value_error() -> None:
    with pytest.raises(MemoryError):
        me.MagicBatchEngine(2**63 - 1, 5, 1e-10)


class _Gate:
    """A slowed constructor for `ls.MagicBatchEngine`: counts the constructions that started."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, hold: float = 0.0) -> None:
        self.entered, self.release = threading.Event(), threading.Event()
        self.started = 0
        gate, real = self, ls.MagicBatchEngine

        class Slow(real):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any) -> None:
                gate.started += 1
                gate.entered.set()
                gate.release.wait(hold or _WAIT_S)
                super().__init__(*args)

        monkeypatch.setattr(ls, "MagicBatchEngine", Slow)


def test_close_waits_for_an_engine_creation_that_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gate(monkeypatch)
    store = LiveExpansions(2, 5, 1e-10)
    creator = threading.Thread(target=lambda: store.engine)
    creator.start()
    assert gate.entered.wait(_WAIT_S)
    closer = threading.Thread(target=store.close)
    closer.start()
    time.sleep(0.3)  # close() has to queue behind the creation, not slip past it
    gate.release.set()
    creator.join(_WAIT_S)
    closer.join(_WAIT_S)
    assert not creator.is_alive() and not closer.is_alive()
    assert store._engine is None  # close() ran after the creation finished and dropped the engine


def test_racing_first_accesses_to_a_store_create_exactly_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gate(monkeypatch, hold=0.5)
    store = LiveExpansions(2, 5, 1e-10)
    out: list[Any] = []
    threads = [threading.Thread(target=lambda: out.append(store.engine)) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    gate.release.set()
    for t in threads:
        t.join(_WAIT_S)
    assert gate.started == 1, f"{gate.started} engines were constructed for 8 racing accesses"
    assert len(out) == 8 and all(e is out[0] for e in out)
    store.close()
