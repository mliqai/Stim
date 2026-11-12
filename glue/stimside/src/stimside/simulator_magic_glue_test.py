"""Run 5 Stage B (glue cuts (a)/(b)/(c), D-21): the faster paths are exactly the reference.

(a) factor lists the simulator built (E1-memo objects, registered) skip re-encoding via a cached
    `OwnedFactorTable`; any other list, however frozen it looks, takes the full path.
(b) an `OwnedFactorTable` is private and read-only; a forged one (wrong seal) is copied as usual.
(c) `_as_i32` / `_list_indices` / `addr` keep their single-read, private-copy contracts.
D-21: the vectorized fault-channel draw and postselect bits are bit-identical to the loops.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

import stimside.simulator_magic as sm
from stimside._magic import live_store as ls
from stimside._magic.channels import (
    CompiledCliffordFactorList,
    sample_categorical_branch,
)
from stimside._magic.gates import CliffordFaultChannel, CompiledPauliAxis, FaultBranch
from stimside._native import magic_engine as me
from stimside.simulator_magic import MagicsideSimulator

_CIRC = stim.Circuit("H 0 1 2\nI[T] 0\nCX 0 1\nI[T] 1 2\nM 0 1 2")

pytestmark = pytest.mark.skipif(
    not ls.engine_numerics_available(), reason="magic engine library is unavailable"
)


def _axis(n_words: int = 1) -> CompiledPauliAxis:
    a = np.zeros(n_words, np.uint64)
    a[0] = 1
    return CompiledPauliAxis(a, np.zeros(n_words, np.uint64), 0)


def _frozen_lookalike() -> CompiledCliffordFactorList:
    ax = _axis()
    ax.a_words.flags.writeable = False
    ax.b_words.flags.writeable = False
    return CompiledCliffordFactorList(1, None, (ax,))


def test_only_registered_lists_take_the_trusted_path() -> None:
    store = ls.LiveExpansions(4, 3, 1e-10)
    look = _frozen_lookalike()
    assert store._trusted_table([look], 1) is None  # read-only words are not enough
    ls.register_own_list(look)
    owned = store._trusted_table([look], 1)
    assert type(owned) is me.OwnedFactorTable
    assert store._trusted_table([look], 1) is owned  # cached by identity
    assert store._trusted_table([look, _frozen_lookalike()], 1) is None


@pytest.mark.parametrize("hostile", [None, (None,), "x"])
def test_hostile_lists_never_raise_in_the_trust_check(hostile: object) -> None:
    store = ls.LiveExpansions(2, 3, 1e-10)
    lists = [CompiledCliffordFactorList(1, None, hostile)]  # type: ignore[arg-type]
    assert store._trusted_table(lists, 1) is None


def test_an_owned_table_is_read_only_and_never_aliases_the_lists() -> None:
    fac = CompiledCliffordFactorList(1, None, (_axis(),))
    own = me.own_factor_table(ls.factor_table([fac], 1), 1)
    for arr in (*own.cols, *own.axes):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flags.writeable = True
        assert not np.shares_memory(arr, fac.r_axes[0].a_words)


def test_a_forged_or_mismatched_owned_table_is_refused() -> None:
    fac = CompiledCliffordFactorList(1, None, (_axis(),))
    own = me.own_factor_table(ls.factor_table([fac], 1), 1)
    forged = me.OwnedFactorTable(own.cols, own.axes, 1, object())
    eng = me.MagicBatchEngine(2, 3, 1e-10)
    try:
        with pytest.raises(TypeError, match="own_factor_table"):
            eng.inject_cliffords(forged, np.zeros(2, np.int32), False)
        wide = me.OwnedFactorTable(own.cols, own.axes, 2, own.seal)
        with pytest.raises(TypeError, match="own_factor_table"):
            eng.inject_cliffords(wide, np.zeros(2, np.int32), False)
        assert eng.inject_cliffords(own, np.zeros(2, np.int32), False) is None
    finally:
        eng.close()


def test_the_simulator_registers_its_memoized_factor_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", True)
    monkeypatch.setattr(sm, "_REF_MEMO_ENABLED", True)
    sim = MagicsideSimulator(_CIRC, batch_size=8, seed=1)
    sim.interactive_do(stim.Circuit("H 0 1 2\nI[T] 0"))
    mask = np.ones((1, 8), bool)
    mask[0, ::2] = False
    for _ in range(3):
        sim.do_conditional_clifford(stim.CircuitInstruction("CX", [0, 1]), mask)
    assert len(sim._live_exps._tables) >= 1


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.uint8, np.uint16])
def test_as_i32_narrow_dtypes_still_give_a_private_int32_copy(dtype: type) -> None:
    src = np.array([0, 1, 2], dtype=dtype)
    out = me._as_i32(src, (3,), "x")
    src[:] = 3
    assert out.dtype == np.int32 and out.tolist() == [0, 1, 2] and out.flags.owndata
    with pytest.raises(ValueError, match="shape"):
        me._as_i32(src, (4,), "x")


def test_list_indices_int32_fast_path_is_a_clipped_private_copy() -> None:
    src = np.array([-7, 0, 1, 5, 2**31 - 1], np.int32)
    out = me._list_indices(src, 2, 5)
    assert out.tolist() == [-1, 0, 1, 2, 2] and out.dtype == np.int32
    src[:] = 0
    assert out.tolist() == [-1, 0, 1, 2, 2]
    assert me._list_indices(src.astype(np.int64) + 2**40, 2, 5).tolist() == [2] * 5


def test_addr_still_refuses_strided_and_gives_null_for_empty() -> None:
    with pytest.raises(TypeError):
        me.addr(np.zeros(8)[::2])
    assert me.addr(np.zeros(0)) == 0
    a = np.zeros(3)
    assert me.addr(a) == a.ctypes.data


@pytest.mark.parametrize("seed", range(20))
def test_categorical_batch_matches_the_scalar_loop_and_rng_state(seed: int) -> None:
    rs = np.random.default_rng(seed)
    probs = list(rs.random(int(rs.integers(1, 15))) / 15)
    n = int(rs.integers(1, 130))
    r1, r2 = np.random.default_rng(seed + 100), np.random.default_rng(seed + 100)
    ref = [sample_categorical_branch(probs, r1) for _ in range(n)]
    assert sm._categorical_batch(probs, r2, n).tolist() == ref
    assert r1.bit_generator.state == r2.bit_generator.state


def _fault_run(engine: bool, monkeypatch: pytest.MonkeyPatch, batch: int) -> bytes:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    sim = MagicsideSimulator(_CIRC, batch_size=batch, seed=5)
    sim.interactive_do(stim.Circuit("H 0 1 2\nI[T] 0"))
    for k in range(5):
        ch = CliffordFaultChannel(
            0, (FaultBranch(0.2, "H", (0,)), FaultBranch(0.15, "CX", (0, 1))), k
        )
        sim._fire_fault_channel_live(k, ch)
    bits = (np.arange(batch) % 2).astype(np.int64)
    sim.postselect_z_batch(2, np.ones(batch, bool), bits)
    out = [sim._log_branch_masses.tobytes(), repr(sim.np_rng.bit_generator.state).encode()]
    for s in range(batch):
        _, e = sim.get_qfe_expansion(s)
        out += [e.weights.tobytes(), e.h_words.tobytes(), e.A_words.tobytes()]
    return b"".join(out)


@pytest.mark.parametrize("batch", [1, 7, 64])
def test_vectorized_fault_draws_and_postselect_agree_with_the_list_store(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _fault_run(True, monkeypatch, batch) == _fault_run(False, monkeypatch, batch)
