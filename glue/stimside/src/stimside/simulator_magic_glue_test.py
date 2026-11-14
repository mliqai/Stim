"""Glue cuts (a)/(b)/(c) and D-21: the faster paths are exactly the reference.

(a) factor lists the simulator built (E1-memo objects, registered) skip re-encoding via a cached
    `OwnedFactorTable`; any other list, however frozen it looks, takes the full path.
(b) an `OwnedFactorTable` is private and read-only; a forged one (wrong seal) raises TypeError.
(c) `_as_i32` / `_list_indices` / `addr` keep their single-read, private-copy contracts.
D-21: the vectorized fault-channel draw and postselect bits are bit-identical to the loops.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

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
    lists = [CompiledCliffordFactorList(1, None, cast(Any, hostile))]
    assert store._trusted_table(lists, 1) is None


def test_an_owned_table_is_read_only_and_never_aliases_the_lists() -> None:
    fac = CompiledCliffordFactorList(1, None, (_axis(),))
    own = me.own_factor_table(ls.factor_table([fac], 1), 1)
    assert own.seal is None
    for arr in (*own.cols, *own.axes):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flags.writeable = True
        assert not np.shares_memory(arr, fac.r_axes[0].a_words)


def test_a_forged_or_mismatched_owned_table_is_refused() -> None:
    fac = CompiledCliffordFactorList(1, None, (_axis(),))
    own = me.own_factor_table(ls.factor_table([fac], 1), 1)
    forged = me.OwnedFactorTable(own.cols, own.axes, 1, object())
    spoof_same = me.OwnedFactorTable(own.cols, own.axes, 1, own.seal)
    evil_cols = (
        np.ones(10, np.uint8), np.ones(1, np.int32), np.zeros(1, np.int32),
        np.zeros(1, np.int32), np.zeros(1, np.int32),
    )
    evil_axes = (np.zeros((1, 1), np.uint64), np.zeros((1, 1), np.uint64), np.zeros(100, np.uint8))
    spoof_oob = me.OwnedFactorTable(evil_cols, evil_axes, 1, own.seal)
    wide = me.OwnedFactorTable(own.cols, own.axes, 2, own.seal)
    wide_real = me.own_factor_table(
        ls.factor_table([CompiledCliffordFactorList(1, None, (_axis(2),))], 2), 2
    )

    class _SubOwned(me.OwnedFactorTable):
        pass

    calls = [0]

    class _EvilSubOwned(me.OwnedFactorTable):
        def __hash__(self) -> int:
            calls[0] += 1
            return hash(own)

        def __eq__(self, other: object) -> bool:
            calls[0] += 1
            return True

    eng = me.MagicBatchEngine(2, 3, 1e-10)
    try:
        for bad in (
            forged, spoof_same, spoof_oob, wide, wide_real,
            _SubOwned(own.cols, own.axes, 1, own.seal),
            _EvilSubOwned(own.cols, own.axes, 1, own.seal),
        ):
            with pytest.raises(TypeError, match="own_factor_table"):
                eng.inject_cliffords(bad, np.zeros(2, np.int32), False)
        assert calls[0] == 0
        assert eng.inject_cliffords(own, np.zeros(2, np.int32), False) is None
    finally:
        eng.close()


def test_owned_table_object_setattr_cannot_poison_inject_cliffords() -> None:
    ft = ls.factor_table([CompiledCliffordFactorList(1, None, (_axis(),))], 1)
    own = me.own_factor_table(ft, 1)
    evil_cols = (
        np.ones(10, np.uint8), np.ones(1, np.int32), np.zeros(1, np.int32),
        np.zeros(1, np.int32), np.zeros(1, np.int32),
    )
    evil_axes = (
        np.zeros((1, 2), np.uint64), np.zeros((1, 2), np.uint64), np.zeros(100, np.uint8),
    )
    a_x, b_x = np.zeros(1, np.uint64), np.array([1], np.uint64)
    eng_ref = me.MagicBatchEngine(2, 3, 1e-10)
    eng = me.MagicBatchEngine(2, 3, 1e-10)
    eng2 = me.MagicBatchEngine(2, 70, 1e-10)
    try:
        for e in (eng_ref, eng):
            e.apply_pauli_mask(a_x, b_x, 0, np.ones(2, np.uint8))
        assert eng_ref.inject_cliffords(own, np.zeros(2, np.int32), False) is None
        object.__setattr__(own, "cols", evil_cols)
        object.__setattr__(own, "axes", evil_axes)
        object.__setattr__(own, "n_words", 2)
        assert eng.inject_cliffords(own, np.zeros(2, np.int32), False) is None
        ref_s, got_s = eng_ref.export_shot(0), eng.export_shot(0)
        assert got_s.weights.tobytes() == ref_s.weights.tobytes()
        assert got_s.h_words.tobytes() == ref_s.h_words.tobytes()
        with pytest.raises(TypeError, match="own_factor_table"):
            eng2.inject_cliffords(own, np.zeros(2, np.int32), False)
    finally:
        eng_ref.close()
        eng.close()
        eng2.close()


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


def test_trusted_table_cache_lru_eviction_and_cross_nwords_refusal() -> None:
    st = ls.LiveExpansions(2, 5, 1e-10)
    st_big = ls.LiveExpansions(2, 130, 1e-10)
    facs = [
        CompiledCliffordFactorList(1, None, (_axis(1),))
        for _ in range(ls._TABLE_CACHE_MAX + 2)
    ]
    for f in facs:
        ls.register_own_list(f)
    try:
        first = st._trusted_table([facs[0]], 1)
        assert first is not None
        for i in range(1, ls._TABLE_CACHE_MAX):
            assert st._trusted_table([facs[i]], 1) is not None
        assert len(st._tables) == ls._TABLE_CACHE_MAX
        assert st._trusted_table([facs[0]], 1) is first  # promote facs[0] so facs[1] becomes LRU
        assert st._trusted_table([facs[ls._TABLE_CACHE_MAX]], 1) is not None
        assert len(st._tables) == ls._TABLE_CACHE_MAX
        assert (id(facs[0]),) in st._tables
        assert (id(facs[1]),) not in st._tables
        assert (id(facs[ls._TABLE_CACHE_MAX]),) in st._tables
        assert st_big._trusted_table([facs[0]], 3) is None
    finally:
        st.close()
        st_big.close()


def test_trusted_table_and_frozen_key_snapshot_sequences_once() -> None:
    good = CompiledCliffordFactorList(1, None, (_axis(2),))
    ls.register_own_list(good)
    evil = CompiledCliffordFactorList(
        1, None, (CompiledPauliAxis(np.array([9, 9], np.uint64), np.zeros(2, np.uint64), 0),)
    )

    class _FlakySeq(Sequence[CompiledCliffordFactorList]):
        def __init__(self, first: CompiledCliffordFactorList) -> None:
            self.n = 0
            self.first = first

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int | slice) -> Any:
            return self.first

        def __iter__(self) -> Any:
            self.n += 1
            yield self.first if self.n == 1 else evil

    class _EvilTuple(tuple[Any, ...]):
        n = 0

        def __iter__(self) -> Any:
            type(self).n += 1
            yield good if type(self).n == 1 else evil

    st = ls.LiveExpansions(2, 70, 1e-10)
    try:
        res = st._trusted_table(_FlakySeq(good), 2)
        assert type(res) is me.OwnedFactorTable
        assert res.axes[0].shape == (1, 2) and res.axes[0].tolist() == [[1, 0]]
        assert st._trusted_table([good], 2) is res
        st._tables.clear()
        res_t = st._trusted_table(cast(Any, _EvilTuple((good,))), 2)
        assert type(res_t) is me.OwnedFactorTable and res_t.axes[0].tolist() == [[1, 0]]
        st._tables.clear()
        assert st._trusted_table(_FlakySeq(evil), 2) is None
        assert len(st._tables) == 0
        gen = (x for x in [good])
        assert ls._frozen_key(cast(Any, gen)) == (id(good),)
        assert ls._frozen_key(cast(Any, (x for x in [evil]))) is None
    finally:
        st.close()


def test_as_i32_uint32_overflow_and_list_indices_subclass_uint64_edges() -> None:
    for bad_u32 in (2**31, 2**32 - 1):
        with pytest.raises(ValueError, match="int32"):
            me._as_i32(np.array([bad_u32], dtype=np.uint32), (1,), "x")
    ok_u32 = me._as_i32(np.array([0, 2**31 - 1], dtype=np.uint32), (2,), "x")
    assert ok_u32.dtype == np.int32 and ok_u32.tolist() == [0, 2**31 - 1]

    class _EvilI32(np.ndarray):
        @property
        def dtype(self) -> np.dtype[Any]:
            return np.dtype(np.int32)

    evil = np.array([2**40], dtype=np.int64).view(_EvilI32)
    assert me._list_indices(evil, 5, 1).tolist() == [5]
    for dt in (np.uint64, np.dtype(">u8")):
        u64 = np.array([0, 2**63, 2**64 - 1], dtype=dt)
        assert me._list_indices(u64, 2, 3).tolist() == [0, 2, 2]


def test_categorical_batch_strict_lt_and_compile_factors_memo_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ZeroRng:
        def random(self, n: int) -> np.ndarray:
            return np.zeros(n, dtype=np.float64)

    out = sm._categorical_batch([0.0, 0.5], cast(Any, _ZeroRng()), 3)
    assert out.tolist() == [2, 2, 2]

    monkeypatch.setattr(sm, "_REF_MEMO_MAX_ENTRIES", 0)
    sim = MagicsideSimulator(stim.Circuit("H 0\nM 0"), batch_size=2, seed=1)
    fac = sim._compile_factors("H", (0,))
    assert id(fac) not in ls._OWN_LISTS


def _forced_and_sync_digest(
    monkeypatch: pytest.MonkeyPatch, engine: bool, *, sync: bool
) -> bytes:
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
    circ = stim.Circuit("H 0\nI[T] 0\nI[CLIFFORD_FAULT: (0.4, X) (0.4, Z)] 0\nM 0")
    reals = None if sync else [{"faults": {0: 1}}, {}, {"faults": {0: 2}}, {}]
    sim = MagicsideSimulator(
        circ, batch_size=4, seed=5, fault_realizations=reals, sync_tableside_rng=sync
    )
    if reals is not None:
        sim.interactive_do(circ)
        sim.finish_interactive_run()
    else:
        sim.run()
    out = [sim._log_branch_masses.tobytes(), *(c.tobytes() for c in sim._measurement_columns)]
    out.append(repr(sim.np_rng.bit_generator.state).encode())
    for s in range(4):
        _, e = sim.get_qfe_expansion(s)
        out += [e.weights.tobytes(), e.h_words.tobytes(), e.A_words.tobytes()]
    return b"".join(out)


def test_forced_fault_realizations_and_sync_rng_match_list_store_under_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _forced_and_sync_digest(monkeypatch, True, sync=False) == _forced_and_sync_digest(
        monkeypatch, False, sync=False
    )
    assert _forced_and_sync_digest(monkeypatch, True, sync=True) == _forced_and_sync_digest(
        monkeypatch, False, sync=True
    )


@pytest.mark.parametrize("dt", [np.int8, np.int64, np.uint8, np.uint64, np.dtype(">i4")])
def test_postselect_non_unit_bits_agree_across_integer_dtypes(
    dt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    vals = [0, 1, 2, 3, 4, 5, 6, 7] if np.dtype(dt).kind == "u" else [0, 1, 2, 3, -1, -2, 4, -3]
    bits = np.array(vals, dtype=dt)
    mask = np.array([True, False, True, True, True, True, False, True])
    outs: list[bytes] = []
    for engine in (False, True):
        monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", engine)
        sim = MagicsideSimulator(stim.Circuit("H 0\nI[T] 0\nM 0"), batch_size=8, seed=9)
        sim.interactive_do(stim.Circuit("H 0\nI[T] 0"))
        sim.postselect_z_batch(0, mask, bits)
        parts = [sim._log_branch_masses.tobytes()]
        for s in range(8):
            parts.append(sim.get_qfe_expansion(s)[1].weights.tobytes())
        outs.append(b"".join(parts))
    assert outs[0] == outs[1]


def test_own_factor_table_and_inject_reject_evil_int_nwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EvilInt(int):
        def __ne__(self, other: object) -> bool:
            return False

    ft_1w = ls.factor_table([CompiledCliffordFactorList(1, None, (_axis(1),))], 1)
    for bad_nw in (_EvilInt(1), True, 1.0, 0, -1):
        with pytest.raises(ValueError, match="n_words"):
            me.own_factor_table(ft_1w, cast(Any, bad_nw))

    own_1w = me.own_factor_table(ft_1w, 1)
    eng_1w = me.MagicBatchEngine(2, 3, 1e-10)
    eng_2w = me.MagicBatchEngine(2, 70, 1e-10)
    called = [False]
    orig_inject = eng_2w._lib.mbe_inject_cliffords

    def _spy(*args: object) -> int:
        called[0] = True
        return int(orig_inject(*args))

    try:
        monkeypatch.setattr(eng_2w._lib, "mbe_inject_cliffords", _spy)
        # Registry entry with non-exact int n_words must raise TypeError on both 1w and 2w engines.
        monkeypatch.setitem(me._OWNED_TABLES, own_1w, (own_1w.cols, own_1w.axes, _EvilInt(1)))
        for eng in (eng_1w, eng_2w):
            with pytest.raises(TypeError, match="own_factor_table"):
                eng.inject_cliffords(own_1w, np.zeros(2, np.int32), False)
        assert not called[0]
        # Non-exact int engine.n_words must also raise TypeError before touching C++.
        monkeypatch.setitem(me._OWNED_TABLES, own_1w, (own_1w.cols, own_1w.axes, 1))
        eng_2w.n_words = cast(Any, _EvilInt(2))
        with pytest.raises(TypeError, match="own_factor_table"):
            eng_2w.inject_cliffords(own_1w, np.zeros(2, np.int32), False)
        assert not called[0]
    finally:
        eng_1w.close()
        eng_2w.close()

