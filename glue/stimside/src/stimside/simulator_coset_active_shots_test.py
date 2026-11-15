"""D-32 & R3-F7: `active_shots` read-only MappingProxyType, fresh-copy isolation, and export safety.

`active_shots` returns a read-only `MappingProxyType` of fresh `MRQFState.copy()` instances
materialized at read time. Mapping mutations raise `TypeError`, writes to returned `MRQFState`
copies never affect the simulator or engine, and tampered internal `_active_shots` cache entries
are replaced rather than written through during `_sync_cpp_to_py()`.
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
import stim

from stimside._coset.mrqf_state import MRQFState
from stimside._native import coset_kernels
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator

_B = 16
_TEXT = "H 0 1 2 3\nTICK\nM 0 1 2 3 4 5"


def _cpp_missing() -> bool:
    native_dir = Path(coset_kernels.__file__).resolve().parent
    has_so = (native_dir / "libcoset_kernels.so").exists()
    can_build = (native_dir / "coset_kernels.cpp").exists() and os.access(native_dir, os.W_OK)
    return not (has_so or can_build)


pytestmark = pytest.mark.skipif(_cpp_missing(), reason="coset C++ kernel library is unavailable")


def _mk() -> CosetsideSimulator:
    c = stim.Circuit(_TEXT)
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=_B)
    sim = CosetsideSimulator(c, compiled_op_handler=h, batch_size=_B, seed=3)
    if not sim._use_cpp:
        pytest.skip("Cosetside C++ engine is unavailable")
    return sim


def _rank(d: int) -> MRQFState:
    st = MRQFState(6, 1)
    st._ensure_capacity(max(1, d))
    st._d = d
    st.P = list(range(d))
    for j in range(d):
        st._A_buf[j, 0] = np.uint64(1 << j)
    st._recompute_a_union()
    st._refresh_views()
    return st


def _bits(sim: CosetsideSimulator) -> bytes:
    return b"".join(
        sim.peek_pauli_batch([0, 1, 2, 3, 4, 5], p).tobytes() for p in "XYZ"
    )


@pytest.mark.parametrize("key", [0, 2, _B, _B + 5, 10**6, -1, -3, 2.0, True, "x"])
def test_active_shots_mapping_rejects_assignment_and_deletion(key: object) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    before = _bits(sim)
    shots = sim.active_shots
    assert isinstance(shots, MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[Any, MRQFState], shots)[key] = MRQFState(6, 1)
    with pytest.raises(TypeError):
        del cast(dict[Any, MRQFState], shots)[key]
    assert _bits(sim) == before == _bits(ref)
    sim.run()
    ref.run()
    assert sim.get_detector_flips().tobytes() == ref.get_detector_flips().tobytes()


@pytest.mark.parametrize("mode", ["cpp", "py", "sync"])
def test_active_shots_is_readonly_mapping_proxy_in_all_modes(mode: str) -> None:
    c = stim.Circuit(_TEXT)
    h = LeakageUint8Coset().compile_op_handler(circuit=c, batch_size=_B)
    sim = CosetsideSimulator(
        c, compiled_op_handler=h, batch_size=_B, seed=3, sync_tableside_rng=(mode == "sync")
    )
    if mode == "py":
        sim._use_cpp = False
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    shots = sim.active_shots
    assert isinstance(shots, MappingProxyType)
    assert len(shots) > 0
    for mut in (
        lambda: cast(Any, shots).__setitem__(2, _rank(1)),
        lambda: cast(Any, shots).__delitem__(0),
        lambda: cast(Any, shots).clear(),
        lambda: cast(Any, shots).pop(0),
        lambda: cast(Any, shots).update({2: _rank(1)}),
        lambda: cast(Any, shots).setdefault(2, _rank(1)),
    ):
        with pytest.raises((TypeError, AttributeError)):
            mut()


def test_active_shots_materializes_fresh_copies_and_isolates_mutations() -> None:
    for use_cpp in (True, False):
        sim, ref = _mk(), _mk()
        sim._use_cpp = use_cpp
        ref._use_cpp = use_cpp
        m = np.ones((4, _B), bool)
        m[:, ::3] = False
        for s in (sim, ref):
            s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
        before = _bits(sim)
        read1 = sim.active_shots
        read2 = sim.active_shots
        b0 = min(read1)
        st1, st2 = read1[b0], read2[b0]
        assert st1 is not st2 and st1 is not sim._active_shots[b0]
        st1.check_invariants()
        st1.h_words[0] ^= np.uint64(0x55)
        st1.a_union[0] ^= np.uint64(0xAA)
        st1.A_words[0, 0] ^= np.uint64(1)
        st1.ell[0] ^= np.uint8(1)
        st1.P.append(99)
        st1._A_buf = np.zeros((0, 1), dtype=np.uint64)
        assert _bits(sim) == before == _bits(ref)
        assert sim.active_shots[b0].h_words.tobytes() == ref.active_shots[b0].h_words.tobytes()
        assert sim.active_shots[b0].P == ref.active_shots[b0].P


def test_export_cache_entry_with_short_buffers_is_replaced_on_sync() -> None:
    for swap in (
        ("h_words", np.zeros(0, np.uint64)),
        ("_A_buf", np.zeros((0, 1), np.uint64)),
        ("a_union", np.zeros(1, np.uint8)),
    ):
        sim, ref = _mk(), _mk()
        m = np.ones((4, _B), bool)
        m[:, ::3] = False
        for s in (sim, ref):
            s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
        sim._sync_cpp_to_py()
        hostile = MRQFState(6, 1)
        setattr(hostile, *swap)
        sim._active_shots[0] = hostile
        sim._cpp_dirty = True
        exported = sim.active_shots[0]
        assert exported is not hostile and sim._active_shots[0] is not hostile
        assert getattr(hostile, swap[0]).shape == swap[1].shape
        expect = ref.active_shots[0]
        assert exported.d == expect.d and exported.P == expect.P
        assert exported.h_words.tobytes() == expect.h_words.tobytes()
        assert exported.A_words.tobytes() == expect.A_words.tobytes()


class _SubArray(np.ndarray):
    pass


class _SubMRQF(MRQFState):
    pass


class _EvilInt(int):
    def __int__(self) -> int:
        return self

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return True

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return True


def test_mrqf_state_copy_is_deeply_independent() -> None:
    src = _rank(3)
    src.h_words[0] = np.uint64(0b101)
    src._ell_buf[:3] = np.array([1, 2, 3], dtype=np.uint8)
    src._Gamma_buf[0, 1] = src._Gamma_buf[1, 0] = 1
    cp = src.copy()
    assert cp is not src and cp.P is not src.P and cp.P == src.P and cp.d == 3
    for a, b in (
        (cp.h_words, src.h_words),
        (cp.a_union, src.a_union),
        (cp._A_buf, src._A_buf),
        (cp._ell_buf, src._ell_buf),
        (cp._Gamma_buf, src._Gamma_buf),
        (cp.A_words, src.A_words),
        (cp.ell, src.ell),
        (cp.Gamma, src.Gamma),
    ):
        assert not np.shares_memory(a, b)
        assert np.array_equal(a, b)
    cp.h_words[0] ^= np.uint64(1)
    cp.a_union[0] ^= np.uint64(1)
    cp.A_words[0, 0] ^= np.uint64(8)
    cp.ell[0] ^= np.uint8(1)
    cp.Gamma[0, 1] ^= np.uint8(1)
    cp.P[0] = 5
    assert int(src.h_words[0]) == 0b101
    assert src.P == [0, 1, 2] and int(src.ell[0]) == 1 and int(src.Gamma[0, 1]) == 1
    src.check_invariants()


def _tampered_export(kind: str) -> MRQFState:
    if kind == "readonly_h":
        st = MRQFState(6, 1)
        st.h_words.flags.writeable = False
        return st
    if kind == "subclass_h":
        st = MRQFState(6, 1)
        st.h_words = st.h_words.view(_SubArray)
        return st
    if kind == "cap_zero":
        st = MRQFState(6, 1)
        st._cap = 0
        st._A_buf = np.zeros((0, 1), dtype=np.uint64)
        st._ell_buf = np.zeros(0, dtype=np.uint8)
        st._Gamma_buf = np.zeros((0, 0), dtype=np.uint8)
        return st
    if kind == "cap_bool_one_row":
        st = MRQFState(6, 1)
        st._cap = cast(Any, True)
        st._A_buf = np.zeros((1, 1), dtype=np.uint64)
        st._ell_buf = np.zeros(1, dtype=np.uint8)
        st._Gamma_buf = np.zeros((1, 1), dtype=np.uint8)
        return st
    if kind == "d_bool":
        st = MRQFState(6, 1)
        st._d = cast(Any, True)
        return st
    if kind in ("nwords_float", "nwords_bool"):
        st = MRQFState(6, 1)
        st._cap = 1
        st._A_buf = np.zeros((1, 1), dtype=np.uint64)
        st._ell_buf = np.zeros(1, dtype=np.uint8)
        st._Gamma_buf = np.zeros((1, 1), dtype=np.uint8)
        st.n_words = cast(Any, 1.0 if kind == "nwords_float" else True)
        return st
    if kind == "aliased_h_au":
        st = MRQFState(6, 1)
        shared = np.zeros(1, dtype=np.uint64)
        st.h_words = shared
        st.a_union = shared
        return st
    if kind == "aliased_views":
        st = MRQFState(6, 1)
        buf = np.zeros(2, dtype=np.uint64)
        st.h_words = buf[:1]
        st.a_union = buf[:1]
        return st
    if kind == "nwords_mismatch":
        st = MRQFState(6, 1)
        st.n_words = 99
        return st
    if kind == "d_gt_cap":
        st = MRQFState(6, 1)
        st._d = st._cap + 1
        return st
    if kind == "d_neg":
        st = MRQFState(6, 1)
        st._d = -1
        return st
    return _SubMRQF(6, 1)


@pytest.mark.parametrize(
    "kind",
    [
        "readonly_h", "subclass_h", "cap_zero", "cap_bool_one_row", "d_bool",
        "nwords_float", "nwords_bool", "aliased_h_au", "aliased_views",
        "nwords_mismatch", "d_gt_cap", "d_neg", "subclass",
    ],
)
def test_export_cache_with_readonly_subclass_or_tampered_cap_is_replaced(kind: str) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    sim._sync_cpp_to_py()
    hostile = _tampered_export(kind)
    sim._active_shots[0] = hostile
    sim._cpp_dirty = True
    exported = sim.active_shots[0]
    assert exported is not hostile and sim._active_shots[0] is not hostile
    assert type(exported) is MRQFState
    assert exported.h_words.flags.writeable and type(exported.h_words) is np.ndarray
    assert type(exported.n_words) is int
    assert not np.shares_memory(exported.h_words, exported.a_union)
    expect = ref.active_shots[0]
    assert exported.d == expect.d and exported.P == expect.P
    exported.check_invariants()


def test_mrqf_ensure_capacity_rejects_nonpositive_cap() -> None:
    for bad_cap in (0, -2, False, True, 1.5, np.int64(4), _EvilInt(0)):
        st = MRQFState(6, 1)
        st._cap = cast(Any, bad_cap)
        with pytest.raises(ValueError, match="capacity"):
            st._ensure_capacity(10)


def test_cosetside_simulator_refuses_copy_deepcopy_and_pickle() -> None:
    sim = _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    for fn in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="CosetsideSimulator"):
            fn(sim)
    assert sim.peek_pauli_batch([0, 5]).shape == (2, _B)


@pytest.mark.parametrize("kind", ["nwords_evil_int", "gamma_aliased_ell"])
def test_export_cache_with_evil_int_nwords_or_gamma_alias_is_replaced(kind: str) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    sim._sync_cpp_to_py()
    hostile = MRQFState(6, 1)
    if kind == "nwords_evil_int":
        hostile.n_words = cast(Any, _EvilInt(1))
    else:
        cap = hostile._cap
        raw = np.zeros(cap * cap + 16, dtype=np.uint8)
        hostile._Gamma_buf = raw[: cap * cap].reshape(cap, cap)
        hostile._ell_buf = raw[cap - 1 : 2 * cap - 1]
    sim._active_shots[0] = hostile
    sim._cpp_dirty = True
    exported = sim.active_shots[0]
    assert exported is not hostile and sim._active_shots[0] is not hostile
    assert type(exported) is MRQFState and type(exported.n_words) is int
    assert not np.shares_memory(exported._ell_buf, exported._Gamma_buf)
    expect = ref.active_shots[0]
    assert exported.d == expect.d and exported.P == expect.P
    exported.check_invariants()

