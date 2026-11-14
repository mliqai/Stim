"""R3-F7: hostile `active_shots` entries never reach `engine_import_shot` / `engine_export_shot`.

`active_shots` is the live public dict. Before the fix an out-of-range key or a short buffer in
an entry was handed to C++ unchecked (heap reads/writes under ASan). Now every entry is checked
and copied before the engine is cleared, so a bad entry raises ValueError, changes nothing, and a
held reference with short buffers is replaced on export instead of being written through.
"""

from __future__ import annotations

import copy
import ctypes
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import stim

from stimside._coset.mrqf_state import MRQFState
from stimside._native import coset_kernels
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator, _owned_block

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


def _short(attr: str, value: object, d: int = 3) -> Callable[[], MRQFState]:
    def make() -> MRQFState:
        st = _rank(d)
        setattr(st, attr, value)
        return st

    return make


def _with(**kw: object) -> Callable[[], MRQFState]:
    def make() -> MRQFState:
        st = _rank(2)
        for k, v in kw.items():
            setattr(st, k, v)
        return st

    return make


_BAD_STATES: dict[str, Callable[[], MRQFState]] = {
    "d_big": _with(_d=1000),
    "d_neg": _with(_d=-1),
    "P_short": _with(P=[0]),
    "P_neg": _with(P=[-5, 1]),
    "P_ge_nq": _with(P=[0, 40]),
    "h_short": _short("h_words", np.zeros(0, np.uint64)),
    "h_dtype": _short("h_words", np.zeros(8, np.uint8)),
    "au_short": _short("a_union", np.zeros(0, np.uint64)),
    "A_short": _short("_A_buf", np.zeros((1, 1), np.uint64)),
    "A_wide": _short("_A_buf", np.zeros((4, 2), np.uint64)),
    "ell_short": _short("_ell_buf", np.zeros(1, np.uint8)),
    "gamma_short": _short("_Gamma_buf", np.zeros((1, 1), np.uint8)),
}


def _bits(sim: CosetsideSimulator) -> bytes:
    return b"".join(
        sim.peek_pauli_batch([0, 1, 2, 3, 4, 5], p).tobytes() for p in "XYZ"
    )


@pytest.mark.parametrize("key", [_B, _B + 5, 10**6, -1, -3])
def test_an_out_of_range_key_raises_and_changes_nothing(key: int) -> None:
    sim, ref = _mk(), _mk()
    before = _bits(sim)
    sim.active_shots[key] = MRQFState(6, 1)
    with pytest.raises(ValueError, match="out of range"):
        sim.peek_pauli_batch([0, 5])
    del sim.active_shots[key]
    assert _bits(sim) == before == _bits(ref)
    sim.run()
    ref.run()
    assert sim.get_detector_flips().tobytes() == ref.get_detector_flips().tobytes()


@pytest.mark.parametrize("kind", list(_BAD_STATES))
def test_an_entry_with_a_bad_rank_index_or_buffer_raises_before_the_engine_is_touched(
    kind: str,
) -> None:
    sim = _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    before = _bits(sim)
    shots = sim.active_shots
    saved = dict(shots)
    shots[2] = _BAD_STATES[kind]()
    with pytest.raises(ValueError, match=r"active_shots\[2\]"):
        sim.peek_pauli_batch([0, 5])
    shots.clear()
    shots.update(saved)
    assert _bits(sim) == before


def test_a_valid_hand_built_entry_is_imported() -> None:
    sim = _mk()
    sim.active_shots[2] = _rank(3)
    sim.peek_pauli_batch([0, 5])
    assert set(sim.active_shots) == {2} and sim.active_shots[2].d == 3
    sim.active_shots[2].check_invariants()


@pytest.mark.parametrize(
    "swap",
    [
        ("h_words", np.zeros(0, np.uint64)),
        ("_A_buf", np.zeros((0, 1), np.uint64)),
        ("a_union", np.zeros(1, np.uint8)),
    ],
    ids=["h_words", "A_buf", "a_union_dtype"],
)
def test_a_held_reference_with_short_buffers_is_replaced_on_export(
    swap: tuple[str, np.ndarray],
) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    held = sim.active_shots
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    hostile = MRQFState(6, 1)
    setattr(hostile, *swap)
    held[0] = hostile  # not re-pushed (held across a kernel); the next export writes shot 0
    exported = sim.active_shots[0]
    assert exported is not hostile
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


def test_bad_entry_preserves_cpp_engine_state_without_reimport() -> None:
    sim = _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    before = _bits(sim)
    shots = sim.active_shots
    saved = dict(shots)
    shots[1] = _rank(1)
    shots[2] = _BAD_STATES["h_short"]()
    with pytest.raises(ValueError, match=r"active_shots\[2\]"):
        sim.peek_pauli_batch([0, 5])
    out_idx = np.empty(_B, dtype=np.int32)
    out_d = np.empty(_B, dtype=np.int32)
    assert sim._cpp_lib is not None and sim._cpp_eng is not None
    n_act = int(
        sim._cpp_lib.engine_get_active_shot_indices(
            sim._cpp_eng,
            out_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_d.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    )
    assert n_act == len(saved) and set(out_idx[:n_act].tolist()) == set(saved)
    sim._active_shots.clear()
    sim._active_shots.update(saved)
    sim._py_dirty = False
    assert _bits(sim) == before


def test_owned_block_and_shot_import_do_not_alias_caller_memory() -> None:
    src = np.array([1], dtype=np.uint64)
    blk = _owned_block(src, np.uint64, (1,), "h")
    src[0] = 777
    assert int(blk[0]) == 1

    sim = _mk()
    st0 = _rank(0)
    h_src = np.array([1], dtype=np.uint64)
    st0.h_words = h_src
    st1 = _rank(0)

    class _MutatingAu:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            h_src[0] = 0
            return np.zeros(1, dtype=np.uint64)

    st1.a_union = cast(Any, _MutatingAu())
    sim.active_shots[0] = st0
    sim.active_shots[1] = st1
    assert int(sim.peek_pauli_batch([0], "Z")[0, 0]) == -1


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
def test_held_reference_with_readonly_subclass_or_tampered_cap_is_replaced(kind: str) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    held = sim.active_shots
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    hostile = _tampered_export(kind)
    held[0] = hostile
    exported = sim.active_shots[0]
    assert exported is not hostile and type(exported) is MRQFState
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


def test_rank_d_boundaries_exact_nq_and_out_of_range() -> None:
    sim = _mk()
    sim.active_shots[2] = _rank(6)
    sim.peek_pauli_batch([0, 5])
    assert sim.active_shots[2].d == 6

    for bad_d, bad_p in ((-1, []), (7, [0, 1, 2, 3, 4, 5, 0])):
        s_bad = _mk()
        st = _rank(2)
        st._d = bad_d
        st.P = bad_p
        s_bad.active_shots[2] = st
        with pytest.raises(ValueError, match=rf"rank d={bad_d}"):
            s_bad.peek_pauli_batch([0, 5])


def test_cosetside_simulator_refuses_copy_deepcopy_and_pickle() -> None:
    sim = _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    sim.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    for fn in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="CosetsideSimulator"):
            fn(sim)
    assert sim.peek_pauli_batch([0, 5]).shape == (2, _B)


def test_duplicate_nonint_p_key_d_and_oversized_words_raise() -> None:
    bad_ps: list[object] = [
        [0, 0], [0.5, 1.5], [0.0, 1.0], [False, True], [0, True], ["0", "1"], [0, 2**70],
        np.array([False, True]), np.array([0.0, 1.0]),
        np.array([0, 0], dtype=np.int64), np.array([-1, 0], dtype=np.int64),
        np.array([0, 40], dtype=np.int64), np.array([0, 6], dtype=np.uint64),
    ]
    for bad_p in bad_ps:
        sim = _mk()
        st = _rank(2)
        cast(Any, st).P = bad_p
        sim.active_shots[2] = st
        with pytest.raises(ValueError):
            sim.peek_pauli_batch([0, 5])

    for bad_key in (2.0, True, False, "x"):
        sim = _mk()
        cast(dict[Any, MRQFState], sim.active_shots)[bad_key] = _rank(1)
        with pytest.raises((TypeError, ValueError)):
            sim.peek_pauli_batch([0, 5])

    for bad_d in (True, False, 1.0, "1"):
        sim = _mk()
        st = _rank(1)
        cast(Any, st)._d = bad_d
        sim.active_shots[2] = st
        with pytest.raises((TypeError, ValueError)):
            sim.peek_pauli_batch([0, 5])

    for attr in ("h_words", "a_union"):
        sim = _mk()
        st = _rank(1)
        setattr(st, attr, np.zeros(2, dtype=np.uint64))
        sim.active_shots[2] = st
        with pytest.raises(ValueError):
            sim.peek_pauli_batch([0, 5])

    sim_ok = _mk()
    cast(dict[Any, MRQFState], sim_ok.active_shots)[np.int64(3)] = _rank(1)
    assert sim_ok.peek_pauli_batch([0], "Z").shape == (1, _B)


@pytest.mark.parametrize("kind", ["nwords_evil_int", "gamma_aliased_ell"])
def test_held_reference_with_evil_int_nwords_or_gamma_alias_is_replaced(kind: str) -> None:
    sim, ref = _mk(), _mk()
    m = np.ones((4, _B), bool)
    m[:, ::3] = False
    held = sim.active_shots
    for s in (sim, ref):
        s.do_conditional_clifford(stim.CircuitInstruction("H", [0, 1, 2, 3]), m)
    hostile = MRQFState(6, 1)
    if kind == "nwords_evil_int":
        hostile.n_words = cast(Any, _EvilInt(1))
    else:
        cap = hostile._cap
        raw = np.zeros(cap * cap + 16, dtype=np.uint8)
        hostile._Gamma_buf = raw[: cap * cap].reshape(cap, cap)
        hostile._ell_buf = raw[cap - 1 : 2 * cap - 1]
    held[0] = hostile
    exported = sim.active_shots[0]
    assert exported is not hostile and type(exported) is MRQFState
    assert type(exported.n_words) is int
    assert not np.shares_memory(exported._ell_buf, exported._Gamma_buf)
    expect = ref.active_shots[0]
    assert exported.d == expect.d and exported.P == expect.P
    exported.check_invariants()

