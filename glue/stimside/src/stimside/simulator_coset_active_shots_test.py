"""R3-F7: hostile `active_shots` entries never reach `engine_import_shot` / `engine_export_shot`.

`active_shots` is the live public dict. Before the fix an out-of-range key or a short buffer in
an entry was handed to C++ unchecked (heap reads/writes under ASan). Now every entry is checked
and copied before the engine is cleared, so a bad entry raises ValueError, changes nothing, and a
held reference with short buffers is replaced on export instead of being written through.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

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
