"""Hostile-caller tests of the engine's native boundary (S4b-1 fix stage).

* The C++ runs with the GIL released, so it must never read a buffer another thread can still
  write: the wrapper copies every array (`_as`) and `mbe_inject_cliffords` copies its table and
  shot list once more before reading them. Both layers are tested: deterministically, by
  changing the caller's arrays at the very moment the native call starts, and by racing a thread
  against the raw ABI (which bypasses the wrapper's copies).
* No C++ exception may cross `extern "C"` (that is `std::terminate`): sizes no allocation can
  satisfy come back as an error code. These probes run in a subprocess because a regression is
  a dead process, not a failed assertion.
* An out-of-memory error in the middle of a Clifford command names the shot it stopped at.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import CompiledCliffordFactorList, apply_clifford_factors
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, expansion_to_arrays, factor_table
from stimside._magic.live_store_test import (
    _first_difference,
    _nonzero_axis,
    random_state,
)
from stimside._native import magic_engine as me
from stimside._native.magic_engine_test import _SpyLib

_SRC = Path(me.__file__).resolve().parents[2]  # the directory that contains the `stimside` package
_N, _B, _ATOL = 70, 6, 1e-10


def _under_asan() -> bool:
    """ASan's and TSan's allocators abort the process on an impossible `new` before C++ can
    throw (the name is historical: it detects either sanitizer)."""
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__asan_init") or hasattr(libc, "__tsan_init")
    except OSError:
        return False


_NO_ASAN = pytest.mark.skipif(
    _under_asan(),
    reason="the ASan/TSan allocator aborts on impossible allocations instead of throwing",
)


def _run_child(script: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-c", script, str(_SRC), *args],
        capture_output=True, text=True, env=env, timeout=timeout, check=False,
    )


def _lists(rng: np.random.Generator) -> list[CompiledCliffordFactorList]:
    def ax() -> CompiledPauliAxis:
        return CompiledPauliAxis(*_nonzero_axis(rng, _N))

    return [
        CompiledCliffordFactorList(3, ax(), (ax(), ax())),
        CompiledCliffordFactorList(1, None, (ax(),)),
    ]


def _store_with(states: list[pq.ShotExpansion]) -> LiveExpansions:
    store = LiveExpansions(len(states), _N, _ATOL)
    for b, st in enumerate(states):
        store[b] = st
    return store


def _hooked(store: LiveExpansions, hook: Any, name: str) -> None:
    """Run `hook()` just before the native call `name` starts (after the wrapper's checks)."""
    eng = store.engine
    spy = _SpyLib(eng._lib, lambda n, _a: hook() if n == name else None)
    eng._lib = spy  # type: ignore[assignment]


def test_the_clifford_command_reads_a_private_copy_of_every_caller_array() -> None:
    rng = np.random.default_rng(1)
    states = [random_state(rng, _N, 4, "plain") for _ in range(_B)]
    lists = _lists(rng)
    which = np.array([0, 1, 0, -1, 1, 0], dtype=np.int32)
    expect = [s.copy() for s in states]
    for s, w in enumerate(which):
        if w >= 0:
            expect[s] = apply_clifford_factors(
                expect[s], lists[w], inverse=True, check_unitary=True, atol=_ATOL
            )
    table = factor_table(lists, 2)
    store = _store_with(states)

    def scribble() -> None:  # what a racing thread could do to the caller's buffers
        which[:] = -1
        for arr in (table.has_p0, table.n_r, table.v_fwd, table.v_inv, table.axis_start):
            arr[:] = 0
        table.axes_a[:] = 0
        table.axes_b[:] = 0
        table.axes_kappa[:] = 0

    _hooked(store, scribble, "mbe_inject_cliffords")
    assert store.engine.inject_cliffords(table, which, True) is None
    assert (which == -1).all() and not table.axes_a.any()  # the scribble really ran
    for i, want in enumerate(expect):
        assert _first_difference(want, store[i]) is None, f"shot {i}"
    store.close()


def test_the_pauli_mask_command_reads_private_copies_of_the_words_and_the_mask() -> None:
    rng = np.random.default_rng(2)
    states = [random_state(rng, _N, 4, "plain") for _ in range(_B)]
    a, b, kap = _nonzero_axis(rng, _N)
    mask = np.array([True, False, True, True, False, True])
    expect = [
        pq.apply_pauli(s, a, b, kap) if m else s.copy() for s, m in zip(states, mask, strict=True)
    ]
    store = _store_with(states)

    def scribble() -> None:
        a[:] = 0
        b[:] = 0
        mask[:] = False

    _hooked(store, scribble, "mbe_apply_pauli_mask")
    store.engine.apply_pauli_mask(a, b, kap, mask)
    assert not a.any() and not mask.any()  # the scribble really ran
    for i, want in enumerate(expect):
        assert _first_difference(want, store[i]) is None, f"shot {i}"
    store.close()


def test_import_shot_reads_private_copies_of_the_shot_arrays() -> None:
    rng = np.random.default_rng(3)
    st = random_state(rng, _N, 8, "d")
    want = st.copy()
    arrays = expansion_to_arrays(st)
    store = LiveExpansions(1, _N, _ATOL)

    def scribble() -> None:
        for arr in (arrays.weights, arrays.h_words, arrays.ell, arrays.A_words, arrays.Gamma):
            arr[...] = 0

    _hooked(store, scribble, "mbe_import_shot")
    store.engine.import_shot(0, arrays)
    assert not arrays.weights.any()  # the scribble really ran
    assert _first_difference(want, store[0]) is None
    store.close()


_RACE = r"""
import sys, threading, numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._magic.channels import CompiledCliffordFactorList as CL
from stimside._magic.gates import CompiledPauliAxis as AX
from stimside._magic.live_store import LiveExpansions, factor_table
from stimside._magic.live_store_test import _nonzero_axis, random_state
from stimside._native.magic_engine import addr

n, B, mode = 70, 64, sys.argv[2]
rng = np.random.default_rng(4)
st = random_state(rng, n, 64, "d")
store = LiveExpansions(B, n, 1e-10)
for b in range(B):
    store[b] = st.copy()
eng = store.engine
t = factor_table([CL(3, None, ()), CL(1, AX(*_nonzero_axis(rng, n)), ())], eng.n_words)
which = np.zeros(B, dtype=np.int32) if mode != "axes" else np.ones(B, dtype=np.int32)
target, at = {"shot_list": (which, B - 1), "n_r": (t.n_r, 0), "axes": (t.axes_a[0], 0)}[mode]
# a list index / axis count too big for the table, or (axes) another valid Pauli word
w0 = int(target[0])
bad, good = {"shot_list": (1 << 30, 0), "n_r": (5, 0), "axes": (w0 ^ 8, w0)}[mode]
stop = [False]


def flip():
    while not stop[0]:
        target[at] = bad
        target[at] = good


th = threading.Thread(target=flip)
th.start()
for _ in range(400):
    err, errd = np.zeros(4, np.int64), np.zeros(2)
    rc = eng._lib.mbe_inject_cliffords(     # the raw ABI: the wrapper's copies are not involved
        eng._h(), 2, t.axes_kappa.shape[0], addr(t.has_p0), addr(t.n_r), addr(t.v_fwd),
        addr(t.v_inv), addr(t.axis_start), addr(t.axes_a), addr(t.axes_b), addr(t.axes_kappa),
        addr(which), 0, addr(err), addr(errd))
    assert rc in (-1, -2) or rc >= 0, rc
    if mode == "axes":  # every shot got the same list: they must still be identical
        ref = eng.export_shot(0)
        for b in range(1, B):
            o = eng.export_shot(b)
            assert all(x.tobytes() == y.tobytes() for x, y in zip(o[5:], ref[5:])), b
stop[0] = True
th.join()
print("RACE DONE")
"""


@pytest.mark.parametrize("mode", ["shot_list", "n_r", "axes"])
def test_a_thread_rewriting_the_callers_buffers_during_the_raw_command_cannot_crash_it(
    mode: str,
) -> None:
    run = _run_child(_RACE, mode)
    assert run.returncode == 0 and "RACE DONE" in run.stdout, (run.returncode, run.stderr[-800:])


_HUGE = r"""
import math, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._magic.live_store import LiveExpansions
from stimside._native.magic_engine import addr

store = LiveExpansions(2, 5, 1e-10)
eng = store.engine
lib, h = eng._lib, eng._h()
sl = np.full(2, -1, dtype=np.int32)
err, errd = np.zeros(4, np.int64), np.zeros(2)
for n_lists in (0, 1):
    for n_axes in (2**59, 2**55, 2**63, 2**63 + 1, 2**64 - 1):  # 2**63: 2 * n_axes wraps to 0
        rc = lib.mbe_inject_cliffords(h, n_lists, n_axes, *([None] * 8), addr(sl), 0,
                                      addr(err), addr(errd))
        print("inject", n_lists, n_axes, rc, flush=True)
        assert rc in (-2, -3), rc
assert eng.export_shot(0).chi == 1   # the engine is intact
store.close()
print("HUGE OK")
"""


@_NO_ASAN
def test_sizes_no_allocation_can_satisfy_return_an_error_code_instead_of_terminating() -> None:
    run = _run_child(_HUGE)
    assert run.returncode == 0 and "HUGE OK" in run.stdout, (run.returncode, run.stderr[-800:])


class _FailsAt3:
    """Stand-in for the CDLL: the command 'ran out of memory' while at shot 3."""

    def __init__(self, lib: Any) -> None:
        self._lib = lib

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lib, name)

    def mbe_inject_cliffords(self, *args: Any) -> int:
        (ctypes.c_int64 * 4).from_address(args[-2])[2] = 3  # out_err[2]: the shot it was at
        return -3


def test_an_out_of_memory_error_names_the_shot_and_is_not_reported_as_before_running() -> None:
    store = LiveExpansions(5, 5, 1e-10)
    store.engine._lib = _FailsAt3(store.engine._lib)  # type: ignore[assignment]
    fac = CompiledCliffordFactorList(1, None, ())
    with pytest.raises(MemoryError, match=r"shot 3.*before it were updated") as exc:
        store.inject_factor_lists([fac], np.zeros(5, dtype=np.int32), False)
    assert "before running" not in str(exc.value)
    store.close()


_OOM = r"""
import resource, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._magic.channels import CompiledCliffordFactorList as CL
from stimside._magic.live_store import LiveExpansions, arrays_to_expansion
from stimside._native.magic_engine import ShotArrays

n, B, big = 5, 3, 3_000_000
store = LiveExpansions(B, n, 1e-10)
huge = ShotArrays(big, 0, True, 0.0, 1e-10, np.ones(big, np.complex128),
                  np.zeros((big, 1), np.uint64), np.zeros((big, 0), np.uint8),
                  np.zeros((0, 1), np.uint64), np.zeros(1, np.uint64), np.zeros((0, 0), np.uint8),
                  np.zeros(0, np.int32), np.zeros(0, np.uint8))
store[2] = arrays_to_expansion(n, huge)
del huge
vm = int(open("/proc/self/statm").read().split()[0]) * resource.getpagesize()
hard = resource.getrlimit(resource.RLIMIT_AS)[1]
resource.setrlimit(resource.RLIMIT_AS, (vm + 40 * 2**20, hard))   # the 72 MB staged copy cannot fit
try:
    store.inject_factor_lists([CL(1, None, ())], np.zeros(B, np.int32), False)
except MemoryError as exc:
    msg = str(exc)
else:
    msg = "NO ERROR"
finally:
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
c = 2 ** -0.5
assert "shot 2" in msg, msg
for b in (0, 1):   # committed before the failure: multiplied by omega
    assert store[b].weights[0] == complex(c, c), store[b].weights
assert store[2].chi == big and store[2].weights[0] == 1.0   # the failing shot is untouched
print("OOM OK", msg)
"""


@_NO_ASAN
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and RLIMIT_AS")
def test_a_real_out_of_memory_stops_at_the_failing_shot_after_committing_the_earlier_ones() -> None:
    run = _run_child(_OOM)
    assert run.returncode == 0 and "OOM OK" in run.stdout, (run.returncode, run.stderr[-800:])
