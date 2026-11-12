"""A mask is copied first: the call checks and uses its own private copy (R3-F11).

`do_conditional_clifford` and `do_unmatched_noisy_reset_z` check the shape of a mask and later
hand it to a C++ kernel that reads `rows * batch_size` bytes. `np.ascontiguousarray(mask,
dtype=np.uint8)` returns the caller's OWN array for a C-contiguous uint8 mask and re-reads its
shape and dtype at that moment, so a caller that changed the array in between (a signal handler
flipping `mask.dtype`, a second thread calling `mask.resize(..., refcheck=False)`, an
object-dtype `__bool__` hook) made the check and the kernel disagree: ASan heap-buffer-overflow at
coset_kernels.cpp:629 / :891. Both methods now take one private copy (`_own_mask`), check that
copy and use only that copy.

The tests are deterministic: a hook on an instance method fires once, after the shape check and
before the mask is used, and changes the caller's array there directly, from a real signal
handler (`signal.raise_signal`), or from a real thread that is joined before the hook returns.
They check that

* a mask changed in place after the check (zeroed, re-viewed as uint16, shrunk) does not change
  the result, and a kernel is only ever handed a buffer holding at least the bytes it reads (a
  spy checks that and stands in for the kernel, so a regression is an AssertionError, not an
  out-of-bounds read inside the test run);
* the kernel is never handed the caller's own buffer, and the caller's arrays are not modified;
* `_own_mask` returns a private, C-ordered, plain, writable array of the caller's dtype;
* masks of every accepted dtype and layout give exactly the results of round 3 (digests pinned
  on the code before this fix, which converted with `np.asarray` only), error outcomes included.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import signal
import threading
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import stim

from stimside import simulator_coset as _sim_mod
from stimside._native import coset_kernels
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.simulator_coset import CosetsideSimulator, _own_mask

_DCC_KERNEL = "engine_batch_inject_general_clifford"
_RESET_KERNEL = "engine_batch_unmatched_noisy_reset_z"
_N = 6  # qubits of every circuit below
_B = 8  # batch size of the hook tests (even: a uint8 mask can be re-viewed as uint16)
_H4 = stim.CircuitInstruction("H", [0, 1, 2, 3])
_CX2 = stim.CircuitInstruction("CX", [0, 1, 2, 3])
_CX3 = stim.CircuitInstruction("CX", [0, 1, 1, 2, 2, 3])
_X4 = stim.CircuitInstruction("X", [0, 1, 2, 3])
_TAIL = "\nTICK\nM 0 1 2 3 4 5"
_MIXED = "X_ERROR(1) 0\nM 0\nCX rec[-1] 1 2 3\nTICK\nM 1 2 3 4 5"  # rec-controlled + quantum
_PATHS = [True, False]
_PATH_IDS = ["kernel", "python"]


class _MaskSpy:
    """Stands in for the kernels library; checks the mask buffers before the real kernel runs.

    `extents` maps the address of every array `np.ascontiguousarray` returned to its size in
    bytes. A mask the kernel is given must start at such an address and the array must hold at
    least the bytes the kernel reads (`num_groups * B` / `n_qubits * B`); otherwise this raises
    AssertionError instead of calling the kernel."""

    def __init__(self, lib: ctypes.CDLL, extents: dict[int, int], batch: int) -> None:
        self._lib, self._extents, self._batch = lib, extents, batch
        self.is_input: set[int] = set()  # addresses where np.ascontiguousarray made no new array
        self.mask_ptrs: dict[str, list[int]] = {_DCC_KERNEL: [], _RESET_KERNEL: []}

    def _check(self, name: str, ptr: Any, rows: int) -> None:
        addr = ctypes.cast(ptr, ctypes.c_void_p).value or 0
        have, need = self._extents.get(addr), rows * self._batch
        if have is None or have < need:
            raise AssertionError(f"{name} reads {need} mask bytes from a buffer of {have}")
        self.mask_ptrs[name].append(addr)

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if name not in self.mask_ptrs:
            return fn

        def spied(*args: Any) -> Any:
            if name == _DCC_KERNEL:
                self._check(name, args[5], args[1])  # (eng, num_g, arity, n_fac, t_groups, cm)
            else:
                self._check(name, args[1], _N)  # (eng, reset_zero, reset_one, ...)
                self._check(name, args[2], _N)
            return fn(*args)

        return spied


def _sim(
    text: str,
    *,
    use_cpp: bool = True,
    batch: int = _B,
    watch: pytest.MonkeyPatch | None = None,
    allow_sync_fallback: bool = True,
) -> tuple[CosetsideSimulator, _MaskSpy | None]:
    """A simulator; with `watch`, np.ascontiguousarray is recorded and the kernels are spied."""
    cpp_ok = _coset_cpp_available()
    sync_rng: bool | None = None
    if not cpp_ok:
        if use_cpp or not allow_sync_fallback:
            pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
        sync_rng = True
    circuit = stim.Circuit(text)
    handler = LeakageUint8Coset().compile_op_handler(circuit=circuit, batch_size=batch)
    sim = CosetsideSimulator(
        circuit,
        compiled_op_handler=handler,
        batch_size=batch,
        seed=5,
        sync_tableside_rng=sync_rng,
    )
    if not use_cpp:
        sim._use_cpp = False
    elif sim._cpp_lib is None:
        pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
    if watch is None or sim._cpp_lib is None:
        return sim, None
    extents: dict[int, int] = {}
    kept: list[np.ndarray] = []
    real = np.ascontiguousarray
    spy = _MaskSpy(sim._cpp_lib, extents, batch)

    def recording(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        kept.append(out)  # alive for the whole test, so no address is reused
        extents[out.ctypes.data] = out.nbytes
        if out is args[0]:
            spy.is_input.add(out.ctypes.data)
        return out

    watch.setattr(np, "ascontiguousarray", recording)
    sim._cpp_lib = spy  # type: ignore[assignment]
    return sim, spy


def _digest(sim: CosetsideSimulator) -> str:
    """Frames, active shots, step cursor and the Pauli expectations of every qubit."""
    xs, zs, _, _, _ = sim._flip_simulator.to_numpy(output_xs=True, output_zs=True)
    shots = sorted((b, st.d, st.h_words.tolist()) for b, st in sim.active_shots.items())
    peeks = np.stack([sim.peek_pauli_batch(list(range(_N)), p) for p in "XYZ"])
    parts = (xs.tobytes(), zs.tobytes(), repr(shots).encode(), peeks.tobytes())
    return hashlib.sha256(b"|".join(parts) + repr(sim._step_cursor).encode()).hexdigest()[:16]


def _fire_once(sim: CosetsideSimulator, method: str, fire: Callable[[], None]) -> None:
    """Run `fire()` the first time `sim.<method>` is called, just before it runs."""
    real, done = getattr(sim, method), []

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not done:
            done.append(True)
            fire()
        return real(*args, **kwargs)

    setattr(sim, method, wrapper)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _mutate(arr: np.ndarray, kind: str) -> None:
    """Change the caller's array in place the way another thread or a handler could."""
    if kind == "fill":
        arr[...] = 0
    elif kind == "dtype":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the `dtype` setter is deprecated in newer numpy
            arr.dtype = np.uint16
    else:
        arr.resize((1, arr.shape[-1]), refcheck=False)


def _via_signal(action: Callable[[], None]) -> None:
    """Run `action` in a real signal handler of this (main) thread, before this returns."""
    sig = getattr(signal, "SIGUSR1", None)
    assert sig is not None
    ran: list[bool] = []
    old = signal.signal(sig, lambda *_: (action(), ran.append(True)))
    try:
        signal.raise_signal(sig)
        assert ran, "the signal handler did not run"
    finally:
        signal.signal(sig, old)


def _via_thread(action: Callable[[], None]) -> None:
    """Run `action` in a real second thread, joined before this returns."""
    failure: list[BaseException] = []

    def target() -> None:
        try:
            action()
        except (AssertionError, TypeError, ValueError) as exc:  # re-raised in the caller's thread
            failure.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


_NO_SIGUSR1 = not (hasattr(signal, "SIGUSR1") and hasattr(signal, "raise_signal"))
_SKIP_NO_SIGUSR1 = pytest.mark.skipif(_NO_SIGUSR1, reason="SIGUSR1 requires POSIX")


def _coset_cpp_missing() -> bool:
    native_dir = Path(coset_kernels.__file__).resolve().parent
    has_so = (native_dir / "libcoset_kernels.so").exists()
    can_compile = (native_dir / "coset_kernels.cpp").exists() and os.access(native_dir, os.W_OK)
    return not (has_so or can_compile)


def _coset_cpp_available() -> bool:
    if _coset_cpp_missing():
        return False
    for mod in (coset_kernels, _sim_mod):
        getter = getattr(mod, "get_coset_kernels_lib", None)
        if getter is None:
            return False
        try:
            if getter() is None:
                return False
        except (OSError, RuntimeError):
            return False
    return True


_SKIP_NO_CPP = pytest.mark.skipif(
    _coset_cpp_missing(), reason="coset C++ kernel library is unavailable"
)


def _require_cpp_spy(spy: _MaskSpy | None) -> _MaskSpy:
    if spy is None:
        pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
    assert spy is not None
    return spy


_VIA: dict[str, Callable[[Callable[[], None]], None]] = {
    "hook": lambda action: action(), "signal": _via_signal, "thread": _via_thread,
}
_VIA_PARAMS = [
    pytest.param("signal", marks=_SKIP_NO_SIGUSR1) if name == "signal" else name
    for name in _VIA
]
_PATH_PARAMS = [pytest.param(True, marks=_SKIP_NO_CPP), False]
_MUTATIONS = ["fill", "dtype", "resize"]
_DTYPES = [np.uint8, np.bool_]
_DTYPE_IDS = ["uint8", "bool"]
# (circuit text, ops run interactively first, op (None: the op after those), mask rows, id)
_DCC_CASES = [
    ("H 0 1 2 3" + _TAIL, 0, _H4, 4, "h_disjoint"),
    ("CX 0 1 2 3" + _TAIL, 0, _CX2, 2, "cx_disjoint"),
    ("CX 0 1 1 2 2 3" + _TAIL, 0, _CX3, 3, "cx_overlapping"),
    ("X 0 1 2 3" + _TAIL, 0, _X4, 4, "pauli_x"),
    (_MIXED, 2, None, 2, "mixed_record_controlled"),
]


def _some_failed(rows: int, batch: int, dtype: Any) -> np.ndarray:
    mask = np.random.default_rng(rows).random((rows, batch)) < 0.5
    mask[:, 0], mask[:, -1] = False, True  # every row has a failed and a met shot
    return np.array(mask, dtype=dtype)


def _dcc_run(
    case: tuple, mask: np.ndarray, use_cpp: bool, change: Callable[[], None] | None,
    watch: pytest.MonkeyPatch | None,
) -> tuple[str, _MaskSpy | None]:
    """The state digest, or the error's name (a uint8 mask is not accepted by every path)."""
    text, prelude, op, _, _ = case
    sim, spy = _sim(text, use_cpp=use_cpp, watch=watch)
    circuit = stim.Circuit(text)
    for i in range(prelude):
        sim.interactive_do(circuit[i])
    if change is not None:
        _fire_once(sim, "_next_step_meta", change)  # runs right after the shape check
    try:
        sim.do_conditional_clifford(op or circuit[prelude], mask)
    except TypeError as exc:
        return type(exc).__name__, spy
    return _digest(sim), spy


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", _MUTATIONS)
@pytest.mark.parametrize("via", _VIA_PARAMS)
@pytest.mark.parametrize("case", _DCC_CASES, ids=[c[4] for c in _DCC_CASES])
def test_a_condition_met_mask_changed_after_the_check_is_not_what_the_call_uses(
    case: tuple, via: str, kind: str, dtype: Any, use_cpp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mask is zeroed / re-viewed as uint16 / shrunk to one row once the shape check has
    passed (by a hook, a signal handler or a thread). The call must act on the mask it checked:
    the result of an unmutated run."""
    rows = case[3]
    expected, _ = _dcc_run(case, _some_failed(rows, _B, dtype), use_cpp, None, None)
    mask = _some_failed(rows, _B, dtype)

    def change() -> None:
        _VIA[via](lambda: _mutate(mask, kind))

    got, spy = _dcc_run(case, mask, use_cpp, change, monkeypatch)
    assert got == expected
    if use_cpp and case[4] in ("h_disjoint", "cx_disjoint", "mixed_record_controlled") and (
        not got.endswith("Error")
    ):
        assert len(_require_cpp_spy(spy).mask_ptrs[_DCC_KERNEL]) == 1  # ran, on a checked copy


def _reset_run(
    which: str, mask: np.ndarray, use_cpp: bool, change: Callable[[], None] | None,
    watch: pytest.MonkeyPatch | None, pauli: str = "Z",
) -> tuple[str, _MaskSpy | None]:
    """`which` mask is `mask` (the one `change` alters), the other one is all False."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, use_cpp=use_cpp, watch=watch)
    p_lit = cast(Literal["X", "Y", "Z"], pauli)
    if which == "zero":  # reset_zero acts on qubits in |-1> of `pauli`: put them there first
        sim.do_unmatched_noisy_reset_z(
            np.zeros((_N, _B), bool), np.ones((_N, _B), bool), pauli=p_lit
        )
    zeros = np.zeros((_N, _B), mask.dtype)
    if change is not None:
        _fire_once(sim, "_current_packed_words", change)  # runs right after the shape checks
    try:
        r0, r1 = (mask, zeros) if which == "zero" else (zeros, mask)
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli=p_lit)
    except TypeError as exc:
        return type(exc).__name__, spy
    return _digest(sim), spy


@pytest.mark.parametrize("pauli", ["Z", "X", "Y"])
@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", _MUTATIONS)
@pytest.mark.parametrize("via", _VIA_PARAMS)
@pytest.mark.parametrize("which", ["zero", "one"])
def test_a_reset_mask_changed_after_the_check_is_not_what_the_call_uses(
    which: str, via: str, kind: str, dtype: Any, use_cpp: bool, pauli: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, _ = _reset_run(which, _some_failed(_N, _B, dtype), use_cpp, None, None, pauli=pauli)
    mask = _some_failed(_N, _B, dtype)

    def change() -> None:
        _VIA[via](lambda: _mutate(mask, kind))

    got, spy = _reset_run(which, mask, use_cpp, change, monkeypatch, pauli=pauli)
    assert got == expected
    if use_cpp and pauli == "Z":  # X/Y run on the Python path; Z ran on the kernel
        ptrs = _require_cpp_spy(spy).mask_ptrs[_RESET_KERNEL]
        assert len(ptrs) == (4 if which == "zero" else 2)


@pytest.mark.parametrize("pauli", ["Z", "X", "Y"])
@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
def test_both_reset_masks_changed_after_the_check_are_not_what_the_call_uses(
    use_cpp: bool, pauli: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One hook shrinks BOTH caller masks (a forgotten copy of either one shows)."""

    def run(mutate: bool) -> str:
        r0, r1 = _some_failed(_N, _B, np.uint8), _some_failed(_N, _B, np.uint8)
        r1 = r1[::-1].copy()
        sim, _ = _sim("H 0 1 2 3" + _TAIL, use_cpp=use_cpp, watch=monkeypatch)
        if mutate:
            _fire_once(
                sim, "_current_packed_words", lambda: [_mutate(m, "resize") for m in (r0, r1)]
            )
        p_lit = cast(Literal["X", "Y", "Z"], pauli)
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli=p_lit)
        return _digest(sim)

    assert run(True) == run(False)


@_SKIP_NO_CPP
def test_the_kernel_is_never_handed_the_callers_own_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a C-contiguous uint8 mask `np.ascontiguousarray` returns the caller's array itself;
    the kernels must be given the private copy instead."""
    mask = _some_failed(4, _B, np.uint8)
    sim, spy = _sim("H 0 1 2 3" + _TAIL, watch=monkeypatch)
    spy = _require_cpp_spy(spy)
    sim.do_conditional_clifford(_H4, mask)
    assert spy.mask_ptrs[_DCC_KERNEL] and mask.ctypes.data not in spy.mask_ptrs[_DCC_KERNEL]
    r0, r1 = _some_failed(_N, _B, np.uint8), _some_failed(_N, _B, np.uint8)
    sim, spy = _sim("H 0 1 2 3" + _TAIL, watch=monkeypatch)
    spy = _require_cpp_spy(spy)
    sim.do_unmatched_noisy_reset_z(r0, r1)
    ptrs = spy.mask_ptrs[_RESET_KERNEL]
    assert len(ptrs) == 2 and not {r0.ctypes.data, r1.ctypes.data} & set(ptrs)


@_SKIP_NO_CPP
def test_a_uint8_mask_is_copied_once_the_kernel_gets_the_copy_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private copy is already C-ordered uint8, so `np.ascontiguousarray` hands it on as it
    is: one copy per mask, not a copy of a copy (a Fortran-ordered copy would be copied again)."""
    sim, spy = _sim("H 0 1 2 3" + _TAIL, watch=monkeypatch)
    spy = _require_cpp_spy(spy)
    sim.do_conditional_clifford(_H4, _some_failed(4, _B, np.uint8))
    sim.do_unmatched_noisy_reset_z(_some_failed(_N, _B, np.uint8), _some_failed(_N, _B, np.uint8))
    ptrs = spy.mask_ptrs[_DCC_KERNEL] + spy.mask_ptrs[_RESET_KERNEL]
    assert len(ptrs) == 3 and set(ptrs) <= spy.is_input


class _Hook:
    """An object-dtype element: reading its truth value or `int()` rewrites the caller's arrays
    (every element becomes False) while numpy converts the mask, as a `__bool__` hook could."""

    def __init__(self, value: bool, arrays: list[np.ndarray]) -> None:
        self.value, self.arrays = value, arrays

    def _hit(self) -> bool:
        for arr in self.arrays:
            arr[...] = False
        return self.value

    def __bool__(self) -> bool:
        return self._hit()

    def __int__(self) -> int:
        return int(self._hit())


def _object_mask(values: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    mask = np.empty(values.shape, dtype=object)
    for idx in np.ndindex(*values.shape):
        mask[idx] = _Hook(bool(values[idx]), arrays)
    arrays.append(mask)
    return mask


@_SKIP_NO_CPP
def test_an_object_mask_with_hooks_is_converted_once_and_that_copy_is_what_is_used() -> None:
    """The elements' `__bool__` runs while the mask is converted (here: into the private copy,
    for the kernel path) and rewrites the caller's array; the copy holds what they reported."""
    values = _some_failed(4, _B, np.bool_)

    def run(mask: Any) -> str:
        sim, _ = _sim("H 0 1 2 3" + _TAIL)
        if sim._cpp_lib is None:
            pytest.skip("Cosetside C++ kernel library (_cpp_lib) is unavailable")
        sim.do_conditional_clifford(_H4, mask)
        return _digest(sim)

    assert run(_object_mask(values, [])) == run(values.astype(object))


# ---- layouts: the caller's arrays are not touched, and valid masks act as they did ----


def _strided(a: np.ndarray, dtype: Any) -> np.ndarray:
    big = np.zeros((a.shape[0] * 2, a.shape[1] * 2), dtype)
    big[::2, ::2] = a
    return big[::2, ::2]


def _reversed(a: np.ndarray) -> np.ndarray:
    return a.astype(np.uint8)[::-1, ::-1].copy()[::-1, ::-1]


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


# name -> function(bool array of the mask's logical content) -> what the caller passes
_LAYOUTS: dict[str, Callable[[np.ndarray], Any]] = {
    "bool": lambda a: a.copy(),
    "uint8": lambda a: a.astype(np.uint8),
    "uint8_255": lambda a: np.where(a, 255, 0).astype(np.uint8),
    "uint16": lambda a: a.astype(np.uint16),
    "int64_256": lambda a: np.where(a, 256, 0),
    "float64": lambda a: a.astype(np.float64),
    "object_bool": lambda a: a.astype(object),
    "fortran_bool": lambda a: np.asfortranarray(a),
    "fortran_uint8": lambda a: np.asfortranarray(a.astype(np.uint8)),
    "strided_bool": lambda a: _strided(a, np.bool_),
    "strided_uint8": lambda a: _strided(a, np.uint8),
    "reversed_uint8": _reversed,
    "read_only_bool": lambda a: _read_only(a.copy()),
    "read_only_uint8": lambda a: _read_only(a.astype(np.uint8)),
    "list": lambda a: a.tolist(),
    "masked_array": lambda a: np.ma.masked_array(a, mask=False),
}


def _snapshot(arr: Any) -> tuple:
    plain = np.asarray(arr)
    flags = plain.flags
    return (type(arr), plain.dtype, plain.shape, plain.strides, flags.writeable, plain.tobytes())


@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
@pytest.mark.parametrize("layout", [name for name in _LAYOUTS if name != "list"])
def test_the_callers_mask_arrays_are_not_modified_by_either_method(
    layout: str, use_cpp: bool
) -> None:
    make, rng = _LAYOUTS[layout], np.random.default_rng(4)
    for text, op, rows in (("H 0 1 2 3", _H4, 4), ("CX 0 1 2 3", _CX2, 2), ("X 0 1 2 3", _X4, 4)):
        mask = make(rng.random((rows, _B)) < 0.5)
        before = _snapshot(mask)
        sim, _ = _sim(text + _TAIL, use_cpp=use_cpp)
        with contextlib.suppress(TypeError, ValueError), _quiet():  # an unusable dtype: as before
            sim.do_conditional_clifford(op, mask)
        assert _snapshot(mask) == before
    r0, r1 = make(rng.random((_N, _B)) < 0.4), make(rng.random((_N, _B)) < 0.4)
    before = (_snapshot(r0), _snapshot(r1))
    sim, _ = _sim("H 0 1 2 3" + _TAIL, use_cpp=use_cpp)
    for pauli in "ZXY":
        with contextlib.suppress(TypeError, ValueError), _quiet():
            sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)  # type: ignore[arg-type]
        assert (_snapshot(r0), _snapshot(r1)) == before


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_own_mask_is_a_private_c_ordered_plain_copy_of_the_callers_dtype(layout: str) -> None:
    caller = _LAYOUTS[layout](np.random.default_rng(2).random((3, 8)) < 0.5)
    got, plain = _own_mask(caller), np.asarray(caller)
    assert type(got) is np.ndarray and got.shape == plain.shape and got.dtype == plain.dtype
    assert got.flags.c_contiguous and got.flags.writeable and got.flags.owndata
    assert not np.shares_memory(got, plain)
    assert np.array_equal(got, plain)


def test_own_mask_of_a_uint8_c_contiguous_array_is_not_the_array_numpy_would_pass_on() -> None:
    caller = np.ones((3, 8), np.uint8)
    assert np.ascontiguousarray(caller, dtype=np.uint8) is caller  # why the copy is needed
    got = _own_mask(caller)
    assert got is not caller and np.ascontiguousarray(got, dtype=np.uint8) is got
    got[...] = 0
    assert caller.all()  # writing the copy leaves the caller alone


def test_own_mask_runs_a_legacy_array_hook_once_without_a_copy_keyword_warning() -> None:
    calls: list[int] = []

    class Legacy:
        def __array__(self, dtype: Any = None) -> np.ndarray:  # no `copy` parameter
            calls.append(1)
            return np.ones((2, 3), np.bool_)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _own_mask(Legacy())
    assert len(calls) == 1 and got.shape == (2, 3)


def test_own_mask_drops_an_ndarray_subclass_and_never_runs_its_hooks() -> None:
    seen: list[str] = []

    class Sub(np.ndarray):
        def __getitem__(self, item: Any) -> Any:
            seen.append("getitem")
            return super().__getitem__(item)

    got = _own_mask(np.ones((2, 3), np.bool_).view(Sub))
    assert type(got) is np.ndarray and got.shape == (2, 3) and seen == []


# ---- every dtype / layout of a valid mask acts exactly as it did in round 3 ----

# (id, circuit text, op, rows) : every mask call below runs on a fresh simulator at step 0
_STEPS: list[tuple[str, str, stim.CircuitInstruction, int]] = [
    ("h4", "H 0 1 2 3", _H4, 4), ("cx2", "CX 0 1 2 3", _CX2, 2),
    ("cx3", "CX 0 1 1 2 2 3", _CX3, 3), ("x4", "X 0 1 2 3", _X4, 4),
]
_RESET_PAULIS = "ZXY"


def _attempt(run: Callable[[], str]) -> str:
    try:
        with _quiet():
            return run()
    except (TypeError, ValueError) as exc:  # the error outcome is part of what is pinned
        return type(exc).__name__


def _scramble(sim: CosetsideSimulator, rng: np.random.Generator, batch: int) -> None:
    """Random X and Z frames on every qubit, so that a different mask gives different frames."""
    for pauli in "XZ":
        sim.apply_pauli(pauli, rng.random((_N, batch)) < 0.5)  # type: ignore[arg-type]


def _outcomes(layout: str, use_cpp: bool, batch: int) -> list[str]:
    """One outcome per call (a state digest, or the name of the error it raised), each on a
    fresh simulator, with valid-shape masks in `layout`: four conditional Cliffords (all-true,
    and random masks), a record-controlled one, and a reset of each Pauli."""
    make, rng = _LAYOUTS[layout], np.random.default_rng(31)
    out: list[str] = []

    def dcc(text: str, op: stim.CircuitInstruction, rows: int, prelude: int, full: bool) -> str:
        sim, _ = _sim(text, use_cpp=use_cpp, batch=batch, allow_sync_fallback=False)
        for i in range(prelude):
            sim.interactive_do(stim.Circuit(text)[i])
        _scramble(sim, rng, batch)
        got = rng.random((rows, batch)) < 0.5
        sim.do_conditional_clifford(op, make(np.ones_like(got) if full else got))
        return _digest(sim)

    for _, text, op, rows in _STEPS:
        out.append(_attempt(lambda t=text, o=op, r=rows: dcc(t + _TAIL, o, r, 0, False)))
        out.append(_attempt(lambda t=text, o=op, r=rows: dcc(t + _TAIL, o, r, 0, True)))
    mixed = stim.Circuit(_MIXED)
    out.append(_attempt(lambda: dcc(_MIXED, mixed[2], 2, 2, False)))

    def reset(pauli: str, p: float) -> str:
        sim, _ = _sim("H 0 1 2 3" + _TAIL, use_cpp=use_cpp, batch=batch, allow_sync_fallback=False)
        _scramble(sim, rng, batch)
        r0, r1 = (make(rng.random((_N, batch)) < p) for _ in range(2))
        sim.do_unmatched_noisy_reset_z(r0, r1, pauli=pauli)  # type: ignore[arg-type]
        return _digest(sim)

    out += [_attempt(lambda p=pauli: reset(p, 0.3)) for pauli in _RESET_PAULIS]
    return out


def _outcome(layout: str, use_cpp: bool, batch: int) -> str:
    return hashlib.sha256(repr(_outcomes(layout, use_cpp, batch)).encode()).hexdigest()[:12]


# (layout, use_cpp, batch) -> outcome digest, taken on the round-3 code (`np.asarray` only)
_PINNED: dict[tuple[str, bool, int], str] = {
    ("bool", True, 1): "bc4be2e99214",
    ("bool", True, 7): "421b7467b576",
    ("bool", True, 64): "b30e3618adea",
    ("bool", False, 1): "bc4be2e99214",
    ("bool", False, 7): "421b7467b576",
    ("bool", False, 64): "b30e3618adea",
    ("uint8", True, 1): "9fbf76ad4efb",
    ("uint8", True, 7): "8c2c3871a30b",
    ("uint8", True, 64): "be0c94290819",
    ("uint8", False, 1): "9fbf76ad4efb",
    ("uint8", False, 7): "13ebcf7beb98",
    ("uint8", False, 64): "2504364e0c49",
    ("uint8_255", True, 1): "9fbf76ad4efb",
    ("uint8_255", True, 7): "d85f91658f76",
    ("uint8_255", True, 64): "24669d21b51c",
    ("uint8_255", False, 1): "9fbf76ad4efb",
    ("uint8_255", False, 7): "d85f91658f76",
    ("uint8_255", False, 64): "24669d21b51c",
    ("uint16", True, 1): "9fbf76ad4efb",
    ("uint16", True, 7): "8c2c3871a30b",
    ("uint16", True, 64): "be0c94290819",
    ("uint16", False, 1): "9fbf76ad4efb",
    ("uint16", False, 7): "13ebcf7beb98",
    ("uint16", False, 64): "2504364e0c49",
    ("int64_256", True, 1): "0b235aa80601",
    ("int64_256", True, 7): "f366c5660af4",
    ("int64_256", True, 64): "9aa35756b049",
    ("int64_256", False, 1): "298d1fbcad25",
    ("int64_256", False, 7): "5312195c19b4",
    ("int64_256", False, 64): "d31fcf9ae97f",
    ("float64", True, 1): "9aa82bc84503",
    ("float64", True, 7): "e81c3ef19084",
    ("float64", True, 64): "a6cca96a45eb",
    ("float64", False, 1): "d6ac1209dc5a",
    ("float64", False, 7): "13b2009c889c",
    ("float64", False, 64): "1bc9fd0ba1c9",
    ("object_bool", True, 1): "9fbf76ad4efb",
    ("object_bool", True, 7): "8c2c3871a30b",
    ("object_bool", True, 64): "be0c94290819",
    ("object_bool", False, 1): "9fbf76ad4efb",
    ("object_bool", False, 7): "13ebcf7beb98",
    ("object_bool", False, 64): "2504364e0c49",
    ("fortran_bool", True, 1): "bc4be2e99214",
    ("fortran_bool", True, 7): "421b7467b576",
    ("fortran_bool", True, 64): "b30e3618adea",
    ("fortran_bool", False, 1): "bc4be2e99214",
    ("fortran_bool", False, 7): "421b7467b576",
    ("fortran_bool", False, 64): "b30e3618adea",
    ("fortran_uint8", True, 1): "9fbf76ad4efb",
    ("fortran_uint8", True, 7): "8c2c3871a30b",
    ("fortran_uint8", True, 64): "be0c94290819",
    ("fortran_uint8", False, 1): "9fbf76ad4efb",
    ("fortran_uint8", False, 7): "13ebcf7beb98",
    ("fortran_uint8", False, 64): "2504364e0c49",
    ("strided_bool", True, 1): "bc4be2e99214",
    ("strided_bool", True, 7): "421b7467b576",
    ("strided_bool", True, 64): "b30e3618adea",
    ("strided_bool", False, 1): "bc4be2e99214",
    ("strided_bool", False, 7): "421b7467b576",
    ("strided_bool", False, 64): "b30e3618adea",
    ("strided_uint8", True, 1): "9fbf76ad4efb",
    ("strided_uint8", True, 7): "8c2c3871a30b",
    ("strided_uint8", True, 64): "be0c94290819",
    ("strided_uint8", False, 1): "9fbf76ad4efb",
    ("strided_uint8", False, 7): "13ebcf7beb98",
    ("strided_uint8", False, 64): "2504364e0c49",
    ("reversed_uint8", True, 1): "9fbf76ad4efb",
    ("reversed_uint8", True, 7): "8c2c3871a30b",
    ("reversed_uint8", True, 64): "be0c94290819",
    ("reversed_uint8", False, 1): "9fbf76ad4efb",
    ("reversed_uint8", False, 7): "13ebcf7beb98",
    ("reversed_uint8", False, 64): "2504364e0c49",
    ("read_only_bool", True, 1): "bc4be2e99214",
    ("read_only_bool", True, 7): "421b7467b576",
    ("read_only_bool", True, 64): "b30e3618adea",
    ("read_only_bool", False, 1): "bc4be2e99214",
    ("read_only_bool", False, 7): "421b7467b576",
    ("read_only_bool", False, 64): "b30e3618adea",
    ("read_only_uint8", True, 1): "9fbf76ad4efb",
    ("read_only_uint8", True, 7): "8c2c3871a30b",
    ("read_only_uint8", True, 64): "be0c94290819",
    ("read_only_uint8", False, 1): "9fbf76ad4efb",
    ("read_only_uint8", False, 7): "13ebcf7beb98",
    ("read_only_uint8", False, 64): "2504364e0c49",
    ("list", True, 1): "bc4be2e99214",
    ("list", True, 7): "421b7467b576",
    ("list", True, 64): "b30e3618adea",
    ("list", False, 1): "bc4be2e99214",
    ("list", False, 7): "421b7467b576",
    ("list", False, 64): "b30e3618adea",
    ("masked_array", True, 1): "bc4be2e99214",
    ("masked_array", True, 7): "421b7467b576",
    ("masked_array", True, 64): "b30e3618adea",
    ("masked_array", False, 1): "bc4be2e99214",
    ("masked_array", False, 7): "421b7467b576",
    ("masked_array", False, 64): "b30e3618adea",
}


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("use_cpp", _PATH_PARAMS, ids=_PATH_IDS)
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_masks_of_every_dtype_and_layout_act_as_they_did_before_the_copy(
    layout: str, use_cpp: bool, batch: int
) -> None:
    assert _outcome(layout, use_cpp, batch) == _PINNED[(layout, use_cpp, batch)], _outcomes(
        layout, use_cpp, batch
    )
