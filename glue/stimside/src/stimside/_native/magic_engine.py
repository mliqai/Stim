"""ctypes ABI and handle wrapper for the `mbe_*` MagicBatchEngine entry points.

The engine is C++ code in the same `libmagic_kernels.so` as the `magic_*` kernels (see
`magic_engine.hpp`), but it is declared here, with its own `_SPECS` and ABI sentinel, so the
`magic_kernels._SPECS` table, its sentinel and the tests that enumerate it stay as they are.

This module speaks arrays only (no `ShotExpansion`: `_native` may not import `_magic`); the
`_magic` layer converts between expansions and these arrays. The engine owns the
shots it holds: `export_shot` returns copies and `import_shot` copies in, so no Python object
can alias or outlive engine state. A wrapper is never pickled.

Thread safety and handle misuse. ctypes drops the GIL during every `mbe_*` call and the C++ takes
no locks, so two threads inside one engine (a `close()` or `import_shot` during `apply_pauli_mask`,
or an import between the header and the export of one shot) would free or resize memory another
thread reads. Each wrapper therefore holds a private lock around every native call: one engine
shared by threads is serialised, independent engines never wait for each other.

A native section is not re-entrant, because Python code can run on the thread that is inside it
(a signal handler, a garbage-collection finalizer, an `__array__` hook). Such code that calls into
the same engine gets a `RuntimeError`, except `close()`, which is deferred until the section
exits, so nothing frees an engine a pending native call is about to use. (The lock itself is an
`RLock` plus a `_busy` flag rather than a plain lock plus an owner id: a handler that fires right
after the lock was taken, before the flag is set, then simply runs a complete section of its own
instead of deadlocking, and a `fork()` from a handler cannot deadlock on its own thread.)

A `fork()` takes no lock in the parent: a parent that took the engine locks one after the other
could deadlock against a handler that nests engines in the other order. Instead the child, which
has only the forking thread, gives every engine and every `LiveExpansions` a fresh lock and clears
its busy / close-pending / creating flags, so it never inherits a lock held by a thread that does
not exist there. An engine that another thread was inside a native section of at the fork is
*poisoned* in the child: its memory is a copy-on-fork snapshot, possibly mid-update, so any use
raises `RuntimeError` and it is never freed there. An engine that was idle at the fork, or is
created afterwards, works normally in the child.

Limits (found, not fixed; the engine path is private and OFF by default): the wrapper is not
async-signal-safe, so a signal handler or finalizer on the owner thread must not touch an engine
beyond what is refused above, nor nest engine access while another thread is inside a different
engine in the opposite order (two threads would then wait on each other).

The raw `mbe_*` ABI trusts its caller (live handle, buffers sized from `mbe_shot_header`, a mask
of `batch_size` bytes, Pauli words of `n_words`); the wrapper is the only caller and enforces that
contract: integer sizes and indices, a positive finite `atol`, aligned contiguous buffers, Pauli
words without padding bits, and a clear `RuntimeError` for any use after `close()`.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import itertools
import math
import operator
import os
import sys
import threading
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from stimside._native._kernel_loader import AbiSpec, get_lib, resolve_so
from stimside._native.magic_kernels import w_addr

_LIB_INSTANCE: ctypes.CDLL | None = None

_VP, _SZ, _DBL = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
_I32, _INT, _U8 = ctypes.c_int32, ctypes.c_int, ctypes.c_uint8

# Pointer parameters are c_void_p, as for the magic_* kernels: callers pass integer addresses.
_SPECS: tuple[AbiSpec, ...] = (
    ("mbe_abi_v1", [], _I32),
    ("mbe_abi_v2", [], _I32),
    ("mbe_abi_v3", [], _I32),
    ("mbe_abi_v4", [], _I32),
    ("mbe_abi_v5", [], _I32),
    ("mbe_live_engines", [], _SZ),
    ("mbe_create", [_SZ, _SZ, _DBL], _VP),
    ("mbe_destroy", [_VP], None),
    ("mbe_reset", [_VP], _INT),
    ("mbe_import_shot", [_VP, _SZ, _SZ, _SZ, _I32, _DBL, _DBL, *[_VP] * 8], _INT),
    ("mbe_shot_header", [_VP, _SZ, _VP, _VP], _INT),
    ("mbe_export_shot", [_VP, _SZ, *[_VP] * 8], _INT),
    ("mbe_apply_pauli_mask", [_VP, _VP, _VP, _U8, _VP], ctypes.c_int64),
    (
        "mbe_inject_cliffords",
        [_VP, _SZ, _SZ, *[_VP] * 9, _I32, _VP, _VP],
        ctypes.c_int64,
    ),
    ("mbe_live_shots", [_VP], ctypes.c_int64),
    ("mbe_partition", [_VP, _I32, _VP], _INT),
    (
        "mbe_apply_block",
        [_VP, _I32, _SZ, _SZ, _DBL, _DBL, *[_VP] * 7, _I32, ctypes.c_int64, *[_VP] * 3],
        ctypes.c_int64,
    ),
    (
        "mbe_measure_split",
        [_VP, _VP, _VP, _U8, _I32, _I32, _I32, _VP, _VP, _U8, *[_VP] * 8],
        ctypes.c_int64,
    ),
    ("mbe_measure_finish", [_VP, _SZ, *[_VP] * 5], ctypes.c_int64),
    ("mbe_peek", [_VP, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 4], ctypes.c_int64),
    ("mbe_postselect", [_VP, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 6], ctypes.c_int64),
    (
        "mbe_unmatched_apply",
        [_VP, _VP, _VP, _U8, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 6],
        ctypes.c_int64,
    ),
    ("mbe_cmul_scalar", [_SZ, _VP, _DBL, _DBL], None),
    ("mbe_cmul_form", [_SZ, _VP, _DBL, _DBL, _I32], None),
    ("mbe_norm_sq", [_SZ, _VP], _DBL),
)


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile)


def get_magic_engine_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """The shared library with the `mbe_*` ABI bound (compiled on first use if missing/stale)."""
    return get_lib(
        sys.modules[__name__], "magic_kernels", "mbe_abi_v5", _SPECS,
        force_recompile=force_recompile,
    )


class ShotArrays(NamedTuple):
    """One shot in the tight `ShotExpansion` layout (shapes follow from `chi`, `d`, `n_words`)."""

    chi: int
    d: int
    canonical: bool
    log_mass: float
    atol: float
    weights: Any  # complex128 (chi,)
    h_words: Any  # uint64 (chi, nw)
    ell: Any  # uint8 (chi, d)
    A_words: Any  # uint64 (d, nw)
    a_union: Any  # uint64 (nw,)
    Gamma: Any  # uint8 (d, d)
    pivots: Any  # int32 (d,)
    ell_odd: Any  # uint8 (d,)


class FactorTable(NamedTuple):
    """Compiled Clifford factor lists in the engine's layout: list `l` owns the axes
    `axis_start[l] : axis_start[l] + has_p0[l] + n_r[l]` of `axes_*` (`p0` first, then the `r`
    axes). `v_fwd` / `v_inv` are the omega index of the closing multiply, -1 for none."""

    has_p0: Any  # uint8 (L,)
    n_r: Any  # int32 (L,)
    v_fwd: Any  # int32 (L,)
    v_inv: Any  # int32 (L,)
    axis_start: Any  # int32 (L,)
    axes_a: Any  # uint64 (A, nw)
    axes_b: Any  # uint64 (A, nw)
    axes_kappa: Any  # uint8 (A,)


class FactorFailure(NamedTuple):
    """Why `inject_cliffords` stopped at `shot` (earlier shots were updated, later ones not)."""

    shot: int
    kind: int  # FAIL_* below
    op: int  # 0 apply_pauli, 1 apply_quarter_turn
    aux: int
    aux2: int
    pre: float
    post: float
    z: float = 0.0  # S5b Born-split failures only: the shot's total mass z


FAIL_NON_FINITE_WEIGHTS, FAIL_NON_FINITE_NORM, FAIL_KERNEL = 1, 2, 3
FAIL_UNITARITY, FAIL_NOT_HERMITIAN, FAIL_PADDING = 4, 5, 6
FAIL_MERGE_NON_FINITE, FAIL_MAX_COMPONENTS = 7, 8
FAIL_BAD_NORM, FAIL_ROTATION, FAIL_ZERO_CHOSEN = 9, 10, 11  # measurements (S5a)
# Born-split checks of one shot (S5b: `FactorFailure.pre` / `post` / `z` carry the masses)
FAIL_NON_FINITE_MASS, FAIL_NEGATIVE_MASS, FAIL_CONSERVATION, FAIL_ZERO_POSTSELECT = 12, 13, 14, 15
# `SplitResult.cls`: what `measure_split` found for each shot.
CLS_DEAD, CLS_TRIVIAL, CLS_ACTIVE, CLS_UNSEEN = 0, 1, 2, 3
# `FactorFailure.op`: the phase_qfe function whose exception text the failure is reported as.
OP_NAMES = (
    "apply_pauli", "apply_quarter_turn", "apply_pauli_t", "apply_ccz", "apply_compiled_block",
    "merge_frames", "project_diagonal",
)


class BlockTable(NamedTuple):
    """A compiled block in the engine's layout (see `mbe_apply_block`).

    `mode` 0: the `axes_*` rows are the branches of a pauli_sum with coefficients `axes_w_ri`
    (the block's scalar already applied, as the reference does). `mode` 1: step `i` is
    `step_kind[i]` (0 P1, 1 P2, 2 N1, 3 N2) with `step_sign[i]` over row `step_start[i]` (and
    the seven rows after it for N2, whose 8 weights are `axes_w_ri`); `g_re`, `g_im` multiply
    the weights first.
    """

    mode: int
    g_re: float
    g_im: float
    step_kind: Any  # int32 (S,)
    step_sign: Any  # int32 (S,)
    step_start: Any  # int32 (S,)
    axes_a: Any  # uint64 (A, nw)
    axes_b: Any  # uint64 (A, nw)
    axes_kappa: Any  # uint8 (A,)
    axes_w_ri: Any  # float64 (A, 2)


class BlockResult(NamedTuple):
    """`apply_block`: `failure` is None on success; `groups`, `live` = groups run, shots with
    chi > 0 before the block."""

    failure: FactorFailure | None
    groups: int
    live: int


class SplitResult(NamedTuple):
    """`measure_split`: per shot `cls` (CLS_*; CLS_UNSEEN past a failure), the bit of a trivial
    shot, the pre-norm `z` and raw masses `z01[:, m]` of an active one, and the rows `chi2[:, m]` of
    each projected branch. `failure` is set when the split stopped at `failure.shot`; `oom` when
    it ran out of memory (the shots before it were classified and their kicks applied).
    `token` is unique to this split (see `MagicBatchEngine.measure_finish`)."""

    cls: Any
    bit: Any
    z: Any
    z01: Any
    chi2: Any
    n_active: int
    failure: FactorFailure | None
    oom: bool
    token: int = 0


class FinishResult(NamedTuple):
    """`measure_finish`: `failure` None when every active shot was committed; `phase` 1 = nothing
    changed, 2 = the failing reset kick left the shots before it committed."""

    failure: FactorFailure | None
    phase: int


class PostselectResult(NamedTuple):
    """`postselect`: `done[i]` = list entry i was committed (a dead shot is skipped, not done),
    `z01[i]` its clipped masses (z0, z1). `failure.shot` is the list position that failed (the
    entries before it are committed); `oom_at` the position an out-of-memory stopped at, else
    None."""

    done: Any
    z01: Any
    failure: FactorFailure | None
    oom_at: int | None


class UnmatchedResult(NamedTuple):
    """`unmatched_apply`: `failure.shot` / `oom_at` are list positions, `phase` 1 = the projection
    of that entry failed (the shot is as before), 2 = its kick failed (the projection is
    committed). The entries before it are finished, the later ones untouched."""

    failure: FactorFailure | None
    oom_at: int | None
    phase: int


_MAX_SIZE = 2**63 - 1  # ctypes would wrap a size_t argument at 2**64
_SPLIT_TOKENS = itertools.count(1)  # one per `measure_split`, across all engines (atomic `next`)


def addr(arr: Any) -> int:
    """`magic_kernels.addr`, refusing buffers the C++ must not load through (it reads uint64,
    int32 and double directly: an unaligned or strided buffer is undefined behaviour)."""
    flags = arr.flags
    if not (flags.c_contiguous and flags.aligned):
        raise TypeError("native buffers must be aligned and C-contiguous")
    return int(arr.ctypes.data) if arr.size > 0 else 0  # = `_kernels_addr(arr)`, inlined


def as_integer(v: Any, what: str) -> int:
    """`v` as a Python int; floats, strings, None and bools are refused (no silent truncation)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {type(v).__name__}")
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {type(v).__name__}") from None


def _valid_atol(atol: Any) -> float:
    """A positive finite float, as `phase_qfe.validate_atol` (`_native` cannot import `_magic`)."""
    scalar = not isinstance(atol, (bool, np.bool_)) and np.ndim(atol) == 0
    try:
        val = float(atol) if scalar else math.nan
    except (TypeError, ValueError):
        val = math.nan
    if not (math.isfinite(val) and val > 0.0):
        raise ValueError(f"atol must be a finite positive float, got {atol!r}")
    return val


def _check_padding(words: Any, n_qubits: int, what: str) -> None:
    rem = n_qubits & 63
    if rem and int(words[-1]) >> rem:
        raise ValueError(f"Non-zero upper padding bits in {what} for n_qubits={n_qubits}")


def _as(arr: Any, dtype: Any, shape: tuple[int, ...], what: str) -> Any:
    """A private, aligned, C-contiguous copy of `arr` as `dtype` (always a copy).

    The C++ runs with the GIL released: a buffer the caller still owns could be changed by
    another thread between the C++'s validation of it and its use. So the C++ only ever sees
    these copies (and, for the Clifford command, copies them once more before reading).
    """
    out = np.array(arr, dtype=dtype, order="C", copy=True)
    if out.shape != shape:
        raise ValueError(f"{what} must have shape {shape}, got {out.shape}")
    return out


_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
_FITS_I32 = frozenset(np.dtype(t) for t in (np.int8, np.int16, np.int32, np.uint8, np.uint16))
_NO_KICK = np.zeros(0, np.uint64)  # empty, so `addr` gives NULL; never written


def _int32_value(v: Any, what: str) -> int:
    """`v` as a Python int that fits an int32 (ctypes would wrap it silently)."""
    i = as_integer(v, what)
    if not _I32_MIN <= i <= _I32_MAX:
        raise ValueError(f"{what} must fit an int32, got {i}")
    return i


def _as_i32(arr: Any, shape: tuple[int, ...], what: str) -> Any:
    """`_as(arr, np.int32, ...)` that refuses anything but an integer that fits an int32.

    numpy's `astype`/`array(dtype=np.int32)` wraps an int64 column silently (2**32 + 1 would run
    as 1) and truncates a float one (NaN and inf become INT_MIN), so a non-integer dtype (float,
    complex, bool, object, ...) is a TypeError and an out-of-range integer a ValueError. An empty
    array of any dtype is accepted (`np.asarray([])` is float64); `compile_*` give int32.
    """
    out = np.array(arr, order="C", copy=True)  # read the caller's data once; check this copy
    if out.size and out.dtype.kind not in "iu":
        raise TypeError(f"{what} must be an integer array, got dtype {out.dtype}")
    fits = out.dtype in _FITS_I32  # every value of these dtypes fits: no range scan
    if out.size and not fits and (int(out.min()) < _I32_MIN or int(out.max()) > _I32_MAX):
        raise ValueError(f"{what} must fit an int32")
    if out.dtype == np.int32 and out.flags.aligned:  # `out` is already a private copy
        if out.shape != shape:
            raise ValueError(f"{what} must have shape {shape}, got {out.shape}")
        return out
    return _as(out, np.int32, shape, what)


def _list_indices(shot_list: Any, n_lists: int, batch: int) -> Any:
    """`shot_list` as a private int32 array: every value below 0 means "skip" (as the reference's
    `which >= 0`), every value from `n_lists` up becomes `n_lists` (the native refusal), so a
    wide integer is never wrapped into a valid list index by the int32 conversion."""
    if type(shot_list) is np.ndarray and shot_list.dtype == np.int32:
        own = _as(shot_list, np.int32, (batch,), "shot_list")  # int32 needs no narrowing
        np.clip(own, -1, n_lists, out=own)
        return own
    raw = np.array(shot_list, order="C", copy=True)
    if raw.shape != (batch,):
        raise ValueError(f"shot_list must have shape {(batch,)}, got {raw.shape}")
    if raw.dtype.kind == "u":
        return np.minimum(raw, np.uint64(n_lists)).astype(np.int32)
    wide = _as(raw, np.int64, (batch,), "shot_list")
    return np.clip(wide, -1, n_lists).astype(np.int32)


_FORK_OWNERS: weakref.WeakSet[Any] = weakref.WeakSet()  # objects a forked child must renew


def _owned_columns(
    table: FactorTable, nw: int
) -> tuple[int, int, tuple[Any, ...], tuple[Any, ...]]:
    """Private, checked copies of a factor table's columns and axes (the C++ reads only these)."""
    n_lists = int(table.has_p0.shape[0])
    t = (
        _as(table.has_p0, np.uint8, (n_lists,), "has_p0"),
        _as_i32(table.n_r, (n_lists,), "n_r"),
        _as_i32(table.v_fwd, (n_lists,), "v_fwd"),
        _as_i32(table.v_inv, (n_lists,), "v_inv"),
        _as_i32(table.axis_start, (n_lists,), "axis_start"),
    )
    n_axes = int(table.axes_kappa.shape[0])
    axes = (
        _as(table.axes_a, np.uint64, (n_axes, nw), "axes_a"),
        _as(table.axes_b, np.uint64, (n_axes, nw), "axes_b"),
        _as(table.axes_kappa, np.uint8, (n_axes,), "axes_kappa"),
    )
    return n_lists, n_axes, t, axes


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class OwnedFactorTable:
    """A factor table held by stimside alone: private copies made once by `own_factor_table`,
    read-only, never handed out. `inject_cliffords` uses it without copying it again (glue cut
    (b)); no caller array is ever behind it, so no caller can change it while the GIL is released.
    """

    cols: tuple[Any, ...]
    axes: tuple[Any, ...]
    n_words: int
    seal: object = None


# Module-private weak registry: only `own_factor_table` registers instances and stores their
# immutable buffers + n_words; `eq=False` prevents forged tables from matching, and storing the
# arrays here prevents `object.__setattr__(own, "cols"/"axes", ...)` from poisoning C++ calls.
_OWNED_TABLES: weakref.WeakKeyDictionary[
    OwnedFactorTable, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], int]
] = weakref.WeakKeyDictionary()


def _immutable(arr: Any) -> Any:
    """A copy of `arr` over a `bytes` buffer: numpy refuses to make it writable again (unlike
    an array that owns its data, whose `writeable` flag anyone can turn back on)."""
    out = np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)
    assert out.flags.c_contiguous and out.flags.aligned and not out.flags.writeable
    return out


def own_factor_table(table: FactorTable, nw: int) -> OwnedFactorTable:
    """Copy and check `table` once (as `inject_cliffords` would), into immutable buffers."""
    if type(nw) is not int or nw < 1:
        raise ValueError(f"n_words must be a positive int, got {nw!r}")
    _, _, cols, axes = _owned_columns(table, nw)
    owned = OwnedFactorTable(
        tuple(_immutable(a) for a in cols), tuple(_immutable(a) for a in axes), nw
    )
    _OWNED_TABLES[owned] = (owned.cols, owned.axes, owned.n_words)
    return owned


def track_locks_for_fork(owner: Any) -> None:
    """Have `owner._reinit_after_fork()` run in every forked child (it renews `owner._lock`).

    Weakly referenced. Only the child touches the table, and only after the fork, so the parent
    takes no lock and cannot deadlock; the table has no lock of its own (it could itself be
    inherited held) and a single set operation is atomic under the GIL.
    """
    _FORK_OWNERS.add(owner)


def _reinit_owners_in_child() -> None:
    for owner in list(_FORK_OWNERS):  # the forking thread is the only one: nothing changes it
        owner._reinit_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_owners_in_child)


class MagicBatchEngine:
    """Handle to a C++ batch engine holding `batch_size` shots over `n_qubits` qubits."""

    def __init__(self, batch_size: int, n_qubits: int, atol: float) -> None:
        self._handle: int | None = None
        # Per engine: independent engines never contend. Re-entrant only so that a thread that
        # already owns it (a handler, a finalizer) can be recognised by `_busy` and refused.
        self._lock = threading.RLock()
        self._busy = False  # a native section is running (only ever seen set by its own thread)
        self._close_pending = False  # `close()` was asked for from inside that section
        self._owner = 0  # id of the thread running that section
        self._poisoned = False  # forked while another thread ran a section: unusable here
        self._pending_token = 0  # token of the split `measure_finish` may use (0: none)
        batch = as_integer(batch_size, "batch_size")
        nq = as_integer(n_qubits, "n_qubits")
        if not (0 <= batch <= _MAX_SIZE and 1 <= nq <= _MAX_SIZE):
            raise ValueError(f"need 0 <= batch_size, 1 <= n_qubits < 2**63, got {batch}, {nq}")
        tol = _valid_atol(atol)
        self._lib = get_magic_engine_lib()
        handle = self._lib.mbe_create(batch, nq, tol)
        if not handle:
            raise MemoryError("mbe_create failed")
        self._handle = int(handle)
        self.batch_size, self.n_qubits, self.atol = batch, nq, tol
        self.n_words = max(1, (self.n_qubits + 63) // 64)
        track_locks_for_fork(self)

    def _reinit_after_fork(self) -> None:
        """Run in a forked child, which has only the forking thread: renew the lock and flags.

        An engine that another thread was inside a section of was being updated by a thread that
        is gone: its memory is a snapshot taken mid-update. It is poisoned (handle forgotten,
        never freed here). A section of the forking thread itself (a fork from a handler) is not
        inside a native call: it goes on, and its own exit clears the flags.
        """
        self._lock = threading.RLock()
        if self._busy and self._owner != threading.get_ident():
            self._poisoned, self._handle = True, None
            self._busy = self._close_pending = False

    @contextlib.contextmanager
    def _section(self) -> Iterator[None]:
        """Exclusive use of the native handle. Never re-entered by the thread that is inside.

        Python code can run on that thread between reading the handle and the native call (a
        signal handler, a finalizer): it gets a `RuntimeError`, and its `close()` is deferred to
        the end of the section. Other threads wait on the lock and never see `_busy` set.
        """
        with self._lock:
            nested = self._busy
            try:
                if nested:
                    raise RuntimeError("re-entrant call into a MagicBatchEngine native section")
                self._busy, self._owner = True, threading.get_ident()
                yield
            finally:
                if not nested:
                    self._busy = False
                    if self._close_pending:
                        self._close_pending = False
                        self._destroy()

    def _destroy(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._lib.mbe_destroy(handle)

    def close(self) -> None:
        """Free the native engine (idempotent; waits for a native call in another thread).

        From inside a native section of the calling thread (a handler or finalizer that fired
        there) the free is deferred until the section ends.
        """
        with self._lock:
            if self._busy:
                self._close_pending = True
            else:
                self._destroy()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:  # not after a failed __init__
            self.close()

    def __getstate__(self) -> None:
        raise TypeError("MagicBatchEngine owns native state and cannot be pickled or copied")

    def _h(self) -> int:
        if self._handle is None:
            if self._poisoned:
                raise RuntimeError(
                    "MagicBatchEngine was inside a native call when the process forked: its "
                    "state is not usable in the child"
                )
            raise RuntimeError("MagicBatchEngine is closed")
        return self._handle

    def _shot_index(self, b: int) -> int:
        i = as_integer(b, "shot index")
        if not 0 <= i < self.batch_size:
            raise IndexError(f"shot {i} out of range for batch_size={self.batch_size}")
        return i

    def reset(self) -> None:
        """Every shot back to `ShotExpansion.zero` (capacity is kept)."""
        with self._section():
            rc = self._lib.mbe_reset(self._h())
        if rc == 2:
            raise MemoryError("mbe_reset out of memory")
        if rc != 0:
            raise RuntimeError("mbe_reset failed")

    def import_shot(self, b: int, s: ShotArrays) -> None:
        """Copy shot `b` in. Array shapes are validated first: the C++ trusts them."""
        if not isinstance(s, ShotArrays):
            raise TypeError(f"expected ShotArrays, got {type(s).__name__}")
        b, nw = self._shot_index(b), self.n_words
        chi, d = as_integer(s.chi, "chi"), as_integer(s.d, "d")
        if not 0 <= chi <= _MAX_SIZE or not 0 <= d <= self.n_qubits:
            raise ValueError(f"invalid chi={chi}, d={d} for n_qubits={self.n_qubits}")
        atol = _valid_atol(s.atol)
        keep = (
            _as(s.weights, np.complex128, (chi,), "weights"),
            _as(s.h_words, np.uint64, (chi, nw), "h_words"),
            _as(s.ell, np.uint8, (chi, d), "ell"),
            _as(s.A_words, np.uint64, (d, nw), "A_words"),
            _as(s.a_union, np.uint64, (nw,), "a_union"),
            _as(s.Gamma, np.uint8, (d, d), "Gamma"),
            _as_i32(s.pivots, (d,), "pivots"),
            _as(s.ell_odd, np.uint8, (d,), "ell_odd"),
        )
        w, *rest = keep
        with self._section():
            rc = self._lib.mbe_import_shot(
                self._h(), b, chi, d, int(bool(s.canonical)), float(s.log_mass), atol,
                w_addr(w), *(addr(x) for x in rest),
            )
        del keep  # the arrays above must outlive the call
        if rc != 0:
            if rc == 2:
                raise MemoryError("mbe_import_shot out of memory")
            raise ValueError("mbe_import_shot rejected its arguments")

    def export_shot(self, b: int) -> ShotArrays:
        """Copy shot `b` out (fresh arrays; the engine keeps its own)."""
        b, nw = self._shot_index(b), self.n_words
        cd = np.zeros(3, dtype=np.uint64)
        lm = np.zeros(2, dtype=np.float64)
        with self._section():  # the sizes read from the header must still hold for the export
            if self._lib.mbe_shot_header(self._h(), b, addr(cd), addr(lm)) != 0:
                raise RuntimeError("mbe_shot_header failed")
            chi, d = int(cd[0]), int(cd[1])
            out = ShotArrays(
                chi, d, bool(cd[2]), float(lm[0]), float(lm[1]),
                np.empty(chi, dtype=np.complex128), np.empty((chi, nw), dtype=np.uint64),
                np.empty((chi, d), dtype=np.uint8), np.empty((d, nw), dtype=np.uint64),
                np.empty(nw, dtype=np.uint64), np.empty((d, d), dtype=np.uint8),
                np.empty(d, dtype=np.int32), np.empty(d, dtype=np.uint8),
            )
            rc = self._lib.mbe_export_shot(
                self._h(), b, w_addr(out.weights), addr(out.h_words), addr(out.ell),
                addr(out.A_words), addr(out.a_union), addr(out.Gamma), addr(out.pivots),
                addr(out.ell_odd),
            )
        if rc != 0:
            raise RuntimeError("mbe_export_shot failed")
        return out

    def apply_pauli_mask(self, a_words: Any, b_words: Any, kappa: int, mask: Any) -> None:
        """`P = i**kappa X(a) Z(b)` on every shot with `mask` set and chi > 0, in shot order."""
        nw = self.n_words
        a_w = _as(a_words, np.uint64, (nw,), "a_words")
        b_w = _as(b_words, np.uint64, (nw,), "b_words")
        _check_padding(a_w, self.n_qubits, "a_words")
        _check_padding(b_w, self.n_qubits, "b_words")
        m = _as(mask, np.bool_, (self.batch_size,), "mask").view(np.uint8)
        kap = int(kappa) & 3
        with self._section():
            bad = self._lib.mbe_apply_pauli_mask(self._h(), addr(a_w), addr(b_w), kap, addr(m))
        if bad >= 0:
            raise FloatingPointError("Non-finite weights in apply_pauli")
        if bad == -3:
            raise MemoryError("mbe_apply_pauli_mask out of memory (shots before it were updated)")
        if bad != -1:
            raise RuntimeError("mbe_apply_pauli_mask: invalid engine handle")

    def inject_cliffords(
        self, table: FactorTable | OwnedFactorTable, shot_list: Any, inverse: bool
    ) -> FactorFailure | None:
        """Apply factor list `shot_list[s]` (-1: none) to every shot, in shot order.

        What `apply_clifford_factors(..., check_unitary=True)` does per shot, committed shot by
        shot: on a failure the shots before it are updated, it and the later ones are not, and
        the failure is returned (the caller raises the reference exception). None on success.
        An `OwnedFactorTable` (built by `own_factor_table`) is used as is; any other table is
        copied first.
        """
        nw = self.n_words
        if isinstance(table, OwnedFactorTable):
            entry = None if type(table) is not OwnedFactorTable else _OWNED_TABLES.get(table)
            if entry is None or type(entry[2]) is not int or type(nw) is not int or entry[2] != nw:
                raise TypeError(
                    "an OwnedFactorTable comes only from own_factor_table(.., n_words)"
                )
            t, axes, _ = entry
            n_lists, n_axes = int(t[0].shape[0]), int(axes[2].shape[0])
        else:
            n_lists, n_axes, t, axes = _owned_columns(table, nw)
        shots = _list_indices(shot_list, n_lists, self.batch_size)
        err, errd = np.zeros(4, dtype=np.int64), np.zeros(2, dtype=np.float64)
        t0, t1, t2, t3, t4 = t
        a0, a1, a2 = axes
        with self._section():
            bad = self._lib.mbe_inject_cliffords(
                self._h(), n_lists, n_axes, addr(t0), addr(t1), addr(t2), addr(t3), addr(t4),
                addr(a0), addr(a1), addr(a2), addr(shots), int(bool(inverse)), addr(err),
                addr(errd),
            )
        if bad == -1:
            return None
        if bad == -3:  # the C++ reports the shot it was at: the ones before it are committed
            raise MemoryError(
                f"mbe_inject_cliffords ran out of memory at shot {int(err[2])}: the shots before "
                "it were updated, it and the later ones were not"
            )
        if bad < 0:
            raise RuntimeError(f"mbe_inject_cliffords failed before running ({bad})")
        return FactorFailure(int(bad), *(int(x) for x in err), *(float(x) for x in errd))

    def live_shots(self) -> int:
        """Number of shots with chi > 0."""
        with self._section():
            n = int(self._lib.mbe_live_shots(self._h()))
        if n < 0:
            raise RuntimeError("mbe_live_shots: invalid engine handle")
        return n

    def partition(self, dedupe: bool) -> Any:
        """`rep[s]` = lowest shot of shot s's group of bit-identical states, -1 when chi == 0
        (each live shot is its own group when `dedupe` is False)."""
        rep = np.empty(self.batch_size, dtype=np.int64)
        with self._section():
            rc = self._lib.mbe_partition(self._h(), int(bool(dedupe)), addr(rep))
        if rc == -3:
            raise MemoryError("mbe_partition out of memory")
        if rc != 0:
            raise RuntimeError(f"mbe_partition failed ({rc})")
        return rep

    def apply_block(
        self, block: BlockTable, dedupe: bool, max_components: int | None
    ) -> BlockResult:
        """`apply_compiled_block(check_unitary=True)` on every group of identical shots, the
        result copied to each member, then the `max_components` bound (see `mbe_apply_block`).

        On a failure the groups before the failing one are updated and it and the later ones
        are not (for `FAIL_MAX_COMPONENTS` the failing group is updated as well).
        """
        nw, n_axes = self.n_words, int(block.axes_kappa.shape[0])
        n_steps = int(block.step_kind.shape[0])
        t = (
            _as_i32(block.step_kind, (n_steps,), "step_kind"),
            _as_i32(block.step_sign, (n_steps,), "step_sign"),
            _as_i32(block.step_start, (n_steps,), "step_start"),
            _as(block.axes_a, np.uint64, (n_axes, nw), "axes_a"),
            _as(block.axes_b, np.uint64, (n_axes, nw), "axes_b"),
            _as(block.axes_kappa, np.uint8, (n_axes,), "axes_kappa"),
            _as(block.axes_w_ri, np.float64, (n_axes, 2), "axes_w_ri"),
        )
        mode = _int32_value(block.mode, "block mode")
        mc = _MAX_SIZE if max_components is None else as_integer(max_components, "max_components")
        mc = min(max(mc, -1), _MAX_SIZE)  # INT64_MAX: no bound; any negative bound: always hit
        err, errd = np.zeros(4, dtype=np.int64), np.zeros(2, dtype=np.float64)
        info = np.zeros(2, dtype=np.int64)
        with self._section():
            bad = self._lib.mbe_apply_block(
                self._h(), mode, n_steps, n_axes, float(block.g_re), float(block.g_im),
                *(addr(x) for x in t), int(bool(dedupe)), mc, addr(err), addr(errd), addr(info),
            )
        groups, live = int(info[0]), int(info[1])
        if bad == -1:
            return BlockResult(None, groups, live)
        if bad == -3:
            raise MemoryError("mbe_apply_block ran out of memory (groups before it were updated)")
        if bad < 0:
            raise RuntimeError(f"mbe_apply_block failed before running ({bad})")
        fail = FactorFailure(int(bad), *(int(x) for x in err), *(float(x) for x in errd))
        return BlockResult(fail, groups, live)


    def measure_split(
        self, obs: tuple[Any, Any, int], keep_rot: bool,
        reset: tuple[Any, Any, int, bool] | None,
    ) -> SplitResult:
        """First half of measuring the Pauli `obs = (a_words, b_words, kappa)` on every shot.

        Nothing of a shot changes except the reset kicks of trivial shots (`reset` = the kick's
        `(a_words, b_words, kappa)` and `reset_flip_ref`). Both projected branches of every shot
        that needs a Born draw are kept for `measure_finish` (see `mbe_measure_split`). The
        observable is not validated here: the engine does it, after classifying, as the
        reference does.
        """
        nw, batch = self.n_words, self.batch_size
        a_w, b_w = (_as(x, np.uint64, (nw,), n) for x, n in ((obs[0], "a"), (obs[1], "b")))
        k_a = k_b = _NO_KICK  # address 0: read only for a reset (the engine accepts NULL otherwise)
        k_kap, flip_ref = 0, False
        if reset is not None:
            k_a = _as(reset[0], np.uint64, (nw,), "kick a")
            k_b = _as(reset[1], np.uint64, (nw,), "kick b")
            k_kap, flip_ref = int(reset[2]) & 3, bool(reset[3])
        cls = np.full(batch, CLS_UNSEEN, dtype=np.uint8)
        bit = np.zeros(batch, dtype=np.uint8)
        z, z01 = np.zeros(batch), np.zeros((batch, 2))
        chi2 = np.zeros((batch, 2), dtype=np.int64)
        err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
        token = next(_SPLIT_TOKENS)
        with self._section():
            self._pending_token = 0  # the native call drops the previous split in every case
            rc = self._lib.mbe_measure_split(
                self._h(), addr(a_w), addr(b_w), int(obs[2]) & 3, int(bool(keep_rot)),
                int(reset is not None), int(flip_ref), addr(k_a), addr(k_b), k_kap, addr(cls),
                addr(bit), addr(z), addr(z01), addr(chi2), addr(err), addr(errd), addr(info),
            )
            if rc == 0 and info[0] > 0:  # the engine keeps this split for `measure_finish`
                self._pending_token = token
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_measure_split failed before running ({rc})")
        fail = None
        if rc == 1:
            fail = FactorFailure(int(info[1]), *(int(x) for x in err), *(float(x) for x in errd))
        return SplitResult(cls, bit, z, z01, chi2, int(info[0]), fail, rc == -3, token)

    def measure_finish(
        self, outcome: Any, chosen: Any, split: SplitResult | None = None
    ) -> FinishResult:
        """Second half: commit outcome `outcome[i]` (bool) of the i-th active shot of the last
        `measure_split`, whose chosen mass is `chosen[i]` (see `mbe_measure_finish`). Any other
        engine call in between drops the pending split (RuntimeError here).

        `split` is the `SplitResult` the outcomes were drawn from: when another split replaced
        it (another thread's, say) the call is refused with a RuntimeError before the engine is
        touched, so the outcomes of one split can never commit the branches of another. (Without
        `split` the outcomes are applied to whatever split is pending; the simulator always
        passes it.)
        """
        n = int(np.shape(outcome)[0])
        out = _as(np.asarray(outcome, dtype=np.bool_), np.uint8, (n,), "outcome")
        ch = _as(chosen, np.float64, (n,), "chosen")
        err, errd, info = np.zeros(4, np.int64), np.zeros(2), np.zeros(2, np.int64)
        with self._section():
            if split is not None and split.token != self._pending_token:
                raise RuntimeError(
                    "mbe_measure_finish: no pending measure_split matches this split (another "
                    "split replaced it, or it was already finished)"
                )
            self._pending_token = 0  # used up in every case, as in the engine
            rc = self._lib.mbe_measure_finish(
                self._h(), n, addr(out), addr(ch), addr(err), addr(errd), addr(info)
            )
        if rc == -3:
            raise MemoryError(
                "mbe_measure_finish ran out of memory "
                + ("(no shot was changed)" if info[0] == 1 else "(shots before it were committed)")
            )
        if rc == -2:
            raise RuntimeError("mbe_measure_finish: no pending measure_split for this engine")
        if rc == -4:
            raise ValueError("mbe_measure_finish: outcome / chosen do not match the active shots")
        if rc == 0:
            return FinishResult(None, 0)
        fail = FactorFailure(int(info[1]), *(int(x) for x in err), *(float(x) for x in errd))
        return FinishResult(fail, int(info[0]))

    def _born_args(self, obs: tuple[Any, Any, int], shots: Any) -> tuple[Any, Any, int, Any]:
        """The observable words and the list of shots as private arrays (the C++ reads these)."""
        nw = self.n_words
        a_w, b_w = (_as(x, np.uint64, (nw,), n) for x, n in ((obs[0], "a"), (obs[1], "b")))
        n = int(np.shape(shots)[0])
        return a_w, b_w, int(obs[2]) & 3, _as_i32(shots, (n,), "shots")

    @staticmethod
    def _born_failure(rc: int, info: Any, err: Any, errd: Any) -> FactorFailure | None:
        if rc != 1:
            return None
        return FactorFailure(int(info[0]), *(int(x) for x in err), *(float(x) for x in errd))

    def peek(self, obs: tuple[Any, Any, int], shots: Any) -> tuple[Any, FactorFailure | None]:
        """`_obs_p1` of `obs = (a_words, b_words, kappa)` on every listed shot (see `mbe_peek`).

        Returns (p1, failure): p1[i] belongs to shots[i] and is valid before `failure.shot`
        (a position in `shots`); nothing changes. An out-of-memory is a MemoryError.
        """
        a_w, b_w, kap, lst = self._born_args(obs, shots)
        p1 = np.zeros(lst.shape[0])
        err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(1, np.int64)
        with self._section():
            rc = self._lib.mbe_peek(
                self._h(), addr(a_w), addr(b_w), kap, addr(lst), lst.shape[0], addr(p1),
                addr(err), addr(errd), addr(info),
            )
        if rc == -3:
            raise MemoryError(f"mbe_peek ran out of memory at list entry {int(info[0])}")
        if rc not in (0, 1):
            raise RuntimeError(f"mbe_peek failed before running ({rc})")
        return p1, self._born_failure(rc, info, err, errd)

    def postselect(
        self, obs: tuple[Any, Any, int], shots: Any, desired: Any
    ) -> PostselectResult:
        """`postselect_z_batch`'s loop: shot shots[i] is projected onto bit desired[i] & 1 (dead
        shots skipped), in order, each committed before the next (see `mbe_postselect`)."""
        a_w, b_w, kap, lst = self._born_args(obs, shots)
        n = lst.shape[0]
        want = _as(np.asarray(desired, dtype=np.uint8), np.uint8, (n,), "desired")
        done, z01 = np.zeros(n, np.uint8), np.zeros((n, 2))
        err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(1, np.int64)
        with self._section():
            self._pending_token = 0  # the native call drops a pending split
            rc = self._lib.mbe_postselect(
                self._h(), addr(a_w), addr(b_w), kap, addr(lst), n, addr(want), addr(done),
                addr(z01), addr(err), addr(errd), addr(info),
            )
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_postselect failed before running ({rc})")
        oom = int(info[0]) if rc == -3 else None
        return PostselectResult(done, z01, self._born_failure(rc, info, err, errd), oom)

    def unmatched_apply(
        self, obs: tuple[Any, Any, int], kick: tuple[Any, Any, int], shots: Any,
        bits: Any, measure: Any, kick_flags: Any,
    ) -> UnmatchedResult:
        """Second half of an unmatched noisy reset: per listed shot, project onto `bits[i]` where
        `measure[i]`, then kick where `kick_flags[i]` and the shot is alive (see
        `mbe_unmatched_apply`)."""
        a_w, b_w, kap, lst = self._born_args(obs, shots)
        n, nw = lst.shape[0], self.n_words
        k_a, k_b = (_as(x, np.uint64, (nw,), n_) for x, n_ in ((kick[0], "ka"), (kick[1], "kb")))
        flags = tuple(_as(np.asarray(x, dtype=np.uint8), np.uint8, (n,), nm)
                      for x, nm in ((bits, "bits"), (measure, "measure"), (kick_flags, "kicks")))
        err, errd, info = np.zeros(4, np.int64), np.zeros(3), np.zeros(2, np.int64)
        with self._section():
            self._pending_token = 0
            rc = self._lib.mbe_unmatched_apply(
                self._h(), addr(a_w), addr(b_w), kap, addr(k_a), addr(k_b), int(kick[2]) & 3,
                addr(lst), n, *(addr(f) for f in flags), addr(err), addr(errd), addr(info),
            )
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_unmatched_apply failed before running ({rc})")
        fail = self._born_failure(rc, info, err, errd)
        return UnmatchedResult(fail, int(info[0]) if rc == -3 else None, int(info[1]))


_SELF_CHECK_FACTORS = (
    0.5 + 0.5j, -0.7071067811865476 + 0.7071067811865476j, 1j, -1.0 + 0.0j, 0.25 + 0.0j,
)


# The numpy side of each recipe the engine copies (module level, so that a test can substitute a
# deliberately different one and see the guard refuse the engine).
def _numpy_norm_sq(w: Any) -> Any:
    return np.float64(np.sum(w.real * w.real + w.imag * w.imag))


def _numpy_form_a(w: Any, c: complex) -> Any:
    """`w * c`: array times a Python complex (the omega multiply; `weights *= g`, n >= 2)."""
    return w * c


def _numpy_form_b(w: Any, c: complex) -> Any:
    """`np.complex128(c) * w`: the scalar on the left (`coeffs[m] * weights`)."""
    return np.complex128(c) * w


def _numpy_in_place(one: Any, c: complex) -> Any:
    """`w *= c` on a one-element array, which numpy computes unfused."""
    out = one.copy()
    out *= c
    return out


@functools.cache
def numerics_match_numpy() -> bool:
    """True when the engine's complex multiplies and norm_sq reproduce numpy bit for bit.

    numpy's FMA contraction and pairwise summation are properties of the numpy build and the
    CPU, so the engine checks them once on fixed inputs (with zeros and -0.0) before use: the
    pairwise norm, form A (`w * c`), form B (`np.complex128(c) * w`) and the unfused in-place
    multiply of a single element. False too when the library cannot be loaded or built (the
    caller then keeps the Python path).
    """
    try:
        lib = get_magic_engine_lib()
    except (OSError, RuntimeError):  # missing/unloadable library, or the rebuild failed
        return False

    def same(got: Any, ref: Any) -> bool:
        return bool(np.array_equal(got.view(np.uint64), np.asarray(ref).view(np.uint64)))

    rng = np.random.default_rng(7)
    for n in (1, 3, 8, 9, 64, 130, 4097):
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w[::5], w[1::7] = complex(0.0, -0.0), complex(-0.0, 0.0)
        got = np.float64(lib.mbe_norm_sq(n, w_addr(w)))
        if not same(got, _numpy_norm_sq(w)):
            return False
        for c in _SELF_CHECK_FACTORS:
            for form, ref in ((0, _numpy_form_a(w, c)), (1, _numpy_form_b(w, c))):
                mine = w.copy()
                lib.mbe_cmul_form(n, w_addr(mine), c.real, c.imag, form)
                if not same(mine, ref):
                    return False
            for k in range(min(n, 12)):  # the single-element in-place quirk (form 2)
                one = w[k : k + 1].copy()
                lib.mbe_cmul_form(1, w_addr(one), c.real, c.imag, 2)
                if not same(one, _numpy_in_place(w[k : k + 1], c)):
                    return False
    return True
