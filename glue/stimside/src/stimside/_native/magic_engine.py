"""ctypes ABI and handle wrapper for the `mbe_*` MagicBatchEngine entry points.

Array-only wrapper around the native batch engine in `libmagic_kernels.so` (`_native` may not
import `_magic`). `export_shot` returns fresh arrays and `import_shot` copies in. Every native
call runs inside a per-engine non-re-entrant section (`_section()`: `RLock` + `_busy`; nested
`close()` is deferred; an interrupt as it is entered leaves the engine usable once the exception
is released). After `os.fork()`, the child renews locks and poisons any engine that another thread
was inside at fork time. All caller inputs are validated and copied (or immutable) before C++ runs.
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
_I32, _I64, _INT, _U8 = ctypes.c_int32, ctypes.c_int64, ctypes.c_int, ctypes.c_uint8
_U8_DT, _U64_DT, _I32_DT = np.dtype(np.uint8), np.dtype(np.uint64), np.dtype(np.int32)
_F64_DT, _BOOL_DT = np.dtype(np.float64), np.dtype(np.bool_)

# Pointer parameters are c_void_p, as for the magic_* kernels: callers pass integer addresses.
_SPECS: tuple[AbiSpec, ...] = (
    *((f"mbe_abi_v{v}", [], _I32) for v in range(1, 7)),
    ("mbe_live_engines", [], _SZ),
    ("mbe_create", [_SZ, _SZ, _DBL], _VP),
    ("mbe_destroy", [_VP], None),
    ("mbe_reset", [_VP], _INT),
    ("mbe_import_shot", [_VP, _SZ, _SZ, _SZ, _I32, _DBL, _DBL, *[_VP] * 8], _INT),
    ("mbe_shot_header", [_VP, _SZ, _VP, _VP], _INT),
    ("mbe_export_shot", [_VP, _SZ, *[_VP] * 8], _INT),
    ("mbe_apply_pauli_mask", [_VP, _VP, _VP, _U8, _VP], _I64),
    ("mbe_apply_pauli_noise_batch", [_VP, _SZ, _SZ, _SZ, *[_VP] * 9], _I64),
    ("mbe_inject_cliffords", [_VP, _SZ, _SZ, *[_VP] * 9, _I32, _VP, _VP], _I64),
    ("mbe_live_shots", [_VP], _I64),
    ("mbe_partition", [_VP, _I32, _VP], _INT),
    ("mbe_apply_block", [_VP, _I32, _SZ, _SZ, _DBL, _DBL, *[_VP] * 7, _I32, _I64, *[_VP] * 3],
     _I64),
    ("mbe_measure_split", [_VP, _VP, _VP, _U8, _I32, _I32, _I32, _VP, _VP, _U8, *[_VP] * 8], _I64),
    ("mbe_measure_finish", [_VP, _SZ, *[_VP] * 5], _I64),
    ("mbe_measure_step",
     [_VP, _SZ, _SZ, _VP, _VP, _U8, _I32, _I32, _I32, _VP, _VP, _U8, *[_VP] * 11], _I64),
    ("mbe_peek", [_VP, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 4], _I64),
    ("mbe_postselect", [_VP, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 6], _I64),
    ("mbe_unmatched_apply", [_VP, _VP, _VP, _U8, _VP, _VP, _U8, _VP, _SZ, *[_VP] * 6], _I64),
    ("mbe_cmul_scalar", [_SZ, _VP, _DBL, _DBL], None),
    ("mbe_cmul_form", [_SZ, _VP, _DBL, _DBL, _I32], None),
    ("mbe_norm_sq", [_SZ, _VP], _DBL),
)


def _resolve_magic_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "magic_kernels", force_recompile)


def get_magic_engine_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """The shared library with the `mbe_*` ABI bound (compiled on first use if missing/stale)."""
    return get_lib(
        sys.modules[__name__], "magic_kernels", "mbe_abi_v6", _SPECS,
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
    """Compiled Clifford factor lists in the engine's flat layout (`p0` first, then `r` axes)."""

    has_p0: Any  # uint8 (L,)
    n_r: Any  # int32 (L,)
    v_fwd: Any  # int32 (L,)
    v_inv: Any  # int32 (L,)
    axis_start: Any  # int32 (L,)
    axes_a: Any  # uint64 (A, nw)
    axes_b: Any  # uint64 (A, nw)
    axes_kappa: Any  # uint8 (A,)


class FactorFailure(NamedTuple):
    """Failure details reported by an engine command at `shot`."""

    shot: int
    kind: int
    op: int
    aux: int
    aux2: int
    pre: float
    post: float
    z: float = 0.0


FAIL_NON_FINITE_WEIGHTS, FAIL_NON_FINITE_NORM, FAIL_KERNEL = 1, 2, 3
FAIL_UNITARITY, FAIL_NOT_HERMITIAN, FAIL_PADDING = 4, 5, 6
FAIL_MERGE_NON_FINITE, FAIL_MAX_COMPONENTS = 7, 8
FAIL_BAD_NORM, FAIL_ROTATION, FAIL_ZERO_CHOSEN = 9, 10, 11
FAIL_NON_FINITE_MASS, FAIL_NEGATIVE_MASS, FAIL_CONSERVATION, FAIL_ZERO_POSTSELECT = 12, 13, 14, 15
CLS_DEAD, CLS_TRIVIAL, CLS_ACTIVE, CLS_UNSEEN = 0, 1, 2, 3
OP_NAMES = (
    "apply_pauli", "apply_quarter_turn", "apply_pauli_t", "apply_ccz", "apply_compiled_block",
    "merge_frames", "project_diagonal",
)


class BlockTable(NamedTuple):
    """Compiled block in engine layout (`mode` 0: pauli_sum; `mode` 1: sequential steps)."""

    mode: int
    g_re: float
    g_im: float
    step_kind: Any
    step_sign: Any
    step_start: Any
    axes_a: Any
    axes_b: Any
    axes_kappa: Any
    axes_w_ri: Any


class BlockResult(NamedTuple):
    """`apply_block` outcome: optional failure plus (groups, live) counts."""

    failure: FactorFailure | None
    groups: int
    live: int


class SplitResult(NamedTuple):
    """`measure_split` outcome with per-shot class, trivial bit, active masses, and split token."""

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
    """`measure_finish` outcome (`phase` 1: staging failed; 2: reset kick failed)."""

    failure: FactorFailure | None
    phase: int


class PostselectResult(NamedTuple):
    """`postselect` outcome: `done`, `z01`, optional failure, and optional OOM list index."""

    done: Any
    z01: Any
    failure: FactorFailure | None
    oom_at: int | None


class UnmatchedResult(NamedTuple):
    """`unmatched_apply` outcome: optional failure, optional OOM index, and failure phase."""

    failure: FactorFailure | None
    oom_at: int | None
    phase: int


_MAX_SIZE = 2**63 - 1  # ctypes would wrap a size_t argument at 2**64
_SPLIT_TOKENS = itertools.count(1)  # one per `measure_split`, across all engines (atomic `next`)


def addr(arr: Any) -> int:
    """Buffer address; refuses unaligned or non-C-contiguous arrays."""
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
    """Private, aligned, C-contiguous copy of `arr` with dtype and shape verified."""
    out = np.array(arr, dtype=dtype, order="C", copy=True)
    if out.shape != shape:
        raise ValueError(f"{what} must have shape {shape}, got {out.shape}")
    return out


_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
_FITS_I32 = frozenset(np.dtype(t) for t in (np.int8, np.int16, np.int32, np.uint8, np.uint16))


def _int32_value(v: Any, what: str) -> int:
    """`v` as a Python int that fits an int32 (ctypes would wrap it silently)."""
    i = as_integer(v, what)
    if not _I32_MIN <= i <= _I32_MAX:
        raise ValueError(f"{what} must fit an int32, got {i}")
    return i


def _as_i32(arr: Any, shape: tuple[int, ...], what: str) -> Any:
    """Private int32 copy refusing non-integer dtypes and out-of-range values."""
    out = np.array(arr, order="C", copy=True)
    if out.size and out.dtype.kind not in "iu":
        raise TypeError(f"{what} must be an integer array, got dtype {out.dtype}")
    fits = out.dtype in _FITS_I32
    if out.size and not fits and (int(out.min()) < _I32_MIN or int(out.max()) > _I32_MAX):
        raise ValueError(f"{what} must fit an int32")
    if out.dtype == np.int32 and out.flags.aligned:
        if out.shape != shape:
            raise ValueError(f"{what} must have shape {shape}, got {out.shape}")
        return out
    return _as(out, np.int32, shape, what)


def _list_indices(shot_list: Any, n_lists: int, batch: int) -> Any:
    """Private int32 shot list clipped to `[-1, n_lists]` so wide ints cannot wrap."""
    if type(shot_list) is np.ndarray and shot_list.dtype == np.int32:
        own = _as(shot_list, np.int32, (batch,), "shot_list")
        np.clip(own, -1, n_lists, out=own)
        return own
    raw = np.array(shot_list, order="C", copy=True)
    if raw.shape != (batch,):
        raise ValueError(f"shot_list must have shape {(batch,)}, got {raw.shape}")
    if raw.dtype.kind == "u":
        return np.minimum(raw, np.uint64(n_lists)).astype(np.int32)
    return np.clip(_as(raw, np.int64, (batch,), "shot_list"), -1, n_lists).astype(np.int32)


_FORK_OWNERS: weakref.WeakSet[Any] = weakref.WeakSet()  # objects a forked child must renew


def _owned_columns(
    table: FactorTable, nw: int
) -> tuple[int, int, tuple[Any, ...], tuple[Any, ...]]:
    """Private, checked copies of a factor table's columns and axes (the C++ reads only these)."""
    nl, na = int(table.has_p0.shape[0]), int(table.axes_kappa.shape[0])
    t = (
        _as(table.has_p0, _U8_DT, (nl,), "has_p0"), _as_i32(table.n_r, (nl,), "n_r"),
        _as_i32(table.v_fwd, (nl,), "v_fwd"), _as_i32(table.v_inv, (nl,), "v_inv"),
        _as_i32(table.axis_start, (nl,), "axis_start"),
    )
    axes = (
        _as(table.axes_a, _U64_DT, (na, nw), "axes_a"),
        _as(table.axes_b, _U64_DT, (na, nw), "axes_b"),
        _as(table.axes_kappa, _U8_DT, (na,), "axes_kappa"),
    )
    return nl, na, t, axes


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class OwnedFactorTable:
    """Immutable factor table held by stimside alone: copied once by `own_factor_table`."""

    cols: tuple[Any, ...]
    axes: tuple[Any, ...]
    n_words: int
    seal: object = None


# Weak registries mapping owned handles to immutable buffers/n_words (immune to setattr/forging).
_OWNED_TABLES: weakref.WeakKeyDictionary[
    OwnedFactorTable, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], int]
] = weakref.WeakKeyDictionary()
_OWNED_TABLE_ADDRS: weakref.WeakKeyDictionary[
    OwnedFactorTable, tuple[tuple[Any, ...], tuple[Any, ...], int, int, tuple[int, ...]]
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class OwnedAxis:
    """Immutable Pauli axis held by stimside alone: copied once by `own_axis`, never mutated."""

    a_words: Any
    b_words: Any
    kappa: int
    n_words: int


_OWNED_AXES: weakref.WeakKeyDictionary[
    OwnedAxis, tuple[Any, Any, int, int, int, int]
] = weakref.WeakKeyDictionary()
_Axis = OwnedAxis | tuple[Any, Any, int]


def _immutable(arr: Any) -> Any:
    """A copy of `arr` over a `bytes` buffer: numpy refuses to make it writable again."""
    out = np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)
    assert out.flags.c_contiguous and out.flags.aligned and not out.flags.writeable
    return out


def own_factor_table(table: FactorTable, nw: int) -> OwnedFactorTable:
    """Copy and check `table` once (as `inject_cliffords` would), into immutable buffers."""
    if type(nw) is not int or nw < 1:
        raise ValueError(f"n_words must be a positive int, got {nw!r}")
    n_lists, n_axes, cols, axes = _owned_columns(table, nw)
    owned = OwnedFactorTable(
        tuple(_immutable(a) for a in cols), tuple(_immutable(a) for a in axes), nw
    )
    _OWNED_TABLES[owned] = (owned.cols, owned.axes, owned.n_words)
    _OWNED_TABLE_ADDRS[owned] = (
        owned.cols, owned.axes, n_lists, n_axes,
        tuple(addr(x) for x in (*owned.cols, *owned.axes)),
    )
    return owned


def own_axis(a_words: Any, b_words: Any, kappa: int, nw: int) -> OwnedAxis:
    """Copy and check a Pauli axis once into immutable bytes-backed buffers with cached pointers."""
    if type(nw) is not int or nw < 1:
        raise ValueError(f"n_words must be a positive int, got {nw!r}")
    a_imm = _immutable(_as(a_words, np.uint64, (nw,), "a_words"))
    b_imm = _immutable(_as(b_words, np.uint64, (nw,), "b_words"))
    kap = int(kappa) & 3
    owned = OwnedAxis(a_imm, b_imm, kap, nw)
    _OWNED_AXES[owned] = (a_imm, b_imm, addr(a_imm), addr(b_imm), kap, nw)
    return owned


def track_locks_for_fork(owner: Any) -> None:
    """Have `owner._reinit_after_fork()` run in every forked child (it renews `owner._lock`)."""
    _FORK_OWNERS.add(owner)


def _reinit_owners_in_child() -> None:
    for owner in list(_FORK_OWNERS):
        owner._reinit_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_owners_in_child)


class MagicBatchEngine:
    """Handle to a C++ batch engine holding `batch_size` shots over `n_qubits` qubits."""

    def __init__(self, batch_size: int, n_qubits: int, atol: float) -> None:
        self._handle: int | None = None
        self._lock = threading.RLock()
        self._busy = self._close_pending = self._poisoned = False
        self._owner = self._pending_token = 0
        batch, nq = as_integer(batch_size, "batch_size"), as_integer(n_qubits, "n_qubits")
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
        self._init_buffers()
        track_locks_for_fork(self)

    def _init_buffers(self) -> None:
        b, nw = self.batch_size, self.n_words
        self._err, self._errd = np.zeros(4, np.int64), np.zeros(3, _F64_DT)
        self._info, self._cd, self._lm = np.zeros(2, np.int64), np.zeros(3, _U64_DT), np.zeros(2)
        self._obs_a, self._obs_b = np.zeros(nw, _U64_DT), np.zeros(nw, _U64_DT)
        self._kick_a, self._kick_b = np.zeros(nw, _U64_DT), np.zeros(nw, _U64_DT)
        self._mask_u8, self._shots_i32 = np.zeros(b, _U8_DT), np.zeros(b, _I32_DT)
        self._outc_u8, self._chosen_f64 = np.zeros(b, _U8_DT), np.zeros(b, _F64_DT)
        self._ms_cls, self._ms_bit = np.empty(b, _U8_DT), np.empty(b, _U8_DT)
        self._ms_z, self._ms_z01 = np.empty(b, _F64_DT), np.empty((b, 2), _F64_DT)
        self._ms_chi2, self._ms_dlog = np.empty((b, 2), np.int64), np.empty(b, _F64_DT)
        self._err_a, self._errd_a = addr(self._err), addr(self._errd)
        self._info_a, self._cd_a, self._lm_a = addr(self._info), addr(self._cd), addr(self._lm)
        self._obs_a_a, self._obs_b_a = addr(self._obs_a), addr(self._obs_b)
        self._kick_a_a, self._kick_b_a = addr(self._kick_a), addr(self._kick_b)
        self._mask_a, self._shots_a = addr(self._mask_u8), addr(self._shots_i32)
        self._outc_a, self._chosen_a = addr(self._outc_u8), addr(self._chosen_f64)
        ms = (self._ms_cls, self._ms_bit, self._ms_z, self._ms_z01, self._ms_chi2, self._ms_dlog)
        self._ms_a = tuple(addr(x) for x in ms)
        self._noise_bufs: tuple[tuple[Any, ...], tuple[int, ...]] | None = None

    def _reinit_after_fork(self) -> None:
        """Run in a forked child: renew the lock and poison engines held by other threads."""
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

        The `try`/`finally` is inside the generator, so a `KeyboardInterrupt` (or another
        exception from a signal handler) raised as the section is entered does not leave `_busy`
        set for good: once the exception is released, the generator is closed and its `finally`
        runs. A hand-written `__enter__`/`__exit__` pair has no such fallback: an exception raised
        after `__enter__` set `_busy` and before the `with` body starts skips `__exit__` for good.
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

    def _fail(self, shot: int, n_d: int = 2) -> FactorFailure:
        e, d = self._err, self._errd
        return FactorFailure(
            shot, int(e[0]), int(e[1]), int(e[2]), int(e[3]),
            float(d[0]), float(d[1]), float(d[2]) if n_d == 3 else 0.0,
        )

    @staticmethod
    def _copy_axis(fa: bool, fb: bool, da: Any, db: Any, a: Any, b: Any) -> None:
        if fa:
            np.copyto(da, a)
        if fb:
            np.copyto(db, b)

    def _copy_obs_kick(self, obs: tuple[Any, ...], kick: tuple[Any, ...]) -> None:
        self._copy_axis(obs[5], obs[6], self._obs_a, self._obs_b, obs[0], obs[1])
        self._copy_axis(kick[5], kick[6], self._kick_a, self._kick_b, kick[0], kick[1])

    def _prep_axis(
        self, obs: _Axis, kick: bool = False, na: str = "a", nb: str = "b"
    ) -> tuple[Any, Any, int, int, int, bool, bool]:
        nw = self.n_words
        if isinstance(obs, OwnedAxis):
            ent = None if type(obs) is not OwnedAxis else _OWNED_AXES.get(obs)
            if ent is None or type(ent[5]) is not int or type(nw) is not int or ent[5] != nw:
                raise TypeError("an OwnedAxis comes only from own_axis(.., n_words)")
            return ent[0], ent[1], ent[2], ent[3], ent[4], False, False
        da, db = (self._kick_a, self._kick_b) if kick else (self._obs_a, self._obs_b)
        da_a, db_a = (self._kick_a_a, self._kick_b_a) if kick else (self._obs_a_a, self._obs_b_a)
        ra, rb, rk = obs
        kap, sh = int(rk) & 3, (nw,)
        fa = type(ra) is np.ndarray and ra.dtype == _U64_DT and ra.shape == sh == da.shape
        fb = type(rb) is np.ndarray and rb.dtype == _U64_DT and rb.shape == sh == db.shape
        ab = ra if fa else _as(ra, np.uint64, sh, na)
        bb = rb if fb else _as(rb, np.uint64, sh, nb)
        return ab, bb, da_a if fa else addr(ab), db_a if fb else addr(bb), kap, fa, fb

    def _prep_shots(self, shots: Any) -> tuple[Any, int, int, bool]:
        s, n, cap = shots, int(np.shape(shots)[0]), self._shots_i32.shape[0]
        fs = type(s) is np.ndarray and s.dtype == _I32_DT and s.shape == (n,) and n <= cap
        buf = s if fs else _as_i32(s, (n,), "shots")
        return buf, n, (self._shots_a if n else 0) if fs else addr(buf), fs

    def _destroy(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._lib.mbe_destroy(handle)

    def close(self) -> None:
        """Free the native engine (idempotent; defers destroy if called inside a section)."""
        with self._lock:
            if self._busy:
                self._close_pending = True
            else:
                self._destroy()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self.close()

    def __getstate__(self) -> None:
        raise TypeError("MagicBatchEngine owns native state and cannot be pickled or copied")

    def _h(self) -> int:
        if self._handle is None:
            if self._poisoned:
                raise RuntimeError("MagicBatchEngine was inside a native call when the process "
                                   "forked: its state is not usable in the child")
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
            _as(s.h_words, _U64_DT, (chi, nw), "h_words"), _as(s.ell, _U8_DT, (chi, d), "ell"),
            _as(s.A_words, _U64_DT, (d, nw), "A_words"), _as(s.a_union, _U64_DT, (nw,), "a_union"),
            _as(s.Gamma, _U8_DT, (d, d), "Gamma"), _as_i32(s.pivots, (d,), "pivots"),
            _as(s.ell_odd, _U8_DT, (d,), "ell_odd"),
        )
        w, *rest = keep
        with self._section():
            rc = self._lib.mbe_import_shot(
                self._h(), b, chi, d, int(bool(s.canonical)), float(s.log_mass), atol,
                w_addr(w), *(addr(x) for x in rest),
            )
        del keep
        if rc == 2:
            raise MemoryError("mbe_import_shot out of memory")
        if rc != 0:
            raise ValueError("mbe_import_shot rejected its arguments")

    def export_shot(self, b: int) -> ShotArrays:
        """Copy shot `b` out (fresh arrays; the engine keeps its own)."""
        b, nw = self._shot_index(b), self.n_words
        with self._section():
            if self._lib.mbe_shot_header(self._h(), b, self._cd_a, self._lm_a) != 0:
                raise RuntimeError("mbe_shot_header failed")
            chi, d = int(self._cd[0]), int(self._cd[1])
            out = ShotArrays(
                chi, d, bool(self._cd[2]), float(self._lm[0]), float(self._lm[1]),
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

    def _apply_pauli_mask_obs(self, obs: _Axis, mask: Any) -> None:
        m, b, mu = mask, self.batch_size, self._mask_u8
        ab, bb, aa, ba, kap, fa, fb = self._prep_axis(obs, False, "a_words", "b_words")
        fm = type(m) is np.ndarray and m.dtype == _BOOL_DT and m.shape == (b,) == mu.shape
        mb = m if fm else _as(m, np.bool_, (b,), "mask").view(np.uint8)
        ma = self._mask_a if fm else addr(mb)
        with self._section():
            self._copy_axis(fa, fb, self._obs_a, self._obs_b, ab, bb)
            if fm:
                np.copyto(self._mask_u8, mb)
            _check_padding(self._obs_a if fa else ab, self.n_qubits, "a_words")
            _check_padding(self._obs_b if fb else bb, self.n_qubits, "b_words")
            bad = self._lib.mbe_apply_pauli_mask(self._h(), aa, ba, kap, ma)
        if bad >= 0:
            raise FloatingPointError("Non-finite weights in apply_pauli")
        if bad == -3:
            raise MemoryError("mbe_apply_pauli_mask out of memory (shots before it were updated)")
        if bad != -1:
            raise RuntimeError("mbe_apply_pauli_mask: invalid engine handle")

    def apply_pauli_mask(self, a_words: Any, b_words: Any, kappa: int, mask: Any) -> None:
        """`P = i**kappa X(a) Z(b)` on every shot with `mask` set and chi > 0, in shot order."""
        self._apply_pauli_mask_obs((a_words, b_words, kappa), mask)

    def apply_pauli_noise_batch(
        self, x2x: Any, x2z: Any, x_kap: Any, z2x: Any, z2z: Any, z_kap: Any, xm: Any, zm: Any,
    ) -> None:
        """Apply batch X-then-Z Pauli noise masks pulled back through `(x2x..z_kap)`."""
        nq, b, nw = self.n_qubits, self.batch_size, self.n_words
        specs = (
            (x2x, _U64_DT, (nq, nw), "x2x"), (x2z, _U64_DT, (nq, nw), "x2z"),
            (x_kap, _U8_DT, (nq,), "x_kap"), (z2x, _U64_DT, (nq, nw), "z2x"),
            (z2z, _U64_DT, (nq, nw), "z2z"), (z_kap, _U8_DT, (nq,), "z_kap"),
            (xm, _BOOL_DT, (nq, b), "xm"), (zm, _BOOL_DT, (nq, b), "zm"),
        )
        fast = (nq * b <= (1 << 20) and b == self._mask_u8.shape[0] and nw == self._obs_a.shape[0]
                and all(type(a) is np.ndarray and a.dtype == dt and a.shape == sh
                        for a, dt, sh, _ in specs))
        copies = None if fast else tuple(_as(a, dt, sh, nm) for a, dt, sh, nm in specs)
        addrs = None if fast else tuple(addr(c) for c in copies)  # type: ignore[union-attr]
        with self._section():
            if fast:
                nb = self._noise_bufs
                if nb is None or nb[0][0].shape != (nq, nw) or nb[0][6].shape != (nq, b):
                    bufs = tuple(np.empty(sh, dt) for _, dt, sh, _ in specs)
                    nb = self._noise_bufs = (bufs, tuple(addr(c) for c in bufs))
                for dst, (src, _, _, _) in zip(nb[0], specs):
                    np.copyto(dst, src)
                addrs = nb[1]
            assert addrs is not None
            self._pending_token = 0
            bad = self._lib.mbe_apply_pauli_noise_batch(self._h(), nq, b, nw, *addrs, self._err_a)
            ek, ea1, ea2 = (int(self._err[0]), int(self._err[2]), int(self._err[3]))
        del copies
        if bad >= 0:
            if ek == FAIL_NOT_HERMITIAN:
                raise ValueError(
                    f"Pauli operator P(a, b, kappa={ea2}) must be Hermitian (<a,b>={ea1})"
                )
            if ek == FAIL_PADDING:
                raise ValueError(
                    f"Non-zero upper padding bits in packed uint64 word for n_qubits={nq}"
                )
            raise FloatingPointError("Non-finite weights in apply_pauli")
        if bad == -3:
            raise MemoryError("mbe_apply_pauli_noise_batch out of memory (shots before updated)")
        if bad != -1:
            raise RuntimeError(f"mbe_apply_pauli_noise_batch failed before running ({bad})")

    def inject_cliffords(
        self, table: FactorTable | OwnedFactorTable, shot_list: Any, inverse: bool
    ) -> FactorFailure | None:
        """Apply factor list `shot_list[s]` (-1: none) to every shot, in shot order."""
        nw, b = self.n_words, self.batch_size
        addrs: tuple[int, ...] | None = None
        if isinstance(table, OwnedFactorTable):
            entry = None if type(table) is not OwnedFactorTable else _OWNED_TABLES.get(table)
            if entry is None or type(entry[2]) is not int or type(nw) is not int or entry[2] != nw:
                raise TypeError("an OwnedFactorTable comes only from own_factor_table(.., n_words)")
            t, axes, _ = entry
            cached = _OWNED_TABLE_ADDRS.get(table)
            if cached is not None and cached[0] is t and cached[1] is axes:
                _, _, n_lists, n_axes, addrs = cached
            else:
                n_lists, n_axes = int(t[0].shape[0]), int(axes[2].shape[0])
        else:
            n_lists, n_axes, t, axes = _owned_columns(table, nw)
        s, sb = shot_list, self._shots_i32
        fs = type(s) is np.ndarray and s.dtype == _I32_DT and s.shape == (b,) == sb.shape
        shots = None if fs else _list_indices(s, n_lists, b)
        if addrs is None:
            addrs = tuple(addr(x) for x in (*t, *axes))
        sa, inv = self._shots_a if fs else addr(shots), int(bool(inverse))
        with self._section():
            if fs:
                np.copyto(self._shots_i32, s)
            bad = self._lib.mbe_inject_cliffords(
                self._h(), n_lists, n_axes, *addrs, sa, inv, self._err_a, self._errd_a
            )
            if bad == -1:
                return None
            err_s = int(self._err[2]) if bad == -3 else 0
            fail = self._fail(bad) if bad >= 0 else None
        if bad == -3:
            raise MemoryError(f"mbe_inject_cliffords ran out of memory at shot {err_s}: the shots "
                              "before it were updated, it and the later ones were not")
        if bad < 0:
            raise RuntimeError(f"mbe_inject_cliffords failed before running ({bad})")
        return fail

    def live_shots(self) -> int:
        """Number of shots with chi > 0."""
        with self._section():
            n = int(self._lib.mbe_live_shots(self._h()))
        if n < 0:
            raise RuntimeError("mbe_live_shots: invalid engine handle")
        return n

    def partition(self, dedupe: bool) -> Any:
        """`rep[s]` = lowest shot of shot s's group of bit-identical states, -1 when chi == 0."""
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
        """`apply_compiled_block(check_unitary=True)` per group of identical shots."""
        nw, n_axes = self.n_words, int(block.axes_kappa.shape[0])
        n_steps = int(block.step_kind.shape[0])
        t = (
            _as_i32(block.step_kind, (n_steps,), "step_kind"),
            _as_i32(block.step_sign, (n_steps,), "step_sign"),
            _as_i32(block.step_start, (n_steps,), "step_start"),
            _as(block.axes_a, _U64_DT, (n_axes, nw), "axes_a"),
            _as(block.axes_b, _U64_DT, (n_axes, nw), "axes_b"),
            _as(block.axes_kappa, _U8_DT, (n_axes,), "axes_kappa"),
            _as(block.axes_w_ri, _F64_DT, (n_axes, 2), "axes_w_ri"),
        )
        mode = _int32_value(block.mode, "block mode")
        mc = _MAX_SIZE if max_components is None else as_integer(max_components, "max_components")
        mc = min(max(mc, -1), _MAX_SIZE)
        g_re, g_im, dd = float(block.g_re), float(block.g_im), int(bool(dedupe))
        with self._section():
            bad = self._lib.mbe_apply_block(
                self._h(), mode, n_steps, n_axes, g_re, g_im, *(addr(x) for x in t), dd, mc,
                self._err_a, self._errd_a, self._info_a,
            )
            groups, live = int(self._info[0]), int(self._info[1])
            fail = self._fail(bad) if bad >= 0 else None
        if bad == -1:
            return BlockResult(None, groups, live)
        if bad == -3:
            raise MemoryError("mbe_apply_block ran out of memory (groups before it were updated)")
        if bad < 0:
            raise RuntimeError(f"mbe_apply_block failed before running ({bad})")
        return BlockResult(fail, groups, live)

    def _prep_reset(
        self, reset: tuple[Any, ...] | None
    ) -> tuple[Any, Any, int, int, int, bool, bool, bool]:
        if reset is None:
            return None, None, 0, 0, 0, False, False, False
        ko, flip = (reset[0], bool(reset[1])) if len(reset) == 2 else (reset[:3], bool(reset[3]))
        return (*self._prep_axis(ko, True, "kick a", "kick b"), flip)

    def measure_split(
        self, obs: _Axis, keep_rot: bool, reset: tuple[Any, ...] | None
    ) -> SplitResult:
        """First half of measuring `obs` on every shot (fresh non-aliasing result arrays)."""
        batch, o, k = self.batch_size, self._prep_axis(obs), self._prep_reset(reset)
        cls, bit = np.full(batch, CLS_UNSEEN, _U8_DT), np.zeros(batch, _U8_DT)
        z, z01, chi2 = np.zeros(batch), np.zeros((batch, 2)), np.zeros((batch, 2), np.int64)
        token = next(_SPLIT_TOKENS)
        with self._section():
            self._copy_obs_kick(o, k)
            self._pending_token = 0
            rc = self._lib.mbe_measure_split(
                self._h(), o[2], o[3], o[4], int(bool(keep_rot)), int(reset is not None),
                int(k[7]), k[2], k[3], k[4], addr(cls), addr(bit), addr(z), addr(z01), addr(chi2),
                self._err_a, self._errd_a, self._info_a,
            )
            n_act = int(self._info[0])
            if rc == 0 and n_act > 0:
                self._pending_token = token
            fail = self._fail(int(self._info[1])) if rc == 1 else None
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_measure_split failed before running ({rc})")
        return SplitResult(cls, bit, z, z01, chi2, n_act, fail, rc == -3, token)

    def measure_step(
        self, obs: _Axis, keep_rot: bool, reset: tuple[Any, ...] | None,
        fn_addr: int, state_addr: int, log_masses: Any = None,
    ) -> tuple[int, Any, Any]:
        """Single-call measurement: split, mass check, Born draw, finish."""
        b, nw = self.batch_size, self.n_words
        if log_masses is not None and (
            type(log_masses) is not np.ndarray or log_masses.dtype != _F64_DT
            or log_masses.shape != (b,)
        ):
            raise ValueError("log_masses must be a float64 array of shape (batch_size,)")
        o, k = self._prep_axis(obs), self._prep_reset(reset)
        with self._section():
            self._copy_obs_kick(o, k)
            self._pending_token = 0
            rc = self._lib.mbe_measure_step(
                self._h(), b, nw, o[2], o[3], o[4], int(bool(keep_rot)), int(reset is not None),
                int(k[7]), k[2], k[3], k[4], fn_addr, state_addr, *self._ms_a,
                self._err_a, self._errd_a, self._info_a,
            )
            n_act, info1 = int(self._info[0]), int(self._info[1])
            if log_masses is not None and (info1 if rc == 0 else rc in (1, 2, 3, 4, -3)):
                log_masses[self._ms_cls == CLS_DEAD] = -math.inf
            if log_masses is not None and ((rc == 0 and n_act) or rc == 4):
                log_masses += self._ms_dlog
            if rc == 0:
                return 0, self._ms_bit.view(np.bool_).copy(), None
            if rc in (1, 2):
                fail = self._fail(info1) if rc == 1 else None
                return rc, None, SplitResult(
                    self._ms_cls.copy(), self._ms_bit.copy(), self._ms_z.copy(),
                    self._ms_z01.copy(), self._ms_chi2.copy(), n_act, fail, False, 0,
                )
            if rc in (3, 4):
                return rc, None, self._fail(info1)
        if rc == -3:
            if info1 == 0:
                raise MemoryError("mbe_measure_split ran out of memory (no shot was measured)")
            msg = "(no shot was changed)" if info1 == 1 else "(shots before it were committed)"
            raise MemoryError(f"mbe_measure_finish ran out of memory {msg}")
        raise RuntimeError(f"mbe_measure_step failed before running ({rc})")

    def measure_finish(
        self, outcome: Any, chosen: Any, split: SplitResult | None = None
    ) -> FinishResult:
        """Second half: commit outcome `outcome[i]` of each active shot of the last split."""
        o, c, n, cap = outcome, chosen, int(np.shape(outcome)[0]), self._outc_u8.shape[0]
        fo = type(o) is np.ndarray and o.dtype == _BOOL_DT and o.shape == (n,) and n <= cap
        fc = type(c) is np.ndarray and c.dtype == _F64_DT and c.shape == (n,) and n <= cap
        out = o if fo else _as(np.asarray(o, dtype=np.bool_), np.uint8, (n,), "outcome")
        ch = c if fc else _as(c, np.float64, (n,), "chosen")
        out_a = (self._outc_a if n else 0) if fo else addr(out)
        ch_a = (self._chosen_a if n else 0) if fc else addr(ch)
        with self._section():
            if split is not None and split.token != self._pending_token:
                raise RuntimeError("mbe_measure_finish: no pending measure_split matches this "
                                   "split (another split replaced it, or it was already finished)")
            self._pending_token = 0
            if fo and n:
                np.copyto(self._outc_u8[:n], out)
            if fc and n:
                np.copyto(self._chosen_f64[:n], ch)
            rc = self._lib.mbe_measure_finish(
                self._h(), n, out_a, ch_a, self._err_a, self._errd_a, self._info_a
            )
            if rc == 0:
                return FinishResult(None, 0)
            phase = int(self._info[0])
            fail = self._fail(int(self._info[1])) if rc > 0 else None
        if rc == -3:
            raise MemoryError(
                "mbe_measure_finish ran out of memory "
                + ("(no shot was changed)" if phase == 1 else "(shots before it were committed)")
            )
        if rc == -2:
            raise RuntimeError("mbe_measure_finish: no pending measure_split for this engine")
        if rc == -4:
            raise ValueError("mbe_measure_finish: outcome / chosen do not match the active shots")
        return FinishResult(fail, phase)

    def peek(self, obs: _Axis, shots: Any) -> tuple[Any, FactorFailure | None]:
        """`_obs_p1` of `obs` on every listed shot (fresh non-aliasing `p1`)."""
        ab, bb, aa, ba, kap, fa, fb = self._prep_axis(obs)
        lst, n, la, fs = self._prep_shots(shots)
        p1 = np.zeros(n)
        with self._section():
            self._copy_axis(fa, fb, self._obs_a, self._obs_b, ab, bb)
            if fs and n:
                np.copyto(self._shots_i32[:n], lst)
            rc = self._lib.mbe_peek(
                self._h(), aa, ba, kap, la, n, addr(p1), self._err_a, self._errd_a, self._info_a
            )
            if rc == 0:
                return p1, None
            at = int(self._info[0])
            fail = self._fail(at, 3) if rc == 1 else None
        if rc == -3:
            raise MemoryError(f"mbe_peek ran out of memory at list entry {at}")
        if rc != 1:
            raise RuntimeError(f"mbe_peek failed before running ({rc})")
        return p1, fail

    def postselect(self, obs: _Axis, shots: Any, desired: Any) -> PostselectResult:
        """Project each listed shot onto `desired[i] & 1` (fresh non-aliasing result arrays)."""
        ab, bb, aa, ba, kap, fa, fb = self._prep_axis(obs)
        lst, n, la, fs = self._prep_shots(shots)
        d, cap = desired, self._outc_u8.shape[0]
        fw = type(d) is np.ndarray and d.dtype == _U8_DT and d.shape == (n,) and n <= cap
        want = d if fw else _as(np.asarray(d, dtype=np.uint8), np.uint8, (n,), "desired")
        wa = (self._outc_a if n else 0) if fw else addr(want)
        done, z01 = np.zeros(n, np.uint8), np.zeros((n, 2))
        with self._section():
            self._copy_axis(fa, fb, self._obs_a, self._obs_b, ab, bb)
            if fs and n:
                np.copyto(self._shots_i32[:n], lst)
            if fw and n:
                np.copyto(self._outc_u8[:n], want)
            self._pending_token = 0
            rc = self._lib.mbe_postselect(
                self._h(), aa, ba, kap, la, n, wa, addr(done), addr(z01),
                self._err_a, self._errd_a, self._info_a,
            )
            at = int(self._info[0]) if rc in (1, -3) else 0
            fail = self._fail(at, 3) if rc == 1 else None
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_postselect failed before running ({rc})")
        return PostselectResult(done, z01, fail, at if rc == -3 else None)

    def unmatched_apply(
        self, obs: _Axis, kick: _Axis, shots: Any, bits: Any, measure: Any, kick_flags: Any,
    ) -> UnmatchedResult:
        """Second half of an unmatched noisy reset: project and kick listed shots."""
        ab, bb, aa, ba, kap, fa, fb = self._prep_axis(obs)
        ka_b, kb_b, ka_a, kb_a, k_kap, fka, fkb = self._prep_axis(kick, True, "ka", "kb")
        lst, n, la, fs = self._prep_shots(shots)
        flags = tuple(
            _as(np.asarray(x, dtype=np.uint8), np.uint8, (n,), nm)
            for x, nm in ((bits, "bits"), (measure, "measure"), (kick_flags, "kicks"))
        )
        with self._section():
            self._copy_axis(fa, fb, self._obs_a, self._obs_b, ab, bb)
            self._copy_axis(fka, fkb, self._kick_a, self._kick_b, ka_b, kb_b)
            if fs and n:
                np.copyto(self._shots_i32[:n], lst)
            self._pending_token = 0
            rc = self._lib.mbe_unmatched_apply(
                self._h(), aa, ba, kap, ka_a, kb_a, k_kap, la, n, *(addr(f) for f in flags),
                self._err_a, self._errd_a, self._info_a,
            )
            at, phase = int(self._info[0]), int(self._info[1])
            fail = self._fail(at, 3) if rc == 1 else None
        if rc not in (0, 1, -3):
            raise RuntimeError(f"mbe_unmatched_apply failed before running ({rc})")
        return UnmatchedResult(fail, at if rc == -3 else None, phase)


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
    """True when engine complex multiplies and norm_sq match numpy bit-for-bit (False on error)."""
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
