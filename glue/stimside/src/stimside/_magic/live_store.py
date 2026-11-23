"""Engine-backed store of the live per-shot expansions (`LiveExpansions` proxy over `_live_exps`).

Reading `store[b]` exports a fresh `ShotExpansion` and `store[b] = exp` imports a copy, so no
expansion aliases engine state. Supports integer indexing, `len`, iteration, and `copy()`.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from stimside._magic import phase_qfe as _pq
from stimside._magic.channels import CompiledCliffordFactorList, apply_clifford_factors
from stimside._magic.phase_qfe import K1_CANCEL_REL_TOL, ShotExpansion
from stimside._magic.query import QueryBudgetExceededError
from stimside._native import magic_engine as _me
from stimside._native.magic_engine import MagicBatchEngine, own_axis, own_factor_table


def expansion_to_arrays(exp: ShotExpansion) -> _me.ShotArrays:
    """The arrays of `exp` as the engine stores them (views, not copies)."""
    assert exp.ell_odd is not None
    return _me.ShotArrays(
        exp.chi, exp.d, bool(exp._is_canonical_merged), float(exp.log_mass), float(exp.atol),
        exp.weights, exp.h_words, exp.ell, exp.A_words, exp.a_union, exp.Gamma, exp.pivots,
        exp.ell_odd,
    )


def arrays_to_expansion(n_qubits: int, s: _me.ShotArrays) -> ShotExpansion:
    """A `ShotExpansion` owning exactly the exported arrays `s`."""
    return ShotExpansion(
        n_qubits, s.d, s.A_words, s.a_union, s.Gamma, s.pivots, s.weights, s.h_words, s.ell,
        s.ell_odd, s.log_mass, s.atol, s.canonical,
    )


def engine_numerics_available() -> bool:
    """False when numpy does not match engine numerics or the native library cannot load."""
    return _me.numerics_match_numpy()


def _axes_of(fac: CompiledCliffordFactorList) -> list[Any]:
    return ([] if fac.p0 is None else [fac.p0]) + list(fac.r_axes)


def _is_integer(v: Any) -> bool:
    """An `int` or numpy integer, not a bool."""
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _well_formed_axis(ax: Any, n_words: int) -> bool:
    """True when `ax` has `uint64` `(n_words,)` word arrays and int-convertible `kappa`."""
    if not all(hasattr(ax, name) for name in ("a_words", "b_words", "kappa")):
        return False
    for w in (ax.a_words, ax.b_words):
        if not (isinstance(w, np.ndarray) and w.dtype == np.uint64 and w.shape == (n_words,)):
            return False
    try:
        int(ax.kappa)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def well_formed_lists(lists: Sequence[CompiledCliffordFactorList], n_words: int) -> bool:
    """True when every list has integer `v` and uint64 `(n_words,)` axes; never raises."""
    ok = False
    with contextlib.suppress(Exception):
        ok = all(
            _is_integer(fac.v) and all(_well_formed_axis(ax, n_words) for ax in _axes_of(fac))
            for fac in lists
        )
    return ok


def factor_table(lists: Sequence[CompiledCliffordFactorList], n_words: int) -> _me.FactorTable:
    """Compile factor lists into the engine's flat table layout (`p0` first, then `r` axes)."""
    if not well_formed_lists(lists, n_words):
        raise ValueError(
            f"factor lists need uint64 word arrays of shape ({n_words},), integer v and kappa"
        )
    return _encode_lists(lists, n_words)


_TABLE_CACHE_MAX = 4096
_OWN_LISTS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_OWN_AXES: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_Axis = _me.OwnedAxis | tuple[Any, Any, int]


def register_own_list(fac: CompiledCliffordFactorList) -> None:
    """Mark `fac` (frozen by the simulator's memo) as trusted for cached owned tables."""
    _OWN_LISTS[id(fac)] = fac


def register_own_axis(ax: Any) -> None:
    """Mark `ax` (frozen by the simulator's memo) as trusted for cached owned axes."""
    _OWN_AXES[id(ax)] = ax


def _frozen_key(lists: Sequence[CompiledCliffordFactorList]) -> tuple[Any, ...] | None:
    """Identity key when every list is registered via `register_own_list`, else None."""
    lists_t = lists if type(lists) is tuple else tuple(lists)
    key = tuple(id(fac) for fac in lists_t)
    return key if all(_OWN_LISTS.get(k) is fac for k, fac in zip(key, lists_t)) else None


def _encode_lists(lists: Sequence[CompiledCliffordFactorList], n_words: int) -> _me.FactorTable:
    axes: list[Any] = []
    cols: list[list[int]] = [[], [], [], [], []]  # has_p0, n_r, v_fwd, v_inv, axis_start
    for fac in lists:
        cols[4].append(len(axes))
        cols[0].append(fac.p0 is not None)
        cols[1].append(len(fac.r_axes))
        cols[2].append(int(fac.v) & 7 if fac.v else -1)  # `if fac.v`: even 8 multiplies by omega^0
        cols[3].append((-int(fac.v)) & 7 if fac.v else -1)
        axes.extend(_axes_of(fac))
    a = np.zeros((len(axes), n_words), dtype=np.uint64)
    b = np.zeros((len(axes), n_words), dtype=np.uint64)
    for i, ax in enumerate(axes):
        a[i], b[i] = ax.a_words, ax.b_words
    kappa = np.array([int(ax.kappa) & 3 for ax in axes], dtype=np.uint8)
    return _me.FactorTable(
        np.array(cols[0], dtype=np.uint8), np.array(cols[1], dtype=np.int32),
        np.array(cols[2], dtype=np.int32), np.array(cols[3], dtype=np.int32),
        np.array(cols[4], dtype=np.int32), a, b, kappa,
    )


def raise_factor_failure(fail: _me.FactorFailure, n_qubits: int, atol: float) -> None:
    """Raise what the per-shot reference function raises for this failure."""
    name = _me.OP_NAMES[fail.op]
    if fail.kind == _me.FAIL_NON_FINITE_WEIGHTS:
        raise FloatingPointError(f"Non-finite weights in {name}")
    if fail.kind == _me.FAIL_NON_FINITE_NORM:
        raise FloatingPointError(f"Non-finite input norm in {name}")
    if fail.kind == _me.FAIL_MERGE_NON_FINITE:
        raise FloatingPointError("Non-finite weights encountered in merge_frames")
    if fail.kind == _me.FAIL_KERNEL:
        raise FloatingPointError(f"magic_apply_quarter_turn returned error code {fail.aux}")
    if fail.kind == _me.FAIL_NOT_HERMITIAN:
        raise ValueError(
            f"Pauli operator P(a, b, kappa={fail.aux2}) must be Hermitian (<a,b>={fail.aux})"
        )
    if fail.kind == _me.FAIL_PADDING:
        raise ValueError(
            f"Non-zero upper padding bits in packed uint64 word for n_qubits={n_qubits}"
        )
    if fail.kind == _me.FAIL_UNITARITY:
        _pq.check_unitary_norm(fail.pre, fail.post, atol, name)  # raises with the reference text
    if fail.kind == _me.FAIL_ZERO_CHOSEN:  # `1.0 / math.sqrt(cz)` of the reference
        if fail.aux2 == 0:
            raise ZeroDivisionError("float division by zero")
        raise ValueError("math domain error")  # a negative mass: born_outcomes never gives one
    raise RuntimeError(f"unknown factor failure {fail}")


def raise_split_failure(res: _me.SplitResult, n_qubits: int, atol: float) -> None:
    """Raise what `measure_expansions_batch` raises for a failed or OOM `measure_split`."""
    if res.oom:
        raise MemoryError("mbe_measure_split ran out of memory (no shot was measured)")
    fail = res.failure
    if fail is None:
        return
    active = res.cls == _me.CLS_ACTIVE
    if fail.kind == _me.FAIL_BAD_NORM:
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={res.z[active]!r}")
    if fail.kind == _me.FAIL_ROTATION:
        raise FloatingPointError(f"apply_quarter_turn failed while measuring shot {fail.aux}")
    raise_factor_failure(fail, n_qubits, atol)


def raise_born_failure(fail: _me.FactorFailure, n_qubits: int, atol: float) -> None:
    """Raise reference errors for one shot's failure from `peek`, `postselect`, or `unmatched`."""
    k = fail.kind
    if k == _me.FAIL_BAD_NORM:
        raise FloatingPointError(f"Invalid pre-measurement norm squared z={fail.z!r}")
    if k == _me.FAIL_NON_FINITE_MASS:
        raise FloatingPointError(
            f"Non-finite projected Born masses: z0={fail.pre!r}, z1={fail.post!r}"
        )
    if k == _me.FAIL_NEGATIVE_MASS:
        raise FloatingPointError(
            f"Significantly negative projected Born mass: z0={fail.pre!r}, z1={fail.post!r}, "
            f"z={fail.z!r}"
        )
    if k == _me.FAIL_CONSERVATION:
        z0, z1 = fail.pre, fail.post
        raise FloatingPointError(
            f"M2 Born mass conservation violated: z0={z0:.16e}, z1={z1:.16e}, "
            f"z0+z1={z0 + z1:.16e}, z={fail.z:.16e}, tol={atol:.3e}"
        )
    if k == _me.FAIL_ZERO_POSTSELECT:
        raise _pq.ImpossiblePostselectionError(
            f"Requested postselection outcome {fail.aux} has zero Born probability "
            f"(z_{fail.aux}={fail.pre:.3e}, total z={fail.z:.3e})"
        )
    raise_factor_failure(fail, n_qubits, atol)


def born_outcomes(
    res: _me.SplitResult, atol: float, rng: np.random.Generator,
    sample_outcome: Callable[[int, float], int] | None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Born mass checks and outcome sampling over active shots; returns (out, z0, z1, chosen)."""
    idx = np.flatnonzero(res.cls == _me.CLS_ACTIVE)
    z, z0r, z1r = res.z[idx], res.z01[idx, 0], res.z01[idx, 1]
    scale = np.maximum(1.0, z)
    if not np.all(np.isfinite(z0r) & np.isfinite(z1r)):
        raise FloatingPointError(f"Non-finite projected Born masses: z0={z0r!r}, z1={z1r!r}")
    if np.any((z0r < -1e-12 * scale) | (z1r < -1e-12 * scale)):
        raise FloatingPointError(f"Significantly negative projected Born mass: {z0r!r}, {z1r!r}")
    z0, z1 = np.where(z0r > 0.0, z0r, 0.0), np.where(z1r > 0.0, z1r, 0.0)  # max(0.0, z_m)
    if not np.all(np.abs((z0 + z1) - z) <= atol * scale):
        raise FloatingPointError(f"M2 Born mass conservation violated: {z0!r}, {z1!r}, z={z!r}")
    if np.any((z <= 0.0) | ((z0 + z1) <= 0.0)):
        raise _pq.ImpossiblePostselectionError(
            "Cannot sample measurement from a zero-norm expansion"
        )
    zero_floor = K1_CANCEL_REL_TOL * z
    if sample_outcome is None:
        u = rng.random(len(idx))  # one draw per active shot, pinned ones included
        p1 = np.minimum(1.0, np.maximum(0.0, z1 / (z0 + z1)))
        out = np.where(z0 <= zero_floor, True, np.where(z1 <= zero_floor, False, u < p1))
    else:
        out = np.array([bool(sample_outcome(i, min(1.0, max(0.0, z1_i / z_i))))
                        for i, (z1_i, z_i) in enumerate(zip(z1.tolist(), z.tolist()))])
    chosen = np.where(out, z1, z0)
    if sample_outcome is not None:
        chi_chosen = res.chi2[idx, out.astype(np.intp)]
        if np.any((chosen <= zero_floor) | (chi_chosen == 0)):
            raise _pq.ImpossiblePostselectionError(
                "Sampled measurement outcome has zero Born probability"
            )
    return out, z0, z1, chosen


_KIND = {"P1": 0, "P2": 1, "N1": 2, "N2": 3}


def _empty_block_table(mode: int, g_sc: complex, nw: int) -> _me.BlockTable:
    z32, zu64 = np.zeros(0, np.int32), np.zeros((0, nw), np.uint64)
    return _me.BlockTable(
        mode, g_sc.real, g_sc.imag, z32, z32, z32, zu64, zu64, np.zeros(0, np.uint8),
        np.zeros((0, 2), np.float64),
    )


def _pauli_sum_table(block: _pq.CompiledPauliBlock, g_sc: complex) -> _me.BlockTable:
    """Engine pauli_sum encoding with coefficients `g_sc * coeffs[nz]`."""
    assert block.coeffs is not None and block.a_words is not None
    assert block.b_words is not None and block.kappas is not None
    nz = np.flatnonzero(np.abs(block.coeffs) > 0.0)
    z32, coeffs = np.zeros(0, np.int32), np.asarray(g_sc * block.coeffs[nz], np.complex128)
    return _me.BlockTable(
        0, 1.0, 0.0, z32, z32, z32, block.a_words[nz], block.b_words[nz], block.kappas[nz],
        np.ascontiguousarray(coeffs).view(np.float64).reshape(-1, 2),
    )


def _step_rows(step: _pq.CompiledPauliBlockStep, nq: int) -> tuple[Any, Any, Any, Any]:
    """Weights, `a`, `b`, and kappas of a step's axes (8 rows for N2, else 1)."""
    if step.kind == "N2":
        return _pq._normalize_ccz_branches(step.paulis, nq)
    a_w, b_w, kap = _pq._unpack_pauli_args(*step.paulis[0], nq)
    return np.zeros(1, np.complex128), a_w[None, :], b_w[None, :], np.array([kap], np.uint8)


def _sequential_table(
    block: _pq.CompiledPauliBlock, g_sc: complex, nq: int
) -> _me.BlockTable | None:
    """Engine encoding of a sequential block, or None for nested/unknown steps."""
    nw, n_rows = _pq.n_words_for_qubits(nq), 0
    kinds, signs, starts, parts = [], [], [], []
    for step in block.steps:
        if not isinstance(step, _pq.CompiledPauliBlockStep) or step.kind not in _KIND:
            return None
        kinds.append(_KIND[step.kind])
        signs.append(int(step.sign))
        starts.append(n_rows)
        parts.append(_step_rows(step, nq))
        n_rows += parts[-1][1].shape[0]
    def cat(i: int, dt: Any, sh: tuple[int, ...]) -> Any:
        return np.concatenate([p[i] for p in parts]).astype(dt) if parts else np.zeros(sh, dt)
    w_ri = np.ascontiguousarray(cat(0, np.complex128, (0,))).view(np.float64).reshape(-1, 2)
    return _me.BlockTable(
        1, g_sc.real, g_sc.imag, np.array(kinds, np.int32), np.array(signs, np.int32),
        np.array(starts, np.int32), cat(1, np.uint64, (0, nw)), cat(2, np.uint64, (0, nw)),
        cat(3, np.uint8, (0,)), w_ri,
    )


def block_table(block: Any, n_qubits: int) -> _me.BlockTable | None:
    """Encode `block` for the engine, or None if invalid/nested (caller uses reference path)."""
    if not isinstance(block, _pq.CompiledPauliBlock):
        return None
    table: _me.BlockTable | None = None
    with contextlib.suppress(Exception):  # the reference raises its own error for such a block
        _pq._validate_compiled_block(block, n_qubits)
        g_sc = _pq._coerce_finite_complex(block.global_scalar, "block.global_scalar")
        if block.mode == "pauli_sum" and not block.is_identity:
            table = _pauli_sum_table(block, g_sc)
        elif block.is_identity:
            table = _empty_block_table(1, g_sc, _pq.n_words_for_qubits(n_qubits))
        else:
            table = _sequential_table(block, g_sc, n_qubits)
    return table


class LiveExpansions(Sequence[ShotExpansion]):
    """`list[ShotExpansion]`-like view of the engine's shots (copy out on read, in on write)."""

    def __init__(self, batch_size: int, n_qubits: int, atol: float) -> None:
        self.batch_size = _me.as_integer(batch_size, "batch_size")
        self.n_qubits = _me.as_integer(n_qubits, "n_qubits")
        if self.batch_size < 0 or self.n_qubits < 1:
            raise ValueError(f"need batch_size >= 0, n_qubits >= 1, got {batch_size}, {n_qubits}")
        self.atol = _pq.validate_atol(atol)
        self.failed_shot: int | None = None
        self._tables: OrderedDict[Any, Any] = OrderedDict()
        self._axes: OrderedDict[int, tuple[_me.OwnedAxis, Any]] = OrderedDict()
        self._engine: MagicBatchEngine | None = None
        self._lock, self._creating = threading.RLock(), False
        _me.track_locks_for_fork(self)

    def _reinit_after_fork(self) -> None:
        """Run in a forked child, which has only the forking thread: renew the lock and flag."""
        self._lock = threading.RLock()
        self._creating = False

    @property
    def engine(self) -> MagicBatchEngine:
        """Native engine created lazily on first use (thread-safe, non-re-entrant)."""
        with self._lock:
            if self._engine is None:
                if self._creating:
                    raise RuntimeError("re-entrant creation of the LiveExpansions engine")
                self._creating = True
                try:
                    self._engine = MagicBatchEngine(self.batch_size, self.n_qubits, self.atol)
                finally:
                    self._creating = False
            return self._engine

    def copy(self) -> LiveExpansions:
        """An independent store holding copies of the same shots."""
        new = type(self)(self.batch_size, self.n_qubits, self.atol)
        if self._engine is not None:  # no engine yet: every shot is still the zero state
            for b in range(self.batch_size):
                new[b] = self[b]
        return new

    def reset(self) -> None:
        """Every shot back to `ShotExpansion.zero` (what `clear()` does for the list)."""
        engine = self._engine
        if engine is not None:
            engine.reset()

    def close(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def __len__(self) -> int:
        return self.batch_size

    @overload
    def __getitem__(self, b: int) -> ShotExpansion: ...
    @overload
    def __getitem__(self, b: slice) -> list[ShotExpansion]: ...
    def __getitem__(self, b: int | slice) -> ShotExpansion | list[ShotExpansion]:
        if isinstance(b, slice):
            return [self[i] for i in range(*b.indices(self.batch_size))]
        return arrays_to_expansion(self.n_qubits, self.engine.export_shot(self._index(b)))

    def _index(self, b: Any) -> int:
        """A shot index as for a list: integers only (floats are an error), negatives wrap."""
        if not isinstance(b, (int, np.integer)):
            raise TypeError(f"shot index must be an integer, got {type(b).__name__}")
        i = int(b)
        return i + self.batch_size if i < 0 else i

    def __setitem__(self, b: int, exp: ShotExpansion) -> None:
        i = self._index(b)
        if not isinstance(exp, ShotExpansion):
            raise TypeError(f"expected ShotExpansion, got {type(exp).__name__}")
        if exp.n_qubits != self.n_qubits:
            raise ValueError(f"shot has n_qubits={exp.n_qubits}, engine has {self.n_qubits}")
        self.engine.import_shot(i, expansion_to_arrays(exp))

    def _owned_axis(self, ax: Any) -> _me.OwnedAxis | tuple[Any, Any, int]:
        aid = id(ax)
        reg = _OWN_AXES.get(aid)
        if reg is not None and reg is ax:
            cache = self._axes
            hit = cache.get(aid)
            if hit is not None and hit[1] is ax:
                cache.move_to_end(aid)
                return hit[0]
            nw = self.engine.n_words
            if _well_formed_axis(ax, nw):
                owned = own_axis(ax.a_words, ax.b_words, ax.kappa, nw)
                cache[aid] = (owned, ax)
                if len(cache) > _TABLE_CACHE_MAX:
                    cache.popitem(last=False)
                return owned
        return (ax.a_words, ax.b_words, ax.kappa)

    def apply_pauli_mask(self, axis: Any, mask: NDArray[np.bool_]) -> None:
        """`apply_pauli(shot, axis)` on every shot of `mask` with chi > 0, in shot order."""
        self.engine._apply_pauli_mask_obs(self._owned_axis(axis), mask)

    def apply_pauli_noise_batch(
        self, ref_tab: Any, xm: NDArray[np.bool_], zm: NDArray[np.bool_]
    ) -> None:
        self.engine.apply_pauli_noise_batch(
            ref_tab.x2x, ref_tab.x2z, ref_tab.x_kap, ref_tab.z2x, ref_tab.z2z, ref_tab.z_kap, xm, zm
        )

    def inject_factor_lists(
        self, lists: Sequence[CompiledCliffordFactorList], shot_list: Any, inverse: bool
    ) -> None:
        """Apply `lists[shot_list[s]]` per shot in shot order (fallback per-shot if malformed)."""
        self.failed_shot = None
        eng = self.engine
        table: _me.FactorTable | _me.OwnedFactorTable | None = self._trusted_table(
            lists, eng.n_words
        )
        if table is None:
            if not well_formed_lists(lists, eng.n_words):
                self._inject_per_shot(lists, shot_list, inverse)
                return
            table = _encode_lists(lists, eng.n_words)
        fail = eng.inject_cliffords(table, shot_list, inverse)
        if fail is not None:
            self.failed_shot = fail.shot
            raise_factor_failure(fail, self.n_qubits, self.atol)

    def _trusted_table(
        self, lists: Sequence[CompiledCliffordFactorList], nw: int
    ) -> _me.OwnedFactorTable | None:
        """Cached `OwnedFactorTable` for registered factor lists, else None."""
        lists_t = lists if type(lists) is tuple else tuple(lists)
        key = _frozen_key(lists_t)
        if key is None:
            return None
        cache = self._tables
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit[0]
        if not well_formed_lists(lists_t, nw):
            return None
        owned = own_factor_table(_encode_lists(lists_t, nw), nw)
        cache[key] = (owned, lists_t)
        if len(cache) > _TABLE_CACHE_MAX:
            cache.popitem(last=False)
        return owned

    def _inject_per_shot(
        self, lists: Sequence[CompiledCliffordFactorList], shot_list: Any, inverse: bool
    ) -> None:
        which = np.asarray(shot_list)
        for s in np.flatnonzero(which >= 0).tolist():
            try:
                self[s] = apply_clifford_factors(
                    self[s], lists[int(which[s])], inverse=inverse, check_unitary=True,
                    atol=self.atol,
                )
            except Exception:
                self.failed_shot = s
                raise

    def apply_block(
        self, table: _me.BlockTable, max_components: int | None, dedupe: bool = True
    ) -> tuple[int, int]:
        """Apply compiled block per group of identical live shots; returns (groups, live)."""
        self.failed_shot = None
        res = self.engine.apply_block(table, dedupe, max_components)
        fail = res.failure
        if fail is None:
            return res.groups, res.live
        self.failed_shot = fail.shot
        if fail.kind == _me.FAIL_MAX_COMPONENTS:
            raise QueryBudgetExceededError(
                f"Expansion exceeded max_components={max_components} (got chi={fail.aux})"
            )
        raise_factor_failure(fail, self.n_qubits, self.atol)
        raise AssertionError("unreachable")

    def axis_is_encodable(self, axis: Any) -> bool:
        """True when `axis` has the `uint64` word arrays the engine takes."""
        hit = self._axes.get(id(axis))
        if hit is not None and hit[1] is axis:
            return True
        return _well_formed_axis(axis, self.engine.n_words)

    def measure_split(self, obs: _Axis, keep_rot: bool, reset: Any) -> _me.SplitResult:
        self.failed_shot = None
        return self.engine.measure_split(obs, keep_rot, reset)

    def measure_step(
        self, obs: _Axis, keep_rot: bool, reset: Any, fn_a: int, st_a: int, log_m: Any = None,
    ) -> tuple[int, Any, Any]:
        self.failed_shot = None
        return self.engine.measure_step(obs, keep_rot, reset, fn_a, st_a, log_m)

    def measure_finish(self, outcome: Any, chosen: Any, split: Any = None) -> _me.FinishResult:
        return self.engine.measure_finish(outcome, chosen, split)

    def peek(self, obs: _Axis, shots: Any) -> tuple[NDArray[np.float64], _me.FactorFailure | None]:
        return self.engine.peek(obs, shots)

    def postselect(self, obs: _Axis, shots: Any, desired: Any) -> _me.PostselectResult:
        return self.engine.postselect(obs, shots, desired)

    def unmatched_apply(
        self, obs: _Axis, kick: _Axis, shots: Any, bits: Any, measure: Any, kicks: Any,
    ) -> _me.UnmatchedResult:
        return self.engine.unmatched_apply(obs, kick, shots, bits, measure, kicks)

    def __getstate__(self) -> None:
        raise TypeError("LiveExpansions owns native engine state and cannot be pickled or copied")
