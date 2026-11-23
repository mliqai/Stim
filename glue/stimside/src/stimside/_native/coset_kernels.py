"""Shared-library loader for `coset_kernels.cpp`.

ABI notes:
- ABI v2 (``coset_kernels_abi_v2() == 2``): void-returning entry points validate pointers,
  dimensions, and qubit/target ranges and return silently on invalid input without returning an
  error code. On a healthy engine (``fail_kind == 0``), ``engine_batch_inject_general_clifford``
  and ``engine_batch_unmatched_noisy_reset_z`` zero non-null output kick flags on invalid input
  (with non-null ``ptr``), whereas ``engine_batch_mixed_substep`` and ``engine_batch_evict`` leave
  accumulator flags untouched on invalid input; no v2 kernel touches flags when ``ptr == NULL``.
  ``engine_set_rng_states`` copies in place (no heap allocation) and returns ``0`` on success or
  ``< 0`` on ``NULL``/error; ``engine_get_rng_states`` copies owned RNG states with
  ``n_shots == batch_size`` check. ``engine_get_active_shot_indices`` returns ``-1`` on ``NULL``.
- ABI v3 (``coset_kernels_abi_v3() == 3``): additive ``int``-returning entry points validate
  engine-owned dimensions and return ``< 0`` on invalid arguments or allocation failure.
- Failed-engine & exception safety: ``engine_failed(ptr)`` returns ``0`` (healthy), ``1``
  (``std::bad_alloc``), ``2`` (other C++ exception), or ``-1`` (``NULL``);
  ``engine_create_last_error()`` reports ``0``/``1``/``2`` for the last ``engine_create`` on the
  current thread. When ``fail_kind != 0``, v2 void kernels zero non-null output/accumulator kick
  flags and return early, while ``int``-returning exports return ``< 0`` (``-2`` on
  ``engine_get_active_shot_indices``, ``engine_num_qubits``, ``engine_batch_size``,
  ``engine_n_words``, and ``engine_unitary_step_num_groups``; negative codes on v3 kernels) and
  zero non-null output flags. ``get_coset_kernels_lib()`` wraps the CDLL so caught C++ exceptions
  raise ``MemoryError`` (``1``) or ``RuntimeError`` (``> 0``).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, cast

from stimside._native._kernel_loader import (
    AbiSpec,
    f64_p,
    get_lib,
    i8_p,
    i32_p,
    i64_p,
    int_p,
    resolve_so,
    u8_p,
    u64_p,
    void_p,
)

_c_int, _c_i32, _c_i64, _c_u8 = ctypes.c_int, ctypes.c_int32, ctypes.c_int64, ctypes.c_uint8


class CosetCallBlock(ctypes.Structure):
    _fields_ = [
        ("n_q", _c_i32), ("n_b", _c_i32), ("n_w", _c_i32), ("n_packed_bytes", _c_i32),
        ("m_buf_rows", _c_i32), ("any_x", _c_i32), ("any_z", _c_i32), ("flags", _c_i32),
        ("scratch_x_mask", u8_p), ("scratch_z_mask", u8_p), ("m_buf", u8_p),
    ]


call_block_p = ctypes.POINTER(CosetCallBlock)

_LIB_INSTANCE: ctypes.CDLL | None = None
_WRAPPED_CACHE: tuple[ctypes.CDLL, ctypes.CDLL] | None = None

_SPECS: tuple[AbiSpec, ...] = (
    ("coset_kernels_abi_v2", [], _c_int),
    ("coset_kernels_abi_v3", [], _c_int),
    ("coset_kernels_abi_v4", [], _c_int),
    ("engine_create", [_c_int, _c_int, u64_p], void_p),
    ("engine_destroy", [void_p], None),
    ("engine_set_rng_states", [void_p, u64_p], _c_int),
    ("engine_get_rng_states", [void_p, _c_int, u64_p], _c_int),
    ("engine_clear", [void_p], None),
    ("engine_get_active_shot_indices", [void_p, i32_p, i32_p], _c_int),
    ("engine_export_shot_v3", [
        void_p, _c_int, _c_int, _c_int, u64_p, u64_p, u64_p, u8_p, u8_p, i32_p,
    ], _c_int),
    ("engine_batch_inject_general_clifford", [
        void_p, _c_int, _c_int, _c_int, i64_p, u8_p, u8_p, u8_p, u8_p, u8_p, u8_p, u64_p, u64_p,
        u8_p, u8_p, u8_p, u64_p, u64_p, u8_p, u64_p, u64_p, u8_p, u8_p, int_p, int_p,
    ], None),
    ("engine_batch_meas_step_v4", [
        void_p, _c_int, _c_int, u8_p, i64_p, u8_p, i64_p, i32_p, u64_p, u64_p, u64_p,
        _c_int, i32_p, i64_p, _c_int, i32_p, i64_p, u64_p, u64_p, u8_p, call_block_p,
    ], _c_int),
    ("engine_cond_pauli_noise_v4", [
        void_p, _c_int, _c_int, i64_p, _c_int, f64_p, u8_p, u8_p, void_p, void_p, call_block_p,
    ], _c_int),
    ("engine_batch_mixed_substep", [
        void_p, _c_int, _c_i64, _c_u8, _c_i64, u64_p, u64_p, u64_p, _c_int, _c_int, i64_p,
        _c_int, i64_p, u8_p, u8_p, u8_p, u8_p, int_p, int_p,
    ], None),
    ("engine_batch_evict", [void_p, u64_p, u64_p, u8_p, u8_p, u8_p, u8_p, int_p, int_p], None),
    ("engine_batch_unmatched_noisy_reset_z", [
        void_p, u8_p, u8_p, u8_p, u64_p, u64_p, u8_p, u64_p, u8_p, u8_p, int_p, int_p,
    ], None),
    ("engine_peek_pauli_batch", [
        void_p, _c_int, i64_p, _c_int, u8_p, u8_p, u64_p, u64_p, u64_p, u64_p, u8_p, u8_p, i8_p,
    ], None),
    ("engine_num_qubits", [void_p], _c_int),
    ("engine_batch_size", [void_p], _c_int),
    ("engine_n_words", [void_p], _c_int),
    ("engine_postselect_batch", [
        void_p, _c_i64, _c_int, _c_int, u8_p, _c_int, u8_p, _c_int, _c_int, u8_p, _c_int, _c_int,
        u8_p, _c_int, _c_int, u64_p, u64_p, u64_p, u64_p, u8_p, u8_p,
        _c_int, _c_int, u8_p, _c_int, _c_int, u8_p, int_p, int_p,
    ], _c_int),
    ("engine_project_pauli", [
        void_p, _c_int, _c_int, _c_int, u8_p, _c_int, _c_int, u8_p, _c_int, i64_p, _c_int, _c_int,
        u8_p, _c_int, _c_int, u8_p, _c_int, _c_int, u64_p, u64_p, u64_p, u64_p, u8_p, u8_p,
        _c_int, _c_int, u8_p, _c_int, _c_int, u8_p, int_p, int_p,
    ], _c_int),
    ("engine_register_unitary_step", [
        void_p, _c_int, _c_int, _c_int, i64_p, u8_p, u8_p, u8_p, u64_p, u64_p, u8_p, u8_p, u8_p,
        u64_p, u64_p, u8_p, _c_int, _c_int, u64_p, u64_p,
    ], _c_int),
    ("engine_unitary_step_num_groups", [void_p, _c_int], _c_int),
    ("engine_register_unitary_group_evict", [
        void_p, _c_int, _c_int, _c_int, _c_int, u64_p, u64_p,
    ], _c_int),
    ("engine_inject_group_v3", [
        void_p, _c_int, _c_int, _c_int, u8_p, _c_int, _c_int, u8_p, _c_int, _c_int, u8_p,
        _c_int, _c_int, u8_p, _c_int, _c_int, u8_p, int_p, int_p,
    ], _c_int),
    ("engine_batch_inject_unitary_v4", [
        void_p, _c_int, _c_int, _c_int, u8_p, u8_p, u8_p, u8_p, call_block_p,
    ], _c_int),
    ("engine_failed", [void_p], _c_int),
    ("engine_create_last_error", [], _c_int),
)

_NO_GUARD: frozenset[str] = frozenset({
    "coset_kernels_abi_v2", "coset_kernels_abi_v3", "coset_kernels_abi_v4",
    "engine_destroy", "engine_failed", "engine_create_last_error",
})
_VOID_FNS: frozenset[str] = frozenset({
    "engine_clear", "engine_batch_inject_general_clifford", "engine_batch_mixed_substep",
    "engine_batch_evict", "engine_batch_unmatched_noisy_reset_z", "engine_peek_pauli_batch",
})
_INT_FNS: frozenset[str] = frozenset({
    "engine_set_rng_states", "engine_get_rng_states", "engine_get_active_shot_indices",
    "engine_export_shot_v3", "engine_num_qubits", "engine_batch_size", "engine_n_words",
    "engine_postselect_batch", "engine_project_pauli", "engine_register_unitary_step",
    "engine_unitary_step_num_groups", "engine_register_unitary_group_evict",
    "engine_inject_group_v3", "engine_batch_inject_unitary_v4",
    "engine_batch_meas_step_v4", "engine_cond_pauli_noise_v4",
})

_SPEC_NAMES: frozenset[str] = frozenset(spec[0] for spec in _SPECS)
assert {"engine_create"} | _NO_GUARD | _VOID_FNS | _INT_FNS == _SPEC_NAMES
assert len(_NO_GUARD) + len(_VOID_FNS) + len(_INT_FNS) + 1 == len(_SPEC_NAMES)


def _raise_for_err(err: int) -> None:
    if err == 1:
        raise MemoryError("coset_kernels C++ allocation failed")
    if err > 0:
        raise RuntimeError(f"coset_kernels C++ exception (code={err})")


def _wrap_create(raw_fn: Any, err_fn: Any) -> Any:
    def guarded(*args: Any) -> Any:
        ptr = raw_fn(*args)
        if not ptr:
            _raise_for_err(int(err_fn()))
        return ptr

    g: Any = guarded
    g.argtypes = raw_fn.argtypes
    g.restype = raw_fn.restype
    return g


def _wrap_void(raw_fn: Any, failed_fn: Any) -> Any:
    def guarded(ptr: Any, *args: Any) -> None:
        raw_fn(ptr, *args)
        if ptr:
            _raise_for_err(int(failed_fn(ptr)))

    g: Any = guarded
    g.argtypes = raw_fn.argtypes
    g.restype = raw_fn.restype
    return g


def _wrap_int(raw_fn: Any, failed_fn: Any) -> Any:
    def guarded(ptr: Any, *args: Any) -> int:
        rc = int(raw_fn(ptr, *args))
        if rc < 0 and ptr:
            _raise_for_err(int(failed_fn(ptr)))
        return rc

    g: Any = guarded
    g.argtypes = raw_fn.argtypes
    g.restype = raw_fn.restype
    return g


class _CosetKernelsLib:
    """CDLL proxy checking engine failure state after guarded C++ calls."""

    def __init__(self, raw: ctypes.CDLL) -> None:
        self._raw = raw
        failed_fn = raw.engine_failed
        self.engine_create = _wrap_create(raw.engine_create, raw.engine_create_last_error)
        for name in _NO_GUARD:
            setattr(self, name, getattr(raw, name))
        for name in _VOID_FNS:
            setattr(self, name, _wrap_void(getattr(raw, name), failed_fn))
        for name in _INT_FNS:
            setattr(self, name, _wrap_int(getattr(raw, name), failed_fn))


def _resolve_coset_kernels_so(force_recompile: bool = False) -> Path:
    return resolve_so(__file__, "coset_kernels", force_recompile)


def get_coset_kernels_lib(force_recompile: bool = False) -> ctypes.CDLL:
    """Locate (or compile if needed) and return the cached ctypes CDLL for coset_kernels.cpp."""
    global _WRAPPED_CACHE
    raw = get_lib(
        sys.modules[__name__],
        "coset_kernels",
        "coset_kernels_abi_v4",
        _SPECS,
        force_recompile=force_recompile,
    )
    if not force_recompile and _WRAPPED_CACHE is not None and _WRAPPED_CACHE[0] is raw:
        return _WRAPPED_CACHE[1]
    wrapped = cast(ctypes.CDLL, _CosetKernelsLib(raw))
    _WRAPPED_CACHE = (raw, wrapped)
    return wrapped
