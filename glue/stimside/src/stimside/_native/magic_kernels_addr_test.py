# Copyright 2026 Google LLC
"""Integer-address kernel calls (WP-24 B2 / E3): `addr`/`w_addr`, `c_void_p`, no typed pointers."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest

from stimside._magic import phase_qfe as pq
from stimside._native import _kernel_loader as kl
from stimside._native import magic_kernels as mk


def _typed_pointer_params() -> list[tuple[str, int]]:
    lib = mk.get_magic_kernels_lib()
    return [
        (name, i)
        for name, _, _ in mk._SPECS
        for i, t in enumerate(getattr(lib, name).argtypes)
        if isinstance(t, type) and issubclass(t, ctypes._Pointer)
    ]


def _shot(n: int = 5) -> pq.ShotExpansion:
    e = pq.apply_pauli_t(pq.ShotExpansion.zero(n), [1] + [0] * (n - 1), [0] * n, 0)
    return pq.apply_pauli_t(e, [0, 1] + [0] * (n - 2), [0] * n, 0)


def test_every_pointer_parameter_is_bound_as_c_void_p() -> None:
    lib = mk.get_magic_kernels_lib()
    assert _typed_pointer_params() == []
    n_void = sum(
        t is ctypes.c_void_p for name, _, _ in mk._SPECS for t in getattr(lib, name).argtypes
    )
    assert n_void > 100  # the typed pointers did not just vanish from the specs


def test_addr_is_the_data_address_and_zero_for_empty_arrays() -> None:
    a = np.arange(12, dtype=np.uint64).reshape(3, 4)
    assert mk.addr(a) == a.ctypes.data == ctypes.addressof(kl.c_ptr(a, kl.u64_p).contents)
    assert mk.addr(a[1:]) == a.ctypes.data + 4 * 8
    assert mk.addr(np.empty((0, 4), np.uint64)) == 0
    assert mk.w_addr(np.empty(0, np.complex128)) == 0


def test_w_addr_returns_the_complex128_data_address() -> None:
    w = np.ones((3, 2), np.complex128)
    assert mk.w_addr(w) == w.ctypes.data


def _unaligned_complex128() -> np.ndarray:
    raw = np.zeros(4 * 16 + 1, np.uint8)
    return raw[1:].view(np.complex128)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4, np.complex64),
        np.ones(4, np.float64),
        np.ones(8, np.complex128)[::2],
        np.ones(4, ">c16"),
        _unaligned_complex128(),
        np.array(1 + 1j),
    ],
    ids=["complex64", "float64", "strided", "big_endian", "unaligned", "zero_dim"],
)
def test_w_addr_rejects_what_c_w_ptr_rejects(bad: np.ndarray) -> None:
    with pytest.raises(TypeError, match="complex128"):
        mk.w_addr(bad)
    with pytest.raises(TypeError, match="complex128"):
        kl.c_w_ptr(bad)


def test_integer_addresses_and_typed_pointers_give_identical_kernel_output() -> None:
    lib = mk.get_magic_kernels_lib()
    a = pq.coerce_pauli_words([1, 0, 1, 0, 0], 5)
    b = pq.coerce_pauli_words([0, 1, 1, 0, 0], 5)
    results = []
    for typed in (False, True):
        e = _shot()
        ptr = (lambda x, t: kl.c_ptr(x, t)) if typed else (lambda x, t: mk.addr(x))
        w = kl.c_w_ptr(e.weights) if typed else mk.w_addr(e.weights)
        lib.magic_apply_pauli_batch(
            e.chi, e.d, e.n_words, w, ptr(e.h_words, kl.u64_p), ptr(e.ell, kl.u8_p),
            ptr(e.A_words, kl.u64_p), ptr(e.a_union, kl.u64_p), ptr(a, kl.u64_p),
            ptr(b, kl.u64_p), ctypes.c_uint8(1),
        )
        results.append([getattr(e, f).tobytes() for f in ("weights", "h_words", "ell")])
    assert results[0] == results[1]


def test_kernel_wrappers_build_no_typed_pointer_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"cast": 0, "c_ptr": 0}
    real_cast = ctypes.cast

    def counting_cast(*args: Any) -> Any:
        calls["cast"] += 1
        return real_cast(*args)

    def counting_c_ptr(*args: Any) -> Any:
        calls["c_ptr"] += 1
        return kl.c_ptr(*args)

    monkeypatch.setattr(ctypes, "cast", counting_cast)
    monkeypatch.setattr(pq, "_c_ptr", counting_c_ptr)
    e = _shot()
    e = pq.apply_pauli(e, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], 1)
    e = pq.apply_quarter_turn(e, [0, 0, 1, 0, 0], [0] * 5, 0)
    e = pq.apply_pauli_t(e, [0, 0, 0, 1, 0], [0] * 5, 0)
    e, _z = pq.project_diagonal(e, [0, 0, 0, 0, 1])
    pq.measure_expansions_batch([e], ([0, 0, 1, 0, 0], [0] * 5, 0), rng=np.random.default_rng(0))
    assert calls == {"cast": 0, "c_ptr": 0}


def test_typed_pointer_helpers_stay_importable_from_phase_qfe() -> None:
    # magic_kernels_test (frozen) builds typed pointers through these names.
    a = np.zeros(2, np.uint64)
    assert ctypes.addressof(pq._c_ptr(a, pq._c_u64_p).contents) == a.ctypes.data


def test_no_complex128_array_reaches_the_unchecked_addr_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Weights must always go through the strict `w_addr`; the plain `addr` does no dtype check.
    unchecked: list[Any] = []
    checked: list[Any] = []
    real_addr, real_w_addr = mk.addr, mk.w_addr

    def spy_addr(a: Any) -> int:
        if a.dtype.kind == "c":
            unchecked.append(a)
        return real_addr(a)

    def spy_w_addr(w: Any) -> int:
        checked.append(w)
        return real_w_addr(w)

    monkeypatch.setattr(pq, "_addr", spy_addr)
    monkeypatch.setattr(pq, "_w_addr", spy_w_addr)
    z1 = [0, 0, 0, 0, 1]
    steps = {
        "apply_pauli": lambda e: pq.apply_pauli(e, [1, 0, 0, 0, 0], z1, 0),
        "apply_quarter_turn": lambda e: pq.apply_quarter_turn(e, [0, 0, 1, 0, 0], [0] * 5, 0),
        "apply_pauli_t": lambda e: pq.apply_pauli_t(e, [0, 0, 0, 1, 0], [0] * 5, 0),
        "apply_ccz": lambda e: pq.apply_ccz(e, _ccz_axes()),
        "project_diagonal": lambda e: pq.project_diagonal(e, z1)[0],
        "merge_frames": pq.merge_frames,
        "measure_expansions_batch": lambda e: pq.measure_expansions_batch(
            [e], ([0, 0, 1, 0, 0], [0] * 5, 0), rng=np.random.default_rng(0))[3][0],
    }
    for name, step in steps.items():
        shot = _shot()
        unchecked.clear()
        checked.clear()
        step(shot)
        assert checked, f"{name} never used w_addr"
        assert not unchecked, f"{name} passed a complex array to the unchecked addr()"


def _ccz_axes() -> list[Any]:
    def z(i: int) -> tuple[list[int], list[int], int]:
        return [0] * 5, [int(j == i) for j in range(5)], 0

    return [z(2), z(3), z(4)]
