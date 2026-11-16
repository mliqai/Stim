"""ABI, wrapper, sentinel rebuild, and RLIMIT_AS failure tests for `coset_kernels`."""

from __future__ import annotations

import copy
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stimside._native import _kernel_loader as kl
from stimside._native import coset_kernels as ck

_SRC = Path(ck.__file__).resolve().parents[2]
_SENTINEL_U64 = np.uint64(0xAAAAAAAAAAAAAAAA)
_SENTINEL_U8 = np.uint8(0xAA)
_SENTINEL_I32 = np.int32(-1431655766)


def _under_sanitizer() -> bool:
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__asan_init") or hasattr(libc, "__tsan_init")
    except OSError:
        return False


def _ptr(arr: np.ndarray[Any, Any], c_type: Any) -> Any:
    return arr.ctypes.data_as(c_type) if arr.size > 0 else ctypes.cast(0, c_type)


def _make_sentinel_export_bufs(nw: int, cap_d: int) -> dict[str, np.ndarray[Any, Any]]:
    m = max(1, cap_d)
    return {
        "h": np.full(nw, _SENTINEL_U64, dtype=np.uint64),
        "au": np.full(nw, _SENTINEL_U64, dtype=np.uint64),
        "A": np.full((m, nw), _SENTINEL_U64, dtype=np.uint64),
        "ell": np.full(m, _SENTINEL_U8, dtype=np.uint8),
        "Gamma": np.full((m, m), _SENTINEL_U8, dtype=np.uint8),
        "P": np.full(m, _SENTINEL_I32, dtype=np.int32),
    }


def _assert_sentinels_untouched(bufs: dict[str, np.ndarray[Any, Any]]) -> None:
    assert np.all(bufs["h"] == _SENTINEL_U64)
    assert np.all(bufs["au"] == _SENTINEL_U64)
    assert np.all(bufs["A"] == _SENTINEL_U64)
    assert np.all(bufs["ell"] == _SENTINEL_U8)
    assert np.all(bufs["Gamma"] == _SENTINEL_U8)
    assert np.all(bufs["P"] == _SENTINEL_I32)


def _call_export_v3(
    fn: Any, eng: Any, b: int, cap_w: int, cap_d: int, bufs: dict[str, np.ndarray[Any, Any]]
) -> int:
    return int(
        fn(
            eng, b, cap_w, cap_d,
            _ptr(bufs["h"], kl.u64_p), _ptr(bufs["au"], kl.u64_p),
            _ptr(bufs["A"], kl.u64_p), _ptr(bufs["ell"], kl.u8_p),
            _ptr(bufs["Gamma"], kl.u8_p), _ptr(bufs["P"], kl.i32_p),
        )
    )


def _make_shot_inputs(nq: int, nw: int, d: int) -> tuple[np.ndarray[Any, Any], ...]:
    h = np.zeros(nw, dtype=np.uint64)
    h[0] = np.uint64(0x5A)
    A = np.zeros((d, nw), dtype=np.uint64)
    for r in range(d):
        A[r, r >> 6] = np.uint64(1) << np.uint64(r & 63)
    au = np.bitwise_or.reduce(A, axis=0) if d > 0 else np.zeros(nw, dtype=np.uint64)
    ell = (np.arange(d, dtype=np.uint8) * 3) & np.uint8(3)
    Gamma = np.zeros((d, d), dtype=np.uint8)
    for r in range(d):
        for c in range(r + 1, d):
            val = np.uint8((r + c) & 1)
            Gamma[r, c] = val
            Gamma[c, r] = val
    P = np.arange(d, dtype=np.int32)
    return h, au, A, ell, Gamma, P


def test_abi_versions_and_export_shot_v3_matches_v2() -> None:
    lib = ck.get_coset_kernels_lib()
    assert lib.coset_kernels_abi_v2() == 2
    assert lib.coset_kernels_abi_v3() == 3

    rng = np.arange(1, 3, dtype=np.uint64)
    eng = lib.engine_create(70, 2, rng.ctypes.data_as(kl.u64_p))
    try:
        nw = lib.engine_n_words(eng)
        assert nw == 2
        h0, au0, A0, ell0, G0, P0 = _make_shot_inputs(70, nw, 0)
        lib.engine_import_shot(
            eng, 0, 0, _ptr(h0, kl.u64_p), _ptr(au0, kl.u64_p),
            _ptr(A0, kl.u64_p), _ptr(ell0, kl.u8_p), _ptr(G0, kl.u8_p), _ptr(P0, kl.i32_p),
        )
        h1, au1, A1, ell1, G1, P1 = _make_shot_inputs(70, nw, 5)
        lib.engine_import_shot(
            eng, 1, 5, _ptr(h1, kl.u64_p), _ptr(au1, kl.u64_p),
            _ptr(A1, kl.u64_p), _ptr(ell1, kl.u8_p), _ptr(G1, kl.u8_p), _ptr(P1, kl.i32_p),
        )

        v3_d0 = _make_sentinel_export_bufs(nw, 0)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 0, nw, 0, v3_d0) == 0
        assert np.array_equal(v3_d0["h"], h0) and np.array_equal(v3_d0["au"], au0)

        v2 = _make_sentinel_export_bufs(nw, 5)
        lib.engine_export_shot(
            eng, 1, _ptr(v2["h"], kl.u64_p), _ptr(v2["au"], kl.u64_p),
            _ptr(v2["A"], kl.u64_p), _ptr(v2["ell"], kl.u8_p),
            _ptr(v2["Gamma"], kl.u8_p), _ptr(v2["P"], kl.i32_p),
        )
        v3 = _make_sentinel_export_bufs(nw, 5)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, nw, 5, v3) == 5
        for key in ("h", "au", "A", "ell", "Gamma", "P"):
            assert np.array_equal(v3[key], v2[key]), key
        assert np.array_equal(v3["A"], A1) and np.array_equal(v3["Gamma"], G1)
    finally:
        lib.engine_destroy(eng)


def test_export_shot_v3_error_codes_leave_sentinel_buffers_untouched() -> None:
    lib = ck.get_coset_kernels_lib()
    rng = np.arange(1, 3, dtype=np.uint64)
    eng = lib.engine_create(70, 2, rng.ctypes.data_as(kl.u64_p))
    try:
        nw = lib.engine_n_words(eng)
        h1, au1, A1, ell1, G1, P1 = _make_shot_inputs(70, nw, 4)
        lib.engine_import_shot(
            eng, 1, 4, _ptr(h1, kl.u64_p), _ptr(au1, kl.u64_p),
            _ptr(A1, kl.u64_p), _ptr(ell1, kl.u8_p), _ptr(G1, kl.u8_p), _ptr(P1, kl.i32_p),
        )

        # -4: cap_d < s.d writes nothing
        b_small_d = _make_sentinel_export_bufs(nw, 3)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, nw, 3, b_small_d) == -4
        _assert_sentinels_untouched(b_small_d)

        # -3: cap_words < n_words writes nothing
        b_small_w = _make_sentinel_export_bufs(nw, 4)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, nw - 1, 4, b_small_w) == -3
        _assert_sentinels_untouched(b_small_w)

        # -2: invalid shot index or negative caps
        b_idx = _make_sentinel_export_bufs(nw, 4)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, -1, nw, 4, b_idx) == -2
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 2, nw, 4, b_idx) == -2
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, -1, 4, b_idx) == -2
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, nw, -1, b_idx) == -2
        _assert_sentinels_untouched(b_idx)

        # -1: null ptr/out_h/out_au or null d>0 buffers
        b_null = _make_sentinel_export_bufs(nw, 4)
        null_u64, null_u8, null_i32 = (
            ctypes.cast(0, kl.u64_p), ctypes.cast(0, kl.u8_p), ctypes.cast(0, kl.i32_p),
        )
        assert int(lib.engine_export_shot_v3(
            0, 1, nw, 4, _ptr(b_null["h"], kl.u64_p), _ptr(b_null["au"], kl.u64_p),
            _ptr(b_null["A"], kl.u64_p), _ptr(b_null["ell"], kl.u8_p),
            _ptr(b_null["Gamma"], kl.u8_p), _ptr(b_null["P"], kl.i32_p),
        )) == -1
        assert int(lib.engine_export_shot_v3(
            eng, 1, nw, 4, null_u64, _ptr(b_null["au"], kl.u64_p),
            _ptr(b_null["A"], kl.u64_p), _ptr(b_null["ell"], kl.u8_p),
            _ptr(b_null["Gamma"], kl.u8_p), _ptr(b_null["P"], kl.i32_p),
        )) == -1
        assert int(lib.engine_export_shot_v3(
            eng, 1, nw, 4, _ptr(b_null["h"], kl.u64_p), _ptr(b_null["au"], kl.u64_p),
            null_u64, _ptr(b_null["ell"], kl.u8_p),
            _ptr(b_null["Gamma"], kl.u8_p), _ptr(b_null["P"], kl.i32_p),
        )) == -1
        assert int(lib.engine_export_shot_v3(
            eng, 1, nw, 4, _ptr(b_null["h"], kl.u64_p), _ptr(b_null["au"], kl.u64_p),
            _ptr(b_null["A"], kl.u64_p), null_u8,
            _ptr(b_null["Gamma"], kl.u8_p), _ptr(b_null["P"], kl.i32_p),
        )) == -1
        assert int(lib.engine_export_shot_v3(
            eng, 1, nw, 4, _ptr(b_null["h"], kl.u64_p), _ptr(b_null["au"], kl.u64_p),
            _ptr(b_null["A"], kl.u64_p), _ptr(b_null["ell"], kl.u8_p),
            null_u8, _ptr(b_null["P"], kl.i32_p),
        )) == -1
        assert int(lib.engine_export_shot_v3(
            eng, 1, nw, 4, _ptr(b_null["h"], kl.u64_p), _ptr(b_null["au"], kl.u64_p),
            _ptr(b_null["A"], kl.u64_p), _ptr(b_null["ell"], kl.u8_p),
            _ptr(b_null["Gamma"], kl.u8_p), null_i32,
        )) == -1
        _assert_sentinels_untouched(b_null)
    finally:
        lib.engine_destroy(eng)


def test_stale_so_missing_export_shot_v3_triggers_sentinel_rebuild(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("requires C++ compiler")
    src = tmp_path / "src"
    native = src / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# marker\n", encoding="utf-8")
    (src / "stimside" / "__init__.py").write_text("", encoding="utf-8")
    (native / "__init__.py").write_text("", encoding="utf-8")
    for fname in ("_kernel_loader.py", "coset_kernels.py", "coset_kernels.cpp"):
        shutil.copy2(_SRC / "stimside" / "_native" / fname, native / fname)

    cpp_full = (native / "coset_kernels.cpp").read_text(encoding="utf-8")
    stale_cpp = cpp_full.replace("engine_export_shot_v3", "engine_export_shot_v3_disabled")
    assert stale_cpp != cpp_full
    stale_src = native / "stale_coset_kernels.cpp"
    stale_src.write_text(stale_cpp, encoding="utf-8")
    so_path = native / "libcoset_kernels.so"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fPIC", "-shared", str(stale_src), "-o", str(so_path)],
        check=True,
    )
    stale_src.unlink()
    newer_ns = (native / "coset_kernels.cpp").stat().st_mtime_ns + 10_000_000_000
    os.utime(so_path, ns=(newer_ns, newer_ns))

    probe = (
        "from stimside._native import coset_kernels as ck; "
        "lib = ck.get_coset_kernels_lib(); "
        "assert hasattr(lib, 'engine_export_shot_v3'); "
        "print(lib.coset_kernels_abi_v3())"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(src)}
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=False
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "3"


def test_proxy_attribute_error_copy_and_specs_partition() -> None:
    lib = ck.get_coset_kernels_lib()
    with pytest.raises(AttributeError):
        _ = lib.nonexistent_coset_kernel_symbol  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = lib.does_not_exist  # type: ignore[attr-defined]
    copied = copy.copy(lib)
    assert copied is not lib
    assert copied.coset_kernels_abi_v3() == 3

    spec_names = frozenset(spec[0] for spec in ck._SPECS)
    assert ck._SPEC_NAMES == spec_names
    assert {"engine_create"} | ck._NO_GUARD | ck._VOID_FNS | ck._INT_FNS == spec_names
    assert len(ck._NO_GUARD) + len(ck._VOID_FNS) + len(ck._INT_FNS) + 1 == len(spec_names)


class _StubFn:
    def __init__(self, impl: Any, argtypes: Any, restype: Any) -> None:
        self._impl = impl
        self.argtypes = list(argtypes)
        self.restype = restype

    def __call__(self, *args: Any) -> Any:
        return self._impl(*args)


class _FakeRaw:
    def __init__(self) -> None:
        self.create_ptr = 123
        self.create_err = 0
        self.failed_code = 0
        self.int_rc = 0
        self.void_calls: list[str] = []
        self.int_calls: list[str] = []

        for name, argtypes, restype in ck._SPECS:
            if name == "engine_create":
                fn: Any = lambda *a: self.create_ptr
            elif name == "engine_create_last_error":
                fn = lambda: self.create_err
            elif name == "engine_failed":
                fn = lambda ptr: self.failed_code
            elif name in ck._VOID_FNS:
                fn = lambda *a, _n=name: self.void_calls.append(_n)
            elif name in ck._INT_FNS:
                fn = lambda *a, _n=name: (self.int_calls.append(_n), self.int_rc)[1]
            else:
                fn = lambda *a: 0
            setattr(self, name, _StubFn(fn, argtypes, restype))


def test_coset_kernels_lib_fake_raw_kills_m5_m5b_m5c() -> None:
    raw = _FakeRaw()
    lib = ck._CosetKernelsLib(raw)  # type: ignore[arg-type]

    # M5c: engine_create NULL check
    assert lib.engine_create(2, 1, None) == 123
    raw.create_ptr = 0
    raw.create_err = 1
    with pytest.raises(MemoryError, match="allocation failed"):
        lib.engine_create(2, 1, None)
    raw.create_err = 2
    with pytest.raises(RuntimeError, match="code=2"):
        lib.engine_create(2, 1, None)

    # M5: every _VOID_FNS checks engine_failed when ptr != 0
    for name in sorted(ck._VOID_FNS):
        fn = getattr(lib, name)
        raw.failed_code = 1
        with pytest.raises(MemoryError, match="allocation failed"):
            fn(99)
        raw.failed_code = 2
        with pytest.raises(RuntimeError, match="code=2"):
            fn(99)
        fn(0)  # ptr == 0 does not check engine_failed
        raw.failed_code = 0
        fn(99)

    # M5b: every _INT_FNS checks engine_failed iff rc < 0 and ptr != 0
    for name in sorted(ck._INT_FNS):
        fn = getattr(lib, name)
        raw.int_rc = 0
        raw.failed_code = 1
        assert fn(99) == 0  # rc >= 0 does not raise
        raw.int_rc = -2
        raw.failed_code = 0
        assert fn(99) == -2  # healthy negative rc returned
        raw.failed_code = 1
        with pytest.raises(MemoryError, match="allocation failed"):
            fn(99)
        raw.failed_code = 2
        with pytest.raises(RuntimeError, match="code=2"):
            fn(99)
        assert fn(0) == -2  # ptr == 0 does not query engine_failed


_RLIMIT_CHILD = r"""
import ctypes, resource, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._native import _kernel_loader as kl
from stimside._native import coset_kernels as ck

lib = ck.get_coset_kernels_lib()
raw = lib._raw

# Cap address space at current VmSize + 256 MiB so 2+ GiB allocations fail with bad_alloc.
vm_kb = 0
with open("/proc/self/status", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("VmSize:"):
            vm_kb = int(line.split()[1])
            break
cap = max(vm_kb * 1024 + 256 * 1024 * 1024, 512 * 1024 * 1024)
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
if hard != resource.RLIM_INFINITY:
    cap = min(cap, hard)
resource.setrlimit(resource.RLIMIT_AS, (cap, hard))

# 1. Verify engine_create OOM raises MemoryError (M5c on real C++)
big_rng = np.arange(1, 4001, dtype=np.uint64)
try:
    lib.engine_create(64000, 4000, big_rng.ctypes.data_as(kl.u64_p))
    raise AssertionError("expected MemoryError from huge engine_create")
except MemoryError:
    pass
assert lib.engine_create_last_error() == 1

# 2. Create a small engine and OOM engine_import_shot (M5 on real C++)
rng = np.arange(1, 2, dtype=np.uint64)
eng = lib.engine_create(200_000, 1, rng.ctypes.data_as(kl.u64_p))
nw = lib.engine_n_words(eng)
h = np.zeros(nw, np.uint64)
au = np.zeros(nw, np.uint64)
A = np.zeros((1, nw), np.uint64)
ell = np.zeros(1, np.uint8)
G = np.zeros((1, 1), np.uint8)
P = np.zeros(200_000, np.int32)
p64 = lambda a: a.ctypes.data_as(kl.u64_p)
p8 = lambda a: a.ctypes.data_as(kl.u8_p)
p32 = lambda a: a.ctypes.data_as(kl.i32_p)
pi64 = lambda a: a.ctypes.data_as(kl.i64_p)
pi8 = lambda a: a.ctypes.data_as(kl.i8_p)
try:
    lib.engine_import_shot(eng, 0, 200_000, p64(h), p64(au), p64(A), p8(ell), p8(G), p32(P))
    raise AssertionError("expected MemoryError from 200k import")
except MemoryError:
    pass
assert lib.engine_failed(eng) == 1

# 3. Every guarded export raises MemoryError on failed engine; raw returns < 0 & keeps sentinels
out_idx = np.full(1, -99, np.int32)
out_d = np.full(1, -99, np.int32)
sh = np.full(nw, 0xAA, np.uint64)
sau = np.full(nw, 0xAA, np.uint64)
sA = np.full((1, nw), 0xAA, np.uint64)
sell = np.full(1, 0xAA, np.uint8)
sG = np.full((1, 1), 0xAA, np.uint8)
sP = np.full(1, -1431655766, np.int32)
tg = np.zeros(1, np.int64)
u8_nb = np.full(1, 0xAA, np.uint8)
u8_1 = np.full(1, 0xAA, np.uint8)
i8_1 = np.full((1, 1), 77, np.int8)
ax, az = ctypes.c_int(9), ctypes.c_int(9)
pw = np.full(nw, 0xAA, np.uint64)
kp = np.full(1, 0xAA, np.uint8)
N = 200_000

guarded_calls = [
    ("engine_num_qubits", lambda: lib.engine_num_qubits(eng)),
    ("engine_batch_size", lambda: lib.engine_batch_size(eng)),
    ("engine_n_words", lambda: lib.engine_n_words(eng)),
    ("engine_clear", lambda: lib.engine_clear(eng)),
    ("engine_set_rng_states", lambda: lib.engine_set_rng_states(eng, p64(rng))),
    ("engine_get_rng_states", lambda: lib.engine_get_rng_states(eng, 1, p64(sh))),
    ("engine_get_active_shot_indices",
     lambda: lib.engine_get_active_shot_indices(eng, p32(out_idx), p32(out_d))),
    ("engine_export_shot",
     lambda: lib.engine_export_shot(
         eng, 0, p64(sh), p64(sau), p64(sA), p8(sell), p8(sG), p32(sP))),
    ("engine_export_shot_v3",
     lambda: lib.engine_export_shot_v3(
         eng, 0, nw, 1, p64(sh), p64(sau), p64(sA), p8(sell), p8(sG), p32(sP))),
    ("engine_import_shot",
     lambda: lib.engine_import_shot(
         eng, 0, 0, p64(h), p64(au), p64(A), p8(ell), p8(G), p32(P))),
    ("engine_unitary_step_num_groups", lambda: lib.engine_unitary_step_num_groups(eng, 0)),
    ("engine_register_unitary_step",
     lambda: lib.engine_register_unitary_step(
         eng, 0, 1, 0, pi64(tg), p8(u8_1), p8(u8_1), p8(u8_1), p64(h), p64(h),
         p8(u8_1), p8(u8_1), p8(u8_1), p64(h), p64(h), p8(u8_1), N, nw, p64(pw), p64(pw))),
    ("engine_register_unitary_group_evict",
     lambda: lib.engine_register_unitary_group_evict(eng, 0, 0, N, nw, p64(pw), p64(pw))),
    ("engine_batch_det_meas_step",
     lambda: lib.engine_batch_det_meas_step(
         eng, 0, pi64(tg), p8(u8_1), p64(h), p8(u8_nb), p8(u8_1),
         p64(pw), p64(pw), p8(u8_nb), p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_mixed_substep",
     lambda: lib.engine_batch_mixed_substep(
         eng, 0, 0, 0, 0, p64(h), p64(h), p64(h), 0, 0, pi64(tg), 0, pi64(tg),
         p8(u8_nb), p8(u8_1), p8(u8_nb), p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_evict",
     lambda: lib.engine_batch_evict(
         eng, p64(pw), p64(pw), p8(u8_nb), p8(u8_nb),
         p8(u8_nb), p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_unmatched_noisy_reset_z",
     lambda: lib.engine_batch_unmatched_noisy_reset_z(
         eng, p8(u8_nb), p8(u8_nb), p8(u8_nb), p64(pw), p64(pw), p8(kp),
         p64(pw), p8(u8_nb), p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_peek_pauli_batch",
     lambda: lib.engine_peek_pauli_batch(
         eng, 1, pi64(tg), 2, p8(u8_nb), p8(u8_nb),
         p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp), pi8(i8_1))),
    ("engine_peek_pauli_batch_v3",
     lambda: lib.engine_peek_pauli_batch_v3(
         eng, 1, pi64(tg), 2, N, 1, p8(u8_nb), N, 1, p8(u8_nb), N, nw,
         p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp), 1, 1, pi8(i8_1))),
    ("engine_postselect_batch",
     lambda: lib.engine_postselect_batch(
         eng, 0, 2, 1, p8(u8_1), 1, p8(u8_1), N, 1, p8(u8_nb), N, 1, p8(u8_nb), N, nw,
         p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_project_pauli",
     lambda: lib.engine_project_pauli(
         eng, 2, N, 1, p8(u8_nb), N, 1, p8(u8_nb), -1, pi64(tg),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), N, nw,
         p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_inject_group_v3",
     lambda: lib.engine_inject_group_v3(
         eng, 0, 0, 1, p8(u8_1), N, 1, p8(u8_nb), N, 1, p8(u8_nb),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_inject_general_clifford_v3",
     lambda: lib.engine_batch_inject_general_clifford_v3(
         eng, 0, 1, 1, p8(u8_1), N, 1, p8(u8_nb), N, 1, p8(u8_nb),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_evict_v3",
     lambda: lib.engine_batch_evict_v3(
         eng, N, nw, p64(pw), p64(pw), N, 1, p8(u8_nb), N, 1, p8(u8_nb),
         N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_inject_general_clifford",
     lambda: lib.engine_batch_inject_general_clifford(
         eng, 0, 1, 0, pi64(tg), p8(u8_1), p8(u8_nb), p8(u8_nb),
         p8(u8_1), p8(u8_1), p8(u8_1), p64(h), p64(h), p8(u8_1),
         p8(u8_1), p8(u8_1), p64(h), p64(h), p8(u8_1), p64(pw), p64(pw),
         p8(u8_nb), p8(u8_nb), ctypes.byref(ax), ctypes.byref(az))),
]
for name, fn in guarded_calls:
    try:
        fn()
        raise AssertionError(f"{name} did not raise MemoryError on failed engine")
    except MemoryError:
        pass

# 4. Raw unguarded int exports return < 0 and leave sentinels untouched
assert raw.engine_num_qubits(eng) == -2
assert raw.engine_batch_size(eng) == -2
assert raw.engine_n_words(eng) == -2
assert raw.engine_set_rng_states(eng, p64(rng)) == -2
assert raw.engine_get_rng_states(eng, 1, p64(sh)) == -2
assert raw.engine_unitary_step_num_groups(eng, 0) == -2
assert raw.engine_get_active_shot_indices(eng, p32(out_idx), p32(out_d)) == -2
assert raw.engine_export_shot_v3(
    eng, 0, nw, 1, p64(sh), p64(sau), p64(sA), p8(sell), p8(sG), p32(sP)) == -6
assert raw.engine_peek_pauli_batch_v3(
    eng, 1, pi64(tg), 2, N, 1, p8(u8_nb), N, 1, p8(u8_nb), N, nw,
    p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp), 1, 1, pi8(i8_1)) == -8
assert raw.engine_postselect_batch(
    eng, 0, 2, 1, p8(u8_1), 1, p8(u8_1), N, 1, p8(u8_nb), N, 1, p8(u8_nb), N, nw,
    p64(pw), p64(pw), p64(pw), p64(pw), p8(kp), p8(kp),
    N, 1, p8(u8_nb), N, 1, p8(u8_nb), ctypes.byref(ax), ctypes.byref(az)) == -9
assert ax.value == 0 and az.value == 0
assert np.all(out_idx == -99) and np.all(out_d == -99)
assert np.all(sh == 0xAA) and np.all(sau == 0xAA) and np.all(sA == 0xAA)
assert np.all(sell == 0xAA) and np.all(sG == 0xAA) and np.all(sP == -1431655766)
assert np.all(u8_nb == 0xAA) and np.all(u8_1 == 0xAA)
assert np.all(i8_1 == 77)
lib.engine_destroy(eng)
print("RLIMIT_AS_OK")
"""


@pytest.mark.skipif(_under_sanitizer(), reason="sanitizer shadow memory conflicts with RLIMIT_AS")
def test_rlimit_as_oom_and_sticky_failed_engine_guard() -> None:
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(_SRC),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _RLIMIT_CHILD, str(_SRC)],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "RLIMIT_AS_OK"


def test_engine_owns_rng_states_and_validates_set_get_args() -> None:
    lib = ck.get_coset_kernels_lib()
    rng1 = np.array([11, 22], dtype=np.uint64)
    eng = lib.engine_create(4, 2, rng1.ctypes.data_as(kl.u64_p))
    try:
        rng1[:] = 999
        del rng1
        out = np.zeros(2, dtype=np.uint64)
        assert lib.engine_get_rng_states(eng, 2, out.ctypes.data_as(kl.u64_p)) == 0
        assert np.array_equal(out, np.array([11, 22], dtype=np.uint64))

        null_u64 = ctypes.cast(0, kl.u64_p)
        assert lib.engine_set_rng_states(eng, null_u64) == -1
        assert lib.engine_get_rng_states(eng, 1, out.ctypes.data_as(kl.u64_p)) == -3
        assert lib.engine_get_rng_states(eng, 2, null_u64) == -1
        assert lib.engine_failed(eng) == 0

        rng2 = np.array([33, 44], dtype=np.uint64)
        assert lib.engine_set_rng_states(eng, rng2.ctypes.data_as(kl.u64_p)) == 0
        rng2[:] = 0
        assert lib.engine_get_rng_states(eng, 2, out.ctypes.data_as(kl.u64_p)) == 0
        assert np.array_equal(out, np.array([33, 44], dtype=np.uint64))
    finally:
        lib.engine_destroy(eng)
