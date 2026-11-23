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


def _under_tsan() -> bool:
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "__tsan_init")
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


def _seed_active_d_shot(
    lib: Any, eng: Any, nq: int, bs: int, nw: int, b: int, d: int
) -> tuple[np.ndarray[Any, Any], ...]:
    h = np.zeros(nw, dtype=np.uint64)
    h[0] = np.uint64(0x5A)
    A = np.zeros((d, nw), dtype=np.uint64)
    for r in range(d):
        A[r, r >> 6] = np.uint64(1) << np.uint64(r & 63)
    au = np.bitwise_or.reduce(A, axis=0) if d > 0 else np.zeros(nw, dtype=np.uint64)
    nf = d + 1
    tg = np.zeros(1, dtype=np.int64)
    cm = np.ones((1, bs), dtype=np.uint8)
    cm[0, b] = 0
    xs = np.zeros((nq, bs), dtype=np.uint8)
    zs = np.zeros((nq, bs), dtype=np.uint8)
    z_pw = np.zeros((nq, nw), dtype=np.uint64)
    ox = np.zeros((nq, bs), dtype=np.uint8)
    oz = np.zeros((nq, bs), dtype=np.uint8)
    ax, az = ctypes.c_int(0), ctypes.c_int(0)
    fk = np.ones(nf, dtype=np.uint8)
    fk[0] = 0
    zf = np.zeros(nf, dtype=np.uint8)
    a1 = np.zeros((nf, nw), dtype=np.uint64)
    b1 = np.zeros((nf, nw), dtype=np.uint64)
    kap1 = np.zeros(nf, dtype=np.uint8)
    a1[0] = h
    for r in range(d):
        a1[r + 1] = A[r]
        b1[r + 1, ((r + 1) % d) >> 6] = np.uint64(1) << np.uint64(((r + 1) % d) & 63)
        kap1[r + 1] = np.uint8(r & 3)
    b2 = np.zeros((nf, nw), dtype=np.uint64)
    lib.engine_batch_inject_general_clifford(
        eng, 1, 1, nf, _ptr(tg, kl.i64_p), _ptr(cm, kl.u8_p),
        _ptr(xs, kl.u8_p), _ptr(zs, kl.u8_p), _ptr(fk, kl.u8_p),
        _ptr(zf, kl.u8_p), _ptr(zf, kl.u8_p),
        _ptr(a1, kl.u64_p), _ptr(b1, kl.u64_p), _ptr(kap1, kl.u8_p),
        _ptr(zf, kl.u8_p), _ptr(zf, kl.u8_p),
        _ptr(b2, kl.u64_p), _ptr(b2, kl.u64_p), _ptr(zf, kl.u8_p),
        _ptr(z_pw, kl.u64_p), _ptr(z_pw, kl.u64_p),
        _ptr(ox, kl.u8_p), _ptr(oz, kl.u8_p), ctypes.byref(ax), ctypes.byref(az),
    )
    P = np.arange(d, dtype=np.int32)
    return h, au, A, P


def test_abi_versions_and_export_shot_v3_matches_v2() -> None:
    lib = ck.get_coset_kernels_lib()
    assert lib.coset_kernels_abi_v2() == 2
    assert lib.coset_kernels_abi_v3() == 3
    assert lib.coset_kernels_abi_v4() == 4

    rng = np.arange(1, 3, dtype=np.uint64)
    eng = lib.engine_create(70, 2, rng.ctypes.data_as(kl.u64_p))
    try:
        nw = lib.engine_n_words(eng)
        assert nw == 2
        h1, au1, A1, P1 = _seed_active_d_shot(lib, eng, 70, 2, nw, 1, 5)

        v3_d0 = _make_sentinel_export_bufs(nw, 0)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 0, nw, 0, v3_d0) == 0
        assert np.array_equal(v3_d0["h"], np.zeros(nw, dtype=np.uint64))
        assert np.array_equal(v3_d0["au"], np.zeros(nw, dtype=np.uint64))

        v3 = _make_sentinel_export_bufs(nw, 5)
        assert _call_export_v3(lib.engine_export_shot_v3, eng, 1, nw, 5, v3) == 5
        assert np.array_equal(v3["h"], h1)
        assert np.array_equal(v3["au"], au1)
        assert np.array_equal(v3["A"], A1)
        assert np.array_equal(v3["P"], P1)
        assert np.all(v3["ell"] < 4)
        assert np.array_equal(v3["Gamma"], v3["Gamma"].T)
        assert int(np.sum(v3["Gamma"])) > 0
    finally:
        lib.engine_destroy(eng)


def test_export_shot_v3_error_codes_leave_sentinel_buffers_untouched() -> None:
    lib = ck.get_coset_kernels_lib()
    rng = np.arange(1, 3, dtype=np.uint64)
    eng = lib.engine_create(70, 2, rng.ctypes.data_as(kl.u64_p))
    try:
        nw = lib.engine_n_words(eng)
        _seed_active_d_shot(lib, eng, 70, 2, nw, 1, 4)

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
    # Place under site-packages without setup.py so resolve_so() does not trigger source-tree
    # staleness; only load_cdll()'s full sentinel tuple can trigger the rebuild (kills K10).
    src = tmp_path / "site-packages"
    native = src / "stimside" / "_native"
    native.mkdir(parents=True)
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
import resource, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from stimside._native import _kernel_loader as kl
from stimside._native import coset_kernels as ck

lib = ck.get_coset_kernels_lib()

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

# Verify engine_create OOM raises MemoryError using a properly sized rng buffer (L-7 safe).
big_rng = np.arange(1, 4001, dtype=np.uint64)
try:
    lib.engine_create(64000, 4000, big_rng.ctypes.data_as(kl.u64_p))
    raise AssertionError("expected MemoryError from huge engine_create")
except MemoryError:
    pass
assert lib.engine_create_last_error() == 1
print("RLIMIT_AS_OK")
"""


@pytest.mark.skipif(
    _under_sanitizer() or not Path("/proc/self/status").is_file(),
    reason="needs Linux /proc/self/status and no sanitizer shadow memory",
)
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


_FI_STICKY_CPP = r"""
#include <cstdlib>
#include <new>
static thread_local long g_countdown = -1;
static thread_local long g_count = 0;
extern "C" void fi_arm(long n) { g_countdown = n; g_count = 0; }
extern "C" long fi_count() { return g_count; }
extern "C" void fi_disarm() { g_countdown = -1; }
static void* alloc(size_t sz) {
    g_count++;
    if (g_countdown == 0) { g_countdown = -1; throw std::bad_alloc(); }
    if (g_countdown > 0) --g_countdown;
    void* p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new(size_t sz) { return alloc(sz); }
void* operator new[](size_t sz) { return alloc(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
"""

_FI_STICKY_CHILD = r"""
import ctypes, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
fi = ctypes.CDLL(sys.argv[2])
fi.fi_arm.argtypes = [ctypes.c_long]
fi.fi_count.restype = ctypes.c_long
from stimside._native import _kernel_loader as kl
from stimside._native import coset_kernels as ck

lib = ck.get_coset_kernels_lib()
raw = lib._raw
N, B = 66, 2
p64 = lambda a: a.ctypes.data_as(kl.u64_p)
p8 = lambda a: a.ctypes.data_as(kl.u8_p)
p32 = lambda a: a.ctypes.data_as(kl.i32_p)
pi64 = lambda a: a.ctypes.data_as(kl.i64_p)
pi8 = lambda a: a.ctypes.data_as(kl.i8_p)

rng = np.array([11, 22], dtype=np.uint64)
eng = lib.engine_create(N, B, p64(rng))
nw = lib.engine_n_words(eng)
assert nw == 2

# Seed shot 0 (d=1 active) and shot 1 (d=0, h!=0 healed) via live kernels.
tg = np.array([0], np.int64)
fk = np.array([1], np.uint8)
f_u8 = np.zeros(1, np.uint8)
f_a = np.zeros((1, nw), np.uint64); f_a[0, 0] = 1
f_b = np.zeros((1, nw), np.uint64); f_b[0, 0] = 1
x2x = np.zeros((N, nw), np.uint64); x2x[0, 0] = 1
x2z = np.zeros((N, nw), np.uint64); x2z[0, 0] = 1
z2x = np.zeros((N, nw), np.uint64)
z2z = np.zeros((N, nw), np.uint64); z2z[0, 0] = 1; z2z[1, 0] = 1
kp = np.zeros(N, np.uint8)
z_pw = np.zeros((N, nw), np.uint64)
z_xs = np.zeros((N, B), np.uint8)
z_zs = np.zeros((N, B), np.uint8)
z_ox = np.zeros((N, B), np.uint8)
z_oz = np.zeros((N, B), np.uint8)
s_ax, s_az = ctypes.c_int(0), ctypes.c_int(0)
cm_s1 = np.array([[1, 0]], np.uint8)
lib.engine_batch_inject_general_clifford(
    eng, 1, 1, 1, pi64(tg), p8(cm_s1), p8(z_xs), p8(z_zs), p8(fk),
    p8(f_u8), p8(f_u8), p64(f_a), p64(z_pw[:1]), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(z_pw[:1]), p64(z_pw[:1]), p8(f_u8),
    p64(z_pw), p64(z_pw), p8(z_ox), p8(z_oz), ctypes.byref(s_ax), ctypes.byref(s_az),
)
m_tmp = np.zeros(B, np.uint8)
lib.engine_batch_mixed_substep(
    eng, 0, 0, 0, 0, p64(f_a[0]), p64(f_a[0]), p64(f_a[0]), 0, 0, pi64(tg), 0, pi64(tg),
    p8(z_xs), p8(m_tmp), p8(z_ox), p8(z_oz), ctypes.byref(s_ax), ctypes.byref(s_az),
)
cm_s0 = np.array([[0, 1]], np.uint8)
lib.engine_batch_inject_general_clifford(
    eng, 1, 1, 1, pi64(tg), p8(cm_s0), p8(z_xs), p8(z_zs), p8(fk),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p64(z_pw), p64(z_pw), p8(z_ox), p8(z_oz), ctypes.byref(s_ax), ctypes.byref(s_az),
)

# Register a valid 1-group, arity=1, n_factors=1 unitary step (step_id=0) before failing engine.
sid = lib.engine_register_unitary_step(
    eng, 1, 1, 1, pi64(tg), p8(fk), p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8), N, nw, p64(x2z), p64(z2z),
)
assert sid == 0

# Trigger real std::bad_alloc inside engine_register_unitary_step -> fail_kind == 1.
fi.fi_arm(0)
rc_oom = int(raw.engine_register_unitary_step(
    eng, 1, 1, 1, pi64(tg), p8(fk), p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8), N, nw, p64(x2z), p64(z2z),
))
fi.fi_disarm()
assert rc_oom == -9 and int(raw.engine_failed(eng)) == 1

# Properly sized sentinel buffers & valid effect-producing inputs for all 21 guarded exports.
out_rng = np.full(B, 0xAAAAAAAAAAAAAAAA, np.uint64)
out_idx = np.full(B, -99, np.int32)
out_d = np.full(B, -99, np.int32)
sh = np.full(nw, 0xAAAAAAAAAAAAAAAA, np.uint64)
sau = np.full(nw, 0xAAAAAAAAAAAAAAAA, np.uint64)
sA = np.full((1, nw), 0xAAAAAAAAAAAAAAAA, np.uint64)
sell = np.full(1, 0xAA, np.uint8)
sG = np.full((1, 1), 0xAA, np.uint8)
sP = np.full(1, -1431655766, np.int32)
xs = np.zeros((N, B), np.uint8); xs[0, :] = 1
zs = np.zeros((N, B), np.uint8)
xs_orig = xs.copy()
ox = np.full((N, B), 0xAA, np.uint8)
oz = np.full((N, B), 0xAA, np.uint8)
cm_row = np.zeros(B, np.uint8)
cm_mat = np.zeros((1, B), np.uint8)
mref = np.zeros(1, np.uint8)
b_sub = np.zeros((1, nw), np.uint64); b_sub[0, 0] = 1
m_mat = np.full((1, B), 9, np.uint8)
m_shots = np.full(B, 9, np.uint8)
r0 = np.zeros((N, B), np.uint8); r0[0, :] = 1
r1 = np.zeros((N, B), np.uint8)
fm = np.ones(B, np.uint8)
db = np.zeros(B, np.uint8)
out_exp = np.full((1, B), 77, np.int8)

xs_p = np.zeros((N, (B + 7) >> 3), np.uint8); xs_p[0, 0] = 3
xs_p_orig = xs_p.copy()
is_r = np.zeros(1, np.uint8)
n64, n32, nu64 = ctypes.cast(0, kl.i64_p), ctypes.cast(0, kl.i32_p), ctypes.cast(0, kl.u64_p)
nu8, p_n = ctypes.cast(0, kl.u8_p), np.array([0.5], np.float64)
bg_ct = np.random.default_rng(0).bit_generator.ctypes
bg_fn, bg_st = ctypes.cast(bg_ct.next_double, kl.void_p), kl.void_p(bg_ct.state_address)
v4_blk = ck.CosetCallBlock(N, B, nw, (B + 7) >> 3, 1, 7, 9, 0, p8(ox), p8(oz), p8(m_mat))

ax, az = ctypes.c_int(7), ctypes.c_int(9)
def reset_flags():
    ax.value, az.value = 7, 9
def assert_zero_flags(name):
    assert ax.value == 0 and az.value == 0, (name, ax.value, az.value)
def assert_sentinels(name):
    assert np.all(out_rng == np.uint64(0xAAAAAAAAAAAAAAAA)), name
    assert np.all(out_idx == -99) and np.all(out_d == -99), name
    assert np.all(sh == np.uint64(0xAAAAAAAAAAAAAAAA)), name
    assert np.all(sau == np.uint64(0xAAAAAAAAAAAAAAAA)), name
    assert np.all(sA == np.uint64(0xAAAAAAAAAAAAAAAA)), name
    assert np.all(sell == 0xAA) and np.all(sG == 0xAA) and np.all(sP == -1431655766), name
    assert np.all(ox == 0xAA) and np.all(oz == 0xAA), name
    assert np.all(m_mat == 9) and np.all(m_shots == 9) and np.all(out_exp == 77), name
    assert np.array_equal(xs, xs_orig) and np.array_equal(xs_p, xs_p_orig), name

# 1. All 16 raw int exports: exact negative rc, zeroed flags, untouched sentinels, 0 allocs.
rng2 = np.array([33, 44], np.uint64)
assert int(raw.engine_num_qubits(eng)) == -2
assert int(raw.engine_batch_size(eng)) == -2
assert int(raw.engine_n_words(eng)) == -2
assert int(raw.engine_set_rng_states(eng, p64(rng2))) == -2
assert int(raw.engine_get_rng_states(eng, B, p64(out_rng))) == -2
assert int(raw.engine_get_active_shot_indices(eng, p32(out_idx), p32(out_d))) == -2
assert int(raw.engine_export_shot_v3(
    eng, 0, nw, 1, p64(sh), p64(sau), p64(sA), p8(sell), p8(sG), p32(sP))) == -6
assert int(raw.engine_unitary_step_num_groups(eng, 0)) == -2

fi.fi_arm(10**9)
rc_reg = int(raw.engine_register_unitary_step(
    eng, 1, 1, 1, pi64(tg), p8(fk), p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8), N, nw, p64(x2z), p64(z2z)))
assert rc_reg == -9 and fi.fi_count() == 0
fi.fi_arm(10**9)
rc_ev = int(raw.engine_register_unitary_group_evict(eng, 0, 1, N, nw, p64(x2z), p64(z2z)))
assert rc_ev == -7 and fi.fi_count() == 0
fi.fi_disarm()

reset_flags()
assert int(raw.engine_inject_group_v3(
    eng, 0, 0, B, p8(cm_row), N, B, p8(xs), N, B, p8(zs),
    N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))) == -7
assert_zero_flags("engine_inject_group_v3")
nblk = ctypes.cast(0, ck.call_block_p)
for bp in (ctypes.byref(v4_blk), nblk):
    v4_blk.any_x, v4_blk.any_z = 7, 9
    assert int(raw.engine_batch_inject_unitary_v4(
        eng, 0, 1, B, p8(cm_mat), nu8, p8(xs_p), p8(xs_p), bp)) == -9
    assert bp is nblk or (v4_blk.any_x == 0 and v4_blk.any_z == 0)
reset_flags()
assert int(raw.engine_postselect_batch(
    eng, 0, 2, B, p8(fm), B, p8(db), N, B, p8(xs), N, B, p8(zs), N, nw,
    p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp),
    N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))) == -9
assert_zero_flags("engine_postselect_batch")
reset_flags()
assert int(raw.engine_project_pauli(
    eng, 2, N, B, p8(r0), N, B, p8(r1), -1, pi64(tg), N, B, p8(xs), N, B, p8(zs), N, nw,
    p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp),
    N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))) == -10
assert_zero_flags("engine_project_pauli")
for bp in (ctypes.byref(v4_blk), nblk):
    v4_blk.any_x, v4_blk.any_z = 7, 9
    assert int(raw.engine_batch_meas_step_v4(
        eng, 1, 0, p8(is_r), pi64(tg), p8(mref), n64, n32, p64(b_sub), nu64, nu64,
        0, n32, n64, 0, n32, n64, p64(x2z), p64(z2z), p8(xs_p), bp)) == -9
    assert bp is nblk or (v4_blk.any_x == 0 and v4_blk.any_z == 0)
    v4_blk.any_x, v4_blk.any_z = 7, 9
    assert int(raw.engine_cond_pauli_noise_v4(
        eng, 0, 1, pi64(tg), 1, p_n.ctypes.data_as(kl.f64_p), nu8, p8(cm_mat), bg_fn, bg_st, bp
    )) == -9
    assert bp is nblk or (v4_blk.any_x == 0 and v4_blk.any_z == 0)
assert_sentinels("raw_int_exports")

# 2. All 6 raw void exports: zeroed flags, untouched sentinels.
raw.engine_peek_pauli_batch(
    eng, 1, pi64(tg), 2, p8(xs), p8(zs),
    p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp), pi8(out_exp))
reset_flags()
raw.engine_batch_mixed_substep(
    eng, 0, 0, 0, 0, p64(b_sub), p64(b_sub), p64(b_sub), 0, 0, pi64(tg), 0, pi64(tg),
    p8(xs), p8(m_shots), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))
assert_zero_flags("engine_batch_mixed_substep")
reset_flags()
raw.engine_batch_evict(
    eng, p64(x2z), p64(z2z), p8(xs), p8(zs), p8(ox), p8(oz),
    ctypes.byref(ax), ctypes.byref(az))
assert_zero_flags("engine_batch_evict")
reset_flags()
raw.engine_batch_unmatched_noisy_reset_z(
    eng, p8(r0), p8(r1), p8(xs), p64(z2x), p64(z2z), p8(kp),
    p64(x2z), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))
assert_zero_flags("engine_batch_unmatched_noisy_reset_z")
reset_flags()
raw.engine_batch_inject_general_clifford(
    eng, 1, 1, 1, pi64(tg), p8(cm_mat), p8(xs), p8(zs), p8(fk),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
    p64(x2z), p64(z2z), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))
assert_zero_flags("engine_batch_inject_general_clifford")
# engine_clear writes no caller buffer and allocates nothing; with all readers guarding
# fail_kind != 0, removing only its early return is unobservable at the raw ABI.
raw.engine_clear(eng)
assert_sentinels("raw_void_exports")

# 3. All 21 guarded wrapper calls raise MemoryError on the failed engine.
wrapped_calls = [
    ("engine_num_qubits", lambda: lib.engine_num_qubits(eng)),
    ("engine_batch_size", lambda: lib.engine_batch_size(eng)),
    ("engine_n_words", lambda: lib.engine_n_words(eng)),
    ("engine_clear", lambda: lib.engine_clear(eng)),
    ("engine_set_rng_states", lambda: lib.engine_set_rng_states(eng, p64(rng2))),
    ("engine_get_rng_states", lambda: lib.engine_get_rng_states(eng, B, p64(out_rng))),
    ("engine_get_active_shot_indices",
     lambda: lib.engine_get_active_shot_indices(eng, p32(out_idx), p32(out_d))),
    ("engine_export_shot_v3",
     lambda: lib.engine_export_shot_v3(
         eng, 0, nw, 1, p64(sh), p64(sau), p64(sA), p8(sell), p8(sG), p32(sP))),
    ("engine_unitary_step_num_groups", lambda: lib.engine_unitary_step_num_groups(eng, 0)),
    ("engine_register_unitary_step",
     lambda: lib.engine_register_unitary_step(
         eng, 1, 1, 1, pi64(tg), p8(fk), p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
         p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8), N, nw, p64(x2z), p64(z2z))),
    ("engine_register_unitary_group_evict",
     lambda: lib.engine_register_unitary_group_evict(eng, 0, 1, N, nw, p64(x2z), p64(z2z))),
    ("engine_batch_meas_step_v4",
     lambda: lib.engine_batch_meas_step_v4(
         eng, 1, 0, p8(is_r), pi64(tg), p8(mref), n64, n32, p64(b_sub), nu64, nu64,
         0, n32, n64, 0, n32, n64, p64(x2z), p64(z2z), p8(xs_p), ctypes.byref(v4_blk))),
    ("engine_cond_pauli_noise_v4",
     lambda: lib.engine_cond_pauli_noise_v4(
         eng, 0, 1, pi64(tg), 1, p_n.ctypes.data_as(kl.f64_p), nu8, p8(cm_mat),
         bg_fn, bg_st, ctypes.byref(v4_blk))),
    ("engine_batch_mixed_substep",
     lambda: lib.engine_batch_mixed_substep(
         eng, 0, 0, 0, 0, p64(b_sub), p64(b_sub), p64(b_sub), 0, 0, pi64(tg), 0, pi64(tg),
         p8(xs), p8(m_shots), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_evict",
     lambda: lib.engine_batch_evict(
         eng, p64(x2z), p64(z2z), p8(xs), p8(zs), p8(ox), p8(oz),
         ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_unmatched_noisy_reset_z",
     lambda: lib.engine_batch_unmatched_noisy_reset_z(
         eng, p8(r0), p8(r1), p8(xs), p64(z2x), p64(z2z), p8(kp),
         p64(x2z), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_peek_pauli_batch",
     lambda: lib.engine_peek_pauli_batch(
         eng, 1, pi64(tg), 2, p8(xs), p8(zs),
         p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp), pi8(out_exp))),
    ("engine_postselect_batch",
     lambda: lib.engine_postselect_batch(
         eng, 0, 2, B, p8(fm), B, p8(db), N, B, p8(xs), N, B, p8(zs), N, nw,
         p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp),
         N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_project_pauli",
     lambda: lib.engine_project_pauli(
         eng, 2, N, B, p8(r0), N, B, p8(r1), -1, pi64(tg), N, B, p8(xs), N, B, p8(zs), N, nw,
         p64(x2x), p64(x2z), p64(z2x), p64(z2z), p8(kp), p8(kp),
         N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_inject_group_v3",
     lambda: lib.engine_inject_group_v3(
         eng, 0, 0, B, p8(cm_row), N, B, p8(xs), N, B, p8(zs),
         N, B, p8(ox), N, B, p8(oz), ctypes.byref(ax), ctypes.byref(az))),
    ("engine_batch_inject_unitary_v4",
     lambda: lib.engine_batch_inject_unitary_v4(
         eng, 0, 1, B, p8(cm_mat), nu8, p8(xs_p), p8(xs_p), ctypes.byref(v4_blk))),
    ("engine_batch_inject_general_clifford",
     lambda: lib.engine_batch_inject_general_clifford(
         eng, 1, 1, 1, pi64(tg), p8(cm_mat), p8(xs), p8(zs), p8(fk),
         p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
         p8(f_u8), p8(f_u8), p64(f_a), p64(f_b), p8(f_u8),
         p64(x2z), p64(z2z), p8(ox), p8(oz), ctypes.byref(ax), ctypes.byref(az))),
]
assert {n for n, _ in wrapped_calls} == (ck._VOID_FNS | ck._INT_FNS)
assert len(wrapped_calls) == 22
for name, fn in wrapped_calls:
    try:
        fn()
        raise AssertionError(f"{name} did not raise MemoryError on failed engine")
    except MemoryError:
        pass
assert_sentinels("wrapped_exports")
lib.engine_destroy(eng)
print("STICKY_22_OK")
"""


@pytest.mark.skipif(
    _under_tsan() or shutil.which("c++") is None,
    reason="needs a C++ compiler and no TSan runtime (TSan owns operator new)",
)
def test_failed_engine_all_25_exports_refuse_work_raw_and_wrapped(tmp_path: Path) -> None:
    """All 22 guarded exports refuse valid work on a failed engine (raw + wrapped)."""
    cpp, so = tmp_path / "fi_sticky.cpp", tmp_path / "libfi_sticky.so"
    cpp.write_text(_FI_STICKY_CPP, encoding="utf-8")
    subprocess.run(["c++", "-shared", "-fPIC", "-O1", str(cpp), "-o", str(so)], check=True)
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    ld_preload = f"{so} {existing_preload}" if existing_preload else str(so)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(_SRC),
        "LD_PRELOAD": ld_preload,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if existing_preload:
        asan_opts = env.get("ASAN_OPTIONS", "")
        if "verify_asan_link_order=0" not in asan_opts:
            env["ASAN_OPTIONS"] = (
                f"{asan_opts}:verify_asan_link_order=0"
                if asan_opts
                else "verify_asan_link_order=0"
            )
    proc = subprocess.run(
        [sys.executable, "-c", _FI_STICKY_CHILD, str(_SRC), str(so)],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "STICKY_22_OK"


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
