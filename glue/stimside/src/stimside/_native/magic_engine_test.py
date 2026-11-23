"""ABI, lifetime and numerics tests for the `mbe_*` MagicBatchEngine entry points.

The engine C code lives in `magic_engine.hpp` (compiled into `libmagic_kernels.so`) and is bound
in `magic_engine.py` with its own `_SPECS`; these tests pin that table to the C prototypes, check
that every address argument is a live, 64-bit-wide integer address, that engines do not leak,
and that the engine's complex multiply / `norm_sq` equal numpy's bit for bit.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stimside._native import magic_engine as me
from stimside._native import magic_kernels as mk

_NATIVE = Path(me.__file__).resolve().parent
_SRC = _NATIVE.parents[1]  # the directory that contains the `stimside` package
_PKG_ROOT = _SRC.parent  # the project root (MANIFEST.in, setup.py)
_FOUR_GB = 2**32
_SCALARS = {
    "size_t": ctypes.c_size_t, "uint8_t": ctypes.c_uint8, "int": ctypes.c_int,
    "double": ctypes.c_double, "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64,
}


def _hpp_prototypes() -> dict[str, tuple[str, list[str]]]:
    """`name -> (return type, [C parameter types])` of the `extern "C"` block of the header."""
    src = (_NATIVE / "magic_engine.hpp").read_text()
    body = src[src.index('extern "C" {'):]
    out: dict[str, tuple[str, list[str]]] = {}
    for m in re.finditer(r"^([\w\*]+)\s+(\w+)\(([^)]*)\)\s*\{", body, re.MULTILINE):
        params = []
        for p in m.group(3).split(","):
            toks = p.replace("const", " ").replace("*", " * ").split()
            if toks and toks != ["void"]:
                params.append("".join(toks[:-1]))
        out[m.group(2)] = (m.group(1), params)
    return out


def test_specs_match_the_c_prototypes_and_every_pointer_is_bound_c_void_p() -> None:
    protos = _hpp_prototypes()
    assert {n for n, _, _ in me._SPECS} == set(protos) and len(protos) == 28
    assert all(n.startswith("mbe_") for n in protos)
    lib = me.get_magic_engine_lib()
    abis = (lib.mbe_abi_v1(), lib.mbe_abi_v2(), lib.mbe_abi_v3(), lib.mbe_abi_v4(),
            lib.mbe_abi_v5(), lib.mbe_abi_v6())
    assert abis == (1, 2, 3, 4, 5, 6)
    n_ptr = 0
    for name, (ret, params) in protos.items():
        fn = getattr(lib, name)
        assert len(fn.argtypes) == len(params), name
        for i, (ctype, bound) in enumerate(zip(params, fn.argtypes)):
            if ctype.endswith("*"):
                n_ptr += 1
                assert bound is ctypes.c_void_p, f"{name}[{i}] {ctype} bound as {bound}"
            else:
                assert bound is _SCALARS[ctype], f"{name}[{i}] {ctype} bound as {bound}"
        want = None if ret == "void" else ctypes.c_void_p if ret == "void*" else _SCALARS[ret]
        assert fn.restype is want, f"{name} restype"
    assert n_ptr >= 34  # the table really has pointer parameters to check


def test_engine_symbols_are_kept_out_of_the_magic_kernel_table() -> None:
    assert not any(n.startswith("mbe_") for n, _, _ in mk._SPECS)
    assert not any(n.startswith("magic_") for n, _, _ in me._SPECS)
    cpp = (_NATIVE / "magic_kernels.cpp").read_text()
    assert cpp.count('#include "magic_engine.hpp"') == 1  # one translation unit, one include
    assert cpp.rstrip().endswith('#include "magic_engine.hpp"')


def test_packaging_lists_the_header_by_explicit_name_only() -> None:
    manifest = (_PKG_ROOT / "MANIFEST.in").read_text()
    assert "include src/stimside/_native/magic_engine.hpp" in manifest.splitlines()
    assert "*.h" not in manifest and "*.hpp" not in manifest


class _SpyLib:
    """Wraps the engine's CDLL: records every integer address handed to an `mbe_*` pointer
    parameter and lets a hook look at the call just before it runs."""

    def __init__(self, lib: Any, before: Any = None) -> None:
        self._lib, self._before = lib, before
        self.min_addr: dict[str, int] = {}
        self.seen: set[str] = set()
        self._ptr_pos = {n: [i for i, t in enumerate(a) if t is ctypes.c_void_p]
                         for n, a, _ in me._SPECS}

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._lib, name)
        if not name.startswith("mbe_"):
            return fn

        def call(*args: Any) -> Any:
            self.seen.add(name)
            ptrs = [args[i] for i in self._ptr_pos[name] if args[i]]
            assert all(isinstance(p, int) for p in ptrs), f"{name}: non-integer address"
            if ptrs:
                self.min_addr[name] = min(self.min_addr.get(name, 2**64), *ptrs)
            if self._before is not None:
                self._before(name, args)
            return fn(*args)

        return call


def _hermitian_words(rng: np.random.Generator, n: int, density: float) -> tuple[Any, Any, int]:
    from stimside._magic import phase_qfe as pq

    a = pq.pack_bit_vector((rng.random(n) < density).astype(np.uint8), n)
    b = pq.pack_bit_vector((rng.random(n) < density).astype(np.uint8), n)
    return a, b, int(pq.word_dot_parity(a, b)) & 3  # kappa parity must equal <a, b>


def _t_state(rng: np.random.Generator, n: int, n_t: int) -> Any:
    from stimside._magic import phase_qfe as pq

    e = pq.ShotExpansion.zero(n)
    for _ in range(n_t):
        a, b, kap = _hermitian_words(rng, n, 0.3)
        e = pq.apply_pauli_t(e, a, b, kap, dag=False)
    return e


def _a_block(rng: np.random.Generator, n: int) -> Any:
    from stimside._magic.channels import MagicPhaseTerm, MagicTermRecord
    from stimside._magic.gates import CompiledPauliAxis
    from stimside._magic.plan import compile_open_block

    ax = CompiledPauliAxis(*_hermitian_words(rng, n, 0.4))
    rec = MagicTermRecord(MagicPhaseTerm((ax,), "T", 1))
    return compile_open_block([rec], n_qubits=n)[0].block  # type: ignore[attr-defined]


def _one_of_each(lib_hook: Any = None) -> tuple[bytes, _SpyLib]:
    """Drive every `mbe_*` entry point once; returns a digest of the observable results."""
    from stimside._magic.channels import CompiledCliffordFactorList
    from stimside._magic.gates import CompiledPauliAxis
    from stimside._magic.live_store import (
        LiveExpansions,
        block_table,
        expansion_to_arrays,
        factor_table,
    )

    n = 70
    store = LiveExpansions(4, n, 1e-10)
    eng = store.engine
    spy = _SpyLib(eng._lib, lib_hook)
    eng._lib = spy  # type: ignore[assignment]
    rng = np.random.default_rng(4)
    e = _t_state(rng, n, 3)
    e.log_mass = -0.5
    eng.import_shot(1, expansion_to_arrays(e))
    a, b, kap = _hermitian_words(rng, n, 0.5)
    eng.apply_pauli_mask(a, b, kap, np.array([True, True, False, True]))
    from stimside.util.reference_chp import ReferenceCHPTableau
    rt = ReferenceCHPTableau(n)
    xm, zm = np.zeros((n, 4), bool), np.zeros((n, 4), bool)
    xm[0, 1] = zm[1, 1] = True
    eng.apply_pauli_noise_batch(rt.x2x, rt.x2z, rt.x_kap, rt.z2x, rt.z2z, rt.z_kap, xm, zm)
    axis = CompiledPauliAxis(a, b, kap)
    fac = CompiledCliffordFactorList(3, axis, (axis,))
    which = np.array([-1, 0, 0, -1], np.int32)
    eng.inject_cliffords(factor_table([fac], eng.n_words), which, True)
    out = eng.export_shot(1)
    live = eng.live_shots()
    block = block_table(_a_block(rng, n), n)
    assert block is not None
    res = eng.apply_block(block, True, 7)
    rep = eng.partition(True)
    out_b = eng.export_shot(1)
    # S5a: one measurement (a rotating observable, a reset kick) split and finished
    ax2 = CompiledPauliAxis(*_hermitian_words(rng, n, 0.5))
    sp = eng.measure_split((ax2.a_words, ax2.b_words, ax2.kappa), False, (a, b, kap, False))
    chosen = np.where(sp.z01[sp.cls == 2, 0] > 0, sp.z01[sp.cls == 2, 0], sp.z01[sp.cls == 2, 1])
    fin = eng.measure_finish(np.zeros(sp.n_active, bool), chosen) if sp.n_active else None
    bg = np.random.PCG64(12345)
    fn_a = ctypes.cast(bg.ctypes.next_double, ctypes.c_void_p).value
    assert fn_a is not None
    eng.measure_step((ax2.a_words, ax2.b_words, ax2.kappa), False, (a, b, kap, False), fn_a,
                     int(bg.ctypes.state_address))
    out_m = eng.export_shot(1)
    # S5b: a peek, a postselection and one unmatched-reset apply
    ob = (a, b, kap)
    p1, _pf = eng.peek(ob, np.arange(4, dtype=np.int32))
    ps = eng.postselect(ob, np.array([0, 1, 3], np.int32), np.array([0, 1, 0], np.uint8))
    um = eng.unmatched_apply(
        ob, (a, b, kap), np.array([1, 2], np.int32), [1, 0], [1, 1], [1, 0]
    )
    out_s5b = eng.export_shot(1)
    eng.reset()
    out0 = eng.export_shot(0)
    w = (rng.standard_normal(9) + 1j * rng.standard_normal(9)).astype(np.complex128)
    spy.mbe_norm_sq(9, mk.w_addr(w))
    spy.mbe_cmul_scalar(9, mk.w_addr(w), 0.5, 0.5)
    spy.mbe_cmul_form(9, mk.w_addr(w), 0.5, -0.25, 1)
    digest = b"".join(x.tobytes() for x in (*out[5:], *out_b[5:], *out_m[5:], *out0[5:], w, rep,
                                              p1, *out_s5b[5:]))
    digest += repr((out[:5], live, res.groups, res.live, res.failure)).encode()
    digest += repr((sp.cls.tolist(), sp.z.tolist(), sp.z01.tolist(), sp.n_active, fin)).encode()
    digest += repr((_pf, ps.done.tolist(), ps.z01.tolist(), ps.failure, um)).encode()
    n_live = spy.mbe_live_engines()
    store.close()
    assert spy.mbe_live_engines() == n_live - 1
    return digest, spy


def _child_main() -> None:
    if int(np.empty(1, np.uint8).ctypes.data) <= _FOUR_GB:
        print(json.dumps({"skip": "numpy arrays are not placed above 4 GB by this allocator"}))
        return
    digest, spy = _one_of_each()
    print(json.dumps({"digest": digest.hex(), "min_addr": spy.min_addr, "seen": sorted(spy.seen)}))


def test_every_engine_pointer_parameter_accepts_addresses_above_4_gb() -> None:
    # With the mmap threshold at 1 byte glibc serves every numpy buffer from mmap, far above
    # 4 GB, so a pointer parameter bound as c_int (or an int cast) would truncate the address.
    env = dict(
        os.environ, MALLOC_MMAP_THRESHOLD_="1", PYTHONPATH=str(_SRC),
        PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
    )
    code = f"import {__name__} as t; t._child_main()"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=240, check=False)
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-3000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])
    pointer_fns = {n for n, a, _ in me._SPECS if any(t is ctypes.c_void_p for t in a)}
    assert set(result["seen"]) >= pointer_fns - {"mbe_create"}
    low = {k: hex(v) for k, v in result["min_addr"].items() if v <= _FOUR_GB}
    assert not low, f"arguments below 4 GB: {low}"
    assert bytes.fromhex(result["digest"]) == _one_of_each()[0]  # same results in-process


def test_every_address_argument_is_backed_by_a_live_array_at_call_time() -> None:
    owners: dict[int, weakref.ref[Any]] = {}
    real_addr, real_w_addr = me.addr, me.w_addr

    def tracked(fn: Any) -> Any:
        def wrapper(arr: Any) -> int:
            p = fn(arr)
            if p:
                owners[p] = weakref.ref(arr)
            return p

        return wrapper

    violations: list[str] = []
    checked: list[int] = []

    def before(name: str, args: tuple[Any, ...]) -> None:
        gc.collect()
        churn = [np.empty(k, np.uint8) for k in (8, 100, 4096, 1 << 16) for _ in range(4)]
        del churn  # reuse of a dead array's memory would be visible to the kernel
        for i, p in enumerate(args):
            ref = owners.get(p) if isinstance(p, int) and p else None
            if ref is not None:
                checked.append(i)
                if ref() is None:
                    violations.append(f"{name} arg {i}: address {p:#x} owner is dead")

    me.addr, me.w_addr = tracked(real_addr), tracked(real_w_addr)
    try:
        _one_of_each(before)
    finally:
        me.addr, me.w_addr = real_addr, real_w_addr
    assert not violations, violations
    assert len(checked) >= 12  # the check really looked at many address arguments


def test_create_destroy_loop_returns_the_live_engine_count_to_its_start() -> None:
    lib = me.get_magic_engine_lib()
    gc.collect()
    start = lib.mbe_live_engines()
    engines = [me.MagicBatchEngine(8, 70, 1e-10) for _ in range(50)]
    assert lib.mbe_live_engines() == start + 50
    for e in engines[:25]:
        e.close()
        e.close()  # idempotent
    assert lib.mbe_live_engines() == start + 25
    del engines
    gc.collect()  # __del__ frees the rest
    assert lib.mbe_live_engines() == start


def test_a_closed_engine_refuses_every_command_and_a_failed_init_is_harmless() -> None:
    e = me.MagicBatchEngine(2, 5, 1e-10)
    e.close()
    for call in (e.reset, lambda: e.export_shot(0),
                 lambda: e.apply_pauli_mask(np.zeros(1, np.uint64), np.zeros(1, np.uint64), 0,
                                            np.zeros(2, bool))):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    start = me.get_magic_engine_lib().mbe_live_engines()
    for bad in ((-1, 5), (4, 0)):
        with pytest.raises(ValueError):
            me.MagicBatchEngine(*bad, 1e-10)
    gc.collect()  # __del__ of the half-built objects must not raise or free anything
    assert me.get_magic_engine_lib().mbe_live_engines() == start


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def _asan_loaded() -> bool:
    with open("/proc/self/maps") as f:
        return "libasan" in f.read()


@pytest.mark.skipif(_asan_loaded(), reason="ASan quarantines freed memory: RSS is no leak signal")
def test_create_import_destroy_cycles_do_not_leak_memory() -> None:
    from stimside._magic.live_store import expansion_to_arrays

    n = 130
    arrays = expansion_to_arrays(_t_state(np.random.default_rng(0), n, 6))  # chi 64, ~2.5 KB

    def cycle() -> None:
        eng = me.MagicBatchEngine(64, n, 1e-10)
        for b in range(64):
            eng.import_shot(b, arrays)
        eng.reset()
        eng.close()

    for _ in range(20):
        cycle()
    base = _rss_mb()
    for _ in range(300):  # a leak of one engine (~160 KB) per cycle would add ~48 MB
        cycle()
    assert _rss_mb() - base < 12.0


def test_import_validates_array_shapes_before_the_c_code_trusts_them() -> None:
    from stimside._magic import phase_qfe as pq
    from stimside._magic.live_store import expansion_to_arrays

    eng = me.MagicBatchEngine(2, 70, 1e-10)
    good = expansion_to_arrays(pq.ShotExpansion.zero(70))
    bads: tuple[tuple[str, Any], ...] = (
        ("weights", np.ones(3, np.complex128)),
        ("h_words", np.zeros((1, 1), np.uint64)),
        ("a_union", np.zeros(5, np.uint64)),
        ("A_words", np.zeros((2, 2), np.uint64)),
    )
    for field, bad in bads:
        replacement: dict[str, Any] = {field: bad}
        with pytest.raises(ValueError, match=field):
            eng.import_shot(0, good._replace(**replacement))
    with pytest.raises(ValueError):
        eng.import_shot(0, good._replace(d=71))
    with pytest.raises(IndexError):
        eng.import_shot(2, good)
    with pytest.raises(ValueError, match="mask"):
        eng.apply_pauli_mask(np.zeros(2, np.uint64), np.zeros(2, np.uint64), 0, np.zeros(3, bool))
    eng.import_shot(1, good)  # the engine still works after the rejected calls
    eng.close()


def test_a_missing_symbol_is_a_clear_error_and_loading_never_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_compile(*a: Any, **k: Any) -> None:
        raise AssertionError("unexpected compile")

    monkeypatch.setattr(subprocess, "run", no_compile)
    monkeypatch.setattr(me, "_LIB_INSTANCE", None)
    lib = me.get_magic_engine_lib()  # an up-to-date library is only loaded, never rebuilt
    assert lib.mbe_abi_v1() == 1 and me.get_magic_engine_lib() is lib
    eng = me.MagicBatchEngine(2, 5, 1e-10)
    eng.close()


_STALE_CODE = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "from stimside._native import magic_engine as me\n"
    "lib = me.get_magic_engine_lib()\n"
    "print('OK', lib.mbe_abi_v1())\n"
)


def test_a_stale_library_without_the_engine_symbols_is_rebuilt_on_first_use(
    tmp_path: Path,
) -> None:
    """An installed library built before the engine existed exports `magic_*` but no `mbe_*`."""
    site = tmp_path / "site-packages"
    keep = shutil.ignore_patterns("__pycache__", "*.so", "*_test.py")
    shutil.copytree(_NATIVE.parent, site / "stimside", ignore=keep)
    native = site / "stimside" / "_native"
    for other in native.glob("*.cpp"):
        if other.name != "magic_kernels.cpp":
            other.unlink()  # only the library under test is rebuilt
    stub = tmp_path / "stale.cpp"
    names = [n for n, _, _ in mk._SPECS]
    body = "".join(f"int {n}() {{ return 0; }}\n" for n in names)
    stub.write_text('extern "C" {\n' + body + "}\n")
    so = native / "libmagic_kernels.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(stub), "-o", str(so)], check=True)
    assert not hasattr(ctypes.CDLL(str(so)), "mbe_abi_v1")
    env = {**os.environ, "STIMSIDE_PORTABLE_BUILD": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", _STALE_CODE, str(site)], capture_output=True,
                         text=True, env=env, timeout=600, check=False)
    assert out.returncode == 0 and out.stdout.split() == ["OK", "1"], out.stderr


# ---------------------------------------------------------------------------------------------
# numpy recipes: complex multiply (FMA form) and norm_sq (pairwise sum), bit for bit
# ---------------------------------------------------------------------------------------------

_MULTIPLIERS = (
    1 + 0j, 1j, -1 + 0j, -1j, 0.5 + 0.5j, -0.5 + 0.5j, 0.7071067811865476 + 0.7071067811865476j,
    -0.7071067811865476 - 0.7071067811865476j, 0.25 + 0.0j, 1.5 - 2.25j, 0.0 + 0.0j,
)


def _with_zeros(rng: np.random.Generator, n: int) -> np.ndarray:
    w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    if n > 2:
        w[::5] = complex(0.0, -0.0)
        w[1::7] = complex(-0.0, 0.0)
        w[2::11] = complex(-0.0, -0.0)
    return w.astype(np.complex128)


def test_complex_multiply_equals_numpy_including_signed_zeros() -> None:
    lib = me.get_magic_engine_lib()
    rng = np.random.default_rng(1)
    unfused_mismatches = checked = 0
    for n in list(range(1, 40)) + [64, 127, 128, 129, 1000]:
        for c in _MULTIPLIERS:
            w = _with_zeros(rng, n)
            ref, mine = w * c, w.copy()
            lib.mbe_cmul_scalar(n, mk.w_addr(mine), c.real, c.imag)
            assert np.array_equal(mine.view(np.uint64), ref.view(np.uint64)), (n, c)
            plain = w.copy()
            plain.real = w.real * c.real - w.imag * c.imag
            plain.imag = w.real * c.imag + w.imag * c.real
            unfused_mismatches += not np.array_equal(plain.view(np.uint64), ref.view(np.uint64))
            checked += 1
    assert checked >= 480
    assert unfused_mismatches > 50  # the FMA form is needed: an unfused product really differs


def test_norm_sq_equals_numpy_pairwise_sum_for_every_length() -> None:
    lib = me.get_magic_engine_lib()
    rng = np.random.default_rng(2)
    sequential_mismatches = 0
    lengths = list(range(300)) + [511, 512, 513, 1000, 4095, 4096, 4097, 5000]
    for n in lengths:
        w = _with_zeros(rng, n) * float(rng.choice([1e-3, 1.0, 1e3]))
        want = float(np.sum(w.real * w.real + w.imag * w.imag)) if n else 0.0
        got = lib.mbe_norm_sq(n, mk.w_addr(w) if n else 0)
        assert np.float64(got).view(np.uint64) == np.float64(want).view(np.uint64), n
        seq = 0.0
        for i in range(n):
            seq += float(w.real[i] * w.real[i] + w.imag[i] * w.imag[i])
        sequential_mismatches += seq != want
    assert len(lengths) * 4 >= 1000
    assert sequential_mismatches > 100  # a sequential loop would not match numpy
