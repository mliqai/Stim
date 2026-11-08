# Copyright 2026 Google LLC
"""Pins for the integer-address kernel ABI (WP-24 B2 fix): pointer width, typed pointers, lifetimes.

`magic_kernels_addr_test.py` covers `addr`/`w_addr` themselves. These tests pin what a caller can
get wrong with raw addresses: a pointer parameter bound narrower than 64 bits (crashes only for
arrays above 4 GB), a typed ctypes pointer left on a call path, an address taken from a temporary
(use-after-free in C++), a wrong pointee dtype, and a packer that stops owning its buffers.
"""

from __future__ import annotations

import ast
import ctypes
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stimside._magic import phase_qfe as pq
from stimside._native import _kernel_loader as kl
from stimside._native import magic_kernels as mk

_PKG_ROOT = Path(mk.__file__).resolve().parents[1]
_SCALARS = {
    "size_t": ctypes.c_size_t, "uint8_t": ctypes.c_uint8, "int": ctypes.c_int,
    "double": ctypes.c_double, "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64,
}
_POINTEE = {  # C pointee type -> numpy dtypes an address argument may point at
    "double": ("float64", "complex128"), "uint64_t": ("uint64",), "uint8_t": ("uint8",),
    "int32_t": ("int32",), "int64_t": ("int64",), "size_t": (),
}
_BYREF = type(ctypes.byref(ctypes.c_int(0)))
_FOUR_GB = 2**32
Protos = dict[str, tuple[str, list[tuple[str, str]]]]


def _c_prototypes() -> Protos:
    """`name -> (return type, [(C type, parameter name)])` of every `extern "C"` kernel."""
    src = Path(mk.__file__).with_suffix(".cpp").read_text()
    body = src[src.index('extern "C" {'):]
    out: Protos = {}
    for m in re.finditer(r"^(\w+)\s+(magic_\w+)\(([^)]*)\)\s*\{", body, re.MULTILINE):
        params = [p.replace("const", " ").split() for p in m.group(3).split(",") if p.strip()]
        out[m.group(2)] = (m.group(1), [("".join(p[:-1]), p[-1]) for p in params])
    return out


_KERNELS = {n for n in _c_prototypes() if not n.startswith("magic_abi")}


def test_every_kernel_pointer_parameter_is_c_void_p_and_no_other_parameter_is() -> None:
    protos = _c_prototypes()
    assert {n for n, _, _ in mk._SPECS} == set(protos)  # the table covers every exported kernel
    lib = mk.get_magic_kernels_lib()
    n_ptr = 0
    for name, (ret, params) in protos.items():
        fn = getattr(lib, name)
        assert len(fn.argtypes) == len(params), name
        for i, ((ctype, pname), bound) in enumerate(zip(params, fn.argtypes)):
            where = f"{name}[{i}] {ctype} {pname}"
            if ctype.endswith("*"):
                n_ptr += 1
                assert bound is ctypes.c_void_p, f"{where} bound as {bound}"
            else:
                assert bound is _SCALARS[ctype], f"{where} bound as {bound}"
        assert fn.restype is (None if ret == "void" else _SCALARS[ret]), f"{name} restype"
    assert n_ptr > 150  # the prototype parser did not silently lose the pointers


def _ax(n: int, i: int) -> list[int]:
    return [int(j == i) for j in range(n)]


def _t_state(n: int, k: int = 3, offset: int = 0) -> pq.ShotExpansion:
    e = pq.ShotExpansion.zero(n)
    for i in range(k):
        e = pq.apply_pauli_t(e, _ax(n, (i + offset) % n), [0] * n, 0)
    return e


def _choi_operands(n: int) -> tuple[pq.ShotExpansion, pq.ShotExpansion]:
    nw = pq.n_words_for_qubits(n + 2)
    left = pq.ShotExpansion.from_component(pq.PhaseQFEComponent(
        n_qubits=n + 2, d=1, h_words=np.zeros(nw, np.uint64),
        A_words=pq.unit_bit_words(n + 1, nw).reshape(1, nw), ell=np.zeros(1, np.uint8),
        Gamma=np.zeros((1, 1), np.uint8), pivots=np.array([n + 1], np.int32),
    ))
    return left, _t_state(n + 1, 3)


def _digest(h: Any, shots: Any) -> None:
    for e in shots:
        for f in ("weights", "h_words", "ell", "A_words", "Gamma", "pivots", "ell_odd"):
            h.update(np.ascontiguousarray(getattr(e, f)).tobytes())


def _exercise_all_kernels(n: int = 5) -> str:
    """Drive every magic kernel entry point (both batch-measure paths too); return a digest."""
    h, z = hashlib.sha256(), [0] * n
    e = pq.apply_quarter_turn(_t_state(n), _ax(n, 0), z, 0)  # apply_quarter_turn_tol; d = 1
    e = pq.apply_pauli(e, _ax(n, 1), _ax(n, 0), 1)  # apply_pauli_batch; b overlaps A: ell shifts
    e = pq.apply_pauli_t(e, _ax(n, 3), z, 0)  # branch_pauli_t
    e = pq.apply_pauli_t(e, _ax(n, 0), _ax(n, 0), 1)  # Y on the pivot: canonicalizes
    e = pq.apply_quarter_turn(e, _ax(n, 2), z, 0)  # d = 2: every geometry array is non-empty
    e = pq.apply_ccz(e, [(z, _ax(n, i), 0) for i in (2, 3, 4)])  # branch_ccz
    e = pq.apply_ccz(e, [(_ax(n, i), z, 0) for i in (0, 1, 2)])  # X axes hit the pivot column
    loose = pq.apply_pauli_t(e, _ax(n, 0), z, 0, merge=False)  # not canonical: merging has work
    e = pq.merge_frames(loose)  # canonicalize_and_merge
    p, _ = pq.project_diagonal(e, _ax(n, 4))  # project_diagonal
    left, right = _choi_operands(n)
    right = pq.apply_quarter_turn(right, _ax(n + 1, 0), [0] * (n + 1), 0)
    c = pq.contract_choi_expansions(left, right, n_op=n + 1, q_act=[n], right_is_choi=False)
    _digest(h, [e, p, c])
    for obs in ((z, _ax(n, 3), 0), (_ax(n, 0), z, 0)):  # diagonal / quarter-turn rotation
        out, z0, z1, posts = pq.measure_expansions_batch(
            [e, e.copy()], obs, rng=np.random.default_rng(3))  # measure_batch
        h.update(out.tobytes() + z0.tobytes() + z1.tobytes())
        _digest(h, posts)
    src = pq.apply_quarter_turn(_t_state(n), _ax(n, 2), z, 0)
    out_chi, _, bufs, out_ptrs = pq._alloc_cpp_geom_bufs(src.chi, src.n_words, src.d + 1)
    a, b = pq.coerce_pauli_words(_ax(n, 0), n), pq.coerce_pauli_words(z, n)
    rc = mk.get_magic_kernels_lib().magic_apply_quarter_turn(  # the legacy 23-argument symbol
        src.chi, src.d, n, src.n_words, *pq._cpp_shot_in_ptrs(src),
        pq._addr(a), pq._addr(b), ctypes.c_uint8(0), *out_ptrs,
    )
    assert rc == 0 and out_chi.value > 0
    h.update(b"".join(x[: out_chi.value].tobytes() for x in bufs[:2]))
    return h.hexdigest()


class _Addr(int):
    """An address from `addr`/`w_addr` that remembers a weak reference to its memory owner."""

    ref: Any
    desc: str
    dtype: str
    is_w: bool


class _KernelSpy:
    """Wraps `pq._addr`/`_w_addr` and every kernel; checks each pointer argument at call time."""

    def __init__(self, protos: Protos) -> None:
        self.protos = protos
        self.before_call: Any = None
        self.seen: dict[str, int] = {}
        self.min_addr: dict[str, int] = {}
        self.covered: set[tuple[str, int]] = set()  # (kernel, position) given a non-NULL pointer
        self.last_args: list[_Addr] = []

    def tag(self, arr: Any, a: int, is_w: bool) -> _Addr:
        assert a == (arr.ctypes.data if arr.size else 0), "address is not the array data pointer"
        assert arr.flags.c_contiguous and arr.flags.aligned, f"non-contiguous {arr.shape}"
        root = arr
        while isinstance(root.base, np.ndarray):
            root = root.base
        t = _Addr(a)
        t.ref, t.dtype, t.is_w = weakref.ref(root), str(arr.dtype), is_w
        t.desc = f"shape={arr.shape} dtype={arr.dtype}"
        return t

    def check(self, name: str, args: tuple[Any, ...]) -> None:
        params = self.protos[name][1]
        assert len(args) == len(params), name
        self.last_args = []
        for i, (arg, (ctype, pname)) in enumerate(zip(args, params)):
            where = f"{name}[{i}] {ctype} {pname}"
            assert not isinstance(arg, ctypes._Pointer), f"{where}: typed ctypes pointer"
            if ctype.endswith("*") and isinstance(arg, _BYREF):
                self.covered.add((name, i))
            if not ctype.endswith("*") or arg is None or isinstance(arg, _BYREF):
                continue  # scalar, NULL outcome, or a byref out-parameter
            if arg != 0:
                self.covered.add((name, i))
            assert isinstance(arg, _Addr), f"{where}: not an address from addr()/w_addr()"
            self.last_args.append(arg)
            if arg == 0:
                continue
            assert arg.ref() is not None, f"{where}: address of a freed array ({arg.desc})"
            assert arg.dtype in _POINTEE[ctype[:-1]], f"{where}: points at {arg.desc}"
            assert arg.is_w == (arg.dtype == "complex128"), f"{where}: unchecked complex array"
            self.min_addr[name] = min(self.min_addr.get(name, int(arg)), int(arg))
        self.seen[name] = self.seen.get(name, 0) + 1


def _install_spy(monkeypatch: pytest.MonkeyPatch) -> _KernelSpy:
    spy = _KernelSpy(_c_prototypes())
    real_addr, real_w_addr = pq._addr, pq._w_addr
    monkeypatch.setattr(pq, "_addr", lambda a: spy.tag(a, real_addr(a), False))
    monkeypatch.setattr(pq, "_w_addr", lambda w: spy.tag(w, real_w_addr(w), True))
    lib = mk.get_magic_kernels_lib()

    def wrap(name: str, real: Any) -> Any:
        def wrapped(*args: Any) -> Any:
            spy.check(name, args)
            if spy.before_call is not None:
                spy.before_call(name, args)
            return real(*args)

        return wrapped

    for name in _KERNELS:
        monkeypatch.setattr(lib, name, wrap(name, getattr(lib, name)))
    return spy


@pytest.mark.parametrize("n", [5, 70])
def test_every_kernel_call_gets_live_correctly_typed_integer_addresses(
    monkeypatch: pytest.MonkeyPatch, n: int,
) -> None:
    spy = _install_spy(monkeypatch)
    _exercise_all_kernels(n)
    assert set(spy.seen) == _KERNELS  # no kernel went unchecked
    # every pointer parameter of every kernel got a real (non-NULL) address or a byref at least
    # once: a mistyped parameter cannot hide behind an always-NULL argument
    every = {
        (name, i) for name in _KERNELS for i, (t, _) in enumerate(spy.protos[name][1])
        if t.endswith("*")
    }
    assert every - spy.covered == set()


def test_no_ctypes_cast_or_typed_pointer_is_built_by_any_kernel_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def count(label: str, real: Any) -> Any:
        def counted(*args: Any) -> Any:
            built.append(label)
            return real(*args)
        return counted

    monkeypatch.setattr(ctypes, "cast", count("cast", ctypes.cast))
    for mod, name in ((kl, "c_ptr"), (kl, "c_w_ptr"), (pq, "_c_ptr")):
        monkeypatch.setattr(mod, name, count(name, getattr(mod, name)))
    _exercise_all_kernels()  # includes apply_ccz and contract_choi_expansions
    assert built == []


@pytest.mark.parametrize("n", [5, 64, 70, 130])
def test_typed_pointer_and_integer_address_paths_agree_for_ccz_and_contract_choi(
    monkeypatch: pytest.MonkeyPatch, n: int,
) -> None:
    def run() -> list[bytes]:
        e = _t_state(n, 4, offset=n - 2)
        axes = [([0] * n, _ax(n, (n - 1 - i) % n), 0) for i in range(3)]
        left, right = _choi_operands(n)
        outs = [
            pq.apply_ccz(e, axes),
            pq.contract_choi_expansions(left, right, n_op=n + 1, q_act=[n], right_is_choi=False),
        ]
        return [
            np.ascontiguousarray(getattr(o, f)).tobytes()
            for o in outs for f in ("weights", "h_words", "ell", "A_words", "Gamma", "pivots")
        ]

    by_int = run()
    assert len(by_int[0]) > 0 and len(by_int[6]) > 0  # both kernels produced rows
    monkeypatch.setattr(pq, "_addr", lambda a: kl.c_ptr(a, kl.u8_p))
    monkeypatch.setattr(pq, "_w_addr", kl.c_w_ptr)
    assert run() == by_int


def _held(arg: ast.expr) -> bool:
    """A name, an attribute chain, or a constant/name item of one: something a frame still holds."""
    if isinstance(arg, ast.Subscript):
        return _held(arg.value) and isinstance(arg.slice, (ast.Constant, ast.Name))
    return isinstance(arg, ast.Name) or (isinstance(arg, ast.Attribute) and _held(arg.value))


def _addr_sites(path: Path) -> list[tuple[int, str, ast.expr]]:
    return [
        (n.lineno, n.func.id, n.args[0]) for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("_addr", "_w_addr", "addr", "w_addr") and n.args
    ]


def test_no_address_is_taken_from_a_temporary_at_any_call_site() -> None:
    files = [p for p in sorted(_PKG_ROOT.rglob("*.py")) if not p.name.endswith("_test.py")]
    sites = [(p, *s) for p in files for s in _addr_sites(p)]
    assert len(sites) >= 30, "the AST audit found too few call sites to mean anything"
    bad = [f"{p.name}:{ln}: {fn}({ast.unparse(a)})" for p, ln, fn, a in sites if not _held(a)]
    assert bad == []


def test_the_ast_audit_flags_copies_calls_and_slices() -> None:
    cases = [("x", True), ("e.w", True), ("b[1]", True), ("a.copy()", False),
             ("np.ascontiguousarray(a)", False), ("a[1:]", False), ("a + b", False),
             ("f(a).x", False), ("a.copy()[0]", False)]
    for text, ok in cases:
        assert _held(ast.parse(text, mode="eval").body) is ok, text


def test_the_spy_flags_a_freed_array_a_wrong_pointee_and_a_typed_pointer() -> None:
    spy = _KernelSpy(_c_prototypes())
    w, u64, u8 = np.ones(2, np.complex128), np.zeros(2, np.uint64), np.zeros(2, np.uint8)
    tw, t64, t8 = spy.tag(w, mk.w_addr(w), True), spy.tag(u64, mk.addr(u64), False), None
    t8 = spy.tag(u8, mk.addr(u8), False)
    ok = (1, 1, 1, tw, t64, t8, t64, t64, t64, t64, 0)  # magic_apply_pauli_batch
    spy.check("magic_apply_pauli_batch", ok)
    tmp = np.zeros(2, np.uint64)
    dead = spy.tag(tmp, mk.addr(tmp), False)
    del tmp
    gc.collect()
    with pytest.raises(AssertionError, match="freed array"):
        spy.check("magic_apply_pauli_batch", (*ok[:6], dead, *ok[7:]))
    with pytest.raises(AssertionError, match="points at"):
        spy.check("magic_apply_pauli_batch", (*ok[:4], t8, *ok[5:]))
    with pytest.raises(AssertionError, match="unchecked complex"):
        spy.check("magic_apply_pauli_batch", (*ok[:3], spy.tag(w, mk.addr(w), False), *ok[4:]))
    with pytest.raises(AssertionError, match="typed ctypes pointer"):
        spy.check("magic_apply_pauli_batch", (*ok[:4], kl.c_ptr(u64, kl.u64_p), *ok[5:]))
    with pytest.raises(AssertionError, match="not an address"):
        spy.check("magic_apply_pauli_batch", (*ok[:4], int(t64), *ok[5:]))


@pytest.mark.parametrize("rotated", [False, True], ids=["diagonal", "quarter_turn_rotation"])
def test_measure_batch_run_alone_keeps_every_array_it_points_into_alive(
    monkeypatch: pytest.MonkeyPatch, rotated: bool,
) -> None:
    n = 5
    spy = _install_spy(monkeypatch)
    captured: list[Any] = []
    real_pack = pq._measure_batch_kernel

    def capturing(*args: Any) -> Any:
        run, outs = real_pack(*args)
        captured.append(run)
        return run, outs

    monkeypatch.setattr(pq, "_measure_batch_kernel", capturing)
    obs = (_ax(n, 0), [0] * n, 0) if rotated else ([0] * n, _ax(n, 3), 0)
    z = [0] * n  # a quarter turn makes d = 1: every geometry array is then non-empty
    shots = [pq.apply_quarter_turn(_t_state(n, k), _ax(n, 0), z, 0) for k in (2, 3)]
    pq.measure_expansions_batch(shots, obs, rng=np.random.default_rng(1))
    gc.collect()  # the caller's frame is gone: only `run` may still own the packed arrays
    run = captured.pop()
    run(None)  # the spy raises if any pointer argument's owner is dead
    args = list(spy.last_args)
    assert len(args) == 24 and all(a != 0 for a in args)  # 11 in, ra/rb/diag, 10 out
    outcome = np.array([1, 0], np.uint8)
    run(outcome)
    assert [a.ref() is outcome for a in spy.last_args].count(True) == 1
    del run, outcome
    gc.collect()
    assert all(a.ref() is None for a in args), "owners outlive `run`: the liveness check is blind"


def test_measure_batch_is_given_one_canonical_flag_byte_per_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 5
    seen: list[bytes] = []
    lib = mk.get_magic_kernels_lib()
    real = lib.magic_measure_batch

    def reading(*args: Any) -> Any:
        seen.append(ctypes.string_at(args[5], args[0]))  # shot_canon: uint8[n_shots]
        return real(*args)

    monkeypatch.setattr(lib, "magic_measure_batch", reading)
    flags = [False, True, True, True, False, True]
    exps = []
    for merge in flags:
        e = pq.ShotExpansion.zero(n)
        for _ in range(2):
            e = pq.apply_pauli_t(e, _ax(n, 0), [0] * n, 0, merge=merge)
        exps.append(e)
    assert [e._is_canonical_merged for e in exps] == flags
    pq.measure_expansions_batch(exps, (_ax(n, 1), [0] * n, 0), rng=np.random.default_rng(2))
    assert seen[0] == bytes(int(f) for f in flags)


def _stack_poisoner(protos: Protos) -> Any:
    """Return `f(name, args)` that fills the stack slots a kernel call will use with junk.

    A pointer bound narrower than 64 bits only writes the low half of its 8-byte stack slot
    (parameters beyond the sixth integer one), so whether the stale upper half happens to equal
    the real address decides if the call crashes. `getpid` ignores any arguments, so calling it
    through the kernel's own signature with a non-canonical address in every pointer slot makes
    that outcome deterministic without touching C++.
    """
    getpid = ctypes.cast(ctypes.CDLL(None).getpid, ctypes.c_void_p).value
    junk = 0xFFFF_DEAD_0000_0000  # non-canonical: any dereference of it faults
    dummies = {}
    for name, (_, params) in protos.items():
        types = [ctypes.c_void_p if t.endswith("*") else _SCALARS[t] for t, _ in params]
        dummies[name] = (ctypes.CFUNCTYPE(ctypes.c_int, *types)(getpid), types)

    def poison(name: str, args: tuple[Any, ...]) -> None:
        fn, types = dummies[name]
        fn(*[junk if t is ctypes.c_void_p else 0 for t in types])

    return poison


def _child_main() -> None:
    if int(np.empty(1, np.uint8).ctypes.data) <= _FOUR_GB:
        print(json.dumps({"skip": "numpy arrays are not placed above 4 GB by this allocator"}))
        return
    spy = _install_spy(pytest.MonkeyPatch())
    spy.before_call = _stack_poisoner(spy.protos)
    digest = _exercise_all_kernels(5) + _exercise_all_kernels(70)
    print(json.dumps({"digest": digest, "min_addr": spy.min_addr, "seen": sorted(spy.seen)}))


def test_every_kernel_runs_on_arrays_located_above_4_gb() -> None:
    # With the mmap threshold at 1 byte glibc serves every numpy buffer from mmap, which lives
    # far above 4 GB, so a pointer parameter bound as c_int would truncate the address.
    env = dict(
        os.environ, MALLOC_MMAP_THRESHOLD_="1", PYTHONPATH=str(_PKG_ROOT.parent),
        PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
    )
    code = f"import {__name__} as t; t._child_main()"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=240, check=False)
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-3000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])
    assert set(result["seen"]) == _KERNELS
    low = {k: hex(v) for k, v in result["min_addr"].items() if v <= _FOUR_GB}
    assert not low and set(result["min_addr"]) == _KERNELS, f"arguments below 4 GB: {low}"
    assert result["digest"] == _exercise_all_kernels(5) + _exercise_all_kernels(70)
