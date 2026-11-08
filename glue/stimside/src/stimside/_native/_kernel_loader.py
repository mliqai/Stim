"""Shared locate-or-compile logic for the ctypes kernel libraries (coset / tableside / marginal_dem / magic)."""

from __future__ import annotations

import _ctypes
from collections.abc import Callable, Sequence
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import tempfile
from types import ModuleType
from typing import Any

AbiSpec = tuple[str, Sequence[Any], Any]

u8_p = ctypes.POINTER(ctypes.c_uint8)
u32_p = ctypes.POINTER(ctypes.c_uint32)
u64_p = ctypes.POINTER(ctypes.c_uint64)
i8_p = ctypes.POINTER(ctypes.c_int8)
i32_p = ctypes.POINTER(ctypes.c_int32)
i64_p = ctypes.POINTER(ctypes.c_int64)
f64_p = ctypes.POINTER(ctypes.c_double)
size_p = ctypes.POINTER(ctypes.c_size_t)
int_p = ctypes.POINTER(ctypes.c_int)
void_p = ctypes.c_void_p


def c_ptr(arr: Any, ptr_t: Any) -> Any:
    return ctypes.cast(arr.ctypes.data, ptr_t) if arr is not None and arr.size > 0 else ctypes.cast(0, ptr_t)


def c_w_ptr(w: Any) -> Any:
    # No coercion here: a converted temporary would be freed before C++ reads it.
    if (
        w.dtype.char != "D"  # "D" is complex128
        or not w.dtype.isnative
        or w.ndim == 0
        or not (w.flags.c_contiguous and w.flags.aligned)
    ):
        raise TypeError(
            "weights must be a native-endian, aligned, C-contiguous complex128 array with at "
            f"least one axis, got dtype {w.dtype}, shape {w.shape}"
        )
    return c_ptr(w.view("float64"), f64_p)


def get_cpp_compile_and_link_args(
    include_march_native: bool = True,
) -> tuple[list[str], list[str]]:
    machine = (platform.machine() or platform.processor() or "").lower()
    compile_args = [
        "-std=c++17", "-O3", "-fPIC", "-fno-strict-aliasing",
        "-ffp-contract=off",  # no FMA: keeps exact references and native/portable builds in step
        "-g0",
    ]
    if (
        include_march_native
        and os.environ.get("STIMSIDE_PORTABLE_BUILD") != "1"
        and any(
            arch in machine
            for arch in ("x86_64", "amd64", "i686", "i386", "arm64", "aarch64")
        )
    ):
        compile_args.append("-march=native")
    return compile_args, ["-O3", "-shared"]


def _source_mtime_ns(cpp_path: Path) -> int:
    """Newest mtime among the translation unit and the headers beside it.

    Only regular files that are not symlinks count: a link to a directory or to a procfs file
    would report an mtime that the loader's own temp files keep moving, rebuilding every start.
    """
    mtimes = [cpp_path.stat().st_mtime_ns]
    for hpp in cpp_path.parent.glob("*.hpp"):
        try:
            st = hpp.lstat()
        except OSError:  # e.g. deleted just now
            continue
        if stat.S_ISREG(st.st_mode):
            mtimes.append(st.st_mtime_ns)
    return max(mtimes)


def resolve_so(
    module_file: str,
    name: str,
    force_recompile: bool = False,
) -> Path:
    """Locate lib<name>.so next to `module_file`; compile from `<name>.cpp` if missing or stale."""
    here = Path(module_file).resolve().parent
    cpp_path = here / f"{name}.cpp"
    so_path = here / f"lib{name}.so"

    in_source_tree = (
        "site-packages" not in here.parts
        and "dist-packages" not in here.parts
        and (here.parent.parent.parent / "setup.py").exists()
    )
    need_compile = (
        force_recompile
        or not so_path.exists()
        or (
            in_source_tree
            and cpp_path.exists()
            and os.access(here, os.W_OK)
            and _source_mtime_ns(cpp_path) > so_path.stat().st_mtime_ns
        )
    )
    if need_compile:
        if not cpp_path.exists():
            raise FileNotFoundError(
                f"Compiled shared library '{so_path.name}' not found in {here} "
                f"and source '{cpp_path.name}' is missing. "
                f"Please run 'pip install -e .' or 'pip install .' in Stim/glue/stimside."
            )
        if not os.access(here, os.W_OK):
            raise PermissionError(
                f"Cannot compile '{so_path.name}' into read-only directory {here}. "
                f"Please run 'pip install -e .' or 'pip install .' in Stim/glue/stimside."
            )
        tmp_so = here / f".lib{name}.{os.getpid()}.tmp.so"
        cxx = shlex.split(os.environ.get("CXX") or "c++") or ["c++"]
        c_native, link_args = get_cpp_compile_and_link_args(include_march_native=True)
        c_generic, _ = get_cpp_compile_and_link_args(include_march_native=False)
        cmd_variants = [[*cxx, *c_native, *link_args, str(cpp_path), "-o", str(tmp_so)]]
        if c_generic != c_native:
            cmd_variants.append([*cxx, *c_generic, *link_args, str(cpp_path), "-o", str(tmp_so)])
        last_exc: Exception | None = None
        lock_digest = hashlib.sha1(os.fsencode(so_path)).hexdigest()[:16]
        uid = os.getuid() if hasattr(os, "getuid") else 0
        lock_path = Path(tempfile.gettempdir()) / f".stimside_build_{uid}_{lock_digest}.lock"
        lf = None
        try:
            import fcntl  # POSIX-only, and only needed when compiling.

            fd = os.open(
                lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            lf = os.fdopen(fd, "a+b")
            if os.fstat(fd).st_uid == uid:  # Never wait on a lock file another user created.
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # Build without the cross-process lock.
        try:
            if (
                not force_recompile
                and so_path.exists()
                and (not in_source_tree or _source_mtime_ns(cpp_path) <= so_path.stat().st_mtime_ns)
            ):
                return so_path
            for cmd in cmd_variants:
                try:
                    src_mtime_ns = _source_mtime_ns(cpp_path)
                    subprocess.run(cmd, check=True)
                    os.utime(tmp_so, ns=(src_mtime_ns, src_mtime_ns))
                    os.replace(tmp_so, so_path)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if tmp_so.exists():
                        tmp_so.unlink(missing_ok=True)
        finally:
            if lf is not None:
                lf.close()
        if last_exc is not None and (force_recompile or not so_path.exists()):
            raise RuntimeError(
                f"Failed to compile {cpp_path} with {cmd_variants[0]}. "
                f"Please build/install stimside via 'pip install -e .' or 'pip install .'."
            ) from last_exc
    return so_path


def load_cdll(
    resolve: Callable[..., Path], sentinel: str | Sequence[str] | None = None
) -> ctypes.CDLL:
    """Load the resolved library; recompile once if loading fails or a name is not exported.

    `sentinel` is one exported name or a sequence of them. Raises RuntimeError rather than
    return a rebuilt library that still lacks one.
    """
    names = (sentinel,) if isinstance(sentinel, str) else tuple(sentinel or ())
    first_err: OSError | None = None
    try:
        lib = ctypes.CDLL(str(resolve()))
        for name in names:
            getattr(lib, name)
        return lib
    except AttributeError:
        # Else dlopen hands this stale library back for the rebuilt one (same path).
        _ctypes.dlclose(lib._handle)
    except OSError as exc:
        first_err = exc
    try:
        rebuilt_path = resolve(force_recompile=True)
    except Exception as exc:
        if first_err is None:
            raise
        if exc.__cause__ is None:
            raise exc from first_err
        # Keep the compiler error as the cause; still report the initial load error.
        raise RuntimeError(f"{exc} (initial load error: {first_err})") from exc
    lib = ctypes.CDLL(str(rebuilt_path))
    if missing := [name for name in names if not hasattr(lib, name)]:
        # dlopen handed back the stale library: another handle keeps it loaded (or the source
        # lacks it).
        raise RuntimeError(
            f"{lib._name} does not export {', '.join(missing)} even after a rebuild: restart "
            "Python if an older copy of it was loaded earlier in this process."
        )
    return lib


def get_lib(
    mod: ModuleType,
    name: str,
    sentinel: str,
    specs: Sequence[AbiSpec],
    *,
    force_recompile: bool = False,
) -> ctypes.CDLL:
    """Locate/compile, configure ctypes ABI, cache on ``mod._LIB_INSTANCE``, and return the CDLL."""
    cached = getattr(mod, "_LIB_INSTANCE", None)
    if not force_recompile and cached is not None:
        return cached
    if force_recompile:
        mod._LIB_INSTANCE = None  # type: ignore[attr-defined]
    resolve_fn = getattr(mod, f"_resolve_{name}_so")
    resolver = (
        (lambda force_recompile=True: resolve_fn(force_recompile=True))
        if force_recompile
        else resolve_fn
    )
    lib = load_cdll(resolver, tuple(dict.fromkeys((sentinel, *(spec[0] for spec in specs)))))
    for fn_name, argtypes, restype in specs:
        fn = getattr(lib, fn_name)
        fn.argtypes = list(argtypes)
        fn.restype = restype
    mod._LIB_INSTANCE = lib  # type: ignore[attr-defined]
    return lib
