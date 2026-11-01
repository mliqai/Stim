"""Shared locate-or-compile logic for the ctypes kernel libraries (coset / tableside / marginal_dem / magic)."""

from __future__ import annotations

import _ctypes
from collections.abc import Callable, Sequence
import ctypes
try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

AbiSpec = tuple[str, Sequence[Any], Any]


def get_cpp_compile_and_link_args(
    include_march_native: bool = True,
) -> tuple[list[str], list[str]]:
    system = platform.system()
    machine = (platform.machine() or platform.processor() or "").lower()
    if system.startswith("Win"):
        return ["/std:c++17", "/O2"], []
    compile_args = ["-std=c++17", "-O3", "-fPIC", "-fno-strict-aliasing", "-g0"]
    if include_march_native and any(
        arch in machine
        for arch in ("x86_64", "amd64", "i686", "i386", "arm64", "aarch64")
    ):
        compile_args.append("-march=native")
    return compile_args, ["-O3", "-shared"]


def find_prebuilt_so(here: Path, lib_stem: str, cpp_path: Path) -> Path | None:
    patterns = (f"{lib_stem}.so", f"{lib_stem}*.so", f"{lib_stem}*.dylib", f"{lib_stem}*.pyd", f"{lib_stem}*.dll")
    search_dirs: list[Path] = [here]

    rel_pkg = Path(here.parent.name, here.name)
    repo_root = here.parent.parent.parent
    if (repo_root / "setup.py").exists():
        for build_dir_name in ("python_build_stimside", "build"):
            build_root = repo_root / build_dir_name
            if build_root.is_dir():
                for sub in sorted(build_root.glob(f"lib*/{rel_pkg.as_posix()}")):
                    search_dirs.append(sub)

    for sys_entry in sys.path:
        if not sys_entry:
            continue
        candidate_dir = Path(sys_entry) / rel_pkg
        if candidate_dir != here and candidate_dir.is_dir():
            search_dirs.append(candidate_dir)

    cpp_mtime = cpp_path.stat().st_mtime if cpp_path.exists() else None
    for d in search_dirs:
        for pat in patterns:
            for match in sorted(d.glob(pat)):
                if not match.is_file():
                    continue
                if d != here and cpp_mtime is not None and match.stat().st_mtime < cpp_mtime:
                    continue
                return match
    return None


def resolve_so(
    module_file: str,
    name: str,
    force_recompile: bool = False,
    finder: Callable[[Path, str, Path], Path | None] = find_prebuilt_so,
) -> Path:
    """Locate lib<name>.so for <name>.cpp next to `module_file` (or a prebuilt copy); compile if missing/stale."""
    here = Path(module_file).resolve().parent
    cpp_path = here / f"{name}.cpp"
    so_path = here / f"lib{name}.so"

    if not force_recompile:
        found = finder(here, f"lib{name}", cpp_path)
        if found is not None:
            if found.parent != here and os.access(here, os.W_OK):
                tmp_copy = here / f".lib{name}.{os.getpid()}.tmp.so"
                try:
                    shutil.copy2(found, tmp_copy)
                    os.replace(tmp_copy, so_path)
                    found = so_path
                except OSError:
                    tmp_copy.unlink(missing_ok=True)
            so_path = found

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
            and cpp_path.stat().st_mtime > so_path.stat().st_mtime
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
        so_path = here / f"lib{name}.so"
        tmp_so = here / f".lib{name}.{os.getpid()}.tmp.so"
        cxx = (
            shutil.which(os.environ.get("CXX", ""))
            or os.environ.get("CXX")
            or shutil.which("g++")
            or shutil.which("c++")
            or shutil.which("clang++")
            or "g++"
        )
        c_native, link_args = get_cpp_compile_and_link_args(include_march_native=True)
        c_generic, _ = get_cpp_compile_and_link_args(include_march_native=False)
        cmd_variants = [[cxx, *c_native, *link_args, str(cpp_path), "-o", str(tmp_so)]]
        if c_generic != c_native:
            cmd_variants.append([cxx, *c_generic, *link_args, str(cpp_path), "-o", str(tmp_so)])
        last_exc: Exception | None = None
        with cpp_path.open("rb") as lf:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                if (
                    not force_recompile
                    and so_path.exists()
                    and (not in_source_tree or cpp_path.stat().st_mtime <= so_path.stat().st_mtime)
                ):
                    return so_path
                for cmd in cmd_variants:
                    try:
                        subprocess.run(cmd, check=True)
                        target_mtime = max(time.time(), cpp_path.stat().st_mtime + 1.0)
                        os.utime(tmp_so, (target_mtime, target_mtime))
                        os.replace(tmp_so, so_path)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if tmp_so.exists():
                            tmp_so.unlink(missing_ok=True)
            finally:
                if fcntl is not None:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        if last_exc is not None and (force_recompile or not so_path.exists()):
            raise RuntimeError(
                f"Failed to compile {cpp_path} with {cmd_variants[0]}. "
                f"Please build/install stimside via 'pip install -e .' or 'pip install .'."
            ) from last_exc
    return so_path


def load_cdll(resolve: Callable[..., Path], sentinel: str | None = None) -> ctypes.CDLL:
    """Load the resolved library; recompile once if loading fails or `sentinel` is not exported.

    Raises RuntimeError rather than return a rebuilt library that still lacks `sentinel`.
    """
    try:
        lib = ctypes.CDLL(str(resolve()))
        if sentinel is not None:
            getattr(lib, sentinel)
        return lib
    except AttributeError:
        # Else dlopen hands this stale library back for the rebuilt one (same path).
        _ctypes.dlclose(lib._handle)
    except OSError:
        pass
    lib = ctypes.CDLL(str(resolve(force_recompile=True)))
    if sentinel is not None and not hasattr(lib, sentinel):
        # dlopen handed back the stale library: another handle keeps it loaded (or the source lacks it).
        raise RuntimeError(
            f"{lib._name} does not export {sentinel} even after a rebuild: restart Python if an "
            "older copy of it was loaded earlier in this process."
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
    lib = load_cdll(resolver, sentinel)
    for fn_name, argtypes, restype in specs:
        fn = getattr(lib, fn_name)
        fn.argtypes = list(argtypes)
        fn.restype = restype
    mod._LIB_INSTANCE = lib  # type: ignore[attr-defined]
    return lib
