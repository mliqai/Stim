"""Shared locate-or-compile logic for the ctypes kernel libraries.

Loader policy:
- Installed package (``site-packages``, ``dist-packages``, ``pip install --target`` trees with
  ``stimside*.dist-info``, or paths outside ``.../src/stimside/_native`` with ``setup.py``):
  prebuilt ``lib<name>.so`` is loaded directly without checking source mtimes or ``.build_id``
  sidecars and without invoking a compiler.
- Missing ABI symbol or unloadable installed library: ``load_cdll`` triggers a one-time rebuild
  only if ``<name>.cpp`` is present and the directory is writable; otherwise it raises a clear
  error instructing the user to run ``pip install``. If an older copy remains loaded in the
  process, it raises ``RuntimeError`` advising to restart Python.
- Writable source checkout (``.../src/stimside/_native`` with ``setup.py`` and no
  ``stimside*.dist-info``): rebuilds only when ``lib<name>.so`` is missing, older than
  ``<name>.cpp`` or regular non-symlink ``*.hpp`` headers, mismatched against the source hash in
  ``.lib<name>.build_id``, mismatched against recorded ``"machine"`` or ``/proc/cpuinfo``
  ``"cpu_flags"`` digest in ``"native"`` mode, or when ``STIMSIDE_NATIVE_BUILD=1`` /
  ``STIMSIDE_PORTABLE_BUILD=1`` explicitly requests the opposite mode. With no flag set, flags
  never trigger a rebuild and source-triggered rebuilds keep the recorded mode (defaulting to
  ``"portable"``). Missing or invalid/legacy ``.lib<name>.build_id`` sidecars are adopted under
  the directory lock via ``os.link`` as ``"unknown"`` without overwriting valid v2 records or
  rebuilding; ``RecursionError`` during JSON parsing falls back cleanly.
- Concurrency and failure safety: builds acquire a bounded non-blocking ``flock`` on ``_native``
  waiting up to ``600`` s by default (configurable via ``STIMSIDE_BUILD_LOCK_TIMEOUT_S``, falling
  back with ``RuntimeWarning`` on invalid values). Lock timeouts and once-per-library failed
  rebuild fallbacks to an existing ``lib<name>.so`` emit ``RuntimeWarning`` via
  ``warnings.warn``; temp library and sidecar files get unique random names inside ``_native``
  (mode subject to the umask, as compiler output) and are atomically ``os.replace``d (only
  failed sidecar writes after a rebuild warn once per library on ``stderr``).
"""

from __future__ import annotations

import _ctypes
from collections.abc import Callable, Sequence
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import time
import warnings
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


_VALID_BUILD_MODES = frozenset({"portable", "native", "unknown"})
_SOURCE_CHECKOUT_SUFFIX = Path("src/stimside/_native").parts
_LOCK_MAX_RETRIES: int = 12000
_LOCK_RETRY_DELAY_S: float = 0.05
_WARNED_FAILED_REBUILDS: set[Path] = set()
_WARNED_FAILED_BUILD_IDS: set[Path] = set()


def _requested_build_mode() -> str | None:
    if os.environ.get("STIMSIDE_PORTABLE_BUILD") == "1":
        return "portable"
    if os.environ.get("STIMSIDE_NATIVE_BUILD") == "1":
        return "native"
    return None


def _current_machine() -> str:
    return (platform.machine() or platform.processor() or "").lower()


def _mkstemp_with_umask_mode(so_path: Path, suffix: str, mode: int) -> Path:
    """Create a unique empty temp file in ``_native``; the umask applies to ``mode``.

    ``tempfile.mkstemp`` always uses 0600, which would leave built libraries and records unreadable
    for other users; the compiler's own output files get ``mode`` minus the umask.
    """
    for _ in range(100):
        tmp = so_path.with_name(f".{so_path.stem}.{os.getpid()}.{os.urandom(6).hex()}{suffix}")
        try:
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode))
        except FileExistsError:
            continue
        return tmp
    raise FileExistsError(errno.EEXIST, "no unique temporary file name found", str(so_path))


def _cpu_flags_digest() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        if key.strip().lower() in ("flags", "features"):
            tokens = sorted(set(val.split()))
            if tokens:
                return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]
            return ""
    return ""


def _arch_supports_march_native() -> bool:
    machine = _current_machine()
    return any(
        arch in machine
        for arch in ("x86_64", "amd64", "i686", "i386", "arm64", "aarch64")
    )


def _compile_args_for_mode(
    mode: str,
    include_march_native: bool = True,
) -> tuple[list[str], list[str]]:
    compile_args = [
        "-std=c++17", "-O3", "-fPIC", "-fno-strict-aliasing",
        "-ffp-contract=off",  # no FMA: keeps exact references and native/portable builds in step
        "-g0",
    ]
    if include_march_native and mode == "native" and _arch_supports_march_native():
        compile_args.append("-march=native")
    return compile_args, ["-O3", "-shared"]


def get_cpp_compile_and_link_args(
    include_march_native: bool = True,
) -> tuple[list[str], list[str]]:
    mode = "native" if _requested_build_mode() == "native" else "portable"
    return _compile_args_for_mode(mode, include_march_native=include_march_native)


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


def _build_id_path(so_path: Path) -> Path:
    return so_path.with_name(f".{so_path.stem}.build_id")


def _compute_source_hash(cpp_path: Path) -> str:
    """Compute SHA-256 hash over source and regular non-symlink headers beside it."""
    h = hashlib.sha256(b"stimside-source-hash-v2\0")
    cpp_bytes = cpp_path.read_bytes()
    h.update(f"cpp:{cpp_path.name}\0{len(cpp_bytes)}\0".encode())
    h.update(cpp_bytes)
    for hpp in sorted(cpp_path.parent.glob("*.hpp")):
        try:
            st = hpp.lstat()
            if not stat.S_ISREG(st.st_mode):
                continue
            hpp_bytes = hpp.read_bytes()
        except OSError:
            continue
        h.update(f"hpp:{hpp.name}\0{len(hpp_bytes)}\0".encode())
        h.update(hpp_bytes)
    return h.hexdigest()


def _read_build_record(so_path: Path) -> tuple[str, str, str | None, str | None] | None:
    """Return ``(source_hash, mode, machine, cpu_flags)`` from v2 JSON, or ``None``."""
    try:
        data = json.loads(_build_id_path(so_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    ver = data.get("version")
    src_hash = data.get("source_hash")
    mode = data.get("mode")
    if (
        type(ver) is not int
        or ver != 2
        or not isinstance(src_hash, str)
        or not src_hash
        or not isinstance(mode, str)
        or mode not in _VALID_BUILD_MODES
    ):
        return None
    machine = data.get("machine")
    cpu_flags = data.get("cpu_flags")
    if "machine" in data and not isinstance(machine, str):
        return None
    if "cpu_flags" in data and not isinstance(cpu_flags, str):
        return None
    return src_hash, mode, machine, cpu_flags


def _write_build_record(so_path: Path, source_hash: str, mode: str) -> None:
    """Atomically write v2 JSON sidecar describing requested build inputs (not binary hash)."""
    record: dict[str, Any] = {"mode": mode, "source_hash": source_hash, "version": 2}
    if mode == "native":
        record["machine"] = _current_machine()
        record["cpu_flags"] = _cpu_flags_digest()
    payload = json.dumps(record, sort_keys=True)
    tmp_id = _mkstemp_with_umask_mode(so_path, ".tmp.build_id", 0o666)
    try:
        tmp_id.write_text(f"{payload}\n", encoding="utf-8")
        os.replace(tmp_id, _build_id_path(so_path))
    except BaseException:
        tmp_id.unlink(missing_ok=True)
        raise


def _adopt_build_record(so_path: Path, source_hash: str) -> None:
    """Adopt a missing/legacy sidecar as ``unknown`` without replacing a valid v2 record."""
    bid_path = _build_id_path(so_path)
    payload = json.dumps(
        {"mode": "unknown", "source_hash": source_hash, "version": 2}, sort_keys=True
    )
    tmp_id: Path | None = None
    try:
        tmp_id = _mkstemp_with_umask_mode(so_path, ".tmp.build_id", 0o666)
        tmp_id.write_text(f"{payload}\n", encoding="utf-8")
        try:
            os.link(tmp_id, bid_path)
            return
        except FileExistsError:
            if _read_build_record(so_path) is None:
                os.replace(tmp_id, bid_path)
            return
        except OSError:
            if not bid_path.exists() or _read_build_record(so_path) is None:
                os.replace(tmp_id, bid_path)
            return
    except OSError:
        return
    finally:
        if tmp_id is not None:
            tmp_id.unlink(missing_ok=True)


def _is_source_checkout(here: Path) -> bool:
    return (
        "site-packages" not in here.parts
        and "dist-packages" not in here.parts
        and here.parts[-3:] == _SOURCE_CHECKOUT_SUFFIX
        and (here.parent.parent.parent / "setup.py").exists()
        and not any(here.parent.parent.glob("stimside[-_]*.dist-info"))
    )


def _plan_source_build(
    so_path: Path,
    cpp_path: Path,
    in_source_tree: bool,
    requested_mode: str | None,
    force_recompile: bool,
) -> tuple[bool, bool, str, str | None]:
    """Decide whether to rebuild or adopt a missing sidecar and choose the build mode."""
    rec = _read_build_record(so_path) if in_source_tree else None
    rec_mode = rec[1] if rec is not None else None
    keep_mode = rec_mode if rec_mode in ("portable", "native") else "portable"
    effective_mode = requested_mode if requested_mode is not None else keep_mode
    if force_recompile or not so_path.exists():
        return True, False, effective_mode, None
    if not (in_source_tree and cpp_path.exists() and os.access(so_path.parent, os.W_OK)):
        return False, False, "portable", None
    if _source_mtime_ns(cpp_path) > so_path.stat().st_mtime_ns:
        return True, False, effective_mode, None
    src_hash = _compute_source_hash(cpp_path)
    if rec is None:
        if requested_mode is None:
            return False, True, "unknown", src_hash
        return True, False, requested_mode, None
    if rec[0] != src_hash:
        return True, False, effective_mode, None
    if requested_mode is not None and requested_mode != rec_mode:
        return True, False, requested_mode, None
    if rec_mode == "native" and (rec[2] != _current_machine() or rec[3] != _cpu_flags_digest()):
        return True, False, effective_mode, None
    return False, False, rec_mode or "portable", None


def _lock_max_retries() -> int:
    raw = os.environ.get("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "").strip()
    if not raw:
        return max(0, _LOCK_MAX_RETRIES)
    delay = _LOCK_RETRY_DELAY_S
    try:
        timeout_s = float(raw)
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("must be finite and non-negative")
        if timeout_s == 0.0:
            return 0
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError("retry delay must be positive")
        return math.ceil(timeout_s / delay)
    except (ValueError, OverflowError):
        warnings.warn(
            f"stimside warning: invalid STIMSIDE_BUILD_LOCK_TIMEOUT_S={raw!r}; "
            "using default build lock timeout",
            RuntimeWarning,
            stacklevel=3,
        )
        return max(0, _LOCK_MAX_RETRIES)


def _acquire_dir_lock(here: Path) -> int | None:
    """Lock ``here`` with bounded non-blocking ``flock``; return fd or None to build lockless.

    When the lock is skipped (missing ``fcntl``, foreign owner, unsupported filesystem such as
    NFSv4 ``EBADF``, or timeout), concurrent builds remain correct because each process compiles
    to a unique ``.lib<name>.<pid>.*.tmp.so`` and atomically ``os.replace``s it.
    """
    uid = os.getuid() if hasattr(os, "getuid") else 0
    lock_fd: int | None = None
    try:
        import fcntl  # POSIX-only, and only needed when compiling.

        dir_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        lock_fd = os.open(here, dir_flags)
        if os.fstat(lock_fd).st_uid != uid:  # Never wait on another user's lock.
            fd, lock_fd = lock_fd, None
            os.close(fd)
            return None
        max_retries = _lock_max_retries()
        for attempt in range(max_retries + 1):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                if attempt == max_retries:
                    warnings.warn(
                        f"stimside warning: timed out waiting for build lock on {here}; "
                        "proceeding to build without cross-process lock",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    fd, lock_fd = lock_fd, None
                    os.close(fd)
                    return None
                time.sleep(_LOCK_RETRY_DELAY_S)
    except (ImportError, OSError):
        if lock_fd is not None:
            fd, lock_fd = lock_fd, None
            try:
                os.close(fd)
            except OSError:
                pass
        return None
    except BaseException:
        if lock_fd is not None:
            fd, lock_fd = lock_fd, None
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return lock_fd


def _warn_failed_build_id_once(so_path: Path, exc: Exception) -> None:
    if so_path not in _WARNED_FAILED_BUILD_IDS:
        _WARNED_FAILED_BUILD_IDS.add(so_path)
        sys.stderr.write(
            f"stimside warning: failed to write build record for {so_path.name} ({exc}); "
            f"using newly built library {so_path}\n"
        )


def _compile_so_variants(
    cpp_path: Path,
    so_path: Path,
    in_source_tree: bool,
    effective_mode: str,
) -> tuple[Exception | None, list[str]]:
    """Compile ``cpp_path`` into ``so_path`` via unique ``.tmp.so`` with fallback variant."""
    c_primary, link_args = _compile_args_for_mode(effective_mode, include_march_native=True)
    c_generic, _ = _compile_args_for_mode(effective_mode, include_march_native=False)
    cxx = shlex.split(os.environ.get("CXX") or "c++") or ["c++"]
    c_variants = [c_primary] if c_generic == c_primary else [c_primary, c_generic]
    primary_cmd: list[str] | None = None
    last_exc: Exception | None = None
    for c_args in c_variants:
        tmp_so: Path | None = None
        try:
            tmp_so = _mkstemp_with_umask_mode(so_path, ".tmp.so", 0o777)
            cmd = [*cxx, *c_args, *link_args, str(cpp_path), "-o", str(tmp_so)]
            if primary_cmd is None:
                primary_cmd = cmd
            src_mtime_ns = _source_mtime_ns(cpp_path)
            src_hash = _compute_source_hash(cpp_path) if in_source_tree else None
            subprocess.run(cmd, check=True)
            os.utime(tmp_so, ns=(src_mtime_ns, src_mtime_ns))
            os.replace(tmp_so, so_path)
            if src_hash is not None:
                # Record requested mode even on portable fallback so later runs do not loop.
                try:
                    _write_build_record(so_path, src_hash, effective_mode)
                except OSError as exc:
                    _warn_failed_build_id_once(so_path, exc)
            return None, primary_cmd
        except (OSError, subprocess.SubprocessError) as exc:
            last_exc = exc
            if tmp_so is not None:
                tmp_so.unlink(missing_ok=True)
        except BaseException:
            if tmp_so is not None:
                tmp_so.unlink(missing_ok=True)
            raise
    fallback_cmd = primary_cmd or [*cxx, *c_primary, *link_args, str(cpp_path), "-o", str(so_path)]
    return last_exc, fallback_cmd


def _warn_failed_rebuild_once(so_path: Path, exc: Exception) -> None:
    if so_path not in _WARNED_FAILED_REBUILDS:
        _WARNED_FAILED_REBUILDS.add(so_path)
        warnings.warn(
            f"stimside warning: failed to rebuild {so_path.name} ({exc}); "
            f"using existing library {so_path}",
            RuntimeWarning,
            stacklevel=3,
        )


def resolve_so(
    module_file: str,
    name: str,
    force_recompile: bool = False,
) -> Path:
    """Locate lib<name>.so next to `module_file`; compile from `<name>.cpp` if missing/stale."""
    here = Path(module_file).resolve().parent
    cpp_path = here / f"{name}.cpp"
    so_path = here / f"lib{name}.so"
    in_source_tree = _is_source_checkout(here)
    requested_mode = _requested_build_mode()
    needs_build, needs_adopt, _, _ = _plan_source_build(
        so_path, cpp_path, in_source_tree, requested_mode, force_recompile
    )
    if not needs_build and not needs_adopt:
        return so_path
    if needs_build:
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
    lock_fd = _acquire_dir_lock(here)
    last_exc: Exception | None = None
    primary_cmd: list[str] = []
    try:
        needs_build, needs_adopt, effective_mode, src_hash = _plan_source_build(
            so_path, cpp_path, in_source_tree, requested_mode, force_recompile
        )
        if not needs_build:
            if needs_adopt and src_hash is not None:
                _adopt_build_record(so_path, src_hash)
            return so_path
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
        last_exc, primary_cmd = _compile_so_variants(
            cpp_path, so_path, in_source_tree, effective_mode
        )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    if last_exc is not None:
        if force_recompile or not so_path.exists():
            raise RuntimeError(
                f"Failed to compile {cpp_path} with {primary_cmd}. "
                f"Please build/install stimside via 'pip install -e .' or 'pip install .'."
            ) from last_exc
        _warn_failed_rebuild_once(so_path, last_exc)
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
