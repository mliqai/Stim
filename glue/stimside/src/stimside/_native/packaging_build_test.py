"""Packaging, D-39/D-42 build flags, loader policy, build-ID sidecar, and C-8 lock tests."""

from __future__ import annotations

import errno
import functools
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import threading
import time
import types
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import pytest

import stimside
from stimside._native import _kernel_loader, coset_kernels, magic_kernels

_CPP_NAMES = (
    "coset_kernels.cpp",
    "magic_kernels.cpp",
    "marginal_dem_kernels.cpp",
    "tableside_kernels.cpp",
)
_HPP_NAMES = ("qfe_core.hpp", "magic_engine.hpp")
_SO_NAMES = (
    "libcoset_kernels.so",
    "libmagic_kernels.so",
    "libmarginal_dem_kernels.so",
    "libtableside_kernels.so",
)
_KERNEL_NAMES = (
    "coset_kernels",
    "magic_kernels",
    "marginal_dem_kernels",
    "tableside_kernels",
)
_OFFLINE_SKIP_HINTS = (
    "failed to resolve requirements",
    "no solution found when resolving",
    "not found in the cache",
    "because the network was disabled",
    "because networking is disabled",
    "failed to fetch",
    "failed to download",
    "offline mode",
)
_BACKEND_FAILURE_MARKERS = (
    "the build backend returned an error",
    "traceback (most recent call last)",
    "running build_ext",
    "setuptools.build_meta",
    "error: command",
    "compileerror",
    "error: #error",
)


def _find_uv() -> str | None:
    return shutil.which("uv") or (
        str(Path.home() / ".local" / "bin" / "uv")
        if (Path.home() / ".local" / "bin" / "uv").is_file()
        else None
    )


def _source_pkg_root() -> Path | None:
    parents = Path(stimside.__file__).resolve().parents
    if len(parents) < 3:
        return None
    root = parents[2]
    if (
        (root / "pyproject.toml").is_file()
        and (root / "setup.py").is_file()
        and (root / "src" / "stimside" / "__init__.py").is_file()
    ):
        return root
    return None


def _cxx_cmd() -> list[str]:
    return shlex.split(os.environ.get("CXX") or "c++") or ["c++"]


def _require_cxx() -> list[str]:
    cxx = _cxx_cmd()
    if shutil.which(cxx[0]) is None:
        pytest.skip(f"C++ compiler {cxx[0]!r} not available")
    return cxx


def _make_source_native(
    tmp_path: Path, name: str, cpp_src: str
) -> tuple[Path, Path, functools.partial[Path]]:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / f"{name}.cpp"
    cpp.write_text(cpp_src, encoding="utf-8")
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / f"{name}.py"), name)
    return native, cpp, resolve


def _read_sidecar_json(so_path: Path) -> dict[str, object]:
    sidecar = _kernel_loader._build_id_path(so_path)
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _seed_dirty_native_copy(pkg_copy: Path) -> None:
    native = pkg_copy / "src" / "stimside" / "_native"
    for name in _KERNEL_NAMES:
        so = native / f"lib{name}.so"
        bid = native / f".lib{name}.build_id"
        if not so.is_file() or not bid.is_file():
            _kernel_loader.resolve_so(str(native / f"{name}.py"), name)
        assert so.is_file(), f"missing {so}"
        assert bid.is_file(), f"missing {bid}"


def _is_offline_uv_prereq_error(stderr: str) -> bool:
    """Return True only for offline uv resolver errors before build-backend execution."""
    err_low = stderr.lower()
    if any(marker in err_low for marker in _BACKEND_FAILURE_MARKERS):
        return False
    return any(hint in err_low for hint in _OFFLINE_SKIP_HINTS)


def _run_uv_build_offline(uv_bin: str, args: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(
        [uv_bin, "build", "--offline", *args],
        capture_output=True, text=True, env=env, timeout=600, check=False,
    )
    if proc.returncode != 0:
        if _is_offline_uv_prereq_error(proc.stderr):
            pytest.skip(f"offline uv build prerequisites unavailable: {proc.stderr.strip()[:200]}")
        assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout


def _assert_wheel_ok(whl_path: Path, plat: str) -> None:
    assert re.search(rf"-py3-none-{re.escape(plat)}\.whl$", whl_path.name), whl_path.name
    with zipfile.ZipFile(whl_path) as zf:
        wnames = set(zf.namelist())
        wheel_meta = next(n for n in wnames if n.endswith(".dist-info/WHEEL"))
        assert f"Tag: py3-none-{plat}" in zf.read(wheel_meta).decode("utf-8")
    assert "stimside/golden_digests.json" in wnames
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in wnames)
    for name in _CPP_NAMES + _HPP_NAMES + _SO_NAMES:
        assert f"stimside/_native/{name}" in wnames
    assert not any(
        n.endswith((".build_id", ".lock"))
        or ".tmp." in Path(n).name
        or ".stimside_build_" in Path(n).name
        for n in wnames
    )


@pytest.mark.parametrize(
    "portable,native,include_flag,machine,expect_march",
    [
        (None, None, True, "x86_64", False),
        (None, "0", True, "x86_64", False),
        (None, "", True, "x86_64", False),
        (None, "yes", True, "x86_64", False),
        ("0", None, True, "aarch64", False),
        ("1", None, True, "x86_64", False),
        (None, "1", True, "x86_64", True),
        ("0", "1", True, "arm64", True),
        ("1", "1", True, "x86_64", False),
        (None, "1", False, "x86_64", False),
        (None, "1", True, "riscv64", False),
    ],
)
def test_d39_flag_truth_table(
    monkeypatch: pytest.MonkeyPatch,
    portable: str | None,
    native: str | None,
    include_flag: bool,
    machine: str,
    expect_march: bool,
) -> None:
    if portable is None:
        monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    else:
        monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", portable)
    if native is None:
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    else:
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", native)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    c_args, l_args = _kernel_loader.get_cpp_compile_and_link_args(include_flag)
    assert "-ffp-contract=off" in c_args
    assert ("-march=native" in c_args) is expect_march
    assert l_args == ["-O3", "-shared"]


def test_sdist_and_wheel_contents_and_wheel_tag(tmp_path: Path) -> None:
    pkg_root = _source_pkg_root()
    if pkg_root is None:
        pytest.skip("packaging build test requires a source checkout")
    uv_bin = _find_uv()
    if uv_bin is None:
        pytest.skip("uv binary not available for offline sdist/wheel build")
    _require_cxx()
    pkg_copy = tmp_path / "pkg"
    ignore = shutil.ignore_patterns(
        "build", "dist", "python_build_stimside", "*.egg-info",
        "__pycache__", ".pytest_cache", ".benchmarks",
    )
    shutil.copytree(pkg_root, pkg_copy, ignore=ignore)
    _seed_dirty_native_copy(pkg_copy)
    dist_dir, dist_direct, tmp_dir = tmp_path / "dist", tmp_path / "dist_direct", tmp_path / "tmp"
    tmp_dir.mkdir()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_dir)}
    env.pop("STIMSIDE_NATIVE_BUILD", None)
    _run_uv_build_offline(uv_bin, ["--out-dir", str(dist_dir), str(pkg_copy)], env)
    sdists = list(dist_dir.glob("*.tar.gz"))
    wheels = list(dist_dir.glob("*.whl"))
    assert len(sdists) == 1 and len(wheels) == 1

    with tarfile.open(sdists[0], "r:gz") as tf:
        sdist_rel = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
    for req in ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml", "setup.py"):
        assert req in sdist_rel
    assert "src/stimside/golden_digests.json" in sdist_rel
    for name in _CPP_NAMES + _HPP_NAMES:
        assert f"src/stimside/_native/{name}" in sdist_rel
    assert not any(n.endswith((".so", ".build_id")) or ".tmp." in Path(n).name for n in sdist_rel)

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    _assert_wheel_ok(wheels[0], plat)

    # Refresh one sidecar/mtime to stale values before direct wheel build to verify sync.
    coset_cpp = pkg_copy / "src" / "stimside" / "_native" / "coset_kernels.cpp"
    coset_so = pkg_copy / "src" / "stimside" / "_native" / "libcoset_kernels.so"
    coset_bid = _kernel_loader._build_id_path(coset_so)
    os.utime(coset_so, ns=(1_000_000_000, 1_000_000_000))
    coset_bid.write_text(
        json.dumps({"mode": "native", "source_hash": "0" * 64, "version": 2}) + "\n",
        encoding="utf-8",
    )

    # Also build a wheel directly from the dirty copy holding all 4 .so and .build_id files.
    _run_uv_build_offline(uv_bin, ["--wheel", "--out-dir", str(dist_direct), str(pkg_copy)], env)
    direct_wheels = list(dist_direct.glob("*.whl"))
    assert len(direct_wheels) == 1
    _assert_wheel_ok(direct_wheels[0], plat)
    synced_rec = _read_sidecar_json(coset_so)
    assert synced_rec["version"] == 2 and synced_rec["mode"] == "portable"
    assert synced_rec["source_hash"] == _kernel_loader._compute_source_hash(coset_cpp)
    assert coset_so.stat().st_mtime_ns == _kernel_loader._source_mtime_ns(coset_cpp)


def test_b2_loader_policy_installed_vs_missing_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    site = tmp_path / "site-packages" / "stimside" / "_native"
    site.mkdir(parents=True)
    real_so = coset_kernels._resolve_coset_kernels_so()
    so_copy = site / "libcoset_kernels.so"
    shutil.copy2(real_so, so_copy)
    cpp_copy = site / "coset_kernels.cpp"
    cpp_copy.write_text("// edited newer source\n", encoding="utf-8")
    newer_ns = so_copy.stat().st_mtime_ns + 10_000_000_000
    os.utime(cpp_copy, ns=(newer_ns, newer_ns))
    monkeypatch.setattr(coset_kernels, "__file__", str(site / "coset_kernels.py"))
    with mock.patch.object(subprocess, "run") as mock_run:
        assert coset_kernels._resolve_coset_kernels_so() == so_copy
    mock_run.assert_not_called()

    stub = tmp_path / "site_stub" / "stimside" / "_native"
    stub.mkdir(parents=True)
    stub_cpp = stub / "tiny_b2.cpp"
    stub_cpp.write_text('extern "C" int tiny_b2_v1() { return 1; }\n', encoding="utf-8")
    subprocess.run(
        [*cxx, "-shared", "-fPIC", str(stub_cpp), "-o", str(stub / "libtiny_b2.so")],
        check=True,
    )
    stub_cpp.write_text(
        'extern "C" int tiny_b2_v1() { return 1; }\n'
        'extern "C" int tiny_b2_v2() { return 2; }\n',
        encoding="utf-8",
    )
    resolve = functools.partial(_kernel_loader.resolve_so, str(stub / "tiny_b2.py"), "tiny_b2")
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    lib = _kernel_loader.load_cdll(resolve, ("tiny_b2_v1", "tiny_b2_v2"))
    assert lib.tiny_b2_v2() == 2
    assert not (stub / ".libtiny_b2.build_id").exists()

    nocpp = tmp_path / "site_nocpp" / "stimside" / "_native"
    nocpp.mkdir(parents=True)
    subprocess.run(
        [*cxx, "-shared", "-fPIC", "-x", "c++", "-", "-o", str(nocpp / "libtiny_nc.so")],
        input='extern "C" int tiny_nc_v1() { return 1; }\n', text=True, check=True,
    )
    resolve_nc = functools.partial(_kernel_loader.resolve_so, str(nocpp / "tiny_nc.py"), "tiny_nc")
    with pytest.raises(FileNotFoundError, match="pip install"):
        _kernel_loader.load_cdll(resolve_nc, ("tiny_nc_v1", "tiny_nc_v2"))

    ro = tmp_path / "site_ro" / "stimside" / "_native"
    ro.mkdir(parents=True)
    subprocess.run(
        [*cxx, "-shared", "-fPIC", "-x", "c++", "-", "-o", str(ro / "libtiny_ro.so")],
        input='extern "C" int tiny_ro_v1() { return 1; }\n', text=True, check=True,
    )
    (ro / "tiny_ro.cpp").write_text(
        'extern "C" int tiny_ro_v1() { return 1; }\n'
        'extern "C" int tiny_ro_v2() { return 2; }\n',
        encoding="utf-8",
    )
    resolve_ro = functools.partial(_kernel_loader.resolve_so, str(ro / "tiny_ro.py"), "tiny_ro")
    ro.chmod(0o555)
    try:
        if not os.access(ro, os.W_OK):
            with pytest.raises(PermissionError, match="pip install"):
                _kernel_loader.load_cdll(resolve_ro, ("tiny_ro_v1", "tiny_ro_v2"))
    finally:
        ro.chmod(0o755)


def _eval_symbol_in_subprocess(so: Path, sym: str) -> int:
    code = f"import ctypes; print(ctypes.CDLL({str(so)!r}).{sym}())"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return int(out.stdout.strip())


def test_b3_build_id_rebuilds_on_older_edit_and_flag_toggle_without_lookalikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, cpp, resolve = _make_source_native(
        tmp_path,
        "tiny_b3",
        '#include "tiny_b3.hpp"\nextern "C" int tiny_b3_val() { return B3_VAL; }\n',
    )
    hpp = native / "tiny_b3.hpp"
    hpp.write_text("#define B3_VAL 10\n", encoding="utf-8")
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    rec_portable = _read_sidecar_json(so)
    assert rec_portable["version"] == 2 and rec_portable["mode"] == "portable"

    # Lookalike *.hpp symlink / directory must not trigger rebuild.
    (native / "junk_dir.hpp").mkdir()
    (native / "junk_link.hpp").symlink_to(tmp_path / "missing.hpp")
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()

    # Older-mtime .hpp edit triggers rebuild via source_hash mismatch.
    old_ns = so.stat().st_mtime_ns - 5_000_000_000
    hpp.write_text("#define B3_VAL 22\n", encoding="utf-8")
    os.utime(hpp, ns=(old_ns, old_ns))
    so = resolve()
    assert so.stat().st_mtime_ns == max(cpp.stat().st_mtime_ns, old_ns)
    assert _eval_symbol_in_subprocess(so, "tiny_b3_val") == 22

    # Older-mtime .cpp edit also triggers rebuild via source_hash mismatch.
    cpp.write_text(
        '#include "tiny_b3.hpp"\nextern "C" int tiny_b3_val() { return B3_VAL + 1; }\n',
        encoding="utf-8",
    )
    os.utime(cpp, ns=(old_ns, old_ns))
    so = resolve()
    assert _eval_symbol_in_subprocess(so, "tiny_b3_val") == 23

    # Explicit STIMSIDE_NATIVE_BUILD=1 switches mode to native; unsetting keeps native sticky.
    rec_before = _read_sidecar_json(so)
    assert rec_before["source_hash"] != rec_portable["source_hash"]
    assert rec_before["mode"] == "portable"
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    resolve()
    rec_native = _read_sidecar_json(so)
    assert rec_native["mode"] == "native"
    assert rec_native["source_hash"] == rec_before["source_hash"]
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()
    assert not any("build_id" in s[0] for s in magic_kernels._SPECS)


def test_b4_zero_lock_files_and_multiprocess_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    native, _, _ = _make_source_native(
        tmp_path, "tiny_c8", 'extern "C" int tiny_c8_v1() { return 7; }\n'
    )
    tmp_dir = tmp_path / "sys_tmp"
    tmp_dir.mkdir()
    compiles_dir = tmp_path / "compiles"
    compiles_dir.mkdir()
    wrapper = tmp_path / "cxx.sh"
    cxx_quoted = " ".join(shlex.quote(c) for c in cxx)
    wrapper.write_text(
        f'#!/bin/sh\ntouch "{compiles_dir}/c_$$_$(date +%s%N)"\nexec {cxx_quoted} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    go_flag = tmp_path / "go"
    src_root = str(Path(stimside.__file__).resolve().parents[1])
    code = (
        "import ctypes, pathlib, time\n"
        "from stimside._native import _kernel_loader\n"
        f"go = pathlib.Path({str(go_flag)!r})\n"
        "while not go.exists():\n"
        "    time.sleep(0.01)\n"
        f"so = _kernel_loader.resolve_so({str(native / 'tiny_c8.py')!r}, 'tiny_c8')\n"
        "print('OK:' + str(ctypes.CDLL(str(so)).tiny_c8_v1()))\n"
    )
    env = {
        **os.environ,
        "CXX": str(wrapper),
        "PYTHONPATH": src_root,
        "TMPDIR": str(tmp_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        for _ in range(4)
    ]
    go_flag.write_text("1", encoding="utf-8")
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        assert out.strip() == "OK:7", out
    assert len(list(compiles_dir.iterdir())) == 1
    assert list(tmp_dir.iterdir()) == []
    assert list(native.glob("*lock*")) == []
    assert list(native.glob("*.tmp.*")) == []
    assert (native / ".libtiny_c8.build_id").is_file()


@pytest.mark.parametrize("umask", [0o022, 0o002])
def test_loader_built_library_and_record_modes_follow_the_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, umask: int
) -> None:
    """Other users must be able to load what the loader builds (temp files are not left 0600)."""
    _require_cxx()
    _, _, resolve = _make_source_native(
        tmp_path, "tiny_mode", 'extern "C" int tiny_mode_v1() { return 1; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    old = os.umask(umask)
    try:
        so = resolve()
    finally:
        os.umask(old)
    assert stat.S_IMODE(so.stat().st_mode) == 0o777 & ~umask
    record = _kernel_loader._build_id_path(so)
    assert stat.S_IMODE(record.stat().st_mode) == 0o666 & ~umask
    assert list(so.parent.glob("*.tmp.*")) == []


def test_b3_same_length_older_mtime_cpp_edit_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    old_src = 'extern "C" int tiny_l08_val() { return 10; }\n'
    new_src = 'extern "C" int tiny_l08_val() { return 22; }\n'
    assert len(old_src) == len(new_src)
    _, cpp, resolve = _make_source_native(tmp_path, "tiny_l08", old_src)
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    assert _eval_symbol_in_subprocess(so, "tiny_l08_val") == 10
    old_ns = so.stat().st_mtime_ns - 5_000_000_000
    cpp.write_text(new_src, encoding="utf-8")
    os.utime(cpp, ns=(old_ns, old_ns))
    so = resolve()
    assert _eval_symbol_in_subprocess(so, "tiny_l08_val") == 22


def test_b3_readonly_source_checkout_with_stale_so_does_not_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, cpp, resolve = _make_source_native(
        tmp_path, "tiny_l14", 'extern "C" int tiny_l14_val() { return 1; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    cpp.write_text('extern "C" int tiny_l14_val() { return 2; }\n', encoding="utf-8")
    os.utime(cpp, ns=(newer_ns, newer_ns))
    native.chmod(0o555)
    try:
        if os.access(native, os.W_OK):
            pytest.skip("cannot make directory read-only for this user")
        with mock.patch.object(subprocess, "run") as mock_run:
            assert resolve() == so
        mock_run.assert_not_called()
    finally:
        native.chmod(0o755)


def test_d42_sticky_flags_source_checkout_and_installed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, cpp, resolve = _make_source_native(
        tmp_path, "tiny_d42", 'extern "C" int tiny_d42_val() { return 1; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    real_run = subprocess.run
    seen: list[list[str]] = []

    def spy_run(cmd: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[object]:
        seen.append(list(cmd))
        return real_run(cmd, *a, **kw)  # type: ignore[arg-type]

    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        # Row 1: portable build -> no-var (0), PORTABLE=1 (0), NATIVE=1 (1 native).
        so = resolve()
        assert len(seen) == 1 and "-march=native" not in seen[-1]
        assert _read_sidecar_json(so)["mode"] == "portable"
        assert resolve() == so and len(seen) == 1
        monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1")
        assert resolve() == so and len(seen) == 1
        monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD")
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert resolve() == so and len(seen) == 2
        if _kernel_loader._arch_supports_march_native():
            assert "-march=native" in seen[-1]
        assert _read_sidecar_json(so)["mode"] == "native"

        # Row 2: native build -> no-var (0, stays native), NATIVE=1 (0), source edit keeps native.
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
        assert resolve() == so and len(seen) == 2
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert resolve() == so and len(seen) == 2
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
        cpp.write_text('extern "C" int tiny_d42_val() { return 2; }\n', encoding="utf-8")
        newer_ns = so.stat().st_mtime_ns + 5_000_000_000
        os.utime(cpp, ns=(newer_ns, newer_ns))
        assert resolve() == so and len(seen) == 3
        if _kernel_loader._arch_supports_march_native():
            assert "-march=native" in seen[-1]
        assert _read_sidecar_json(so)["mode"] == "native"

        # Explicit PORTABLE=1 (alone or beating NATIVE=1) switches back to portable once.
        monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1")
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert resolve() == so and len(seen) == 4
        assert "-march=native" not in seen[-1]
        assert _read_sidecar_json(so)["mode"] == "portable"

        # Row 4: hand-replaced .so with kept .build_id (fresh mtime) -> no-var (0), NATIVE=1 (1).
        monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD")
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
        hand_tmp = native / "hand.so"
        shutil.copy2(so, hand_tmp)
        os.utime(hand_tmp, ns=(newer_ns + 1_000_000, newer_ns + 1_000_000))
        os.replace(hand_tmp, so)
        assert resolve() == so and len(seen) == 4
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert resolve() == so and len(seen) == 5
        assert _read_sidecar_json(so)["mode"] == "native"

        # Row 5: installed site-packages -> 0 compiles under no-var and NATIVE=1, no sidecar.
        site = tmp_path / "site-packages" / "stimside" / "_native"
        site.mkdir(parents=True)
        site_so = site / "libtiny_d42.so"
        shutil.copy2(so, site_so)
        shutil.copy2(cpp, site / "tiny_d42.cpp")
        site_resolve = functools.partial(
            _kernel_loader.resolve_so, str(site / "tiny_d42.py"), "tiny_d42"
        )
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
        assert site_resolve() == site_so and len(seen) == 5
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert site_resolve() == site_so and len(seen) == 5
        assert not _kernel_loader._build_id_path(site_so).exists()


def test_d42_missing_and_legacy_v1_build_id_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_adopt", 'extern "C" int tiny_adopt_val() { return 1; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    sidecar = _kernel_loader._build_id_path(so)

    # Row 3 (no .build_id, fresh .so): no-var adopts without rebuilding, then NATIVE=1 builds once.
    sidecar.unlink()
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()
    adopted = _read_sidecar_json(so)
    assert adopted["version"] == 2 and adopted["mode"] == "unknown"
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    resolve()
    assert _read_sidecar_json(so)["mode"] == "native"

    # Legacy v1 plain-hex .build_id: adopted without rebuilding, then PORTABLE=1 builds once.
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    sidecar.write_text("a" * 64 + "\n", encoding="utf-8")
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()
    assert _read_sidecar_json(so)["mode"] == "unknown"
    monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1")
    resolve()
    assert _read_sidecar_json(so)["mode"] == "portable"

    # Missing .build_id with immediate explicit NATIVE=1 rebuilds once into native.
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD")
    sidecar.unlink()
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    resolve()
    assert _read_sidecar_json(so)["mode"] == "native"

    # Adopted unknown mode + source edit with no env var defaults to portable (D-39).
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    sidecar.unlink()
    resolve()
    assert _read_sidecar_json(so)["mode"] == "unknown"
    cpp.write_text('extern "C" int tiny_adopt_val() { return 2; }\n', encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))
    resolve()
    assert _read_sidecar_json(so)["mode"] == "portable"


def test_d42_march_native_fallback_does_not_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    _, _, resolve = _make_source_native(
        tmp_path, "tiny_noloop", 'extern "C" int tiny_noloop_val() { return 9; }\n'
    )
    inv_log = tmp_path / "cxx_calls.log"
    cxx_quoted = " ".join(shlex.quote(c) for c in cxx)
    wrapper = tmp_path / "cxx_nomarch.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{inv_log}"\n'
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "-march=native" ]; then exit 1; fi\n'
        "done\n"
        f'exec {cxx_quoted} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("CXX", str(wrapper))
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    so = resolve()
    assert _eval_symbol_in_subprocess(so, "tiny_noloop_val") == 9
    assert len(inv_log.read_text(encoding="utf-8").splitlines()) == 2
    assert _read_sidecar_json(so)["mode"] == "native"

    # Second run with NATIVE=1 and third with unset env must perform 0 compiles.
    assert resolve() == so
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    assert resolve() == so
    assert len(inv_log.read_text(encoding="utf-8").splitlines()) == 2


def test_b2_2_stricter_source_checkout_rejects_vendor_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    proj = tmp_path / "proj"
    vendor_native = proj / "vendor" / "stimside" / "_native"
    vendor_native.mkdir(parents=True)
    (proj / "setup.py").write_text("# unrelated project setup.py\n", encoding="utf-8")
    cpp = vendor_native / "tiny_v.cpp"
    so = vendor_native / "libtiny_v.so"
    cpp.write_text('extern "C" int tiny_v_val() { return 1; }\n', encoding="utf-8")
    subprocess.run([*cxx, "-shared", "-fPIC", str(cpp), "-o", str(so)], check=True)
    cpp.write_text('extern "C" int tiny_v_val() { return 2; }\n', encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))
    resolve = functools.partial(
        _kernel_loader.resolve_so, str(vendor_native / "tiny_v.py"), "tiny_v"
    )

    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
        monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
        assert resolve() == so
    mock_run.assert_not_called()
    assert not _kernel_loader._build_id_path(so).exists()


def test_b2_3_failed_rebuild_warns_once_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_warn", 'extern "C" int tiny_warn_val() { return 3; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()

    cpp.write_text("#error intentional_b2_3_failure\n", encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        assert resolve() == so
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    first_msg = str(caught[0].message)
    assert "stimside warning: failed to rebuild libtiny_warn.so" in first_msg
    assert "using existing library" in first_msg

    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always", RuntimeWarning)
        assert resolve() == so
    assert caught_again == []


def test_b2_4_bounded_flock_wait_warns_and_builds_lockless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_to", 'extern "C" int tiny_to_val() { return 7; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", raising=False)
    monkeypatch.setattr(_kernel_loader, "_LOCK_MAX_RETRIES", 2)
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.01)

    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            so = resolve()
    finally:
        os.close(holder_fd)

    matches = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "stimside warning: timed out waiting for build lock" in str(w.message)
    ]
    assert len(matches) == 1
    assert len(caught) == 1
    assert _eval_symbol_in_subprocess(so, "tiny_to_val") == 7
    assert list(native.glob("*lock*")) == []
    assert list(native.glob("*.tmp.*")) == []


def test_r8_r2_offline_uv_error_classifier_does_not_skip_compile_failures() -> None:
    offline_stderr = (
        "Building source distribution...\n"
        "  Failed to resolve requirements from `build-system.requires`\n"
        "  No solution found when resolving: `setuptools>=77`\n"
        "  Because setuptools was not found in the cache and the network was disabled.\n"
    )
    compile_stderr = (
        "running build_ext\n"
        "coset_kernels.cpp:1:2: error: #error INTENTIONAL_COMPILE_ERROR_REV3\n"
        "error: Command '['c++', ...]' returned non-zero exit status 1.\n"
        "The build backend returned an error\n"
        "Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
    )
    setup_tb_stderr = (
        "Traceback (most recent call last):\n"
        "  File 'setup.py', line 1\n"
        "RuntimeError: not found in the cache and offline mode inside setup.py\n"
        "The build backend returned an error\n"
    )
    compile_hint_stderr = (
        "running build_ext\n"
        "coset_kernels.cpp:1:2: error: #error not found in the cache\n"
        "The build backend returned an error\n"
    )
    assert _is_offline_uv_prereq_error(offline_stderr) is True
    assert _is_offline_uv_prereq_error(compile_stderr) is False
    assert _is_offline_uv_prereq_error(setup_tb_stderr) is False
    assert _is_offline_uv_prereq_error(compile_hint_stderr) is False

    fake_offline = subprocess.CompletedProcess(["uv"], 1, stdout="", stderr=offline_stderr)
    with (
        mock.patch.object(subprocess, "run", return_value=fake_offline),
        pytest.raises(pytest.skip.Exception),
    ):
        _run_uv_build_offline("uv", [], {})

    for bad_stderr in (compile_stderr, setup_tb_stderr, compile_hint_stderr):
        fake_err = subprocess.CompletedProcess(["uv"], 1, stdout="", stderr=bad_stderr)
        with (
            mock.patch.object(subprocess, "run", return_value=fake_err),
            pytest.raises(AssertionError),
        ):
            _run_uv_build_offline("uv", [], {})


def test_r8_r2_b2_3_assertion_error_propagates_from_resolve_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_ae", 'extern "C" int tiny_ae_val() { return 1; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    cpp.write_text('extern "C" int tiny_ae_val() { return 2; }\n', encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))
    with (
        mock.patch.object(subprocess, "run", side_effect=AssertionError("mocked boom")),
        pytest.raises(AssertionError, match="mocked boom"),
    ):
        resolve()


def test_r8_r2_b2_4_ebadf_fast_breaks_without_retries_or_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fcntl = pytest.importorskip("fcntl")
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    calls = 0
    sleeps = 0

    def fake_flock(_fd: int, _op: int) -> None:
        nonlocal calls
        calls += 1
        raise OSError(errno.EBADF, "Bad file descriptor")

    def fake_sleep(_s: float) -> None:
        nonlocal sleeps
        sleeps += 1

    monkeypatch.setattr(fcntl, "flock", fake_flock)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    fds_before = len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").is_dir() else None
    capsys.readouterr()
    assert _kernel_loader._acquire_dir_lock(native) is None
    err = capsys.readouterr().err
    fds_after = len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").is_dir() else None
    assert calls == 1
    assert sleeps == 0
    assert err == ""
    assert fds_after == fds_before


def test_r8_r2_b2_build_id_dir_error_keeps_native_so_without_fallback_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cxx()
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_bid_dir", 'extern "C" int tiny_bid_dir_val() { return 5; }\n'
    )
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    bid_dir = native / ".libtiny_bid_dir.build_id"
    bid_dir.mkdir()
    cmds: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd: list[str], *args: object, **kwargs: object) -> object:
        cmds.append(list(cmd))
        return real_run(cmd, *args, **kwargs)  # type: ignore[arg-type]

    _kernel_loader._WARNED_FAILED_BUILD_IDS.discard(native / "libtiny_bid_dir.so")
    capsys.readouterr()
    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        so = resolve()
    err = capsys.readouterr().err
    assert len(cmds) == 1
    assert "-march=native" in cmds[0]
    assert "failed to write build record for libtiny_bid_dir.so" in err
    assert "failed to rebuild" not in err
    assert _eval_symbol_in_subprocess(so, "tiny_bid_dir_val") == 5


def test_r8_r2_acquire_dir_lock_no_double_close_on_os_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    real_close = os.close
    closed_calls: list[int] = []

    def fail_close(fd: int) -> None:
        closed_calls.append(fd)
        try:
            real_close(fd)
        finally:
            raise OSError(errno.EIO, "close failed")

    monkeypatch.setattr(os, "getuid", lambda: 999_999)
    monkeypatch.setattr(os, "close", fail_close)
    assert _kernel_loader._acquire_dir_lock(native) is None
    assert len(closed_calls) == 1


def test_r8_delta_acquire_dir_lock_timeout_no_double_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fcntl = pytest.importorskip("fcntl")
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    real_close = os.close
    closed_calls: list[int] = []

    def fail_close(fd: int) -> None:
        closed_calls.append(fd)
        try:
            real_close(fd)
        finally:
            raise OSError(errno.EIO, "close failed on timeout")

    def busy_flock(_fd: int, _op: int) -> None:
        raise OSError(errno.EAGAIN, "busy")

    monkeypatch.setattr(fcntl, "flock", busy_flock)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.delenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", raising=False)
    monkeypatch.setattr(_kernel_loader, "_LOCK_MAX_RETRIES", 1)
    monkeypatch.setattr(os, "close", fail_close)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        assert _kernel_loader._acquire_dir_lock(native) is None
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "timed out waiting for build lock" in str(caught[0].message)
    assert len(closed_calls) == 1


def _make_stub_package(root: Path, pkg_root: Path) -> Path:
    for fname in ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml", "setup.py"):
        shutil.copy2(pkg_root / fname, root / fname)
    shutil.copy2(pkg_root / "MANIFEST.in", root / "MANIFEST.in")
    pkg = root / "src" / "stimside"
    native = pkg / "_native"
    native.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (pkg / "golden_digests.json").write_text("{}\n", encoding="utf-8")
    (native / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        pkg_root / "src" / "stimside" / "_native" / "_kernel_loader.py",
        native / "_kernel_loader.py",
    )
    for hpp in _HPP_NAMES:
        (native / hpp).write_text("// stub header\n", encoding="utf-8")
    for kname in _KERNEL_NAMES:
        (native / f"{kname}.cpp").write_text(
            f'extern "C" int {kname}_stub() {{ return 1; }}\n',
            encoding="utf-8",
        )
    return native


def test_sdist_and_wheel_native_build_syncs_cpu_identity(tmp_path: Path) -> None:
    pkg_root = _source_pkg_root()
    if pkg_root is None:
        pytest.skip("packaging build test requires a source checkout")
    uv_bin = _find_uv()
    if uv_bin is None:
        pytest.skip("uv binary not available for offline sdist/wheel build")
    _require_cxx()
    stub_pkg = tmp_path / "stub_pkg"
    stub_pkg.mkdir()
    native = _make_stub_package(stub_pkg, pkg_root)
    dist_dir, tmp_dir = tmp_path / "dist", tmp_path / "tmp"
    tmp_dir.mkdir()
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_dir),
        "STIMSIDE_NATIVE_BUILD": "1",
    }
    env.pop("STIMSIDE_PORTABLE_BUILD", None)
    _run_uv_build_offline(uv_bin, ["--wheel", "--out-dir", str(dist_dir), str(stub_pkg)], env)
    assert len(list(dist_dir.glob("*.whl"))) == 1
    expected_machine = _kernel_loader._current_machine()
    expected_flags = _kernel_loader._cpu_flags_digest()
    for kname in _KERNEL_NAMES:
        cpp = native / f"{kname}.cpp"
        so = native / f"lib{kname}.so"
        rec = _read_sidecar_json(so)
        assert rec["version"] == 2
        assert rec["mode"] == "native"
        assert rec["source_hash"] == _kernel_loader._compute_source_hash(cpp)
        assert rec["machine"] == expected_machine
        assert rec["cpu_flags"] == expected_flags
        assert so.stat().st_mtime_ns == _kernel_loader._source_mtime_ns(cpp)
    with mock.patch.object(subprocess, "run") as mock_run:
        for kname in _KERNEL_NAMES:
            resolved = _kernel_loader.resolve_so(str(native / f"{kname}.py"), kname)
            assert resolved == native / f"lib{kname}.so"
    mock_run.assert_not_called()


def test_setup_sync_extensions_inplace_vs_non_inplace_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg_root = _source_pkg_root()
    if pkg_root is None:
        pytest.skip("setup sync test requires a source checkout")
    setup_copy = tmp_path / "setup.py"
    shutil.copy2(pkg_root / "setup.py", setup_copy)
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    shutil.copy2(
        pkg_root / "src" / "stimside" / "_native" / "_kernel_loader.py",
        native / "_kernel_loader.py",
    )
    cpp = native / "tiny_sync.cpp"
    cpp.write_text('extern "C" int tiny_sync_val() { return 1; }\n', encoding="utf-8")

    st_mod = types.ModuleType("setuptools")
    cmd_mod = types.ModuleType("setuptools.command")
    bw_mod = types.ModuleType("setuptools.command.bdist_wheel")
    be_mod = types.ModuleType("setuptools.command.build_ext")

    class _FakeExt:
        def __init__(self, name: str, sources: list[str], **_kw: object) -> None:
            self.name = name
            self.sources = sources

    class _FakeBdistWheel:
        def get_tag(self) -> tuple[str, str, str]:
            return ("cp313", "cp313", "linux_x86_64")

    class _FakeBuildExt:
        pass

    st_mod.Extension = _FakeExt  # type: ignore[attr-defined]
    st_mod.setup = lambda **_kw: None  # type: ignore[attr-defined]
    bw_mod.bdist_wheel = _FakeBdistWheel  # type: ignore[attr-defined]
    be_mod.build_ext = _FakeBuildExt  # type: ignore[attr-defined]
    fake_mods = {
        "setuptools": st_mod,
        "setuptools.command": cmd_mod,
        "setuptools.command.bdist_wheel": bw_mod,
        "setuptools.command.build_ext": be_mod,
    }
    mod_name = "_test_stimside_setup_sync"
    with mock.patch.dict(sys.modules, fake_mods):
        spec = importlib.util.spec_from_file_location(mod_name, setup_copy)
        assert spec is not None and spec.loader is not None
        setup_mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(setup_mod)
        finally:
            sys.modules.pop(mod_name, None)
            sys.modules.pop("_stimside_kernel_loader", None)

    build_lib = tmp_path / "build_lib"
    built_so = build_lib / "stimside" / "_native" / "libtiny_sync.so"
    built_so.parent.mkdir(parents=True)
    built_so.write_bytes(b"fake-so")
    ext = setup_mod.StimsideBuildExt.__new__(setup_mod.StimsideBuildExt)
    ext.build_lib = str(build_lib)
    ext.distribution = types.SimpleNamespace(package_dir={"": "src"})
    ext.extensions = [
        _FakeExt("stimside._native.libtiny_sync", ["src/stimside/_native/tiny_sync.cpp"])
    ]

    monkeypatch.setattr(os, "replace", mock.Mock(side_effect=OSError("disk full")))
    ext.inplace = True
    with pytest.raises(OSError, match="disk full"):
        ext._sync_extensions_to_source_dir()
    ext.inplace = False
    ext._sync_extensions_to_source_dir()


@pytest.mark.parametrize(
    "make_payload",
    [
        lambda h: json.dumps({"version": 3, "source_hash": h, "mode": "portable"}).encode(),
        lambda h: json.dumps({"version": 1, "source_hash": h, "mode": "portable"}).encode(),
        lambda h: json.dumps({"version": 2, "source_hash": h, "mode": "bogus"}).encode(),
        lambda _h: b"[1, 2, 3]",
        lambda h: json.dumps(
            {"version": 2, "source_hash": h, "mode": "native", "machine": 123}
        ).encode(),
        lambda h: json.dumps(
            {"version": 2, "source_hash": h, "mode": "native", "cpu_flags": ["avx2"]}
        ).encode(),
        lambda _h: (
            b"[" * max(20_000, sys.getrecursionlimit() + 500)
            + b"1"
            + b"]" * max(20_000, sys.getrecursionlimit() + 500)
        ),
        lambda _h: (
            b'{"a":' * max(20_000, sys.getrecursionlimit() + 500)
            + b"1"
            + b"}" * max(20_000, sys.getrecursionlimit() + 500)
        ),
    ],
    ids=[
        "version_3",
        "version_1",
        "bad_mode",
        "json_list",
        "bad_machine_type",
        "bad_cpu_flags_type",
        "deep_list_recursion",
        "deep_dict_recursion",
    ],
)
def test_invalid_and_deeply_nested_build_records_adopted_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_payload: object
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_badrec", 'extern "C" int tiny_badrec_val() { return 11; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    good_hash = _kernel_loader._compute_source_hash(cpp)
    sidecar = _kernel_loader._build_id_path(so)
    sidecar.write_bytes(make_payload(good_hash))  # type: ignore[operator]
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()
    adopted = _read_sidecar_json(so)
    assert adopted == {"version": 2, "mode": "unknown", "source_hash": good_hash}


def test_non_checkout_layouts_and_dist_info_never_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)

    layouts = [
        ("no_setup", False, None),
        ("site-packages/repo", True, None),
        ("dist-packages/repo", True, None),
        ("target_proj", True, "stimside-0.1.0.dist-info"),
    ]
    for rel_root, create_setup, meta_dir in layouts:
        root = tmp_path / rel_root
        native = root / "src" / "stimside" / "_native"
        native.mkdir(parents=True)
        if create_setup:
            (root / "setup.py").write_text("# setup\n", encoding="utf-8")
        if meta_dir is not None:
            (root / "src" / meta_dir).mkdir()
        cpp = native / "tiny_nc.cpp"
        so = native / "libtiny_nc.so"
        cpp.write_text('extern "C" int tiny_nc_val() { return 1; }\n', encoding="utf-8")
        subprocess.run([*cxx, "-shared", "-fPIC", str(cpp), "-o", str(so)], check=True)
        cpp.write_text('extern "C" int tiny_nc_val() { return 2; }\n', encoding="utf-8")
        newer_ns = so.stat().st_mtime_ns + 5_000_000_000
        os.utime(cpp, ns=(newer_ns, newer_ns))
        resolve = functools.partial(
            _kernel_loader.resolve_so, str(native / "tiny_nc.py"), "tiny_nc"
        )
        with mock.patch.object(subprocess, "run") as mock_run:
            assert resolve() == so
            monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
            assert resolve() == so
            monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
        mock_run.assert_not_called()
        assert not _kernel_loader._build_id_path(so).exists()

    # Positive control: editable checkout with .egg-info in src/ still rebuilds as a checkout.
    ed_root = tmp_path / "editable_proj"
    ed_native = ed_root / "src" / "stimside" / "_native"
    ed_native.mkdir(parents=True)
    (ed_root / "setup.py").write_text("# setup\n", encoding="utf-8")
    (ed_root / "src" / "stimside.egg-info").mkdir()
    ed_cpp = ed_native / "tiny_ed.cpp"
    ed_so = ed_native / "libtiny_ed.so"
    ed_cpp.write_text('extern "C" int tiny_ed_val() { return 1; }\n', encoding="utf-8")
    subprocess.run([*cxx, "-shared", "-fPIC", str(ed_cpp), "-o", str(ed_so)], check=True)
    ed_cpp.write_text('extern "C" int tiny_ed_val() { return 2; }\n', encoding="utf-8")
    newer_ns = ed_so.stat().st_mtime_ns + 5_000_000_000
    os.utime(ed_cpp, ns=(newer_ns, newer_ns))
    ed_res = functools.partial(_kernel_loader.resolve_so, str(ed_native / "tiny_ed.py"), "tiny_ed")
    assert ed_res() == ed_so
    assert _eval_symbol_in_subprocess(ed_so, "tiny_ed_val") == 2
    assert _read_sidecar_json(ed_so)["mode"] == "portable"


def test_installed_forced_recompile_ignores_build_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxx = _require_cxx()
    site = tmp_path / "site-packages" / "stimside" / "_native"
    site.mkdir(parents=True)
    cpp = site / "tiny_inst.cpp"
    so = site / "libtiny_inst.so"
    cpp.write_text('extern "C" int tiny_inst_val() { return 4; }\n', encoding="utf-8")
    subprocess.run([*cxx, "-shared", "-fPIC", str(cpp), "-o", str(so)], check=True)
    _kernel_loader._build_id_path(so).write_text(
        json.dumps({"version": 2, "source_hash": "0" * 64, "mode": "native"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    cmds: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[object]:
        cmds.append(list(cmd))
        return real_run(cmd, *a, **kw)  # type: ignore[arg-type]

    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        rebuilt = _kernel_loader.resolve_so(
            str(site / "tiny_inst.py"), "tiny_inst", force_recompile=True
        )
    assert rebuilt == so
    assert len(cmds) == 1
    assert "-march=native" not in cmds[0]


def test_adoption_write_oserror_is_swallowed_and_never_replaces_valid_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, cpp, resolve = _make_source_native(
        tmp_path, "tiny_adopt_safe", 'extern "C" int tiny_adopt_safe_val() { return 6; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    sidecar = _kernel_loader._build_id_path(so)
    src_hash = _kernel_loader._compute_source_hash(cpp)

    # Adoption write OSError (e.g. .build_id is a directory) must not fail resolve_so.
    sidecar.unlink()
    sidecar.mkdir()
    assert resolve() == so
    assert list(native.glob("*.tmp.*")) == []
    sidecar.rmdir()

    # Existing valid v2 native record must never be overwritten by _adopt_build_record.
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    resolve()
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    assert _read_sidecar_json(so)["mode"] == "native"
    _kernel_loader._adopt_build_record(so, src_hash)
    assert _read_sidecar_json(so)["mode"] == "native"
    assert list(native.glob("*.tmp.*")) == []

    # Legacy v1 plain-hex record is still upgraded by _adopt_build_record.
    sidecar.write_text("a" * 64 + "\n", encoding="utf-8")
    _kernel_loader._adopt_build_record(so, src_hash)
    assert _read_sidecar_json(so)["mode"] == "unknown"

    # Simulated race: unlocked check sees missing record, concurrent native builder writes record.
    sidecar.unlink()
    orig_read = _kernel_loader._read_build_record
    calls = 0

    def racing_read(path: Path) -> _kernel_loader._BuildRecord | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            _kernel_loader._write_build_record(so, src_hash, "native")
            return None
        return orig_read(path)

    monkeypatch.setattr(_kernel_loader, "_read_build_record", racing_read)
    assert resolve() == so
    assert calls == 2
    assert _read_sidecar_json(so)["mode"] == "native"


def test_failed_compile_removes_temp_so_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_cleanup", 'extern "C" int tiny_cleanup_val() { return 1; }\n'
    )
    failing_cxx = tmp_path / "fail_cxx.sh"
    failing_cxx.write_text(
        '#!/bin/sh\nout=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'printf "partial" > "$out"\nexit 1\n',
        encoding="utf-8",
    )
    failing_cxx.chmod(0o755)
    monkeypatch.setenv("CXX", str(failing_cxx))
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    with pytest.raises(RuntimeError, match="Failed to compile"):
        resolve()
    assert list(native.glob("*.tmp.*")) == []


def test_lock_timeout_default_600s_env_override_and_warning_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    assert _kernel_loader._LOCK_MAX_RETRIES == 12000
    assert _kernel_loader._LOCK_MAX_RETRIES * _kernel_loader._LOCK_RETRY_DELAY_S == pytest.approx(
        600.0
    )
    monkeypatch.delenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", raising=False)
    assert _kernel_loader._lock_max_retries() == 12000

    native, cpp, resolve = _make_source_native(
        tmp_path, "tiny_env_to", 'extern "C" int tiny_env_to_val() { return 8; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)

    # Invalid timeout env values emit RuntimeWarning pointing at caller and fall back to default.
    for bad_val in ("-1", "-inf", "abc", "nan", "1e308"):
        monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", bad_val)
        with warnings.catch_warnings(record=True) as caught_bad:
            warnings.simplefilter("always", RuntimeWarning)
            assert _kernel_loader._lock_max_retries() == 12000
        assert len(caught_bad) == 1 and caught_bad[0].filename == __file__
        assert issubclass(caught_bad[0].category, RuntimeWarning)

    # Zero timeout breaks immediately without sleeping.
    monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "0")
    sleeps = 0

    def count_sleep(_s: float) -> None:
        nonlocal sleeps
        sleeps += 1

    monkeypatch.setattr(time, "sleep", count_sleep)
    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with warnings.catch_warnings(record=True) as caught_zero:
            warnings.simplefilter("always", RuntimeWarning)
            assert _kernel_loader._acquire_dir_lock(native) is None
        assert sleeps == 0
        assert len(caught_zero) == 1

        # Positive short timeout via env overrides the 600s default.
        monkeypatch.undo()
        monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "0.05")
        monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
        monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
        with warnings.catch_warnings(record=True) as caught_short:
            warnings.simplefilter("always", RuntimeWarning)
            so = resolve()
        assert len(caught_short) == 1
        assert math.isclose(_kernel_loader._LOCK_RETRY_DELAY_S, 0.05)
        assert _eval_symbol_in_subprocess(so, "tiny_env_to_val") == 8
    finally:
        os.close(holder_fd)

    # PYTHONWARNINGS=ignore suppresses failed-rebuild warnings in a fresh subprocess.
    cpp.write_text("#error intentional_filter_check\n", encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))
    src_root = str(Path(stimside.__file__).resolve().parents[1])
    code = (
        "from stimside._native import _kernel_loader\n"
        f"_kernel_loader.resolve_so({str(native / 'tiny_env_to.py')!r}, 'tiny_env_to')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": src_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONWARNINGS": "ignore",
    }
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True, text=True, env=env, timeout=30, check=False,
    )
    assert proc.returncode == 0
    assert "stimside warning:" not in proc.stderr


def test_native_build_records_and_checks_machine_and_cpu_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    _, _, resolve = _make_source_native(
        tmp_path, "tiny_cpu", 'extern "C" int tiny_cpu_val() { return 12; }\n'
    )
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    rec = _read_sidecar_json(so)
    assert rec["mode"] == "native"
    assert rec["machine"] == _kernel_loader._current_machine()
    assert rec["cpu_flags"] == _kernel_loader._cpu_flags_digest()

    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    with mock.patch.object(subprocess, "run") as mock_run:
        assert resolve() == so
    mock_run.assert_not_called()

    # Changing CPU flags triggers a native rebuild even with env vars unset.
    real_run = subprocess.run
    cmds: list[list[str]] = []

    def spy_run(cmd: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[object]:
        cmds.append(list(cmd))
        return real_run(cmd, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(_kernel_loader, "_cpu_flags_digest", lambda: "deadbeef01234567")
    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        assert resolve() == so
    assert len(cmds) == 1
    if _kernel_loader._arch_supports_march_native():
        assert "-march=native" in cmds[0]
    rec2 = _read_sidecar_json(so)
    assert rec2["mode"] == "native" and rec2["cpu_flags"] == "deadbeef01234567"

    # Changing machine architecture also triggers a native rebuild.
    other_arch = "aarch64" if _kernel_loader._current_machine() != "aarch64" else "x86_64"
    monkeypatch.setattr(platform, "machine", lambda: other_arch)
    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        assert resolve() == so
    assert len(cmds) == 2
    assert "-march=native" in cmds[1]
    assert _read_sidecar_json(so)["machine"] == other_arch

    # Unit check of _cpu_flags_digest order-independence and missing-file handling.
    monkeypatch.undo()
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: "flags : sse2 avx2\n")
    d1 = _kernel_loader._cpu_flags_digest()
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: "Features : avx2 sse2 sse2\n")
    assert _kernel_loader._cpu_flags_digest() == d1 and len(d1) == 16
    monkeypatch.setattr(Path, "read_text", mock.Mock(side_effect=OSError("no procfs")))
    assert _kernel_loader._cpu_flags_digest() == ""


def test_lock_inf_timeout_waits_for_release_and_notice_fires_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    for inf_val in ("inf", "+inf", "infinity", "INF"):
        monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", inf_val)
        with warnings.catch_warnings(record=True) as caught_inf:
            warnings.simplefilter("always", RuntimeWarning)
            assert _kernel_loader._lock_max_retries() is None
        assert caught_inf == []
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_inf", 'extern "C" int tiny_inf_val() { return 19; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "inf")
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.01)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 0.02)

    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    fcntl.flock(holder_fd, fcntl.LOCK_EX)

    def _release_later() -> None:
        time.sleep(0.06)
        os.close(holder_fd)

    releaser = threading.Thread(target=_release_later)
    releaser.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            so = resolve()
    finally:
        releaser.join()

    assert caught == []
    err_lines = [
        line for line in capsys.readouterr().err.splitlines() if "waiting for build lock" in line
    ]
    assert len(err_lines) == 1
    assert str(native) in err_lines[0]
    assert "STIMSIDE_BUILD_LOCK_TIMEOUT_S" in err_lines[0]
    assert f"uid {os.getuid()}" in err_lines[0]
    assert _eval_symbol_in_subprocess(so, "tiny_inf_val") == 19


def test_lock_wait_notice_and_shared_deadline_across_four_libs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    native, _, resolve0 = _make_source_native(
        tmp_path, "tiny_sh0", 'extern "C" int tiny_sh0_val() { return 10; }\n'
    )
    resolvers = [resolve0]
    for i in range(1, 4):
        name = f"tiny_sh{i}"
        (native / f"{name}.cpp").write_text(
            f'extern "C" int {name}_val() {{ return {10 + i}; }}\n', encoding="utf-8"
        )
        mod = native / f"{name}.py"
        mod.write_text("# stub\n", encoding="utf-8")
        resolvers.append(functools.partial(_kernel_loader.resolve_so, str(mod), name))

    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "0.05")
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.01)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 0.02)

    real_sleep = time.sleep
    sleeps = 0

    def spy_sleep(s: float) -> None:
        nonlocal sleeps
        sleeps += 1
        real_sleep(s)

    monkeypatch.setattr(time, "sleep", spy_sleep)
    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with warnings.catch_warnings(record=True) as caught0:
            warnings.simplefilter("always", RuntimeWarning)
            so0 = resolvers[0]()
        assert sleeps >= 1
        assert len(caught0) == 1 and caught0[0].filename == __file__
        err0 = capsys.readouterr().err
        assert err0.count("waiting for build lock") == 1
        assert "STIMSIDE_BUILD_LOCK_TIMEOUT_S" in err0

        for idx, res in enumerate(resolvers[1:], start=1):
            sleeps = 0
            with warnings.catch_warnings(record=True) as caught_i:
                warnings.simplefilter("always", RuntimeWarning)
                so_i = res()
            assert sleeps == 0, f"lib {idx} should reuse exhausted process deadline"
            assert len(caught_i) == 1 and caught_i[0].filename == __file__
            assert capsys.readouterr().err == ""
            assert _eval_symbol_in_subprocess(so_i, f"tiny_sh{idx}_val") == 10 + idx
    finally:
        os.close(holder_fd)
    assert _eval_symbol_in_subprocess(so0, "tiny_sh0_val") == 10


def test_adoption_does_not_wait_for_held_dir_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_lockless_adopt", 'extern "C" int tiny_la_val() { return 23; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    _kernel_loader._build_id_path(so).unlink()

    sleeps = 0

    def spy_sleep(_s: float) -> None:
        nonlocal sleeps
        sleeps += 1

    monkeypatch.setattr(time, "sleep", spy_sleep)
    monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "0.05")
    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with (
            mock.patch.object(subprocess, "run") as mock_run,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always", RuntimeWarning)
            assert resolve() == so
    finally:
        os.close(holder_fd)

    mock_run.assert_not_called()
    assert sleeps == 0
    assert caught == []
    assert _read_sidecar_json(so)["mode"] == "unknown"


def test_cpu_flags_digest_filters_to_isa_tokens_and_intersects_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = "flags : sse2 avx2 fma\n"
    noisy = "flags : sse2 avx2 fma hypervisor ibpb ibrs ssbd stibp tsc_known_freq apic smep\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: base)
    d_base = _kernel_loader._cpu_flags_digest()
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: noisy)
    assert _kernel_loader._cpu_flags_digest() == d_base and len(d_base) == 16

    with_avx512 = "flags : sse2 avx2 fma avx512f\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: with_avx512)
    d_avx512 = _kernel_loader._cpu_flags_digest()
    assert d_avx512 != d_base and len(d_avx512) == 16

    hybrid = (
        "processor : 0\nflags : sse2 avx2 fma avx512f hypervisor\n\n"
        "processor : 1\nflags : sse2 avx2 fma ibpb\n"
    )
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: hybrid)
    assert _kernel_loader._cpu_flags_digest() == d_base

    arm_clean = "Features : fp asimd crc32 sve2\n"
    arm_noisy = "Features : fp asimd crc32 sve2 evtstrm cpuid\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: arm_clean)
    d_arm = _kernel_loader._cpu_flags_digest()
    monkeypatch.setattr(Path, "read_text", lambda self, **_kw: arm_noisy)
    assert _kernel_loader._cpu_flags_digest() == d_arm and len(d_arm) == 16


def test_readonly_checkout_native_foreign_cpu_warns_once_and_werror_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses read-only directory mode checks")
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_ro_cpu", 'extern "C" int tiny_ro_cpu_val() { return 31; }\n'
    )
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    bid = _kernel_loader._build_id_path(so)
    rec = _read_sidecar_json(so)
    rec["cpu_flags"] = "0000000000000000"
    bid.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")

    native.chmod(0o555)
    try:
        with mock.patch.object(subprocess, "run") as mock_run:
            # Under -W error, both call 1 and call 2 must raise (once-only flag not set yet).
            for _ in range(2):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    with pytest.raises(RuntimeWarning, match="different CPU"):
                        resolve()
            # Under normal warnings, emits once with caller filename, then stays silent.
            with warnings.catch_warnings(record=True) as caught1:
                warnings.simplefilter("always", RuntimeWarning)
                assert resolve() == so
            assert len(caught1) == 1
            assert caught1[0].filename == __file__
            assert "different CPU" in str(caught1[0].message)
            with warnings.catch_warnings(record=True) as caught2:
                warnings.simplefilter("always", RuntimeWarning)
                assert resolve() == so
            assert caught2 == []
        mock_run.assert_not_called()
    finally:
        native.chmod(0o755)

    # Matching native or portable build in read-only checkout emits no warning.
    monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1")
    resolve()
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD")
    native.chmod(0o555)
    try:
        with warnings.catch_warnings(record=True) as caught_ok:
            warnings.simplefilter("always", RuntimeWarning)
            assert resolve() == so
        assert caught_ok == []
    finally:
        native.chmod(0o755)


def test_failed_rebuild_warning_reraises_under_werror_and_points_at_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_werr", 'extern "C" int tiny_werr_val() { return 4; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    cpp.write_text("#error intentional_werror_check\n", encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))

    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="failed to rebuild"):
                resolve()

    with warnings.catch_warnings(record=True) as caught1:
        warnings.simplefilter("always", RuntimeWarning)
        assert resolve() == so
    assert len(caught1) == 1
    assert caught1[0].filename == __file__

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always", RuntimeWarning)
        assert resolve() == so
    assert caught2 == []


def test_stale_temp_files_removed_under_lock_keeping_fresh_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_stale", 'extern "C" int tiny_stale_val() { return 42; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", raising=False)

    old_t = time.time() - 1200.0
    old_so = native / ".libtiny_stale.111.aaaaaa.tmp.so"
    old_bid = native / ".libtiny_stale.111.aaaaaa.tmp.build_id"
    fresh_so = native / ".libtiny_stale.222.bbbbbb.tmp.so"
    other_so = native / ".libother.111.cccccc.tmp.so"
    target = tmp_path / "outside.txt"
    link_so = native / ".libtiny_stale.333.dddddd.tmp.so"
    for p in (old_so, old_bid, fresh_so, other_so, target):
        p.write_text("x", encoding="utf-8")
    for p in (old_so, old_bid, other_so, target):
        os.utime(p, (old_t, old_t))
    link_so.symlink_to(target)
    try:
        os.utime(link_so, (old_t, old_t), follow_symlinks=False)
    except (NotImplementedError, OSError):
        pass

    so = resolve()
    assert not old_so.exists()
    assert not old_bid.exists()
    assert fresh_so.exists()
    assert other_so.exists()
    assert link_so.is_symlink() and target.exists()
    assert _eval_symbol_in_subprocess(so, "tiny_stale_val") == 42


def test_adopt_exdev_fallback_preserves_valid_native_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    native, cpp, resolve = _make_source_native(
        tmp_path, "tiny_exdev", 'extern "C" int tiny_exdev_val() { return 5; }\n'
    )
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    src_hash = _kernel_loader._compute_source_hash(cpp)

    def fail_exdev(_src: object, _dst: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", fail_exdev)
    _kernel_loader._adopt_build_record(so, src_hash)
    assert _read_sidecar_json(so)["mode"] == "native"
    assert list(native.glob("*.tmp.*")) == []

    _kernel_loader._build_id_path(so).unlink()
    _kernel_loader._adopt_build_record(so, src_hash)
    assert _read_sidecar_json(so)["mode"] == "unknown"
    assert list(native.glob("*.tmp.*")) == []


class _BusyThenFreeFcntl:
    """Stands in for ``fcntl``: the lock is busy for the first three calls."""

    LOCK_EX, LOCK_NB = 2, 4

    def __init__(self) -> None:
        self.calls = 0

    def flock(self, _fd: int, _flags: int) -> None:
        self.calls += 1
        if self.calls <= 3:
            raise OSError(errno.EAGAIN, "lock is busy")


class _FailingStream:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.writes = 0

    def write(self, _text: str) -> int:
        self.writes += 1
        raise self.exc

    def flush(self) -> None:
        raise self.exc


def _unusable_stream(tmp_path: Path, kind: str) -> object:
    closed = (tmp_path / "closed_stderr.txt").open("w", encoding="utf-8")
    closed.close()
    broken = _FailingStream(BrokenPipeError("stderr pipe is gone"))
    return {"none": None, "closed": closed, "broken_pipe": broken}[kind]


def _wait_for_busy_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: object
) -> _BusyThenFreeFcntl:
    """Wait for a lock that is busy three times, with ``stream`` as stderr and the notice due."""
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.001)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 0.0)
    monkeypatch.setattr(_kernel_loader, "_LOCK_WAIT_SPENT_S", {})
    lock = _BusyThenFreeFcntl()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", stream)
        assert _kernel_loader._wait_for_flock(0, tmp_path, 0, math.inf, None, lock) is True
    return lock


@pytest.mark.parametrize("stream_kind", ["none", "closed"])
def test_lock_notice_is_best_effort_when_stderr_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream_kind: str
) -> None:
    lock = _wait_for_busy_lock(tmp_path, monkeypatch, _unusable_stream(tmp_path, stream_kind))
    assert lock.calls == 4  # the wait went on past the notice and took the lock


def test_lock_notice_is_tried_once_when_stderr_is_a_broken_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _FailingStream(BrokenPipeError("stderr pipe is gone"))
    lock = _wait_for_busy_lock(tmp_path, monkeypatch, stream)
    assert (lock.calls, stream.writes) == (4, 1)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, RuntimeError])
def test_lock_notice_hides_only_stream_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc_type: type[BaseException]
) -> None:
    """An interrupt, or an exception raised by a signal handler, still ends the wait."""
    with pytest.raises(exc_type):
        _wait_for_busy_lock(tmp_path, monkeypatch, _FailingStream(exc_type("stop")))


@pytest.mark.parametrize("stream_kind", ["none", "closed", "broken_pipe"])
def test_build_record_warning_is_best_effort_when_stderr_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream_kind: str
) -> None:
    so_path = tmp_path / "libtiny.so"
    monkeypatch.setattr(_kernel_loader, "_WARNED_FAILED_BUILD_IDS", set())
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", _unusable_stream(tmp_path, stream_kind))
        _kernel_loader._warn_failed_build_id_once(so_path, OSError("read-only"))
    assert so_path in _kernel_loader._WARNED_FAILED_BUILD_IDS


def test_lock_wait_budget_cleared_in_forked_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("os.fork not available")
    key = tmp_path.resolve()
    monkeypatch.setitem(_kernel_loader._LOCK_WAIT_SPENT_S, key, 999.0)
    pid = os.fork()
    if pid == 0:
        os._exit(0 if _kernel_loader._LOCK_WAIT_SPENT_S == {} else 1)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert _kernel_loader._LOCK_WAIT_SPENT_S[key] == 999.0


def test_lock_wait_budget_reset_after_compile_and_distinct_exhausted_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    native, _, resolve0 = _make_source_native(
        tmp_path, "tiny_rst0", 'extern "C" int tiny_rst0_val() { return 1; }\n'
    )
    (native / "tiny_rst1.cpp").write_text(
        'extern "C" int tiny_rst1_val() { return 2; }\n', encoding="utf-8"
    )
    mod1 = native / "tiny_rst1.py"
    mod1.write_text("# stub\n", encoding="utf-8")
    resolve1 = functools.partial(_kernel_loader.resolve_so, str(mod1), "tiny_rst1")

    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setenv("STIMSIDE_BUILD_LOCK_TIMEOUT_S", "0.03")
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.01)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 10.0)
    monkeypatch.setattr(_kernel_loader, "_LOCK_WAIT_SPENT_S", {})

    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with warnings.catch_warnings(record=True) as w0:
            warnings.simplefilter("always", RuntimeWarning)
            resolve0()
        assert len(w0) == 1 and "timed out waiting for build lock" in str(w0[0].message)
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always", RuntimeWarning)
            resolve1()
        assert len(w1) == 1
        assert "budget already exhausted" in str(w1[0].message)
        assert "timed out waiting for build lock" not in str(w1[0].message)
    finally:
        os.close(holder_fd)

    # Once the lock is free, a successful locked rebuild refills the wait budget.
    monkeypatch.setitem(_kernel_loader._LOCK_WAIT_SPENT_S, native.resolve(), 0.02)
    resolve0(force_recompile=True)
    assert native.resolve() not in _kernel_loader._LOCK_WAIT_SPENT_S


def test_readonly_legacy_native_record_same_cpu_no_warning_and_writable_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cxx()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses read-only directory mode checks")
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_leg_ro", 'extern "C" int tiny_leg_ro_val() { return 7; }\n'
    )
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD")
    rec = _read_sidecar_json(so)
    assert rec["cpu_flags_v"] == 2
    rec.pop("cpu_flags_v")
    rec["cpu_flags"] = "legacy_full_cpuinfo_digest"
    bid = _kernel_loader._build_id_path(so)
    bid.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")

    # Read-only checkout with legacy same-machine native record must not falsely warn.
    native.chmod(0o555)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert resolve() == so
    finally:
        native.chmod(0o755)

    # Different machine on legacy record still warns on read-only checkout.
    rec["machine"] = "other-arch"
    bid.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
    native.chmod(0o555)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="different CPU"):
                resolve()
    finally:
        native.chmod(0o755)

    # Writable checkout upgrades legacy record to cpu_flags_v == 2.
    rec["machine"] = _kernel_loader._current_machine()
    bid.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
    assert resolve() == so
    assert _read_sidecar_json(so)["cpu_flags_v"] == 2


def test_stale_temp_cleanup_respects_age_floor_and_pid_liveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    so = tmp_path / "libtiny_floor.so"
    now = time.time()
    dead_recent = tmp_path / ".libtiny_floor.999999.aaaaaa.tmp.so"
    live_old = tmp_path / f".libtiny_floor.{os.getpid()}.bbbbbb.tmp.so"
    dead_old = tmp_path / ".libtiny_floor.999999.cccccc.tmp.so"
    dead_mid = tmp_path / ".libtiny_floor.999999.dddddd.tmp.so"
    for p in (dead_recent, live_old, dead_old, dead_mid):
        p.write_text("x", encoding="utf-8")
    os.utime(dead_recent, (now - 300.0, now - 300.0))
    os.utime(live_old, (now - 1200.0, now - 1200.0))
    os.utime(dead_old, (now - 1200.0, now - 1200.0))
    os.utime(dead_mid, (now - 900.0, now - 900.0))

    # User timeout smaller than 600 s floor (or 0.0) still keeps 300 s file and live PID.
    _kernel_loader._cleanup_stale_temp_files(so, 1.0)
    assert dead_recent.exists()
    assert live_old.exists()
    assert not dead_old.exists()
    assert not dead_mid.exists()

    # Larger user timeout raises the cutoff above 600 s; 0.0 falls back to default 600 s floor.
    dead_mid.write_text("x", encoding="utf-8")
    os.utime(dead_mid, (now - 900.0, now - 900.0))
    _kernel_loader._cleanup_stale_temp_files(so, 1200.0)
    assert dead_mid.exists()
    _kernel_loader._cleanup_stale_temp_files(so, 0.0)
    assert not dead_mid.exists()

    # PermissionError from os.kill distinguishes same-user synthetic PIDs from other users.
    st_mine = live_old.lstat()
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: (_ for _ in ()).throw(PermissionError()))
    assert _kernel_loader._pid_is_alive(dead_recent.name, st_mine) is False
    fake_other = os.stat_result(
        tuple(999_999 if i == 4 else v for i, v in enumerate(st_mine))
    )
    assert _kernel_loader._pid_is_alive(dead_recent.name, fake_other) is True


def test_adopt_fallback_lock_and_exdev_invalid_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fcntl = pytest.importorskip("fcntl")
    so = tmp_path / "libtiny_adopt_lock.so"
    so.write_bytes(b"fake")
    bid = _kernel_loader._build_id_path(so)
    bid.write_text("legacy-invalid\n", encoding="utf-8")

    # While directory lock is held, unlocked adoption skips replacement instead of clobbering.
    holder_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        _kernel_loader._adopt_build_record(so, "h123")
        assert bid.read_text(encoding="utf-8") == "legacy-invalid\n"
        # Caller already holding lock (locked=True) upgrades without self-conflict.
        _kernel_loader._adopt_build_record(so, "h123", locked=True)
        assert _read_sidecar_json(so)["mode"] == "unknown"
    finally:
        os.close(holder_fd)

    # EXDEV on an existing invalid sidecar still upgrades when unlocked and lock is free.
    bid.write_text("legacy-invalid\n", encoding="utf-8")
    monkeypatch.setattr(
        os, "link", lambda _s, _d: (_ for _ in ()).throw(OSError(errno.EXDEV, "exdev"))
    )
    _kernel_loader._adopt_build_record(so, "h456")
    assert _read_sidecar_json(so)["mode"] == "unknown"


def test_lock_notice_wording_and_timeout_warning_closed_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.001)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 0.0)
    monkeypatch.setattr(_kernel_loader, "_LOCK_WAIT_SPENT_S", {})
    lock = _BusyThenFreeFcntl()
    assert _kernel_loader._wait_for_flock(0, tmp_path, os.getuid(), math.inf, None, lock) is True
    err = capsys.readouterr().err
    assert "waiting for build lock" in err
    assert "STIMSIDE_BUILD_LOCK_TIMEOUT_S" in err
    assert "shorten the wait" not in err

    # Closed stderr during lock-timeout warning must not raise ValueError.
    holder_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stderr", _unusable_stream(tmp_path, "closed"))
            assert (
                _kernel_loader._acquire_dir_lock(tmp_path, timeout_s=0.0, max_retries=0) is None
            )
    finally:
        os.close(holder_fd)


def test_wait_for_flock_accumulates_spent_budget_and_caps_on_retry_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path.resolve()
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.002)
    monkeypatch.setattr(_kernel_loader, "_LOCK_NOTICE_DELAY_S", 10.0)
    monkeypatch.setattr(_kernel_loader, "_LOCK_WAIT_SPENT_S", {key: 0.05})
    lock = _BusyThenFreeFcntl()
    assert _kernel_loader._wait_for_flock(0, tmp_path, 0, 1.0, 10, lock) is True
    assert _kernel_loader._LOCK_WAIT_SPENT_S[key] > 0.05

    class _AlwaysBusyFcntl:
        LOCK_EX, LOCK_NB = 2, 4

        def flock(self, _fd: int, _flags: int) -> None:
            raise OSError(errno.EAGAIN, "busy")

    monkeypatch.setattr(_kernel_loader, "_LOCK_WAIT_SPENT_S", {})
    assert (
        _kernel_loader._wait_for_flock(0, tmp_path, 0, 1.0, 1, _AlwaysBusyFcntl()) is False
    )
    assert _kernel_loader._LOCK_WAIT_SPENT_S[key] >= 1.0
