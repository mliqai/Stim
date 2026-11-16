"""Packaging, D-39 build flags, loader policy, build-ID sidecar, and C-8 lock tests."""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
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


def _find_uv() -> str | None:
    return shutil.which("uv") or (
        str(Path.home() / ".local" / "bin" / "uv")
        if (Path.home() / ".local" / "bin" / "uv").is_file()
        else None
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
    uv_bin = _find_uv()
    if uv_bin is None:
        pytest.skip("uv binary not available for offline sdist/wheel build")
    pkg_root = Path(stimside.__file__).resolve().parents[2]
    pkg_copy = tmp_path / "pkg"
    ignore = shutil.ignore_patterns(
        "build", "dist", "python_build_stimside", "*.egg-info",
        "__pycache__", ".pytest_cache", "*.so", "*.build_id", ".benchmarks",
    )
    shutil.copytree(pkg_root, pkg_copy, ignore=ignore)
    dist_dir, tmp_dir = tmp_path / "dist", tmp_path / "tmp"
    tmp_dir.mkdir()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_dir)}
    env.pop("STIMSIDE_NATIVE_BUILD", None)
    proc = subprocess.run(
        [uv_bin, "build", "--offline", "--out-dir", str(dist_dir), str(pkg_copy)],
        capture_output=True, text=True, env=env, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
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
    assert not any(n.endswith((".so", ".build_id")) for n in sdist_rel)

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    assert re.search(rf"-py3-none-{re.escape(plat)}\.whl$", wheels[0].name), wheels[0].name
    with zipfile.ZipFile(wheels[0]) as zf:
        wnames = set(zf.namelist())
        wheel_meta = next(n for n in wnames if n.endswith(".dist-info/WHEEL"))
        assert f"Tag: py3-none-{plat}" in zf.read(wheel_meta).decode("utf-8")
    assert "stimside/golden_digests.json" in wnames
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in wnames)
    for name in _CPP_NAMES + _HPP_NAMES + _SO_NAMES:
        assert f"stimside/_native/{name}" in wnames
    assert not any(
        n.endswith((".build_id", ".lock")) or ".stimside_build_" in Path(n).name
        for n in wnames
    )


def test_b2_loader_policy_installed_vs_missing_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    with mock.patch.object(subprocess, "run", side_effect=AssertionError("must not compile")):
        assert coset_kernels._resolve_coset_kernels_so() == so_copy

    stub = tmp_path / "site_stub" / "stimside" / "_native"
    stub.mkdir(parents=True)
    stub_cpp = stub / "tiny_b2.cpp"
    stub_cpp.write_text('extern "C" int tiny_b2_v1() { return 1; }\n', encoding="utf-8")
    subprocess.run(
        ["c++", "-shared", "-fPIC", str(stub_cpp), "-o", str(stub / "libtiny_b2.so")],
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
        ["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(nocpp / "libtiny_nc.so")],
        input='extern "C" int tiny_nc_v1() { return 1; }\n', text=True, check=True,
    )
    resolve_nc = functools.partial(_kernel_loader.resolve_so, str(nocpp / "tiny_nc.py"), "tiny_nc")
    with pytest.raises(FileNotFoundError, match="pip install"):
        _kernel_loader.load_cdll(resolve_nc, ("tiny_nc_v1", "tiny_nc_v2"))

    ro = tmp_path / "site_ro" / "stimside" / "_native"
    ro.mkdir(parents=True)
    subprocess.run(
        ["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(ro / "libtiny_ro.so")],
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
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / "tiny_b3.cpp"
    hpp = native / "tiny_b3.hpp"
    hpp.write_text("#define B3_VAL 10\n", encoding="utf-8")
    cpp.write_text(
        '#include "tiny_b3.hpp"\nextern "C" int tiny_b3_val() { return B3_VAL; }\n',
        encoding="utf-8",
    )
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / "tiny_b3.py"), "tiny_b3")
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    sidecar = native / ".libtiny_b3.build_id"
    assert sidecar.is_file()
    id_portable = sidecar.read_text(encoding="utf-8").strip()

    # Lookalike *.hpp symlink / directory must not trigger rebuild.
    (native / "junk_dir.hpp").mkdir()
    (native / "junk_link.hpp").symlink_to(tmp_path / "missing.hpp")
    with mock.patch.object(subprocess, "run", side_effect=AssertionError("lookalike rebuilt")):
        assert resolve() == so

    # Older-mtime .hpp edit triggers rebuild via build_id mismatch.
    old_ns = so.stat().st_mtime_ns - 5_000_000_000
    hpp.write_text("#define B3_VAL 22\n", encoding="utf-8")
    os.utime(hpp, ns=(old_ns, old_ns))
    so = resolve()
    assert so.stat().st_mtime_ns == max(cpp.stat().st_mtime_ns, old_ns)
    assert _eval_symbol_in_subprocess(so, "tiny_b3_val") == 22

    # Older-mtime .cpp edit also triggers rebuild via build_id mismatch.
    cpp.write_text(
        '#include "tiny_b3.hpp"\nextern "C" int tiny_b3_val() { return B3_VAL + 1; }\n',
        encoding="utf-8",
    )
    os.utime(cpp, ns=(old_ns, old_ns))
    so = resolve()
    assert _eval_symbol_in_subprocess(so, "tiny_b3_val") == 23

    # Toggling STIMSIDE_NATIVE_BUILD=1 updates build_id and rebuilds without touching mtime.
    id_before = sidecar.read_text(encoding="utf-8").strip()
    assert id_before != id_portable
    monkeypatch.setenv("STIMSIDE_NATIVE_BUILD", "1")
    resolve()
    assert sidecar.read_text(encoding="utf-8").strip() != id_before
    assert not any("build_id" in s[0] for s in magic_kernels._SPECS)


def test_b4_zero_lock_files_and_multiprocess_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    (native / "tiny_c8.cpp").write_text('extern "C" int tiny_c8_v1() { return 7; }\n')
    tmp_dir = tmp_path / "sys_tmp"
    tmp_dir.mkdir()
    compiles_dir = tmp_path / "compiles"
    compiles_dir.mkdir()
    wrapper = tmp_path / "cxx.sh"
    wrapper.write_text(
        f'#!/bin/sh\ntouch "{compiles_dir}/c_$$_$(date +%s%N)"\nexec c++ "$@"\n',
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
