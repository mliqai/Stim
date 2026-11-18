"""Packaging, D-39/D-42 build flags, loader policy, build-ID sidecar, and C-8 lock tests."""

from __future__ import annotations

import errno
import functools
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
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

    # Also build a wheel directly from the dirty copy holding all 4 .so and .build_id files.
    _run_uv_build_offline(uv_bin, ["--wheel", "--out-dir", str(dist_direct), str(pkg_copy)], env)
    direct_wheels = list(dist_direct.glob("*.whl"))
    assert len(direct_wheels) == 1
    _assert_wheel_ok(direct_wheels[0], plat)


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cxx()
    _, cpp, resolve = _make_source_native(
        tmp_path, "tiny_warn", 'extern "C" int tiny_warn_val() { return 3; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    so = resolve()
    capsys.readouterr()

    cpp.write_text("#error intentional_b2_3_failure\n", encoding="utf-8")
    newer_ns = so.stat().st_mtime_ns + 5_000_000_000
    os.utime(cpp, ns=(newer_ns, newer_ns))

    assert resolve() == so
    first_err = capsys.readouterr().err
    assert "stimside warning: failed to rebuild libtiny_warn.so" in first_err
    assert "using existing library" in first_err

    assert resolve() == so
    second_err = capsys.readouterr().err
    assert "stimside warning: failed to rebuild libtiny_warn.so" not in second_err


def test_b2_4_bounded_flock_wait_warns_and_builds_lockless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cxx()
    fcntl = pytest.importorskip("fcntl")
    native, _, resolve = _make_source_native(
        tmp_path, "tiny_to", 'extern "C" int tiny_to_val() { return 7; }\n'
    )
    monkeypatch.delenv("STIMSIDE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("STIMSIDE_PORTABLE_BUILD", raising=False)
    monkeypatch.setattr(_kernel_loader, "_LOCK_MAX_RETRIES", 2)
    monkeypatch.setattr(_kernel_loader, "_LOCK_RETRY_DELAY_S", 0.01)

    holder_fd = os.open(native, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        capsys.readouterr()
        so = resolve()
        err = capsys.readouterr().err
    finally:
        os.close(holder_fd)

    assert err.count("stimside warning: timed out waiting for build lock") == 1
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    monkeypatch.setattr(_kernel_loader, "_LOCK_MAX_RETRIES", 1)
    monkeypatch.setattr(os, "close", fail_close)
    capsys.readouterr()
    assert _kernel_loader._acquire_dir_lock(native) is None
    err = capsys.readouterr().err
    assert "timed out waiting for build lock" in err
    assert len(closed_calls) == 1
