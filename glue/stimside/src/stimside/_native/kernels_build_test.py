from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

import stimside
from stimside._native import (
    _kernel_loader,
    coset_kernels,
    magic_kernels,
    marginal_dem_kernels,
    tableside_kernels,
)


@pytest.fixture(autouse=True)
def _isolate_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_tmp = tmp_path / "_lock_tmp"
    lock_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(lock_tmp))


KERNELS = pytest.mark.parametrize(
    "mod,name",
    [
        (coset_kernels, "coset_kernels"),
        (tableside_kernels, "tableside_kernels"),
        (marginal_dem_kernels, "marginal_dem_kernels"),
        (magic_kernels, "magic_kernels"),
    ],
    ids=["coset", "tableside", "marginal_dem", "magic"],
)


def _resolve(mod, name: str) -> Path:
    return getattr(mod, f"_resolve_{name}_so")()


def _forbid_subprocess(msg: str):
    return mock.patch.object(subprocess, "run", side_effect=AssertionError(msg))


@KERNELS
def test_prebuilt_kernels_load_without_runtime_subprocess(mod, name) -> None:
    # Ensure kernels are present (either from pip install or fallback build),
    # then verify that loading from disk never invokes subprocess.run.
    _resolve(mod, name)
    mod._LIB_INSTANCE = None
    with _forbid_subprocess("subprocess.run should not be called when .so is already built"):
        assert getattr(mod, f"get_{name}_lib")() is not None


@KERNELS
def test_wheel_install_resolution_ignores_cpp_mtime_and_missing_cpp(
    mod, name, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pkg = tmp_path / "site-packages" / "stimside" / "_native"
    fake_pkg.mkdir(parents=True)
    shutil.copy2(_resolve(mod, name), fake_pkg / f"lib{name}.so")
    monkeypatch.setattr(mod, "__file__", str(fake_pkg / f"{name}.py"))
    with _forbid_subprocess("subprocess.run should not be called in site-packages"):
        assert _resolve(mod, name) == fake_pkg / f"lib{name}.so"


def test_source_tree_ignores_foreign_so_on_sys_path_and_compiles_own_cpp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign_site_native = tmp_path / "installed_site" / "stimside" / "_native"
    foreign_site_native.mkdir(parents=True)
    foreign_cpp = tmp_path / "foreign.cpp"
    foreign_cpp.write_text(
        'extern "C" int tiny_foreign_abi_v1() { return 1; }\n'
        'extern "C" int foreign_build_marker() { return 42; }\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["c++", "-std=c++17", "-O2", "-fPIC", "-shared", str(foreign_cpp), "-o", str(foreign_site_native / "libtiny_foreign.so")],
        check=True,
    )

    fake_src_native = tmp_path / "repo" / "src" / "stimside" / "_native"
    fake_src_native.mkdir(parents=True)
    (tmp_path / "repo" / "setup.py").write_text("# dummy\n", encoding="utf-8")
    local_cpp = fake_src_native / "tiny_foreign.cpp"
    local_cpp.write_text('extern "C" int tiny_foreign_abi_v1() { return 1; }\n', encoding="utf-8")
    older_ns = (foreign_site_native / "libtiny_foreign.so").stat().st_mtime_ns - 5_000_000_000
    os.utime(local_cpp, ns=(older_ns, older_ns))

    monkeypatch.syspath_prepend(str(tmp_path / "installed_site"))
    so = _kernel_loader.resolve_so(str(fake_src_native / "tiny_foreign.py"), "tiny_foreign")
    assert so == fake_src_native / "libtiny_foreign.so"
    lib = ctypes.CDLL(str(so))
    assert lib.tiny_foreign_abi_v1() == 1
    assert not hasattr(lib, "foreign_build_marker")


@KERNELS
def test_missing_so_and_cpp_raises_clear_file_not_found(
    mod, name, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_native = tmp_path / "empty_native"
    empty_native.mkdir(parents=True)
    monkeypatch.setattr(mod, "__file__", str(empty_native / f"{name}.py"))
    with pytest.raises(FileNotFoundError, match="pip install"):
        _resolve(mod, name)


def _stale_tiny_library(tmp_path: Path) -> tuple[functools.partial[Path], Path]:
    # A prebuilt library from an older layout, not older than the .cpp (so only the sentinel
    # can tell).
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / "tiny.cpp"
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / "tiny.py"), "tiny")
    cpp.write_text('extern "C" int tiny_abi_v1() { return 1; }\n', encoding="utf-8")
    so = resolve(force_recompile=True)
    cpp.write_text('extern "C" int tiny_abi_v2() { return 2; }\n', encoding="utf-8")
    t_ns = cpp.stat().st_mtime_ns + 10_000_000_000
    os.utime(so, ns=(t_ns, t_ns))
    return resolve, so


def test_library_without_the_sentinel_is_rebuilt_and_used(tmp_path: Path) -> None:
    # The rebuilt library must be returned, not the stale one already loaded.
    resolve, _ = _stale_tiny_library(tmp_path)
    assert _kernel_loader.load_cdll(resolve, "tiny_abi_v2").tiny_abi_v2() == 2


def test_stale_library_that_stays_loaded_raises_instead_of_being_used(tmp_path: Path) -> None:
    resolve, so = _stale_tiny_library(tmp_path)
    held = ctypes.CDLL(str(so))  # another handle, so the stale library cannot be unloaded
    with pytest.raises(RuntimeError, match="restart Python"):
        _kernel_loader.load_cdll(resolve, "tiny_abi_v2")
    assert not hasattr(held, "tiny_abi_v2")


def test_loader_uses_hidden_tmp_so_and_never_uses_system_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / "tiny2.cpp"
    cpp.write_text('extern "C" int tiny2_abi_v1() { return 1; }\n', encoding="utf-8")

    # 1. resolve_so must ignore both hidden .lib*.tmp.so and anything in tempfile.gettempdir().
    fake_tmp = tmp_path / "fake_sys_tmp"
    fake_tmp.mkdir()
    (fake_tmp / "libtiny2.so").write_bytes(b"bad")
    (native / ".libtiny2.999.tmp.so").write_bytes(b"partial")
    monkeypatch.setenv("TMPDIR", str(fake_tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(fake_tmp))

    # 2. Compilation writes to a dot-prefixed temp file and never leaves lock files in package dir.
    seen_out: list[str] = []
    real_run = subprocess.run

    def spy_run(cmd, *a, **kw):
        out_name = Path(cmd[cmd.index("-o") + 1]).name
        seen_out.append(out_name)
        return real_run(cmd, *a, **kw)

    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        so = _kernel_loader.resolve_so(str(native / "tiny2.py"), "tiny2")
    assert so == native / "libtiny2.so"
    assert seen_out and all(n.startswith(".libtiny2.") for n in seen_out)
    assert list(native.glob("*lock*")) == []

    # 3. Unwritable package dir with missing .so raises instead of falling back to /tmp.
    so.unlink()
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    with _forbid_subprocess("must not compile into /tmp when package dir is read-only"):
        with pytest.raises(PermissionError, match="pip install"):
            _kernel_loader.resolve_so(str(native / "tiny2.py"), "tiny2")


def test_cxx_wrapper_edit_race_and_corrupt_so_error_chaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / "tiny_race.cpp"
    cpp.write_text('extern "C" int tiny_race_abi_v1() { return 1; }\n', encoding="utf-8")

    # 1. CXX containing compiler + flags parses cleanly via shlex.split, and an edit saved while
    # the compiler is running leaves .cpp strictly newer than the stamped .so so the next call rebuilds.
    monkeypatch.setenv("CXX", "c++ -w")
    real_run = subprocess.run

    def edit_during_compile(cmd, *a, **kw):
        res = real_run(cmd, *a, **kw)
        old_ns = cpp.stat().st_mtime_ns
        cpp.write_text(
            'extern "C" int tiny_race_abi_v1() { return 1; }\n'
            'extern "C" int tiny_race_edited() { return 9; }\n',
            encoding="utf-8",
        )
        new_ns = max(cpp.stat().st_mtime_ns, old_ns + 1_000_000)
        os.utime(cpp, ns=(new_ns, new_ns))
        return res

    with mock.patch.object(subprocess, "run", side_effect=edit_during_compile):
        so1 = _kernel_loader.resolve_so(str(native / "tiny_race.py"), "tiny_race")
    assert so1.stat().st_mtime_ns < cpp.stat().st_mtime_ns

    so2 = _kernel_loader.resolve_so(str(native / "tiny_race.py"), "tiny_race")
    assert so2.stat().st_mtime_ns == cpp.stat().st_mtime_ns
    assert ctypes.CDLL(str(so2)).tiny_race_edited() == 9
    with _forbid_subprocess("second call after clean stamp must not recompile"):
        assert _kernel_loader.resolve_so(str(native / "tiny_race.py"), "tiny_race") == so2

    # 2. Corrupt .so in read-only site-packages chains the initial dlopen OSError onto PermissionError.
    ro_pkg = tmp_path / "site-packages" / "stimside" / "_native"
    ro_pkg.mkdir(parents=True)
    (ro_pkg / "corrupt.cpp").write_text('extern "C" int corrupt_abi_v1() { return 1; }\n', encoding="utf-8")
    (ro_pkg / "libcorrupt.so").write_bytes(b"not an ELF file")
    ro_resolve = functools.partial(_kernel_loader.resolve_so, str(ro_pkg / "corrupt.py"), "corrupt")
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    with pytest.raises(PermissionError) as exc_info:
        _kernel_loader.load_cdll(ro_resolve, "corrupt_abi_v1")
    assert isinstance(exc_info.value.__cause__, OSError)


def test_coset_kernels_exports_and_checks_abi_sentinel() -> None:
    coset_kernels._LIB_INSTANCE = None
    lib = coset_kernels.get_coset_kernels_lib()
    assert lib.coset_kernels_abi_v2() == 2


def test_packaging_single_version_source_and_no_sdist_so_globs() -> None:
    root = Path(stimside.__file__).resolve().parents[2]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    setup_src = (root / "setup.py").read_text(encoding="utf-8")
    assert (root / "LICENSE").is_file()
    assert "include LICENSE" in manifest
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject
    assert '"setuptools>=77"' in pyproject
    assert "self.force = True" in setup_src  # rebuild whenever compile flags may differ
    assert '"numpy>=2"' in pyproject
    assert '"stim>=1.16"' in pyproject
    assert '"sinter>=1.16"' in pyproject
    assert "*.so" not in pyproject
    assert "*.so" not in setup_src
    assert "global-exclude *.so" in manifest
    assert 'version = {attr = "stimside.__version__"}' in pyproject
    assert "__version__" not in setup_src
    assert 'version = "' not in pyproject
    assert "golden_digests.json" in manifest
    assert "golden_digests.json" in pyproject
    assert "*.h" not in manifest and "*.h" not in pyproject


def test_resolve_so_tolerates_flock_ebadf_and_portable_build_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1")
    c_args, _ = _kernel_loader.get_cpp_compile_and_link_args(include_march_native=True)
    assert "-march=native" not in c_args

    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    (native / "tiny_nfsv4.cpp").write_text('extern "C" int tiny_nfsv4_abi_v1() { return 7; }\n', encoding="utf-8")

    fcntl = pytest.importorskip("fcntl")  # the loader imports it lazily when compiling
    monkeypatch.setattr(
        fcntl,
        "flock",
        mock.Mock(side_effect=OSError(errno.EBADF, "Bad file descriptor on NFSv4")),
    )
    so = _kernel_loader.resolve_so(str(native / "tiny_nfsv4.py"), "tiny_nfsv4", force_recompile=True)
    assert ctypes.CDLL(str(so)).tiny_nfsv4_abi_v1() == 7
    assert list(native.glob("*lock*")) == []


@pytest.mark.parametrize("foreign_owner", [False, True])
def test_resolve_so_only_waits_on_its_own_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_owner: bool
) -> None:
    fcntl = pytest.importorskip("fcntl")
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    (native / "tiny_lock.cpp").write_text('extern "C" int tiny_lock_abi_v1() { return 5; }\n', encoding="utf-8")
    flock = mock.Mock()
    monkeypatch.setattr(fcntl, "flock", flock)
    if foreign_owner:  # The lock file we create then looks like another user's.
        real_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    so = _kernel_loader.resolve_so(str(native / "tiny_lock.py"), "tiny_lock", force_recompile=True)
    assert ctypes.CDLL(str(so)).tiny_lock_abi_v1() == 5
    assert flock.called is not foreign_owner


def test_load_cdll_keeps_compiler_error_when_rebuild_also_fails(tmp_path: Path) -> None:
    bogus = tmp_path / "libbogus.so"
    bogus.write_bytes(b"not a shared library")
    compiler_err = FileNotFoundError(2, "No such file or directory", "/nonexistent/c++")

    def resolve(force_recompile: bool = False) -> Path:
        if not force_recompile:
            return bogus
        raise RuntimeError("Failed to compile bogus.cpp") from compiler_err

    with pytest.raises(RuntimeError, match="initial load error") as info:
        _kernel_loader.load_cdll(resolve)
    assert info.value.__cause__ is not None
    assert info.value.__cause__.__cause__ is compiler_err


def test_import_stimside_without_fcntl() -> None:
    import sys

    src_dir = str(Path(stimside.__file__).resolve().parents[1])
    code = (
        "import sys; sys.modules['fcntl'] = None\n"
        "import stimside, stimside._native._kernel_loader\n"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": src_dir, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


STALE = pytest.mark.parametrize(
    "mod,name,missing",
    [
        (coset_kernels, "coset_kernels", "engine_peek_pauli_batch"),
        (tableside_kernels, "tableside_kernels", "fast_format_stim_op"),
        (marginal_dem_kernels, "marginal_dem_kernels", "marginal_free_batch_result"),
        (magic_kernels, "magic_kernels", "magic_measure_batch"),
    ],
    ids=["coset", "tableside", "marginal_dem", "magic"],
)
_FIRST_USE = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "from stimside._native import {name}\n"
    "{name}.get_{name}_lib()\n"
    "print('OK')\n"
)


def _stale_site_packages(
    tmp_path: Path, mod, name: str, missing: str, *, rebuildable: bool, writable: bool
) -> Path:
    """A package copy outside any source tree whose lib<name>.so lacks the symbol `missing`.

    The stub library exports the sentinel and every `_SPECS` name except `missing`, like a
    library built before `missing` was added. A rebuildable copy also ships a stub `<name>.cpp`
    that exports everything, so the rebuild takes well under a second.
    """
    pkg_src = Path(stimside.__file__).parent
    site = tmp_path / "site-packages"  # no setup.py above it, so never "in a source tree"
    ignore = shutil.ignore_patterns("__pycache__", "*.so", "*_test.py", "*.cpp", "*.hpp")
    shutil.copytree(pkg_src, site / "stimside", ignore=ignore)
    native = site / "stimside" / "_native"
    real_cpp = (pkg_src / "_native" / f"{name}.cpp").read_text(encoding="utf-8")
    exported = set(re.findall(r"\b(\w+_abi_v\d+)\(", real_cpp)) | {s[0] for s in mod._SPECS}
    assert missing in exported

    def stub(names: set[str]) -> str:
        body = "".join(f"int {n}() {{ return 0; }}\n" for n in sorted(names))
        return 'extern "C" {\n' + body + "}"

    (tmp_path / "stale.cpp").write_text(stub(exported - {missing}), encoding="utf-8")
    so = native / f"lib{name}.so"
    stale_cpp = tmp_path / "stale.cpp"
    subprocess.run(["c++", "-shared", "-fPIC", str(stale_cpp), "-o", str(so)], check=True)
    if rebuildable:
        (native / f"{name}.cpp").write_text(stub(exported), encoding="utf-8")
    if not writable:
        native.chmod(0o555)
        if os.access(native, os.W_OK):  # root ignores permission bits: nothing is read-only
            native.chmod(0o755)
            pytest.skip("cannot make a directory read-only for this user (running as root?)")
    return site


def _first_use(site: Path, name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "STIMSIDE_PORTABLE_BUILD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env["TMPDIR"] = str(tmp_path)
    code = _FIRST_USE.format(name=name)
    if name == "magic_kernels":
        code += "magic_kernels.get_magic_measure_batch()\n"  # the symbol the stale library lacks
    cmd = [sys.executable, "-c", code, str(site)]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600, check=False)


@STALE
def test_stale_installed_library_is_rebuilt_in_place_when_source_is_present(
    mod, name: str, missing: str, tmp_path: Path
) -> None:
    site = _stale_site_packages(tmp_path, mod, name, missing, rebuildable=True, writable=True)
    out = _first_use(site, name, tmp_path)
    assert out.returncode == 0 and out.stdout.split()[0] == "OK", out.stderr


@STALE
@pytest.mark.parametrize("rebuildable", [False, True], ids=["no_cpp", "read_only"])
def test_stale_installed_library_that_cannot_be_rebuilt_raises_a_clear_error(
    mod, name: str, missing: str, rebuildable: bool, tmp_path: Path
) -> None:
    site = _stale_site_packages(
        tmp_path, mod, name, missing, rebuildable=rebuildable, writable=not rebuildable
    )
    try:
        err = _first_use(site, name, tmp_path)
    finally:  # pytest must be able to delete its tmp dirs
        (site / "stimside" / "_native").chmod(0o755)
    assert err.returncode != 0 and "AttributeError" not in err.stderr
    assert "pip install" in err.stderr


def test_load_cdll_rebuilds_when_only_a_later_name_of_a_sentinel_sequence_is_missing(
    tmp_path: Path,
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    cpp = native / "tiny2.cpp"
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / "tiny2.py"), "tiny2")
    cpp.write_text('extern "C" int tiny2_a() { return 1; }\n', encoding="utf-8")
    so = resolve(force_recompile=True)
    cpp.write_text(  # source gained tiny2_b; the .so stays stamped newer, so only names tell
        'extern "C" int tiny2_a() { return 1; }\nextern "C" int tiny2_b() { return 2; }\n',
        encoding="utf-8",
    )
    t_ns = cpp.stat().st_mtime_ns + 10_000_000_000
    os.utime(so, ns=(t_ns, t_ns))
    lib = _kernel_loader.load_cdll(resolve, ("tiny2_a", "tiny2_b"))
    assert lib.tiny2_a() == 1 and lib.tiny2_b() == 2


def test_load_cdll_error_names_every_name_missing_after_rebuild(tmp_path: Path) -> None:
    resolve, _ = _stale_tiny_library(tmp_path)
    with pytest.raises(RuntimeError, match="tiny_nope, tiny_nada"):
        _kernel_loader.load_cdll(resolve, ("tiny_abi_v2", "tiny_nope", "tiny_nada"))


@pytest.mark.parametrize("portable", [False, True])
def test_compile_flags_disable_fma_contraction(
    monkeypatch: pytest.MonkeyPatch, portable: bool
) -> None:
    # The exact tracer reference and native/portable agreement need no FMA contraction.
    monkeypatch.setenv("STIMSIDE_PORTABLE_BUILD", "1" if portable else "0")
    for native in (True, False):
        c_args, _ = _kernel_loader.get_cpp_compile_and_link_args(include_march_native=native)
        assert "-ffp-contract=off" in c_args


def test_header_edit_alone_makes_the_library_stale_and_a_rebuild_stamps_the_newest_input(
    tmp_path: Path,
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    (native / "tiny_h.cpp").write_text(
        '#include "tiny_h.hpp"\nextern "C" int tiny_h_abi_v1() { return TINY_VALUE; }\n',
        encoding="utf-8",
    )
    hpp = native / "tiny_h.hpp"
    hpp.write_text("#define TINY_VALUE 1\n", encoding="utf-8")
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / "tiny_h.py"), "tiny_h")
    so = resolve(force_recompile=True)
    with _forbid_subprocess("an unchanged source tree must not recompile"):
        resolve()
    before = so.read_bytes()
    hpp.write_text("#define TINY_VALUE 2222\n", encoding="utf-8")
    t_ns = so.stat().st_mtime_ns + 10_000_000_000
    os.utime(hpp, ns=(t_ns, t_ns))  # only the header is newer than the library
    resolve()
    assert so.read_bytes() != before and so.stat().st_mtime_ns == t_ns
    with _forbid_subprocess("the rebuilt library is stamped with the header's mtime"):
        resolve()


def test_dangling_header_symlink_beside_the_source_does_not_break_resolve_or_rebuild(
    tmp_path: Path,
) -> None:
    native = tmp_path / "src" / "stimside" / "_native"
    native.mkdir(parents=True)
    (tmp_path / "setup.py").write_text("# dummy\n", encoding="utf-8")
    (native / "tiny_d.cpp").write_text('extern "C" int tiny_d_abi_v1() { return 7; }\n')
    resolve = functools.partial(_kernel_loader.resolve_so, str(native / "tiny_d.py"), "tiny_d")
    so = resolve(force_recompile=True)
    (native / ".#tiny_d.hpp").symlink_to("user@host.1234")  # an Emacs lock file: dangling
    with _forbid_subprocess("an up-to-date tree with a dangling header link must not recompile"):
        assert resolve() == so
    t_ns = so.stat().st_mtime_ns + 10_000_000_000
    os.utime(native / "tiny_d.cpp", ns=(t_ns, t_ns))
    assert ctypes.CDLL(str(resolve())).tiny_d_abi_v1() == 7  # a stale source still rebuilds
    assert so.stat().st_mtime_ns == t_ns


_LOAD_ALL_KERNELS = (
    "from stimside._native import coset_kernels, magic_kernels\n"
    "from stimside._native import marginal_dem_kernels, tableside_kernels\n"
    "coset_kernels.get_coset_kernels_lib()\n"
    "magic_kernels.get_magic_kernels_lib()\n"
    "marginal_dem_kernels.get_marginal_dem_kernels_lib()\n"
    "tableside_kernels.get_tableside_kernels_lib()\n"
)


def _link_to_dir(native: Path, tmp_path: Path) -> None:
    (native / "x.hpp").symlink_to(".")


def _link_to_tmp_dir(native: Path, tmp_path: Path) -> None:
    (native / "x.hpp").symlink_to(tmp_path)


def _link_to_dangling(native: Path, tmp_path: Path) -> None:
    (native / "x.hpp").symlink_to(tmp_path / "does_not_exist")


def _link_to_procfs_file(native: Path, tmp_path: Path) -> None:
    if not Path("/proc/self/stat").exists():
        pytest.skip("no procfs")
    (native / "x.hpp").symlink_to("/proc/self/stat")  # its mtime is always "now"


def _link_to_newer_regular_file(native: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.h"
    target.write_text("#pragma once\n", encoding="utf-8")
    link = native / "x.hpp"
    link.symlink_to(target)
    _make_newer_than(target, native, follow_symlinks=True)
    _make_newer_than(link, native, follow_symlinks=False)


def _newer_directory_named_hpp(native: Path, tmp_path: Path) -> None:
    (native / "x.hpp").mkdir()
    _make_newer_than(native / "x.hpp", native, follow_symlinks=True)


def _make_newer_than(path: Path, native: Path, *, follow_symlinks: bool) -> None:
    t_ns = max(p.stat().st_mtime_ns for p in native.glob("lib*.so")) + 10_000_000_000
    os.utime(path, ns=(t_ns, t_ns), follow_symlinks=follow_symlinks)


@pytest.mark.parametrize(
    "make_junk",
    [
        _link_to_dir,
        _link_to_tmp_dir,
        _link_to_dangling,
        _link_to_procfs_file,
        _link_to_newer_regular_file,
        _newer_directory_named_hpp,
    ],
    ids=["to_native_dir", "to_tmp_dir", "dangling", "to_procfs", "to_newer_file", "newer_dir"],
)
def test_header_lookalikes_that_are_not_regular_files_never_trigger_a_rebuild(
    make_junk, tmp_path: Path
) -> None:
    """A `*.hpp` symlink or directory must not count as a header, in any fresh process."""
    pkg_src = Path(stimside.__file__).parent
    src = tmp_path / "tree" / "src" / "stimside"
    shutil.copytree(pkg_src, src, ignore=shutil.ignore_patterns("__pycache__", "*_test.py"))
    (tmp_path / "tree" / "setup.py").write_text("# dummy\n", encoding="utf-8")
    native, compiles = src / "_native", tmp_path / "compiles.log"
    wrapper = tmp_path / "cxx.sh"
    wrapper.write_text(f'#!/bin/sh\necho x >> "{compiles}"\nexec c++ "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    (tmp_path / "tmp").mkdir()
    env = {**os.environ, "CXX": str(wrapper), "PYTHONPATH": str(src.parent)}
    env.update(TMPDIR=str(tmp_path / "tmp"), PYTHONDONTWRITEBYTECODE="1")

    def load_in_fresh_process() -> int:
        before = compiles.read_text(encoding="utf-8").count("x") if compiles.exists() else 0
        cmd = [sys.executable, "-c", _LOAD_ALL_KERNELS]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, check=False)
        assert out.returncode == 0, out.stderr
        return compiles.read_text(encoding="utf-8").count("x") - before if compiles.exists() else 0

    load_in_fresh_process()  # may compile if a prebuilt library is stale: not under test
    make_junk(native, tmp_path)
    assert [load_in_fresh_process() for _ in range(3)] == [0, 0, 0]
