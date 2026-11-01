from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
import shutil
import subprocess
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


def test_top_level_stimside_exports() -> None:
    expected = {
        "__version__",
        "BaseDecoder",
        "BranchAndBoundDecoder",
        "CallableDecoder",
        "CliffordFaultChannel",
        "CompiledLeakageUint8Magic",
        "CompiledMagicsideSampler",
        "CosetsideSampler",
        "CosetsideSimulator",
        "EvaluateQueries",
        "FaultBranch",
        "FlipsideSampler",
        "FlipsideSimulator",
        "LeakageDecoder",
        "LeakageUint8Magic",
        "MagicsideSampler",
        "MagicsideSimulator",
        "MarginalDecoder",
        "MarginalLeakageDemGenerator",
        "PrepareQueries",
        "PreparedQueryPlan",
        "QueryBudget",
        "QueryBudgetExceededError",
        "QueryDiagnostics",
        "QueryResult",
        "TablesideSampler",
        "TablesideSimulator",
        "evaluate_queries",
        "prepare_queries",
    }
    assert len(expected) == 29
    assert len(stimside.__all__) == len(set(stimside.__all__))
    assert set(stimside.__all__) == expected
    for sym in expected:
        assert hasattr(stimside, sym), f"Missing top-level export {sym}"
    from stimside.op_handlers.leakage_handlers import leakage_uint8_magic
    from stimside import sampler_magic, simulator_magic
    from stimside._magic import compile as magic_compile, query as magic_query

    assert stimside.MagicsideSimulator is simulator_magic.MagicsideSimulator
    assert stimside.MagicsideSampler is sampler_magic.MagicsideSampler
    assert stimside.CompiledMagicsideSampler is sampler_magic.CompiledMagicsideSampler
    assert stimside.LeakageUint8Magic is leakage_uint8_magic.LeakageUint8Magic
    assert stimside.CompiledLeakageUint8Magic is leakage_uint8_magic.CompiledLeakageUint8Magic
    assert stimside.CliffordFaultChannel is magic_compile.CliffordFaultChannel
    assert stimside.FaultBranch is magic_compile.FaultBranch
    assert stimside.PrepareQueries is magic_query.PrepareQueries
    assert stimside.EvaluateQueries is magic_query.EvaluateQueries
    assert stimside.prepare_queries is magic_query.prepare_queries
    assert stimside.evaluate_queries is magic_query.evaluate_queries
    assert stimside.QueryBudget is magic_query.QueryBudget
    assert stimside.QueryBudgetExceededError is magic_query.QueryBudgetExceededError
    assert stimside.PreparedQueryPlan is magic_query.PreparedQueryPlan
    assert stimside.QueryResult is magic_query.QueryResult
    assert stimside.QueryDiagnostics is magic_query.QueryDiagnostics


@KERNELS
def test_prebuilt_kernels_load_without_runtime_subprocess(mod, name) -> None:
    # Ensure kernels are present (either from pip install / python_build_stimside or fallback build),
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


@KERNELS
def test_source_tree_discovers_prebuilt_so_from_sys_path_without_subprocess(
    mod, name, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_site_native = tmp_path / "installed_site" / "stimside" / "_native"
    fake_site_native.mkdir(parents=True)
    shutil.copy2(_resolve(mod, name), fake_site_native / f"lib{name}.so")
    fake_src_native = tmp_path / "repo" / "src" / "stimside" / "_native"
    fake_src_native.mkdir(parents=True)
    (tmp_path / "repo" / "setup.py").write_text("# dummy\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "installed_site"))
    monkeypatch.setattr(mod, "__file__", str(fake_src_native / f"{name}.py"))
    with _forbid_subprocess("subprocess.run should not be called when prebuilt .so is on sys.path"):
        assert _resolve(mod, name).exists()


@KERNELS
def test_missing_so_and_cpp_raises_clear_file_not_found(
    mod, name, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_native = tmp_path / "empty_native"
    empty_native.mkdir(parents=True)
    monkeypatch.setattr(mod, "_find_prebuilt_so_candidate", lambda *a, **kw: None)
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
    t = cpp.stat().st_mtime + 10
    os.utime(so, (t, t))
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

    # 1. find_prebuilt_so must ignore both hidden .lib*.tmp.so and anything in tempfile.gettempdir().
    fake_tmp = tmp_path / "fake_sys_tmp"
    fake_tmp.mkdir()
    (fake_tmp / "libtiny2.so").write_bytes(b"bad")
    (native / ".libtiny2.999.tmp.so").write_bytes(b"partial")
    monkeypatch.setenv("TMPDIR", str(fake_tmp))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(fake_tmp))
    assert _kernel_loader.find_prebuilt_so(native, "libtiny2", cpp) is None

    # 2. Compilation writes to a dot-prefixed temp file invisible to find_prebuilt_so.
    seen_out: list[str] = []
    real_run = subprocess.run

    def spy_run(cmd, *a, **kw):
        out_name = Path(cmd[cmd.index("-o") + 1]).name
        seen_out.append(out_name)
        return real_run(cmd, *a, **kw)

    with mock.patch.object(subprocess, "run", side_effect=spy_run):
        so = _kernel_loader.resolve_so(str(native / "tiny2.py"), "tiny2", force_recompile=True)
    assert so == native / "libtiny2.so"
    assert seen_out and all(n.startswith(".libtiny2.") for n in seen_out)
    assert list(native.glob("*lock*")) == []

    # 3. Unwritable package dir with missing .so raises instead of falling back to /tmp.
    so.unlink()
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    with _forbid_subprocess("must not compile into /tmp when package dir is read-only"):
        with pytest.raises(PermissionError, match="pip install"):
            _kernel_loader.resolve_so(str(native / "tiny2.py"), "tiny2")


def test_coset_kernels_exports_and_checks_abi_sentinel() -> None:
    coset_kernels._LIB_INSTANCE = None
    lib = coset_kernels.get_coset_kernels_lib()
    assert lib.coset_kernels_abi_v1() == 1


def test_packaging_single_version_source_and_no_sdist_so_globs() -> None:
    root = Path(stimside.__file__).resolve().parents[2]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    setup_src = (root / "setup.py").read_text(encoding="utf-8")
    assert "*.so" not in pyproject
    assert "*.so" not in setup_src
    assert "global-exclude *.so" in manifest
    assert 'version = {attr = "stimside.__version__"}' in pyproject
    assert "__version__" not in setup_src
    assert 'version = "' not in pyproject
