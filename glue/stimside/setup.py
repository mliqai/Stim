# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_kernel_loader():
    loader_path = PROJECT_ROOT / "src" / "stimside" / "_native" / "_kernel_loader.py"
    spec = importlib.util.spec_from_file_location("_stimside_kernel_loader", loader_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_LOADER = _load_kernel_loader()


class StimsideBuildExt(build_ext):
    """Builds C++ ctypes shared libraries (.so) without Python ABI suffixes."""

    def get_ext_filename(self, ext_name: str) -> str:
        parts = ext_name.split(".")
        return os.path.join(*parts) + ".so"

    def get_export_symbols(self, ext: Extension) -> list[str]:
        # Pure C++ extern "C" shared libraries loaded via ctypes do not define PyInit_<name>.
        return []

    def _capture_source_mtimes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ext in self.extensions:
            mtimes = [
                _KERNEL_LOADER._source_mtime_ns(PROJECT_ROOT / src)
                for src in ext.sources
                if (PROJECT_ROOT / src).exists()
            ]
            if mtimes:
                out[ext.name] = max(mtimes)
        return out

    def _capture_build_ids(self) -> dict[str, tuple[str, str]]:
        mode = "native" if _KERNEL_LOADER._requested_build_mode() == "native" else "portable"
        out: dict[str, tuple[str, str]] = {}
        for ext in self.extensions:
            if not ext.sources:
                continue
            cpp_path = PROJECT_ROOT / ext.sources[0]
            if cpp_path.exists():
                out[ext.name] = (_KERNEL_LOADER._compute_source_hash(cpp_path), mode)
        return out

    def run(self) -> None:
        # The in-tree build_base persists across builds and its up-to-date check ignores
        # compile-flag changes (STIMSIDE_NATIVE_BUILD / STIMSIDE_PORTABLE_BUILD), so force.
        # Build release wheels from the sdist: persistent build_base can retain deleted files.
        self.force = True
        src_mtimes = self._capture_source_mtimes()
        build_records = self._capture_build_ids()
        super().run()
        self._sync_extensions_to_source_dir(src_mtimes, build_records)

    def copy_extensions_to_source(self) -> None:
        self._sync_extensions_to_source_dir()

    def _sync_extensions_to_source_dir(
        self,
        src_mtimes: dict[str, int] | None = None,
        build_records: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        package_dir_map = self.distribution.package_dir or {"": "src"}
        root_src = package_dir_map.get("", "src")
        captured = src_mtimes if src_mtimes is not None else self._capture_source_mtimes()
        captured_records = (
            build_records if build_records is not None else self._capture_build_ids()
        )
        for ext in self.extensions:
            rel_ext_path = self.get_ext_filename(ext.name)
            built_path = (Path(self.build_lib) / rel_ext_path).resolve()
            if not built_path.exists():
                continue
            src_mtime_ns = captured.get(ext.name)
            if src_mtime_ns is not None:
                os.utime(built_path, ns=(src_mtime_ns, src_mtime_ns))
            dest_path = (PROJECT_ROOT / root_src / rel_ext_path).resolve()
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if built_path != dest_path:
                    tmp_dest = dest_path.with_name(f".{dest_path.name}.{os.getpid()}.tmp")
                    shutil.copy2(built_path, tmp_dest)
                    if src_mtime_ns is not None:
                        os.utime(tmp_dest, ns=(src_mtime_ns, src_mtime_ns))
                    os.replace(tmp_dest, dest_path)
                record = captured_records.get(ext.name)
                if record is not None:
                    _KERNEL_LOADER._write_build_record(dest_path, record[0], record[1])
            except OSError:
                if self.inplace:
                    raise


class StimsideBdistWheel(bdist_wheel):
    """Tag wheels as py3-none-<platform> (ctypes C++ libraries use no Python C-API)."""

    def get_tag(self) -> tuple[str, str, str]:
        _, _, plat = super().get_tag()
        return "py3", "none", plat


_EXTENSIONS = (
    ("stimside._native.libcoset_kernels", "src/stimside/_native/coset_kernels.cpp"),
    ("stimside._native.libtableside_kernels", "src/stimside/_native/tableside_kernels.cpp"),
    ("stimside._native.libmarginal_dem_kernels", "src/stimside/_native/marginal_dem_kernels.cpp"),
    ("stimside._native.libmagic_kernels", "src/stimside/_native/magic_kernels.cpp"),
)


def _get_extensions() -> list[Extension]:
    compile_args, link_args = _KERNEL_LOADER.get_cpp_compile_and_link_args()
    return [
        Extension(
            name=ext_name,
            sources=[cpp_src],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
        for ext_name, cpp_src in _EXTENSIONS
    ]


setup(
    ext_modules=_get_extensions(),
    cmdclass={"build_ext": StimsideBuildExt, "bdist_wheel": StimsideBdistWheel},
    options={"build": {"build_base": "python_build_stimside"}},
)
