# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tests for the build_ext.py template.

The template reads ``solution.json`` from the working directory and extracts
compile options via the Solution model.  Tests here exercise the template by
exec-ing it in a controlled tmp directory with ``torch.utils.cpp_extension``
and ``sol_execbench.core`` mocked out.
"""

import ast
import json
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "sol_execbench"
    / "driver"
    / "templates"
    / "build_ext.py"
)


def _make_solution_json(compile_options: dict | None = None) -> str:
    """Create a minimal valid solution JSON string with given compile_options."""
    solution = {
        "name": "test_solution",
        "definition": "test_def",
        "author": "test",
        "spec": {
            "languages": ["hip_cpp"],
            "target_hardware": ["LOCAL"],
            "entry_point": "main.hip::run",
            "destination_passing_style": True,
            "compile_options": compile_options,
        },
        "sources": [{"path": "main.hip", "content": "// test source"}],
    }
    return json.dumps(solution)


def _exec_build_ext(
    cwd: Path,
    compile_options: dict | None = None,
) -> MagicMock:
    """Write solution.json and execute the build_ext template in *cwd* with ext.load mocked.

    Returns the ``ext.load`` mock so callers can inspect how it was called.
    """
    # Write solution.json
    (cwd / "solution.json").write_text(_make_solution_json(compile_options))

    script = _TEMPLATE_PATH.read_text()

    # Stub torch.utils.cpp_extension so we don't need a real torch install
    mock_ext = MagicMock()
    fake_torch: Any = types.ModuleType("torch")
    fake_torch_utils: Any = types.ModuleType("torch.utils")
    fake_torch_utils_cpp: Any = types.ModuleType("torch.utils.cpp_extension")
    fake_torch_utils_cpp.load = mock_ext.load
    fake_torch.utils = fake_torch_utils
    fake_torch_utils.cpp_extension = fake_torch_utils_cpp

    saved_modules = {}
    for mod_name in ("torch", "torch.utils", "torch.utils.cpp_extension"):
        saved_modules[mod_name] = sys.modules.get(mod_name)

    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_torch_utils
    sys.modules["torch.utils.cpp_extension"] = fake_torch_utils_cpp

    old_cwd = Path.cwd()
    previous_cxx = os.environ.get("CXX")
    try:
        os.chdir(cwd)
        exec(compile(script, "build_ext.py", "exec"), {"__builtins__": __builtins__})
    finally:
        os.chdir(old_cwd)
        if previous_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = previous_cxx
        for mod_name, orig in saved_modules.items():
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig

    return mock_ext


class TestTemplateAST:
    """Verify the raw template is valid Python."""

    def test_raw_template_parses(self):
        source = _TEMPLATE_PATH.read_text()
        tree = ast.parse(source, filename="build_ext.py")
        assert isinstance(tree, ast.Module)
        assert len(tree.body) > 0

    def test_template_has_expected_imports(self):
        source = _TEMPLATE_PATH.read_text()
        tree = ast.parse(source, filename="build_ext.py")
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                import_names.append(node.module)
        assert "os" not in import_names
        assert "sol_execbench.core.platform.runtime" in import_names
        assert "torch.utils.cpp_extension" in import_names
        assert "sol_execbench.core.data.solution" in import_names


class TestSourceDiscovery:
    """Tests for source file collection logic."""

    @pytest.mark.parametrize("suffix", [".hip", ".cpp", ".cc", ".cxx", ".c"])
    def test_collects_supported_extensions(self, tmp_path, suffix):
        (tmp_path / f"kernel{suffix}").write_text("// source")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 1
        assert sources[0].endswith(suffix)

    def test_ignores_non_source_files(self, tmp_path):
        (tmp_path / "kernel.hip").write_text("// source")
        (tmp_path / "legacy.cu").write_text("// ignored")
        (tmp_path / "readme.md").write_text("docs")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "helper.py").write_text("pass")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 1

    def test_collects_multiple_sources(self, tmp_path):
        (tmp_path / "kernel.hip").write_text("// hip")
        (tmp_path / "utils.cpp").write_text("// cpp")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 2

    def test_no_sources_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="No HIP/C\\+\\+ source files"):
            _exec_build_ext(tmp_path)

    def test_cu_files_are_ignored(self, tmp_path):
        (tmp_path / "kernel.cu").write_text("// legacy source")
        with pytest.raises(RuntimeError, match="No HIP/C\\+\\+ source files"):
            _exec_build_ext(tmp_path)


class TestCompileOptions:
    """Tests for compile option extraction."""

    def test_default_hip_cflags(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {})
        hip_cflags = mock.load.call_args.kwargs["extra_cuda_cflags"]
        assert hip_cflags == ["-O3"]

    def test_no_compile_options_gives_empty_flags(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        hip_cflags = mock.load.call_args.kwargs["extra_cuda_cflags"]
        assert hip_cflags == []

    def test_custom_hip_cflags(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"hip_cflags": ["--offload-arch=gfx1200"]})
        hip_cflags = mock.load.call_args.kwargs["extra_cuda_cflags"]
        assert hip_cflags == ["--offload-arch=gfx1200"]

    def test_cflags_default_empty(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["extra_cflags"] == []

    def test_custom_cflags(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"cflags": ["-Wall"]})
        assert mock.load.call_args.kwargs["extra_cflags"] == ["-Wall"]

    def test_custom_ld_flags(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"ld_flags": ["-lrocblas"]})
        assert mock.load.call_args.kwargs["extra_ldflags"] == ["-lrocblas"]

    def test_ld_flags_default_empty(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {})
        assert mock.load.call_args.kwargs["extra_ldflags"] == []

    def test_dangerous_compile_options_rejected_before_extension_load(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")

        with pytest.raises(ValueError, match="runtime linker paths"):
            _exec_build_ext(
                tmp_path,
                {"ld_flags": ["-Wl,-rpath,/tmp/lib"]},
            )


class TestIncludePaths:
    """Tests for extra_include_paths construction."""

    def test_only_staging_directory_included(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        includes = mock.load.call_args.kwargs["extra_include_paths"]
        assert includes == [str(tmp_path)]


class TestRocmRootResolution:
    def test_template_rebases_portable_rocm_flags(self):
        source = _TEMPLATE_PATH.read_text()

        assert "discover_rocm_root" in source
        assert 'ENVIRON.setdefault("CXX"' in source
        assert 'flag == "-I/opt/rocm/include"' in source
        assert 'flag == "-L/opt/rocm/lib"' in source


class TestSoRename:
    """Tests for the .so rename logic after compilation."""

    def test_renames_suffixed_so(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        suffixed = tmp_path / "benchmark_kernel.cpython-312-x86_64-linux-gnu.so"
        suffixed.write_bytes(b"ELF-fake")
        _exec_build_ext(tmp_path)
        assert (tmp_path / "benchmark_kernel.so").exists()
        assert not suffixed.exists()

    def test_already_named_correctly(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"ELF-fake")
        _exec_build_ext(tmp_path)
        assert (tmp_path / "benchmark_kernel.so").exists()

    def test_no_so_raises(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        with pytest.raises(FileNotFoundError, match="benchmark_kernel.so not produced"):
            _exec_build_ext(tmp_path)


class TestExtLoad:
    """Tests for ext.load call arguments."""

    def test_load_called_with_correct_name(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["name"] == "benchmark_kernel"

    def test_build_directory_is_cwd(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["build_directory"] == str(tmp_path)

    def test_verbose_enabled(self, tmp_path):
        (tmp_path / "k.hip").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["verbose"] is True
