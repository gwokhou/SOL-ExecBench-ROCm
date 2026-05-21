# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for sol_execbench.driver.problem_packager — ProblemPackager."""

import ast
import json
from pathlib import Path

import pytest

from sol_execbench.core import (
    BenchmarkConfig,
    Definition,
    Solution,
    Trace,
    Workload,
)
from sol_execbench.driver import problem_packager
from sol_execbench.driver.problem_packager import ProblemPackager, _get_local_gfx

# ── Fixtures ──────────────────────────────────────────────────────────────────

_DEFINITION_DICT = {
    "name": "test_vecadd",
    "axes": {"n": {"type": "const", "value": 256}},
    "inputs": {
        "x": {"shape": ["n"], "dtype": "float32"},
        "y": {"shape": ["n"], "dtype": "float32"},
    },
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": "import torch\ndef run(x, y):\n    return x + y",
}

_WORKLOAD_DICTS = [
    {
        "axes": {},
        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        "uuid": "wkl-0001",
    },
    {
        "axes": {},
        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
        "uuid": "wkl-0002",
    },
]

_PYTHON_SOLUTION_DICT = {
    "name": "vecadd_python",
    "definition": "test_vecadd",
    "author": "test",
    "spec": {
        "languages": ["pytorch"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::run",
    },
    "sources": [{"path": "kernel.py", "content": "def run(x, y):\n    return x + y\n"}],
}

_HIP_SOLUTION_DICT = {
    "name": "vecadd_hip",
    "definition": "test_vecadd",
    "author": "test",
    "spec": {
        "languages": ["hip_cpp"],
        "target_hardware": ["gfx1200"],
        "entry_point": "kernel.hip::vecadd",
    },
    "sources": [
        {"path": "kernel.hip", "content": "// placeholder HIP kernel\n"},
    ],
}


@pytest.fixture
def definition() -> Definition:
    return Definition(**_DEFINITION_DICT)


@pytest.fixture
def workloads() -> list[Workload]:
    return [Workload(**w) for w in _WORKLOAD_DICTS]


@pytest.fixture
def python_solution() -> Solution:
    return Solution(**_PYTHON_SOLUTION_DICT)


@pytest.fixture
def hip_solution() -> Solution:
    return Solution(**_HIP_SOLUTION_DICT)


@pytest.fixture
def config() -> BenchmarkConfig:
    return BenchmarkConfig(lock_clocks=False)


def _make_packager(
    tmp_path: Path,
    definition: Definition,
    workloads: list[Workload],
    solution: Solution,
    config: BenchmarkConfig,
) -> ProblemPackager:
    return ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=tmp_path / "staging",
        keep_output_dir=True,
    )


# ── Init: files written to staging ────────────────────────────────────────────


class TestInit:
    def test_definition_json_written(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        path = pkg.output_dir / "definition.json"
        assert path.exists()
        assert json.loads(path.read_text())["name"] == "test_vecadd"

    def test_workload_jsonl_written(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        path = pkg.output_dir / "workload.jsonl"
        assert path.exists()
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_solution_json_written(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        path = pkg.output_dir / "solution.json"
        assert path.exists()
        assert json.loads(path.read_text())["name"] == "vecadd_python"

    def test_config_json_written(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        path = pkg.output_dir / "config.json"
        assert path.exists()
        parsed = json.loads(path.read_text())
        assert parsed["lock_clocks"] is False

    def test_python_sources_written(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        kernel = pkg.output_dir / "kernel.py"
        assert kernel.exists()
        assert "def run(x, y):" in kernel.read_text()

    def test_hip_sources_written(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        kernel = pkg.output_dir / "kernel.hip"
        assert kernel.exists()
        assert "HIP kernel" in kernel.read_text()


# ── compile() ─────────────────────────────────────────────────────────────────


class TestCompile:
    def test_returns_command_and_artifact_path(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        cmd, artifact_path = pkg.compile()
        assert cmd == ["python", "build_ext.py"]
        assert artifact_path.endswith("benchmark_kernel.so")

    def test_build_ext_staged(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        pkg.compile()
        build_ext = pkg.output_dir / "build_ext.py"
        assert build_ext.exists()
        ast.parse(build_ext.read_text())

    def test_offload_arch_injected_for_gfx1200(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        pkg.compile()
        sol = json.loads((pkg.output_dir / "solution.json").read_text())
        hip_cflags = sol["spec"]["compile_options"]["hip_cflags"]
        assert hip_cflags == ["--offload-arch=gfx1200"]

    def test_offload_arch_not_duplicated_when_explicit(
        self, tmp_path, definition, workloads, config
    ):
        sol_dict = {
            **_HIP_SOLUTION_DICT,
            "spec": {
                **_HIP_SOLUTION_DICT["spec"],
                "compile_options": {"hip_cflags": ["--offload-arch=gfx1200"]},
            },
        }
        solution = Solution(**sol_dict)
        pkg = _make_packager(tmp_path, definition, workloads, solution, config)
        pkg.compile()
        sol = json.loads((pkg.output_dir / "solution.json").read_text())
        hip_cflags = sol["spec"]["compile_options"]["hip_cflags"]
        assert hip_cflags == ["--offload-arch=gfx1200"]

    def test_offload_arch_injected_for_local_target(
        self, tmp_path, definition, workloads, config, monkeypatch
    ):
        sol_dict = {
            **_HIP_SOLUTION_DICT,
            "spec": {
                **_HIP_SOLUTION_DICT["spec"],
                "target_hardware": ["LOCAL"],
            },
        }
        monkeypatch.setattr(problem_packager, "_get_local_gfx", lambda: "gfx1200")
        solution = Solution(**sol_dict)
        pkg = _make_packager(tmp_path, definition, workloads, solution, config)
        pkg.compile()
        sol = json.loads((pkg.output_dir / "solution.json").read_text())
        hip_cflags = sol["spec"]["compile_options"]["hip_cflags"]
        assert hip_cflags == ["--offload-arch=gfx1200"]

    def test_asserts_on_python_solution(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        with pytest.raises(AssertionError, match="HIP/C\\+\\+"):
            pkg.compile()


# ── execute() ─────────────────────────────────────────────────────────────────


class TestExecute:
    def test_returns_command(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        cmd = pkg.execute()
        assert cmd == ["python", "eval_driver.py"]

    def test_eval_driver_staged(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        pkg.execute()
        driver = pkg.output_dir / "eval_driver.py"
        assert driver.exists()
        ast.parse(driver.read_text())

    def test_raises_without_so_for_cpp(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        with pytest.raises(FileNotFoundError, match="benchmark_kernel.so"):
            pkg.execute()

    def test_succeeds_with_so_for_cpp(
        self, tmp_path, definition, workloads, hip_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, hip_solution, config)
        # Simulate a compiled artifact.
        (pkg.output_dir / "benchmark_kernel.so").write_bytes(b"\x00")
        cmd = pkg.execute()
        assert cmd == ["python", "eval_driver.py"]


class TestLocalGfxDetection:
    def test_get_local_gfx_uses_rocm_agent_enumerator(self, monkeypatch):
        def fake_check_output(cmd, **kwargs):
            assert kwargs["text"] is True
            assert cmd == ["rocm_agent_enumerator", "-name"]
            return "gfx000\ngfx1200\n"

        monkeypatch.setattr(
            problem_packager.subprocess, "check_output", fake_check_output
        )
        assert _get_local_gfx() == "gfx1200"

    def test_get_local_gfx_falls_back_to_rocminfo(self, monkeypatch):
        def fake_check_output(cmd, **kwargs):
            if cmd == ["rocm_agent_enumerator", "-name"]:
                raise FileNotFoundError
            assert cmd == ["rocminfo"]
            return "Agent 2\n  Name:                    gfx1200\n"

        monkeypatch.setattr(
            problem_packager.subprocess, "check_output", fake_check_output
        )
        assert _get_local_gfx() == "gfx1200"

    def test_get_local_gfx_returns_none_when_no_target_found(self, monkeypatch):
        def fake_check_output(cmd, **kwargs):
            if cmd == ["rocm_agent_enumerator", "-name"]:
                return "gfx000\n"
            return "Name: CPU\n"

        monkeypatch.setattr(
            problem_packager.subprocess, "check_output", fake_check_output
        )
        assert _get_local_gfx() is None


# ── convert_stdout_to_traces() ────────────────────────────────────────────────


class TestConvertStdoutToTraces:
    def _make_trace_json(self, uuid: str = "wkl-0001") -> str:
        return json.dumps(
            {
                "definition": "test_vecadd",
                "workload": {
                    "axes": {},
                    "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
                    "uuid": uuid,
                },
                "solution": "vecadd_python",
                "evaluation": {
                    "status": "PASSED",
                    "environment": {"hardware": "AMD Instinct MI300X (gfx942)"},
                    "timestamp": "2026-01-01T00:00:00",
                    "correctness": {
                        "max_absolute_error": 0.0,
                        "max_relative_error": 0.0,
                    },
                    "performance": {
                        "latency_ms": 0.1,
                        "reference_latency_ms": 0.2,
                        "speedup_factor": 2.0,
                    },
                },
            }
        )

    def test_parses_single_trace(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        stdout = self._make_trace_json()
        traces = pkg.convert_stdout_to_traces(stdout)
        assert len(traces) == 1
        assert isinstance(traces[0], Trace)
        assert traces[0].definition == "test_vecadd"

    def test_parses_multiple_traces(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        stdout = (
            self._make_trace_json("wkl-0001") + "\n" + self._make_trace_json("wkl-0002")
        )
        traces = pkg.convert_stdout_to_traces(stdout)
        assert len(traces) == 2

    def test_skips_non_json_lines(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        stdout = "some library noise\n" + self._make_trace_json() + "\nmore noise\n"
        traces = pkg.convert_stdout_to_traces(stdout)
        assert len(traces) == 1

    def test_returns_empty_for_no_traces(
        self, tmp_path, definition, workloads, python_solution, config
    ):
        pkg = _make_packager(tmp_path, definition, workloads, python_solution, config)
        traces = pkg.convert_stdout_to_traces("no json here\njust noise\n")
        assert traces == []
