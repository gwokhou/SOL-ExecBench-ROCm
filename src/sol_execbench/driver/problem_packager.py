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

"""Package a problem (definition + workloads + solution) into a staging directory.

Produces shell commands that the CLI can run directly via subprocess to compile
HIP/C++ solutions and evaluate them on the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from pathlib import Path

from ..core import (
    BenchmarkConfig,
    Definition,
    Solution,
    SupportedHardware,
    SupportedLanguages,
    Trace,
    Workload,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_CPP_LANGUAGES = {
    SupportedLanguages.HIP_CPP,
    SupportedLanguages.HIPBLAS,
    SupportedLanguages.MIOPEN,
    SupportedLanguages.CK,
    SupportedLanguages.ROCWMMA,
}


def _first_gfx_target(lines: list[str]) -> str | None:
    """Return the first concrete AMD gfx target from command output lines."""
    for line in lines:
        match = re.search(r"\bgfx[0-9a-fA-F]+\b", line)
        if match and match.group(0) != "gfx000":
            return match.group(0)
    return None


def _get_local_gfx() -> str | None:
    """Detect the local AMD GPU gfx target using ROCm tooling."""
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        target = _first_gfx_target(out.splitlines())
        if target:
            return target
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return _first_gfx_target(out.splitlines())
    except Exception:
        return None


def _gfx_to_offload_arch(gfx: str) -> str:
    """Convert an AMD gfx target string to a HIP offload architecture flag."""
    return f"--offload-arch={gfx}"


class ProblemPackager:
    """Stage files for compilation and execution, returning commands for the CLI."""

    def __init__(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
        config: BenchmarkConfig,
        output_dir: Path,
        keep_output_dir: bool = False,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_output_dir = keep_output_dir
        self._closed = False

        self.definition = definition
        self.workloads = workloads
        self.solution = solution
        self.config = config

        # Write problem files to staging directory up front.
        (self.output_dir / "definition.json").write_text(definition.model_dump_json())
        (self.output_dir / "workload.jsonl").write_text(
            "\n".join(w.model_dump_json() for w in workloads)
        )
        (self.output_dir / "solution.json").write_text(solution.model_dump_json())
        (self.output_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config))
        )
        self._write_sources()

    def __del__(self):
        self.close()

    def __enter__(self) -> "ProblemPackager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the staging directory when this packager owns cleanup."""
        if self._closed:
            return
        self._closed = True
        if not self.keep_output_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)

    @property
    def _is_cpp(self) -> bool:
        return any(lang in _CPP_LANGUAGES for lang in self.solution.spec.languages)

    def _inject_offload_arch_flags(self, sol_dict: dict) -> dict:
        """Auto-inject HIP offload architecture flags when none are explicit."""
        spec = sol_dict["spec"]
        compile_options = dict(spec.get("compile_options") or {})
        hip_cflags = list(compile_options.get("hip_cflags", []))

        if any(
            flag.startswith(("--offload-arch", "-offload-arch", "--amdgpu-target"))
            for flag in hip_cflags
        ):
            return sol_dict

        offload_arches: list[str] = []
        target_hw = set(spec.get("target_hardware", []))

        for hardware in SupportedHardware:
            value = hardware.value
            if value != SupportedHardware.LOCAL.value and value in target_hw:
                offload_arches.append(value)

        if SupportedHardware.LOCAL.value in target_hw:
            local_gfx = _get_local_gfx()
            if local_gfx:
                offload_arches.append(local_gfx)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique = [gfx for gfx in offload_arches if not (gfx in seen or seen.add(gfx))]
        if unique:
            compile_options["hip_cflags"] = [
                _gfx_to_offload_arch(gfx) for gfx in unique
            ] + hip_cflags
            spec["compile_options"] = compile_options
            sol_dict["spec"] = spec

        return sol_dict

    def _write_sources(self) -> None:
        """Write solution source files to the staging directory."""
        for src in self.solution.sources:
            dest = self.output_dir / src.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.content)

    def compile(self) -> tuple[list[str], str]:
        """Stage compilation files and return (command, artifact_path).

        Writes build_ext.py, solution.json, and HIP/C++ source files to
        output_dir. Injects offload architecture flags for the target hardware.

        The CLI should run the command in output_dir.
        After success, the artifact (benchmark_kernel.so) will be at artifact_path.
        """
        assert self._is_cpp, (
            f"compile() only handles HIP/C++ solutions, "
            f"got languages={self.solution.spec.languages}"
        )

        sol_dict = json.loads(self.solution.model_dump_json())
        sol_dict = self._inject_offload_arch_flags(sol_dict)

        # Overwrite solution.json with injected offload architecture flags.
        (self.output_dir / "solution.json").write_text(json.dumps(sol_dict))
        (self.output_dir / "build_ext.py").write_text(
            (_TEMPLATES_DIR / "build_ext.py").read_text()
        )

        cmd = ["python", "build_ext.py"]
        artifact_path = str(self.output_dir / "benchmark_kernel.so")

        return cmd, artifact_path

    def execute(self) -> list[str]:
        """Stage execution files and return the command to run.

        Writes eval_driver.py, definition.json, workload.jsonl, solution.json
        to output_dir. For Python solutions, also writes source files. For C++
        solutions, expects benchmark_kernel.so to already exist in output_dir
        (produced by a prior compile() call).

        The CLI should run the command in output_dir.
        Trace JSON will be emitted on stdout (one JSON object per line).
        """
        if self._is_cpp:
            so_path = self.output_dir / "benchmark_kernel.so"
            if not so_path.exists():
                raise FileNotFoundError(
                    f"benchmark_kernel.so not found at {so_path} — "
                    "run compile() first for HIP/C++ solutions"
                )

        (self.output_dir / "eval_driver.py").write_text(
            (_TEMPLATES_DIR / "eval_driver.py").read_text()
        )

        return ["python", "eval_driver.py"]

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        """Parse JSONL stdout from eval_driver.py into Trace objects.

        Each line starting with '{' is parsed as a Trace JSON object.
        Non-JSON lines (library noise redirected to stderr) are skipped.
        """
        traces = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                traces.append(Trace(**json.loads(line)))
        return traces
