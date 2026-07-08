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

"""Package a problem (definition + workloads + solution) into a staging directory.

Produces shell commands that the CLI can run directly via subprocess to compile
HIP/C++ solutions and evaluate them on the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..core.bench.config import BenchmarkConfig
from ..core.data.definition import Definition
from ..core.data.solution import CompileOptions, NATIVE_ROCM_LANGUAGES
from ..core.data.solution import Solution, SupportedHardware
from ..core.data.trace import Trace
from ..core.data.workload import SafetensorsInput, Workload
from ..core.text_utils import ordered_unique

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
        self._stage_safetensors_inputs()

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
        return any(
            lang in NATIVE_ROCM_LANGUAGES for lang in self.solution.spec.languages
        )

    def _inject_offload_arch_flags(self, sol_dict: dict) -> dict:
        """Auto-inject HIP offload architecture flags when none are explicit."""
        spec = sol_dict["spec"]
        compile_options = dict(spec.get("compile_options") or {})
        if "hip_cflags" not in compile_options:
            # Preserve the CompileOptions default (e.g. -O3) instead of
            # clobbering it with an empty list when no flags were specified.
            compile_options["hip_cflags"] = list(CompileOptions().hip_cflags)
        hip_cflags = list(compile_options["hip_cflags"])

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

        unique = ordered_unique(offload_arches)
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

    def _stage_safetensors_inputs(self) -> None:
        """Expose repo-local safetensors blobs under their workload paths."""
        for workload in self.workloads:
            for input_spec in workload.inputs.values():
                if not isinstance(input_spec, SafetensorsInput):
                    continue
                source, relative_path = self._resolve_stageable_safetensors(
                    input_spec.path
                )
                if source is None or relative_path is None:
                    continue
                dest = self.output_dir / relative_path
                if dest.exists() or dest.is_symlink():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dest.symlink_to(source)
                except OSError:
                    shutil.copy2(source, dest)

    def _resolve_stageable_safetensors(
        self, raw_path: str
    ) -> tuple[Path | None, Path | None]:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            return None, None

        roots = [_repo_root()]
        env_root = os.environ.get("FLASHINFER_TRACE_DIR")
        if env_root:
            roots.insert(0, Path(env_root))

        parts = path.parts
        for root in roots:
            for start in range(len(parts)):
                source = root / Path(*parts[start:])
                if source.is_file():
                    return source.resolve(), path
        return None, None

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

        sol_dict = self._inject_offload_arch_flags(
            self.solution.model_dump(mode="json")
        )

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
