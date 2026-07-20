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
import sys
import subprocess
from pathlib import Path

from ..core.bench.config import BenchmarkConfig
from ..core.bench.clock_lock import acquire_clock_lock
from ..core.data.solution import NATIVE_ROCM_LANGUAGES
from .build_config import (
    first_gfx_target as _first_gfx_target_core,
    gfx_to_offload_arch as _gfx_to_offload_arch_core,
    get_local_gfx,
    inject_offload_arch_flags,
)
from .staging import (
    Definition,
    Solution,
    Workload,
    resolve_stageable_safetensors,
    stage_definition_files,
    stage_safetensors_inputs,
    stage_solution_sources,
)
from .trace_output import Trace, parse_trace_jsonl
from .problem_packager_lifecycle import ProblemPackagerLifecycle

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _first_gfx_target(lines: list[str]) -> str | None:
    """Return the first concrete AMD gfx target from command output lines."""
    return _first_gfx_target_core(lines)


def _get_local_gfx() -> str | None:
    """Detect the local AMD GPU gfx target using ROCm tooling."""
    return get_local_gfx(check_output=subprocess.check_output)


def _gfx_to_offload_arch(gfx: str) -> str:
    """Convert an AMD gfx target string to a HIP offload architecture flag."""
    return _gfx_to_offload_arch_core(gfx)


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
        self._lifecycle = ProblemPackagerLifecycle(output_dir, keep_output_dir)
        self.definition = definition
        self.workloads = workloads
        self.solution = solution
        self.config = config
        stage_definition_files(definition, self.output_dir)
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
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is not None:
            lifecycle.close_safely()

    def __enter__(self) -> "ProblemPackager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lifecycle.close_for_context(exc_value)

    def close(self) -> None:
        """Release the staging directory when this packager owns cleanup."""
        self._lifecycle.close()

    @property
    def _is_cpp(self) -> bool:
        return any(
            lang in NATIVE_ROCM_LANGUAGES for lang in self.solution.spec.languages
        )

    def _inject_offload_arch_flags(
        self, sol_dict: dict[str, object]
    ) -> dict[str, object]:
        """Auto-inject HIP offload architecture flags when none are explicit."""
        return inject_offload_arch_flags(sol_dict, local_gfx_getter=_get_local_gfx)

    def _write_sources(self) -> None:
        """Write solution source files to the staging directory."""
        stage_solution_sources(self.solution, self.output_dir)

    def _stage_safetensors_inputs(self) -> None:
        """Expose repo-local safetensors blobs under their workload paths."""
        stage_safetensors_inputs(
            self.workloads,
            self.output_dir,
            repo_root=_repo_root(),
            flashinfer_trace_dir=os.environ.get("FLASHINFER_TRACE_DIR"),
        )

    def _resolve_stageable_safetensors(
        self, raw_path: str
    ) -> tuple[Path | None, Path | None]:
        return resolve_stageable_safetensors(
            raw_path,
            repo_root=_repo_root(),
            flashinfer_trace_dir=os.environ.get("FLASHINFER_TRACE_DIR"),
        )

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

        cmd = [sys.executable, "build_ext.py"]
        artifact_path = str(self.output_dir / "benchmark_kernel.so")
        return cmd, artifact_path

    def execute(self) -> list[str]:
        """Stage execution files and return the command to run.

        Writes a trusted reference worker, an orchestration launcher, and the
        candidate-only eval driver.  Reference code is loaded only by the
        trusted worker; the candidate process receives reference cases through
        authenticated, pickle-free IPC.

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

        if self._lifecycle.closed:
            raise RuntimeError("ProblemPackager is already closed")
        if self.config.lock_clocks:
            self._lifecycle.acquire_clock_lock(acquire_clock_lock)

        for name in (
            "eval_driver.py",
            "reference_worker.py",
            "evaluation_orchestrator.py",
        ):
            (self.output_dir / name).write_text((_TEMPLATES_DIR / name).read_text())

        return [sys.executable, "evaluation_orchestrator.py"]

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        """Parse JSONL stdout from eval_driver.py into Trace objects.

        Each line starting with '{' is parsed as a Trace JSON object.
        Non-JSON lines (library noise redirected to stderr) are skipped.
        """
        return parse_trace_jsonl(stdout)
