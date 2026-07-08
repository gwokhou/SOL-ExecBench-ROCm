# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compilation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.bench.io import flashinfer_safetensors_env
from ...core.bench.stderr import filter_benign_rocm_stderr


@dataclass(frozen=True)
class CompilePhaseResult:
    attempted: bool
    succeeded: bool
    artifact_path: Path | None
    stdout: str
    filtered_stderr: str
    returncode: int


def _compile_command(packager: Any, output_path: Path) -> tuple[list[str], Path]:
    make_compile_cmd = getattr(packager, "_make_compile_cmd", None)
    if make_compile_cmd is not None:
        return make_compile_cmd(output_path), output_path

    cmd, artifact_path = packager.compile()
    return cmd, Path(artifact_path)


def run_compile_phase(
    packager: Any,
    *,
    staging_dir: Path,
    compile_timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CompilePhaseResult:
    """Compile a staged HIP/C++ solution and return subprocess diagnostics."""

    if not packager._is_cpp:
        return CompilePhaseResult(
            attempted=False,
            succeeded=False,
            artifact_path=None,
            stdout="",
            filtered_stderr="",
            returncode=0,
        )

    artifact_path = staging_dir / "benchmark_kernel.so"
    cmd, artifact_path = _compile_command(packager, artifact_path)
    env = env_builder({**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    proc = runner(
        cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=compile_timeout,
        env=env,
    )

    return CompilePhaseResult(
        attempted=True,
        succeeded=proc.returncode == 0,
        artifact_path=artifact_path,
        stdout=proc.stdout,
        filtered_stderr=filter_benign_rocm_stderr(proc.stderr),
        returncode=proc.returncode,
    )
