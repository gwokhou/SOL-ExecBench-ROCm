# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from rich.console import Console

from ...core.bench.io import flashinfer_safetensors_env
from ...core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from ...core.platform.runtime import resolve_rocm_tool
from ..sidecars.profile import _profile_output_directory
from .diagnostics import (
    NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
    _DIAGNOSTIC_TAIL_LIMIT,
    _diagnostic_tail,
    _no_trace_diagnostics_sidecar_path,
    _write_no_trace_diagnostics_sidecar,
)

__all__ = [
    "NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION",
    "_DIAGNOSTIC_TAIL_LIMIT",
    "_diagnostic_tail",
    "_no_trace_diagnostics_sidecar_path",
    "_run_evaluation_command",
    "_run_profiled_evaluation",
    "_timeout_output_text",
    "_write_no_trace_diagnostics_sidecar",
]

console = Console(stderr=True)


def _timeout_output_text(output: str | bytes | None) -> str:
    """Return timeout output as text regardless of subprocess typing."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _run_evaluation_command(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run the staged evaluation command with the standard ROCm allocator env."""

    env = env_builder({**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    return runner(
        eval_cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _run_profiled_evaluation(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    rocprofv3_available: bool | None = None,
    profile_collector: Callable[
        ..., Rocprofv3ProfileResult
    ] = collect_rocprofv3_profile,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure."""

    output_directory = _profile_output_directory(output_file, staging_dir)
    request = Rocprofv3ProfileRequest(
        application_command=tuple(eval_cmd),
        output_directory=output_directory,
        output_file="profile",
        working_directory=staging_dir,
        timeout_seconds=timeout,
    )
    if rocprofv3_available is None:
        rocprofv3_available = resolve_rocm_tool(ROCPROFV3_EXECUTABLE) is not None
    profile_result = profile_collector(
        request,
        rocprofv3_available=rocprofv3_available,
        runner=lambda command, cwd, timeout_seconds: subprocess_run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env_builder(
                {
                    **os.environ,
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "SOL_EXECBENCH_GRACEFUL_EXIT": "1",
                }
            ),
        ),
    )
    if profile_result.succeeded:
        profiled_proc = subprocess.CompletedProcess(
            args=list(profile_result.command),
            returncode=profile_result.returncode or 0,
            stdout=profile_result.stdout,
            stderr=profile_result.stderr,
        )
        return profiled_proc, profile_result
    return None, profile_result
