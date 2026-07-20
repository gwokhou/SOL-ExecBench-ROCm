# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from rich.console import Console

from ...core.bench.io import flashinfer_safetensors_env
from ...core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    ProfileRunner,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from ...core.platform.runtime import resolve_rocm_tool
from ...core.process.environment import sanitized_subprocess_env
from ...core.process.subprocesses import (
    TextSubprocessRunner,
    run_in_process_group_bounded,
)
from ..sidecars.profile import _profile_output_directory
from .diagnostics import (
    NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
    _DIAGNOSTIC_TAIL_LIMIT,
    NoTraceDiagnostics,
    _diagnostic_tail,
    _no_trace_diagnostics_sidecar_path,
    _write_no_trace_diagnostics_sidecar,
)

__all__ = [
    "NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION",
    "NoTraceDiagnostics",
    "_DIAGNOSTIC_TAIL_LIMIT",
    "_diagnostic_tail",
    "_no_trace_diagnostics_sidecar_path",
    "_run_evaluation_command",
    "_run_profiled_evaluation",
    "_timeout_output_text",
    "_write_no_trace_diagnostics_sidecar",
]

console = Console(stderr=True)
EnvironmentBuilder = Callable[[Mapping[str, str]], dict[str, str]]


class ProfileCollector(Protocol):
    """Exact injected collector contract for profiled evaluation."""

    def __call__(
        self,
        request: Rocprofv3ProfileRequest,
        *,
        rocprofv3_available: bool = True,
        runner: ProfileRunner | None = None,
    ) -> Rocprofv3ProfileResult: ...


def _evaluation_env(
    staging_dir: Path,
    env_builder: EnvironmentBuilder,
    *,
    graceful_exit: bool = False,
) -> dict[str, str]:
    (staging_dir / ".tmp").mkdir(exist_ok=True)
    base = dict(os.environ)
    if graceful_exit:
        base["SOL_EXECBENCH_GRACEFUL_EXIT"] = "1"
    sanitized = sanitized_subprocess_env(base, staging_dir=staging_dir)
    return sanitized_subprocess_env(env_builder(sanitized), staging_dir=staging_dir)


def _run_command(
    command: list[str],
    *,
    cwd: Path | None,
    timeout: int | None,
    env: Mapping[str, str],
    runner: TextSubprocessRunner | None,
) -> subprocess.CompletedProcess[str]:
    if runner is None:
        return run_in_process_group_bounded(command, cwd=cwd, timeout=timeout, env=env)
    return runner(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


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
    env_builder: EnvironmentBuilder = flashinfer_safetensors_env,
    runner: TextSubprocessRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the staged evaluation command with the standard ROCm allocator env."""

    env = _evaluation_env(staging_dir, env_builder)
    return _run_command(
        eval_cmd,
        cwd=staging_dir,
        timeout=timeout,
        env=env,
        runner=runner,
    )


def _run_profiled_evaluation(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    env_builder: EnvironmentBuilder = flashinfer_safetensors_env,
    subprocess_run: TextSubprocessRunner | None = None,
    rocprofv3_available: bool | None = None,
    profile_collector: ProfileCollector = collect_rocprofv3_profile,
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
        runner=lambda command, cwd, timeout_seconds: _run_command(
            list(command),
            cwd=cwd,
            timeout=timeout_seconds,
            env=_evaluation_env(staging_dir, env_builder, graceful_exit=True),
            runner=subprocess_run,
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
