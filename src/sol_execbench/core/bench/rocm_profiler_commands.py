# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler command construction and process runners."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler_models import ROCPROFV3_EXECUTABLE


ProfilerRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ProfileRunner = Callable[
    [Sequence[str], Path | None, int | None], subprocess.CompletedProcess[str]
]


def build_rocprofv3_command(
    application_command: Sequence[str],
    *,
    output_directory: str,
    output_file: str,
    executable: str = ROCPROFV3_EXECUTABLE,
    include_hip_runtime: bool = True,
) -> list[str]:
    """Build a `rocprofv3` command for kernel timing evidence collection."""
    return build_rocprofv3_profile_command(
        application_command,
        output_directory=output_directory,
        output_file=output_file,
        executable=executable,
        include_hip_runtime=include_hip_runtime,
        output_format="csv",
    )


def build_rocprofv3_profile_command(
    application_command: Sequence[str],
    *,
    output_directory: str,
    output_file: str,
    executable: str = ROCPROFV3_EXECUTABLE,
    include_hip_runtime: bool = True,
    output_format: str = "rocpd",
) -> list[str]:
    """Build a `rocprofv3` command for optional diagnostic artifacts."""
    if not application_command:
        raise ValueError("application_command must not be empty")

    command = [
        executable,
        "--kernel-trace",
        "--output-format",
        output_format,
        "--output-directory",
        output_directory,
        "--output-file",
        output_file,
    ]
    if include_hip_runtime:
        command.insert(2, "--hip-runtime-trace")
    return [*command, "--", *application_command]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SOL_EXECBENCH_GRACEFUL_EXIT": "1"}
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _default_profile_runner(
    command: Sequence[str],
    working_directory: Path | None,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SOL_EXECBENCH_GRACEFUL_EXIT": "1"}
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        cwd=working_directory,
        timeout=timeout_seconds,
        env=env,
    )


default_runner = _default_runner
default_profile_runner = _default_profile_runner
