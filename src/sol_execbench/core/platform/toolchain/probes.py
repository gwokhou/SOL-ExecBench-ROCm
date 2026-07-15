# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded toolchain probes."""

from __future__ import annotations

import shutil
import subprocess

from ..environment import ProbeCompletedProcess
from ..runtime import resolve_tool_path
from ...text_utils import text_tail
from .models import (
    DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    ProbeRunner,
    ToolchainProbeResult,
    ToolchainStatus,
    Which,
)


def probe_toolchain_tool(
    tool_id: str,
    binary: str,
    command: list[str],
    *,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
) -> ToolchainProbeResult:
    """Run one bounded toolchain probe."""

    resolved_path = resolve_tool_path(binary, which=which)
    path = str(resolved_path) if resolved_path is not None else None
    if path is None:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            status=ToolchainStatus.UNAVAILABLE,
            timeout_seconds=timeout_seconds,
        )
    effective_command = [path, *command[1:]]
    effective_runner = runner or run_probe
    try:
        completed = effective_runner(effective_command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=effective_command,
            path=path,
            status=ToolchainStatus.FAILED,
            stdout_tail=text_tail(exc.stdout),
            stderr_tail=text_tail(exc.stderr),
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=effective_command,
            path=path,
            status=ToolchainStatus.FAILED,
            stderr_tail=text_tail(str(exc)),
            timeout_seconds=timeout_seconds,
        )

    return ToolchainProbeResult(
        tool_id=tool_id,
        command=effective_command,
        path=path,
        status=(
            ToolchainStatus.AVAILABLE
            if completed.returncode == 0
            else ToolchainStatus.FAILED
        ),
        returncode=completed.returncode,
        stdout_tail=text_tail(completed.stdout),
        stderr_tail=text_tail(completed.stderr),
        timeout_seconds=timeout_seconds,
    )


def run_probe(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return ProbeCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
