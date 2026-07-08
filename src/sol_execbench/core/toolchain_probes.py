# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded toolchain probes."""

from __future__ import annotations

import shutil
import subprocess

from .environment import ProbeCompletedProcess
from .text_utils import text_tail as _tail
from .toolchain_models import (
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

    path = which(binary)
    if path is None:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            status=ToolchainStatus.UNAVAILABLE,
            timeout_seconds=timeout_seconds,
        )
    effective_runner = runner or run_probe
    try:
        completed = effective_runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            path=path,
            status=ToolchainStatus.FAILED,
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            path=path,
            status=ToolchainStatus.FAILED,
            stderr_tail=_tail(str(exc)),
            timeout_seconds=timeout_seconds,
        )

    return ToolchainProbeResult(
        tool_id=tool_id,
        command=command,
        path=path,
        status=(
            ToolchainStatus.AVAILABLE
            if completed.returncode == 0
            else ToolchainStatus.FAILED
        ),
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
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
