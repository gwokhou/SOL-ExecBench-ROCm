# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from rich.console import Console

from sol_execbench.cli.evaluation.diagnostics import (
    _DIAGNOSTIC_TAIL_LIMIT,
    NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
    NoTraceDiagnostics,
    _diagnostic_tail,
    _no_trace_diagnostics_sidecar_path,
    _write_no_trace_diagnostics_sidecar,
)
from sol_execbench.cli.sidecars.profile import _profile_output_directory
from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    acquire_clock_lock,
)
from sol_execbench.core.bench.io import flashinfer_safetensors_env
from sol_execbench.core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    ProfileRunner,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_GRACEFUL_EXIT,
    sanitized_subprocess_env,
)
from sol_execbench.core.process.subprocesses import (
    TextSubprocessRunner,
    run_in_process_group_bounded,
)

__all__ = [
    "NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION",
    "_DIAGNOSTIC_TAIL_LIMIT",
    "NoTraceDiagnostics",
    "_diagnostic_tail",
    "_no_trace_diagnostics_sidecar_path",
    "_run_evaluation_command",
    "_run_profiled_evaluation",
    "_timeout_output_text",
    "_write_no_trace_diagnostics_sidecar",
]

console = Console(stderr=True)
logger = logging.getLogger(__name__)
EnvironmentBuilder = Callable[[Mapping[str, str]], dict[str, str]]
ClockLocker = Callable[[], ClockLockLease]


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
        base[ENV_SOL_EXECBENCH_GRACEFUL_EXIT] = "1"
    sanitized = sanitized_subprocess_env(base, staging_dir=staging_dir)
    return sanitized_subprocess_env(
        env_builder(sanitized),
        staging_dir=staging_dir,
    )


def _run_command(
    command: list[str],
    *,
    cwd: Path | None,
    timeout: int | None,
    env: Mapping[str, str],
    runner: TextSubprocessRunner | None,
) -> subprocess.CompletedProcess[str]:
    if runner is None:
        return run_in_process_group_bounded(
            command,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
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
    clock_locker: ClockLocker = acquire_clock_lock,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure.

    The rocprofv3 collection is wrapped in a best-effort STABLE_PEAK clock lock.

    Why we lock the clock: on gfx1200 (RDNA4) the SQ perf counter
    ``SQ_WAVE_CYCLES`` (event 24) reads exactly zero under the default ``AUTO``
    power policy, because the dVFS shader-clock transitions suppress its
    increment. That silently corrupts every derived occupancy/stall metric
    (``MeanOccupancyPerCU``, ``OccupancyPercent``, ``WAVE_DEP_WAIT``,
    ``WAVE_ISSUE_WAIT``). Holding a stable power state removes the transitions
    so the counter accumulates normally; ``STABLE_PEAK`` is chosen over AMD's
    ``STABLE_STD`` because it keeps a representative high SCLK/MCLK mix, whereas
    STD collapses MCLK and bandwidth-starves the kernel (see issue for data).
    Background: https://github.com/ROCm/rocm-systems/issues/8523 and AMD's
    stable-power-state guidance in the rocprofiler-sdk limitations doc.

    The lock comes from :func:`acquire_clock_lock`: idempotent (an outer
    STABLE_PEAK is preserved, never released by this inner acquire) and
    best-effort -- if the lock is unsupported or unavailable, profiling still
    runs unlocked and a warning is logged that gfx1200 counters may be
    unreliable.
    """
    output_directory = _profile_output_directory(output_file, staging_dir)
    request = Rocprofv3ProfileRequest(
        application_command=tuple(eval_cmd),
        output_directory=output_directory,
        output_file="profile",
        working_directory=staging_dir,
        timeout_seconds=timeout,
    )
    if rocprofv3_available is None:
        rocprofv3_available = (
            resolve_rocm_tool(ROCPROFV3_EXECUTABLE) is not None
        )
    # gfx1200: SQ_WAVE_CYCLES reads zero under AUTO/dVFS (ROCm issue #8523,
    # https://github.com/ROCm/rocm-systems/issues/8523). Hold STABLE_PEAK for
    # the rocprofv3 collection so the counter and its derived occupancy/stall
    # metrics are valid. Idempotent vs an outer lock; best-effort skip below.
    with clock_locker() as clock_lease:
        if not clock_lease.locked:
            logger.warning(
                "rocprofv3 profiling is running without a STABLE_PEAK clock "
                "lock; on gfx1200 SQ_WAVE_CYCLES and derived occupancy/stall "
                "metrics may read zero (ROCm issue #8523).",
            )
        profile_result = profile_collector(
            request,
            rocprofv3_available=rocprofv3_available,
            runner=lambda command, cwd, timeout_seconds: _run_command(
                list(command),
                cwd=cwd,
                timeout=timeout_seconds,
                env=_evaluation_env(
                    staging_dir, env_builder, graceful_exit=True
                ),
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
