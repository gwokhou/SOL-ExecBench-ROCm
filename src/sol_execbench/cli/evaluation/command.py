# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from rich.console import Console

from sol_execbench.cli.evaluation.diagnostics import (
    _DIAGNOSTIC_TAIL_LIMIT,
    NoTraceDiagnostics,
    _diagnostic_tail,
    _no_trace_diagnostics_sidecar_path,
    _write_no_trace_diagnostics_sidecar,
)
from sol_execbench.cli.sidecars.profile import _profile_output_directory
from sol_execbench.core.bench.clock_lock import (
    ClockLockAcquirer,
    acquire_clock_lock,
)
from sol_execbench.core.bench.io import flashinfer_safetensors_env
from sol_execbench.core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    ProfileRunner,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_counters,
    collect_rocprofv3_profile,
)
from sol_execbench.core.evidence.runtime_evidence.collectors import (
    collect_runtime_gpu_telemetry,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_DEVICE,
    ENV_SOL_EXECBENCH_GRACEFUL_EXIT,
    sanitized_subprocess_env,
)
from sol_execbench.core.process.subprocesses import (
    TextSubprocessRunner,
    run_in_process_group_bounded,
)

__all__ = [
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
ProfileApplicationPreparer = Callable[[], None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileLifecycle:
    """Injected clock and replay-restaging lifecycle hooks."""

    clock_locker: ClockLockAcquirer = acquire_clock_lock
    prepare_profiled_application: ProfileApplicationPreparer | None = None


_DEFAULT_PROFILE_LIFECYCLE = ProfileLifecycle()


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


def _profile_command_runner(
    staging_dir: Path,
    env_builder: EnvironmentBuilder,
    subprocess_run: TextSubprocessRunner | None,
    prepare_profiled_application: ProfileApplicationPreparer | None,
) -> ProfileRunner:
    """Build the profiler callback around the evaluation process policy."""

    def run(
        command: Sequence[str],
        cwd: Path | None,
        timeout_seconds: int | None,
    ) -> subprocess.CompletedProcess[str]:
        if prepare_profiled_application is not None and "--" in command:
            prepare_profiled_application()
        return _run_command(
            list(command),
            cwd=cwd,
            timeout=timeout_seconds,
            env=_evaluation_env(
                staging_dir,
                env_builder,
                graceful_exit=True,
            ),
            runner=subprocess_run,
        )

    return run


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
    counter_mode: bool = False,
    lifecycle: ProfileLifecycle = _DEFAULT_PROFILE_LIFECYCLE,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure.

    The rocprofv3 collection is wrapped in a best-effort STABLE_PEAK clock lock.

    The lock comes from :func:`acquire_clock_lock`: idempotent (an outer
    STABLE_PEAK is preserved, never released by this inner acquire) and
    best-effort -- if the lock is unsupported or unavailable, profiling still
    runs unlocked and a warning records that the environment is not suitable
    for formal counter evidence.
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

    # Hold STABLE_PEAK across collection so replay passes share one audited
    # clock policy. Idempotent vs an outer lock; best-effort skip below.
    with lifecycle.clock_locker() as clock_lease:
        if not clock_lease.locked:
            logger.warning(
                "rocprofv3 profiling is running without a STABLE_PEAK clock "
                "lock; this run cannot be formal gfx1200 counter evidence.",
            )
        collector = (
            collect_rocprofv3_counters if counter_mode else profile_collector
        )
        pre_snapshot = _replay_snapshot("pre") if counter_mode else None
        profile_result = collector(
            request,
            rocprofv3_available=rocprofv3_available,
            runner=_profile_command_runner(
                staging_dir,
                env_builder,
                subprocess_run,
                lifecycle.prepare_profiled_application,
            ),
        )
        post_snapshot = _replay_snapshot("post") if counter_mode else None
        if counter_mode:
            profile_result = replace(
                profile_result,
                environment_snapshots=tuple(
                    snapshot
                    for snapshot in (pre_snapshot, post_snapshot)
                    if snapshot is not None
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


def _replay_snapshot(
    phase: Literal["pre", "post"],
) -> RuntimeGPUTelemetry | None:
    try:
        return collect_runtime_gpu_telemetry(
            phase=phase,
            device_index=_replay_device_index(),
        )
    except (OSError, TypeError, ValueError):
        logger.warning("Unable to collect %s replay GPU telemetry", phase)
        return None


def _replay_device_index() -> int:
    selected = os.environ.get(ENV_SOL_EXECBENCH_DEVICE, "cuda:0")
    namespace, separator, index = selected.partition(":")
    if namespace != "cuda" or separator != ":" or not index.isdecimal():
        raise ValueError(f"unsupported replay device: {selected!r}")
    return int(index)
