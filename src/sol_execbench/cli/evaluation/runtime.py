# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation runtime orchestration helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from sol_execbench.cli.evaluation import command as cli_evaluation
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.core.bench.performance_model.timing_evidence import (
    RAW_TIMING_FILENAME,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.reports.relative_metrics import apply_reference_speedups


class EvaluationPackager(Protocol):
    """Minimal staged-package behavior needed by runtime evaluation."""

    def restage_trusted_reference(self) -> None:
        """Restore the worker-only reference for one replay process."""
        ...

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        """Parse staged-driver standard output into traces."""
        ...


class EvaluationRuntimeFailureReason(StrEnum):
    """Stable machine-readable reasons for executions without trace output."""

    TIMEOUT = "evaluation_timeout"
    FAILED_NO_STDOUT = "evaluation_failed_no_stdout"
    NO_PARSEABLE_TRACES = "no_parseable_traces"


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeSuccess:
    """Successful runtime execution with parsed traces and diagnostics."""

    traces: list[Trace]
    stdout: str
    stderr: str
    filtered_stderr: str
    returncode: int
    profile_result: Rocprofv3ProfileResult | None
    profile_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeNoTraceFailure:
    """Common diagnostics for a classified execution failure."""

    reason: ClassVar[EvaluationRuntimeFailureReason]
    message: str
    stdout: str
    stderr: str
    filtered_stderr: str
    returncode: int
    profile_result: Rocprofv3ProfileResult | None
    profile_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeTimeout(EvaluationRuntimeNoTraceFailure):
    """Execution that exceeded its timeout without producing traces."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.TIMEOUT
    )


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeFailedNoStdout(EvaluationRuntimeNoTraceFailure):
    """Failed execution that produced no standard output."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.FAILED_NO_STDOUT
    )


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeNoParseableTraces(EvaluationRuntimeNoTraceFailure):
    """Execution whose output contained no parseable traces."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.NO_PARSEABLE_TRACES
    )


EvaluationRuntimeResult = (
    EvaluationRuntimeSuccess
    | EvaluationRuntimeTimeout
    | EvaluationRuntimeFailedNoStdout
    | EvaluationRuntimeNoParseableTraces
)


def _profile_fallback_reason(
    profile_result: Rocprofv3ProfileResult | None,
) -> str | None:
    if profile_result is None:
        return None
    return profile_result.skipped_reason or profile_result.failed_reason


def _run_profiled_or_none(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: ProfileMode,
) -> tuple[
    subprocess.CompletedProcess[str] | None,
    Rocprofv3ProfileResult | None,
]:
    if profile not in {ProfileMode.ROCPROFV3, ProfileMode.ROCPROFV3_COUNTERS}:
        return None, None
    if profile == ProfileMode.ROCPROFV3_COUNTERS:
        return None, None
    return cli_evaluation._run_profiled_evaluation(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
    )


def run_evaluation_runtime(
    packager: EvaluationPackager,
    *,
    eval_cmd: list[str],
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: ProfileMode,
) -> EvaluationRuntimeResult:
    """Run evaluation and classify subprocess outcomes without CLI side effects."""
    profiled_proc, profile_result = _run_profiled_or_none(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
    )
    try:
        proc = profiled_proc or cli_evaluation._run_evaluation_command(
            eval_cmd,
            staging_dir=staging_dir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = cli_evaluation._timeout_output_text(exc.stdout)
        stderr = cli_evaluation._timeout_output_text(exc.stderr)
        return EvaluationRuntimeTimeout(
            message=f"Evaluation timed out after {timeout}s",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            filtered_stderr=filter_benign_rocm_stderr(stderr),
            profile_result=profile_result,
            profile_fallback_reason=_profile_fallback_reason(profile_result),
        )

    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        return EvaluationRuntimeFailedNoStdout(
            message="Evaluation failed",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
            profile_fallback_reason=_profile_fallback_reason(profile_result),
        )

    traces = packager.convert_stdout_to_traces(proc.stdout)
    if not traces:
        return EvaluationRuntimeNoParseableTraces(
            message="No traces produced",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
            profile_fallback_reason=_profile_fallback_reason(profile_result),
        )

    apply_reference_speedups(traces)
    if profile == ProfileMode.ROCPROFV3_COUNTERS:
        profile_result = _collect_counter_replay(
            packager,
            eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
        )

    return EvaluationRuntimeSuccess(
        traces=traces,
        stdout=proc.stdout,
        stderr=proc.stderr,
        filtered_stderr=filtered_stderr,
        returncode=proc.returncode,
        profile_result=profile_result,
        profile_fallback_reason=_profile_fallback_reason(profile_result),
    )


def _collect_counter_replay(
    packager: EvaluationPackager,
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
) -> Rocprofv3ProfileResult:
    """Collect counter passes after canonical timing and preserve its samples."""
    timing_path = staging_dir / RAW_TIMING_FILENAME
    canonical_timing = (
        timing_path.read_bytes() if timing_path.is_file() else None
    )
    try:
        _, result = cli_evaluation._run_profiled_evaluation(
            eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
            counter_mode=True,
            lifecycle=cli_evaluation.ProfileLifecycle(
                prepare_profiled_application=(
                    packager.restage_trusted_reference
                ),
            ),
        )
    finally:
        if canonical_timing is not None:
            timing_path.write_bytes(canonical_timing)
        elif timing_path.is_file():
            timing_path.unlink()
    return result
