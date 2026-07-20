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

from ...core.bench.rocm_profiler import Rocprofv3ProfileResult
from ...core.bench.stderr import filter_benign_rocm_stderr
from ...core.data.trace import Trace
from ...core.reports.relative_metrics import apply_reference_speedups
from . import command as cli_evaluation
from .profile_mode import PROFILE_ROCPROFV3


class EvaluationPackager(Protocol):
    """Minimal staged-package behavior needed by runtime evaluation."""

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]: ...


class EvaluationRuntimeFailureReason(StrEnum):
    """Stable machine-readable reasons for executions without trace output."""

    TIMEOUT = "evaluation_timeout"
    FAILED_NO_STDOUT = "evaluation_failed_no_stdout"
    NO_PARSEABLE_TRACES = "no_parseable_traces"


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeSuccess:
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
    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.TIMEOUT
    )


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeFailedNoStdout(EvaluationRuntimeNoTraceFailure):
    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.FAILED_NO_STDOUT
    )


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeNoParseableTraces(EvaluationRuntimeNoTraceFailure):
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
    profile: str,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult | None]:
    if profile != PROFILE_ROCPROFV3:
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
    profile: str,
) -> EvaluationRuntimeResult:
    """Run evaluation and classify subprocess outcomes without CLI side effects."""

    profiled_proc, profile_result = _run_profiled_or_none(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
    )
    fallback_reason = _profile_fallback_reason(profile_result)

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
            profile_fallback_reason=fallback_reason,
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
            profile_fallback_reason=fallback_reason,
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
            profile_fallback_reason=fallback_reason,
        )

    apply_reference_speedups(traces)

    return EvaluationRuntimeSuccess(
        traces=traces,
        stdout=proc.stdout,
        stderr=proc.stderr,
        filtered_stderr=filtered_stderr,
        returncode=proc.returncode,
        profile_result=profile_result,
        profile_fallback_reason=fallback_reason,
    )
