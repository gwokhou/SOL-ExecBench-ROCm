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
    EVALUATION_INCOMPLETE = "evaluation_incomplete"


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRuntimeSuccess:
    """Successful runtime execution with parsed traces and diagnostics."""

    traces: list[Trace]
    stdout: str
    stderr: str
    filtered_stderr: str
    returncode: int
    profile_result: Rocprofv3ProfileResult | None
    profile_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
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


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRuntimeTimeout(EvaluationRuntimeNoTraceFailure):
    """Execution that exceeded its timeout without producing traces."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.TIMEOUT
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRuntimeFailedNoStdout(EvaluationRuntimeNoTraceFailure):
    """Failed execution that produced no standard output."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.FAILED_NO_STDOUT
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRuntimeNoParseableTraces(EvaluationRuntimeNoTraceFailure):
    """Execution whose output contained no parseable traces."""

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.NO_PARSEABLE_TRACES
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRuntimeIncomplete(EvaluationRuntimeNoTraceFailure):
    """Execution whose trace count or exit code cannot be trusted.

    The eval driver emits exactly one trace per workload and always exits 0, so
    a trace-count mismatch or a non-zero exit (despite parseable output) means
    the run did not complete every workload — a partial run, a mid-run crash,
    or forged/injected traces (audit eval_driver.py:34 / cli-c2 / cli-c3).  The
    separate release-scoring path already enforces trace identity and count;
    the evaluate path previously did not.
    """

    reason: ClassVar[EvaluationRuntimeFailureReason] = (
        EvaluationRuntimeFailureReason.EVALUATION_INCOMPLETE
    )


EvaluationRuntimeResult = (
    EvaluationRuntimeSuccess
    | EvaluationRuntimeTimeout
    | EvaluationRuntimeFailedNoStdout
    | EvaluationRuntimeNoParseableTraces
    | EvaluationRuntimeIncomplete
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


def _failure_result[FailureT: EvaluationRuntimeNoTraceFailure](
    cls: type[FailureT],
    message: str,
    proc: subprocess.CompletedProcess[str],
    *,
    filtered_stderr: str,
    profile_result: Rocprofv3ProfileResult | None,
) -> FailureT:
    """Build a classified failure carrying the standard proc diagnostics."""
    return cls(
        message=message,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        filtered_stderr=filtered_stderr,
        profile_result=profile_result,
        profile_fallback_reason=_profile_fallback_reason(profile_result),
    )


def _run_eval_subprocess(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    timeout: int,
    profiled_proc: subprocess.CompletedProcess[str] | None,
    profile_result: Rocprofv3ProfileResult | None,
) -> subprocess.CompletedProcess[str] | EvaluationRuntimeTimeout:
    """Run the eval subprocess, classifying a timeout as a failure result."""
    try:
        return profiled_proc or cli_evaluation._run_evaluation_command(
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


def _trace_count_guard(
    proc: subprocess.CompletedProcess[str],
    traces: list[Trace],
    expected_trace_count: int | None,
    *,
    filtered_stderr: str,
    profile_result: Rocprofv3ProfileResult | None,
) -> EvaluationRuntimeIncomplete | None:
    """Return an incomplete classification when a run cannot be trusted.

    The eval driver emits one trace per workload and always exits 0, so a
    trace-count mismatch or a non-zero exit means the run did not complete
    every workload — a partial run, a mid-run crash, or forged/injected traces
    (audit eval_driver.py:34 / cli-c2 / cli-c3).  Returns ``None`` when no
    guard is active or the run is complete and clean.
    """
    if expected_trace_count is None:
        return None
    if len(traces) == expected_trace_count and proc.returncode == 0:
        return None
    if len(traces) != expected_trace_count:
        message = (
            f"Expected {expected_trace_count} trace(s) but parsed "
            f"{len(traces)}; evaluation did not complete every workload"
        )
    else:
        message = (
            f"Evaluation exited with code {proc.returncode} despite "
            f"complete trace output; result cannot be trusted"
        )
    return _failure_result(
        EvaluationRuntimeIncomplete,
        message,
        proc,
        filtered_stderr=filtered_stderr,
        profile_result=profile_result,
    )


def run_evaluation_runtime(
    packager: EvaluationPackager,
    *,
    eval_cmd: list[str],
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: ProfileMode,
    expected_trace_count: int | None = None,
) -> EvaluationRuntimeResult:
    """Run evaluation and classify subprocess outcomes without CLI side effects."""
    profiled_proc, profile_result = _run_profiled_or_none(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
    )
    proc_or_timeout = _run_eval_subprocess(
        eval_cmd,
        staging_dir=staging_dir,
        timeout=timeout,
        profiled_proc=profiled_proc,
        profile_result=profile_result,
    )
    if isinstance(proc_or_timeout, EvaluationRuntimeTimeout):
        return proc_or_timeout
    proc = proc_or_timeout

    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        return _failure_result(
            EvaluationRuntimeFailedNoStdout,
            "Evaluation failed",
            proc,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
        )

    traces = packager.convert_stdout_to_traces(proc.stdout)
    incomplete = _trace_count_guard(
        proc,
        traces,
        expected_trace_count,
        filtered_stderr=filtered_stderr,
        profile_result=profile_result,
    )
    if incomplete is not None:
        return incomplete
    if not traces:
        return _failure_result(
            EvaluationRuntimeNoParseableTraces,
            "No traces produced",
            proc,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
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
