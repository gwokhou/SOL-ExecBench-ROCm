# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation runtime orchestration helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import json
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from sol_execbench.cli.evaluation import command as cli_evaluation
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.core.bench.performance_model.timing_evidence import (
    RAW_TIMING_FILENAME,
)
from sol_execbench.core.bench.reference_protocol import (
    REFERENCE_ATTESTATION_CONFIG_FILE,
    REFERENCE_ATTESTATION_RECEIPT_FILE,
    ReferenceProtocolError,
    verify_reference_attestation,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.data.trace import EvaluationStatus, Trace
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
    del eval_cmd, staging_dir, output_file, timeout, profile
    return None, None


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


def _prepare_worker_attestation(
    staging_dir: Path,
) -> tuple[str, Path, Path, bytes] | None:
    if not staging_dir.is_dir():
        return None
    key_hex = secrets.token_hex(32)
    config_path = staging_dir / REFERENCE_ATTESTATION_CONFIG_FILE
    receipt_path = staging_dir / REFERENCE_ATTESTATION_RECEIPT_FILE
    receipt_path.unlink(missing_ok=True)
    config_bytes = json.dumps(
        {"key": key_hex, "receipt": REFERENCE_ATTESTATION_RECEIPT_FILE},
        sort_keys=True,
    ).encode()
    config_path.write_bytes(config_bytes)
    return key_hex, config_path, receipt_path, config_bytes


def _consume_worker_attestation(
    state: tuple[str, Path, Path, bytes] | None,
) -> tuple[dict[str, object] | None, str | None]:
    if state is None:
        return None, None
    key_hex, config_path, receipt_path, config_bytes = state
    try:
        if config_path.is_file() and config_path.read_bytes() == config_bytes:
            return None, None
        if not receipt_path.is_file():
            return (
                None,
                "Trusted reference worker attestation receipt is missing",
            )
        envelope = json.loads(receipt_path.read_text())
        payload = verify_reference_attestation(envelope, key_hex=key_hex)
        return payload, None
    except (OSError, ValueError, ReferenceProtocolError) as exc:
        return None, f"Trusted reference worker attestation failed: {exc}"
    finally:
        config_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)


def _verify_trace_identities(
    packager: EvaluationPackager,
    traces: list[Trace],
) -> str | None:
    definition = getattr(packager, "definition", None)
    solution = getattr(packager, "solution", None)
    workloads = getattr(packager, "workloads", None)
    if (
        definition is None
        or solution is None
        or not isinstance(workloads, list)
    ):
        return None
    if len(traces) != len(workloads):
        return (
            f"Expected {len(workloads)} trace(s) for staged workloads "
            f"but parsed {len(traces)}"
        )
    def_name = getattr(definition, "name", None)
    sol_name = getattr(solution, "name", None)
    for trace, expected_workload in zip(traces, workloads, strict=True):
        expected_uuid = getattr(expected_workload, "uuid", None)
        if (
            trace.definition != def_name
            or trace.solution != sol_name
            or trace.workload.uuid != expected_uuid
        ):
            return (
                "Parsed trace identity does not match staged "
                "(definition, solution, workload.uuid)"
            )
    return None


def _verify_one_passed_trace_attestation(
    trace: Trace,
    workloads_att: Mapping[str, object],
    cfg: Mapping[str, object],
    attested_def: object,
) -> str | None:
    if trace.definition != attested_def:
        return "Trace definition disagrees with reference worker attestation"
    wl_att = workloads_att.get(trace.workload.uuid)
    if not isinstance(wl_att, dict):
        return f"Missing worker attestation for workload {trace.workload.uuid}"
    if (
        wl_att.get("all_correctness_rounds") is not True
        or wl_att.get("correctness_rounds_completed") != 10
        or wl_att.get("timing_case_issued") is not True
    ):
        return f"Workload {trace.workload.uuid} did not complete trusted reference stages"
    perf = trace.evaluation.performance if trace.evaluation else None
    ref_lat = wl_att.get("reference_latency_ms")
    passed_vals = wl_att.get("timed_validations_passed")
    trials = cfg.get("trials")
    warmup = cfg.get("warmup_runs")
    if (
        perf is None
        or not isinstance(ref_lat, (int, float))
        or not isinstance(passed_vals, int)
        or not isinstance(trials, int)
        or not isinstance(warmup, int)
    ):
        return "Malformed worker attestation metrics"
    if abs(perf.reference_latency_ms - float(ref_lat)) > 1e-6:
        return "Trace reference_latency_ms disagrees with worker attestation"
    timed_sum = sum(perf.timed_iterations_per_trial)
    if timed_sum < max(trials, 1) or passed_vals != trials * warmup + timed_sum:
        return "Trace timed validation count disagrees with worker attestation"
    return None


def _verify_worker_attestation(
    traces: list[Trace],
    payload: dict[str, object] | None,
    attestation_error: str | None,
) -> str | None:
    if attestation_error is not None:
        return attestation_error
    if payload is None:
        return None
    if payload.get("pending_timing_validation") is not False:
        return "Reference worker exited with an unvalidated timing iteration"
    workloads_raw = payload.get("workloads")
    cfg_raw = payload.get("config")
    if not isinstance(workloads_raw, Mapping) or not isinstance(
        cfg_raw, Mapping
    ):
        return "Reference worker attestation payload is incomplete"
    workloads_att: Mapping[str, object] = {
        str(k): v for k, v in workloads_raw.items()
    }
    cfg: Mapping[str, object] = {str(k): v for k, v in cfg_raw.items()}
    attested_def = payload.get("definition")
    for trace in traces:
        if (
            trace.evaluation is not None
            and trace.evaluation.status is EvaluationStatus.PASSED
        ):
            failure = _verify_one_passed_trace_attestation(
                trace,
                workloads_att,
                cfg,
                attested_def,
            )
            if failure is not None:
                return failure
    return None


_CompletedClassification = (
    tuple[list[Trace], str]
    | EvaluationRuntimeFailedNoStdout
    | EvaluationRuntimeNoParseableTraces
    | EvaluationRuntimeIncomplete
)


def _classify_completed_evaluation(
    packager: EvaluationPackager,
    proc: subprocess.CompletedProcess[str],
    *,
    expected_trace_count: int | None,
    profile_result: Rocprofv3ProfileResult | None,
    attestation_payload: dict[str, object] | None,
    attestation_error: str | None,
) -> _CompletedClassification:
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
    integrity_failure = _verify_trace_identities(packager, traces) or (
        _verify_worker_attestation(
            traces, attestation_payload, attestation_error
        )
        if proc.returncode == 0
        else None
    )
    if integrity_failure is not None:
        return _failure_result(
            EvaluationRuntimeIncomplete,
            integrity_failure,
            proc,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
        )
    return traces, filtered_stderr


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
    if profiled_proc is None and profile_result is not None:
        packager.restage_trusted_reference()
    attestation_state = (
        _prepare_worker_attestation(staging_dir)
        if profiled_proc is None
        else None
    )
    proc_or_timeout = _run_eval_subprocess(
        eval_cmd,
        staging_dir=staging_dir,
        timeout=timeout,
        profiled_proc=profiled_proc,
        profile_result=profile_result,
    )
    attestation_payload, attestation_error = _consume_worker_attestation(
        attestation_state,
    )
    if isinstance(proc_or_timeout, EvaluationRuntimeTimeout):
        return proc_or_timeout
    proc = proc_or_timeout
    classified = _classify_completed_evaluation(
        packager,
        proc,
        expected_trace_count=expected_trace_count,
        profile_result=profile_result,
        attestation_payload=attestation_payload,
        attestation_error=attestation_error,
    )
    if not isinstance(classified, tuple):
        return classified
    traces, filtered_stderr = classified

    apply_reference_speedups(traces)
    if profile in {ProfileMode.ROCPROFV3, ProfileMode.ROCPROFV3_COUNTERS}:
        profile_result = _collect_post_canonical_profile(
            packager,
            eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
            counter_mode=profile is ProfileMode.ROCPROFV3_COUNTERS,
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


def _collect_post_canonical_profile(
    packager: EvaluationPackager,
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    counter_mode: bool,
) -> Rocprofv3ProfileResult:
    """Collect profiler evidence after canonical timing and preserve its samples."""
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
            counter_mode=counter_mode,
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


def _collect_counter_replay(
    packager: EvaluationPackager,
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
) -> Rocprofv3ProfileResult:
    """Collect counter passes after canonical timing and preserve its samples."""
    return _collect_post_canonical_profile(
        packager,
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        counter_mode=True,
    )
