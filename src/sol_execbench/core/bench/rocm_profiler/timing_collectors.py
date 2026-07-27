# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler timing parsing and live timing collection."""

import subprocess
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler.commands import (
    ProfilerRunner,
    build_rocprofv3_command,
    default_runner,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    DefaultTimingSelection,
    Rocprofv3CollectionRequest,
    Rocprofv3CollectionResult,
    Rocprofv3TimingEvidence,
    SourceTimingRequest,
)
from sol_execbench.core.bench.rocm_profiler.timing_evidence import (
    build_compact_timing_evidence,
    build_timing_evidence,
    read_overhead_calibration,
    select_default_timing,
)
from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    select_timing_policy,
    timing_policy_for_languages,
)
from sol_execbench.core.text_utils import subprocess_text


def collect_rocprofv3_timing(
    request: Rocprofv3CollectionRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
    calibration_path: Path | None = None,
) -> Rocprofv3CollectionResult:
    """Collect live `rocprofv3` timing evidence for a command.

    The runner is injectable so unit tests can exercise live collection without
    requiring a GPU or installed profiler. Non-profiler policies return explicit
    fallback metadata instead of masquerading as kernel activity timing.
    """
    selection = select_default_timing(
        request.policy,
        rocprofv3_available=rocprofv3_available,
    )
    if not selection.profiler_backed:
        return Rocprofv3CollectionResult(evidence=None, selection=selection)

    request.output_directory.mkdir(parents=True, exist_ok=True)
    command = build_rocprofv3_command(
        request.application_command,
        output_directory=str(request.output_directory),
        output_file=request.output_file,
        executable=request.executable,
        include_hip_runtime=request.include_hip_runtime,
    )
    try:
        completed = _run_profiler_command(request, command, runner)
    except subprocess.TimeoutExpired as exc:
        return _fallback_result(
            request,
            f"rocprofv3 command timed out after {request.timeout_seconds:g} seconds",
            command=command,
            csv_path=find_rocprofv3_csv(
                request.output_directory,
                request.output_file,
            ),
            stdout=subprocess_text(exc.stdout),
            stderr=subprocess_text(exc.stderr),
        )
    return _result_from_completed_run(
        request,
        selection,
        command,
        completed,
        calibration_path,
    )


def _result_from_completed_run(
    request: Rocprofv3CollectionRequest,
    selection: DefaultTimingSelection,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    calibration_path: Path | None,
) -> Rocprofv3CollectionResult:
    """Turn one completed profiler process into evidence or explicit fallback."""
    csv_path = find_rocprofv3_csv(request.output_directory, request.output_file)
    if completed.returncode != 0:
        return _fallback_result(
            request,
            f"rocprofv3 command failed with exit code {completed.returncode}",
            command=command,
            csv_path=csv_path,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    if csv_path is None:
        return _fallback_result(
            request,
            "rocprofv3 did not produce a CSV timing output",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    evidence, compacted_kernel_rows = _build_collection_evidence(
        request,
        csv_path,
        calibration_path,
    )
    if _requires_kernel_activity(request, evidence, compacted_kernel_rows):
        return _fallback_result(
            request,
            "rocprofv3 did not produce kernel activity rows",
            command=command,
            csv_path=csv_path,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    return Rocprofv3CollectionResult(
        evidence=evidence,
        selection=selection,
        command=tuple(command),
        csv_path=csv_path,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _run_profiler_command(
    request: Rocprofv3CollectionRequest,
    command: list[str],
    runner: ProfilerRunner | None,
) -> subprocess.CompletedProcess[str]:
    if runner is not None:
        return runner(command)
    return default_runner(command, timeout_seconds=request.timeout_seconds)


def _fallback_result(
    request: Rocprofv3CollectionRequest,
    reason: str,
    *,
    command: list[str],
    csv_path: Path | None = None,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> Rocprofv3CollectionResult:
    """Return a consistently labelled non-profiler timing result."""
    selection = DefaultTimingSelection(
        policy=select_timing_policy(
            request.policy.source_type,
            profiler_available=False,
        ),
        profiler_backed=False,
        fallback_applied=True,
        reason=reason,
    )
    return Rocprofv3CollectionResult(
        evidence=None,
        selection=selection,
        command=tuple(command),
        csv_path=csv_path,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _build_collection_evidence(
    request: Rocprofv3CollectionRequest,
    csv_path: Path,
    calibration_path: Path | None,
) -> tuple[Rocprofv3TimingEvidence, int | None]:
    """Parse the profiler output in its requested compact or full form."""
    profiler_overhead_ms = read_overhead_calibration(
        calibration_path,
        expected_gpu_architecture=request.gpu_architecture,
        expected_profiler_executable=request.executable,
        expected_clock_locked=request.clock_locked,
    )
    if request.compact_rows:
        return build_compact_timing_evidence(
            csv_path=csv_path,
            policy=request.policy,
            tool_version=request.tool_version,
            gpu_architecture=request.gpu_architecture,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            min_measurement_time_seconds=request.min_measurement_time_seconds,
            trial_count=request.trial_count,
            clock_locked=request.clock_locked,
            profiler_overhead_ms=profiler_overhead_ms,
        )
    return build_timing_evidence(
        csv_content=csv_path.read_text(),
        policy=request.policy,
        tool_version=request.tool_version,
        gpu_architecture=request.gpu_architecture,
        warmup_runs=request.warmup_runs,
        iterations=request.iterations,
        min_measurement_time_seconds=request.min_measurement_time_seconds,
        trial_count=request.trial_count,
        clock_locked=request.clock_locked,
        profiler_overhead_ms=profiler_overhead_ms,
    ), None


def _requires_kernel_activity(
    request: Rocprofv3CollectionRequest,
    evidence: Rocprofv3TimingEvidence,
    compacted_kernel_rows: int | None,
) -> bool:
    if request.policy.activity_domain != TimingActivityDomain.KERNEL_ACTIVITY:
        return False
    if (compacted_kernel_rows or 0) > 0:
        return False
    return not any(row.is_kernel_activity for row in evidence.parsed_rows)


def collect_source_timing_evidence(
    request: SourceTimingRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
) -> Rocprofv3CollectionResult:
    """Select source-specific timing policy and collect evidence when supported."""
    policy = timing_policy_for_languages(
        request.languages,
        profiler_available=True,
    )
    collection_request = Rocprofv3CollectionRequest(
        application_command=request.application_command,
        output_directory=request.output_directory,
        output_file=request.output_file,
        policy=policy,
        tool_version=request.tool_version,
        gpu_architecture=request.gpu_architecture,
        executable=request.executable,
        warmup_runs=request.warmup_runs,
        iterations=request.iterations,
        min_measurement_time_seconds=request.min_measurement_time_seconds,
        trial_count=request.trial_count,
        clock_locked=request.clock_locked,
        timeout_seconds=request.timeout_seconds,
    )
    return collect_rocprofv3_timing(
        collection_request,
        rocprofv3_available=rocprofv3_available,
        runner=runner,
    )


def find_rocprofv3_csv(output_directory: Path, output_file: str) -> Path | None:
    """Return the best matching rocprofv3 CSV output, if present."""
    candidates = sorted(output_directory.glob(f"{output_file}*.csv"))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.endswith("_kernel_trace.csv"):
            return candidate
    return candidates[0]
