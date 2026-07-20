# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler timing parsing and live timing collection."""

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
    run = runner or default_runner
    completed = run(command)
    csv_path = find_rocprofv3_csv(request.output_directory, request.output_file)
    if completed.returncode != 0:
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason=f"rocprofv3 command failed with exit code {completed.returncode}",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
            csv_path=csv_path,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    if csv_path is None:
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason="rocprofv3 did not produce a CSV timing output",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    profiler_overhead_ms = read_overhead_calibration(calibration_path)
    compacted_kernel_rows: int | None = None
    if request.compact_rows:
        evidence, compacted_kernel_rows = build_compact_timing_evidence(
            policy=request.policy,
            csv_path=csv_path,
            tool_version=request.tool_version,
            gpu_architecture=request.gpu_architecture,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            min_measurement_time_seconds=request.min_measurement_time_seconds,
            trial_count=request.trial_count,
            clock_locked=request.clock_locked,
            profiler_overhead_ms=profiler_overhead_ms,
        )
    else:
        evidence = build_timing_evidence(
            policy=request.policy,
            csv_content=csv_path.read_text(),
            tool_version=request.tool_version,
            gpu_architecture=request.gpu_architecture,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            min_measurement_time_seconds=request.min_measurement_time_seconds,
            trial_count=request.trial_count,
            clock_locked=request.clock_locked,
            profiler_overhead_ms=profiler_overhead_ms,
        )
    if (
        request.policy.activity_domain == TimingActivityDomain.KERNEL_ACTIVITY
        and (compacted_kernel_rows or 0) <= 0
        and not any(row.is_kernel_activity for row in evidence.parsed_rows)
    ):
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason="rocprofv3 did not produce kernel activity rows",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
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


def collect_source_timing_evidence(
    request: SourceTimingRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
) -> Rocprofv3CollectionResult:
    """Select source-specific timing policy and collect evidence when supported."""
    policy = timing_policy_for_languages(request.languages, profiler_available=True)
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
    )
    return collect_rocprofv3_timing(
        collection_request,
        rocprofv3_available=rocprofv3_available,
        runner=runner,
    )


def find_rocprofv3_csv(output_directory: Path, output_file: str) -> Path | None:
    candidates = sorted(output_directory.glob(f"{output_file}*.csv"))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.endswith("_kernel_trace.csv"):
            return candidate
    return candidates[0]


_find_rocprofv3_csv = find_rocprofv3_csv
