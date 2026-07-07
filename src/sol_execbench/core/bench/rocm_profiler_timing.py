# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler timing parsing and live timing collection."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler_commands import (
    ProfilerRunner,
    build_rocprofv3_command,
    default_runner,
)
from sol_execbench.core.bench.rocm_profiler_models import (
    ROCPROFV3_EXECUTABLE,
    DefaultTimingSelection,
    Rocprofv3CollectionRequest,
    Rocprofv3CollectionResult,
    Rocprofv3TimingEvidence,
    Rocprofv3TimingRow,
)
from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
    timing_policy_for_languages,
    select_timing_policy,
)


def select_default_timing(
    policy: TimingPolicy,
    *,
    rocprofv3_available: bool,
) -> DefaultTimingSelection:
    """Resolve whether the default path can use profiler-backed timing."""
    if policy.backend == TimingBackend.ROCPROFV3 and rocprofv3_available:
        return DefaultTimingSelection(
            policy=policy,
            profiler_backed=True,
            fallback_applied=False,
            reason="rocprofv3 is available for the selected timing policy",
        )

    if policy.backend != TimingBackend.ROCPROFV3:
        reason = (
            f"selected policy backend is {policy.backend.value}, not rocprofv3 "
            "kernel activity timing"
        )
    else:
        reason = "rocprofv3 is unavailable for the selected timing policy"
    fallback_policy = replace(
        select_timing_policy(
            policy.source_type,
            profiler_available=False,
        ),
        reason=reason,
    )
    return DefaultTimingSelection(
        policy=fallback_policy,
        profiler_backed=False,
        fallback_applied=True,
        reason=reason,
    )


def parse_rocprofv3_csv(content: str) -> tuple[Rocprofv3TimingRow, ...]:
    """Parse representative `rocprofv3` CSV content into timing rows."""
    reader = csv.DictReader(content.splitlines())
    rows: list[Rocprofv3TimingRow] = []
    for raw_row in reader:
        normalized = {_normalize_header(key): value for key, value in raw_row.items()}
        name = _first_value(normalized, "kernelname", "name", "function", "operation")
        domain = _first_value(normalized, "domain", "kind", "type", "category")
        duration_ns = _duration_ns(normalized)
        if name is None or domain is None or duration_ns is None:
            continue
        rows.append(
            Rocprofv3TimingRow(
                name=name,
                domain=domain,
                duration_ns=duration_ns,
                raw={key: value for key, value in raw_row.items() if key is not None},
            )
        )
    return tuple(rows)


def summarize_rocprofv3_csv(path: Path) -> tuple[int, float]:
    """Stream a `rocprofv3` CSV file into kernel row count and duration."""
    kernel_rows = 0
    duration_ns = 0.0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            normalized = {
                _normalize_header(key): value for key, value in raw_row.items()
            }
            domain = _first_value(normalized, "domain", "kind", "type", "category")
            row_duration_ns = _duration_ns(normalized)
            if domain is None or row_duration_ns is None:
                continue
            if "kernel" not in _normalize_header(domain):
                continue
            kernel_rows += 1
            duration_ns += row_duration_ns
    return kernel_rows, duration_ns


def build_compact_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_path: Path,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> tuple[Rocprofv3TimingEvidence, int]:
    """Build summary timing evidence without materializing every CSV row."""
    kernel_rows, duration_ns = summarize_rocprofv3_csv(csv_path)
    parsed_rows = ()
    if kernel_rows > 0:
        parsed_rows = (
            Rocprofv3TimingRow(
                name="compacted_kernel_activity",
                domain="KERNEL_DISPATCH",
                duration_ns=duration_ns,
                raw={"compacted_kernel_activity_rows": str(kernel_rows)},
            ),
        )
    evidence = Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parsed_rows,
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )
    return evidence, kernel_rows


def build_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_content: str,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> Rocprofv3TimingEvidence:
    """Build derived profiler timing evidence from parsed CSV content."""
    return Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parse_rocprofv3_csv(csv_content),
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )


def read_overhead_calibration(
    calibration_path: Path | None,
) -> float | None:
    """Read profiler overhead calibration value from a JSON sidecar.

    Returns the ``overhead_ms`` value if the file exists and is valid,
    otherwise None. Never raises — logs warnings on parse errors.
    """
    if calibration_path is None or not calibration_path.exists():
        return None
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        overhead = payload.get("overhead_ms")
        return float(overhead) if overhead is not None else None
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to read overhead calibration from %s: %s", calibration_path, exc
        )
        return None


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
    *,
    application_command: Sequence[str],
    languages: Sequence[str],
    output_directory: Path,
    output_file: str,
    tool_version: str,
    gpu_architecture: str,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
    executable: str = ROCPROFV3_EXECUTABLE,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
) -> Rocprofv3CollectionResult:
    """Select source-specific timing policy and collect evidence when supported."""
    policy = timing_policy_for_languages(languages, profiler_available=True)
    request = Rocprofv3CollectionRequest(
        application_command=tuple(application_command),
        output_directory=output_directory,
        output_file=output_file,
        policy=policy,
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        executable=executable,
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
    )
    return collect_rocprofv3_timing(
        request,
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


def _normalize_header(header: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def _first_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return None


def _duration_ns(row: dict[str, str]) -> float | None:
    for key in (
        "durationns",
        "durationnsec",
        "durationnanoseconds",
        "duration",
    ):
        value = row.get(key)
        if value:
            return float(value)

    start = _first_numeric(row, "starttimestamp", "startns", "begin")
    end = _first_numeric(row, "endtimestamp", "endns", "end")
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def _first_numeric(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value:
            return float(value)
    return None


# Compatibility aliases for the old monolithic module's private helper names.
_read_overhead_calibration = read_overhead_calibration
_find_rocprofv3_csv = find_rocprofv3_csv
normalize_header = _normalize_header
first_value = _first_value
duration_ns = _duration_ns
first_numeric = _first_numeric
