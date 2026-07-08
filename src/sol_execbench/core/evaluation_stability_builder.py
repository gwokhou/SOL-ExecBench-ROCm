# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evaluation stability diagnostic sidecar helpers."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any

from sol_execbench.core.evaluation_stability_models import (
    SOURCE_CHECKSUM_KEYS,
    STABILITY_STATUS_PRIORITY,
    EvaluationStabilityReport,
    RuntimeDistribution,
    StabilitySourceRef,
    StabilityStatusTotals,
    StabilityWorkload,
)
from sol_execbench.core.trust_summary import utc_timestamp


def build_evaluation_stability_report(
    *,
    timing_evidence: list[dict[str, Any]],
    source_paths: list[Path | None] | None = None,
    created_at: str | None = None,
    noise_cv_threshold: float = 0.10,
    min_samples: int = 3,
) -> EvaluationStabilityReport:
    source_paths = source_paths or [None] * len(timing_evidence)
    sources: list[StabilitySourceRef] = []
    workloads: list[StabilityWorkload] = []
    totals = StabilityStatusTotals()

    for index, payload in enumerate(timing_evidence):
        source_id = f"timing_{index + 1}"
        path = source_paths[index] if index < len(source_paths) else None
        sources.append(_source_ref(source_id, payload, path))
        workload = _build_workload(
            source_id,
            payload,
            noise_cv_threshold=noise_cv_threshold,
            min_samples=min_samples,
        )
        totals.add(workload.stability_status)
        workloads.append(workload)

    report = EvaluationStabilityReport(
        created_at=created_at or utc_timestamp(),
        sources=sources,
        status_totals=totals,
        workloads=workloads,
    )
    return report.with_checksum()


def _build_workload(
    source_id: str,
    payload: dict[str, Any],
    *,
    noise_cv_threshold: float,
    min_samples: int,
) -> StabilityWorkload:
    samples = _duration_samples(payload)
    distribution = _runtime_distribution(samples)
    reasons = _reason_codes(payload, distribution, noise_cv_threshold, min_samples)
    status = _primary_status(reasons)
    return StabilityWorkload(
        source_id=source_id,
        workload_ref=_workload_ref(payload, source_id),
        backend=_optional_str(payload.get("backend")),
        activity_domain=_optional_str(payload.get("activity_domain")),
        warmup_runs=_optional_int(payload.get("warmup_runs")),
        measured_repeat_count=distribution.count,
        clock_locked=_optional_bool(payload.get("clock_locked")),
        synchronization_policy=_synchronization_policy(payload),
        stability_status=status,
        reason_codes=reasons,
        runtime_distribution=distribution,
        source_trace_refs=_trace_refs(payload),
    )


def _reason_codes(
    payload: dict[str, Any],
    distribution: RuntimeDistribution,
    noise_cv_threshold: float,
    min_samples: int,
) -> list[str]:
    reasons: list[str] = []
    backend = str(payload.get("backend", "")).lower()
    activity_domain = str(payload.get("activity_domain", "")).lower()
    if backend == "unsupported" or activity_domain == "unsupported":
        reasons.append("backend_unsupported")
    if distribution.count == 0:
        reasons.append("missing_timing")
    elif distribution.count < min_samples:
        reasons.append("insufficient_samples")
    if payload.get("clock_locked") is False:
        reasons.append("clock_unlocked")
    # Detect concurrent GPU processes
    concurrent_processes = payload.get("concurrent_gpu_processes")
    if isinstance(concurrent_processes, list) and concurrent_processes:
        reasons.append("gpu_contention")
    # Detect PID lock contention
    if payload.get("pid_lock_contention") is True:
        reasons.append("multi_instance_interference")
    if payload.get("fallback_applied") is True or backend in {
        "device_events",
        "pytorch_profiler",
    }:
        reasons.append("profiler_overhead_risk")
    if (
        distribution.coefficient_of_variation is not None
        and distribution.coefficient_of_variation > noise_cv_threshold
    ):
        reasons.append("noisy")
    if not reasons:
        reasons.append("stable")
    return reasons


def _primary_status(reasons: list[str]) -> str:
    for status in STABILITY_STATUS_PRIORITY:
        if status in reasons:
            return status
    return "stable"


def _duration_samples(payload: dict[str, Any]) -> list[float]:
    for key in ("runtime_ms_distribution", "durations_ms", "measurements_ms"):
        value = payload.get(key)
        if isinstance(value, list):
            samples: list[float] = []
            for item in value:
                sample = _positive_float(item)
                if sample is not None:
                    samples.append(sample)
            return samples

    rows = payload.get("parsed_rows")
    if isinstance(rows, list):
        durations = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("is_kernel_activity") is False:
                continue
            duration = _positive_float(row.get("duration_ms"))
            if duration is not None:
                durations.append(duration)
        if durations:
            return durations

    duration = _positive_float(
        payload.get("kernel_duration_ms") or payload.get("runtime_ms")
    )
    return [duration] if duration is not None else []


def _runtime_distribution(samples: list[float]) -> RuntimeDistribution:
    if not samples:
        return RuntimeDistribution()
    ordered = sorted(samples)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    stddev = math.sqrt(variance)
    med = float(median(ordered))
    cv = stddev / mean if mean else None
    return RuntimeDistribution(
        count=count,
        min_ms=ordered[0],
        max_ms=ordered[-1],
        mean_ms=mean,
        median_ms=med,
        population_stddev_ms=stddev,
        coefficient_of_variation=cv,
        selected_runtime_ms=med,
    )


def _source_ref(
    source_id: str,
    payload: dict[str, Any],
    path: Path | None,
) -> StabilitySourceRef:
    return StabilitySourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=_optional_str(payload.get("schema_version")),
        checksum=_checksum(payload),
    )


def _checksum(payload: dict[str, Any]) -> str | None:
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            checksum = value.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _workload_ref(payload: dict[str, Any], fallback: str) -> str:
    for key in ("workload_uuid", "workload_id", "problem_id", "trace_ref"):
        value = payload.get(key)
        if value:
            return str(value)
    return fallback


def _trace_refs(payload: dict[str, Any]) -> list[str]:
    refs = payload.get("source_trace_refs")
    if isinstance(refs, list):
        return [str(ref) for ref in refs if ref]
    ref = payload.get("trace_ref")
    return [str(ref)] if ref else []


def _synchronization_policy(payload: dict[str, Any]) -> str:
    value = payload.get("synchronization_policy")
    if isinstance(value, str):
        return value
    backend = str(payload.get("backend", "")).lower()
    if backend == "rocprofv3":
        return "profiler-completed-command"
    if backend == "device_events":
        return "hip-backed-device-event-synchronize"
    return "unspecified"


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
