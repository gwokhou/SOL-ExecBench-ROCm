# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evaluation stability diagnostic sidecar helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median

from sol_execbench.core.reports.evaluation_stability.models import (
    SOURCE_CHECKSUM_KEYS,
    STABILITY_STATUS_PRIORITY,
    EvaluationStabilityInputs,
    EvaluationStabilityReport,
    RuntimeDistribution,
    StabilitySourceRef,
    StabilityStatusTotals,
    StabilityWorkload,
)
from sol_execbench.core.reports.trust_summary import utc_timestamp


def build_evaluation_stability_report(
    inputs: EvaluationStabilityInputs | None = None,
    *,
    timing_evidence: Sequence[Mapping[str, object]] | None = None,
    source_paths: Sequence[Path | None] | None = None,
    created_at: str | None = None,
    noise_cv_threshold: float = 0.10,
    min_samples: int = 3,
) -> EvaluationStabilityReport:
    """Build a timing-quality report from one explicit input request."""
    if inputs is not None:
        _reject_legacy_arguments(
            timing_evidence,
            source_paths,
            created_at,
            noise_cv_threshold,
            min_samples,
        )
    elif timing_evidence is None:
        raise TypeError("timing_evidence or EvaluationStabilityInputs is required")
    else:
        inputs = EvaluationStabilityInputs(
            timing_evidence=timing_evidence,
            source_paths=source_paths or (),
            created_at=created_at,
            noise_cv_threshold=noise_cv_threshold,
            min_samples=min_samples,
        )
    resolved_source_paths = inputs.source_paths or (None,) * len(inputs.timing_evidence)
    sources: list[StabilitySourceRef] = []
    workloads: list[StabilityWorkload] = []
    totals = StabilityStatusTotals()

    for index, payload in enumerate(inputs.timing_evidence):
        source_id = f"timing_{index + 1}"
        path = (
            resolved_source_paths[index] if index < len(resolved_source_paths) else None
        )
        sources.append(_source_ref(source_id, payload, path))
        workload = _build_workload(
            source_id,
            payload,
            noise_cv_threshold=inputs.noise_cv_threshold,
            min_samples=inputs.min_samples,
        )
        totals.add(workload.stability_status)
        workloads.append(workload)

    report = EvaluationStabilityReport(
        created_at=inputs.created_at or utc_timestamp(),
        sources=sources,
        status_totals=totals,
        workloads=workloads,
    )
    return report.with_checksum()


def _reject_legacy_arguments(
    timing_evidence: object,
    source_paths: object,
    created_at: object,
    noise_cv_threshold: float,
    min_samples: int,
) -> None:
    if (
        timing_evidence is not None
        or source_paths is not None
        or created_at is not None
        or noise_cv_threshold != 0.10
        or min_samples != 3
    ):
        raise TypeError(
            "pass either EvaluationStabilityInputs or legacy keyword arguments, not both"
        )


def _build_workload(
    source_id: str,
    payload: Mapping[str, object],
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
    payload: Mapping[str, object],
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


def _duration_samples(payload: Mapping[str, object]) -> list[float]:
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
            normalized_row = _string_keyed_mapping(row)
            if normalized_row is None:
                continue
            if normalized_row.get("is_kernel_activity") is False:
                continue
            duration = _positive_float(normalized_row.get("duration_ms"))
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
    payload: Mapping[str, object],
    path: Path | None,
) -> StabilitySourceRef:
    return StabilitySourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=_optional_str(payload.get("schema_version")),
        checksum=_checksum(payload),
    )


def _checksum(payload: Mapping[str, object]) -> str | None:
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        nested_payload = _string_keyed_mapping(value)
        if nested_payload is not None:
            checksum = nested_payload.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _workload_ref(payload: Mapping[str, object], fallback: str) -> str:
    for key in ("workload_uuid", "workload_id", "problem_id", "trace_ref"):
        value = payload.get(key)
        if value:
            return str(value)
    return fallback


def _trace_refs(payload: Mapping[str, object]) -> list[str]:
    refs = payload.get("source_trace_refs")
    if isinstance(refs, list):
        return [str(ref) for ref in refs if ref]
    ref = payload.get("trace_ref")
    return [str(ref)] if ref else []


def _synchronization_policy(payload: Mapping[str, object]) -> str:
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


def _string_keyed_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}
