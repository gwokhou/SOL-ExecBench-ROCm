# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_mapping_list,
    path_str_or_none,
)
from sol_execbench.core.dataset.profiler_timing_coverage_models import (
    OOM_LOG_MARKERS,
    ProfilerTimingEvidenceSummary,
)


def _find_problem_timing_evidence(
    category: str,
    problem_path: str,
    evidence_dirs: list[Path],
) -> ProfilerTimingEvidenceSummary | None:
    problem_name = Path(problem_path).name
    candidates = [
        evidence_dir / category / f"{problem_name}.timing.json"
        for evidence_dir in evidence_dirs
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        return _load_timing_evidence_summary(candidate)
    return None


def _load_timing_evidence_summary(path: Path) -> ProfilerTimingEvidenceSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _profiler_timing_summary_from_payload(payload, path=path.as_posix())
    if summary is None:
        raise ValueError(f"timing evidence must be a JSON object: {path}")
    return summary


def _profiler_timing_summary_from_payload(
    payload: object,
    *,
    path: str,
) -> ProfilerTimingEvidenceSummary | None:
    if not isinstance(payload, Mapping):
        return None

    evidence = path_dict(payload, "evidence")
    selection = path_dict(payload, "selection")
    policy = path_dict(selection, "policy")
    metadata = path_dict(payload, "replacement_metadata")
    backend = path_get(evidence, "backend") or path_get(policy, "backend")
    activity_domain = path_get(evidence, "activity_domain") or path_get(
        policy, "activity_domain"
    )
    csv_path = path_get(payload, "csv_path")
    failure_reason = path_get(metadata, "failure_reason")
    fallback_reason = path_get(selection, "reason")
    return ProfilerTimingEvidenceSummary(
        path=path,
        profiler_collected=path_get(payload, "profiler_collected") is True,
        backend=str(backend) if backend is not None else None,
        activity_domain=str(activity_domain) if activity_domain is not None else None,
        csv_path=str(csv_path) if csv_path else None,
        kernel_duration_ms=_float_or_none(path_get(evidence, "kernel_duration_ms")),
        kernel_activity_rows=_kernel_activity_rows(evidence),
        full_workload_coverage=_full_workload_coverage(metadata),
        profiled_workload_count=_int_or_none(
            path_get(metadata, "profiled_workload_count")
        ),
        expected_workload_count=_int_or_none(
            path_get(metadata, "expected_workload_count")
        ),
        trace_status_counts=_trace_status_counts(metadata),
        replacement_failure_reason=str(failure_reason)
        if failure_reason is not None
        else None,
        fallback_reason=str(fallback_reason) if fallback_reason is not None else None,
        blocker_class=_blocker_class(payload),
        reference_override=_reference_override(metadata),
    )


def _has_profiler_backed_kernel_activity(
    evidence: ProfilerTimingEvidenceSummary,
) -> bool:
    return (
        evidence.profiler_collected
        and evidence.backend == "rocprofv3"
        and evidence.kernel_activity_rows > 0
        and evidence.full_workload_coverage
    )


def _has_partial_profiler_kernel_activity(
    evidence: ProfilerTimingEvidenceSummary,
) -> bool:
    return (
        evidence.profiler_collected
        and evidence.backend == "rocprofv3"
        and evidence.kernel_activity_rows > 0
        and not evidence.full_workload_coverage
    )


def _is_profiler_replacement_attempt(evidence: ProfilerTimingEvidenceSummary) -> bool:
    return (
        evidence.backend == "rocprofv3"
        or evidence.profiled_workload_count is not None
        or evidence.expected_workload_count is not None
    )


def _full_workload_coverage(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return True
    return metadata.get("full_workload_coverage") is True


def _trace_status_counts(metadata: dict[str, Any]) -> dict[str, int]:
    counts = path_dict(metadata, "trace_status_counts")
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        count = _int_or_none(value)
        if isinstance(key, str) and count is not None:
            normalized[key] = count
    return dict(sorted(normalized.items()))


def _reference_override(metadata: dict[str, Any]) -> dict[str, Any] | None:
    reference_override = path_dict(metadata, "reference_override")
    if not reference_override:
        return None
    return {
        str(key): value
        for key, value in sorted(
            reference_override.items(), key=lambda item: str(item[0])
        )
    }


def _blocker_class(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    normalized_payload: dict[str, Any] = {
        str(key): value for key, value in payload.items()
    }
    details = _failure_trace_details(normalized_payload)
    if not any(detail.get("oom_detected") is True for detail in details):
        return None
    oom_phases = {
        str(detail.get("phase"))
        for detail in details
        if detail.get("oom_detected") is True
    }
    metadata = path_dict(normalized_payload, "replacement_metadata")
    trace_counts = _trace_status_counts(metadata)
    source_workloads = _source_workloads(normalized_payload)
    has_profiler_gap = "PROFILER_BLOCKED" in trace_counts or any(
        workload.get("status") == "profiler_blocked" for workload in source_workloads
    )
    if "correctness" in oom_phases:
        return "profiler_closure_oom_blocked"
    if has_profiler_gap:
        return "memory_oom_with_profiler_gap"
    if "gen_inputs" in oom_phases:
        return "gen_inputs_oom_blocked"
    if "user_function" in oom_phases:
        return "user_solution_oom"
    return "reference_oom_blocked"


def _failure_trace_details(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_workloads = _source_workloads(payload)
    if source_workloads:
        details: list[dict[str, Any]] = []
        for workload in source_workloads:
            replacement_path = workload.get("replacement_path")
            slice_payload = (
                _load_optional_json(Path(replacement_path))
                if isinstance(replacement_path, str)
                else None
            )
            details.extend(_payload_failure_details(slice_payload))
        return details
    return _payload_failure_details(payload)


def _source_workloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return path_mapping_list(payload, "evidence.source_workloads")


def _payload_failure_details(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    stdout = path_str_or_none(payload, "stdout")
    details: list[dict[str, Any]] = []
    if stdout is not None:
        for line in stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            evaluation = path_dict(record, "evaluation")
            status = path_get(evaluation, "status")
            if status == "PASSED" or not isinstance(status, str):
                continue
            log = str(path_get(evaluation, "log") or "")
            details.append(
                {
                    "status": status,
                    "phase": _failure_phase(log),
                    "oom_detected": _is_oom_log(log),
                }
            )
    stderr = path_str_or_none(payload, "stderr")
    if stderr is not None and _is_oom_log(stderr):
        details.append(
            {
                "status": "PROFILER_BLOCKED",
                "phase": _failure_phase(stderr),
                "oom_detected": True,
            }
        )
    return details


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _is_oom_log(log: str) -> bool:
    return any(marker in log for marker in OOM_LOG_MARKERS)


def _failure_phase(log: str) -> str:
    if log.startswith("gen_inputs failed:"):
        return "gen_inputs"
    if log.startswith("Reference run() failed:"):
        return "reference"
    if log.startswith("User function failed:"):
        return "user_function"
    if "compute_error_stats" in log or "correctness.py" in log:
        return "correctness"
    return "unknown"


def _is_memory_oom_blocker(blocker_class: str | None) -> bool:
    return blocker_class in {
        "reference_oom_blocked",
        "gen_inputs_oom_blocked",
        "user_solution_oom",
        "profiler_closure_oom_blocked",
        "memory_oom_with_profiler_gap",
        "reference_oom_with_profiler_gap",
    }


def _kernel_activity_rows(evidence: dict[str, Any]) -> int:
    parsed_rows = path_mapping_list(evidence, "parsed_rows")
    if parsed_rows:
        return sum(
            1 for row in parsed_rows if path_get(row, "is_kernel_activity") is True
        )
    duration = _float_or_none(path_get(evidence, "kernel_duration_ms"))
    return 1 if duration is not None and duration > 0 else 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
