# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded artifact parsing for profile summary sidecars."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.profile_summary.hints import derive_bottleneck_hints
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryBottleneckHint,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PROFILE_SUMMARY_MAX_PARSE_BYTES = 1_000_000
_PROFILE_SUMMARY_MAX_ROWS = 10_000


class StructuredProfileEvidence(BaseModelWithDocstrings):
    """Structured profile evidence parsed from bounded profiler artifacts."""

    model_config = _MODEL_CONFIG

    workload_metrics: list[ProfileSummaryStructuredMetric] = Field(default_factory=list)
    kernel_metrics: list[ProfileSummaryKernelMetric] = Field(default_factory=list)
    bottleneck_hints: list[ProfileSummaryBottleneckHint] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


def structured_profile_evidence(
    profile_result: Rocprofv3ProfileResult,
) -> StructuredProfileEvidence:
    """Parse bounded structured profile evidence from registered artifacts."""

    workload_id = _metadata_workload_id(profile_result)
    workload_metrics = [
        ProfileSummaryStructuredMetric(
            name="artifact_coverage_status",
            value=profile_result.artifact_coverage_status,
            source="rocprofv3_profile_metadata",
            workload_id=workload_id,
            parse_status="available",
        )
    ]
    kernel_metrics: list[ProfileSummaryKernelMetric] = []
    parse_warnings: list[str] = []

    for artifact in profile_result.artifacts:
        if artifact.kind == "trace_csv":
            parsed_metrics, warnings = _parse_trace_csv_artifact(artifact.path)
            kernel_metrics.extend(parsed_metrics)
            parse_warnings.extend(warnings)
        elif artifact.kind == "counter_csv":
            parsed_metrics, warnings = _parse_counter_csv_artifact(artifact.path)
            kernel_metrics.extend(parsed_metrics)
            parse_warnings.extend(warnings)
        elif artifact.kind == "agent_info_csv":
            _, warnings = _parse_limited_csv(artifact.path)
            parse_warnings.extend(warnings)
        elif artifact.kind == "metadata_json":
            parsed_metrics, warnings = _parse_metadata_json_artifact(
                artifact.path,
                workload_id=workload_id,
            )
            workload_metrics.extend(parsed_metrics)
            parse_warnings.extend(warnings)
        elif artifact.kind in {
            "diagnostic_json",
            "rocpd",
            "perfetto_trace",
            "otf2_trace",
        }:
            parse_warnings.append(
                f"{artifact.path.name}: {artifact.kind} artifacts are "
                "citation-only in sol_execbench.profile_summary.v2"
            )
        elif artifact.kind == "other":
            parse_warnings.append(
                f"{artifact.path.name}: unsupported profiler artifact kind other"
            )

    return StructuredProfileEvidence(
        workload_metrics=workload_metrics,
        kernel_metrics=kernel_metrics,
        bottleneck_hints=derive_bottleneck_hints(kernel_metrics),
        parse_warnings=parse_warnings,
    )


def _metadata_workload_id(profile_result: Rocprofv3ProfileResult) -> str | None:
    for artifact in profile_result.artifacts:
        if artifact.kind != "metadata_json" or not artifact.path.is_file():
            continue
        try:
            if artifact.path.stat().st_size > _PROFILE_SUMMARY_MAX_PARSE_BYTES:
                continue
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            value = payload.get("workload_id") or payload.get("workload")
            if isinstance(value, str):
                return value
    return None


def _parse_trace_csv_artifact(
    path: Path,
) -> tuple[list[ProfileSummaryKernelMetric], list[str]]:
    rows, warnings = _parse_limited_csv(path)
    metrics: list[ProfileSummaryKernelMetric] = []
    for row in rows:
        normalized = {_normalize_key(key): value for key, value in row.items()}
        domain = _first_text(normalized, "domain", "kind", "type", "category")
        if domain is None or "kernel" not in _normalize_key(domain):
            continue
        duration_ns = _first_number(
            normalized,
            "durationns",
            "durationnsec",
            "durationnanoseconds",
            "duration",
        )
        if duration_ns is None:
            continue
        kernel_name = _first_text(
            normalized,
            "name",
            "kernelname",
            "function",
            "operation",
        )
        metrics.append(
            ProfileSummaryKernelMetric(
                kernel_name=kernel_name or "unknown_kernel",
                name="kernel_duration_ms",
                value=duration_ns / 1_000_000.0,
                unit="ms",
                source=path.name,
                artifact=path.name,
            )
        )
    return metrics, warnings


def _parse_counter_csv_artifact(
    path: Path,
) -> tuple[list[ProfileSummaryKernelMetric], list[str]]:
    rows, warnings = _parse_limited_csv(path)
    metrics: list[ProfileSummaryKernelMetric] = []
    for row in rows:
        normalized = {_normalize_key(key): value for key, value in row.items()}
        name = _first_text(normalized, "metric", "name", "counter")
        value = _first_number(normalized, "value", "countervalue", "result")
        if name is None or value is None:
            continue
        unit = _first_text(normalized, "unit", "units")
        kernel_name = _first_text(normalized, "kernel", "kernelname") or path.stem
        metrics.append(
            ProfileSummaryKernelMetric(
                kernel_name=kernel_name,
                name=name,
                value=value,
                unit=unit,
                source=path.name,
                artifact=path.name,
            )
        )
    return metrics, warnings


def _parse_metadata_json_artifact(
    path: Path,
    *,
    workload_id: str | None,
) -> tuple[list[ProfileSummaryStructuredMetric], list[str]]:
    warnings = _artifact_parse_preflight(path)
    if warnings:
        return [], warnings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [], [f"{path.name}: malformed JSON profile metadata ({exc})"]
    if not isinstance(payload, dict):
        return [], [f"{path.name}: metadata JSON is not an object"]

    metrics: list[ProfileSummaryStructuredMetric] = []
    source_workload_id = workload_id
    for key, unit in (
        ("kernel_dispatches", "count"),
        ("dispatch_count", "count"),
        ("kernel_count", "count"),
    ):
        value = payload.get(key)
        if isinstance(value, int | float | str | bool) or value is None:
            number_or_value = _coerce_scalar(value)
            if number_or_value is None:
                continue
            metrics.append(
                ProfileSummaryStructuredMetric(
                    name="kernel_dispatch_count",
                    value=number_or_value,
                    unit=unit,
                    source=path.name,
                    workload_id=source_workload_id,
                    artifact=path.name,
                    parse_status="available",
                )
            )
            break
    return metrics, []


def _parse_limited_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    warnings = _artifact_parse_preflight(path)
    if warnings:
        return [], warnings
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, str]] = []
            for index, row in enumerate(reader):
                if index >= _PROFILE_SUMMARY_MAX_ROWS:
                    return rows, [
                        (
                            f"{path.name}: CSV row limit reached; metrics are "
                            f"derived from the first {_PROFILE_SUMMARY_MAX_ROWS} "
                            f"rows only and may be incomplete"
                        )
                    ]
                rows.append({key or "": value for key, value in row.items()})
            return rows, []
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [], [f"{path.name}: malformed CSV profile artifact ({exc})"]


def _artifact_parse_preflight(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path.name}: profiler artifact is missing"]
    try:
        if path.stat().st_size > _PROFILE_SUMMARY_MAX_PARSE_BYTES:
            return [f"{path.name}: profiler artifact exceeds parse byte limit"]
    except OSError as exc:
        return [f"{path.name}: profiler artifact stat failed ({exc})"]
    return []


def _normalize_key(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _first_text(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _first_number(row: dict[str, str], *keys: str) -> int | float | None:
    for key in keys:
        value = row.get(key)
        number = _coerce_scalar(value)
        if isinstance(number, int | float):
            return number
    return None


def _finite_or_none(value: int | float) -> int | float | None:
    """Pass through finite numbers; reject NaN/Inf so they never reach sidecar JSON."""

    return value if math.isfinite(value) else None


def _coerce_scalar(value: object) -> int | float | str | None:
    # bool is intentionally rejected: Python bool is an int subclass, but a JSON
    # `true` / CSV "true" is not a numeric metric value. Non-finite floats
    # (NaN/Inf) are also rejected so they never reach the sidecar JSON.
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return _finite_or_none(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        integer = int(stripped)
    except ValueError:
        pass
    else:
        return integer
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return _finite_or_none(number)
