# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler command and evidence helpers."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
    select_timing_policy,
)
from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT


ROCPROFV3_EXECUTABLE = "rocprofv3"
ROCPROFV3_EVIDENCE_SCHEMA_VERSION = "sol_execbench.rocprofv3_timing.v1"


@dataclass(frozen=True)
class Rocprofv3TimingRow:
    """One normalized timing row parsed from `rocprofv3` CSV output."""

    name: str
    domain: str
    duration_ns: float
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000.0

    @property
    def is_kernel_activity(self) -> bool:
        """Whether this row represents kernel activity."""
        normalized = _normalize_header(self.domain)
        return "kernel" in normalized

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row payload."""
        return {
            "name": self.name,
            "domain": self.domain,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ms,
            "is_kernel_activity": self.is_kernel_activity,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class Rocprofv3TimingEvidence:
    """Derived profiler timing evidence for one profiled command."""

    tool_version: str
    gpu_architecture: str
    activity_domain: TimingActivityDomain
    aggregation_rule: str
    backend: TimingBackend
    interpretation: str
    parsed_rows: tuple[Rocprofv3TimingRow, ...]
    fallback_applied: bool = False
    fallback_reason: str | None = None
    schema_version: str = ROCPROFV3_EVIDENCE_SCHEMA_VERSION
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def kernel_duration_ms(self) -> float:
        """Aggregate kernel activity duration in milliseconds."""
        return sum(row.duration_ms for row in self.parsed_rows if row.is_kernel_activity)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence payload."""
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "canonical_output": self.canonical_output,
            "tool_version": self.tool_version,
            "gpu_architecture": self.gpu_architecture,
            "activity_domain": self.activity_domain.value,
            "aggregation_rule": self.aggregation_rule,
            "backend": self.backend.value,
            "interpretation": self.interpretation,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "kernel_duration_ms": self.kernel_duration_ms,
            "parsed_rows": [row.to_dict() for row in self.parsed_rows],
        }


@dataclass(frozen=True)
class DefaultTimingSelection:
    """Policy-aware decision for profiler-backed timing or fallback."""

    policy: TimingPolicy
    profiler_backed: bool
    fallback_applied: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable selection payload."""
        return {
            "policy": self.policy.to_dict(),
            "profiler_backed": self.profiler_backed,
            "fallback_applied": self.fallback_applied,
            "reason": self.reason,
        }


def build_rocprofv3_command(
    application_command: Sequence[str],
    *,
    output_directory: str,
    output_file: str,
    executable: str = ROCPROFV3_EXECUTABLE,
    include_hip_runtime: bool = True,
) -> list[str]:
    """Build a `rocprofv3` command for kernel timing evidence collection."""
    if not application_command:
        raise ValueError("application_command must not be empty")

    command = [
        executable,
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        output_directory,
        "--output-file",
        output_file,
    ]
    if include_hip_runtime:
        command.insert(2, "--hip-runtime-trace")
    return [*command, "--", *application_command]


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

    fallback_policy = select_timing_policy(
        policy.source_type,
        profiler_available=False,
    )
    if policy.backend != TimingBackend.ROCPROFV3:
        reason = (
            f"selected policy backend is {policy.backend.value}, not rocprofv3 "
            "kernel activity timing"
        )
    else:
        reason = "rocprofv3 is unavailable for the selected timing policy"
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


def build_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_content: str,
    tool_version: str,
    gpu_architecture: str,
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
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
    )


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
