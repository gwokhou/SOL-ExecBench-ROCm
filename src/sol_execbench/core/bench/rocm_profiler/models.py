# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler request, result, and evidence models."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
)
from sol_execbench.core.evidence import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS
from sol_execbench.core.text_utils import text_tail


ROCPROFV3_EXECUTABLE = "rocprofv3"
ROCPROFV3_EVIDENCE_SCHEMA_VERSION = SCHEMA_VERSIONS["rocprofv3_timing"]
ROCPROFV3_PROFILE_SCHEMA_VERSION = SCHEMA_VERSIONS["rocprofv3_profile"]
ROCPROF_REASON_ARTIFACTS_REGISTERED = "rocprof_artifacts_registered"
ROCPROF_REASON_NO_REGISTERED_ARTIFACTS = "rocprof_no_registered_artifacts"
ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED = "rocprof_diagnostic_log_registered"
ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE = "rocprof_partial_artifact_coverage"
ROCPROF_REASON_COMMAND_FAILED = "rocprof_command_failed"
ROCPROF_REASON_COMMAND_TIMEOUT = "rocprof_command_timeout"
ROCPROF_REASON_UNAVAILABLE = "rocprof_unavailable"
# Artifact kinds that are diagnostic/opaque only and never count as profiler data.
# Single source of truth; profile_summary.py reuses this for status/limitations.
_NON_DATA_ARTIFACT_KINDS = frozenset({"diagnostic_json", "other"})
ROCPROF_WARNING_NO_PROFILER_DATA_ARTIFACTS = (
    "rocprofv3 returned success but produced no profiler data artifacts"
)
ROCPROF_WARNING_INCOMPLETE_ARTIFACT_COVERAGE = (
    "rocprofv3 registered artifacts, but coverage is incomplete or only opaque "
    "artifacts were discovered"
)


class Rocprofv3ProfileStatus(StrEnum):
    """Closed lifecycle states for optional profiler collection."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


_PROFILE_ARTIFACT_SUFFIXES = {
    ".csv",
    ".db",
    ".json",
    ".otf2",
    ".pftrace",
    ".rocpd",
    ".sqlite",
    ".sqlite3",
    ".trace",
}
_PROFILE_OUTPUT_DIR_NAMES = {
    "rocprofiler",
    "rocprofiler-sdk",
    "rocprofv3",
    "rocprofv3-results",
    "roctracer",
}


@dataclass(frozen=True, slots=True)
class Rocprofv3ProfileArtifact:
    """One profiler artifact registered from a `rocprofv3` output directory."""

    path: Path
    kind: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "path": str(self.path),
            "kind": self.kind,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Rocprofv3ProfileRequest:
    """Request to collect optional diagnostic `rocprofv3` artifacts."""

    application_command: tuple[str, ...]
    output_directory: Path
    output_file: str
    working_directory: Path | None = None
    executable: str = ROCPROFV3_EXECUTABLE
    include_hip_runtime: bool = True
    output_format: str = "rocpd"
    timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class Rocprofv3ProfileResult:
    """Result metadata for optional `rocprofv3` artifact collection."""

    status: Rocprofv3ProfileStatus
    command: tuple[str, ...]
    output_directory: Path
    output_file: str
    artifacts: tuple[Rocprofv3ProfileArtifact, ...] = ()
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None
    failed_reason: str | None = None
    working_directory: Path | None = None
    timeout_seconds: int | None = None
    profiler_available: bool | None = None
    artifact_coverage_status: str | None = None
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    output_format: str | None = None
    profiler_data_artifacts: bool = False
    output_directory_listing: tuple[str, ...] = ()
    schema_version: str = field(init=False, default=ROCPROFV3_PROFILE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Reject contradictory status and reason combinations."""
        if self.status is Rocprofv3ProfileStatus.SUCCESS:
            if self.skipped_reason is not None or self.failed_reason is not None:
                raise ValueError("successful profiling cannot include failure reasons")
        elif self.status is Rocprofv3ProfileStatus.UNAVAILABLE:
            if self.skipped_reason is None or self.failed_reason is not None:
                raise ValueError("unavailable profiling requires only a skipped reason")
        elif self.status is Rocprofv3ProfileStatus.FAILED:
            if self.failed_reason is None or self.skipped_reason is not None:
                raise ValueError("failed profiling requires only a failed reason")
        elif self.skipped_reason is not None:
            raise ValueError("partial profiling cannot include a skipped reason")

    @property
    def succeeded(self) -> bool:
        """Whether profiler collection completed with registered artifacts."""
        return self.status is Rocprofv3ProfileStatus.SUCCESS

    @property
    def has_profiler_data(self) -> bool:
        """Whether any registered artifact carries profiler data.

        Diagnostic/opaque kinds (``diagnostic_json``, ``other``) do not count.
        Centralizes the data-vs-diagnostic classification so consumers
        (e.g. profile_summary) read this instead of re-deriving it.
        """
        return has_profiler_data_artifact(self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic sidecar payload."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "diagnostic_only": True,
            "score_authority": False,
            "command": list(self.command),
            "working_directory": (
                str(self.working_directory)
                if self.working_directory is not None
                else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "output_directory": str(self.output_directory),
            "output_file": self.output_file,
            "profiler_available": self.profiler_available,
            "artifact_coverage_status": self.artifact_coverage_status,
            "output_format": self.output_format,
            "profiler_data_artifacts": self.profiler_data_artifacts,
            "output_directory_listing": list(self.output_directory_listing),
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "returncode": self.returncode,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
            "skipped_reason": self.skipped_reason,
            "failed_reason": self.failed_reason,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row payload."""
        return {
            "name": self.name,
            "domain": self.domain,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ms,
            "is_kernel_activity": self.is_kernel_activity,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True, slots=True)
class Rocprofv3TimingEvidence:
    """Derived profiler timing evidence for one profiled command."""

    tool_version: str
    gpu_architecture: str
    activity_domain: TimingActivityDomain
    aggregation_rule: str
    backend: TimingBackend
    interpretation: str
    parsed_rows: tuple[Rocprofv3TimingRow, ...]
    warmup_runs: int | None = None
    iterations: int | None = None
    min_measurement_time_seconds: float | None = None
    trial_count: int | None = None
    clock_locked: bool | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    profiler_overhead_ms: float | None = None
    schema_version: str = field(init=False, default=ROCPROFV3_EVIDENCE_SCHEMA_VERSION)
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def kernel_duration_ms(self) -> float:
        """Aggregate kernel activity duration in milliseconds."""
        return sum(
            row.duration_ms for row in self.parsed_rows if row.is_kernel_activity
        )

    def to_dict(self) -> dict[str, Any]:
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
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
            "min_measurement_time_seconds": self.min_measurement_time_seconds,
            "trial_count": self.trial_count,
            "clock_locked": self.clock_locked,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "profiler_overhead_ms": self.profiler_overhead_ms,
            "kernel_duration_ms": self.kernel_duration_ms,
            "parsed_rows": [row.to_dict() for row in self.parsed_rows],
        }


@dataclass(frozen=True, slots=True)
class DefaultTimingSelection:
    """Policy-aware decision for profiler-backed timing or fallback."""

    policy: TimingPolicy
    profiler_backed: bool
    fallback_applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable selection payload."""
        return {
            "policy": self.policy.to_dict(),
            "profiler_backed": self.profiler_backed,
            "fallback_applied": self.fallback_applied,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Rocprofv3CollectionRequest:
    """Request to collect live profiler timing evidence for one command."""

    application_command: tuple[str, ...]
    output_directory: Path
    output_file: str
    policy: TimingPolicy
    tool_version: str
    gpu_architecture: str
    executable: str = ROCPROFV3_EXECUTABLE
    include_hip_runtime: bool = True
    warmup_runs: int | None = None
    iterations: int | None = None
    min_measurement_time_seconds: float | None = None
    trial_count: int | None = None
    clock_locked: bool | None = None
    compact_rows: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceTimingRequest:
    """Immutable source-level inputs used to select a timing policy."""

    application_command: tuple[str, ...]
    languages: tuple[str, ...]
    output_directory: Path
    output_file: str
    tool_version: str
    gpu_architecture: str
    executable: str = ROCPROFV3_EXECUTABLE
    warmup_runs: int | None = None
    iterations: int | None = None
    min_measurement_time_seconds: float | None = None
    trial_count: int | None = None
    clock_locked: bool | None = None


@dataclass(frozen=True, slots=True)
class Rocprofv3CollectionResult:
    """Result of live profiler collection or explicit fallback routing."""

    evidence: Rocprofv3TimingEvidence | None
    selection: DefaultTimingSelection
    command: tuple[str, ...] = ()
    csv_path: Path | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        """Keep evidence presence consistent with fallback routing."""
        if self.evidence is None and not self.selection.fallback_applied:
            raise ValueError("profiler-backed selection requires timing evidence")
        if self.evidence is not None and (
            self.selection.fallback_applied or not self.selection.profiler_backed
        ):
            raise ValueError("fallback selection cannot include profiler evidence")

    @property
    def profiler_collected(self) -> bool:
        """Whether live profiler evidence was collected and parsed."""
        return self.evidence is not None and not self.selection.fallback_applied

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable collection result payload."""
        return {
            "profiler_collected": self.profiler_collected,
            "selection": self.selection.to_dict(),
            "command": list(self.command),
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


def _tail(text: str, *, max_chars: int = 4096) -> str:
    return text_tail(text, limit=max_chars)


def _normalize_header(header: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def is_profiler_data_artifact(artifact: Rocprofv3ProfileArtifact) -> bool:
    return artifact.kind not in _NON_DATA_ARTIFACT_KINDS


def has_profiler_data_artifact(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
) -> bool:
    return any(is_profiler_data_artifact(artifact) for artifact in artifacts)
