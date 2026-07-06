# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict profile summary content models."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class ProfileSummaryMetric(BaseModelWithDocstrings):
    """One bounded normalized profile metric."""

    model_config = _MODEL_CONFIG

    name: str
    """Stable metric name."""
    value: int | float | str | bool | None
    """JSON scalar metric value."""
    unit: str | None = None
    """Optional unit label."""
    source: str
    """Source label for this metric."""


class ProfileSummaryStructuredMetric(BaseModelWithDocstrings):
    """One bounded workload-level profile metric."""

    model_config = _MODEL_CONFIG

    name: str
    """Stable metric name."""
    value: int | float | str | bool | None
    """JSON scalar metric value."""
    unit: str | None = None
    """Optional unit label."""
    source: str
    """Metric source label."""
    workload_id: str | None = None
    """Optional workload identity."""
    artifact: str | None = None
    """Compact source artifact label."""
    parse_status: str = "available"
    """Bounded parse status for the source metric."""


class ProfileSummaryKernelMetric(BaseModelWithDocstrings):
    """One bounded kernel-level profile metric."""

    model_config = _MODEL_CONFIG

    kernel_name: str
    """Kernel or source bucket name."""
    name: str
    """Stable metric name."""
    value: int | float | str | bool | None
    """JSON scalar metric value."""
    unit: str | None = None
    """Optional unit label."""
    source: str
    """Metric source label."""
    artifact: str | None = None
    """Compact source artifact label."""
    parse_status: str = "available"
    """Bounded parse status for the source metric."""


class ProfileSummaryBottleneckHint(BaseModelWithDocstrings):
    """Conservative diagnostic bottleneck hint."""

    model_config = _MODEL_CONFIG

    category: Literal[
        "compute_bound",
        "memory_l2_bound",
        "lds_bound",
        "launch_overhead",
        "insufficient_counters",
        "unknown",
    ]
    """Closed diagnostic hint category."""
    severity: Literal["low", "medium", "high", "unknown"] = "low"
    """Conservative severity label."""
    confidence: Literal["low", "medium", "high"] = "low"
    """Confidence in this diagnostic hint."""
    message: str
    """Bounded human-readable diagnostic message."""
    source_metrics: list[str] = Field(default_factory=list)
    """Source metric names used to derive the hint."""
    evidence_artifacts: list[str] = Field(default_factory=list)
    """Compact artifact labels supporting the hint."""


class ProfileSummaryArtifactCitation(BaseModelWithDocstrings):
    """Compact profile-summary artifact citation."""

    model_config = _MODEL_CONFIG

    kind: str
    """Artifact kind such as trace, profile_metadata, or profiler_artifact."""
    label: str
    """Compact artifact label."""
    path: str | None = None
    """Compact path, normally a file name."""
    sha256: str | None = None
    """Artifact checksum when available."""
    status: str | None = None
    """Optional artifact status."""
    size_bytes: int | None = Field(default=None, ge=0)
    """Artifact size in bytes when available."""


class ProfileSummaryContent(BaseModelWithDocstrings):
    """Compact normalized profiler metadata summary."""

    model_config = _MODEL_CONFIG

    profiler_status: str | None = None
    """Raw profiler result status when available."""
    profiler_available: bool | None = None
    """Whether rocprofv3 was available to the producer."""
    artifact_coverage_status: str | None = None
    """Bounded profiler artifact coverage status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable profiler result reason codes."""
    warnings: list[str] = Field(default_factory=list)
    """Bounded profiler result warnings."""
    command: list[str] = Field(default_factory=list)
    """Profiler command with bounded argv values."""
    output_file: str | None = None
    """Profiler output-file prefix."""
    artifact_count: int = Field(ge=0)
    """Number of registered profiler artifacts."""
    artifact_kinds: dict[str, int] = Field(default_factory=dict)
    """Registered profiler artifact counts by kind."""
    returncode: int | None = None
    """Profiler command return code when available."""
    timeout_seconds: int | None = None
    """Profiler timeout when available."""
    skipped_reason: str | None = None
    """Bounded skipped reason."""
    failed_reason: str | None = None
    """Bounded failed reason."""
    metrics: list[ProfileSummaryMetric] = Field(default_factory=list)
    """Bounded normalized profile metrics derived from metadata."""
    workload_metrics: list[ProfileSummaryStructuredMetric] = Field(default_factory=list)
    """Structured workload-level profile metrics."""
    kernel_metrics: list[ProfileSummaryKernelMetric] = Field(default_factory=list)
    """Structured kernel-level profile metrics."""
    bottleneck_hints: list[ProfileSummaryBottleneckHint] = Field(default_factory=list)
    """Conservative diagnostic bottleneck hints."""
    parse_warnings: list[str] = Field(default_factory=list)
    """Bounded artifact parse warnings."""
