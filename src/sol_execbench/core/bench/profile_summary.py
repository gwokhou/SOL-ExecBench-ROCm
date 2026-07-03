# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only profile summary sidecar contract."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.contract import SOL_EXECBENCH_CONTRACT_VERSION
from sol_execbench.core.dataset.checksums import sha256_file
from sol_execbench.core.trust_summary import utc_timestamp


PROFILE_SUMMARY_SCHEMA_VERSION = "sol_execbench.profile_summary.v1"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PROFILE_SUMMARY_MAX_PARSE_BYTES = 1_000_000
_PROFILE_SUMMARY_MAX_ROWS = 10_000


class ProfileSummaryStatus(str, Enum):
    """Aggregate profile summary availability."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ProfileSummaryReasonCode(str, Enum):
    """Stable reason-code vocabulary for profile summary generation."""

    PROFILE_SUMMARY_GENERATED = "profile_summary_generated"
    PROFILE_PARTIAL = "profile_partial"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    NO_PROFILE_RESULT = "no_profile_result"


class ProfileSummaryFreshnessStatus(str, Enum):
    """Freshness validation status for a profile summary."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ProfileSummaryGovernanceStatus(str, Enum):
    """Diagnostic governance status for a profile summary sidecar."""

    USABLE_DIAGNOSTIC = "usable_diagnostic"
    STALE_DIAGNOSTIC = "stale_diagnostic"
    UNAVAILABLE = "unavailable"
    INVALID_DIAGNOSTIC = "invalid_diagnostic"


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


class ProfileSummaryIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated profile summary sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_contract_version: str
    """SOL evaluator contract version used by the producer."""
    sol_version: str | None = None
    """Preferred HIP-facing alias for the SOL contract version."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    run_id: str | None = None
    """Optional run identity."""

    @model_validator(mode="after")
    def _aliases_match(self) -> ProfileSummaryIdentity:
        if (
            self.sol_version is not None
            and self.sol_version != self.sol_contract_version
        ):
            raise ValueError("sol_version must match sol_contract_version")
        return self


class ProfileSummaryAuthority(BaseModelWithDocstrings):
    """Authority boundary for profile summary sidecars."""

    model_config = _MODEL_CONFIG

    diagnostic_only: Literal[True] = True
    correctness_authority: Literal[False] = False
    performance_authority: Literal[False] = False
    timing_authority: Literal[False] = False
    score_authority: Literal[False] = False
    evidence_tier_authority: Literal[False] = False
    confirmed_improvement_authority: Literal[False] = False
    release_gate_authority: Literal[False] = False
    cutover_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False
    claim_upgrade_authority: Literal[False] = False


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


class ProfileSummaryFreshnessValidation(BaseModelWithDocstrings):
    """Result of validating profile summary freshness identity."""

    model_config = _MODEL_CONFIG

    status: ProfileSummaryFreshnessStatus
    """Freshness result."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes explaining stale or unknown status."""


class ProfileSummaryGovernanceGuardrail(BaseModelWithDocstrings):
    """Authority boundary after optional profile-summary governance checks."""

    model_config = _MODEL_CONFIG

    status: ProfileSummaryGovernanceStatus
    """Diagnostic governance status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes for unavailable, stale, or invalid states."""
    diagnostic_only: Literal[True] = True
    correctness_authority: Literal[False] = False
    performance_authority: Literal[False] = False
    timing_authority: Literal[False] = False
    score_authority: Literal[False] = False
    evidence_tier_authority: Literal[False] = False
    confirmed_improvement_authority: Literal[False] = False
    release_gate_authority: Literal[False] = False
    cutover_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False
    claim_upgrade_authority: Literal[False] = False


class ProfileSummarySidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only sidecar for normalized profiler metadata."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.profile_summary.v1"] = (
        PROFILE_SUMMARY_SCHEMA_VERSION
    )
    status: ProfileSummaryStatus
    reason_code: ProfileSummaryReasonCode
    identity: ProfileSummaryIdentity
    authority: ProfileSummaryAuthority = Field(default_factory=ProfileSummaryAuthority)
    summary: ProfileSummaryContent
    limitations: list[str] = Field(default_factory=list)
    artifact_citations: list[ProfileSummaryArtifactCitation] = Field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""

        return self.model_dump(mode="json")


def build_profile_summary_sidecar(
    *,
    profile_result: Rocprofv3ProfileResult | None,
    trace_path: str | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    artifact_citations: Sequence[ProfileSummaryArtifactCitation] = (),
) -> ProfileSummarySidecar:
    """Build a bounded diagnostic profile summary from rocprofv3 metadata."""

    status = _status_for_profile_result(profile_result)
    reason_code = _reason_code_for_profile_result(profile_result, status)
    return ProfileSummarySidecar(
        status=status,
        reason_code=reason_code,
        identity=ProfileSummaryIdentity(
            generated_at=generated_at or utc_timestamp(),
            sol_contract_version=SOL_EXECBENCH_CONTRACT_VERSION,
            sol_version=SOL_EXECBENCH_CONTRACT_VERSION,
            trace_path=_compact_path(trace_path),
            run_id=run_id,
        ),
        summary=_profile_summary_content(profile_result),
        limitations=_limitations(profile_result),
        artifact_citations=list(artifact_citations),
    )


def profile_summary_artifact_citation_from_path(
    *,
    kind: str,
    path: Path,
    label: str | None = None,
    status: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ProfileSummaryArtifactCitation:
    """Build a compact citation from a profile-summary artifact path."""

    checksum = (
        sha256
        if sha256 is not None
        else (sha256_file(path) if path.is_file() else None)
    )
    return ProfileSummaryArtifactCitation(
        kind=kind,
        label=label or path.name,
        path=path.name,
        sha256=checksum,
        status=status,
        size_bytes=size_bytes,
    )


def validate_profile_summary_freshness(
    sidecar: ProfileSummarySidecar,
    *,
    trace_path: str | None = None,
    sol_contract_version: str = SOL_EXECBENCH_CONTRACT_VERSION,
    sol_version: str | None = None,
    run_id: str | None = None,
) -> ProfileSummaryFreshnessValidation:
    """Classify whether a profile summary identity matches expected run identity."""

    reasons: list[str] = []
    identity = sidecar.identity
    expected_sol_version = sol_version or sol_contract_version
    if (identity.sol_version or identity.sol_contract_version) != expected_sol_version:
        reasons.append("sol_contract_version_mismatch")
    _match_optional(
        reasons, "trace_path", identity.trace_path, _compact_path(trace_path)
    )
    _match_optional(reasons, "run_id", identity.run_id, run_id)
    if reasons:
        return ProfileSummaryFreshnessValidation(
            status=ProfileSummaryFreshnessStatus.STALE,
            reason_codes=reasons,
        )
    if trace_path is None and run_id is None:
        return ProfileSummaryFreshnessValidation(
            status=ProfileSummaryFreshnessStatus.UNKNOWN,
            reason_codes=["insufficient_expected_identity"],
        )
    return ProfileSummaryFreshnessValidation(
        status=ProfileSummaryFreshnessStatus.CURRENT
    )


def evaluate_profile_summary_governance(
    *,
    sidecar: ProfileSummarySidecar | None,
    freshness: ProfileSummaryFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> ProfileSummaryGovernanceGuardrail:
    """Return diagnostic-only governance state for an optional profile summary."""

    if parse_error is not None:
        return ProfileSummaryGovernanceGuardrail(
            status=ProfileSummaryGovernanceStatus.INVALID_DIAGNOSTIC,
            reason_codes=["sidecar_parse_error"],
        )
    if sidecar is None:
        return ProfileSummaryGovernanceGuardrail(
            status=ProfileSummaryGovernanceStatus.UNAVAILABLE,
            reason_codes=["sidecar_missing"],
        )
    if (
        freshness is not None
        and freshness.status == ProfileSummaryFreshnessStatus.STALE
    ):
        return ProfileSummaryGovernanceGuardrail(
            status=ProfileSummaryGovernanceStatus.STALE_DIAGNOSTIC,
            reason_codes=freshness.reason_codes or ["sidecar_stale"],
        )
    if (
        freshness is not None
        and freshness.status == ProfileSummaryFreshnessStatus.UNKNOWN
    ):
        return ProfileSummaryGovernanceGuardrail(
            status=ProfileSummaryGovernanceStatus.UNAVAILABLE,
            reason_codes=freshness.reason_codes or ["sidecar_freshness_unknown"],
        )
    return ProfileSummaryGovernanceGuardrail(
        status=ProfileSummaryGovernanceStatus.USABLE_DIAGNOSTIC,
    )


def _status_for_profile_result(
    profile_result: Rocprofv3ProfileResult | None,
) -> ProfileSummaryStatus:
    if profile_result is None:
        return ProfileSummaryStatus.UNAVAILABLE
    if profile_result.status == "success" and profile_result.has_profiler_data:
        return ProfileSummaryStatus.AVAILABLE
    if profile_result.status in {"success", "partial"}:
        # Successful process execution without profiler data is partial diagnostics,
        # not missing and not full profile availability.
        return ProfileSummaryStatus.PARTIAL
    if profile_result.status in {"failed", "unavailable"}:
        return ProfileSummaryStatus.PARTIAL
    return ProfileSummaryStatus.UNAVAILABLE


def _reason_code_for_profile_result(
    profile_result: Rocprofv3ProfileResult | None,
    status: ProfileSummaryStatus,
) -> ProfileSummaryReasonCode:
    if profile_result is None:
        return ProfileSummaryReasonCode.NO_PROFILE_RESULT
    if status == ProfileSummaryStatus.AVAILABLE:
        return ProfileSummaryReasonCode.PROFILE_SUMMARY_GENERATED
    if status == ProfileSummaryStatus.PARTIAL:
        return ProfileSummaryReasonCode.PROFILE_PARTIAL
    return ProfileSummaryReasonCode.PROFILE_UNAVAILABLE


def _profile_summary_content(
    profile_result: Rocprofv3ProfileResult | None,
) -> ProfileSummaryContent:
    if profile_result is None:
        return ProfileSummaryContent(artifact_count=0)
    artifact_kinds: dict[str, int] = {}
    total_size_bytes = 0
    for artifact in profile_result.artifacts:
        artifact_kinds[artifact.kind] = artifact_kinds.get(artifact.kind, 0) + 1
        total_size_bytes += artifact.size_bytes
    metrics = [
        ProfileSummaryMetric(
            name="artifact_count",
            value=len(profile_result.artifacts),
            unit="count",
            source="rocprofv3_profile_metadata",
        ),
        ProfileSummaryMetric(
            name="artifact_total_size_bytes",
            value=total_size_bytes,
            unit="bytes",
            source="rocprofv3_profile_metadata",
        ),
    ]
    structured = _structured_profile_evidence(profile_result)
    return ProfileSummaryContent(
        profiler_status=profile_result.status,
        profiler_available=profile_result.profiler_available,
        artifact_coverage_status=profile_result.artifact_coverage_status,
        reason_codes=list(profile_result.reason_codes),
        warnings=list(profile_result.warnings),
        command=list(profile_result.command),
        output_file=profile_result.output_file,
        artifact_count=len(profile_result.artifacts),
        artifact_kinds=dict(sorted(artifact_kinds.items())),
        returncode=profile_result.returncode,
        timeout_seconds=profile_result.timeout_seconds,
        skipped_reason=profile_result.skipped_reason,
        failed_reason=profile_result.failed_reason,
        metrics=metrics,
        workload_metrics=structured.workload_metrics,
        kernel_metrics=structured.kernel_metrics,
        bottleneck_hints=structured.bottleneck_hints,
        parse_warnings=structured.parse_warnings,
    )


def _limitations(profile_result: Rocprofv3ProfileResult | None) -> list[str]:
    limitations = [
        "Profile summary is diagnostic adapter input only.",
        "Canonical Trace JSONL remains the authority for correctness, timing, scoring, and status.",
        "Raw rocprofv3 metadata remains in the separate profile sidecar when present.",
    ]
    if profile_result is None:
        limitations.append("No rocprofv3 profile result was supplied.")
        return limitations
    if profile_result.status != "success":
        limitations.append(f"rocprofv3 profile status is {profile_result.status}.")
    elif not profile_result.has_profiler_data:
        if profile_result.artifact_coverage_status != "diagnostic_logs_only":
            limitations.append(
                "rocprofv3 profile completed without profiler data artifacts."
            )
    if profile_result.artifact_coverage_status == "diagnostic_logs_only":
        limitations.append(
            "rocprofv3 produced diagnostic logs but no profiler data artifacts."
        )
    return limitations


class _StructuredProfileEvidence(BaseModelWithDocstrings):
    """Internal structured profile evidence accumulator."""

    model_config = _MODEL_CONFIG

    workload_metrics: list[ProfileSummaryStructuredMetric] = Field(default_factory=list)
    kernel_metrics: list[ProfileSummaryKernelMetric] = Field(default_factory=list)
    bottleneck_hints: list[ProfileSummaryBottleneckHint] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


def _structured_profile_evidence(
    profile_result: Rocprofv3ProfileResult,
) -> _StructuredProfileEvidence:
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
                f"{artifact.path.name}: {artifact.kind} artifacts are citation-only in profile_summary.sidecar.v1"
            )
        elif artifact.kind == "other":
            parse_warnings.append(
                f"{artifact.path.name}: unsupported profiler artifact kind other"
            )

    return _StructuredProfileEvidence(
        workload_metrics=workload_metrics,
        kernel_metrics=kernel_metrics,
        bottleneck_hints=_derive_bottleneck_hints(kernel_metrics),
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


def _derive_bottleneck_hints(
    kernel_metrics: Sequence[ProfileSummaryKernelMetric],
) -> list[ProfileSummaryBottleneckHint]:
    counter_metrics = [
        metric for metric in kernel_metrics if metric.name != "kernel_duration_ms"
    ]
    if not counter_metrics:
        return [
            ProfileSummaryBottleneckHint(
                category="insufficient_counters",
                severity="low",
                confidence="high",
                message="No bounded counter artifact was available for bottleneck classification.",
            )
        ]

    by_name: dict[str, list[ProfileSummaryKernelMetric]] = {}
    for metric in counter_metrics:
        by_name.setdefault(_normalize_key(metric.name), []).append(metric)
    hints: list[ProfileSummaryBottleneckHint] = []
    l2_metrics = by_name.get("l2cachehitrate", [])
    if l2_metrics:
        low_l2 = [metric for metric in l2_metrics if _is_low_l2_hit_rate(metric)]
        if low_l2:
            hints.append(
                ProfileSummaryBottleneckHint(
                    category="memory_l2_bound",
                    severity="medium",
                    confidence="low",
                    message="Low L2 hit-rate counter suggests memory/L2 pressure.",
                    source_metrics=[metric.name for metric in low_l2],
                    evidence_artifacts=_metric_artifacts(low_l2),
                )
            )
    lds_metrics = by_name.get("ldsbankconflict", [])
    if lds_metrics and any(
        (_numeric_value(metric.value) or 0) > 0 for metric in lds_metrics
    ):
        hints.append(
            ProfileSummaryBottleneckHint(
                category="lds_bound",
                severity="low",
                confidence="low",
                message="LDS bank conflict counter is present and non-zero.",
                source_metrics=[metric.name for metric in lds_metrics],
                evidence_artifacts=_metric_artifacts(lds_metrics),
            )
        )
    valu_metrics = by_name.get("sqinstsvalu", [])
    if valu_metrics:
        hints.append(
            ProfileSummaryBottleneckHint(
                category="compute_bound",
                severity="low",
                confidence="low",
                message="VALU instruction counter is present without stronger memory or launch evidence.",
                source_metrics=[metric.name for metric in valu_metrics],
                evidence_artifacts=_metric_artifacts(valu_metrics),
            )
        )
    if hints:
        return hints
    return [
        ProfileSummaryBottleneckHint(
            category="unknown",
            severity="unknown",
            confidence="low",
            message="Counter artifact was parsed, but no conservative bottleneck rule matched.",
            source_metrics=[metric.name for metric in counter_metrics],
            evidence_artifacts=_metric_artifacts(counter_metrics),
        )
    ]


def _is_low_l2_hit_rate(metric: ProfileSummaryKernelMetric) -> bool:
    """Conservatively flag a low L2 hit rate, normalizing percent vs fraction units.

    rocprof can report L2_CACHE_HIT_RATE as a percent (0-100) or a fraction (0-1);
    a fixed ``< 60`` threshold would flag every fraction value. The unit hint
    selects the threshold, falling back to the value's magnitude.
    """
    value = _numeric_value(metric.value)
    if value is None:
        return False
    unit = _normalize_key(metric.unit)
    if unit in {"fraction", "ratio"}:
        return value < 0.6
    if unit in {"percent", "pct"}:
        return value < 60.0
    return value < 0.6 if value <= 1.0 else value < 60.0


def _metric_artifacts(metrics: Sequence[ProfileSummaryKernelMetric]) -> list[str]:
    artifacts = {metric.artifact for metric in metrics if metric.artifact is not None}
    return sorted(artifacts)


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


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _finite_or_none(float(value))
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return _finite_or_none(number)
    return None


def _compact_path(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).name


def _match_optional(
    reasons: list[str],
    field: str,
    actual: str | None,
    expected: str | None,
) -> None:
    if expected is not None and actual != expected:
        reasons.append(f"{field}_mismatch")
