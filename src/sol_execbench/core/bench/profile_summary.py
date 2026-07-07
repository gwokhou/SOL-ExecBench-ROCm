# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only profile summary sidecar contract."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticSidecarAuthority,
    classify_diagnostic_governance,
    classify_freshness,
    compact_path,
    match_optional,
    match_required_optional,
)
from sol_execbench.core.bench.profile_summary_artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.profile_summary_models import (
    ProfileSummaryArtifactCitation,
    ProfileSummaryBottleneckHint,
    ProfileSummaryContent,
    ProfileSummaryKernelMetric,
    ProfileSummaryMetric,
    ProfileSummaryStructuredMetric,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.checksums import sha256_file
from sol_execbench.core.trust_summary import utc_timestamp


PROFILE_SUMMARY_SCHEMA_VERSION = "sol_execbench.profile_summary.v2"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PROFILE_SUMMARY_MODEL_EXPORTS = (
    ProfileSummaryBottleneckHint,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)


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


class ProfileSummaryIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated profile summary sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_version: str
    """Producer/runtime SOL version or HIP-facing supported SOL tag."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    run_id: str | None = None
    """Optional run identity."""


class ProfileSummaryFreshnessValidation(BaseModelWithDocstrings):
    """Result of validating profile summary freshness identity."""

    model_config = _MODEL_CONFIG

    status: ProfileSummaryFreshnessStatus
    """Freshness result."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes explaining stale or unknown status."""


class ProfileSummaryGovernanceGuardrail(DiagnosticSidecarAuthority):
    """Authority boundary after optional profile-summary governance checks."""

    status: ProfileSummaryGovernanceStatus
    """Diagnostic governance status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes for unavailable, stale, or invalid states."""


class ProfileSummarySidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only sidecar for normalized profiler metadata."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.profile_summary.v2"] = (
        PROFILE_SUMMARY_SCHEMA_VERSION
    )
    status: ProfileSummaryStatus
    reason_code: ProfileSummaryReasonCode
    identity: ProfileSummaryIdentity
    authority: Literal["diagnostic"] = "diagnostic"
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
            sol_version=SOL_EXECBENCH_RELEASE,
            trace_path=compact_path(trace_path),
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
    sol_version: str | None = None,
    run_id: str | None = None,
) -> ProfileSummaryFreshnessValidation:
    """Classify whether a profile summary identity matches expected run identity."""

    reasons: list[str] = []
    identity = sidecar.identity
    match_required_optional(reasons, "sol_version", identity.sol_version, sol_version)
    match_optional(reasons, "trace_path", identity.trace_path, compact_path(trace_path))
    match_optional(reasons, "run_id", identity.run_id, run_id)
    any_expected = (
        trace_path is not None or run_id is not None or sol_version is not None
    )
    status_value, reason_codes = classify_freshness(reasons, any_expected=any_expected)
    return ProfileSummaryFreshnessValidation(
        status=ProfileSummaryFreshnessStatus(status_value),
        reason_codes=reason_codes,
    )


def evaluate_profile_summary_governance(
    *,
    sidecar: ProfileSummarySidecar | None,
    freshness: ProfileSummaryFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> ProfileSummaryGovernanceGuardrail:
    """Return diagnostic-only governance state for an optional profile summary."""

    status_value, reason_codes = classify_diagnostic_governance(
        sidecar_present=sidecar is not None,
        freshness_status=(freshness.status.value if freshness is not None else None),
        freshness_reason_codes=(
            freshness.reason_codes if freshness is not None else None
        ),
        parse_error=parse_error,
    )
    return ProfileSummaryGovernanceGuardrail(
        status=ProfileSummaryGovernanceStatus(status_value),
        reason_codes=reason_codes,
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
    structured = structured_profile_evidence(profile_result)
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
