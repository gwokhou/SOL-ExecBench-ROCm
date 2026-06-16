# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only profile summary sidecar contract."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.contract import SOL_EXECBENCH_CONTRACT_VERSION
from sol_execbench.core.dataset.checksums import sha256_file
from sol_execbench.core.trust_summary import utc_timestamp


PROFILE_SUMMARY_SCHEMA_VERSION = "sol_execbench.profile_summary.v1"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


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


class ProfileSummaryIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated profile summary sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_contract_version: str
    """SOL evaluator contract version used by the producer."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    run_id: str | None = None
    """Optional run identity."""


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
    )


def validate_profile_summary_freshness(
    sidecar: ProfileSummarySidecar,
    *,
    trace_path: str | None = None,
    sol_contract_version: str = SOL_EXECBENCH_CONTRACT_VERSION,
    run_id: str | None = None,
) -> ProfileSummaryFreshnessValidation:
    """Classify whether a profile summary identity matches expected run identity."""

    reasons: list[str] = []
    identity = sidecar.identity
    if identity.sol_contract_version != sol_contract_version:
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
    if profile_result.status == "success" and profile_result.artifacts:
        return ProfileSummaryStatus.AVAILABLE
    if profile_result.status == "success":
        # Succeeded but registered no artifacts: partial diagnostics, not missing.
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
    return ProfileSummaryContent(
        profiler_status=profile_result.status,
        profiler_available=profile_result.profiler_available,
        command=list(profile_result.command),
        output_file=profile_result.output_file,
        artifact_count=len(profile_result.artifacts),
        artifact_kinds=dict(sorted(artifact_kinds.items())),
        returncode=profile_result.returncode,
        timeout_seconds=profile_result.timeout_seconds,
        skipped_reason=profile_result.skipped_reason,
        failed_reason=profile_result.failed_reason,
        metrics=metrics,
    )


def _limitations(profile_result: Rocprofv3ProfileResult | None) -> list[str]:
    limitations = [
        "Profile summary is diagnostic adapter input only.",
        "Canonical Trace JSONL remains the authority for correctness, timing, scoring, and status.",
        "Raw rocprofv3 metadata remains in the separate profile sidecar when present.",
    ]
    if profile_result is None:
        limitations.append("No rocprofv3 profile result was supplied.")
    elif profile_result.status != "success":
        limitations.append(f"rocprofv3 profile status is {profile_result.status}.")
    elif not profile_result.artifacts:
        limitations.append("rocprofv3 profile completed without registered artifacts.")
    return limitations


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
