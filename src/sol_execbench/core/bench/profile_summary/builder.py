# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Profile summary sidecar construction."""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.bench.diagnostic_sidecar import compact_path
from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryArtifactCitation,
    ProfileSummaryContent,
    ProfileSummaryMetric,
)
from sol_execbench.core.bench.profile_summary.sidecar_models import (
    ProfileSummaryIdentity,
    ProfileSummaryReasonCode,
    ProfileSummarySidecar,
    ProfileSummaryStatus,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.data.contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.timestamps import utc_timestamp


def build_profile_summary_sidecar(
    *,
    profile_result: Rocprofv3ProfileResult | None,
    trace_path: str | None = None,
    run_id: str | None = None,
    sol_version: str | None = None,
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
            sol_version=sol_version or SOL_EXECBENCH_RELEASE,
            trace_path=compact_path(trace_path),
            run_id=run_id,
        ),
        summary=_profile_summary_content(profile_result),
        limitations=_limitations(profile_result),
        artifact_citations=list(artifact_citations),
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
