# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Freshness and governance checks for profile summary sidecars."""

from __future__ import annotations

from sol_execbench.core.bench.diagnostic_sidecar import (
    classify_diagnostic_governance,
    classify_freshness,
    compact_path,
    match_optional,
    match_required_optional,
)
from sol_execbench.core.bench.profile_summary.sidecar_models import (
    ProfileSummaryFreshnessValidation,
    ProfileSummaryGovernanceGuardrail,
    ProfileSummarySidecar,
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
    status, reason_codes = classify_freshness(reasons, any_expected=any_expected)
    return ProfileSummaryFreshnessValidation(
        status=status,
        reason_codes=reason_codes,
    )


def evaluate_profile_summary_governance(
    *,
    sidecar: ProfileSummarySidecar | None,
    freshness: ProfileSummaryFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> ProfileSummaryGovernanceGuardrail:
    """Return diagnostic-only governance state for an optional profile summary."""

    status, reason_codes = classify_diagnostic_governance(
        sidecar_present=sidecar is not None,
        freshness_status=freshness.status if freshness is not None else None,
        freshness_reason_codes=(
            freshness.reason_codes if freshness is not None else None
        ),
        parse_error=parse_error,
    )
    return ProfileSummaryGovernanceGuardrail(
        status=status,
        reason_codes=reason_codes,
    )
