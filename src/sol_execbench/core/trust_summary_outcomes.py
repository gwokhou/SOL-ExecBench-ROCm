# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Outcome builders for trust summary reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_int,
    path_mapping_list,
    path_str_or_none,
)
from sol_execbench.core.report_payloads import report_source_view
from sol_execbench.core.trust_summary_models import TrustOutcome, TrustSourceRef

SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "amd_sol_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)


def consistency_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    """Build the consistency outcome."""
    if payload is None:
        return TrustOutcome(
            key="internally_consistent",
            status="evidence_missing",
            reason_codes=["consistency_report_missing"],
            next_steps=["Generate consistency_report.v1."],
        )
    blockers = finding_total(payload, "blocker")
    status = "internally_consistent" if blockers == 0 else "blocked"
    return TrustOutcome(
        key="internally_consistent",
        status=status,
        reason_codes=[
            "no_consistency_blockers" if blockers == 0 else "consistency_blockers"
        ],
        next_steps=[] if blockers == 0 else ["Resolve consistency blocker findings."],
    )


def stability_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    """Build the evaluation stability outcome."""
    if payload is None:
        return TrustOutcome(
            key="stable_enough_to_interpret",
            status="evidence_missing",
            reason_codes=["evaluation_stability_missing"],
            next_steps=["Generate evaluation_stability.v1."],
        )
    totals = path_dict(payload, "status_totals")
    risky = [
        key
        for key in (
            "noisy",
            "insufficient_samples",
            "missing_timing",
            "clock_unlocked",
            "profiler_overhead_risk",
            "backend_unsupported",
        )
        if path_int(totals, key, default=0) > 0
    ]
    return TrustOutcome(
        key="stable_enough_to_interpret",
        status="stable_enough"
        if not risky and path_int(totals, "stable", default=0) > 0
        else "blocked",
        reason_codes=["stable_timing"] if not risky else risky,
        next_steps=[]
        if not risky
        else [
            "Collect stable timing evidence with documented clock and backend policy."
        ],
    )


def claim_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    """Build the claim-upgrade outcome."""
    if payload is None:
        return TrustOutcome(
            key="claim_upgrade",
            status="diagnostic_only",
            reason_codes=["claim_upgrade_missing"],
            next_steps=["Generate claim_upgrade.v1."],
        )
    highest = str(path_get(payload, "highest_eligible_claim") or "diagnostic_only")
    status = "claim_upgrade_blocked" if highest == "diagnostic_only" else highest
    return TrustOutcome(
        key="claim_upgrade",
        status=status,
        reason_codes=[f"highest:{highest}"],
        next_steps=[]
        if highest != "diagnostic_only"
        else ["Satisfy claim-upgrade prerequisites before stronger claims."],
    )


def evidence_outcome(
    *,
    execution_closure: dict[str, Any] | None,
    paper_denominator: dict[str, Any] | None,
    matrix_report: dict[str, Any] | None,
    amd_score_report: dict[str, Any] | None,
    amd_sol_report: dict[str, Any] | None,
    solar_derivation: dict[str, Any] | None,
    amd_bound_sanity: dict[str, Any] | None,
) -> TrustOutcome:
    """Build the evidence completeness outcome."""
    missing = [
        name
        for name, payload in {
            "execution_closure": execution_closure,
            "paper_denominator": paper_denominator,
            "matrix_report": matrix_report,
            "amd_score_report": amd_score_report,
            "amd_sol_report": amd_sol_report,
            "solar_derivation": solar_derivation,
            "amd_bound_sanity": amd_bound_sanity,
        }.items()
        if payload is None
    ]
    next_steps = [f"Provide {name} evidence." for name in missing]
    next_steps.append(
        "Future CDNA3-family validation, including MI300X (gfx942), "
        "native-host validation, and paper-scale validation need explicit hardware evidence."
    )
    return TrustOutcome(
        key="evidence_completeness",
        status="evidence_missing" if missing else "reviewable",
        reason_codes=[f"missing:{name}" for name in missing]
        or ["required_refs_present"],
        next_steps=next_steps if missing else next_steps[-1:],
    )


def overall_status(outcomes: list[TrustOutcome]) -> str:
    """Aggregate outcome statuses into the report status."""
    statuses = {outcome.status for outcome in outcomes}
    if "blocked" in statuses or "claim_upgrade_blocked" in statuses:
        return "claim_upgrade_blocked"
    if "evidence_missing" in statuses:
        return "evidence_missing"
    return "reviewable"


def finding_total(payload: dict[str, Any], severity: str) -> int:
    """Return a finding count from summary totals or finding records."""
    totals = path_dict(payload, "summary.finding_totals")
    if totals:
        return path_int(totals, severity, default=0)
    return sum(
        1
        for finding in path_mapping_list(payload, "findings")
        if path_get(finding, "severity") == severity
    )


def source_ref(
    source_id: str, payload: dict[str, Any], path: object | None
) -> TrustSourceRef:
    """Build a source reference with schema and checksum metadata."""
    source = report_source_view(payload, source_name=source_id)
    return TrustSourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=source.schema_version,
        checksum=source.checksum or checksum(payload),
    )


def checksum(payload: dict[str, Any]) -> str | None:
    """Extract the first known checksum field from a source payload."""
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            checksum_value = path_str_or_none(value, "value")
            if checksum_value is not None:
                return checksum_value
        if isinstance(value, str):
            return value
    return None


def mapping_or_none(payload: object) -> dict[str, Any] | None:
    """Return a string-keyed mapping or None for non-object payloads."""
    if not isinstance(payload, Mapping):
        return None
    return {str(key): value for key, value in payload.items()}
