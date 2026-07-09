# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Research trust summary sidecar helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.reports.trust_summary.io import (
    load_json,
    write_trust_summary_reports,
)
from sol_execbench.core.reports.trust_summary.models import (
    CLAIM_BOUNDARY_TEXT,
    TRUST_SUMMARY_SCHEMA_VERSION,
    TrustOutcome,
    TrustSourceRef,
    TrustSummaryClaimBoundary,
    TrustSummaryReport,
)
from sol_execbench.core.reports.trust_summary.outcomes import (
    claim_outcome as _claim_outcome,
    consistency_outcome as _consistency_outcome,
    evidence_outcome as _evidence_outcome,
    mapping_or_none as _mapping_or_none,
    overall_status as _overall_status,
    source_ref as _source_ref,
    stability_outcome as _stability_outcome,
)
from sol_execbench.core.reports.trust_summary.rendering import (
    render_trust_summary_markdown,
)
from sol_execbench.core.utils import utc_timestamp


def build_trust_summary_report(
    *,
    consistency_report: dict[str, Any] | None = None,
    evaluation_stability: dict[str, Any] | None = None,
    claim_upgrade: dict[str, Any] | None = None,
    execution_closure: dict[str, Any] | None = None,
    paper_denominator: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_report: dict[str, Any] | None = None,
    solar_derivation: dict[str, Any] | None = None,
    amd_bound_sanity: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> TrustSummaryReport:
    """Build a deterministic trust summary report."""
    source_paths = source_paths or {}
    payloads = {
        "consistency_report": _mapping_or_none(consistency_report),
        "evaluation_stability": _mapping_or_none(evaluation_stability),
        "claim_upgrade": _mapping_or_none(claim_upgrade),
        "execution_closure": _mapping_or_none(execution_closure),
        "paper_denominator": _mapping_or_none(paper_denominator),
        "matrix_report": _mapping_or_none(matrix_report),
        "amd_score_report": _mapping_or_none(amd_score_report),
        "amd_sol_report": _mapping_or_none(amd_sol_report),
        "solar_derivation": _mapping_or_none(solar_derivation),
        "amd_bound_sanity": _mapping_or_none(amd_bound_sanity),
    }
    outcomes = [
        _consistency_outcome(payloads["consistency_report"]),
        _stability_outcome(payloads["evaluation_stability"]),
        _claim_outcome(payloads["claim_upgrade"]),
        _evidence_outcome(
            execution_closure=payloads["execution_closure"],
            paper_denominator=payloads["paper_denominator"],
            matrix_report=payloads["matrix_report"],
            amd_score_report=payloads["amd_score_report"],
            amd_sol_report=payloads["amd_sol_report"],
            solar_derivation=payloads["solar_derivation"],
            amd_bound_sanity=payloads["amd_bound_sanity"],
        ),
    ]
    next_steps = sorted({step for outcome in outcomes for step in outcome.next_steps})
    report = TrustSummaryReport(
        created_at=created_at or utc_timestamp(),
        sources=[
            _source_ref(source_id, payload, source_paths.get(source_id))
            for source_id, payload in payloads.items()
            if payload is not None
        ],
        outcomes=outcomes,
        overall_status=_overall_status(outcomes),
        next_steps=next_steps,
    )
    return report.with_checksum()


__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "TRUST_SUMMARY_SCHEMA_VERSION",
    "TrustOutcome",
    "TrustSourceRef",
    "TrustSummaryClaimBoundary",
    "TrustSummaryReport",
    "build_trust_summary_report",
    "load_json",
    "render_trust_summary_markdown",
    "write_trust_summary_reports",
]
