# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Cross-report consistency diagnostic sidecar helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.reports.consistency.checks import (
    dedupe_findings as _dedupe_findings,
    find_checksum_mismatches as _find_checksum_mismatches,
    find_claim_boundary_violations as _find_claim_boundary_violations,
    find_denominator_closure_drift as _find_denominator_closure_drift,
    find_matrix_runtime_attempted as _find_matrix_runtime_attempted,
    find_missing_derived_evidence_scored as _find_missing_derived_evidence_scored,
    mapping_or_none as _mapping_or_none,
    records as _records,
    source_ref as _source_ref,
)
from sol_execbench.core.reports.consistency.io import write_consistency_reports
from sol_execbench.core.reports.consistency.models import (
    CLAIM_BOUNDARY_TEXT,
    CONSISTENCY_REPORT_SCHEMA_VERSION,
    ConsistencyClaimBoundary,
    ConsistencyFinding,
    ConsistencyFindingTotals,
    ConsistencyReport,
    ConsistencySourceRef,
    ConsistencySummary,
)
from sol_execbench.core.reports.consistency.rendering import render_consistency_markdown
from sol_execbench.core.reports.trust_summary import load_json, utc_timestamp


def build_consistency_report(
    *,
    execution_closure: dict[str, Any] | None = None,
    paper_denominator: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    runtime_evidence: dict[str, Any] | None = None,
    static_evidence: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_report: dict[str, Any] | None = None,
    solar_derivation: dict[str, Any] | None = None,
    amd_bound_sanity: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> ConsistencyReport:
    """Build a deterministic cross-report consistency diagnostic report."""
    source_paths = source_paths or {}
    payloads = {
        "execution_closure": _mapping_or_none(execution_closure),
        "paper_denominator": _mapping_or_none(paper_denominator),
        "matrix_report": _mapping_or_none(matrix_report),
        "runtime_evidence": _mapping_or_none(runtime_evidence),
        "static_evidence": _mapping_or_none(static_evidence),
        "amd_score_report": _mapping_or_none(amd_score_report),
        "amd_sol_report": _mapping_or_none(amd_sol_report),
        "solar_derivation": _mapping_or_none(solar_derivation),
        "amd_bound_sanity": _mapping_or_none(amd_bound_sanity),
    }
    sources = [
        _source_ref(source_id, payload, source_paths.get(source_id))
        for source_id, payload in payloads.items()
        if payload is not None
    ]

    findings: list[ConsistencyFinding] = []
    closure_records = _records(execution_closure)
    denominator_workloads = _records(paper_denominator, key="workloads")
    findings.extend(
        _find_denominator_closure_drift(closure_records, denominator_workloads)
    )
    findings.extend(_find_matrix_runtime_attempted(matrix_report, closure_records))
    findings.extend(
        _find_missing_derived_evidence_scored(closure_records, amd_score_report)
    )
    findings.extend(_find_checksum_mismatches(payloads))
    findings.extend(_find_claim_boundary_violations(payloads))

    findings = _dedupe_findings(findings)
    totals = ConsistencyFindingTotals()
    for finding in findings:
        totals.add(finding.severity)

    report = ConsistencyReport(
        created_at=created_at or utc_timestamp(),
        sources=sorted(sources, key=lambda source: source.source_id),
        summary=ConsistencySummary(
            sources_checked=len(sources),
            findings_total=len(findings),
            finding_totals=totals,
        ),
        findings=findings,
    )
    return report.with_checksum()


__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "CONSISTENCY_REPORT_SCHEMA_VERSION",
    "ConsistencyClaimBoundary",
    "ConsistencyFinding",
    "ConsistencyFindingTotals",
    "ConsistencyReport",
    "ConsistencySourceRef",
    "ConsistencySummary",
    "build_consistency_report",
    "load_json",
    "render_consistency_markdown",
    "utc_timestamp",
    "write_consistency_reports",
]
