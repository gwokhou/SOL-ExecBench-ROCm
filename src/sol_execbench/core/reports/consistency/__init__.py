# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Cross-report consistency diagnostic sidecar helpers."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

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
    ConsistencyInputs,
    ConsistencyReport,
    ConsistencySourceRef,
    ConsistencySummary,
)
from sol_execbench.core.reports.consistency.rendering import render_consistency_markdown
from sol_execbench.core.reports.trust_summary import load_json
from sol_execbench.core.timestamps import utc_timestamp as _utc_timestamp


def build_consistency_report(
    inputs: ConsistencyInputs | None = None,
    *,
    execution_closure: Mapping[str, object] | None = None,
    paper_denominator: Mapping[str, object] | None = None,
    matrix_report: Mapping[str, object] | None = None,
    runtime_evidence: Mapping[str, object] | None = None,
    static_evidence: Mapping[str, object] | None = None,
    amd_score_report: Mapping[str, object] | None = None,
    amd_sol_report: Mapping[str, object] | None = None,
    solar_derivation: Mapping[str, object] | None = None,
    amd_bound_sanity: Mapping[str, object] | None = None,
    source_paths: Mapping[str, Path | None] | None = None,
    created_at: str | None = None,
) -> ConsistencyReport:
    """Build a deterministic cross-report consistency diagnostic report."""
    if inputs is not None:
        _reject_legacy_arguments(
            execution_closure,
            paper_denominator,
            matrix_report,
            runtime_evidence,
            static_evidence,
            amd_score_report,
            amd_sol_report,
            solar_derivation,
            amd_bound_sanity,
            source_paths,
            created_at,
        )
    else:
        inputs = ConsistencyInputs(
            execution_closure=execution_closure,
            paper_denominator=paper_denominator,
            matrix_report=matrix_report,
            runtime_evidence=runtime_evidence,
            static_evidence=static_evidence,
            amd_score_report=amd_score_report,
            amd_sol_report=amd_sol_report,
            solar_derivation=solar_derivation,
            amd_bound_sanity=amd_bound_sanity,
            source_paths=source_paths or {},
            created_at=created_at,
        )
    return _build_consistency_report(inputs)


def _build_consistency_report(inputs: ConsistencyInputs) -> ConsistencyReport:
    """Evaluate normalized report inputs without legacy call-shape concerns."""
    payloads = {
        "execution_closure": _mapping_or_none(inputs.execution_closure),
        "paper_denominator": _mapping_or_none(inputs.paper_denominator),
        "matrix_report": _mapping_or_none(inputs.matrix_report),
        "runtime_evidence": _mapping_or_none(inputs.runtime_evidence),
        "static_evidence": _mapping_or_none(inputs.static_evidence),
        "amd_score_report": _mapping_or_none(inputs.amd_score_report),
        "amd_sol_report": _mapping_or_none(inputs.amd_sol_report),
        "solar_derivation": _mapping_or_none(inputs.solar_derivation),
        "amd_bound_sanity": _mapping_or_none(inputs.amd_bound_sanity),
    }
    sources = [
        _source_ref(source_id, payload, inputs.source_paths.get(source_id))
        for source_id, payload in payloads.items()
        if payload is not None
    ]

    findings: list[ConsistencyFinding] = []
    closure_records = _records(payloads["execution_closure"])
    denominator_workloads = _records(payloads["paper_denominator"], key="workloads")
    findings.extend(
        _find_denominator_closure_drift(closure_records, denominator_workloads)
    )
    findings.extend(
        _find_matrix_runtime_attempted(payloads["matrix_report"], closure_records)
    )
    findings.extend(
        _find_missing_derived_evidence_scored(
            closure_records, payloads["amd_score_report"]
        )
    )
    findings.extend(_find_checksum_mismatches(payloads))
    findings.extend(_find_claim_boundary_violations(payloads))

    findings = _dedupe_findings(findings)
    totals = ConsistencyFindingTotals()
    for finding in findings:
        totals.add(finding.severity)

    report = ConsistencyReport(
        created_at=inputs.created_at or _utc_timestamp(),
        sources=sorted(sources, key=lambda source: source.source_id),
        summary=ConsistencySummary(
            sources_checked=len(sources),
            findings_total=len(findings),
            finding_totals=totals,
        ),
        findings=findings,
    )
    return report.with_checksum()


def _reject_legacy_arguments(*values: object) -> None:
    if any(value is not None for value in values):
        raise TypeError(
            "pass either ConsistencyInputs or legacy keyword arguments, not both"
        )


__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "CONSISTENCY_REPORT_SCHEMA_VERSION",
    "ConsistencyClaimBoundary",
    "ConsistencyFinding",
    "ConsistencyFindingTotals",
    "ConsistencyInputs",
    "ConsistencyReport",
    "ConsistencySourceRef",
    "ConsistencySummary",
    "build_consistency_report",
    "load_json",
    "render_consistency_markdown",
    "write_consistency_reports",
]
