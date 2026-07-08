# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.data.path_access import path_dict, path_get, path_mapping_list, path_str_or_none
from sol_execbench.core.dataset.manifest import utc_timestamp
from sol_execbench.core.dataset.parity_gap_models import (
    EVIDENCE_KEYS,
    EvidenceCompleteness,
    ParityGapCategory,
    ParityGapDenominators,
    ParityGapReport,
)
from sol_execbench.core.dataset.parity_gap_records import (
    _add_blocker,
    _amd_score_record,
    _category_counts,
    _execution_closure_record,
    _final_blockers,
    _readiness_workload_record,
    _record_ref,
    _score_category,
    _source,
)


def build_parity_gap_report(
    *,
    manifest: dict[str, Any] | None,
    inventory: dict[str, Any],
    readiness: dict[str, Any],
    ready_subset: dict[str, Any] | None,
    execution_closure: dict[str, Any],
    amd_score_report: dict[str, Any] | None = None,
    source_paths: dict[str, Path] | None = None,
    created_at: str | None = None,
) -> ParityGapReport:
    source_paths = source_paths or {}
    categories: dict[str, ParityGapDenominators] = {}
    blockers: dict[str, dict[str, Any]] = {}
    workload_to_category: dict[str, str] = {}

    for category in path_mapping_list(inventory, "categories"):
        category_name = path_str_or_none(category, "name") or "unknown"
        counts = _category_counts(category_name, categories)
        denoms = path_dict(category, "denominators")
        counts.discovered += int(
            path_get(denoms, "discovered_problems", default=0) or 0
        )
        counts.parsed += int(path_get(denoms, "parsed_problems", default=0) or 0)

    for workload_payload in path_mapping_list(readiness, "workloads"):
        workload = _readiness_workload_record(workload_payload)
        counts = _category_counts(workload.category, categories)
        if workload.workload_uuid:
            workload_to_category[workload.workload_uuid] = workload.category
        if workload.status == "ready":
            counts.ready += 1
        else:
            counts.blocked += 1
            for reason in workload.reasons:
                _add_blocker(
                    blockers,
                    reason_code=str(
                        path_get(reason, "code", default="unknown_readiness_blocker")
                    ),
                    category=workload.category,
                    example_ref=_record_ref(workload_payload),
                    next_action=str(
                        path_get(
                            reason,
                            "next_action",
                            default="Review readiness blocker.",
                        )
                    ),
                )

    evidence_present = dict.fromkeys(EVIDENCE_KEYS, 0)
    evidence_missing = dict.fromkeys(EVIDENCE_KEYS, 0)
    for record_payload in path_mapping_list(execution_closure, "records"):
        record = _execution_closure_record(record_payload)
        counts = _category_counts(record.category, categories)
        status = record.closure_status
        if record.workload_uuid:
            workload_to_category[record.workload_uuid] = record.category
        if status == "not_attempted":
            counts.not_attempted += 1
        if status == "filtered":
            continue
        if status == "skipped_existing_pass":
            counts.skipped += 1
        if status in {
            "attempted_passed",
            "attempted_failed",
            "missing_trace",
            "derived_evidence_missing",
        }:
            counts.attempted += 1
        if status == "attempted_passed" or record.trace_status == "PASSED":
            counts.passed += 1
        if status in {"attempted_failed", "missing_trace"} or (
            record.trace_status not in {None, "PASSED"}
        ):
            counts.failed += 1
        if record.trace_ref:
            evidence_present["trace"] += 1
        elif status not in {"filtered", "not_attempted"}:
            evidence_missing["trace"] += 1
        evidence_map = {
            "timing": "timing_evidence",
            "amd_native_score": "amd_score",
            "amd_sol": "amd_sol_bound",
            "solar_derivation": "solar_derivation",
        }
        for evidence_key, ref_key in evidence_map.items():
            if record.evidence_refs.get(ref_key):
                evidence_present[evidence_key] += 1
        for gap in record.evidence_gaps:
            if gap.startswith("timing"):
                evidence_missing["timing"] += 1
            elif gap.startswith("amd_score"):
                evidence_missing["amd_native_score"] += 1
            elif gap.startswith("amd_sol"):
                evidence_missing["amd_sol"] += 1
            elif gap.startswith("solar"):
                evidence_missing["solar_derivation"] += 1
            _add_blocker(
                blockers,
                reason_code=gap,
                category=record.category,
                example_ref=_record_ref(record_payload),
                next_action="Generate or attach the missing derived evidence artifact.",
            )
        if status in {"attempted_failed", "missing_trace"}:
            _add_blocker(
                blockers,
                reason_code=status,
                category=record.category,
                example_ref=_record_ref(record_payload),
                next_action="Inspect CLI logs, trace output, and runner environment.",
            )

    if amd_score_report is not None:
        for score_payload in path_mapping_list(amd_score_report, "scores"):
            score = _amd_score_record(score_payload)
            counts = _category_counts(
                _score_category(score, workload_to_category), categories
            )
            warnings = " ".join(score.warnings)
            if score.supported and "degraded" in warnings.lower():
                counts.degraded += 1
            elif score.supported:
                counts.scored += 1
            else:
                counts.unscored += 1
            if score.evidence_refs.get("timing"):
                evidence_present["timing"] += 1
            if score.evidence_refs.get("sol_bound"):
                evidence_present["amd_sol"] += 1
            if score.derived_evidence_refs.get(
                "formula"
            ) or score.derived_evidence_refs.get("coverage"):
                evidence_present["solar_derivation"] += 1
            evidence_present["amd_native_score"] += 1

    suite = ParityGapDenominators()
    category_reports = []
    for name in sorted(categories):
        suite.merge(categories[name])
        category_reports.append(
            ParityGapCategory(name=name, denominators=categories[name])
        )

    sources = {
        "manifest": _source(
            manifest,
            path=source_paths.get("manifest"),
            checksum_key="manifest_checksum",
        ),
        "inventory": _source(
            inventory,
            path=source_paths.get("inventory"),
            checksum_key="inventory_checksum",
        ),
        "readiness": _source(
            readiness,
            path=source_paths.get("readiness"),
            checksum_key="readiness_checksum",
        ),
        "ready_subset": _source(
            ready_subset,
            path=source_paths.get("ready_subset"),
            checksum_key="ready_subset_checksum",
        ),
        "execution_closure": _source(
            execution_closure, path=source_paths.get("execution_closure")
        ),
        "amd_score_report": _source(
            amd_score_report, path=source_paths.get("amd_score_report")
        ),
    }

    report = ParityGapReport(
        created_at=created_at or utc_timestamp(),
        sources=sources,
        suite=suite,
        categories=category_reports,
        blockers=_final_blockers(blockers),
        evidence_completeness=EvidenceCompleteness(
            present=evidence_present,
            missing=evidence_missing,
        ),
    )
    return report.with_checksum()
