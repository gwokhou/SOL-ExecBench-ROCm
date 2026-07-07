# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Final assembly for paper denominator reports."""

from __future__ import annotations

from sol_execbench.core.dataset.manifest import utc_timestamp
from sol_execbench.core.dataset.paper_denominator_evidence import (
    _next_hints,
    _sorted_evidence_gaps,
    _sorted_reason_buckets,
)
from sol_execbench.core.dataset.paper_denominator_models import (
    PaperDenominatorCategory,
    PaperDenominatorReport,
    PaperDenominatorSources,
)
from sol_execbench.core.dataset.paper_denominator_rollups import _empty_rollup
from sol_execbench.core.dataset.paper_denominator_sources import _artifact_source, _source
from sol_execbench.core.dataset.paper_denominator_state import PaperDenominatorBuildState


def assemble_paper_denominator_report(
    state: PaperDenominatorBuildState,
    *,
    created_at: str | None = None,
) -> PaperDenominatorReport:
    suite = _empty_rollup()
    category_reports = []
    for name in sorted(state.categories):
        suite.merge(state.categories[name])
        category_reports.append(
            PaperDenominatorCategory(name=name, rollup=state.categories[name])
        )

    reason_buckets = _sorted_reason_buckets(state.reason_groups)
    report = PaperDenominatorReport(
        created_at=created_at or utc_timestamp(),
        sources=PaperDenominatorSources(
            manifest=_source(
                state.manifest,
                path=state.source_paths.get("manifest"),
                checksum_keys=("manifest_checksum",),
            ),
            inventory=_source(
                state.inventory,
                path=state.source_paths.get("inventory"),
                checksum_keys=("inventory_checksum",),
            ),
            readiness=_source(
                state.readiness,
                path=state.source_paths.get("readiness"),
                checksum_keys=("readiness_checksum",),
            ),
            ready_subset=_source(
                state.ready_subset,
                path=state.source_paths.get("ready_subset"),
                checksum_keys=("ready_subset_checksum",),
            ),
            execution_closure=_source(
                state.execution_closure,
                path=state.source_paths.get("execution_closure"),
                checksum_keys=("execution_closure_checksum",),
            ),
            amd_score_report=_source(
                state.amd_score_report,
                path=state.source_paths.get("amd_score_report"),
                checksum_keys=(
                    "amd_native_score_checksum",
                    "amd_score_checksum",
                    "report_checksum",
                ),
            ),
            amd_sol_artifacts=sorted(
                [_artifact_source(artifact) for artifact in state.amd_sol_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            solar_artifacts=sorted(
                [_artifact_source(artifact) for artifact in state.solar_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
        ),
        suite=suite,
        categories=category_reports,
        problems=sorted(
            state.problems.values(),
            key=lambda problem: (
                problem.category,
                problem.problem_id,
                problem.problem_path or "",
            ),
        ),
        workloads=sorted(
            state.workloads.values(),
            key=lambda workload: (
                workload.category,
                workload.problem_id,
                workload.row_index if workload.row_index is not None else -1,
                workload.workload_uuid or "",
            ),
        ),
        reason_buckets=reason_buckets,
        evidence_gaps=_sorted_evidence_gaps(state.evidence_groups),
        next_evidence_hints=_next_hints(reason_buckets),
    )
    return report.with_checksum()
