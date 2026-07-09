# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Rollup helpers for paper denominator reports."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.dataset.paper_denominator.models import (
    PaperDenominatorProblem,
    PaperDenominatorRollup,
)


def _empty_rollup() -> PaperDenominatorRollup:
    return PaperDenominatorRollup()


def _category_rollup(
    categories: dict[str, PaperDenominatorRollup],
    category: str,
) -> PaperDenominatorRollup:
    return categories.setdefault(category, _empty_rollup())


def _problem_rollup(
    problems: dict[tuple[str, str], PaperDenominatorProblem],
    *,
    category: str,
    problem_id: str,
    problem_path: str | None,
) -> PaperDenominatorRollup:
    key = (category, problem_id)
    if key not in problems:
        problems[key] = PaperDenominatorProblem(
            category=category,
            problem_id=problem_id,
            problem_path=problem_path,
            rollup=_empty_rollup(),
        )
    return problems[key].rollup


def _record_ref(record: dict[str, Any]) -> str:
    problem = str(record.get("problem_id") or record.get("problem_path") or "unknown")
    uuid = record.get("workload_uuid")
    row = record.get("row_index")
    return f"{problem}#{uuid or row}"


def _readiness_state(status: str) -> str:
    lowered = status.lower()
    if lowered == "ready":
        return "ready"
    if "unsupported" in lowered:
        return "unsupported"
    return "blocked"


def _closure_state(status: str) -> str | None:
    if status == "skipped_existing_pass":
        return "skipped"
    if status == "missing_trace":
        return "attempted_failed"
    if status == "derived_evidence_missing":
        return "deferred"
    if status in {
        "attempted_passed",
        "attempted_failed",
        "filtered",
        "not_attempted",
    }:
        return status
    return None
