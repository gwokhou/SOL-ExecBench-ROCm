# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Build stages for paper denominator reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sol_execbench.core.data.path_access import (
    path_get,
    path_mapping_list,
    path_str_or_none,
)
from sol_execbench.core.dataset.paper_denominator.evidence import (
    _add_missing_evidence,
    _add_reason,
)
from sol_execbench.core.dataset.paper_denominator.models import (
    DENOMINATOR_STATE_KEYS,
    REQUIRED_RECORD_EVIDENCE_REFS,
    PaperDenominatorStateTotals,
    PaperDenominatorWorkload,
)
from sol_execbench.core.dataset.paper_denominator.rollups import (
    _category_rollup,
    _closure_state,
    _problem_rollup,
    _readiness_state,
    _record_ref,
)
from sol_execbench.core.dataset.paper_denominator.state import (
    PaperDenominatorBuildState,
)


@dataclass(frozen=True)
class InventoryProblemRecord:
    category: str
    problem_id: str
    problem_path: str | None
    workloads: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReadinessWorkloadRecord:
    category: str
    problem_id: str
    problem_path: str | None
    workload_uuid: str | None
    row_index: int | None
    status: str
    reasons: list[dict[str, Any]] = field(default_factory=list)


def _inventory_problem_record(payload: object) -> InventoryProblemRecord:
    problem_path = path_str_or_none(payload, "problem_path")
    problem_id = path_str_or_none(payload, "problem_id") or problem_path or "unknown"
    workloads = path_mapping_list(payload, "workloads")
    return InventoryProblemRecord(
        category=path_str_or_none(payload, "category") or "unknown",
        problem_id=problem_id,
        problem_path=problem_path,
        workloads=workloads,
    )


def _readiness_workload_record(payload: object) -> ReadinessWorkloadRecord:
    problem_path = path_str_or_none(payload, "problem_path")
    problem_id = path_str_or_none(payload, "problem_id") or problem_path or "unknown"
    reasons = path_mapping_list(payload, "reasons")
    return ReadinessWorkloadRecord(
        category=path_str_or_none(payload, "category") or "unknown",
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=path_str_or_none(payload, "workload_uuid"),
        row_index=_optional_int(path_get(payload, "row_index")),
        status=path_str_or_none(payload, "status") or "blocked",
        reasons=reasons,
    )


def seed_inventory(state: PaperDenominatorBuildState) -> None:
    for category in state.inventory.get("categories", []):
        category_name = str(category.get("name", "unknown"))
        _category_rollup(state.categories, category_name)

    for problem_payload in state.inventory.get("problems", []):
        problem = _inventory_problem_record(problem_payload)
        _problem_rollup(
            state.problems,
            category=problem.category,
            problem_id=problem.problem_id,
            problem_path=problem.problem_path,
        )
        for workload_record in problem.workloads:
            workload_uuid = workload_record.get("uuid")
            key = (
                problem.problem_id,
                workload_record.get("row_index"),
                str(workload_uuid) if workload_uuid else None,
            )
            state.workloads.setdefault(
                key,
                PaperDenominatorWorkload(
                    category=problem.category,
                    problem_id=problem.problem_id,
                    problem_path=problem.problem_path,
                    workload_uuid=str(workload_uuid) if workload_uuid else None,
                    row_index=workload_record.get("row_index"),
                ),
            )


def merge_readiness(state: PaperDenominatorBuildState) -> None:
    for record_payload in state.readiness.get("workloads", []):
        record = _readiness_workload_record(record_payload)
        denominator_state = _readiness_state(record.status)
        example_ref = _record_ref(
            record_payload if isinstance(record_payload, Mapping) else {}
        )
        for rollup in (
            _category_rollup(state.categories, record.category),
            _problem_rollup(
                state.problems,
                category=record.category,
                problem_id=record.problem_id,
                problem_path=record.problem_path,
            ),
        ):
            rollup.workloads += 1
            rollup.states.add(denominator_state)
        for reason in record.reasons:
            code = str(reason.get("code", f"{denominator_state}_readiness"))
            _add_reason(
                state.reason_groups,
                reason_code=code,
                state=denominator_state,
                example_ref=example_ref,
                next_evidence=reason.get("next_action"),
            )
        key = (
            record.problem_id,
            record.row_index,
            record.workload_uuid,
        )
        workload = state.workloads.setdefault(
            key,
            PaperDenominatorWorkload(
                category=record.category,
                problem_id=record.problem_id,
                problem_path=record.problem_path,
                workload_uuid=record.workload_uuid,
                row_index=record.row_index,
            ),
        )
        workload.category = record.category
        workload.problem_id = record.problem_id
        workload.problem_path = record.problem_path
        workload.workload_uuid = record.workload_uuid
        workload.row_index = record.row_index
        workload.readiness_status = record.status
        workload.states = PaperDenominatorStateTotals(**{denominator_state: 1})


def merge_execution_closure(state: PaperDenominatorBuildState) -> None:
    for record in state.execution_closure.get("records", []):
        category = str(record.get("category", "unknown"))
        problem_id = str(
            record.get("problem_id") or record.get("problem_path") or "unknown"
        )
        problem_path = record.get("problem_path")
        status = str(record.get("closure_status"))
        denominator_state = _closure_state(status)
        key = (
            problem_id,
            record.get("row_index"),
            str(record.get("workload_uuid")) if record.get("workload_uuid") else None,
        )
        workload = state.workloads.setdefault(
            key,
            PaperDenominatorWorkload(
                category=category,
                problem_id=problem_id,
                problem_path=str(problem_path) if problem_path else None,
                workload_uuid=str(record.get("workload_uuid"))
                if record.get("workload_uuid")
                else None,
                row_index=record.get("row_index"),
            ),
        )
        workload.category = category
        workload.problem_id = problem_id
        workload.problem_path = str(problem_path) if problem_path else None
        workload.workload_uuid = (
            str(record.get("workload_uuid")) if record.get("workload_uuid") else None
        )
        workload.row_index = record.get("row_index")
        workload.closure_status = status
        example_ref = _record_ref(record)
        if denominator_state:
            for rollup in (
                _category_rollup(state.categories, category),
                _problem_rollup(
                    state.problems,
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_path) if problem_path else None,
                ),
            ):
                rollup.states.add(denominator_state)
            workload.states.add(denominator_state)
        for reason in record.get("filter_reasons", []):
            _add_reason(
                state.reason_groups,
                reason_code=str(reason),
                state="filtered",
                example_ref=example_ref,
            )
        for reason in record.get("readiness_reason_codes", []):
            _add_reason(
                state.reason_groups,
                reason_code=str(reason),
                state=denominator_state or status,
                example_ref=example_ref,
            )
        for gap in record.get("evidence_gaps", []):
            gap_code = str(gap)
            if gap_code not in workload.evidence_gaps:
                workload.evidence_gaps.append(gap_code)
            _add_missing_evidence(
                reason_groups=state.reason_groups,
                evidence_groups=state.evidence_groups,
                reason_code=gap_code,
                example_ref=example_ref,
            )
        evidence_refs = record.get("evidence_refs") or {}
        if status not in {"filtered", "not_attempted"}:
            for ref_key, reason_code in REQUIRED_RECORD_EVIDENCE_REFS.items():
                if evidence_refs.get(ref_key):
                    continue
                if reason_code not in workload.evidence_gaps:
                    workload.evidence_gaps.append(reason_code)
                    _add_missing_evidence(
                        reason_groups=state.reason_groups,
                        evidence_groups=state.evidence_groups,
                        reason_code=reason_code,
                        example_ref=example_ref,
                    )
        if workload.evidence_gaps:
            for rollup in (
                _category_rollup(state.categories, category),
                _problem_rollup(
                    state.problems,
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_path) if problem_path else None,
                ),
            ):
                rollup.states.add("evidence_missing")
            workload.states.add("evidence_missing")


def add_missing_artifact_evidence(state: PaperDenominatorBuildState) -> None:
    for score in (state.amd_score_report or {}).get("scores", []):
        uuid = score.get("workload_uuid")
        if score.get("supported") is not True:
            reason_code = "amd_score_evidence_missing"
            ref = str(uuid or score.get("definition") or "unknown")
            _add_missing_evidence(
                reason_groups=state.reason_groups,
                evidence_groups=state.evidence_groups,
                reason_code=reason_code,
                example_ref=ref,
            )

    if state.amd_score_report is None:
        _add_missing_evidence(
            reason_groups=state.reason_groups,
            evidence_groups=state.evidence_groups,
            reason_code="amd_score_evidence_missing",
            example_ref="amd_score_report",
            next_evidence="Attach a bounded AMD score report ref/checksum before upgrading claims.",
        )
    if not state.amd_sol_artifacts:
        _add_missing_evidence(
            reason_groups=state.reason_groups,
            evidence_groups=state.evidence_groups,
            reason_code="amd_sol_evidence_missing",
            example_ref="amd_sol_artifacts",
            next_evidence="Attach bounded AMD SOL artifact refs/checksums before upgrading claims.",
        )
    if not state.solar_artifacts:
        _add_missing_evidence(
            reason_groups=state.reason_groups,
            evidence_groups=state.evidence_groups,
            reason_code="solar_derivation_missing",
            example_ref="solar_artifacts",
            next_evidence="Attach bounded SOLAR derivation refs/checksums before upgrading claims.",
        )


def finalize_rollups(state: PaperDenominatorBuildState) -> None:
    for workload in state.workloads.values():
        if any(getattr(workload.states, key) for key in DENOMINATOR_STATE_KEYS):
            continue
        workload.states.add("not_attempted")
        for rollup in (
            _category_rollup(state.categories, workload.category),
            _problem_rollup(
                state.problems,
                category=workload.category,
                problem_id=workload.problem_id,
                problem_path=workload.problem_path,
            ),
        ):
            rollup.states.add("not_attempted")

    for problem in state.problems.values():
        problem.rollup.problems = 1
        problem.rollup.workloads = sum(
            1
            for workload in state.workloads.values()
            if workload.category == problem.category
            and workload.problem_id == problem.problem_id
        )

    for category_rollup in state.categories.values():
        category_rollup.problems = 0
        category_rollup.workloads = 0
    for problem in state.problems.values():
        category_rollup = _category_rollup(state.categories, problem.category)
        category_rollup.problems += 1
        category_rollup.workloads += problem.rollup.workloads


def _optional_str(value: object) -> str | None:
    return str(value) if value else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return None
