# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sol_execbench.core.data.path_access import (
    path_bool,
    path_dict,
    path_get,
    path_int_or_none,
    path_mapping_list,
    path_str_list,
    path_str_or_none,
)
from sol_execbench.core.dataset.parity_gap_models import (
    DENOMINATOR_KEYS,
    ParityAmdScoreRecord,
    ParityExecutionClosureRecord,
    ParityGapBlocker,
    ParityGapDenominators,
    ParityGapSource,
    ParityReadinessWorkloadRecord,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checksum(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return path_str_or_none(value, "value")
    if isinstance(value, str):
        return value
    return None


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | None,
    checksum_key: str | None = None,
) -> ParityGapSource:
    if payload is None:
        return ParityGapSource(path=str(path) if path else None)
    return ParityGapSource(
        path=str(path) if path else None,
        schema_version=path_str_or_none(payload, "schema_version"),
        checksum=_checksum(payload, checksum_key) if checksum_key else None,
    )


def _default_counts() -> dict[str, int]:
    return dict.fromkeys(DENOMINATOR_KEYS, 0)


def _category_counts(
    name: str, categories: dict[str, ParityGapDenominators]
) -> ParityGapDenominators:
    if name not in categories:
        categories[name] = ParityGapDenominators()
    return categories[name]


def _record_ref(record: dict[str, Any]) -> str:
    problem = (
        path_get(record, "problem_id") or path_get(record, "problem_path") or "unknown"
    )
    uuid = path_get(record, "workload_uuid")
    row = path_get(record, "row_index")
    return f"{problem}#{uuid or row}"


def _add_blocker(
    groups: dict[str, dict[str, Any]],
    *,
    reason_code: str,
    category: str,
    example_ref: str,
    next_action: str,
) -> None:
    group = groups.setdefault(
        reason_code,
        {
            "reason_code": reason_code,
            "count": 0,
            "categories": set(),
            "example_refs": [],
            "next_actions": set(),
        },
    )
    group["count"] += 1
    group["categories"].add(category)
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)
    group["next_actions"].add(next_action)


def _final_blockers(groups: dict[str, dict[str, Any]]) -> list[ParityGapBlocker]:
    blockers = []
    for group in groups.values():
        blockers.append(
            ParityGapBlocker(
                reason_code=group["reason_code"],
                count=group["count"],
                categories=sorted(group["categories"]),
                example_refs=sorted(group["example_refs"]),
                next_actions=sorted(group["next_actions"]),
            )
        )
    return sorted(blockers, key=lambda blocker: (-blocker.count, blocker.reason_code))


def _score_category(
    score: ParityAmdScoreRecord,
    workload_to_category: dict[str, str],
) -> str:
    uuid = str(score.workload_uuid or "")
    return workload_to_category.get(uuid, "unknown")


def _readiness_workload_record(payload: object) -> ParityReadinessWorkloadRecord:
    problem_path = path_str_or_none(payload, "problem_path")
    problem_id = path_str_or_none(payload, "problem_id") or problem_path or "unknown"
    return ParityReadinessWorkloadRecord(
        category=path_str_or_none(payload, "category") or "unknown",
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=path_str_or_none(payload, "workload_uuid"),
        row_index=path_int_or_none(payload, "row_index"),
        status=path_str_or_none(payload, "status") or "unknown",
        reasons=path_mapping_list(payload, "reasons"),
    )


def _execution_closure_record(payload: object) -> ParityExecutionClosureRecord:
    problem_path = path_str_or_none(payload, "problem_path")
    problem_id = path_str_or_none(payload, "problem_id") or problem_path or "unknown"
    return ParityExecutionClosureRecord(
        category=path_str_or_none(payload, "category") or "unknown",
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=path_str_or_none(payload, "workload_uuid"),
        row_index=path_int_or_none(payload, "row_index"),
        closure_status=path_str_or_none(payload, "closure_status") or "unknown",
        trace_status=path_str_or_none(payload, "trace_status"),
        trace_ref=path_str_or_none(payload, "trace_ref"),
        evidence_refs=path_dict(payload, "evidence_refs"),
        evidence_gaps=path_str_list(payload, "evidence_gaps"),
    )


def _amd_score_record(payload: object) -> ParityAmdScoreRecord:
    return ParityAmdScoreRecord(
        definition=path_str_or_none(payload, "definition") or "unknown",
        workload_uuid=path_str_or_none(payload, "workload_uuid"),
        supported=path_bool(payload, "supported"),
        warnings=path_str_list(payload, "warnings"),
        evidence_refs=path_dict(payload, "evidence_refs"),
        derived_evidence_refs=path_dict(payload, "derived_evidence_refs"),
    )
