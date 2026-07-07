# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Readiness reason, blocker, and severity helpers."""

from __future__ import annotations

from .inventory import ProblemInventoryRecord, WorkloadInventoryRecord
from .readiness_models import (
    READINESS_SEVERITY,
    ReadinessBlockerReport,
    ReadinessReason,
)

def _reason(
    code: str, message: str, next_action: str, evidence_path: str | None = None
) -> ReadinessReason:
    return ReadinessReason(
        code=code, message=message, next_action=next_action, evidence_path=evidence_path
    )


def _blocker(
    *,
    code: str,
    blocker_type: str,
    problem: ProblemInventoryRecord,
    workload: WorkloadInventoryRecord,
    message: str,
    next_action: str,
    evidence_path: str | None = None,
) -> ReadinessBlockerReport:
    return ReadinessBlockerReport(
        code=code,
        blocker_type=blocker_type,
        problem_id=problem.problem_id,
        problem_path=problem.problem_path,
        workload_uuid=workload.uuid,
        row_index=workload.row_index,
        evidence_path=evidence_path,
        message=message,
        next_action=next_action,
    )


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "schema_input_blocked"
    return min(statuses, key=lambda status: READINESS_SEVERITY[status])
