# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Internal state for workload readiness classification."""

from __future__ import annotations

from dataclasses import dataclass, field

from .inventory import ProblemInventoryRecord, WorkloadInventoryRecord
from .readiness_models import (
    LayeredEvidence,
    ReadinessBlockerReport,
    ReadinessClass,
    ReadinessReason,
    WorkloadReadinessRecord,
)
from .readiness_reports import _blocker, _reason


@dataclass
class ReadinessClassificationState:
    problem: ProblemInventoryRecord
    workload: WorkloadInventoryRecord
    status: str = "ready"
    readiness_class: str = ReadinessClass.PYTORCH_COMPATIBLE
    reasons: list[ReadinessReason] = field(default_factory=list)
    blockers: list[ReadinessBlockerReport] = field(default_factory=list)
    layers: LayeredEvidence = field(default_factory=LayeredEvidence)

    def add_reason(
        self,
        code: str,
        message: str,
        next_action: str,
        evidence_path: str | None = None,
    ) -> None:
        self.reasons.append(
            _reason(code, message, next_action, evidence_path)
        )

    def add_blocker(
        self,
        *,
        code: str,
        blocker_type: str,
        message: str,
        next_action: str,
        evidence_path: str | None = None,
    ) -> None:
        self.blockers.append(
            _blocker(
                code=code,
                blocker_type=blocker_type,
                problem=self.problem,
                workload=self.workload,
                message=message,
                next_action=next_action,
                evidence_path=evidence_path,
            )
        )

    def add_default_ready_reason(self) -> None:
        if self.reasons or self.status != "ready":
            return
        self.add_reason(
            "ready_to_attempt_rocm_execution",
            "No static blocker found; ready to attempt local ROCm execution.",
            "Run bounded execution closure in Phase 55.",
            self.problem.problem_path,
        )

    def finish_record(self) -> WorkloadReadinessRecord:
        self.add_default_ready_reason()
        return WorkloadReadinessRecord(
            category=self.problem.category,
            problem_id=self.problem.problem_id,
            problem_path=self.problem.problem_path,
            workload_uuid=self.workload.uuid,
            row_index=self.workload.row_index,
            status=self.status,
            readiness_class=self.readiness_class,
            reasons=self.reasons,
            blocker_reports=self.blockers,
            layered_evidence=self.layers,
        )
