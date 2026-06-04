# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Ready-subset sidecar generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp
from .readiness import DatasetReadiness

READY_SUBSET_SCHEMA_VERSION = "sol_execbench.ready_subset.v1"


class ReadySubsetWorkloadRef(BaseModel):
    uuid: str | None
    row_index: int
    readiness_class: str
    readiness_status: str
    closure_inputs: dict[str, str | int | None]


class ReadySubsetExclusionReason(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None
    row_index: int
    readiness_class: str
    readiness_status: str
    reason_codes: list[str]
    blocker_types: list[str]
    message: str


class ReadySubsetProblemRef(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    workloads: list[ReadySubsetWorkloadRef]


class ReadySubsetClaimBoundary(BaseModel):
    ready_to_attempt_rocm_execution: bool
    execution_success: bool = False
    hardware_validation: bool = False
    paper_level_validation: bool = False
    hosted_leaderboard_parity: bool = False
    upstream_solar_equivalence: bool = False
    score_authority: bool = False


class ReadySubsetDenominator(BaseModel):
    total_workloads: int
    included_workloads: int
    excluded_workloads: int


class ReadySubset(BaseModel):
    schema_version: str = READY_SUBSET_SCHEMA_VERSION
    created_at: str
    dataset_root: str
    readiness_checksum: str | None = None
    selected_categories: tuple[str, ...]
    denominator: ReadySubsetDenominator
    included_workloads: int
    excluded_workloads: int
    problems: list[ReadySubsetProblemRef]
    exclusions: list[ReadySubsetExclusionReason] = Field(default_factory=list)
    claim_boundary: ReadySubsetClaimBoundary
    ready_subset_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ReadySubset":
        payload = self.model_dump(mode="json")
        payload["ready_subset_checksum"] = None
        return self.model_copy(update={"ready_subset_checksum": DatasetManifestChecksum(value=stable_json_checksum(payload))})

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def build_ready_subset(
    readiness: DatasetReadiness,
    *,
    dataset_root: Path,
    created_at: str | None = None,
) -> ReadySubset:
    grouped: dict[str, list[ReadySubsetWorkloadRef]] = defaultdict(list)
    problem_meta: dict[str, tuple[str, str]] = {}
    exclusions: list[ReadySubsetExclusionReason] = []
    included = 0
    excluded = 0
    for record in readiness.workloads:
        if record.status == "ready":
            included += 1
            grouped[record.problem_id].append(
                ReadySubsetWorkloadRef(
                    uuid=record.workload_uuid,
                    row_index=record.row_index,
                    readiness_class=record.readiness_class,
                    readiness_status=record.status,
                    closure_inputs={
                        "category": record.category,
                        "problem_id": record.problem_id,
                        "problem_path": record.problem_path,
                        "workload_uuid": record.workload_uuid,
                        "row_index": record.row_index,
                        "readiness_checksum": readiness.readiness_checksum.value
                        if readiness.readiness_checksum
                        else None,
                    },
                )
            )
            problem_meta[record.problem_id] = (record.category, record.problem_path)
        else:
            excluded += 1
            reason_codes = [reason.code for reason in record.reasons]
            blocker_types = sorted(
                {blocker.blocker_type for blocker in record.blocker_reports}
            )
            exclusions.append(
                ReadySubsetExclusionReason(
                    category=record.category,
                    problem_id=record.problem_id,
                    problem_path=record.problem_path,
                    workload_uuid=record.workload_uuid,
                    row_index=record.row_index,
                    readiness_class=record.readiness_class,
                    readiness_status=record.status,
                    reason_codes=reason_codes,
                    blocker_types=blocker_types,
                    message="; ".join(reason.message for reason in record.reasons),
                )
            )
    problems = [
        ReadySubsetProblemRef(category=problem_meta[problem_id][0], problem_id=problem_id, problem_path=problem_meta[problem_id][1], workloads=sorted(workloads, key=lambda item: (item.row_index, item.uuid or "")))
        for problem_id, workloads in sorted(grouped.items())
    ]
    subset = ReadySubset(
        created_at=created_at or utc_timestamp(),
        dataset_root=Path(dataset_root).as_posix(),
        readiness_checksum=readiness.readiness_checksum.value
        if readiness.readiness_checksum
        else None,
        selected_categories=readiness.selected_categories,
        denominator=ReadySubsetDenominator(
            total_workloads=included + excluded,
            included_workloads=included,
            excluded_workloads=excluded,
        ),
        included_workloads=included,
        excluded_workloads=excluded,
        problems=problems,
        exclusions=sorted(
            exclusions,
            key=lambda item: (item.problem_id, item.row_index, item.workload_uuid or ""),
        ),
        claim_boundary=ReadySubsetClaimBoundary(ready_to_attempt_rocm_execution=included > 0),
    )
    return subset.with_checksum()


def write_ready_subset(subset: ReadySubset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(subset.to_json(), encoding="utf-8")
