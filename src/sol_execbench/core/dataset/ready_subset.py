# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Ready-subset sidecar generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp
from .readiness import DatasetReadiness

READY_SUBSET_SCHEMA_VERSION = "sol_execbench.ready_subset.v1"


class ReadySubsetWorkloadRef(BaseModel):
    uuid: str | None
    row_index: int


class ReadySubsetProblemRef(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    workloads: list[ReadySubsetWorkloadRef]


class ReadySubsetClaimBoundary(BaseModel):
    ready_to_attempt_rocm_execution: bool
    execution_success: bool = False
    paper_level_validation: bool = False
    hosted_leaderboard_parity: bool = False
    upstream_solar_equivalence: bool = False


class ReadySubset(BaseModel):
    schema_version: str = READY_SUBSET_SCHEMA_VERSION
    created_at: str
    dataset_root: str
    readiness_checksum: str | None = None
    selected_categories: tuple[str, ...]
    included_workloads: int
    excluded_workloads: int
    problems: list[ReadySubsetProblemRef]
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
    included = 0
    excluded = 0
    for record in readiness.workloads:
        if record.status == "ready":
            included += 1
            grouped[record.problem_id].append(ReadySubsetWorkloadRef(uuid=record.workload_uuid, row_index=record.row_index))
            problem_meta[record.problem_id] = (record.category, record.problem_path)
        else:
            excluded += 1
    problems = [
        ReadySubsetProblemRef(category=problem_meta[problem_id][0], problem_id=problem_id, problem_path=problem_meta[problem_id][1], workloads=sorted(workloads, key=lambda item: (item.row_index, item.uuid or "")))
        for problem_id, workloads in sorted(grouped.items())
    ]
    subset = ReadySubset(created_at=created_at or utc_timestamp(), dataset_root=Path(dataset_root).as_posix(), readiness_checksum=readiness.readiness_checksum.value if readiness.readiness_checksum else None, selected_categories=readiness.selected_categories, included_workloads=included, excluded_workloads=excluded, problems=problems, claim_boundary=ReadySubsetClaimBoundary(ready_to_attempt_rocm_execution=included > 0))
    return subset.with_checksum()


def write_ready_subset(subset: ReadySubset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(subset.to_json(), encoding="utf-8")
