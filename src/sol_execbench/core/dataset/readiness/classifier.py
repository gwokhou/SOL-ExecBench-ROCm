# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""ROCm readiness classification builders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..inventory import (
    DatasetInventory,
    ProblemInventoryRecord,
    WorkloadInventoryRecord,
)
from sol_execbench.core.timestamps import utc_timestamp
from .classification_handlers import (
    classify_cuda_solution,
    classify_custom_inputs,
    classify_flashinfer,
    classify_missing_reference,
    classify_nvidia_dsl,
    classify_nvidia_reference,
    classify_quant,
    classify_safetensors_and_low_precision,
    classify_schema_failure,
    classify_unsupported_dtype,
)
from .classification_state import ReadinessClassificationState
from .hints import _solution_runtime_hints
from .models import (
    DatasetReadiness,
    DatasetReadinessClaimBoundary,
    ProblemReadinessRecord,
    WorkloadReadinessRecord,
)
from .reports import _worst_status


def classify_workload_readiness(
    problem: ProblemInventoryRecord,
    workload: WorkloadInventoryRecord,
    *,
    dataset_root: Path,
) -> WorkloadReadinessRecord:
    state = ReadinessClassificationState(problem=problem, workload=workload)
    solution_hints = _solution_runtime_hints(problem, dataset_root)

    handlers = (
        lambda: classify_unsupported_dtype(state),
        lambda: classify_schema_failure(state),
        lambda: classify_missing_reference(state),
        lambda: classify_flashinfer(
            state, dataset_root=dataset_root, solution_hints=solution_hints
        ),
        lambda: classify_nvidia_reference(state),
        lambda: classify_cuda_solution(state, solution_hints=solution_hints),
        lambda: classify_nvidia_dsl(state, solution_hints=solution_hints),
        lambda: classify_quant(state),
        lambda: classify_custom_inputs(state),
        lambda: classify_safetensors_and_low_precision(
            state, dataset_root=dataset_root, solution_hints=solution_hints
        ),
    )
    for handler in handlers:
        if handler():
            break

    return state.finish_record()


def classify_rocm_readiness(
    inventory: DatasetInventory,
    *,
    dataset_root: Path,
    created_at: str | None = None,
) -> DatasetReadiness:
    workload_records: list[WorkloadReadinessRecord] = []
    by_problem: dict[str, list[WorkloadReadinessRecord]] = defaultdict(list)
    for problem in inventory.problems:
        if not problem.workloads:
            synthetic = WorkloadInventoryRecord(
                uuid=None,
                row_index=0,
                schema_status="schema_failure",
                schema_failure=problem.schema_failure or "no parsed workloads",
            )
            record = classify_workload_readiness(
                problem, synthetic, dataset_root=dataset_root
            )
            workload_records.append(record)
            by_problem[problem.problem_id].append(record)
            continue
        for workload in problem.workloads:
            record = classify_workload_readiness(
                problem, workload, dataset_root=dataset_root
            )
            workload_records.append(record)
            by_problem[problem.problem_id].append(record)

    problem_records: list[ProblemReadinessRecord] = []
    problem_lookup = {problem.problem_id: problem for problem in inventory.problems}
    for problem_id in sorted(by_problem):
        records = by_problem[problem_id]
        counts: dict[str, int] = {}
        for record in records:
            counts[record.status] = counts.get(record.status, 0) + 1
        problem = problem_lookup[problem_id]
        problem_records.append(
            ProblemReadinessRecord(
                category=problem.category,
                problem_id=problem_id,
                problem_path=problem.problem_path,
                status=_worst_status([record.status for record in records]),
                workload_count=len(records),
                status_counts=dict(sorted(counts.items())),
            )
        )

    blocker_reports = [
        blocker for record in workload_records for blocker in record.blocker_reports
    ]
    readiness = DatasetReadiness(
        created_at=created_at or utc_timestamp(),
        inventory_checksum=inventory.inventory_checksum.value
        if inventory.inventory_checksum
        else None,
        selected_categories=inventory.selected_categories,
        problems=problem_records,
        workloads=workload_records,
        blocker_reports=blocker_reports,
        claim_boundary=DatasetReadinessClaimBoundary(
            ready_to_attempt_rocm_execution=any(
                record.status == "ready" for record in workload_records
            )
        ),
    )
    return readiness.with_checksum()
