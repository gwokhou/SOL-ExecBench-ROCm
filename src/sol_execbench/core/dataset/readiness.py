# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static ROCm readiness classification for dataset inventory sidecars."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .checksums import stable_json_checksum
from .inventory import DatasetInventory, ProblemInventoryRecord, WorkloadInventoryRecord
from .manifest import DatasetManifestChecksum, utc_timestamp

READINESS_SCHEMA_VERSION = "sol_execbench.rocm_readiness.v1"

READINESS_SEVERITY: dict[str, int] = {
    "schema_input_blocked": 0,
    "unsupported_nvidia_only_path": 1,
    "custom_input_blocked": 2,
    "dtype_blocked": 3,
    "runtime_blocked": 4,
    "needs_hardware_evidence": 5,
    "ready": 6,
}

LOW_PRECISION_DTYPES = {
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "float4_e2m1fn_x2",
}

class ReadinessReason(BaseModel):
    code: str
    evidence_path: str | None = None
    next_action: str
    message: str


class LayeredEvidence(BaseModel):
    schema_known: str = "ok"
    input_generation: str = "ok"
    reference_execution: str = "ready_to_attempt"
    candidate_execution: str = "not_evaluated"
    hardware_validation: str = "not_required"


class WorkloadReadinessRecord(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None
    row_index: int
    status: str
    reasons: list[ReadinessReason] = Field(default_factory=list)
    layered_evidence: LayeredEvidence = Field(default_factory=LayeredEvidence)


class ProblemReadinessRecord(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    status: str
    workload_count: int
    status_counts: dict[str, int]


class DatasetReadiness(BaseModel):
    schema_version: str = READINESS_SCHEMA_VERSION
    created_at: str
    inventory_checksum: str | None = None
    selected_categories: tuple[str, ...]
    problems: list[ProblemReadinessRecord]
    workloads: list[WorkloadReadinessRecord]
    readiness_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetReadiness":
        payload = self.model_dump(mode="json")
        payload["readiness_checksum"] = None
        return self.model_copy(
            update={"readiness_checksum": DatasetManifestChecksum(value=stable_json_checksum(payload))}
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _reason(code: str, message: str, next_action: str, evidence_path: str | None = None) -> ReadinessReason:
    return ReadinessReason(code=code, message=message, next_action=next_action, evidence_path=evidence_path)


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "schema_input_blocked"
    return min(statuses, key=lambda status: READINESS_SEVERITY[status])


def _reference_has_nvidia_blocker(problem: ProblemInventoryRecord) -> bool:
    return bool(problem.definition and problem.definition.reference_runtime_hints)


def _low_precision_or_quant(problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord) -> bool:
    dtypes = set(workload.input_dtypes.values()) | set(workload.output_dtypes.values())
    return problem.category == "Quant" or bool(dtypes & LOW_PRECISION_DTYPES)


def classify_workload_readiness(
    problem: ProblemInventoryRecord,
    workload: WorkloadInventoryRecord,
    *,
    dataset_root: Path,
) -> WorkloadReadinessRecord:
    reasons: list[ReadinessReason] = []
    layers = LayeredEvidence()
    status = "ready"

    if problem.schema_status != "parsed" or workload.schema_status != "parsed":
        layers.schema_known = "blocked"
        status = "schema_input_blocked"
        reasons.append(_reason("workload_schema_failure" if workload.schema_status != "parsed" else "definition_schema_failure", workload.schema_failure or problem.schema_failure or "schema parse failed", "Fix or exclude malformed canonical dataset entry.", problem.problem_path))
    elif not problem.reference_available:
        status = "schema_input_blocked"
        layers.reference_execution = "blocked"
        reasons.append(_reason("missing_reference", "Reference source is not available.", "Restore reference source before execution attempts.", problem.problem_path))
    elif _reference_has_nvidia_blocker(problem):
        status = "unsupported_nvidia_only_path"
        layers.reference_execution = "blocked"
        reasons.append(_reason("nvidia_cuda_runtime_hint", "Static NVIDIA/CUDA runtime hint detected.", "Port or exclude NVIDIA-only reference path.", problem.problem_path))
    elif workload.uses_custom_inputs:
        status = "custom_input_blocked"
        layers.input_generation = "blocked"
        reasons.append(_reason("custom_input_requires_evaluator_support", "Custom input generation requires evaluator support and must not be random-substituted.", "Use execution-time custom input support or exclude from ready subset.", problem.problem_path))
    else:
        missing_safetensors = []
        dataset_root_resolved = Path(dataset_root).resolve()
        for ref in workload.safetensors_refs:
            raw_path = Path(ref["path"])
            reason_code = "safetensors_asset_missing"
            if raw_path.is_absolute():
                reason_code = "safetensors_path_outside_dataset_root"
                missing_safetensors.append((ref, reason_code))
                continue
            candidate = (dataset_root_resolved / raw_path).resolve()
            if not candidate.is_relative_to(dataset_root_resolved):
                reason_code = "safetensors_path_outside_dataset_root"
                missing_safetensors.append((ref, reason_code))
                continue
            if not candidate.exists():
                missing_safetensors.append((ref, reason_code))
        if missing_safetensors:
            status = "runtime_blocked"
            layers.input_generation = "needs_asset"
            for ref, reason_code in missing_safetensors:
                reasons.append(
                    _reason(
                        reason_code,
                        f"Safetensors asset {ref['path']} for key {ref['tensor_key']} is unavailable inside the dataset root.",
                        "Acquire asset inside the dataset root or configure a safe blob root before execution.",
                        ref["path"],
                    )
                )
        elif _low_precision_or_quant(problem, workload):
            status = "needs_hardware_evidence"
            layers.hardware_validation = "needed"
            reasons.append(_reason("low_precision_requires_hardware_evidence", "Low-precision or Quant workload needs hardware validation evidence before validation claims.", "Collect hardware evidence during execution closure.", problem.problem_path))

    if not reasons and status == "ready":
        reasons.append(_reason("ready_to_attempt_rocm_execution", "No static blocker found; ready to attempt local ROCm execution.", "Run bounded execution closure in Phase 55.", problem.problem_path))

    return WorkloadReadinessRecord(category=problem.category, problem_id=problem.problem_id, problem_path=problem.problem_path, workload_uuid=workload.uuid, row_index=workload.row_index, status=status, reasons=reasons, layered_evidence=layers)


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
            synthetic = WorkloadInventoryRecord(uuid=None, row_index=0, schema_status="schema_failure", schema_failure=problem.schema_failure or "no parsed workloads")
            record = classify_workload_readiness(problem, synthetic, dataset_root=dataset_root)
            workload_records.append(record)
            by_problem[problem.problem_id].append(record)
            continue
        for workload in problem.workloads:
            record = classify_workload_readiness(problem, workload, dataset_root=dataset_root)
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
        problem_records.append(ProblemReadinessRecord(category=problem.category, problem_id=problem_id, problem_path=problem.problem_path, status=_worst_status([record.status for record in records]), workload_count=len(records), status_counts=dict(sorted(counts.items()))))

    readiness = DatasetReadiness(created_at=created_at or utc_timestamp(), inventory_checksum=inventory.inventory_checksum.value if inventory.inventory_checksum else None, selected_categories=inventory.selected_categories, problems=problem_records, workloads=workload_records)
    return readiness.with_checksum()


def write_dataset_readiness(readiness: DatasetReadiness, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(readiness.to_json(), encoding="utf-8")
