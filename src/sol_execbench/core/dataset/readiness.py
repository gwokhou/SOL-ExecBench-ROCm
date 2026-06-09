# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Static ROCm readiness classification for dataset inventory sidecars."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .checksums import stable_json_checksum
from .inventory import DatasetInventory, ProblemInventoryRecord, WorkloadInventoryRecord
from .low_precision import (
    CDNA4_VALIDATION_DEFERRED_CODE,
    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
)
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
BLACKWELL_LOW_PRECISION_DTYPES = {
    "float4_e2m1",
    "float4_e2m1fn_x2",
    "nvfp4",
    "mxfp4",
}


class ReadinessClass:
    PYTORCH_COMPATIBLE = "pytorch_compatible"
    ROCM_PORT_NEEDED = "rocm_port_needed"
    FLASHINFER_SPECIFIC = "flashinfer_specific"
    NVFP4_BLACKWELL_SPECIFIC = "nvfp4_blackwell_specific"
    UNSUPPORTED = "unsupported"
    BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"


class ReadinessBlockerReport(BaseModel):
    code: str
    blocker_type: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None = None
    row_index: int
    evidence_path: str | None = None
    message: str
    next_action: str


class DatasetReadinessClaimBoundary(BaseModel):
    ready_to_attempt_rocm_execution: bool
    execution_success: bool = False
    hardware_validation: bool = False
    paper_level_validation: bool = False
    hosted_leaderboard_parity: bool = False
    upstream_solar_equivalence: bool = False
    score_authority: bool = False


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
    readiness_class: str
    reasons: list[ReadinessReason] = Field(default_factory=list)
    blocker_reports: list[ReadinessBlockerReport] = Field(default_factory=list)
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
    blocker_reports: list[ReadinessBlockerReport] = Field(default_factory=list)
    claim_boundary: DatasetReadinessClaimBoundary
    readiness_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetReadiness":
        payload = self.model_dump(mode="json")
        payload["readiness_checksum"] = None
        return self.model_copy(
            update={
                "readiness_checksum": DatasetManifestChecksum(
                    value=stable_json_checksum(payload)
                )
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


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


def _reference_has_nvidia_blocker(problem: ProblemInventoryRecord) -> bool:
    return bool(problem.definition and problem.definition.reference_runtime_hints)


def _low_precision_or_quant(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> bool:
    dtypes = set(workload.input_dtypes.values()) | set(workload.output_dtypes.values())
    return problem.category == "Quant" or bool(dtypes & LOW_PRECISION_DTYPES)


def _blackwell_low_precision(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> bool:
    dtypes = set(workload.input_dtypes.values()) | set(workload.output_dtypes.values())
    identity = f"{problem.problem_id} {problem.problem_path}".lower()
    return (
        problem.category == "Quant"
        or bool(dtypes & BLACKWELL_LOW_PRECISION_DTYPES)
        or "blackwell" in identity
        or "nvfp4" in identity
        or "mxfp4" in identity
    )


def _solution_runtime_hints(
    problem: ProblemInventoryRecord, dataset_root: Path
) -> set[str]:
    hints: set[str] = set()
    problem_dir = Path(dataset_root) / problem.problem_path
    for filename in problem.solution_files:
        lowered = filename.lower()
        if lowered.endswith((".cu", ".cuh", ".so")) or "cuda" in lowered:
            hints.add("cuda_kernel")
        if "cute" in lowered or "cutile" in lowered or "cutlass" in lowered:
            hints.add("nvidia_dsl")
        path = problem_dir / filename
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".py",
            ".cu",
            ".cuh",
            ".cpp",
            ".hip",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in text for token in ("cuda", "cublas", "cutlass", "nvrtc")):
            hints.add("cuda_kernel")
        if any(
            token in text for token in ("flashinfer", "paged_decode", "paged attention")
        ):
            hints.add("flashinfer_runtime")
        if any(token in text for token in ("cute_dsl", "cutile", "cutlass")):
            hints.add("nvidia_dsl")
        if any(
            token in text for token in ("nvfp4", "mxfp4", "float4_e2m1", "blackwell")
        ):
            hints.add("blackwell_low_precision")
    return hints


def _unsupported_dtype_failure(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> str | None:
    failure = " ".join(
        item
        for item in (problem.schema_failure, workload.schema_failure)
        if item is not None
    ).lower()
    if (
        "unsupported dtype" in failure
        or "dtype" in failure
        and "unsupported" in failure
    ):
        return problem.schema_failure or workload.schema_failure or "unsupported dtype"
    return None


def classify_workload_readiness(
    problem: ProblemInventoryRecord,
    workload: WorkloadInventoryRecord,
    *,
    dataset_root: Path,
) -> WorkloadReadinessRecord:
    reasons: list[ReadinessReason] = []
    blockers: list[ReadinessBlockerReport] = []
    layers = LayeredEvidence()
    status = "ready"
    readiness_class = ReadinessClass.PYTORCH_COMPATIBLE
    solution_hints = _solution_runtime_hints(problem, dataset_root)
    is_flashinfer = (
        problem.category == "FlashInfer-Bench"
        or "flashinfer" in problem.problem_path.lower()
        or "flashinfer_runtime" in solution_hints
    )

    unsupported_dtype = _unsupported_dtype_failure(problem, workload)
    if unsupported_dtype is not None:
        layers.schema_known = "blocked"
        status = "dtype_blocked"
        readiness_class = ReadinessClass.UNSUPPORTED
        message = f"Unsupported dtype in migrated schema: {unsupported_dtype}"
        reasons.append(
            _reason(
                "unsupported_dtype",
                message,
                "Exclude or add an explicit ROCm dtype compatibility path.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="unsupported_dtype",
                blocker_type="unsupported_dtype",
                problem=problem,
                workload=workload,
                message=message,
                next_action="Exclude or add an explicit ROCm dtype compatibility path.",
                evidence_path=problem.problem_path,
            )
        )
    elif problem.schema_status != "parsed" or workload.schema_status != "parsed":
        layers.schema_known = "blocked"
        status = "schema_input_blocked"
        readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
        code = (
            "workload_schema_failure"
            if workload.schema_status != "parsed"
            else "definition_schema_failure"
        )
        message = (
            workload.schema_failure or problem.schema_failure or "schema parse failed"
        )
        reasons.append(
            _reason(
                code,
                message,
                "Fix or exclude malformed canonical dataset entry.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code=code,
                blocker_type="missing_evidence",
                problem=problem,
                workload=workload,
                message=message,
                next_action="Fix or exclude malformed canonical dataset entry.",
                evidence_path=problem.problem_path,
            )
        )
    elif not problem.reference_available:
        status = "schema_input_blocked"
        readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
        layers.reference_execution = "blocked"
        reasons.append(
            _reason(
                "missing_reference",
                "Reference source is not available.",
                "Restore reference source before execution attempts.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="missing_reference",
                blocker_type="missing_evidence",
                problem=problem,
                workload=workload,
                message="Reference source is not available.",
                next_action="Restore reference source before execution attempts.",
                evidence_path=problem.problem_path,
            )
        )
    elif is_flashinfer:
        status = "runtime_blocked"
        readiness_class = ReadinessClass.FLASHINFER_SPECIFIC
        layers.reference_execution = "blocked"
        reasons.append(
            _reason(
                "flashinfer_runtime_assumption",
                "FlashInfer Trace workload depends on FlashInfer-specific runtime semantics.",
                "Route through a dedicated ROCm FlashInfer compatibility/port path before execution.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="flashinfer_runtime_assumption",
                blocker_type="flashinfer_runtime_assumption",
                problem=problem,
                workload=workload,
                message="FlashInfer Trace workload depends on FlashInfer-specific runtime semantics.",
                next_action="Route through a dedicated ROCm FlashInfer compatibility/port path before execution.",
                evidence_path=problem.problem_path,
            )
        )
    elif _reference_has_nvidia_blocker(problem):
        status = "unsupported_nvidia_only_path"
        readiness_class = ReadinessClass.ROCM_PORT_NEEDED
        layers.reference_execution = "blocked"
        reasons.append(
            _reason(
                "nvidia_cuda_runtime_hint",
                "Static NVIDIA/CUDA runtime hint detected.",
                "Port or exclude NVIDIA-only reference path.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="nvidia_cuda_runtime_hint",
                blocker_type="cuda_kernel_dependency",
                problem=problem,
                workload=workload,
                message="Static NVIDIA/CUDA runtime hint detected.",
                next_action="Port or exclude NVIDIA-only reference path.",
                evidence_path=problem.problem_path,
            )
        )
    elif "cuda_kernel" in solution_hints:
        status = "unsupported_nvidia_only_path"
        readiness_class = ReadinessClass.ROCM_PORT_NEEDED
        layers.candidate_execution = "blocked"
        reasons.append(
            _reason(
                "cuda_solution_dependency",
                "Migrated solution still contains CUDA/NVIDIA kernel dependencies.",
                "Port candidate solution to HIP, Triton ROCm, or a ROCm library before execution.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="cuda_solution_dependency",
                blocker_type="cuda_kernel_dependency",
                problem=problem,
                workload=workload,
                message="Migrated solution still contains CUDA/NVIDIA kernel dependencies.",
                next_action="Port candidate solution to HIP, Triton ROCm, or a ROCm library before execution.",
                evidence_path=problem.problem_path,
            )
        )
    elif "nvidia_dsl" in solution_hints:
        status = "unsupported_nvidia_only_path"
        readiness_class = ReadinessClass.UNSUPPORTED
        layers.candidate_execution = "blocked"
        reasons.append(
            _reason(
                "unsupported_nvidia_dsl",
                "Migrated solution depends on NVIDIA-specific DSL/runtime code.",
                "Replace with a ROCm-native solution path before execution.",
                problem.problem_path,
            )
        )
        blockers.append(
            _blocker(
                code="unsupported_nvidia_dsl",
                blocker_type="cuda_kernel_dependency",
                problem=problem,
                workload=workload,
                message="Migrated solution depends on NVIDIA-specific DSL/runtime code.",
                next_action="Replace with a ROCm-native solution path before execution.",
                evidence_path=problem.problem_path,
            )
        )
    elif workload.uses_custom_inputs:
        if problem.definition and problem.definition.custom_inputs_entrypoint:
            layers.input_generation = "ready_to_generate"
        else:
            status = "custom_input_blocked"
            readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
            layers.input_generation = "blocked"
            reasons.append(
                _reason(
                    "custom_input_requires_evaluator_support",
                    "Custom input workload is missing a valid definition.custom_inputs_entrypoint.",
                    "Restore the benchmark-defined custom input entrypoint before execution attempts.",
                    problem.problem_path,
                )
            )
            blockers.append(
                _blocker(
                    code="custom_input_requires_evaluator_support",
                    blocker_type="missing_evidence",
                    problem=problem,
                    workload=workload,
                    message="Custom input workload is missing a valid definition.custom_inputs_entrypoint.",
                    next_action="Restore the benchmark-defined custom input entrypoint before execution attempts.",
                    evidence_path=problem.problem_path,
                )
            )
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
            readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
            layers.input_generation = "needs_asset"
            for ref, reason_code in missing_safetensors:
                message = f"Safetensors asset {ref['path']} for key {ref['tensor_key']} is unavailable inside the dataset root."
                reasons.append(
                    _reason(
                        reason_code,
                        message,
                        "Acquire asset inside the dataset root or configure a safe blob root before execution.",
                        ref["path"],
                    )
                )
                blockers.append(
                    _blocker(
                        code=reason_code,
                        blocker_type="missing_blob",
                        problem=problem,
                        workload=workload,
                        message=message,
                        next_action="Acquire asset inside the dataset root or configure a safe blob root before execution.",
                        evidence_path=ref["path"],
                    )
                )
        elif (
            _blackwell_low_precision(problem, workload)
            or "blackwell_low_precision" in solution_hints
        ):
            status = "needs_hardware_evidence"
            readiness_class = ReadinessClass.NVFP4_BLACKWELL_SPECIFIC
            layers.hardware_validation = "needed"
            reasons.append(
                _reason(
                    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
                    "Phase 134 CPU semantic compatibility path is available for NVFP4/Blackwell low-precision metadata, packing, and fallback behavior.",
                    "Use the compatibility path for migrated definitions, then collect CDNA4 hardware evidence before validation claims.",
                    problem.problem_path,
                )
            )
            reasons.append(
                _reason(
                    CDNA4_VALIDATION_DEFERRED_CODE,
                    "NVFP4/Blackwell low-precision workload still needs CDNA4 hardware evidence before validation claims.",
                    "Collect CDNA4 hardware evidence before validation or performance claims.",
                    problem.problem_path,
                )
            )
            blockers.append(
                _blocker(
                    code=CDNA4_VALIDATION_DEFERRED_CODE,
                    blocker_type="low_precision_format_dependency",
                    problem=problem,
                    workload=workload,
                    message="NVFP4/Blackwell low-precision workload still needs CDNA4 hardware evidence before validation claims.",
                    next_action="Collect CDNA4 hardware evidence before validation or performance claims.",
                    evidence_path=problem.problem_path,
                )
            )
        elif _low_precision_or_quant(problem, workload):
            status = "needs_hardware_evidence"
            readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
            layers.hardware_validation = "needed"
            reasons.append(
                _reason(
                    "low_precision_requires_hardware_evidence",
                    "Low-precision or Quant workload needs hardware validation evidence before validation claims.",
                    "Collect hardware evidence during execution closure.",
                    problem.problem_path,
                )
            )
            blockers.append(
                _blocker(
                    code="low_precision_requires_hardware_evidence",
                    blocker_type="low_precision_format_dependency",
                    problem=problem,
                    workload=workload,
                    message="Low-precision or Quant workload needs hardware validation evidence before validation claims.",
                    next_action="Collect hardware evidence during execution closure.",
                    evidence_path=problem.problem_path,
                )
            )

    if not reasons and status == "ready":
        reasons.append(
            _reason(
                "ready_to_attempt_rocm_execution",
                "No static blocker found; ready to attempt local ROCm execution.",
                "Run bounded execution closure in Phase 55.",
                problem.problem_path,
            )
        )

    return WorkloadReadinessRecord(
        category=problem.category,
        problem_id=problem.problem_id,
        problem_path=problem.problem_path,
        workload_uuid=workload.uuid,
        row_index=workload.row_index,
        status=status,
        readiness_class=readiness_class,
        reasons=reasons,
        blocker_reports=blockers,
        layered_evidence=layers,
    )


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


def write_dataset_readiness(readiness: DatasetReadiness, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(readiness.to_json(), encoding="utf-8")
