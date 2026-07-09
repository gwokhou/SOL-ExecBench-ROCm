# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset inventory sidecar models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


INVENTORY_SCHEMA_VERSION = "sol_execbench.dataset_inventory.v1"


class InventoryDiagnostic(BaseModel):
    code: str
    severity: str = "error"
    category: str | None = None
    problem_path: str | None = None
    row_index: int | None = None
    message: str


class InventoryDenominators(BaseModel):
    discovered_problems: int = 0
    parsed_problems: int = 0
    parsed_workloads: int = 0
    schema_failures: int = 0
    missing_required_files: int = 0

    def add(self, other: "InventoryDenominators") -> None:
        self.discovered_problems += other.discovered_problems
        self.parsed_problems += other.parsed_problems
        self.parsed_workloads += other.parsed_workloads
        self.schema_failures += other.schema_failures
        self.missing_required_files += other.missing_required_files


class WorkloadInventoryRecord(BaseModel):
    uuid: str | None
    row_index: int
    schema_status: str
    schema_failure: str | None = None
    axes: dict[str, int] = Field(default_factory=dict)
    input_kinds: dict[str, str] = Field(default_factory=dict)
    input_kind_counts: dict[str, int] = Field(default_factory=dict)
    uses_custom_inputs: bool = False
    uses_safetensors: bool = False
    safetensors_refs: list[dict[str, str]] = Field(default_factory=list)
    input_dtypes: dict[str, str] = Field(default_factory=dict)
    output_dtypes: dict[str, str] = Field(default_factory=dict)
    resolved_input_shapes: dict[str, list[int] | None] = Field(default_factory=dict)
    resolved_output_shapes: dict[str, list[int] | None] = Field(default_factory=dict)
    shape_status: str = "unknown"


class ProblemDefinitionInventory(BaseModel):
    name: str | None = None
    op_type: str | None = None
    op_family_hint: str = "unknown"
    op_family_source: str = "unknown"
    direction_hint: str = "unknown"
    direction_source: str = "unknown"
    input_dtypes: list[str] = Field(default_factory=list)
    output_dtypes: list[str] = Field(default_factory=list)
    input_shapes: dict[str, list[str] | None] = Field(default_factory=dict)
    output_shapes: dict[str, list[str] | None] = Field(default_factory=dict)
    custom_inputs_entrypoint: str | None = None
    reference_runtime_hints: list[str] = Field(default_factory=list)
    reference_runtime_false_positive_evidence: list[dict] = Field(default_factory=list)


class ProblemInventoryRecord(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    definition_path: str
    workload_path: str
    reference_path: str | None = None
    reference_available: bool = False
    solution_files: list[str] = Field(default_factory=list)
    schema_status: str
    schema_failure: str | None = None
    definition: ProblemDefinitionInventory | None = None
    workload_count: int = 0
    workloads: list[WorkloadInventoryRecord] = Field(default_factory=list)


class CategoryInventoryRecord(BaseModel):
    name: str
    denominators: InventoryDenominators


class DatasetInventory(BaseModel):
    schema_version: str = INVENTORY_SCHEMA_VERSION
    created_at: str
    root: str
    manifest_path: str | None = None
    selected_categories: tuple[str, ...]
    categories: list[CategoryInventoryRecord]
    problems: list[ProblemInventoryRecord]
    denominators: InventoryDenominators
    diagnostics: list[InventoryDiagnostic]
    inventory_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetInventory":
        return self.model_copy(
            update={
                "inventory_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "inventory_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
