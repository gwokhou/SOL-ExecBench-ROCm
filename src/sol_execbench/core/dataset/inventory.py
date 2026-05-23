# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset inventory sidecar generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import CustomInput, SafetensorsInput, Workload

from .categories import validate_categories
from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp

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
        payload = self.model_dump(mode="json")
        payload["inventory_checksum"] = None
        return self.model_copy(
            update={"inventory_checksum": DatasetManifestChecksum(value=stable_json_checksum(payload))}
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_definition(path: Path) -> tuple[Definition | None, str | None]:
    try:
        return Definition.model_validate_json(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def _load_workload(line: str) -> tuple[Workload | None, str | None]:
    try:
        return Workload.model_validate_json(line), None
    except Exception as exc:
        return None, str(exc)


def _op_family(definition: Definition) -> tuple[str, str]:
    if definition.op_type:
        return definition.op_type, "definition.op_type"
    lowered = definition.name.lower()
    for hint in ("matmul", "attention", "conv", "moe", "mamba", "embedding", "norm"):
        if hint in lowered:
            return hint, "definition.name"
    return "unknown", "unknown"


def _direction(definition: Definition) -> tuple[str, str]:
    text = f"{definition.name} {definition.description or ''}".lower()
    if "backward" in text or "grad" in text:
        return "backward", "definition.name_or_description"
    if "forward" in text:
        return "forward", "definition.name_or_description"
    return "unknown", "unknown"


NVIDIA_RUNTIME_HINTS = ("cupy", "cuda.c", "cuda runtime", "nvrtc", "cublas", "cutlass")


def _reference_runtime_hints(definition: Definition, reference_path: Path | None) -> list[str]:
    text = definition.reference.lower()
    if reference_path is not None and reference_path.is_file():
        text += "\n" + reference_path.read_text(encoding="utf-8").lower()
    return sorted(hint for hint in NVIDIA_RUNTIME_HINTS if hint in text)


def _definition_record(definition: Definition, reference_path: Path | None) -> ProblemDefinitionInventory:
    family, family_source = _op_family(definition)
    direction, direction_source = _direction(definition)
    return ProblemDefinitionInventory(
        name=definition.name,
        op_type=definition.op_type,
        op_family_hint=family,
        op_family_source=family_source,
        direction_hint=direction,
        direction_source=direction_source,
        input_dtypes=[str(spec.dtype.value) for spec in definition.inputs.values()],
        output_dtypes=[str(spec.dtype.value) for spec in definition.outputs.values()],
        input_shapes={name: spec.shape for name, spec in definition.inputs.items()},
        output_shapes={name: spec.shape for name, spec in definition.outputs.items()},
        custom_inputs_entrypoint=definition.custom_inputs_entrypoint,
        reference_runtime_hints=_reference_runtime_hints(definition, reference_path),
    )


def _shape_dict(shapes: dict[str, tuple[int, ...] | None]) -> dict[str, list[int] | None]:
    return {name: list(shape) if shape is not None else None for name, shape in shapes.items()}


def _workload_record(definition: Definition, workload: Workload, row_index: int) -> WorkloadInventoryRecord:
    input_kinds = {name: spec.type for name, spec in workload.inputs.items()}
    counts = Counter(input_kinds.values())
    safetensors_refs = [
        {"input": name, "path": spec.path, "tensor_key": spec.tensor_key}
        for name, spec in workload.inputs.items()
        if isinstance(spec, SafetensorsInput)
    ]
    shape_status = "resolved"
    try:
        input_shapes = _shape_dict(definition.get_input_shapes(workload.axes))
        output_shapes = _shape_dict(definition.get_output_shapes(workload.axes))
    except Exception:
        shape_status = "unresolved"
        input_shapes = {}
        output_shapes = {}
    return WorkloadInventoryRecord(
        uuid=workload.uuid,
        row_index=row_index,
        schema_status="parsed",
        axes=workload.axes,
        input_kinds=input_kinds,
        input_kind_counts=dict(sorted(counts.items())),
        uses_custom_inputs=any(isinstance(spec, CustomInput) for spec in workload.inputs.values()),
        uses_safetensors=bool(safetensors_refs),
        safetensors_refs=safetensors_refs,
        input_dtypes={name: spec.dtype.value for name, spec in definition.inputs.items()},
        output_dtypes={name: spec.dtype.value for name, spec in definition.outputs.items()},
        resolved_input_shapes=input_shapes,
        resolved_output_shapes=output_shapes,
        shape_status=shape_status,
    )


def _solution_files(problem_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in problem_dir.iterdir()
        if path.is_file() and (path.name.startswith("solution") or path.suffix == ".so")
    )


def build_dataset_inventory(
    root: Path,
    *,
    categories: tuple[str, ...] | None = None,
    manifest_path: Path | None = None,
    created_at: str | None = None,
) -> DatasetInventory:
    root = Path(root)
    selected = validate_categories(categories)
    problems: list[ProblemInventoryRecord] = []
    category_records: list[CategoryInventoryRecord] = []
    diagnostics: list[InventoryDiagnostic] = []
    total = InventoryDenominators()

    for category in selected:
        cat_denoms = InventoryDenominators()
        category_dir = root / category
        if not category_dir.is_dir():
            diagnostics.append(InventoryDiagnostic(code="missing_category", category=category, path=None, message=f"Missing category: {category}"))
            category_records.append(CategoryInventoryRecord(name=category, denominators=cat_denoms))
            continue
        for problem_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            cat_denoms.discovered_problems += 1
            definition_path = problem_dir / "definition.json"
            workload_path = problem_dir / "workload.jsonl"
            problem_path = _rel(problem_dir, root)
            missing = [path.name for path in (definition_path, workload_path) if not path.is_file()]
            if missing:
                cat_denoms.missing_required_files += len(missing)
                diagnostics.append(InventoryDiagnostic(code="missing_required_file", category=category, problem_path=problem_path, message=", ".join(missing)))
                problems.append(
                    ProblemInventoryRecord(
                        category=category,
                        problem_id=problem_path,
                        problem_path=problem_path,
                        definition_path=_rel(definition_path, root),
                        workload_path=_rel(workload_path, root),
                        schema_status="missing_required_file",
                        schema_failure=", ".join(missing),
                    )
                )
                continue

            definition, failure = _load_definition(definition_path)
            if definition is None:
                cat_denoms.schema_failures += 1
                diagnostics.append(InventoryDiagnostic(code="definition_schema_failure", category=category, problem_path=problem_path, message=failure or "definition parse failed"))
                problems.append(
                    ProblemInventoryRecord(
                        category=category,
                        problem_id=problem_path,
                        problem_path=problem_path,
                        definition_path=_rel(definition_path, root),
                        workload_path=_rel(workload_path, root),
                        schema_status="schema_failure",
                        schema_failure=failure,
                    )
                )
                continue

            workloads: list[WorkloadInventoryRecord] = []
            for row_index, line in enumerate(workload_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                workload, workload_failure = _load_workload(line)
                if workload is None:
                    cat_denoms.schema_failures += 1
                    diagnostics.append(InventoryDiagnostic(code="workload_schema_failure", category=category, problem_path=problem_path, row_index=row_index, message=workload_failure or "workload parse failed"))
                    workloads.append(WorkloadInventoryRecord(uuid=None, row_index=row_index, schema_status="schema_failure", schema_failure=workload_failure))
                    continue
                cat_denoms.parsed_workloads += 1
                workloads.append(_workload_record(definition, workload, row_index))

            cat_denoms.parsed_problems += 1
            reference_path = problem_dir / "reference.py"
            problems.append(
                ProblemInventoryRecord(
                    category=category,
                    problem_id=problem_path,
                    problem_path=problem_path,
                    definition_path=_rel(definition_path, root),
                    workload_path=_rel(workload_path, root),
                    reference_path=_rel(reference_path, root)
                    if reference_path.exists()
                    else None,
                    reference_available=reference_path.exists() or bool(definition.reference),
                    solution_files=_solution_files(problem_dir),
                    schema_status="parsed",
                    definition=_definition_record(definition, reference_path),
                    workload_count=len(workloads),
                    workloads=workloads,
                )
            )

        total.add(cat_denoms)
        category_records.append(CategoryInventoryRecord(name=category, denominators=cat_denoms))

    inventory = DatasetInventory(created_at=created_at or utc_timestamp(), root=root.as_posix(), manifest_path=manifest_path.as_posix() if manifest_path else None, selected_categories=selected, categories=category_records, problems=problems, denominators=total, diagnostics=diagnostics)
    return inventory.with_checksum()


def write_dataset_inventory(inventory: DatasetInventory, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.to_json(), encoding="utf-8")
