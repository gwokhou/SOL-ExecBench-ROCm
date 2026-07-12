# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Dataset inventory sidecar generation."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.categories import validate_categories
from sol_execbench.core.dataset.inventory.io import (
    load_definition as _load_definition,
    load_workload as _load_workload,
    rel as _rel,
    solution_files as _solution_files,
    write_dataset_inventory,
)
from sol_execbench.core.dataset.inventory.models import (
    INVENTORY_SCHEMA_VERSION,
    CategoryInventoryRecord,
    DatasetInventory,
    InventoryDenominators,
    InventoryDiagnostic,
    ProblemDefinitionInventory,
    ProblemInventoryRecord,
    WorkloadInventoryRecord,
)
from sol_execbench.core.dataset.inventory.records import (
    definition_record as _definition_record,
    workload_record as _workload_record,
)
from sol_execbench.core.dataset.inventory.reference_hints import (
    NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS,
    NVIDIA_RUNTIME_BLOCKER_HINTS,
    ReferenceRuntimeHintEvidence,
)
from sol_execbench.core.timestamps import utc_timestamp


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
            diagnostics.append(
                InventoryDiagnostic(
                    code="missing_category",
                    category=category,
                    message=f"Missing category: {category}",
                )
            )
            category_records.append(
                CategoryInventoryRecord(name=category, denominators=cat_denoms)
            )
            continue
        for problem_dir in sorted(
            path for path in category_dir.iterdir() if path.is_dir()
        ):
            cat_denoms.discovered_problems += 1
            definition_path = problem_dir / "definition.json"
            workload_path = problem_dir / "workload.jsonl"
            problem_path = _rel(problem_dir, root)
            missing = [
                path.name
                for path in (definition_path, workload_path)
                if not path.is_file()
            ]
            if missing:
                cat_denoms.missing_required_files += len(missing)
                diagnostics.append(
                    InventoryDiagnostic(
                        code="missing_required_file",
                        category=category,
                        problem_path=problem_path,
                        message=", ".join(missing),
                    )
                )
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
                diagnostics.append(
                    InventoryDiagnostic(
                        code="definition_schema_failure",
                        category=category,
                        problem_path=problem_path,
                        message=failure or "definition parse failed",
                    )
                )
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
            for row_index, line in enumerate(
                workload_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                workload, workload_failure = _load_workload(line)
                if workload is None:
                    cat_denoms.schema_failures += 1
                    diagnostics.append(
                        InventoryDiagnostic(
                            code="workload_schema_failure",
                            category=category,
                            problem_path=problem_path,
                            row_index=row_index,
                            message=workload_failure or "workload parse failed",
                        )
                    )
                    workloads.append(
                        WorkloadInventoryRecord(
                            uuid=None,
                            row_index=row_index,
                            schema_status="schema_failure",
                            schema_failure=workload_failure,
                        )
                    )
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
                    reference_available=reference_path.exists()
                    or bool(definition.reference),
                    solution_files=_solution_files(problem_dir),
                    schema_status="parsed",
                    definition=_definition_record(definition, reference_path),
                    workload_count=len(workloads),
                    workloads=workloads,
                )
            )

        total.add(cat_denoms)
        category_records.append(
            CategoryInventoryRecord(name=category, denominators=cat_denoms)
        )

    inventory = DatasetInventory(
        created_at=created_at or utc_timestamp(),
        root=root.as_posix(),
        manifest_path=manifest_path.as_posix() if manifest_path else None,
        selected_categories=selected,
        categories=category_records,
        problems=problems,
        denominators=total,
        diagnostics=diagnostics,
    )
    return inventory.with_checksum()


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "NVIDIA_LEXICAL_FALSE_POSITIVE_HINTS",
    "NVIDIA_RUNTIME_BLOCKER_HINTS",
    "CategoryInventoryRecord",
    "DatasetInventory",
    "InventoryDenominators",
    "InventoryDiagnostic",
    "ProblemDefinitionInventory",
    "ProblemInventoryRecord",
    "ReferenceRuntimeHintEvidence",
    "WorkloadInventoryRecord",
    "build_dataset_inventory",
    "write_dataset_inventory",
]
