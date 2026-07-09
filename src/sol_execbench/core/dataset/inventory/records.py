# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Record builders for dataset inventory sidecars."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import CustomInput, SafetensorsInput, Workload
from sol_execbench.core.dataset.inventory.models import (
    ProblemDefinitionInventory,
    WorkloadInventoryRecord,
)
from sol_execbench.core.dataset.inventory.reference_hints import (
    classify_reference_runtime_hints,
)


def op_family(definition: Definition) -> tuple[str, str]:
    """Infer an operation family hint for a problem definition."""
    if definition.op_type:
        return definition.op_type, "definition.op_type"
    lowered = definition.name.lower()
    for hint in ("matmul", "attention", "conv", "moe", "mamba", "embedding", "norm"):
        if hint in lowered:
            return hint, "definition.name"
    return "unknown", "unknown"


def direction(definition: Definition) -> tuple[str, str]:
    """Infer forward/backward direction from definition metadata."""
    text = f"{definition.name} {definition.description or ''}".lower()
    if "backward" in text or "grad" in text:
        return "backward", "definition.name_or_description"
    if "forward" in text:
        return "forward", "definition.name_or_description"
    return "unknown", "unknown"


def definition_record(
    definition: Definition, reference_path: Path | None
) -> ProblemDefinitionInventory:
    """Build inventory metadata for a parsed definition."""
    family, family_source = op_family(definition)
    direction_hint, direction_source = direction(definition)
    blocker_hints, false_positive_evidence = classify_reference_runtime_hints(
        definition, reference_path
    )
    return ProblemDefinitionInventory(
        name=definition.name,
        op_type=definition.op_type,
        op_family_hint=family,
        op_family_source=family_source,
        direction_hint=direction_hint,
        direction_source=direction_source,
        input_dtypes=[str(spec.dtype.value) for spec in definition.inputs.values()],
        output_dtypes=[str(spec.dtype.value) for spec in definition.outputs.values()],
        input_shapes={name: spec.shape for name, spec in definition.inputs.items()},
        output_shapes={name: spec.shape for name, spec in definition.outputs.items()},
        custom_inputs_entrypoint=definition.custom_inputs_entrypoint,
        reference_runtime_hints=blocker_hints,
        reference_runtime_false_positive_evidence=[
            dataclasses.asdict(evidence) for evidence in false_positive_evidence
        ],
    )


def shape_dict(
    shapes: dict[str, tuple[int, ...] | None],
) -> dict[str, list[int] | None]:
    """Convert tuple shapes to JSON-friendly lists."""
    return {
        name: list(shape) if shape is not None else None
        for name, shape in shapes.items()
    }


def workload_record(
    definition: Definition, workload: Workload, row_index: int
) -> WorkloadInventoryRecord:
    """Build inventory metadata for a parsed workload row."""
    input_kinds: dict[str, str] = {
        name: spec.type for name, spec in workload.inputs.items()
    }
    counts = Counter(input_kinds.values())
    safetensors_refs = [
        {"input": name, "path": spec.path, "tensor_key": spec.tensor_key}
        for name, spec in workload.inputs.items()
        if isinstance(spec, SafetensorsInput)
    ]
    shape_status = "resolved"
    try:
        input_shapes = shape_dict(definition.get_input_shapes(workload.axes))
        output_shapes = shape_dict(definition.get_output_shapes(workload.axes))
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
        uses_custom_inputs=any(
            isinstance(spec, CustomInput) for spec in workload.inputs.values()
        ),
        uses_safetensors=bool(safetensors_refs),
        safetensors_refs=safetensors_refs,
        input_dtypes={
            name: spec.dtype.value for name, spec in definition.inputs.items()
        },
        output_dtypes={
            name: spec.dtype.value for name, spec in definition.outputs.items()
        },
        resolved_input_shapes=input_shapes,
        resolved_output_shapes=output_shapes,
        shape_status=shape_status,
    )
