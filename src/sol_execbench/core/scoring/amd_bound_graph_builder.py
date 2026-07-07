# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public builder for structured AMD bound graphs."""

from __future__ import annotations

import ast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph_annotations import _annotate_family_graph
from sol_execbench.core.scoring.amd_bound_graph_ast import _AstBoundGraphExtractor
from sol_execbench.core.scoring.amd_bound_graph_fx import _try_fx_bound_graph
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundTensor,
    BoundTensorRole,
)

def build_bound_graph(definition: Definition, workload: Workload) -> BoundGraph:
    """Build a structured bound graph for a concrete definition/workload pair."""
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    tensors = _declared_tensors(definition, input_shapes, output_shapes)
    warnings: list[str] = []

    fx_graph = _try_fx_bound_graph(
        definition, workload, input_shapes, output_shapes, tensors
    )
    if fx_graph is not None:
        return fx_graph

    warnings.append("dynamic_trace_failed")

    try:
        tree = ast.parse(definition.reference, mode="exec")
    except SyntaxError as exc:
        raise ValueError(
            f"Reference must be valid Python code for graph extraction: {exc}"
        ) from exc

    extractor = _AstBoundGraphExtractor(
        definition=definition,
        initial_tensors=tensors,
        output_names=tuple(definition.outputs.keys()),
        output_shapes=output_shapes,
    )
    nodes, extracted_tensors, edges, extractor_warnings = extractor.extract(tree)
    warnings.extend(extractor_warnings)

    return _annotate_family_graph(
        BoundGraph(
            definition=definition.name,
            workload_uuid=workload.uuid,
            nodes=nodes,
            tensors=extracted_tensors,
            edges=edges,
            warnings=tuple(dict.fromkeys(warnings)),
        )
    )

def _declared_tensors(
    definition: Definition,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
) -> dict[str, BoundTensor]:
    tensors: dict[str, BoundTensor] = {}
    for name, spec in definition.inputs.items():
        tensor_id = f"input:{name}"
        tensors[tensor_id] = BoundTensor(
            tensor_id=tensor_id,
            name=name,
            role=BoundTensorRole.INPUT,
            shape=input_shapes[name],
            dtype=spec.dtype.value,
            producer_node_id=None,
            source=f"definition.inputs.{name}",
        )
    for name, spec in definition.outputs.items():
        tensor_id = f"output:{name}"
        tensors[tensor_id] = BoundTensor(
            tensor_id=tensor_id,
            name=name,
            role=BoundTensorRole.OUTPUT,
            shape=output_shapes[name],
            dtype=spec.dtype.value,
            producer_node_id=None,
            source=f"definition.outputs.{name}",
        )
    return tensors
