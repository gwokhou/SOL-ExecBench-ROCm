# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conversion of provider graphs into the AMD bound-graph IR."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph.annotations import (
    _annotate_family_graph,
)
from sol_execbench.core.scoring.amd_bound_graph.fx_helpers import (
    _classification_family,
    _classify_fx_node,
    _first_input_dtype,
    _first_input_shape,
    _flatten_fx_output_tensor_ids,
    _fx_input_tensor_ids,
    _fx_node_attributes,
    _fx_node_name,
    _fx_source_expression,
    _fx_tensor_meta,
    _static_data_movement_confidence,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
)


def bound_graph_from_fx_graph(
    traced: Any,
    definition: Definition,
    workload: Workload,
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
    *,
    trace_source: str,
) -> BoundGraph | None:
    """Convert a shape-annotated FX graph from an approved semantic provider."""
    tensors = dict(declared_tensors)
    nodes: list[BoundGraphNode] = []
    edges: list[BoundEdge] = []
    warnings: list[str] = []
    node_outputs: dict[Any, str] = {}
    metadata_complete = True

    def append_fx_node(fx_node: Any) -> None:
        nonlocal metadata_complete
        func_name, classification, warning = _classify_fx_node(fx_node)
        if warning is not None:
            warnings.append(warning)

        node_id = f"op_{len(nodes) + 1}"
        input_tensor_ids = tuple(
            dict.fromkeys(
                tensor_id
                for tensor_id in _fx_input_tensor_ids(fx_node, node_outputs, definition)
                if tensor_id in tensors
            )
        )
        output_tensor_id = f"tmp:{node_id}:0"
        output_shape, output_dtype = _fx_tensor_meta(fx_node)
        if trace_source == "torch.export" and (
            output_shape is None or output_dtype is None
        ):
            metadata_complete = False
            return
        resolved_output_shape = (
            output_shape
            if output_shape is not None
            else _first_input_shape(input_tensor_ids, tensors, output_shapes)
        )
        attributes = {
            "trace_source": trace_source,
            **_fx_node_attributes(fx_node, func_name, classification),
        }
        confidence, rationale = _static_data_movement_confidence(
            classification,
            attributes,
            input_tensor_ids,
            tensors,
            resolved_output_shape,
            output_shape is not None,
        )
        tensors[output_tensor_id] = BoundTensor(
            tensor_id=output_tensor_id,
            name=output_tensor_id,
            role=BoundTensorRole.INTERMEDIATE,
            shape=resolved_output_shape,
            dtype=output_dtype
            or _first_input_dtype(input_tensor_ids, tensors, definition),
            producer_node_id=node_id,
            source=_fx_source_expression(fx_node),
        )
        nodes.append(
            BoundGraphNode(
                node_id=node_id,
                op_family=_classification_family(classification),
                op_name=func_name,
                source_expression=_fx_source_expression(fx_node),
                input_tensor_ids=input_tensor_ids,
                output_tensor_ids=(output_tensor_id,),
                attributes=attributes,
                confidence=confidence,
                rationale=rationale,
                conversion_status="not_converted",
            )
        )
        node_outputs[fx_node] = output_tensor_id
        for input_tensor_id in input_tensor_ids:
            edges.append(
                BoundEdge(
                    edge_id=f"edge_{len(edges) + 1}",
                    source_tensor_id=input_tensor_id,
                    target_node_id=node_id,
                    role="input",
                )
            )

    input_names = iter(definition.inputs)
    matched_output_names: set[str] = set()
    for fx_node in traced.graph.nodes:
        if fx_node.op == "placeholder":
            name = str(fx_node.target)
            if name not in definition.inputs:
                name = next(input_names, "")
            if name not in definition.inputs:
                return None
            node_outputs[fx_node] = f"input:{name}"
        elif fx_node.op in {"call_function", "call_method", "call_module"}:
            if _fx_node_name(fx_node) in {
                "aten._assert_tensor_metadata.default",
                "_assert_tensor_metadata",
                "torch._assert_tensor_metadata",
            }:
                continue
            if _fx_node_name(fx_node) == "builtins.getattr":
                if len(fx_node.args) >= 2 and fx_node.args[1] in {"T", "mT"}:
                    source_ids = _fx_input_tensor_ids(fx_node, node_outputs, definition)
                    if source_ids:
                        node_outputs[fx_node] = source_ids[0]
                continue
            if _fx_node_name(fx_node) in {
                "getitem",
                "_operator.getitem",
            } and _fx_tensor_meta(fx_node) == (None, None):
                continue
            append_fx_node(fx_node)
        elif fx_node.op == "output":
            output_tensor_ids = [
                tensor_id
                for tensor_id in _flatten_fx_output_tensor_ids(
                    fx_node.args, node_outputs
                )
                if tensor_id in tensors
            ]
            for index, output_name in enumerate(definition.outputs):
                if index >= len(output_tensor_ids):
                    break
                tensor_id = f"output:{output_name}"
                source_tensor = tensors[output_tensor_ids[index]]
                declared_output = tensors[tensor_id]
                if trace_source == "torch.export" and (
                    source_tensor.shape != declared_output.shape
                    or source_tensor.dtype != declared_output.dtype
                ):
                    return None
                tensors[tensor_id] = replace(
                    declared_output,
                    producer_node_id=source_tensor.producer_node_id,
                    source=source_tensor.tensor_id,
                )
                matched_output_names.add(output_name)

    if not nodes or not metadata_complete:
        return None
    if trace_source == "torch.export" and matched_output_names != set(
        definition.outputs
    ):
        return None

    return _annotate_family_graph(
        BoundGraph(
            definition=definition.name,
            workload_uuid=workload.uuid,
            nodes=tuple(nodes),
            tensors=tensors,
            edges=tuple(edges),
            warnings=tuple(dict.fromkeys(warnings)),
        )
    )
