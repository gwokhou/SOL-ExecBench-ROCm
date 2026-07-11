# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""torch.fx extraction for AMD bound graphs."""

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
    _torch_dtype,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
)


def _try_fx_bound_graph(
    definition: Definition,
    workload: Workload,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
) -> BoundGraph | None:
    """Trace the reference with torch.fx and convert common nodes to BoundGraph."""
    try:
        import torch
        from torch.fx import Node, symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception:
        return None

    namespace: dict[str, Any] = {"torch": torch}
    try:
        exec(
            compile(definition.reference, f"<{definition.name}.reference>", "exec"),
            namespace,
        )
        run = namespace["run"]
        sample_inputs = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                sample_inputs.append(getattr(workload.inputs.get(name), "value", 0))
            else:
                sample_inputs.append(torch.zeros(shape, dtype=dtype, device="meta"))
        traced = symbolic_trace(run)
        ShapeProp(traced).propagate(*sample_inputs)
    except Exception:
        return None

    tensors = dict(declared_tensors)
    nodes: list[BoundGraphNode] = []
    edges: list[BoundEdge] = []
    warnings: list[str] = []
    node_outputs: dict[Node, str] = {}

    def append_fx_node(fx_node: Node) -> None:
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
        tensors[output_tensor_id] = BoundTensor(
            tensor_id=output_tensor_id,
            name=output_tensor_id,
            role=BoundTensorRole.INTERMEDIATE,
            shape=output_shape
            or _first_input_shape(input_tensor_ids, tensors, output_shapes),
            dtype=output_dtype
            or _first_input_dtype(input_tensor_ids, tensors, definition),
            producer_node_id=node_id,
            source=_fx_source_expression(fx_node),
        )
        bound_node = BoundGraphNode(
            node_id=node_id,
            op_family=_classification_family(classification),
            op_name=func_name,
            source_expression=_fx_source_expression(fx_node),
            input_tensor_ids=input_tensor_ids,
            output_tensor_ids=(output_tensor_id,),
            attributes={
                "trace_source": "torch.fx",
                **_fx_node_attributes(fx_node, func_name, classification),
            },
            confidence=classification.confidence,
            rationale=classification.rationale,
            conversion_status="not_converted",
        )
        nodes.append(bound_node)
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

    for fx_node in traced.graph.nodes:
        if fx_node.op == "placeholder":
            node_outputs[fx_node] = f"input:{fx_node.target}"
        elif fx_node.op in {"call_function", "call_method", "call_module"}:
            if _fx_node_name(fx_node) == "builtins.getattr":
                # ``Tensor.T`` is represented as ``getattr(tensor, "T")`` by
                # FX. It is a logical layout view, not a GPU operation, but
                # its source tensor remains a real GEMM operand. Preserve the
                # data dependency so downstream shape/FLOP inference receives
                # both operands instead of silently degrading the GEMM.
                if len(fx_node.args) >= 2 and fx_node.args[1] in {"T", "mT"}:
                    source_ids = _fx_input_tensor_ids(fx_node, node_outputs, definition)
                    if source_ids:
                        node_outputs[fx_node] = source_ids[0]
                continue
            if _fx_node_name(fx_node) in {
                "getitem",
                "_operator.getitem",
            } and _fx_tensor_meta(fx_node) == (None, None):
                # Indexing x.shape/stride metadata is host-side bookkeeping,
                # not a GPU operator in the bound graph.
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
                tensors[tensor_id] = replace(
                    tensors[tensor_id],
                    producer_node_id=source_tensor.producer_node_id,
                    source=source_tensor.tensor_id,
                )

    if not nodes:
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
