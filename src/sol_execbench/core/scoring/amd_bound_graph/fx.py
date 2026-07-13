# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Torch semantic graph providers converted into the bound-graph IR."""

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
        from torch.fx import symbolic_trace
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

    return _bound_graph_from_fx_graph(
        traced,
        definition,
        workload,
        output_shapes,
        declared_tensors,
        trace_source="torch.fx",
    )


def _try_torch_export_bound_graph(
    definition: Definition,
    workload: Workload,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
) -> BoundGraph | None:
    """Export a complete ATen graph, failing closed when export cannot prove it."""
    try:
        import torch
    except Exception:
        return None

    namespace: dict[str, Any] = {"torch": torch}
    try:
        exec(
            compile(definition.reference, f"<{definition.name}.reference>", "exec"),
            namespace,
        )
        run = namespace["run"]

        class ReferenceModule(torch.nn.Module):
            def forward(self, *args: Any) -> Any:
                return run(*args)

        sample_inputs: list[Any] = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                # Data-dependent dimensions cannot receive authority from this
                # concrete export attempt.
                return None
            sample_inputs.append(torch.zeros(shape, dtype=dtype, device="cpu"))
        exported = torch.export.export(
            ReferenceModule(), tuple(sample_inputs), strict=True
        )
    except Exception:
        return None

    return _bound_graph_from_fx_graph(
        exported.graph_module,
        definition,
        workload,
        output_shapes,
        declared_tensors,
        trace_source="torch.export",
    )


def _bound_graph_from_fx_graph(
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
            # Do not borrow a shape from a neighbouring tensor for authority.
            # Missing FakeTensor output metadata invalidates this semantic
            # capture rather than silently turning it into an AST-like guess.
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
        bound_node = BoundGraphNode(
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
