# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""torch.fx extraction for AMD bound graphs."""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import Any

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_classification import (
    CallClassification as _CallClassification,
    classify_call as _classify_call,
    movement_kind_for_name as _movement_kind_for_name,
)
from sol_execbench.core.scoring.amd_bound_graph_annotations import _annotate_family_graph
from sol_execbench.core.scoring.amd_bound_graph_common import (
    _MISSING,
    _axis_from_values,
    _classification_family,
    _convolution_attributes,
    _fx_tensor_meta,
    _memory_bound_call_attributes,
    _moe_call_attributes,
    _ssm_mamba_call_attributes,
    _target_dtype_from_values,
)
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

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
        func_name = _fx_node_name(fx_node)
        classification = _classify_call(func_name)
        if fx_node.op == "call_function" and fx_node.target is operator.matmul:
            func_name = "@"
            classification = _CallClassification(
                OpFamily.GEMM.value,
                EstimateConfidence.SUPPORTED,
                "recognized traced matrix multiply",
            )
        if fx_node.op == "call_function" and fx_node.target in {
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.pow,
        }:
            classification = _CallClassification(
                OpFamily.ELEMENTWISE.value,
                EstimateConfidence.INEXACT,
                "recognized traced elementwise operation",
            )
        if classification is None:
            warnings.append(f"unsupported_operator:{func_name or '<unknown>'}")
            classification = _CallClassification(
                OpFamily.UNSUPPORTED.value,
                EstimateConfidence.UNSUPPORTED,
                "unsupported traced operation preserved as graph evidence",
            )

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

def _torch_dtype(torch: Any, dtype: DType) -> Any:
    return {
        DType.FLOAT64: torch.float64,
        DType.FLOAT32: torch.float32,
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.INT64: torch.int64,
        DType.INT32: torch.int32,
        DType.INT16: torch.int16,
        DType.INT8: torch.int8,
        DType.BOOL: torch.bool,
        DType.FLOAT8_E4M3FN: torch.float16,
        DType.FLOAT8_E5M2: torch.float16,
        DType.FLOAT4_E2M1: torch.float16,
        DType.FLOAT4_E2M1FN_X2: torch.float16,
    }[dtype]


def _fx_node_name(node: Any) -> str:
    target = node.target
    if isinstance(target, str):
        return target
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        module = target.__module__
        name = target.__name__
        if module == "_operator":
            return name
        if module == "torch._C._linalg" and name.startswith("linalg_"):
            return f"torch.linalg.{name.removeprefix('linalg_')}"
        return f"{module}.{name}"
    return str(target)


def _fx_input_tensor_ids(
    node: Any,
    node_outputs: dict[Any, str],
    definition: Definition,
) -> tuple[str, ...]:
    result: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif value in node_outputs:
            result.append(node_outputs[value])
        elif isinstance(value, str) and value in definition.inputs:
            result.append(f"input:{value}")

    collect(node.args)
    collect(node.kwargs)
    return tuple(result)


def _flatten_fx_output_tensor_ids(
    value: Any, node_outputs: dict[Any, str]
) -> tuple[str, ...]:
    result: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)
        elif item in node_outputs:
            result.append(node_outputs[item])

    collect(value)
    return tuple(result)

def _fx_source_expression(node: Any) -> str:
    if (
        node.op == "call_function"
        and node.target is operator.matmul
        and len(node.args) >= 2
    ):
        return f"{node.args[0]} @ {node.args[1]}"
    func_name = _fx_node_name(node)
    args = ", ".join(str(arg) for arg in node.args)
    return f"{func_name}({args})"


def _fx_node_attributes(
    node: Any,
    func_name: str,
    classification: _CallClassification,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    movement_kind = _movement_kind_for_name(leaf_name)
    if movement_kind is not None:
        attributes["movement_kind"] = movement_kind

    op_family = _classification_family(classification)
    if op_family in {
        OpFamily.REDUCTION,
        OpFamily.NORMALIZATION,
        OpFamily.SOFTMAX,
    }:
        axis = _axis_from_values(node.args[1:], node.kwargs)
        if axis is not _MISSING:
            attributes["dim"] = axis
            attributes["axis_source"] = "attribute"

    target_dtype = _target_dtype_from_values(leaf_name, node.args[1:], node.kwargs)
    if target_dtype is not None:
        attributes["target_dtype"] = target_dtype
    if op_family == OpFamily.CONVOLUTION:
        attributes.update(
            _convolution_attributes(leaf_name, node.args, node.kwargs, node)
        )
    if op_family == OpFamily.EMBEDDING_POSITIONAL:
        attributes.update(
            _memory_bound_call_attributes(leaf_name, node.args, node.kwargs, node)
        )
    if op_family == OpFamily.MOE:
        attributes.update(_moe_call_attributes(leaf_name, node.args, node.kwargs))
    if op_family == OpFamily.SSM_MAMBA:
        attributes.update(_ssm_mamba_call_attributes(leaf_name))
    return attributes

def _first_input_shape(
    input_tensor_ids: tuple[str, ...],
    tensors: dict[str, BoundTensor],
    output_shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    for tensor_id in input_tensor_ids:
        tensor = tensors.get(tensor_id)
        if tensor and tensor.shape is not None:
            return tensor.shape
    for shape in output_shapes.values():
        if shape is not None:
            return shape
    return None


def _first_input_dtype(
    input_tensor_ids: tuple[str, ...],
    tensors: dict[str, BoundTensor],
    definition: Definition,
) -> str:
    for tensor_id in input_tensor_ids:
        tensor = tensors.get(tensor_id)
        if tensor:
            return tensor.dtype
    if definition.outputs:
        return next(iter(definition.outputs.values())).dtype.value
    return "unknown"
