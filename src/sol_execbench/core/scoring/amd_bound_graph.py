# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured AMD bound graph IR for SOL/SOLAR-derived analysis."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


class BoundTensorRole(str, Enum):
    """Role of a tensor in a bound graph."""

    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"


class OpFamily(str, Enum):
    """Paper-aligned operation family for SOLAR graph extraction."""

    ATTENTION = "attention"
    MOE = "moe"
    NORMALIZATION = "normalization"
    EMBEDDING_POSITIONAL = "embedding_positional"
    LINEAR_PROJECTION = "linear_projection"
    GEMM = "gemm"
    MLP_ACTIVATION = "mlp_activation"
    CONVOLUTION = "convolution"
    SSM_MAMBA = "ssm_mamba"
    SOFTMAX = "softmax"
    REDUCTION = "reduction"
    ELEMENTWISE = "elementwise"
    DATA_MOVEMENT = "data_movement"
    DTYPE_CONVERSION = "dtype_conversion"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BoundTensor:
    """Tensor metadata bound to a concrete workload."""

    tensor_id: str
    name: str
    role: BoundTensorRole
    shape: tuple[int, ...] | None
    dtype: str
    producer_node_id: str | None
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "role": self.role.value,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "producer_node_id": self.producer_node_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class BoundEdge:
    """Producer/consumer edge between a tensor and an operation."""

    edge_id: str
    source_tensor_id: str
    target_node_id: str
    role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source_tensor_id": self.source_tensor_id,
            "target_node_id": self.target_node_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class BoundGraphNode:
    """Operation node in a structured AMD bound graph."""

    node_id: str
    op_family: OpFamily
    op_name: str
    source_expression: str
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    attributes: dict[str, object]
    confidence: EstimateConfidence
    rationale: str
    einsum_hint: str | None = None
    conversion_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "op_family": self.op_family.value,
            "op_name": self.op_name,
            "source_expression": self.source_expression,
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_ids": list(self.output_tensor_ids),
            "attributes": dict(self.attributes),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "einsum_hint": self.einsum_hint,
            "conversion_status": self.conversion_status,
        }


@dataclass(frozen=True)
class BoundGraph:
    """Structured bound graph for one Definition and Workload."""

    definition: str
    workload_uuid: str
    nodes: tuple[BoundGraphNode, ...]
    tensors: dict[str, BoundTensor]
    edges: tuple[BoundEdge, ...]
    warnings: tuple[str, ...]
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": {
                tensor_id: tensor.to_dict()
                for tensor_id, tensor in sorted(self.tensors.items())
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
            "derived": self.derived,
        }


@dataclass(frozen=True)
class _CallClassification:
    op_family: OpFamily
    confidence: EstimateConfidence
    rationale: str


_CALL_CLASSIFIERS: tuple[tuple[set[str], _CallClassification], ...] = (
    (
        {"matmul", "mm", "bmm"},
        _CallClassification(OpFamily.GEMM, EstimateConfidence.SUPPORTED, "recognized matrix multiply"),
    ),
    (
        {"linear"},
        _CallClassification(
            OpFamily.LINEAR_PROJECTION,
            EstimateConfidence.SUPPORTED,
            "recognized linear projection",
        ),
    ),
    (
        {"sum", "mean", "amax", "max", "amin", "min", "var", "std"},
        _CallClassification(
            OpFamily.REDUCTION,
            EstimateConfidence.INEXACT,
            "recognized reduction with conservative later-modeling semantics",
        ),
    ),
    (
        {"layer_norm", "group_norm", "rms_norm", "norm", "rsqrt"},
        _CallClassification(
            OpFamily.NORMALIZATION,
            EstimateConfidence.INEXACT,
            "recognized normalization-like operation",
        ),
    ),
    (
        {"softmax", "log_softmax"},
        _CallClassification(
            OpFamily.SOFTMAX,
            EstimateConfidence.INEXACT,
            "recognized softmax-like operation",
        ),
    ),
    (
        {"relu", "gelu", "silu", "sigmoid", "tanh", "exp", "sqrt"},
        _CallClassification(
            OpFamily.MLP_ACTIVATION,
            EstimateConfidence.INEXACT,
            "recognized activation operation",
        ),
    ),
    (
        {
            "t",
            "transpose",
            "permute",
            "view",
            "reshape",
            "flatten",
            "contiguous",
            "unsqueeze",
            "squeeze",
            "expand",
            "broadcast_to",
        },
        _CallClassification(
            OpFamily.DATA_MOVEMENT,
            EstimateConfidence.INEXACT,
            "recognized view or data-movement operation",
        ),
    ),
    (
        {"to", "type", "float", "half", "bfloat16", "double", "bool", "int", "long"},
        _CallClassification(
            OpFamily.DTYPE_CONVERSION,
            EstimateConfidence.INEXACT,
            "recognized dtype conversion operation",
        ),
    ),
)

_LOGICAL_VIEW_NAMES = {
    "t",
    "transpose",
    "permute",
    "view",
    "reshape",
    "flatten",
    "unsqueeze",
    "squeeze",
}
_BROADCAST_VIEW_NAMES = {"expand", "broadcast_to"}
_MATERIALIZED_MOVEMENT_NAMES = {"contiguous"}
_DTYPE_METHOD_TARGETS = {
    "float": DType.FLOAT32.value,
    "half": DType.FLOAT16.value,
    "bfloat16": DType.BFLOAT16.value,
    "double": DType.FLOAT64.value,
    "bool": DType.BOOL.value,
    "int": DType.INT32.value,
    "long": DType.INT64.value,
}


def build_bound_graph(definition: Definition, workload: Workload) -> BoundGraph:
    """Build a structured bound graph for a concrete definition/workload pair."""
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    tensors = _declared_tensors(definition, input_shapes, output_shapes)
    warnings: list[str] = []

    fx_graph = _try_fx_bound_graph(definition, workload, input_shapes, output_shapes, tensors)
    if fx_graph is not None:
        return fx_graph

    warnings.append("dynamic_trace_failed")

    try:
        tree = ast.parse(definition.reference, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Reference must be valid Python code for graph extraction: {exc}") from exc

    extractor = _AstBoundGraphExtractor(
        definition=definition,
        initial_tensors=tensors,
        output_names=tuple(definition.outputs.keys()),
        output_shapes=output_shapes,
    )
    nodes, extracted_tensors, edges, extractor_warnings = extractor.extract(tree)
    warnings.extend(extractor_warnings)

    return _annotate_attention_graph(BoundGraph(
        definition=definition.name,
        workload_uuid=workload.uuid,
        nodes=nodes,
        tensors=extracted_tensors,
        edges=edges,
        warnings=tuple(dict.fromkeys(warnings)),
    ))


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
        exec(compile(definition.reference, f"<{definition.name}.reference>", "exec"), namespace)
        run = namespace["run"]
        sample_inputs = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                sample_inputs.append(torch.tensor(0, dtype=dtype))
            else:
                sample_inputs.append(torch.zeros(shape, dtype=dtype))
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
                OpFamily.GEMM,
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
                OpFamily.ELEMENTWISE,
                EstimateConfidence.INEXACT,
                "recognized traced elementwise operation",
            )
        if classification is None:
            warnings.append(f"unsupported_operator:{func_name or '<unknown>'}")
            classification = _CallClassification(
                OpFamily.UNSUPPORTED,
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
            shape=output_shape or _first_input_shape(input_tensor_ids, tensors, output_shapes),
            dtype=output_dtype or _first_input_dtype(input_tensor_ids, tensors, definition),
            producer_node_id=node_id,
            source=_fx_source_expression(fx_node),
        )
        bound_node = BoundGraphNode(
            node_id=node_id,
            op_family=classification.op_family,
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
                for tensor_id in _flatten_fx_output_tensor_ids(fx_node.args, node_outputs)
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

    return _annotate_attention_graph(BoundGraph(
        definition=definition.name,
        workload_uuid=workload.uuid,
        nodes=tuple(nodes),
        tensors=tensors,
        edges=tuple(edges),
        warnings=tuple(dict.fromkeys(warnings)),
    ))


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


def _flatten_fx_output_tensor_ids(value: Any, node_outputs: dict[Any, str]) -> tuple[str, ...]:
    result: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)
        elif item in node_outputs:
            result.append(node_outputs[item])

    collect(value)
    return tuple(result)


def _fx_tensor_meta(node: Any) -> tuple[tuple[int, ...] | None, str | None]:
    meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    if meta is None:
        return None, None
    shape = tuple(int(dim) for dim in meta.shape) if getattr(meta, "shape", None) is not None else None
    dtype = str(meta.dtype).removeprefix("torch.") if getattr(meta, "dtype", None) is not None else None
    return shape, dtype


def _fx_source_expression(node: Any) -> str:
    if node.op == "call_function" and node.target is operator.matmul and len(node.args) >= 2:
        return f"{node.args[0]} @ {node.args[1]}"
    func_name = _fx_node_name(node)
    args = ", ".join(str(arg) for arg in node.args)
    return f"{func_name}({args})"


def _fx_node_attributes(
    node: Any,
    func_name: str,
    classification: _CallClassification,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    movement_kind = _movement_kind_for_name(leaf_name)
    if movement_kind is not None:
        attributes["movement_kind"] = movement_kind

    if classification.op_family in {
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


def _annotate_attention_graph(graph: BoundGraph) -> BoundGraph:
    """Promote visible dense attention chains into attention-family graph nodes."""
    nodes = list(graph.nodes)
    warnings = list(graph.warnings)
    for qk_index, qk_node in enumerate(nodes):
        qk_dims = _attention_qk_dims(graph, qk_node)
        if qk_dims is None:
            continue

        current_node = qk_node
        qk_attrs: dict[str, object] = {
            **qk_node.attributes,
            **qk_dims,
            "subrole": "qk_scores",
            "axis_source": "tensor_shapes",
            "mask_semantics": "not_applicable",
        }
        nodes[qk_index] = replace(
            qk_node,
            op_family=OpFamily.ATTENTION,
            attributes=qk_attrs,
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention QK score matmul",
        )

        mask_node = _next_consumer_node(nodes, current_node, graph, families={OpFamily.ELEMENTWISE})
        if mask_node is not None:
            mask_index = nodes.index(mask_node)
            mask_semantics = _attention_mask_semantics(graph, mask_node, qk_node)
            mask_attrs = {
                **mask_node.attributes,
                **qk_dims,
                "subrole": "scale_or_mask",
                "axis_source": "tensor_shapes",
                "mask_semantics": mask_semantics,
            }
            confidence = (
                EstimateConfidence.INEXACT
                if mask_semantics == "partial"
                else mask_node.confidence
            )
            nodes[mask_index] = replace(
                mask_node,
                op_family=OpFamily.ATTENTION,
                attributes=mask_attrs,
                confidence=confidence,
                rationale="recognized attention scale or mask application",
            )
            current_node = nodes[mask_index]
            if mask_semantics == "partial":
                warnings.append("inexact_operator:attention_mask")

        softmax_node = _next_consumer_node(
            nodes,
            current_node,
            graph,
            families={OpFamily.SOFTMAX},
        )
        if softmax_node is None:
            continue
        softmax_index = nodes.index(softmax_node)
        softmax_axis = softmax_node.attributes.get("dim", softmax_node.attributes.get("axis"))
        softmax_attrs = {
            **softmax_node.attributes,
            **qk_dims,
            "subrole": "softmax",
            "axis": softmax_axis,
            "axis_source": softmax_node.attributes.get("axis_source", "missing"),
        }
        nodes[softmax_index] = replace(
            softmax_node,
            op_family=OpFamily.ATTENTION,
            attributes=softmax_attrs,
            confidence=(
                EstimateConfidence.SUPPORTED
                if softmax_axis is not None
                else EstimateConfidence.INEXACT
            ),
            rationale="recognized attention softmax over score axis",
        )

        pv_node = _next_consumer_node(
            nodes,
            nodes[softmax_index],
            graph,
            families={OpFamily.GEMM},
        )
        pv_dims = _attention_pv_dims(graph, pv_node, qk_dims) if pv_node else None
        if pv_node is None or pv_dims is None:
            continue
        pv_index = nodes.index(pv_node)
        nodes[pv_index] = replace(
            pv_node,
            op_family=OpFamily.ATTENTION,
            attributes={
                **pv_node.attributes,
                **pv_dims,
                "subrole": "pv_aggregation",
                "axis_source": "tensor_shapes",
            },
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention PV aggregation matmul",
        )

        output_node = _next_consumer_node(
            nodes,
            nodes[pv_index],
            graph,
            families={OpFamily.GEMM, OpFamily.LINEAR_PROJECTION},
        )
        if output_node is None:
            continue
        output_index = nodes.index(output_node)
        nodes[output_index] = replace(
            output_node,
            op_family=OpFamily.ATTENTION,
            attributes={
                **output_node.attributes,
                **pv_dims,
                "subrole": "output_projection",
                "axis_source": "tensor_shapes",
            },
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention output projection",
        )

    dynamic_attention = _dynamic_attention_evidence(graph, nodes)
    if dynamic_attention is not None:
        nodes.append(dynamic_attention)
        warnings.append("unsupported_operator:dynamic_attention_axes")

    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))


def _attention_qk_dims(graph: BoundGraph, node: BoundGraphNode) -> dict[str, int] | None:
    if node.op_family != OpFamily.GEMM or len(node.input_tensor_ids) < 2:
        return None
    lhs = graph.tensors.get(node.input_tensor_ids[0])
    rhs = graph.tensors.get(node.input_tensor_ids[1])
    out = graph.tensors.get(node.output_tensor_ids[0]) if node.output_tensor_ids else None
    if lhs is None or rhs is None or out is None:
        return None
    if lhs.shape is None or rhs.shape is None or out.shape is None:
        return None
    if len(lhs.shape) != 4 or len(rhs.shape) != 4 or len(out.shape) != 4:
        return None
    batch, heads, sequence_q, head_dim = lhs.shape
    rhs_batch, rhs_heads, rhs_head_dim, sequence_k = rhs.shape
    if (batch, heads, sequence_q, sequence_k) != out.shape:
        return None
    if (batch, heads, head_dim) != (rhs_batch, rhs_heads, rhs_head_dim):
        return None
    return {
        "batch": int(batch),
        "heads": int(heads),
        "sequence_q": int(sequence_q),
        "sequence_k": int(sequence_k),
        "head_dim": int(head_dim),
    }


def _attention_pv_dims(
    graph: BoundGraph,
    node: BoundGraphNode | None,
    qk_dims: dict[str, int],
) -> dict[str, int] | None:
    if node is None or node.op_family != OpFamily.GEMM or len(node.input_tensor_ids) < 2:
        return None
    rhs = graph.tensors.get(node.input_tensor_ids[1])
    out = graph.tensors.get(node.output_tensor_ids[0]) if node.output_tensor_ids else None
    if rhs is None or out is None or rhs.shape is None or out.shape is None:
        return None
    expected_rhs = (
        qk_dims["batch"],
        qk_dims["heads"],
        qk_dims["sequence_k"],
        qk_dims["head_dim"],
    )
    expected_out = (
        qk_dims["batch"],
        qk_dims["heads"],
        qk_dims["sequence_q"],
        qk_dims["head_dim"],
    )
    if rhs.shape != expected_rhs or out.shape != expected_out:
        return None
    return dict(qk_dims)


def _attention_mask_semantics(
    graph: BoundGraph,
    node: BoundGraphNode,
    qk_node: BoundGraphNode,
) -> str:
    qk_outputs = set(qk_node.output_tensor_ids)
    for tensor_id in node.input_tensor_ids:
        if tensor_id in qk_outputs:
            continue
        tensor = graph.tensors.get(tensor_id)
        if tensor is not None and "mask" in tensor.name.lower():
            return "partial"
    return "scale"


def _next_consumer_node(
    nodes: list[BoundGraphNode],
    producer: BoundGraphNode,
    graph: BoundGraph,
    *,
    families: set[OpFamily],
) -> BoundGraphNode | None:
    produced = set(producer.output_tensor_ids)
    consumers = [
        node
        for node in nodes
        if node.node_id != producer.node_id
        and node.op_family in families
        and produced.intersection(node.input_tensor_ids)
    ]
    if consumers:
        return min(consumers, key=lambda item: item.node_id)

    producer_tensors = {
        tensor_id
        for tensor_id, tensor in graph.tensors.items()
        if tensor.producer_node_id == producer.node_id
    }
    for node in nodes:
        if node.node_id != producer.node_id and node.op_family in families:
            if producer_tensors.intersection(node.input_tensor_ids):
                return node
    return None


def _dynamic_attention_evidence(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
) -> BoundGraphNode | None:
    has_dynamic = any(node.op_family == OpFamily.UNSUPPORTED for node in nodes)
    names = {tensor.name for tensor in graph.tensors.values()}
    has_qkv = {"q", "k", "v"} <= names
    if not has_dynamic or not has_qkv:
        return None
    if any(node.op_family == OpFamily.ATTENTION for node in nodes):
        return None
    return BoundGraphNode(
        node_id=f"op_{len(nodes) + 1}",
        op_family=OpFamily.ATTENTION,
        op_name="dynamic_attention_axes",
        source_expression="dynamic attention axes",
        input_tensor_ids=tuple(
            tensor_id
            for tensor_id, tensor in sorted(graph.tensors.items())
            if tensor.name in {"q", "k", "v"}
        ),
        output_tensor_ids=(),
        attributes={
            "subrole": "dynamic_attention_axes",
            "axis_source": "missing",
            "dynamic_axes": True,
        },
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale="unsupported dynamic attention axes prevent static sequence modeling",
    )


class _AstBoundGraphExtractor:
    def __init__(
        self,
        definition: Definition,
        initial_tensors: dict[str, BoundTensor],
        output_names: tuple[str, ...],
        output_shapes: dict[str, tuple[int, ...] | None],
    ) -> None:
        self.definition = definition
        self.tensors = dict(initial_tensors)
        self.env = {name: f"input:{name}" for name in definition.inputs}
        self.output_names = output_names
        self.output_shapes = output_shapes
        self.nodes: list[BoundGraphNode] = []
        self.edges: list[BoundEdge] = []
        self.warnings: list[str] = []

    def extract(
        self,
        tree: ast.Module,
    ) -> tuple[
        tuple[BoundGraphNode, ...],
        dict[str, BoundTensor],
        tuple[BoundEdge, ...],
        tuple[str, ...],
    ]:
        run_func = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
            None,
        )
        if run_func is None:
            raise ValueError("Reference must define a top-level function named 'run'")

        for statement in run_func.body:
            self._process_statement(statement)

        return (
            tuple(self.nodes),
            self.tensors,
            tuple(self.edges),
            tuple(dict.fromkeys(self.warnings)),
        )

    def _process_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            value_tensor_ids = self._process_expr(statement.value)
            for target in statement.targets:
                self._bind_target(target, value_tensor_ids)
        elif isinstance(statement, ast.AnnAssign):
            value_tensor_ids = self._process_expr(statement.value) if statement.value else ()
            self._bind_target(statement.target, value_tensor_ids)
        elif isinstance(statement, ast.Return):
            self._bind_outputs(self._process_expr(statement.value) if statement.value else ())
        elif isinstance(statement, ast.Expr):
            self._process_expr(statement.value)
        elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Raise)):
            expression = ast.unparse(statement)
            op_name = statement.__class__.__name__.lower()
            self._append_node(
                op_family=OpFamily.UNSUPPORTED,
                op_name=op_name,
                source_expression=expression,
                input_tensor_ids=(),
                confidence=EstimateConfidence.UNSUPPORTED,
                rationale="unsupported dynamic control flow preserved as graph evidence",
                attributes={},
            )
            self.warnings.append(f"unsupported_operator:{op_name}")

    def _process_expr(self, expression: ast.AST | None) -> tuple[str, ...]:
        if expression is None:
            return ()
        if isinstance(expression, ast.Name):
            tensor_id = self.env.get(expression.id)
            return (tensor_id,) if tensor_id is not None else ()
        if isinstance(expression, ast.Tuple):
            result: list[str] = []
            for element in expression.elts:
                result.extend(self._process_expr(element))
            return tuple(result)
        if isinstance(expression, ast.BinOp):
            return self._process_binop(expression)
        if isinstance(expression, ast.Call):
            return self._process_call(expression)
        if isinstance(expression, ast.UnaryOp):
            return self._process_expr(expression.operand)
        if isinstance(expression, ast.Subscript):
            return self._process_expr(expression.value)
        if isinstance(expression, ast.Attribute):
            return self._process_expr(expression.value)
        return ()

    def _process_binop(self, node: ast.BinOp) -> tuple[str, ...]:
        input_tensor_ids = self._process_expr(node.left) + self._process_expr(node.right)
        if isinstance(node.op, ast.MatMult):
            op_family = OpFamily.GEMM
            confidence = EstimateConfidence.SUPPORTED
            rationale = "recognized @ matrix multiply"
            op_name = "@"
        else:
            op_family = OpFamily.ELEMENTWISE
            confidence = EstimateConfidence.INEXACT
            rationale = "recognized elementwise binary operation"
            op_name = node.op.__class__.__name__.lower()
        return self._append_node(
            op_family=op_family,
            op_name=op_name,
            source_expression=ast.unparse(node),
            input_tensor_ids=input_tensor_ids,
            confidence=confidence,
            rationale=rationale,
            attributes={},
        )

    def _process_call(self, node: ast.Call) -> tuple[str, ...]:
        input_tensor_ids: list[str] = []
        if isinstance(node.func, ast.Attribute):
            input_tensor_ids.extend(self._process_expr(node.func.value))
        for arg in node.args:
            input_tensor_ids.extend(self._process_expr(arg))
        for keyword in node.keywords:
            input_tensor_ids.extend(self._process_expr(keyword.value))

        func_name = _call_name(node.func)
        classification = _classify_call(func_name)
        if classification is None:
            self.warnings.append(f"unsupported_operator:{func_name or '<unknown>'}")
            return self._append_node(
                op_family=OpFamily.UNSUPPORTED,
                op_name=func_name or "<unknown>",
                source_expression=ast.unparse(node),
                input_tensor_ids=tuple(input_tensor_ids),
                confidence=EstimateConfidence.UNSUPPORTED,
                rationale="unsupported call preserved as graph evidence",
                attributes={},
            )

        return self._append_node(
            op_family=classification.op_family,
            op_name=func_name,
            source_expression=ast.unparse(node),
            input_tensor_ids=tuple(input_tensor_ids),
            confidence=classification.confidence,
            rationale=classification.rationale,
            attributes=_ast_call_attributes(node, func_name, classification),
        )

    def _append_node(
        self,
        op_family: OpFamily,
        op_name: str,
        source_expression: str,
        input_tensor_ids: tuple[str, ...],
        confidence: EstimateConfidence,
        rationale: str,
        attributes: dict[str, object] | None,
    ) -> tuple[str, ...]:
        node_id = f"op_{len(self.nodes) + 1}"
        output_tensor_id = f"tmp:{node_id}:0"
        node_attributes = attributes or {}
        output_tensor = BoundTensor(
            tensor_id=output_tensor_id,
            name=output_tensor_id,
            role=BoundTensorRole.INTERMEDIATE,
            shape=self._default_intermediate_shape(input_tensor_ids),
            dtype=str(
                node_attributes.get("target_dtype")
                or self._default_intermediate_dtype(input_tensor_ids)
            ),
            producer_node_id=node_id,
            source=source_expression,
        )
        self.tensors[output_tensor_id] = output_tensor
        node = BoundGraphNode(
            node_id=node_id,
            op_family=op_family,
            op_name=op_name,
            source_expression=source_expression,
            input_tensor_ids=tuple(dict.fromkeys(input_tensor_ids)),
            output_tensor_ids=(output_tensor_id,),
            attributes=node_attributes,
            confidence=confidence,
            rationale=rationale,
            conversion_status="not_converted",
        )
        self.nodes.append(node)
        for input_tensor_id in node.input_tensor_ids:
            self.edges.append(
                BoundEdge(
                    edge_id=f"edge_{len(self.edges) + 1}",
                    source_tensor_id=input_tensor_id,
                    target_node_id=node_id,
                    role="input",
                )
            )
        return (output_tensor_id,)

    def _bind_target(self, target: ast.AST, tensor_ids: tuple[str, ...]) -> None:
        if isinstance(target, ast.Name) and tensor_ids:
            self.env[target.id] = tensor_ids[0]
        elif isinstance(target, ast.Tuple):
            for element, tensor_id in zip(target.elts, tensor_ids):
                if isinstance(element, ast.Name):
                    self.env[element.id] = tensor_id

    def _bind_outputs(self, tensor_ids: tuple[str, ...]) -> None:
        for index, output_name in enumerate(self.output_names):
            if index >= len(tensor_ids):
                break
            tensor_id = f"output:{output_name}"
            source_tensor_id = tensor_ids[index]
            source_tensor = self.tensors.get(source_tensor_id)
            producer_node_id = source_tensor.producer_node_id if source_tensor else None
            self.tensors[tensor_id] = replace(
                self.tensors[tensor_id],
                producer_node_id=producer_node_id,
                source=source_tensor_id,
            )

    def _default_intermediate_shape(self, input_tensor_ids: tuple[str, ...]) -> tuple[int, ...] | None:
        for tensor_id in input_tensor_ids:
            tensor = self.tensors.get(tensor_id)
            if tensor and tensor.shape is not None:
                return tensor.shape
        for shape in self.output_shapes.values():
            if shape is not None:
                return shape
        return None

    def _default_intermediate_dtype(self, input_tensor_ids: tuple[str, ...]) -> str:
        for tensor_id in input_tensor_ids:
            tensor = self.tensors.get(tensor_id)
            if tensor:
                return tensor.dtype
        if self.definition.outputs:
            return next(iter(self.definition.outputs.values())).dtype.value
        return "unknown"


def _classify_call(func_name: str) -> _CallClassification | None:
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    for names, classification in _CALL_CLASSIFIERS:
        if leaf_name in names or func_name in names:
            return classification
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


_MISSING = object()


def _movement_kind_for_name(leaf_name: str) -> str | None:
    if leaf_name in _BROADCAST_VIEW_NAMES:
        return "broadcast_view"
    if leaf_name in _MATERIALIZED_MOVEMENT_NAMES:
        return "materialized"
    if leaf_name in _LOGICAL_VIEW_NAMES:
        return "logical_view"
    return None


def _axis_from_values(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    for name in ("dim", "axis"):
        if name in kwargs:
            return _literal_value(kwargs[name])
    for arg in args:
        value = _literal_value(arg)
        if value is not _MISSING:
            return value
    return _MISSING


def _target_dtype_from_values(
    leaf_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    if leaf_name in _DTYPE_METHOD_TARGETS:
        return _DTYPE_METHOD_TARGETS[leaf_name]
    if leaf_name not in {"to", "type"}:
        return None
    if "dtype" in kwargs:
        dtype = _normalize_dtype_value(kwargs["dtype"])
        if dtype is not None:
            return dtype
    for arg in args:
        dtype = _normalize_dtype_value(arg)
        if dtype is not None:
            return dtype
    return None


def _ast_call_attributes(
    node: ast.Call,
    func_name: str,
    classification: _CallClassification,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    movement_kind = _movement_kind_for_name(leaf_name)
    if movement_kind is not None:
        attributes["movement_kind"] = movement_kind

    keyword_values = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    if classification.op_family in {
        OpFamily.REDUCTION,
        OpFamily.NORMALIZATION,
        OpFamily.SOFTMAX,
    }:
        positional_args = tuple(node.args if isinstance(node.func, ast.Attribute) else node.args[1:])
        axis = _axis_from_values(positional_args, keyword_values)
        if axis is not _MISSING:
            attributes["dim"] = axis
            attributes["axis_source"] = "attribute"

    target_dtype = _target_dtype_from_values(
        leaf_name,
        tuple(node.args if isinstance(node.func, ast.Attribute) else node.args[1:]),
        keyword_values,
    )
    if target_dtype is not None:
        attributes["target_dtype"] = target_dtype
    return attributes


def _literal_value(value: Any) -> object:
    if isinstance(value, ast.Constant) and isinstance(value.value, (int, type(None))):
        return value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        operand = _literal_value(value.operand)
        if isinstance(operand, int):
            return -operand
    if isinstance(value, ast.Tuple | ast.List):
        elements = tuple(_literal_value(element) for element in value.elts)
        if all(element is None or isinstance(element, int) for element in elements):
            return elements
        return _MISSING
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, tuple | list) and all(item is None or isinstance(item, int) for item in value):
        return tuple(value)
    return _MISSING


def _normalize_dtype_value(value: Any) -> str | None:
    if isinstance(value, ast.Attribute):
        return _normalize_dtype_name(value.attr)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return _normalize_dtype_name(value.value)
    return _normalize_dtype_name(str(value).removeprefix("torch."))


def _normalize_dtype_name(raw: str) -> str | None:
    name = raw.removeprefix("torch.").lower()
    aliases = {
        "float": DType.FLOAT32.value,
        "float32": DType.FLOAT32.value,
        "half": DType.FLOAT16.value,
        "float16": DType.FLOAT16.value,
        "bfloat16": DType.BFLOAT16.value,
        "double": DType.FLOAT64.value,
        "float64": DType.FLOAT64.value,
        "bool": DType.BOOL.value,
        "int": DType.INT32.value,
        "int32": DType.INT32.value,
        "long": DType.INT64.value,
        "int64": DType.INT64.value,
        "int16": DType.INT16.value,
        "int8": DType.INT8.value,
    }
    return aliases.get(name)
