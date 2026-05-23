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

    return BoundGraph(
        definition=definition.name,
        workload_uuid=workload.uuid,
        nodes=nodes,
        tensors=extracted_tensors,
        edges=edges,
        warnings=tuple(dict.fromkeys(warnings)),
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
            attributes={"trace_source": "torch.fx"},
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

    return BoundGraph(
        definition=definition.name,
        workload_uuid=workload.uuid,
        nodes=tuple(nodes),
        tensors=tensors,
        edges=tuple(edges),
        warnings=tuple(dict.fromkeys(warnings)),
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
            )

        return self._append_node(
            op_family=classification.op_family,
            op_name=func_name,
            source_expression=ast.unparse(node),
            input_tensor_ids=tuple(input_tensor_ids),
            confidence=classification.confidence,
            rationale=classification.rationale,
        )

    def _append_node(
        self,
        op_family: OpFamily,
        op_name: str,
        source_expression: str,
        input_tensor_ids: tuple[str, ...],
        confidence: EstimateConfidence,
        rationale: str,
    ) -> tuple[str, ...]:
        node_id = f"op_{len(self.nodes) + 1}"
        output_tensor_id = f"tmp:{node_id}:0"
        output_tensor = BoundTensor(
            tensor_id=output_tensor_id,
            name=output_tensor_id,
            role=BoundTensorRole.INTERMEDIATE,
            shape=self._default_intermediate_shape(input_tensor_ids),
            dtype=self._default_intermediate_dtype(input_tensor_ids),
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
            attributes={},
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
