# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AST fallback extraction for AMD bound graphs."""

from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.scoring.amd_bound_estimate.classification import (
    classify_call as _classify_call,
)
from sol_execbench.core.scoring.amd_bound_graph.common import (
    _ast_call_attributes,
    _classification_family,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundEdge,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence


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
        self.constants: dict[str, object] = {}
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
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run"
            ),
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
            constant = self._static_value(statement.value)
            if constant is not _STATIC_MISSING:
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        self.constants[target.id] = constant
                return
            value_tensor_ids = self._process_expr(statement.value)
            for target in statement.targets:
                self._bind_target(target, value_tensor_ids)
        elif isinstance(statement, ast.AnnAssign):
            value_tensor_ids = (
                self._process_expr(statement.value) if statement.value else ()
            )
            self._bind_target(statement.target, value_tensor_ids)
        elif isinstance(statement, ast.Return):
            self._bind_outputs(
                self._process_expr(statement.value) if statement.value else ()
            )
        elif isinstance(statement, ast.Expr):
            self._process_expr(statement.value)
        elif isinstance(statement, ast.If):
            condition = self._static_value(statement.test)
            if isinstance(condition, bool):
                for nested in statement.body if condition else statement.orelse:
                    self._process_statement(nested)
            else:
                self._append_unsupported_control_flow(statement)
        elif isinstance(statement, ast.For):
            values = self._static_range(statement.iter)
            if values is None or not isinstance(statement.target, ast.Name):
                self._append_unsupported_control_flow(statement)
            else:
                previous = self.constants.get(statement.target.id, _STATIC_MISSING)
                for value in values:
                    self.constants[statement.target.id] = value
                    for nested in statement.body:
                        self._process_statement(nested)
                if previous is _STATIC_MISSING:
                    self.constants.pop(statement.target.id, None)
                else:
                    self.constants[statement.target.id] = previous
                for nested in statement.orelse:
                    self._process_statement(nested)
        elif isinstance(statement, (ast.While, ast.Raise)):
            self._append_unsupported_control_flow(statement)

    def _append_unsupported_control_flow(self, statement: ast.stmt) -> None:
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
        if isinstance(expression, (ast.Tuple, ast.List)):
            result: list[str] = []
            for element in expression.elts:
                result.extend(self._process_expr(element))
            return tuple(result)
        if isinstance(expression, ast.BinOp):
            return self._process_binop(expression)
        if isinstance(expression, ast.Compare):
            input_tensor_ids: tuple[str, ...] = self._process_expr(expression.left)
            for comparator in expression.comparators:
                input_tensor_ids += self._process_expr(comparator)
            return self._append_node(
                op_family=OpFamily.ELEMENTWISE,
                op_name=expression.ops[0].__class__.__name__.lower(),
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized elementwise comparison",
                attributes={},
            )
        if isinstance(expression, ast.Call):
            return self._process_call(expression)
        if isinstance(expression, ast.UnaryOp):
            input_tensor_ids = self._process_expr(expression.operand)
            if not input_tensor_ids:
                return ()
            return self._append_node(
                op_family=OpFamily.ELEMENTWISE,
                op_name=expression.op.__class__.__name__.lower(),
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized elementwise unary operation",
                attributes={},
            )
        if isinstance(expression, ast.Subscript):
            if isinstance(
                expression.value, ast.Attribute
            ) and expression.value.attr in {
                "shape",
                "stride",
            }:
                return ()
            input_tensor_ids = self._process_expr(expression.value)
            if not input_tensor_ids:
                return ()
            return self._append_node(
                op_family=OpFamily.DATA_MOVEMENT,
                op_name="getitem",
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized materialized tensor indexing operation",
                attributes={"movement_kind": "materialized"},
            )
        if isinstance(expression, ast.Attribute):
            return self._process_expr(expression.value)
        return ()

    def _process_binop(self, node: ast.BinOp) -> tuple[str, ...]:
        input_tensor_ids = self._process_expr(node.left) + self._process_expr(
            node.right
        )
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
        func_name = _call_name(node.func)
        if func_name.rsplit(".", maxsplit=1)[-1] in {
            "dim",
            "ndimension",
            "numel",
            "size",
            "stride",
            "is_contiguous",
            "len",
        }:
            return ()
        input_tensor_ids: list[str] = []
        if isinstance(node.func, ast.Attribute):
            input_tensor_ids.extend(self._process_expr(node.func.value))
        elif isinstance(node.func, ast.Name) and node.func.id in self.env:
            input_tensor_ids.extend(self._process_expr(node.func))
        for arg in node.args:
            input_tensor_ids.extend(self._process_expr(arg))
        for keyword in node.keywords:
            input_tensor_ids.extend(self._process_expr(keyword.value))

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
            op_family=_classification_family(classification),
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
        attributes: dict[str, Any] | None,
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

    def _static_range(self, expression: ast.AST) -> tuple[int, ...] | None:
        if (
            not isinstance(expression, ast.Call)
            or _call_name(expression.func) != "range"
        ):
            return None
        values = [self._static_value(arg) for arg in expression.args]
        if not values or any(not isinstance(value, int) for value in values):
            return None
        int_values = [value for value in values if isinstance(value, int)]
        try:
            if len(int_values) == 1:
                result = tuple(range(int_values[0]))
            elif len(int_values) == 2:
                result = tuple(range(int_values[0], int_values[1]))
            elif len(int_values) == 3:
                result = tuple(range(int_values[0], int_values[1], int_values[2]))
            else:
                return None
        except (TypeError, ValueError):
            return None
        return result if len(result) <= 1024 else None

    def _static_value(self, expression: ast.AST) -> object:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            return self.constants.get(expression.id, _STATIC_MISSING)
        if isinstance(expression, ast.UnaryOp):
            operand = self._static_value(expression.operand)
            if operand is _STATIC_MISSING:
                return _STATIC_MISSING
            if isinstance(expression.op, ast.Not):
                return not bool(operand)
            if isinstance(expression.op, ast.USub) and isinstance(operand, int | float):
                return -operand
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            left = self._static_value(expression.left)
            right = self._static_value(expression.comparators[0])
            if left is _STATIC_MISSING or right is _STATIC_MISSING:
                return _STATIC_MISSING
            op = expression.ops[0]
            left_value: Any = left
            right_value: Any = right
            try:
                if isinstance(op, ast.Eq):
                    return left_value == right_value
                if isinstance(op, ast.NotEq):
                    return left_value != right_value
                if isinstance(op, ast.Lt):
                    return left_value < right_value
                if isinstance(op, ast.LtE):
                    return left_value <= right_value
                if isinstance(op, ast.Gt):
                    return left_value > right_value
                if isinstance(op, ast.GtE):
                    return left_value >= right_value
            except TypeError:
                return _STATIC_MISSING
        return _STATIC_MISSING

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

    def _default_intermediate_shape(
        self, input_tensor_ids: tuple[str, ...]
    ) -> tuple[int, ...] | None:
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


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


_STATIC_MISSING = object()
