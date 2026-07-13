# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AST fallback extraction for AMD bound graphs."""

from __future__ import annotations

import ast
import math
import operator
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
        self.sequences: dict[str, tuple[str, ...]] = {}
        self.output_names = output_names
        self.output_shapes = output_shapes
        self.nodes: list[BoundGraphNode] = []
        self.edges: list[BoundEdge] = []
        self.warnings: list[str] = []
        self.functions: dict[str, ast.FunctionDef] = {}
        self.class_names: set[str] = set()
        self._helper_returns: list[list[str]] = []
        self._call_stack: list[str] = []
        self._dynamic_control_depth = 0

    def extract(
        self,
        tree: ast.Module,
    ) -> tuple[
        tuple[BoundGraphNode, ...],
        dict[str, BoundTensor],
        tuple[BoundEdge, ...],
        tuple[str, ...],
    ]:
        self.functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
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
            if isinstance(statement.value, ast.List) and not statement.value.elts:
                value_tensor_ids = self._process_expr(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        self.sequences[target.id] = value_tensor_ids
                        self.env.pop(target.id, None)
                        self.constants.pop(target.id, None)
                return
            constant = self._static_value(statement.value)
            if constant is not _STATIC_MISSING:
                for target in statement.targets:
                    self._bind_constant(target, constant)
                return
            if isinstance(statement.value, (ast.List, ast.Tuple)):
                value_tensor_ids = self._process_expr(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        self.sequences[target.id] = value_tensor_ids
                        self.env.pop(target.id, None)
                        self.constants.pop(target.id, None)
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
            tensor_ids = self._process_expr(statement.value) if statement.value else ()
            if self._helper_returns:
                self._helper_returns[-1].extend(tensor_ids)
            else:
                self._bind_outputs(tensor_ids)
        elif isinstance(statement, ast.Expr):
            self._process_expr(statement.value)
        elif isinstance(statement, ast.AugAssign):
            value_tensor_ids = self._process_binop(
                ast.BinOp(left=statement.target, op=statement.op, right=statement.value)
            )
            self._bind_target(statement.target, value_tensor_ids)
        elif isinstance(statement, ast.If):
            condition = self._static_value(statement.test)
            if isinstance(condition, bool):
                for nested in statement.body if condition else statement.orelse:
                    self._process_statement(nested)
            else:
                self._append_unsupported_control_flow(statement)
                self._process_dynamic_statements((*statement.body, *statement.orelse))
        elif isinstance(statement, ast.For):
            values = self._static_range(statement.iter)
            if values is None or not isinstance(statement.target, ast.Name):
                self._append_unsupported_control_flow(statement)
                self._process_dynamic_statements(tuple(statement.body))
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
        elif isinstance(statement, ast.While):
            self._append_unsupported_control_flow(statement)
            self._process_dynamic_statements((*statement.body, *statement.orelse))
        elif isinstance(statement, ast.Try):
            self._append_unsupported_control_flow(statement)
            nested_statements: list[ast.stmt] = [
                *statement.body,
                *statement.orelse,
                *statement.finalbody,
            ]
            for handler in statement.handlers:
                nested_statements.extend(handler.body)
            self._process_dynamic_statements(tuple(nested_statements))
        elif isinstance(statement, ast.With):
            for item in statement.items:
                self._process_expr(item.context_expr)
            for nested in statement.body:
                self._process_statement(nested)
        elif isinstance(statement, ast.Raise):
            self._append_unsupported_statement(statement)
        elif isinstance(statement, ast.FunctionDef):
            self.functions[statement.name] = statement

    def _process_dynamic_statements(self, statements: tuple[ast.stmt, ...]) -> None:
        self._dynamic_control_depth += 1
        try:
            for statement in statements:
                self._process_statement(statement)
        finally:
            self._dynamic_control_depth -= 1

    def _append_unsupported_control_flow(self, statement: ast.stmt) -> None:
        op_name = statement.__class__.__name__.lower()
        self.warnings.append(f"inexact_control_flow:{op_name}")

    def _append_unsupported_statement(self, statement: ast.stmt) -> None:
        op_name = statement.__class__.__name__.lower()
        self._append_node(
            op_family=OpFamily.UNSUPPORTED,
            op_name=op_name,
            source_expression=ast.unparse(statement),
            input_tensor_ids=(),
            confidence=EstimateConfidence.UNSUPPORTED,
            rationale="unsupported statement preserved as graph evidence",
            attributes={},
        )
        self.warnings.append(f"unsupported_operator:{op_name}")

    def _process_expr(self, expression: ast.AST | None) -> tuple[str, ...]:
        if expression is None:
            return ()
        if isinstance(expression, ast.Name):
            tensor_id = self.env.get(expression.id)
            if tensor_id is not None:
                return (tensor_id,)
            return self.sequences.get(expression.id, ())
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
        if isinstance(expression, ast.BoolOp):
            input_tensor_ids: tuple[str, ...] = ()
            for value in expression.values:
                input_tensor_ids += self._process_expr(value)
            if not input_tensor_ids:
                return ()
            return self._append_node(
                op_family=OpFamily.ELEMENTWISE,
                op_name=expression.op.__class__.__name__.lower(),
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized elementwise boolean operation",
                attributes={},
            )
        if isinstance(expression, ast.IfExp):
            input_tensor_ids = self._process_expr(expression.body) + self._process_expr(
                expression.orelse
            )
            if not input_tensor_ids:
                return ()
            return self._append_node(
                op_family=OpFamily.ELEMENTWISE,
                op_name="where",
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized conditional selection expression",
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
            dynamic_index = not self._is_static_index(expression.slice)
            return self._append_node(
                op_family=OpFamily.DATA_MOVEMENT,
                op_name="getitem",
                source_expression=ast.unparse(expression),
                input_tensor_ids=input_tensor_ids,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized materialized tensor indexing operation",
                attributes={
                    "movement_kind": "materialized",
                    **({"dynamic_index": True} if dynamic_index else {}),
                },
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
            confidence = (
                EstimateConfidence.SUPPORTED
                if self._has_exact_elementwise_shapes(input_tensor_ids)
                else EstimateConfidence.INEXACT
            )
            rationale = (
                "recognized exact-shape elementwise binary operation"
                if confidence == EstimateConfidence.SUPPORTED
                else "recognized elementwise binary operation"
            )
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

    def _is_static_index(self, expression: ast.AST) -> bool:
        if isinstance(expression, ast.Constant):
            return True
        if isinstance(expression, ast.Name):
            return expression.id in self.constants
        if isinstance(expression, (ast.Tuple, ast.List)):
            return all(self._is_static_index(element) for element in expression.elts)
        if isinstance(expression, ast.Slice):
            return all(
                value is None or self._is_static_index(value)
                for value in (expression.lower, expression.upper, expression.step)
            )
        if isinstance(expression, ast.UnaryOp):
            return self._is_static_index(expression.operand)
        return False

    def _has_exact_elementwise_shapes(self, input_tensor_ids: tuple[str, ...]) -> bool:
        if not input_tensor_ids:
            return False
        shapes = [self.tensors[tensor_id].shape for tensor_id in input_tensor_ids]
        return shapes[0] is not None and all(shape == shapes[0] for shape in shapes)

    def _process_call(self, node: ast.Call) -> tuple[str, ...]:
        func_name = _call_name(node.func)
        if isinstance(node.func, ast.Name) and func_name in self.functions:
            return self._process_helper_call(node, self.functions[func_name])
        if isinstance(node.func, ast.Name) and func_name in self.class_names:
            return ()
        if self._process_sequence_call(node, func_name):
            return ()
        if _is_host_metadata_call(node, func_name):
            return ()
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
            if not _is_dtype_metadata(arg):
                input_tensor_ids.extend(self._process_expr(arg))
        for keyword in node.keywords:
            if not _is_dtype_metadata(keyword.value):
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

        attributes = _ast_call_attributes(node, func_name, classification)
        if (
            func_name.rsplit(".", maxsplit=1)[-1] in {"to", "type"}
            and "target_dtype" not in attributes
        ):
            target_dtype = self._dtype_from_metadata(node)
            if target_dtype is not None:
                attributes["target_dtype"] = target_dtype
        if (
            func_name.rsplit(".", maxsplit=1)[-1] == "type_as"
            and "target_dtype" not in attributes
            and len(input_tensor_ids) >= 2
        ):
            target_tensor = self.tensors.get(input_tensor_ids[-1])
            if target_tensor is not None:
                attributes["target_dtype"] = target_tensor.dtype

        return self._append_node(
            op_family=_classification_family(classification),
            op_name=func_name,
            source_expression=ast.unparse(node),
            input_tensor_ids=tuple(input_tensor_ids),
            confidence=classification.confidence,
            rationale=classification.rationale,
            attributes=attributes,
            output_count=self._call_output_count(node, func_name, input_tensor_ids),
        )

    def _process_helper_call(
        self, node: ast.Call, function: ast.FunctionDef
    ) -> tuple[str, ...]:
        if function.name in self._call_stack:
            self.warnings.append(
                f"unsupported_operator:recursive_helper:{function.name}"
            )
            return ()

        saved_env = self.env
        saved_constants = self.constants
        saved_sequences = self.sequences
        saved_functions = self.functions
        self.env = dict(saved_env)
        self.constants = dict(saved_constants)
        self.sequences = dict(saved_sequences)
        self.functions = dict(saved_functions)
        positional = list(node.args)
        keywords = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
        }
        for index, parameter in enumerate(function.args.args):
            value = (
                positional[index]
                if index < len(positional)
                else keywords.get(parameter.arg)
            )
            if value is None:
                continue
            static_value = self._static_value(value)
            if static_value is not _STATIC_MISSING:
                self.constants[parameter.arg] = static_value
                continue
            tensor_ids = self._process_expr(value)
            if len(tensor_ids) == 1:
                self.env[parameter.arg] = tensor_ids[0]
            elif tensor_ids:
                self.sequences[parameter.arg] = tensor_ids

        self._call_stack.append(function.name)
        self._helper_returns.append([])
        try:
            for statement in function.body:
                self._process_statement(statement)
            result = tuple(dict.fromkeys(self._helper_returns[-1]))
        finally:
            self._helper_returns.pop()
            self._call_stack.pop()
            self.env = saved_env
            self.constants = saved_constants
            self.sequences = saved_sequences
            self.functions = saved_functions
        return result

    def _process_sequence_call(self, node: ast.Call, func_name: str) -> bool:
        leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
        if leaf_name not in {"append", "extend"} or not isinstance(
            node.func, ast.Attribute
        ):
            return False
        receiver = node.func.value
        values: list[str] = []
        for arg in node.args:
            values.extend(self._process_expr(arg))
        if isinstance(receiver, ast.Name):
            existing = self.sequences.get(receiver.id, ())
            self.sequences[receiver.id] = (*existing, *values)
        return True

    def _call_output_count(
        self,
        node: ast.Call,
        func_name: str,
        input_tensor_ids: list[str],
    ) -> int:
        leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
        if leaf_name in {"topk", "sort"}:
            return 2
        if leaf_name == "chunk":
            position = 0 if isinstance(node.func, ast.Attribute) else 1
            value = (
                self._static_value(node.args[position])
                if position < len(node.args)
                else _STATIC_MISSING
            )
            return value if isinstance(value, int) and 0 < value <= 1024 else 1
        if leaf_name in {"split", "tensor_split"}:
            position = 0 if isinstance(node.func, ast.Attribute) else 1
            value = (
                self._static_value(node.args[position])
                if position < len(node.args)
                else _STATIC_MISSING
            )
            if isinstance(value, tuple) and value:
                return min(len(value), 1024)
        if leaf_name == "unbind" and input_tensor_ids:
            tensor = self.tensors.get(input_tensor_ids[0])
            if tensor is not None and tensor.shape:
                raw_dim = (
                    node.args[0]
                    if isinstance(node.func, ast.Attribute) and node.args
                    else None
                )
                dim = self._static_value(raw_dim) if raw_dim is not None else 0
                if isinstance(dim, int):
                    normalized = dim if dim >= 0 else dim + len(tensor.shape)
                    if 0 <= normalized < len(tensor.shape):
                        return min(tensor.shape[normalized], 1024)
        return 1

    def _append_node(
        self,
        op_family: OpFamily,
        op_name: str,
        source_expression: str,
        input_tensor_ids: tuple[str, ...],
        confidence: EstimateConfidence,
        rationale: str,
        attributes: dict[str, Any] | None,
        output_count: int = 1,
    ) -> tuple[str, ...]:
        node_id = f"op_{len(self.nodes) + 1}"
        output_tensor_ids = tuple(
            f"tmp:{node_id}:{index}" for index in range(max(1, output_count))
        )
        node_attributes = attributes or {}
        if self._dynamic_control_depth:
            node_attributes = {
                **node_attributes,
                "dynamic_control_flow": True,
            }
            if confidence == EstimateConfidence.SUPPORTED:
                confidence = EstimateConfidence.INEXACT
                rationale = f"{rationale}; dynamic execution multiplicity is unresolved"
        output_shape = self._default_intermediate_shape(input_tensor_ids)
        if op_family == OpFamily.REDUCTION:
            output_shape = self._reduction_output_shape(
                input_tensor_ids, node_attributes, output_shape
            )
        for output_tensor_id in output_tensor_ids:
            output_tensor = BoundTensor(
                tensor_id=output_tensor_id,
                name=output_tensor_id,
                role=BoundTensorRole.INTERMEDIATE,
                shape=output_shape,
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
            output_tensor_ids=output_tensor_ids,
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
        return output_tensor_ids

    def _dtype_from_metadata(self, node: ast.Call) -> str | None:
        values = [*node.args, *(keyword.value for keyword in node.keywords)]
        for value in values:
            if not _is_dtype_metadata(value):
                continue
            assert isinstance(value, ast.Attribute)
            tensor_ids = self._process_expr(value.value)
            if tensor_ids and tensor_ids[0] in self.tensors:
                return self.tensors[tensor_ids[0]].dtype
        return None

    def _reduction_output_shape(
        self,
        input_tensor_ids: tuple[str, ...],
        attributes: dict[str, Any],
        fallback: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if not input_tensor_ids:
            return fallback
        input_tensor = self.tensors.get(input_tensor_ids[0])
        input_shape = input_tensor.shape if input_tensor is not None else None
        if input_shape is None:
            return fallback
        raw_axis = attributes.get("dim")
        if raw_axis is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(raw_axis, int):
            axes = (raw_axis,)
        elif isinstance(raw_axis, tuple) and all(
            isinstance(axis, int) for axis in raw_axis
        ):
            axes = raw_axis
        else:
            return fallback
        normalized = {axis + len(input_shape) if axis < 0 else axis for axis in axes}
        if any(axis < 0 or axis >= len(input_shape) for axis in normalized):
            return fallback
        if attributes.get("keepdim") is True:
            return tuple(
                1 if index in normalized else dimension
                for index, dimension in enumerate(input_shape)
            )
        return tuple(
            dimension
            for index, dimension in enumerate(input_shape)
            if index not in normalized
        )

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
        # Large data-dependent loops are represented by one inexact body instance.
        # Fully unrolling expert/token loops makes the fallback graph enormous while
        # adding no new operator evidence.
        return result if len(result) <= 32 else None

    def _static_value(self, expression: ast.AST) -> object:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            return self.constants.get(expression.id, _STATIC_MISSING)
        if isinstance(expression, (ast.Tuple, ast.List)):
            values = tuple(self._static_value(element) for element in expression.elts)
            if all(value is not _STATIC_MISSING for value in values):
                return values
            return _STATIC_MISSING
        if isinstance(expression, ast.Attribute):
            if expression.attr == "shape":
                tensor_ids = self._process_expr(expression.value)
                if tensor_ids:
                    tensor = self.tensors.get(tensor_ids[0])
                    if tensor is not None and tensor.shape is not None:
                        return tensor.shape
            return _STATIC_MISSING
        if isinstance(expression, ast.Subscript):
            value = self._static_value(expression.value)
            index = self._static_value(expression.slice)
            if value is _STATIC_MISSING or index is _STATIC_MISSING:
                return _STATIC_MISSING
            if not isinstance(value, (tuple, list, str)) or not isinstance(index, int):
                return _STATIC_MISSING
            try:
                return value[index]
            except (IndexError, KeyError, TypeError):
                return _STATIC_MISSING
        if isinstance(expression, ast.BinOp):
            left = self._static_value(expression.left)
            right = self._static_value(expression.right)
            if left is _STATIC_MISSING or right is _STATIC_MISSING:
                return _STATIC_MISSING
            operation = _STATIC_BINARY_OPERATORS.get(type(expression.op))
            if operation is None:
                return _STATIC_MISSING
            try:
                return operation(left, right)
            except (ArithmeticError, TypeError, ValueError):
                return _STATIC_MISSING
        if isinstance(expression, ast.UnaryOp):
            operand = self._static_value(expression.operand)
            if operand is _STATIC_MISSING:
                return _STATIC_MISSING
            if isinstance(expression.op, ast.Not):
                return not bool(operand)
            if isinstance(expression.op, ast.USub) and isinstance(operand, int | float):
                return -operand
            if isinstance(expression.op, ast.UAdd) and isinstance(operand, int | float):
                return operand
        if isinstance(expression, ast.BoolOp):
            values = [self._static_value(value) for value in expression.values]
            if any(value is _STATIC_MISSING for value in values):
                return _STATIC_MISSING
            if isinstance(expression.op, ast.And):
                return all(bool(value) for value in values)
            if isinstance(expression.op, ast.Or):
                return any(bool(value) for value in values)
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
        if isinstance(expression, ast.Call):
            func_name = _call_name(expression.func)
            leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
            if isinstance(expression.func, ast.Attribute) and leaf_name in {
                "dim",
                "ndimension",
                "numel",
                "size",
            }:
                tensor_ids = self._process_expr(expression.func.value)
                tensor = self.tensors.get(tensor_ids[0]) if tensor_ids else None
                shape = tensor.shape if tensor is not None else None
                if shape is not None:
                    if leaf_name in {"dim", "ndimension"}:
                        return len(shape)
                    if leaf_name == "numel":
                        return math.prod(shape)
                    if not expression.args:
                        return shape
                    dim = self._static_value(expression.args[0])
                    if isinstance(dim, int):
                        try:
                            return shape[dim]
                        except IndexError:
                            return _STATIC_MISSING
            values = [self._static_value(arg) for arg in expression.args]
            if any(value is _STATIC_MISSING for value in values):
                return _STATIC_MISSING
            function = _STATIC_CALLS.get(func_name)
            if function is not None:
                try:
                    return function(*values)
                except (ArithmeticError, TypeError, ValueError):
                    return _STATIC_MISSING
        return _STATIC_MISSING

    def _bind_target(self, target: ast.AST, tensor_ids: tuple[str, ...]) -> None:
        if isinstance(target, ast.Name) and tensor_ids:
            self.env[target.id] = tensor_ids[0]
            self.constants.pop(target.id, None)
            self.sequences.pop(target.id, None)
        elif isinstance(target, ast.Tuple):
            if len(tensor_ids) == 1:
                # ``chunk``/``split`` return several logical views.  AST
                # extraction cannot prove each partition extent, so retain the
                # common producer for every target while the node itself stays
                # explicitly inexact.  This preserves downstream dependency
                # evidence without inventing exact shapes or traffic.
                tensor_ids = tensor_ids * len(target.elts)
            for element, tensor_id in zip(target.elts, tensor_ids):
                if isinstance(element, ast.Name):
                    self.env[element.id] = tensor_id
        elif isinstance(target, ast.Subscript) and tensor_ids:
            base_tensor_ids = self._process_expr(target.value)
            if not base_tensor_ids:
                return
            updated_tensor_ids = self._append_node(
                op_family=OpFamily.DATA_MOVEMENT,
                op_name="setitem",
                source_expression=f"{ast.unparse(target)} = <value>",
                input_tensor_ids=(*base_tensor_ids, *tensor_ids),
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized indexed tensor update",
                attributes={"movement_kind": "materialized", "mutation": True},
            )
            root = _root_name(target.value)
            if root is not None:
                self.env[root] = updated_tensor_ids[0]

    def _bind_constant(self, target: ast.AST, value: object) -> None:
        if isinstance(target, ast.Name):
            self.constants[target.id] = value
            self.env.pop(target.id, None)
            self.sequences.pop(target.id, None)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple):
            for element, item in zip(target.elts, value):
                self._bind_constant(element, item)

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


def _is_dtype_metadata(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "dtype"


def _is_host_metadata_call(node: ast.Call, func_name: str) -> bool:
    if func_name.startswith("math.") or func_name in {
        "torch.device",
        "torch.finfo",
        "torch.nn.attention.sdpa_kernel",
    }:
        return True
    if isinstance(node.func, ast.Name) and func_name in {
        "bool",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "max",
        "min",
        "range",
        "reversed",
        "slice",
        "zip",
    }:
        return True
    return func_name.rsplit(".", maxsplit=1)[-1] in {
        "cpu",
        "index",
        "item",
        "numpy",
        "tolist",
    }


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


_STATIC_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_STATIC_CALLS: dict[str, Any] = {
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "math.ceil": math.ceil,
    "math.floor": math.floor,
    "math.log": math.log,
    "math.sqrt": math.sqrt,
}


_STATIC_MISSING = object()
