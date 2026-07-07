# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Graph extraction helpers for AMD SOL v1 artifacts."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import RandomInput, Workload
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraphNode, OpFamily
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.amd_sol_models import GraphNode

def extract_graph(definition: Definition) -> tuple[GraphNode, ...]:
    """Extract a conservative operation graph from reference code."""
    try:
        workload = _minimal_workload_for_definition(definition)
        bound_graph = build_bound_graph(definition, workload)
        return tuple(_graph_node_from_bound_node(node) for node in bound_graph.nodes)
    except Exception:
        tree = ast.parse(definition.reference, mode="exec")
        visitor = _GraphVisitor()
        visitor.visit(tree)
        return tuple(visitor.nodes)

def _minimal_workload_for_definition(definition: Definition) -> Workload:
    axes = {name: 1 for name in definition.var_axes}
    return Workload(
        axes=axes,
        inputs={name: RandomInput() for name in definition.inputs},
        uuid=f"{definition.name}-graph-extraction",
    )


def _graph_node_from_bound_node(node: BoundGraphNode) -> GraphNode:
    return GraphNode(
        node_id=node.node_id,
        op_type=_legacy_op_type(node.op_family),
        expression=node.source_expression,
        confidence=node.confidence,
        rationale=node.rationale,
    )


def _legacy_op_type(op_family: OpFamily) -> str:
    if op_family in {OpFamily.GEMM, OpFamily.LINEAR_PROJECTION}:
        return "matmul"
    if op_family == OpFamily.MLP_ACTIVATION:
        return "activation"
    if op_family == OpFamily.DTYPE_CONVERSION:
        return "data_movement"
    return {
        OpFamily.NORMALIZATION: "normalization",
        OpFamily.SOFTMAX: "softmax",
        OpFamily.REDUCTION: "reduction",
        OpFamily.ELEMENTWISE: "elementwise",
        OpFamily.DATA_MOVEMENT: "data_movement",
        OpFamily.UNSUPPORTED: "unsupported",
    }.get(op_family, "unsupported")


class _GraphVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []

    def visit_Call(self, node: ast.Call) -> None:
        expression = ast.unparse(node)
        func_name = _call_name(node.func)
        analyzer = _classify_call(func_name)
        if analyzer is None:
            self._append("unsupported", expression, EstimateConfidence.UNSUPPORTED, "unsupported call")
        elif analyzer.op_type == "ignore":
            pass
        else:
            self._append(
                analyzer.op_type,
                expression,
                analyzer.confidence,
                analyzer.rationale,
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self._append("@", ast.unparse(node), EstimateConfidence.SUPPORTED, "recognized @ matmul")
        elif isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            self._append(
                "elementwise",
                ast.unparse(node),
                EstimateConfidence.INEXACT,
                "recognized elementwise binary operation",
            )
        self.generic_visit(node)

    def _append(
        self,
        op_type: str,
        expression: str,
        confidence: EstimateConfidence,
        rationale: str,
    ) -> None:
        self.nodes.append(
            GraphNode(
                node_id=f"op_{len(self.nodes) + 1}",
                op_type="matmul" if op_type == "@" else op_type,
                expression=expression,
                confidence=confidence,
                rationale=rationale,
            )
        )


@dataclass(frozen=True)
class _CallAnalyzer:
    op_type: str
    confidence: EstimateConfidence
    rationale: str


_CALL_ANALYZERS: tuple[tuple[set[str], _CallAnalyzer], ...] = (
    (
        {"matmul", "mm", "bmm"},
        _CallAnalyzer("matmul", EstimateConfidence.SUPPORTED, "recognized matmul call"),
    ),
    (
        {"sum", "mean", "amax", "max", "amin", "min", "var", "std"},
        _CallAnalyzer(
            "reduction",
            EstimateConfidence.INEXACT,
            "recognized reduction call with conservative estimate",
        ),
    ),
    (
        {"layer_norm", "group_norm", "rms_norm", "norm", "rsqrt"},
        _CallAnalyzer(
            "normalization",
            EstimateConfidence.INEXACT,
            "recognized normalization-like call with conservative estimate",
        ),
    ),
    (
        {"softmax", "log_softmax"},
        _CallAnalyzer(
            "softmax",
            EstimateConfidence.INEXACT,
            "recognized softmax-like call with conservative estimate",
        ),
    ),
    (
        {"relu", "gelu", "silu", "sigmoid", "tanh", "exp", "sqrt"},
        _CallAnalyzer(
            "activation",
            EstimateConfidence.INEXACT,
            "recognized activation call with conservative estimate",
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
        _CallAnalyzer(
            "data_movement",
            EstimateConfidence.INEXACT,
            "recognized view or data-movement call",
        ),
    ),
    (
        {"to", "type", "float", "half", "bfloat16"},
        _CallAnalyzer("ignore", EstimateConfidence.INEXACT, "ignored dtype/device conversion"),
    ),
)


def _classify_call(func_name: str) -> _CallAnalyzer | None:
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    for names, analyzer in _CALL_ANALYZERS:
        if leaf_name in names or func_name in names:
            return analyzer
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""
