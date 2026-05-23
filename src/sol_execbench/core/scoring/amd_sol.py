# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMD speed-of-light bound artifacts for benchmark workloads."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import prod

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    EstimateConfidence,
    HardwareValidationStatus as HardwareValidationStatus,
    default_amd_hardware_models as load_default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import BoundGraphNode, OpFamily, build_bound_graph


AMD_SOL_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v1"


@dataclass(frozen=True)
class GraphNode:
    """Normalized operation graph node."""

    node_id: str
    op_type: str
    expression: str
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "expression": self.expression,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class WorkEstimate:
    """Estimated FLOPs and bytes for one graph node."""

    node_id: str
    flops: float
    bytes_accessed: float
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "flops": self.flops,
            "bytes_accessed": self.bytes_accessed,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class OpSolBound:
    """Per-operation AMD speed-of-light bound."""

    node_id: str
    compute_bound_ms: float
    memory_bound_ms: float
    sol_bound_ms: float
    limiting_resource: str
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "limiting_resource": self.limiting_resource,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AmdSolBoundArtifact:
    """Auditable AMD SOL bound artifact for one workload."""

    definition: str
    workload_uuid: str
    hardware_model: AmdHardwareModel
    graph_nodes: tuple[GraphNode, ...]
    work_estimates: tuple[WorkEstimate, ...]
    op_bounds: tuple[OpSolBound, ...]
    schema_version: str = AMD_SOL_SCHEMA_VERSION
    derived: bool = True

    @property
    def aggregate_sol_bound_ms(self) -> float:
        """Aggregate SOL bound as the sum of per-op bounds."""
        return sum(bound.sol_bound_ms for bound in self.op_bounds)

    @property
    def coverage_summary(self) -> AmdSolCoverageSummary:
        """Derived coverage summary for this artifact."""
        return summarize_amd_sol_coverage(self.graph_nodes, self.work_estimates)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model": self.hardware_model.to_dict(),
            "graph_nodes": [node.to_dict() for node in self.graph_nodes],
            "work_estimates": [estimate.to_dict() for estimate in self.work_estimates],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_sol_bound_ms": self.aggregate_sol_bound_ms,
            "coverage_summary": self.coverage_summary.to_dict(),
        }


@dataclass(frozen=True)
class AmdSolCoverageSummary:
    """Derived AMD SOL coverage summary for one graph or artifact."""

    total_ops: int
    supported_ops: int
    inexact_ops: int
    unsupported_ops: int
    op_type_counts: dict[str, int]
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "derived": self.derived,
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "inexact_ops": self.inexact_ops,
            "unsupported_ops": self.unsupported_ops,
            "op_type_counts": dict(self.op_type_counts),
        }


def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    """Return built-in AMD hardware model entries."""
    return load_default_amd_hardware_models()


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


def estimate_work(
    definition: Definition,
    workload: Workload,
    graph_nodes: tuple[GraphNode, ...],
) -> tuple[WorkEstimate, ...]:
    """Estimate FLOPs and bytes for graph nodes."""
    try:
        bound_graph = build_bound_graph(definition, workload)
        rich_estimates = estimate_bound_work(bound_graph)
        if graph_nodes and len(graph_nodes) != len(rich_estimates):
            raise ValueError(
                "legacy graph node count does not match rich bound estimate count"
            )
        return tuple(
            _work_estimate_from_rich_estimate(
                estimate,
                node_id=graph_nodes[index].node_id if graph_nodes else estimate.node_id,
            )
            for index, estimate in enumerate(rich_estimates)
        )
    except Exception as exc:
        return _legacy_estimate_work(
            definition,
            workload,
            graph_nodes,
            fallback_reason=f"rich bound estimate failed: {exc}",
        )


def _legacy_estimate_work(
    definition: Definition,
    workload: Workload,
    graph_nodes: tuple[GraphNode, ...],
    *,
    fallback_reason: str,
) -> tuple[WorkEstimate, ...]:
    """Legacy whole-definition estimator retained as explicit exceptional fallback."""
    axes = definition.get_resolved_axes_values(workload.axes)
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    tensor_bytes = _tensor_bytes(definition, input_shapes, output_shapes)
    output_numel = sum(prod(shape) for shape in output_shapes.values() if shape)
    input_numel = sum(prod(shape) for shape in input_shapes.values() if shape)
    reduction_dim = _largest_reduction_dim(definition, axes)
    estimates: list[WorkEstimate] = []

    for node in graph_nodes:
        if node.op_type == "matmul" and output_numel and reduction_dim:
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=float(2 * output_numel * reduction_dim),
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.SUPPORTED,
                    rationale=(
                        "legacy fallback: matmul FLOPs estimated as 2 * output "
                        f"elements * reduction dimension ({fallback_reason})"
                    ),
                )
            )
        elif node.op_type in {"elementwise", "activation"} and output_numel:
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=float(output_numel),
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.INEXACT,
                    rationale=(
                        f"legacy fallback: {node.op_type} work estimated as one "
                        f"operation per output element ({fallback_reason})"
                    ),
                )
            )
        elif node.op_type == "reduction" and (input_numel or output_numel):
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=float(max(input_numel, output_numel)),
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.INEXACT,
                    rationale=(
                        "legacy fallback: reduction work conservatively estimated "
                        f"from input elements ({fallback_reason})"
                    ),
                )
            )
        elif node.op_type == "normalization" and (input_numel or output_numel):
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=float(4 * max(input_numel, output_numel)),
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.INEXACT,
                    rationale=(
                        "legacy fallback: normalization-like work conservatively "
                        "estimates reductions, scaling, and elementwise application "
                        f"({fallback_reason})"
                    ),
                )
            )
        elif node.op_type == "softmax" and (input_numel or output_numel):
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=float(5 * max(input_numel, output_numel)),
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.INEXACT,
                    rationale=(
                        "legacy fallback: softmax-like work conservatively estimates "
                        f"max, exp, sum, and normalization passes ({fallback_reason})"
                    ),
                )
            )
        elif node.op_type == "data_movement":
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=0.0,
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.INEXACT,
                    rationale=(
                        "legacy fallback: data movement or view-like operation "
                        f"modeled as zero-FLOP tensor traffic ({fallback_reason})"
                    ),
                )
            )
        else:
            estimates.append(
                WorkEstimate(
                    node_id=node.node_id,
                    flops=0.0,
                    bytes_accessed=float(tensor_bytes),
                    confidence=EstimateConfidence.UNSUPPORTED,
                    rationale=(
                        f"legacy fallback: unsupported operation estimate for "
                        f"{node.op_type} ({fallback_reason})"
                    ),
                )
            )
    return tuple(estimates)


def _work_estimate_from_rich_estimate(
    estimate: OperatorWorkEstimate,
    *,
    node_id: str,
) -> WorkEstimate:
    return WorkEstimate(
        node_id=node_id,
        flops=estimate.flops,
        bytes_accessed=estimate.total_bytes,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
    )


def summarize_amd_sol_coverage(
    graph_nodes: tuple[GraphNode, ...],
    work_estimates: tuple[WorkEstimate, ...],
) -> AmdSolCoverageSummary:
    """Summarize analyzer coverage for a graph and its work estimates."""
    op_type_counts: dict[str, int] = {}
    for node in graph_nodes:
        op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

    return AmdSolCoverageSummary(
        total_ops=len(graph_nodes),
        supported_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.SUPPORTED
        ),
        inexact_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.INEXACT
        ),
        unsupported_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.UNSUPPORTED
        ),
        op_type_counts=op_type_counts,
    )


def build_amd_sol_bound_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
) -> AmdSolBoundArtifact:
    """Build an auditable AMD SOL bound artifact."""
    graph_nodes = extract_graph(definition)
    work_estimates = estimate_work(definition, workload, graph_nodes)
    op_bounds = tuple(
        _bound_for_estimate(estimate, hardware_model) for estimate in work_estimates
    )
    return AmdSolBoundArtifact(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model=hardware_model,
        graph_nodes=graph_nodes,
        work_estimates=work_estimates,
        op_bounds=op_bounds,
    )


def _minimal_workload_for_definition(definition: Definition) -> Workload:
    axes = {name: 1 for name in definition.var_axes}
    return Workload(
        axes=axes,
        inputs={name: {"type": "random"} for name in definition.inputs},
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


def _dtype_bytes(dtype: DType) -> float:
    return {
        DType.FLOAT64: 8.0,
        DType.FLOAT32: 4.0,
        DType.FLOAT16: 2.0,
        DType.BFLOAT16: 2.0,
        DType.FLOAT8_E4M3FN: 1.0,
        DType.FLOAT8_E5M2: 1.0,
        DType.FLOAT4_E2M1: 0.5,
        DType.FLOAT4_E2M1FN_X2: 0.5,
        DType.INT64: 8.0,
        DType.INT32: 4.0,
        DType.INT16: 2.0,
        DType.INT8: 1.0,
        DType.BOOL: 1.0,
    }[dtype]


def _tensor_bytes(
    definition: Definition,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
) -> float:
    total = 0.0
    for name, shape in input_shapes.items():
        if shape:
            total += prod(shape) * _dtype_bytes(definition.inputs[name].dtype)
    for name, shape in output_shapes.items():
        if shape:
            total += prod(shape) * _dtype_bytes(definition.outputs[name].dtype)
    return total


def _largest_reduction_dim(definition: Definition, axes: dict[str, int]) -> int:
    """Infer the common matmul K dimension from the first shaped input."""
    for spec in definition.inputs.values():
        if spec.shape:
            axis_name = spec.shape[-1]
            if axis_name.isdigit():
                return int(axis_name)
            return axes.get(axis_name, 0)
    return 0


def _bound_for_estimate(
    estimate: WorkEstimate,
    hardware_model: AmdHardwareModel,
) -> OpSolBound:
    compute_bound_ms = (
        estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
        if hardware_model.peak_tflops > 0
        else 0.0
    )
    memory_bound_ms = (
        estimate.bytes_accessed
        / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0)
        * 1000.0
        if hardware_model.memory_bandwidth_gbps > 0
        else 0.0
    )
    limiting_resource = "compute" if compute_bound_ms >= memory_bound_ms else "memory"
    return OpSolBound(
        node_id=estimate.node_id,
        compute_bound_ms=compute_bound_ms,
        memory_bound_ms=memory_bound_ms,
        sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
        limiting_resource=limiting_resource,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
    )
