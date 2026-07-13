# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Matrix and convolution AMD bound work estimators."""

from __future__ import annotations


from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

from sol_execbench.core.scoring.amd_bound_estimate.common import (
    _convolution_dims,
    _estimate_tensors,
    _infer_gemm_dims,
    _join_rationale,
    _node_tensors,
    _sum_tensor_bytes,
    _unsupported_estimate,
)


def _gemm_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors = _node_tensors(graph, node.input_tensor_ids)
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    warnings: list[str] = []
    rationale_parts: list[str] = []
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes

    dims = _infer_gemm_dims(
        input_tensors,
        output_tensors,
        transpose_rhs=node.op_family == OpFamily.LINEAR_PROJECTION,
    )
    if dims is None:
        if not input_tensors and not output_tensors:
            return _unsupported_estimate(
                node,
                rationale="unsupported GEMM estimate: all key tensors are unresolved",
                warnings=("unsupported_operator:gemm_missing_tensors",),
            )
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=node.op_family,
            op_name=node.op_name,
            formula_kind="gemm_flops",
            formula="2*M*N*K",
            formula_inputs={},
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "GEMM semantics recognized but missing shape evidence for M/N/K",
                rationale_parts,
            ),
            warnings=tuple(warnings or ("inexact_operator:gemm_missing_shape",)),
        )

    if "B" in dims:
        formula_kind = "batched_gemm_flops"
        formula = "2*B*M*N*K"
        flops = float(2 * dims["B"] * dims["M"] * dims["N"] * dims["K"])
    else:
        formula_kind = "gemm_flops"
        formula = "2*M*N*K"
        flops = float(2 * dims["M"] * dims["N"] * dims["K"])

    if node.confidence != EstimateConfidence.SUPPORTED:
        warnings.append("inexact_operator:gemm_graph_shape_unresolved")
    confidence = (
        EstimateConfidence.SUPPORTED
        if node.confidence == EstimateConfidence.SUPPORTED and not warnings
        else EstimateConfidence.INEXACT
    )
    dtypes = {tensor.dtype for tensor in (*input_tensors, *output_tensors)}
    if any(dtype.startswith("float8_") for dtype in dtypes):
        # The current gfx12 probe contract has no validated FP8 WMMA
        # instruction/throughput probe. Keep the operation modeled, but do not
        # let a generic 2*M*N*K count promote it into authority evidence.
        confidence = EstimateConfidence.INEXACT
        warnings.append("inexact_operator:gemm_fp8_matrix_probe_unavailable")
        rationale_parts.append("FP8 matrix instruction calibration is unavailable")
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind=formula_kind,
        formula=formula,
        formula_inputs=dict(dims),
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            "GEMM FLOPs estimated from input/output tensor shapes",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        warnings=tuple(warnings),
    )


def _convolution_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes
    dims, missing = _convolution_dims(input_tensors, output_tensors, node)
    warnings.extend(f"inexact_operator:convolution_missing_{item}" for item in missing)
    rationale_parts.extend(f"missing convolution {item}" for item in missing)
    if dims is None:
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.CONVOLUTION,
            op_name=node.op_name,
            formula_kind="convolution_flops",
            formula="2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements",
            formula_inputs={},
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "convolution semantics recognized but static metadata is incomplete",
                rationale_parts,
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    flops = float(
        2
        * dims["N"]
        * dims["C_out"]
        * dims["output_spatial_elements"]
        * (dims["C_in"] // dims["groups"])
        * dims["kernel_elements"]
    )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.CONVOLUTION,
        op_name=node.op_name,
        formula_kind="convolution_flops",
        formula="2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements",
        formula_inputs=dims,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.SUPPORTED
        if not warnings
        else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "convolution FLOPs estimated from input, weight, output, and grouping metadata",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        warnings=tuple(dict.fromkeys(warnings)),
    )
