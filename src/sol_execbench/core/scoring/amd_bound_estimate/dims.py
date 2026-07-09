# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Dimension inference helpers for AMD bound estimators."""

from __future__ import annotations

from math import prod

from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraphNode,
    BoundTensor,
)


def infer_gemm_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
) -> dict[str, int] | None:
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) == 2 and len(rhs_shape) == 2 and len(out_shape) >= 2:
        return {
            "M": int(lhs_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    if len(lhs_shape) >= 3 and len(rhs_shape) == 2 and len(out_shape) >= 3:
        batch = prod(out_shape[:-2])
        return {
            "B": int(batch),
            "M": int(out_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    if len(lhs_shape) >= 3 and len(rhs_shape) >= 3 and len(out_shape) >= 3:
        batch = prod(out_shape[:-2])
        return {
            "B": int(batch),
            "M": int(out_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    return None


def convolution_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> tuple[dict[str, int] | None, list[str]]:
    missing: list[str] = []
    dimensionality = node.attributes.get("dimensionality")
    if not isinstance(dimensionality, int) or dimensionality not in {1, 2, 3}:
        missing.append("dimensionality")
    for key in ("stride", "padding", "dilation", "groups", "output_spatial"):
        if key not in node.attributes:
            missing.append(key)
    if len(input_tensors) < 2:
        missing.append("input_or_weight")
        return None, missing
    if not output_tensors:
        missing.append("output")
        return None, missing
    input_shape = input_tensors[0].shape
    weight_shape = input_tensors[1].shape
    output_shape = output_tensors[0].shape
    if input_shape is None:
        missing.append("input_shape")
    if weight_shape is None:
        missing.append("kernel_shape")
    if output_shape is None:
        missing.append("output_shape")
    groups = node.attributes.get("groups")
    if not isinstance(groups, int) or groups <= 0:
        missing.append("groups")
    output_spatial = node.attributes.get("output_spatial")
    if not isinstance(output_spatial, tuple):
        missing.append("output_spatial")
    if missing:
        return None, list(dict.fromkeys(missing))
    assert isinstance(dimensionality, int)
    assert isinstance(groups, int)
    assert isinstance(output_spatial, tuple)
    assert (
        input_shape is not None
        and weight_shape is not None
        and output_shape is not None
    )
    if (
        len(input_shape) != dimensionality + 2
        or len(weight_shape) != dimensionality + 2
        or len(output_shape) != dimensionality + 2
        or len(output_spatial) != dimensionality
    ):
        return None, list(dict.fromkeys([*missing, "dimensionality_shape_match"]))
    batch = int(input_shape[0])
    input_channels = int(input_shape[1])
    output_channels = int(output_shape[1])
    kernel_elements = int(prod(weight_shape[2:]))
    if input_channels % groups != 0:
        return None, list(dict.fromkeys([*missing, "groups"]))
    if int(weight_shape[0]) != output_channels:
        return None, list(dict.fromkeys([*missing, "kernel_shape"]))
    expected_channels_per_group = input_channels // groups
    if int(weight_shape[1]) != expected_channels_per_group:
        return None, list(dict.fromkeys([*missing, "groups"]))
    return {
        "N": batch,
        "C_in": input_channels,
        "C_out": output_channels,
        "groups": groups,
        "output_spatial_elements": int(prod(output_spatial)),
        "kernel_elements": kernel_elements,
        "dimensionality": dimensionality,
    }, []


def attention_dims_from_node_or_shapes(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> dict[str, int] | None:
    dims = attention_dims_from_attributes(node)
    if dims is not None:
        return dims
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
        return None
    subrole = str(node.attributes.get("subrole") or "")
    if subrole == "qk_scores":
        return {
            "B": int(lhs_shape[0]),
            "H": int(lhs_shape[1]),
            "S_q": int(lhs_shape[2]),
            "S_k": int(rhs_shape[3]),
            "D": int(lhs_shape[3]),
        }
    if subrole == "pv_aggregation":
        return {
            "B": int(out_shape[0]),
            "H": int(out_shape[1]),
            "S_q": int(out_shape[2]),
            "S_k": int(rhs_shape[2]),
            "D": int(out_shape[3]),
        }
    return None


def attention_dims_from_attributes(node: BoundGraphNode) -> dict[str, int] | None:
    keys = {
        "B": "batch",
        "H": "heads",
        "S_q": "sequence_q",
        "S_k": "sequence_k",
        "D": "head_dim",
    }
    result: dict[str, int] = {}
    for formula_name, attribute_name in keys.items():
        value = node.attributes.get(attribute_name)
        if not isinstance(value, int):
            return None
        result[formula_name] = value
    return result


def attention_output_projection_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> dict[str, int] | None:
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) == 4 and len(rhs_shape) == 2 and len(out_shape) == 4:
        return {
            "M": int(lhs_shape[0] * lhs_shape[1] * lhs_shape[2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    dims = attention_dims_from_attributes(node)
    if dims is not None:
        return {
            "M": int(dims["B"] * dims["H"] * dims["S_q"]),
            "N": int(dims["D"]),
            "K": int(dims["D"]),
        }
    return infer_gemm_dims(input_tensors, output_tensors)
