# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR-owned shape-domain contracts (stage 1: concrete instantiation)."""

from solar.shapes.domain import (
    Assumption,
    AssumptionOp,
    DimExpr,
    IntervalDim,
    ShapeDomain,
    ShapeReasonCode,
    ShapeSymbol,
    ShapeVerdict,
    TensorBinding,
    TensorRole,
    bind_tensors,
    certify_structure,
    structural_projection,
)

__all__ = [
    "Assumption",
    "AssumptionOp",
    "DimExpr",
    "IntervalDim",
    "ShapeDomain",
    "ShapeReasonCode",
    "ShapeSymbol",
    "ShapeVerdict",
    "TensorBinding",
    "TensorRole",
    "bind_tensors",
    "certify_structure",
    "structural_projection",
]
