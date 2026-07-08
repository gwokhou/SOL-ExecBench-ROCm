# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reusable schema models for workload definitions."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class AxisConst(BaseModelWithDocstrings):
    """Constant axis with a fixed value."""

    type: Literal["const"] = "const"
    """The type identifier for constant axes."""
    value: NonNegativeInt
    """The constant integer value of this axis dimension."""
    description: Optional[str] = None
    """An optional human-readable description explaining the purpose of this axis."""


class AxisVar(BaseModel):
    """Variable axis that can be specified at runtime."""

    type: Literal["var"] = "var"
    """The type identifier for variable axes."""
    description: Optional[str] = Field(default=None)
    """An optional human-readable description explaining the purpose of this axis."""


class AxisExpr(BaseModel):
    """Expression axis that can be specified at runtime."""

    type: Literal["expr"] = "expr"
    """The type identifier for expression axes."""
    expression: NonEmptyString
    """The mathematical expression that defines the value of this axis."""
    description: Optional[str] = Field(default=None)
    """An optional human-readable description explaining the purpose of this axis."""


class DType(str, Enum):
    """Supported data types for tensors."""

    FLOAT64 = "float64"
    """64-bit IEEE 754 floating point."""
    FLOAT32 = "float32"
    """32-bit IEEE 754 floating point."""
    FLOAT16 = "float16"
    """16-bit IEEE 754 half-precision floating point."""
    BFLOAT16 = "bfloat16"
    """16-bit Brain Floating Point format."""
    FLOAT8_E4M3FN = "float8_e4m3fn"
    """8-bit floating point with 4 exponent bits and 3 mantissa bits."""
    FLOAT8_E5M2 = "float8_e5m2"
    """8-bit floating point with 5 exponent bits and 2 mantissa bits."""
    FLOAT4_E2M1 = "float4_e2m1"
    """4-bit floating point with 2 exponent bits and 1 mantissa bit."""
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    """4-bit floating point with 2 exponent bits and 1 mantissa bit, packed into a single byte."""
    INT64 = "int64"
    """64-bit signed integer."""
    INT32 = "int32"
    """32-bit signed integer."""
    INT16 = "int16"
    """16-bit signed integer."""
    INT8 = "int8"
    """8-bit signed integer."""
    BOOL = "bool"
    """Boolean type."""


class TensorSpec(BaseModelWithDocstrings):
    """Specification for an input, output, or scalar tensor."""

    shape: Optional[list[NonEmptyString]]
    """List of axis names defining the tensor shape. None for scalar values."""
    dtype: DType
    """The data type of all elements in this tensor."""
    description: Optional[str] = None
    """An optional human-readable description of this tensor's purpose and usage."""


AxisSpec = Union[AxisConst, AxisVar, AxisExpr]
"""Union type representing all possible axis specifications."""
