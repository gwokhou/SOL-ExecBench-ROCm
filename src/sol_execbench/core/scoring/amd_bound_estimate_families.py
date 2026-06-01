# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family dispatch helpers for AMD bound work estimates."""

from __future__ import annotations

from enum import Enum


class EstimateDispatchFamily(str, Enum):
    """Formula group used to estimate a bound graph operation family."""

    ATTENTION = "attention"
    CONVOLUTION = "convolution"
    EMBEDDING_POSITIONAL = "embedding_positional"
    MOE = "moe"
    SSM_MAMBA = "ssm_mamba"
    GEMM = "gemm"
    ELEMENTWISE = "elementwise"
    ACTIVATION = "activation"
    REDUCTION = "reduction"
    NORMALIZATION = "normalization"
    SOFTMAX = "softmax"
    DATA_MOVEMENT = "data_movement"
    DTYPE_CONVERSION = "dtype_conversion"
    UNSUPPORTED = "unsupported"


_DISPATCH_BY_OP_FAMILY_VALUE = {
    "attention": EstimateDispatchFamily.ATTENTION,
    "convolution": EstimateDispatchFamily.CONVOLUTION,
    "embedding_positional": EstimateDispatchFamily.EMBEDDING_POSITIONAL,
    "moe": EstimateDispatchFamily.MOE,
    "ssm_mamba": EstimateDispatchFamily.SSM_MAMBA,
    "gemm": EstimateDispatchFamily.GEMM,
    "linear_projection": EstimateDispatchFamily.GEMM,
    "elementwise": EstimateDispatchFamily.ELEMENTWISE,
    "mlp_activation": EstimateDispatchFamily.ACTIVATION,
    "reduction": EstimateDispatchFamily.REDUCTION,
    "normalization": EstimateDispatchFamily.NORMALIZATION,
    "softmax": EstimateDispatchFamily.SOFTMAX,
    "data_movement": EstimateDispatchFamily.DATA_MOVEMENT,
    "dtype_conversion": EstimateDispatchFamily.DTYPE_CONVERSION,
}


def estimate_dispatch_family(op_family: object) -> EstimateDispatchFamily:
    """Return the estimate formula group for an operation family enum/value."""
    value = getattr(op_family, "value", op_family)
    return _DISPATCH_BY_OP_FAMILY_VALUE.get(str(value), EstimateDispatchFamily.UNSUPPORTED)


def estimate_dispatch_groups() -> dict[EstimateDispatchFamily, tuple[str, ...]]:
    """Return operation-family values grouped by estimate dispatch family."""
    groups: dict[EstimateDispatchFamily, list[str]] = {
        family: [] for family in EstimateDispatchFamily
    }
    for op_family_value, dispatch_family in _DISPATCH_BY_OP_FAMILY_VALUE.items():
        groups[dispatch_family].append(op_family_value)
    return {
        dispatch_family: tuple(sorted(op_family_values))
        for dispatch_family, op_family_values in groups.items()
    }
