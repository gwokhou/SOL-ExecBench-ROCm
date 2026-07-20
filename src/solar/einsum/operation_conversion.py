# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed stage models for PyTorch-to-einsum operation conversion."""

from __future__ import annotations

from dataclasses import dataclass

from solar.einsum.ops.base import EinsumOp

REDUCTION_OPS_WITH_DIM = frozenset(
    {
        "sum",
        "mean",
        "prod",
        "max",
        "min",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "logsumexp",
        "norm",
        "std",
        "var",
        "all",
        "any",
        "nansum",
        "nanmean",
    }
)

FORCE_ATEN_SEMANTICS_OPS = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "convtranspose1d",
        "convtranspose2d",
        "convtranspose3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "scaled_dot_product_attention",
    }
)


@dataclass(frozen=True, slots=True)
class OperationRepresentation:
    """Stable result of the semantic-representation conversion stage."""

    equation: str
    operands: dict[str, list[str]]
    elementwise_op: str
    reduction_op: str
    is_real_einsum: bool
    is_einsum_supportable: bool

    @classmethod
    def from_einsum_op(cls, operation: EinsumOp) -> OperationRepresentation:
        return cls(
            equation=operation.equation,
            operands={operand.name: operand.dims for operand in operation.operands},
            elementwise_op=operation.elementwise_op,
            reduction_op=operation.reduction_op,
            is_real_einsum=operation.is_real_einsum,
            is_einsum_supportable=operation.is_einsum_supportable,
        )


def default_operation_representation() -> OperationRepresentation:
    """Return the legacy permissive representation used after tracing gaps."""
    return OperationRepresentation(
        equation="",
        operands={},
        elementwise_op="mul",
        reduction_op="add",
        is_real_einsum=True,
        is_einsum_supportable=True,
    )


__all__ = [
    "FORCE_ATEN_SEMANTICS_OPS",
    "REDUCTION_OPS_WITH_DIM",
    "OperationRepresentation",
    "default_operation_representation",
]
