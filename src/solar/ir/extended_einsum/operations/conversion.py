# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Stable operation facts used by extended-einsum IR conversion."""

from __future__ import annotations

from dataclasses import dataclass

from solar.ir.extended_einsum.operations.handlers.base import EinsumOp

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
    },
)


@dataclass(frozen=True, slots=True)
class OperationRepresentation:
    """One handler-derived extended-einsum operation representation."""

    equation: str
    operands: dict[str, list[str]]
    elementwise_op: str
    reduction_op: str
    is_real_einsum: bool
    is_einsum_supportable: bool

    @classmethod
    def from_einsum_op(cls, operation: EinsumOp) -> OperationRepresentation:
        """Copy stable IR fields from a registered operation handler result."""
        return cls(
            equation=operation.equation,
            operands={item.name: item.dims for item in operation.operands},
            elementwise_op=operation.elementwise_op,
            reduction_op=operation.reduction_op,
            is_real_einsum=operation.is_real_einsum,
            is_einsum_supportable=operation.is_einsum_supportable,
        )


def default_operation_representation() -> OperationRepresentation:
    """Return the conservative identity-like representation for pass-throughs."""
    return OperationRepresentation("", {}, "none", "none", False, True)


__all__ = [
    "REDUCTION_OPS_WITH_DIM",
    "OperationRepresentation",
    "default_operation_representation",
]
