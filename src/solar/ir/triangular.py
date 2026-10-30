# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static triangular selection from structured calls, without tensor sampling."""

from collections.abc import Mapping

from solar.ir.access import AccessReason, TensorAccess
from solar.ir.access_views import UnsupportedAccessError, slice_access
from solar.ir.contracts import operation_attributes, operation_operands
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    SemanticValueDecodeError,
    decode_semantic_value,
)
from solar.types import DynamicValue

TRIANGULAR_TARGETS = frozenset({"tril", "triu"})


def _clamp_row(value: int, rows: int) -> int:
    return min(rows, max(0, value))


def triangular_diagonal(semantic: Mapping[str, DynamicValue]) -> int:
    """Read a concrete diagonal from either independently extracted dialect."""
    args = operation_operands(semantic)
    diagonal = operation_attributes(semantic).get(
        "diagonal", args[1] if len(args) > 1 else 0
    )
    try:
        diagonal = decode_semantic_value(
            diagonal, (), policy=EXTENDED_VALUE_POLICY
        )
    except SemanticValueDecodeError as exc:
        raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX) from exc
    if type(diagonal) is not int:
        raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
    return diagonal


def triangular_accesses(
    view: TensorAccess,
    semantic: Mapping[str, DynamicValue],
    *,
    max_regions: int,
) -> tuple[TensorAccess, ...]:
    """Represent copied input cells as disjoint rows and one full rectangle.

    Work depends on the sloping boundary, not on matrix or batch element count.
    Full/empty and narrow triangular domains therefore remain cheap at huge
    nominal sizes. All regions retain the original logical base and epoch.
    """
    if max_regions <= 0:
        raise ValueError("triangle region budget must be positive")
    if len(view.shape) < 2:
        raise ValueError("triangular selection requires at least two axes")
    target = semantic.get("target")
    if target not in TRIANGULAR_TARGETS:
        raise ValueError("triangular selection requires a triangular target")
    diagonal = triangular_diagonal(semantic)
    rows, columns = view.shape[-2:]
    if view.elements == 0:
        return ()
    if target == "tril":
        first = _clamp_row(-diagonal, rows)
        last = _clamp_row(columns - diagonal - 1, rows)
        full_start, full_end = last, rows
    else:
        first = _clamp_row(1 - diagonal, rows)
        last = _clamp_row(columns - diagonal, rows)
        full_start, full_end = 0, first
    count = last - first + (full_end > full_start)
    if count > max_regions:
        raise UnsupportedAccessError(AccessReason.REGION_BUDGET)
    regions = []
    for row in range(first, last):
        start, end = (
            (0, row + diagonal + 1)
            if target == "tril"
            else (row + diagonal, columns)
        )
        strip = slice_access(view, len(view.shape) - 2, row, row + 1)
        regions.append(slice_access(strip, len(view.shape) - 1, start, end))
    if full_end > full_start:
        regions.append(
            slice_access(view, len(view.shape) - 2, full_start, full_end)
        )
    return tuple(regions)
