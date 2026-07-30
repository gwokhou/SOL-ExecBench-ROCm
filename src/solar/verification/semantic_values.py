"""Typed decoding for serialized IR semantic values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from solar.types import DynamicValue

type LiteralKey = Literal["literal", "value"]


class SemanticValueErrorKind(StrEnum):
    """Stable failure categories for dialect-specific error translation."""

    MISSING_TENSOR = "missing_tensor"
    INVALID_DTYPE = "invalid_dtype"
    INVALID_LAYOUT = "invalid_layout"
    INVALID_VALUE = "invalid_value"


class SemanticValueDecodeError(ValueError):
    """A serialized semantic value cannot be decoded exactly."""

    def __init__(
        self,
        kind: SemanticValueErrorKind,
        detail: object = None,
    ) -> None:
        """Initialize a categorized decoder failure."""
        super().__init__(kind.value)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SemanticValuePolicy:
    """Dialect choices for otherwise shared semantic value decoding."""

    literal_keys: tuple[LiteralKey, ...]
    decode_bare_memory_formats: bool = False


ATEN_VALUE_POLICY = SemanticValuePolicy(
    literal_keys=("value",),
    decode_bare_memory_formats=True,
)
EXTENDED_VALUE_POLICY = SemanticValuePolicy(
    literal_keys=("literal", "value"),
)


def decode_semantic_value(
    value: DynamicValue,
    operands: Sequence[DynamicValue],
    *,
    policy: SemanticValuePolicy,
) -> DynamicValue:
    """Decode one recursively serialized IR value."""
    if policy.decode_bare_memory_formats:
        memory_format = _memory_format(value)
        if memory_format is not None:
            return memory_format
    match value:
        case list() as items:
            return [
                decode_semantic_value(item, operands, policy=policy)
                for item in items
            ]
        case tuple() as items:
            return tuple(
                decode_semantic_value(item, operands, policy=policy)
                for item in items
            )
        case Mapping() as encoded:
            return _decode_mapping(encoded, operands, policy)
        case _:
            return value


def _decode_mapping(
    encoded: Mapping[str, DynamicValue],
    operands: Sequence[DynamicValue],
    policy: SemanticValuePolicy,
) -> DynamicValue:
    import torch

    if "tensor" in encoded:
        index = int(encoded["tensor"])
        if index not in range(len(operands)):
            raise SemanticValueDecodeError(
                SemanticValueErrorKind.MISSING_TENSOR,
                index,
            )
        return operands[index]
    if "dtype" in encoded:
        dtype = getattr(torch, str(encoded["dtype"]), None)
        if not isinstance(dtype, torch.dtype):
            raise SemanticValueDecodeError(
                SemanticValueErrorKind.INVALID_DTYPE,
                encoded["dtype"],
            )
        return dtype
    if "device" in encoded:
        return torch.device(str(encoded["device"]))
    if "layout" in encoded:
        layout = getattr(torch, str(encoded["layout"]), None)
        if not isinstance(layout, torch.layout):
            raise SemanticValueDecodeError(
                SemanticValueErrorKind.INVALID_LAYOUT,
                encoded["layout"],
            )
        return layout
    for key in policy.literal_keys:
        if key in encoded:
            return _decode_literal(encoded[key], policy)
    if "slice" in encoded:
        parts = [
            decode_semantic_value(item, operands, policy=policy)
            for item in encoded["slice"]
        ]
        return slice(*parts)
    raise SemanticValueDecodeError(SemanticValueErrorKind.INVALID_VALUE)


def _decode_literal(
    value: DynamicValue,
    policy: SemanticValuePolicy,
) -> DynamicValue:
    if value == "__ellipsis__":
        return Ellipsis
    if policy.decode_bare_memory_formats:
        memory_format = _memory_format(value)
        if memory_format is not None:
            return memory_format
    return value


def _memory_format(value: DynamicValue) -> DynamicValue | None:
    if not isinstance(value, str):
        return None
    import torch

    return {
        "preserve_format": torch.preserve_format,
        "contiguous_format": torch.contiguous_format,
    }.get(value)


__all__ = [
    "ATEN_VALUE_POLICY",
    "EXTENDED_VALUE_POLICY",
    "SemanticValueDecodeError",
    "SemanticValueErrorKind",
    "SemanticValuePolicy",
    "decode_semantic_value",
]
