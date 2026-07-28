"""Shared contracts for the Torchview extended-einsum converter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PathLike = str | Path


class ConversionError(ValueError):
    """The PyTorch graph cannot be converted without approximation."""


@dataclass(frozen=True, slots=True)
class ConvertedTensorMetadata:
    """Normalized tensor metadata collected while converting one layer."""

    tensor_names: dict[str, list[str]]
    tensor_types: dict[str, list[str]]
    tensor_shapes: dict[str, list[list[int]]]
    tensor_dtypes: dict[str, Any]
    activation_connections: list[str]
    output_connections: list[str]
    additional_info: dict[str, Any]
