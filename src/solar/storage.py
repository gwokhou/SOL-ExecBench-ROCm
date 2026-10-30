"""Tensor-storage identity and injectivity shared by tracing and verification."""

from typing import Annotated

from pydantic import Field

from solar.types import DynamicValue

type NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


def same_tensor_storage(left: DynamicValue, right: DynamicValue) -> bool:
    """Compare storage objects without exposing device addresses in artifacts."""
    import torch

    if not isinstance(left, torch.Tensor) or not isinstance(
        right, torch.Tensor
    ):
        return False
    if left is right:
        return True
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def injective_strides(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """Prove element uniqueness for ordinary strided layouts; false is unknown."""
    if len(shape) != len(strides) or any(n < 0 for n in (*shape, *strides)):
        return False
    if 0 in shape:
        return True
    covered = 1
    for stride, size in sorted(zip(strides, shape, strict=True)):
        if size <= 1:
            continue
        if stride < covered:
            return False
        covered += (size - 1) * stride
    return True
