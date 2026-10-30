"""Compute dtype attribution independent from output storage casts."""

from collections.abc import Mapping

from solar.analysis.resource_targets import (
    _COMPOSITE_SFU_OPS,
    _SFU_OPS,
    _VALU_OPS,
)
from solar.ir.contracts import operation_attributes, operation_operands
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    decode_semantic_value,
)
from solar.precision import normalize_dtype
from solar.types import DynamicValue


def _binary_compute_dtype(
    semantic: Mapping[str, DynamicValue],
    shapes: list[DynamicValue],
    dtypes: list[DynamicValue],
    fallback: str,
) -> DynamicValue:
    """Resolve promotion using shape-only operands; no storage or device access."""
    import torch

    operands = operation_operands(semantic)
    if len(operands) < 2:
        return None
    tensors = []
    for shape, raw in zip(shapes, dtypes, strict=True):
        name = str(raw).removeprefix("torch.")
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            name = {
                "fp16": "float16",
                "bf16": "bfloat16",
                "fp32": "float32",
                "fp64": "float64",
            }.get(normalize_dtype(raw, fallback), name)
            dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"cannot prove arithmetic dtype {raw}")
        tensors.append(
            torch.empty((0,) if shape else (), dtype=dtype, device="meta")
        )
    values = [
        decode_semantic_value(value, tensors, policy=EXTENDED_VALUE_POLICY)
        for value in operands[:2]
    ]
    return torch.result_type(*values)


def resource_compute_dtype(
    semantic: Mapping[str, DynamicValue],
    target: str,
    input_shapes: list[DynamicValue],
    input_dtypes: list[DynamicValue],
    output_dtypes: list[DynamicValue],
    compute_precision: str | None,
    fallback_precision: str,
) -> DynamicValue:
    """Use computation promotion, retaining separate contraction precision."""
    dtype = compute_precision or (
        input_dtypes[0]
        if input_dtypes
        else (output_dtypes[0] if output_dtypes else fallback_precision)
    )
    binary = target in {
        "add",
        "sub",
        "mul",
        "maximum",
        "minimum",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    }
    if compute_precision is None and binary:
        dtype = (
            _binary_compute_dtype(
                semantic, input_shapes, input_dtypes, fallback_precision
            )
            or dtype
        )
    elif (
        compute_precision is None
        and target in _VALU_OPS | _SFU_OPS | _COMPOSITE_SFU_OPS
        and not (semantic.get("effects") or {}).get("mutates")
        and "out" not in operation_attributes(semantic)
        and output_dtypes
        and normalize_dtype(output_dtypes[0], fallback_precision) != "bool"
    ):
        dtype = output_dtypes[0]
    return dtype
