"""Static spatial contribution counts for resource accounting."""

from collections.abc import Mapping

from solar.ir.contracts import operation_attributes, operation_operands
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    decode_semantic_value,
)
from solar.types import DynamicValue


def transposed_convolution_macs(
    inputs: list[DynamicValue],
    outputs: list[DynamicValue],
    semantic: Mapping[str, DynamicValue],
) -> int:
    """Count input/kernel products whose coordinates reach the actual output."""
    activation, weight = inputs[:2]
    if len(activation) < 3 or len(weight) != len(activation):
        raise ValueError(
            "transposed convolution requires static spatial shapes"
        )
    rank = len(activation) - 2
    attributes = operation_attributes(semantic)
    operands = operation_operands(semantic)

    def parameter(name: str, slot: int, default: int) -> list[int]:
        raw = attributes.get(name)
        if raw is None:
            raw = operands[slot] if len(operands) > slot else default
        value = decode_semantic_value(raw, (), policy=EXTENDED_VALUE_POLICY)
        return (
            [int(value)] * rank
            if isinstance(value, int)
            else [int(item) for item in value]
        )

    stride = parameter("stride", 3, 1)
    padding = parameter("padding", 4, 0)
    dilation = parameter(
        "dilation", 5 if semantic.get("exact_target") == "convolution" else 7, 1
    )
    if not all(len(value) == rank for value in (stride, padding, dilation)):
        raise ValueError(
            "transposed convolution spatial parameter rank mismatch"
        )
    spatial = 1
    for size, kernel, output, step, pad, dil in zip(
        activation[2:],
        weight[2:],
        outputs[0][2:],
        stride,
        padding,
        dilation,
        strict=True,
    ):
        if step <= 0 or dil <= 0 or pad < 0:
            raise ValueError("invalid transposed convolution geometry")
        # Count input/kernel contributions landing inside the returned output.
        # Output-padding changes the admitted coordinates, not the dense
        # number of output positions times all kernel coefficients.
        pairs = sum(
            max(
                0,
                min(int(size) - 1, (int(output) - 1 + pad - k * dil) // step)
                - max(0, -((k * dil - pad) // step))
                + 1,
            )
            for k in range(int(kernel))
        )
        spatial *= pairs
    return int(activation[0]) * int(activation[1]) * int(weight[1]) * spatial
