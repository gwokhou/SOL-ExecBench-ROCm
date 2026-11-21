"""Static spatial, reduction, normalization, and attention geometry counts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from solar.ir.contracts import (
    operation_attributes,
    operation_operands,
    resolve_operand_tensor_slots,
)
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    decode_semantic_value,
)
from solar.types import DynamicValue


def _unwrap_value(value: DynamicValue) -> DynamicValue:
    if isinstance(value, Mapping) and set(value) & {"value", "dtype"}:
        return _unwrap_value(value.get("value", value.get("dtype")))
    if isinstance(value, (list, tuple)):
        return [_unwrap_value(item) for item in value]
    return value


def _shape_prod(shape: Sequence[DynamicValue]) -> int:
    return int(
        math.prod(
            int(dim["lower"]) if isinstance(dim, Mapping) else int(dim)
            for dim in shape
        )
    )


def transposed_convolution_macs(
    inputs: list[DynamicValue],
    outputs: list[DynamicValue],
    semantic: Mapping[str, DynamicValue],
) -> int:
    """Count input/kernel products whose coordinates reach the actual output."""
    activation, weight = inputs[:2]
    if (
        not isinstance(activation, list)
        or not isinstance(weight, list)
        or len(activation) < 3
        or len(weight) != len(activation)
    ):
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
            if isinstance(value, int) and not isinstance(value, bool)
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
    out0 = outputs[0] if outputs and isinstance(outputs[0], list) else []
    spatial = 1
    for size, kernel, output, step, pad, dil in zip(
        activation[2:],
        weight[2:],
        out0[2:],
        stride,
        padding,
        dilation,
        strict=True,
    ):
        if step <= 0 or dil <= 0 or pad < 0:
            raise ValueError("invalid transposed convolution geometry")
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


def contraction_macs(
    target: str,
    input_shapes: list[DynamicValue],
    output_shapes: list[DynamicValue],
    semantic: Mapping[str, DynamicValue],
) -> int:
    """Return exact dense MAC count for canonical public contractions."""
    if not input_shapes or not output_shapes:
        return 0
    output_n = max(
        (_shape_prod(s) for s in output_shapes if isinstance(s, list)),
        default=0,
    )
    arity = len(input_shapes)
    if target in {"mm", "bmm", "matmul", "linear"}:
        slots = resolve_operand_tensor_slots(semantic, (0, 1), arity)
        if slots is None:
            return 0
        left = input_shapes[slots[0]]
        return (
            output_n * int(left[-1]) if isinstance(left, list) and left else 0
        )
    if target == "addmm":
        slots = resolve_operand_tensor_slots(semantic, (1, 2), arity)
        if slots is None:
            return 0
        left = input_shapes[slots[0]]
        return (
            output_n * int(left[-1]) if isinstance(left, list) and left else 0
        )
    if target.startswith("conv_transpose"):
        slots = resolve_operand_tensor_slots(semantic, (0, 1), arity)
        if slots is None:
            return 0
        selected = [input_shapes[slots[0]], input_shapes[slots[1]]]
        return transposed_convolution_macs(selected, output_shapes, semantic)
    if target.startswith("conv"):
        slots = resolve_operand_tensor_slots(semantic, (0, 1), arity)
        if slots is None:
            return 0
        weight = input_shapes[slots[1]]
        if not isinstance(weight, list) or len(weight) < 3:
            return 0
        return output_n * _shape_prod(weight[1:])
    return 0


def reduction_groups(
    shape: Sequence[DynamicValue] | None,
    semantic: Mapping[str, DynamicValue],
) -> int:
    """Return the number of independent reduction groups for ``shape``."""
    if not shape:
        return 1
    kwargs = operation_attributes(semantic)
    dim = _unwrap_value(kwargs.get("dim"))
    if dim is None:
        for argument in operation_operands(semantic):
            if isinstance(argument, Mapping) and "tensor" in argument:
                continue
            candidate = _unwrap_value(argument)
            if type(candidate) is int or (
                isinstance(candidate, (list, tuple))
                and all(type(item) is int for item in candidate)
            ):
                dim = candidate
                break
    if dim is None or isinstance(dim, bool):
        return 1
    if type(dim) is int:
        dims = [dim]
    elif isinstance(dim, (list, tuple)):
        if dim and not all(type(item) is int for item in dim):
            return 1
        dims = [item for item in dim if type(item) is int]
    else:
        return 1
    rank = len(shape)
    axes = {item % rank for item in dims}
    reduced = math.prod(_shape_prod([shape[axis]]) for axis in axes)
    return max(1, _shape_prod(shape) // max(1, int(reduced)))


def _layer_or_rms_groups(
    input_shapes: Sequence[DynamicValue],
    output_n: int,
    semantic: Mapping[str, DynamicValue],
) -> tuple[int, bool]:
    kwargs = operation_attributes(semantic)
    operands = operation_operands(semantic)
    norm_shape = _unwrap_value(kwargs.get("normalized_shape"))
    if norm_shape is None and len(operands) >= 2:
        cand = _unwrap_value(operands[1])
        if type(cand) is int or (
            isinstance(cand, (list, tuple))
            and all(type(x) is int for x in cand)
        ):
            norm_shape = cand
    reduced = 0
    if type(norm_shape) is int:
        reduced = max(1, norm_shape)
    elif isinstance(norm_shape, (list, tuple)) and norm_shape:
        reduced = max(1, math.prod(int(x) for x in norm_shape))
    elif len(input_shapes) >= 2 and isinstance(input_shapes[1], list):
        weight_shape = input_shapes[1]
        x_shape = input_shapes[0] if isinstance(input_shapes[0], list) else []
        tail = (
            x_shape[-len(weight_shape) :]
            if weight_shape and len(x_shape) >= len(weight_shape)
            else weight_shape
        )
        reduced = max(1, _shape_prod(tail)) if tail else 1
    elif input_shapes and isinstance(input_shapes[0], list) and input_shapes[0]:
        reduced = max(1, _shape_prod(input_shapes[0][-1:]))
    has_weight = (
        isinstance(operands[2], Mapping) and "tensor" in operands[2]
        if len(operands) >= 3
        else (
            isinstance(kwargs.get("weight"), Mapping)
            and "tensor" in kwargs["weight"]
            if "weight" in kwargs
            else len(input_shapes) >= 2
        )
    )
    groups = max(1, output_n // max(1, reduced)) if output_n > 0 else 0
    return groups, has_weight


def normalization_counts(
    target: str,
    input_shapes: Sequence[DynamicValue],
    output_shapes: Sequence[DynamicValue],
    output_n: int,
    semantic: Mapping[str, DynamicValue],
) -> tuple[int, int, int]:
    """Return ``(reduction_ops, sfu_ops, valu_ops)`` for non-softmax norms."""
    kwargs = operation_attributes(semantic)
    operands = operation_operands(semantic)
    x_shape = (
        input_shapes[0]
        if input_shapes and isinstance(input_shapes[0], list)
        else []
    )
    if target in {"layer_norm", "rms_norm"}:
        groups, has_weight = _layer_or_rms_groups(
            input_shapes, output_n, semantic
        )
        combines = max(0, output_n - groups)
        if target == "rms_norm":
            valu = (3 if has_weight else 2) * output_n
            return combines, groups, valu
        return 2 * combines, groups, 5 * output_n
    if target == "group_norm":
        batch = _shape_prod([x_shape[0]]) if len(x_shape) >= 1 else 1
        raw_groups = _unwrap_value(
            kwargs.get("num_groups", kwargs.get("group"))
        )
        if type(raw_groups) is not int:
            exact = str(semantic.get("exact_target", ""))
            idx = 6 if exact == "native_group_norm" else 1
            cand = _unwrap_value(operands[idx]) if len(operands) > idx else None
            raw_groups = cand if type(cand) is int else None
        if type(raw_groups) is int and raw_groups > 0:
            groups = batch * raw_groups
        elif len(output_shapes) >= 2 and isinstance(output_shapes[1], list):
            groups = max(1, _shape_prod(output_shapes[1]))
        else:
            groups = batch
        groups = min(output_n, max(1, groups)) if output_n > 0 else 0
        return 2 * max(0, output_n - groups), groups, 5 * output_n
    channels = _shape_prod([x_shape[1]]) if len(x_shape) >= 2 else 1
    channels = min(output_n, max(1, channels)) if output_n > 0 else 0
    raw_train = _unwrap_value(kwargs.get("training"))
    if not isinstance(raw_train, bool) and len(operands) >= 6:
        cand = _unwrap_value(operands[5])
        if isinstance(cand, bool):
            raw_train = cand
    training = bool(raw_train) if isinstance(raw_train, bool) else False
    if not training:
        return 0, channels, 2 * output_n
    return 2 * max(0, output_n - channels), channels, 5 * output_n


def attention_counts(
    input_shapes: Sequence[DynamicValue],
    output_shapes: Sequence[DynamicValue],
    semantic: Mapping[str, DynamicValue],
) -> tuple[int, int, int, int] | None:
    """Return ``(mfma_flops, reduction_ops, sfu_ops, valu_ops)`` for SDPA."""
    arity = len(input_shapes)
    slots = resolve_operand_tensor_slots(semantic, (0, 1, 2), arity)
    if slots is not None:
        q_shape = input_shapes[slots[0]]
        k_shape = input_shapes[slots[1]]
        v_shape: DynamicValue = input_shapes[slots[2]]
    else:
        q_shape = input_shapes[0] if arity > 0 else []
        k_shape = input_shapes[1] if arity > 1 else []
        v_shape = input_shapes[2] if arity > 2 else None
    if (
        not isinstance(q_shape, list)
        or not isinstance(k_shape, list)
        or len(q_shape) < 2
        or len(k_shape) < 2
    ):
        return None
    q_seq = _shape_prod([q_shape[-2]])
    q_width = _shape_prod([q_shape[-1]])
    k_seq = _shape_prod([k_shape[-2]])
    batch_heads = _shape_prod(q_shape[:-2]) if len(q_shape) > 2 else 1
    q_rows = batch_heads * q_seq
    if isinstance(v_shape, list) and len(v_shape) >= 2:
        v_width = _shape_prod([v_shape[-1]])
    elif (
        output_shapes
        and isinstance(output_shapes[0], list)
        and len(output_shapes[0]) >= 2
    ):
        v_width = _shape_prod([output_shapes[0][-1]])
    else:
        v_width = q_width
    kwargs = operation_attributes(semantic)
    operands = operation_operands(semantic)
    raw_causal = _unwrap_value(kwargs.get("is_causal"))
    if not isinstance(raw_causal, bool) and len(operands) >= 6:
        cand = _unwrap_value(operands[5])
        if isinstance(cand, bool):
            raw_causal = cand
    if raw_causal is True:
        diag = min(q_seq, k_seq)
        per_head = diag * (diag + 1) // 2 + max(0, q_seq - diag) * k_seq
        score_elements = batch_heads * per_head
    else:
        score_elements = q_rows * k_seq
    active_rows = q_rows if k_seq > 0 else 0
    mfma_flops = 2 * score_elements * (q_width + v_width)
    reduction_ops = 2 * max(0, score_elements - active_rows)
    return mfma_flops, reduction_ops, score_elements, 2 * score_elements
