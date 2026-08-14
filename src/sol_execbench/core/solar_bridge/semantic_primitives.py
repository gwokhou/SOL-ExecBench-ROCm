# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parse primitive SOLAR operations into typed performance descriptors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from sol_execbench.core.bench.performance_model.models import (
    CrossEntropyDescriptor,
    CrossEntropyReduction,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    IndexedReadDescriptor,
    IndexedReadOperation,
    IndexedUpdateDescriptor,
    IndexedUpdateOperation,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SoftmaxDescriptor,
    SoftmaxOperation,
    TensorDType,
    TransposeDescriptor,
)
from sol_execbench.core.data.definition_models import DType

_SIMPLE_OPS = frozenset(
    {"abs", "add", "clamp", "div", "mul", "neg", "relu", "sub"}
)
_TRANSCENDENTAL_OPS = frozenset({"cos", "exp", "log", "sigmoid", "sin", "tanh"})
_COMPOSITE_OPS = frozenset({"gelu"})
ELEMENTWISE_TYPES = _SIMPLE_OPS | _TRANSCENDENTAL_OPS | _COMPOSITE_OPS
TRANSPOSE_TYPES = frozenset({"permute", "transpose"})
REDUCTION_TYPES = {
    "sum": ReductionOperation.SUM,
    "mean": ReductionOperation.MEAN,
    "rms_norm": ReductionOperation.RMS_NORM,
    "layer_norm": ReductionOperation.LAYER_NORM,
    "native_layer_norm": ReductionOperation.LAYER_NORM,
}
SOFTMAX_TYPES = {
    "softmax": SoftmaxOperation.SOFTMAX,
    "_softmax": SoftmaxOperation.SOFTMAX,
    "log_softmax": SoftmaxOperation.LOG_SOFTMAX,
    "_log_softmax": SoftmaxOperation.LOG_SOFTMAX,
}
INDEXED_READ_TYPES = {
    "gather": IndexedReadOperation.GATHER,
    "index_select": IndexedReadOperation.INDEX_SELECT,
    "embedding": IndexedReadOperation.EMBEDDING,
}
INDEXED_UPDATE_TYPES = {
    "scatter": IndexedUpdateOperation.SCATTER,
    "index_copy": IndexedUpdateOperation.INDEX_COPY,
    "index_put": IndexedUpdateOperation.INDEX_PUT,
    "scatter_add": IndexedUpdateOperation.SCATTER_ADD,
    "index_add": IndexedUpdateOperation.INDEX_ADD,
}
MATMUL_TYPES = frozenset({"bmm", "gemm", "matmul", "mm"})


def elementwise_descriptor(
    layers: list[tuple[Mapping[str, object], str]],
) -> ElementwiseDescriptor | None:
    """Parse a homogeneous elementwise layer group."""
    if any(
        not _tensor_metadata(layer, "inputs")
        or not _tensor_metadata(layer, "outputs")
        for layer, _operation in layers
    ):
        return None
    outputs = [
        metadata
        for layer, _operation in layers
        for metadata in _tensor_metadata(layer, "outputs")
    ]
    if not outputs:
        return None
    shape, dtype = max(outputs, key=lambda item: _shape_elements(item[0]))
    if dtype not in {TensorDType.FLOAT32, TensorDType.BFLOAT16}:
        return None
    tensors = [
        metadata
        for layer, _operation in layers
        for direction in ("inputs", "outputs")
        for metadata in _tensor_metadata(layer, direction)
    ]
    if not tensors or any(item[1] is not dtype for item in tensors):
        return None
    counts: dict[ElementwiseOperationClass, float] = {}
    for layer, operation in layers:
        group = _elementwise_group(operation)
        layer_outputs = _tensor_metadata(layer, "outputs")
        amount = sum(_shape_elements(item[0]) for item in layer_outputs)
        counts[group] = counts.get(group, 0.0) + float(amount)
    return ElementwiseDescriptor(
        shape=shape,
        dtype=dtype,
        operations=counts,
    )


def _elementwise_group(operation: str) -> ElementwiseOperationClass:
    if operation in _SIMPLE_OPS:
        return ElementwiseOperationClass.SIMPLE
    if operation in _TRANSCENDENTAL_OPS:
        return ElementwiseOperationClass.TRANSCENDENTAL
    return ElementwiseOperationClass.COMPOSITE


def transpose_descriptor(
    layer: Mapping[str, object],
) -> TransposeDescriptor | None:
    """Parse one transpose or permute layer."""
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) != 1 or len(outputs) != 1:
        return None
    input_shape, dtype = inputs[0]
    output_shape, output_dtype = outputs[0]
    if (
        len(input_shape) != 2
        or output_shape != [input_shape[1], input_shape[0]]
        or output_dtype is not dtype
    ):
        return None
    rows, columns = input_shape
    return TransposeDescriptor(
        rows=rows,
        columns=columns,
        dtype=dtype,
        element_bytes=_dtype_bytes(dtype),
        input_strides=(columns, 1),
        output_strides=(rows, 1),
    )


def reduction_descriptor(
    layer: Mapping[str, object],
    operation: ReductionOperation,
) -> ReductionDescriptor | None:
    """Parse one reduction-family layer."""
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if not inputs or not outputs:
        return None
    input_shape, input_dtype = inputs[0]
    output_shape, output_dtype = outputs[0]
    if (
        not input_shape
        or input_dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}
        or output_dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}
    ):
        return None
    expected_outputs = (
        {tuple(input_shape)}
        if operation
        in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
        else {
            tuple(input_shape[:-1]),
            (*input_shape[:-1], 1),
        }
    )
    if tuple(output_shape) not in expected_outputs:
        return None
    if operation is ReductionOperation.LAYER_NORM:
        normalized_shape = _attributes(layer).get("normalized_shape")
        if normalized_shape not in (
            input_shape[-1],
            (input_shape[-1],),
            [input_shape[-1]],
        ):
            return None
    return ReductionDescriptor(
        operation=operation,
        outer_rows=max(math.prod(input_shape[:-1]), 1),
        reduction_width=input_shape[-1],
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


def matmul_descriptor(
    layer: Mapping[str, object],
    *,
    batched: bool,
) -> MatmulDescriptor | None:
    """Parse one matrix-multiplication layer."""
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) < 2 or len(outputs) != 1:
        return None
    left, right = inputs[0], inputs[1]
    output_shape, output_dtype = outputs[0]
    if left[1] is not right[1] or left[1] not in {
        TensorDType.FLOAT16,
        TensorDType.FLOAT32,
    }:
        return None
    if output_dtype not in {left[1], TensorDType.FLOAT32}:
        return None
    expected_rank = 3 if batched else 2
    if any(len(shape) != expected_rank for shape in (left[0], right[0])):
        return None
    batch = left[0][0] if batched else 1
    m, k = left[0][-2:]
    right_k, n = right[0][-2:]
    if (
        k != right_k
        or output_shape[-2:] != [m, n]
        or (batched and right[0][0] != batch)
        or (batched and output_shape[0] != batch)
        or (not batched and len(output_shape) != 2)
    ):
        return None
    return MatmulDescriptor(
        batch=batch,
        m=m,
        n=n,
        k=k,
        leading_dimension_a=k,
        leading_dimension_b=n,
        leading_dimension_c=n,
        input_dtype=left[1],
        output_dtype=output_dtype,
        contiguous=bool(_attributes(layer).get("contiguous", True)),
        batch_stride_a=(m * k if batched else None),
        batch_stride_b=(k * n if batched else None),
        batch_stride_c=(m * n if batched else None),
    )


def softmax_descriptor(
    layer: Mapping[str, object],
    operation: SoftmaxOperation,
) -> SoftmaxDescriptor | None:
    """Parse one softmax-family layer."""
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0] != outputs[0]:
        return None
    shape, dtype = inputs[0]
    if dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}:
        return None
    axis = _normalized_axis(_attributes(layer).get("dim"), len(shape))
    if axis != len(shape) - 1:
        return None
    return SoftmaxDescriptor(
        operation=operation,
        outer_rows=max(math.prod(shape[:-1]), 1),
        reduction_width=shape[-1],
        input_dtype=dtype,
        output_dtype=dtype,
    )


def cross_entropy_descriptor(
    layer: Mapping[str, object],
) -> CrossEntropyDescriptor | None:
    """Parse one native cross-entropy layer."""
    raw = _raw_tensor_metadata(layer, "inputs")
    outputs = _raw_tensor_metadata(layer, "outputs")
    if len(raw) < 2 or len(outputs) != 1:
        return None
    logits_shape, logits_dtype = raw[0]
    target_shape, target_dtype = raw[1]
    if (
        len(logits_shape) != 2
        or target_shape != [logits_shape[0]]
        or logits_dtype not in {DType.BFLOAT16, DType.FLOAT32}
        or target_dtype not in {DType.INT32, DType.INT64}
    ):
        return None
    attributes = _attributes(layer)
    reduction = attributes.get("reduction")
    if reduction not in {"mean", "sum"}:
        return None
    weight = attributes.get("weight")
    if weight is not None and weight != "None":
        return None
    label_smoothing = attributes.get("label_smoothing", 0.0)
    if (
        isinstance(label_smoothing, bool)
        or not isinstance(label_smoothing, (int, float))
        or float(label_smoothing) != 0.0
    ):
        return None
    normalized_logits_dtype = (
        TensorDType.BFLOAT16
        if logits_dtype is DType.BFLOAT16
        else TensorDType.FLOAT32
    )
    return CrossEntropyDescriptor(
        rows=logits_shape[0],
        classes=logits_shape[1],
        logits_dtype=normalized_logits_dtype,
        target_dtype=target_dtype,
        reduction=CrossEntropyReduction(str(reduction)),
    )


def decomposed_cross_entropy_descriptor(
    layers: list[tuple[Mapping[str, object], str]],
) -> CrossEntropyDescriptor | None:
    """Parse the admitted log-softmax plus NLL decomposition."""
    if [operation for _layer, operation in layers] != [
        "log_softmax",
        "nll_loss_forward",
    ]:
        return None
    log_softmax, nll_loss = layers[0][0], layers[1][0]
    if softmax_descriptor(
        log_softmax, SoftmaxOperation.LOG_SOFTMAX
    ) is None or not _canonical_nll_loss_arguments(nll_loss):
        return None
    logits = _raw_tensor_metadata(log_softmax, "inputs")
    log_probabilities = _raw_tensor_metadata(log_softmax, "outputs")
    nll_inputs = _raw_tensor_metadata(nll_loss, "inputs")
    nll_outputs = _raw_tensor_metadata(nll_loss, "outputs")
    if (
        len(logits) != 1
        or len(log_probabilities) != 1
        or len(nll_inputs) < 2
        or not nll_outputs
        or nll_inputs[0] != log_probabilities[0]
        or nll_outputs[0][0] != []
    ):
        return None
    logits_shape, logits_dtype = logits[0]
    target_shape, target_dtype = nll_inputs[1]
    if (
        len(logits_shape) != 2
        or target_shape != [logits_shape[0]]
        or logits_dtype not in {DType.BFLOAT16, DType.FLOAT32}
        or target_dtype not in {DType.INT32, DType.INT64}
    ):
        return None
    normalized_logits_dtype = (
        TensorDType.BFLOAT16
        if logits_dtype is DType.BFLOAT16
        else TensorDType.FLOAT32
    )
    return CrossEntropyDescriptor(
        rows=logits_shape[0],
        classes=logits_shape[1],
        logits_dtype=normalized_logits_dtype,
        target_dtype=target_dtype,
        reduction=CrossEntropyReduction.MEAN,
    )


def _canonical_nll_loss_arguments(layer: Mapping[str, object]) -> bool:
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        return False
    arguments = semantic.get("arguments")
    if not isinstance(arguments, list) or len(arguments) < 5:
        return False
    return (
        _unwrap_semantic_value(arguments[2]) is None
        and _unwrap_semantic_value(arguments[3]) == 1
        and _unwrap_semantic_value(arguments[4]) == -100
    )


def indexed_read_descriptor(
    layer: Mapping[str, object],
    operation: IndexedReadOperation,
) -> IndexedReadDescriptor | None:
    """Parse one indexed-read layer."""
    inputs = _raw_tensor_metadata(layer, "inputs")
    if len(inputs) < 2:
        return None
    source, index = inputs[0], inputs[1]
    if operation is IndexedReadOperation.EMBEDDING:
        source, index = inputs[0], inputs[1]
        axis = 0
    else:
        axis = _normalized_axis(
            _attributes(layer).get("dim"),
            len(source[0]),
        )
    if (
        axis is None
        or source[1] not in {DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32}
        or index[1] not in {DType.INT32, DType.INT64}
    ):
        return None
    return IndexedReadDescriptor(
        operation=operation,
        source_shape=source[0],
        index_shape=index[0],
        axis=axis,
        payload_dtype=TensorDType(source[1].value),
        index_dtype=index[1],
        element_bytes=_dtype_bytes(TensorDType(source[1].value)),
    )


def indexed_update_descriptor(
    layer: Mapping[str, object],
    operation: IndexedUpdateOperation,
) -> IndexedUpdateDescriptor | None:
    """Parse one indexed-update layer."""
    inputs = _raw_tensor_metadata(layer, "inputs")
    if len(inputs) < 3:
        return None
    output, index = inputs[0], inputs[1]
    axis = _normalized_axis(_attributes(layer).get("dim"), len(output[0]))
    if (
        axis is None
        or output[1] is not DType.FLOAT32
        or index[1] not in {DType.INT32, DType.INT64}
    ):
        return None
    return IndexedUpdateDescriptor(
        operation=operation,
        output_shape=output[0],
        index_shape=index[0],
        axis=axis,
        payload_dtype=TensorDType.FLOAT32,
        index_dtype=index[1],
        atomic=operation
        in {
            IndexedUpdateOperation.SCATTER_ADD,
            IndexedUpdateOperation.INDEX_ADD,
        },
    )


def primitive_layer_descriptor(
    layer: Mapping[str, object],
    operation: str,
) -> object | None:
    """Dispatch one layer to its primitive descriptor parser."""
    if operation in ELEMENTWISE_TYPES:
        return elementwise_descriptor([(layer, operation)])
    if operation in TRANSPOSE_TYPES:
        return transpose_descriptor(layer)
    if operation in REDUCTION_TYPES:
        return reduction_descriptor(layer, REDUCTION_TYPES[operation])
    if operation in MATMUL_TYPES:
        return matmul_descriptor(layer, batched=operation == "bmm")
    if operation in SOFTMAX_TYPES:
        return softmax_descriptor(layer, SOFTMAX_TYPES[operation])
    if operation == "cross_entropy":
        return cross_entropy_descriptor(layer)
    if operation in INDEXED_READ_TYPES:
        return indexed_read_descriptor(layer, INDEXED_READ_TYPES[operation])
    if operation in INDEXED_UPDATE_TYPES:
        return indexed_update_descriptor(
            layer,
            INDEXED_UPDATE_TYPES[operation],
        )
    return None


def _attributes(layer: Mapping[str, object]) -> Mapping[str, object]:
    value = layer.get("attributes")
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        return {}
    kwargs = semantic.get("kwargs", semantic.get("attributes"))
    result = (
        {
            str(name): _unwrap_semantic_value(item)
            for name, item in kwargs.items()
        }
        if isinstance(kwargs, Mapping)
        else {}
    )
    arguments = semantic.get("arguments", semantic.get("operands"))
    positional = [
        _unwrap_semantic_value(item)
        for item in (arguments if isinstance(arguments, list) else [])
        if not (isinstance(item, Mapping) and "tensor" in item)
    ]
    operation = str(layer.get("type", "")).lower()
    if operation in {
        *SOFTMAX_TYPES,
        *INDEXED_READ_TYPES,
        *INDEXED_UPDATE_TYPES,
    }:
        if "dim" not in result and positional:
            result["dim"] = positional[0]
    elif operation == "layer_norm" and positional:
        result.setdefault("normalized_shape", positional[0])
    return result


def _unwrap_semantic_value(value: object) -> object:
    if isinstance(value, Mapping) and "value" in value:
        mapping = cast("Mapping[str, object]", value)
        return _unwrap_semantic_value(mapping["value"])
    if isinstance(value, list):
        return [_unwrap_semantic_value(item) for item in value]
    return value


def _normalized_axis(value: object, rank: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    normalized = value + rank if value < 0 else value
    return normalized if 0 <= normalized < rank else None


def tensor_names(
    layer: Mapping[str, object],
    direction: str,
) -> list[str]:
    """Return validated tensor names for one layer direction."""
    names = layer.get("tensor_names")
    if not isinstance(names, Mapping):
        return []
    values = names.get(direction)
    return (
        [str(value) for value in values]
        if isinstance(values, list)
        and all(isinstance(value, str) for value in values)
        else []
    )


def _raw_tensor_metadata(
    layer: Mapping[str, object],
    direction: str,
) -> list[tuple[list[int], DType]]:
    shapes = layer.get("tensor_shapes")
    dtypes = layer.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        return []
    raw_shapes = shapes.get(direction)
    raw_dtypes = dtypes.get(direction)
    if not isinstance(raw_shapes, list) or not isinstance(raw_dtypes, list):
        return []
    if len(raw_shapes) != len(raw_dtypes):
        return []
    result: list[tuple[list[int], DType]] = []
    for shape, dtype in zip(raw_shapes, raw_dtypes, strict=True):
        if not (
            isinstance(shape, list)
            and all(
                isinstance(dimension, int) and dimension > 0
                for dimension in shape
            )
        ):
            return []
        try:
            result.append(
                (
                    cast("list[int]", shape),
                    DType(str(dtype).removeprefix("torch.")),
                )
            )
        except ValueError:
            return []
    return result


def _tensor_metadata(
    layer: Mapping[str, object],
    direction: str,
) -> list[tuple[list[int], TensorDType]]:
    shapes = layer.get("tensor_shapes")
    dtypes = layer.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        return []
    raw_shapes = shapes.get(direction)
    raw_dtypes = dtypes.get(direction)
    if not isinstance(raw_shapes, list) or not isinstance(raw_dtypes, list):
        return []
    if len(raw_shapes) != len(raw_dtypes):
        return []
    result: list[tuple[list[int], TensorDType]] = []
    for shape, dtype in zip(raw_shapes, raw_dtypes, strict=True):
        if (
            isinstance(shape, list)
            and shape
            and all(
                isinstance(dimension, int) and dimension > 0
                for dimension in shape
            )
        ):
            try:
                normalized_dtype = str(dtype).removeprefix("torch.")
                result.append(
                    (
                        cast("list[int]", shape),
                        TensorDType(normalized_dtype),
                    )
                )
            except ValueError:
                return []
    return result


def _shape_elements(shape: list[int]) -> int:
    return math.prod(shape)


def _dtype_bytes(dtype: TensorDType) -> int:
    return 4 if dtype is TensorDType.FLOAT32 else 2


__all__ = [
    "ELEMENTWISE_TYPES",
    "INDEXED_READ_TYPES",
    "INDEXED_UPDATE_TYPES",
    "MATMUL_TYPES",
    "REDUCTION_TYPES",
    "SOFTMAX_TYPES",
    "TRANSPOSE_TYPES",
    "cross_entropy_descriptor",
    "decomposed_cross_entropy_descriptor",
    "elementwise_descriptor",
    "indexed_read_descriptor",
    "indexed_update_descriptor",
    "matmul_descriptor",
    "primitive_layer_descriptor",
    "reduction_descriptor",
    "softmax_descriptor",
    "tensor_names",
    "transpose_descriptor",
]
