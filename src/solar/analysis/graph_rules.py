# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Declarative operation sets used by graph accounting stages."""

BOOL_DTYPES = frozenset({"torch.bool", "bool"})

RECOMPUTABLE_OPERAND_TARGETS = frozenset(
    {
        "abs",
        "amax",
        "clamp",
        "contiguous",
        "detach",
        "div",
        "expand",
        "flatten",
        "mul",
        "permute",
        "reshape",
        "squeeze",
        "to",
        "transpose",
        "unsqueeze",
        "view",
    }
)

LOW_PRECISION_DEQUANT_DTYPES = frozenset(
    {
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "int8",
        "int4",
    }
)

TRANSPARENT_OPS = frozenset(
    {
        "expand",
        "expand_as",
        "view",
        "reshape",
        "contiguous",
        "transpose",
        "permute",
        "t",
        "unsqueeze",
        "squeeze",
        "flatten",
        "unfold",
        "unflatten",
        "chunk",
        "split",
        "tensor_split",
    }
)

QUANTIZED_PAYLOAD_PASSTHROUGH = frozenset(
    {
        "contiguous",
        "detach",
        "expand",
        "flatten",
        "permute",
        "reshape",
        "squeeze",
        "transpose",
        "unsqueeze",
        "view",
    }
)

ZERO_COMPUTE_OPS = frozenset(
    {
        "embedding",
        "embedding_bag",
        "expand",
        "expand_as",
        "view",
        "reshape",
        "contiguous",
        "transpose",
        "permute",
        "t",
        "unsqueeze",
        "squeeze",
        "flatten",
        "unfold",
        "unflatten",
        "__getitem__",
        "narrow",
        "slice",
        "select",
        "__setitem__",
        "scatter",
        "scatter_",
        "index_copy",
        "index_copy_",
        "index_put",
        "index_put_",
        "cat",
        "concat",
        "stack",
        "chunk",
        "split",
        "tensor_split",
        "repeat",
        "repeat_interleave",
        "tile",
        "roll",
        "flip",
        "pad",
        "constant_pad_nd",
        "clone",
        "copy_",
        "to",
        "type",
        "type_as",
        "float",
        "half",
        "bfloat16",
        "int",
    }
)

ZERO_COPY_VIEW_OPS = TRANSPARENT_OPS
SLICE_VIEW_OPS = frozenset({"__getitem__", "narrow", "slice", "select"})
SCATTER_OPS = frozenset(
    {
        "__setitem__",
        "scatter",
        "scatter_",
        "index_copy",
        "index_copy_",
        "index_put",
        "index_put_",
    }
)

__all__ = [
    "BOOL_DTYPES",
    "LOW_PRECISION_DEQUANT_DTYPES",
    "QUANTIZED_PAYLOAD_PASSTHROUGH",
    "RECOMPUTABLE_OPERAND_TARGETS",
    "SCATTER_OPS",
    "SLICE_VIEW_OPS",
    "TRANSPARENT_OPS",
    "ZERO_COMPUTE_OPS",
    "ZERO_COPY_VIEW_OPS",
]
