# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reviewed canonical operation families for AMD resource accounting."""

_VIEW_OPS = frozenset(
    {
        "alias",
        "detach",
        "diagonal",
        "expand",
        "flatten",
        "getitem",
        "identity",
        "narrow",
        "permute",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "transpose",
        "unsqueeze",
        "view",
    },
)
_MEMORY_ONLY_OPS = frozenset(
    {
        "cat",
        "chunk",
        "clone",
        "contiguous",
        "copy",
        "copy_",
        "empty",
        "empty_like",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "pad",
        "repeat",
        "repeat_interleave",
        "scalar_tensor",
        "split",
        "stack",
        "tensor_split",
        "vstack",
        "zeros",
        "zeros_like",
    },
)
_MFMA_OPS = frozenset(
    {
        "addmm",
        "bmm",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "linear",
        "matmul",
        "mm",
    },
)
_SFU_OPS = frozenset(
    {
        "cos",
        "exp",
        "exp2",
        "floor",
        "log",
        "log2",
        "pow",
        "rsqrt",
        "sin",
        "sqrt",
        "tanh",
    },
)
_LOSS_OPS = frozenset(
    {
        "cross_entropy",
        "cross_entropy_loss",
        "kl_div",
        "nll_loss",
        "nll_loss_forward",
        "xlogy",
    },
)
_COMPOSITE_SFU_OPS = frozenset(
    {
        "elu",
        "gelu",
        "hardsigmoid",
        "hardswish",
        "mish",
        "sigmoid",
        "silu",
        "softplus",
    },
)
_REDUCTION_OPS = frozenset(
    {
        "all",
        "amax",
        "amin",
        "any",
        "argmax",
        "argmin",
        "logsumexp",
        "mean",
        "prod",
        "sum",
    }
)
_VARIANCE_OPS = frozenset({"std", "std_mean", "var", "var_mean"})
_NORMALIZATION_OPS = frozenset(
    {
        "batch_norm",
        "group_norm",
        "layer_norm",
        "log_softmax",
        "rms_norm",
        "softmax",
    },
)
_ATOMIC_OPS = frozenset(
    {
        "__setitem__",
        "index_add",
        "index_copy",
        "index_put",
        "scatter",
        "scatter_add",
    },
)
_SCAN_SORT_OPS = frozenset(
    {"argsort", "cummax", "cummin", "cumprod", "cumsum", "sort", "topk"},
)
_CONVERSION_OPS = frozenset(
    {
        "bfloat16",
        "dequantize",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "float",
        "half",
        "int",
        "long",
        "quantize_per_channel",
        "quantize_per_tensor",
        "to",
        "type",
        "type_as",
    },
)
_INDEX_OPS = frozenset(
    {"embedding", "embedding_bag", "gather", "index_select", "tril", "triu"},
)
_VALU_OPS = frozenset(
    {
        "abs",
        "add",
        "bitwise_and",
        "bitwise_not",
        "clamp",
        "div",
        "eq",
        "ge",
        "gt",
        "le",
        "leaky_relu",
        "lt",
        "maximum",
        "masked_fill",
        "minimum",
        "mul",
        "ne",
        "neg",
        "ones_like",
        "relu",
        "square",
        "sub",
        "where",
        "zeros_like",
    },
)
