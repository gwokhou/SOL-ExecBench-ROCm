# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Torchview metadata fields accepted by the reviewed extractor."""

MODULE_ATTR_NAMES = (
    "module",
    "pytorch_module",
    "op",
    "operation",
    "target",
    "_module",
    "wrapped_module",
)

GEOMETRIC_ATTRS = frozenset(
    {
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "output_padding",
        "normalized_shape",
        "output_size",
    },
)

BOOLEAN_ATTRS = frozenset(
    {
        "inplace",
        "affine",
        "elementwise_affine",
        "track_running_stats",
        "ceil_mode",
        "count_include_pad",
        "return_indices",
        "sparse",
    },
)

__all__ = ["BOOLEAN_ATTRS", "GEOMETRIC_ATTRS", "MODULE_ATTR_NAMES"]
