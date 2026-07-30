# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handlers for reduction operations.

This module provides einsum handlers for:
- sum, mean, prod
- max, min, amax, amin
- argmax, argmin
- logsumexp, norm
"""

import string
from typing import Any

from solar.ir.extended_einsum.operations.handlers.base import (
    EinsumOp,
    EinsumOperand,
    EinsumOpHandler,
)
from solar.types import TensorShape, TensorShapes


class ReductionHandler(EinsumOpHandler):
    """Handler for reduction operations.

    Based on PyTorch documentation:
    https://docs.pytorch.org/docs/stable/nn.functional.html

    All these operations support dim and keepdim parameters.
    """

    supported_ops = (
        # Standard reductions
        "sum",
        "mean",
        "prod",
        # Value reductions
        "max",
        "min",
        "amax",
        "amin",
        # Index reductions (return indices, not values)
        "argmax",
        "argmin",
        # Special reductions
        "logsumexp",
        "norm",
        # Statistical reductions
        "std",
        "var",
        # Boolean reductions
        "all",
        "any",
        # NaN-aware reductions
        "nansum",
        "nanmean",
    )

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for reduction operation."""
        input_shape = (
            tensor_shapes.inputs[0] if tensor_shapes.num_inputs > 0 else None
        )

        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")

        # Get reduction dimensions
        dims = kwargs.get("dims")
        if dims is None:
            dims = kwargs.get("reduce_dims")

        keepdim = bool(kwargs.get("keepdim", False))

        # Normalize op name
        op_type = op_name.lower()
        if op_type in {"amax", "amin"}:
            op_type = op_type[1:]  # amax -> max, amin -> min

        # Pass the observed output shape so the handler can distinguish the
        # binary elementwise overloads of min/max (`torch.min(x, other)`),
        # whose output rank matches the input, from the genuine reduce-all
        # case. Without this, dims=None
        # would unconditionally collapse to a scalar — see kbl_l2/{83,93}.
        out_shape = (
            tensor_shapes.outputs[0] if tensor_shapes.num_outputs > 0 else None
        )
        return self._generate_reduction_einsum(
            input_shape,
            op_type,
            dims,
            keepdim,
            output_shape=out_shape,
        )

    def _generate_reduction_einsum(
        self,
        shape: TensorShape,
        op_type: str = "sum",
        dims: list[int] | None = None,
        keepdim: bool = False,
        output_shape: TensorShape | None = None,
    ) -> EinsumOp:
        """Generate extended-einsum notation for a reduction."""
        ndims = len(shape)
        input_labels = list(string.ascii_uppercase[:ndims])
        normalized_dims = _normalized_dims(dims, ndims)
        output_labels = _reduction_output_labels(
            input_labels,
            op_type=op_type,
            dims=normalized_dims,
            keepdim=keepdim,
            output_shape=output_shape,
        )

        operands = [
            EinsumOperand("Input", input_labels, is_output=False),
            EinsumOperand("Output", output_labels, is_output=True),
        ]

        equation = f"{''.join(input_labels)}->{''.join(output_labels)}"

        # Map reduction op_type to appropriate reduction_op
        reduction_op_map = {
            # Standard reductions
            "sum": "add",
            "mean": "add",  # Mean is sum then divide
            "prod": "mul",
            # Value reductions
            "max": "max",
            "min": "min",
            "amax": "max",
            "amin": "min",
            # Index reductions
            "argmax": "max",
            "argmin": "min",
            # Special reductions
            "logsumexp": "add",
            "norm": "add",
            # Statistical reductions
            "std": "add",  # std involves sum of squared differences
            "var": "add",  # var involves sum of squared differences
            # Boolean reductions
            "all": "and",
            "any": "or",
            # NaN-aware reductions
            "nansum": "add",
            "nanmean": "add",
        }

        # Binary elementwise overload of min/max keeps the input rank — no
        # axis is collapsed, so there's no reduction. Mark it as a plain
        # elementwise op.
        is_binary_elementwise = (
            normalized_dims is None
            and op_type in {"min", "max"}
            and len(output_labels) == ndims
        )
        if is_binary_elementwise:
            elementwise_op = op_type
            reduction_op = "none"
        else:
            elementwise_op = "copy"
            reduction_op = reduction_op_map.get(op_type, "add")

        return EinsumOp(
            operands=operands,
            equation=equation,
            name=op_type,
            is_real_einsum=False,
            elementwise_op=elementwise_op,
            reduction_op=reduction_op,
        )


def _normalized_dims(
    dims: list[int] | None,
    rank: int,
) -> list[int] | None:
    if dims is None:
        return None
    return [
        dimension if dimension >= 0 else rank + dimension for dimension in dims
    ]


def _reduction_output_labels(
    input_labels: list[str],
    *,
    op_type: str,
    dims: list[int] | None,
    keepdim: bool,
    output_shape: TensorShape | None,
) -> list[str]:
    if dims is None:
        rank_preserving = (
            output_shape is not None
            and len(output_shape) == len(input_labels)
            and op_type in {"min", "max"}
        )
        return input_labels.copy() if rank_preserving or keepdim else []
    if keepdim:
        return input_labels.copy()
    return [
        label for index, label in enumerate(input_labels) if index not in dims
    ]


__all__ = ["ReductionHandler"]
