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

"""Handlers for matrix multiplication operations.

This module provides einsum handlers for:
- matmul (matrix multiplication)
- linear (fully connected layer)
- bmm (batch matrix multiplication)
- mm (2D matrix multiplication)
"""

from typing import Any

from solar.ir.extended_einsum.operations.handlers.base import (
    EinsumOp,
    EinsumOperand,
    EinsumOpHandler,
)
from solar.ir.extended_einsum.operations.handlers.registry import (
    get_global_registry,
)
from solar.types import TensorShape, TensorShapes


class MatmulHandler(EinsumOpHandler):
    """Handler for matmul operations."""

    supported_ops = ("matmul", "mm")

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = (
            tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        )

        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")

        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]

        return self._generate_matmul_einsum(input_shape, weight_shape)

    def _generate_matmul_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape,
    ) -> EinsumOp:
        """Generate einsum labels for every PyTorch matmul rank combination."""
        input_rank = len(input_shape)
        other_rank = len(other_shape)
        input_batch_rank = max(0, input_rank - 2)
        other_batch_rank = max(0, other_rank - 2)
        output_batch_rank = max(input_batch_rank, other_batch_rank)
        output_batch = [f"B{index}" for index in range(output_batch_rank)]
        input_batch = (
            output_batch[-input_batch_rank:] if input_batch_rank else []
        )
        other_batch = (
            output_batch[-other_batch_rank:] if other_batch_rank else []
        )
        input_dims = input_batch + (["K"] if input_rank == 1 else ["M", "K"])
        other_dims = other_batch + (["K"] if other_rank == 1 else ["K", "N"])
        output_dims = [
            *output_batch,
            *(["M"] if input_rank > 1 else []),
            *(["N"] if other_rank > 1 else []),
        ]
        operands = [
            EinsumOperand("Input", input_dims, is_output=False),
            EinsumOperand("Weight", other_dims, is_output=False),
            EinsumOperand("Output", output_dims, is_output=True),
        ]
        return EinsumOp(
            operands=operands,
            equation=(
                f"{''.join(input_dims)},{''.join(other_dims)}"
                f"->{''.join(output_dims)}"
            ),
            name="matmul",
            elementwise_op="mul",
            reduction_op="add",
        )


class LinearHandler(EinsumOpHandler):
    """Handler for linear (fully connected) layers."""

    supported_ops = ("linear",)

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for linear layer."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = (
            tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        )

        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")

        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]

        return self._generate_linear_einsum(input_shape, weight_shape)

    def _generate_linear_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
    ) -> EinsumOp:
        """Generate einsum for linear layer.

        Args:
            input_shape: Shape of input tensor.
            weight_shape: Shape of weight tensor.

        Returns:
            EinsumOp for the linear operation.

        """
        # Linear: input @ weight.T
        batch_dims = len(input_shape) - 1
        batch_letters = [f"B{i}" for i in range(batch_dims)]

        input_dims = batch_letters + ["K"]
        weight_dims = ["N", "K"]  # Weight is [out_features, in_features]
        output_dims = batch_letters + ["N"]

        operands = [
            EinsumOperand("Input", input_dims, is_output=False),
            EinsumOperand("Weight", weight_dims, is_output=False),
            EinsumOperand("Output", output_dims, is_output=True),
        ]

        input_str = "".join(input_dims)
        weight_str = "".join(weight_dims)
        output_str = "".join(output_dims)
        equation = f"{input_str},{weight_str}->{output_str}"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="linear",
            elementwise_op="mul",
            reduction_op="add",
        )


class BmmHandler(EinsumOpHandler):
    """Handler for batch matrix multiplication."""

    supported_ops = ("bmm",)

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication."""
        input_shape = tensor_shapes.inputs[0]
        weight_shape = (
            tensor_shapes.inputs[1] if tensor_shapes.num_inputs > 1 else None
        )

        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")

        return self._generate_bmm_einsum(input_shape, weight_shape)

    def _generate_bmm_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape,
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication.

        bmm: [B, M, K] x [B, K, N] -> [B, M, N]
        """
        operands = [
            EinsumOperand("Input", ["B", "M", "K"], is_output=False),
            EinsumOperand("Weight", ["B", "K", "N"], is_output=False),
            EinsumOperand("Output", ["B", "M", "N"], is_output=True),
        ]

        equation = "BMK,BKN->BMN"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="bmm",
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(MatmulHandler)
_registry.register_handler(LinearHandler)
_registry.register_handler(BmmHandler)


__all__ = ["BmmHandler", "LinearHandler", "MatmulHandler"]
