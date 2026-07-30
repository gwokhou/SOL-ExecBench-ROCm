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

"""Handlers for attention operations.

This module provides einsum handlers for:
- scaled_dot_product_attention (SDPA)
- flex_attention
- multi_head_attention_forward

Based on PyTorch's scaled_dot_product_attention:
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

from typing import Any

from solar.ir.extended_einsum.operations.handlers.base import (
    EinsumOp,
    EinsumOperand,
    EinsumOpHandler,
)
from solar.types import TensorShape, TensorShapes


class ScaledDotProductAttentionHandler(EinsumOpHandler):
    """Handler for scaled dot-product attention.

    This handler can either:
    1. Generate a single fused einsum equation (BHQD,BHKD,BHKV->BHQV)
    2. Expand into a subgraph of operations for detailed analysis

    The expansion follows PyTorch's reference implementation:
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value
    """

    supported_ops = (
        "scaled_dot_product_attention",
        "sdpa",
        "attention",
    )

    # Mark this as an expandable operation
    is_expandable = True

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for scaled dot-product attention.

        Gets shapes from tensor_shapes.inputs (Q, K, V).
        No weights are involved in SDPA.
        """
        if tensor_shapes.num_inputs < 3:
            raise ValueError(
                f"SDPA requires 3 input shapes (Q, K, V). Got {tensor_shapes.num_inputs}",
            )

        query_shape = tensor_shapes.inputs[0]
        key_shape = tensor_shapes.inputs[1]
        value_shape = tensor_shapes.inputs[2]

        is_causal = kwargs.get("is_causal", False)

        return self._generate_sdpa_einsum(
            query_shape,
            key_shape,
            value_shape,
            is_causal,
        )

    def _generate_sdpa_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape,
        is_causal: bool = False,
    ) -> EinsumOp:
        """Generate fused einsum for scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Shape notation (following PyTorch docs):
            Query: (N, ..., Hq, L, E)  -> simplified to (B, H, Q, D)
            Key:   (N, ..., H, S, E)   -> simplified to (B, H, K, D)
            Value: (N, ..., H, S, Ev)  -> simplified to (B, H, K, V)
            Output:(N, ..., Hq, L, Ev) -> simplified to (B, H, Q, V)

        Where:
            B = batch (N and any extra batch dims)
            H = number of heads
            Q = query sequence length (L)
            K = key/value sequence length (S)
            D = embedding dimension (E)
            V = value embedding dimension (Ev)
        """
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]

        # Fused equation: combines Q@K^T and result@V
        equation = "BHQD,BHKD,BHKV->BHQV"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="scaled_dot_product_attention",
            elementwise_op="mul",
            reduction_op="add",
        )


class FlexAttentionHandler(EinsumOpHandler):
    """Handler for flex_attention (similar to SDPA)."""

    supported_ops = ("flex_attention",)

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        if tensor_shapes.num_inputs < 3:
            raise ValueError(
                f"flex_attention requires 3 input shapes (Q, K, V). "
                f"Got {tensor_shapes.num_inputs}",
            )

        query = tensor_shapes.inputs[0]
        key = tensor_shapes.inputs[1]
        value = tensor_shapes.inputs[2]

        return self._generate_flex_attention_einsum(query, key, value)

    def _generate_flex_attention_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape,
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]

        equation = "BHQD,BHKD,BHKV->BHQV"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="flex_attention",
            elementwise_op="mul",
            reduction_op="add",
        )


class MultiHeadAttentionHandler(EinsumOpHandler):
    """Handler for multi-head attention forward."""

    supported_ops = ("multi_head_attention_forward", "multihead_attention")

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        """Generate einsum for multi-head attention."""
        if tensor_shapes.num_inputs < 1:
            raise ValueError(f"Missing Input shape for {op_name}")

        input_shape = tensor_shapes.inputs[0]
        return self._generate_mha_einsum(input_shape)

    def _generate_mha_einsum(self, input_shape: TensorShape) -> EinsumOp:
        """Generate einsum for multi-head attention.

        MHA combines Q, K, V projections with attention computation.
        We represent it as a single attention-like operation.
        """
        operands = [
            EinsumOperand("Input", ["B", "S", "D"], is_output=False),
            EinsumOperand("Output", ["B", "S", "D"], is_output=True),
        ]

        equation = "BSD->BSD"

        return EinsumOp(
            operands=operands,
            equation=equation,
            name="multi_head_attention_forward",
            is_real_einsum=False,  # Composite operation
            elementwise_op="mul",
            reduction_op="add",
        )


__all__ = [
    "FlexAttentionHandler",
    "MultiHeadAttentionHandler",
    "ScaledDotProductAttentionHandler",
]
