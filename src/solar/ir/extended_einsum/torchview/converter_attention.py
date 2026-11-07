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

"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        tensor_names: {inputs: [...], outputs: [...]}
        tensor_shapes: {inputs: [...], outputs: [...]}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from solar.composition import BoundComponent

PathLike = str | Path


@dataclass(frozen=True, slots=True, kw_only=True)
class _AttentionCore:
    """Normalized QK-scale-softmax-AV expansion metadata."""

    node_id: str
    query_name: str
    key_name: str
    value_name: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    scores_shape: list[int]
    output_shape: list[int]
    qk_equation: str
    qk_weight_role: str
    qk_weight_dims: list[str]
    av_weight_role: str
    activation_dtype: Any
    output_dtype: Any
    qk_connections: list[str]
    value_connections: list[str]
    output_connections: list[str]
    scale_factor: str | None = None
    softmax_dimension: int | None = None

    @property
    def qk_id(self) -> str:
        """Return the query-key matmul node ID."""
        return f"{self.node_id}.qk_matmul"

    @property
    def scale_id(self) -> str:
        """Return the scale node ID."""
        return f"{self.node_id}.scale"

    @property
    def softmax_id(self) -> str:
        """Return the softmax node ID."""
        return f"{self.node_id}.softmax"

    @property
    def av_id(self) -> str:
        """Return the attention-value matmul node ID."""
        return f"{self.node_id}.av_matmul"


@dataclass(frozen=True, slots=True, kw_only=True)
class _MHAContext:
    """Projection and shared-core metadata for one MHA expansion."""

    core: _AttentionCore
    sequence: int
    batch: int
    dimension: int
    activation_name: str
    activation_dtype: Any
    weight_dtype: Any
    output_dtype: Any
    input_weight_shape: list[int] | None
    output_weight_shape: list[int] | None
    input_connections: list[str]
    output_connections: list[str]

    @property
    def input_projection_id(self) -> str:
        """Return the input projection node ID."""
        return f"{self.core.node_id}.in_proj"

    @property
    def output_projection_id(self) -> str:
        """Return the output projection node ID."""
        return f"{self.core.node_id}.out_proj"


def _attention_qk_layer(context: _AttentionCore) -> dict[str, Any]:
    """Emit the query-key contraction layer."""
    return {
        "type": "matmul",
        "einsum_equation": context.qk_equation,
        "elementwise_op": "mul",
        "reduction_op": "add",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "tensor_names": {
            "inputs": [context.query_name, context.key_name],
            "outputs": [f"{context.qk_id}.Output"],
        },
        "tensor_types": {
            "inputs": ["input", "input"],
            "outputs": ["output"],
        },
        "tensor_shapes": {
            "inputs": [context.query_shape, context.key_shape],
            "outputs": [context.scores_shape],
        },
        "operands": {
            "Input": ["B", "H", "Q", "D"],
            context.qk_weight_role: context.qk_weight_dims,
            "Output": ["B", "H", "Q", "K"],
        },
        "tensor_dtypes": {
            "inputs": [context.activation_dtype, context.activation_dtype],
            "outputs": [context.activation_dtype],
        },
        "connections": {
            "inputs": context.qk_connections,
            "outputs": [context.scale_id],
        },
    }


def _attention_unary_layer(
    context: _AttentionCore,
    *,
    softmax: bool,
) -> dict[str, Any]:
    """Emit the scale or softmax layer over attention scores."""
    node_id = context.softmax_id if softmax else context.scale_id
    predecessor = context.scale_id if softmax else context.qk_id
    successor = context.av_id if softmax else context.softmax_id
    operation = "softmax" if softmax else "mul"
    layer: dict[str, Any] = {
        "type": operation,
        "einsum_equation": "BHQK->BHQK",
        "elementwise_op": operation,
        "reduction_op": "none",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {
            "inputs": [f"{predecessor}.Output"],
            "outputs": [f"{node_id}.Output"],
        },
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "tensor_shapes": {
            "inputs": [context.scores_shape],
            "outputs": [context.scores_shape],
        },
        "operands": {
            "Input": ["B", "H", "Q", "K"],
            "Output": ["B", "H", "Q", "K"],
        },
        "tensor_dtypes": {
            "inputs": [context.activation_dtype],
            "outputs": [context.activation_dtype],
        },
        "connections": {"inputs": [predecessor], "outputs": [successor]},
    }
    if softmax and context.softmax_dimension is not None:
        layer["additional_info"] = {"dim": context.softmax_dimension}
    elif not softmax and context.scale_factor is not None:
        layer["additional_info"] = {"scale_factor": context.scale_factor}
    return layer


def _attention_av_layer(context: _AttentionCore) -> dict[str, Any]:
    """Emit the attention-value contraction layer."""
    return {
        "type": "matmul",
        "einsum_equation": "BHQK,BHKV->BHQV",
        "elementwise_op": "mul",
        "reduction_op": "add",
        "is_real_einsum": True,
        "is_einsum_supportable": True,
        "tensor_names": {
            "inputs": [f"{context.softmax_id}.Output", context.value_name],
            "outputs": [f"{context.av_id}.Output"],
        },
        "tensor_types": {
            "inputs": ["input", "input"],
            "outputs": ["output"],
        },
        "tensor_shapes": {
            "inputs": [context.scores_shape, context.value_shape],
            "outputs": [context.output_shape],
        },
        "operands": {
            "Input": ["B", "H", "Q", "K"],
            context.av_weight_role: ["B", "H", "K", "V"],
            "Output": ["B", "H", "Q", "V"],
        },
        "tensor_dtypes": {
            "inputs": [context.activation_dtype, context.activation_dtype],
            "outputs": [context.output_dtype],
        },
        "connections": {
            "inputs": [context.softmax_id, *context.value_connections],
            "outputs": context.output_connections,
        },
    }


def _attention_core_layers(
    context: _AttentionCore,
) -> dict[str, dict[str, Any]]:
    """Emit the four shared scaled-dot-product attention layers."""
    return {
        context.qk_id: _attention_qk_layer(context),
        context.scale_id: _attention_unary_layer(context, softmax=False),
        context.softmax_id: _attention_unary_layer(context, softmax=True),
        context.av_id: _attention_av_layer(context),
    }


class AttentionOperationConverter(BoundComponent):
    """Expand attention and groupwise-convolution operations."""

    def _should_expand_mha(self, node_data: dict[str, Any]) -> bool:
        """Check if this is a multi_head_attention_forward that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {
            "multi_head_attention_forward",
            "multihead_attention",
        }

    def _should_expand_sdpa(self, node_data: dict[str, Any]) -> bool:
        """Check if this is a scaled_dot_product_attention that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return not self._strict and node_type in {
            "scaled_dot_product_attention",
            "sdpa",
            "attention",
        }

    def _should_expand_lstm(self, node_data: dict[str, Any]) -> bool:
        """Check if this is an LSTM that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type == "lstm"

    def _should_expand_gru(self, node_data: dict[str, Any]) -> bool:
        """Check if this is a GRU that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type == "gru"

    @staticmethod
    def _attention_connections(
        node_id: str,
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Collect canonical attention predecessor and successor IDs."""
        inputs = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id not in info.get("consumers", []):
                continue
            start_id = start_node_id_map.get(info["original_id"])
            if start_id and start_id not in inputs:
                inputs.append(start_id)
        return sorted(inputs), sorted(op_graph.successors(node_id))

    def _sdpa_context(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> _AttentionCore:
        """Normalize one native scaled-dot-product attention node."""
        input_shapes = node_data.get("input_shapes") or []
        if len(input_shapes) < 3:
            raise ValueError(
                f"SDPA requires 3 inputs (Q, K, V). Got: {input_shapes}"
            )
        query = list(input_shapes[0])
        key = list(input_shapes[1])
        value = list(input_shapes[2])
        batch, heads, query_length, dimension = query[:4]
        key_length = key[2]
        value_width = value[3]
        outputs = node_data.get("output_shapes") or []
        output_shape = (
            list(outputs[0])
            if outputs
            else [batch, heads, query_length, value_width]
        )
        input_dtypes = node_data.get("input_dtypes") or []
        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        output_dtypes = node_data.get("output_dtypes") or []
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        inputs, successors = self._attention_connections(
            node_id, op_graph, start_nodes_info, start_node_id_map
        )
        return _AttentionCore(
            node_id=node_id,
            query_name=(
                f"{inputs[0]}.Output" if inputs else f"{node_id}.Query"
            ),
            key_name=(
                f"{inputs[1]}.Output" if len(inputs) > 1 else f"{node_id}.Key"
            ),
            value_name=(
                f"{inputs[2]}.Output" if len(inputs) > 2 else f"{node_id}.Value"
            ),
            query_shape=query,
            key_shape=key,
            value_shape=value,
            scores_shape=[batch, heads, query_length, key_length],
            output_shape=output_shape,
            qk_equation="BHQD,BHKD->BHQK",
            qk_weight_role="Weight",
            qk_weight_dims=["B", "H", "K", "D"],
            av_weight_role="Weight",
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            qk_connections=inputs[:2],
            value_connections=inputs[2:3],
            output_connections=successors,
            scale_factor=f"1/sqrt({dimension})",
            softmax_dimension=-1,
        )

    def _expand_sdpa(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand native scaled-dot-product attention into four layers."""
        context = self._sdpa_context(
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
        return (
            _attention_core_layers(context),
            context.av_id,
            {0: context.qk_id, 1: context.qk_id, 2: context.av_id},
        )

    @staticmethod
    def _as_list(value: Any, default: list[int]) -> list[Any]:
        """Normalize scalar/list convolution args to a list."""
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _mha_weight_shapes(
        input_shapes: list[Any],
        input_types: list[str],
        dimension: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Identify the input and output projection weight matrices."""
        input_weight = None
        output_weight = None
        for index, kind in enumerate(input_types):
            if kind != "weight" or index >= len(input_shapes):
                continue
            shape = input_shapes[index]
            if not isinstance(shape, list) or len(shape) != 2:
                continue
            if shape == [3 * dimension, dimension]:
                input_weight = shape
            elif shape == [dimension, dimension]:
                output_weight = shape
        return input_weight, output_weight

    @staticmethod
    def _mha_dtypes(
        node_data: dict[str, Any],
        input_types: list[str],
    ) -> tuple[Any, Any, Any]:
        """Return activation, weight, and output dtypes for MHA."""
        inputs = node_data.get("input_dtypes") or []
        activation = inputs[0] if inputs else "torch.float32"
        weight = next(
            (
                inputs[index]
                for index, kind in enumerate(input_types)
                if kind == "weight" and index < len(inputs)
            ),
            activation,
        )
        outputs = node_data.get("output_dtypes") or []
        return activation, weight, outputs[0] if outputs else activation

    def _mha_context(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> _MHAContext:
        """Normalize one multi-head-attention node for expansion."""
        input_shapes = node_data.get("input_shapes") or []
        activation_shape = input_shapes[0] if input_shapes else []
        if len(activation_shape) < 3:
            raise ValueError(
                f"MHA requires [S,B,D] input. Got: {activation_shape}"
            )
        sequence, batch, dimension = activation_shape[:3]
        input_types = [
            str(value).lower() for value in (node_data.get("input_types") or [])
        ]
        input_weight, output_weight = self._mha_weight_shapes(
            input_shapes, input_types, dimension
        )
        activation_dtype, weight_dtype, output_dtype = self._mha_dtypes(
            node_data, input_types
        )
        inputs, outputs = self._attention_connections(
            node_id, op_graph, start_nodes_info, start_node_id_map
        )
        activation_name = (
            f"{inputs[0]}.Output" if inputs else f"{node_id}.Input"
        )
        projected_name = (
            f"{node_id}.in_proj.Output" if input_weight else activation_name
        )
        core = _AttentionCore(
            node_id=node_id,
            query_name=projected_name,
            key_name=projected_name,
            value_name=projected_name,
            query_shape=[batch, 1, sequence, dimension],
            key_shape=[batch, 1, dimension, sequence],
            value_shape=[batch, 1, sequence, dimension],
            scores_shape=[batch, 1, sequence, sequence],
            output_shape=[batch, 1, sequence, dimension],
            qk_equation="BHQD,BHDK->BHQK",
            qk_weight_role="Input_1",
            qk_weight_dims=["B", "H", "D", "K"],
            av_weight_role="Input_1",
            activation_dtype=activation_dtype,
            output_dtype=activation_dtype,
            qk_connections=[f"{node_id}.in_proj"]
            if input_weight
            else inputs[:1],
            value_connections=[],
            output_connections=(
                [f"{node_id}.out_proj"] if output_weight else outputs
            ),
        )
        return _MHAContext(
            core=core,
            sequence=sequence,
            batch=batch,
            dimension=dimension,
            activation_name=activation_name,
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            input_weight_shape=input_weight,
            output_weight_shape=output_weight,
            input_connections=inputs,
            output_connections=outputs,
        )

    @staticmethod
    def _mha_projection_layer(
        context: _MHAContext,
        *,
        output: bool,
    ) -> dict[str, Any]:
        """Emit the input or output linear projection of MHA."""
        node_id = (
            context.output_projection_id
            if output
            else context.input_projection_id
        )
        weight_shape = (
            context.output_weight_shape
            if output
            else context.input_weight_shape
        )
        output_width = context.dimension if output else 3 * context.dimension
        input_name = (
            f"{context.core.av_id}.Output"
            if output
            else context.activation_name
        )
        return {
            "type": "linear",
            "einsum_equation": "MK,NK->MN",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": {
                "Input": ["M", "K"],
                "Weight": ["N", "K"],
                "Output": ["M", "N"],
            },
            "tensor_names": {
                "inputs": [
                    input_name,
                    f"{node_id}.Weight",
                ],
                "outputs": [f"{node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [
                    [context.sequence * context.batch, context.dimension],
                    weight_shape,
                ],
                "outputs": [[context.sequence * context.batch, output_width]],
            },
            "tensor_dtypes": {
                "inputs": [context.activation_dtype, context.weight_dtype],
                "outputs": [
                    context.output_dtype if output else context.activation_dtype
                ],
            },
            "connections": {
                "inputs": (
                    [context.core.av_id]
                    if output
                    else context.input_connections[:1]
                ),
                "outputs": (
                    context.output_connections
                    if output
                    else [context.core.qk_id]
                ),
            },
        }

    def _expand_mha(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand MHA into optional projections and a shared SDPA core."""
        context = self._mha_context(
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
        subgraph: dict[str, dict[str, Any]] = {}
        if context.input_weight_shape:
            subgraph[context.input_projection_id] = self._mha_projection_layer(
                context, output=False
            )
        subgraph.update(_attention_core_layers(context.core))
        final_node_id = context.core.av_id
        if context.output_weight_shape:
            subgraph[context.output_projection_id] = self._mha_projection_layer(
                context, output=True
            )
            final_node_id = context.output_projection_id
        input_mapping = {0: context.input_projection_id}
        if len(context.input_connections) > 1:
            input_mapping[1] = context.input_projection_id
        if len(context.input_connections) > 2:
            input_mapping[2] = context.input_projection_id
        return subgraph, final_node_id, input_mapping
