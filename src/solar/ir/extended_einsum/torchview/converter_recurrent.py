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

from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)

PathLike = str | Path


@dataclass(frozen=True)
class _RecurrentContext:
    """Normalized graph and tensor metadata for one recurrent expansion."""

    node_id: str
    sequence: int
    batch: int
    input_width: int
    hidden_width: int
    gate_width: int
    activation_shape: list[int]
    input_weight_shape: list[int]
    hidden_weight_shape: list[int]
    final_shape: list[int]
    activation_dtype: Any
    weight_dtype: Any
    output_dtype: Any
    input_connections: list[str]
    output_connections: list[str]
    input_name: str
    hidden_name: str

    @property
    def input_linear_id(self) -> str:
        """Return the input-projection node ID."""
        return f"{self.node_id}.ih_linear"

    @property
    def hidden_linear_id(self) -> str:
        """Return the hidden-projection node ID."""
        return f"{self.node_id}.hh_linear"

    @property
    def gates_id(self) -> str:
        """Return the final gate-combination node ID."""
        return f"{self.node_id}.gates"


class ConverterRecurrentMixin(ConverterMixinContract):
    """Expand recurrent neural-network operations."""

    @staticmethod
    def _recurrent_weight_shapes(
        input_shapes: list[Any],
        input_types: list[str],
        input_width: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Identify input and hidden weight matrices by their inner width."""
        input_weight = None
        hidden_weight = None
        for index, kind in enumerate(input_types):
            if kind != "weight" or index >= len(input_shapes):
                continue
            shape = input_shapes[index]
            if not isinstance(shape, list) or len(shape) != 2:
                continue
            if shape[1] == input_width and input_weight is None:
                input_weight = shape
            elif hidden_weight is None:
                hidden_weight = shape
        return input_weight, hidden_weight

    @staticmethod
    def _recurrent_connections(
        node_id: str,
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Collect canonical recurrent predecessor and successor IDs."""
        inputs = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id not in info.get("consumers", []):
                continue
            start_id = start_node_id_map.get(info["original_id"])
            if start_id and start_id not in inputs:
                inputs.append(start_id)
        return sorted(inputs), sorted(op_graph.successors(node_id))

    def _recurrent_context(
        self,
        operation: str,
        gate_count: int,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> _RecurrentContext:
        """Normalize one GRU/LSTM node into shared expansion metadata."""
        input_shapes = node_data.get("input_shapes") or []
        activation_shape = input_shapes[0] if input_shapes else []
        if len(activation_shape) < 3:
            raise ValueError(
                f"{operation} requires [S,B,I] input. Got: {activation_shape}"
            )
        sequence, batch, input_width = activation_shape[:3]
        input_types = [
            str(value).lower() for value in (node_data.get("input_types") or [])
        ]
        input_weight, hidden_weight = self._recurrent_weight_shapes(
            input_shapes, input_types, input_width
        )
        hidden_width = (
            hidden_weight[1]
            if hidden_weight
            else input_weight[0] // gate_count
            if input_weight
            else input_width
        )
        input_dtypes = node_data.get("input_dtypes") or []
        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (
                input_dtypes[index]
                for index, kind in enumerate(input_types)
                if kind == "weight" and index < len(input_dtypes)
            ),
            activation_dtype,
        )
        outputs = node_data.get("output_shapes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        inputs, successors = self._recurrent_connections(
            node_id, op_graph, start_nodes_info, start_node_id_map
        )
        gate_width = gate_count * hidden_width
        return _RecurrentContext(
            node_id=node_id,
            sequence=sequence,
            batch=batch,
            input_width=input_width,
            hidden_width=hidden_width,
            gate_width=gate_width,
            activation_shape=list(activation_shape),
            input_weight_shape=input_weight or [gate_width, input_width],
            hidden_weight_shape=hidden_weight or [gate_width, hidden_width],
            final_shape=(
                list(outputs[0]) if outputs else [sequence, batch, hidden_width]
            ),
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            input_connections=inputs,
            output_connections=successors,
            input_name=(
                f"{inputs[0]}.Output" if inputs else f"{node_id}.Input"
            ),
            hidden_name=(
                f"{inputs[1]}.Output"
                if len(inputs) > 1
                else f"{node_id}.Hidden"
            ),
        )

    @staticmethod
    def _recurrent_linear_layer(
        context: _RecurrentContext,
        *,
        hidden: bool,
    ) -> dict[str, Any]:
        """Emit one input-to-hidden or hidden-to-hidden projection."""
        node_id = (
            context.hidden_linear_id if hidden else context.input_linear_id
        )
        input_rank = "H" if hidden else "I"
        input_shape = (
            [context.sequence, context.batch, context.hidden_width]
            if hidden
            else context.activation_shape
        )
        weight_shape = (
            context.hidden_weight_shape
            if hidden
            else context.input_weight_shape
        )
        input_name = context.hidden_name if hidden else context.input_name
        connection = (
            context.input_connections[1:2]
            if hidden
            else context.input_connections[:1]
        )
        return {
            "type": "linear",
            "einsum_equation": f"SB{input_rank},G{input_rank}->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    input_name,
                    f"{context.node_id}.Weight_{'hh' if hidden else 'ih'}",
                ],
                "outputs": [f"{node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [input_shape, weight_shape],
                "outputs": [
                    [context.sequence, context.batch, context.gate_width]
                ],
            },
            "tensor_dtypes": {
                "inputs": [context.activation_dtype, context.weight_dtype],
                "outputs": [context.output_dtype],
            },
            "operands": {
                "Input": ["S", "B", input_rank],
                "Weight": ["G", input_rank],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": connection,
                "outputs": [context.gates_id],
            },
        }

    @staticmethod
    def _recurrent_gate_layer(context: _RecurrentContext) -> dict[str, Any]:
        """Emit the final elementwise gate-combination layer."""
        gate_shape = [context.sequence, context.batch, context.gate_width]
        return {
            "type": "sigmoid",
            "einsum_equation": "SBH->SBH",
            "elementwise_op": "sigmoid",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    f"{context.input_linear_id}.Output",
                    f"{context.hidden_linear_id}.Output",
                ],
                "outputs": [f"{context.gates_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [gate_shape, gate_shape],
                "outputs": [context.final_shape],
            },
            "tensor_dtypes": {
                "inputs": [context.output_dtype, context.output_dtype],
                "outputs": [context.output_dtype],
            },
            "operands": {
                "Input": ["S", "B", "H"],
                "Output": ["S", "B", "H"],
            },
            "connections": {
                "inputs": [
                    context.input_linear_id,
                    context.hidden_linear_id,
                ],
                "outputs": context.output_connections,
            },
        }

    def _expand_recurrent(
        self,
        operation: str,
        gate_count: int,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand one recurrent node using the shared GRU/LSTM structure."""
        context = self._recurrent_context(
            operation,
            gate_count,
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
        subgraph = {
            context.input_linear_id: self._recurrent_linear_layer(
                context, hidden=False
            ),
            context.hidden_linear_id: self._recurrent_linear_layer(
                context, hidden=True
            ),
            context.gates_id: self._recurrent_gate_layer(context),
        }
        input_mapping = {0: context.input_linear_id}
        if len(context.input_connections) > 1:
            input_mapping[1] = context.hidden_linear_id
        return subgraph, context.gates_id, input_mapping

    def _expand_lstm(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand an LSTM into two projections and one gate layer."""
        return self._expand_recurrent(
            "LSTM",
            4,
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )

    def _expand_gru(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand a GRU into two projections and one gate layer."""
        return self._expand_recurrent(
            "GRU",
            3,
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
