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

from pathlib import Path
from typing import Any

import networkx as nx

from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)

PathLike = str | Path


class ConverterRecurrentMixin(ConverterMixinContract):
    """Expand recurrent neural-network operations."""

    def _expand_lstm(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand LSTM into a subgraph of linear operations.

        LSTM decomposes into (per timestep, summed over S steps):
          1. ih_linear: input @ W_ih^T   [S*B, I] @ [4H, I] -> [S*B, 4H]
          2. hh_linear: hidden @ W_hh^T  [S*B, H] @ [4H, H] -> [S*B, 4H]
          3. gate ops (sigmoid, tanh) — elementwise, not real einsums
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_types = [
            str(t).lower() for t in (node_data.get("input_types") or [])
        ]
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (
                input_dtypes[i]
                for i, t in enumerate(input_types)
                if t == "weight" and i < len(input_dtypes)
            ),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"LSTM requires [S,B,I] input. Got: {act_shape}")
        S, B, I = act_shape[0], act_shape[1], act_shape[2]  # noqa: N806

        # Find weight shapes
        w_ih_shape = None  # [4H, I]
        w_hh_shape = None  # [4H, H]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[1] == I and w_ih_shape is None:
                        w_ih_shape = ws
                    elif w_hh_shape is None:
                        w_hh_shape = ws

        H = (  # noqa: N806 - matches the einsum rank label
            w_hh_shape[1]
            if w_hh_shape
            else (w_ih_shape[0] // 4 if w_ih_shape else I)
        )

        input_connections = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(op_graph.successors(node_id))

        subgraph: dict[str, dict[str, Any]] = {}

        ih_id = f"{node_id}.ih_linear"
        hh_id = f"{node_id}.hh_linear"
        gates_id = f"{node_id}.gates"

        input_mapping: dict[int, str] = {0: ih_id}
        if len(input_connections) > 1:
            input_mapping[1] = hh_id

        act_input_name = (
            f"{input_connections[0]}.Output"
            if input_connections
            else f"{node_id}.Input"
        )
        hidden_input_name = (
            f"{input_connections[1]}.Output"
            if len(input_connections) > 1
            else f"{node_id}.Hidden"
        )

        # Hidden state shape — h0 is [num_layers*num_directions, B, H].
        # Use the shape from input_shapes[1] if available.
        input_shapes[1] if len(input_shapes) > 1 else [1, B, H]

        G = 4 * H  # noqa: N806 - matches the einsum rank label

        # 1. ih_linear: input @ W_ih^T — [S, B, I] @ [G, I] -> [S, B, G]
        ih_output_shape = [S, B, G]
        subgraph[ih_id] = {
            "type": "linear",
            "einsum_equation": "SBI,GI->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [act_input_name, f"{node_id}.Weight_ih"],
                "outputs": [f"{ih_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(act_shape), w_ih_shape or [G, I]],
                "outputs": [ih_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "I"],
                "Weight": ["G", "I"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[:1],
                "outputs": [gates_id],
            },
        }

        # 2. hh_linear: hidden @ W_hh^T
        # The hidden projection runs once per timestep but we represent the
        # total work as [S, B, H] @ [G, H] -> [S, B, G] so MACs reflect
        # S steps of B×H×G multiplies.
        hh_input_shape = [S, B, H]
        hh_output_shape = [S, B, G]
        subgraph[hh_id] = {
            "type": "linear",
            "einsum_equation": "SBH,GH->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [hidden_input_name, f"{node_id}.Weight_hh"],
                "outputs": [f"{hh_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [hh_input_shape, w_hh_shape or [G, H]],
                "outputs": [hh_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "H"],
                "Weight": ["G", "H"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[1:2]
                if len(input_connections) > 1
                else [],
                "outputs": [gates_id],
            },
        }

        # 3. Gate ops (sigmoid/tanh) — elementwise, combines ih + hh results
        final_shape = list(output_shapes[0]) if output_shapes else [S, B, H]
        subgraph[gates_id] = {
            "type": "sigmoid",
            "einsum_equation": "SBH->SBH",
            "elementwise_op": "sigmoid",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{ih_id}.Output", f"{hh_id}.Output"],
                "outputs": [f"{gates_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [ih_output_shape, hh_output_shape],
                "outputs": [final_shape],
            },
            "tensor_dtypes": {
                "inputs": [out_dtype, out_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "H"],
                "Output": ["S", "B", "H"],
            },
            "connections": {
                "inputs": [ih_id, hh_id],
                "outputs": output_connections,
            },
        }

        final_node_id = gates_id
        return subgraph, final_node_id, input_mapping

    def _expand_gru(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand GRU into a subgraph of linear operations.

        GRU decomposes into (per timestep, summed over S steps):
          1. ih_linear: input @ W_ih^T   [S*B, I] @ [3H, I] -> [S*B, 3H]
          2. hh_linear: hidden @ W_hh^T  [S*B, H] @ [3H, H] -> [S*B, 3H]
          3. gate ops (sigmoid, tanh) — elementwise, not real einsums
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_types = [
            str(t).lower() for t in (node_data.get("input_types") or [])
        ]
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (
                input_dtypes[i]
                for i, t in enumerate(input_types)
                if t == "weight" and i < len(input_dtypes)
            ),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"GRU requires [S,B,I] input. Got: {act_shape}")
        S, B, I = act_shape[0], act_shape[1], act_shape[2]  # noqa: N806

        w_ih_shape = None  # [3H, I]
        w_hh_shape = None  # [3H, H]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[1] == I and w_ih_shape is None:
                        w_ih_shape = ws
                    elif w_hh_shape is None:
                        w_hh_shape = ws

        H = (  # noqa: N806 - matches the einsum rank label
            w_hh_shape[1]
            if w_hh_shape
            else (w_ih_shape[0] // 3 if w_ih_shape else I)
        )

        input_connections = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(op_graph.successors(node_id))

        subgraph: dict[str, dict[str, Any]] = {}

        ih_id = f"{node_id}.ih_linear"
        hh_id = f"{node_id}.hh_linear"
        gates_id = f"{node_id}.gates"

        input_mapping: dict[int, str] = {0: ih_id}
        if len(input_connections) > 1:
            input_mapping[1] = hh_id

        act_input_name = (
            f"{input_connections[0]}.Output"
            if input_connections
            else f"{node_id}.Input"
        )
        hidden_input_name = (
            f"{input_connections[1]}.Output"
            if len(input_connections) > 1
            else f"{node_id}.Hidden"
        )

        input_shapes[1] if len(input_shapes) > 1 else [1, B, H]

        G = 3 * H  # noqa: N806 - matches the einsum rank label

        # 1. ih_linear: input @ W_ih^T — [S, B, I] @ [G, I] -> [S, B, G]
        ih_output_shape = [S, B, G]
        subgraph[ih_id] = {
            "type": "linear",
            "einsum_equation": "SBI,GI->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [act_input_name, f"{node_id}.Weight_ih"],
                "outputs": [f"{ih_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(act_shape), w_ih_shape or [G, I]],
                "outputs": [ih_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "I"],
                "Weight": ["G", "I"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[:1],
                "outputs": [gates_id],
            },
        }

        # 2. hh_linear: hidden @ W_hh^T
        # The hidden projection runs once per timestep but we represent the
        # total work as [S, B, H] @ [G, H] -> [S, B, G] so MACs reflect
        # S steps of B×H×G multiplies.
        hh_input_shape = [S, B, H]
        hh_output_shape = [S, B, G]
        subgraph[hh_id] = {
            "type": "linear",
            "einsum_equation": "SBH,GH->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [hidden_input_name, f"{node_id}.Weight_hh"],
                "outputs": [f"{hh_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [hh_input_shape, w_hh_shape or [G, H]],
                "outputs": [hh_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "H"],
                "Weight": ["G", "H"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[1:2]
                if len(input_connections) > 1
                else [],
                "outputs": [gates_id],
            },
        }

        # 3. Gate ops (sigmoid/tanh) — elementwise, combines ih + hh results
        final_shape = list(output_shapes[0]) if output_shapes else [S, B, H]
        subgraph[gates_id] = {
            "type": "sigmoid",
            "einsum_equation": "SBH->SBH",
            "elementwise_op": "sigmoid",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{ih_id}.Output", f"{hh_id}.Output"],
                "outputs": [f"{gates_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [ih_output_shape, hh_output_shape],
                "outputs": [final_shape],
            },
            "tensor_dtypes": {
                "inputs": [out_dtype, out_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input": ["S", "B", "H"],
                "Output": ["S", "B", "H"],
            },
            "connections": {
                "inputs": [ih_id, hh_id],
                "outputs": output_connections,
            },
        }

        final_node_id = gates_id
        return subgraph, final_node_id, input_mapping
