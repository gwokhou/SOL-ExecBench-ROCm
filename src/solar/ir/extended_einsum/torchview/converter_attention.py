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


class ConverterAttentionMixin(ConverterMixinContract):
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

    def _expand_sdpa(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand scaled_dot_product_attention into a subgraph of operations.

        Based on PyTorch's reference implementation:
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight = torch.softmax(attn_weight, dim=-1)
            return attn_weight @ value

        Returns:
            Tuple of (subgraph_layers_dict, final_node_id, input_mapping)
            input_mapping maps input index -> subgraph node that receives it
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        node_data.get("module_args", {})
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        if len(input_shapes) < 3:
            raise ValueError(
                f"SDPA requires 3 inputs (Q, K, V). Got: {input_shapes}"
            )

        query_shape = list(input_shapes[0])  # [B, H, Q, D]
        key_shape = list(input_shapes[1])  # [B, H, K, D]
        value_shape = list(input_shapes[2])  # [B, H, K, V]
        output_shape = list(output_shapes[0]) if output_shapes else None

        # Infer dimensions
        B = query_shape[0]  # noqa: N806 - matches the einsum rank label
        H = query_shape[1]  # noqa: N806 - matches the einsum rank label
        Q_len = query_shape[2]  # noqa: N806 - matches the einsum rank label
        D = query_shape[3]  # noqa: N806 - matches the einsum rank label
        K_len = key_shape[2]  # noqa: N806 - matches the einsum rank label
        V_dim = value_shape[3]  # noqa: N806 - matches the einsum rank label

        # Intermediate shapes
        scores_shape = [B, H, Q_len, K_len]  # Q @ K^T
        final_output_shape = (
            output_shape if output_shape else [B, H, Q_len, V_dim]
        )

        # Build input connections
        input_connections = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)

        output_connections = sorted(op_graph.successors(node_id))

        subgraph: dict[str, dict[str, Any]] = {}

        # Node IDs for subgraph
        qk_node_id = f"{node_id}.qk_matmul"
        scale_node_id = f"{node_id}.scale"
        softmax_node_id = f"{node_id}.softmax"
        av_node_id = f"{node_id}.av_matmul"

        # Build input mapping: which predecessor input goes to which subgraph node
        # Q (input 0) -> qk_matmul
        # K (input 1) -> qk_matmul
        # V (input 2) -> av_matmul
        input_mapping: dict[int, str] = {
            0: qk_node_id,  # Q -> qk_matmul
            1: qk_node_id,  # K -> qk_matmul
            2: av_node_id,  # V -> av_matmul
        }

        # 1. Q @ K^T -> attention scores
        # Einsum: BHQD,BHKD->BHQK (D is contracted)
        subgraph[qk_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQD,BHKD->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    (
                        f"{input_connections[0]}.Output"
                        if input_connections
                        else f"{node_id}.Query"
                    ),
                    (
                        f"{input_connections[1]}.Output"
                        if len(input_connections) > 1
                        else f"{node_id}.Key"
                    ),
                ],
                "outputs": [f"{qk_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [query_shape, key_shape],
                "outputs": [scores_shape],
            },
            # Operands drive the AF graph builder; without them the
            # layer is silently dropped (cf. commit 8162f29 for linear).
            "operands": {
                "Input": ["B", "H", "Q", "D"],
                "Weight": ["B", "H", "K", "D"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": (
                    input_connections[:2]
                    if len(input_connections) >= 2
                    else input_connections
                ),
                "outputs": [scale_node_id],
            },
        }
        # 2. Scale by 1/sqrt(d_k)
        subgraph[scale_node_id] = {
            "type": "mul",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{qk_node_id}.Output"],
                "outputs": [f"{scale_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [qk_node_id],
                "outputs": [softmax_node_id],
            },
            "additional_info": {
                "scale_factor": f"1/sqrt({D})",
            },
        }

        # 3. Softmax over K dimension (dim=-1)
        subgraph[softmax_node_id] = {
            "type": "softmax",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "softmax",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{scale_node_id}.Output"],
                "outputs": [f"{softmax_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [scale_node_id],
                "outputs": [av_node_id],
            },
            "additional_info": {
                "dim": -1,
            },
        }

        # 4. Attention weights @ V -> output
        # Einsum: BHQK,BHKV->BHQV (K is contracted)
        subgraph[av_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQK,BHKV->BHQV",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    f"{softmax_node_id}.Output",
                    (
                        f"{input_connections[2]}.Output"
                        if len(input_connections) > 2
                        else f"{node_id}.Value"
                    ),
                ],
                "outputs": [f"{av_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape, value_shape],
                "outputs": [final_output_shape],
            },
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Weight": ["B", "H", "K", "V"],
                "Output": ["B", "H", "Q", "V"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": [softmax_node_id]
                + (
                    input_connections[2:3] if len(input_connections) > 2 else []
                ),
                "outputs": output_connections,
            },
        }

        return subgraph, av_node_id, input_mapping

    @staticmethod
    def _as_list(value: Any, default: list[int]) -> list[Any]:
        """Normalize scalar/list convolution args to a list."""
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _expand_mha(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand multi_head_attention_forward into a subgraph.

        MHA decomposes into:
          1. in_proj (linear): input @ in_proj_weight^T  [S*B,D] @ [D,3D] -> [S*B,3D]
          2. qk_matmul: Q @ K^T  [B,H,S,D/H] x [B,H,D/H,S] -> [B,H,S,S]
          3. scale: * 1/sqrt(d_k)
          4. softmax
          5. av_matmul: attn @ V  [B,H,S,S] x [B,H,S,D/H] -> [B,H,S,D/H]
          6. out_proj (linear): concat @ out_proj_weight^T  [S*B,D] @ [D,D] -> [S*B,D]

        The head count cancels for MACs: B*H*S*S*(D/H) = B*S*S*D.
        We use num_heads=1 equivalent shapes so standard equations work.
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

        # Parse activation shape: [S, B, D]
        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"MHA requires [S,B,D] input. Got: {act_shape}")
        S, B, D = act_shape[0], act_shape[1], act_shape[2]  # noqa: N806

        # Find weight shapes by type
        in_proj_w_shape = None  # [3D, D]
        out_proj_w_shape = None  # [D, D]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[0] == 3 * D and ws[1] == D:
                        in_proj_w_shape = ws
                    elif ws[0] == D and ws[1] == D:
                        out_proj_w_shape = ws

        # Derived shapes for sub-nodes
        [S, B, 3 * D]  # after in_proj
        # Use single-head equivalent: [B, 1, S, D] so H cancels in cost
        # Use Q/K labels for sequence dims to avoid repeated dim in equations
        q_shape = [B, 1, S, D]
        k_transposed_shape = [B, 1, D, S]  # K^T for matmul handler convention
        v_shape = [B, 1, S, D]
        scores_shape = [B, 1, S, S]  # shapes still use S,S (same value)
        attn_out_shape = [B, 1, S, D]
        list(output_shapes[0]) if output_shapes else [S, B, D]

        # Build connections
        input_connections = sorted(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(op_graph.successors(node_id))

        subgraph: dict[str, dict[str, Any]] = {}

        in_proj_id = f"{node_id}.in_proj"
        qk_id = f"{node_id}.qk_matmul"
        scale_id = f"{node_id}.scale"
        softmax_id = f"{node_id}.softmax"
        av_id = f"{node_id}.av_matmul"
        out_proj_id = f"{node_id}.out_proj"

        input_mapping: dict[int, str] = {0: in_proj_id}
        if len(input_connections) > 1:
            input_mapping[1] = in_proj_id
        if len(input_connections) > 2:
            input_mapping[2] = in_proj_id

        act_input_name = (
            f"{input_connections[0]}.Output"
            if input_connections
            else f"{node_id}.Input"
        )
        in_proj_w_name = f"{node_id}.in_proj.Weight"
        out_proj_w_name = f"{node_id}.out_proj.Weight"

        # 1. in_proj: input @ in_proj_weight^T
        #    Weight is [N, K] = [3D, D]. Equation MK,NK->MN matches handler convention.
        in_proj_input_shape = [S * B, D]
        in_proj_output_shape = [S * B, 3 * D]
        if in_proj_w_shape:
            subgraph[in_proj_id] = {
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
                    "inputs": [act_input_name, in_proj_w_name],
                    "outputs": [f"{in_proj_id}.Output"],
                },
                "tensor_types": {
                    "inputs": ["input", "weight"],
                    "outputs": ["output"],
                },
                "tensor_shapes": {
                    "inputs": [in_proj_input_shape, in_proj_w_shape],
                    "outputs": [in_proj_output_shape],
                },
                "tensor_dtypes": {
                    "inputs": [act_dtype, weight_dtype],
                    "outputs": [act_dtype],
                },
                "connections": {
                    "inputs": input_connections[:1],
                    "outputs": [qk_id],
                },
            }

        # 2. qk_matmul: Q @ K^T
        subgraph[qk_id] = {
            "type": "matmul",
            "einsum_equation": "BHQD,BHDK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": {
                "Input": ["B", "H", "Q", "D"],
                "Input_1": ["B", "H", "D", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_names": {
                "inputs": [
                    f"{in_proj_id}.Output"
                    if in_proj_w_shape
                    else act_input_name,
                    f"{in_proj_id}.Output"
                    if in_proj_w_shape
                    else act_input_name,
                ],
                "outputs": [f"{qk_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [q_shape, k_transposed_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [in_proj_id]
                if in_proj_w_shape
                else input_connections[:1],
                "outputs": [scale_id],
            },
        }

        # 3. scale
        subgraph[scale_id] = {
            "type": "mul",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_names": {
                "inputs": [f"{qk_id}.Output"],
                "outputs": [f"{scale_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [qk_id],
                "outputs": [softmax_id],
            },
        }

        # 4. softmax
        subgraph[softmax_id] = {
            "type": "softmax",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "softmax",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_names": {
                "inputs": [f"{scale_id}.Output"],
                "outputs": [f"{softmax_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [scale_id],
                "outputs": [av_id],
            },
        }

        # 5. av_matmul: attn @ V
        subgraph[av_id] = {
            "type": "matmul",
            "einsum_equation": "BHQK,BHKV->BHQV",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": {
                "Input": ["B", "H", "Q", "K"],
                "Input_1": ["B", "H", "K", "V"],
                "Output": ["B", "H", "Q", "V"],
            },
            "tensor_names": {
                "inputs": [
                    f"{softmax_id}.Output",
                    f"{in_proj_id}.Output"
                    if in_proj_w_shape
                    else act_input_name,
                ],
                "outputs": [f"{av_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape, v_shape],
                "outputs": [attn_out_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [softmax_id],
                "outputs": [out_proj_id]
                if out_proj_w_shape
                else output_connections,
            },
        }

        # 6. out_proj: Weight is [N, K] = [D, D]. Equation MK,NK->MN.
        final_node_id = av_id
        if out_proj_w_shape:
            out_proj_input_shape = [S * B, D]
            out_proj_output_shape = [S * B, D]
            subgraph[out_proj_id] = {
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
                    "inputs": [f"{av_id}.Output", out_proj_w_name],
                    "outputs": [f"{out_proj_id}.Output"],
                },
                "tensor_types": {
                    "inputs": ["input", "weight"],
                    "outputs": ["output"],
                },
                "tensor_shapes": {
                    "inputs": [out_proj_input_shape, out_proj_w_shape],
                    "outputs": [out_proj_output_shape],
                },
                "tensor_dtypes": {
                    "inputs": [act_dtype, weight_dtype],
                    "outputs": [out_dtype],
                },
                "connections": {
                    "inputs": [av_id],
                    "outputs": output_connections,
                },
            }
            final_node_id = out_proj_id

        return subgraph, final_node_id, input_mapping
