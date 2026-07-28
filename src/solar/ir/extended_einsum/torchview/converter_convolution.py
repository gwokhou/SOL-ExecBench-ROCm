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
from solar.types import TensorShapes

PathLike = str | Path


class ConverterConvolutionMixin(ConverterMixinContract):
    """Expand reviewed groupwise-convolution operations."""

    def _should_expand_groupwise_conv(self, node_data: dict[str, Any]) -> bool:
        """Check if this is a group-wise convolution that needs reshape expansion.

        Currently only conv1d / conv2d are expanded via the reshape pass.
        Conv3d and conv-transpose variants rely on the AF graph builder's
        union-find canonicalization to handle the C_out vs C_out/groups
        split, since the reshape path doesn't survive the parameter-tensor
        weight (still 4-/5-dim original) <-> 5-/6-dim grouped einsum mismatch.
        """
        node_type = str(node_data.get("type", "")).lower()
        if node_type not in ("conv1d", "conv2d"):
            return False

        module_args = node_data.get("module_args") or {}
        groups = int(module_args.get("groups", 1))
        if groups <= 1:
            return False

        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        in_channels = int(
            module_args.get(
                "in_channels",
                input_shapes[0][1]
                if input_shapes and len(input_shapes[0]) > 1
                else 0,
            )
        )
        out_channels = int(
            module_args.get(
                "out_channels",
                (
                    output_shapes[0][1]
                    if output_shapes and len(output_shapes[0]) > 1
                    else 0
                ),
            )
        )

        # Depthwise conv is handled directly by the conv handler.
        return not (groups == in_channels and groups == out_channels)

    def _expand_groupwise_conv(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand group-wise conv into input view, grouped conv, and output view."""
        module_args = node_data.get("module_args") or {}
        groups = int(module_args.get("groups", 1))
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        node_type = str(node_data.get("type", "conv2d")).lower()
        is_2d = node_type == "conv2d"

        input_shape = list(input_shapes[0]) if len(input_shapes) > 0 else []
        weight_shape = list(input_shapes[1]) if len(input_shapes) > 1 else []
        output_shape = list(output_shapes[0]) if output_shapes else []

        B = input_shape[0]  # noqa: N806 - matches the einsum rank label
        C_in = input_shape[1]  # noqa: N806 - matches the einsum rank label
        O_total = (  # noqa: N806 - matches the einsum rank label
            weight_shape[0] if weight_shape else output_shape[1]
        )
        I = C_in // groups  # noqa: N806 - matches the einsum rank label
        O_pg = O_total // groups  # noqa: N806 - matches the einsum rank label

        if is_2d:
            H, W = input_shape[2], input_shape[3]  # noqa: N806
            KH, KW = weight_shape[2], weight_shape[3]  # noqa: N806
            H_out, W_out = output_shape[2], output_shape[3]  # noqa: N806
            reshaped_input = [B, groups, I, H, W]
            reshaped_weight = [groups, O_pg, I, KH, KW]
            reshaped_output = [B, groups, O_pg, H_out, W_out]
            reshape_in_eq = "ABCD->AE0E1CD"
            reshape_in_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "E1", "C", "D"],
            }
            reshape_out_eq = "ABCDE->AF0DE"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D", "E"],
                "Output": ["A", "F0", "D", "E"],
            }
            fallback_conv_equation = "BGI(P+R)(Q+S),GOIRS->BGOPQ"
        else:
            L = input_shape[2]  # noqa: N806 - matches the einsum rank label
            KL = weight_shape[2]  # noqa: N806 - matches the einsum rank label
            L_out = output_shape[2]  # noqa: N806 - einsum rank label
            reshaped_input = [B, groups, I, L]
            reshaped_weight = [groups, O_pg, I, KL]
            reshaped_output = [B, groups, O_pg, L_out]
            reshape_in_eq = "ABC->ADE0C"
            reshape_in_operands = {
                "Input": ["A", "B", "C"],
                "Output": ["A", "D", "E0", "C"],
            }
            reshape_out_eq = "ABCD->AE0D"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "D"],
            }
            fallback_conv_equation = "BGI(P+R),GOIR->BGOP"

        raw_input_connections = list(
            (node_data.get("connections") or {}).get("inputs") or []
        )
        if not raw_input_connections:
            raw_input_connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in raw_input_connections:
                    raw_input_connections.append(original_id)

        # Strict tensor→producer resolution (see notes in the twin block of
        # `_convert_operation`). Fall back to op_graph.predecessors only when
        # exactly enough unmatched predecessors remain to fill all deferred
        # slots — raise on ambiguity rather than silently positional-guess.
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        op_predecessors = list(op_graph.predecessors(node_id))
        input_connections: list[str | None] = []
        assigned_preds: set[str] = set()
        deferred_indices: list[int] = []
        input_types_raw = list(node_data.get("input_types") or [])
        for idx, conn_id in enumerate(raw_input_connections):
            mapped = start_node_id_map.get(conn_id, conn_id)
            itype = (
                str(input_types_raw[idx]).lower()
                if idx < len(input_types_raw)
                else "input"
            )
            if itype == "weight":
                input_connections.append(mapped)
                continue
            if mapped in start_node_id_map.values() or mapped in op_graph.nodes:
                input_connections.append(mapped)
                assigned_preds.add(mapped)
                continue
            producer = tensor_to_producer.get(conn_id)
            if producer is not None and producer in op_graph.nodes:
                input_connections.append(producer)
                assigned_preds.add(producer)
                continue
            input_connections.append(None)
            deferred_indices.append(idx)
        if deferred_indices:
            unmatched_preds = [
                p for p in op_predecessors if p not in assigned_preds
            ]
            if len(unmatched_preds) == len(deferred_indices):
                for d_idx, pred in zip(
                    deferred_indices, unmatched_preds, strict=False
                ):
                    input_connections[d_idx] = pred
            elif len(unmatched_preds) == 0:
                # Genuinely producerless input — fall through to literal name.
                for d_idx in deferred_indices:
                    input_connections[d_idx] = start_node_id_map.get(
                        raw_input_connections[d_idx],
                        raw_input_connections[d_idx],
                    )
            else:
                raise ValueError(
                    f"_convert_operation({node_id!r}, conv path): ambiguous "
                    f"predecessor resolution; deferred={len(deferred_indices)}, "
                    f"unmatched_preds={len(unmatched_preds)}."
                )

        output_connections = list(op_graph.successors(node_id))
        if not output_connections:
            raw_outs = list(
                (node_data.get("connections") or {}).get("outputs") or []
            )
            output_connections = [c for c in raw_outs if c in op_graph.nodes]

        input_types = node_data.get("input_types") or []
        activation_conn = None
        weight_conn = None
        for idx, conn in enumerate(input_connections):
            itype = input_types[idx] if idx < len(input_types) else "input"
            if str(itype).lower() == "weight" or "parameter-tensor" in str(
                conn
            ):
                if weight_conn is None:
                    weight_conn = conn
            elif activation_conn is None:
                activation_conn = conn
        if activation_conn is None and input_connections:
            activation_conn = input_connections[0]
        if weight_conn is None and len(input_connections) > 1:
            weight_conn = input_connections[1]

        reshape_in_id = f"{node_id}.reshape_input"
        conv_id = f"{node_id}.groupwise_conv"
        reshape_out_id = f"{node_id}.reshape_output"

        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = (
            input_dtypes[1] if len(input_dtypes) > 1 else activation_dtype
        )
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        weight_tensor_name = (
            f"{weight_conn}.Output" if weight_conn else f"{conv_id}.Weight"
        )

        reshape_in_layer = {
            "type": "view",
            "einsum_equation": reshape_in_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    (
                        f"{activation_conn}.Output"
                        if activation_conn
                        else f"{reshape_in_id}.Input"
                    )
                ],
                "outputs": [f"{reshape_in_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(input_shape)],
                "outputs": [reshaped_input],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype],
                "outputs": [activation_dtype],
            },
            "operands": reshape_in_operands,
            "connections": {
                "inputs": [activation_conn] if activation_conn else [],
                "outputs": [conv_id],
            },
        }

        stride = self._as_list(
            module_args.get("stride"), [1, 1] if is_2d else [1]
        )
        padding = self._as_list(
            module_args.get("padding"), [0, 0] if is_2d else [0]
        )
        dilation = self._as_list(
            module_args.get("dilation"), [1, 1] if is_2d else [1]
        )

        conv_ts = TensorShapes(
            inputs=[reshaped_input, reshaped_weight],
            outputs=[reshaped_output],
        )
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op(
                node_type,
                conv_ts,
                module_args=module_args,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            conv_equation = einsum_op.equation
            conv_operands = {
                operand.name: operand.dims for operand in einsum_op.operands
            }
        except Exception:  # noqa: BLE001 - optional backend fallback
            conv_equation = fallback_conv_equation
            if is_2d:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R", "Q+S"],
                    "Weight": ["G", "O", "I", "R", "S"],
                    "Output": ["B", "G", "O", "P", "Q"],
                }
            else:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R"],
                    "Weight": ["G", "O", "I", "R"],
                    "Output": ["B", "G", "O", "P"],
                }

        conv_layer = {
            "type": node_type,
            "einsum_equation": conv_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{reshape_in_id}.Output", weight_tensor_name],
                "outputs": [f"{conv_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_input, reshaped_weight],
                "outputs": [reshaped_output],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype, weight_dtype],
                "outputs": [output_dtype],
            },
            "operands": conv_operands,
            "connections": {
                "inputs": (
                    [reshape_in_id, weight_conn]
                    if weight_conn
                    else [reshape_in_id]
                ),
                "outputs": [reshape_out_id],
            },
        }

        reshape_out_layer = {
            "type": "view",
            "einsum_equation": reshape_out_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{conv_id}.Output"],
                "outputs": [f"{reshape_out_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_output],
                "outputs": [list(output_shape)],
            },
            "tensor_dtypes": {
                "inputs": [output_dtype],
                "outputs": [output_dtype],
            },
            "operands": reshape_out_operands,
            "connections": {
                "inputs": [conv_id],
                "outputs": output_connections,
            },
        }

        subgraph = {
            reshape_in_id: reshape_in_layer,
            conv_id: conv_layer,
            reshape_out_id: reshape_out_layer,
        }

        input_mapping = {}
        if activation_conn:
            input_mapping[0] = reshape_in_id
        if weight_conn:
            input_mapping[1] = conv_id

        return subgraph, reshape_out_id, input_mapping
