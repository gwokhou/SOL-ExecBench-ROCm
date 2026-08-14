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
from solar.types import TensorShapes

PathLike = str | Path


@dataclass(frozen=True)
class _GroupwiseGeometry:
    node_type: str
    is_2d: bool
    module_args: dict[str, Any]
    input_shape: list[int]
    weight_shape: list[int]
    output_shape: list[int]
    reshaped_input: list[int]
    reshaped_weight: list[int]
    reshaped_output: list[int]
    reshape_input_equation: str
    reshape_input_operands: dict[str, list[str]]
    reshape_output_equation: str
    reshape_output_operands: dict[str, list[str]]
    fallback_conv_equation: str


@dataclass(frozen=True)
class _GroupwiseConnections:
    activation: str | None
    weight: str | None
    outputs: list[str]


@dataclass(frozen=True)
class _GroupwiseLayerIds:
    reshape_input: str
    convolution: str
    reshape_output: str


@dataclass(frozen=True)
class _GroupwiseDTypes:
    activation: str
    weight: str
    output: str


def _groupwise_2d_geometry(
    module_args: dict[str, Any],
    input_shape: list[int],
    weight_shape: list[int],
    output_shape: list[int],
    groups: int,
    channels_per_group: int,
    outputs_per_group: int,
) -> _GroupwiseGeometry:
    height, width = input_shape[2:4]
    kernel_height, kernel_width = weight_shape[2:4]
    output_height, output_width = output_shape[2:4]
    return _GroupwiseGeometry(
        node_type="conv2d",
        is_2d=True,
        module_args=module_args,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        reshaped_input=[
            input_shape[0],
            groups,
            channels_per_group,
            height,
            width,
        ],
        reshaped_weight=[
            groups,
            outputs_per_group,
            channels_per_group,
            kernel_height,
            kernel_width,
        ],
        reshaped_output=[
            input_shape[0],
            groups,
            outputs_per_group,
            output_height,
            output_width,
        ],
        reshape_input_equation="ABCD->AE0E1CD",
        reshape_input_operands={
            "Input": ["A", "B", "C", "D"],
            "Output": ["A", "E0", "E1", "C", "D"],
        },
        reshape_output_equation="ABCDE->AF0DE",
        reshape_output_operands={
            "Input": ["A", "B", "C", "D", "E"],
            "Output": ["A", "F0", "D", "E"],
        },
        fallback_conv_equation="BGI(P+R)(Q+S),GOIRS->BGOPQ",
    )


def _groupwise_1d_geometry(
    node_type: str,
    module_args: dict[str, Any],
    input_shape: list[int],
    weight_shape: list[int],
    output_shape: list[int],
    groups: int,
    channels_per_group: int,
    outputs_per_group: int,
) -> _GroupwiseGeometry:
    return _GroupwiseGeometry(
        node_type=node_type,
        is_2d=False,
        module_args=module_args,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        reshaped_input=[
            input_shape[0],
            groups,
            channels_per_group,
            input_shape[2],
        ],
        reshaped_weight=[
            groups,
            outputs_per_group,
            channels_per_group,
            weight_shape[2],
        ],
        reshaped_output=[
            input_shape[0],
            groups,
            outputs_per_group,
            output_shape[2],
        ],
        reshape_input_equation="ABC->ADE0C",
        reshape_input_operands={
            "Input": ["A", "B", "C"],
            "Output": ["A", "D", "E0", "C"],
        },
        reshape_output_equation="ABCD->AE0D",
        reshape_output_operands={
            "Input": ["A", "B", "C", "D"],
            "Output": ["A", "E0", "D"],
        },
        fallback_conv_equation="BGI(P+R),GOIR->BGOP",
    )


def _groupwise_geometry(node_data: dict[str, Any]) -> _GroupwiseGeometry:
    module_args = node_data.get("module_args") or {}
    groups = int(module_args.get("groups", 1))
    input_shapes = node_data.get("input_shapes") or []
    output_shapes = node_data.get("output_shapes") or []
    input_shape = list(input_shapes[0]) if input_shapes else []
    weight_shape = list(input_shapes[1]) if len(input_shapes) > 1 else []
    output_shape = list(output_shapes[0]) if output_shapes else []
    node_type = str(node_data.get("type", "conv2d")).lower()
    output_channels = weight_shape[0] if weight_shape else output_shape[1]
    common_args = (
        module_args,
        input_shape,
        weight_shape,
        output_shape,
        groups,
        input_shape[1] // groups,
        output_channels // groups,
    )
    if node_type == "conv2d":
        return _groupwise_2d_geometry(*common_args)
    return _groupwise_1d_geometry(node_type, *common_args)


def _groupwise_ids(node_id: str) -> _GroupwiseLayerIds:
    return _GroupwiseLayerIds(
        reshape_input=f"{node_id}.reshape_input",
        convolution=f"{node_id}.groupwise_conv",
        reshape_output=f"{node_id}.reshape_output",
    )


def _groupwise_dtypes(node_data: dict[str, Any]) -> _GroupwiseDTypes:
    inputs = node_data.get("input_dtypes") or []
    outputs = node_data.get("output_dtypes") or []
    activation = inputs[0] if inputs else "torch.float32"
    return _GroupwiseDTypes(
        activation=activation,
        weight=inputs[1] if len(inputs) > 1 else activation,
        output=outputs[0] if outputs else activation,
    )


def _fallback_conv_operands(is_2d: bool) -> dict[str, list[str]]:
    if is_2d:
        return {
            "Input": ["B", "G", "I", "P+R", "Q+S"],
            "Weight": ["G", "O", "I", "R", "S"],
            "Output": ["B", "G", "O", "P", "Q"],
        }
    return {
        "Input": ["B", "G", "I", "P+R"],
        "Weight": ["G", "O", "I", "R"],
        "Output": ["B", "G", "O", "P"],
    }


def _select_data_connections(
    node_data: dict[str, Any], input_connections: list[str | None]
) -> tuple[str | None, str | None]:
    input_types = node_data.get("input_types") or []
    activation = None
    weight = None
    for index, connection in enumerate(input_connections):
        input_type = input_types[index] if index < len(input_types) else "input"
        is_weight = str(input_type).lower() == "weight"
        is_parameter = "parameter-tensor" in str(connection)
        if is_weight or is_parameter:
            weight = weight or connection
        else:
            activation = activation or connection
    if activation is None and input_connections:
        activation = input_connections[0]
    if weight is None and len(input_connections) > 1:
        weight = input_connections[1]
    return activation, weight


def _make_groupwise_view_layer(
    geometry: _GroupwiseGeometry,
    connections: _GroupwiseConnections,
    ids: _GroupwiseLayerIds,
    dtypes: _GroupwiseDTypes,
    *,
    input_view: bool,
) -> dict[str, Any]:
    if input_view:
        layer_id = ids.reshape_input
        source = connections.activation
        source_name = f"{source}.Output" if source else f"{layer_id}.Input"
        input_shape = geometry.input_shape
        output_shape = geometry.reshaped_input
        equation = geometry.reshape_input_equation
        operands = geometry.reshape_input_operands
        input_dtype = output_dtype = dtypes.activation
        layer_connections = {
            "inputs": [source] if source else [],
            "outputs": [ids.convolution],
        }
    else:
        layer_id = ids.reshape_output
        source_name = f"{ids.convolution}.Output"
        input_shape = geometry.reshaped_output
        output_shape = geometry.output_shape
        equation = geometry.reshape_output_equation
        operands = geometry.reshape_output_operands
        input_dtype = output_dtype = dtypes.output
        layer_connections = {
            "inputs": [ids.convolution],
            "outputs": connections.outputs,
        }
    return {
        "type": "view",
        "einsum_equation": equation,
        "elementwise_op": "copy",
        "reduction_op": "none",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {
            "inputs": [source_name],
            "outputs": [f"{layer_id}.Output"],
        },
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "tensor_shapes": {"inputs": [input_shape], "outputs": [output_shape]},
        "tensor_dtypes": {"inputs": [input_dtype], "outputs": [output_dtype]},
        "operands": operands,
        "connections": layer_connections,
    }


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

    @staticmethod
    def _raw_groupwise_inputs(
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
    ) -> list[str]:
        connections = list(
            (node_data.get("connections") or {}).get("inputs") or []
        )
        if not connections:
            connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id not in info.get("consumers", []):
                continue
            original_id = info["original_id"]
            if original_id not in connections:
                connections.append(original_id)
        return connections

    def _resolve_groupwise_inputs(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_node_id_map: dict[str, str],
        raw_connections: list[str],
    ) -> list[str | None]:
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        predecessors = list(op_graph.predecessors(node_id))
        input_types = list(node_data.get("input_types") or [])
        resolved: list[str | None] = []
        assigned: set[str] = set()
        deferred: list[int] = []

        for index, connection in enumerate(raw_connections):
            mapped = start_node_id_map.get(connection, connection)
            input_type = (
                str(input_types[index]).lower()
                if index < len(input_types)
                else "input"
            )
            if input_type == "weight":
                resolved.append(mapped)
            elif (
                mapped in start_node_id_map.values() or mapped in op_graph.nodes
            ):
                resolved.append(mapped)
                assigned.add(mapped)
            elif (
                producer := tensor_to_producer.get(connection)
            ) is not None and producer in op_graph.nodes:
                resolved.append(producer)
                assigned.add(producer)
            else:
                resolved.append(None)
                deferred.append(index)

        unmatched = [item for item in predecessors if item not in assigned]
        if len(unmatched) == len(deferred):
            for index, predecessor in zip(deferred, unmatched, strict=False):
                resolved[index] = predecessor
        elif not unmatched:
            for index in deferred:
                connection = raw_connections[index]
                resolved[index] = start_node_id_map.get(connection, connection)
        elif deferred:
            raise ValueError(
                f"_convert_operation({node_id!r}, conv path): ambiguous "
                f"predecessor resolution; deferred={len(deferred)}, "
                f"unmatched_preds={len(unmatched)}."
            )
        return resolved

    def _groupwise_connections(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> _GroupwiseConnections:
        raw_inputs = self._raw_groupwise_inputs(
            node_id, node_data, op_graph, start_nodes_info
        )
        inputs = self._resolve_groupwise_inputs(
            node_id, node_data, op_graph, start_node_id_map, raw_inputs
        )
        outputs = list(op_graph.successors(node_id))
        if not outputs:
            raw_outputs = list(
                (node_data.get("connections") or {}).get("outputs") or []
            )
            outputs = [item for item in raw_outputs if item in op_graph.nodes]
        activation, weight = _select_data_connections(node_data, inputs)
        return _GroupwiseConnections(activation, weight, outputs)

    def _groupwise_conv_spec(
        self, geometry: _GroupwiseGeometry
    ) -> tuple[str, dict[str, list[str]]]:
        default = [1, 1] if geometry.is_2d else [1]
        stride = self._as_list(geometry.module_args.get("stride"), default)
        padding_default = [0, 0] if geometry.is_2d else [0]
        padding = self._as_list(
            geometry.module_args.get("padding"), padding_default
        )
        dilation = self._as_list(geometry.module_args.get("dilation"), default)
        tensor_shapes = TensorShapes(
            inputs=[geometry.reshaped_input, geometry.reshaped_weight],
            outputs=[geometry.reshaped_output],
        )
        try:
            operation = self._einsum_analyzer.get_einsum_op(
                geometry.node_type,
                tensor_shapes,
                module_args=geometry.module_args,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        except Exception:  # noqa: BLE001 - optional backend fallback
            return (
                geometry.fallback_conv_equation,
                _fallback_conv_operands(geometry.is_2d),
            )
        return (
            operation.equation,
            {operand.name: operand.dims for operand in operation.operands},
        )

    def _make_groupwise_conv_layer(
        self,
        geometry: _GroupwiseGeometry,
        connections: _GroupwiseConnections,
        ids: _GroupwiseLayerIds,
        dtypes: _GroupwiseDTypes,
    ) -> dict[str, Any]:
        equation, operands = self._groupwise_conv_spec(geometry)
        weight_name = (
            f"{connections.weight}.Output"
            if connections.weight
            else f"{ids.convolution}.Weight"
        )
        input_connections = [ids.reshape_input]
        if connections.weight:
            input_connections.append(connections.weight)
        return {
            "type": geometry.node_type,
            "einsum_equation": equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{ids.reshape_input}.Output", weight_name],
                "outputs": [f"{ids.convolution}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [
                    geometry.reshaped_input,
                    geometry.reshaped_weight,
                ],
                "outputs": [geometry.reshaped_output],
            },
            "tensor_dtypes": {
                "inputs": [dtypes.activation, dtypes.weight],
                "outputs": [dtypes.output],
            },
            "operands": operands,
            "connections": {
                "inputs": input_connections,
                "outputs": [ids.reshape_output],
            },
        }

    def _expand_groupwise_conv(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], str, dict[int, str]]:
        """Expand group-wise conv into input view, grouped conv, and output view."""
        geometry = _groupwise_geometry(node_data)
        connections = self._groupwise_connections(
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
        ids = _groupwise_ids(node_id)
        dtypes = _groupwise_dtypes(node_data)
        subgraph = {
            ids.reshape_input: _make_groupwise_view_layer(
                geometry, connections, ids, dtypes, input_view=True
            ),
            ids.convolution: self._make_groupwise_conv_layer(
                geometry, connections, ids, dtypes
            ),
            ids.reshape_output: _make_groupwise_view_layer(
                geometry, connections, ids, dtypes, input_view=False
            ),
        }
        input_mapping = {}
        if connections.activation:
            input_mapping[0] = ids.reshape_input
        if connections.weight:
            input_mapping[1] = ids.convolution
        return subgraph, ids.reshape_output, input_mapping
