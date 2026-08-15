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

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import networkx as nx

from solar.composition import BoundComponent
from solar.types import TensorShapes

PathLike = str | Path
_LinearEntry = tuple[int, str, Any, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class _LinearSplitContext:
    """Normalized inputs used to emit a linear matmul and bias add."""

    node_id: str
    add_node_id: str
    output_connections: list[str]
    output_shapes: list[list[int]]
    matmul_output_shape: list[int]
    activation: _LinearEntry | None
    weight: _LinearEntry | None
    bias: _LinearEntry | None
    weight_shape: list[int] | None
    bias_shape: list[int] | None
    activation_dtype: Any
    weight_dtype: Any
    output_dtype: Any
    start_node_id_map: dict[str, str]


class GraphNormalizer(BoundComponent):
    """Normalize split, linear, and entry-node graph structure."""

    def _should_split_linear_with_bias(self, node_data: dict[str, Any]) -> bool:
        """Check if this is a linear layer with bias that should be split."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        if node_type != "linear":
            return False

        input_shapes = node_data.get("input_shapes") or []
        input_types = [
            str(t).lower() for t in (node_data.get("input_types") or [])
        ]

        # Prefer explicit tensor typing: one activation + at least two weight inputs,
        # with at least one rank-1 weight as bias.
        if input_types:
            weight_indices = [
                i for i, t in enumerate(input_types) if t == "weight"
            ]
            input_indices = [
                i for i, t in enumerate(input_types) if t == "input"
            ]
            has_rank1_weight = any(
                i < len(input_shapes)
                and isinstance(input_shapes[i], list)
                and len(input_shapes[i]) == 1
                for i in weight_indices
            )
            if (
                len(input_indices) >= 1
                and len(weight_indices) >= 2
                and has_rank1_weight
            ):
                return True
            # Don't early-return False here; input_types can be incomplete
            # in some traced graphs. Fall through to fallback checks.

        # Fallback without input_types: x, weight, bias by shape rank pattern.
        if len(input_shapes) >= 3:
            has_rank1 = any(
                isinstance(s, list) and len(s) == 1 for s in input_shapes
            )
            has_rank2_or_more = any(
                isinstance(s, list) and len(s) >= 2 for s in input_shapes
            )
            if has_rank1 and has_rank2_or_more:
                return True

        # Fallback: infer from metadata/notes text when shape info is incomplete.
        module_args = node_data.get("module_args") or {}
        if bool(module_args.get("bias", False)):
            return True

        notes_blob = " ".join(
            str(v)
            for v in (
                node_data.get("notes"),
                module_args.get("raw_attributes"),
                module_args.get("function_name"),
            )
            if v is not None
        ).lower()
        return "bias" in notes_blob

    def _validate_input_types_alignment(
        self, node_id: str, node_data: dict[str, Any]
    ) -> None:
        """Ensure input_types aligns 1:1 with input_shapes for op nodes.

        When the torchview graph collapses multiple tensor inputs into
        fewer connection nodes (e.g. ``cat`` receiving two tensors via a
        single hidden-tensor node), ``input_types`` will be shorter than
        ``input_shapes``.  Pad with ``'input'`` to restore alignment,
        since the missing entries are always activation (non-weight)
        tensors.

        If ``input_types`` is *longer* than ``input_shapes``, that
        indicates a graph construction bug and we raise immediately.
        """
        input_shapes = node_data.get("input_shapes") or []
        input_types = node_data.get("input_types") or []
        if len(input_types) < len(input_shapes):
            input_types = list(input_types) + ["input"] * (
                len(input_shapes) - len(input_types)
            )
            node_data["input_types"] = input_types
        elif len(input_types) > len(input_shapes):
            node_type = node_data.get("type", "unknown")
            raise ValueError(
                f"Node '{node_id}' (type={node_type}) has more input_types "
                f"({len(input_types)}) than input_shapes ({len(input_shapes)}). "
                "This indicates a graph construction bug."
            )

    def _linear_typed_inputs(
        self,
        connections: list[str],
        shapes: list[Any],
        input_types: list[Any],
    ) -> list[tuple[int, str, Any, str]]:
        """Apply the reviewed functional-linear tensor roles."""
        parameter_indices = self._PARAMETER_TENSOR_INDICES["linear"]
        return [
            (
                index,
                connection,
                shapes[index] if index < len(shapes) else None,
                (
                    "weight"
                    if index in parameter_indices
                    else str(input_types[index])
                    if index < len(input_types)
                    else "input"
                ),
            )
            for index, connection in enumerate(connections)
        ]

    @staticmethod
    def _canonical_linear_input(
        entry: tuple[int, str, Any, str],
        start_node_id_map: dict[str, str],
    ) -> str:
        """Resolve one traced tensor node to its emitted start/op ID."""
        return start_node_id_map.get(entry[1], entry[1])

    @staticmethod
    def _synthetic_tensor_call(input_count: int) -> dict[str, Any]:
        """Return exact positional semantics for a synthetic tensor operation."""
        return {
            "call_arguments": [
                {"tensor": index} for index in range(input_count)
            ],
            "call_kwargs": {},
        }

    @staticmethod
    def _linear_input_connections(
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> list[str]:
        """Collect ordered tensor/start connections for functional linear."""
        connections = list(
            (node_data.get("connections") or {}).get("inputs") or [],
        )
        for predecessor in op_graph.predecessors(node_id):
            if predecessor not in connections:
                connections.append(predecessor)
        for info in start_nodes_info:
            start_id = start_node_id_map.get(info["original_id"])
            if (
                node_id in info.get("consumers", [])
                and start_id
                and start_id not in connections
            ):
                connections.append(start_id)
        return connections

    @staticmethod
    def _select_linear_entries(
        typed_inputs: list[_LinearEntry],
    ) -> tuple[_LinearEntry | None, _LinearEntry | None, _LinearEntry | None]:
        """Select activation, weight matrix, and bias entries."""
        activation = next(
            (
                entry
                for entry in typed_inputs
                if entry[3] != "weight" and "parameter-tensor" not in entry[1]
            ),
            typed_inputs[0] if typed_inputs else None,
        )
        weights = [
            entry
            for entry in typed_inputs
            if entry[3] == "weight" or "parameter-tensor" in entry[1]
        ]
        bias = next(
            (
                entry
                for entry in weights
                if isinstance(entry[2], list) and len(entry[2]) == 1
            ),
            weights[-1] if len(weights) >= 2 else None,
        )
        matrix = next(
            (
                entry
                for entry in weights
                if bias is None
                or entry[1] != bias[1]
                and isinstance(entry[2], list)
                and len(entry[2]) >= 2
            ),
            next(
                (
                    entry
                    for entry in weights
                    if bias is None or entry[1] != bias[1]
                ),
                None,
            ),
        )
        return activation, matrix, bias

    def _linear_split_context(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> _LinearSplitContext:
        """Normalize graph, dtype, and tensor-role inputs for a split."""
        shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        connections = self._linear_input_connections(
            node_id, node_data, op_graph, start_nodes_info, start_node_id_map
        )
        outputs = list(op_graph.successors(node_id))
        if not outputs:
            outputs = [
                item
                for item in (node_data.get("connections") or {}).get(
                    "outputs", []
                )
                if item in op_graph.nodes
            ]
        input_types = node_data.get("input_types") or []
        input_dtypes = node_data.get("input_dtypes") or []
        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (
                input_dtypes[index]
                for index, kind in enumerate(input_types)
                if str(kind).lower() == "weight" and index < len(input_dtypes)
            ),
            activation_dtype,
        )
        output_dtypes = node_data.get("output_dtypes") or []
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        activation, weight, bias = self._select_linear_entries(
            self._linear_typed_inputs(connections, shapes, input_types)
        )
        return _LinearSplitContext(
            node_id=node_id,
            add_node_id=f"{node_id}.bias_add",
            output_connections=outputs,
            output_shapes=cast("list[list[int]]", output_shapes),
            matmul_output_shape=(
                cast("list[int]", output_shapes[0]) if output_shapes else []
            ),
            activation=activation,
            weight=weight,
            bias=bias,
            weight_shape=(
                cast("list[int]", list(weight[2]))
                if weight and isinstance(weight[2], list)
                else None
            ),
            bias_shape=(
                cast("list[int]", list(bias[2]))
                if bias and isinstance(bias[2], list)
                else None
            ),
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            start_node_id_map=start_node_id_map,
        )

    def _linear_matmul_equation(
        self,
        context: _LinearSplitContext,
    ) -> tuple[str, dict[str, list[str]]]:
        """Return the analyzed or fallback matmul equation and operands."""
        inputs: list[list[int]] = [
            cast("list[int]", list(entry[2]))
            for entry in (context.activation, context.weight)
            if entry and isinstance(entry[2], list)
        ]
        tensor_shapes = TensorShapes(
            inputs=inputs, outputs=context.output_shapes
        )
        try:
            operation = self._einsum_analyzer.get_einsum_op(
                "linear", tensor_shapes
            )
            return operation.equation, {
                operand.name: list(operand.dims)
                for operand in operation.operands
            }
        except Exception:  # noqa: BLE001 - optional backend fallback
            input_shape = context.activation[2] if context.activation else []
            batch_count = len(input_shape) - 1 if input_shape else 0
            batch = [f"B{index}" for index in range(batch_count)]
            input_dims = "".join([*batch, "K"])
            output_dims = "".join([*batch, "N"])
            tokens = lambda value: re.findall(r"[A-Z]\d*", value)  # noqa: E731
            return (
                f"{input_dims},NK->{output_dims}",
                {
                    "Input": tokens(input_dims),
                    "Weight": ["N", "K"],
                    "Output": tokens(output_dims),
                },
            )

    def _linear_matmul_layer(
        self,
        context: _LinearSplitContext,
        equation: str,
        operands: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Emit the matmul half of a split linear operation."""
        names: list[str] = []
        shapes: list[list[Any]] = []
        connections: list[str] = []
        for entry in (context.activation, context.weight):
            if entry is None:
                continue
            emitted_id = self._canonical_linear_input(
                entry, context.start_node_id_map
            )
            names.append(f"{emitted_id}.Output")
            connections.append(emitted_id)
            if isinstance(entry[2], list):
                shapes.append(list(entry[2]))
        layer: dict[str, Any] = {
            "type": "linear",
            "einsum_equation": equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": operands,
            "tensor_names": {
                "inputs": names,
                "outputs": [f"{context.node_id}.Output"],
            },
            "tensor_types": {
                "inputs": [
                    "input" if index == 0 else "weight"
                    for index in range(len(names))
                ],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": shapes,
                "outputs": [list(context.matmul_output_shape)]
                if context.matmul_output_shape
                else [],
            },
            "tensor_dtypes": {
                "inputs": [context.activation_dtype, context.weight_dtype],
                "outputs": [context.output_dtype],
            },
            "connections": {
                "inputs": connections,
                "outputs": [context.add_node_id],
            },
        }
        if context.weight_shape:
            layer["additional_info"] = {
                "weights": [
                    {"name": "Weight", "shape": list(context.weight_shape)}
                ]
            }
        return layer

    @staticmethod
    def _bias_add_equation(
        output_shape: list[int],
        bias_shape: list[int] | None,
    ) -> tuple[str, dict[str, list[str]] | None]:
        """Return the broadcast-add equation for the split bias."""
        if output_shape and bias_shape and len(bias_shape) == 1:
            labels = string.ascii_uppercase[: len(output_shape)]
            return (
                f"{labels},{labels[-1]}->{labels}",
                {
                    "Input": list(labels),
                    "Weight": [labels[-1]],
                    "Output": list(labels),
                },
            )
        if output_shape:
            labels = string.ascii_uppercase[: len(output_shape)]
            return (
                f"{labels}->{labels}",
                {"Input": list(labels), "Output": list(labels)},
            )
        return "", None

    def _bias_add_layer(
        self,
        context: _LinearSplitContext,
    ) -> dict[str, Any]:
        """Emit the bias-add half of a split linear operation."""
        equation, operands = self._bias_add_equation(
            context.matmul_output_shape, context.bias_shape
        )
        names = [f"{context.node_id}.Output"]
        shapes = (
            [list(context.matmul_output_shape)]
            if context.matmul_output_shape
            else []
        )
        connections = [context.node_id]
        if context.bias and context.bias_shape:
            bias_id = self._canonical_linear_input(
                context.bias, context.start_node_id_map
            )
            names.append(f"{bias_id}.Output")
            shapes.append(list(context.bias_shape))
            connections.append(bias_id)
        layer: dict[str, Any] = {
            "type": "add",
            "einsum_equation": equation,
            "elementwise_op": "add",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": operands,
            "tensor_names": {
                "inputs": names,
                "outputs": [f"{context.add_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"] + (["weight"] if len(names) > 1 else []),
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": shapes,
                "outputs": [list(context.output_shapes[0])]
                if context.output_shapes
                else [],
            },
            "tensor_dtypes": {
                "inputs": [context.output_dtype, context.weight_dtype],
                "outputs": [context.output_dtype],
            },
            "connections": {
                "inputs": connections,
                "outputs": context.output_connections,
            },
            "module_args": self._synthetic_tensor_call(len(names)),
        }
        if context.bias_shape:
            layer["additional_info"] = {
                "weights": [{"name": "Bias", "shape": list(context.bias_shape)}]
            }
        return layer

    def _split_linear_with_bias(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a biased linear operation into matmul and add layers."""
        context = self._linear_split_context(
            node_id,
            node_data,
            op_graph,
            start_nodes_info,
            start_node_id_map,
        )
        equation, operands = self._linear_matmul_equation(context)
        return (
            self._linear_matmul_layer(context, equation, operands),
            self._bias_add_layer(context),
        )

    def _redirect_expanded_outputs(
        self,
        result: dict[str, Any],
        expanded_input_map: dict[str, dict[int, str]],
    ) -> None:
        """Redirect predecessors to the correct expanded subgraph entries."""
        for original_id, input_mapping in expanded_input_map.items():
            for layer_id, layer in result["layers"].items():
                connections = layer.get("connections", {})
                outputs = connections.get("outputs", [])
                if original_id not in outputs:
                    continue
                connections["outputs"] = [
                    (
                        self._find_entry_node_for_predecessor(
                            result,
                            layer_id,
                            original_id,
                            input_mapping,
                        )
                        if output == original_id
                        else output
                    )
                    for output in outputs
                ]

    @staticmethod
    def _remap_split_tensor_name(
        name: str,
        layer_id: str,
        node_id_remap: dict[str, str],
    ) -> str:
        """Remap one split tensor name without creating a self-loop."""
        for original_id, final_id in node_id_remap.items():
            if final_id == layer_id:
                continue
            if name == f"{original_id}.Output" or name.startswith(
                f"{original_id}.Output_"
            ):
                return name.replace(f"{original_id}.", f"{final_id}.", 1)
        return name

    def _redirect_split_inputs(
        self,
        result: dict[str, Any],
        node_id_remap: dict[str, str],
    ) -> None:
        """Redirect downstream connections and tensor names to final nodes."""
        for layer_id, layer in result["layers"].items():
            connections = layer.get("connections", {})
            connections["inputs"] = [
                (
                    node_id_remap[input_id]
                    if input_id in node_id_remap
                    and node_id_remap[input_id] != layer_id
                    else input_id
                )
                for input_id in connections.get("inputs", [])
            ]
            tensor_names = layer.get("tensor_names", {})
            if tensor_names:
                tensor_names["inputs"] = [
                    self._remap_split_tensor_name(name, layer_id, node_id_remap)
                    for name in tensor_names.get("inputs", [])
                ]

    def _fix_split_connections(
        self,
        result: dict[str, Any],
        node_id_remap: dict[str, str],
        expanded_input_map: dict[str, dict[int, str]] | None = None,
    ) -> None:
        """Reconnect graph edges after split and expanded operations."""
        expanded = expanded_input_map or {}
        if not node_id_remap and not expanded:
            return
        self._redirect_expanded_outputs(result, expanded)
        self._redirect_split_inputs(result, node_id_remap)
