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

"""Exact semantic serialization for make_fx reference graphs."""

from __future__ import annotations

import operator
from typing import Any

import torch


class ReferenceGraphSerializer:
    """Serialize the canonical make_fx graph, including explicit backward ops."""

    def serialize_fx_reference(
        self,
        graph_module: Any,
        model_name: str,
    ) -> dict[str, Any]:
        """Serialize a make_fx-captured reference, including explicit backward ops."""
        return self._serialize_graph(graph_module, model_name)

    @staticmethod
    def _graph_output_values(nodes: list[Any]) -> list[Any]:
        output_node = next(node for node in nodes if node.op == "output")
        raw_outputs = output_node.args[0]
        return (
            list(raw_outputs)
            if isinstance(raw_outputs, (tuple, list))
            else [raw_outputs]
        )

    @staticmethod
    def _serialize_argument(value: Any, inputs: list[Any]) -> Any:
        import torch.fx

        if isinstance(value, torch.fx.Node):
            return {"tensor": inputs.index(value)}
        if isinstance(value, (tuple, list)):
            return [
                ReferenceGraphSerializer._serialize_argument(item, inputs)
                for item in value
            ]
        if isinstance(value, torch.dtype):
            return {"dtype": str(value).replace("torch.", "")}
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if isinstance(value, torch.layout):
            return {"layout": str(value).removeprefix("torch.")}
        if value is torch.preserve_format:
            return "preserve_format"
        if value is torch.contiguous_format:
            return "contiguous_format"
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"value": value}
        return {"value": str(value)}

    @staticmethod
    def _tensor_metadata(value: Any) -> list[tuple[list[int], str]]:
        if isinstance(value, torch.Tensor):
            return [(list(value.shape), str(value.dtype))]
        if isinstance(value, (tuple, list)):
            result: list[tuple[list[int], str]] = []
            for item in value:
                result.extend(ReferenceGraphSerializer._tensor_metadata(item))
            return result
        return []

    @classmethod
    def _canonical_aten_target(
        cls,
        node: Any,
        target_name: str,
        input_nodes: list[Any],
    ) -> str:
        aliases = {
            "_log_softmax": "log_softmax",
            "_safe_softmax": "softmax",
            "_softmax": "softmax",
            "_unsafe_view": "view",
            "max": "amax",
            "min": "amin",
            "native_batch_norm": "batch_norm",
            "native_group_norm": "group_norm",
            "native_layer_norm": "layer_norm",
            "t": "transpose",
        }
        if target_name != "convolution":
            return aliases.get(target_name.rstrip("_"), target_name.rstrip("_"))
        if not input_nodes:
            raise RuntimeError("ATen convolution has no tensor input")
        metadata = cls._tensor_metadata(input_nodes[0].meta.get("val"))
        if not metadata or len(metadata[0][0]) not in {3, 4, 5}:
            raise RuntimeError("ATen convolution input rank is unsupported")
        dimensions = len(metadata[0][0]) - 2
        transposed = bool(node.args[6]) if len(node.args) > 6 else False
        prefix = "conv_transpose" if transposed else "conv"
        return f"{prefix}{dimensions}d"

    @staticmethod
    def _schema_effects(
        node: Any,
        input_nodes: list[Any],
        *,
        target_name: str,
        exact_target: str,
        output_arity: int,
    ) -> dict[str, Any]:
        """Translate an ATen FunctionSchema alias contract to executable IR."""
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            raise RuntimeError(
                f"AOT target has no FunctionSchema: {node.target}",
            )

        def tensor_indices(value: Any) -> set[int]:
            import torch.fx

            if isinstance(value, torch.fx.Node):
                return {input_nodes.index(value)}
            if isinstance(value, (tuple, list)):
                return {
                    index for item in value for index in tensor_indices(item)
                }
            if isinstance(value, dict):
                return {
                    index
                    for item in value.values()
                    for index in tensor_indices(item)
                }
            return set()

        positional = list(node.args)
        aliases_by_input: dict[int, set[str]] = {}
        mutations: set[int] = set()
        for position, argument in enumerate(schema.arguments):
            if position < len(positional):
                value = positional[position]
            elif argument.name in node.kwargs:
                value = node.kwargs[argument.name]
            else:
                continue
            indices = tensor_indices(value)
            alias_info = argument.alias_info
            if alias_info is None:
                continue
            for index in indices:
                aliases_by_input.setdefault(index, set()).update(
                    str(item) for item in alias_info.before_set
                )
                aliases_by_input[index].update(
                    str(item) for item in alias_info.after_set
                )
                if alias_info.is_write:
                    mutations.add(index)

        aliases: list[dict[str, int]] = []
        for output_index, returned in enumerate(schema.returns):
            if output_index >= output_arity or returned.alias_info is None:
                continue
            returned_sets = {
                *(str(item) for item in returned.alias_info.before_set),
                *(str(item) for item in returned.alias_info.after_set),
            }
            for input_index, input_sets in aliases_by_input.items():
                if returned_sets & input_sets:
                    aliases.append(
                        {"output": output_index, "input": input_index},
                    )

        if target_name.endswith("_") and not mutations and input_nodes:
            raise RuntimeError(
                f"mutating ATen target lacks a schema write effect: {node.target}",
            )
        return {
            "mutates": sorted(mutations),
            "aliases": aliases,
            "atomic": exact_target in {"scatter", "index_put", "index_add"},
            "opaque_library_call": False,
        }

    @staticmethod
    def _start_layer(
        node: Any,
        metadata: list[tuple[list[int], str]],
    ) -> dict[str, Any]:
        if len(metadata) != 1:
            raise RuntimeError(f"FX placeholder {node.name} is not one tensor")
        return {
            "type": "start",
            "phase": "input",
            "semantic_op": {
                "kind": "input",
                "target": "input",
                "arguments": [],
                "kwargs": {},
            },
            "tensor_names": {"inputs": [], "outputs": [node.name]},
            "tensor_shapes": {"inputs": [], "outputs": [metadata[0][0]]},
            "tensor_dtypes": {"inputs": [], "outputs": [metadata[0][1]]},
            "connections": {
                "inputs": [],
                "outputs": [item.name for item in node.users],
            },
        }

    @staticmethod
    def _getitem_source_name(
        node: Any,
        node_output_names: dict[Any, list[str]],
    ) -> str:
        import torch.fx

        source, selected = node.args
        if not isinstance(source, torch.fx.Node) or not isinstance(
            selected,
            int,
        ):
            raise RuntimeError(
                f"FX getitem {node.name} is not a fixed tensor selection",
            )
        try:
            return node_output_names.get(source, [])[selected]
        except IndexError as exc:
            raise RuntimeError(
                f"FX getitem {node.name} selects an unavailable output",
            ) from exc

    @staticmethod
    def _identity_layer(
        node: Any,
        metadata: list[tuple[list[int], str]],
        selected_name: str,
    ) -> dict[str, Any]:
        if len(metadata) != 1:
            raise RuntimeError(f"FX getitem {node.name} is not one tensor")
        return {
            "type": "identity",
            "phase": "reference",
            "semantic_op": {
                "kind": "aten",
                "target": "identity",
                "overload": "default",
                "arguments": [{"tensor": 0}],
                "kwargs": {},
                "effects": {
                    "mutates": [],
                    "aliases": [{"output": 0, "input": 0}],
                    "atomic": False,
                    "opaque_library_call": False,
                },
            },
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "einsum_equation": "",
            "elementwise_op": "none",
            "reduction_op": "none",
            "tensor_names": {"inputs": [selected_name], "outputs": [node.name]},
            "tensor_shapes": {
                "inputs": [metadata[0][0]],
                "outputs": [metadata[0][0]],
            },
            "tensor_dtypes": {
                "inputs": [metadata[0][1]],
                "outputs": [metadata[0][1]],
            },
            "connections": {
                "inputs": [node.args[0].name],
                "outputs": [
                    item.name for item in node.users if item.op != "output"
                ],
            },
        }

    @classmethod
    def _tensor_fields(
        cls,
        node: Any,
        input_nodes: list[Any],
        output_tensor_names: list[str],
        metadata: list[tuple[list[int], str]],
        node_output_names: dict[Any, list[str]],
    ) -> dict[str, Any]:
        return {
            "tensor_names": {
                "inputs": [
                    node_output_names[predecessor][0]
                    for predecessor in input_nodes
                ],
                "outputs": output_tensor_names,
            },
            "tensor_shapes": {
                "inputs": [
                    shape
                    for predecessor in input_nodes
                    for shape, _ in cls._tensor_metadata(
                        predecessor.meta.get("val"),
                    )
                ],
                "outputs": [shape for shape, _ in metadata],
            },
            "tensor_dtypes": {
                "inputs": [
                    dtype
                    for predecessor in input_nodes
                    for _, dtype in cls._tensor_metadata(
                        predecessor.meta.get("val"),
                    )
                ],
                "outputs": [dtype for _, dtype in metadata],
            },
            "connections": {
                "inputs": [predecessor.name for predecessor in input_nodes],
                "outputs": [
                    item.name for item in node.users if item.op != "output"
                ],
            },
        }

    def _aten_layer(
        self,
        node: Any,
        metadata: list[tuple[list[int], str]],
        node_output_names: dict[Any, list[str]],
    ) -> tuple[list[str], dict[str, Any]]:
        target_text = str(node.target)
        parts = target_text.split(".")
        if len(parts) < 3 or parts[-3] != "aten":
            raise RuntimeError(
                f"FX graph contains non-ATen target: {target_text}",
            )
        target_name, overload = parts[-2:]
        input_nodes = list(node.all_input_nodes)
        exact_target = self._canonical_aten_target(
            node,
            target_name,
            input_nodes,
        )
        if any(
            len(node_output_names.get(predecessor, [])) != 1
            for predecessor in input_nodes
        ):
            raise RuntimeError(
                f"FX node {node.name} consumes a structured tensor value; "
                "explicit getitem lowering is required",
            )
        output_names = (
            [node.name]
            if len(metadata) == 1
            else [f"{node.name}.{index}" for index in range(len(metadata))]
        )
        if not output_names:
            raise RuntimeError(f"FX node {node.name} has no tensor outputs")
        semantic = {
            "kind": "aten",
            "target": exact_target,
            "exact_target": target_name,
            "overload": overload,
            "arguments": [
                self._serialize_argument(item, input_nodes)
                for item in node.args
            ],
            "kwargs": {
                str(key): self._serialize_argument(value, input_nodes)
                for key, value in node.kwargs.items()
            },
            "effects": self._schema_effects(
                node,
                input_nodes,
                target_name=target_name,
                exact_target=exact_target,
                output_arity=len(metadata),
            ),
        }
        layer = {
            "type": exact_target,
            "phase": "reference",
            "semantic_op": semantic,
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "einsum_equation": "",
            "elementwise_op": "none",
            "reduction_op": "none",
            **self._tensor_fields(
                node,
                input_nodes,
                output_names,
                metadata,
                node_output_names,
            ),
        }
        return output_names, layer

    @staticmethod
    def _graph_signature(
        nodes: list[Any],
        output_names: list[str],
    ) -> dict[str, Any]:
        return {
            "parameters": [],
            "buffers": [],
            "user_inputs": [
                node.name for node in nodes if node.op == "placeholder"
            ],
            "user_outputs": output_names,
            "buffers_to_mutate": {},
            "parameters_to_mutate": {},
            "user_inputs_to_mutate": {},
            "loss_output": None,
            "gradients_to_parameters": {},
            "gradients_to_user_inputs": {},
            "saved_tensors": [],
            "joint_outputs": output_names,
            "gradient_outputs": [],
        }

    def _serialize_graph(
        self,
        graph_module: Any,
        model_name: str,
    ) -> dict[str, Any]:
        import torch.fx

        nodes = list(graph_module.graph.nodes)
        output_values = self._graph_output_values(nodes)
        output_names = [
            node.name
            for node in output_values
            if isinstance(node, torch.fx.Node)
        ]
        if len(output_names) != len(output_values):
            raise RuntimeError("make_fx reference outputs must all be tensors")

        layers: dict[str, Any] = {}
        node_output_names: dict[torch.fx.Node, list[str]] = {}
        for node in nodes:
            if node.op == "output":
                continue
            metadata = self._tensor_metadata(node.meta.get("val"))
            if node.op == "placeholder":
                if not node.users:
                    continue
                node_output_names[node] = [node.name]
                layers[node.name] = self._start_layer(node, metadata)
                continue
            if node.op != "call_function":
                raise RuntimeError(f"unsupported FX node kind: {node.op}")
            if node.target is operator.getitem:
                node_output_names[node] = [node.name]
                selected_name = self._getitem_source_name(
                    node,
                    node_output_names,
                )
                layers[node.name] = self._identity_layer(
                    node,
                    metadata,
                    selected_name,
                )
                continue
            output_tensor_names, layer = self._aten_layer(
                node,
                metadata,
                node_output_names,
            )
            node_output_names[node] = output_tensor_names
            layers[node.name] = layer
        result = {
            "schema_version": 3,
            "model_name": model_name,
            "extraction_kind": "make_fx_reference_v1",
            "joint_graph": False,
            "outputs": output_names,
            "layers": layers,
            "graph_signature": self._graph_signature(nodes, output_names),
        }
        from solar.einsum.semantics import validate_semantic_graph

        validate_semantic_graph(result)
        return result
