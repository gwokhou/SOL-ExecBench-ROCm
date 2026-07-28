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
from pathlib import Path
from typing import Any

import networkx as nx

from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)
from solar.types import TensorShapes

PathLike = str | Path


class ConverterNormalizationMixin(ConverterMixinContract):
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

    def _split_linear_with_bias(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a linear layer with bias into matmul + add operations.

        Returns:
            Tuple of (matmul_layer_dict, add_layer_dict)
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        # Keep original input order from PyTorch graph; don't sort.
        node_connections = (node_data.get("connections") or {}).get(
            "inputs"
        ) or []
        input_connections = list(node_connections)
        for pred in op_graph.predecessors(node_id):
            if pred not in input_connections:
                input_connections.append(pred)
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)

        # Use collapsed op-graph successors so tensor nodes (e.g. hidden-tensor)
        # are not emitted in einsum connections.
        output_connections = list(op_graph.successors(node_id))
        if not output_connections:
            raw_output_connections = list(
                (node_data.get("connections") or {}).get("outputs") or []
            )
            output_connections = [
                c for c in raw_output_connections if c in op_graph.nodes
            ]

        # Extract dtypes from the original node for propagation to sub-nodes.
        input_types = node_data.get("input_types") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (
                input_dtypes[i]
                for i, t in enumerate(input_types)
                if str(t).lower() == "weight" and i < len(input_dtypes)
            ),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        # Infer x/weight/bias from ordered inputs + input_shapes.
        typed_inputs: list[tuple[int, str, Any, str]] = []
        for idx, conn in enumerate(input_connections):
            ishape = input_shapes[idx] if idx < len(input_shapes) else None
            itype = input_types[idx] if idx < len(input_types) else "input"
            typed_inputs.append((idx, conn, ishape, str(itype)))

        activation_entry: tuple[int, str, Any, str] | None = None
        weight_entries: list[tuple[int, str, Any, str]] = []
        for entry in typed_inputs:
            _, conn, _, itype = entry
            if itype == "weight" or "parameter-tensor" in conn:
                weight_entries.append(entry)
            elif activation_entry is None:
                activation_entry = entry

        if activation_entry is None and typed_inputs:
            activation_entry = typed_inputs[0]

        # Bias is normally rank-1 among weight inputs.
        bias_entry: tuple[int, str, Any, str] | None = None
        for entry in weight_entries:
            ishape = entry[2]
            if isinstance(ishape, list) and len(ishape) == 1:
                bias_entry = entry
                break

        # Fallback when rank-based inference fails: last weight is bias.
        if bias_entry is None and len(weight_entries) >= 2:
            bias_entry = weight_entries[-1]

        # Weight matrix is a non-bias weight, preferring rank-2.
        weight_entry: tuple[int, str, Any, str] | None = None
        for entry in weight_entries:
            if bias_entry is not None and entry[1] == bias_entry[1]:
                continue
            ishape = entry[2]
            if isinstance(ishape, list) and len(ishape) >= 2:
                weight_entry = entry
                break
        if weight_entry is None:
            for entry in weight_entries:
                if bias_entry is None or entry[1] != bias_entry[1]:
                    weight_entry = entry
                    break

        weight_shape = (
            list(weight_entry[2])
            if (weight_entry and isinstance(weight_entry[2], list))
            else None
        )
        bias_shape = (
            list(bias_entry[2])
            if (bias_entry and isinstance(bias_entry[2], list))
            else None
        )

        # Intermediate shape (output of matmul, input to add)
        matmul_output_shape = output_shapes[0] if output_shapes else []

        # === MATMUL LAYER ===
        # Get einsum equation for matmul
        matmul_input_shapes_for_equation: list[list[Any]] = []
        if activation_entry and isinstance(activation_entry[2], list):
            matmul_input_shapes_for_equation.append(list(activation_entry[2]))
        if weight_entry and isinstance(weight_entry[2], list):
            matmul_input_shapes_for_equation.append(list(weight_entry[2]))
        matmul_ts = TensorShapes(
            inputs=matmul_input_shapes_for_equation,
            outputs=list(node_data.get("output_shapes") or []),
        )
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op("linear", matmul_ts)
            matmul_equation = einsum_op.equation
            matmul_operands = {o.name: list(o.dims) for o in einsum_op.operands}
        except Exception:  # noqa: BLE001 - optional backend fallback
            # Fallback equation
            batch_dims = len(input_shapes[0]) - 1 if input_shapes else 0
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            input_str = "".join(batch_letters + ["K"])
            weight_str = "NK"
            output_str = "".join(batch_letters + ["N"])
            matmul_equation = f"{input_str},{weight_str}->{output_str}"

            # Parse the fallback equation back into the operand structure.
            # Tokens are an uppercase letter optionally followed by digits (e.g. B0).
            def _toks(s: str) -> list[str]:
                return re.findall(r"[A-Z]\d*", s)

            matmul_operands = {
                "Input": _toks(input_str),
                "Weight": _toks(weight_str),
                "Output": _toks(output_str),
            }

        add_node_id = f"{node_id}.bias_add"

        matmul_input_names: list[str] = []
        matmul_input_shapes_list: list[list[Any]] = []
        matmul_connection_inputs: list[str] = []

        if activation_entry:
            activation_conn_id = activation_entry[1]
            # Activation tensors should reference the canonical start node IDs
            # (e.g. start/start_1) after tensor-node collapse.
            activation_einsum_id = start_node_id_map.get(
                activation_conn_id, activation_conn_id
            )
            matmul_input_names.append(f"{activation_einsum_id}.Output")
            if isinstance(activation_entry[2], list):
                matmul_input_shapes_list.append(list(activation_entry[2]))
            matmul_connection_inputs.append(activation_einsum_id)
        if weight_entry:
            matmul_input_names.append(f"{weight_entry[1]}.Output")
            if isinstance(weight_entry[2], list):
                matmul_input_shapes_list.append(list(weight_entry[2]))
            matmul_connection_inputs.append(weight_entry[1])

        matmul_tensor_names = {
            "inputs": matmul_input_names,
            "outputs": [f"{node_id}.Output"],
        }
        matmul_tensor_types = {
            "inputs": [
                "input" if i == 0 else "weight"
                for i in range(len(matmul_input_names))
            ],
            "outputs": ["output"],
        }
        matmul_tensor_shapes = {
            "inputs": matmul_input_shapes_list,
            "outputs": [list(matmul_output_shape)]
            if matmul_output_shape
            else [],
        }

        matmul_layer: dict[str, Any] = {
            # Keep type as linear so MACs are computed by LinearHandler.
            "type": "linear",
            "einsum_equation": matmul_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            # Operands drive the AF graph builder; without them the layer
            # is silently skipped from the AF einsums list.
            "operands": matmul_operands,
            "tensor_names": matmul_tensor_names,
            "tensor_types": matmul_tensor_types,
            "tensor_shapes": matmul_tensor_shapes,
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": matmul_connection_inputs,
                "outputs": [add_node_id],  # Output goes to bias_add
            },
        }

        if weight_shape:
            matmul_layer["additional_info"] = {
                "weights": [{"name": "Weight", "shape": list(weight_shape)}]
            }

        # === ADD (BIAS) LAYER ===
        # Generate einsum equation for bias add (broadcast add).
        if (
            matmul_output_shape
            and bias_shape
            and len(matmul_output_shape) >= 1
            and len(bias_shape) == 1
        ):
            labels = string.ascii_uppercase[: len(matmul_output_shape)]
            add_equation = f"{labels},{labels[-1]}->{labels}"
            add_operands = {
                "Input": list(labels),
                "Weight": [labels[-1]],
                "Output": list(labels),
            }
        elif matmul_output_shape:
            labels = string.ascii_uppercase[: len(matmul_output_shape)]
            add_equation = f"{labels}->{labels}"
            add_operands = {
                "Input": list(labels),
                "Output": list(labels),
            }
        else:
            add_equation = ""
            add_operands = None

        add_input_names = [f"{node_id}.Output"]
        add_input_shapes_list = (
            [list(matmul_output_shape)] if matmul_output_shape else []
        )
        add_connection_inputs = [node_id]

        if bias_entry and bias_shape:
            add_input_names.append(f"{bias_entry[1]}.Output")
            add_input_shapes_list.append(list(bias_shape))
            add_connection_inputs.append(bias_entry[1])

        add_tensor_names = {
            "inputs": add_input_names,
            "outputs": [f"{add_node_id}.Output"],
        }
        add_tensor_types = {
            "inputs": ["input"]
            + (["weight"] if len(add_input_names) > 1 else []),
            "outputs": ["output"],
        }
        add_tensor_shapes = {
            "inputs": add_input_shapes_list,
            "outputs": [list(output_shapes[0])] if output_shapes else [],
        }

        add_layer: dict[str, Any] = {
            "type": "add",
            "einsum_equation": add_equation,
            "elementwise_op": "add",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": add_operands,
            "tensor_names": add_tensor_names,
            "tensor_types": add_tensor_types,
            "tensor_shapes": add_tensor_shapes,
            "tensor_dtypes": {
                "inputs": [out_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": add_connection_inputs,
                "outputs": output_connections,  # Original outputs
            },
        }

        # Add bias info
        if bias_shape:
            add_layer["additional_info"] = {
                "weights": [{"name": "Bias", "shape": list(bias_shape)}]
            }

        return matmul_layer, add_layer

    def _fix_split_connections(
        self,
        result: dict[str, Any],
        node_id_remap: dict[str, str],
        expanded_input_map: dict[str, dict[int, str]] | None = None,
    ) -> None:
        """Fix connections for layers that reference split/expanded operations.

        When an operation is split/expanded:
        1. Downstream layers that consume the output should reference the final node
        2. Upstream layers (predecessors) should have their outputs updated to
           reference the correct subgraph entry node

        Args:
            result: The einsum graph dictionary being built.
            node_id_remap: Maps original node_id -> final output node_id.
            expanded_input_map: Maps original node_id -> {input_index -> subgraph_node_id}.
        """
        if expanded_input_map is None:
            expanded_input_map = {}

        if not node_id_remap and not expanded_input_map:
            return

        # First pass: Update predecessor outputs for expanded operations
        for original_node_id, input_mapping in expanded_input_map.items():
            # Find all layers that have the original_node_id in their outputs
            for layer_id, layer_data in result["layers"].items():
                connections = layer_data.get("connections", {})
                outputs = connections.get("outputs", [])

                if original_node_id in outputs:
                    # This layer was a predecessor to the expanded node
                    # Find which input index this layer corresponds to
                    # by looking at the subgraph's inputs
                    new_outputs = []
                    for out in outputs:
                        if out == original_node_id:
                            # Determine which subgraph node this layer feeds into
                            # based on which input it provides
                            # We need to find the correct entry node
                            target_node = self._find_entry_node_for_predecessor(
                                result,
                                layer_id,
                                original_node_id,
                                input_mapping,
                            )
                            new_outputs.append(target_node)
                        else:
                            new_outputs.append(out)
                    connections["outputs"] = new_outputs

        # Second pass: Update downstream references
        for layer_id, layer_data in result["layers"].items():
            connections = layer_data.get("connections", {})
            inputs = connections.get("inputs", [])

            # Update input connections to reference final output node
            new_inputs = []
            for inp in inputs:
                # BUGFIX: Don't remap if the current layer is itself the target of the remapping
                # (e.g., don't replace Model.linear -> Model.linear.bias_add in Model.linear.bias_add's own inputs)
                # This prevents creating self-loops in split layers like bias_add
                if inp in node_id_remap and node_id_remap[inp] != layer_id:
                    new_inputs.append(node_id_remap[inp])
                else:
                    new_inputs.append(inp)
            connections["inputs"] = new_inputs

            # Update tensor_names inputs
            tensor_names = layer_data.get("tensor_names", {})
            if tensor_names:
                input_names = tensor_names.get("inputs", [])
                new_input_names = []
                for input_name in input_names:
                    name = input_name
                    for old_id, new_id in node_id_remap.items():
                        # Keep split node self-inputs stable (e.g. bias_add should
                        # consume Model.linear.Output, not its own output).
                        if new_id == layer_id:
                            continue
                        if name == f"{old_id}.Output" or name.startswith(
                            f"{old_id}.Output_"
                        ):
                            name = name.replace(f"{old_id}.", f"{new_id}.", 1)
                            break
                    new_input_names.append(name)
                tensor_names["inputs"] = new_input_names
