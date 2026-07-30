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

from solar.ir.extended_einsum.equations import (
    validate_tensor_names_match_shapes,
)
from solar.ir.extended_einsum.native_registry import (
    NATIVE_OP_REGISTRY,
    canonical_native_target,
)
from solar.ir.extended_einsum.operations.conversion import (
    REDUCTION_OPS_WITH_DIM,
    OperationRepresentation,
    default_operation_representation,
)
from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)
from solar.ir.extended_einsum.torchview.converter_models import (
    ConversionError,
    ConvertedTensorMetadata,
)
from solar.types import TensorShapes

PathLike = str | Path


class ConverterLayersMixin(ConverterMixinContract):
    """Build normalized extended-einsum layers and connections."""

    def _find_entry_node_for_predecessor(
        self,
        result: dict[str, Any],
        predecessor_id: str,
        original_node_id: str,
        input_mapping: dict[int, str],
    ) -> str:
        """Find which subgraph entry node a predecessor should connect to.

        Args:
            result: The einsum graph dictionary.
            predecessor_id: ID of the predecessor layer.
            original_node_id: ID of the original (expanded) node.
            input_mapping: Maps input index -> subgraph node that receives it.

        Returns:
            The subgraph node ID that this predecessor should connect to.
        """
        # Look at the subgraph nodes to find which one has this predecessor in its inputs
        for subgraph_node_id in input_mapping.values():
            if subgraph_node_id in result["layers"]:
                subgraph_layer = result["layers"][subgraph_node_id]
                subgraph_inputs = subgraph_layer.get("connections", {}).get(
                    "inputs", []
                )
                if predecessor_id in subgraph_inputs:
                    return subgraph_node_id

        # Default: return the first entry node (qk_matmul for SDPA)
        if input_mapping:
            return input_mapping.get(0, next(iter(input_mapping.values())))

        return original_node_id

    def _add_start_nodes(
        self,
        result: dict[str, Any],
        start_nodes_info: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Add start nodes to the einsum graph."""
        start_node_id_map: dict[str, str] = {}

        for info in start_nodes_info:
            idx = info["index"]
            start_id = "start" if idx == 0 else f"start_{idx}"
            original_id = info["original_id"]
            start_node_id_map[original_id] = start_id

            output_shapes = info.get("output_shapes") or []
            consumers = info.get("consumers", [])

            # Build tensor_names
            output_names = [f"{start_id}.Output"]
            for i in range(1, len(output_shapes)):
                output_names.append(f"{start_id}.Output_{i}")

            tensor_names = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": output_names,
            }

            # Build tensor_shapes
            tensor_shapes = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": [list(s) for s in output_shapes],
            }

            # Generate einsum equation
            equation = ""
            operands = {}
            if output_shapes and len(output_shapes[0]) > 0:
                dims = len(output_shapes[0])
                labels = [f"{c}{idx}" for c in string.ascii_uppercase[:dims]]
                equation = f"->{''.join(labels)}"
                operands = {start_id: labels}

            layer_dict: dict[str, Any] = {
                "type": "start",
                "source_tensor_id": original_id,
                "source_input_index": info.get("source_input_index"),
                "source_binding": (
                    "exact_source_index"
                    if info.get("source_input_index") is not None
                    else "unbound"
                ),
                "einsum_equation": equation,
                "elementwise_op": "copy",
                "reduction_op": "none",
                "is_real_einsum": False,
                "is_einsum_supportable": False,
                "tensor_names": tensor_names,
                "tensor_types": {
                    "inputs": [],
                    "outputs": ["input" for _ in output_names],
                },
                "tensor_shapes": tensor_shapes,
                "operands": operands,
                "connections": {
                    "inputs": [],
                    "outputs": consumers,
                },
            }

            # Propagate dtype info for start nodes
            output_dtypes = info.get("output_dtypes") or []
            if output_dtypes:
                layer_dict["tensor_dtypes"] = {
                    "inputs": [],
                    "outputs": output_dtypes,
                }

            result["layers"][start_id] = layer_dict

        return start_node_id_map

    def _explicit_einsum_representation(
        self,
        node_id: str,
        shapes: TensorShapes,
        module_args: dict[str, Any],
    ) -> OperationRepresentation:
        parsed_equation = self._parse_einsum_from_raw_attributes(module_args)
        if parsed_equation:
            lhs, rhs = parsed_equation.split("->")
            lhs_parts = lhs.split(",")
            operands = {"Input": list(lhs_parts[0]), "Output": list(rhs)}
            if len(lhs_parts) == 2:
                operands["Weight"] = list(lhs_parts[1])
            return OperationRepresentation(
                equation=parsed_equation,
                operands=operands,
                elementwise_op="mul",
                reduction_op="add",
                is_real_einsum=True,
                is_einsum_supportable=True,
            )
        try:
            operation = self._einsum_analyzer.get_einsum_op(
                "einsum", shapes, module_args=module_args
            )
        except Exception as exc:
            if self._strict:
                raise ConversionError(
                    f"cannot convert einsum layer {node_id}: {exc}"
                ) from exc
            return default_operation_representation()
        return OperationRepresentation.from_einsum_op(operation)

    def _prepare_reduction_arguments(
        self, node_type: str, module_args: dict[str, Any]
    ) -> tuple[list[int] | None, bool]:
        reduce_dims: list[int] | None = None
        keepdim = False
        if node_type in REDUCTION_OPS_WITH_DIM:
            reduce_dims, keepdim = (
                self._parse_reduction_args_from_raw_attributes(module_args)
            )
            if reduce_dims is not None:
                module_args["dim"] = (
                    reduce_dims[0] if len(reduce_dims) == 1 else reduce_dims
                )
                module_args["keepdim"] = keepdim
        if node_type in {"softmax", "log_softmax"} and "dim" not in module_args:
            softmax_dims, _ = self._parse_reduction_args_from_raw_attributes(
                module_args
            )
            if softmax_dims is not None and len(softmax_dims) == 1:
                module_args["dim"] = softmax_dims[0]
        return reduce_dims, keepdim

    def _failed_operation_representation(
        self, node_id: str, node_type: str, error: Exception
    ) -> OperationRepresentation:
        if (
            self._strict
            and canonical_native_target(node_type) in NATIVE_OP_REGISTRY
        ):
            return OperationRepresentation(
                equation="",
                operands={},
                elementwise_op="none",
                reduction_op="none",
                is_real_einsum=False,
                is_einsum_supportable=True,
            )
        if self._strict:
            raise ConversionError(
                f"cannot exactly convert {node_id} ({node_type}): {error}"
            ) from error

        elementwise_op = "mul"
        reduction_op = "add"
        is_real_einsum = True
        if node_type in {"add", "sub", "mul", "div"}:
            elementwise_op = node_type
            reduction_op = "none"
            is_real_einsum = False
        elif node_type in {"sum", "mean"}:
            elementwise_op = "copy"
            is_real_einsum = False
        elif node_type == "prod":
            elementwise_op = "copy"
            reduction_op = "mul"
            is_real_einsum = False
        elif node_type in {"max", "min"}:
            elementwise_op = "copy"
            reduction_op = node_type
            is_real_einsum = False
        return OperationRepresentation(
            equation="",
            operands={},
            elementwise_op=elementwise_op,
            reduction_op=reduction_op,
            is_real_einsum=is_real_einsum,
            is_einsum_supportable=self._is_operation_supportable(node_type),
        )

    def _standard_operation_representation(
        self,
        node_id: str,
        node_type: str,
        shapes: TensorShapes,
        module_args: dict[str, Any],
    ) -> OperationRepresentation:
        reduce_dims, keepdim = self._prepare_reduction_arguments(
            node_type, module_args
        )
        try:
            if reduce_dims is not None:
                operation = self._einsum_analyzer.get_einsum_op(
                    node_type,
                    shapes,
                    module_args=module_args,
                    dims=reduce_dims,
                    keepdim=keepdim,
                )
            else:
                operation = self._einsum_analyzer.get_einsum_op(
                    node_type,
                    shapes,
                    module_args=module_args,
                    keepdim=keepdim,
                )
        except Exception as exc:  # noqa: BLE001 - optional backend fallback
            return self._failed_operation_representation(
                node_id, node_type, exc
            )
        return OperationRepresentation.from_einsum_op(operation)

    def _operation_representation(
        self,
        node_id: str,
        node_type: str,
        shapes: TensorShapes,
        module_args: dict[str, Any],
    ) -> OperationRepresentation:
        if node_type == "einsum":
            return self._explicit_einsum_representation(
                node_id, shapes, module_args
            )
        return self._standard_operation_representation(
            node_id, node_type, shapes, module_args
        )

    @staticmethod
    def _collect_raw_input_connections(
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
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in connections:
                    connections.append(original_id)
        return connections

    @staticmethod
    def _fill_deferred_input_connections(
        node_id: str,
        connections: list[str | None],
        deferred_indices: list[int],
        raw_connections: list[str],
        unmatched_predecessors: list[str],
        start_node_id_map: dict[str, str],
    ) -> None:
        if len(unmatched_predecessors) == len(deferred_indices):
            for index, predecessor in zip(
                deferred_indices, unmatched_predecessors, strict=False
            ):
                connections[index] = predecessor
            return
        if not unmatched_predecessors:
            for index in deferred_indices:
                raw_connection = raw_connections[index]
                connections[index] = start_node_id_map.get(
                    raw_connection, raw_connection
                )
            return
        unresolved = [raw_connections[index] for index in deferred_indices]
        raise ValueError(
            f"_convert_operation({node_id!r}): cannot uniquely resolve inputs "
            f"{unresolved}. deferred={len(deferred_indices)}, unmatched_preds="
            f"{len(unmatched_predecessors)}. Producer attribution is ambiguous — "
            "usually a torchview tracing gap that _build_op_graph's reconciliation "
            "pass couldn't shape-match uniquely."
        )

    def _resolve_input_connections(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        raw_connections: list[str],
        start_node_id_map: dict[str, str],
    ) -> list[str]:
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        input_types = list(node_data.get("input_types") or [])
        connections: list[str | None] = []
        assigned_predecessors: set[str] = set()
        deferred_indices: list[int] = []
        for index, connection_id in enumerate(raw_connections):
            mapped = start_node_id_map.get(connection_id, connection_id)
            input_type = (
                str(input_types[index]).lower()
                if index < len(input_types)
                else "input"
            )
            if input_type == "weight":
                connections.append(mapped)
            elif (
                mapped in start_node_id_map.values() or mapped in op_graph.nodes
            ):
                connections.append(mapped)
                assigned_predecessors.add(mapped)
            elif (
                producer := tensor_to_producer.get(connection_id)
            ) in op_graph.nodes:
                producer_id = str(producer)
                connections.append(producer_id)
                assigned_predecessors.add(producer_id)
            else:
                connections.append(None)
                deferred_indices.append(index)
        if deferred_indices:
            unmatched = [
                predecessor
                for predecessor in op_graph.predecessors(node_id)
                if predecessor not in assigned_predecessors
            ]
            self._fill_deferred_input_connections(
                node_id,
                connections,
                deferred_indices,
                raw_connections,
                unmatched,
                start_node_id_map,
            )
        if any(connection is None for connection in connections):
            raise ValueError(
                f"{node_id}: unresolved tensor input after reconciliation"
            )
        return [str(connection) for connection in connections]

    def _build_converted_tensor_metadata(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        input_connections: list[str],
        raw_connections: list[str],
    ) -> ConvertedTensorMetadata:
        input_types = list(node_data.get("input_types") or [])
        if len(input_types) < len(input_connections):
            input_types.extend(
                ["input"] * (len(input_connections) - len(input_types))
            )
        typed_node_data = {**node_data, "input_types": input_types}
        output_connections = sorted(op_graph.successors(node_id))
        tensor_names = self._build_tensor_names(
            node_id,
            typed_node_data,
            input_connections,
            raw_connections,
            output_connections,
        )
        output_types = list(node_data.get("output_types") or [])
        output_count = len(tensor_names.get("outputs", []))
        if len(output_types) < output_count:
            output_types.extend(["output"] * (output_count - len(output_types)))
        tensor_types = {
            "inputs": list(input_types[: len(tensor_names.get("inputs", []))]),
            "outputs": list(output_types[:output_count]),
        }
        tensor_shapes = self._build_tensor_shapes(node_data)
        is_valid, _ = validate_tensor_names_match_shapes(
            tensor_names, tensor_shapes
        )
        if not is_valid:
            tensor_names, tensor_shapes = self._align_tensor_names_and_shapes(
                tensor_names, tensor_shapes, node_data
            )
            tensor_types["inputs"] = tensor_types["inputs"][
                : len(tensor_names.get("inputs", []))
            ]
            tensor_types["outputs"] = tensor_types["outputs"][
                : len(tensor_names.get("outputs", []))
            ]
        activation_connections = [
            connection
            for index, connection in enumerate(input_connections)
            if not (
                index < len(input_types)
                and str(input_types[index]).lower() == "weight"
            )
        ]
        tensor_dtypes: dict[str, Any] = {}
        if input_dtypes := list(node_data.get("input_dtypes") or []):
            tensor_dtypes["inputs"] = input_dtypes
        if output_dtypes := list(node_data.get("output_dtypes") or []):
            tensor_dtypes["outputs"] = output_dtypes
        return ConvertedTensorMetadata(
            tensor_names=tensor_names,
            tensor_types=tensor_types,
            tensor_shapes=tensor_shapes,
            tensor_dtypes=tensor_dtypes,
            activation_connections=activation_connections,
            output_connections=output_connections,
            additional_info=self._build_additional_info(node_data),
        )

    def _build_converted_layer(
        self,
        node_type_raw: Any,
        node_type: str,
        module_args: dict[str, Any],
        representation: OperationRepresentation,
        metadata: ConvertedTensorMetadata,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": node_type,
            "einsum_equation": representation.equation,
            "elementwise_op": representation.elementwise_op,
            "reduction_op": representation.reduction_op,
            "is_real_einsum": representation.is_real_einsum,
            "is_einsum_supportable": representation.is_einsum_supportable,
            "tensor_names": metadata.tensor_names,
            "tensor_types": metadata.tensor_types,
            "tensor_shapes": metadata.tensor_shapes,
            "operands": representation.operands,
            "connections": {
                "inputs": metadata.activation_connections,
                "outputs": metadata.output_connections,
            },
        }
        source_target = str(node_type_raw).lower().rsplit(".", maxsplit=1)[-1]
        source_target = re.sub(r"_\d+$", "", source_target)
        if source_target.endswith("_") and not source_target.endswith("__"):
            result["mutates_inputs"] = True
        semantic_args = {
            str(key): value
            for key, value in module_args.items()
            if key != "raw_attributes"
            and value is not None
            and isinstance(value, (bool, int, float, str, list, tuple, dict))
        }
        if semantic_args:
            result["module_args"] = semantic_args
        if metadata.tensor_dtypes:
            result["tensor_dtypes"] = metadata.tensor_dtypes
        if metadata.additional_info:
            result["additional_info"] = metadata.additional_info
        if raw_attributes := module_args.get("raw_attributes"):
            result["raw_attributes"] = raw_attributes
        return result

    def _convert_operation(
        self,
        node_id: str,
        node_data: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        start_node_id_map: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a single operation through explicit typed stages."""
        node_type_raw = node_data.get("type", "unknown")
        node_type = self._einsum_analyzer._get_operation_from_name(
            str(node_type_raw)
        )
        module_args = dict(node_data.get("module_args", {}) or {})
        representation = self._operation_representation(
            node_id,
            node_type,
            TensorShapes(
                inputs=list(node_data.get("input_shapes") or []),
                outputs=list(node_data.get("output_shapes") or []),
            ),
            module_args,
        )
        raw_connections = self._collect_raw_input_connections(
            node_id, node_data, op_graph, start_nodes_info
        )
        input_connections = self._resolve_input_connections(
            node_id, node_data, op_graph, raw_connections, start_node_id_map
        )
        metadata = self._build_converted_tensor_metadata(
            node_id,
            node_data,
            op_graph,
            input_connections,
            raw_connections,
        )
        return self._build_converted_layer(
            node_type_raw, node_type, module_args, representation, metadata
        )

    def _build_tensor_names(
        self,
        node_id: str,
        node_data: dict[str, Any],
        input_connections: list[str],
        raw_connections: list[str],
        output_connections: list[str],
    ) -> dict[str, list[str]]:
        """Build tensor names matching input_shapes/output_shapes order.

        Uses input_types to name weight inputs as <node_id>.Weight
        and activation inputs as <predecessor_id>.Output.
        """
        input_names: list[str] = []
        output_names: list[str] = []
        input_types = node_data.get("input_types") or []

        weight_idx = 0
        for i, pred_id in enumerate(input_connections):
            itype = input_types[i] if i < len(input_types) else "input"
            if itype == "weight":
                name = (
                    f"{node_id}.Weight"
                    if weight_idx == 0
                    else f"{node_id}.Weight_{weight_idx}"
                )
                input_names.append(name)
                weight_idx += 1
            else:
                raw_id = (
                    raw_connections[i] if i < len(raw_connections) else pred_id
                )
                producer = self._tensor_to_producer_op.get(raw_id)
                if producer is None:
                    input_names.append(f"{pred_id}.Output")
                else:
                    slot = self._tensor_to_producer_slot.get(raw_id, 0)
                    suffix = "" if slot == 0 else f"_{slot}"
                    input_names.append(f"{producer}.Output{suffix}")

        # Output tensors
        output_names.append(f"{node_id}.Output")
        output_shapes = node_data.get("output_shapes") or []
        for i in range(1, len(output_shapes)):
            output_names.append(f"{node_id}.Output_{i}")

        return {
            "inputs": input_names,
            "outputs": output_names,
        }

    def _build_tensor_shapes(
        self,
        node_data: dict[str, Any],
    ) -> dict[str, list[list[int]]]:
        """Build tensor shapes matching input_shapes/output_shapes order.

        All inputs (activation + weight) are already in input_shapes in arg order.
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        return {
            "inputs": [list(s) for s in input_shapes],
            "outputs": [list(s) for s in output_shapes],
        }

    def _align_tensor_names_and_shapes(
        self,
        tensor_names: dict[str, list[str]],
        tensor_shapes: dict[str, list[list[int]]],
        node_data: dict[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, list[list[int]]]]:
        """Align tensor_names and tensor_shapes to have matching counts.

        When there's a mismatch (e.g., weight_nodes vs weight_shapes have different lengths),
        this method aligns them by using the shapes as the source of truth and generating
        placeholder names if needed, or trimming excess names.
        """
        input_names = tensor_names.get("inputs", [])
        output_names = tensor_names.get("outputs", [])
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])

        # Align inputs
        if len(input_names) != len(input_shapes):
            # Use shapes as source of truth
            if len(input_shapes) > len(input_names):
                # Add placeholder names for missing entries
                node_id = node_data.get("id", "unknown")
                for i in range(len(input_names), len(input_shapes)):
                    input_names.append(f"{node_id}.Input_{i}")
            else:
                # Trim excess names
                input_names = input_names[: len(input_shapes)]

        # Align outputs
        if len(output_names) != len(output_shapes):
            if len(output_shapes) > len(output_names):
                node_id = node_data.get("id", "unknown")
                for i in range(len(output_names), len(output_shapes)):
                    output_names.append(f"{node_id}.Output_{i}")
            else:
                output_names = output_names[: len(output_shapes)]

        return (
            {"inputs": input_names, "outputs": output_names},
            {"inputs": input_shapes, "outputs": output_shapes},
        )

    def _build_additional_info(
        self,
        node_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build additional_info metadata (weight info is in tensor_names/tensor_shapes)."""
        return {}

    def _is_operation_supportable(self, op_type: str) -> bool:
        """Check if extended-einsum can express an operation."""
        op = op_type.lower()

        canonical = canonical_native_target(op)
        return canonical in NATIVE_OP_REGISTRY
