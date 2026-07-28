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

import json
import shutil
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from solar.artifacts.yaml import NoAliasDumper
from solar.ir.extended_einsum.torchview.af_graph_builder import (
    build_af_graph_from_dict,
)
from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)
from solar.ir.extended_einsum.torchview.converter_models import (
    ConversionError,
    PathLike,
)
from solar.ir.extended_einsum.torchview.reviewed_handlers import (
    expand_reviewed_handlers,
)
from solar.ir.extended_einsum.torchview.semantics import (
    annotate_semantics,
    validate_semantic_graph,
)
from solar.ir.extended_einsum.torchview.taco import add_taco_expressions


class ConverterPipelineMixin(ConverterMixinContract):
    """Orchestrate loading, conversion, validation, and publication."""

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
        enable_rename: bool = False,
    ) -> dict[str, Any] | None:
        """Convert a PyTorch graph to einsum representation.

        This method:
        1. Loads the PyTorch graph
        2. Builds an operation-only graph (collapsing tensor nodes)
        3. Converts operations to einsum notation
        4. Writes einsum_graph.yaml
        5. Optionally renames ranks using BFS and writes einsum_graph_renamed.yaml

        Args:
            pytorch_graph_path: Path to pytorch_graph.yaml (or legacy JSON).
            output_dir: Directory to write output files.
            copy_graph: If True, copy input graph to output directory.
            expand_complex_ops: If True, attempt to expand complex operations.
            enable_rename: If True, run BFS rank renaming; otherwise copy einsum_graph.yaml as renamed.

        Returns:
            The einsum graph dictionary, or None on failure.
        """
        src = Path(pytorch_graph_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            if self._debug:
                print(f"Debug: PyTorch graph not found: {src}")
            return None

        pytorch_graph = self._load_pytorch_graph(src)
        if not pytorch_graph:
            return None

        if copy_graph:
            self._copy_input_graph(src, out_dir, pytorch_graph)

        einsum_graph = self._convert_loaded_graph(
            pytorch_graph, expand_complex_ops=expand_complex_ops
        )
        self._publish_einsum_graph(
            einsum_graph, out_dir, enable_rename=enable_rename
        )
        return einsum_graph

    def _convert_loaded_graph(
        self,
        pytorch_graph: dict[str, Any],
        *,
        expand_complex_ops: bool,
    ) -> dict[str, Any]:
        """Convert a validated in-memory operator graph to semantic einsum."""
        op_graph, start_nodes_info, param_nodes_info = self._build_op_graph(
            pytorch_graph
        )
        if expand_complex_ops:
            op_graph = (
                self._expand_reviewed_ops(op_graph)
                if self._strict
                else self._expand_complex_ops(op_graph)
            )
        einsum_graph = self._build_einsum_graph(
            pytorch_graph, op_graph, start_nodes_info, param_nodes_info
        )
        einsum_graph = annotate_semantics(einsum_graph, strict=self._strict)
        if self._strict:
            self._validate_exact_graph(einsum_graph)
            validate_semantic_graph(einsum_graph)
        einsum_graph = add_taco_expressions(einsum_graph)
        self._validate_tensor_shape_consistency(einsum_graph)
        return einsum_graph

    def _publish_einsum_graph(
        self,
        einsum_graph: dict[str, Any],
        out_dir: Path,
        *,
        enable_rename: bool,
    ) -> None:
        """Publish semantic and optional AccelForge artifacts."""
        out_path = out_dir / "einsum_graph.yaml"
        self._write_yaml(out_path, einsum_graph, no_aliases=True)
        if self._debug:
            print(f"✅ Wrote einsum graph: {out_path}")
        renamed_path = out_dir / "einsum_graph_renamed.yaml"
        shutil.copy2(out_path, renamed_path)
        if self._debug:
            mode = (
                "requested legacy rename is obsolete"
                if enable_rename
                else "rename disabled"
            )
            print(f"✅ Copied einsum graph as renamed ({mode}): {renamed_path}")
        try:
            af_graph = build_af_graph_from_dict(einsum_graph)
        except (RuntimeError, ValueError) as exc:
            if not self._strict:
                raise
            # AccelForge's legacy cost IR cannot encode every executable
            # schema-v3 operation (notably gather/scatter and multi-output
            # primitives).  The exact semantic graph has already passed the
            # strict validator above, so retain it and record that the
            # secondary AF projection is unavailable instead of inventing a
            # copy/einsum surrogate.
            einsum_graph["af_emission"] = {
                "status": "not_applicable",
                "reason": "extended_semantics_not_representable",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            self._write_yaml(out_path, einsum_graph, no_aliases=True)
            shutil.copy2(out_path, renamed_path)
            return
        af_to_write = {
            k: v for k, v in af_graph.items() if not k.startswith("_")
        }
        out_path = out_dir / "af_einsum_graph.yaml"
        self._write_yaml(out_path, af_to_write, no_aliases=False)
        if self._debug:
            print(f"✅ Wrote AccelForge graph: {out_path}")
            diagnostics = af_graph.get("_diagnostics") or []
            for d in diagnostics:
                print(f"  af_graph diagnostic: {d}")

    @staticmethod
    def _write_yaml(
        path: Path, value: dict[str, Any], *, no_aliases: bool
    ) -> None:
        """Serialize one graph with PyYAML's safe repository dumper."""
        dumper = NoAliasDumper if no_aliases else yaml.SafeDumper
        with path.open("w") as stream:
            yaml.dump(
                value,
                stream,
                Dumper=dumper,
                sort_keys=False,
                default_flow_style=False,
            )

    def _copy_input_graph(
        self,
        src: Path,
        out_dir: Path,
        pytorch_graph: dict[str, Any],
    ) -> None:
        """Copy input graph to output directory."""
        try:
            dst = out_dir / "pytorch_graph.yaml"
            if src.suffix.lower() in {".yaml", ".yml"}:
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            elif not dst.exists():
                with open(dst, "w") as f:
                    yaml.dump(
                        pytorch_graph,
                        f,
                        Dumper=NoAliasDumper,
                        sort_keys=False,
                        default_flow_style=False,
                    )
        except Exception:  # noqa: BLE001 - optional backend fallback
            if self._debug:
                print(
                    "Debug: Failed to copy/write canonical pytorch_graph.yaml"
                )

    def _load_pytorch_graph(self, path: Path) -> dict[str, Any] | None:
        """Load PyTorch graph from YAML or JSON file.

        Args:
            path: Path to the graph file.

        Returns:
            The graph dictionary, or None on failure.
        """
        try:
            suffix = path.suffix.lower()

            if suffix in {".yaml", ".yml"}:
                with open(path) as f:
                    data = yaml.safe_load(f)
            elif suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
            else:
                if self._debug:
                    print(f"Debug: Unsupported file extension: {path.suffix}")
                return None

            if isinstance(data, dict) and "layers" in data:
                return data
            if isinstance(data, list):
                return self._convert_node_list(data, model_name=path.stem)

            if self._debug:
                print(f"Debug: Unexpected structure in {path}")
            return None

        except Exception as exc:  # noqa: BLE001 - optional backend fallback
            if self._debug:
                print(f"Debug: Failed to load PyTorch graph: {exc}")
            return None

    def _convert_node_list(
        self,
        nodes: list[dict[str, Any]],
        *,
        model_name: str,
    ) -> dict[str, Any]:
        """Convert legacy node list format to structured graph dictionary."""
        layers: dict[str, Any] = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("name") or "unknown"
            layers[node_id] = {
                "type": node.get("node_type", node.get("type", "unknown")),
                "node_class": node.get("node_class", "UnknownNode"),
                "input_shapes": node.get("input_shapes", []) or [],
                "output_shapes": node.get("output_shapes", []) or [],
                "weight_nodes": node.get("weight_nodes", []) or [],
                "weight_shapes": node.get("weight_shapes", []) or [],
                "module_args": node.get("module_args", {}) or {},
                "connections": {
                    "inputs": node.get("input_nodes", []) or [],
                    "outputs": node.get("output_nodes", []) or [],
                },
            }
        return {"model_name": model_name, "layers": layers}

    def _build_op_graph(
        self,
        pytorch_graph: dict[str, Any],
    ) -> tuple[nx.DiGraph, list[dict[str, Any]], list[dict[str, Any]]]:
        """Build operation-only graph by collapsing tensor nodes.

        The input PyTorch graph is typically bipartite (TensorNodes and
        Function/Module nodes). This method collapses tensors and connects
        producer operations to consumer operations.

        Args:
            pytorch_graph: The PyTorch graph dictionary.

        Returns:
            Tuple of (operation graph, start node information, parameter node info).
        """
        layers = pytorch_graph.get("layers") or {}
        tensor_ids, op_ids, auxiliary_ids, parameter_ids = (
            self._partition_nodes(layers)
        )

        # Repair every known torchview tracing quirk in one place: dropped
        # scalar edges, orphan/dead-end tensor pairs, and fp32-overridden
        # output dtypes. After this call, ``layers`` is the cleaned source
        # of truth for downstream graph construction and handlers.
        self._repair_torchview_quirks(layers, op_ids, tensor_ids)
        # The repair pass may materialize producerless tensor-valued keyword
        # arguments that torchview recorded only in ``raw_attributes``.  They
        # are real external inputs, so refresh the partitions before emitting
        # start nodes instead of leaving phantom tensor names in the graph.
        tensor_ids, op_ids, auxiliary_ids, parameter_ids = (
            self._partition_nodes(layers)
        )

        graph = nx.DiGraph()
        for op_id in op_ids:
            graph.add_node(op_id, **(layers.get(op_id) or {}))

        # Collect auxiliary tensor info for start nodes (model inputs only)
        start_nodes_info = self._collect_start_node_info(
            layers, auxiliary_ids, op_ids
        )

        # Collect parameter tensor info separately (model weights)
        param_nodes_info = self._collect_start_node_info(
            layers, parameter_ids, op_ids
        )

        for tensor_id in tensor_ids:
            tensor_data = layers.get(tensor_id) or {}
            conns = tensor_data.get("connections") or {}
            producers = list(conns.get("inputs") or [])
            consumers = list(conns.get("outputs") or [])

            if len(producers) == 1 and producers[0] in op_ids:
                self._tensor_to_producer_op[tensor_id] = producers[0]

            for producer in producers:
                for consumer in consumers:
                    if (
                        producer in op_ids
                        and consumer in op_ids
                        and producer != consumer
                    ):
                        graph.add_edge(producer, consumer)

        # Fallback: use direct connections if no tensor nodes
        if not tensor_ids:
            for op_id in op_ids:
                conns = (layers.get(op_id) or {}).get("connections") or {}
                outputs = list(conns.get("outputs") or [])
                for out_id in outputs:
                    if out_id in op_ids and out_id != op_id:
                        graph.add_edge(op_id, out_id)

        return graph, start_nodes_info, param_nodes_info

    def _partition_nodes(
        self,
        layers: dict[str, Any],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Partition nodes into tensor, operation, and auxiliary categories.

        Args:
            layers: The layers dictionary from the PyTorch graph.

        Returns:
            Tuple of (tensor_ids, op_ids, auxiliary_tensor_ids, parameter_tensor_ids).
        """
        tensor_ids: list[str] = []
        op_ids: list[str] = []
        auxiliary_ids: list[str] = []

        parameter_ids: list[str] = []

        for node_id, data in (layers or {}).items():
            node_class = (data.get("node_class") or "").lower()
            node_type = (data.get("type") or "").lower()

            # Any *-tensor/TensorNode should be treated as a tensor-side node,
            # never an operation node.
            if "tensornode" in node_class or "tensor" in node_type:
                if node_type == "auxiliary-tensor":
                    auxiliary_ids.append(node_id)
                elif node_type == "parameter-tensor":
                    parameter_ids.append(node_id)
                else:
                    tensor_ids.append(node_id)
            else:
                op_ids.append(node_id)

        return tensor_ids, op_ids, auxiliary_ids, parameter_ids

    def _collect_start_node_info(
        self,
        layers: dict[str, Any],
        auxiliary_ids: list[str],
        op_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Collect information about auxiliary tensors to create start nodes."""
        start_nodes_info: list[dict[str, Any]] = []

        for idx, aux_id in enumerate(auxiliary_ids):
            aux_data = layers.get(aux_id) or {}
            conns = aux_data.get("connections") or {}
            output_shapes = aux_data.get("output_shapes") or []
            consumers = list(conns.get("outputs") or [])
            # Filter to only include operation nodes
            valid_consumers = [c for c in consumers if c in op_ids]

            output_dtypes = aux_data.get("output_dtypes") or []

            start_nodes_info.append(
                {
                    "original_id": aux_id,
                    "index": idx,
                    "output_shapes": output_shapes,
                    "output_dtypes": output_dtypes,
                    "consumers": valid_consumers,
                    "recovered_from": (aux_data.get("module_args") or {}).get(
                        "recovered_from"
                    ),
                }
            )

        return start_nodes_info

    def _expand_complex_ops(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations using GraphExpander (best-effort)."""
        if not graph.nodes:
            return graph

        try:
            from solar.ir.extended_einsum.torchview.graph_expander import (
                GraphExpander,
            )

            expander = GraphExpander(
                debug=self._debug,
                enable_agent=self._enable_agent,
                api_key=self._api_key,
                cache_dir=self._cache_dir,
                fail_closed=self._strict,
            )
            return expander.expand(graph)
        except Exception as exc:
            if self._strict:
                raise ConversionError(
                    f"complex-operation expansion failed: {exc}"
                ) from exc
            return graph

    def _expand_reviewed_ops(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Delegate formal handler trust and expansion to its dedicated stage."""
        try:
            return expand_reviewed_handlers(
                graph,
                handler_directory=self._cache_dir,
                debug=self._debug,
            )
        except Exception as exc:
            raise ConversionError(
                f"reviewed handler expansion failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_exact_graph(einsum_graph: dict[str, Any]) -> None:
        """Reject every incomplete or approximate layer in official mode."""
        failures: list[str] = []
        for layer_id, layer in (einsum_graph.get("layers") or {}).items():
            if str(layer.get("type", "")).lower() == "start":
                dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
                shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
                if len(dtypes) != len(shapes):
                    failures.append(f"{layer_id}: missing explicit input dtype")
                continue
            if layer.get("is_einsum_supportable") is not True:
                failures.append(f"{layer_id}: unsupported operation")
            semantic = layer.get("semantic_op") or {}
            if semantic.get("kind") == "einsum" and not layer.get(
                "einsum_equation"
            ):
                failures.append(f"{layer_id}: empty einsum equation")
            dtypes = layer.get("tensor_dtypes") or {}
            shapes = layer.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                if len(dtypes.get(side) or []) != len(shapes.get(side) or []):
                    failures.append(
                        f"{layer_id}: missing explicit {side} dtype metadata"
                    )
        if failures:
            raise ConversionError(
                "strict conversion refused an untrusted graph:\n- "
                + "\n- ".join(failures)
            )

    def _build_einsum_graph(
        self,
        pytorch_graph: dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: list[dict[str, Any]],
        param_nodes_info: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build einsum graph dictionary from operation graph."""
        result: dict[str, Any] = {
            "model_name": pytorch_graph.get("model_name", "pytorch_model"),
            "layers": {},
        }

        # Add start nodes from auxiliary tensors (model inputs only)
        start_node_id_map = self._add_start_nodes(result, start_nodes_info)

        # Combine start + param info for ID mapping in _convert_operation.
        # Parameter nodes don't get their own einsum layers.
        all_source_nodes_info = list(start_nodes_info) + list(
            param_nodes_info or []
        )

        # Map parameter node original IDs so _convert_operation can find them
        for info in param_nodes_info or []:
            original_id = info["original_id"]
            start_node_id_map[original_id] = original_id

        # Map hidden-tensor IDs to their producer op so all downstream
        # code (connections, tensor_names) resolves them automatically.
        for tensor_id, producer_op in self._tensor_to_producer_op.items():
            if tensor_id not in start_node_id_map:
                start_node_id_map[tensor_id] = producer_op

        # Track node ID remapping for split/expanded operations
        # Maps original node_id -> final output node_id
        node_id_remap: dict[str, str] = {}

        # Track expanded nodes' input mappings
        # Maps original node_id -> {input_index -> subgraph_node_id}
        expanded_input_map: dict[str, dict[int, str]] = {}

        # Convert each operation to einsum representation
        for node_id in op_graph.nodes():
            node_data = dict(op_graph.nodes[node_id] or {})
            self._validate_input_types_alignment(node_id, node_data)

            # Check if this is a linear layer with bias that should be split
            if self._should_split_linear_with_bias(node_data):
                matmul_layer, add_layer = self._split_linear_with_bias(
                    node_id,
                    node_data,
                    op_graph,
                    all_source_nodes_info,
                    start_node_id_map,
                )
                result["layers"][node_id] = matmul_layer
                add_node_id = f"{node_id}.bias_add"
                result["layers"][add_node_id] = add_layer
                # Remap: original node_id outputs now come from add_node_id
                node_id_remap[node_id] = add_node_id

            # Check if this is a group-wise conv that needs reshape expansion
            elif self._should_expand_groupwise_conv(node_data):
                subgraph_layers, final_node_id, input_mapping = (
                    self._expand_groupwise_conv(
                        node_id,
                        node_data,
                        op_graph,
                        start_nodes_info,
                        start_node_id_map,
                    )
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is MHA that should be expanded
            elif self._should_expand_mha(node_data):
                subgraph_layers, final_node_id, input_mapping = (
                    self._expand_mha(
                        node_id,
                        node_data,
                        op_graph,
                        start_nodes_info,
                        start_node_id_map,
                    )
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is LSTM that should be expanded
            elif self._should_expand_lstm(node_data):
                subgraph_layers, final_node_id, input_mapping = (
                    self._expand_lstm(
                        node_id,
                        node_data,
                        op_graph,
                        start_nodes_info,
                        start_node_id_map,
                    )
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is GRU that should be expanded
            elif self._should_expand_gru(node_data):
                subgraph_layers, final_node_id, input_mapping = (
                    self._expand_gru(
                        node_id,
                        node_data,
                        op_graph,
                        start_nodes_info,
                        start_node_id_map,
                    )
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is SDPA that should be expanded
            elif self._should_expand_sdpa(node_data):
                subgraph_layers, final_node_id, input_mapping = (
                    self._expand_sdpa(
                        node_id,
                        node_data,
                        op_graph,
                        start_nodes_info,
                        start_node_id_map,
                    )
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                # Remap: original node_id outputs now come from final subgraph node
                node_id_remap[node_id] = final_node_id
                # Store input mapping for predecessor updates
                expanded_input_map[node_id] = input_mapping

            else:
                layer_dict = self._convert_operation(
                    node_id,
                    node_data,
                    op_graph,
                    start_nodes_info,
                    start_node_id_map,
                )
                result["layers"][node_id] = layer_dict

        # Fix connections for split/expanded operations
        self._fix_split_connections(result, node_id_remap, expanded_input_map)

        return result
