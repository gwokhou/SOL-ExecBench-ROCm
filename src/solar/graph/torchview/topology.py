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

"""TorchView graph processor for extracting layer information from PyTorch models.

This module provides functionality to process torchview ComputationGraph objects
and extract detailed information about model layers, following Google's Python
style guide.

The output format matches the original process_torchview_graph.py output:
- node_id: Hierarchical node identifier (e.g., "Model.linear_0")
- node_type: Operation type (e.g., "linear", "conv2d", "matmul")
- node_class: Actual node class (e.g., "FunctionNode", "TensorNode", "ModuleNode")
- input_nodes: List of input node IDs (connections from predecessors)
- output_nodes: List of output node IDs (connections to successors)
- input_shapes: List of input tensor shapes
- output_shapes: List of output tensor shapes
- input_dtypes: List of input tensor data types
- output_dtypes: List of output tensor data types
- input_types: List of input tensor types
- output_types: List of output tensor types
- module_args: Dictionary of module configuration arguments
"""

import re
from typing import Any

from torch import nn

from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.processor_contract import TorchviewProcessorContract


class TorchviewTopologyMixin(TorchviewProcessorContract):
    """Recover graph edges, hierarchy, and stable node identities."""

    def _extract_from_edge_list(
        self, computation_graph: Any, original_model: nn.Module | None = None
    ) -> list[NodeInfo]:
        """Extract nodes from the edge_list of the computation graph.

        This method properly tracks node relationships (input_nodes, output_nodes)
        by processing the edge list to build connection information.

        Args:
            computation_graph: ComputationGraph with edge_list.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects with proper connection information.
        """
        computation_nodes = {}  # original_id -> node object
        node_order = []  # Preserve order of discovery

        # Step 1: Collect all unique nodes from edges
        for i, edge in enumerate(computation_graph.edge_list):
            if len(edge) < 2:
                raise ValueError(
                    f"Edge at index {i} has fewer than 2 nodes: {edge}. "
                    f"Expected format: (source_node, target_node)."
                )
            if len(edge) > 2:
                raise ValueError(
                    f"Edge at index {i} has more than 2 nodes: {len(edge)} nodes found. "
                    f"Expected exactly 2 nodes per edge: (source_node, target_node)."
                )

            source_node, target_node = edge[0], edge[1]

            # Add nodes to computation_nodes dict
            for node in (source_node, target_node):
                self._validate_node_type(node)
                original_id = str(getattr(node, "node_id", id(node)))
                if original_id not in computation_nodes:
                    computation_nodes[original_id] = node
                    node_order.append(original_id)

        if self.debug:
            print(f"  Found {len(computation_nodes)} unique computation nodes")

        # Step 2: Pre-scan all nodes to discover which module names have duplicates
        # This allows us to add indices consistently (Linear_0, Linear_1, Linear_2)
        self._prescan_module_hierarchy(computation_nodes, node_order)

        # Step 3: Generate clean IDs and hierarchical names for all nodes
        result = []
        for original_id in node_order:
            node = computation_nodes[original_id]
            clean_id = self._generate_clean_id(node)
            hierarchical_name = self._generate_hierarchical_name(node)

            self._original_to_clean_id[original_id] = clean_id
            self._original_to_hierarchical[original_id] = hierarchical_name

            node_info = self._extract_node_info(node, clean_id, original_model)
            # Add hierarchical_name to module_args
            node_info.module_args["hierarchical_name"] = hierarchical_name
            result.append(node_info)

        # Step 3: Build relationships from edge list using the ID mapping
        id_to_node_info = {node.node_id: node for node in result}

        for edge in computation_graph.edge_list:
            if len(edge) >= 2:
                source_node, target_node = edge[0], edge[1]

                source_original_id = str(
                    getattr(source_node, "node_id", id(source_node))
                )
                target_original_id = str(
                    getattr(target_node, "node_id", id(target_node))
                )

                source_clean_id = self._original_to_clean_id.get(
                    source_original_id
                )
                target_clean_id = self._original_to_clean_id.get(
                    target_original_id
                )

                if source_clean_id and target_clean_id:
                    source_info = id_to_node_info.get(source_clean_id)
                    target_info = id_to_node_info.get(target_clean_id)

                    if source_info and target_info:
                        # Add connection if not already present
                        if target_clean_id not in source_info.output_nodes:
                            source_info.output_nodes.append(target_clean_id)
                        if source_clean_id not in target_info.input_nodes:
                            target_info.input_nodes.append(source_clean_id)

        # Reorder input_nodes to match positional arg order using ordered_input_nodes
        # from the patched torchview FunctionNode
        for original_id in node_order:
            node = computation_nodes[original_id]
            ordered_inputs = getattr(node, "ordered_input_nodes", None)
            if ordered_inputs:
                ordered_node_id = self._original_to_clean_id.get(original_id)
                if ordered_node_id:
                    node_info = id_to_node_info.get(ordered_node_id)
                    if node_info and node_info.input_nodes:
                        # Build ordered input_nodes from ordered_input_nodes
                        ordered_clean_ids = []
                        for input_node in ordered_inputs:
                            inp_orig_id = str(
                                getattr(input_node, "node_id", id(input_node))
                            )
                            inp_clean_id = self._original_to_clean_id.get(
                                inp_orig_id
                            )
                            if (
                                inp_clean_id
                                and inp_clean_id in node_info.input_nodes
                            ):
                                ordered_clean_ids.append(inp_clean_id)
                        # Append any remaining input_nodes not in ordered list
                        for inp_id in node_info.input_nodes:
                            if inp_id not in ordered_clean_ids:
                                ordered_clean_ids.append(inp_id)
                        node_info.input_nodes = ordered_clean_ids

        # Populate input_types and output_types based on connected node types.
        # Input classification:
        #   'input-tensor', 'auxiliary-tensor', 'hidden-tensor' -> 'input'
        #   'parameter-tensor' -> 'weight'
        #   FunctionNode/ModuleNode predecessor -> 'input'
        # Output classification:
        #   'output-tensor', 'auxiliary-tensor', 'hidden-tensor' -> 'output'
        #   FunctionNode/ModuleNode successor -> 'output'
        #   'parameter-tensor' -> ASSERT FAIL (invalid)
        weight_tensor_types = {"parameter-tensor"}

        for node_info in result:
            # input_types
            input_types = []
            for inp_id in node_info.input_nodes:
                inp_node = id_to_node_info.get(inp_id)
                if inp_node and inp_node.type.lower() in weight_tensor_types:
                    input_types.append("weight")
                else:
                    input_types.append("input")
            node_info.input_types = input_types

            # output_types
            output_types = []
            for out_id in node_info.output_nodes:
                out_node = id_to_node_info.get(out_id)
                if out_node and out_node.type.lower() == "parameter-tensor":
                    raise ValueError(
                        f"Output node {out_id} of {node_info.node_id} is "
                        "parameter-tensor, which should only appear as input."
                    )
                output_types.append("output")
            node_info.output_types = output_types

        if self.debug:
            nodes_with_inputs = sum(1 for n in result if n.input_nodes)
            nodes_with_outputs = sum(1 for n in result if n.output_nodes)
            print(f"  Nodes with input connections: {nodes_with_inputs}")
            print(f"  Nodes with output connections: {nodes_with_outputs}")

        # Step 4: Apply parameters from original model if provided
        # This handles both FunctionNode and ModuleNode cases
        if original_model:
            self._apply_model_parameters(
                result, original_model, computation_nodes
            )

        return result

    def _validate_node_type(self, node: Any) -> None:
        """Validate that a node is one of the expected computation node types.

        Args:
            node: Node object to validate.

        Raises:
            TypeError: If node is not one of the valid node types.
        """
        node_class = type(node).__name__
        if node_class not in self._VALID_NODE_TYPES:
            raise TypeError(
                f"Invalid node type: {node_class}. "
                f"Expected one of {self._VALID_NODE_TYPES}."
            )

    def _prescan_module_hierarchy(
        self, computation_nodes: dict[str, Any], node_order: list[str]
    ) -> None:
        """Pre-scan all nodes to discover which module names have duplicates.

        This allows consistent indexing where all instances of a duplicated
        module name get indices (Linear_0, Linear_1, Linear_2) instead of
        (Linear, Linear_1, Linear_2).

        Also tracks module names that appear multiple times in ANY hierarchy path,
        so they get consistent indexing across all nodes.

        Args:
            computation_nodes: Dict mapping original_id to node objects.
            node_order: List of original_ids in discovery order.
        """
        # Temporary tracker to count unique instances at each level
        temp_tracker: dict[tuple[str, str], set[int]] = {}

        # Track module names that appear multiple times in any single hierarchy
        # (e.g., EncoderLayer appears 3 times in EncoderLayer.EncoderLayer.EncoderLayer)
        if not hasattr(self, "_names_repeated_in_any_path"):
            self._names_repeated_in_any_path = set()

        for original_id in node_order:
            node = computation_nodes[original_id]
            hierarchy_with_ids = self._get_module_hierarchy_with_ids(node)

            # Check if any module name appears multiple times in this hierarchy
            name_counts: dict[str, int] = {}
            for module_name, _ in hierarchy_with_ids:
                name_counts[module_name] = name_counts.get(module_name, 0) + 1

            # Track names that repeat in any path
            for name, count in name_counts.items():
                if count > 1:
                    self._names_repeated_in_any_path.add(name)

            parent_path = "root"
            for module_name, obj_id in hierarchy_with_ids:
                key = (parent_path, module_name)

                if key not in temp_tracker:
                    temp_tracker[key] = set()

                temp_tracker[key].add(obj_id)

                # Build parent_path for next level (use module_name without index for now)
                parent_path = f"{parent_path}.{module_name}"

        # Mark keys that have duplicates (multiple different obj_ids at same level)
        for key, obj_ids in temp_tracker.items():
            if len(obj_ids) > 1:
                self._module_has_duplicates.add(key)

    def _generate_clean_id(self, node: Any) -> str:
        """Generate a flat node ID with Model prefix.

        Format: Model.<opname>_<count>

        Args:
            node: Node object.

        Returns:
            Flat node ID string.
        """
        node_name = getattr(node, "name", type(node).__name__.lower())

        # Use flat naming: Model.<op_name>_<count>
        op_key = f"Model.{node_name}"
        if op_key not in self._node_counter:
            self._node_counter[op_key] = 0
            count = 0
        else:
            self._node_counter[op_key] += 1
            count = self._node_counter[op_key]

        return (
            f"Model.{node_name}_{count}" if count > 0 else f"Model.{node_name}"
        )

    def _generate_hierarchical_name(self, node: Any) -> str:
        """Generate a hierarchical name showing the full module path.

        Format: Model.<level0name>_<idx>.<level1name>_<idx>.<opname>

        Args:
            node: Node object.

        Returns:
            Hierarchical name string.
        """
        node_name = getattr(node, "name", type(node).__name__.lower())

        # Build hierarchical path from parent ModuleNodes (with their object IDs)
        hierarchy_with_ids = self._get_module_hierarchy_with_ids(node)

        # Convert hierarchy to indexed names
        indexed_hierarchy = self._index_hierarchy(hierarchy_with_ids)

        # Build full path: Model.<indexed_hierarchy>.<node_name>
        if indexed_hierarchy:
            base_path = "Model." + ".".join(indexed_hierarchy)
        else:
            base_path = "Model"

        # Add counter for the operation name uniqueness within this hierarchy
        if not hasattr(self, "_hierarchical_counter"):
            self._hierarchical_counter = {}

        op_key = f"{base_path}.{node_name}"
        if op_key not in self._hierarchical_counter:
            self._hierarchical_counter[op_key] = 0
            count = 0
        else:
            self._hierarchical_counter[op_key] += 1
            count = self._hierarchical_counter[op_key]

        op_name_indexed = f"{node_name}_{count}" if count > 0 else node_name
        return f"{base_path}.{op_name_indexed}"

    def _get_module_hierarchy_with_ids(
        self,
        node: Any,
        visited: set[int] | None = None,
    ) -> list[tuple[str, int]]:
        """Trace up parent chain to find ModuleNode hierarchy with object IDs.

        Args:
            node: Node object.
            visited: Set of visited node IDs to prevent cycles.

        Returns:
            List of (module_name, object_id) tuples from root to immediate parent.
        """
        if visited is None:
            visited = set()

        node_id = id(node)
        if node_id in visited:
            return []
        visited.add(node_id)

        parents = list(getattr(node, "parents", []))

        # Look for ModuleNode parent first
        for parent in parents:
            parent_class = type(parent).__name__
            if parent_class == "ModuleNode":
                parent_name = getattr(parent, "name", "unknown")
                parent_obj_id = id(parent)
                # Recurse to get the ModuleNode's containment hierarchy
                parent_hierarchy = self._get_module_hierarchy_with_ids(
                    parent, visited
                )
                return parent_hierarchy + [(parent_name, parent_obj_id)]

        # No direct ModuleNode parent - trace through TensorNode parents only
        # (TensorNodes are part of module containment, FunctionNodes are data flow)
        for parent in parents:
            parent_class = type(parent).__name__
            if parent_class == "TensorNode":
                # Continue tracing through TensorNode to find containing ModuleNode
                parent_hierarchy = self._get_module_hierarchy_with_ids(
                    parent, visited
                )
                if parent_hierarchy:
                    return parent_hierarchy

        # No module container found
        return []

    def _index_hierarchy(
        self, hierarchy_with_ids: list[tuple[str, int]]
    ) -> list[str]:
        """Convert hierarchy with IDs to indexed names.

        Tracks module names at each level and assigns indices when names repeat.
        Always adds index suffix when there are duplicates at that level, OR when
        the same module name appears multiple times in ANY hierarchy path.

        Args:
            hierarchy_with_ids: List of (module_name, object_id) tuples.

        Returns:
            List of indexed module names (e.g., ['MultiHeadAttention', 'Linear_0']).
        """
        if not hierarchy_with_ids:
            return []

        # Track seen (parent_path, name) -> {obj_id: index}
        # This allows us to assign consistent indices based on first-seen order
        if not hasattr(self, "_module_index_tracker"):
            self._module_index_tracker = {}

        # Track which (parent_path, name) keys have duplicates
        if not hasattr(self, "_module_has_duplicates"):
            self._module_has_duplicates = set()

        # Track names that repeat in any path (set by prescan)
        if not hasattr(self, "_names_repeated_in_any_path"):
            self._names_repeated_in_any_path = set()

        result = []
        parent_path = "root"

        # Track indices for names that repeat across any hierarchy path
        # This ensures consistent indexing even for nodes at different depths
        path_name_indices: dict[str, int] = {}

        for module_name, obj_id in hierarchy_with_ids:
            key = (parent_path, module_name)

            if key not in self._module_index_tracker:
                self._module_index_tracker[key] = {}

            tracker = self._module_index_tracker[key]

            if obj_id not in tracker:
                # Assign next index for this module name at this level
                tracker[obj_id] = len(tracker)
                # If this is the second or later instance, mark as having duplicates
                if len(tracker) > 1:
                    self._module_has_duplicates.add(key)

            idx = tracker[obj_id]

            # Check if this name repeats in any hierarchy path (from prescan)
            if module_name in self._names_repeated_in_any_path:
                # Use path-local index for names that can repeat in hierarchies
                if module_name not in path_name_indices:
                    path_name_indices[module_name] = 0
                else:
                    path_name_indices[module_name] += 1
                indexed_name = f"{module_name}_{path_name_indices[module_name]}"
            elif key in self._module_has_duplicates:
                # Use global index for duplicates at the same level
                indexed_name = f"{module_name}_{idx}"
            else:
                indexed_name = module_name

            result.append(indexed_name)
            parent_path = f"{parent_path}.{indexed_name}"

        return result

    def _extract_from_visual_graph(self, visual_graph: Any) -> list[NodeInfo]:
        """Extract nodes from the visual graph representation.

        Args:
            visual_graph: graphviz.Digraph object.

        Returns:
            List of NodeInfo objects.
        """
        if not hasattr(visual_graph, "source"):
            return []

        nodes: dict[str, NodeInfo] = {}
        edges: list[tuple[str, str]] = []

        # Parse graphviz source
        self._parse_graphviz_source(visual_graph.source, nodes, edges)

        # Build relationships
        for source_id, target_id in edges:
            if source_id in nodes and target_id in nodes:
                nodes[source_id].output_nodes.append(nodes[target_id].node_id)
                nodes[target_id].input_nodes.append(nodes[source_id].node_id)

        return list(nodes.values())

    def _parse_graphviz_source(
        self,
        source: str,
        nodes: dict[str, NodeInfo],
        edges: list[tuple[str, str]],
    ) -> None:
        """Parse graphviz source to extract nodes and edges.

        Args:
            source: Graphviz source string.
            nodes: Dictionary to populate with nodes.
            edges: List to populate with edges.
        """
        lines = source.split("\n")
        current_node_def = ""
        in_node_definition = False
        node_id: str | None = None

        for raw_line in lines:
            line = raw_line.strip()

            # Check for node definition
            if re.match(r"^\d+\s+\[label=<", line):
                in_node_definition = True
                current_node_def = line
                node_match = re.match(r"^(\d+)", line)
                if node_match is None:
                    continue
                node_id = node_match.group(1)
            elif in_node_definition and line.endswith("]"):
                current_node_def += " " + line
                in_node_definition = False
                if node_id is None:
                    current_node_def = ""
                    continue

                # Parse node definition
                node_info = self._parse_node_definition(
                    node_id, current_node_def
                )
                if node_info:
                    nodes[node_id] = node_info
                current_node_def = ""
            elif in_node_definition:
                current_node_def += " " + line

            # Check for edge definition
            elif "->" in line and "[" not in line:
                edge_match = re.match(r"^(\d+)\s*->\s*(\d+)", line)
                if edge_match:
                    edges.append((edge_match.group(1), edge_match.group(2)))

    def _parse_node_definition(
        self, node_id: str, node_def: str
    ) -> NodeInfo | None:
        """Parse a node definition from graphviz source.

        Args:
            node_id: Node ID from graphviz.
            node_def: Node definition string.

        Returns:
            NodeInfo object if successfully parsed, None otherwise.
        """
        try:
            # Extract node type
            node_type = "Unknown"
            type_match = re.search(
                r"<TD[^>]*>([^<]*)<BR/>depth:\d+</TD>", node_def
            )
            if type_match:
                node_type = type_match.group(1).strip()

            # Extract shapes
            input_shapes = []
            output_shapes = []

            if node_type in ["input-tensor", "output-tensor"]:
                shape_match = re.search(r"<TD>\(([^)]+)\)</TD>", node_def)
                if shape_match:
                    shape_str = shape_match.group(1)
                    shape = [int(x.strip()) for x in shape_str.split(",")]
                    if node_type == "input-tensor":
                        output_shapes.append(shape)
                    else:
                        input_shapes.append(shape)

            # Create node ID
            hierarchical_id = f"Model.{node_type}_{node_id}"

            # Extract dtypes (empty for visual graph extraction)
            input_dtypes: list[str] = []
            output_dtypes: list[str] = []

            return NodeInfo(
                node_id=hierarchical_id,
                type=node_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
            )

        except Exception as e:  # noqa: BLE001 - optional backend fallback
            if self.debug:
                print(f"Error parsing node {node_id}: {e}")
            return None
