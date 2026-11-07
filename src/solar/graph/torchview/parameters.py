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

from typing import Any

from torch import nn

from solar.composition import BoundComponent
from solar.graph.torchview.models import NodeInfo


class ParameterBinder(BoundComponent):
    """Apply concrete model parameters to extracted nodes."""

    def _apply_model_parameters(
        self,
        layer_nodes: list[NodeInfo],
        model: nn.Module,
        computation_nodes: dict[str, Any] | None = None,
    ) -> None:
        """Apply direct and shape-matched model parameters to graph nodes."""
        if computation_nodes:
            self._apply_direct_module_nodes(layer_nodes, computation_nodes)
        modules_by_type = self._collect_modules_by_type(model)
        if self.debug:
            count = sum(len(modules) for modules in modules_by_type.values())
            print(f"  Found {count} PyTorch modules")
        for module_type, modules in modules_by_type.items():
            self._match_module_group(layer_nodes, module_type, modules)

    def _apply_direct_module_nodes(
        self,
        layer_nodes: list[NodeInfo],
        computation_nodes: dict[str, Any],
    ) -> None:
        """Apply modules referenced directly by Torchview ModuleNodes."""
        by_id = {node.node_id: node for node in layer_nodes}
        for original_id, node in computation_nodes.items():
            if type(node).__name__ != "ModuleNode":
                continue
            clean_id = self._original_to_clean_id.get(original_id)
            node_info = by_id.get(clean_id) if clean_id else None
            if node_info is None or node_info.node_id in self._processed_nodes:
                continue
            pytorch_module = self._get_pytorch_module(node)
            if pytorch_module is None:
                continue
            self._apply_module_to_node(
                node_info,
                pytorch_module,
                type(pytorch_module).__name__,
            )
            self._processed_nodes.add(node_info.node_id)
            if self.debug:
                print(f"  Applied module from ModuleNode: {node_info.node_id}")

    @staticmethod
    def _collect_modules_by_type(
        model: nn.Module,
    ) -> dict[str, list[tuple[str, nn.Module]]]:
        """Group non-root model modules by concrete class name."""
        modules_by_type: dict[str, list[tuple[str, nn.Module]]] = {}
        for name, module in model.named_modules():
            if not name:
                continue
            module_type = type(module).__name__
            modules_by_type.setdefault(module_type, []).append((name, module))
        return modules_by_type

    def _match_module_group(
        self,
        layer_nodes: list[NodeInfo],
        module_type: str,
        modules: list[tuple[str, nn.Module]],
    ) -> None:
        """Match one concrete module family to compatible graph nodes."""
        candidates = [
            node
            for node in layer_nodes
            if node.node_class in ("FunctionNode", "ModuleNode")
            and module_type.lower() in node.type.lower()
            and node.node_id not in self._processed_nodes
        ]
        if self.debug and modules:
            print(
                f"  Matching {len(modules)} {module_type} modules to "
                f"{len(candidates)} nodes"
            )
        if module_type == "Linear" and len(modules) > len(candidates) > 0:
            self._match_linear_modules_by_shape(modules, candidates)
            return
        self._match_modules_sequentially(module_type, modules, candidates)

    def _match_modules_sequentially(
        self,
        module_type: str,
        modules: list[tuple[str, nn.Module]],
        candidates: list[NodeInfo],
    ) -> None:
        """Apply unmatched modules and candidate nodes in stable order."""
        available = iter(
            node
            for node in candidates
            if node.node_id not in self._processed_nodes
        )
        for module_name, module in modules:
            if module_name in self._matched_modules:
                continue
            target = next(available, None)
            if target is None:
                return
            self._apply_module_to_node(target, module, module_type)
            self._processed_nodes.add(target.node_id)
            self._matched_modules.add(module_name)

    def _match_linear_modules_by_shape(
        self,
        modules_list: list[tuple[str, nn.Module]],
        candidate_nodes: list[NodeInfo],
    ) -> None:
        """Match Linear modules to nodes using shape-based matching.

        Args:
            modules_list: List of (name, module) tuples.
            candidate_nodes: List of candidate NodeInfo objects.
        """
        for node in candidate_nodes:
            if node.node_id in self._processed_nodes:
                continue

            # Get expected dimensions from node shapes
            if node.input_shapes and node.output_shapes:
                input_shape = node.input_shapes[0]
                output_shape = node.output_shapes[0]

                if len(input_shape) > 0 and len(output_shape) > 0:
                    expected_in_features = input_shape[-1]
                    expected_out_features = output_shape[-1]

                    # Find matching Linear module
                    for module_name, module in modules_list:
                        if module_name in self._matched_modules:
                            continue

                        if (
                            hasattr(module, "in_features")
                            and hasattr(module, "out_features")
                            and module.in_features == expected_in_features
                            and module.out_features == expected_out_features
                        ):
                            self._apply_module_to_node(node, module, "Linear")
                            self._processed_nodes.add(node.node_id)
                            self._matched_modules.add(module_name)
                            if self.debug:
                                print(
                                    f"    Shape match: {node.node_id} <-> {module_name}"
                                )
                            break

    def _apply_module_to_node(
        self, node: NodeInfo, module: nn.Module, module_type: str
    ) -> None:
        """Apply module arguments to a node.

        Args:
            node: NodeInfo to update.
            module: PyTorch module with parameters.
            module_type: Type name of the module.
        """
        node.module_args["module_type"] = module_type
        node.module_args.update(self._extract_module_arguments(module))
