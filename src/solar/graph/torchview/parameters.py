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

from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.processor_contract import TorchviewProcessorContract


class TorchviewParametersMixin(TorchviewProcessorContract):
    """Apply concrete model parameters to extracted nodes."""

    def _apply_model_parameters(
        self,
        layer_nodes: list[NodeInfo],
        model: nn.Module,
        computation_nodes: dict[str, Any] | None = None,
    ) -> None:
        """Apply parameters from the original model to extracted nodes.

        This uses shape-based matching to correctly associate PyTorch modules
        with their corresponding function nodes. Also handles ModuleNode cases
        where the node directly references a PyTorch module.

        Args:
            layer_nodes: List of NodeInfo objects to update.
            model: Original PyTorch model.
            computation_nodes: Optional dict mapping original IDs to node objects.
        """
        # First, try to extract from ModuleNode objects directly
        if computation_nodes:
            for original_id, node in computation_nodes.items():
                if type(node).__name__ == "ModuleNode":
                    clean_id = self._original_to_clean_id.get(original_id)
                    if clean_id:
                        node_info = next(
                            (n for n in layer_nodes if n.node_id == clean_id),
                            None,
                        )
                        if (
                            node_info
                            and node_info.node_id not in self._processed_nodes
                        ):
                            pytorch_module = self._get_pytorch_module(node)
                            if pytorch_module:
                                module_type = type(pytorch_module).__name__
                                self._apply_module_to_node(
                                    node_info, pytorch_module, module_type
                                )
                                self._processed_nodes.add(node_info.node_id)
                                if self.debug:
                                    print(
                                        f"  Applied module from ModuleNode: {node_info.node_id}"
                                    )

        # Collect all modules from the model
        modules_by_type: dict[str, list[tuple[str, nn.Module]]] = {}
        for name, module in model.named_modules():
            if name == "":  # Skip root
                continue
            module_type = type(module).__name__
            if module_type not in modules_by_type:
                modules_by_type[module_type] = []
            modules_by_type[module_type].append((name, module))

        if self.debug:
            print(
                f"  Found {sum(len(v) for v in modules_by_type.values())} PyTorch modules"
            )

        # Match modules to nodes by type and shape
        for module_type, modules_list in modules_by_type.items():
            # Find candidate nodes for this module type (both FunctionNode and ModuleNode)
            candidate_nodes = [
                node
                for node in layer_nodes
                if node.node_class in ("FunctionNode", "ModuleNode")
                and module_type.lower() in node.type.lower()
                and node.node_id not in self._processed_nodes
            ]

            if self.debug and modules_list:
                print(
                    f"  Matching {len(modules_list)} {module_type} modules to {len(candidate_nodes)} nodes"
                )

            # Special handling for Linear layers with shape matching
            if (
                module_type == "Linear"
                and len(modules_list) > len(candidate_nodes) > 0
            ):
                self._match_linear_modules_by_shape(
                    modules_list, candidate_nodes
                )
            else:
                # Standard sequential matching
                for _i, (module_name, module) in enumerate(modules_list):
                    if module_name in self._matched_modules:
                        continue

                    target_node = None
                    for node in candidate_nodes:
                        if node.node_id not in self._processed_nodes:
                            target_node = node
                            break

                    if target_node:
                        self._apply_module_to_node(
                            target_node, module, module_type
                        )
                        self._processed_nodes.add(target_node.node_id)
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
