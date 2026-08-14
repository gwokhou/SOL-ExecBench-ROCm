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

from pathlib import Path

import yaml

from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.processor_contract import TorchviewProcessorContract
from solar.types import NodeDict


class TorchviewReportingMixin(TorchviewProcessorContract):
    """Persist and summarize processed Torchview graphs."""

    def _save_pytorch_graph_yaml(
        self,
        layer_nodes: list[NodeInfo],
        filename: Path,
        *,
        model_name: str,
    ) -> None:
        """Save extracted nodes to a structured YAML graph.

        The YAML format:

          model_name: <str>
          layers:
            <node_id>:
              type: <str>
              node_class: <str>
              input_shapes: [...]
              output_shapes: [...]
              input_types: [...]
              output_types: [...]
              module_args: {...}
              connections:
                inputs: [...]
                outputs: [...]

        Args:
            layer_nodes: Extracted nodes.
            filename: Output YAML path.
            model_name: Human-readable model name.
        """
        layers: dict[str, NodeDict] = {}
        graph_dict: NodeDict = {
            "model_name": model_name,
            "layers": layers,
        }

        for node in layer_nodes:
            layers[node.node_id] = node.to_dict()

        with open(filename, "w") as f:
            from solar.artifacts.yaml import NoAliasDumper

            yaml.dump(
                graph_dict,
                f,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
            )

        if self.debug:
            print(f"PyTorch graph YAML saved to {filename}")

    def _print_layer_summary(self, layer_nodes: list[NodeInfo]) -> None:
        """Print summary of extracted layer nodes.

        Args:
            layer_nodes: List of NodeInfo objects.
        """
        print(f"\n{'=' * 80}")
        print(f"EXTRACTED LAYER NODES ({len(layer_nodes)} nodes)")
        print(f"{'=' * 80}")

        for i, node in enumerate(layer_nodes[:5], 1):  # Show first 5
            print(f"\n[{i}] Node ID: {node.node_id}")
            print(f"    Type: {node.type} ({node.node_class})")
            print(f"    Input Nodes: {node.input_nodes}")
            print(f"    Output Nodes: {node.output_nodes}")
            print(f"    Input Shapes: {node.input_shapes}")
            print(f"    Output Shapes: {node.output_shapes}")
            print(f"    Input Dtypes: {node.input_dtypes}")
            print(f"    Output Dtypes: {node.output_dtypes}")
            if node.input_types:
                print(f"    Input Types: {node.input_types}")

        if len(layer_nodes) > 5:
            print(f"\n... and {len(layer_nodes) - 5} more nodes")
