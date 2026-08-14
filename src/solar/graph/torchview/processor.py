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

from torch import nn

from solar.graph.torchview.metadata import TorchviewMetadataMixin
from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.parameters import TorchviewParametersMixin
from solar.graph.torchview.reporting import TorchviewReportingMixin
from solar.graph.torchview.topology import TorchviewTopologyMixin
from solar.types import DynamicValue


class TorchviewProcessor(
    TorchviewMetadataMixin,
    TorchviewTopologyMixin,
    TorchviewParametersMixin,
    TorchviewReportingMixin,
):
    """Processes torchview computation graphs to extract layer information.

    This class provides methods to extract detailed information from torchview
    ComputationGraph objects, including node hierarchies, shapes, connections,
    and PyTorch nn.Module parameters.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the TorchviewProcessor.

        Args:
            debug: Enable debug output for troubleshooting.
        """
        self.debug = debug
        self._processed_nodes: set[str] = set()
        self._matched_modules: set[str] = set()
        self._node_counter: dict[str, int] = {}
        self._original_to_clean_id: dict[str, str] = {}

    def process_graph(
        self,
        computation_graph: DynamicValue,
        output_dir: str,
        kernel_name: str,
        original_model: nn.Module | None = None,
    ) -> list[NodeInfo]:
        """Process a torchview ComputationGraph and save extracted layer nodes.

        Args:
            computation_graph: torchview ComputationGraph object.
            output_dir: Directory to save outputs.
            kernel_name: Name of the kernel for file naming.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects containing extracted layer information.
        """
        if self.debug:
            print(f"Processing torchview graph for {kernel_name}...")

        self._reset_state()

        layer_nodes = self._extract_layer_nodes(
            computation_graph, original_model
        )

        # Save canonical YAML graph (and remove any legacy JSON artifacts).
        output_path = Path(output_dir)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        yaml_filename = output_path / "pytorch_graph.yaml"

        self._save_pytorch_graph_yaml(
            layer_nodes, yaml_filename, model_name=kernel_name
        )

        if self.debug:
            self._print_layer_summary(layer_nodes)

        return layer_nodes

    def _reset_state(self) -> None:
        """Reset internal state for processing a new graph."""
        self._processed_nodes.clear()
        self._matched_modules.clear()
        self._node_counter.clear()
        self._original_to_clean_id.clear()
        self._module_index_tracker: dict[tuple[str, str], dict[int, int]] = {}
        self._module_has_duplicates: set[tuple[str, str]] = set()
        self._names_repeated_in_any_path: set[str] = set()
        self._hierarchical_counter: dict[str, int] = {}
        self._original_to_hierarchical: dict[str, str] = {}

    _VALID_NODE_TYPES = ("TensorNode", "ModuleNode", "FunctionNode")
