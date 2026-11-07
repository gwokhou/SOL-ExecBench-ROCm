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
from threading import RLock
from typing import ClassVar

from torch import nn

from solar.composition import BoundComponent, component_attribute
from solar.graph.torchview.metadata import MetadataExtractor
from solar.graph.torchview.models import NodeInfo, TorchviewProcessorState
from solar.graph.torchview.parameters import ParameterBinder
from solar.graph.torchview.reporting import GraphReporter
from solar.graph.torchview.topology import TopologyBuilder
from solar.types import DynamicValue


class TorchviewProcessor:
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
        self._state = TorchviewProcessorState()
        self._processing_lock = RLock()
        self._components: tuple[BoundComponent, ...] = (
            MetadataExtractor(self),
            TopologyBuilder(self),
            ParameterBinder(self),
            GraphReporter(self),
        )

    def __getattr__(self, name: str) -> DynamicValue:
        """Resolve private processing behavior from composed components."""
        if name in self._STATE_ATTRIBUTES:
            return getattr(self._state, name.removeprefix("_"))
        return component_attribute(self._components, name)

    def __setattr__(self, name: str, value: object) -> None:
        """Route known per-call fields into the active state object."""
        state = self.__dict__.get("_state")
        if state is not None and name in self._STATE_ATTRIBUTES:
            setattr(state, name.removeprefix("_"), value)
            return
        object.__setattr__(self, name, value)

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
        with self._processing_lock:
            if self.debug:
                print(f"Processing torchview graph for {kernel_name}...")

            self._reset_state()

            layer_nodes = self._extract_layer_nodes(
                computation_graph, original_model
            )

            # Save canonical YAML graph (and remove legacy JSON artifacts).
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
        self._state = TorchviewProcessorState()

    _VALID_NODE_TYPES = ("TensorNode", "ModuleNode", "FunctionNode")
    _STATE_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset(
        f"_{name}" for name in TorchviewProcessorState.__dataclass_fields__
    )
