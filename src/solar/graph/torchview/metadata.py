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

import torch
from torch import nn

from solar.composition import BoundComponent
from solar.graph.torchview.attribute_parsing import (
    parse_operation_attributes,
)
from solar.graph.torchview.constants import (
    BOOLEAN_ATTRS,
    GEOMETRIC_ATTRS,
    MODULE_ATTR_NAMES,
)
from solar.graph.torchview.models import NodeInfo
from solar.types import TensorShape


class MetadataExtractor(BoundComponent):
    """Extract node, tensor, module, and function metadata."""

    def _extract_layer_nodes(
        self, computation_graph: Any, original_model: nn.Module | None = None
    ) -> list[NodeInfo]:
        """Extract layer nodes from the computation graph.

        Args:
            computation_graph: torchview ComputationGraph object.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects.
        """
        layer_nodes = []

        # Note: node_hierarchy contains hierarchical ModuleNode info (e.g., nn.Linear
        # with in_features/out_features), but the computation graph is extracted from
        # the flattened edge_list which contains FunctionNodes (e.g., F.linear).
        # The hierarchy is useful for understanding module structure but edge_list
        # provides the actual computation graph with tensor flow.
        # Extract from edge_list (flattened computation graph)
        if (
            hasattr(computation_graph, "edge_list")
            and computation_graph.edge_list
        ):
            if self.debug:
                print(
                    f"Extracting from edge_list ({len(computation_graph.edge_list)} edges)..."
                )
            return self._extract_from_edge_list(
                computation_graph, original_model
            )

        # Parse visual graph as fallback
        if hasattr(computation_graph, "visual_graph"):
            if self.debug:
                print("Parsing visual graph...")
            return self._extract_from_visual_graph(
                computation_graph.visual_graph
            )

        return layer_nodes

    def _extract_node_info(
        self,
        node: Any,
        node_id: str,
        original_model: nn.Module | None = None,
    ) -> NodeInfo:
        """Extract information from a single node.

        Args:
            node: Node object from the computation graph.
            node_id: Unique identifier for the node.
            original_model: Original PyTorch model for dtype inference fallback.

        Returns:
            NodeInfo object containing extracted information.
        """
        node_class_name = type(node).__name__

        # Extract node type
        node_type = self._get_node_type(node, node_class_name)

        # Extract shapes
        input_shapes, output_shapes = self._extract_shapes(node, node_type)

        # Extract dtypes (pass shapes and original_model for dtype inference fallback)
        input_dtypes, output_dtypes = self._extract_dtypes(
            node, node_type, original_model, input_shapes, output_shapes
        )

        # Extract module args (no longer extracting weight_nodes/weight_shapes)
        module_info = self._extract_module_info(node)

        return NodeInfo(
            node_id=node_id,
            type=node_type,
            node_class=node_class_name,
            input_nodes=[],
            output_nodes=[],
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            input_types=[],  # populated later from connection node types
            output_types=[],  # populated later from connection node types
            module_args=module_info["module_args"],
            source_input_index=getattr(
                node,
                "source_input_index",
                getattr(
                    getattr(node, "main_node", None),
                    "source_input_index",
                    None,
                ),
            ),
        )

    def _get_node_type(self, node: Any, node_class: str) -> str:
        """Determine the node type from the node object.

        Args:
            node: Node object.
            node_class: Class name of the node.

        Returns:
            String representing the node type.
        """
        return (
            getattr(node, "operation", None)
            or getattr(node, "op_name", None)
            or getattr(node, "name", None)
            or getattr(node, "_op", None)
            or node_class.lower().replace("node", "")
        )

    def _extract_shapes(
        self, node: Any, node_type: str
    ) -> tuple[list[TensorShape], list[TensorShape]]:
        """Extract input and output shapes from a node."""
        tensor_shapes = self._tensor_node_shapes(node, node_type)
        if tensor_shapes is not None:
            return tensor_shapes
        return (
            self._shapes_from_side(node, "inputs", "input_shape"),
            self._shapes_from_side(node, "outputs", "output_shape"),
        )

    @staticmethod
    def _tensor_node_shapes(
        node: Any,
        node_type: str,
    ) -> tuple[list[TensorShape], list[TensorShape]] | None:
        """Return the shape placement for a concrete TensorNode."""
        if type(node).__name__ != "TensorNode":
            return None
        raw_shape = getattr(node, "tensor_shape", None)
        if raw_shape is None:
            return None
        tensor_shape = (
            list(raw_shape) if hasattr(raw_shape, "__iter__") else [raw_shape]
        )
        if not tensor_shape:
            return None
        node_name = node_type.lower() if node_type else ""
        if node_name in {
            "input-tensor",
            "auxiliary-tensor",
            "parameter-tensor",
        }:
            return [], [tensor_shape]
        if node_name == "output-tensor":
            return [tensor_shape], []
        return [tensor_shape], [tensor_shape]

    @staticmethod
    def _shapes_from_side(
        node: Any,
        collection_name: str,
        fallback_name: str,
    ) -> list[TensorShape]:
        """Extract shapes from one connected side or its legacy attribute."""
        connected = getattr(node, collection_name, None)
        if connected:
            result: list[TensorShape] = []
            for item in connected:
                shape = getattr(item, "tensor_shape", None)
                if shape is None:
                    shape = getattr(item, "shape", None)
                if shape is not None:
                    result.append(list(shape))
            return result
        fallback = getattr(node, fallback_name, None)
        if not fallback:
            return []
        if isinstance(fallback, (list, tuple)):
            return [list(shape) for shape in fallback]
        return [list(fallback)]

    @staticmethod
    def _classify_tensor_node_dtype(
        node_type: str, tensor_dtype: str
    ) -> tuple[list[str], list[str]]:
        """Place a TensorNode dtype on the same sides as its tensor shape."""
        node_name = node_type.lower() if node_type else ""
        if node_name in {
            "input-tensor",
            "auxiliary-tensor",
            "parameter-tensor",
        }:
            return [], [tensor_dtype]
        if node_name == "output-tensor":
            return [tensor_dtype], []
        return [tensor_dtype], [tensor_dtype]

    def _extract_dtypes(
        self,
        node: Any,
        node_type: str,
        original_model: nn.Module | None = None,
        input_shapes: list[TensorShape] | None = None,
        output_shapes: list[TensorShape] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Extract input and output dtypes from a node."""
        tensor_dtype = self._tensor_node_dtype(node)
        if type(node).__name__ == "TensorNode" and tensor_dtype:
            return self._classify_tensor_node_dtype(node_type, tensor_dtype)

        input_dtypes = self._dtypes_from_side(node, "inputs")
        output_dtypes = self._dtypes_from_side(node, "outputs")
        if not input_dtypes:
            input_dtypes = self._legacy_dtypes(node, "input_dtype")
        if not output_dtypes:
            output_dtypes = self._legacy_dtypes(node, "output_dtype")
        if original_model is not None:
            default_dtype = self._model_default_dtype(original_model)
            self._pad_dtypes(input_dtypes, input_shapes, default_dtype)
            self._pad_dtypes(output_dtypes, output_shapes, default_dtype)
        return input_dtypes, output_dtypes

    @staticmethod
    def _tensor_like_dtype(obj: Any) -> str | None:
        """Extract a dtype string from a Torchview tensor-like object."""
        if obj is None:
            return None
        tensor_dtype = getattr(obj, "tensor_dtype", None)
        if tensor_dtype is not None:
            return str(tensor_dtype)
        dtype = getattr(obj, "dtype", None)
        if isinstance(dtype, (torch.dtype, str)):
            return str(dtype)
        tensor = getattr(obj, "tensor", None)
        if tensor is not None and hasattr(tensor, "dtype"):
            return str(tensor.dtype)
        if isinstance(obj, torch.Tensor):
            return str(obj.dtype)
        return None

    def _tensor_node_dtype(self, node: Any) -> str | None:
        """Recover a TensorNode dtype, including its output fallback."""
        if type(node).__name__ != "TensorNode":
            return None
        direct = self._tensor_like_dtype(node)
        if direct:
            return direct
        for output in getattr(node, "outputs", None) or ():
            dtype = self._tensor_like_dtype(output)
            if dtype:
                return dtype
        return None

    def _dtypes_from_side(self, node: Any, name: str) -> list[str]:
        """Extract every available dtype from one connected tensor side."""
        return [
            dtype
            for item in (getattr(node, name, None) or ())
            if (dtype := self._tensor_like_dtype(item)) is not None
        ]

    @staticmethod
    def _legacy_dtypes(node: Any, name: str) -> list[str]:
        """Normalize a legacy singular-or-sequence dtype attribute."""
        value = getattr(node, name, None)
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            return [str(dtype) for dtype in value]
        return [str(value)]

    @staticmethod
    def _model_default_dtype(model: nn.Module) -> str:
        """Return the first model parameter dtype for missing graph metadata."""
        return next(
            (
                str(parameter.dtype)
                for parameter in model.parameters()
                if parameter.dtype is not None
            ),
            "torch.float32",
        )

    @staticmethod
    def _pad_dtypes(
        dtypes: list[str],
        shapes: list[TensorShape] | None,
        default_dtype: str,
    ) -> None:
        """Pad dtype metadata to the corresponding tensor-shape count."""
        while len(dtypes) < len(shapes or ()):
            dtypes.append(default_dtype)

    def _extract_module_info(self, node: Any) -> dict[str, Any]:
        """Extract module argument information from a node.

        Args:
            node: Node object.

        Returns:
            Dictionary with module_args.
        """
        module_info: dict[str, Any] = {"module_args": {}}

        node_class = type(node).__name__

        if node_class == "ModuleNode":
            self._extract_module_node_info(node, module_info)
        elif node_class == "FunctionNode":
            self._extract_function_node_info(node, module_info)

        return module_info

    def _extract_module_node_info(
        self, node: Any, module_info: dict[str, Any]
    ) -> None:
        """Extract information from a ModuleNode.

        Args:
            node: ModuleNode object.
            module_info: Dictionary to populate with extracted information.
        """
        module = self._get_pytorch_module(node)

        if module is not None:
            module_info["module_args"] = self._extract_module_arguments(module)
        else:
            # Fall back to torchview's serialized module attributes.
            if hasattr(node, "attributes") and node.attributes:
                parsed = self._parse_module_attributes_string(node.attributes)
                if parsed:
                    module_info["module_args"] = parsed

    def _get_pytorch_module(self, node: Any) -> nn.Module | None:
        """Get the PyTorch module from a node object.

        Args:
            node: Node object that may contain a PyTorch module.

        Returns:
            PyTorch module if found, None otherwise.
        """
        # Try common attribute names
        for attr_name in MODULE_ATTR_NAMES:
            if hasattr(node, attr_name):
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, nn.Module):
                    return attr_value
        return None

    def _extract_module_arguments(self, module: nn.Module) -> dict[str, Any]:
        """Extract module configuration arguments.

        Args:
            module: PyTorch module.

        Returns:
            Dictionary of module arguments.
        """
        args: dict[str, Any] = {"module_type": type(module).__name__}

        # Extract common attributes based on module type
        for attr_name in dir(module):
            if attr_name.startswith("_") or callable(
                getattr(module, attr_name)
            ):
                continue

            try:
                value = getattr(module, attr_name)

                # Handle special attribute types
                if attr_name in GEOMETRIC_ATTRS and hasattr(value, "__iter__"):
                    args[attr_name] = list(value)
                elif attr_name in BOOLEAN_ATTRS:
                    args[attr_name] = bool(value)
                elif isinstance(value, (int, float, str, bool)):
                    args[attr_name] = value
                elif attr_name == "bias":
                    args[attr_name] = value is not None

            except Exception:  # noqa: BLE001,S112 - optional backend fallback
                continue

        return args

    def _parse_module_attributes_string(
        self, attributes: str
    ) -> dict[str, Any]:
        """Parse torchview ModuleNode attributes string.

        Format: "Linear(training=False, in_features=64, out_features=64)"

        Args:
            attributes: Stringified module attributes from torchview.

        Returns:
            Dictionary of parsed module arguments.
        """
        result: dict[str, Any] = {}

        if not attributes:
            return result

        # The leading token is the serialized module type.

        match = re.match(r"(\w+)\((.*)\)", attributes)
        if not match:
            return result

        module_type = match.group(1)
        args_str = match.group(2)

        result["module_type"] = module_type

        # Parse key=value pairs
        # Handle nested parentheses for tuples like kernel_size=(3, 3)
        for kv_match in re.finditer(r"(\w+)=((?:\([^)]*\))|[^,]+)", args_str):
            key = kv_match.group(1)
            value_str = kv_match.group(2).strip()

            # Parse the value
            try:
                if value_str == "True":
                    result[key] = True
                elif value_str == "False":
                    result[key] = False
                elif value_str == "None":
                    result[key] = None
                elif value_str.startswith("(") and value_str.endswith(")"):
                    # Tuple like (3, 3)
                    result[key] = eval(value_str)
                elif (
                    "." in value_str
                    and not value_str.replace(".", "")
                    .replace("-", "")
                    .isdigit()
                ):
                    # String with dots (like torch.float32)
                    result[key] = value_str
                elif value_str.replace(".", "").replace("-", "").isdigit():
                    # Number
                    if "." in value_str:
                        result[key] = float(value_str)
                    else:
                        result[key] = int(value_str)
                else:
                    result[key] = value_str
            except Exception:  # noqa: BLE001 - optional backend fallback
                result[key] = value_str

        return result

    def _extract_function_node_info(
        self, node: Any, module_info: dict[str, Any]
    ) -> None:
        """Extract information from a FunctionNode.

        Uses input_types from torchview (if available) to identify which
        input tensors are activations ('input') vs parameters ('weight').
        Falls back to raw_attributes parsing for older torchview versions.

        Args:
            node: FunctionNode object.
            module_info: Dictionary to populate with extracted information.
        """
        node_name = getattr(node, "name", "").lower()
        module_info["module_args"]["function_name"] = node_name

        # Extract from torchview 'attributes' field (contains stringified args/kwargs)
        # This is populated when collect_attributes=True in draw_graph()
        if hasattr(node, "attributes") and node.attributes:
            parsed_args = self._parse_torchview_attributes(
                node.attributes, node_name
            )
            if parsed_args:
                module_info["module_args"].update(parsed_args)

        # Extract scalar kwargs as module_args
        if hasattr(node, "kwargs") and node.kwargs:
            for key, value in node.kwargs.items():
                if not hasattr(value, "shape"):
                    module_info["module_args"][key] = value

    def _parse_torchview_attributes(
        self,
        attributes: str,
        node_name: str,
    ) -> dict[str, Any]:
        """Parse exact and diagnostic arguments from Torchview text."""
        if not attributes:
            return {}
        result: dict[str, Any] = {"raw_attributes": attributes}
        try:
            arguments, kwargs = self._eval_attributes_string(attributes)
            if arguments is None:
                return result
            result.update(
                parse_operation_attributes(
                    node_name,
                    arguments,
                    kwargs or {},
                )
            )
        except Exception as exc:  # noqa: BLE001 - optional backend fallback
            if self.debug:
                print(
                    "Warning: Failed to parse attributes for "
                    f"{node_name}: {exc}"
                )
        return result

    @staticmethod
    def _replace_balanced_calls(
        text: str, func_name: str, replacement: str
    ) -> str:
        """Replace func_name(...) using balanced parenthesis matching."""
        result = []
        i = 0
        tag = func_name + "("
        while i < len(text):
            if text[i : i + len(tag)] == tag:
                depth = 1
                j = i + len(tag)
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                result.append(replacement)
                i = j
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    def _eval_attributes_string(
        self,
        attributes: str,
    ) -> tuple[list[Any] | None, dict[str, Any] | None]:
        """Safely evaluate torchview attributes string.

        Replaces Tensor(...) and slice(...) with placeholders, quotes bare
        keyword dict keys, then evaluates the string.

        Args:
            attributes: Stringified attributes from torchview.

        Returns:
            Tuple of (args_list, kwargs_dict) or (None, None) on failure.
        """
        try:
            processed = attributes

            processed = self._replace_balanced_calls(
                processed, "Tensor", "{'tensor_placeholder': True}"
            )
            processed = re.sub(
                r"torch\.(\w+)",
                lambda match: repr(f"__torch_{match.group(1)}__"),
                processed,
            )
            processed = processed.replace("Ellipsis", repr("__ellipsis__"))
            processed = processed.replace("...", repr("__ellipsis__"))

            # Quote bare keyword keys: {tensor: v, a: v} -> {"tensor": v, "a": v}
            processed = re.sub(
                r"(?<=[{,])\s*([a-zA-Z_]\w*)\s*:", r' "\1":', processed
            )

            parsed = eval(
                processed,
                {"__builtins__": {}},
                {
                    "inf": float("inf"),
                    "nan": float("nan"),
                    "slice": lambda start=None, stop=None, step=None: {
                        "slice": [start, stop, step]
                    },
                },
            )

            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                args_list = (
                    parsed[0] if isinstance(parsed[0], (list, tuple)) else []
                )
                kwargs_dict = parsed[1] if isinstance(parsed[1], dict) else {}
                return list(args_list), kwargs_dict

            return None, None

        except Exception as e:  # noqa: BLE001 - optional backend fallback
            if self.debug:
                print(f"Warning: Failed to eval attributes: {e}")
            return None, None
