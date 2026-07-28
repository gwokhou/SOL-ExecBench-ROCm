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

from solar.graph.torchview.constants import (
    BOOLEAN_ATTRS,
    GEOMETRIC_ATTRS,
    MODULE_ATTR_NAMES,
)
from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.processor_contract import TorchviewProcessorContract
from solar.types import TensorShape


class TorchviewMetadataMixin(TorchviewProcessorContract):
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
        #
        #  Extract from node hierarchy if available (hierarchical module info)
        # if hasattr(computation_graph, 'node_hierarchy') and computation_graph.node_hierarchy:
        #     if self._is_hierarchy_useful(computation_graph.node_hierarchy):
        #         if self.debug:
        #             print("Extracting from node_hierarchy...")
        #         return self._extract_from_hierarchy(
        #             computation_graph.node_hierarchy, 'Model'
        #         )

        # Extract from edge_list (flattened computation graph)
        if (
            hasattr(computation_graph, "edge_list")
            and computation_graph.edge_list
        ):
            if self.debug:
                print(
                    f"Extracting from edge_list ({len(computation_graph.edge_list)} edges)..."
                )
            layer_nodes = self._extract_from_edge_list(
                computation_graph, original_model
            )
            return layer_nodes

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
        """Extract input and output shapes from a node.

        Args:
            node: Node object.
            node_type: Type of the node for special handling.
            original_model: Optional model used to recover module dtypes.
            input_shapes: Optional normalized input shapes.
            output_shapes: Optional normalized output shapes.
            original_model: Optional model used to recover module dtypes.
            input_shapes: Optional normalized input shapes.
            output_shapes: Optional normalized output shapes.

        Returns:
            Tuple of (input_shapes, output_shapes).
        """
        input_shapes = []
        output_shapes = []
        node_class = type(node).__name__

        # Special handling for TensorNode - check tensor_shape first
        if node_class == "TensorNode":
            tensor_shape = None
            if hasattr(node, "tensor_shape") and node.tensor_shape is not None:
                if hasattr(node.tensor_shape, "__iter__"):
                    tensor_shape = list(node.tensor_shape)
                else:
                    tensor_shape = [node.tensor_shape]

            if tensor_shape:
                node_name = node_type.lower() if node_type else ""
                if node_name in {
                    "input-tensor",
                    "auxiliary-tensor",
                    "parameter-tensor",
                }:
                    output_shapes = [tensor_shape]
                elif node_name == "output-tensor":
                    input_shapes = [tensor_shape]
                elif node_name == "hidden-tensor":
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                else:
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                return input_shapes, output_shapes

        # Extract input shapes from inputs attribute
        if hasattr(node, "inputs") and node.inputs:
            for inp in node.inputs:
                if (
                    hasattr(inp, "tensor_shape")
                    and inp.tensor_shape is not None
                ):
                    input_shapes.append(list(inp.tensor_shape))
                elif hasattr(inp, "shape"):
                    input_shapes.append(list(inp.shape))
        elif hasattr(node, "input_shape") and node.input_shape:
            if isinstance(node.input_shape, (list, tuple)):
                input_shapes = [list(s) for s in node.input_shape]
            else:
                input_shapes = [list(node.input_shape)]

        # Extract output shapes from outputs attribute
        if hasattr(node, "outputs") and node.outputs:
            for out in node.outputs:
                if (
                    hasattr(out, "tensor_shape")
                    and out.tensor_shape is not None
                ):
                    output_shapes.append(list(out.tensor_shape))
                elif hasattr(out, "shape"):
                    output_shapes.append(list(out.shape))
        elif hasattr(node, "output_shape") and node.output_shape:
            if isinstance(node.output_shape, (list, tuple)):
                output_shapes = [list(s) for s in node.output_shape]
            else:
                output_shapes = [list(node.output_shape)]

        return input_shapes, output_shapes

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
        """Extract input and output dtypes from a node.

        Args:
            node: Node object.
            node_type: Type of the node for special handling.
            original_model: Optional model used to recover module dtypes.
            input_shapes: Optional normalized input shapes.
            output_shapes: Optional normalized output shapes.

        Returns:
            Tuple of (input_dtypes, output_dtypes).
        """
        input_dtypes = []
        output_dtypes = []
        node_class = type(node).__name__

        # Helper function to extract dtype from a tensor-like object
        def get_dtype(obj: Any) -> str | None:
            """Extract dtype string from a tensor-like object."""
            if obj is None:
                return None

            # Check for tensor_dtype attribute (torchview convention)
            if hasattr(obj, "tensor_dtype") and obj.tensor_dtype is not None:
                dtype = obj.tensor_dtype
                if isinstance(dtype, torch.dtype) or dtype is not None:
                    return str(dtype)

            # Check for dtype attribute directly
            if hasattr(obj, "dtype"):
                dtype = obj.dtype
                if isinstance(dtype, torch.dtype):
                    return str(dtype)
                elif isinstance(dtype, str):
                    return dtype

            # Check for tensor attribute that might have dtype
            if hasattr(obj, "tensor") and obj.tensor is not None:
                tensor_obj = obj.tensor
                if hasattr(tensor_obj, "dtype"):
                    return str(tensor_obj.dtype)

            # Check if it's a torch.Tensor directly
            if isinstance(obj, torch.Tensor):
                return str(obj.dtype)

            return None

        # Special handling for TensorNode - mirror shape extraction logic
        if node_class == "TensorNode":
            tensor_dtype = None

            # Try tensor_dtype attribute first (torchview convention)
            if hasattr(node, "tensor_dtype") and node.tensor_dtype is not None:
                tensor_dtype = str(node.tensor_dtype)
            # Try dtype attribute
            elif hasattr(node, "dtype") and node.dtype is not None:
                tensor_dtype = str(node.dtype)
            # Try tensor attribute
            elif hasattr(node, "tensor") and node.tensor is not None:
                if hasattr(node.tensor, "dtype"):
                    tensor_dtype = str(node.tensor.dtype)
            # Try getting from inputs/outputs if available
            elif hasattr(node, "outputs") and node.outputs:
                for out in node.outputs:
                    dtype = get_dtype(out)
                    if dtype:
                        tensor_dtype = dtype
                        break

            # Assign dtype to the same tensor sides used by shape extraction.
            if tensor_dtype:
                return self._classify_tensor_node_dtype(node_type, tensor_dtype)

        # Extract input dtypes from inputs attribute (mirroring shape extraction)
        if hasattr(node, "inputs") and node.inputs:
            for inp in node.inputs:
                dtype = get_dtype(inp)
                if dtype:
                    input_dtypes.append(dtype)
                else:
                    # If we can't get dtype from the input object, try to infer from shape
                    # by checking if it has tensor_shape and we can get dtype from that
                    if hasattr(inp, "tensor_shape") and hasattr(
                        inp, "tensor_dtype"
                    ):
                        dtype = get_dtype(inp)
                        if dtype:
                            input_dtypes.append(dtype)

        # Extract output dtypes from outputs attribute (mirroring shape extraction)
        if hasattr(node, "outputs") and node.outputs:
            for out in node.outputs:
                dtype = get_dtype(out)
                if dtype:
                    output_dtypes.append(dtype)
                else:
                    # Try to get dtype from tensor_shape's source
                    if hasattr(out, "tensor_shape") and hasattr(
                        out, "tensor_dtype"
                    ):
                        dtype = get_dtype(out)
                        if dtype:
                            output_dtypes.append(dtype)

        # Fallback: try input_shape/output_shape attributes (some nodes might have these)
        if (
            not input_dtypes
            and hasattr(node, "input_dtype")
            and node.input_dtype
        ):
            if isinstance(node.input_dtype, (list, tuple)):
                input_dtypes = [str(d) for d in node.input_dtype]
            else:
                input_dtypes = [str(node.input_dtype)]

        if (
            not output_dtypes
            and hasattr(node, "output_dtype")
            and node.output_dtype
        ):
            if isinstance(node.output_dtype, (list, tuple)):
                output_dtypes = [str(d) for d in node.output_dtype]
            else:
                output_dtypes = [str(node.output_dtype)]

        # Fallback: If we have shapes but no dtypes, try to infer from original model
        # This ensures dtype counts match shape counts
        if input_shapes is None:
            input_shapes = []
        if output_shapes is None:
            output_shapes = []

        if original_model is not None:
            if self._cached_default_dtype is None:
                for param in original_model.parameters():
                    if param.dtype is not None:
                        self._cached_default_dtype = str(param.dtype)
                        break
                if self._cached_default_dtype is None:
                    self._cached_default_dtype = "torch.float32"

            default_dtype = self._cached_default_dtype

            while len(input_dtypes) < len(input_shapes):
                input_dtypes.append(default_dtype)

            while len(output_dtypes) < len(output_shapes):
                output_dtypes.append(default_dtype)

        return input_dtypes, output_dtypes

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
            # Fallback: parse the 'attributes' string from torchview
            # Format: "Linear(training=False, in_features=64, out_features=64)"
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

        # Extract module type from the beginning
        # Format: "ModuleType(key=value, ...)"

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
        """Parse torchview stringified attributes to extract function arguments.

        torchview's stringify_attributes() produces strings like:
        - For functions: "[[Tensor(shape=(2, 32, 64), dtype=torch.float32), 1, 2], {}]"
        - This represents [args_list, kwargs_dict]

        We parse this by replacing Tensor(...) with a placeholder and using eval.

        Args:
            attributes: Stringified attributes from torchview.
            node_name: Name of the function (e.g., 'transpose', 'permute').

        Returns:
            Dictionary of parsed arguments.
        """
        result: dict[str, Any] = {}

        if not attributes:
            return result

        # Store raw attributes for debugging
        result["raw_attributes"] = attributes

        try:
            # Parse the attributes string by replacing Tensor(...) with a placeholder
            args_list, kwargs_dict = self._eval_attributes_string(attributes)

            if args_list is None:
                return result
            kwargs_dict = kwargs_dict or {}

            tensor_index = 0

            def semantic_value(value: Any) -> Any:
                nonlocal tensor_index
                if isinstance(value, dict) and value.get("tensor_placeholder"):
                    reference = {"tensor": tensor_index}
                    tensor_index += 1
                    return reference
                if isinstance(value, dict):
                    return {
                        str(key): semantic_value(item)
                        for key, item in value.items()
                    }
                if isinstance(value, (list, tuple)):
                    return [semantic_value(item) for item in value]
                if isinstance(value, str) and value.startswith("__torch_"):
                    name = value.removeprefix("__torch_").removesuffix("__")
                    if name in {
                        "bool",
                        "bfloat16",
                        "float16",
                        "float32",
                        "float64",
                        "int8",
                        "int16",
                        "int32",
                        "int64",
                        "uint8",
                        "float8_e4m3fn",
                        "float8_e5m2",
                    }:
                        return {"dtype": name}
                    return {"value": name}
                return {"value": value}

            # Preserve the exact ordered call signature.  This is the source
            # of truth for executable semantic IR; operation-specific fields
            # below remain useful cost-model diagnostics only.
            result["call_arguments"] = [
                semantic_value(item) for item in args_list
            ]
            result["call_kwargs"] = {
                str(key): semantic_value(value)
                for key, value in kwargs_dict.items()
            }

            # Extract non-tensor arguments (skip index 0 which is usually the input tensor)
            non_tensor_args = [
                arg
                for arg in args_list
                if not isinstance(arg, dict) or "tensor_placeholder" not in arg
            ]
            # Filter out tensor placeholders
            scalar_args = [
                arg
                for arg in non_tensor_args
                if not (isinstance(arg, dict) and "tensor_placeholder" in arg)
            ]

            if node_name == "transpose":
                # transpose(input, dim0, dim1) - extract dim0 and dim1
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if len(int_args) >= 2:
                    result["dim0"] = int_args[0]
                    result["dim1"] = int_args[1]
                    result["transpose_dims"] = [int_args[0], int_args[1]]
                # Also check kwargs
                if kwargs_dict:
                    if "dim0" in kwargs_dict:
                        result["dim0"] = kwargs_dict["dim0"]
                    if "dim1" in kwargs_dict:
                        result["dim1"] = kwargs_dict["dim1"]
                    if "dim0" in result and "dim1" in result:
                        result["transpose_dims"] = [
                            result["dim0"],
                            result["dim1"],
                        ]

            elif node_name == "permute":
                # permute(input, dims) or permute(input, *dims)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result["permute_dims"] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(
                        isinstance(d, int) for d in arg
                    ):
                        result["permute_dims"] = list(arg)
                        break
                # Check kwargs
                if kwargs_dict and "dims" in kwargs_dict:
                    result["permute_dims"] = list(kwargs_dict["dims"])

            elif node_name == "t":
                # t() is always transpose(0, 1) for 2D tensors
                result["dim0"] = 0
                result["dim1"] = 1
                result["transpose_dims"] = [1, 0]

            elif node_name in ("view", "reshape"):
                # view(input, *sizes) or reshape(input, shape)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result["target_shape"] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(
                        isinstance(d, int) for d in arg
                    ):
                        result["target_shape"] = list(arg)
                        break

            elif node_name in (
                "mean",
                "sum",
                "logsumexp",
                "prod",
                "amax",
                "amin",
                "any",
                "all",
                "norm",
                "std",
                "var",
            ):
                # Reduction ops: func(input, dim, keepdim=False)
                # scalar_args may contain dim as int or list of ints
                if kwargs_dict:
                    if "dim" in kwargs_dict:
                        dim_val = kwargs_dict["dim"]
                        result["dim"] = (
                            [dim_val]
                            if isinstance(dim_val, int)
                            else list(dim_val)
                        )
                    if "keepdim" in kwargs_dict:
                        result["keepdim"] = kwargs_dict["keepdim"]
                # dim can also be a positional arg
                if "dim" not in result:
                    for arg in scalar_args:
                        if isinstance(arg, int):
                            result["dim"] = [arg]
                            break
                        elif isinstance(arg, (list, tuple)) and all(
                            isinstance(d, int) for d in arg
                        ):
                            result["dim"] = list(arg)
                            break

        except Exception as e:  # noqa: BLE001 - optional backend fallback
            if self.debug:
                print(
                    f"Warning: Failed to parse attributes for {node_name}: {e}"
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
                    "slice": lambda start=None, stop=None, step=None: {
                        "slice": [start, stop, step]
                    }
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
