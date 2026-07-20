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

"""Backward pass graph extraction through pinned AOTAutograd semantics."""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn as nn
import yaml

from solar.common.utils import ensure_directory


class BackwardProcessor:
    """Extract backward computation graphs from PyTorch models.

    The supported path serializes and verifies the joint forward/backward graph
    produced by the pinned PyTorch 2.11 AOTAutograd API.
    """

    def __init__(self, debug: bool = False):
        """Initialize BackwardProcessor.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug

    def extract_backward_graph(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target: torch.Tensor,
        output_dir: str,
        model_name: str = "model",
    ) -> Optional[Dict[str, Any]]:
        """Extract backward computation graph.

        Args:
            model: PyTorch model.
            inputs: List of input tensors for forward pass.
            loss_fn: Loss function that takes (output, target) -> scalar loss.
            target: Target tensor for loss computation.
            output_dir: Directory to save backward graph.
            model_name: Name for the model (used in file naming).

        Returns:
            Dictionary containing backward graph data, or None if failed.
        """
        if not str(torch.__version__).startswith("2.11.0"):
            raise RuntimeError(
                "formal backward extraction requires the pinned PyTorch 2.11.0 AOTAutograd API"
            )
        from torch._functorch.aot_autograd import aot_export_module

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(parameter.is_floating_point())
        inputs_with_grad: list[Any] = []
        for value in inputs:
            if isinstance(value, torch.Tensor):
                clone = value.detach().clone()
                if clone.is_floating_point() or clone.is_complex():
                    clone.requires_grad_(True)
                inputs_with_grad.append(clone)
            else:
                inputs_with_grad.append(value)

        class JointWrapper(nn.Module):
            def __init__(
                self,
                wrapped: nn.Module,
                objective: Callable[..., torch.Tensor],
                objective_target: Any,
            ):
                super().__init__()
                self.model = wrapped
                self.objective = objective
                if isinstance(objective_target, torch.Tensor):
                    self.register_buffer("objective_target", objective_target)
                else:
                    self.objective_target = objective_target

            def forward(self, *args: Any) -> tuple[torch.Tensor]:
                output = self.model(*args)
                loss = self.objective(output, self.objective_target)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise RuntimeError(
                        "backward objective must return one scalar tensor"
                    )
                return (loss,)

        wrapper = JointWrapper(model, loss_fn, target).eval()
        joint_module, signature = aot_export_module(
            wrapper,
            tuple(inputs_with_grad),
            trace_joint=True,
            output_loss_index=0,
        )
        graph = self._serialize_aot_joint_graph(joint_module, signature, model_name)
        destination = ensure_directory(output_dir) / "joint_graph.yaml"
        destination.write_text(yaml.safe_dump(graph, sort_keys=False))
        serialized_graph = yaml.safe_load(destination.read_text())
        self._verify_aot_joint_gradients(
            wrapper,
            joint_module,
            serialized_graph,
            signature,
            tuple(inputs_with_grad),
        )
        return serialized_graph

    @staticmethod
    def _serialize_argument(value: Any, inputs: list[Any]) -> Any:
        import torch.fx

        if isinstance(value, torch.fx.Node):
            return {"tensor": inputs.index(value)}
        if isinstance(value, (tuple, list)):
            return [
                BackwardProcessor._serialize_argument(item, inputs) for item in value
            ]
        if isinstance(value, torch.dtype):
            return {"dtype": str(value).replace("torch.", "")}
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if value is torch.preserve_format:
            return "preserve_format"
        if value is torch.contiguous_format:
            return "contiguous_format"
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"value": value}
        return {"value": str(value)}

    @staticmethod
    def _tensor_metadata(value: Any) -> list[tuple[list[int], str]]:
        if isinstance(value, torch.Tensor):
            return [(list(value.shape), str(value.dtype))]
        if isinstance(value, (tuple, list)):
            result: list[tuple[list[int], str]] = []
            for item in value:
                result.extend(BackwardProcessor._tensor_metadata(item))
            return result
        return []

    @staticmethod
    def _schema_effects(
        node: Any,
        input_nodes: list[Any],
        *,
        target_name: str,
        exact_target: str,
        output_arity: int,
    ) -> dict[str, Any]:
        """Translate an ATen FunctionSchema alias contract to executable IR."""
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            raise RuntimeError(f"AOT target has no FunctionSchema: {node.target}")

        def tensor_indices(value: Any) -> set[int]:
            import torch.fx

            if isinstance(value, torch.fx.Node):
                return {input_nodes.index(value)}
            if isinstance(value, (tuple, list)):
                return {index for item in value for index in tensor_indices(item)}
            if isinstance(value, dict):
                return {
                    index for item in value.values() for index in tensor_indices(item)
                }
            return set()

        positional = list(node.args)
        aliases_by_input: dict[int, set[str]] = {}
        mutations: set[int] = set()
        for position, argument in enumerate(schema.arguments):
            if position < len(positional):
                value = positional[position]
            elif argument.name in node.kwargs:
                value = node.kwargs[argument.name]
            else:
                continue
            indices = tensor_indices(value)
            alias_info = argument.alias_info
            if alias_info is None:
                continue
            for index in indices:
                aliases_by_input.setdefault(index, set()).update(
                    str(item) for item in alias_info.before_set
                )
                aliases_by_input[index].update(
                    str(item) for item in alias_info.after_set
                )
                if alias_info.is_write:
                    mutations.add(index)

        aliases: list[dict[str, int]] = []
        for output_index, returned in enumerate(schema.returns):
            if output_index >= output_arity or returned.alias_info is None:
                continue
            returned_sets = {
                *(str(item) for item in returned.alias_info.before_set),
                *(str(item) for item in returned.alias_info.after_set),
            }
            for input_index, input_sets in aliases_by_input.items():
                if returned_sets & input_sets:
                    aliases.append({"output": output_index, "input": input_index})

        if target_name.endswith("_") and not mutations and input_nodes:
            raise RuntimeError(
                f"mutating ATen target lacks a schema write effect: {node.target}"
            )
        return {
            "mutates": sorted(mutations),
            "aliases": aliases,
            "atomic": exact_target in {"scatter", "index_put", "index_add"},
            "opaque_library_call": False,
        }

    def _serialize_aot_joint_graph(
        self, graph_module: Any, signature: Any, model_name: str
    ) -> Dict[str, Any]:
        import torch.fx

        nodes = list(graph_module.graph.nodes)
        output_node = next(node for node in nodes if node.op == "output")
        output_values = list(output_node.args[0])
        output_names = [
            node.name for node in output_values if isinstance(node, torch.fx.Node)
        ]
        backward = signature.backward_signature
        gradient_names = set(backward.gradients_to_parameters) | set(
            backward.gradients_to_user_inputs
        )
        forward_names = set(signature.user_outputs)

        # Mark every node needed for the forward output.  Remaining call nodes
        # belong to the captured backward program; shared saved values stay forward.
        forward_nodes: set[torch.fx.Node] = set()

        def visit(node: torch.fx.Node) -> None:
            if node in forward_nodes:
                return
            forward_nodes.add(node)
            for predecessor in node.all_input_nodes:
                visit(predecessor)

        for node in nodes:
            if node.name in forward_names:
                visit(node)

        layers: dict[str, Any] = {}
        node_output_names: dict[torch.fx.Node, list[str]] = {}
        for node in nodes:
            if node.op == "output":
                continue
            metadata = self._tensor_metadata(node.meta.get("val"))
            if node.op == "placeholder":
                if len(metadata) != 1:
                    raise RuntimeError(f"AOT placeholder {node.name} is not one tensor")
                node_output_names[node] = [node.name]
                layers[node.name] = {
                    "type": "start",
                    "phase": "input",
                    "semantic_op": {
                        "kind": "input",
                        "target": "input",
                        "arguments": [],
                        "kwargs": {},
                    },
                    "tensor_names": {"inputs": [], "outputs": [node.name]},
                    "tensor_shapes": {
                        "inputs": [],
                        "outputs": [item[0] for item in metadata],
                    },
                    "tensor_dtypes": {
                        "inputs": [],
                        "outputs": [item[1] for item in metadata],
                    },
                    "connections": {
                        "inputs": [],
                        "outputs": [item.name for item in node.users],
                    },
                }
                continue
            if node.op != "call_function":
                raise RuntimeError(f"unsupported AOT node kind: {node.op}")
            if node.target is operator.getitem:
                source, selected = node.args
                if not isinstance(source, torch.fx.Node) or not isinstance(
                    selected, int
                ):
                    raise RuntimeError(
                        f"AOT getitem {node.name} is not a fixed tensor selection"
                    )
                source_names = node_output_names.get(source) or []
                try:
                    selected_name = source_names[selected]
                except IndexError as exc:
                    raise RuntimeError(
                        f"AOT getitem {node.name} selects an unavailable output"
                    ) from exc
                node_output_names[node] = [node.name]
                layers[node.name] = {
                    "type": "identity",
                    "phase": "forward" if node in forward_nodes else "backward",
                    "semantic_op": {
                        "kind": "aten",
                        "target": "identity",
                        "overload": "default",
                        "arguments": [{"tensor": 0}],
                        "kwargs": {},
                        "effects": {
                            "mutates": [],
                            "aliases": [{"output": 0, "input": 0}],
                            "atomic": False,
                            "opaque_library_call": False,
                        },
                    },
                    "is_real_einsum": False,
                    "is_einsum_supportable": True,
                    "einsum_equation": "",
                    "elementwise_op": "none",
                    "reduction_op": "none",
                    "tensor_names": {
                        "inputs": [selected_name],
                        "outputs": [node.name],
                    },
                    "tensor_shapes": {
                        "inputs": [metadata[0][0]],
                        "outputs": [metadata[0][0]],
                    },
                    "tensor_dtypes": {
                        "inputs": [metadata[0][1]],
                        "outputs": [metadata[0][1]],
                    },
                    "connections": {
                        "inputs": [source.name],
                        "outputs": [
                            item.name for item in node.users if item.op != "output"
                        ],
                    },
                }
                continue
            target_text = str(node.target)
            parts = target_text.split(".")
            if len(parts) < 3 or parts[-3] != "aten":
                raise RuntimeError(f"AOT graph contains non-ATen target: {target_text}")
            target_name = parts[-2]
            overload = parts[-1]
            exact_target = {"t": "transpose"}.get(
                target_name.rstrip("_"), target_name.rstrip("_")
            )
            input_nodes = list(node.all_input_nodes)
            if any(
                len(node_output_names.get(predecessor) or []) != 1
                for predecessor in input_nodes
            ):
                raise RuntimeError(
                    f"AOT node {node.name} consumes a structured tensor value; "
                    "explicit getitem lowering is required"
                )
            output_tensor_names = (
                [node.name]
                if len(metadata) == 1
                else [f"{node.name}.{i}" for i in range(len(metadata))]
            )
            if not output_tensor_names:
                raise RuntimeError(f"AOT node {node.name} has no tensor outputs")
            node_output_names[node] = output_tensor_names
            semantic = {
                "kind": "aten",
                "target": exact_target,
                "overload": overload,
                "arguments": [
                    self._serialize_argument(item, input_nodes) for item in node.args
                ],
                "kwargs": {
                    str(key): self._serialize_argument(value, input_nodes)
                    for key, value in node.kwargs.items()
                },
                "effects": self._schema_effects(
                    node,
                    input_nodes,
                    target_name=target_name,
                    exact_target=exact_target,
                    output_arity=len(metadata),
                ),
            }
            layers[node.name] = {
                "type": target_name.rstrip("_"),
                "phase": "forward" if node in forward_nodes else "backward",
                "semantic_op": semantic,
                "is_real_einsum": False,
                "is_einsum_supportable": True,
                "einsum_equation": "",
                "elementwise_op": "none",
                "reduction_op": "none",
                "tensor_names": {
                    "inputs": [
                        node_output_names[predecessor][0] for predecessor in input_nodes
                    ],
                    "outputs": output_tensor_names,
                },
                "tensor_shapes": {
                    "inputs": [
                        item
                        for predecessor in input_nodes
                        for item, _ in self._tensor_metadata(
                            predecessor.meta.get("val")
                        )
                    ],
                    "outputs": [item[0] for item in metadata],
                },
                "tensor_dtypes": {
                    "inputs": [
                        dtype
                        for predecessor in input_nodes
                        for _, dtype in self._tensor_metadata(
                            predecessor.meta.get("val")
                        )
                    ],
                    "outputs": [item[1] for item in metadata],
                },
                "connections": {
                    "inputs": [predecessor.name for predecessor in input_nodes],
                    "outputs": [
                        item.name for item in node.users if item.op != "output"
                    ],
                },
            }

        saved = sorted(
            node.name
            for node in forward_nodes
            if any(
                user not in forward_nodes and user.op != "output" for user in node.users
            )
        )
        result = {
            "schema_version": 3,
            "model_name": model_name,
            "joint_graph": True,
            "outputs": output_names,
            "layers": layers,
            "graph_signature": {
                "parameters": list(signature.parameters),
                "buffers": list(signature.buffers),
                "user_inputs": list(signature.user_inputs),
                "user_outputs": list(signature.user_outputs),
                "buffers_to_mutate": dict(signature.buffers_to_mutate),
                "parameters_to_mutate": dict(signature.parameters_to_mutate),
                "user_inputs_to_mutate": dict(signature.user_inputs_to_mutate),
                "loss_output": backward.loss_output,
                "gradients_to_parameters": dict(backward.gradients_to_parameters),
                "gradients_to_user_inputs": dict(backward.gradients_to_user_inputs),
                "saved_tensors": saved,
                "joint_outputs": output_names,
                "gradient_outputs": sorted(gradient_names),
            },
        }
        from solar.einsum.semantics import validate_semantic_graph

        validate_semantic_graph(result)
        return result

    @staticmethod
    def _verify_aot_joint_gradients(
        wrapper: nn.Module,
        joint_module: Any,
        graph: dict[str, Any],
        signature: Any,
        inputs: tuple[Any, ...],
    ) -> None:
        named_parameters = dict(wrapper.named_parameters())
        named_buffers = dict(wrapper.named_buffers())
        placeholders = [
            node for node in joint_module.graph.nodes if node.op == "placeholder"
        ]
        user_iter = iter(inputs)
        arguments: list[Any] = []
        for placeholder in placeholders:
            if placeholder.name in signature.inputs_to_parameters:
                arguments.append(
                    named_parameters[signature.inputs_to_parameters[placeholder.name]]
                )
            elif placeholder.name in signature.inputs_to_buffers:
                arguments.append(
                    named_buffers[signature.inputs_to_buffers[placeholder.name]]
                )
            else:
                arguments.append(next(user_iter))

        def clone_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                result = value.detach().clone()
                result.requires_grad_(value.requires_grad)
                return result
            return value

        raw_arguments = [clone_value(value) for value in arguments]
        serialized_arguments = [clone_value(value) for value in arguments]
        actual_outputs = tuple(joint_module(*raw_arguments))
        output_node = next(
            node for node in joint_module.graph.nodes if node.op == "output"
        )
        output_names = [node.name for node in output_node.args[0]]
        actual_by_name = dict(zip(output_names, actual_outputs))

        from solar.verification import EinsumGraphExecutor

        serialized_outputs = EinsumGraphExecutor(graph)(*serialized_arguments)
        serialized_outputs = (
            tuple(serialized_outputs)
            if isinstance(serialized_outputs, (tuple, list))
            else (serialized_outputs,)
        )
        if len(serialized_outputs) != len(actual_outputs):
            raise RuntimeError("serialized AOT joint output arity mismatch")
        serialized_by_name = dict(zip(output_names, serialized_outputs))
        for output_name, actual_value in actual_by_name.items():
            serialized_value = serialized_by_name[output_name]
            torch.testing.assert_close(serialized_value, actual_value, equal_nan=True)
        for raw_argument, serialized_argument in zip(
            raw_arguments, serialized_arguments
        ):
            if isinstance(raw_argument, torch.Tensor):
                torch.testing.assert_close(
                    serialized_argument, raw_argument, equal_nan=True
                )

        def storage_relation(values: list[Any]) -> tuple[tuple[bool, ...], ...]:
            def aliases(left: Any, right: Any) -> bool:
                if not isinstance(left, torch.Tensor) or not isinstance(
                    right, torch.Tensor
                ):
                    return False
                return (
                    left is right
                    or left.untyped_storage()._cdata == right.untyped_storage()._cdata
                )

            return tuple(
                tuple(aliases(left, right) for right in values) for left in values
            )

        raw_values = [*raw_arguments, *actual_outputs]
        serialized_values = [*serialized_arguments, *serialized_outputs]
        if storage_relation(raw_values) != storage_relation(serialized_values):
            raise RuntimeError("serialized AOT joint alias relationships drifted")

        reference_loss = wrapper(*inputs)[0]
        differentiable: list[torch.Tensor] = [
            parameter for parameter in wrapper.parameters() if parameter.requires_grad
        ] + [
            value
            for value in inputs
            if isinstance(value, torch.Tensor) and value.requires_grad
        ]
        expected = torch.autograd.grad(
            reference_loss, differentiable, allow_unused=True
        )
        gradient_names = [
            *signature.backward_signature.gradients_to_parameters,
            *signature.backward_signature.gradients_to_user_inputs,
        ]
        actual = [serialized_by_name[name] for name in gradient_names]
        if len(actual) != len(expected):
            raise RuntimeError("AOT joint gradient arity mismatch")
        for expected_value, actual_value in zip(expected, actual):
            if expected_value is None or actual_value is None:
                if expected_value is not actual_value:
                    raise RuntimeError("AOT joint graph omitted a required gradient")
                continue
            torch.testing.assert_close(actual_value, expected_value, equal_nan=True)
