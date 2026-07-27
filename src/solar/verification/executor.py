"""Lazy PyTorch executor for SOLAR semantic einsum graphs."""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping, Sequence

from solar.common.types import DynamicValue
from solar.einsum.semantics import (
    SemanticGraphError,
    validate_semantic_graph,
)
from solar.verification.errors import EinsumExecutionError

_TOKEN = re.compile(r"[A-Za-z][0-9]*")
_UNHANDLED = object()


def torch_equation(equation: str) -> str:
    """Map SOLAR rank tokens, including ``A0``, to PyTorch rank letters."""
    ranks_only = equation.replace("->", "")
    if (
        not equation
        or "->" not in equation
        or any(character in ranks_only for character in "()+-")
    ):
        raise EinsumExecutionError(
            f"unsupported extended einsum equation: {equation!r}",
        )
    tokens: list[str] = []
    for token in _TOKEN.findall(equation):
        if token not in tokens:
            tokens.append(token)
    alphabet = string.ascii_letters
    if len(tokens) > len(alphabet):
        raise EinsumExecutionError(
            "einsum uses more ranks than torch can represent",
        )
    mapping = dict(zip(tokens, alphabet, strict=False))
    return _TOKEN.sub(lambda match: mapping[match.group(0)], equation)


def _shapes(layer: Mapping[str, DynamicValue]) -> list[tuple[int, ...]]:
    outputs = (layer.get("tensor_shapes") or {}).get("outputs") or []
    return [tuple(int(dimension) for dimension in shape) for shape in outputs]


def _decode_semantic_argument(
    argument: DynamicValue,
    operands: Sequence[DynamicValue],
    layer_id: str,
) -> DynamicValue:
    import torch

    if argument == "preserve_format":
        return torch.preserve_format
    if argument == "contiguous_format":
        return torch.contiguous_format
    if isinstance(argument, list):
        return [
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument
        ]
    if isinstance(argument, tuple):
        return tuple(
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument
        )
    if not isinstance(argument, Mapping):
        return argument
    if "tensor" in argument:
        index = int(argument["tensor"])
        if index < 0 or index >= len(operands):
            raise EinsumExecutionError(
                f"layer {layer_id} references missing tensor argument {index}",
            )
        return operands[index]
    if "dtype" in argument:
        dtype = getattr(torch, str(argument["dtype"]), None)
        if not isinstance(dtype, torch.dtype):
            raise EinsumExecutionError(
                f"layer {layer_id} references invalid dtype "
                f"{argument['dtype']!r}",
            )
        return dtype
    if "device" in argument:
        return torch.device(str(argument["device"]))
    if "layout" in argument:
        layout = getattr(torch, str(argument["layout"]), None)
        if not isinstance(layout, torch.layout):
            raise EinsumExecutionError(
                f"layer {layer_id} references invalid layout "
                f"{argument['layout']!r}",
            )
        return layout
    if "value" in argument:
        value = argument["value"]
        if value == "__ellipsis__":
            return Ellipsis
        if value == "preserve_format":
            return torch.preserve_format
        if value == "contiguous_format":
            return torch.contiguous_format
        return value
    if "slice" in argument:
        values = [
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument["slice"]
        ]
        return slice(*values)
    raise EinsumExecutionError(
        f"layer {layer_id} has an invalid semantic argument",
    )


class EinsumGraphExecutor:
    """Execute the exact subset of extended einsum understood by SOLAR."""

    def __init__(
        self,
        graph: Mapping[str, DynamicValue],
        *,
        check_shapes: bool = True,
    ) -> None:
        """Validate and configure an executable semantic graph."""
        try:
            validate_semantic_graph(graph)
        except SemanticGraphError as exc:
            raise EinsumExecutionError(str(exc)) from exc
        layers = graph.get("layers") or {}
        if not isinstance(layers, Mapping) or not layers:
            raise EinsumExecutionError("einsum graph has no layers")
        self.layers = dict(layers)
        declared_outputs = graph.get("outputs")
        if declared_outputs is None:
            declared_outputs = (graph.get("graph_signature") or {}).get(
                "joint_outputs",
            )
        self.declared_outputs = (
            [str(name) for name in declared_outputs]
            if isinstance(declared_outputs, list)
            else None
        )
        self.check_shapes = check_shapes
        self._validate_layers()

    def _validate_layers(self) -> None:
        for layer_id, layer in self.layers.items():
            if not isinstance(layer, Mapping):
                raise EinsumExecutionError(f"layer {layer_id} is not a mapping")
            if str(layer.get("type", "")).lower() == "start":
                continue
            dtypes = layer.get("tensor_dtypes") or {}
            shapes = layer.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                if len(dtypes.get(side) or []) != len(shapes.get(side) or []):
                    raise EinsumExecutionError(
                        f"layer {layer_id} lacks explicit per-tensor "
                        "dtype metadata",
                    )

    def __call__(self, *inputs: DynamicValue) -> DynamicValue:
        """Execute the graph for positional inputs and return declared outputs."""
        values, start_ids, produced, input_index = self._bind_start_inputs(
            inputs,
        )
        input_index = self._bind_external_inputs(
            inputs,
            input_index,
            start_ids,
            produced,
            values,
        )
        if input_index != len(inputs):
            raise EinsumExecutionError(
                f"graph consumes {input_index} inputs but reference supplied "
                f"{len(inputs)}",
            )
        consumed = self._execute_pending(values, start_ids)
        return self._terminal_outputs(values, produced, consumed)

    def _bind_start_inputs(
        self,
        inputs: Sequence[DynamicValue],
    ) -> tuple[
        dict[str, DynamicValue],
        list[str],
        set[str],
        int,
    ]:
        values: dict[str, DynamicValue] = {}
        produced = {
            name
            for layer in self.layers.values()
            for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
        }
        start_ids = [
            layer_id
            for layer_id, layer in self.layers.items()
            if str(layer.get("type", "")).lower() == "start"
        ]
        input_index = 0
        for layer_id in start_ids:
            names = (self.layers[layer_id].get("tensor_names") or {}).get(
                "outputs",
            ) or []
            for name in names:
                if input_index >= len(inputs):
                    raise EinsumExecutionError(
                        "not enough inputs for graph start tensors",
                    )
                values[str(name)] = inputs[input_index]
                input_index += 1
        return values, start_ids, produced, input_index

    def _bind_external_inputs(
        self,
        inputs: Sequence[DynamicValue],
        input_index: int,
        start_ids: Sequence[str],
        produced: set[str],
        values: dict[str, DynamicValue],
    ) -> int:
        external_names: list[str] = []
        for layer_id, layer in self.layers.items():
            if layer_id in start_ids:
                continue
            for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                if (
                    name not in produced
                    and name not in values
                    and name not in external_names
                ):
                    external_names.append(str(name))
        for name in external_names:
            if input_index >= len(inputs):
                raise EinsumExecutionError(f"missing external tensor {name}")
            values[name] = inputs[input_index]
            input_index += 1
        return input_index

    def _execute_pending(
        self,
        values: dict[str, DynamicValue],
        start_ids: Sequence[str],
    ) -> set[str]:
        pending = {
            key: value
            for key, value in self.layers.items()
            if key not in start_ids
        }
        consumed: set[str] = set()
        while pending:
            progressed = False
            for layer_id, layer in list(pending.items()):
                names = _tensor_names(layer, "inputs")
                if not all(name in values for name in names):
                    continue
                result = self._execute_layer(
                    layer_id,
                    layer,
                    [values[name] for name in names],
                )
                self._store_outputs(layer_id, layer, result, values)
                consumed.update(names)
                del pending[layer_id]
                progressed = True
            if not progressed:
                missing = {
                    layer_id: [
                        name
                        for name in _tensor_names(layer, "inputs")
                        if name not in values
                    ]
                    for layer_id, layer in pending.items()
                }
                raise EinsumExecutionError(
                    f"unresolvable graph dependencies: {missing}",
                )
        return consumed

    def _store_outputs(
        self,
        layer_id: str,
        layer: Mapping[str, DynamicValue],
        result: DynamicValue,
        values: dict[str, DynamicValue],
    ) -> None:
        import torch

        output_names = _tensor_names(layer, "outputs")
        results = (
            list(result) if isinstance(result, (tuple, list)) else [result]
        )
        if len(output_names) != len(results):
            raise EinsumExecutionError(
                f"layer {layer_id} returned {len(results)} outputs, "
                f"expected {len(output_names)}",
            )
        expected_shapes = _shapes(layer)
        for index, (output_name, output) in enumerate(
            zip(output_names, results, strict=True),
        ):
            if not isinstance(output, torch.Tensor):
                raise EinsumExecutionError(
                    f"layer {layer_id} output {index} is not a tensor",
                )
            if (
                self.check_shapes
                and tuple(output.shape) != expected_shapes[index]
            ):
                raise EinsumExecutionError(
                    f"layer {layer_id} output {index} produced "
                    f"{tuple(output.shape)}, expected {expected_shapes[index]}",
                )
            values[output_name] = output

    def _terminal_outputs(
        self,
        values: dict[str, DynamicValue],
        produced: set[str],
        consumed: set[str],
    ) -> DynamicValue:
        if self.declared_outputs is not None:
            missing = [
                name for name in self.declared_outputs if name not in values
            ]
            if missing:
                raise EinsumExecutionError(
                    f"graph declares unavailable outputs: {missing}",
                )
            ordered = self.declared_outputs
        else:
            terminal = {
                name
                for name in produced
                if name not in consumed and name in values
            }
            ordered = [
                name
                for layer in self.layers.values()
                for name in _tensor_names(layer, "outputs")
                if name in terminal
            ]
        if not ordered:
            raise EinsumExecutionError("einsum graph has no terminal output")
        outputs = tuple(values[name] for name in ordered)
        return outputs[0] if len(outputs) == 1 else outputs

    def _execute_layer(
        self,
        layer_id: str,
        layer: Mapping[str, DynamicValue],
        operands: Sequence[DynamicValue],
    ) -> DynamicValue:
        import torch

        semantic = layer["semantic_op"]
        if semantic["kind"] == "einsum":
            return torch.einsum(
                torch_equation(str(semantic["equation"])),
                *operands,
            )
        arguments = [
            _decode_semantic_argument(item, operands, layer_id)
            for item in semantic.get("arguments") or []
        ]
        kwargs = {
            str(key): _decode_semantic_argument(value, operands, layer_id)
            for key, value in (semantic.get("kwargs") or {}).items()
        }
        target = str(semantic.get("target", ""))
        handlers = (
            _execute_exact_aten,
            _execute_mutation,
            _execute_arithmetic,
            _execute_shape,
            _execute_indexing,
            _execute_functional,
            _execute_quantized,
            _execute_aten_fallback,
        )
        for handler in handlers:
            result = handler(
                target,
                arguments,
                kwargs,
                semantic,
                layer_id,
                _shapes(layer),
            )
            if result is not _UNHANDLED:
                return result
        raise EinsumExecutionError(
            f"operation {target!r} at {layer_id} is not executable exactly",
        )


def _tensor_names(
    layer: Mapping[str, DynamicValue],
    side: str,
) -> list[str]:
    return [
        str(name)
        for name in ((layer.get("tensor_names") or {}).get(side) or [])
    ]


def _execute_exact_aten(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del target, layer_id, output_shapes
    import torch

    exact_target = semantic.get("exact_target")
    if semantic.get("kind") != "aten" or not isinstance(exact_target, str):
        return _UNHANDLED
    packet = getattr(torch.ops.aten, exact_target, None)
    overload_name = str(semantic.get("overload", "default"))
    overload = getattr(packet, overload_name, None) if packet else None
    if overload is None:
        raise EinsumExecutionError(
            f"ATen operation {exact_target}.{overload_name} is unavailable",
        )
    return overload(*arguments, **kwargs)


def _execute_mutation(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del output_shapes
    if not (semantic.get("effects") or {}).get("mutates"):
        return _UNHANDLED
    if not arguments:
        raise EinsumExecutionError(
            f"mutating operation {target!r} at {layer_id} has no receiver",
        )
    method = getattr(arguments[0], f"{target}_", None)
    if method is None:
        raise EinsumExecutionError(
            f"mutating operation {target!r} at {layer_id} is unavailable",
        )
    return method(*arguments[1:], **kwargs)


def _execute_arithmetic(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch
    import torch.nn.functional as functional

    operations = {**_binary_operations(), **_unary_operations()}
    if target in operations:
        return operations[target](*arguments, **kwargs)
    if target in {"mm", "bmm", "matmul", "addmm", "where"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "masked_fill":
        return arguments[0].masked_fill(*arguments[1:], **kwargs)
    if target == "cumsum":
        return torch.cumsum(*arguments, **kwargs)
    if target in {"softmax", "log_softmax"}:
        return getattr(functional, target)(*arguments, **kwargs)
    if target in {
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "logsumexp",
    }:
        return getattr(torch, target)(*arguments, **kwargs)
    return _UNHANDLED


def _binary_operations() -> dict[str, Callable[..., DynamicValue]]:
    import torch

    return {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "eq": torch.eq,
        "ge": torch.ge,
        "gt": torch.gt,
        "le": torch.le,
        "lt": torch.lt,
        "ne": torch.ne,
        "pow": torch.pow,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
        "bitwise_and": torch.bitwise_and,
    }


def _unary_operations() -> dict[str, Callable[..., DynamicValue]]:
    import torch
    import torch.nn.functional as functional

    return {
        "abs": torch.abs,
        "bitwise_not": torch.bitwise_not,
        "cos": torch.cos,
        "elu": functional.elu,
        "exp": torch.exp,
        "gelu": functional.gelu,
        "hardsigmoid": functional.hardsigmoid,
        "hardswish": functional.hardswish,
        "leaky_relu": functional.leaky_relu,
        "log": torch.log,
        "mish": functional.mish,
        "neg": torch.neg,
        "relu": functional.relu,
        "rsqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "silu": functional.silu,
        "sin": torch.sin,
        "sqrt": torch.sqrt,
        "square": torch.square,
        "tanh": torch.tanh,
    }


def _execute_shape(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del semantic
    import torch

    if target == "identity":
        return arguments[0]
    if target == "to":
        return arguments[0].to(*arguments[1:], **kwargs)
    if target in {"bfloat16", "float", "half", "int", "long"}:
        return getattr(arguments[0], target)()
    if target in {"type_as", "clone", "detach"}:
        return getattr(arguments[0], target)(*arguments[1:], **kwargs)
    if target in {"view", "reshape"}:
        if len(arguments) > 1:
            return getattr(arguments[0], target)(*arguments[1:], **kwargs)
        shape = kwargs.pop("shape", output_shapes[0])
        return getattr(arguments[0], target)(tuple(shape))
    if target == "flatten":
        return torch.flatten(*arguments, **kwargs)
    if target == "contiguous":
        return arguments[0].contiguous(**kwargs)
    if target in {
        "squeeze",
        "unsqueeze",
        "permute",
        "repeat",
        "repeat_interleave",
        "expand",
    }:
        return getattr(arguments[0], target)(*arguments[1:], **kwargs)
    if target == "transpose":
        if len(arguments) == 1 and not kwargs:
            if arguments[0].ndim != 2:
                raise EinsumExecutionError(
                    f"layer {layer_id} requires explicit transpose dimensions",
                )
            return arguments[0].t()
        return torch.transpose(*arguments, **kwargs)
    if target in {"cat", "stack"}:
        if arguments and isinstance(arguments[0], (list, tuple)):
            return getattr(torch, target)(*arguments, **kwargs)
        return getattr(torch, target)(arguments, **kwargs)
    if target == "vstack":
        return torch.vstack(*arguments, **kwargs)
    if target in {"chunk", "split"}:
        return getattr(torch, target)(*arguments, **kwargs)
    return _UNHANDLED


def _execute_indexing(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch

    if target in {"gather", "scatter", "index_select", "select", "narrow"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "getitem":
        index = arguments[1]
        if isinstance(index, list) and any(
            isinstance(item, slice) or item is None or item is Ellipsis
            for item in index
        ):
            index = tuple(index)
        return arguments[0][index]
    if target == "slice":
        dimension = int(kwargs.get("dim", 0))
        slices = [slice(None)] * arguments[0].ndim
        slices[dimension] = slice(
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("step"),
        )
        return arguments[0][tuple(slices)]
    return _UNHANDLED


def _execute_functional(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch.nn.functional as functional

    if target == "linear" or target.startswith("conv_transpose"):
        return getattr(functional, target)(*arguments, **kwargs)
    if target in {
        "conv1d",
        "conv2d",
        "conv3d",
        "batch_norm",
        "group_norm",
        "instance_norm",
        "layer_norm",
        "embedding",
        "embedding_bag",
        "dropout",
        "max_pool2d",
        "scaled_dot_product_attention",
    }:
        return getattr(functional, target)(*arguments, **kwargs)
    return _UNHANDLED


def _execute_quantized(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch

    if target in {
        "quantize_per_tensor",
        "quantize_per_channel",
        "fake_quantize_per_tensor_affine",
        "fake_quantize_per_channel_affine",
    }:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "dequantize":
        return arguments[0].dequantize()
    if target in {"ones_like", "zeros_like"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "clamp":
        return torch.clamp(*arguments, **kwargs)
    return _UNHANDLED


def _execute_aten_fallback(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: list[tuple[int, ...]],
) -> DynamicValue:
    del layer_id, output_shapes
    import torch

    if not target.isidentifier() or not hasattr(torch.ops.aten, target):
        return _UNHANDLED
    packet = getattr(torch.ops.aten, target)
    overload_name = str(semantic.get("overload", "default"))
    overload = getattr(packet, overload_name, None)
    if overload is None:
        raise EinsumExecutionError(
            f"ATen operation {target}.{overload_name} is unavailable",
        )
    return overload(*arguments, **kwargs)
