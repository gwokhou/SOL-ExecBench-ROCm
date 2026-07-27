# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned, fail-closed semantics for executable SOLAR graphs."""

# Semantic target classifications deliberately mirror the executor dispatch.
# pylint: disable=duplicate-code,too-many-locals,import-outside-toplevel,too-many-branches

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solar.schema_versions import EINSUM_GRAPH_SCHEMA_VERSION


class SemanticGraphError(ValueError):
    """A graph does not contain complete, executable operation semantics."""


def _collect_tensor_references(value: Any, references: set[int]) -> None:
    """Add every serialized tensor reference nested under ``value``."""
    if isinstance(value, Mapping):
        if "tensor" in value:
            references.add(int(value["tensor"]))
            return
        for item in value.values():
            _collect_tensor_references(item, references)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_references(item, references)


SUPPORTED_ATEN_TARGETS = frozenset(
    {
        "__and__",
        "__invert__",
        "abs",
        "add",
        "addmm",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "batch_norm",
        "bitwise_and",
        "bitwise_not",
        "cat",
        "chunk",
        "clamp",
        "clone",
        "contiguous",
        "copy",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "cos",
        "cumsum",
        "dequantize",
        "detach",
        "div",
        "elu",
        "embedding",
        "embedding_bag",
        "exp",
        "expand",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "float",
        "flatten",
        "gather",
        "gelu",
        "group_norm",
        "getitem",
        "hardsigmoid",
        "hardswish",
        "half",
        "identity",
        "index_add",
        "index_copy",
        "index_put",
        "index_select",
        "int",
        "layer_norm",
        "linear",
        "log",
        "log_softmax",
        "logsumexp",
        "long",
        "maximum",
        "matmul",
        "masked_fill",
        "mean",
        "minimum",
        "mish",
        "mm",
        "bmm",
        "mul",
        "narrow",
        "neg",
        "ones_like",
        "permute",
        "pow",
        "prod",
        "quantize_per_channel",
        "quantize_per_tensor",
        "relu",
        "repeat",
        "repeat_interleave",
        "reshape",
        "rsqrt",
        "scaled_dot_product_attention",
        "scatter",
        "scatter_add",
        "select",
        "sigmoid",
        "silu",
        "sin",
        "slice",
        "softmax",
        "split",
        "sqrt",
        "square",
        "squeeze",
        "stack",
        "sub",
        "sum",
        "tanh",
        "transpose",
        "to",
        "type_as",
        "unsqueeze",
        "view",
        "vstack",
        "where",
        "zeros_like",
    },
)


def validate_semantic_graph(graph: Mapping[str, Any]) -> None:
    """Validate the latest graph contract without accepting legacy schemas."""
    if int(graph.get("schema_version", 0)) != EINSUM_GRAPH_SCHEMA_VERSION:
        raise SemanticGraphError(
            "einsum graph must use current "
            f"schema_version={EINSUM_GRAPH_SCHEMA_VERSION}",
        )
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise SemanticGraphError("einsum graph has no layers")
    for layer_id, layer in layers.items():
        _validate_semantic_layer(str(layer_id), layer)


def _validate_semantic_layer(layer_id: str, layer: Any) -> None:
    if not isinstance(layer, Mapping):
        raise SemanticGraphError(f"layer {layer_id} is not a mapping")
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    names = layer.get("tensor_names") or {}
    for side in ("inputs", "outputs"):
        arity = len(shapes.get(side) or [])
        if (
            len(dtypes.get(side) or []) != arity
            or len(names.get(side) or []) != arity
        ):
            raise SemanticGraphError(
                f"layer {layer_id} lacks explicit {side} "
                "name/shape/dtype metadata",
            )
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        raise SemanticGraphError(f"layer {layer_id} has no semantic_op")
    kind = str(semantic.get("kind", ""))
    if kind == "input":
        return
    if kind == "einsum":
        equation = str(semantic.get("equation", ""))
        if not equation or "->" not in equation:
            raise SemanticGraphError(
                f"layer {layer_id} has no exact einsum equation",
            )
        return
    if kind != "aten":
        raise SemanticGraphError(
            f"layer {layer_id} is not executable exactly",
        )
    _validate_aten_semantics(layer_id, semantic, names)


def _validate_aten_semantics(
    layer_id: str,
    semantic: Mapping[str, Any],
    names: Mapping[str, Any],
) -> None:
    target = str(semantic.get("target", ""))
    if target not in SUPPORTED_ATEN_TARGETS:
        import torch

        if not target.isidentifier() or not hasattr(torch.ops.aten, target):
            raise SemanticGraphError(
                f"layer {layer_id} uses unsupported exact operation {target!r}",
            )
    if not isinstance(semantic.get("arguments"), list):
        raise SemanticGraphError(
            f"layer {layer_id} lacks explicit arguments",
        )
    arguments = semantic.get("arguments") or []
    kwargs = semantic.get("kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid keyword arguments",
        )
    input_arity = len(names.get("inputs") or [])
    referenced_tensors: set[int] = set()
    _collect_tensor_references(arguments, referenced_tensors)
    _collect_tensor_references(kwargs, referenced_tensors)
    if any(index < 0 or index >= input_arity for index in referenced_tensors):
        raise SemanticGraphError(
            f"layer {layer_id} references a tensor outside its input metadata",
        )
    if input_arity and referenced_tensors != set(range(input_arity)):
        raise SemanticGraphError(
            f"layer {layer_id} does not preserve every ordered tensor argument",
        )
    _validate_effects(layer_id, semantic, names, input_arity)
    _validate_required_parameters(
        layer_id,
        target,
        arguments,
        kwargs,
    )


def _validate_effects(
    layer_id: str,
    semantic: Mapping[str, Any],
    names: Mapping[str, Any],
    input_arity: int,
) -> None:
    effects = semantic.get("effects")
    if not isinstance(effects, Mapping):
        raise SemanticGraphError(f"layer {layer_id} lacks explicit effects")
    mutations = effects.get("mutates")
    aliases = effects.get("aliases")
    if not isinstance(mutations, list) or not isinstance(aliases, list):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid mutation/alias effects",
        )
    if any(int(index) < 0 or int(index) >= input_arity for index in mutations):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid mutation target",
        )
    output_arity = len(names.get("outputs") or [])
    for alias in aliases:
        if (
            not isinstance(alias, Mapping)
            or int(alias.get("input", -1)) not in range(input_arity)
            or int(alias.get("output", -1)) not in range(output_arity)
        ):
            raise SemanticGraphError(
                f"layer {layer_id} has invalid alias effect",
            )


def _validate_required_parameters(
    layer_id: str,
    target: str,
    arguments: list[Any],
    kwargs: Mapping[str, Any],
) -> None:
    required_parameters = {
        "chunk": (("chunks", 2),),
        "expand": (("sizes", 2),),
        "gather": (("dim", 3),),
        "getitem": (("item", 2),),
        "index_copy": (("dim", 4),),
        "index_select": (("dim", 3),),
        "log_softmax": (("dim", 2),),
        "logsumexp": (("dim", 2),),
        "narrow": (("dim", 4), ("start", 4), ("length", 4)),
        "permute": (("dims", 2),),
        "repeat": (("repeats", 2),),
        "select": (("dim", 3), ("index", 3)),
        "softmax": (("dim", 2),),
        "split": (("split_size_or_sections", 2),),
        "unsqueeze": (("dim", 2),),
    }
    missing = [
        key
        for key, positional_arity in required_parameters.get(target, ())
        if key not in kwargs and len(arguments) < positional_arity
    ]
    if target == "slice" and not (
        "dim" in kwargs
        and any(key in kwargs for key in ("start", "end", "step"))
    ):
        missing.append("explicit slice bounds")
    if missing:
        raise SemanticGraphError(
            f"layer {layer_id} lacks exact {target} parameters: "
            + ", ".join(missing),
        )


__all__ = [
    "EINSUM_GRAPH_SCHEMA_VERSION",
    "SUPPORTED_ATEN_TARGETS",
    "SemanticGraphError",
    "validate_semantic_graph",
]
