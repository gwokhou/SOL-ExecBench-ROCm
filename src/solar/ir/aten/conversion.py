# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact ATen intermediate representation backend."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from solar.graph.contracts import OperatorGraphArtifact
from solar.ir.bindings import bind_graph
from solar.ir.contracts import IRGraphArtifact, IRKind
from solar.schema_versions import ATEN_IR_SCHEMA_VERSION


class AtenIRError(ValueError):
    """An ATen IR graph is incomplete or cannot replay its source exactly."""


_UNINITIALIZED_ALLOCATION_TARGETS = frozenset(
    {"empty", "empty_like", "empty_strided", "new_empty"},
)


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
        "topk",
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


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    output_dir: str | Path,
) -> IRGraphArtifact:
    """Persist the canonical operator graph as one validated ATen IR artifact."""
    traced = _load_operator_graph(operator)
    converted = deepcopy(dict(traced))
    converted["ir_kind"] = "aten"
    converted["schema_version"] = ATEN_IR_SCHEMA_VERSION
    bind_graph(converted, operator)
    validate_aten_graph(converted)
    path = Path(output_dir) / "aten_graph.yaml"
    path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return IRGraphArtifact(path, IRKind.ATEN)


def validate_aten_graph(graph: Mapping[str, Any]) -> None:
    """Validate one exact ATen IR graph without accepting legacy schemas."""
    if graph.get("ir_kind") != IRKind.ATEN.value:
        raise AtenIRError("graph is not ATen IR")
    if graph.get("schema_version") != ATEN_IR_SCHEMA_VERSION:
        raise AtenIRError(
            "ATen graph must use current "
            f"schema_version={ATEN_IR_SCHEMA_VERSION}",
        )
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise AtenIRError("ATen graph has no layers")
    for layer_id, layer in layers.items():
        _validate_aten_layer(str(layer_id), layer)


def _load_operator_graph(operator: OperatorGraphArtifact) -> Mapping[str, Any]:
    traced = operator.document.data
    if traced.get("extraction_kind") != "make_fx_reference_v1":
        raise RuntimeError("ATen graph provenance is not trusted")
    return traced


def _validate_aten_layer(layer_id: str, layer: Any) -> None:
    if not isinstance(layer, Mapping):
        raise AtenIRError(f"layer {layer_id} is not a mapping")
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    for side in ("inputs", "outputs"):
        arity = len(shapes.get(side) or [])
        if (
            len(dtypes.get(side) or []) != arity
            or len(names.get(side) or []) != arity
        ):
            raise AtenIRError(
                f"layer {layer_id} lacks explicit {side} "
                "name/shape/dtype metadata",
            )
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        raise AtenIRError(f"layer {layer_id} has no semantic_op")
    kind = str(semantic.get("kind", ""))
    if kind == "input":
        return
    if kind == "einsum":
        equation = str(semantic.get("equation", ""))
        if not equation or "->" not in equation:
            raise AtenIRError(f"layer {layer_id} has no exact einsum equation")
        return
    if kind != "aten":
        raise AtenIRError(f"layer {layer_id} is not executable exactly")
    _validate_aten_operation(layer_id, semantic, names)


def _validate_aten_operation(
    layer_id: str,
    semantic: Mapping[str, Any],
    names: Mapping[str, Any],
) -> None:
    target = str(semantic.get("target", ""))
    if target in _UNINITIALIZED_ALLOCATION_TARGETS:
        raise AtenIRError(
            f"layer {layer_id} uses unsupported exact operation {target!r}",
        )
    if target not in SUPPORTED_ATEN_TARGETS:
        import torch

        if not target.isidentifier() or not hasattr(torch.ops.aten, target):
            raise AtenIRError(
                f"layer {layer_id} uses unsupported exact operation {target!r}",
            )
    arguments = semantic.get("arguments")
    if not isinstance(arguments, list):
        raise AtenIRError(f"layer {layer_id} lacks explicit arguments")
    kwargs = semantic.get("kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise AtenIRError(f"layer {layer_id} has invalid keyword arguments")
    input_arity = len(names.get("inputs") or [])
    references: set[int] = set()
    _collect_tensor_references(arguments, references)
    _collect_tensor_references(kwargs, references)
    if any(index < 0 or index >= input_arity for index in references):
        raise AtenIRError(
            f"layer {layer_id} references a tensor outside its input metadata",
        )
    if input_arity and references != set(range(input_arity)):
        raise AtenIRError(
            f"layer {layer_id} does not preserve every ordered tensor argument",
        )
    _validate_effects(layer_id, semantic, names, input_arity)
    _validate_required_parameters(
        layer_id,
        target,
        arguments,
        kwargs,
        output_arity=len(names.get("outputs") or []),
    )


def _collect_tensor_references(value: Any, references: set[int]) -> None:
    if isinstance(value, Mapping):
        if "tensor" in value:
            references.add(int(value["tensor"]))
            return
        for item in value.values():
            _collect_tensor_references(item, references)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_references(item, references)


def _validate_effects(
    layer_id: str,
    semantic: Mapping[str, Any],
    names: Mapping[str, Any],
    input_arity: int,
) -> None:
    effects = semantic.get("effects")
    if not isinstance(effects, Mapping):
        raise AtenIRError(f"layer {layer_id} lacks explicit effects")
    mutations = effects.get("mutates")
    aliases = effects.get("aliases")
    if not isinstance(mutations, list) or not isinstance(aliases, list):
        raise AtenIRError(
            f"layer {layer_id} has invalid mutation/alias effects",
        )
    if any(int(index) < 0 or int(index) >= input_arity for index in mutations):
        raise AtenIRError(f"layer {layer_id} has invalid mutation target")
    output_arity = len(names.get("outputs") or [])
    for alias in aliases:
        if (
            not isinstance(alias, Mapping)
            or int(alias.get("input", -1)) not in range(input_arity)
            or int(alias.get("output", -1)) not in range(output_arity)
        ):
            raise AtenIRError(f"layer {layer_id} has invalid alias effect")


def _validate_required_parameters(
    layer_id: str,
    target: str,
    arguments: list[Any],
    kwargs: Mapping[str, Any],
    *,
    output_arity: int,
) -> None:
    required = {
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
        "topk": (("k", 2),),
        "unsqueeze": (("dim", 2),),
    }
    missing = [
        key
        for key, arity in required.get(target, ())
        if key not in kwargs and len(arguments) < arity
    ]
    explicit_slice_dim = len(arguments) >= 2 or "dim" in kwargs
    explicit_slice_bounds = len(arguments) >= 3 or any(
        key in kwargs for key in ("start", "end", "step")
    )
    if target == "slice" and not (explicit_slice_dim and explicit_slice_bounds):
        missing.append("explicit slice bounds")
    if target == "topk" and output_arity != 2:
        missing.append("two output slots")
    if target in {"chunk", "split"} and output_arity < 1:
        missing.append("output slots")
    if missing:
        raise AtenIRError(
            f"layer {layer_id} lacks exact {target} parameters: "
            + ", ".join(missing),
        )


__all__ = [
    "SUPPORTED_ATEN_TARGETS",
    "AtenIRError",
    "convert_operator_graph",
    "validate_aten_graph",
]
