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

"""Principled AccelForge einsum-graph builder for Solar.

Replaces the historical multi-pass rank renamer + AF emit + conflict mint
with a single topologically-ordered union-find
traversal over the stage-2 einsum-graph dict produced by ``_build_einsum_graph``.

Algorithm
=========
A *rank* is a named tensor dimension identified by its physical size. Two
operand positions belong to the same rank iff:

1. **Within-layer same atomic label + same size**: e.g. matmul's
   ``Input=[A,B]``, ``Weight=[B,C]``, ``Output=[A,C]`` — the ``B`` in
   Input and Weight is the same rank (the reduction dim).
2. **Cross-layer producer→consumer position match + same size**: the
   predecessor's output dim ``i`` and the successor's input dim ``i`` are
   the same physical tensor dim.

Two positions with the same label but different sizes are SEPARATE ranks
(handles reduction-with-keepdim cases such as
``Min(dim=1, keepdim=True)``: ``ABCD->ABCD`` with B sized 64 → 1).

Composite labels like ``P+R`` (conv kernel stencil) stay as their own
component in the union-find; the *iterator expression* in their projection
uses the canonical names of the sub-atoms.

The emitted AccelForge graph contains:
- ``rank_sizes``: one entry per equivalence class (canonical name ``R0``,
  ``R1``, ... assigned in topological order of first appearance)
- ``einsums``: one per real layer, with dict-form projections (list form
  when the rank/iter is the trivial identity) referencing canonical names
- top-level ``renames``, ``bits_per_value``, ``persistent_tensors`` blocks

Invariants guaranteed
=====================
1. Every rank name maps to exactly one size — no ``Rk`` reuse across
   layers with different physical sizes.
2. Every tensor accessed in multiple einsums has the same rank tuple.
3. Names are ISL-safe (only ``[A-Za-z0-9_]``).
4. Composite-label positions get their own rank; iterators reference the
   canonical names of sub-atoms.

Usage
=====
From a dict (typical solar internal use)::

    af = build_af_graph_from_dict(einsum_graph)

From an einsum_graph.yaml on disk::

    af = build_af_graph_from_yaml(
        "einsum_graph.yaml", output_path="af_einsum_graph.yaml"
    )
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from solar.ir.extended_einsum.torchview.axis_mapping import (
    AxisMappingRequest,
    derive_axis_mapping,
)

# Op-type classification

# Solar layer-types treated as explicit copy operations in the AF graph.
# Only used to fast-path the "start" entry-point pseudo-node here — non-start
# layers from this set are emitted without ``is_copy_operation`` so AF lets
# the mapper handle them normally (setting True triggers AF's "No backing
# TensorHolder" on shape-changing views; setting False explodes the pmapping
# join space for decoder-class graphs).
_OUTPUT_ROLE_PATTERN = re.compile(r"^Output(?:_\d+)?$")
_INPUT_ROLE_PATTERN = re.compile(r"^(?:Input|Weight)(?:_\d+)?$")
_NON_ID_CHAR = re.compile(r"[^A-Za-z0-9_]")

from solar.ir.extended_einsum.torchview.af_model import (
    _is_input_role,
    _is_output_role,
)

_SHAPE_OP_TYPES: set[str] = {
    "transpose",
    "permute",
    "contiguous",
    "squeeze",
    "unsqueeze",
    "expand",
    "__getitem__",
    "__get__",
    "view",
    "reshape",
}


@dataclass
class _Alias:
    root_tensor: str
    """Tensor name in the surviving (root) producer's output."""

    root_layer: str
    """Name of the layer that produces ``root_tensor``."""

    root_role: str
    """Role under which ``root_layer`` exposes ``root_tensor`` (typically 'Output')."""

    root_dims: list[str]
    """Operand labels of the root producer's output, in order."""

    root_shape: list[int]
    """Shape of the root producer's output, in order."""

    in_to_out: list[list[int]]
    """For each input position of the elided shape-op, the list of output
    positions it maps to. Empty list means the input position was dropped
    (squeeze); multiple means it was broadcast (expand). Lengths and
    indices reference the shape-op's OWN input/output positions, not the
    physical root's (those are reachable transitively via this struct's
    ``root_dims``)."""

    out_to_in: list[int | None]
    """For each output position of the elided shape-op, the input position
    it came from, or None if introduced (unsqueeze)."""


def _derive_pos_mapping(
    layer_name: str,
    layer: dict[str, Any],
    in_dims: list[str],
    in_shape: list[int],
    out_dims: list[str],
    out_shape: list[int],
) -> tuple[list[int | None], list[list[int]]] | None:
    """Return the first safe axis mapping for an elide-able shape op."""
    del layer_name
    return derive_axis_mapping(
        AxisMappingRequest(
            operation=str(layer.get("type", "")),
            input_dims=in_dims,
            input_shape=in_shape,
            output_dims=out_dims,
            output_shape=out_shape,
        )
    )


@dataclass(frozen=True)
class _ShapeOpCandidate:
    """Validated positional metadata for one elide-able shape operation."""

    name: str
    predecessor: str
    output_tensor: str
    input_dims: list[str]
    input_shape: list[int]
    output_to_input: list[int | None]
    input_to_output: list[list[int]]


def _shape_product(shape: list[int]) -> int | None:
    """Return an integer shape product, or ``None`` for malformed metadata."""
    product = 1
    try:
        for size in shape:
            product *= int(size)
    except (TypeError, ValueError):
        return None
    return product


def _shape_op_candidate(
    name: str,
    layer: dict[str, Any],
    diagnostics: list[str],
) -> _ShapeOpCandidate | None:
    """Validate and normalize one potential shape-op alias."""
    if layer.get("is_real_einsum", True):
        return None
    operation = str(layer.get("type", ""))
    if operation not in _SHAPE_OP_TYPES:
        return None
    operands = layer.get("operands") or {}
    inputs = [
        role
        for role in operands
        if _is_input_role(role)
        or (not _is_output_role(role) and role != "start")
    ]
    outputs = [role for role in operands if _is_output_role(role)]
    predecessors = (layer.get("connections") or {}).get("inputs") or []
    if len(inputs) != 1 or len(outputs) != 1 or len(predecessors) != 1:
        return None
    shapes = layer.get("tensor_shapes") or {}
    input_shapes = shapes.get("inputs") or []
    output_shapes = shapes.get("outputs") or []
    output_names = (layer.get("tensor_names") or {}).get("outputs") or []
    if not input_shapes or not output_shapes or not output_names:
        return None
    input_shape = list(input_shapes[0])
    output_shape = list(output_shapes[0])
    if operation != "expand":
        input_product = _shape_product(input_shape)
        output_product = _shape_product(output_shape)
        if input_product is None or output_product is None:
            diagnostics.append(
                f"layer {name!r}: non-integer shape; emit normally"
            )
            return None
        if input_product != output_product:
            diagnostics.append(
                f"layer {name!r}: shape product mismatch "
                f"(in={input_shape}, out={output_shape}); emit normally"
            )
            return None
    input_dims = list(operands.get(inputs[0]) or [])
    output_dims = list(operands.get(outputs[0]) or [])
    mapping = _derive_pos_mapping(
        name, layer, input_dims, input_shape, output_dims, output_shape
    )
    if mapping is None:
        diagnostics.append(
            f"layer {name!r} ({operation}): could not derive a safe "
            f"projection rewrite (in={input_dims}/{input_shape}, "
            f"out={output_dims}/{output_shape}); emit normally"
        )
        return None
    output_to_input, input_to_output = mapping
    return _ShapeOpCandidate(
        name=name,
        predecessor=str(predecessors[0]),
        output_tensor=str(output_names[0]),
        input_dims=input_dims,
        input_shape=input_shape,
        output_to_input=output_to_input,
        input_to_output=input_to_output,
    )


def _primary_output(
    name: str,
    layer: dict[str, Any],
) -> tuple[str, str, list[str], list[int]] | None:
    """Return the primary producer output role, tensor, dims, and shape."""
    operands = layer.get("operands") or {}
    role = next((item for item in operands if _is_output_role(item)), None)
    if role is None and operands:
        role = next(iter(operands))
    names = (layer.get("tensor_names") or {}).get("outputs") or []
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    if role is None or not names or not shapes:
        return None
    return role, str(names[0]), list(operands.get(role) or []), list(shapes[0])


def _compose_shape_alias(
    upstream: _Alias,
    candidate: _ShapeOpCandidate,
) -> _Alias:
    """Compose one candidate mapping with its already-elided predecessor."""
    output_to_input = [
        (
            upstream.out_to_in[index]
            if index is not None and index < len(upstream.out_to_in)
            else None
        )
        for index in candidate.output_to_input
    ]
    input_to_output: list[list[int]] = [
        [] for _ in range(len(upstream.root_dims))
    ]
    for output_index, input_index in enumerate(output_to_input):
        if input_index is not None and 0 <= input_index < len(input_to_output):
            input_to_output[input_index].append(output_index)
    return _Alias(
        root_tensor=upstream.root_tensor,
        root_layer=upstream.root_layer,
        root_role=upstream.root_role,
        root_dims=list(upstream.root_dims),
        root_shape=list(upstream.root_shape),
        in_to_out=input_to_output,
        out_to_in=output_to_input,
    )


def _candidate_alias(
    candidate: _ShapeOpCandidate,
    layers: dict[str, dict[str, Any]],
    aliases: dict[str, _Alias],
) -> _Alias | None:
    """Resolve one candidate to its surviving root producer."""
    producer = layers.get(candidate.predecessor)
    if producer is None:
        return None
    primary = _primary_output(candidate.predecessor, producer)
    if primary is None:
        return None
    role, tensor, dims, shape = primary
    if tensor in aliases:
        return _compose_shape_alias(aliases[tensor], candidate)
    return _Alias(
        root_tensor=tensor,
        root_layer=candidate.predecessor,
        root_role=role,
        root_dims=dims,
        root_shape=shape,
        in_to_out=candidate.input_to_output,
        out_to_in=candidate.output_to_input,
    )


def _build_shape_op_aliases(
    layers: dict[str, dict[str, Any]],
) -> tuple[dict[str, _Alias], set[str], list[str]]:
    """Build aliases for safe shape ops in topological layer order."""
    aliases: dict[str, _Alias] = {}
    elided: set[str] = set()
    diagnostics: list[str] = []
    for name, layer in layers.items():
        candidate = _shape_op_candidate(name, layer, diagnostics)
        if candidate is None:
            continue
        alias = _candidate_alias(candidate, layers, aliases)
        if alias is None:
            continue
        aliases[candidate.output_tensor] = alias
        elided.add(name)
    return aliases, elided, diagnostics


def _rewrite_alias_inputs(
    name: str,
    layer: dict[str, Any],
    aliases: dict[str, _Alias],
    diagnostics: list[str],
    rewrite_sequence: int,
) -> int:
    """Rewrite one surviving layer's inputs to their shape-op roots."""
    predecessors = list((layer.get("connections") or {}).get("inputs") or [])
    operands = layer.get("operands") or {}
    names = (layer.get("tensor_names") or {}).get("inputs") or []
    shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    input_roles = [role for role in operands if not _is_output_role(role)]
    for slot in range(len(predecessors)):
        tensor_name = names[slot] if slot < len(names) else None
        if tensor_name is None or tensor_name not in aliases:
            continue
        alias = aliases[tensor_name]
        predecessors[slot] = alias.root_layer
        if slot < len(names):
            names[slot] = alias.root_tensor
        if slot < len(shapes):
            shapes[slot] = list(alias.root_shape)
        if slot >= len(input_roles):
            continue
        role = input_roles[slot]
        current_dims = list(operands.get(role) or [])
        if len(current_dims) != len(alias.out_to_in):
            diagnostics.append(
                f"layer {name!r}: alias-rewrite skipped for role {role!r} "
                f"— operand width {len(current_dims)} != alias width "
                f"{len(alias.out_to_in)}."
            )
            continue
        new_dims: list[str | None] = [None] * len(alias.root_dims)
        for output_index, input_index in enumerate(alias.out_to_in):
            if input_index is None or not 0 <= input_index < len(new_dims):
                continue
            if new_dims[input_index] is None:
                new_dims[input_index] = current_dims[output_index]
        for index, label in enumerate(new_dims):
            if label is None:
                new_dims[index] = f"squeeze_{name}_{rewrite_sequence}_{index}"
                rewrite_sequence += 1
        operands[role] = [str(value) for value in new_dims]
    layer["operands"] = operands
    _store_rewritten_inputs(layer, predecessors, names, shapes)
    return rewrite_sequence


def _store_rewritten_inputs(
    layer: dict[str, Any],
    predecessors: list[str],
    names: list[str],
    shapes: list[list[int]],
) -> None:
    """Write rewritten positional input metadata back when present."""
    if (layer.get("connections") or {}).get("inputs") is not None:
        layer["connections"]["inputs"] = predecessors
    if (layer.get("tensor_names") or {}).get("inputs") is not None:
        layer["tensor_names"]["inputs"] = names
    if (layer.get("tensor_shapes") or {}).get("inputs") is not None:
        layer["tensor_shapes"]["inputs"] = shapes


def _rewrite_elided_layer(
    name: str,
    layer: dict[str, Any],
    aliases: dict[str, _Alias],
    elided: set[str],
    diagnostics: list[str],
    rewrite_sequence: int,
) -> tuple[dict[str, Any], int]:
    """Return one surviving layer with all elided aliases bypassed."""
    rewritten = copy.deepcopy(layer)
    rewrite_sequence = _rewrite_alias_inputs(
        name,
        rewritten,
        aliases,
        diagnostics,
        rewrite_sequence,
    )
    connections = rewritten.get("connections") or {}
    outputs = [
        output
        for output in list(connections.get("outputs") or [])
        if output not in elided
    ]
    if connections.get("outputs") is not None:
        connections["outputs"] = outputs
    return rewritten, rewrite_sequence


def _apply_shape_op_elision(
    layers: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Elide safe shape ops and reconnect consumers to root producers."""
    aliases, elided, diagnostics = _build_shape_op_aliases(layers)
    if not elided:
        return layers, diagnostics
    rewritten: dict[str, dict[str, Any]] = {}
    rewrite_sequence = 0
    for name, layer in layers.items():
        if name in elided:
            continue
        rewritten[name], rewrite_sequence = _rewrite_elided_layer(
            name,
            layer,
            aliases,
            elided,
            diagnostics,
            rewrite_sequence,
        )
    return rewritten, diagnostics


# ---------------------------------------------------------------------------
# Operand normalization (multi-input correctness gate)
# ---------------------------------------------------------------------------


def _operand_roles(
    operands: dict[str, Any],
    input_slots: int,
    output_slots: int,
) -> tuple[list[str], list[str]]:
    """Classify existing operand roles using the AF positional convention."""
    inputs: list[str] = []
    outputs: list[str] = []
    for role in operands:
        if _is_output_role(role):
            outputs.append(role)
        elif _is_input_role(role) or len(inputs) < input_slots:
            inputs.append(role)
        elif len(outputs) < output_slots:
            outputs.append(role)
        else:
            inputs.append(role)
    return inputs, outputs


def _synthesize_operand(
    operands: dict[str, Any],
    diagnostics: list[str],
    *,
    layer_name: str,
    slot: int,
    role_name: str,
    slot_shape: list[int],
    template_dims: list[str],
    template_shape: list[int],
    suffix: str,
) -> None:
    """Add one missing positional operand role when shape data is complete."""
    while role_name in operands:
        role_name += "_x"
    if (
        template_dims
        and template_shape
        and slot_shape
        and len(template_dims) == len(slot_shape) == len(template_shape)
    ):
        labels = list(template_dims)
        for dimension, (actual, template) in enumerate(
            zip(slot_shape, template_shape, strict=True)
        ):
            if int(actual) != int(template):
                labels[dimension] = f"{labels[dimension]}_{suffix}{slot}"
    elif slot_shape:
        labels = [f"{role_name}_d{index}" for index in range(len(slot_shape))]
    else:
        diagnostics.append(
            f"layer {layer_name!r}: cannot synthesize role {role_name!r} "
            f"— no shape info available for slot {slot}."
        )
        return
    operands[role_name] = labels
    diagnostics.append(
        f"layer {layer_name!r}: synthesized {role_name}={labels} "
        f"for missing slot {slot} (shape={slot_shape})."
    )


def _normalize_input_operands(
    name: str,
    operands: dict[str, Any],
    roles: list[str],
    shapes: list[list[int]],
    types: list[str],
    diagnostics: list[str],
) -> None:
    """Synthesize undeclared non-weight input slots."""
    template_dims = list(operands.get(roles[0]) or []) if roles else []
    template_shape = list(shapes[0]) if shapes else []
    slot_count = len(types) if types else len(shapes)
    for slot in range(len(roles), slot_count):
        if slot < len(types) and types[slot] == "weight":
            continue
        slot_shape = list(shapes[slot]) if slot < len(shapes) else []
        _synthesize_operand(
            operands,
            diagnostics,
            layer_name=name,
            slot=slot,
            role_name=f"Input_{slot}",
            slot_shape=slot_shape,
            template_dims=template_dims,
            template_shape=template_shape,
            suffix="s",
        )


def _normalize_output_operands(
    name: str,
    operands: dict[str, Any],
    roles: list[str],
    shapes: list[list[int]],
    diagnostics: list[str],
) -> None:
    """Synthesize undeclared output slots."""
    template_dims = list(operands.get(roles[0]) or []) if roles else []
    template_shape = list(shapes[0]) if shapes else []
    for slot in range(len(roles), len(shapes)):
        _synthesize_operand(
            operands,
            diagnostics,
            layer_name=name,
            slot=slot,
            role_name=f"Output_{slot}",
            slot_shape=list(shapes[slot]),
            template_dims=template_dims,
            template_shape=template_shape,
            suffix="o",
        )


def _normalize_layer_operands(
    name: str,
    layer: dict[str, Any],
    diagnostics: list[str],
) -> None:
    """Normalize the positional operand roles for one layer in place."""
    operands = layer.get("operands")
    if not operands:
        return
    shapes = layer.get("tensor_shapes") or {}
    input_shapes = shapes.get("inputs") or []
    output_shapes = shapes.get("outputs") or []
    input_types = (layer.get("tensor_types") or {}).get("inputs") or []
    input_roles, output_roles = _operand_roles(
        operands,
        max(len(input_shapes), len(input_types)),
        len(output_shapes),
    )
    _normalize_input_operands(
        name,
        operands,
        input_roles,
        input_shapes,
        input_types,
        diagnostics,
    )
    _normalize_output_operands(
        name,
        operands,
        output_roles,
        output_shapes,
        diagnostics,
    )


def _normalized_operands(layers: dict[str, dict[str, Any]]) -> list[str]:
    """Normalize every layer and return all synthesis diagnostics."""
    diagnostics: list[str] = []
    for name, layer in layers.items():
        _normalize_layer_operands(name, layer, diagnostics)
    return diagnostics


def _normalize_operands(layers: dict[str, dict[str, Any]]) -> list[str]:
    """Normalize operand roles in place and return synthesis diagnostics."""
    return _normalized_operands(layers)
