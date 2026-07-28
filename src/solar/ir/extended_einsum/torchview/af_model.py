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

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

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


def _is_output_role(role: str) -> bool:
    return bool(_OUTPUT_ROLE_PATTERN.match(role))


def _is_input_role(role: str) -> bool:
    return bool(_INPUT_ROLE_PATTERN.match(role))


def _parse_atoms(label: str) -> list[str]:
    """Split a (possibly-composite) dim label like ``P+R`` into atoms ``[P, R]``."""
    if "+" not in label:
        return [label]
    return [tok.strip() for tok in label.split("+") if tok.strip()]


def _sanitize(name: str) -> str:
    """Make a tensor/einsum name safe for ISL identifiers.

    ISL identifiers must match ``[A-Za-z_][A-Za-z0-9_]*``. Solar emits names
    like ``Model.parameter-tensor`` (dots and hyphens). Replace every
    non-identifier character with underscore; prepend ``_`` if the name
    begins with a digit.
    """
    if not isinstance(name, str):
        return name
    s = _NON_ID_CHAR.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _bits_from_dtype(dtype_str: str) -> int | None:
    """Translate a torch dtype string (e.g. ``'torch.float16'``) to bit width."""
    if not isinstance(dtype_str, str):
        return None
    s = dtype_str.replace("torch.", "").lower()
    mapping = {
        "float64": 64,
        "double": 64,
        "complex128": 128,
        "complex64": 64,
        "float32": 32,
        "tf32": 32,
        "bfloat16": 16,
        "float16": 16,
        "half": 16,
        "int64": 64,
        "long": 64,
        "int32": 32,
        "int": 32,
        "int16": 16,
        "short": 16,
        "int8": 8,
        "uint8": 8,
        "byte": 8,
        "bool": 1,
    }
    return mapping.get(s)


# Data model


@dataclass(frozen=True)
class AxisKey:
    """Unique identifier for one dim of one operand in one layer."""

    layer: str
    role: str
    pos: int


@dataclass
class Axis:
    """Describe one graph axis and its concrete extent."""

    key: AxisKey
    label: str  # raw label as emitted by solar (e.g. "B" or "P+R")
    size: int


# Union-find with size-checked unions


class UnionFind:
    """Maintain equivalence classes for connected operand axes."""

    def __init__(self) -> None:
        """Initialize an empty disjoint-set structure."""
        self._parent: dict[AxisKey, AxisKey] = {}
        self._first_seen_order: dict[AxisKey, int] = {}
        self._counter = 0

    def add(self, x: AxisKey) -> None:
        """Add an axis if it is not already present."""
        if x not in self._parent:
            self._parent[x] = x
            self._first_seen_order[x] = self._counter
            self._counter += 1

    def find(self, x: AxisKey) -> AxisKey:
        """Return the canonical representative for an axis."""
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        cur = x
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, x: AxisKey, y: AxisKey) -> AxisKey:
        """Merge two axis classes and return their representative."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        # Stable canonical: keep the one seen earliest (topological priority).
        if self._first_seen_order[rx] <= self._first_seen_order[ry]:
            self._parent[ry] = rx
            return rx
        self._parent[rx] = ry
        return ry


# Builder


@dataclass
class BuildContext:
    """Mutable state shared by AF graph construction phases."""

    layers: dict[str, dict[str, Any]]
    """Topologically-ordered dict from einsum_graph stage-2 output."""

    axes: dict[AxisKey, Axis] = field(default_factory=dict)
    """All (layer, role, pos) → Axis."""

    uf: UnionFind = field(default_factory=UnionFind)
    """Union-find over AxisKey."""

    canonical_name: dict[AxisKey, str] = field(default_factory=dict)
    """Axis → canonical name (R0, R1, ...)."""

    rank_sizes: dict[str, int] = field(default_factory=dict)
    """canonical_name → size."""

    role_to_shape_index: dict[tuple[str, str], tuple[str, int]] = field(
        default_factory=dict
    )
    """(layer, role) → ('inputs' or 'outputs', index)."""

    diagnostics: list[str] = field(default_factory=list)


def _build_role_to_shape_index(
    layers: dict[str, dict[str, Any]],
    ctx: BuildContext,
) -> None:
    """Map (layer, role) → (which-tensor-shapes-list, index-into-list).

    Walks operand keys in YAML insertion order. Input-like roles consume
    successive ``tensor_shapes.inputs`` entries; output-like roles consume
    ``tensor_shapes.outputs``. Unknown role names (e.g. "start" pseudo-node)
    pick whichever bucket still has slots. When the indexed slot is
    out-of-range we still assign an index so the emitter can classify; the
    size is inferred later in ``_collect_axes`` from same-layer atom labels.
    """
    for layer_name, layer in layers.items():
        operands = layer.get("operands") or {}
        shapes_in = layer.get("tensor_shapes", {}).get("inputs") or []
        shapes_out = layer.get("tensor_shapes", {}).get("outputs") or []
        next_in = 0
        next_out = 0
        for role in operands:
            kind: str | None = None
            if _is_output_role(role):
                kind = "outputs"
            elif _is_input_role(role):
                kind = "inputs"
            else:
                if next_in < len(shapes_in):
                    kind = "inputs"
                elif next_out < len(shapes_out):
                    kind = "outputs"
                else:
                    kind = "inputs"
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}: role {role!r} unclassified "
                        f"with no remaining shape slots; defaulting to input."
                    )
            if kind == "outputs":
                ctx.role_to_shape_index[(layer_name, role)] = (
                    "outputs",
                    next_out,
                )
                next_out += 1
            else:
                ctx.role_to_shape_index[(layer_name, role)] = (
                    "inputs",
                    next_in,
                )
                next_in += 1


def _collect_axes(ctx: BuildContext) -> None:
    """Populate ctx.axes for every (layer, role, pos).

    When solar's ``tensor_shapes`` is missing an entry for a role (e.g.
    matmul Weight: 2 input operands but only 1 input shape recorded),
    infer the size from same-layer atom labels — any atom appearing
    atomically in another operand of the same layer carries its size.
    Composite labels (``P+R``) are skipped (need atom-level info).
    """
    for layer_name, layer in ctx.layers.items():
        operands = layer.get("operands") or {}
        shapes_in = layer.get("tensor_shapes", {}).get("inputs") or []
        shapes_out = layer.get("tensor_shapes", {}).get("outputs") or []

        label_size: dict[str, int] = {}
        for role, dims in operands.items():
            ki = ctx.role_to_shape_index.get((layer_name, role))
            if ki is None:
                continue
            kind, idx = ki
            shapes_list = shapes_in if kind == "inputs" else shapes_out
            if idx >= len(shapes_list):
                continue
            shape = shapes_list[idx]
            for pos, label in enumerate(dims):
                if pos >= len(shape):
                    break
                if "+" not in label and label not in label_size:
                    label_size[label] = int(shape[pos])

        for role, dims in operands.items():
            key_to_kind = ctx.role_to_shape_index.get((layer_name, role))
            shape: list[int] | None = None
            if key_to_kind is not None:
                kind, idx = key_to_kind
                shapes_list = shapes_in if kind == "inputs" else shapes_out
                if idx < len(shapes_list):
                    shape = shapes_list[idx]
            if shape is None:
                inferred: list[int | None] = []
                for label in dims:
                    if "+" in label:
                        inferred.append(None)
                    else:
                        inferred.append(label_size.get(label))
                if all(s is not None for s in inferred):
                    shape = [
                        int(value) for value in inferred if value is not None
                    ]
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}, role {role!r}: inferred shape "
                        f"{shape} from same-layer atom labels."
                    )
                else:
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}, role {role!r}: no tensor_shape "
                        f"and couldn't infer; skipping."
                    )
                    continue
            if shape is None:
                raise ValueError(
                    f"missing shape metadata for {layer_name}.{role}"
                )
            n = min(len(dims), len(shape))
            for pos in range(n):
                key = AxisKey(layer_name, role, pos)
                ctx.axes[key] = Axis(
                    key=key, label=dims[pos], size=int(shape[pos])
                )
                ctx.uf.add(key)


def _within_layer_union(ctx: BuildContext) -> None:
    """Phase 1: union axes within a layer that share (atomic label, size).

    Composite labels (containing ``+``) stay as their own identity. Grouping
    by ``(label, size)`` ensures that when a layer like ``div`` has three
    "B"-labeled positions with sizes ``[64, 1, 64]``, the two size-64
    entries unify even though they're not adjacent in YAML order.
    """
    by_layer: dict[str, list[AxisKey]] = defaultdict(list)
    for k in ctx.axes:
        by_layer[k.layer].append(k)
    for keys in by_layer.values():
        groups: dict[tuple[str, int], list[AxisKey]] = defaultdict(list)
        for k in keys:
            ax = ctx.axes[k]
            groups[(ax.label, ax.size)].append(k)
        for group in groups.values():
            for a in group[1:]:
                ctx.uf.union(group[0], a)


def _input_like_roles_in_order(
    operands: dict[str, Any],
    layer_name: str,
    role_to_shape_index: dict[tuple[str, str], tuple[str, int]],
    tensor_types_inputs: list[str] | None = None,
    skip_weight_typed: bool = False,
) -> list[str]:
    """Ordered list of input-like roles (those mapping to tensor_shapes.inputs).

    Catches solar's custom input roles (``Target`` for loss functions,
    ``Hidden_in``/``Cell_in`` for RNN/LSTM/GRU, etc.) — anything mapped to
    an ``inputs`` slot is treated as an input regardless of name.

    When ``skip_weight_typed`` is True and ``tensor_types_inputs`` is
    available, drop roles tagged as ``"weight"`` — these don't consume
    a predecessor (they're emitted as fresh ``W{n}`` tensors). Without
    skipping, the cross-layer union pairs a weight role with the next
    predecessor and mis-unions the pred's output axes with the scalar
    weight's axes — the L2/84 (`Gemm + BatchNorm + scale*x + Softmax`)
    crash signature.
    """
    out = []
    input_pos = 0
    for role in operands:
        idx = role_to_shape_index.get((layer_name, role))
        if idx is None or idx[0] != "inputs":
            continue
        if skip_weight_typed and tensor_types_inputs is not None:
            role_type = (
                tensor_types_inputs[input_pos]
                if input_pos < len(tensor_types_inputs)
                else None
            )
            input_pos += 1
            if role_type == "weight":
                continue
        else:
            input_pos += 1
        out.append(role)
    return out


def _primary_output_role(
    pred_operands: dict[str, Any],
    role_to_shape_index: dict[tuple[str, str], tuple[str, int]],
    pred_name: str,
) -> str | None:
    """Return the predecessor's primary output role name."""
    for cand in pred_operands:
        if _is_output_role(cand):
            return cand
    # Fallback for pseudo-nodes (e.g. "start").
    for cand in pred_operands:
        idx = role_to_shape_index.get((pred_name, cand))
        if idx is not None and idx[0] == "outputs":
            return cand
    return None


def _cross_layer_union(ctx: BuildContext) -> None:
    """Phase 2: union axes across producer→consumer connections (pos-wise)."""
    for layer_name, layer in ctx.layers.items():
        preds = (layer.get("connections") or {}).get("inputs") or []
        operands = layer.get("operands") or {}
        # connections.inputs lists only the non-weight predecessors. Match
        # that by skipping weight-typed roles when pairing roles ↔ preds —
        # otherwise the cross-layer union for `mul(scale, x)` mis-pairs
        # ``scale`` with batch_norm and never unions batch_norm's actual
        # axes with the mul input's, producing distinct canonical names
        # across einsums that AF's pydantic schema rejects.
        tensor_types_inputs = (layer.get("tensor_types") or {}).get(
            "inputs"
        ) or []
        input_roles = _input_like_roles_in_order(
            operands,
            layer_name,
            ctx.role_to_shape_index,
            tensor_types_inputs=tensor_types_inputs,
            skip_weight_typed=True,
        )
        for i, role in enumerate(input_roles):
            if i >= len(preds):
                break
            pred = preds[i]
            pred_layer = ctx.layers.get(pred)
            if pred_layer is None:
                continue
            pred_operands = pred_layer.get("operands") or {}
            pred_output_role = _primary_output_role(
                pred_operands, ctx.role_to_shape_index, pred
            )
            if pred_output_role is None:
                continue
            pred_dims = pred_operands.get(pred_output_role, [])
            cur_dims = operands.get(role, [])
            n = min(len(pred_dims), len(cur_dims))
            for pos in range(n):
                a = AxisKey(pred, pred_output_role, pos)
                b = AxisKey(layer_name, role, pos)
                if a not in ctx.axes or b not in ctx.axes:
                    continue
                if ctx.axes[a].size == ctx.axes[b].size:
                    ctx.uf.union(a, b)


def _assign_canonical_names(ctx: BuildContext) -> None:
    """Phase 3: assign R0, R1, ... to equivalence classes in topological order."""
    counter = 0
    seen_roots: dict[AxisKey, str] = {}
    for layer_name, layer in ctx.layers.items():
        operands = layer.get("operands") or {}
        for role, dims in operands.items():
            for pos in range(len(dims)):
                key = AxisKey(layer_name, role, pos)
                if key not in ctx.axes:
                    continue
                root = ctx.uf.find(key)
                if root not in seen_roots:
                    name = f"R{counter}"
                    counter += 1
                    seen_roots[root] = name
                    ctx.rank_sizes[name] = ctx.axes[key].size
                else:
                    existing = ctx.rank_sizes[seen_roots[root]]
                    if existing != ctx.axes[key].size:
                        ctx.diagnostics.append(
                            f"size mismatch in component {seen_roots[root]}: "
                            f"{existing} vs {ctx.axes[key].size} at {key}"
                        )
                ctx.canonical_name[key] = seen_roots[root]


# AF YAML emission


def _build_iter_expr_for_layer(
    ctx: BuildContext, layer_name: str
) -> dict[str, str]:
    """Per-layer ``atom_letter → lowercase canonical_iter_var`` map.

    Used by ``_projection_for_axis`` to rewrite composite labels (``P+R``)
    into iterator expressions over canonical rank names (e.g. ``r5+r7``).
    """
    operands = ctx.layers[layer_name].get("operands") or {}
    atomic_to_iter: dict[str, str] = {}
    for role, dims in operands.items():
        for pos, label in enumerate(dims):
            atoms = _parse_atoms(label)
            if len(atoms) == 1:
                atom = atoms[0]
                key = AxisKey(layer_name, role, pos)
                canonical = ctx.canonical_name.get(key)
                if canonical is not None and atom not in atomic_to_iter:
                    atomic_to_iter[atom] = canonical.lower()
    # Synthesize names for atoms referenced only inside composites.
    for dims in operands.values():
        for _pos, label in enumerate(dims):
            atoms = _parse_atoms(label)
            for atom in atoms:
                if atom not in atomic_to_iter:
                    atomic_to_iter[atom] = (
                        f"x_{layer_name.replace('.', '_')}_{atom}".lower()
                    )
                    ctx.diagnostics.append(
                        f"layer {layer_name!r}: atom {atom!r} in composite "
                        f"{label!r} has no atomic anchor; using fallback iter."
                    )
    return atomic_to_iter


def _projection_for_axis(
    ctx: BuildContext, key: AxisKey, atomic_iter_map: dict[str, str]
) -> str:
    """Iterator expression for one operand position."""
    label = ctx.axes[key].label
    atoms = _parse_atoms(label)
    if len(atoms) == 1:
        return ctx.canonical_name[key].lower()
    return "+".join(atomic_iter_map[a] for a in atoms)


def _bits_for_role(
    layer: dict[str, Any],
    role: str,
    ctx_idx: tuple[str, int] | None = None,
) -> int | None:
    """Return bits-per-value for one operand role with sensible fallbacks."""
    dtypes = layer.get("tensor_dtypes") or {}
    if ctx_idx is None:
        for kind in ("outputs", "inputs"):
            dlist = dtypes.get(kind) or []
            if dlist:
                return _bits_from_dtype(dlist[0])
        return None
    kind, idx = ctx_idx
    dtype_list = dtypes.get(kind) or []
    if idx < len(dtype_list):
        return _bits_from_dtype(dtype_list[idx])
    if dtype_list:
        return _bits_from_dtype(dtype_list[0])
    other = "outputs" if kind == "inputs" else "inputs"
    other_list = dtypes.get(other) or []
    if other_list:
        return _bits_from_dtype(other_list[0])
    return None
