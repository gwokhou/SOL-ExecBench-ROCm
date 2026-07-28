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
    """Return (out_to_in, in_to_out) for an elide-able shape op.

    Returns None when the rewrite is not safe (e.g. genuine reshape, or
    shape ambiguity we can't resolve from operands alone).
    """
    n_in = len(in_dims)
    n_out = len(out_dims)
    op_type = layer.get("type")

    # contiguous / identity-view (shapes match positionally).
    if n_in == n_out and in_shape == out_shape:
        return list(range(n_in)), [[i] for i in range(n_in)]

    # Pure label permutation (same multiset of labels, same multiset of sizes).
    # transpose / permute usually fall here. solar's transpose has identical
    # labels with reordered sizes ("AB->AB" but shape transposed) — so we
    # PREFER shape-based matching over label-based.
    if n_in == n_out and sorted(in_shape) == sorted(out_shape):
        # Try label-based permutation first when the labels are a true
        # multiset permutation distinct from identity.
        if sorted(in_dims) == sorted(out_dims) and in_dims != out_dims:
            used: set[int] = set()
            o2i: list[int | None] | None = []
            for j, lbl in enumerate(out_dims):
                hit = None
                for i, ilbl in enumerate(in_dims):
                    if i in used:
                        continue
                    if ilbl == lbl and in_shape[i] == out_shape[j]:
                        hit = i
                        break
                if hit is None:
                    o2i = None
                    break
                used.add(hit)
                o2i.append(hit)
            if o2i is not None:
                i2o: list[list[int]] = [[] for _ in range(n_in)]
                for j, i in enumerate(o2i):
                    if i is not None:
                        i2o[i].append(j)
                return o2i, i2o
        # Shape-based positional permutation. Greedy unique match by size;
        # bail out if sizes aren't unique enough to derive an unambiguous
        # permutation (caller will emit the op normally).
        used2: set[int] = set()
        o2i2: list[int | None] = []
        ok = True
        for j in range(n_out):
            hit = None
            for i in range(n_in):
                if i in used2:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit = i
                    break
            if hit is None:
                ok = False
                break
            used2.add(hit)
            o2i2.append(hit)
        if ok:
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i2):
                if i is not None:
                    i2o[i].append(j)
            return o2i2, i2o
        return None

    # squeeze: input has size-1 dims that are dropped in output.
    if op_type == "squeeze" and n_out <= n_in:
        ok = True
        used3: set[int] = set()
        o2i3: list[int | None] = []
        for j in range(n_out):
            hit = None
            for i in range(n_in):
                if i in used3:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit = i
                    break
            if hit is None:
                ok = False
                break
            used3.add(hit)
            o2i3.append(hit)
        if ok and all(in_shape[i] == 1 for i in range(n_in) if i not in used3):
            # Every unmatched input position must be size 1 (the dropped axes).
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i3):
                if i is not None:
                    i2o[i].append(j)
            return o2i3, i2o
        return None

    # unsqueeze: output has size-1 dims that aren't in input.
    if op_type == "unsqueeze" and n_in <= n_out:
        ok = True
        used4: set[int] = set()
        o2i4: list[int | None] = []
        for j in range(n_out):
            if out_shape[j] == 1:
                # Either it's a true unsqueeze-introduced axis, or a
                # preserved size-1 from input — prefer to match an unused
                # size-1 input first so order is stable.
                hit = None
                for i in range(n_in):
                    if i in used4:
                        continue
                    if in_shape[i] == 1:
                        hit = i
                        break
                if hit is not None:
                    used4.add(hit)
                    o2i4.append(hit)
                else:
                    o2i4.append(None)
                continue
            hit2 = None
            for i in range(n_in):
                if i in used4:
                    continue
                if in_shape[i] == out_shape[j]:
                    hit2 = i
                    break
            if hit2 is None:
                ok = False
                break
            used4.add(hit2)
            o2i4.append(hit2)
        if ok and len(used4) == n_in:
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i4):
                if i is not None:
                    i2o[i].append(j)
            return o2i4, i2o
        return None

    # expand: a size-1 input dim is broadcast to a larger output dim.
    if op_type == "expand" and n_in == n_out:
        o2i5: list[int | None] = list(range(n_in))
        i2o: list[list[int]] = [[j] for j in range(n_in)]
        return o2i5, i2o

    # torchview emits Tensor.T as ``__get__``.  Tensor.T reverses dimensions;
    # do this before the exact-shape branch because a square transpose has the
    # same shape but is not an identity mapping.
    if op_type == "__get__" and n_in == n_out and in_shape[::-1] == out_shape:
        o2i_get: list[int | None] = list(reversed(range(n_in)))
        i2o_get: list[list[int]] = [[] for _ in range(n_in)]
        for output_axis, input_axis in enumerate(o2i_get):
            if input_axis is None:
                raise ValueError("getitem axis mapping is incomplete")
            i2o_get[input_axis].append(output_axis)
        return o2i_get, i2o_get

    # __getitem__: only safe when the access selects every element along every
    # axis (no-op). Detect by exact shape equality.
    if op_type == "__getitem__":
        if n_in == n_out and in_shape == out_shape:
            return list(range(n_in)), [[i] for i in range(n_in)]
        # Same total size but different shape. Try the permutation derivation
        # above for conservative legacy traces.
        try:
            prod_in = 1
            for s in in_shape:
                prod_in *= s
            prod_out = 1
            for s in out_shape:
                prod_out *= s
        except Exception:  # noqa: BLE001 - malformed optional metadata
            return None
        if (
            prod_in == prod_out
            and n_in == n_out
            and sorted(in_shape) == sorted(out_shape)
        ):
            used5: set[int] = set()
            o2i6: list[int | None] = []
            ok = True
            for j in range(n_out):
                hit = None
                for i in range(n_in):
                    if i in used5:
                        continue
                    if in_shape[i] == out_shape[j]:
                        hit = i
                        break
                if hit is None:
                    ok = False
                    break
                used5.add(hit)
                o2i6.append(hit)
            if ok:
                i2o = [[] for _ in range(n_in)]
                for j, i in enumerate(o2i6):
                    if i is not None:
                        i2o[i].append(j)
                return o2i6, i2o
        return None

    # view / reshape: only elide when it's a pure no-op (same shape) OR a
    # squeeze-or-unsqueeze of size-1 dims. Axis collapse / split is left
    # to AF as a real op.
    if op_type in ("view", "reshape"):
        # Identity reshape (shape unchanged).
        if n_in == n_out and in_shape == out_shape:
            return list(range(n_in)), [[i] for i in range(n_in)]
        # Reshape that only adds/drops size-1 dims (product preserved).
        # Build the mapping by walking the non-unit dim sequences in both
        # sides — they must match in order.
        in_nonunit = [(i, s) for i, s in enumerate(in_shape) if s != 1]
        out_nonunit = [(j, s) for j, s in enumerate(out_shape) if s != 1]
        if len(in_nonunit) == len(out_nonunit) and all(
            a[1] == b[1] for a, b in zip(in_nonunit, out_nonunit, strict=False)
        ):
            o2i7: list[int | None] = [None] * n_out
            for (i, _), (j, _) in zip(in_nonunit, out_nonunit, strict=False):
                o2i7[j] = i
            # Pair leftover size-1 input dims to leftover size-1 output
            # dims in order; remaining are introduced (None on out side)
            # or dropped (no entry on out side).
            unmatched_in = [
                i
                for i in range(n_in)
                if i not in {x for x in o2i7 if x is not None}
            ]
            unmatched_out = [j for j in range(n_out) if o2i7[j] is None]
            for k in range(min(len(unmatched_in), len(unmatched_out))):
                o2i7[unmatched_out[k]] = unmatched_in[k]
            # Validate every input dim is either matched (one or more
            # output slot points to it) or is a dropped size-1.
            i2o = [[] for _ in range(n_in)]
            for j, i in enumerate(o2i7):
                if i is not None:
                    i2o[i].append(j)
            for i in range(n_in):
                if not i2o[i] and in_shape[i] != 1:
                    return None
            return o2i7, i2o
        return None

    return None


def _build_shape_op_aliases(
    layers: dict[str, dict[str, Any]],
) -> tuple[dict[str, _Alias], set[str], list[str]]:
    """Return (alias_table, elided_set, diagnostics).

    Walks layers in topological order and records, for each elide-able
    pure-shape-op layer, an Alias entry keyed by the layer's primary
    output tensor name. Chained shape ops compose via the table — when
    an op's predecessor is itself elided, we look up the predecessor's
    alias and propagate the root forward.
    """
    aliases: dict[str, _Alias] = {}
    elided: set[str] = set()
    diagnostics: list[str] = []

    for name, layer in layers.items():
        if layer.get("is_real_einsum", True):
            continue
        op_type = layer.get("type")
        if op_type not in _SHAPE_OP_TYPES:
            continue

        operands = layer.get("operands") or {}
        in_roles = [
            r
            for r in operands
            if _is_input_role(r) or (not _is_output_role(r) and r != "start")
        ]
        out_roles = [r for r in operands if _is_output_role(r)]
        # Skip multi-input or no-output shape ops.
        if len(in_roles) != 1 or len(out_roles) != 1:
            continue
        preds = (layer.get("connections") or {}).get("inputs") or []
        if len(preds) != 1:
            continue

        in_role = in_roles[0]
        out_role = out_roles[0]
        in_dims = list(operands.get(in_role) or [])
        out_dims = list(operands.get(out_role) or [])
        shapes = layer.get("tensor_shapes", {}) or {}
        in_shape_list = shapes.get("inputs") or []
        out_shape_list = shapes.get("outputs") or []
        if not in_shape_list or not out_shape_list:
            continue
        in_shape = list(in_shape_list[0])
        out_shape = list(out_shape_list[0])

        # Defensive: total size must be preserved (expand explicitly excepted).
        if op_type != "expand":
            try:
                pi = 1
                for s in in_shape:
                    pi *= int(s)
                po = 1
                for s in out_shape:
                    po *= int(s)
            except Exception:  # noqa: BLE001 - malformed optional metadata
                diagnostics.append(
                    f"layer {name!r}: non-integer shape; emit normally"
                )
                continue
            if pi != po:
                diagnostics.append(
                    f"layer {name!r}: shape product mismatch "
                    f"(in={in_shape}, out={out_shape}); emit normally"
                )
                continue

        # Detect partial __getitem__ (slice with stride/length != full
        # range). For now, only elide when shapes equate or are a pure
        # permutation (handled inside _derive_pos_mapping).
        # Compute the rewrite.
        derived = _derive_pos_mapping(
            name, layer, in_dims, in_shape, out_dims, out_shape
        )
        if derived is None:
            diagnostics.append(
                f"layer {name!r} ({op_type}): could not derive a safe "
                f"projection rewrite (in={in_dims}/{in_shape}, "
                f"out={out_dims}/{out_shape}); emit normally"
            )
            continue
        out_to_in, in_to_out = derived

        # Resolve predecessor's primary output tensor + role.
        pred_name = preds[0]
        pred_layer = layers.get(pred_name)
        if pred_layer is None:
            continue
        pred_operands = pred_layer.get("operands") or {}
        pred_out_role: str | None = None
        for cand in pred_operands:
            if _is_output_role(cand):
                pred_out_role = cand
                break
        if pred_out_role is None and pred_operands:
            # pseudo-node (e.g. "start") — fall back to its first operand.
            pred_out_role = next(iter(pred_operands))
        if pred_out_role is None:
            continue

        pred_output_tensor_name = (
            (pred_layer.get("tensor_names") or {}).get("outputs") or [None]
        )[0]
        if pred_output_tensor_name is None:
            continue
        pred_output_tensor_name = str(pred_output_tensor_name)
        pred_out_shape_list = (pred_layer.get("tensor_shapes") or {}).get(
            "outputs"
        ) or []
        if not pred_out_shape_list:
            continue
        pred_out_shape = list(pred_out_shape_list[0])
        pred_out_dims = list(pred_operands.get(pred_out_role) or [])

        my_output_tensor_name = (
            (layer.get("tensor_names") or {}).get("outputs") or [None]
        )[0]
        if my_output_tensor_name is None:
            continue

        # If the predecessor is itself elided, follow the chain.
        if pred_output_tensor_name in aliases:
            up = aliases[pred_output_tensor_name]
            # Compose: out_to_in points at our input positions; those map
            # via the predecessor's alias.out_to_in to the chain root.
            composed: list[int | None] = []
            for j in range(len(out_to_in)):
                k = out_to_in[j]
                if k is None:
                    composed.append(None)
                else:
                    if k < len(up.out_to_in):
                        composed.append(up.out_to_in[k])
                    else:
                        composed.append(None)
            composed_i2o: list[list[int]] = [
                [] for _ in range(len(up.root_dims))
            ]
            for j, i in enumerate(composed):
                if i is not None and 0 <= i < len(composed_i2o):
                    composed_i2o[i].append(j)
            aliases[my_output_tensor_name] = _Alias(
                root_tensor=up.root_tensor,
                root_layer=up.root_layer,
                root_role=up.root_role,
                root_dims=list(up.root_dims),
                root_shape=list(up.root_shape),
                in_to_out=composed_i2o,
                out_to_in=composed,
            )
        else:
            aliases[my_output_tensor_name] = _Alias(
                root_tensor=pred_output_tensor_name,
                root_layer=pred_name,
                root_role=pred_out_role,
                root_dims=pred_out_dims,
                root_shape=pred_out_shape,
                in_to_out=in_to_out,
                out_to_in=out_to_in,
            )
        elided.add(name)

    return aliases, elided, diagnostics


def _apply_shape_op_elision(
    layers: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return a rewritten layers dict with shape ops elided.

    For each elide-able layer S:
      - drop S from the output dict
      - for every consumer C whose ``connections.inputs`` lists S, replace
        the entry with the chain root
      - for the consumer operand reading from S, rewrite its dim list to
        align positionally with the root producer's output dims (insert a
        synthetic label for unsqueeze-introduced output dims that the
        consumer carried; drop labels for squeeze-dropped axes; reorder
        for transposes/permutations)
    """
    aliases, elided, diags = _build_shape_op_aliases(layers)
    if not elided:
        return layers, diags

    new_layers: dict[str, dict[str, Any]] = {}
    rewrite_seq = 0

    for name, layer in layers.items():
        if name in elided:
            continue
        new_layer = copy.deepcopy(layer)
        preds = list((new_layer.get("connections") or {}).get("inputs") or [])
        operands = new_layer.get("operands") or {}
        tensor_names = (new_layer.get("tensor_names") or {}).get("inputs") or []
        tensor_shapes = (new_layer.get("tensor_shapes") or {}).get(
            "inputs"
        ) or []
        tensor_dtypes_in = (new_layer.get("tensor_dtypes") or {}).get(
            "inputs"
        ) or []

        # Every role that isn't an explicit output role is an input-like slot
        # (covers Input, Input_1, Weight, Target, Hidden_in, etc.).
        input_roles = [r for r in operands if not _is_output_role(r)]
        # Walk input slot k (1:1 with preds[k] / tensor_names[k] / shapes[k]).
        for k in range(len(preds)):
            preds[k]
            # Walk the alias chain: if pred itself produces an elided
            # tensor we substitute its root.
            in_tensor_name = tensor_names[k] if k < len(tensor_names) else None
            if in_tensor_name is None or in_tensor_name not in aliases:
                continue
            alias = aliases[in_tensor_name]
            preds[k] = alias.root_layer
            if k < len(tensor_names):
                tensor_names[k] = alias.root_tensor
            if k < len(tensor_shapes):
                tensor_shapes[k] = list(alias.root_shape)
            # Rewrite operand labels for this input slot.
            if k < len(input_roles):
                role = input_roles[k]
                cur_dims = list(operands.get(role) or [])
                if len(cur_dims) != len(alias.out_to_in):
                    # Defensive: skip this rewrite (shouldn't happen given
                    # solar's positional convention).
                    diags.append(
                        f"layer {name!r}: alias-rewrite skipped for role "
                        f"{role!r} — operand width {len(cur_dims)} != alias "
                        f"width {len(alias.out_to_in)}."
                    )
                    continue
                new_dims: list[str | None] = [None] * len(alias.root_dims)
                for j, i in enumerate(alias.out_to_in):
                    if i is None or i < 0 or i >= len(new_dims):
                        continue
                    # When multiple output positions map to the same input
                    # (broadcast) — pick the first; the iter at the root
                    # axis is the same dim across the broadcast.
                    if new_dims[i] is None:
                        new_dims[i] = cur_dims[j]
                # Slot in synthetic labels for root positions that the
                # consumer doesn't iterate (because the shape-op squeezed
                # them out / they're size-1).
                for i, label in enumerate(new_dims):
                    if label is None:
                        new_dims[i] = f"squeeze_{name}_{rewrite_seq}_{i}"
                        rewrite_seq += 1
                operands[role] = [str(x) for x in new_dims]

        new_layer["operands"] = operands
        # Write rewritten connections / tensor_names / shapes / dtypes back.
        if (new_layer.get("connections") or {}).get("inputs") is not None:
            new_layer["connections"]["inputs"] = preds
        if (new_layer.get("tensor_names") or {}).get("inputs") is not None:
            new_layer["tensor_names"]["inputs"] = tensor_names
        if (new_layer.get("tensor_shapes") or {}).get("inputs") is not None:
            new_layer["tensor_shapes"]["inputs"] = tensor_shapes

        # Rewrite outgoing-connection lists: each elided layer's name in a
        # successor's connections.outputs should be replaced by the
        # surviving successor or the root layer's successors. We only
        # touch `connections.outputs` for symmetry; AF doesn't read it.
        outs = list((new_layer.get("connections") or {}).get("outputs") or [])
        outs = [o for o in outs if o not in elided]
        if (new_layer.get("connections") or {}).get("outputs") is not None:
            new_layer["connections"]["outputs"] = outs

        _ = tensor_dtypes_in  # currently no dtype rewrite needed; root retains
        new_layers[name] = new_layer

    return new_layers, diags


# ---------------------------------------------------------------------------
# Operand normalization (multi-input correctness gate)
# ---------------------------------------------------------------------------


def _normalize_operands(layers: dict[str, dict[str, Any]]) -> list[str]:
    """Make ``operands`` reflect the true tensor slot count per layer.

    Solar's shape-handlers sometimes emit a single ``Input`` operand role
    for ops that actually consume N tensors (cat, concat, stack). Every
    downstream pass — context indexing, axis collection, cross-layer
    union, AF emit — iterates ``operands`` and would silently drop the
    extra preds. We fix that here, once, by making operands the canonical
    projection-shape map: one role per real input/output tensor slot.

    Source of truth (per AF builder contract):
      - ``tensor_shapes.inputs[k]``, ``tensor_types.inputs[k]`` — slot k's
        shape and role (``"input"`` vs ``"weight"``)
      - ``tensor_shapes.outputs[k]`` — output slot k's shape

    For each layer:
      - If existing input-role count < slot count, synthesize ``Input_k``
        (or ``Weight_k`` per tensor_types) for the missing slots.
      - Same for output roles.

    Synthesized labels are derived from the first existing same-kind role
    as a template, with a per-slot suffix on dims whose size differs from
    the template's size at the same position. Equal-sized dims keep the
    template's label so the union-find correctly merges them (e.g.
    add(x,x) has all dims unified; cat's cat-axis is split). Without a
    template, fresh labels are minted per-dim.

    Mutates ``layers`` in place. Returns diagnostic strings for the
    synthesized roles.
    """
    diags: list[str] = []

    for name, layer in layers.items():
        operands = layer.get("operands")
        if not operands:
            continue
        in_shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
        in_types = (layer.get("tensor_types") or {}).get("inputs") or []
        out_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []

        # Classify each existing role by the same rule
        # ``_build_role_to_shape_index`` uses, so unconventionally-named
        # roles like the entry-point ``start`` (which functions as an
        # output) are recognized and don't trigger spurious synthesis.
        in_roles_existing: list[str] = []
        out_roles_existing: list[str] = []
        n_in_max = max(len(in_shapes), len(in_types))
        for role in operands:
            if _is_output_role(role):
                out_roles_existing.append(role)
            elif _is_input_role(role):
                in_roles_existing.append(role)
            else:
                # Default: fill remaining input slots first, then outputs.
                if len(in_roles_existing) < n_in_max:
                    in_roles_existing.append(role)
                elif len(out_roles_existing) < len(out_shapes):
                    out_roles_existing.append(role)
                else:
                    in_roles_existing.append(role)

        def _synthesize(
            slot: int,
            role_name: str,
            slot_shape: list[int],
            tmpl_dims: list[str],
            tmpl_shape: list[int],
            suffix: str,
            layer_operands: dict[str, Any],
            layer_name: str,
        ) -> None:
            while role_name in layer_operands:
                role_name += "_x"
            if (
                tmpl_dims
                and tmpl_shape
                and slot_shape
                and len(tmpl_dims) == len(slot_shape)
                and len(tmpl_shape) == len(slot_shape)
            ):
                labels = list(tmpl_dims)
                for d in range(len(labels)):
                    if int(slot_shape[d]) != int(tmpl_shape[d]):
                        labels[d] = f"{labels[d]}_{suffix}{slot}"
            elif slot_shape:
                labels = [f"{role_name}_d{d}" for d in range(len(slot_shape))]
            else:
                diags.append(
                    f"layer {layer_name!r}: cannot synthesize role {role_name!r} "
                    f"— no shape info available for slot {slot}."
                )
                return
            layer_operands[role_name] = labels
            diags.append(
                f"layer {layer_name!r}: synthesized {role_name}={labels} "
                f"for missing slot {slot} (shape={slot_shape})."
            )

        # Input slots — only synthesize NON-WEIGHT slots. Weight slots not
        # declared in operands are still correctly emitted as ``W{n}`` by
        # ``_emit_af_workload`` (it inspects tensor_types positionally);
        # adding phantom Weight_k roles for tensors that don't participate
        # in the current einsum's compute inflates the AF mapper's
        # pmapping space and causes OOMs (L2/13: ConvTranspose3d with
        # bias). Real multi-input ops (cat/concat/stack) only need
        # additional non-weight roles.
        tmpl_in_dims = (
            list(operands.get(in_roles_existing[0]) or [])
            if in_roles_existing
            else []
        )
        tmpl_in_shape = list(in_shapes[0]) if in_shapes else []
        for slot in range(len(in_roles_existing), len(in_types) or n_in_max):
            if slot < len(in_types) and in_types[slot] == "weight":
                continue
            slot_shape = list(in_shapes[slot]) if slot < len(in_shapes) else []
            _synthesize(
                slot,
                f"Input_{slot}",
                slot_shape,
                tmpl_in_dims,
                tmpl_in_shape,
                suffix="s",
                layer_operands=operands,
                layer_name=name,
            )

        # Output slots.
        tmpl_out_dims = (
            list(operands.get(out_roles_existing[0]) or [])
            if out_roles_existing
            else []
        )
        tmpl_out_shape = list(out_shapes[0]) if out_shapes else []
        for slot in range(len(out_roles_existing), len(out_shapes)):
            slot_shape = list(out_shapes[slot])
            _synthesize(
                slot,
                f"Output_{slot}",
                slot_shape,
                tmpl_out_dims,
                tmpl_out_shape,
                suffix="o",
                layer_operands=operands,
                layer_name=name,
            )

    return diags
