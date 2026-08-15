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
    AxisKey,
    BuildContext,
    _bits_for_role,
    _build_iter_expr_for_layer,
    _projection_for_axis,
    _sanitize,
)


@dataclass(slots=True, kw_only=True)
class _EmissionCursor:
    """Mutable positional cursor for one layer's input tensor accesses."""

    predecessors: list[str]
    tensor_types: list[str]
    input_index: int = 0
    predecessor_index: int = 0
    weight_index: int = 0

    def _next_weight(self) -> str:
        self.weight_index += 1
        return f"W{self.weight_index}"

    def _input_tensor(
        self,
        rename_target: dict[str, str],
    ) -> tuple[str, str]:
        role_type = (
            self.tensor_types[self.input_index]
            if self.input_index < len(self.tensor_types)
            else None
        )
        self.input_index += 1
        if role_type == "weight":
            return self._next_weight(), "weight"
        if self.predecessor_index < len(self.predecessors):
            predecessor = self.predecessors[self.predecessor_index]
            self.predecessor_index += 1
            key = "input" if "input" not in rename_target else "weight"
            return _sanitize(predecessor), key
        return self._next_weight(), "weight"


def _tensor_identity(
    sanitized_name: str,
    role: str,
    role_kind: str | None,
    cursor: _EmissionCursor,
    rename_target: dict[str, str],
) -> tuple[str, bool, str]:
    """Resolve one operand role to its AF tensor identity and rename kind."""
    if role_kind == "outputs":
        if role in {"Output", "Output_0"}:
            return sanitized_name, True, "output"
        if role.startswith("Output_"):
            suffix = role.split("_", 1)[1] if "_" in role else "0"
            return f"{sanitized_name}_{suffix}", True, "output"
        return sanitized_name, True, "output"
    if role_kind == "inputs":
        tensor_name, rename_key = cursor._input_tensor(rename_target)
        return tensor_name, False, rename_key
    return sanitized_name, True, "output"


def _axis_projection(
    ctx: BuildContext,
    layer_name: str,
    role: str,
    dims: list[str],
    atomic_iter_map: dict[str, str],
) -> list[str] | dict[str, str]:
    """Build and safely demote one tensor-access projection."""
    projection: dict[str, str] = {}
    for position in range(len(dims)):
        key = AxisKey(layer=layer_name, role=role, pos=position)
        if key not in ctx.axes:
            continue
        canonical = ctx.canonical_name[key]
        projection[canonical] = _projection_for_axis(ctx, key, atomic_iter_map)
    can_demote = all(
        value.isidentifier() and rank == value.upper()
        for rank, value in projection.items()
    )
    return list(projection.values()) if can_demote else projection


def _register_rename(
    rename_target: dict[str, str],
    key: str,
    tensor_name: str,
) -> None:
    """Record only the first AF tensor selected for each rename role."""
    rename_target.setdefault(key, tensor_name)


def _layer_accesses(
    ctx: BuildContext,
    layer_name: str,
    layer: dict[str, Any],
    sanitized_name: str,
    cursor: _EmissionCursor,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Emit all raw tensor accesses and rename targets for one layer."""
    accesses: list[dict[str, Any]] = []
    rename_target: dict[str, str] = {}
    atomic_iter_map = _build_iter_expr_for_layer(ctx, layer_name)
    for role, dims in (layer.get("operands") or {}).items():
        context_index = ctx.role_to_shape_index.get((layer_name, role))
        role_kind = context_index[0] if context_index is not None else None
        tensor_name, is_output, rename_key = _tensor_identity(
            sanitized_name,
            role,
            role_kind,
            cursor,
            rename_target,
        )
        access: dict[str, Any] = {
            "name": tensor_name,
            "projection": _axis_projection(
                ctx, layer_name, role, dims, atomic_iter_map
            ),
        }
        if is_output:
            access["output"] = True
        if (bits := _bits_for_role(layer, role, context_index)) is not None:
            access["bits_per_value"] = bits
        accesses.append(access)
        _register_rename(rename_target, rename_key, tensor_name)
    return accesses, rename_target


def _deduplicate_input_accesses(
    accesses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate identical non-output tensor reads."""
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for access in accesses:
        if access.get("output"):
            result.append(access)
            continue
        projection = access["projection"]
        key = (
            access["name"],
            tuple(projection)
            if isinstance(projection, list)
            else tuple(sorted(projection.items())),
        )
        if key not in seen:
            seen.add(key)
            result.append(access)
    return result


def _ensure_entry_input(
    accesses: list[dict[str, Any]],
    sanitized_name: str,
    predecessors: list[str],
) -> bool:
    """Synthesize the model-boundary input for an entry pseudo-node."""
    has_input = any(not access.get("output") for access in accesses)
    has_output = any(access.get("output") for access in accesses)
    if not (has_output and not has_input and not predecessors):
        return False
    output = next(access for access in accesses if access.get("output"))
    projection = output["projection"]
    synthesized: dict[str, Any] = {
        "name": f"{sanitized_name}_in",
        "projection": projection
        if isinstance(projection, list)
        else dict(projection),
    }
    if "bits_per_value" in output:
        synthesized["bits_per_value"] = output["bits_per_value"]
    accesses.insert(0, synthesized)
    return True


def _emit_layer(
    ctx: BuildContext,
    layer_name: str,
    layer: dict[str, Any],
    weight_index: int,
) -> tuple[dict[str, Any] | None, int]:
    """Emit one layer and return the updated global weight counter."""
    if not (layer.get("operands") or {}):
        return None, weight_index
    sanitized_name = _sanitize(layer_name)
    predecessors = (layer.get("connections") or {}).get("inputs") or []
    cursor = _EmissionCursor(
        predecessors=list(predecessors),
        tensor_types=(layer.get("tensor_types") or {}).get("inputs") or [],
        weight_index=weight_index,
    )
    accesses, rename_target = _layer_accesses(
        ctx, layer_name, layer, sanitized_name, cursor
    )
    accesses = _deduplicate_input_accesses(accesses)
    is_entry = _ensure_entry_input(accesses, sanitized_name, predecessors)
    renames = (
        {"input": "Inputs()", "output": "Outputs()", "weight": "Nothing()"}
        if is_entry
        else {
            "input": rename_target.get("input", "Nothing()"),
            "output": rename_target.get("output", sanitized_name),
            "weight": rename_target.get("weight", "Nothing()"),
        }
    )
    entry: dict[str, Any] = {
        "name": sanitized_name,
        "tensor_accesses": accesses,
        "renames": renames,
    }
    if is_entry:
        entry["is_copy_operation"] = True
    return entry, cursor.weight_index


def _canonicalize_rank_order(einsums: list[dict[str, Any]]) -> None:
    """Pin each tensor to the rank order of its first access."""
    canonical: dict[str, list[str]] = {}
    for einsum in einsums:
        for access in einsum["tensor_accesses"]:
            name = str(access["name"])
            projection = access["projection"]
            if isinstance(projection, list):
                ranks = [str(value).upper() for value in projection]
                iterator_map = dict(zip(ranks, projection, strict=False))
            elif isinstance(projection, dict):
                ranks = [str(value) for value in projection]
                iterator_map = {
                    str(rank): str(value) for rank, value in projection.items()
                }
            else:
                continue
            target = canonical.setdefault(name, ranks)
            if ranks != target and all(rank in iterator_map for rank in target):
                access["projection"] = {
                    rank: iterator_map[rank] for rank in target
                }


def _default_bits(einsums: list[dict[str, Any]]) -> int:
    """Return the widest explicitly declared tensor bit width."""
    widths = [
        access.get("bits_per_value")
        for einsum in einsums
        for access in einsum["tensor_accesses"]
        if access.get("bits_per_value") is not None
    ]
    return max(widths) if widths else 32


def _emit_af_workload(ctx: BuildContext, model_name: str) -> dict[str, Any]:
    """Emit one canonical AccelForge workload from a prepared context."""
    del model_name
    einsums: list[dict[str, Any]] = []
    weight_index = 0
    for layer_name, layer in ctx.layers.items():
        entry, weight_index = _emit_layer(ctx, layer_name, layer, weight_index)
        if entry is not None:
            einsums.append(entry)
    _canonicalize_rank_order(einsums)
    workload = {
        "rank_sizes": dict(ctx.rank_sizes),
        "bits_per_value": {"All": _default_bits(einsums)},
        "persistent_tensors": "weight - Intermediates",
        "einsums": einsums,
    }
    return {"workload": workload, "renames": _rename_contract()}


def _rename_contract() -> dict[str, Any]:
    """Return the canonical AF input/output/weight selection contract."""
    return {
        "einsums": [
            {
                "name": "default",
                "tensor_accesses": [
                    {
                        "name": "input",
                        "source": "Inputs & Intermediates",
                        "expected_count": 1,
                    },
                    {
                        "name": "output",
                        "source": "Outputs",
                        "expected_count": 1,
                    },
                    {
                        "name": "weight",
                        "source": "~(input | output)",
                        "expected_count": 1,
                    },
                ],
            }
        ]
    }


def _ghost_scalar_outputs(af: dict[str, Any]) -> None:
    """Give TERMINAL reduction-to-scalar outputs one bounded rank.

    Solar emits ops like ``Model.cross_entropy``, ``Model.mean``,
    ``Model.smooth_l1_loss``, ``Model.kl_div`` with empty output
    projection. An empty-projection output that is a graph TERMINAL (no
    consumer) propagates through AF's join_pmappings as an unconstrained
    schedule, breaking upstream multi-input joins ("No mappings found for
    start <--> start_1"). We give such terminals a single rank borrowed
    from a non-output input access so AF can bound the operation space.

    Two refinements over the historical behavior:

    * **Consumed scalars are left empty.** A scalar output that feeds a
      downstream op (e.g. ``norm`` in ``x / x.norm()``) is a genuine
      intermediate; the consumer reads it with an empty projection too, so
      promoting only one side would make the same tensor carry two rank
      tuples (caught by ``_validate_graph_invariants``). Leaving it empty
      keeps producer and consumer consistent — AF bounds it via the join
      with the consumer's other (ranked) operand.
    * **Smallest rank, not first.** Among the input ranks we pick the
      one with the smallest size. AF rejects an unbounded fresh rank, and
      a fresh size-1 rank is rejected too (ISL "Shape infty"); reusing a
      bounded input rank is required. Choosing the smallest minimizes the
      (already negligible) spurious output-write traffic.
    """
    einsums = af["workload"]["einsums"]
    rank_sizes = af["workload"].get("rank_sizes") or {}
    consumed = {
        ta["name"]
        for e in einsums
        for ta in e["tensor_accesses"]
        if not ta.get("output")
    }
    for e in einsums:
        if e.get("is_copy_operation"):
            continue
        for out_ta in e["tensor_accesses"]:
            if not out_ta.get("output"):
                continue
            proj = out_ta["projection"]
            is_empty = (isinstance(proj, list) and len(proj) == 0) or (
                isinstance(proj, dict) and len(proj) == 0
            )
            if not is_empty or out_ta["name"] in consumed:
                continue
            best_iter = None
            best_size = None
            for ta in e["tensor_accesses"]:
                if ta.get("output"):
                    continue
                p = ta["projection"]
                pairs = (
                    [(v.upper(), v) for v in p]
                    if isinstance(p, list)
                    else list(p.items())
                )
                for rank, it in pairs:
                    sz = rank_sizes.get(
                        rank.upper() if isinstance(rank, str) else rank
                    )
                    if sz is not None and (best_size is None or sz < best_size):
                        best_size, best_iter = sz, it
            if best_iter is not None:
                out_ta["projection"] = [best_iter]
