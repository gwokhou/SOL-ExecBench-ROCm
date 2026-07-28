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


def _emit_af_workload(ctx: BuildContext, model_name: str) -> dict[str, Any]:
    einsums: list[dict[str, Any]] = []
    weight_counter = [0]

    def next_weight_name() -> str:
        weight_counter[0] += 1
        return f"W{weight_counter[0]}"

    for layer_name, layer in ctx.layers.items():
        operands = layer.get("operands") or {}
        if not operands:
            continue
        sanitized_name = _sanitize(layer_name)
        preds = (layer.get("connections") or {}).get("inputs") or []
        atomic_iter_map = _build_iter_expr_for_layer(ctx, layer_name)
        tensor_accesses: list[dict[str, Any]] = []

        rename_target: dict[str, str] = {}
        primary_input_set = False
        primary_weight_set = False

        # Solar annotates each input role as "input" or "weight" in
        # tensor_types.inputs (positional, matching operands' input-role
        # iteration order). connections.inputs only lists the non-weight
        # producers, so we must skip "weight"-typed roles when stepping
        # through preds — otherwise a layer like ``mul(scale, x)`` where
        # operand 0 is a weight and operand 1 is the predecessor tensor
        # ends up assigning the predecessor's name to the weight slot
        # (size 1) and a synthetic W{n} to the tensor slot (full rank),
        # producing pydantic "inconsistent ranks" errors downstream.
        tensor_types_inputs = (layer.get("tensor_types") or {}).get(
            "inputs"
        ) or []
        input_role_index = 0  # position within input-typed roles only
        pred_index = 0
        for role, dims in operands.items():
            ctx_idx = ctx.role_to_shape_index.get((layer_name, role))
            tensor_name: str
            is_output_access = False
            af_rename_key: str | None = None
            role_kind = ctx_idx[0] if ctx_idx is not None else None
            if role_kind == "outputs":
                is_output_access = True
                if role == "Output" or role == "Output_0":
                    tensor_name = sanitized_name
                elif role.startswith("Output_"):
                    n_str = role.split("_", 1)[1] if "_" in role else "0"
                    tensor_name = f"{sanitized_name}_{n_str}"
                else:
                    tensor_name = sanitized_name
                af_rename_key = "output"
            elif role_kind == "inputs":
                # Honor solar's tensor_types tagging when it disagrees with
                # the simple "any input consumes a pred" assumption.
                role_type = (
                    tensor_types_inputs[input_role_index]
                    if input_role_index < len(tensor_types_inputs)
                    else None
                )
                is_weight_role = role_type == "weight"
                if is_weight_role:
                    tensor_name = next_weight_name()
                    af_rename_key = "weight"
                elif pred_index < len(preds):
                    tensor_name = _sanitize(preds[pred_index])
                    pred_index += 1
                    # Match OLD-pipeline convention: the FIRST consumed
                    # pred maps to "input", subsequent preds map to "weight".
                    af_rename_key = (
                        "input" if not primary_input_set else "weight"
                    )
                else:
                    tensor_name = next_weight_name()
                    af_rename_key = "weight"
                input_role_index += 1
            else:
                tensor_name = sanitized_name
                is_output_access = True
                af_rename_key = "output"

            projection: dict[str, str] = {}
            for pos in range(len(dims)):
                key = AxisKey(layer_name, role, pos)
                if key not in ctx.axes:
                    continue
                canonical = ctx.canonical_name[key]
                expr = _projection_for_axis(ctx, key, atomic_iter_map)
                projection[canonical] = expr
            can_demote = all(
                isinstance(v, str) and v.isidentifier() and k == v.upper()
                for k, v in projection.items()
            )
            access: dict[str, Any] = {
                "name": tensor_name,
                "projection": (
                    list(projection.values())
                    if can_demote
                    else dict(projection)
                ),
            }
            if is_output_access:
                access["output"] = True
            bits = _bits_for_role(layer, role, ctx_idx)
            if bits is not None:
                access["bits_per_value"] = bits
            tensor_accesses.append(access)

            if af_rename_key == "input" and not primary_input_set:
                rename_target["input"] = tensor_name
                primary_input_set = True
            elif af_rename_key == "weight" and not primary_weight_set:
                rename_target["weight"] = tensor_name
                primary_weight_set = True
            elif af_rename_key == "output" and "output" not in rename_target:
                rename_target["output"] = tensor_name

        # Dedup multi-read input accesses (e.g. residual ``Add(x, x)`` lists
        # the same predecessor twice). AF requires unique tensor names in
        # an einsum's tensor_accesses, and reading the same tensor twice
        # with the same projection has the same memory cost as reading it
        # once — keep one access. Output accesses are never deduped.
        seen_in: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for ta in tensor_accesses:
            if ta.get("output"):
                deduped.append(ta)
                continue
            # Hash the (name, projection) pair so distinct-projection
            # multi-reads (rare; would require an alias if encountered)
            # still surface as a separate access.
            proj = ta["projection"]
            key = (
                ta["name"],
                tuple(proj)
                if isinstance(proj, list)
                else tuple(sorted(proj.items())),
            )
            if key in seen_in:
                continue
            seen_in.add(key)
            deduped.append(ta)
        tensor_accesses = deduped

        # Entry-point pseudo-nodes ("start") have outputs but no inputs and
        # no predecessors. Synthesize a source-input access so AF has a
        # tensor to read from at the model boundary.
        has_input = any(not ta.get("output") for ta in tensor_accesses)
        has_output = any(ta.get("output") for ta in tensor_accesses)
        is_entry_point = has_output and not has_input and not preds
        if is_entry_point:
            out_ta = next(ta for ta in tensor_accesses if ta.get("output"))
            synth_name = f"{sanitized_name}_in"
            synth: dict[str, Any] = {
                "name": synth_name,
                "projection": (
                    out_ta["projection"]
                    if isinstance(out_ta["projection"], list)
                    else dict(out_ta["projection"])
                ),
            }
            if "bits_per_value" in out_ta:
                synth["bits_per_value"] = out_ta["bits_per_value"]
            tensor_accesses.insert(0, synth)

        if is_entry_point:
            einsum_renames = {
                "input": "Inputs()",
                "output": "Outputs()",
                "weight": "Nothing()",
            }
        else:
            einsum_renames = {
                "input": rename_target.get("input", "Nothing()"),
                "output": rename_target.get("output", sanitized_name),
                "weight": rename_target.get("weight", "Nothing()"),
            }

        einsum_entry: dict[str, Any] = {
            "name": sanitized_name,
            "tensor_accesses": tensor_accesses,
            "renames": einsum_renames,
        }
        if is_entry_point:
            einsum_entry["is_copy_operation"] = True
        einsums.append(einsum_entry)

    # Pin a canonical rank order per tensor. Union-find guarantees rank
    # IDENTITY is consistent across multiple accesses of the same tensor,
    # but the order in the projection may differ if Solar wrote different
    # operand orderings. Pin the first occurrence and rewrite mismatches.
    canonical_rank_order: dict[str, list[str]] = {}
    for e in einsums:
        for ta in e["tensor_accesses"]:
            name = str(ta["name"])
            proj = ta["projection"]
            if isinstance(proj, list):
                ranks = [str(value).upper() for value in proj]
            elif isinstance(proj, dict):
                ranks = [str(value) for value in proj]
            else:
                continue
            if name not in canonical_rank_order:
                canonical_rank_order[name] = ranks
            else:
                target_ranks = canonical_rank_order[name]
                if ranks != target_ranks:
                    iter_map: dict[str, str] = {}
                    if isinstance(proj, list):
                        for r, v in zip(ranks, proj, strict=False):
                            iter_map[str(r)] = str(v)
                    else:
                        for r, v in proj.items():
                            iter_map[str(r)] = str(v)
                    if all(r in iter_map for r in target_ranks):
                        ta["projection"] = {
                            r: iter_map[r] for r in target_ranks
                        }

    all_bits = [
        ta.get("bits_per_value")
        for e in einsums
        for ta in e["tensor_accesses"]
        if ta.get("bits_per_value") is not None
    ]
    default_bits = max(all_bits) if all_bits else 32

    workload = {
        "rank_sizes": dict(ctx.rank_sizes),
        "bits_per_value": {"All": default_bits},
        "persistent_tensors": "weight - Intermediates",
        "einsums": einsums,
    }
    renames = {
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
    return {"workload": workload, "renames": renames}


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
