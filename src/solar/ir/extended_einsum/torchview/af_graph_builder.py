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
from pathlib import Path
from typing import Any

import yaml

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

from solar.ir.extended_einsum.torchview.af_aliases import (
    _apply_shape_op_elision,
    _normalize_operands,
)
from solar.ir.extended_einsum.torchview.af_emission import (
    _emit_af_workload,
    _ghost_scalar_outputs,
)
from solar.ir.extended_einsum.torchview.af_model import (
    BuildContext,
    _assign_canonical_names,
    _build_role_to_shape_index,
    _collect_axes,
    _cross_layer_union,
    _within_layer_union,
)
from solar.ir.extended_einsum.torchview.af_validation import (
    _topological_sort_layers,
    _validate_af_coverage,
    _validate_graph_invariants,
)


def build_af_graph_from_dict(einsum_graph: dict[str, Any]) -> dict[str, Any]:
    """Build the AccelForge einsum graph from an in-memory stage-2 dict.

    The expected input shape is what ``PyTorchToEinsum._build_einsum_graph``
    returns: a dict containing ``"layers": {layer_name: {operands, ...}}``
    plus optional metadata like ``"model_name"``.
    """
    layers = einsum_graph.get("layers") or {}
    if not layers:
        raise ValueError("einsum_graph has no layers")

    layers = _topological_sort_layers(layers)
    layers, elision_diags = _apply_shape_op_elision(layers)
    # Normalize ``operands`` so every real tensor slot has exactly one role
    # — this is the correctness gate that fixes silent multi-input drops
    # (cat/concat/stack) at the AF boundary.
    norm_diags = _normalize_operands(layers)

    ctx = BuildContext(layers=layers)
    if elision_diags:
        ctx.diagnostics.extend(elision_diags)
    if norm_diags:
        ctx.diagnostics.extend(norm_diags)
    _build_role_to_shape_index(layers, ctx)
    _collect_axes(ctx)
    _within_layer_union(ctx)
    _cross_layer_union(ctx)
    _assign_canonical_names(ctx)
    af = _emit_af_workload(ctx, einsum_graph.get("model_name", "model"))
    _ghost_scalar_outputs(af)
    # Note: the historical ``_correct_output_bits`` post-emit dtype repair
    # is no longer needed here — torchview's fp32-override-on-bf16
    # quirk is now repaired at the ``layers`` stage by
    # ``PyTorchToEinsum._repair_torchview_quirks`` (sub-pass C), so the
    # ``tensor_dtypes`` we ingest are already correct.
    # Post-emit coverage: every Solar pred must end up read by its
    # consumer's AF einsum. Raises on any silent drop.
    _validate_af_coverage(af, layers)
    # Hard-fail correctness gate: rank-tuple consistency, bounded ranks,
    # no orphan reads.
    _validate_graph_invariants(af)
    # Re-derive top-level ``bits_per_value: {All: <max>}`` from the
    # post-emit per-access bits so the energy fallback for tensors without
    # an explicit ``bits_per_value`` annotation matches the widest precision
    # actually used by the workload.
    corrected = [
        ta.get("bits_per_value")
        for e in af["workload"]["einsums"]
        for ta in e["tensor_accesses"]
        if ta.get("bits_per_value") is not None
    ]
    if corrected:
        af["workload"]["bits_per_value"]["All"] = max(corrected)
    if ctx.diagnostics:
        af["_diagnostics"] = list(ctx.diagnostics)
    return af


def build_af_graph_from_yaml(
    einsum_graph_yaml: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build the AccelForge einsum graph from a stage-2 YAML on disk.

    Args:
        einsum_graph_yaml: Path to ``einsum_graph.yaml`` (stage-2 output).
        output_path: Optional path to write the resulting AF YAML to.

    Returns:
        Dict with ``workload`` and ``renames`` keys.
    """
    path = Path(einsum_graph_yaml)
    with open(path) as f:
        graph = yaml.safe_load(f)
    af = build_af_graph_from_dict(graph)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        af_to_write = {k: v for k, v in af.items() if not k.startswith("_")}
        with open(out, "w") as f:
            yaml.dump(af_to_write, f, default_flow_style=False, sort_keys=False)
    return af
