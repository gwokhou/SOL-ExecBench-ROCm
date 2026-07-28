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

from solar.ir.extended_einsum.torchview.af_model import _sanitize


def _validate_af_coverage(
    af: dict[str, Any],
    layers: dict[str, dict[str, Any]],
) -> None:
    """Assert every non-weight Solar pred appears in its consumer's AF einsum.

    Walks ``tensor_types.inputs`` to determine which preds correspond to
    non-weight slots — weight preds (parameter-tensor) are synthesized as
    ``W{n}`` in the AF emit and don't appear by their original name. Each
    non-weight pred's sanitized layer name must appear in the consumer's
    non-output tensor_accesses. Raises on any drop.

    This is the safety net for the cat/concat/stack class of bugs where
    solar emits a unary operand block for a multi-input op and the AF
    emit silently drops the extra preds.
    """
    einsums_by_name = {e["name"]: e for e in af["workload"]["einsums"]}
    errors: list[str] = []
    for layer_name, layer in layers.items():
        sanitized = _sanitize(layer_name)
        e = einsums_by_name.get(sanitized)
        if e is None:
            continue  # elided by shape-op pass, not a coverage error.
        preds = (layer.get("connections") or {}).get("inputs") or []
        if not preds:
            continue
        in_types = (layer.get("tensor_types") or {}).get("inputs") or []
        # Walk slots: pred_index advances only for non-weight slots, mirroring
        # the AF emit's pred-consumption rule. Each non-weight pred must
        # appear by its sanitized name in the consumer's reads.
        expected: list[str] = []
        pred_index = 0
        for _slot, slot_type in enumerate(in_types):
            if slot_type == "weight":
                continue
            if pred_index < len(preds):
                expected.append(_sanitize(preds[pred_index]))
            pred_index += 1
        # When tensor_types is shorter than preds (older graphs), assume
        # remaining preds are non-weight.
        for k in range(len(in_types), len(preds)):
            expected.append(_sanitize(preds[k]))
        read_names = {
            ta["name"] for ta in e["tensor_accesses"] if not ta.get("output")
        }
        missing = [p for p in expected if p not in read_names]
        if missing:
            errors.append(
                f"layer {layer_name!r} (einsum {sanitized!r}): non-weight "
                f"preds {missing} missing from AF tensor_accesses; "
                f"reads={sorted(read_names)}"
            )
    if errors:
        raise RuntimeError(
            "AF graph coverage check failed:\n  " + "\n  ".join(errors)
        )


def _ranks_of_projection(proj: Any) -> tuple[str, ...]:
    """Ordered uppercase canonical-rank names referenced by a projection."""
    if isinstance(proj, list):
        return tuple(str(v).upper() for v in proj)
    if isinstance(proj, dict):
        return tuple(str(k).upper() for k in proj)
    return ()


def _validate_graph_invariants(af: dict[str, Any]) -> None:
    """Hard-fail correctness gate on the emitted AF workload.

    Checks the invariants the union-find construction is supposed to
    guarantee, so any future regression surfaces immediately instead of
    silently producing a wrong energy number:

    1. **One rank tuple per tensor.** Every access of a given tensor name
       must reference the same ordered rank tuple. Catches the ghost-scalar
       producer/consumer mismatch, cross-layer rank divergence, and the
       multi-input (cat/concat) ordering class.
    2. **Bounded ranks.** Every rank referenced in a projection exists in
       ``rank_sizes`` (so AF's ISL can bound the operation space).
    3. **No orphan reads.** Every non-output tensor read is either produced
       by some einsum, a synthesized weight (``W<n>``), or a synthetic
       model-input source (``*_in`` read inside an entry-point copy einsum).

    Raises ``RuntimeError`` listing every violation.
    """
    workload = af["workload"]
    einsums = workload["einsums"]
    rank_sizes = workload.get("rank_sizes") or {}
    errors: list[str] = []

    tuples_by_name: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    producers: set[str] = set()
    for e in einsums:
        for ta in e["tensor_accesses"]:
            tuples_by_name[ta["name"]].add(
                _ranks_of_projection(ta["projection"])
            )
            if ta.get("output"):
                producers.add(ta["name"])

    for name, tset in tuples_by_name.items():
        if len(tset) > 1:
            errors.append(
                f"tensor {name!r} has inconsistent rank tuples across "
                f"accesses: {sorted(tset)}"
            )

    for e in einsums:
        for ta in e["tensor_accesses"]:
            for r in _ranks_of_projection(ta["projection"]):
                if r not in rank_sizes:
                    errors.append(
                        f"einsum {e['name']!r} tensor {ta['name']!r} references "
                        f"rank {r!r} absent from rank_sizes"
                    )

    for e in einsums:
        is_copy = e.get("is_copy_operation")
        for ta in e["tensor_accesses"]:
            if ta.get("output"):
                continue
            nm = str(ta["name"])
            if nm in producers or re.match(r"^W\d+$", nm):
                continue
            if is_copy and nm.endswith("_in"):
                continue
            # torchview placeholders for tensors created by untraced ops
            # (e.g. BERT position-ids from ``torch.arange``) surface as
            # ``*hidden-tensor`` / ``*auxiliary-tensor`` and are legitimate
            # producerless external sources, not dropped edges (a dropped
            # edge has a producer and is repaired in pytorch_to_einsum).
            if re.search(r"(hidden|auxiliary)[-_]tensor", nm):
                continue
            errors.append(
                f"einsum {e['name']!r} reads {nm!r} which has no producer (orphan read)"
            )

    if errors:
        raise RuntimeError(
            "AF graph invariant check failed:\n  " + "\n  ".join(errors)
        )


# ---------------------------------------------------------------------------
# Topological-order normalization
# ---------------------------------------------------------------------------


def _topological_sort_layers(
    layers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return ``layers`` in topological order (Kahn's algorithm).

    Solar's stage-2 output isn't always topo-sorted — e.g. RMSNorm emits
    ``Model.div`` before ``Model.sqrt`` even though div depends on sqrt.
    AF requires producer-before-consumer order; we enforce it here.
    Ties broken by insertion order — stable & deterministic.
    """
    in_deg: dict[str, int] = dict.fromkeys(layers, 0)
    deps_of: dict[str, set[str]] = {name: set() for name in layers}
    successors_of: dict[str, list[str]] = {name: [] for name in layers}
    for name, layer in layers.items():
        preds = (layer.get("connections") or {}).get("inputs") or []
        for p in preds:
            if p in layers and p not in deps_of[name]:
                deps_of[name].add(p)
                successors_of[p].append(name)
                in_deg[name] += 1
    ready = [n for n in layers if in_deg[n] == 0]
    result: dict[str, dict[str, Any]] = {}
    while ready:
        cur = ready.pop(0)
        result[cur] = layers[cur]
        for succ in successors_of[cur]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                ready.append(succ)
    # Cycle / missing node fallback — keep all data, surface to AF.
    if len(result) != len(layers):
        for name, layer in layers.items():
            if name not in result:
                result[name] = layer
    return result
