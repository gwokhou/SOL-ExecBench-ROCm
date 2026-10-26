# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unique external reads of a fixed operator graph, before capacity analysis.

This does not propagate output demand or simplify algebraic expressions. Copies,
effects, dynamic indexing and unreviewed read semantics remain diagnostic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import prod

from solar.analysis.access_contract import ExternalReadReport
from solar.analysis.access_regions import (
    DEFAULT_ACCESS_BUDGET,
    AccessBudget,
    union_accesses,
)
from solar.ir.access import AccessQuality, AccessReason, TensorAccess
from solar.ir.access_artifact import AccessDocument
from solar.ir.access_calls import STATIC_VIEW_TARGETS
from solar.ir.access_graph import COPY_TARGETS, DYNAMIC_INDEX_TARGETS
from solar.ir.access_views import UnsupportedAccessError
from solar.ir.contracts import layer_operation
from solar.ir.triangular import TRIANGULAR_TARGETS, triangular_accesses
from solar.precision import BYTES_PER_ELEMENT
from solar.types import NodeDict

FULL_OPERAND_READS = frozenset(
    {
        "matmul",
        "mm",
        "bmm",
        "linear",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "square",
        "sum",
        "mean",
        "amax",
        "amin",
        "prod",
        "to",
        "float",
        "half",
        "bfloat16",
        "type_as",
        "relu",
        "gelu",
        "sigmoid",
        "tanh",
        "silu",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
    }
)


@dataclass(frozen=True, slots=True)
class ExternalReads:
    """External reads independently of output writes and capacity traffic."""

    quality: AccessQuality
    reasons: tuple[AccessReason, ...]
    budget: AccessBudget
    elements: dict[str, int] = field(default_factory=dict)
    bytes: dict[str, float] = field(default_factory=dict)
    bases: dict[str, NodeDict] = field(default_factory=dict)

    def to_dict(self) -> NodeDict:
        """Keep unproved totals absent instead of claiming a known zero."""
        available = self.quality is not AccessQuality.UNAVAILABLE
        return ExternalReadReport.model_validate(
            {
                "quality": self.quality.value,
                "reasons": [reason.value for reason in self.reasons],
                "scope": "external_reads_fixed_operator_graph",
                "elements": sum(self.elements.values()) if available else None,
                "bytes": sum(self.bytes.values()) if available else None,
                "bases": self.bases,
                "budget": asdict(self.budget),
            }
        ).model_dump(mode="json")


def _blocking_reason(
    layer: NodeDict, access: AccessDocument
) -> AccessReason | None:
    semantic = layer_operation(layer)
    target = str(semantic.get("target", ""))
    effects = semantic.get("effects") or {}
    if any(
        effects.get(key) for key in ("mutates", "atomic", "opaque_library_call")
    ):
        return AccessReason.EFFECT
    if target in COPY_TARGETS:
        return AccessReason.COPY
    if target in DYNAMIC_INDEX_TARGETS:
        return AccessReason.DYNAMIC_INDEX
    if target in STATIC_VIEW_TARGETS:
        reasons = {
            access.reasons[name]
            for name in (layer.get("tensor_names") or {}).get("outputs") or []
        }
        if reasons <= {AccessReason.MATERIALIZED}:
            return None
        return next(iter(sorted(reasons)), AccessReason.UNSUPPORTED_VIEW)
    if semantic.get("kind") == "einsum":
        equation = str(semantic.get("equation", ""))
        inputs = equation.partition("->")[0].split(",")
        if inputs and all(
            term.isalpha() and len(set(term)) == len(term) for term in inputs
        ):
            return None
        return AccessReason.UNKNOWN_READS
    if target not in FULL_OPERAND_READS | TRIANGULAR_TARGETS:
        return AccessReason.UNKNOWN_READS
    return None


def _union_reads(
    reads: Mapping[str, list[TensorAccess]], budget: AccessBudget
) -> ExternalReads:
    elements, byte_counts, bases = {}, {}, {}
    reasons = set()
    quality = AccessQuality.EXACT
    for base, views in sorted(reads.items()):
        result = union_accesses(views, budget=budget)
        elements[base] = result.elements
        byte_counts[base] = result.elements * BYTES_PER_ELEMENT[views[0].dtype]
        bases[base] = {
            "shape": list(views[0].base_shape),
            "dtype": views[0].dtype,
            "elements": result.elements,
            "bytes": byte_counts[base],
            "quality": result.quality.value,
            "reason": result.reason.value,
        }
        reasons.add(result.reason)
        if result.quality is AccessQuality.LOWER_BOUND:
            quality = AccessQuality.LOWER_BOUND
    return ExternalReads(
        quality, tuple(sorted(reasons)), budget, elements, byte_counts, bases
    )


def _operand_accesses(
    layer: NodeDict, view: TensorAccess, budget: AccessBudget
) -> tuple[TensorAccess, ...]:
    semantic = layer_operation(layer)
    if semantic.get("target") in TRIANGULAR_TARGETS:
        return triangular_accesses(view, semantic, max_regions=budget.regions)
    return (view,)


def external_reads(
    layers: Mapping[str, NodeDict],
    live_layers: set[str],
    declared_outputs: set[str],
    access: AccessDocument | None,
    *,
    budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
) -> ExternalReads:
    """Combine proven operand reads and expose all unsupported domains."""
    if access is None or access.source_storage is None:
        return ExternalReads(
            AccessQuality.UNAVAILABLE, (AccessReason.MISSING_STORAGE,), budget
        )
    reads: dict[str, list[TensorAccess]] = defaultdict(list)
    reasons = {
        access.reasons.get(name, AccessReason.UNKNOWN_READS)
        for name in access.source_indices
        if name not in access.tensors
    }
    for layer_id in sorted(live_layers):
        layer = layers[layer_id]
        names = layer.get("tensor_names") or {}
        if layer_id in access.transparent_layers:
            if declared_outputs.intersection(names.get("outputs") or []):
                reasons.add(AccessReason.ALIAS)
            continue
        reason = _blocking_reason(layer, access)
        if reason is not None:
            reasons.add(reason)
            continue
        shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        if shapes and all(
            isinstance(s, list)
            and all(type(n) is int for n in s)
            and prod(s) == 0
            for s in shapes
        ):
            continue
        for name in names.get("inputs") or []:
            if name in access.tensors:
                view = access.tensors[name]
                try:
                    operands = _operand_accesses(layer, view, budget)
                    if operands:
                        reads[view.base_id].extend(operands)
                except UnsupportedAccessError as exc:
                    reasons.add(exc.reason)
    if reasons:
        return ExternalReads(
            AccessQuality.UNAVAILABLE, tuple(sorted(reasons)), budget
        )
    return _union_reads(reads, budget)
