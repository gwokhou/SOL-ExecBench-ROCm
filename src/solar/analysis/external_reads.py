# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unique external reads of a fixed operator graph, before capacity analysis.

This does not propagate output demand or simplify algebraic expressions. Clone
reads its complete operand, while its output remains a separate storage boundary.
Unresolved aliases, effects and unreviewed read semantics remain diagnostic.
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
from solar.analysis.graph_rules import METADATA_INPUT_TARGETS
from solar.ir.access import AccessQuality, AccessReason, TensorAccess
from solar.ir.access_artifact import AccessDocument
from solar.ir.access_calls import STATIC_VIEW_TARGETS
from solar.ir.access_graph import COPY_TARGETS, DYNAMIC_INDEX_TARGETS
from solar.ir.access_views import UnsupportedAccessError
from solar.ir.contracts import (
    layer_operation,
    operation_attributes,
    operation_operands,
)
from solar.ir.extended_einsum.equations import parse_dim_tokens
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    SemanticValueDecodeError,
    decode_semantic_value,
)
from solar.ir.topk_gather import (
    TOPK_GATHER_TARGETS,
    topk_gather_read_proved,
)
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
        "clamp",
        "clone",
        "eq",
        "elu",
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
        "softmax",
        "topk",
        # Batch-1 vocabulary (spec:
        # solar-l3e-external-reads-acceptance): mechanically identical
        # full-operand reads, plus expand/repeat whose external reads equal
        # the operand's unique elements.
        "cat",
        "pow",
        "sin",
        "cos",
        "log2",
        "floor",
        "exp2",
        "log_softmax",
        "softplus",
        "leaky_relu",
        "bitwise_and",
        "xlogy",
        "masked_fill",
        "expand",
        "repeat",
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


def _full_contraction_reads(layer: NodeDict) -> bool:
    """Prove full reads only for unit-scaled addmm and dense 1x1 conv2d."""
    semantic = layer_operation(layer)
    args = operation_operands(semantic)
    attrs = operation_attributes(semantic)

    def argument(name: str, index: int, default: object) -> object:
        return decode_semantic_value(
            attrs.get(name, args[index] if index < len(args) else default),
            (),
            policy=EXTENDED_VALUE_POLICY,
        )

    shapes = layer.get("tensor_shapes") or {}
    inputs, outputs = shapes.get("inputs") or [], shapes.get("outputs") or []
    if (
        not inputs
        or not outputs
        or any(
            type(n) is not int or n <= 0
            for shape in [*inputs, *outputs]
            for n in shape
        )
    ):
        return False
    try:
        if semantic.get("target") == "addmm":
            return (
                argument("alpha", len(args), 1) == 1
                and argument("beta", len(args), 1) == 1
            )
        if semantic.get("target") != "conv2d" or len(inputs) < 2:
            return False
        x, weight = inputs[:2]
        if len(x) != 4 or len(weight) != 4 or weight[2:] != [1, 1]:
            return False
        if outputs != [[x[0], weight[0], x[2], x[3]]] or x[1] != weight[1]:
            return False
        for name, index, expected in (
            ("stride", 3, 1),
            ("padding", 4, 0),
            ("dilation", 5, 1),
        ):
            value = argument(name, index, expected)
            if value != expected and value not in (
                [expected, expected],
                (expected, expected),
            ):
                return False
        aten = semantic.get("exact_target") == "convolution"
        if aten and (
            argument("transposed", 6, False) is not False
            or argument("output_padding", 7, [0, 0]) != [0, 0]
        ):
            return False
        return argument("groups", 8 if aten else 6, 1) == 1
    except SemanticValueDecodeError, TypeError, ValueError:
        return False


def _where_has_external_values(layer: NodeDict, access: AccessDocument) -> bool:
    """A conditional value branch cannot prove a full external value read."""
    names = (layer.get("tensor_names") or {}).get("inputs") or []
    operands = operation_operands(layer_operation(layer))
    if len(operands) != 3:
        return True
    for operand in operands[1:]:
        if isinstance(operand, Mapping) and "tensor" in operand:
            index = operand["tensor"]
            if type(index) is not int or not 0 <= index < len(names):
                return True
            if names[index] in access.tensors:
                return True
    return False


def _blocking_reason(
    layer: NodeDict,
    access: AccessDocument,
    layers: Mapping[str, NodeDict],
    producer_of: Mapping[str, str],
) -> AccessReason | None:
    semantic = layer_operation(layer)
    target = str(semantic.get("target", ""))
    effects = semantic.get("effects") or {}
    if any(
        effects.get(key) for key in ("mutates", "atomic", "opaque_library_call")
    ):
        return AccessReason.EFFECT
    if target in COPY_TARGETS and target != "clone":
        return AccessReason.COPY
    if target in DYNAMIC_INDEX_TARGETS:
        if target in TOPK_GATHER_TARGETS and topk_gather_read_proved(
            layer, layers, access.tensors, producer_of
        ):
            return None
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
        try:
            tokens = [parse_dim_tokens(term) for term in inputs]
        except ValueError:
            return AccessReason.UNKNOWN_READS
        if inputs and all(
            ranks
            and "".join(ranks) == term.upper()
            and all(rank[0].isalpha() for rank in ranks)
            for term, ranks in zip(inputs, tokens, strict=True)
        ):
            return None
        return AccessReason.UNKNOWN_READS
    if target in {"addmm", "conv2d"}:
        return (
            None
            if _full_contraction_reads(layer)
            else AccessReason.UNKNOWN_READS
        )
    if target == "where":
        return (
            AccessReason.UNKNOWN_READS
            if _where_has_external_values(layer, access)
            else None
        )
    if target in METADATA_INPUT_TARGETS:
        return None
    if target == "scalar_tensor":
        operands = operation_operands(semantic)
        if (
            not (layer.get("tensor_names") or {}).get("inputs")
            and len(operands) == 1
            and isinstance(operands[0], Mapping)
            and type(operands[0].get("value")) in (bool, int, float)
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
    if semantic.get("target") in METADATA_INPUT_TARGETS:
        return ()
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
    producer_of: dict[str, str] = {}
    for layer_id, layer in layers.items():
        for name in (layer.get("tensor_names") or {}).get("outputs") or []:
            if name in producer_of:
                raise ValueError(
                    "external reads graph has multiple producers for a tensor"
                )
            producer_of[name] = layer_id
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
        reason = _blocking_reason(layer, access, layers, producer_of)
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
