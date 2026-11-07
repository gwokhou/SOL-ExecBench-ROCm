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
from solar.types import GraphValue, NodeDict

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
        # Batch-2 vocabulary (spec: solar-l3e-batch2): stride-decidable
        # conditional aliases; the copy branch reads its operand once and
        # unions per base, so counting at the layer is sound.
        "contiguous",
        "reshape",
        # Batch-3 spec: normalization reads every tensor operand fully.
        "layer_norm",
        "batch_norm",
        "group_norm",
        # Batch-7 spec: fused losses read both tensor operands exactly —
        # cross_entropy's log_softmax stage already touches every input
        # element and kl_div is elementwise-then-reduce.
        "kl_div",
    }
)

# Batch-7 spec: pool targets whose full read is decidable from window
# coverage (unique external reads are a set union; multiplicity is out of
# scope at this layer).
POOL_WINDOW_TARGETS = frozenset(
    {"max_pool2d", "max_pool2d_with_indices", "avg_pool2d"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
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


def _decode_window_sizes(
    value: object, spatial: tuple[int, ...], *, positive: bool
) -> tuple[int, ...] | None:
    """Normalize one scalar-or-per-dim window argument, or keep it unproved."""
    minimum = 1 if positive else 0
    dims: tuple[int, ...] | None
    if type(value) is int:
        dims = (value,) * len(spatial)
    elif isinstance(value, (list, tuple)) and len(value) == len(spatial):
        decoded: list[int] = []
        for n in value:
            if type(n) is not int:
                return None
            decoded.append(n)
        dims = tuple(decoded)
    else:
        dims = None
    if dims is None:
        return None
    if not all(n >= minimum for n in dims):
        return None
    return dims


def _windows_cover(
    spatial: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> bool:
    """Decide per-dim receptive-field coverage of the input (batch-7 math)."""
    for size, k, s, p, d in zip(
        spatial, kernel, stride, padding, dilation, strict=True
    ):
        extent = d * (k - 1) + 1
        span = size + 2 * p - extent
        if span < 0 or s > extent:
            return False
        windows = span // s + 1
        if (windows - 1) * s - p + extent < size:
            return False
    return True


def _spatial_input_dims(layer: NodeDict) -> tuple[int, ...] | None:
    """Return the two trailing spatial sizes of a 4-D input, or None."""
    shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    if not shapes or len(shapes[0]) != 4:
        return None
    spatial = (shapes[0][2], shapes[0][3])
    if any(type(n) is not int or n <= 0 for n in spatial):
        return None
    return spatial


def _pool_window_reads_cover_input(layer: NodeDict) -> bool:
    """Prove full pool reads when receptive fields cover every input element.

    Batch-7 spec: external reads are a unique-element union, so window
    coverage suffices. ceil_mode stays unproved (torch's trailing-window drop
    rule is not modeled here).
    """
    spatial = _spatial_input_dims(layer)
    if spatial is None:
        return False
    semantic = layer_operation(layer)
    attrs = operation_attributes(semantic)
    args = operation_operands(semantic)

    def argument(name: str, index: int, default: object) -> object:
        return decode_semantic_value(
            attrs.get(name, args[index] if index < len(args) else default),
            (),
            policy=EXTENDED_VALUE_POLICY,
        )

    try:
        kernel = _decode_window_sizes(
            argument("kernel_size", 1, None), spatial, positive=True
        )
        if kernel is None:
            return False
        stride = argument("stride", 2, None)
        stride_dims = (
            kernel
            if stride is None
            else _decode_window_sizes(stride, spatial, positive=True)
        )
        padding = _decode_window_sizes(
            argument("padding", 3, 0), spatial, positive=False
        )
        dilation = _decode_window_sizes(
            argument("dilation", 4, 1), spatial, positive=True
        )
        ceil_mode = argument("ceil_mode", 5, False)
    except SemanticValueDecodeError, TypeError, ValueError:
        return False
    if stride_dims is None or padding is None or dilation is None:
        return False
    if ceil_mode is True:
        return False
    return _windows_cover(spatial, kernel, stride_dims, padding, dilation)


def _conv2d_window_reads_cover_input(layer: NodeDict) -> bool:
    """Prove full conv reads from the same window-coverage union (batch-8).

    The kernel comes from the weight's spatial dims; groups only partition
    input channels and keep the union intact. Transposed convolutions are a
    scatter with a different read pattern and stay unproved.
    """
    spatial = _spatial_input_dims(layer)
    if spatial is None:
        return False
    shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    weight_shape = shapes[1] if len(shapes) > 1 else None
    if (
        not isinstance(weight_shape, list)
        or len(weight_shape) != 4
        or any(type(n) is not int or n < 1 for n in weight_shape[2:])
    ):
        return False
    kernel = tuple(weight_shape[2:])
    semantic = layer_operation(layer)
    attrs = operation_attributes(semantic)
    args = operation_operands(semantic)

    def argument(name: str, index: int, default: object) -> object:
        return decode_semantic_value(
            attrs.get(name, args[index] if index < len(args) else default),
            (),
            policy=EXTENDED_VALUE_POLICY,
        )

    try:
        stride_dims = _decode_window_sizes(
            argument("stride", 3, 1), spatial, positive=True
        )
        padding = _decode_window_sizes(
            argument("padding", 4, 0), spatial, positive=False
        )
        dilation = _decode_window_sizes(
            argument("dilation", 5, 1), spatial, positive=True
        )
        transposed = argument("transposed", 6, False)
        output_padding = _decode_window_sizes(
            argument("output_padding", 7, [0, 0]), spatial, positive=False
        )
    except SemanticValueDecodeError, TypeError, ValueError:
        return False
    if stride_dims is None or padding is None or dilation is None:
        return False
    if transposed is not False or output_padding != (0, 0):
        return False
    return _windows_cover(spatial, kernel, stride_dims, padding, dilation)


def _nll_data_operand_external(
    layer: NodeDict, access: AccessDocument
) -> bool | None:
    """Decide the frozen nll shape, or keep the layer unproved.

    Batch-7 spec: operands must be (data tensor, target tensor, non-tensor
    …); every tensor operand except the target must be internal, otherwise
    the gathered (subset) read stays unproved.
    """
    semantic = layer_operation(layer)
    operands = operation_operands(semantic)
    names = (layer.get("tensor_names") or {}).get("inputs") or []
    if len(operands) < 2:
        return None
    tensor_slots = [
        operand["tensor"]
        for operand in operands
        if isinstance(operand, Mapping) and "tensor" in operand
    ]
    if tensor_slots[:2] != [0, 1]:
        return None
    for index in tensor_slots:
        if index == 1:
            continue
        if not 0 <= index < len(names):
            return None
        if names[index] in access.tensors:
            return True
    return False


def _cross_entropy_has_extra_tensor_operand(layer: NodeDict) -> bool:
    """A weight operand is a per-class gather and cannot claim a full read."""
    operands = operation_operands(layer_operation(layer))
    return (
        sum(
            isinstance(operand, Mapping) and "tensor" in operand
            for operand in operands
        )
        > 2
    )


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
    if target in COPY_TARGETS and target not in {"clone", "contiguous"}:
        return AccessReason.COPY
    if target in DYNAMIC_INDEX_TARGETS:
        if target in TOPK_GATHER_TARGETS and topk_gather_read_proved(
            layer, layers, access.tensors, producer_of
        ):
            return None
        return AccessReason.DYNAMIC_INDEX
    if target in STATIC_VIEW_TARGETS:
        return _static_view_read_reason(layer, access)
    if semantic.get("kind") == "einsum":
        return _einsum_read_reason(semantic)
    if target in {"addmm", "conv2d"}:
        # Batch-8 spec: window coverage generalizes the 1x1 contraction
        # special case; both prove the unique-element union read.
        return (
            None
            if (
                _full_contraction_reads(layer)
                or (
                    target == "conv2d"
                    and _conv2d_window_reads_cover_input(layer)
                )
            )
            else AccessReason.UNKNOWN_READS
        )
    if target == "nll_loss_forward":
        # Batch-7 spec: the target operand is read fully; a gathered data (or
        # weight) operand keeps the layer in the dynamic-index domain.
        data_external = _nll_data_operand_external(layer, access)
        if data_external is None:
            return AccessReason.UNKNOWN_READS
        return AccessReason.DYNAMIC_INDEX if data_external else None
    if target == "cross_entropy":
        # Batch-7 spec: the fused loss reads both tensor operands exactly;
        # a weight operand is a per-class gather and stays unproved.
        if _cross_entropy_has_extra_tensor_operand(layer):
            return AccessReason.UNKNOWN_READS
        return None
    if target in POOL_WINDOW_TARGETS:
        return (
            None
            if _pool_window_reads_cover_input(layer)
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


def transparent_declared_output_aliases_external(
    layer: NodeDict, declared_outputs: set[str], access: AccessDocument
) -> bool:
    """Decide whether a transparent layer's declared output may alias external storage.

    Batch-6 spec: the alias is internal, and provably free of external-storage
    visibility, only when every tensor input is untracked with reason
    MATERIALIZED (graph-owned fresh storage). A tracked input is a view of an
    external base, and any other untracked reason leaves the storage owner
    unproved, so both keep the alias unproved.
    """
    names = layer.get("tensor_names") or {}
    if not declared_outputs.intersection(names.get("outputs") or []):
        return False
    inputs = names.get("inputs") or []
    return not (
        inputs
        and all(
            name not in access.tensors
            and access.reasons.get(name) is AccessReason.MATERIALIZED
            for name in inputs
        )
    )


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
        quality=quality,
        reasons=tuple(sorted(reasons)),
        budget=budget,
        elements=elements,
        bytes=byte_counts,
        bases=bases,
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
            quality=AccessQuality.UNAVAILABLE,
            reasons=(AccessReason.MISSING_STORAGE,),
            budget=budget,
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
            # Batch-6 spec: an internal-materialized declared output alias has
            # no external-storage visibility; only external-reaching or
            # unproved-storage chains keep the alias unproved.
            if transparent_declared_output_aliases_external(
                layer, declared_outputs, access
            ):
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
            quality=AccessQuality.UNAVAILABLE,
            reasons=tuple(sorted(reasons)),
            budget=budget,
        )
    return _union_reads(reads, budget)


def _einsum_read_reason(
    semantic: Mapping[str, GraphValue],
) -> AccessReason | None:
    """Accept only complete static input dimension token sequences."""
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


def _static_view_read_reason(
    layer: NodeDict, access: AccessDocument
) -> AccessReason | None:
    """Keep unresolved view output access reasons visible."""
    reasons = {
        access.reasons[name]
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    # Batch-2 spec: a resolved static logical remap reads exactly its
    # tracked base (reads are attributed via base_id), so STATIC views
    # pass like materialized ones; unsupported or dynamic reasons still
    # block.
    if reasons <= {AccessReason.MATERIALIZED, AccessReason.STATIC}:
        return None
    return next(iter(sorted(reasons)), AccessReason.UNSUPPORTED_VIEW)
