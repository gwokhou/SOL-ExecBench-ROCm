# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned AMD compute-resource accounting for executable einsum graphs.

The counters in this module are hardware independent.  Architecture profiles
map them to conservative, sourced upper rates.  Official analysis is
fail-closed: an executable operation is either classified here or rejected.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from solar.common.constants import normalize_dtype

RESOURCE_MODEL_VERSION = "amd_resource_v1"

_VIEW_OPS = frozenset(
    {
        "detach",
        "expand",
        "flatten",
        "getitem",
        "identity",
        "narrow",
        "permute",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "transpose",
        "unsqueeze",
        "view",
    }
)
_MEMORY_ONLY_OPS = frozenset(
    {
        "cat",
        "chunk",
        "clone",
        "contiguous",
        "copy",
        "copy_",
        "pad",
        "repeat",
        "repeat_interleave",
        "split",
        "stack",
        "tensor_split",
    }
)
_MFMA_OPS = frozenset(
    {
        "addmm",
        "bmm",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "linear",
        "matmul",
        "mm",
    }
)
_SFU_OPS = frozenset(
    {
        "cos",
        "exp",
        "log",
        "pow",
        "rsqrt",
        "sin",
        "sqrt",
        "tanh",
    }
)
_COMPOSITE_SFU_OPS = frozenset(
    {
        "elu",
        "gelu",
        "hardsigmoid",
        "hardswish",
        "mish",
        "sigmoid",
        "silu",
        "softplus",
    }
)
_REDUCTION_OPS = frozenset(
    {"amax", "amin", "argmax", "argmin", "logsumexp", "mean", "prod", "sum"}
)
_VARIANCE_OPS = frozenset({"std", "std_mean", "var", "var_mean"})
_NORMALIZATION_OPS = frozenset(
    {"batch_norm", "group_norm", "layer_norm", "log_softmax", "softmax"}
)
_ATOMIC_OPS = frozenset(
    {
        "__setitem__",
        "index_add",
        "index_copy",
        "index_put",
        "scatter",
        "scatter_add",
    }
)
_SCAN_SORT_OPS = frozenset(
    {"argsort", "cummax", "cummin", "cumprod", "cumsum", "sort", "topk"}
)
_CONVERSION_OPS = frozenset(
    {
        "bfloat16",
        "dequantize",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "float",
        "half",
        "int",
        "long",
        "quantize_per_channel",
        "quantize_per_tensor",
        "to",
        "type",
        "type_as",
    }
)
_INDEX_OPS = frozenset(
    {"embedding", "embedding_bag", "gather", "index_select", "tril", "triu"}
)
_VALU_OPS = frozenset(
    {
        "abs",
        "add",
        "bitwise_and",
        "bitwise_not",
        "clamp",
        "div",
        "eq",
        "ge",
        "gt",
        "le",
        "lt",
        "maximum",
        "masked_fill",
        "minimum",
        "mul",
        "ne",
        "neg",
        "ones_like",
        "relu",
        "square",
        "sub",
        "where",
        "zeros_like",
    }
)


class ResourceClassificationError(ValueError):
    """Raised when a semantic compute node has no exact resource rule."""


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) & {"value", "dtype"}:
        return _unwrap(value.get("value", value.get("dtype")))
    if isinstance(value, (list, tuple)):
        return [_unwrap(item) for item in value]
    return value


def _elements(shapes: list[Any]) -> list[int]:
    result: list[int] = []
    for shape in shapes:
        if isinstance(shape, list):
            result.append(int(math.prod(int(dim) for dim in shape)))
        else:
            result.append(0)
    return result


def _mode(dtype: Any, fallback: str) -> str:
    normalized = normalize_dtype(dtype, fallback)
    if normalized.startswith(("int", "uint")):
        return "integer"
    return normalized


def _accumulation_mode(dtype: Any, fallback: str) -> str:
    source = normalize_dtype(dtype, fallback)
    if source in {"fp16", "bf16", "fp8", "nvfp4"}:
        return f"{source}->fp32"
    if source.startswith(("int", "uint")):
        return f"{source}->int32"
    return f"{source}->{source}"


def _reduction_groups(shape: list[int] | None, semantic: Mapping[str, Any]) -> int:
    if not shape:
        return 1
    kwargs = semantic.get("kwargs") or {}
    dim = _unwrap(kwargs.get("dim"))
    if dim is None:
        positional = [
            _unwrap(argument)
            for argument in (semantic.get("arguments") or [])
            if not (isinstance(argument, Mapping) and "tensor" in argument)
        ]
        if positional:
            dim = positional[0]
    if dim is None:
        return 1
    dims = [dim] if isinstance(dim, int) else list(dim)
    rank = len(shape)
    reduced = 1
    for item in dims:
        index = int(item) % rank
        reduced *= int(shape[index])
    return max(1, int(math.prod(shape)) // max(1, reduced))


@dataclass(frozen=True, slots=True)
class _ResourceContext:
    semantic: Mapping[str, Any]
    target: str
    kind: str
    input_shapes: list[Any]
    output_shapes: list[Any]
    input_elements: list[int]
    output_elements: list[int]
    input_dtypes: list[Any]
    output_dtypes: list[Any]
    dtype: Any
    mode: str
    macs: int
    fallback_precision: str
    strict: bool

    @property
    def input_n(self) -> int:
        return max(self.input_elements, default=0)

    @property
    def output_n(self) -> int:
        return max(self.output_elements, default=0)

    @classmethod
    def from_layer(
        cls,
        layer: Mapping[str, Any],
        *,
        macs: int,
        fallback_precision: str,
        strict: bool,
        compute_precision: str | None,
    ) -> _ResourceContext:
        semantic = layer.get("semantic_op") or {}
        target = str(semantic.get("target") or layer.get("type") or "").lower()
        target = target.rsplit(".", maxsplit=1)[-1]
        if target.endswith("_") and not target.endswith("__"):
            target = target[:-1]
        shapes = layer.get("tensor_shapes") or {}
        input_shapes = list(shapes.get("inputs") or [])
        output_shapes = list(shapes.get("outputs") or [])
        dtypes = layer.get("tensor_dtypes") or {}
        input_dtypes = list(dtypes.get("inputs") or [])
        output_dtypes = list(dtypes.get("outputs") or [])
        dtype = compute_precision or (
            input_dtypes[0]
            if input_dtypes
            else (output_dtypes[0] if output_dtypes else fallback_precision)
        )
        return cls(
            semantic=semantic,
            target=target,
            kind=str(semantic.get("kind", "")),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_elements=_elements(input_shapes),
            output_elements=_elements(output_shapes),
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            dtype=dtype,
            mode=_mode(dtype, fallback_precision),
            macs=macs,
            fallback_precision=fallback_precision,
            strict=strict,
        )


@dataclass(slots=True)
class _ResourceAccumulator:
    work: dict[str, dict[str, int]] = field(default_factory=dict)
    formulas: list[str] = field(default_factory=list)

    def add(self, resource: str, mode: str, amount: int, formula: str) -> None:
        if amount <= 0:
            return
        modes = self.work.setdefault(resource, {})
        modes[mode] = modes.get(mode, 0) + int(amount)
        self.formulas.append(formula)

    def result(self, exemption_reason: str | None = None) -> dict[str, Any]:
        if exemption_reason is not None:
            return {
                "model_version": RESOURCE_MODEL_VERSION,
                "work": {},
                "classification": "exempt",
                "exemption_reason": exemption_reason,
                "formulas": [],
            }
        return {
            "model_version": RESOURCE_MODEL_VERSION,
            "work": {
                name: dict(sorted(modes.items()))
                for name, modes in sorted(self.work.items())
            },
            "classification": "modeled" if self.work else "unclassified",
            "exemption_reason": None,
            "formulas": self.formulas,
        }


@dataclass(frozen=True, slots=True)
class _RuleResult:
    matched: bool
    exemption_reason: str | None = None


_NO_MATCH = _RuleResult(matched=False)
_MATCHED = _RuleResult(matched=True)


def _exempt(reason: str) -> _RuleResult:
    return _RuleResult(matched=True, exemption_reason=reason)


def _mfma_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.kind != "einsum" and not (
        context.target in _MFMA_OPS and context.macs > 0
    ):
        return _NO_MATCH
    accumulator.add(
        "mfma",
        _accumulation_mode(context.dtype, context.fallback_precision),
        2 * int(context.macs),
        "2 * contraction_macs",
    )
    return _MATCHED


def _attention_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target != "scaled_dot_product_attention":
        return _NO_MATCH
    q_shape = context.input_shapes[0] if context.input_shapes else []
    k_shape = context.input_shapes[1] if len(context.input_shapes) > 1 else []
    if len(q_shape) < 2 or len(k_shape) < 2:
        if context.strict:
            raise ResourceClassificationError(
                "scaled_dot_product_attention requires ranked Q/K tensors"
            )
        return _MATCHED
    q_rows = int(math.prod(q_shape[:-1]))
    q_width = int(q_shape[-1])
    k_rows_per_batch = int(k_shape[-2])
    score_elements = q_rows * k_rows_per_batch
    accumulator.add(
        "mfma",
        _accumulation_mode(context.dtype, context.fallback_precision),
        4 * q_rows * k_rows_per_batch * q_width,
        "QK and probability-V contractions",
    )
    accumulator.add(
        "reduction",
        context.mode,
        2 * max(0, score_elements - q_rows),
        "softmax max+sum combines",
    )
    accumulator.add("sfu", context.mode, score_elements, "softmax exponentials")
    accumulator.add("valu", context.mode, 2 * score_elements, "softmax subtract+divide")
    return _MATCHED


def _view_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    del accumulator
    return (
        _exempt("metadata_or_alias_only") if context.target in _VIEW_OPS else _NO_MATCH
    )


def _memory_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    del accumulator
    memory_only = context.target in _MEMORY_ONLY_OPS or context.target in {
        "constant_pad_nd",
        "roll",
        "tile",
        "flip",
    }
    return _exempt("memory_traffic_only") if memory_only else _NO_MATCH


def _atomic_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    effects = context.semantic.get("effects") or {}
    if context.target not in _ATOMIC_OPS and not bool(effects.get("atomic")):
        return _NO_MATCH
    updates = (
        context.input_elements[-1]
        if len(context.input_elements) >= 2 and context.input_elements[-1] > 0
        else max(context.output_n, context.input_n)
    )
    update_dtype = (
        context.input_dtypes[-1] if len(context.input_dtypes) >= 2 else context.dtype
    )
    accumulator.add(
        "atomic",
        _mode(update_dtype, context.fallback_precision),
        updates,
        "one atomic/conflicting update per source element",
    )
    return _MATCHED


def _scan_sort_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _SCAN_SORT_OPS:
        return _NO_MATCH
    accumulator.add(
        "scan_sort",
        context.mode,
        max(context.input_n, context.output_n),
        "one mandatory item visit",
    )
    return _MATCHED


def _conversion_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _CONVERSION_OPS:
        return _NO_MATCH
    source_mode = _mode(
        context.input_dtypes[0] if context.input_dtypes else context.dtype,
        context.fallback_precision,
    )
    destination_mode = _mode(
        context.output_dtypes[0] if context.output_dtypes else context.dtype,
        context.fallback_precision,
    )
    if source_mode == destination_mode:
        return _exempt("same_dtype_conversion_noop")
    accumulator.add(
        "conversion",
        f"{source_mode}->{destination_mode}",
        context.output_n,
        "one conversion per output element",
    )
    if "quantize" in context.target or context.target == "dequantize":
        accumulator.add(
            "valu",
            destination_mode,
            2 * context.output_n,
            "quantization scale and offset",
        )
        if "per_channel" in context.target:
            accumulator.add(
                "reduction",
                source_mode,
                max(0, context.input_n - context.output_n),
                "per-channel/block scale reduction",
            )
    return _MATCHED


def _normalization_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _NORMALIZATION_OPS:
        return _NO_MATCH
    shape = (
        context.output_shapes[0]
        if context.output_shapes and isinstance(context.output_shapes[0], list)
        else None
    )
    groups = _reduction_groups(shape, context.semantic)
    combines = max(0, context.output_n - groups)
    if context.target in {"softmax", "log_softmax"}:
        accumulator.add(
            "reduction", context.mode, 2 * combines, "maximum and sum reductions"
        )
        accumulator.add(
            "sfu", context.mode, context.output_n, "exponential or logarithm"
        )
        accumulator.add(
            "valu", context.mode, 2 * context.output_n, "normalization arithmetic"
        )
    else:
        accumulator.add(
            "reduction", context.mode, 2 * combines, "mean and variance reductions"
        )
        accumulator.add("sfu", context.mode, groups, "inverse square root per group")
        accumulator.add(
            "valu",
            context.mode,
            5 * context.output_n,
            "center, scale, normalize, affine",
        )
    return _MATCHED


def _variance_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _VARIANCE_OPS:
        return _NO_MATCH
    shape = (
        context.input_shapes[0]
        if context.input_shapes and isinstance(context.input_shapes[0], list)
        else None
    )
    groups = _reduction_groups(shape, context.semantic)
    combines = max(0, context.input_n - groups)
    accumulator.add(
        "reduction",
        context.mode,
        2 * combines,
        "mean and squared-deviation combines",
    )
    accumulator.add(
        "valu",
        context.mode,
        2 * context.input_n + groups,
        "center, square, and normalize variance values",
    )
    if context.target in {"std", "std_mean"}:
        accumulator.add(
            "sfu",
            context.mode,
            max(context.output_n, groups),
            "square root per reduction group",
        )
    return _MATCHED


def _reduction_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _REDUCTION_OPS:
        return _NO_MATCH
    shape = (
        context.input_shapes[0]
        if context.input_shapes and isinstance(context.input_shapes[0], list)
        else None
    )
    groups = _reduction_groups(shape, context.semantic)
    combines = max(0, context.input_n - groups)
    accumulator.add(
        "reduction",
        context.mode,
        combines,
        "input elements minus reduction groups",
    )
    if context.target == "mean":
        accumulator.add(
            "valu",
            context.mode,
            max(context.output_n, groups),
            "division per reduction result",
        )
    elif context.target == "logsumexp":
        accumulator.add(
            "sfu",
            context.mode,
            context.input_n + max(context.output_n, groups),
            "exponential and logarithm",
        )
    elif combines == 0:
        return _exempt("degenerate_single_element_reduction")
    return _MATCHED


def _sfu_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _SFU_OPS:
        return _NO_MATCH
    accumulator.add(
        "sfu",
        context.mode,
        context.output_n,
        "one special-function result per output element",
    )
    return _MATCHED


def _composite_sfu_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _COMPOSITE_SFU_OPS:
        return _NO_MATCH
    accumulator.add(
        "sfu",
        context.mode,
        context.output_n,
        "one nonlinear special-function result per output element",
    )
    accumulator.add(
        "valu",
        context.mode,
        2 * context.output_n,
        "nonlinear scale/combine arithmetic",
    )
    return _MATCHED


def _index_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _INDEX_OPS:
        return _NO_MATCH
    accumulator.add(
        "valu",
        "integer",
        context.output_n,
        "one integer address/index operation per output element",
    )
    return _MATCHED


def _valu_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.target not in _VALU_OPS:
        return _NO_MATCH
    accumulator.add(
        "valu",
        context.mode,
        context.output_n,
        "one vector ALU result per output element",
    )
    return _MATCHED


def _macs_fallback_rule(
    context: _ResourceContext, accumulator: _ResourceAccumulator
) -> _RuleResult:
    if context.macs <= 0:
        return _NO_MATCH
    accumulator.add(
        "mfma",
        _accumulation_mode(context.dtype, context.fallback_precision),
        2 * int(context.macs),
        "2 * contraction_macs",
    )
    return _MATCHED


type _ResourceRule = Callable[[_ResourceContext, _ResourceAccumulator], _RuleResult]

_RESOURCE_RULES: tuple[_ResourceRule, ...] = (
    _mfma_rule,
    _attention_rule,
    _view_rule,
    _memory_rule,
    _atomic_rule,
    _scan_sort_rule,
    _conversion_rule,
    _normalization_rule,
    _variance_rule,
    _reduction_rule,
    _sfu_rule,
    _composite_sfu_rule,
    _index_rule,
    _valu_rule,
    _macs_fallback_rule,
)


def classify_layer_resources(
    layer: Mapping[str, Any],
    *,
    macs: int,
    fallback_precision: str,
    strict: bool,
    compute_precision: str | None = None,
) -> dict[str, Any]:
    """Return deterministic resource work for one executable graph layer."""
    context = _ResourceContext.from_layer(
        layer,
        macs=macs,
        fallback_precision=fallback_precision,
        strict=strict,
        compute_precision=compute_precision,
    )
    accumulator = _ResourceAccumulator()
    for rule in _RESOURCE_RULES:
        outcome = rule(context, accumulator)
        if outcome.matched:
            return accumulator.result(outcome.exemption_reason)
    if strict:
        raise ResourceClassificationError(
            f"operation {context.target or '<missing>'!r} has no "
            f"{RESOURCE_MODEL_VERSION} rule"
        )
    return accumulator.result()


def merge_resource_work(
    totals: dict[str, dict[str, int]], layer_work: Mapping[str, Mapping[str, Any]]
) -> None:
    """Add one layer's nested resource counters to graph totals."""
    for resource, modes in layer_work.items():
        target = totals.setdefault(str(resource), {})
        for mode, value in modes.items():
            target[str(mode)] = target.get(str(mode), 0) + int(value)


def validate_resource_work(value: Any) -> dict[str, dict[str, float]]:
    """Validate and normalize serialized resource counters."""
    if not isinstance(value, Mapping):
        raise ValueError("resource_work must be a mapping")
    normalized: dict[str, dict[str, float]] = {}
    for resource, modes in value.items():
        if not isinstance(modes, Mapping) or not modes:
            raise ValueError(f"resource_work.{resource} must be a non-empty mapping")
        normalized[str(resource)] = {}
        for mode, amount in modes.items():
            parsed = float(amount)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError("resource work must be finite and non-negative")
            normalized[str(resource)][str(mode)] = parsed
    return normalized


__all__ = [
    "RESOURCE_MODEL_VERSION",
    "ResourceClassificationError",
    "classify_layer_resources",
    "merge_resource_work",
    "validate_resource_work",
]
