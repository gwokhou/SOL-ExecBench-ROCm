# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned AMD compute-resource accounting for executable IR graphs.

The counters in this module are hardware independent.  Architecture profiles
map them to conservative, sourced upper rates.  Official analysis is
fail-closed: an executable operation is either classified here or rejected.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from solar.analysis.resource_targets import (
    _ATOMIC_OPS,
    _COMPOSITE_SFU_OPS,
    _CONVERSION_OPS,
    _INDEX_OPS,
    _LOSS_OPS,
    _MEMORY_ONLY_OPS,
    _MFMA_OPS,
    _NORMALIZATION_OPS,
    _REDUCTION_OPS,
    _SCAN_SORT_OPS,
    _SFU_OPS,
    _VALU_OPS,
    _VARIANCE_OPS,
    _VIEW_OPS,
)
from solar.ir.contracts import (
    layer_operation,
    operation_attributes,
    operation_operands,
)
from solar.precision import normalize_dtype
from solar.schema_versions import AMD_RESOURCE_MODEL_VERSION

RESOURCE_MODEL_VERSION = AMD_RESOURCE_MODEL_VERSION


def is_mfma_operation(target: str) -> bool:
    """Return whether a canonical operation is modeled as an MFMA contraction."""
    return target in _MFMA_OPS


def mandatory_mfma_macs(
    target: str,
    input_shapes: list[Any],
    output_shapes: list[Any],
    semantic: Mapping[str, Any],
) -> int:
    """Return exact dense MAC count for canonical public contractions."""
    if not input_shapes or not output_shapes:
        return 0
    output_n = max(_elements(output_shapes), default=0)
    if target in {"mm", "bmm", "matmul"} and len(input_shapes) >= 2:
        left = input_shapes[0]
        return output_n * int(left[-1]) if left else 0
    if target == "addmm" and len(input_shapes) >= 3:
        left = input_shapes[1]
        return output_n * int(left[-1]) if left else 0
    if target == "linear" and len(input_shapes) >= 2:
        activation = input_shapes[0]
        return output_n * int(activation[-1]) if activation else 0
    if target.startswith("conv_transpose") and len(input_shapes) >= 2:
        activation, weight = input_shapes[:2]
        if len(activation) < 2 or len(weight) < 3:
            return 0
        groups = int(_unwrap(operation_attributes(semantic).get("groups", 1)))
        kernel = math.prod(int(item) for item in weight[2:])
        return output_n * int(activation[1]) * kernel // max(1, groups)
    if target.startswith("conv") and len(input_shapes) >= 2:
        weight = input_shapes[1]
        if len(weight) < 3:
            return 0
        return output_n * math.prod(int(item) for item in weight[1:])
    return 0


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
            result.append(
                int(
                    math.prod(
                        int(dim["lower"])
                        if isinstance(dim, Mapping)
                        else int(dim)
                        for dim in shape
                    )
                )
            )
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


def _reduction_groups(
    shape: list[int] | None,
    semantic: Mapping[str, Any],
) -> int:
    if not shape:
        return 1
    kwargs = operation_attributes(semantic)
    dim = _unwrap(kwargs.get("dim"))
    if dim is None:
        positional = [
            _unwrap(argument)
            for argument in operation_operands(semantic)
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
        semantic = layer_operation(layer)
        target = str(semantic.get("target") or layer.get("type") or "").lower()
        target = target.rsplit(".", maxsplit=1)[-1]
        if target.endswith("_") and not target.endswith("__"):
            target = target[:-1]
        target = {
            "_adaptive_avg_pool2d": "adaptive_avg_pool2d",
            "_embedding_bag_forward_only": "embedding_bag",
            "_fft_c2c": "fft",
            "_fft_r2c": "fft",
            "_fused_rms_norm": "rms_norm",
            "_to_copy": "to",
            "grid_sampler_2d": "grid_sample",
            "linalg_vector_norm": "vector_norm",
            "max_pool2d_with_indices": "max_pool2d",
            "miopen_batch_norm": "batch_norm",
            "native_batch_norm": "batch_norm",
            "split_with_sizes": "split",
            "upsample_bicubic2d": "interpolate",
            "upsample_bilinear2d": "interpolate",
            "upsample_nearest2d": "interpolate",
        }.get(target, target)
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target != "scaled_dot_product_attention":
        return _NO_MATCH
    q_shape = context.input_shapes[0] if context.input_shapes else []
    k_shape = context.input_shapes[1] if len(context.input_shapes) > 1 else []
    if len(q_shape) < 2 or len(k_shape) < 2:
        if context.strict:
            raise ResourceClassificationError(
                "scaled_dot_product_attention requires ranked Q/K tensors",
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
    accumulator.add(
        "valu",
        context.mode,
        2 * score_elements,
        "softmax subtract+divide",
    )
    return _MATCHED


def _view_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    del accumulator
    return (
        _exempt("metadata_or_alias_only")
        if context.target in _VIEW_OPS
        else _NO_MATCH
    )


def _memory_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
        context.input_dtypes[-1]
        if len(context.input_dtypes) >= 2
        else context.dtype
    )
    accumulator.add(
        "atomic",
        _mode(update_dtype, context.fallback_precision),
        updates,
        "one atomic/conflicting update per source element",
    )
    return _MATCHED


def _scan_sort_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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


def _nonzero_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target != "nonzero":
        return _NO_MATCH
    accumulator.add(
        "scan_sort",
        "integer",
        context.input_n,
        "one predicate/scan visit per input element",
    )
    accumulator.add(
        "valu",
        "integer",
        sum(context.output_elements),
        "one emitted coordinate per nonzero output element",
    )
    return _MATCHED


def _pooling_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target not in {
        "adaptive_avg_pool2d",
        "avg_pool2d",
        "max_pool2d",
    }:
        return _NO_MATCH
    accumulator.add(
        "reduction",
        context.mode,
        max(0, context.input_n - context.output_n),
        "conservative input visits minus pooled outputs",
    )
    if context.target != "max_pool2d":
        accumulator.add(
            "valu",
            context.mode,
            context.output_n,
            "one normalization per average-pool output",
        )
    return _MATCHED


def _resampling_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target not in {"grid_sample", "interpolate"}:
        return _NO_MATCH
    accumulator.add(
        "valu",
        context.mode,
        context.output_n,
        "at least one interpolation/sample selection per output",
    )
    accumulator.add(
        "valu",
        "integer",
        context.output_n,
        "at least one source-coordinate calculation per output",
    )
    return _MATCHED


def _vector_norm_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target != "vector_norm":
        return _NO_MATCH
    groups = max(1, context.output_n)
    accumulator.add(
        "valu",
        context.mode,
        context.input_n,
        "one magnitude/power contribution per input",
    )
    accumulator.add(
        "reduction",
        context.mode,
        max(0, context.input_n - groups),
        "norm contribution combines",
    )
    accumulator.add(
        "sfu",
        context.mode,
        groups,
        "one root/power normalization per norm output",
    )
    return _MATCHED


def _embedding_bag_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target != "embedding_bag":
        return _NO_MATCH
    indices = context.input_elements[0] if context.input_elements else 0
    output_shape = context.output_shapes[0] if context.output_shapes else []
    embedding_width = int(output_shape[-1]) if output_shape else 1
    contributions = indices * embedding_width
    accumulator.add(
        "valu",
        "integer",
        indices,
        "one embedding address calculation per index",
    )
    accumulator.add(
        "reduction",
        context.mode,
        max(0, contributions - context.output_n),
        "embedding contributions minus bag outputs",
    )
    return _MATCHED


def _fft_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target != "fft":
        return _NO_MATCH
    source_mode = normalize_dtype(context.dtype, context.fallback_precision)
    mode = {
        "complex32": "fp16",
        "complex64": "fp32",
        "complex128": "fp64",
    }.get(source_mode, source_mode)
    stages = max(1, math.ceil(math.log2(max(2, context.input_n))))
    accumulator.add(
        "valu",
        mode,
        4 * context.output_n * stages,
        "complex butterfly real additions",
    )
    accumulator.add(
        "valu",
        mode,
        4 * context.output_n * stages,
        "complex butterfly real multiplications",
    )
    return _MATCHED


def _conversion_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
            "reduction",
            context.mode,
            2 * combines,
            "maximum and sum reductions",
        )
        accumulator.add(
            "sfu",
            context.mode,
            context.output_n,
            "exponential or logarithm",
        )
        accumulator.add(
            "valu",
            context.mode,
            2 * context.output_n,
            "normalization arithmetic",
        )
    else:
        accumulator.add(
            "reduction",
            context.mode,
            2 * combines,
            "mean and variance reductions",
        )
        accumulator.add(
            "sfu",
            context.mode,
            groups,
            "inverse square root per group",
        )
        accumulator.add(
            "valu",
            context.mode,
            5 * context.output_n,
            "center, scale, normalize, affine",
        )
    return _MATCHED


def _loss_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
) -> _RuleResult:
    if context.target not in _LOSS_OPS:
        return _NO_MATCH
    output_items = max(1, context.output_n)
    if context.target in {"nll_loss", "nll_loss_forward"}:
        labels = (
            context.input_elements[1]
            if len(context.input_elements) > 1
            else output_items
        )
        accumulator.add(
            "valu",
            context.mode,
            labels,
            "one selected negative log-likelihood value per label",
        )
        accumulator.add(
            "valu",
            "integer",
            labels,
            "one class-index selection per label",
        )
        accumulator.add(
            "reduction",
            context.mode,
            max(0, labels - output_items),
            "loss values minus reduction outputs",
        )
        return _MATCHED
    if context.target in {"cross_entropy", "cross_entropy_loss"}:
        labels = (
            context.input_elements[1]
            if len(context.input_elements) > 1
            else output_items
        )
        rows = max(1, labels)
        accumulator.add(
            "reduction",
            context.mode,
            2 * max(0, context.input_n - rows) + max(0, rows - output_items),
            "log-softmax max/sum and loss reduction combines",
        )
        accumulator.add(
            "sfu",
            context.mode,
            context.input_n,
            "one log-softmax exponential/logarithm per logit",
        )
        accumulator.add(
            "valu",
            context.mode,
            2 * context.input_n + labels,
            "log-softmax normalization and selected loss values",
        )
        accumulator.add(
            "valu",
            "integer",
            labels,
            "one class-index selection per label",
        )
        return _MATCHED
    accumulator.add(
        "sfu",
        context.mode,
        context.output_n,
        "one logarithm per KL/xlogy output element",
    )
    accumulator.add(
        "valu",
        context.mode,
        (3 if context.target == "kl_div" else 1) * context.output_n,
        "KL/xlogy mandatory multiply/subtract arithmetic",
    )
    return _MATCHED


def _variance_rule(
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
        "integer" if context.target in {"all", "any"} else context.mode,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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
    context: _ResourceContext,
    accumulator: _ResourceAccumulator,
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


type _ResourceRule = Callable[
    [_ResourceContext, _ResourceAccumulator],
    _RuleResult,
]

_RESOURCE_RULES: tuple[_ResourceRule, ...] = (
    _mfma_rule,
    _attention_rule,
    _view_rule,
    _memory_rule,
    _atomic_rule,
    _scan_sort_rule,
    _nonzero_rule,
    _pooling_rule,
    _resampling_rule,
    _vector_norm_rule,
    _embedding_bag_rule,
    _fft_rule,
    _conversion_rule,
    _normalization_rule,
    _loss_rule,
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
            f"{RESOURCE_MODEL_VERSION} rule",
        )
    return accumulator.result()


def merge_resource_work(
    totals: dict[str, dict[str, int]],
    layer_work: Mapping[str, Mapping[str, Any]],
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
            raise ValueError(
                f"resource_work.{resource} must be a non-empty mapping",
            )
        normalized[str(resource)] = {}
        for mode, amount in modes.items():
            parsed = float(amount)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(
                    "resource work must be finite and non-negative",
                )
            normalized[str(resource)][str(mode)] = parsed
    return normalized


__all__ = [
    "RESOURCE_MODEL_VERSION",
    "ResourceClassificationError",
    "classify_layer_resources",
    "is_mfma_operation",
    "mandatory_mfma_macs",
    "merge_resource_work",
    "validate_resource_work",
]
