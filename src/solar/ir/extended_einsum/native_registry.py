# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Single capability registry for Extended native semantic operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from solar.errors import (
    NativeAttributeUnsupportedError,
    NativeOperationUnsupportedError,
)
from solar.types import DynamicValue


@dataclass(frozen=True, slots=True)
class NativeEffects:
    """Dialect-independent default effects of a public operation."""

    aliases_input: bool = False
    mutates_input: bool = False
    atomic: bool = False
    opaque_library_call: bool = False


@dataclass(frozen=True, slots=True)
class NativeOpSpec:
    """Complete capability contract for one canonical native operation."""

    target: str
    aliases: frozenset[str]
    input_arity: tuple[int, int | None]
    output_arity: tuple[int, int | None]
    attributes: frozenset[str]
    effects: NativeEffects
    dynamic_shape_rule: str | None
    differentiable: bool
    executor: str
    resource_rule_key: str

    @property
    def test_key(self) -> str:
        """Return the capability-matrix key required for this operation."""
        return self.target


_NO_EFFECTS = NativeEffects()
_COMMON_ATTRIBUTES = frozenset(
    {
        "alpha",
        "approximate",
        "bias",
        "correction",
        "device",
        "dim",
        "dims",
        "dtype",
        "generator",
        "inplace",
        "keepdim",
        "layout",
        "memory_format",
        "eps",
        "out",
        "p",
        "reduction",
        "requires_grad",
        "shape",
        "size",
    }
)


def _spec(
    target: str,
    *,
    aliases: Iterable[str] = (),
    inputs: tuple[int, int | None] = (1, None),
    outputs: tuple[int, int | None] = (1, 1),
    attributes: Iterable[str] = (),
    effects: NativeEffects = _NO_EFFECTS,
    dynamic_shape_rule: str | None = None,
    differentiable: bool = True,
    executor: str = "torch",
    resource_rule_key: str | None = None,
) -> NativeOpSpec:
    return NativeOpSpec(
        target=target,
        aliases=frozenset(aliases),
        input_arity=inputs,
        output_arity=outputs,
        attributes=frozenset(attributes) | _COMMON_ATTRIBUTES,
        effects=effects,
        dynamic_shape_rule=dynamic_shape_rule,
        differentiable=differentiable,
        executor=executor,
        resource_rule_key=resource_rule_key or target,
    )


_VIEW = NativeEffects(aliases_input=True)
_ATOMIC = NativeEffects(atomic=True)
_LIBRARY = NativeEffects(opaque_library_call=True)


def _arithmetic_specs() -> list[NativeOpSpec]:
    return [
        _spec("add", aliases=("__add__", "__radd__"), inputs=(1, 3)),
        _spec("sub", aliases=("__sub__", "__rsub__"), inputs=(1, 2)),
        _spec(
            "mul", aliases=("multiply", "__mul__", "__rmul__"), inputs=(1, 2)
        ),
        _spec("div", aliases=("__truediv__", "__rtruediv__"), inputs=(1, 2)),
        _spec("pow", aliases=("__pow__", "__rpow__"), inputs=(1, 2)),
        _spec(
            "matmul",
            aliases=("__matmul__", "__rmatmul__"),
            inputs=(2, 2),
        ),
        _spec("mm", inputs=(2, 2), resource_rule_key="matmul"),
        _spec("bmm", inputs=(2, 2), resource_rule_key="matmul"),
        _spec("addmm", inputs=(3, 3), resource_rule_key="matmul"),
        _spec("clamp", aliases=("clip",), attributes=("min", "max")),
    ]


def _normalization_sort_specs() -> list[NativeOpSpec]:
    return [
        _spec(
            "rms_norm",
            inputs=(1, 3),
            attributes=("normalized_shape", "weight", "eps"),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec("silu", executor="functional"),
        _spec("softmax", attributes=("dim",), executor="functional"),
        _spec("log_softmax", attributes=("dim",), executor="functional"),
        _spec(
            "sort",
            outputs=(2, 2),
            attributes=("dim", "descending", "stable"),
        ),
        _spec(
            "argsort",
            attributes=("dim", "descending", "stable"),
            differentiable=False,
        ),
        _spec(
            "topk",
            outputs=(2, 2),
            attributes=("k", "dim", "largest", "sorted"),
        ),
        _spec("var", attributes=("dim", "correction", "keepdim")),
        _spec("std", attributes=("dim", "correction", "keepdim")),
        _spec("all", attributes=("dim", "keepdim"), differentiable=False),
        _spec("any", attributes=("dim", "keepdim"), differentiable=False),
    ]


def _pooling_resampling_specs() -> list[NativeOpSpec]:
    return [
        _spec(
            "max_pool2d",
            aliases=("max_pool2d_with_indices",),
            outputs=(1, 2),
            attributes=(
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "ceil_mode",
                "return_indices",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "avg_pool2d",
            attributes=(
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "count_include_pad",
                "divisor_override",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "adaptive_avg_pool2d",
            attributes=("output_size",),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "interpolate",
            attributes=(
                "size",
                "scale_factor",
                "mode",
                "align_corners",
                "recompute_scale_factor",
                "antialias",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "grid_sample",
            inputs=(2, 2),
            attributes=("mode", "padding_mode", "align_corners"),
            executor="functional",
            effects=_LIBRARY,
        ),
    ]


def _indexing_specs() -> list[NativeOpSpec]:
    return [
        _spec("gather", inputs=(2, 2), attributes=("dim", "sparse_grad")),
        _spec(
            "scatter",
            inputs=(2, 3),
            attributes=("dim", "src", "value", "reduce"),
            effects=_ATOMIC,
        ),
        _spec(
            "scatter_add",
            inputs=(3, 3),
            attributes=("dim",),
            effects=_ATOMIC,
        ),
        _spec(
            "index_add",
            inputs=(3, 3),
            attributes=("dim", "alpha"),
            effects=_ATOMIC,
        ),
        _spec("index_select", inputs=(2, 2), attributes=("dim",)),
        _spec(
            "repeat_interleave",
            inputs=(1, 2),
            attributes=("repeats", "dim", "output_size"),
        ),
        _spec(
            "diagonal",
            aliases=("diag",),
            attributes=("offset", "dim1", "dim2"),
            effects=_VIEW,
        ),
        _spec("flip", attributes=("dims",)),
        _spec("roll", attributes=("shifts", "dims")),
        _spec(
            "pad",
            attributes=("pad", "mode", "value"),
            executor="functional",
        ),
    ]


def _attention_loss_specs() -> list[NativeOpSpec]:
    return [
        _spec(
            "scaled_dot_product_attention",
            aliases=("attention", "sdpa"),
            inputs=(3, 4),
            attributes=(
                "attn_mask",
                "dropout_p",
                "is_causal",
                "scale",
                "enable_gqa",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "cross_entropy",
            inputs=(2, 3),
            attributes=(
                "weight",
                "size_average",
                "ignore_index",
                "reduce",
                "reduction",
                "label_smoothing",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
        _spec(
            "embedding_bag",
            inputs=(2, 5),
            outputs=(1, 4),
            attributes=(
                "offsets",
                "max_norm",
                "norm_type",
                "scale_grad_by_freq",
                "mode",
                "sparse",
                "per_sample_weights",
                "include_last_offset",
                "padding_idx",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
    ]


def _spectral_shape_specs() -> list[NativeOpSpec]:
    return [
        _spec(
            "fft",
            aliases=("fft_fft",),
            attributes=("n", "dim", "norm"),
            executor="fft",
            effects=_LIBRARY,
        ),
        _spec(
            "vector_norm",
            aliases=("linalg_vector_norm",),
            attributes=("ord", "dim", "keepdim"),
            executor="linalg",
        ),
        _spec(
            "nonzero",
            outputs=(1, None),
            attributes=("as_tuple",),
            dynamic_shape_rule="nonzero",
            differentiable=False,
        ),
        _spec(
            "conv_transpose2d",
            aliases=("convtranspose2d",),
            inputs=(2, 3),
            attributes=(
                "bias",
                "stride",
                "padding",
                "output_padding",
                "groups",
                "dilation",
            ),
            executor="functional",
            effects=_LIBRARY,
        ),
    ]


def _compatibility_elementwise_specs() -> list[NativeOpSpec]:
    unary = {
        "abs",
        "bitwise_not",
        "cos",
        "exp",
        "exp2",
        "floor",
        "log",
        "log2",
        "neg",
        "rsqrt",
        "sigmoid",
        "sin",
        "sqrt",
        "square",
        "tanh",
        "xlogy",
    }
    comparisons = {
        "bitwise_and",
        "eq",
        "ge",
        "gt",
        "le",
        "lt",
        "maximum",
        "minimum",
        "ne",
    }
    reductions = {
        "amax",
        "amin",
        "argmax",
        "argmin",
        "logsumexp",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
    }
    aliases = {
        "bitwise_and": ("__and__", "__rand__"),
        "bitwise_not": ("__invert__",),
        "eq": ("__eq__",),
        "ge": ("__ge__",),
        "gt": ("__gt__",),
        "le": ("__le__",),
        "lt": ("__lt__",),
        "ne": ("__ne__",),
        "neg": ("__neg__",),
    }
    return [
        _spec(
            target,
            aliases=aliases.get(target, ()),
            outputs=(1, 2) if target in {"max", "min"} else (1, 1),
        )
        for target in sorted(unary | comparisons | reductions)
    ]


def _compatibility_functional_specs() -> list[NativeOpSpec]:
    functional = {
        "batch_norm",
        "elu",
        "gelu",
        "group_norm",
        "hardsigmoid",
        "hardswish",
        "kl_div",
        "layer_norm",
        "leaky_relu",
        "linear",
        "mish",
        "nll_loss",
        "relu",
        "softplus",
    }
    functional_attributes = {
        "batch_norm": (
            "running_mean",
            "running_var",
            "weight",
            "bias",
            "training",
            "momentum",
            "eps",
        ),
        "group_norm": ("num_groups", "weight", "bias", "eps"),
        "kl_div": (
            "size_average",
            "reduce",
            "reduction",
            "log_target",
        ),
        "layer_norm": ("normalized_shape", "weight", "bias", "eps"),
        "nll_loss": (
            "weight",
            "size_average",
            "ignore_index",
            "reduce",
            "reduction",
        ),
        "softplus": ("beta", "threshold"),
    }
    return [
        _spec(
            target,
            attributes=functional_attributes.get(target, ()),
            executor="functional",
            effects=_LIBRARY,
        )
        for target in sorted(functional)
    ]


def _compatibility_shape_specs() -> list[NativeOpSpec]:
    shapes = {
        "cat",
        "chunk",
        "clone",
        "contiguous",
        "detach",
        "expand",
        "flatten",
        "narrow",
        "permute",
        "repeat",
        "reshape",
        "select",
        "slice",
        "split",
        "squeeze",
        "stack",
        "transpose",
        "unsqueeze",
        "view",
    }
    specs = [
        _spec(
            target,
            outputs=(1, None) if target in {"chunk", "split"} else (1, 1),
            effects=_VIEW if target in _VIEW_TARGETS else NativeEffects(),
        )
        for target in sorted(shapes)
    ]
    specs.extend(
        _spec(target, effects=_ATOMIC)
        for target in sorted({"index_copy", "index_put", "where"})
    )
    return specs


def _compatibility_library_specs() -> list[NativeOpSpec]:
    specs = [
        _spec("identity", executor="special"),
        _spec(
            "getitem",
            aliases=("__getitem__",),
            effects=_VIEW,
            executor="special",
        ),
        _spec("masked_fill", executor="method"),
        *(
            _spec(target, executor="method")
            for target in ("bfloat16", "float", "half", "int", "long")
        ),
        _spec("ones_like"),
        _spec("to", executor="method"),
        _spec("type_as", inputs=(2, 2), executor="method"),
        _spec("cumsum"),
        _spec(
            "embedding",
            inputs=(2, 2),
            executor="functional",
            effects=_LIBRARY,
        ),
    ]
    specs.extend(
        _spec(target, inputs=(2, 3), executor="functional", effects=_LIBRARY)
        for target in ("conv1d", "conv2d", "conv3d")
    )
    specs.extend(
        _spec(target, inputs=(2, 3), executor="functional", effects=_LIBRARY)
        for target in ("conv_transpose1d", "conv_transpose3d")
    )
    return specs


def _build_specs() -> tuple[NativeOpSpec, ...]:
    """Build the reviewed public-API capability table."""
    groups = (
        _arithmetic_specs(),
        _normalization_sort_specs(),
        _pooling_resampling_specs(),
        _indexing_specs(),
        _attention_loss_specs(),
        _spectral_shape_specs(),
        _compatibility_elementwise_specs(),
        _compatibility_functional_specs(),
        _compatibility_shape_specs(),
        _compatibility_library_specs(),
    )
    return tuple(spec for group in groups for spec in group)


_VIEW_TARGETS: Final = frozenset(
    {
        "chunk",
        "detach",
        "expand",
        "flatten",
        "narrow",
        "permute",
        "reshape",
        "select",
        "slice",
        "split",
        "squeeze",
        "transpose",
        "unsqueeze",
        "view",
    }
)

NATIVE_OP_SPECS: Final = _build_specs()
NATIVE_OP_REGISTRY: Final[dict[str, NativeOpSpec]] = {
    spec.target: spec for spec in NATIVE_OP_SPECS
}
NATIVE_OP_ALIASES: Final[dict[str, str]] = {
    alias: spec.target for spec in NATIVE_OP_SPECS for alias in spec.aliases
}


def canonical_native_target(target: str) -> str:
    """Normalize a public spelling to one reviewed canonical target."""
    normalized = target.lower().rsplit(".", maxsplit=1)[-1]
    if normalized.endswith("_") and not normalized.endswith("__"):
        normalized = normalized[:-1]
    return NATIVE_OP_ALIASES.get(normalized, normalized)


def native_op_spec(target: str) -> NativeOpSpec:
    """Return one native capability or fail with the stable public error."""
    canonical = canonical_native_target(target)
    try:
        return NATIVE_OP_REGISTRY[canonical]
    except KeyError as exc:
        raise NativeOperationUnsupportedError(
            f"{target!r} is not registered"
        ) from exc


def validate_native_attributes(
    spec: NativeOpSpec,
    attributes: Mapping[str, DynamicValue],
) -> None:
    """Fail closed when an artifact names an unreviewed public parameter."""
    unknown = sorted(set(attributes) - spec.attributes)
    if unknown:
        raise NativeAttributeUnsupportedError(
            f"{spec.target} does not accept {', '.join(unknown)}"
        )


__all__ = [
    "NATIVE_OP_ALIASES",
    "NATIVE_OP_REGISTRY",
    "NATIVE_OP_SPECS",
    "NativeEffects",
    "NativeOpSpec",
    "canonical_native_target",
    "native_op_spec",
    "validate_native_attributes",
]
