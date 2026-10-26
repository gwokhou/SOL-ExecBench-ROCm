# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compile owned static semantics into bounded coordinate relations."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from math import prod
from time import monotonic

from solar.analysis.demand_contracts import DemandBudget, DemandReason
from solar.analysis.graph_models import IntrinsicLayer
from solar.ir.access import base_access, contiguous_strides
from solar.ir.access_calls import STATIC_VIEW_TARGETS, derive_accesses
from solar.ir.access_graph import (
    COPY_TARGETS,
    DYNAMIC_INDEX_TARGETS,
    is_proven_view,
)
from solar.ir.access_views import UnsupportedAccessError
from solar.ir.contracts import layer_operation, operation_operands
from solar.ir.triangular import TRIANGULAR_TARGETS, triangular_diagonal
from solar.precision import normalize_dtype

POINTWISE_TARGETS = frozenset(
    {"add", "sub", "mul", "div", "neg", "abs", "relu", "exp", "log", "square"}
)
RESTRICTED_DTYPES = frozenset({"fp16", "bf16", "fp32", "fp64"})
DEMAND_TIME_CHECK_INTERVAL = 256


class DemandDeclinedError(Exception):
    """Supported operational limits or an explicitly unsupported domain."""

    def __init__(self, reason: DemandReason) -> None:
        """Keep rejection reasons distinct from invalid-data exceptions."""
        super().__init__(reason.value)
        self.reason = reason


@dataclass(slots=True)
class DemandWork:
    """Reserve coordinate/term work before allocating; check time between nodes."""

    budget: DemandBudget
    deadline: float
    cells: int = 0
    terms: int = 0

    def reserve(self, *, cells: int = 0, terms: int = 0) -> None:
        """Reject excessive actual iteration, not a nominal shape by itself."""
        if self.cells + cells > self.budget.cells:
            raise DemandDeclinedError(DemandReason.CELLS)
        if self.terms + terms > self.budget.terms:
            raise DemandDeclinedError(DemandReason.TERMS)
        self.check_time()
        self.cells += cells
        self.terms += terms

    def check_time(self) -> None:
        """The outer worker also enforces its independently bounded deadline."""
        if monotonic() >= self.deadline:
            raise DemandDeclinedError(DemandReason.TIME)


class DemandKind(StrEnum):
    """Only these relations participate in backward pruning."""

    VIEW = "view"
    TRIANGLE = "triangle"
    POINTWISE = "pointwise"
    CONTRACTION = "contraction"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class DemandPlan:
    """Logical flat-index mappings; no tensor values or Python reference code."""

    layer: IntrinsicLayer
    kind: DemandKind
    target: str
    maps: tuple[tuple[int, ...], ...] = ()
    terms: tuple[tuple[int, int, int], ...] = ()
    structural_zeros: frozenset[int] = frozenset()
    boundary: DemandReason | None = None


def checked_items[T](items: Iterable[T], work: DemandWork) -> Iterator[T]:
    """Check deadlines inside bounded loops rather than only between operators."""
    for index, item in enumerate(items):
        if index % DEMAND_TIME_CHECK_INTERVAL == 0:
            work.check_time()
        yield item


def coordinates(
    shape: tuple[int, ...], work: DemandWork
) -> Iterator[tuple[int, ...]]:
    """Iterate only after the caller reserves the product of axis sizes."""
    if 0 in shape:
        return iter(())
    return checked_items(product(*(range(size) for size in shape)), work)


def flat_index(point: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Map a validated in-range logical coordinate to a row-major index."""
    return sum(
        i * s for i, s in zip(point, contiguous_strides(shape), strict=True)
    )


def _boundary(layer: IntrinsicLayer, reason: DemandReason) -> DemandPlan:
    return DemandPlan(
        layer,
        DemandKind.BOUNDARY,
        str(layer_operation(layer.data.layer).get("target", "")),
        boundary=reason,
    )


def _view_plan(layer: IntrinsicLayer, work: DemandWork) -> DemandPlan:
    data = layer.data
    semantic = layer_operation(data.layer)
    if len(data.input_names) != 1 or not is_proven_view(data.layer):
        return _boundary(layer, DemandReason.UNSUPPORTED)
    source = base_access(
        data.input_names[0],
        tuple(data.input_shapes[0]),
        normalize_dtype(data.input_dtypes[0]),
    )
    try:
        views = derive_accesses(
            source, semantic, [tuple(s) for s in data.output_shapes]
        )
    except UnsupportedAccessError:
        return _boundary(layer, DemandReason.UNSUPPORTED)
    work.reserve(cells=sum(data.output_sizes))
    maps = tuple(
        tuple(
            view.offset
            + sum(i * s for i, s in zip(p, view.strides, strict=True))
            for p in coordinates(view.shape, work)
        )
        for view in views
    )
    return DemandPlan(
        layer, DemandKind.VIEW, str(semantic["target"]), maps=maps
    )


def _pointwise_plan(layer: IntrinsicLayer, work: DemandWork) -> DemandPlan:
    data = layer.data
    if len(data.output_sizes) != 1 or not data.input_names:
        return _boundary(layer, DemandReason.UNSUPPORTED)
    shape = tuple(data.output_shapes[0])
    maps = []
    work.reserve(cells=prod(shape) * len(data.input_names))
    for raw in data.input_shapes:
        source = tuple(raw)
        if len(source) > len(shape) or any(
            n not in (1, m)
            for n, m in zip(
                source, shape[len(shape) - len(source) :], strict=True
            )
        ):
            raise ValueError("pointwise broadcast shape is inconsistent")
        maps.append(
            tuple(
                flat_index(
                    tuple(
                        0 if n == 1 else i
                        for i, n in zip(
                            p[len(shape) - len(source) :], source, strict=True
                        )
                    ),
                    source,
                )
                for p in coordinates(shape, work)
            )
        )
    return DemandPlan(
        layer,
        DemandKind.POINTWISE,
        str(layer_operation(data.layer)["target"]),
        maps=tuple(maps),
    )


def _triangle_plan(layer: IntrinsicLayer, work: DemandWork) -> DemandPlan:
    data = layer.data
    semantic = layer_operation(data.layer)
    if (
        len(data.input_names) != 1
        or len(data.output_names) != 1
        or data.input_shapes != data.output_shapes
    ):
        raise ValueError("triangular shape or arity is inconsistent")
    shape = tuple(data.output_shapes[0])
    if len(shape) < 2:
        raise ValueError("triangular operation requires a matrix")
    try:
        diagonal = triangular_diagonal(semantic)
    except UnsupportedAccessError:
        return _boundary(layer, DemandReason.INDEX)
    target = str(semantic["target"])
    work.reserve(cells=prod(shape))
    zeros = frozenset(
        i
        for i, p in enumerate(coordinates(shape, work))
        if (
            p[-1] - p[-2] > diagonal
            if target == "tril"
            else p[-1] - p[-2] < diagonal
        )
    )
    return DemandPlan(
        layer, DemandKind.TRIANGLE, target, structural_zeros=zeros
    )


def _equation(layer: IntrinsicLayer) -> str | None:
    semantic = layer_operation(layer.data.layer)
    if semantic.get("kind") == "einsum":
        return str(semantic.get("equation", ""))
    if semantic.get("target") in {"mm", "matmul"} and all(
        len(s) == 2 for s in layer.data.input_shapes
    ):
        return "ik,kj->ij"
    if semantic.get("target") == "bmm" and all(
        len(s) == 3 for s in layer.data.input_shapes
    ):
        return "bik,bkj->bij"
    return None


def _contraction_plan(
    layer: IntrinsicLayer, work: DemandWork, equation: str
) -> DemandPlan:
    inputs, arrow, output = equation.partition("->")
    terms = inputs.split(",")
    labels = "".join(dict.fromkeys(inputs.replace(",", "")))
    if (
        not arrow
        or len(terms) != 2
        or any(
            not term.isascii()
            or not term.isalpha()
            or len(set(term)) != len(term)
            for term in [*terms, output]
        )
    ):
        return _boundary(layer, DemandReason.UNSUPPORTED)
    data = layer.data
    if len(data.input_shapes) != 2 or len(data.output_shapes) != 1:
        return _boundary(layer, DemandReason.UNSUPPORTED)
    sizes = {}
    for term, shape in zip(
        [*terms, output], [*data.input_shapes, *data.output_shapes], strict=True
    ):
        if len(term) != len(shape):
            raise ValueError("contraction labels and rank differ")
        for label, size in zip(term, shape, strict=True):
            if label in sizes and sizes[label] != size:
                return _boundary(layer, DemandReason.UNSUPPORTED)
            sizes[label] = size
    if not set(output) <= set(labels):
        raise ValueError("contraction output labels have no input")
    count = prod(sizes[label] for label in labels)
    if count != layer.compute.macs:
        return _boundary(layer, DemandReason.UNSUPPORTED)
    work.reserve(
        terms=count, cells=sum(min(size, count) for size in data.input_sizes)
    )
    indices = tuple(
        tuple(labels.index(label) for label in term)
        for term in (output, *terms)
    )
    shapes = [
        tuple(data.output_shapes[0]),
        *(tuple(s) for s in data.input_shapes),
    ]
    triples = tuple(
        _term_indices(p, indices, shapes)
        for p in coordinates(tuple(sizes[label] for label in labels), work)
    )
    return DemandPlan(
        layer,
        DemandKind.CONTRACTION,
        str(layer_operation(data.layer).get("target", "einsum")),
        terms=triples,
    )


def _term_indices(
    point: tuple[int, ...],
    axes: tuple[tuple[int, ...], ...],
    shapes: list[tuple[int, ...]],
) -> tuple[int, int, int]:
    out, left, right = (
        flat_index(tuple(point[i] for i in axis), shape)
        for axis, shape in zip(axes, shapes, strict=True)
    )
    return out, left, right


def _classify_plan(layer: IntrinsicLayer, work: DemandWork) -> DemandPlan:
    """Use structured metadata; unsupported effects/operations retain full demand."""
    data = layer.data
    semantic = layer_operation(data.layer)
    target = str(semantic.get("target", ""))
    effects = semantic.get("effects") or {}
    if any(
        effects.get(key) for key in ("mutates", "atomic", "opaque_library_call")
    ):
        return _boundary(layer, DemandReason.EFFECT)
    if target in COPY_TARGETS:
        return _boundary(layer, DemandReason.COPY)
    if target in DYNAMIC_INDEX_TARGETS:
        return _boundary(layer, DemandReason.INDEX)
    if any(
        normalize_dtype(dtype) not in RESTRICTED_DTYPES
        for dtype in [*data.input_dtypes, *data.output_dtypes]
    ):
        return _boundary(layer, DemandReason.DTYPE)
    if target in STATIC_VIEW_TARGETS:
        return _view_plan(layer, work)
    if target in TRIANGULAR_TARGETS:
        return _triangle_plan(layer, work)
    equation = _equation(layer)
    if equation is not None:
        return _contraction_plan(layer, work, equation)
    if target in POINTWISE_TARGETS:
        return _pointwise_plan(layer, work)
    return _boundary(layer, DemandReason.UNSUPPORTED)


def plan_layer(layer: IntrinsicLayer, work: DemandWork) -> DemandPlan:
    """Reserve the full input domain only when a boundary actually visits it."""
    plan = _classify_plan(layer, work)
    if plan.kind is DemandKind.BOUNDARY:
        work.reserve(cells=sum(layer.data.input_sizes))
    return plan


def pointwise_preserves_zero(
    plan: DemandPlan, zero_operands: tuple[bool, ...]
) -> bool:
    """No division/log/normalization zero inference and no sampled scalar zeros."""
    operands = operation_operands(layer_operation(plan.layer.data.layer))
    if plan.target in {"neg", "abs", "relu", "square"}:
        return len(zero_operands) == 1 and zero_operands[0]
    if len(operands) != 2 or any(
        not isinstance(arg, dict) or set(arg) != {"tensor"} for arg in operands
    ):
        return False
    if plan.target == "mul":
        return any(zero_operands)
    return plan.target in {"add", "sub"} and all(zero_operands)
