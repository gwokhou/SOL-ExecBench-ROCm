# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded structural-zero proof followed by shared-consumer demand union."""

from dataclasses import dataclass
from graphlib import TopologicalSorter
from time import monotonic

from solar.analysis.demand_contracts import (
    DemandReason,
    DemandRequest,
    NumericalDomain,
)
from solar.analysis.demand_plan import (
    DemandDeclinedError,
    DemandKind,
    DemandPlan,
    DemandWork,
    checked_items,
    plan_layer,
    pointwise_preserves_zero,
)
from solar.analysis.graph_context import IntrinsicGraph
from solar.analysis.graph_models import IntrinsicLayer
from solar.ir.access_calls import INDEX_TARGETS
from solar.ir.contracts import layer_operation
from solar.ir.triangular import TRIANGULAR_TARGETS


@dataclass(frozen=True, slots=True)
class DemandCounts:
    """Owned profile-independent counts for a single transformed layer."""

    plan: DemandPlan
    reads: tuple[frozenset[int], ...]
    demanded: tuple[int, ...]
    writes: tuple[int, ...]
    macs: int
    evaluated_outputs: int
    zero_initializations: int


@dataclass(frozen=True, slots=True)
class DemandGraph:
    """Original graph stays untouched; None means a cheap unchanged scenario."""

    layers: tuple[DemandCounts, ...]
    source_demands: dict[str, frozenset[int]]
    boundary_reasons: frozenset[DemandReason]


def _ordered_layers(intrinsic: IntrinsicGraph) -> tuple[IntrinsicLayer, ...]:
    live = {
        layer.data.layer_id: layer
        for layer in intrinsic.layers
        if layer.data.layer_id in intrinsic.topology.live_layer_ids
    }
    producers = intrinsic.topology.tensor_producers
    predecessors = {
        name: {
            producers[tensor]
            for tensor in layer.data.input_names
            if tensor in producers and producers[tensor] in live
        }
        for name, layer in live.items()
    }
    return tuple(
        live[name] for name in TopologicalSorter(predecessors).static_order()
    )


def _prepare_plans(
    intrinsic: IntrinsicGraph, work: DemandWork
) -> tuple[DemandPlan, ...]:
    layers = _ordered_layers(intrinsic)
    for layer in layers:
        data = layer.data
        if any(
            type(n) is not int
            for shape in [*data.input_shapes, *data.output_shapes]
            for n in shape
        ):
            raise DemandDeclinedError(DemandReason.SHAPE)
        # Demand unions, zero sets and output initialization all visit these
        # cells. Reserve their aggregate work before making any such sets.
        work.reserve(cells=4 * sum(data.output_sizes))
    return tuple(plan_layer(layer, work) for layer in layers)


def _forward_zeros(
    plans: tuple[DemandPlan, ...], work: DemandWork
) -> dict[str, frozenset[int]]:
    zeros: dict[str, frozenset[int]] = {}
    for plan in plans:
        work.check_time()
        data = plan.layer.data
        sources = tuple(
            zeros.get(name, frozenset()) for name in data.input_names
        )
        result = [frozenset() for _ in data.output_names]
        if plan.kind is DemandKind.VIEW:
            result = [
                frozenset(
                    i
                    for i, source in enumerate(mapping)
                    if source in sources[0]
                )
                for mapping in plan.maps
            ]
        elif plan.kind is DemandKind.TRIANGLE:
            result = [plan.structural_zeros | sources[0]]
        elif plan.kind is DemandKind.POINTWISE:
            result = [
                frozenset(
                    i
                    for i in checked_items(range(data.output_sizes[0]), work)
                    if pointwise_preserves_zero(
                        plan,
                        tuple(
                            mapping[i] in source
                            for mapping, source in zip(
                                plan.maps, sources, strict=True
                            )
                        ),
                    )
                )
            ]
        elif plan.kind is DemandKind.CONTRACTION:
            nonzero = {
                out
                for out, left, right in checked_items(plan.terms, work)
                if left not in sources[0] and right not in sources[1]
            }
            result = [frozenset(range(data.output_sizes[0])) - nonzero]
        zeros.update(zip(data.output_names, result, strict=True))
    return zeros


def _layer_demands(
    plan: DemandPlan,
    output: tuple[set[int], ...],
    zeros: dict[str, frozenset[int]],
    work: DemandWork,
) -> DemandCounts:
    data = plan.layer.data
    reads = [set() for _ in data.input_names]
    writes = tuple(len(items) for items in output)
    evaluated = sum(writes)
    initialization = sum(
        len(items & zeros[name])
        for items, name in zip(output, data.output_names, strict=True)
    )
    macs = plan.layer.compute.macs
    if plan.kind is DemandKind.BOUNDARY:
        reads = [set(range(size)) for size in data.input_sizes]
        writes = tuple(plan.layer.memory.writes)
        initialization = 0
    elif plan.kind is DemandKind.VIEW:
        for demanded, mapping in zip(output, plan.maps, strict=True):
            reads[0].update(mapping[index] for index in demanded)
        writes = tuple(0 for _ in output)
        evaluated = 0
        initialization = 0
    elif plan.kind is DemandKind.CONTRACTION:
        macs = 0
        left_zeros, right_zeros = (
            zeros.get(name, frozenset()) for name in data.input_names
        )
        for out, left, right in checked_items(plan.terms, work):
            if (
                out in output[0]
                and left not in left_zeros
                and right not in right_zeros
            ):
                reads[0].add(left)
                reads[1].add(right)
                macs += 1
    elif plan.kind is DemandKind.TRIANGLE:
        reads[0] = output[0] - zeros[data.output_names[0]]
        # Retain mask comparison and full zero-initialized materialization.
        # Undemanded cells may be initialized but are never read downstream.
        writes = tuple(data.output_sizes)
        initialization = data.output_sizes[0] - len(reads[0])
        evaluated = data.output_sizes[0]
    elif plan.kind is DemandKind.POINTWISE:
        active = output[0] - zeros[data.output_names[0]]
        reads = [{mapping[index] for index in active} for mapping in plan.maps]
        evaluated = len(active)
    return DemandCounts(
        plan,
        tuple(frozenset(items) for items in reads),
        tuple(len(items) for items in output),
        writes,
        macs,
        evaluated,
        initialization,
    )


def restrict_computation(
    intrinsic: IntrinsicGraph, request: DemandRequest
) -> DemandGraph | None:
    """Prove only the restricted algebraic model; never alter original IR/counts."""
    if request.numerical_domain is not NumericalDomain.FINITE:
        raise DemandDeclinedError(DemandReason.NUMERICS)
    if not any(
        layer_operation(layer.data.layer).get("target")
        in TRIANGULAR_TARGETS | INDEX_TARGETS
        for layer in intrinsic.layers
    ):
        return None
    work = DemandWork(request.budget, monotonic() + request.budget.seconds)
    plans = _prepare_plans(intrinsic, work)
    zeros = _forward_zeros(plans, work)
    demands: dict[str, set[int]] = {}
    for plan in plans:
        for name, size in zip(
            plan.layer.data.output_names,
            plan.layer.data.output_sizes,
            strict=True,
        ):
            demands[name] = (
                set(range(size))
                if name in intrinsic.prepared.declared_graph_outputs
                or plan.kind is DemandKind.BOUNDARY
                else set()
            )
    results = []
    for plan in reversed(plans):
        work.check_time()
        data = plan.layer.data
        result = _layer_demands(
            plan,
            tuple(demands[name] for name in data.output_names),
            zeros,
            work,
        )
        for name, needed in zip(data.input_names, result.reads, strict=True):
            demands.setdefault(name, set()).update(needed)
        results.append(result)
    producers = intrinsic.topology.tensor_producers
    sources = {
        name: frozenset(indices)
        for name, indices in demands.items()
        if name not in producers
    }
    return DemandGraph(
        tuple(reversed(results)),
        sources,
        frozenset(plan.boundary for plan in plans if plan.boundary is not None),
    )
