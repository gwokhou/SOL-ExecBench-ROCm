# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Price separate restricted scenarios without recycling dense capacity proofs."""

from collections import defaultdict
from math import fsum

from solar.analysis.demand import DemandCounts, DemandGraph
from solar.analysis.demand_contracts import (
    DEMAND_ASSUMPTIONS,
    DemandLayer,
    DemandReason,
    DemandRequest,
    DemandResult,
    DemandStatus,
)
from solar.analysis.demand_plan import DemandKind
from solar.analysis.graph_context import IntrinsicGraph
from solar.analysis.resources import merge_resource_work
from solar.analysis.views import (
    AnalysisView,
    ViewAvailability,
    ViewEstimate,
    ViewReason,
    read_analysis_views,
)
from solar.ir.access import AccessQuality, TensorAccess
from solar.precision import BYTES_PER_ELEMENT, dtype_bytes
from solar.rocm.architecture import ArchitectureProfile
from solar.types import NodeDict


def _scaled_work(
    work: NodeDict, numerator: int, denominator: int
) -> dict[str, dict[str, int]]:
    result = {}
    for resource, modes in work.items():
        result[resource] = {}
        for mode, amount in modes.items():
            if denominator == 0:
                if amount:
                    raise ValueError("nonzero work on an empty operation")
                scaled = 0
            else:
                scaled, remainder = divmod(amount * numerator, denominator)
                if remainder:
                    raise ValueError("pointwise resource work is not uniform")
            result[resource][mode] = scaled
    return result


def _layer_report(
    counts: DemandCounts, dense: NodeDict, precision: str
) -> DemandLayer:
    plan, data = counts.plan, counts.plan.layer.data
    work = dense["resources"]["work"]
    if plan.kind is DemandKind.CONTRACTION:
        work = _scaled_work(work, counts.macs, plan.layer.compute.macs)
    elif plan.kind is DemandKind.POINTWISE:
        work = _scaled_work(
            work, counts.evaluated_outputs, data.output_sizes[0]
        )
    reads = tuple(len(items) for items in counts.reads)
    traffic_reads = (
        tuple(0 for _ in reads) if plan.kind is DemandKind.VIEW else reads
    )
    if plan.kind is DemandKind.BOUNDARY:
        traffic_reads = tuple(plan.layer.memory.reads)
    amount = fsum(
        n * dtype_bytes(dtype, precision)
        for n, dtype in zip(
            traffic_reads + counts.writes,
            (*data.input_dtypes, *data.output_dtypes),
            strict=True,
        )
    )
    return DemandLayer(
        target=plan.target,
        boundary=plan.boundary,
        dense_macs=plan.layer.compute.macs,
        macs=counts.macs,
        demanded_outputs=counts.demanded,
        input_reads=traffic_reads,
        output_writes=counts.writes,
        zero_initializations=counts.zero_initializations,
        coordinate_work=counts.evaluated_outputs,
        resource_work=work,
        unfused_bytes=amount,
    )


def _base_address(index: int, view: TensorAccess) -> int:
    coordinates = []
    for size in reversed(view.shape):
        index, remainder = divmod(index, size)
        coordinates.append(remainder)
    if index:
        raise ValueError("demand coordinate exceeds source shape")
    return view.offset + sum(
        i * s for i, s in zip(reversed(coordinates), view.strides, strict=True)
    )


def _external_reads(
    graph: DemandGraph, intrinsic: IntrinsicGraph
) -> tuple[
    dict[str, int], float | None, AccessQuality, tuple[DemandReason, ...]
]:
    access = intrinsic.prepared.access
    if (
        access is None
        or access.source_storage is None
        or graph.boundary_reasons
    ):
        reasons = tuple(sorted(graph.boundary_reasons)) or (
            DemandReason.SOURCE_STORAGE,
        )
        return {}, None, AccessQuality.UNAVAILABLE, reasons
    bases: dict[str, set[int]] = defaultdict(set)
    dtypes = {}
    for name, indices in graph.source_demands.items():
        view = access.tensors.get(name)
        if view is None:
            return (
                {},
                None,
                AccessQuality.UNAVAILABLE,
                (DemandReason.SOURCE_STORAGE,),
            )
        bases[view.base_id].update(
            _base_address(index, view) for index in indices
        )
        dtypes[view.base_id] = view.dtype
    counts = {base: len(items) for base, items in sorted(bases.items())}
    return (
        counts,
        fsum(
            count * BYTES_PER_ELEMENT[dtypes[base]]
            for base, count in counts.items()
        ),
        AccessQuality.EXACT,
        (),
    )


def _output_storage(
    graph: DemandGraph, intrinsic: IntrinsicGraph
) -> tuple[float, bool]:
    """Union returned views in their materialized producer, preserving sharing."""
    aliases: dict[str, tuple[tuple[str, int], ...]] = {}
    dtypes = {}
    written: dict[str, set[int]] = defaultdict(set)
    for counts in graph.layers:
        plan = counts.plan
        data = plan.layer.data
        for name, dtype in zip(
            data.input_names + data.output_names,
            data.input_dtypes + data.output_dtypes,
            strict=True,
        ):
            dtypes[name] = dtype
        for slot, name in enumerate(data.output_names):
            if plan.kind is DemandKind.VIEW:
                source = data.input_names[0]
                previous = aliases.get(source)
                aliases[name] = tuple(
                    previous[index] if previous is not None else (source, index)
                    for index in plan.maps[slot]
                )
            if name in intrinsic.prepared.declared_graph_outputs:
                for index in range(data.output_sizes[slot]):
                    base, address = (
                        aliases[name][index]
                        if name in aliases
                        else (name, index)
                    )
                    written[base].add(address)
    external_alias = bool(
        intrinsic.prepared.declared_graph_outputs
        - set(intrinsic.topology.tensor_producers)
    ) or any(
        name not in intrinsic.topology.tensor_producers for name in written
    )
    return fsum(
        len(indices)
        * dtype_bytes(dtypes[name], intrinsic.prepared.fallback_precision)
        for name, indices in written.items()
    ), external_alias


def _views(
    layers: dict[str, DemandLayer],
    work: dict[str, dict[str, int]],
    profile: ArchitectureProfile,
    read_bytes: float | None,
    writes: float,
    quality: AccessQuality,
) -> dict[AnalysisView, ViewEstimate]:
    resource_seconds = profile.resource_seconds(work)
    bandwidth = profile.memory_bandwidth_bytes_per_second
    operators = {
        name: max(
            max(
                profile.resource_seconds(layer.resource_work).values(),
                default=0.0,
            ),
            layer.unfused_bytes / bandwidth,
        )
        for name, layer in layers.items()
    }
    result = {}
    for kind in AnalysisView:
        reason = (
            ViewReason.CAPACITY_INCOMPLETE
            if kind is AnalysisView.CACHE_AWARE
            else ViewReason.READS_UNPROVED
            if read_bytes is None
            else ViewReason.READY
        )
        available = reason is ViewReason.READY
        amount = (
            fsum(layer.unfused_bytes for layer in layers.values())
            if kind is AnalysisView.UNFUSED
            else (read_bytes or 0.0) + writes
        )
        seconds = (
            fsum(operators.values())
            if kind is AnalysisView.UNFUSED
            else max(
                max(resource_seconds.values(), default=0.0), amount / bandwidth
            )
        )
        result[kind] = ViewEstimate(
            status=ViewAvailability.AVAILABLE
            if available
            else ViewAvailability.UNAVAILABLE,
            reason=reason,
            seconds=seconds if available else None,
            io_bytes=amount if available else None,
            resource_seconds=resource_seconds if available else {},
            operator_seconds=operators
            if available and kind is AnalysisView.UNFUSED
            else {},
            read_quality=quality,
            assumptions=(
                *DEMAND_ASSUMPTIONS,
                f"restricted_{kind.value}",
                "transformed_capacity_requires_new_proof",
            ),
        )
    return result


def demand_report(
    intrinsic: IntrinsicGraph,
    request: DemandRequest,
    dense: NodeDict,
    graph: DemandGraph | None,
    profile: ArchitectureProfile,
) -> DemandResult:
    """Keep original analysis untouched and disclose conditional numeric estimates."""
    if graph is None:
        return DemandResult(
            graph_sha256=intrinsic.prepared.source_sha256,
            constraints=request,
            status=DemandStatus.AVAILABLE,
            reason=DemandReason.NO_CHANGE,
            dense_macs=dense["total"]["macs"],
            macs=dense["total"]["macs"],
            resource_work=dense["total"]["resource_work"],
            read_quality=intrinsic.external_reads.quality,
            views=read_analysis_views(dense["metadata"]["views"]),
        )
    layers = {
        counts.plan.layer.data.layer_id: _layer_report(
            counts,
            dense["layers"][counts.plan.layer.data.layer_id],
            intrinsic.prepared.fallback_precision,
        )
        for counts in graph.layers
    }
    work: dict[str, dict[str, int]] = {}
    for layer in layers.values():
        merge_resource_work(work, layer.resource_work)
    counts, reads, quality, reasons = _external_reads(graph, intrinsic)
    writes, external_alias = _output_storage(graph, intrinsic)
    if external_alias:
        counts, reads, quality = {}, None, AccessQuality.UNAVAILABLE
        reasons = tuple(sorted({*reasons, DemandReason.OUTPUT_ALIAS}))
    return DemandResult(
        graph_sha256=intrinsic.prepared.source_sha256,
        constraints=request,
        status=DemandStatus.AVAILABLE,
        reason=DemandReason.READY,
        layers=layers,
        dense_macs=dense["total"]["macs"],
        macs=sum(layer.macs for layer in layers.values()),
        resource_work=work,
        external_read_elements=counts,
        external_read_bytes=reads,
        output_write_bytes=None if external_alias else writes,
        read_quality=quality,
        read_reasons=reasons,
        views=_views(layers, work, profile, reads, writes, quality),
    )
