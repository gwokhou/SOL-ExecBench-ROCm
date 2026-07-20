# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical serialization for staged graph-analysis results."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import yaml

from solar.analysis.graph_models import (
    AnalysisAccumulator,
    FormalAnalysis,
    GraphIoTotals,
    LowerBound,
)
from solar.analysis.resources import RESOURCE_MODEL_VERSION
from solar.common.types import NodeDict
from solar.common.utils import NoAliasDumper
from solar.rocm.architecture import ArchitectureProfile
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION


class PreparedAnalysisView(Protocol):
    source: Path
    semantic_graph: bool
    requested_precision: str
    fallback_precision: str
    element_size: float
    profile: ArchitectureProfile | None
    strict: bool


def _orojenesis_elements(orojenesis: NodeDict, element_size: float) -> float | None:
    if not orojenesis["layers"] and not orojenesis["chains"]:
        return None
    return sum(
        float(result["selected_capacity"]["point"]["dram_bytes"]) / element_size
        for result in [
            *orojenesis["layers"].values(),
            *orojenesis["chains"].values(),
        ]
        if (result.get("selected_capacity") or {}).get("point")
    )


def _analysis_totals(
    prepared: PreparedAnalysisView,
    accumulator: AnalysisAccumulator,
    io_totals: GraphIoTotals,
    formal: FormalAnalysis,
    lower_bound: LowerBound,
    *,
    start_node_count: int,
    intermediate_tensor_count: int,
) -> NodeDict:
    return {
        "num_layers": len(accumulator.layers),
        "num_start_nodes_filtered": start_node_count,
        "macs": int(accumulator.total_macs),
        "other_ops": int(accumulator.total_other_ops),
        "flops": int(accumulator.total_flops),
        "macs_by_precision": dict(sorted(accumulator.macs_by_precision.items())),
        "resource_work": {
            resource: dict(sorted(modes.items()))
            for resource, modes in sorted(accumulator.resource_work.items())
        },
        "resource_seconds": lower_bound.resource_seconds,
        "compute_resource": lower_bound.compute_resource,
        "unfused_elements": int(accumulator.total_unfused_elems),
        "unfused_bytes": accumulator.total_unfused_bytes,
        "orojenesis_elements": _orojenesis_elements(
            formal.orojenesis, prepared.element_size
        ),
        "fused_elements": io_totals.fused_elements,
        "fused_bytes": formal.audited_fused_bytes,
        "fused_prefetched_elements": io_totals.fused_elements,
        "fused_prefetched_bytes": formal.audited_prefetched_bytes,
        "prefetched_bytes": formal.audited_prefetched_bytes,
        "io_lower_bound_bytes": formal.audited_prefetched_bytes,
        "lower_bound_seconds": lower_bound.seconds,
        "lower_bound_components": lower_bound.components,
        "model_io_elements": int(io_totals.model_io_elements),
        "model_io_bytes": io_totals.model_io_bytes,
        "intermediate_elements": int(accumulator.total_intermediate_elems),
        "intermediate_bytes": accumulator.total_intermediate_bytes,
        "num_intermediate_tensors": intermediate_tensor_count,
        "num_orphaned_layers": len(accumulator.orphaned_layers),
    }


def _analysis_metadata(
    prepared: PreparedAnalysisView,
    accumulator: AnalysisAccumulator,
    formal: FormalAnalysis,
) -> NodeDict:
    return {
        "precision": prepared.requested_precision,
        "fallback_precision": prepared.fallback_precision,
        "bytes_per_element": prepared.element_size,
        "dtype_accounting": (
            "fallback_global" if accumulator.used_dtype_fallback else "per_tensor"
        ),
        "source_graph": str(prepared.source),
        "source_graph_sha256": hashlib.sha256(prepared.source.read_bytes()).hexdigest(),
        "fusion": formal.fusion,
        "orojenesis": formal.orojenesis,
        "bound_kind": (
            "capacity_constrained_tile_aware_v1"
            if formal.formal_bound and prepared.profile is not None
            else "diagnostic"
        ),
        "architecture": (
            prepared.profile.to_dict() if prepared.profile is not None else None
        ),
        "resource_model": {
            "version": RESOURCE_MODEL_VERSION,
            "coverage": dict(accumulator.resource_coverage),
            "fail_closed": prepared.strict,
        },
    }


def build_analysis_result(
    prepared: PreparedAnalysisView,
    accumulator: AnalysisAccumulator,
    io_totals: GraphIoTotals,
    formal: FormalAnalysis,
    lower_bound: LowerBound,
    *,
    start_node_count: int,
    intermediate_tensor_count: int,
) -> NodeDict:
    """Build the canonical graph-analysis payload."""
    return {
        "schema_version": SOLAR_ANALYSIS_SCHEMA_VERSION,
        "layers": accumulator.layers,
        "total": _analysis_totals(
            prepared,
            accumulator,
            io_totals,
            formal,
            lower_bound,
            start_node_count=start_node_count,
            intermediate_tensor_count=intermediate_tensor_count,
        ),
        "metadata": _analysis_metadata(prepared, accumulator, formal),
    }


def write_analysis(output_dir: Path, analysis: NodeDict) -> Path:
    """Write analysis at the pipeline's staged canonical destination."""
    output_path = output_dir / "analysis.yaml"
    with open(output_path, "w") as file:
        yaml.dump(
            analysis,
            file,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
        )
    return output_path


__all__ = ["PreparedAnalysisView", "build_analysis_result", "write_analysis"]
