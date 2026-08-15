# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed values exchanged by the graph-analysis accounting stages."""

from collections import defaultdict
from dataclasses import dataclass, field

from solar.types import GraphValue, NodeDict


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerData:
    """Normalized metadata for one graph layer."""

    layer_id: str
    layer: NodeDict
    op_type: str
    equation: str
    is_real_einsum: bool
    input_layer_ids: list[str]
    output_layer_ids: list[str]
    input_shapes: list[GraphValue]
    output_shapes: list[GraphValue]
    input_types: list[GraphValue]
    output_types: list[GraphValue]
    input_dtypes: list[GraphValue]
    output_dtypes: list[GraphValue]
    input_names: list[str]
    output_names: list[str]
    input_sizes: list[int]
    output_sizes: list[int]


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerCompute:
    """Computed operation counts for one layer."""

    is_real_einsum: bool
    macs: int
    other_ops: int
    flops: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryElements:
    """Element-level memory accounting for one layer."""

    reads: list[int]
    writes: list[int]
    other_ops: int
    orphaned: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryBytes:
    """Byte-level memory accounting for one layer."""

    input_elems: int
    output_elems: int
    unfused_elems: int
    input_bytes: list[float]
    output_bytes: list[float]
    unfused_bytes: float
    used_dtype_fallback: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceAccounting:
    """Per-resource work attributed to one layer."""

    compute_precision: str
    resources: NodeDict


@dataclass(frozen=True, slots=True, kw_only=True)
class InputIo:
    """Classified input traffic for one layer."""

    intermediate_elems: int
    model_elems: int
    intermediate_bytes: float
    model_bytes: float


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputIo:
    """Classified output traffic for one layer."""

    intermediate_elems: int
    model_elems: int
    intermediate_bytes: float
    model_bytes: float
    is_intermediate: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerIo:
    """Combined input and output traffic for one layer."""

    intermediate_elems: int
    intermediate_bytes: float
    model_elems: int
    model_bytes: float
    input_is_intermediate: bool
    output_is_intermediate: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzedLayer:
    """Completed accounting result for one layer."""

    payload: NodeDict
    macs: int
    other_ops: int
    flops: int
    unfused_elems: int
    unfused_bytes: float
    intermediate_elems: int
    intermediate_bytes: float


@dataclass(slots=True, kw_only=True)
class AnalysisAccumulator:
    """Mutable totals accumulated while traversing a graph."""

    layers: dict[str, NodeDict] = field(default_factory=dict)
    total_macs: int = 0
    total_other_ops: int = 0
    total_flops: int = 0
    total_unfused_elems: int = 0
    total_intermediate_elems: int = 0
    total_unfused_bytes: float = 0.0
    total_intermediate_bytes: float = 0.0
    macs_by_precision: dict[str, int] = field(
        default_factory=lambda: defaultdict(int),
    )
    resource_work: dict[str, dict[str, int]] = field(default_factory=dict)
    resource_coverage: dict[str, int] = field(
        default_factory=lambda: {"modeled": 0, "exempt": 0, "unclassified": 0},
    )
    unique_external_inputs: dict[str, int] = field(default_factory=dict)
    unique_external_outputs: dict[str, int] = field(default_factory=dict)
    unique_external_input_bytes: dict[str, float] = field(default_factory=dict)
    unique_external_output_bytes: dict[str, float] = field(default_factory=dict)
    orphaned_layers: set[str] = field(default_factory=set)
    used_dtype_fallback: bool = False

    def record(self, layer_id: str, analyzed: AnalyzedLayer) -> None:
        """Add one analyzed layer to the aggregate totals."""
        self.layers[layer_id] = analyzed.payload
        self.total_macs += analyzed.macs
        self.total_other_ops += analyzed.other_ops
        self.total_flops += analyzed.flops
        self.total_unfused_elems += analyzed.unfused_elems
        self.total_intermediate_elems += analyzed.intermediate_elems
        self.total_unfused_bytes += analyzed.unfused_bytes
        self.total_intermediate_bytes += analyzed.intermediate_bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphIoTotals:
    """Graph-level fused and model-I/O totals."""

    fused_elements: int
    fused_bytes: float
    model_io_elements: int
    model_io_bytes: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FusionPlan:
    """Fusion regions, chains, and supporting proof layers."""

    fusion: NodeDict
    chains: list[list[str]]
    regions: list[NodeDict]
    proof_layers: dict[str, NodeDict]
    unsupported_contraction_layers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalAnalysis:
    """Audited fusion evidence and tile-aware analysis status."""

    fusion: NodeDict | None
    orojenesis: NodeDict
    audited_fused_bytes: float
    audited_prefetched_bytes: float
    tile_aware_bound: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LowerBound:
    """Computed lower bound and its resource components."""

    seconds: float | None
    resource_seconds: dict[str, float]
    compute_resource: str | None
    components: NodeDict | None


__all__ = [
    "AnalysisAccumulator",
    "AnalyzedLayer",
    "FormalAnalysis",
    "FusionPlan",
    "GraphIoTotals",
    "InputIo",
    "LayerCompute",
    "LayerData",
    "LayerIo",
    "LowerBound",
    "MemoryBytes",
    "MemoryElements",
    "OutputIo",
    "ResourceAccounting",
]
