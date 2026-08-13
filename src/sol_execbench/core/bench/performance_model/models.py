# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict contracts for diagnostic-only microarchitecture modeling."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
    DiagnosticConfidence,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.data.base_model import (
    BaseModelWithDocstrings,
    CurrentSchemaModel,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)
from sol_execbench.core.platform.runtime import PCIeTopologyIdentity

PERFORMANCE_MODEL_VERSION = "gfx1200_diagnostic.v7"
CDNA3_PERFORMANCE_MODEL_VERSION = "cdna3_diagnostic.v1"

# Closed architecture/model-version sets. Never widen to a free-form ``str``:
# an unmapped architecture must fail closed rather than silently admit.
SupportedDiagnosticArchitecture = Literal["gfx1200", "gfx942"]
DiagnosticModelVersion = Literal[
    "gfx1200_diagnostic.v7",
    "cdna3_diagnostic.v1",
]

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    use_attribute_docstrings=True,
)


class WorkloadKind(StrEnum):
    """Workload families supported by the current diagnostic model."""

    ELEMENTWISE = "elementwise"
    TRANSPOSE = "transpose"
    REDUCTION = "reduction_norm"
    MATMUL = "matmul"
    SOFTMAX = "softmax"
    CROSS_ENTROPY = "cross_entropy"
    INDEXED_READ = "indexed_read"
    INDEXED_UPDATE = "indexed_update"
    COMPOSITE = "composite_graph"
    TRANSFORMER = "transformer_block"
    CONCURRENT = "concurrent_graph"
    UNSUPPORTED = "unsupported"


class TensorDType(StrEnum):
    """Floating tensor dtypes admitted by the gfx1200 model."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class ElementwiseOperationClass(StrEnum):
    """Calibrated elementwise operation classes."""

    SIMPLE = "simple"
    TRANSCENDENTAL = "transcendental"
    COMPOSITE = "composite"


class ReductionOperation(StrEnum):
    """Reduction and normalization forms admitted by the model."""

    SUM = "sum"
    MEAN = "mean"
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"


class SoftmaxOperation(StrEnum):
    """Supported normalization operations."""

    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"


class CrossEntropyReduction(StrEnum):
    """Supported CrossEntropy output reductions."""

    MEAN = "mean"
    SUM = "sum"


class IndexedReadOperation(StrEnum):
    """Supported indexed-read forms."""

    GATHER = "gather"
    INDEX_SELECT = "index_select"
    EMBEDDING = "embedding"


class IndexedUpdateOperation(StrEnum):
    """Supported indexed-write and atomic-update forms."""

    SCATTER = "scatter"
    INDEX_COPY = "index_copy"
    INDEX_PUT = "index_put"
    SCATTER_ADD = "scatter_add"
    INDEX_ADD = "index_add"


class CalibrationParameterName(StrEnum):
    """Closed scalar parameter vocabulary for the gfx1200 model."""

    DISPATCH_FLOOR_MS = "dispatch_floor_ms"
    VALU_SIMPLE_FP32_PER_MS = "valu_simple_fp32_per_ms"
    VALU_SIMPLE_BF16_PER_MS = "valu_simple_bf16_per_ms"
    VALU_TRANSCENDENTAL_FP32_PER_MS = "valu_transcendental_fp32_per_ms"
    VALU_TRANSCENDENTAL_BF16_PER_MS = "valu_transcendental_bf16_per_ms"
    VALU_COMPOSITE_FP32_PER_MS = "valu_composite_fp32_per_ms"
    VALU_COMPOSITE_BF16_PER_MS = "valu_composite_bf16_per_ms"
    WMMA_F16_F32_FLOP_PER_MS = "wmma_f16_f32_flop_per_ms"
    L2_BYTE_PER_MS = "l2_byte_per_ms"
    L3_BYTE_PER_MS = "l3_byte_per_ms"
    VRAM_BYTE_PER_MS = "vram_byte_per_ms"
    TRANSPOSE_EFFICIENCY = "transpose_efficiency"
    LDS_BYTE_PER_MS = "lds_byte_per_ms"
    LDS_BANK_CONFLICT_PENALTY_MS = "lds_bank_conflict_penalty_ms"
    REDUCTION_OP_PER_MS = "reduction_op_per_ms"
    BARRIER_PENALTY_MS = "barrier_penalty_ms"
    EDGE_WMMA_EFFICIENCY = "edge_wmma_efficiency"
    IRREGULAR_WMMA_EFFICIENCY = "irregular_wmma_efficiency"
    FP32_MATRIX_FLOP_PER_MS = "fp32_matrix_flop_per_ms"
    SOFTMAX_REDUCTION_OP_PER_MS = "softmax_reduction_op_per_ms"
    INDEXED_ADDRESS_OP_PER_MS = "indexed_address_op_per_ms"
    STRIDED_MATMUL_EFFICIENCY = "strided_matmul_efficiency"


class CalibrationUnit(StrEnum):
    """Closed units accepted by calibration parameters."""

    MS = "ms"
    ITEM_PER_MS = "item/ms"
    FLOP_PER_MS = "flop/ms"
    BYTE_PER_MS = "byte/ms"
    MS_PER_EVENT = "ms/event"
    RATIO = "ratio"


class ApplicabilityDimension(StrEnum):
    """Independent variable used by a calibration applicability interval."""

    WORKING_SET_BYTES = "working_set_bytes"
    REDUCTION_WIDTH = "reduction_width"
    TILE_REMAINDER = "tile_remainder"
    ACTIVE_WAVES = "active_waves"
    INDEX_LOCALITY = "index_locality"
    COLLISION_FRACTION = "collision_fraction"
    MAX_MULTIPLICITY = "max_multiplicity"
    ELEMENT_BYTES = "element_bytes"
    RESOURCE_MIX = "resource_mix"
    CONCURRENT_DISPATCHES = "concurrent_dispatches"


class CalibrationSurfaceName(StrEnum):
    """Closed multidimensional calibration surfaces."""

    INDEXED_READ = "indexed_read"
    ATOMIC_UPDATE = "atomic_update"
    RESIDENCY = "residency"
    OVERLAP = "overlap"


class PredictionKind(StrEnum):
    """Prediction level."""

    IR = "ir"
    HW = "hw"


class RatioKind(StrEnum):
    """Closed ratio vocabulary used by attribution."""

    L = "L"
    C = "C"
    R = "R"


class EvidenceReference(BaseModelWithDocstrings):
    """Content-addressed reference to one diagnostic input."""

    model_config = _MODEL_CONFIG

    kind: str
    path: str | None = None
    sha256: SHA256Digest


class DiagnosticModelIdentity(BaseModelWithDocstrings):
    """Hashes only code and resources that can change model output."""

    model_config = _MODEL_CONFIG

    model_version: str = Field(min_length=1)
    policy_files: dict[str, SHA256Digest] = Field(min_length=1)
    counter_semantics_sha256: SHA256Digest
    policy_bundle_sha256: SHA256Digest


class FusionRegion(BaseModelWithDocstrings):
    """One logical dispatch region established by SOLAR."""

    model_config = _MODEL_CONFIG

    region_id: str
    layer_names: list[str] = Field(default_factory=list)


class ElementwiseDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for a pure elementwise fusion region."""

    model_config = _MODEL_CONFIG

    kind: Literal["elementwise"] = "elementwise"
    shape: list[int] = Field(min_length=1)
    dtype: Literal[TensorDType.FLOAT32, TensorDType.BFLOAT16]
    operations: dict[
        ElementwiseOperationClass,
        Annotated[float, Field(gt=0.0)],
    ] = Field(min_length=1)
    contiguous: Literal[True] = True


class TransposeDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for one out-of-place two-dimensional transpose."""

    model_config = _MODEL_CONFIG

    kind: Literal["transpose"] = "transpose"
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    dtype: TensorDType
    element_bytes: int = Field(gt=0)
    input_strides: tuple[int, int]
    output_strides: tuple[int, int]
    permutation: tuple[Literal[1], Literal[0]] = (1, 0)
    contiguous_input: Literal[True] = True
    out_of_place: Literal[True] = True


class ReductionDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for a last-axis reduction or RMSNorm."""

    model_config = _MODEL_CONFIG

    kind: Literal["reduction_norm"] = "reduction_norm"
    operation: ReductionOperation
    outer_rows: int = Field(gt=0)
    reduction_width: int = Field(gt=0)
    input_dtype: Literal[TensorDType.BFLOAT16, TensorDType.FLOAT32]
    output_dtype: Literal[TensorDType.BFLOAT16, TensorDType.FLOAT32]
    accumulation_dtype: Literal[TensorDType.FLOAT32] = TensorDType.FLOAT32
    contiguous: Literal[True] = True
    reduction_axis: Literal[-1] = -1


class MatmulDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for a calibrated GEMM or BMM."""

    model_config = _MODEL_CONFIG

    kind: Literal["matmul"] = "matmul"
    batch: int = Field(default=1, gt=0)
    m: int = Field(gt=0)
    n: int = Field(gt=0)
    k: int = Field(gt=0)
    transpose_a: bool = False
    transpose_b: bool = False
    leading_dimension_a: int = Field(gt=0)
    leading_dimension_b: int = Field(gt=0)
    leading_dimension_c: int = Field(gt=0)
    input_dtype: TensorDType = TensorDType.FLOAT16
    accumulation_dtype: Literal[TensorDType.FLOAT32] = TensorDType.FLOAT32
    output_dtype: TensorDType = TensorDType.FLOAT32
    contiguous: bool = True
    batch_stride_a: int | None = Field(default=None, ge=0)
    batch_stride_b: int | None = Field(default=None, ge=0)
    batch_stride_c: int | None = Field(default=None, ge=0)


class SoftmaxDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for a contiguous last-axis Softmax."""

    model_config = _MODEL_CONFIG

    kind: Literal["softmax"] = "softmax"
    operation: SoftmaxOperation
    outer_rows: int = Field(gt=0)
    reduction_width: int = Field(gt=0)
    input_dtype: Literal[TensorDType.BFLOAT16, TensorDType.FLOAT32]
    output_dtype: Literal[TensorDType.BFLOAT16, TensorDType.FLOAT32]
    contiguous: Literal[True] = True
    reduction_axis: Literal[-1] = -1


class CrossEntropyDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for class-index CrossEntropy."""

    model_config = _MODEL_CONFIG

    kind: Literal["cross_entropy"] = "cross_entropy"
    rows: int = Field(gt=0)
    classes: int = Field(gt=1)
    logits_dtype: Literal[TensorDType.BFLOAT16, TensorDType.FLOAT32]
    target_dtype: Literal[DType.INT32, DType.INT64]
    reduction: CrossEntropyReduction
    contiguous_logits: Literal[True] = True


class IndexedReadDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for one contiguous-source indexed read."""

    model_config = _MODEL_CONFIG

    kind: Literal["indexed_read"] = "indexed_read"
    operation: IndexedReadOperation
    source_shape: list[int] = Field(min_length=1)
    index_shape: list[int] = Field(min_length=1)
    axis: int
    payload_dtype: TensorDType
    index_dtype: Literal[DType.INT32, DType.INT64]
    element_bytes: int = Field(gt=0)
    contiguous_source: Literal[True] = True

    @model_validator(mode="after")
    def axis_is_in_range(self) -> IndexedReadDescriptor:
        """Require a normalized source axis."""
        if not 0 <= self.axis < len(self.source_shape):
            raise ValueError("indexed-read axis is out of range")
        return self


class IndexedUpdateDescriptor(BaseModelWithDocstrings):
    """Strict descriptor for one indexed write or atomic update."""

    model_config = _MODEL_CONFIG

    kind: Literal["indexed_update"] = "indexed_update"
    operation: IndexedUpdateOperation
    output_shape: list[int] = Field(min_length=1)
    index_shape: list[int] = Field(min_length=1)
    axis: int
    payload_dtype: Literal[TensorDType.FLOAT32]
    index_dtype: Literal[DType.INT32, DType.INT64]
    element_bytes: Literal[4] = 4
    contiguous_output: Literal[True] = True
    atomic: bool

    @model_validator(mode="after")
    def operation_matches_effect(self) -> IndexedUpdateDescriptor:
        """Keep overwrite and atomic operations distinct."""
        expected_atomic = self.operation in {
            IndexedUpdateOperation.SCATTER_ADD,
            IndexedUpdateOperation.INDEX_ADD,
        }
        if self.atomic is not expected_atomic:
            raise ValueError("indexed-update atomic effect mismatch")
        if not 0 <= self.axis < len(self.output_shape):
            raise ValueError("indexed-update axis is out of range")
        return self


PrimitiveSemanticDescriptor = Annotated[
    ElementwiseDescriptor
    | TransposeDescriptor
    | ReductionDescriptor
    | MatmulDescriptor
    | SoftmaxDescriptor
    | CrossEntropyDescriptor
    | IndexedReadDescriptor
    | IndexedUpdateDescriptor,
    Field(discriminator="kind"),
]


class CompositeGraphNode(BaseModelWithDocstrings):
    """One admitted primitive node in a bounded semantic DAG."""

    model_config = _MODEL_CONFIG

    node_id: str = Field(min_length=1)
    layer_names: list[str] = Field(min_length=1)
    descriptor: PrimitiveSemanticDescriptor
    input_tensors: list[str] = Field(default_factory=list)
    output_tensors: list[str] = Field(min_length=1)


class CompositeGraphEdge(BaseModelWithDocstrings):
    """One exact producer-consumer tensor dependency."""

    model_config = _MODEL_CONFIG

    producer: str = Field(min_length=1)
    consumer: str = Field(min_length=1)
    tensor: str = Field(min_length=1)
    materialized: bool


class CompositeGraphDescriptor(BaseModelWithDocstrings):
    """Acyclic graph composed only from admitted primitive nodes."""

    model_config = _MODEL_CONFIG

    kind: Literal["composite_graph"] = "composite_graph"
    graph_class: Literal[
        "composite_graph",
        "transformer_block",
        "concurrent_graph",
    ]
    nodes: list[CompositeGraphNode] = Field(min_length=1)
    edges: list[CompositeGraphEdge] = Field(default_factory=list)
    schedule: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def graph_is_complete_and_acyclic(self) -> CompositeGraphDescriptor:
        """Require exact node identity and a topological schedule."""
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("composite graph repeats node_id")
        if self.schedule != list(dict.fromkeys(self.schedule)) or set(
            self.schedule
        ) != set(node_ids):
            raise ValueError("composite graph schedule is incomplete")
        positions = {
            node_id: index for index, node_id in enumerate(self.schedule)
        }
        for edge in self.edges:
            if edge.producer not in positions or edge.consumer not in positions:
                raise ValueError("composite graph edge references unknown node")
            if positions[edge.producer] >= positions[edge.consumer]:
                raise ValueError("composite graph schedule is not topological")
        return self


class UnsupportedDescriptor(BaseModelWithDocstrings):
    """Fail-closed descriptor for an unsupported semantic graph."""

    model_config = _MODEL_CONFIG

    kind: Literal["unsupported"] = "unsupported"
    reason_codes: list[str] = Field(min_length=1)


SemanticDescriptor = Annotated[
    PrimitiveSemanticDescriptor
    | CompositeGraphDescriptor
    | UnsupportedDescriptor,
    Field(discriminator="kind"),
]


class SemanticCharacterization(BaseModelWithDocstrings):
    """Validated semantic workload characterization from SOLAR."""

    model_config = _MODEL_CONFIG

    workload_uuid: str = Field(min_length=1)
    workload_kind: WorkloadKind
    descriptor: SemanticDescriptor
    resource_work: dict[str, dict[str, float]] = Field(default_factory=dict)
    fusion_regions: list[FusionRegion] = Field(default_factory=list)
    semantic_flops: float = Field(ge=0.0)
    semantic_bytes: float = Field(ge=0.0)
    t_sol_ms: float = Field(ge=0.0)
    source: EvidenceReference
    reason_codes: list[str] = Field(default_factory=list)


class ResourceFootprint(BaseModelWithDocstrings):
    """Compiled kernel resource footprint."""

    model_config = _MODEL_CONFIG

    vgpr_count: int | None = Field(default=None, ge=0)
    sgpr_count: int | None = Field(default=None, ge=0)
    lds_bytes: int | None = Field(default=None, ge=0)
    scratch_bytes: int | None = Field(default=None, ge=0)


class CompiledCharacterization(BaseModelWithDocstrings):
    """Static code-object and ISA characterization."""

    model_config = _MODEL_CONFIG

    candidate_sha256: SHA256Digest
    gpu_architecture: str
    kernel_symbol: str
    code_object_sha256: SHA256Digest | None = None
    functional_group_counts: dict[str, int] = Field(default_factory=dict)
    functional_subgroup_counts: dict[str, int] = Field(default_factory=dict)
    observed_matrix_units: list[str] = Field(default_factory=list)
    valu_types: list[str] = Field(default_factory=list)
    footprint: ResourceFootprint = Field(default_factory=ResourceFootprint)
    source: EvidenceReference
    reason_codes: list[str] = Field(default_factory=list)


class DispatchEvidence(BaseModelWithDocstrings):
    """Counter evidence for one runtime dispatch.

    Profiler duration is deliberately absent. Timestamps may establish overlap,
    but prediction code is prohibited from converting their difference into a
    runtime component.
    """

    model_config = _MODEL_CONFIG

    workload_uuid: str
    candidate_sha256: SHA256Digest
    dispatch_id: str
    correlation_id: str | None = None
    queue_id: str | None = None
    stream_id: str | None = None
    kernel_symbol: str
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    iteration_ordinal: int = Field(ge=0)
    replay_phase: Literal["evidence"] = "evidence"
    counter_passes: list[int] = Field(default_factory=list)
    counters: dict[str, float] = Field(default_factory=dict)
    runtime_footprint: ResourceFootprint | None = None
    start_timestamp_ns: int | None = Field(default=None, ge=0)
    end_timestamp_ns: int | None = Field(default=None, ge=0)
    valid: bool = True
    reason_codes: list[str] = Field(default_factory=list)
    evidence_conflicts: list[str] = Field(default_factory=list)
    sources: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> DispatchEvidence:
        """Reject reversed timestamp intervals."""
        if (
            self.start_timestamp_ns is not None
            and self.end_timestamp_ns is not None
            and self.end_timestamp_ns < self.start_timestamp_ns
        ):
            raise ValueError("end_timestamp_ns precedes start_timestamp_ns")
        if not self.valid and not self.reason_codes:
            raise ValueError("invalid dispatch evidence requires reason_codes")
        return self


class DispatchScheduleEdge(BaseModelWithDocstrings):
    """One timestamp-established precedence edge without measured duration."""

    model_config = _MODEL_CONFIG

    predecessor_dispatch_id: str = Field(min_length=1)
    successor_dispatch_id: str = Field(min_length=1)
    reason: Literal["same_lane", "happens_before"]


class PerformanceScheduleEvidence(CurrentSchemaModel):
    """Controlled-replay dispatch topology used by the overlap model."""

    model_config = _MODEL_CONFIG
    current_schema_version = SchemaVersion.PERFORMANCE_SCHEDULE_EVIDENCE

    schema_version: Literal[SchemaVersion.PERFORMANCE_SCHEDULE_EVIDENCE] = (
        SchemaVersion.PERFORMANCE_SCHEDULE_EVIDENCE
    )
    status: DiagnosticSidecarStatus
    workload_uuid: str = Field(min_length=1)
    candidate_sha256: SHA256Digest
    same_process: bool
    same_gpu: bool
    marker_contained: bool
    dispatch_ids: list[str] = Field(min_length=1)
    edges: list[DispatchScheduleEdge] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def topology_is_consistent(self) -> PerformanceScheduleEvidence:
        """Require unique dispatches and explicit unavailable reasons."""
        if len(self.dispatch_ids) != len(set(self.dispatch_ids)):
            raise ValueError("schedule evidence repeats dispatch ID")
        known = set(self.dispatch_ids)
        if any(
            edge.predecessor_dispatch_id not in known
            or edge.successor_dispatch_id not in known
            for edge in self.edges
        ):
            raise ValueError("schedule edge references unknown dispatch")
        scope = self.same_process and self.same_gpu and self.marker_contained
        if self.status is DiagnosticSidecarStatus.AVAILABLE and (
            not scope or self.reason_codes
        ):
            raise ValueError("available schedule requires verified scope")
        if self.status is not DiagnosticSidecarStatus.AVAILABLE and not (
            self.reason_codes
        ):
            raise ValueError("unavailable schedule requires reasons")
        return self


class CalibrationIdentity(BaseModelWithDocstrings):
    """Hardware and toolchain identity bound to a calibration profile."""

    model_config = _MODEL_CONFIG

    gpu_architecture: SupportedDiagnosticArchitecture
    gpu_id: str
    gpu_bdf: str
    pcie_topology: PCIeTopologyIdentity | None = None
    rocm_version: str
    compiler_version: str
    clock_mode: str
    power_profile: str

    @model_validator(mode="after")
    def _topology_terminates_at_gpu(self) -> CalibrationIdentity:
        if (
            self.pcie_topology is not None
            and self.pcie_topology.endpoint_bdf != self.gpu_bdf
        ):
            raise ValueError("PCIe topology does not terminate at gpu_bdf")
        return self


class CalibrationParameter(BaseModelWithDocstrings):
    """One calibrated value with uncertainty and applicability."""

    model_config = _MODEL_CONFIG

    name: CalibrationParameterName
    value: float = Field(gt=0.0)
    unit: CalibrationUnit
    confidence_interval: tuple[float, float]
    applicability: tuple[float, float] | None = None
    applicability_dimension: ApplicabilityDimension | None = None

    @model_validator(mode="after")
    def intervals_are_ordered(self) -> CalibrationParameter:
        """Reject invalid confidence and applicability intervals."""
        lower, upper = self.confidence_interval
        if lower <= 0 or upper < lower or not lower <= self.value <= upper:
            raise ValueError("confidence_interval must contain value")
        if self.applicability is not None:
            start, end = self.applicability
            if start < 0 or end < start:
                raise ValueError("invalid applicability interval")
            if self.applicability_dimension is None:
                raise ValueError(
                    "applicability requires applicability_dimension"
                )
        elif self.applicability_dimension is not None:
            raise ValueError("applicability_dimension requires applicability")
        return self


class CalibrationSurfaceCell(BaseModelWithDocstrings):
    """One bounded cell in a multidimensional calibration surface."""

    model_config = _MODEL_CONFIG

    coordinates: dict[
        ApplicabilityDimension,
        tuple[float, float],
    ] = Field(min_length=1)
    value: float = Field(gt=0.0)
    confidence_interval: tuple[float, float]

    @model_validator(mode="after")
    def intervals_are_ordered(self) -> CalibrationSurfaceCell:
        """Reject invalid coordinate or confidence intervals."""
        lower, upper = self.confidence_interval
        if lower <= 0 or upper < lower or not lower <= self.value <= upper:
            raise ValueError("surface confidence interval must contain value")
        if any(
            start < 0 or end < start for start, end in self.coordinates.values()
        ):
            raise ValueError("surface coordinate interval is invalid")
        return self

    def matches(
        self,
        coordinates: dict[ApplicabilityDimension, float],
    ) -> bool:
        """Return whether every governed coordinate lies in this cell."""
        return set(coordinates) == set(self.coordinates) and all(
            self.coordinates[dimension][0]
            <= value
            <= self.coordinates[dimension][1]
            for dimension, value in coordinates.items()
        )


class CalibrationSurface(BaseModelWithDocstrings):
    """Non-overlapping multidimensional empirical parameter cells."""

    model_config = _MODEL_CONFIG

    name: CalibrationSurfaceName
    unit: CalibrationUnit
    cells: list[CalibrationSurfaceCell] = Field(min_length=1)

    @model_validator(mode="after")
    def cells_do_not_overlap(self) -> CalibrationSurface:
        """Reject ambiguity across cells with the same coordinate axes."""
        if self.name is CalibrationSurfaceName.OVERLAP:
            for cell in self.cells:
                interval = cell.coordinates.get(
                    ApplicabilityDimension.RESOURCE_MIX
                )
                if interval is None or interval[0] != interval[1]:
                    raise ValueError(
                        "overlap resource_mix cells must be measured points"
                    )
        for index, left in enumerate(self.cells):
            for right in self.cells[index + 1 :]:
                if _surface_cells_overlap(left, right):
                    raise ValueError(
                        f"calibration surface overlaps: {self.name}"
                    )
        return self

    def cell(
        self,
        coordinates: dict[ApplicabilityDimension, float],
    ) -> CalibrationSurfaceCell | None:
        """Return the unique matching cell without interpolation."""
        matches = [cell for cell in self.cells if cell.matches(coordinates)]
        if len(matches) > 1:
            raise ValueError(f"ambiguous calibration surface {self.name}")
        return matches[0] if matches else None


def _surface_cells_overlap(
    left: CalibrationSurfaceCell,
    right: CalibrationSurfaceCell,
) -> bool:
    if set(left.coordinates) != set(right.coordinates):
        return False
    return all(
        max(left.coordinates[dimension][0], right.coordinates[dimension][0])
        <= min(left.coordinates[dimension][1], right.coordinates[dimension][1])
        for dimension in left.coordinates
    )


class DiagnosticCalibrationProfile(CurrentSchemaModel):
    """Content-addressed gfx1200 diagnostic calibration."""

    model_config = _MODEL_CONFIG
    current_schema_version = SchemaVersion.DIAGNOSTIC_CALIBRATION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_CALIBRATION] = (
        SchemaVersion.DIAGNOSTIC_CALIBRATION
    )
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    model_version: DiagnosticModelVersion = PERFORMANCE_MODEL_VERSION
    identity: CalibrationIdentity
    parameters: list[CalibrationParameter] = Field(min_length=1)
    surfaces: list[CalibrationSurface] = Field(default_factory=list)
    tuning_evidence_sha256: list[SHA256Digest] = Field(min_length=1)
    parameter_estimation_evidence_sha256: list[SHA256Digest] = Field(
        min_length=1
    )
    probe_evidence_sha256: list[SHA256Digest] = Field(min_length=1)
    configuration_frozen_before_estimation: Literal[True] = True
    bootstrap_seed: int = Field(ge=0)
    bootstrap_replicates: int = Field(ge=1)

    def parameter(
        self,
        name: CalibrationParameterName,
        coordinate: float | None = None,
    ) -> CalibrationParameter | None:
        """Return one named parameter without inventing a fallback."""
        matches = [
            parameter
            for parameter in self.parameters
            if parameter.name == name
            and (
                coordinate is None
                or parameter.applicability is None
                or (
                    parameter.applicability[0]
                    <= coordinate
                    <= parameter.applicability[1]
                )
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous calibration parameter {name} at {coordinate}"
            )
        return matches[0] if matches else None

    def surface(
        self,
        name: CalibrationSurfaceName,
    ) -> CalibrationSurface | None:
        """Return one uniquely named calibration surface."""
        matches = [surface for surface in self.surfaces if surface.name is name]
        if len(matches) > 1:
            raise ValueError(f"duplicate calibration surface {name}")
        return matches[0] if matches else None

    @model_validator(mode="after")
    def parameters_are_unambiguous(self) -> DiagnosticCalibrationProfile:
        """Reject duplicate or overlapping parameter applicability."""
        if set(self.tuning_evidence_sha256) & set(
            self.parameter_estimation_evidence_sha256
        ):
            raise ValueError(
                "tuning and parameter-estimation evidence must be disjoint"
            )
        for parameter in self.parameters:
            expected_unit = _CALIBRATION_PARAMETER_UNITS[parameter.name]
            if parameter.unit is not expected_unit:
                raise ValueError(
                    f"{parameter.name} requires unit {expected_unit}"
                )
            expected_dimension = _CALIBRATION_PARAMETER_DIMENSIONS.get(
                parameter.name
            )
            if parameter.applicability_dimension is not expected_dimension:
                raise ValueError(
                    f"{parameter.name} requires applicability dimension "
                    f"{expected_dimension}"
                )
        for index, left in enumerate(self.parameters):
            for right in self.parameters[index + 1 :]:
                if left.name is right.name and _parameter_intervals_overlap(
                    left,
                    right,
                ):
                    raise ValueError(
                        f"calibration parameter applicability overlaps: "
                        f"{left.name}"
                    )
        return self


def _parameter_intervals_overlap(
    left: CalibrationParameter,
    right: CalibrationParameter,
) -> bool:
    if left.applicability_dimension is not right.applicability_dimension:
        return False
    if left.applicability is None or right.applicability is None:
        return True
    return max(left.applicability[0], right.applicability[0]) <= min(
        left.applicability[1], right.applicability[1]
    )


_CALIBRATION_PARAMETER_UNITS = {
    CalibrationParameterName.DISPATCH_FLOOR_MS: CalibrationUnit.MS,
    CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.VALU_SIMPLE_BF16_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.VALU_TRANSCENDENTAL_FP32_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.VALU_TRANSCENDENTAL_BF16_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.VALU_COMPOSITE_FP32_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.VALU_COMPOSITE_BF16_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS: (
        CalibrationUnit.FLOP_PER_MS
    ),
    CalibrationParameterName.L2_BYTE_PER_MS: CalibrationUnit.BYTE_PER_MS,
    CalibrationParameterName.L3_BYTE_PER_MS: CalibrationUnit.BYTE_PER_MS,
    CalibrationParameterName.VRAM_BYTE_PER_MS: CalibrationUnit.BYTE_PER_MS,
    CalibrationParameterName.TRANSPOSE_EFFICIENCY: CalibrationUnit.RATIO,
    CalibrationParameterName.LDS_BYTE_PER_MS: CalibrationUnit.BYTE_PER_MS,
    CalibrationParameterName.LDS_BANK_CONFLICT_PENALTY_MS: (
        CalibrationUnit.MS_PER_EVENT
    ),
    CalibrationParameterName.REDUCTION_OP_PER_MS: (CalibrationUnit.ITEM_PER_MS),
    CalibrationParameterName.BARRIER_PENALTY_MS: (CalibrationUnit.MS_PER_EVENT),
    CalibrationParameterName.EDGE_WMMA_EFFICIENCY: CalibrationUnit.RATIO,
    CalibrationParameterName.IRREGULAR_WMMA_EFFICIENCY: (CalibrationUnit.RATIO),
    CalibrationParameterName.FP32_MATRIX_FLOP_PER_MS: (
        CalibrationUnit.FLOP_PER_MS
    ),
    CalibrationParameterName.SOFTMAX_REDUCTION_OP_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS: (
        CalibrationUnit.ITEM_PER_MS
    ),
    CalibrationParameterName.STRIDED_MATMUL_EFFICIENCY: (CalibrationUnit.RATIO),
}

_CALIBRATION_PARAMETER_DIMENSIONS = {
    CalibrationParameterName.L2_BYTE_PER_MS: (
        ApplicabilityDimension.WORKING_SET_BYTES
    ),
    CalibrationParameterName.L3_BYTE_PER_MS: (
        ApplicabilityDimension.WORKING_SET_BYTES
    ),
    CalibrationParameterName.VRAM_BYTE_PER_MS: (
        ApplicabilityDimension.WORKING_SET_BYTES
    ),
    CalibrationParameterName.REDUCTION_OP_PER_MS: (
        ApplicabilityDimension.REDUCTION_WIDTH
    ),
    CalibrationParameterName.BARRIER_PENALTY_MS: (
        ApplicabilityDimension.REDUCTION_WIDTH
    ),
    CalibrationParameterName.EDGE_WMMA_EFFICIENCY: (
        ApplicabilityDimension.TILE_REMAINDER
    ),
    CalibrationParameterName.IRREGULAR_WMMA_EFFICIENCY: (
        ApplicabilityDimension.TILE_REMAINDER
    ),
    CalibrationParameterName.SOFTMAX_REDUCTION_OP_PER_MS: (
        ApplicabilityDimension.REDUCTION_WIDTH
    ),
}


class PredictionComponent(BaseModelWithDocstrings):
    """One resource-time component."""

    model_config = _MODEL_CONFIG

    name: str
    time_ms: float = Field(ge=0.0)
    lower_ms: float = Field(ge=0.0)
    upper_ms: float = Field(ge=0.0)
    dispatch_id: str | None = None

    @model_validator(mode="after")
    def interval_contains_time(self) -> PredictionComponent:
        """Require the component interval to contain its estimate."""
        if not self.lower_ms <= self.time_ms <= self.upper_ms:
            raise ValueError("component interval does not contain time_ms")
        return self


class PerformancePrediction(BaseModelWithDocstrings):
    """Diagnostic prediction that cannot carry measured duration."""

    model_config = _MODEL_CONFIG

    kind: PredictionKind
    status: DiagnosticSidecarStatus
    predicted_time_ms: float | None = Field(default=None, ge=0.0)
    lower_ms: float | None = Field(default=None, ge=0.0)
    upper_ms: float | None = Field(default=None, ge=0.0)
    components: list[PredictionComponent] = Field(default_factory=list)
    model_version: DiagnosticModelVersion = PERFORMANCE_MODEL_VERSION
    reason_codes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_estimate(self) -> PerformancePrediction:
        """Keep availability and estimate fields internally consistent."""
        values = (self.predicted_time_ms, self.lower_ms, self.upper_ms)
        if self.status is DiagnosticSidecarStatus.UNAVAILABLE:
            if any(value is not None for value in values):
                raise ValueError("unavailable prediction cannot contain timing")
            return self
        if any(value is None for value in values):
            raise ValueError("available or partial prediction requires timing")
        predicted = self.predicted_time_ms
        lower = self.lower_ms
        upper = self.upper_ms
        if predicted is None or lower is None or upper is None:
            raise ValueError("available or partial prediction requires timing")
        if not lower <= predicted <= upper:
            raise ValueError("prediction interval does not contain estimate")
        return self


class DiagnosticRatio(BaseModelWithDocstrings):
    """One uncertainty-aware L/C/R ratio."""

    model_config = _MODEL_CONFIG

    kind: RatioKind
    status: DiagnosticSidecarStatus
    value: float | None = Field(default=None, ge=0.0)
    lower: float | None = Field(default=None, ge=0.0)
    upper: float | None = Field(default=None, ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_ratio(self) -> DiagnosticRatio:
        """Keep availability and ratio interval internally consistent."""
        values = (self.value, self.lower, self.upper)
        if self.status is DiagnosticSidecarStatus.UNAVAILABLE:
            if any(value is not None for value in values):
                raise ValueError("unavailable ratio cannot contain a value")
            return self
        if any(value is None for value in values):
            raise ValueError("available ratio requires value and interval")
        value = self.value
        lower = self.lower
        upper = self.upper
        if value is None or lower is None or upper is None:
            raise ValueError("available ratio requires value and interval")
        if not lower <= value <= upper:
            raise ValueError("ratio interval does not contain value")
        return self


class PerformanceAttribution(BaseModelWithDocstrings):
    """One bounded attribution or action recommendation."""

    model_config = _MODEL_CONFIG

    code: str
    category: str
    confidence: DiagnosticConfidence
    message: str
    action_code: str | None = None
    evidence: list[str] = Field(default_factory=list)


class WorkloadPerformanceDiagnostic(BaseModelWithDocstrings):
    """Complete diagnostic result for one workload UUID."""

    model_config = _MODEL_CONFIG

    workload_uuid: str
    semantic: SemanticCharacterization
    compiled: list[CompiledCharacterization] = Field(default_factory=list)
    dispatches: list[DispatchEvidence] = Field(default_factory=list)
    schedule: PerformanceScheduleEvidence | None = None
    t_pred_ir: PerformancePrediction
    t_pred_hw: PerformancePrediction
    t_measured_ms: float = Field(ge=0.0)
    t_measured_lower_ms: float = Field(ge=0.0)
    t_measured_upper_ms: float = Field(ge=0.0)
    t_frontier_ms: float | None = Field(default=None, ge=0.0)
    ratios: list[DiagnosticRatio]
    attributions: list[PerformanceAttribution] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def workload_identity_is_consistent(self) -> WorkloadPerformanceDiagnostic:
        """Keep semantic and dispatch evidence bound to this workload."""
        if self.semantic.workload_uuid != self.workload_uuid or any(
            dispatch.workload_uuid != self.workload_uuid
            for dispatch in self.dispatches
        ):
            raise ValueError("workload diagnostic identity mismatch")
        if not (
            self.t_measured_lower_ms
            <= self.t_measured_ms
            <= self.t_measured_upper_ms
        ):
            raise ValueError("measured interval does not contain timing")
        return self


class PerformanceDiagnosticSidecar(CurrentDiagnosticSidecarAuthority):
    """Diagnostic-only microarchitecture sidecar."""

    model_config = _MODEL_CONFIG
    current_schema_version = SchemaVersion.PERFORMANCE_DIAGNOSTIC

    schema_version: Literal[SchemaVersion.PERFORMANCE_DIAGNOSTIC] = (
        SchemaVersion.PERFORMANCE_DIAGNOSTIC
    )
    status: DiagnosticSidecarStatus
    model_version: DiagnosticModelVersion = PERFORMANCE_MODEL_VERSION
    model_identity: DiagnosticModelIdentity
    inference_profile_sha256: SHA256Digest | None = None
    run_id: str
    candidate_sha256: SHA256Digest
    gpu_architecture: str
    calibration_identity: CalibrationIdentity | None = None
    workloads: list[WorkloadPerformanceDiagnostic] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible sidecar data."""
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def evidence_identity_is_consistent(self) -> PerformanceDiagnosticSidecar:
        """Reject candidate, GPU, calibration, and workload contradictions."""
        workload_ids = [workload.workload_uuid for workload in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("performance diagnostic repeats workload UUID")
        if self.model_identity.model_version != self.model_version or (
            self.calibration_identity is not None
            and self.calibration_identity.gpu_architecture
            != self.gpu_architecture
        ):
            raise ValueError("performance diagnostic authority mismatch")
        for workload in self.workloads:
            if any(
                compiled.candidate_sha256 != self.candidate_sha256
                or compiled.gpu_architecture != self.gpu_architecture
                for compiled in workload.compiled
            ) or any(
                dispatch.candidate_sha256 != self.candidate_sha256
                for dispatch in workload.dispatches
            ):
                raise ValueError(
                    "performance diagnostic evidence identity mismatch"
                )
        return self


__all__ = [
    "CDNA3_PERFORMANCE_MODEL_VERSION",
    "PERFORMANCE_MODEL_VERSION",
    "ApplicabilityDimension",
    "CalibrationIdentity",
    "CalibrationParameter",
    "CalibrationParameterName",
    "CalibrationSurface",
    "CalibrationSurfaceCell",
    "CalibrationSurfaceName",
    "CalibrationUnit",
    "CompiledCharacterization",
    "CompositeGraphDescriptor",
    "CompositeGraphEdge",
    "CompositeGraphNode",
    "CrossEntropyDescriptor",
    "CrossEntropyReduction",
    "DiagnosticCalibrationProfile",
    "DiagnosticModelIdentity",
    "DiagnosticModelVersion",
    "DiagnosticRatio",
    "DispatchEvidence",
    "DispatchScheduleEdge",
    "ElementwiseDescriptor",
    "ElementwiseOperationClass",
    "EvidenceReference",
    "FusionRegion",
    "IndexedReadDescriptor",
    "IndexedReadOperation",
    "IndexedUpdateDescriptor",
    "IndexedUpdateOperation",
    "MatmulDescriptor",
    "PerformanceAttribution",
    "PerformanceDiagnosticSidecar",
    "PerformancePrediction",
    "PerformanceScheduleEvidence",
    "PredictionComponent",
    "PredictionKind",
    "RatioKind",
    "ReductionDescriptor",
    "ReductionOperation",
    "ResourceFootprint",
    "SemanticCharacterization",
    "SemanticDescriptor",
    "SoftmaxDescriptor",
    "SoftmaxOperation",
    "SupportedDiagnosticArchitecture",
    "TensorDType",
    "TransposeDescriptor",
    "UnsupportedDescriptor",
    "WorkloadKind",
    "WorkloadPerformanceDiagnostic",
]
