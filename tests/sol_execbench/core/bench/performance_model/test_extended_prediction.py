from __future__ import annotations

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.bench.performance_model.models import (
    ApplicabilityDimension,
    CalibrationIdentity,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationSurface,
    CalibrationSurfaceCell,
    CalibrationSurfaceName,
    CalibrationUnit,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    EvidenceReference,
    FusionRegion,
    IndexedUpdateDescriptor,
    IndexedUpdateOperation,
    SemanticCharacterization,
    SemanticDescriptor,
    SoftmaxDescriptor,
    SoftmaxOperation,
    TensorDType,
    WorkloadKind,
)
from sol_execbench.core.bench.performance_model.prediction import (
    predict_hw,
    predict_ir,
)
from sol_execbench.core.data.definition_models import DType


def _parameter(
    name: CalibrationParameterName,
    value: float,
    unit: CalibrationUnit,
    *,
    applicability: tuple[float, float] | None = None,
    dimension: ApplicabilityDimension | None = None,
) -> CalibrationParameter:
    return CalibrationParameter(
        name=name,
        value=value,
        unit=unit,
        confidence_interval=(value * 0.9, value * 1.1),
        applicability=applicability,
        applicability_dimension=dimension,
    )


def _calibration() -> DiagnosticCalibrationProfile:
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id="gpu-0",
            gpu_bdf="0000:03:00.0",
            rocm_version="7.2",
            compiler_version="hipcc-7.2",
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=[
            _parameter(
                CalibrationParameterName.DISPATCH_FLOOR_MS,
                0.01,
                CalibrationUnit.MS,
            ),
            _parameter(
                CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
                1_000.0,
                CalibrationUnit.ITEM_PER_MS,
            ),
            _parameter(
                CalibrationParameterName.VALU_TRANSCENDENTAL_FP32_PER_MS,
                500.0,
                CalibrationUnit.ITEM_PER_MS,
            ),
            _parameter(
                CalibrationParameterName.SOFTMAX_REDUCTION_OP_PER_MS,
                800.0,
                CalibrationUnit.ITEM_PER_MS,
                applicability=(1.0, 4096.0),
                dimension=ApplicabilityDimension.REDUCTION_WIDTH,
            ),
            _parameter(
                CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS,
                1_000.0,
                CalibrationUnit.ITEM_PER_MS,
            ),
            _parameter(
                CalibrationParameterName.L2_BYTE_PER_MS,
                2_000.0,
                CalibrationUnit.BYTE_PER_MS,
                applicability=(0.0, 2**24),
                dimension=ApplicabilityDimension.WORKING_SET_BYTES,
            ),
        ],
        surfaces=[
            CalibrationSurface(
                name=CalibrationSurfaceName.ATOMIC_UPDATE,
                unit=CalibrationUnit.ITEM_PER_MS,
                cells=[
                    CalibrationSurfaceCell(
                        coordinates={
                            ApplicabilityDimension.COLLISION_FRACTION: (
                                0.0,
                                1.0,
                            ),
                            ApplicabilityDimension.MAX_MULTIPLICITY: (
                                1.0,
                                1024.0,
                            ),
                            ApplicabilityDimension.ELEMENT_BYTES: (4.0, 4.0),
                        },
                        value=400.0,
                        confidence_interval=(360.0, 440.0),
                    )
                ],
            ),
            CalibrationSurface(
                name=CalibrationSurfaceName.OVERLAP,
                unit=CalibrationUnit.RATIO,
                cells=[
                    CalibrationSurfaceCell(
                        coordinates={
                            ApplicabilityDimension.RESOURCE_MIX: (0.0, 0.0),
                            ApplicabilityDimension.CONCURRENT_DISPATCHES: (
                                2.0,
                                2.0,
                            ),
                        },
                        value=1.0,
                        confidence_interval=(0.9, 1.0),
                    ),
                    CalibrationSurfaceCell(
                        coordinates={
                            ApplicabilityDimension.RESOURCE_MIX: (1.0, 1.0),
                            ApplicabilityDimension.CONCURRENT_DISPATCHES: (
                                2.0,
                                2.0,
                            ),
                        },
                        value=0.8,
                        confidence_interval=(0.7, 0.9),
                    ),
                ],
            ),
        ],
        tuning_evidence_sha256=["a" * 64],
        parameter_estimation_evidence_sha256=["b" * 64],
        probe_evidence_sha256=["c" * 64],
        bootstrap_seed=20_260_729,
        bootstrap_replicates=10_000,
    )


def _semantic(
    descriptor: SemanticDescriptor,
    kind: WorkloadKind,
    *,
    semantic_flops: float,
    semantic_bytes: float,
) -> SemanticCharacterization:
    return SemanticCharacterization(
        workload_uuid="workload-1",
        workload_kind=kind,
        descriptor=descriptor,
        fusion_regions=[FusionRegion(region_id="region-0")],
        semantic_flops=semantic_flops,
        semantic_bytes=semantic_bytes,
        t_sol_ms=0.001,
        source=EvidenceReference(kind="solar_analysis", sha256="d" * 64),
    )


def _access_pattern() -> AccessPatternSummary:
    return AccessPatternSummary(
        input_name="indices",
        dtype=DType.INT64,
        element_count=8,
        sampled_element_count=8,
        exact=True,
        minimum_index=0,
        maximum_index=3,
        unique_index_count=4,
        duplicate_fraction=0.5,
        maximum_multiplicity=3,
        multiplicity_histogram={
            "1": 1,
            "2-3": 3,
            "4-7": 0,
            "8-15": 0,
            "16-31": 0,
            "32+": 0,
        },
        adjacent_same_fraction=0.25,
        adjacent_unit_stride_fraction=0.5,
    )


def test_softmax_and_atomic_update_have_registered_predictors() -> None:
    calibration = _calibration()
    softmax = _semantic(
        SoftmaxDescriptor(
            operation=SoftmaxOperation.SOFTMAX,
            outer_rows=4,
            reduction_width=8,
            input_dtype=TensorDType.FLOAT32,
            output_dtype=TensorDType.FLOAT32,
        ),
        WorkloadKind.SOFTMAX,
        semantic_flops=160,
        semantic_bytes=256,
    )
    atomic = _semantic(
        IndexedUpdateDescriptor(
            operation=IndexedUpdateOperation.SCATTER_ADD,
            output_shape=[16],
            index_shape=[8],
            axis=0,
            payload_dtype=TensorDType.FLOAT32,
            index_dtype=DType.INT64,
            atomic=True,
        ),
        WorkloadKind.INDEXED_UPDATE,
        semantic_flops=8,
        semantic_bytes=96,
    )

    softmax_prediction = predict_ir(softmax, calibration)
    atomic_prediction = predict_ir(
        atomic,
        calibration,
        access_patterns=[_access_pattern()],
    )

    assert softmax_prediction.status is DiagnosticSidecarStatus.AVAILABLE
    assert atomic_prediction.status is DiagnosticSidecarStatus.AVAILABLE
    assert "atomic_update" in {
        component.name for component in atomic_prediction.components
    }


def test_overlap_timestamps_change_topology_but_not_dispatch_durations() -> (
    None
):
    semantic = _semantic(
        ElementwiseDescriptor(
            shape=[32],
            dtype=TensorDType.FLOAT32,
            operations={ElementwiseOperationClass.SIMPLE: 32.0},
        ),
        WorkloadKind.CONCURRENT,
        semantic_flops=32,
        semantic_bytes=128,
    )
    first = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel",
        grid=(32, 1, 1),
        workgroup=(32, 1, 1),
        iteration_ordinal=0,
        counters={"SQ_INSTS_VALU": 1, "SQ_WAVES": 1, "FETCH_SIZE": 64},
        start_timestamp_ns=0,
        end_timestamp_ns=10,
    )
    overlapping = first.model_copy(
        update={
            "dispatch_id": "2",
            "queue_id": "1",
            "iteration_ordinal": 1,
            "start_timestamp_ns": 5,
            "end_timestamp_ns": 20,
        }
    )
    stretched = overlapping.model_copy(
        update={"start_timestamp_ns": 1, "end_timestamp_ns": 1_000_000}
    )

    first_prediction = predict_hw(
        semantic,
        [],
        [first, overlapping],
        _calibration(),
    )
    stretched_prediction = predict_hw(
        semantic,
        [],
        [first, stretched],
        _calibration(),
    )

    assert first_prediction.predicted_time_ms == pytest.approx(
        stretched_prediction.predicted_time_ms
    )
