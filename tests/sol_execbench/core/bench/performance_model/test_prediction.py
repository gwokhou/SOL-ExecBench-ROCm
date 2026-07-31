from __future__ import annotations

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model import prediction
from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.models import (
    ApplicabilityDimension,
    CalibrationIdentity,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationUnit,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    EvidenceReference,
    FusionRegion,
    MatmulDescriptor,
    RatioKind,
    SemanticCharacterization,
    TensorDType,
    TransposeDescriptor,
    WorkloadKind,
)
from sol_execbench.core.bench.performance_model.prediction import (
    predict_hw,
    predict_ir,
    validate_calibration_identity,
)


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
    parameters = [
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
            CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS,
            10_000.0,
            CalibrationUnit.FLOP_PER_MS,
        ),
        _parameter(
            CalibrationParameterName.L2_BYTE_PER_MS,
            2_000.0,
            CalibrationUnit.BYTE_PER_MS,
            applicability=(0.0, 2**20),
            dimension=ApplicabilityDimension.WORKING_SET_BYTES,
        ),
        _parameter(
            CalibrationParameterName.TRANSPOSE_EFFICIENCY,
            0.5,
            CalibrationUnit.RATIO,
        ),
        _parameter(
            CalibrationParameterName.EDGE_WMMA_EFFICIENCY,
            0.25,
            CalibrationUnit.RATIO,
            applicability=(1.0, 15.0),
            dimension=ApplicabilityDimension.TILE_REMAINDER,
        ),
    ]
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
        parameters=parameters,
        tuning_evidence_sha256=["a" * 64],
        parameter_estimation_evidence_sha256=["b" * 64],
        probe_evidence_sha256=["c" * 64],
        bootstrap_seed=20_260_729,
        bootstrap_replicates=10_000,
    )


def test_required_parameter_interpolates_adjacent_calibrated_points() -> None:
    calibration = _calibration().model_copy(
        update={
            "parameters": [
                *_calibration().parameters,
                _parameter(
                    CalibrationParameterName.REDUCTION_OP_PER_MS,
                    320.0,
                    CalibrationUnit.ITEM_PER_MS,
                    applicability=(32.0, 32.0),
                    dimension=ApplicabilityDimension.REDUCTION_WIDTH,
                ),
                _parameter(
                    CalibrationParameterName.REDUCTION_OP_PER_MS,
                    640.0,
                    CalibrationUnit.ITEM_PER_MS,
                    applicability=(64.0, 64.0),
                    dimension=ApplicabilityDimension.REDUCTION_WIDTH,
                ),
            ],
        },
    )

    result = prediction._required(
        calibration,
        CalibrationParameterName.REDUCTION_OP_PER_MS,
        coordinate=48.0,
    )

    assert result.value == pytest.approx(480.0)
    assert result.applicability == (48.0, 48.0)


def _semantic(
    kind: WorkloadKind = WorkloadKind.ELEMENTWISE,
) -> SemanticCharacterization:
    descriptor = (
        ElementwiseDescriptor(
            shape=[32],
            dtype=TensorDType.FLOAT32,
            operations={ElementwiseOperationClass.SIMPLE: 32.0},
        )
        if kind is WorkloadKind.ELEMENTWISE
        else TransposeDescriptor(
            rows=4,
            columns=8,
            dtype=TensorDType.FLOAT32,
            element_bytes=4,
            input_strides=(8, 1),
            output_strides=(4, 1),
        )
    )
    return SemanticCharacterization(
        workload_uuid="workload-1",
        workload_kind=kind,
        descriptor=descriptor,
        resource_work={"valu": {"fp32": 32}},
        fusion_regions=[FusionRegion(region_id="region-0")],
        semantic_flops=32,
        semantic_bytes=128,
        t_sol_ms=0.001,
        source=EvidenceReference(kind="solar_analysis", sha256="d" * 64),
    )


def test_predictions_are_deterministic_and_exclude_measured_duration() -> None:
    semantic = _semantic()
    calibration = _calibration()
    dispatch = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel",
        grid=(32, 1, 1),
        workgroup=(32, 1, 1),
        iteration_ordinal=0,
        counter_passes=[1],
        counters={
            "SQ_INSTS_VALU": 32,
            "FETCH_SIZE": 128,
            "SQ_WAVES": 1,
        },
    )

    ir = predict_ir(semantic, calibration)
    hw = predict_hw(semantic, [], [dispatch], calibration)

    assert ir.status is DiagnosticSidecarStatus.AVAILABLE
    assert hw.status is DiagnosticSidecarStatus.AVAILABLE
    assert ir.predicted_time_ms == pytest.approx(0.074)
    assert hw.predicted_time_ms == pytest.approx(1.034)


def test_hw_prediction_matches_static_mangled_symbol_and_warm_gl2_traffic() -> (
    None
):
    semantic = _semantic()
    calibration = _calibration()
    source = EvidenceReference(kind="static_evidence", sha256="e" * 64)
    compiled = CompiledCharacterization(
        candidate_sha256="c" * 64,
        gpu_architecture="gfx1200",
        kernel_symbol="_Z6kernelv",
        functional_group_counts={"Vector ALU": 2},
        source=source,
    )
    dispatch = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel()",
        grid=(32, 1, 1),
        workgroup=(32, 1, 1),
        iteration_ordinal=0,
        counter_passes=[1],
        counters={
            "FETCH_SIZE": 0,
            "GL2C_HIT_SUM": 1,
            "GL2C_MISS_SUM": 1,
            "SQ_WAVES_SUM": 1,
        },
    )
    runtime_helper = dispatch.model_copy(
        update={
            "dispatch_id": "2",
            "kernel_symbol": "__amd_rocclr_copyBuffer",
            "valid": False,
            "reason_codes": ["dispatch_static_kernel_identity_mismatch"],
        },
    )

    prediction = predict_hw(
        semantic,
        [compiled],
        [dispatch, runtime_helper],
        calibration,
    )

    assert prediction.status is DiagnosticSidecarStatus.AVAILABLE
    assert {component.name for component in prediction.components} == {
        "compute",
        "dispatch",
        "memory",
    }
    assert prediction.reason_codes == []


def test_ir_applies_transpose_efficiency() -> None:
    prediction = predict_ir(_semantic(WorkloadKind.TRANSPOSE), _calibration())

    assert prediction.predicted_time_ms == pytest.approx(0.138)


def test_ir_applies_edge_wmma_efficiency() -> None:
    semantic = _semantic().model_copy(
        update={
            "workload_kind": WorkloadKind.MATMUL,
            "descriptor": MatmulDescriptor(
                m=17,
                n=16,
                k=16,
                leading_dimension_a=16,
                leading_dimension_b=16,
                leading_dimension_c=16,
            ),
            "semantic_flops": 320.0,
        }
    )

    prediction = predict_ir(semantic, _calibration())

    assert prediction.predicted_time_ms == pytest.approx(0.138)


def test_calibration_identity_requires_complete_evidence() -> None:
    assert validate_calibration_identity(
        _calibration(),
        gpu_architecture="gfx1200",
        rocm_version="7.2",
        clock_mode="locked",
    ) == [
        "calibration_gpu_id_unverified",
        "calibration_gpu_bdf_unverified",
        "calibration_compiler_version_unverified",
        "calibration_power_profile_unverified",
    ]


def test_frontier_is_unavailable_and_unverified_r_only_reprofiles() -> None:
    semantic = _semantic().model_copy(
        update={
            "descriptor": ElementwiseDescriptor(
                shape=[1],
                dtype=TensorDType.FLOAT32,
                operations={ElementwiseOperationClass.SIMPLE: 32.0},
            ),
            "semantic_flops": 32.0,
            "semantic_bytes": 32.0,
        },
    )
    calibration = _calibration()
    dispatch = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel",
        grid=(1, 1, 1),
        workgroup=(1, 1, 1),
        iteration_ordinal=0,
        counters={"SQ_INSTS_VALU": 1, "SQ_WAVES": 1},
    )
    ir = predict_ir(semantic, calibration)
    hw = predict_hw(semantic, [], [dispatch], calibration)
    ratios = calculate_ratios(
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=1.0,
        t_measured_lower_ms=0.99,
        t_measured_upper_ms=1.01,
        t_sol_ms=semantic.t_sol_ms,
        t_frontier_ms=None,
    )

    assert ratios[0].kind is RatioKind.L
    assert ratios[0].status is DiagnosticSidecarStatus.UNAVAILABLE
    actions = derive_attributions(
        semantic=semantic,
        compiled=[],
        dispatches=[dispatch],
        t_pred_ir=ir,
        t_pred_hw=hw,
        ratios=ratios,
    )
    assert "reprofile_missing_counters" in {
        action.action_code for action in actions
    }


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"queue_id": None}, "dispatch_queue_identity_unverified"),
        (
            {
                "queue_id": "0",
                "start_timestamp_ns": 5,
                "end_timestamp_ns": 20,
            },
            "same_lane_dispatch_overlap",
        ),
    ],
)
def test_hardware_prediction_rejects_unverified_concurrency(
    updates: dict[str, object],
    reason: str,
) -> None:
    first = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel",
        grid=(32, 1, 1),
        workgroup=(32, 1, 1),
        iteration_ordinal=0,
        counters={"SQ_INSTS_VALU": 1, "SQ_WAVES": 1},
        start_timestamp_ns=0,
        end_timestamp_ns=10,
    )
    second = first.model_copy(
        update={
            "dispatch_id": "2",
            "iteration_ordinal": 1,
            "start_timestamp_ns": 10,
            "end_timestamp_ns": 20,
            **updates,
        }
    )

    prediction = predict_hw(
        _semantic(),
        [],
        [first, second],
        _calibration(),
    )

    assert prediction.status is DiagnosticSidecarStatus.UNAVAILABLE
    assert prediction.reason_codes == [reason]


def test_hardware_prediction_accepts_ordered_cross_queue_schedule() -> None:
    first = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        queue_id="0",
        kernel_symbol="kernel",
        grid=(32, 1, 1),
        workgroup=(32, 1, 1),
        iteration_ordinal=0,
        counters={"SQ_INSTS_VALU": 1, "SQ_WAVES": 1},
        start_timestamp_ns=0,
        end_timestamp_ns=10,
    )
    second = first.model_copy(
        update={
            "dispatch_id": "2",
            "queue_id": "1",
            "iteration_ordinal": 1,
            "start_timestamp_ns": 10,
            "end_timestamp_ns": 20,
        }
    )

    prediction = predict_hw(
        _semantic(),
        [],
        [first, second],
        _calibration(),
    )

    assert prediction.status is not DiagnosticSidecarStatus.UNAVAILABLE
