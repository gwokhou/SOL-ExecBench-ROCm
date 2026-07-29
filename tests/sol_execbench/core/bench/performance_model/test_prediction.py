from __future__ import annotations

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    CalibrationParameter,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    EvidenceReference,
    FusionRegion,
    RatioKind,
    SemanticCharacterization,
    WorkloadKind,
)
from sol_execbench.core.bench.performance_model.prediction import (
    predict_hw,
    predict_ir,
    validate_calibration_identity,
)


def _calibration() -> DiagnosticCalibrationProfile:
    values = {
        "dispatch_floor_ms": 0.01,
        "valu_flop_per_ms": 1_000.0,
        "wmma_flop_per_ms": 10_000.0,
        "vram_byte_per_ms": 2_000.0,
        "lds_byte_per_ms": 4_000.0,
        "reduction_op_per_ms": 500.0,
        "transpose_access_efficiency": 0.5,
        "edge_wmma_efficiency": 0.25,
    }
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id="gpu-0",
            rocm_version="7.2",
            compiler_version="hipcc-7.2",
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=[
            CalibrationParameter(
                name=name,
                value=value,
                unit="ms" if name == "dispatch_floor_ms" else "item/ms",
                confidence_interval=(value * 0.9, value * 1.1),
            )
            for name, value in values.items()
        ],
        probe_evidence_sha256=["a" * 64],
        held_out_evidence_sha256=["b" * 64],
    )


def _semantic(kind: WorkloadKind = WorkloadKind.ELEMENTWISE):
    return SemanticCharacterization(
        workload_uuid="workload-1",
        workload_kind=kind,
        shape=[32],
        resource_work={"valu": {"fp32": 32}},
        fusion_regions=[FusionRegion(region_id="region-0")],
        semantic_flops=32,
        semantic_bytes=128,
        t_sol_ms=0.001,
        source=EvidenceReference(
            kind="solar_analysis",
            sha256="a" * 64,
        ),
    )


def test_predictions_are_deterministic_and_do_not_use_measured_duration() -> (
    None
):
    semantic = _semantic()
    calibration = _calibration()
    dispatch = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
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
    hw = predict_hw([], [dispatch], calibration)

    assert ir.status is DiagnosticSidecarStatus.AVAILABLE
    assert hw.status is DiagnosticSidecarStatus.AVAILABLE
    assert ir.predicted_time_ms == pytest.approx(0.074)
    assert hw.predicted_time_ms == pytest.approx(2.058)


@pytest.mark.parametrize(
    ("kind", "shape", "flops", "expected_ms"),
    [
        (WorkloadKind.TRANSPOSE, [32], 32, 0.138),
        (WorkloadKind.MATMUL, [17, 16, 16], 320, 0.138),
    ],
)
def test_ir_applies_workload_efficiency(
    kind: WorkloadKind,
    shape: list[int],
    flops: float,
    expected_ms: float,
) -> None:
    semantic = _semantic(kind).model_copy(
        update={"shape": shape, "semantic_flops": flops},
    )

    prediction = predict_ir(semantic, _calibration())

    assert prediction.predicted_time_ms == pytest.approx(expected_ms)


def test_calibration_identity_requires_complete_evidence() -> None:
    assert validate_calibration_identity(
        _calibration(),
        gpu_architecture="gfx1200",
        rocm_version="7.2",
        clock_mode="locked",
    ) == [
        "calibration_gpu_id_unverified",
        "calibration_compiler_version_unverified",
        "calibration_power_profile_unverified",
    ]


def test_frontier_is_unavailable_and_unverified_r_only_reprofiles() -> None:
    semantic = _semantic().model_copy(
        update={"semantic_flops": 0.0, "semantic_bytes": 0.0},
    )
    calibration = _calibration()
    dispatch = DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        kernel_symbol="kernel",
        grid=(1, 1, 1),
        workgroup=(1, 1, 1),
        iteration_ordinal=0,
        counters={},
    )
    ir = predict_ir(semantic, calibration)
    hw = predict_hw([], [dispatch], calibration)
    ratios = calculate_ratios(
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=1.0,
        timing_noise_ms=0.01,
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
