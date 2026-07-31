from __future__ import annotations

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.inference import (
    CODE_CHANGING_ACTION_CODES,
    DiagnosticInferenceProfile,
    InferenceObservation,
    action_is_admitted,
    apply_conformal_interval,
    build_inference_profile,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticModelIdentity,
    PerformancePrediction,
    PredictionKind,
    WorkloadKind,
)

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)


def _identity() -> DiagnosticModelIdentity:
    return DiagnosticModelIdentity(
        model_version="gfx1200_diagnostic.v6",
        policy_files={"policy.py": "a" * 64},
        counter_semantics_sha256="b" * 64,
        policy_bundle_sha256="c" * 64,
    )


def _observations() -> list[InferenceObservation]:
    return [
        InferenceObservation(
            case_id=f"{kind}:{index}",
            workload_kind=kind,
            measured_ms=1.0,
            base_predicted_ms=1.0,
            base_lower_ms=0.9,
            base_upper_ms=1.1,
            point_features={
                "solar_lower_bound_ms": 1.0,
                "width_64": 0.0,
                "width_128": 0.0,
                "width_256": 0.0,
                "width_512": 0.0,
                "width_1024": 0.0,
                "outer_rows_width_32": float(index + 1),
                "outer_rows_width_64": 0.0,
                "outer_rows_width_128": 0.0,
                "outer_rows_width_256": 0.0,
                "outer_rows_width_512": 0.0,
                "outer_rows_width_1024": 0.0,
            },
        )
        for kind in _FAMILIES
        for index in range(40)
    ]


def test_inference_profile_requires_exact_family_support() -> None:
    profile = build_inference_profile(
        _observations(),
        model_identity=_identity(),
        calibration_profile_sha256="d" * 64,
        calibration_audit_sha256="e" * 64,
        development_corpus_sha256="f" * 64,
    )

    assert {item.case_count for item in profile.conformal} == {40}
    assert {item.point_fit_case_count for item in profile.conformal} == {20}
    assert {item.conformal_case_count for item in profile.conformal} == {20}
    assert all(
        item.point_intercept_ms == pytest.approx(1.0)
        for item in profile.conformal
    )
    assert all(
        set(item.point_feature_names)
        <= {
            "solar_lower_bound_ms",
            "width_64",
            "width_128",
            "width_256",
            "width_512",
            "width_1024",
            "outer_rows_width_32",
            "outer_rows_width_64",
            "outer_rows_width_128",
            "outer_rows_width_256",
            "outer_rows_width_512",
            "outer_rows_width_1024",
        }
        for item in profile.conformal
    )
    assert {item.q95 for item in profile.conformal} == {0.0}
    assert {
        item.action_code for item in profile.action_thresholds
    } == CODE_CHANGING_ACTION_CODES
    assert profile.enabled_action_codes == frozenset()


def test_family_point_model_corrects_hw_prediction() -> None:
    observations = [
        item.model_copy(
            update={
                "measured_ms": 1.0,
                "base_predicted_ms": 2.0,
                "base_lower_ms": 1.8,
                "base_upper_ms": 2.2,
                "point_features": {
                    **item.point_features,
                    "solar_lower_bound_ms": float(
                        item.case_id.rsplit(":", maxsplit=1)[1]
                    ),
                },
            }
        )
        for item in _observations()
    ]
    profile = build_inference_profile(
        observations,
        model_identity=_identity(),
        calibration_profile_sha256="d" * 64,
        calibration_audit_sha256="e" * 64,
        development_corpus_sha256="f" * 64,
    )
    prediction = PerformancePrediction(
        kind=PredictionKind.HW,
        status=DiagnosticSidecarStatus.AVAILABLE,
        predicted_time_ms=2.0,
        lower_ms=1.8,
        upper_ms=2.2,
    )

    calibrated = apply_conformal_interval(
        prediction,
        WorkloadKind.ELEMENTWISE,
        {"solar_lower_bound_ms": 10.0},
        profile,
    )

    assert calibrated.predicted_time_ms == pytest.approx(1.0)
    assert calibrated.lower_ms == pytest.approx(0.9)
    assert calibrated.upper_ms == pytest.approx(1.1)


def test_code_actions_abstain_without_an_admitted_profile() -> None:
    assert action_is_admitted("model_gap_no_kernel_action", {}, None)
    assert not action_is_admitted("restore_wmma_path", {}, None)


def test_old_inference_schema_is_rejected_before_business_validation() -> None:
    profile = build_inference_profile(
        _observations(),
        model_identity=_identity(),
        calibration_profile_sha256="d" * 64,
        calibration_audit_sha256="e" * 64,
        development_corpus_sha256="f" * 64,
    )
    payload = profile.model_dump(mode="json")
    payload["schema_version"] = "unsupported"

    with pytest.raises(ValueError, match="schema_version"):
        DiagnosticInferenceProfile.model_validate(payload)
