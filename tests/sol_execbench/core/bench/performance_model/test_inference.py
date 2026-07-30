from __future__ import annotations

import pytest

from sol_execbench.core.bench.performance_model.inference import (
    CODE_CHANGING_ACTION_CODES,
    DiagnosticInferenceProfile,
    InferenceObservation,
    action_is_admitted,
    build_inference_profile,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticModelIdentity,
    WorkloadKind,
)

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
)


def _identity() -> DiagnosticModelIdentity:
    return DiagnosticModelIdentity(
        model_version="gfx1200_diagnostic.v3",
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
            base_lower_ms=0.9,
            base_upper_ms=1.1,
        )
        for kind in _FAMILIES
        for index in range(20)
    ]


def test_inference_profile_requires_exact_family_support() -> None:
    profile = build_inference_profile(
        _observations(),
        model_identity=_identity(),
        calibration_profile_sha256="d" * 64,
        calibration_audit_sha256="e" * 64,
        development_corpus_sha256="f" * 64,
    )

    assert {item.case_count for item in profile.conformal} == {20}
    assert {item.q95 for item in profile.conformal} == {0.0}
    assert {
        item.action_code for item in profile.action_thresholds
    } == CODE_CHANGING_ACTION_CODES
    assert profile.enabled_action_codes == frozenset()


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
