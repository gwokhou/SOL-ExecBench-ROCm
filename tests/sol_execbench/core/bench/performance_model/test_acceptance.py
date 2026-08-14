from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceArtifactKind,
    DiagnosticAcceptanceCase,
    DiagnosticAcceptanceManifest,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    DiagnosticModelIdentity,
    WorkloadKind,
)
from sol_execbench.core.integrity import stable_json_checksum

_ACTIONS = {
    WorkloadKind.ELEMENTWISE: "stop_launch_bound_search",
    WorkloadKind.TRANSPOSE: "improve_coalescing",
    WorkloadKind.REDUCTION: "reduce_lds_barriers",
    WorkloadKind.MATMUL: "restore_wmma_path",
    WorkloadKind.SOFTMAX: "stop_launch_bound_search",
    WorkloadKind.CROSS_ENTROPY: "improve_coalescing",
    WorkloadKind.INDEXED_READ: "improve_coalescing",
    WorkloadKind.INDEXED_UPDATE: "reduce_atomic_contention",
    WorkloadKind.COMPOSITE: "reduce_dispatch_count",
    WorkloadKind.TRANSFORMER: "restore_fused_attention_path",
    WorkloadKind.CONCURRENT: "remove_extra_traffic",
}


def _identity() -> CalibrationIdentity:
    return CalibrationIdentity(
        gpu_architecture="gfx1200",
        gpu_id="gpu-0",
        gpu_bdf="0000:03:00.0",
        rocm_version="7.2",
        compiler_version="hipcc-7.2",
        clock_mode="locked",
        power_profile="stable_peak",
    )


def _model_identity() -> DiagnosticModelIdentity:
    return DiagnosticModelIdentity(
        model_version="gfx1200_diagnostic.v7",
        policy_files={"policy.py": "d" * 64},
        counter_semantics_sha256="e" * 64,
        policy_bundle_sha256="f" * 64,
    )


def _case(kind: WorkloadKind, index: int) -> DiagnosticAcceptanceCase:
    identity = f"{kind}:{index}"
    action = _ACTIONS[kind]
    return DiagnosticAcceptanceCase(
        case_id=identity,
        pair_id=stable_json_checksum([identity, "pair"]),
        workload_kind=kind,
        evidence_manifest_sha256=stable_json_checksum([identity, "evidence"]),
        performance_diagnostic_sha256=stable_json_checksum(
            [identity, "diagnostic"]
        ),
        predicted_ms=1.05,
        lower_ms=0.9,
        upper_ms=1.1,
        measured_ms=1.0,
        predicted_action_codes=[action],
        gold_action_codes=[action],
    )


def _manifest() -> DiagnosticAcceptanceManifest:
    return DiagnosticAcceptanceManifest(
        artifact_kind=DiagnosticAcceptanceArtifactKind.MANIFEST,
        model_identity=_model_identity(),
        calibration_profile_sha256="a" * 64,
        calibration_identity=_identity(),
        inference_profile_sha256="b" * 64,
        development_corpus_sha256="c" * 64,
        held_out_corpus_sha256="d" * 64,
        enabled_action_codes=sorted(_ACTIONS.values()),
        cases=[_case(kind, index) for kind in _ACTIONS for index in range(20)],
    )


def test_frozen_acceptance_requires_twenty_cases_per_family() -> None:
    result = evaluate_diagnostic_acceptance(_manifest())

    assert result.accepted is True
    assert result.case_count == 220
    assert set(result.family_case_counts.values()) == {20}
    assert set(result.family_empirical_coverage.values()) == {1.0}
    assert result.median_absolute_percentage_error <= 15.0
    assert result.p90_absolute_percentage_error <= 30.0
    assert (
        _manifest().artifact_kind is DiagnosticAcceptanceArtifactKind.MANIFEST
    )
    assert result.artifact_kind is DiagnosticAcceptanceArtifactKind.RESULT


def test_acceptance_family_requires_explicit_artifact_kind() -> None:
    payload = _manifest().model_dump(mode="json")
    payload.pop("artifact_kind")

    with pytest.raises(ValueError, match="artifact_kind"):
        DiagnosticAcceptanceManifest.model_validate(payload)

    payload["artifact_kind"] = DiagnosticAcceptanceArtifactKind.RESULT
    with pytest.raises(ValueError, match="manifest"):
        DiagnosticAcceptanceManifest.model_validate(payload)


def test_supplied_prediction_field_is_rejected_by_corpus_contract() -> None:
    with pytest.raises(ValidationError):
        DiagnosticAcceptanceCase.model_validate(
            {
                **_case(
                    WorkloadKind.ELEMENTWISE,
                    0,
                ).model_dump(mode="json"),
                "primary_attribution_code": "supplied",
            }
        )


def test_wrong_action_fails_acceptance() -> None:
    manifest = _manifest()
    cases = list(manifest.cases)
    for index in range(7):
        cases[index] = cases[index].model_copy(
            update={"gold_action_codes": ["wrong"]}
        )

    result = evaluate_diagnostic_acceptance(
        manifest.model_copy(update={"cases": cases})
    )

    assert result.accepted is False
    assert "held_out_action_quality_gate_failed" in result.reason_codes


def test_control_plane_conformance_does_not_claim_production_quality() -> None:
    manifest = _manifest()
    cases = [
        case.model_copy(update={"measured_ms": 2.0}) for case in manifest.cases
    ]

    production = evaluate_diagnostic_acceptance(
        manifest.model_copy(update={"cases": cases})
    )
    conformance = evaluate_diagnostic_acceptance(
        manifest.model_copy(
            update={
                "purpose": DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
                "cases": cases,
            }
        )
    )

    assert production.accepted is False
    assert conformance.accepted is True
    assert set(conformance.family_empirical_coverage.values()) == {0.0}
    assert (
        conformance.purpose
        is DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
    )
