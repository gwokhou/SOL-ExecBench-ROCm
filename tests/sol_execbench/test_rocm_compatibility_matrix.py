from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.compatibility import (
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    MatrixArtifactReference,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixTarget,
    MatrixToolchainEvidence,
    MatrixValidationScope,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
)


EXPECTED_MATRIX_STATUSES = {
    "host_validated",
    "container_validated",
    "mixed_version",
    "pytorch_wheel_unavailable",
    "runtime_unavailable",
    "not_tested",
}


def _representative_entry() -> MatrixEntry:
    return build_matrix_entry(
        target=MatrixTarget(
            target_id="rocm-7.1-gfx1200-container",
            requested_rocm_user_space_version="7.1.0",
            docker_image_repository="rocm/dev-ubuntu-24.04",
            docker_image_tag="7.1.0-complete",
            pytorch_rocm_target="rocm7.1",
            validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
            intended_gpu_architecture="gfx1200",
        ),
        observed=MatrixObservedEvidence(
            host=MatrixHostEvidence(
                rocm_version="7.1.0",
                driver_version="6.14.0",
                device_nodes=["/dev/kfd", "/dev/dri/renderD128"],
            ),
            container=MatrixContainerEvidence(
                rocm_user_space_version="7.1.0",
                image_repository="rocm/dev-ubuntu-24.04",
                image_tag="7.1.0-complete",
            ),
            python_dependency=MatrixPythonDependencyEvidence(
                python_version="3.12.10",
                torch_version="2.7.1+rocm7.1",
                torch_rocm_target="rocm7.1",
                torch_hip_version="7.1.0",
                triton_rocm_status="installed",
            ),
            toolchain=MatrixToolchainEvidence(
                hipcc_version="HIP version: 7.1.0",
                rocm_agent_enumerator_version="1.0.0",
            ),
            gpu=MatrixGpuEvidence(
                device_count=1,
                device_name="AMD Radeon RX 9070 XT",
                gfx_architecture="gfx1200",
                visible_device_environment={"HIP_VISIBLE_DEVICES": "0"},
            ),
        ),
        status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
        reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
        reason="Container ROCm user-space matched the requested Target.",
        artifacts=[
            MatrixArtifactReference(
                artifact_id="compatibility-probe",
                kind="probe_json",
                path="artifacts/compatibility.json",
                uri="file://artifacts/compatibility.json",
                description="Bounded compatibility probe payload.",
            )
        ],
    )


def test_matrix_entry_serializes_target_and_observed_evidence_separately():
    entry = _representative_entry()
    payload = entry.model_dump(mode="json")

    assert payload["schema_version"] == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    assert payload["schema_version"] == "sol_execbench.rocm_compatibility_matrix.v1"
    assert payload["target"]["requested_rocm_user_space_version"] == "7.1.0"
    assert payload["target"]["docker_image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["target"]["docker_image_tag"] == "7.1.0-complete"
    assert payload["target"]["pytorch_rocm_target"] == "rocm7.1"
    assert payload["target"]["validation_scope"] == "container_user_space"
    assert payload["target"]["intended_gpu_architecture"] == "gfx1200"
    assert payload["observed"]["host"]["rocm_version"] == "7.1.0"
    assert payload["observed"]["container"]["rocm_user_space_version"] == "7.1.0"
    assert payload["observed"]["python_dependency"]["torch_rocm_target"] == "rocm7.1"
    assert payload["observed"]["toolchain"]["hipcc_version"] == "HIP version: 7.1.0"
    assert payload["observed"]["gpu"]["gfx_architecture"] == "gfx1200"
    assert MatrixEntry.model_validate(payload) == entry
    assert entry.to_dict() == payload


def test_matrix_entry_rejects_unknown_top_level_and_nested_fields():
    payload = _representative_entry().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        MatrixEntry.model_validate(payload)

    nested_payload = _representative_entry().model_dump(mode="json")
    nested_payload["observed"]["host"]["unexpected"] = True

    with pytest.raises(ValidationError):
        MatrixEntry.model_validate(nested_payload)


def test_matrix_status_vocabulary_is_locked():
    assert {status.value for status in MatrixCompatibilityStatus} == (
        EXPECTED_MATRIX_STATUSES
    )


def test_matrix_report_contains_entries_and_status_counts():
    entry = _representative_entry()
    report = RocmCompatibilityMatrixReport(
        generated_at="2026-05-28T05:22:46Z",
        entries=[entry],
        status_counts={MatrixCompatibilityStatus.CONTAINER_VALIDATED: 1},
    )
    payload = report.model_dump(mode="json")

    assert payload["schema_version"] == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    assert payload["entries"][0] == entry.model_dump(mode="json")
    assert payload["status_counts"] == {"container_validated": 1}
    assert report.to_dict() == payload


def test_matrix_entry_carries_artifacts_and_diagnostic_claim_boundaries():
    payload = _representative_entry().model_dump(mode="json")

    assert payload["artifacts"] == [
        {
            "artifact_id": "compatibility-probe",
            "kind": "probe_json",
            "path": "artifacts/compatibility.json",
            "uri": "file://artifacts/compatibility.json",
            "description": "Bounded compatibility probe payload.",
        }
    ]
    assert payload["claim_boundary"] == {
        "diagnostic_compatibility_evidence": True,
        "score_authority": False,
        "paper_parity_authority": False,
        "leaderboard_authority": False,
        "container_user_space_validated": True,
        "native_host_validated": False,
        "hardware_validated": True,
    }


@pytest.mark.parametrize(
    "authority_field",
    ["score_authority", "paper_parity_authority", "leaderboard_authority"],
)
def test_matrix_claim_authority_flags_cannot_be_set_true(authority_field):
    payload = _representative_entry().model_dump(mode="json")
    payload["claim_boundary"][authority_field] = True

    with pytest.raises(ValidationError):
        MatrixEntry.model_validate(payload)


def test_matrix_claim_validation_flags_are_explicit_booleans_not_reason_text():
    payload = _representative_entry().model_dump(mode="json")
    payload["claim_boundary"]["native_host_validated"] = "false"

    with pytest.raises(ValidationError):
        MatrixEntry.model_validate(payload)
