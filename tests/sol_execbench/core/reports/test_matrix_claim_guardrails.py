from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.platform.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixTarget,
    MatrixValidationScope,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.reports.matrix_diff import (
    MatrixDiffSeverity,
    diff_matrix_reports,
    matrix_report_diff_to_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _container_target() -> MatrixTarget:
    return MatrixTarget(
        target_id="rocm-7.1-gfx1200-container",
        requested_rocm_user_space_version="7.1.0",
        docker_image_repository="rocm/dev-ubuntu-24.04",
        docker_image_tag="7.1.0-complete",
        pytorch_rocm_target="rocm7.1",
        validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
        intended_gpu_architecture="gfx1200",
    )


def _native_host_target() -> MatrixTarget:
    return MatrixTarget(
        target_id="rocm-7.1-gfx1200-native",
        requested_rocm_user_space_version="7.1.0",
        pytorch_rocm_target="rocm7.1",
        validation_scope=MatrixValidationScope.NATIVE_HOST,
        intended_gpu_architecture="gfx1200",
    )


def _observed_container_stack(
    *, torch_rocm_target: str = "rocm7.1"
) -> MatrixObservedEvidence:
    return MatrixObservedEvidence(
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
            torch_version=f"2.7.1+{torch_rocm_target}",
            torch_rocm_target=torch_rocm_target,
            torch_hip_version="7.1.0",
            triton_rocm_status="installed",
        ),
        gpu=MatrixGpuEvidence(
            device_count=1,
            device_name="AMD Radeon RX 9070 XT",
            gfx_architecture="gfx1200",
        ),
    )


def _observed_native_host_stack() -> MatrixObservedEvidence:
    return MatrixObservedEvidence(
        host=MatrixHostEvidence(
            rocm_version="7.1.0",
            driver_version="6.14.0",
            device_nodes=["/dev/kfd", "/dev/dri/renderD128"],
        ),
        python_dependency=MatrixPythonDependencyEvidence(
            python_version="3.12.10",
            torch_version="2.7.1+rocm7.1",
            torch_rocm_target="rocm7.1",
            torch_hip_version="7.1.0",
        ),
        gpu=MatrixGpuEvidence(
            device_count=1,
            device_name="AMD Radeon RX 9070 XT",
            gfx_architecture="gfx1200",
        ),
    )


def _entry(
    *,
    status: MatrixCompatibilityStatus,
    reason_code: MatrixCompatibilityReasonCode,
    reason: str,
    claim_boundary: MatrixClaimBoundary | None = None,
    observed: MatrixObservedEvidence | None = None,
) -> MatrixEntry:
    return build_matrix_entry(
        target=_container_target(),
        observed=observed or _observed_container_stack(),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=claim_boundary
        or MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )


def _native_host_entry(*, observed: MatrixObservedEvidence) -> MatrixEntry:
    return build_matrix_entry(
        target=_native_host_target(),
        observed=observed,
        status=MatrixCompatibilityStatus.HOST_VALIDATED,
        reason_code=MatrixCompatibilityReasonCode.HOST_NATIVE_VALIDATED,
        reason="Direct native-host ROCm validation matched the requested Target.",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=True,
            hardware_validated=True,
        ),
    )


def test_mixed_version_is_blocked_before_benchmark_by_default():
    entry = _entry(
        status=MatrixCompatibilityStatus.MIXED_VERSION,
        reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
        reason="Target requested rocm7.1 but observed rocm7.0.",
    )

    decision = classify_matrix_entry_for_execution(
        entry, allow_mixed_version_debug=False
    )

    assert decision.status is MatrixCompatibilityStatus.MIXED_VERSION
    assert (
        decision.reason_code is MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH
    )
    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is False
    assert decision.smoke_allowed is False
    assert "blocked before benchmark execution" in decision.reason


def test_mixed_version_debug_override_allows_probe_or_smoke_without_clean_claims():
    entry = _entry(
        status=MatrixCompatibilityStatus.MIXED_VERSION,
        reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
        reason="Observed PyTorch ROCm wheel target does not match requested Target.",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=True,
            native_host_validated=False,
            hardware_validated=True,
        ),
        observed=_observed_container_stack(torch_rocm_target="rocm7.0"),
    )

    decision = classify_matrix_entry_for_execution(
        entry, allow_mixed_version_debug=True
    )

    assert decision.status is MatrixCompatibilityStatus.MIXED_VERSION
    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is True
    assert decision.container_user_space_validated is False
    assert decision.native_host_validated is False
    assert decision.score_authority is False
    assert decision.paper_parity_authority is False
    assert decision.leaderboard_authority is False
    assert "debug override" in decision.reason


def test_pytorch_wheel_unavailable_is_not_a_benchmark_correctness_failure():
    entry = _entry(
        status=MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
        reason_code=MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
        reason="No matching PyTorch ROCm wheel exists for this Target.",
    )

    decision = classify_matrix_entry_for_execution(entry)

    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is False
    assert "dependency stack" in decision.reason


def test_runtime_unavailable_is_not_a_benchmark_correctness_failure():
    entry = _entry(
        status=MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE,
        reason_code=MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE,
        reason="Required ROCm runtime devices were unavailable.",
    )

    decision = classify_matrix_entry_for_execution(entry)

    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is False
    assert "runtime" in decision.reason


def test_not_tested_is_non_authoritative_and_not_benchmark_eligible():
    entry = _entry(
        status=MatrixCompatibilityStatus.NOT_TESTED,
        reason_code=MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        reason="Target has not been tested.",
    )

    decision = classify_matrix_entry_for_execution(entry)

    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is False
    assert decision.container_user_space_validated is False
    assert decision.native_host_validated is False
    assert decision.score_authority is False
    assert decision.paper_parity_authority is False
    assert decision.leaderboard_authority is False


def test_docker_scope_cannot_serialize_host_validated_status():
    with pytest.raises(ValidationError, match="Docker.*host_validated"):
        _entry(
            status=MatrixCompatibilityStatus.HOST_VALIDATED,
            reason_code=MatrixCompatibilityReasonCode.HOST_NATIVE_VALIDATED,
            reason="Incorrectly claimed direct native-host validation.",
            claim_boundary=MatrixClaimBoundary(
                container_user_space_validated=False,
                native_host_validated=True,
                hardware_validated=True,
            ),
        )


def test_docker_container_validated_claims_container_user_space_not_native_host():
    entry = _entry(
        status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
        reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
        reason="container ROCm user-space validated on recorded host driver/devices",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=True,
            native_host_validated=False,
            hardware_validated=True,
        ),
    )
    payload = entry.model_dump(mode="json")

    assert payload["status"] == "container_validated"
    assert payload["reason"] == (
        "container ROCm user-space validated on recorded host driver/devices"
    )
    assert payload["claim_boundary"]["container_user_space_validated"] is True
    assert payload["claim_boundary"]["native_host_validated"] is False


def test_container_validated_rejects_native_host_scope():
    with pytest.raises(ValidationError, match="container_user_space validation scope"):
        build_matrix_entry(
            target=_native_host_target(),
            observed=_observed_container_stack(),
            status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
            reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
            reason="Container validation cannot be claimed from a native-host Target.",
            claim_boundary=MatrixClaimBoundary(
                container_user_space_validated=True,
                native_host_validated=False,
                hardware_validated=True,
            ),
        )


def test_container_validated_rejects_missing_container_evidence():
    with pytest.raises(ValidationError, match="observed container evidence"):
        _entry(
            status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
            reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
            reason="Container validation requires container evidence.",
            claim_boundary=MatrixClaimBoundary(
                container_user_space_validated=True,
                native_host_validated=False,
                hardware_validated=True,
            ),
            observed=MatrixObservedEvidence(
                host=MatrixHostEvidence(rocm_version="7.1.0")
            ),
        )


def test_container_validated_rejects_native_host_claim_flag():
    with pytest.raises(ValidationError, match="native_host_validated=true"):
        _entry(
            status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
            reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
            reason="Container validation cannot claim native-host validation.",
            claim_boundary=MatrixClaimBoundary(
                container_user_space_validated=True,
                native_host_validated=True,
                hardware_validated=True,
            ),
        )


def test_native_host_validated_requires_direct_host_evidence():
    entry = _native_host_entry(observed=_observed_native_host_stack())

    assert entry.status is MatrixCompatibilityStatus.HOST_VALIDATED
    assert entry.claim_boundary.native_host_validated is True
    assert entry.observed.host is not None
    assert entry.observed.host.rocm_version == "7.1.0"
    assert entry.observed.container is None


def test_native_host_validated_rejects_missing_direct_host_evidence():
    with pytest.raises(ValidationError, match="direct native-host evidence"):
        _native_host_entry(
            observed=MatrixObservedEvidence(
                container=MatrixContainerEvidence(rocm_user_space_version="7.1.0")
            )
        )


def test_host_validated_rejects_container_claim_flag():
    with pytest.raises(ValidationError, match="container_user_space_validated=true"):
        build_matrix_entry(
            target=_native_host_target(),
            observed=_observed_native_host_stack(),
            status=MatrixCompatibilityStatus.HOST_VALIDATED,
            reason_code=MatrixCompatibilityReasonCode.HOST_NATIVE_VALIDATED,
            reason="Native-host validation cannot claim container validation.",
            claim_boundary=MatrixClaimBoundary(
                container_user_space_validated=True,
                native_host_validated=True,
                hardware_validated=True,
            ),
        )


def test_matrix_contract_uses_target_and_matrix_entry_not_row_wording():
    source = (
        REPO_ROOT / "src/sol_execbench/core/platform/compatibility.py"
    ).read_text()

    assert "Matrix Entry" in source
    assert "Target" in source
    assert "Row" not in source


def test_container_claim_wording_does_not_overstate_native_host_validation():
    claims = (REPO_ROOT / "docs/CLAIMS.md").read_text()

    assert (
        "container ROCm user-space validated on\nrecorded host driver/devices" in claims
    )
    assert "Docker row is not native host ROCm\nvalidated" in claims


def test_matrix_diff_keeps_claim_boundary_escalation_diagnostic_only():
    old_entry = _entry(
        status=MatrixCompatibilityStatus.NOT_TESTED,
        reason_code=MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        reason="Target has not been tested.",
    )
    new_entry = _entry(
        status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
        reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
        reason="Container ROCm user-space matched the requested Target.",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=True,
            native_host_validated=False,
            hardware_validated=True,
        ),
    )
    old_report = RocmCompatibilityMatrixReport(
        generated_at="2026-05-31T09:00:00Z",
        entries=[old_entry],
        status_counts={MatrixCompatibilityStatus.NOT_TESTED: 1},
    )
    new_report = RocmCompatibilityMatrixReport(
        generated_at="2026-05-31T09:05:00Z",
        entries=[new_entry],
        status_counts={MatrixCompatibilityStatus.CONTAINER_VALIDATED: 1},
    )

    diff = diff_matrix_reports(old_report, new_report)
    payload = diff.to_dict()
    markdown = matrix_report_diff_to_markdown(diff)

    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert payload["native_host_validation_authority"] is False
    assert payload["entry_diffs"][0]["severity"] == (
        MatrixDiffSeverity.CLAIM_BOUNDARY_ESCALATION.value
    )
    assert "Docker/container evidence does not imply native-host validation" in markdown
    assert "score authority" in markdown
    assert "paper-parity authority" in markdown
    assert "leaderboard authority" in markdown
