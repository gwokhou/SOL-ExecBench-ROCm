from __future__ import annotations

from sol_execbench.core.compatibility import (
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
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)


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


def _observed_container_stack(*, torch_rocm_target: str = "rocm7.1") -> MatrixObservedEvidence:
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
    assert decision.reason_code is MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH
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
