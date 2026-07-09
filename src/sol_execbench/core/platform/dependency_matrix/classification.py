"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

import platform

from sol_execbench.core.platform.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixToolchainEvidence,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.platform.dependency_matrix.models import (
    DependencyPreflightResult,
    PytorchDependencyObservation,
    PytorchDependencyPolicy,
)
from sol_execbench.core.platform.dependency_matrix.policy import (
    dependency_policy_evidence_for_target,
)
from sol_execbench.core.platform.docker_matrix import (
    DockerTargetManifestEntry,
    to_matrix_target,
)


def classify_dependency_preflight(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
    allow_mixed_version_debug: bool = False,
) -> DependencyPreflightResult:
    """Classify a selected Target's observed PyTorch ROCm dependency stack."""

    status, reason_code, reason = _classify_observation(
        target=target,
        policy=policy,
        observation=observation,
    )
    entry = build_matrix_entry(
        target=to_matrix_target(target),
        observed=MatrixObservedEvidence(
            container=MatrixContainerEvidence(
                rocm_user_space_version=observation.container_rocm_user_space_version,
                image_repository=target.docker_image_repository,
                image_tag=target.docker_image_tag,
            ),
            python_dependency=_python_dependency_evidence(observation),
            dependency_policy=dependency_policy_evidence_for_target(target),
            toolchain=MatrixToolchainEvidence(
                hipcc_version=observation.hipcc_version,
                toolchain_rocm_version=observation.toolchain_rocm_version,
            ),
        ),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
    return DependencyPreflightResult(
        entry=entry,
        decision=classify_matrix_entry_for_execution(
            entry,
            allow_mixed_version_debug=allow_mixed_version_debug,
        ),
        policy=policy,
    )


def _classify_observation(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
) -> tuple[
    MatrixCompatibilityStatus,
    MatrixCompatibilityReasonCode,
    str,
]:
    if policy.wheel_availability == "unavailable":
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            f"PyTorch ROCm wheels are declared unavailable by policy {policy.policy_id}.",
        )
    if observation.torch_import_error:
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            f"Required torch runtime could not be imported: {observation.torch_import_error}",
        )
    if observation.torch_distribution_version is None:
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            "Required torch distribution is unavailable for this Target policy.",
        )

    mismatch_reason = _mismatch_reason(
        target=target, policy=policy, observation=observation
    )
    if mismatch_reason is not None:
        return (
            MatrixCompatibilityStatus.MIXED_VERSION,
            MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
            mismatch_reason,
        )
    return (
        MatrixCompatibilityStatus.NOT_TESTED,
        MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        (
            "Dependency stack matches the selected Target policy, but no benchmark, "
            "container user-space, native host, or hardware validation has been performed."
        ),
    )


def _mismatch_reason(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
) -> str | None:
    if observation.torch_cuda_version:
        return (
            f"CUDA PyTorch runtime {observation.torch_cuda_version} was observed for "
            f"ROCm Target {target.target_id}."
        )
    if observation.torch_distribution_version != policy.torch_version:
        return (
            f"torch distribution {observation.torch_distribution_version!r} does not "
            f"match policy {policy.torch_version!r}."
        )
    if observation.torch_version and observation.torch_version != policy.torch_version:
        return (
            f"torch runtime version {observation.torch_version!r} does not match "
            f"policy {policy.torch_version!r}."
        )
    if observation.torch_local_version != policy.expected_local_version:
        return (
            f"torch local-version {observation.torch_local_version!r} does not match "
            f"policy {policy.expected_local_version!r}."
        )
    expected_rocm_target = target.pytorch_rocm_target or policy.expected_local_version
    if observation.torch_rocm_target != expected_rocm_target:
        return (
            f"torch ROCm target {observation.torch_rocm_target!r} does not match "
            f"requested {expected_rocm_target!r}."
        )
    expected_hip_prefix = _rocm_major_minor(policy.expected_local_version)
    if (
        expected_hip_prefix is not None
        and observation.torch_hip_version is not None
        and not observation.torch_hip_version.startswith(expected_hip_prefix)
    ):
        return (
            f"torch HIP version {observation.torch_hip_version!r} does not match "
            f"policy {policy.expected_local_version!r}."
        )
    if observation.torchvision_distribution_version != policy.torchvision_version:
        return (
            "torchvision distribution "
            f"{observation.torchvision_distribution_version!r} does not match policy "
            f"{policy.torchvision_version!r}."
        )
    if observation.triton_rocm_status != "installed":
        return (
            f"triton-rocm status {observation.triton_rocm_status!r} does not match "
            "required installed policy."
        )
    if observation.triton_rocm_distribution_version != policy.triton_rocm_version:
        return (
            "triton-rocm distribution "
            f"{observation.triton_rocm_distribution_version!r} does not match policy "
            f"{policy.triton_rocm_version!r}."
        )
    if not _version_matches_expected(
        observation.container_rocm_user_space_version,
        policy.expected_local_version,
    ):
        return (
            "container ROCm user-space version "
            f"{observation.container_rocm_user_space_version!r} does not match policy "
            f"{policy.expected_local_version!r}."
        )
    if not _version_matches_expected(
        observation.toolchain_rocm_version,
        policy.expected_local_version,
    ):
        return (
            f"toolchain ROCm version {observation.toolchain_rocm_version!r} does not "
            f"match policy {policy.expected_local_version!r}."
        )
    return None


def _python_dependency_evidence(
    observation: PytorchDependencyObservation,
) -> MatrixPythonDependencyEvidence:
    return MatrixPythonDependencyEvidence(
        python_version=platform.python_version(),
        torch_distribution_version=observation.torch_distribution_version,
        torch_version=observation.torch_version,
        torch_local_version=observation.torch_local_version,
        torch_rocm_target=observation.torch_rocm_target,
        torch_hip_version=observation.torch_hip_version,
        torch_cuda_version=observation.torch_cuda_version,
        torch_device_available=observation.torch_device_available,
        torch_import_error=observation.torch_import_error,
        torchvision_distribution_version=observation.torchvision_distribution_version,
        triton_rocm_distribution_version=observation.triton_rocm_distribution_version,
        triton_rocm_status=observation.triton_rocm_status,
    )


def _rocm_major_minor(expected_local_version: str) -> str | None:
    if not expected_local_version.startswith("rocm"):
        return None
    parts = expected_local_version.removeprefix("rocm").split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:2])


def _version_matches_expected(version: str | None, expected_local_version: str) -> bool:
    expected = _rocm_major_minor(expected_local_version)
    if version is None or expected is None:
        return True
    return version.startswith(expected)
