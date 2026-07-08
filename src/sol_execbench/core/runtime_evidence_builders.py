"""Runtime compatibility matrix builders."""

from __future__ import annotations

from datetime import UTC, datetime

from sol_execbench.core.compatibility import (
    MatrixArtifactReference,
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixToolchainEvidence,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
)
from sol_execbench.core.dependency_matrix import (
    PytorchDependencyObservation,
    classify_dependency_preflight,
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.docker_matrix import DockerTargetManifestEntry, to_matrix_target
from sol_execbench.core.runtime_evidence_models import RuntimeFailureEvidence


def build_runtime_matrix_entry(
    *,
    target: DockerTargetManifestEntry,
    dependency_observation: PytorchDependencyObservation,
    host: MatrixHostEvidence | None = None,
    container: MatrixContainerEvidence | None = None,
    toolchain: MatrixToolchainEvidence | None = None,
    gpu: MatrixGpuEvidence | None = None,
    runtime_unavailable_reason: str | None = None,
    failure_evidence: list[RuntimeFailureEvidence] | None = None,
    allow_mixed_version_debug: bool = False,
    container_validated: bool = False,
) -> MatrixEntry:
    """Build a diagnostic runtime Matrix Entry for one Docker Target."""
    dependency_result = classify_dependency_preflight(
        target=target,
        policy=load_docker_target_dependency_policy(target),
        observation=dependency_observation,
        allow_mixed_version_debug=allow_mixed_version_debug,
    )
    if runtime_unavailable_reason is not None:
        status = MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
        reason_code = MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
        reason = runtime_unavailable_reason
    elif container_validated and (
        dependency_result.entry.status is MatrixCompatibilityStatus.NOT_TESTED
    ):
        status = MatrixCompatibilityStatus.CONTAINER_VALIDATED
        reason_code = MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED
        reason = (
            "Target-specific Docker wrapper benchmark completed successfully with "
            "matching container dependency and ROCm user-space evidence."
        )
    else:
        status = dependency_result.entry.status
        reason_code = dependency_result.entry.reason_code
        reason = dependency_result.entry.reason

    observed_dependency = dependency_result.entry.observed.python_dependency
    observed_policy = dependency_policy_evidence_for_target(target)
    observed_container = container or MatrixContainerEvidence(
        rocm_user_space_version=dependency_observation.container_rocm_user_space_version,
        image_repository=target.docker_image_repository,
        image_tag=target.docker_image_tag,
    )
    observed_toolchain = toolchain or MatrixToolchainEvidence(
        hipcc_version=dependency_observation.hipcc_version,
        toolchain_rocm_version=dependency_observation.toolchain_rocm_version,
    )
    artifacts = [
        MatrixArtifactReference(
            artifact_id=f"failure-{index + 1}",
            kind=f"runtime_evidence_{failure.category}",
            uri=f"diagnostic://runtime-evidence/{failure.category}/{index + 1}",
            description=failure.message or failure.status,
        )
        for index, failure in enumerate(failure_evidence or [])
    ]

    return build_matrix_entry(
        target=to_matrix_target(target),
        observed=MatrixObservedEvidence(
            host=host,
            container=observed_container,
            python_dependency=observed_dependency,
            dependency_policy=observed_policy,
            toolchain=observed_toolchain,
            gpu=gpu,
        ),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=(
                status is MatrixCompatibilityStatus.CONTAINER_VALIDATED
            ),
            native_host_validated=False,
            hardware_validated=False,
        ),
        artifacts=artifacts,
    )


def build_aggregate_report(
    entries: list[MatrixEntry],
    *,
    generated_at: str | None = None,
) -> RocmCompatibilityMatrixReport:
    """Build an aggregate compatibility matrix report from entries."""
    counts: dict[MatrixCompatibilityStatus, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return RocmCompatibilityMatrixReport(
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        entries=entries,
        status_counts=counts,
    )
