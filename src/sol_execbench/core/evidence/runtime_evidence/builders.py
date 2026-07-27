"""Runtime compatibility matrix builders."""

from __future__ import annotations

from datetime import UTC, datetime

from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeFailureEvidence,
)
from sol_execbench.core.platform.compatibility import (
    MatrixArtifactReference,
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixGPUEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixToolchainEvidence,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
)
from sol_execbench.core.platform.compatibility_evidence_models import (
    MatrixPythonDependencyEvidence,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
    classify_dependency_preflight,
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.platform.docker_matrix import (
    DockerTargetManifestEntry,
    to_matrix_target,
)


def build_runtime_matrix_entry(
    *,
    target: DockerTargetManifestEntry,
    dependency_observation: PytorchDependencyObservation,
    host: MatrixHostEvidence | None = None,
    container: MatrixContainerEvidence | None = None,
    toolchain: MatrixToolchainEvidence | None = None,
    gpu: MatrixGPUEvidence | None = None,
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
    status, reason_code, reason = _matrix_status(
        dependency_result.entry,
        runtime_unavailable_reason,
        container_validated,
    )

    return build_matrix_entry(
        target=to_matrix_target(target),
        observed=_observed_evidence(
            target,
            dependency_observation,
            dependency_result.entry.observed.python_dependency,
            host=host,
            container=container,
            toolchain=toolchain,
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
        artifacts=_failure_artifacts(failure_evidence),
    )


def _matrix_status(
    entry: MatrixEntry,
    runtime_unavailable_reason: str | None,
    container_validated: bool,
) -> tuple[MatrixCompatibilityStatus, MatrixCompatibilityReasonCode, str]:
    if runtime_unavailable_reason is not None:
        return (
            MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE,
            MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE,
            runtime_unavailable_reason,
        )
    if (
        container_validated
        and entry.status is MatrixCompatibilityStatus.NOT_TESTED
    ):
        return (
            MatrixCompatibilityStatus.CONTAINER_VALIDATED,
            MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
            (
                "Target-specific Docker wrapper benchmark completed successfully with "
                "matching container dependency and ROCm user-space evidence."
            ),
        )
    return entry.status, entry.reason_code, entry.reason


def _observed_evidence(
    target: DockerTargetManifestEntry,
    observation: PytorchDependencyObservation,
    python_dependency: MatrixPythonDependencyEvidence | None,
    *,
    host: MatrixHostEvidence | None,
    container: MatrixContainerEvidence | None,
    toolchain: MatrixToolchainEvidence | None,
    gpu: MatrixGPUEvidence | None,
) -> MatrixObservedEvidence:
    return MatrixObservedEvidence(
        host=host,
        container=container
        or MatrixContainerEvidence(
            rocm_user_space_version=observation.container_rocm_user_space_version,
            image_repository=target.docker_image_repository,
            image_tag=target.docker_image_tag,
        ),
        python_dependency=python_dependency,
        dependency_policy=dependency_policy_evidence_for_target(target),
        toolchain=toolchain
        or MatrixToolchainEvidence(
            hipcc_version=observation.hipcc_version,
            toolchain_rocm_version=observation.toolchain_rocm_version,
        ),
        gpu=gpu,
    )


def _failure_artifacts(
    failures: list[RuntimeFailureEvidence] | None,
) -> list[MatrixArtifactReference]:
    return [
        MatrixArtifactReference(
            artifact_id=f"failure-{index + 1}",
            kind=f"runtime_evidence_{failure.category}",
            uri=f"diagnostic://runtime-evidence/{failure.category}/{index + 1}",
            description=failure.message or failure.status,
        )
        for index, failure in enumerate(failures or [])
    ]


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
