"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

from __future__ import annotations

from sol_execbench.core.platform.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.platform.docker_matrix.models import (
    DockerPreflightObservation,
    DockerPreflightResult,
)
from sol_execbench.core.platform.docker_matrix.targets import to_matrix_target


def _observed_device_nodes(observation: DockerPreflightObservation) -> list[str]:
    nodes = []
    if observation.dev_kfd_present and observation.dev_kfd_accessible:
        nodes.append("/dev/kfd")
    if observation.dev_dri_present and observation.dev_dri_accessible:
        nodes.append("/dev/dri")
    return nodes


def _runtime_unavailable_reason(
    observation: DockerPreflightObservation,
) -> str | None:
    docker_context = observation.docker_context or ""
    docker_host = observation.docker_host or ""
    if docker_context == "desktop-linux" or "/.docker/desktop/" in docker_host:
        return (
            f"Docker Desktop context {docker_context!r} ({docker_host}) cannot "
            "provide native Linux ROCm device passthrough."
        )
    if not observation.dev_kfd_present:
        return "/dev/kfd is missing on the host before Docker benchmark execution."
    if not observation.dev_kfd_accessible:
        return "/dev/kfd is not accessible before Docker benchmark execution."
    if not observation.dev_dri_present:
        return "/dev/dri is missing on the host before Docker benchmark execution."
    if not observation.dev_dri_accessible:
        return "/dev/dri is not accessible before Docker benchmark execution."
    if observation.gpu_accessible is False:
        return "GPU access is unavailable before Docker benchmark execution."
    return None


def classify_docker_preflight(
    observation: DockerPreflightObservation,
) -> DockerPreflightResult:
    """Classify Docker runtime observations before benchmark execution."""

    reason = _runtime_unavailable_reason(observation)
    status = (
        MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
        if reason is not None
        else MatrixCompatibilityStatus.NOT_TESTED
    )
    reason_code = (
        MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
        if reason is not None
        else MatrixCompatibilityReasonCode.TARGET_NOT_TESTED
    )
    if reason is None:
        reason = (
            "Docker preflight did not find runtime blockers, but no benchmark, "
            "container user-space, host, or hardware validation has been "
            "performed."
        )
    target = to_matrix_target(observation.selected_target)
    entry = build_matrix_entry(
        target=target,
        observed=MatrixObservedEvidence(
            host=MatrixHostEvidence(
                device_nodes=_observed_device_nodes(observation),
                source="docker_preflight",
            ),
            container=MatrixContainerEvidence(
                image_repository=observation.image_repository,
                image_tag=observation.image_tag,
                image_digest=observation.image_digest,
            ),
            gpu=MatrixGpuEvidence(
                device_count=1 if observation.gpu_accessible else None,
                visible_device_environment=observation.visible_device_environment,
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
    return DockerPreflightResult(
        entry=entry,
        decision=classify_matrix_entry_for_execution(entry),
        build_args=dict(observation.build_args),
    )
