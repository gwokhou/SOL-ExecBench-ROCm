"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

from sol_execbench.core.platform.compatibility import MatrixDependencyPolicyEvidence
from sol_execbench.core.platform.dependency_matrix.models import PytorchDependencyPolicy
from sol_execbench.core.platform.docker_matrix import DockerTargetManifestEntry


def load_docker_target_dependency_policy(
    target: DockerTargetManifestEntry,
) -> PytorchDependencyPolicy:
    """Load and strictly validate dependency policy from a Docker Target."""

    if target.pytorch_dependency_policy is None:
        raise ValueError(
            f"Docker Target {target.target_id!r} does not declare "
            "pytorch_dependency_policy."
        )
    return PytorchDependencyPolicy.model_validate(target.pytorch_dependency_policy)


def dependency_policy_evidence_for_target(
    target: DockerTargetManifestEntry,
) -> MatrixDependencyPolicyEvidence:
    """Convert a Docker Target policy into Matrix Entry policy evidence."""

    policy = load_docker_target_dependency_policy(target)
    return MatrixDependencyPolicyEvidence(
        policy_id=policy.policy_id,
        expected_local_version=policy.expected_local_version,
        uv_index_name=policy.uv_index_name,
        uv_index_url=policy.uv_index_url,
        lock_strategy=policy.lock_strategy,
        suggested_uv_command=policy.suggested_uv_command,
        triton_rocm_version=policy.triton_rocm_version,
        triton_rocm_index_name=policy.triton_rocm_index_name,
        triton_rocm_index_url=policy.triton_rocm_index_url,
    )
