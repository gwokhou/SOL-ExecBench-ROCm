"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from sol_execbench.core.compatibility import MatrixDependencyPolicyEvidence
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.docker_matrix import DockerTargetManifestEntry


_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class PytorchDependencyPolicy(BaseModelWithDocstrings):
    """Strict checked-in PyTorch ROCm dependency policy for one Target."""

    model_config = _MODEL_CONFIG

    policy_id: str
    """Stable dependency policy identifier."""
    wheel_availability: Literal["available", "unavailable"]
    """Whether matching PyTorch ROCm wheels are declared available."""
    torch_version: str
    """Expected torch distribution version, including ROCm local tag."""
    torchvision_version: str
    """Expected torchvision distribution version, including ROCm local tag."""
    expected_local_version: str
    """Expected PyTorch ROCm local-version tag."""
    uv_index_name: str
    """uv index name expected to provide torch and torchvision."""
    uv_index_url: str
    """uv index URL expected to provide torch and torchvision."""
    lock_strategy: str
    """Dependency lock or workflow strategy for this policy."""
    suggested_uv_command: str
    """Auditable uv command or workflow users can run for this policy."""
    triton_rocm_version: str
    """Expected triton-rocm distribution version."""
    triton_rocm_index_name: str
    """uv index name expected to provide triton-rocm."""
    triton_rocm_index_url: str
    """uv index URL expected to provide triton-rocm."""


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
