"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from sol_execbench.core.compatibility import MatrixEntry, MatrixExecutionDecision
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


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


class PytorchDependencyObservation(BaseModelWithDocstrings):
    """Observed installed PyTorch ROCm dependency stack."""

    model_config = _MODEL_CONFIG

    torch_distribution_version: str | None = None
    """Installed torch distribution version."""
    torch_version: str | None = None
    """Observed torch.__version__ value."""
    torch_local_version: str | None = None
    """Observed torch local-version tag, such as rocm7.1."""
    torch_rocm_target: str | None = None
    """Observed PyTorch ROCm target inferred from local-version metadata."""
    torch_hip_version: str | None = None
    """Observed torch.version.hip value."""
    torch_cuda_version: str | None = None
    """Observed torch.version.cuda value."""
    torch_device_available: bool | None = None
    """Whether PyTorch reported device availability."""
    torch_import_error: str | None = None
    """Torch import error when runtime probing failed."""
    torchvision_distribution_version: str | None = None
    """Installed torchvision distribution version."""
    triton_rocm_distribution_version: str | None = None
    """Installed triton-rocm distribution version."""
    triton_rocm_status: str | None = None
    """Observed triton-rocm status, such as installed or missing."""
    container_rocm_user_space_version: str | None = None
    """Observed container ROCm user-space version."""
    hipcc_version: str | None = None
    """Observed hipcc version output."""
    toolchain_rocm_version: str | None = None
    """Observed parsed ROCm toolchain version."""


class DependencyPreflightResult(BaseModelWithDocstrings):
    """Matrix-compatible dependency preflight classification."""

    model_config = _MODEL_CONFIG

    entry: MatrixEntry
    """Diagnostic Matrix Entry produced by dependency classification."""
    decision: MatrixExecutionDecision
    """Pre-benchmark execution decision derived from the Matrix Entry."""
    policy: PytorchDependencyPolicy
    """Selected Target dependency policy used for classification."""

    def to_preview_payload(self) -> dict[str, Any]:
        """Return shell-consumable JSON for dependency preflight classification."""

        entry_payload = self.entry.model_dump(mode="json")
        decision_payload = self.decision.model_dump(mode="json")
        target_payload = entry_payload["target"]
        policy_payload = entry_payload["observed"]["dependency_policy"]
        claim_payload = entry_payload["claim_boundary"]
        return {
            "target_id": target_payload["target_id"],
            "pytorch_rocm_target": target_payload["pytorch_rocm_target"],
            "policy_id": policy_payload["policy_id"],
            "wheel_availability": self.policy.wheel_availability,
            "expected_local_version": policy_payload["expected_local_version"],
            "uv_index_name": policy_payload["uv_index_name"],
            "uv_index_url": policy_payload["uv_index_url"],
            "lock_strategy": policy_payload["lock_strategy"],
            "suggested_uv_command": policy_payload["suggested_uv_command"],
            "torch_version": self.policy.torch_version,
            "torchvision_version": self.policy.torchvision_version,
            "triton_rocm_version": policy_payload["triton_rocm_version"],
            "triton_rocm_index_name": policy_payload["triton_rocm_index_name"],
            "triton_rocm_index_url": policy_payload["triton_rocm_index_url"],
            "status": entry_payload["status"],
            "reason_code": entry_payload["reason_code"],
            "reason": entry_payload["reason"],
            "benchmark_allowed": decision_payload["benchmark_allowed"],
            "probes_allowed": decision_payload["probes_allowed"],
            "smoke_allowed": decision_payload["smoke_allowed"],
            "score_authority": decision_payload["score_authority"],
            "paper_parity_authority": decision_payload["paper_parity_authority"],
            "leaderboard_authority": decision_payload["leaderboard_authority"],
            "container_user_space_validated": decision_payload[
                "container_user_space_validated"
            ],
            "native_host_validated": decision_payload["native_host_validated"],
            "hardware_validated": claim_payload["hardware_validated"],
        }
