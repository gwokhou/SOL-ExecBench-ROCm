# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility matrix target and observed evidence models."""

from __future__ import annotations

from pydantic import Field, model_validator

from sol_execbench.core.platform.compatibility_enums import (
    MATRIX_MODEL_CONFIG,
    MatrixValidationScopeField,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


class MatrixArtifactReference(BaseModelWithDocstrings):
    """Artifact reference associated with a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    artifact_id: str
    """Stable artifact identifier within the Matrix Entry."""
    kind: str
    """Artifact kind, such as docker_log, probe_json, or transcript."""
    path: str | None = None
    """Local artifact path when available."""
    uri: str | None = None
    """Remote artifact URI when available."""
    description: str = ""
    """Human-readable artifact description."""

    @model_validator(mode="after")
    def _require_location(self) -> MatrixArtifactReference:
        if self.path is None and self.uri is None:
            raise ValueError("MatrixArtifactReference requires path or uri.")
        return self


class MatrixTarget(BaseModelWithDocstrings):
    """Requested Target identity for a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    target_id: str
    """Stable Target identifier for the requested validation configuration."""
    requested_rocm_user_space_version: str
    """Requested ROCm user-space version for this Target."""
    docker_image_repository: str | None = None
    """Requested Docker image repository when validation uses a container."""
    docker_image_tag: str | None = None
    """Requested Docker image tag when validation uses a container."""
    pytorch_rocm_target: str | None = None
    """Requested PyTorch ROCm Target, such as rocm7.1."""
    validation_scope: MatrixValidationScopeField
    """Requested validation scope for this Target."""
    intended_gpu_architecture: str | None = None
    """Intended AMD gfx architecture for this Target when known."""


class MatrixHostEvidence(BaseModelWithDocstrings):
    """Observed host-scope ROCm and driver evidence."""

    model_config = MATRIX_MODEL_CONFIG

    rocm_version: str | None = None
    """Observed host ROCm version when detected."""
    driver_version: str | None = None
    """Observed host driver or kernel driver version when detected."""
    device_nodes: list[str] = Field(default_factory=list)
    """Observed host device nodes relevant to ROCm, such as /dev/kfd."""
    source: str | None = None
    """Probe source that produced the host evidence."""


class MatrixContainerEvidence(BaseModelWithDocstrings):
    """Observed container-scope ROCm user-space evidence."""

    model_config = MATRIX_MODEL_CONFIG

    rocm_user_space_version: str | None = None
    """Observed ROCm user-space version inside the container."""
    image_repository: str | None = None
    """Observed container image repository."""
    image_tag: str | None = None
    """Observed container image tag."""
    image_digest: str | None = None
    """Resolved image digest when available."""


class MatrixPythonDependencyEvidence(BaseModelWithDocstrings):
    """Observed Python dependency evidence for a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    python_version: str | None = None
    """Observed Python version."""
    torch_distribution_version: str | None = None
    """Observed installed torch distribution version."""
    torch_version: str | None = None
    """Observed torch.__version__ value."""
    torch_local_version: str | None = None
    """Observed torch local-version tag, such as rocm7.1."""
    torch_rocm_target: str | None = None
    """Observed PyTorch ROCm wheel target when known."""
    torch_hip_version: str | None = None
    """Observed torch.version.hip value."""
    torch_cuda_version: str | None = None
    """Observed torch.version.cuda value."""
    torch_device_available: bool | None = None
    """Whether PyTorch reported ROCm/CUDA device availability."""
    torch_import_error: str | None = None
    """Torch import error when runtime probing failed."""
    torchvision_distribution_version: str | None = None
    """Observed installed torchvision distribution version."""
    triton_rocm_distribution_version: str | None = None
    """Observed installed triton-rocm distribution version."""
    triton_rocm_status: str | None = None
    """Observed Triton ROCm package or availability status."""


class MatrixDependencyPolicyEvidence(BaseModelWithDocstrings):
    """Declared PyTorch ROCm dependency policy for a Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    policy_id: str
    """Stable policy identifier tied to the selected Target."""
    expected_local_version: str
    """Expected PyTorch ROCm local-version tag, such as rocm7.1."""
    uv_index_name: str
    """uv index name expected to provide the PyTorch ROCm wheels."""
    uv_index_url: str
    """uv index URL expected to provide the PyTorch ROCm wheels."""
    lock_strategy: str
    """Dependency lock or workflow strategy for this Target policy."""
    suggested_uv_command: str
    """Auditable uv command or workflow users can run for this policy."""
    triton_rocm_version: str
    """Expected triton-rocm distribution version."""
    triton_rocm_index_name: str
    """uv index name expected to provide triton-rocm."""
    triton_rocm_index_url: str
    """uv index URL expected to provide triton-rocm."""


class MatrixToolchainEvidence(BaseModelWithDocstrings):
    """Observed ROCm toolchain evidence for a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    hipcc_version: str | None = None
    """Observed hipcc version output when available."""
    toolchain_rocm_version: str | None = None
    """Observed ROCm toolchain version when parsed separately."""
    rocm_agent_enumerator_version: str | None = None
    """Observed rocm_agent_enumerator version when available."""
    rocminfo_version: str | None = None
    """Observed rocminfo version when available."""
    tool_statuses: dict[str, str] = Field(default_factory=dict)
    """Per-tool availability or probe status values."""


class MatrixGpuEvidence(BaseModelWithDocstrings):
    """Observed GPU evidence for a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

    device_count: int | None = Field(default=None, ge=0)
    """Observed GPU device count when known."""
    device_name: str | None = None
    """Observed GPU device name when known."""
    gfx_architecture: str | None = None
    """Observed AMD gfx architecture such as gfx1200 or gfx942."""
    visible_device_environment: dict[str, str] = Field(default_factory=dict)
    """Observed GPU visibility environment variables."""


class MatrixObservedEvidence(BaseModelWithDocstrings):
    """Observed evidence separated from requested Matrix Target values."""

    model_config = MATRIX_MODEL_CONFIG

    host: MatrixHostEvidence | None = None
    """Observed host-scope evidence."""
    container: MatrixContainerEvidence | None = None
    """Observed container-scope evidence."""
    python_dependency: MatrixPythonDependencyEvidence | None = None
    """Observed Python dependency evidence."""
    dependency_policy: MatrixDependencyPolicyEvidence | None = None
    """Declared PyTorch ROCm dependency policy for the selected Target."""
    toolchain: MatrixToolchainEvidence | None = None
    """Observed ROCm toolchain evidence."""
    gpu: MatrixGpuEvidence | None = None
    """Observed GPU evidence."""
