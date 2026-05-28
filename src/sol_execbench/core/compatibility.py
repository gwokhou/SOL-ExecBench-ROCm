# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only ROCm compatibility Matrix Entry contract."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION = (
    "sol_execbench.rocm_compatibility_matrix.v1"
)
_MATRIX_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class MatrixCompatibilityStatus(str, Enum):
    """Bounded compatibility status vocabulary for Matrix Entries."""

    HOST_VALIDATED = "host_validated"
    CONTAINER_VALIDATED = "container_validated"
    MIXED_VERSION = "mixed_version"
    PYTORCH_WHEEL_UNAVAILABLE = "pytorch_wheel_unavailable"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    NOT_TESTED = "not_tested"


class MatrixCompatibilityReasonCode(str, Enum):
    """Stable reason-code vocabulary for compatibility Matrix Entries."""

    HOST_NATIVE_VALIDATED = "host_native_validated"
    CONTAINER_USER_SPACE_VALIDATED = "container_user_space_validated"
    TARGET_OBSERVED_MISMATCH = "target_observed_mismatch"
    PYTORCH_ROCM_WHEEL_UNAVAILABLE = "pytorch_rocm_wheel_unavailable"
    ROCM_RUNTIME_UNAVAILABLE = "rocm_runtime_unavailable"
    TARGET_NOT_TESTED = "target_not_tested"


class MatrixValidationScope(str, Enum):
    """Requested validation scope for a Matrix Target."""

    NATIVE_HOST = "native_host"
    CONTAINER_USER_SPACE = "container_user_space"


def _validate_status(value: object) -> object:
    if isinstance(value, str):
        return MatrixCompatibilityStatus(value)
    return value


def _validate_reason_code(value: object) -> object:
    if isinstance(value, str):
        return MatrixCompatibilityReasonCode(value)
    return value


def _validate_validation_scope(value: object) -> object:
    if isinstance(value, str):
        return MatrixValidationScope(value)
    return value


MatrixCompatibilityStatusField = Annotated[
    MatrixCompatibilityStatus,
    BeforeValidator(_validate_status),
]
MatrixCompatibilityReasonCodeField = Annotated[
    MatrixCompatibilityReasonCode,
    BeforeValidator(_validate_reason_code),
]
MatrixValidationScopeField = Annotated[
    MatrixValidationScope,
    BeforeValidator(_validate_validation_scope),
]


class MatrixArtifactReference(BaseModelWithDocstrings):
    """Artifact reference associated with a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

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


class MatrixTarget(BaseModelWithDocstrings):
    """Requested Target identity for a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

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

    model_config = _MATRIX_MODEL_CONFIG

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

    model_config = _MATRIX_MODEL_CONFIG

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

    model_config = _MATRIX_MODEL_CONFIG

    python_version: str | None = None
    """Observed Python version."""
    torch_version: str | None = None
    """Observed torch.__version__ value."""
    torch_rocm_target: str | None = None
    """Observed PyTorch ROCm wheel target when known."""
    torch_hip_version: str | None = None
    """Observed torch.version.hip value."""
    torch_cuda_version: str | None = None
    """Observed torch.version.cuda value."""
    triton_rocm_status: str | None = None
    """Observed Triton ROCm package or availability status."""


class MatrixToolchainEvidence(BaseModelWithDocstrings):
    """Observed ROCm toolchain evidence for a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

    hipcc_version: str | None = None
    """Observed hipcc version output when available."""
    rocm_agent_enumerator_version: str | None = None
    """Observed rocm_agent_enumerator version when available."""
    rocminfo_version: str | None = None
    """Observed rocminfo version when available."""
    tool_statuses: dict[str, str] = Field(default_factory=dict)
    """Per-tool availability or probe status values."""


class MatrixGpuEvidence(BaseModelWithDocstrings):
    """Observed GPU evidence for a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

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

    model_config = _MATRIX_MODEL_CONFIG

    host: MatrixHostEvidence | None = None
    """Observed host-scope evidence."""
    container: MatrixContainerEvidence | None = None
    """Observed container-scope evidence."""
    python_dependency: MatrixPythonDependencyEvidence | None = None
    """Observed Python dependency evidence."""
    toolchain: MatrixToolchainEvidence | None = None
    """Observed ROCm toolchain evidence."""
    gpu: MatrixGpuEvidence | None = None
    """Observed GPU evidence."""


class MatrixClaimBoundary(BaseModelWithDocstrings):
    """Claim boundary fields for a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

    diagnostic_compatibility_evidence: Literal[True] = True
    """Matrix Entries are diagnostic compatibility evidence."""


class MatrixEntry(BaseModelWithDocstrings):
    """Strict diagnostic compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

    schema_version: Literal[ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION] = (
        ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )
    """Compatibility Matrix Entry schema version."""
    target: MatrixTarget
    """Requested Target values for this Matrix Entry."""
    observed: MatrixObservedEvidence
    """Observed host, container, Python dependency, toolchain, and GPU evidence."""
    status: MatrixCompatibilityStatusField
    """Bounded compatibility status for this Matrix Entry."""
    reason_code: MatrixCompatibilityReasonCodeField
    """Stable reason code explaining the status."""
    reason: str
    """Human-readable diagnostic reason."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible Matrix Entry payload."""
        return self.model_dump(mode="json")


class RocmCompatibilityMatrixReport(BaseModelWithDocstrings):
    """Aggregate ROCm compatibility matrix report."""

    model_config = _MATRIX_MODEL_CONFIG

    schema_version: Literal[ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION] = (
        ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )
    """Compatibility matrix report schema version."""
    generated_at: str
    """UTC timestamp when the report was generated."""
    entries: list[MatrixEntry] = Field(default_factory=list)
    """Compatibility Matrix Entry objects."""
    status_counts: dict[MatrixCompatibilityStatusField, int] = Field(
        default_factory=dict
    )
    """Aggregate counts by bounded compatibility status."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible compatibility matrix report payload."""
        return self.model_dump(mode="json")


def build_matrix_entry(
    *,
    target: MatrixTarget,
    observed: MatrixObservedEvidence,
    status: MatrixCompatibilityStatus | str,
    reason_code: MatrixCompatibilityReasonCode | str,
    reason: str,
    artifacts: Sequence[MatrixArtifactReference] = (),
) -> MatrixEntry:
    """Build a strict Matrix Entry from explicit Target and evidence inputs."""

    del artifacts
    return MatrixEntry(
        target=target,
        observed=observed,
        status=status,
        reason_code=reason_code,
        reason=reason,
    )
