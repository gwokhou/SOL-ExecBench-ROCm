# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only ROCm compatibility Matrix Entry contract."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def _require_location(self) -> MatrixArtifactReference:
        if self.path is None and self.uri is None:
            raise ValueError("MatrixArtifactReference requires path or uri.")
        return self


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
    score_authority: Literal[False] = False
    """Compatibility Matrix Entries never grant benchmark score authority."""
    paper_parity_authority: Literal[False] = False
    """Compatibility Matrix Entries never grant paper-parity authority."""
    leaderboard_authority: Literal[False] = False
    """Compatibility Matrix Entries never grant leaderboard authority."""
    container_user_space_validated: bool
    """Whether container ROCm user-space validation evidence is present."""
    native_host_validated: bool
    """Whether direct native-host validation evidence is present."""
    hardware_validated: bool
    """Whether real hardware evidence for the intended architecture is present."""


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
    claim_boundary: MatrixClaimBoundary
    """Explicit diagnostic-only claim boundaries for this Matrix Entry."""
    artifacts: list[MatrixArtifactReference] = Field(default_factory=list)
    """Artifact references supporting this Matrix Entry."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible Matrix Entry payload."""
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def _validate_claim_boundaries(self) -> MatrixEntry:
        target = self.target
        claims = self.claim_boundary

        if target.validation_scope is MatrixValidationScope.CONTAINER_USER_SPACE:
            if self.status is MatrixCompatibilityStatus.HOST_VALIDATED:
                raise ValueError(
                    "Docker/container scoped Matrix Entries cannot use "
                    "status=host_validated; use container_validated for "
                    "container ROCm user-space validation."
                )
            if claims.native_host_validated:
                raise ValueError(
                    "Docker/container scoped Matrix Entries cannot set "
                    "native_host_validated=true."
                )
            if (
                self.status is MatrixCompatibilityStatus.CONTAINER_VALIDATED
                and not claims.container_user_space_validated
            ):
                raise ValueError(
                    "container_validated Docker/container scoped Matrix Entries "
                    "must set container_user_space_validated=true."
                )

        if self.status is MatrixCompatibilityStatus.CONTAINER_VALIDATED:
            if target.validation_scope is not MatrixValidationScope.CONTAINER_USER_SPACE:
                raise ValueError(
                    "container_validated requires container_user_space "
                    "validation scope."
                )
            if self.observed.container is None:
                raise ValueError(
                    "container_validated requires observed container evidence."
                )
            if not claims.container_user_space_validated:
                raise ValueError(
                    "container_validated requires "
                    "container_user_space_validated=true."
                )
            if claims.native_host_validated:
                raise ValueError(
                    "container_validated cannot set native_host_validated=true."
                )

        if self.status is MatrixCompatibilityStatus.HOST_VALIDATED:
            host = self.observed.host
            has_direct_host_evidence = host is not None and bool(
                host.rocm_version or host.driver_version
            )
            if target.validation_scope is not MatrixValidationScope.NATIVE_HOST:
                raise ValueError(
                    "host_validated requires native_host validation scope."
                )
            if not claims.native_host_validated:
                raise ValueError(
                    "host_validated requires native_host_validated=true."
                )
            if not has_direct_host_evidence:
                raise ValueError(
                    "host_validated requires direct native-host evidence with "
                    "a ROCm or driver version."
                )
            if self.observed.container is not None:
                raise ValueError(
                    "host_validated requires direct native-host evidence, not "
                    "Docker/container validation evidence."
                )
            if claims.container_user_space_validated:
                raise ValueError(
                    "host_validated cannot set "
                    "container_user_space_validated=true."
                )

        if claims.container_user_space_validated and self.observed.container is None:
            raise ValueError(
                "container_user_space_validated requires observed container evidence."
            )
        if claims.native_host_validated and self.observed.host is None:
            raise ValueError(
                "native_host_validated requires observed host evidence."
            )

        return self


class MatrixExecutionDecision(BaseModelWithDocstrings):
    """Pre-benchmark execution decision derived from one Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

    status: MatrixCompatibilityStatusField
    """Matrix Entry status used for the decision."""
    reason_code: MatrixCompatibilityReasonCodeField
    """Matrix Entry reason code used for the decision."""
    benchmark_allowed: bool
    """Whether full benchmark execution is allowed for clean validation."""
    probes_allowed: bool
    """Whether bounded diagnostic probes are allowed."""
    smoke_allowed: bool
    """Whether bounded smoke execution is allowed."""
    reason: str
    """Human-readable decision reason."""
    diagnostic_compatibility_evidence: Literal[True] = True
    """Execution decisions are diagnostic compatibility evidence."""
    score_authority: Literal[False] = False
    """Execution decisions never grant benchmark score authority."""
    paper_parity_authority: Literal[False] = False
    """Execution decisions never grant paper-parity authority."""
    leaderboard_authority: Literal[False] = False
    """Execution decisions never grant leaderboard authority."""
    container_user_space_validated: bool
    """Whether the decision permits a clean container user-space claim."""
    native_host_validated: bool
    """Whether the decision permits a clean native-host claim."""


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

    @model_validator(mode="after")
    def _validate_status_counts(self) -> RocmCompatibilityMatrixReport:
        expected_counts: dict[MatrixCompatibilityStatus, int] = {}
        for entry in self.entries:
            expected_counts[entry.status] = expected_counts.get(entry.status, 0) + 1

        if any(count < 0 for count in self.status_counts.values()):
            raise ValueError("status_counts cannot contain negative counts.")
        if self.status_counts != expected_counts:
            raise ValueError("status_counts must match entries by status.")

        return self


def build_matrix_entry(
    *,
    target: MatrixTarget,
    observed: MatrixObservedEvidence,
    status: MatrixCompatibilityStatus | str,
    reason_code: MatrixCompatibilityReasonCode | str,
    reason: str,
    claim_boundary: MatrixClaimBoundary,
    artifacts: Sequence[MatrixArtifactReference] = (),
) -> MatrixEntry:
    """Build a strict Matrix Entry from explicit Target and evidence inputs."""

    return MatrixEntry(
        target=target,
        observed=observed,
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=claim_boundary,
        artifacts=list(artifacts),
    )


def classify_matrix_entry_for_execution(
    entry: MatrixEntry,
    *,
    allow_mixed_version_debug: bool = False,
) -> MatrixExecutionDecision:
    """Classify a Matrix Entry into deterministic pre-benchmark allowances."""

    status = entry.status
    reason_code = entry.reason_code
    claims = entry.claim_boundary

    if status is MatrixCompatibilityStatus.MIXED_VERSION:
        if allow_mixed_version_debug:
            return MatrixExecutionDecision(
                status=status,
                reason_code=reason_code,
                benchmark_allowed=False,
                probes_allowed=True,
                smoke_allowed=True,
                reason=(
                    "Mixed-version Target is running under explicit debug override; "
                    "diagnostic probes or smoke execution may continue, but clean "
                    "validation and benchmark claims remain blocked."
                ),
                container_user_space_validated=False,
                native_host_validated=False,
            )
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=False,
            smoke_allowed=False,
            reason=(
                "Mixed-version Target is blocked before benchmark execution by "
                "default; rerun only with an explicit debug override for probes "
                "or smoke execution."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Requested dependency stack is unavailable for this Target; this "
                "is a compatibility classification, not a benchmark correctness "
                "failure."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Required ROCm runtime or device access is unavailable; this is a "
                "runtime compatibility classification, not a benchmark correctness "
                "failure."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.NOT_TESTED:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Target is not tested and remains non-authoritative; benchmark "
                "eligibility is not implied."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    return MatrixExecutionDecision(
        status=status,
        reason_code=reason_code,
        benchmark_allowed=True,
        probes_allowed=True,
        smoke_allowed=True,
        reason=entry.reason,
        container_user_space_validated=claims.container_user_space_validated,
        native_host_validated=claims.native_host_validated,
    )
