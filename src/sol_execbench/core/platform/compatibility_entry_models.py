# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility matrix entry, report, and execution decision models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.platform.compatibility_enums import (
    MATRIX_MODEL_CONFIG,
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    MatrixCompatibilityReasonCodeField,
    MatrixCompatibilityStatus,
    MatrixCompatibilityStatusField,
    MatrixValidationScope,
)
from sol_execbench.core.platform.compatibility_evidence_models import (
    MatrixArtifactReference,
    MatrixObservedEvidence,
    MatrixTarget,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


class MatrixClaimBoundary(BaseModelWithDocstrings):
    """Claim boundary fields for a compatibility Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

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

    model_config = MATRIX_MODEL_CONFIG

    schema_version: Literal["sol_execbench.rocm_compatibility_matrix.v1"] = (
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
            if (
                target.validation_scope
                is not MatrixValidationScope.CONTAINER_USER_SPACE
            ):
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
                    "container_validated requires container_user_space_validated=true."
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
                raise ValueError("host_validated requires native_host_validated=true.")
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
                    "host_validated cannot set container_user_space_validated=true."
                )

        if claims.container_user_space_validated and self.observed.container is None:
            raise ValueError(
                "container_user_space_validated requires observed container evidence."
            )
        if claims.native_host_validated and self.observed.host is None:
            raise ValueError("native_host_validated requires observed host evidence.")

        return self


class MatrixExecutionDecision(BaseModelWithDocstrings):
    """Pre-benchmark execution decision derived from one Matrix Entry."""

    model_config = MATRIX_MODEL_CONFIG

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

    model_config = MATRIX_MODEL_CONFIG

    schema_version: Literal["sol_execbench.rocm_compatibility_matrix.v1"] = (
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
