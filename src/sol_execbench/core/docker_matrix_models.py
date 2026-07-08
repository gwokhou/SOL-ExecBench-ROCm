"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.compatibility import (
    MatrixCompatibilityStatus,
    MatrixEntry,
    MatrixExecutionDecision,
    MatrixValidationScope,
    MatrixValidationScopeField,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


ROCM_DOCKER_TARGETS_SCHEMA_VERSION = "sol_execbench.rocm_docker_targets.v1"


DEFAULT_DOCKER_TARGET_MANIFEST = (
    Path(__file__).resolve().parents[3] / "docker" / "rocm-targets.json"
)


_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class DockerTargetManifestEntry(BaseModelWithDocstrings):
    """Declared Docker Target entry parsed from the repository manifest."""

    model_config = _MODEL_CONFIG

    target_id: str
    """Stable Target id for this Docker ROCm user-space request."""
    requested_rocm_user_space_version: str
    """Requested ROCm user-space version represented by the Docker image."""
    docker_image_repository: str
    """Requested Docker image repository."""
    docker_image_tag: str
    """Requested Docker image tag."""
    pytorch_rocm_target: str | None = None
    """Expected PyTorch ROCm wheel target for later dependency phases."""
    pytorch_dependency_policy: dict[str, str] | None = None
    """Expected PyTorch ROCm wheel/index policy for this Target."""
    validation_scope: MatrixValidationScopeField
    """Validation scope for this Target."""
    intended_gpu_architecture: str | None = None
    """Intended AMD gfx architecture when the Target is architecture-specific."""

    @model_validator(mode="after")
    def _require_container_scope(self) -> DockerTargetManifestEntry:
        if self.validation_scope is not MatrixValidationScope.CONTAINER_USER_SPACE:
            raise ValueError(
                "Docker Target manifest entries must use container_user_space"
            )
        return self


class DockerTargetManifest(BaseModelWithDocstrings):
    """Repository-owned declared Docker Target manifest."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.rocm_docker_targets.v1"]
    """Docker Target manifest schema version."""
    default_target_id: str
    """Target id selected when the user does not pass a Target."""
    targets: list[DockerTargetManifestEntry]
    """Declared Docker Targets; runtime tag discovery is not performed."""

    @property
    def targets_by_id(self) -> dict[str, DockerTargetManifestEntry]:
        """Return declared Targets keyed by id."""

        return {target.target_id: target for target in self.targets}

    @model_validator(mode="after")
    def _validate_targets(self) -> DockerTargetManifest:
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("Docker Target ids must be unique")
        if self.default_target_id not in ids:
            raise ValueError("default_target_id must reference a declared Target")
        return self


class DockerTargetSelection(BaseModelWithDocstrings):
    """Result of declared or explicit unsafe Docker Target selection."""

    model_config = _MODEL_CONFIG

    target_id: str
    """Selected Target id requested by downstream Docker tooling."""
    target: DockerTargetManifestEntry
    """Selected manifest-style Target entry."""
    unknown_override: bool = False
    """Whether this result came from an explicit unsafe/untested override."""
    status: MatrixCompatibilityStatus | None = None
    """Diagnostic Matrix status for unsafe/untested override selections."""
    entry: MatrixEntry | None = None
    """Diagnostic Matrix Entry for unsafe/untested override selections."""
    decision: MatrixExecutionDecision | None = None
    """Execution decision for unsafe/untested override selections."""


class DockerPreflightObservation(BaseModelWithDocstrings):
    """Structured host observations collected before Docker benchmark execution."""

    model_config = _MODEL_CONFIG

    docker_context: str | None = None
    """Observed Docker context name."""
    docker_host: str | None = None
    """Observed Docker daemon host endpoint."""
    dev_kfd_present: bool
    """Whether `/dev/kfd` exists on the host."""
    dev_kfd_accessible: bool
    """Whether `/dev/kfd` is accessible to the preflight probe."""
    dev_dri_present: bool
    """Whether `/dev/dri` exists on the host."""
    dev_dri_accessible: bool
    """Whether `/dev/dri` is accessible to the preflight probe."""
    gpu_accessible: bool | None = None
    """Whether an optional GPU visibility probe could access the device."""
    selected_target: DockerTargetManifestEntry
    """Selected Docker Target for this preflight."""
    image_repository: str
    """Requested image repository for the selected Target."""
    image_tag: str
    """Requested image tag for the selected Target."""
    image_digest: str | None = None
    """Resolved image digest when available; absence is non-blocking evidence."""
    build_args: dict[str, str]
    """Docker build args associated with the selected Target."""
    visible_device_environment: dict[str, str] = Field(default_factory=dict)
    """Observed GPU visibility environment values."""


class DockerPreflightResult(BaseModelWithDocstrings):
    """Matrix-compatible classification for one Docker preflight observation."""

    model_config = _MODEL_CONFIG

    entry: MatrixEntry
    """Diagnostic Matrix Entry produced by preflight classification."""
    decision: MatrixExecutionDecision
    """Pre-benchmark execution decision derived from the Matrix Entry."""
    build_args: dict[str, str]
    """Docker build args associated with the selected Target."""

    def to_preview_payload(self) -> dict[str, Any]:
        """Return shell-consumable JSON for preflight classification."""

        entry_payload = self.entry.model_dump(mode="json")
        decision_payload = self.decision.model_dump(mode="json")
        target_payload = entry_payload["target"]
        container_payload = entry_payload["observed"]["container"]
        return {
            "target_id": target_payload["target_id"],
            "validation_scope": target_payload["validation_scope"],
            "image_repository": container_payload["image_repository"],
            "image_tag": container_payload["image_tag"],
            "image_digest": container_payload["image_digest"],
            "build_args": self.build_args,
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
        }
