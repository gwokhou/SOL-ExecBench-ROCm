"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixTarget,
    MatrixValidationScope,
    MatrixValidationScopeField,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
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
    validation_scope: MatrixValidationScopeField
    """Validation scope for this Target."""
    intended_gpu_architecture: str | None = None
    """Intended AMD gfx architecture when the Target is architecture-specific."""

    @model_validator(mode="after")
    def _require_container_scope(self) -> DockerTargetManifestEntry:
        if self.validation_scope is not MatrixValidationScope.CONTAINER_USER_SPACE:
            raise ValueError("Docker Target manifest entries must use container_user_space")
        return self


class DockerTargetManifest(BaseModelWithDocstrings):
    """Repository-owned declared Docker Target manifest."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[ROCM_DOCKER_TARGETS_SCHEMA_VERSION]
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
    entry: object | None = None
    """Diagnostic Matrix Entry for unsafe/untested override selections."""
    decision: object | None = None
    """Execution decision for unsafe/untested override selections."""


def load_docker_target_manifest(
    path: str | Path = DEFAULT_DOCKER_TARGET_MANIFEST,
) -> DockerTargetManifest:
    """Load and validate the checked-in Docker Target manifest."""

    payload = json.loads(Path(path).read_text())
    return DockerTargetManifest.model_validate(payload)


def to_matrix_target(target: DockerTargetManifestEntry) -> MatrixTarget:
    """Convert a declared Docker Target entry into the Phase 78 Matrix Target."""

    return MatrixTarget(
        target_id=target.target_id,
        requested_rocm_user_space_version=target.requested_rocm_user_space_version,
        docker_image_repository=target.docker_image_repository,
        docker_image_tag=target.docker_image_tag,
        pytorch_rocm_target=target.pytorch_rocm_target,
        validation_scope=target.validation_scope,
        intended_gpu_architecture=target.intended_gpu_architecture,
    )


def docker_build_args_for_target(
    target: DockerTargetManifestEntry,
) -> dict[str, str]:
    """Return Dockerfile build args for the selected ROCm base image."""

    return {
        "ROCM_DOCKER_IMAGE": target.docker_image_repository,
        "ROCM_DOCKER_TAG": target.docker_image_tag,
    }


def _unsafe_override_entry(
    *,
    target_id: str,
    image_repository: str,
    image_tag: str,
) -> DockerTargetManifestEntry:
    return DockerTargetManifestEntry(
        target_id=f"unsafe-untested-{target_id}",
        requested_rocm_user_space_version="unknown",
        docker_image_repository=image_repository,
        docker_image_tag=image_tag,
        pytorch_rocm_target=None,
        validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
        intended_gpu_architecture=None,
    )


def _not_tested_selection(
    target: DockerTargetManifestEntry,
    *,
    reason: str,
    unknown_override: bool,
) -> DockerTargetSelection:
    matrix_target = to_matrix_target(target)
    entry = build_matrix_entry(
        target=matrix_target,
        observed=MatrixObservedEvidence(
            container=MatrixContainerEvidence(
                image_repository=target.docker_image_repository,
                image_tag=target.docker_image_tag,
                image_digest=None,
            )
        ),
        status=MatrixCompatibilityStatus.NOT_TESTED,
        reason_code=MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
    decision = classify_matrix_entry_for_execution(entry)
    return DockerTargetSelection(
        target_id=target.target_id,
        target=target,
        unknown_override=unknown_override,
        status=MatrixCompatibilityStatus.NOT_TESTED,
        entry=entry,
        decision=decision,
    )


def select_docker_target(
    target_id: str | None = None,
    *,
    manifest_path: str | Path = DEFAULT_DOCKER_TARGET_MANIFEST,
    allow_unknown_override: bool = False,
    override_image_repository: str | None = None,
    override_image_tag: str | None = None,
) -> DockerTargetSelection:
    """Select a declared Docker Target or an explicit unsafe/untested override."""

    manifest = load_docker_target_manifest(manifest_path)
    selected_id = target_id or manifest.default_target_id
    target = manifest.targets_by_id.get(selected_id)
    if target is not None:
        return DockerTargetSelection(target_id=selected_id, target=target)

    if not allow_unknown_override:
        raise ValueError(
            f"Unknown Docker Target {selected_id!r}; choose a declared Target or "
            "pass an explicit unsafe/untested override."
        )
    if not override_image_repository or not override_image_tag:
        raise ValueError(
            "Unknown Docker Target overrides require override_image_repository "
            "and override_image_tag."
        )

    override_target = _unsafe_override_entry(
        target_id=selected_id,
        image_repository=override_image_repository,
        image_tag=override_image_tag,
    )
    return _not_tested_selection(
        override_target,
        reason=(
            f"Unknown Docker Target {selected_id!r} was selected only through an "
            "explicit unsafe/untested override; benchmark eligibility and clean "
            "validation claims remain blocked."
        ),
        unknown_override=True,
    )


def _selection_entry_for_preview(selection: DockerTargetSelection):
    if selection.entry is not None and selection.decision is not None:
        return selection.entry, selection.decision
    return _not_tested_selection(
        selection.target,
        reason=(
            "Declared Docker Target selection preview is diagnostic only; no "
            "runtime, container user-space, host, hardware, or benchmark "
            "validation has been performed."
        ),
        unknown_override=False,
    ).entry, _not_tested_selection(
        selection.target,
        reason=(
            "Declared Docker Target selection preview is diagnostic only; no "
            "runtime, container user-space, host, hardware, or benchmark "
            "validation has been performed."
        ),
        unknown_override=False,
    ).decision


def preview_docker_target_selection(
    target_id: str | None = None,
    *,
    manifest_path: str | Path = DEFAULT_DOCKER_TARGET_MANIFEST,
    allow_unknown_override: bool = False,
    override_image_repository: str | None = None,
    override_image_tag: str | None = None,
    image_digest: str | None = None,
) -> dict[str, Any]:
    """Return shell-consumable JSON data for Docker Target selection."""

    selection = select_docker_target(
        target_id,
        manifest_path=manifest_path,
        allow_unknown_override=allow_unknown_override,
        override_image_repository=override_image_repository,
        override_image_tag=override_image_tag,
    )
    entry, decision = _selection_entry_for_preview(selection)
    payload = entry.model_dump(mode="json")
    build_args = docker_build_args_for_target(selection.target)
    return {
        "target_id": selection.target_id,
        "validation_scope": selection.target.validation_scope.value,
        "image_repository": selection.target.docker_image_repository,
        "image_tag": selection.target.docker_image_tag,
        "image_digest": image_digest,
        "build_args": build_args,
        "status": payload["status"],
        "reason_code": payload["reason_code"],
        "reason": payload["reason"],
        "benchmark_allowed": decision.benchmark_allowed,
        "probes_allowed": decision.probes_allowed,
        "smoke_allowed": decision.smoke_allowed,
        "score_authority": decision.score_authority,
        "paper_parity_authority": decision.paper_parity_authority,
        "leaderboard_authority": decision.leaderboard_authority,
        "container_user_space_validated": decision.container_user_space_validated,
        "native_host_validated": decision.native_host_validated,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    preview.add_argument("--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST)
    preview.add_argument("--target")
    preview.add_argument("--allow-unknown-target", action="store_true")
    preview.add_argument("--override-image-repository")
    preview.add_argument("--override-image-tag")
    preview.add_argument("--image-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Emit shell-consumable Docker Matrix JSON."""

    args = _build_parser().parse_args(argv)
    if args.command == "preview":
        payload = preview_docker_target_selection(
            target_id=args.target,
            manifest_path=args.manifest,
            allow_unknown_override=args.allow_unknown_target,
            override_image_repository=args.override_image_repository,
            override_image_tag=args.override_image_tag,
            image_digest=args.image_digest,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
