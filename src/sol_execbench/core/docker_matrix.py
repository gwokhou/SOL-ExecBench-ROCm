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
    MatrixEntry,
    MatrixExecutionDecision,
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
    pytorch_dependency_policy: dict[str, str] | None = None
    """Expected PyTorch ROCm wheel/index policy for this Target."""
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
    preview_selection = _not_tested_selection(
        selection.target,
        reason=(
            "Declared Docker Target selection preview is diagnostic only; no "
            "runtime, container user-space, host, hardware, or benchmark "
            "validation has been performed."
        ),
        unknown_override=False,
    )
    assert preview_selection.entry is not None
    assert preview_selection.decision is not None
    return preview_selection.entry, preview_selection.decision


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
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST)
    preflight.add_argument("--target")
    preflight.add_argument("--docker-context")
    preflight.add_argument("--docker-host")
    preflight.add_argument("--dev-kfd-present", required=True, type=_parse_bool)
    preflight.add_argument("--dev-kfd-accessible", required=True, type=_parse_bool)
    preflight.add_argument("--dev-dri-present", required=True, type=_parse_bool)
    preflight.add_argument("--dev-dri-accessible", required=True, type=_parse_bool)
    preflight.add_argument("--gpu-accessible", type=_parse_bool)
    preflight.add_argument("--image-digest")
    return parser


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


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
    if args.command == "preflight":
        selection = select_docker_target(args.target, manifest_path=args.manifest)
        observation = DockerPreflightObservation(
            docker_context=args.docker_context,
            docker_host=args.docker_host,
            dev_kfd_present=args.dev_kfd_present,
            dev_kfd_accessible=args.dev_kfd_accessible,
            dev_dri_present=args.dev_dri_present,
            dev_dri_accessible=args.dev_dri_accessible,
            gpu_accessible=args.gpu_accessible,
            selected_target=selection.target,
            image_repository=selection.target.docker_image_repository,
            image_tag=selection.target.docker_image_tag,
            image_digest=args.image_digest,
            build_args=docker_build_args_for_target(selection.target),
        )
        payload = classify_docker_preflight(observation).to_preview_payload()
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
