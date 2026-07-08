"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sol_execbench.core.platform.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixObservedEvidence,
    MatrixTarget,
    MatrixValidationScope,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.platform.docker_matrix_models import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    DockerTargetManifest,
    DockerTargetManifestEntry,
    DockerTargetSelection,
)


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

    args = {
        "ROCM_DOCKER_IMAGE": target.docker_image_repository,
        "ROCM_DOCKER_TAG": target.docker_image_tag,
    }
    policy = target.pytorch_dependency_policy or {}
    if policy:
        args.update(
            {
                "PYTORCH_TORCH_VERSION": policy.get("torch_version", ""),
                "PYTORCH_TORCHVISION_VERSION": policy.get("torchvision_version", ""),
                "PYTORCH_ROCM_INDEX_URL": policy.get("uv_index_url", ""),
                "TRITON_ROCM_VERSION": policy.get("triton_rocm_version", ""),
                "TRITON_ROCM_INDEX_URL": policy.get("triton_rocm_index_url", ""),
            }
        )
    return args


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
