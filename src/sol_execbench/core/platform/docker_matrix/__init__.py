"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

from sol_execbench.core.platform.docker_matrix.cli import main
from sol_execbench.core.platform.docker_matrix.models import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    ROCM_DOCKER_TARGETS_SCHEMA_VERSION,
    DockerPreflightObservation,
    DockerPreflightResult,
    DockerTargetManifest,
    DockerTargetManifestEntry,
    DockerTargetSelection,
)
from sol_execbench.core.platform.docker_matrix.preflight import (
    classify_docker_preflight,
)
from sol_execbench.core.platform.docker_matrix.targets import (
    docker_build_args_for_target,
    load_docker_target_manifest,
    preview_docker_target_selection,
    select_docker_target,
    to_matrix_target,
)

__all__ = [
    "DEFAULT_DOCKER_TARGET_MANIFEST",
    "ROCM_DOCKER_TARGETS_SCHEMA_VERSION",
    "DockerPreflightObservation",
    "DockerPreflightResult",
    "DockerTargetManifest",
    "DockerTargetManifestEntry",
    "DockerTargetSelection",
    "classify_docker_preflight",
    "docker_build_args_for_target",
    "load_docker_target_manifest",
    "main",
    "preview_docker_target_selection",
    "select_docker_target",
    "to_matrix_target",
]


if __name__ == "__main__":
    raise SystemExit(main())
