# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Signed execution-environment identity for release measurements."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol_execbench.core.platform.source_state import (
    GitSourceState,
    verify_git_source_state,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID,
)
from sol_execbench.core.scoring.schema_versions import (
    ReleaseArtifactSchema,
    ReleaseComponentKind,
)

CONTAINER_IMAGE_ID_ENV = ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID
_CONTAINER_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_RELEASE_SOURCE_PATHS = (
    "docker",
    "problems",
    "pyproject.toml",
    "scripts",
    "src",
    "uv.lock",
)


@dataclass(frozen=True, slots=True)
class ReleaseExecutionIdentity:
    """Committed source and immutable container used by one release run."""

    source_revision: str
    container_image_id: str

    def to_dict(self) -> dict[str, object]:
        """Return the strict nested environment payload."""
        return {
            "schema_version": ReleaseArtifactSchema.RELEASE_COMPONENT,
            "artifact_kind": ReleaseComponentKind.ENVIRONMENT,
            "source_revision": self.source_revision,
            "source_tree_clean": True,
            "container_image_id": self.container_image_id,
        }


def current_release_execution_identity(
    source_revision: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReleaseExecutionIdentity:
    """Read the host-resolved immutable image identity passed to the container."""
    environment = os.environ if environ is None else environ
    image_id = environment.get(CONTAINER_IMAGE_ID_ENV, "")
    return _validated_identity(source_revision, image_id)


def verify_release_source_state(
    repository_root: Path,
    *,
    expected_revision: str,
) -> GitSourceState:
    """Require every release-relevant source path to match one commit."""
    return verify_git_source_state(
        repository_root,
        expected_revision=expected_revision,
        paths=_RELEASE_SOURCE_PATHS,
    )


def release_execution_identity_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_source_revision: str,
) -> ReleaseExecutionIdentity:
    """Parse a release environment extension and bind it to its statement."""
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        raise ValueError("release environment payload must be an object")
    raw = data.get("release_execution")
    if not isinstance(raw, Mapping):
        raise ValueError("release environment lacks execution identity")
    if (
        raw.get("schema_version") != ReleaseArtifactSchema.RELEASE_COMPONENT
        or raw.get("artifact_kind") != ReleaseComponentKind.ENVIRONMENT
        or raw.get("source_tree_clean") is not True
    ):
        raise ValueError("release execution environment contract mismatch")
    source_revision = str(raw.get("source_revision", ""))
    if source_revision != expected_source_revision:
        raise ValueError("release environment source revision mismatch")
    return _validated_identity(
        source_revision,
        str(raw.get("container_image_id", "")),
    )


def _validated_identity(
    source_revision: str,
    container_image_id: str,
) -> ReleaseExecutionIdentity:
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("release environment source revision is invalid")
    if _CONTAINER_IMAGE_ID.fullmatch(container_image_id) is None:
        raise ValueError(
            f"{CONTAINER_IMAGE_ID_ENV} must be an immutable sha256 image ID",
        )
    return ReleaseExecutionIdentity(source_revision, container_image_id)


__all__ = [
    "CONTAINER_IMAGE_ID_ENV",
    "ReleaseExecutionIdentity",
    "current_release_execution_identity",
    "release_execution_identity_from_payload",
    "verify_release_source_state",
]
