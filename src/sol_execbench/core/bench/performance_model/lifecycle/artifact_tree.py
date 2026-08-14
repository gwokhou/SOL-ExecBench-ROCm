# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact-tree manifests stored as content-addressed blobs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.schema_versions import (
    DiagnosticLifecycleSchema,
    DiagnosticLifecycleStateKind,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import canonical_json_bytes


class DiagnosticArtifactTreeManifest(CurrentFrozenSchemaModel):
    """One entry manifest and every relative file needed to consume it."""

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    )
    current_artifact_kind = DiagnosticLifecycleStateKind.ARTIFACT_TREE

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    artifact_kind: Literal[DiagnosticLifecycleStateKind.ARTIFACT_TREE] = (
        DiagnosticLifecycleStateKind.ARTIFACT_TREE
    )
    root_path: str = Field(min_length=1)
    artifacts: tuple[DiagnosticLifecycleArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _inventory_is_canonical(self) -> DiagnosticArtifactTreeManifest:
        paths = tuple(item.relative_path for item in self.artifacts)
        if any(not _safe_relative_path(path) for path in paths):
            raise ValueError(
                "artifact-tree members require safe relative paths"
            )
        if not _safe_relative_path(self.root_path):
            raise ValueError("artifact-tree root requires a safe relative path")
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError(
                "artifact-tree inventory must be sorted and unique"
            )
        if self.root_path not in paths:
            raise ValueError("artifact-tree root_path is absent from inventory")
        return self


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def import_artifact_tree(
    *,
    root: Path,
    root_path: Path,
    member_paths: tuple[Path, ...],
    store: BlobStore,
) -> tuple[str, DiagnosticArtifactTreeManifest]:
    """Verify and import one exact regular-file tree into *store*."""
    resolved_root = root.resolve()
    if root_path.is_symlink():
        raise ValueError("artifact-tree root is not regular")
    resolved_entry = root_path.resolve()
    if not resolved_entry.is_relative_to(resolved_root):
        raise ValueError("artifact-tree root escapes its source root")
    artifacts: list[DiagnosticLifecycleArtifact] = []
    resolved_members: dict[str, Path] = {}
    for member_path in member_paths:
        if member_path.is_symlink():
            raise ValueError(
                f"artifact-tree member is not regular: {member_path}"
            )
        path = member_path.resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("artifact-tree member escapes its source root")
        if not path.is_file():
            raise ValueError(f"artifact-tree member is not regular: {path}")
        relative = path.relative_to(resolved_root).as_posix()
        if relative in resolved_members:
            raise ValueError("artifact-tree member paths must be unique")
        resolved_members[relative] = path
    for relative, path in sorted(resolved_members.items()):
        digest = store.put_file(path)
        artifacts.append(
            DiagnosticLifecycleArtifact(
                relative_path=relative,
                sha256=digest,
                size_bytes=path.stat().st_size,
            )
        )
    manifest = DiagnosticArtifactTreeManifest(
        root_path=resolved_entry.relative_to(resolved_root).as_posix(),
        artifacts=tuple(artifacts),
    )
    digest = store.put_bytes(
        canonical_json_bytes(manifest.model_dump(mode="json"))
    )
    return digest, manifest


__all__ = ["DiagnosticArtifactTreeManifest", "import_artifact_tree"]
