# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic dataset acquisition and layout manifest models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .checksums import stable_json_checksum
from .layout import LayoutCategory, LayoutDiagnostic, inspect_dataset_layout
from sol_execbench.core.utils import utc_timestamp

MANIFEST_SCHEMA_VERSION = "sol_execbench.dataset_manifest.v1"


class DatasetManifestSource(BaseModel):
    """Source metadata for an acquisition or local-layout manifest."""

    kind: str = "huggingface_dataset"
    repo_id: str = "nvidia/SOL-ExecBench"
    revision: str | None = None
    local_provenance: str | None = None


class DatasetManifestRoot(BaseModel):
    """Dataset root path metadata."""

    path: str
    path_kind: str = "relative_or_absolute"


class DatasetManifestClaimBoundary(BaseModel):
    """Machine-readable claim boundary for Phase 53 artifacts."""

    acquisition_or_layout_complete: bool
    rocm_readiness: bool = False
    execution_success: bool = False
    paper_level_validation: bool = False
    hosted_leaderboard_parity: bool = False
    upstream_solar_equivalence: bool = False


class DatasetManifestChecksum(BaseModel):
    """Checksum metadata."""

    algorithm: str = "sha256"
    value: str


class DatasetManifest(BaseModel):
    """Sidecar-only dataset acquisition and layout manifest."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    created_at: str
    source: DatasetManifestSource
    root: DatasetManifestRoot
    selected_categories: tuple[str, ...]
    categories: tuple[LayoutCategory, ...]
    diagnostics: tuple[LayoutDiagnostic, ...]
    claim_boundary: DatasetManifestClaimBoundary
    manifest_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetManifest":
        """Return a copy with a stable checksum over the manifest payload."""

        payload = self.model_dump(mode="json")
        payload["manifest_checksum"] = None
        checksum = stable_json_checksum(payload)
        return self.model_copy(
            update={
                "manifest_checksum": DatasetManifestChecksum(value=checksum),
            }
        )

    def to_json(self) -> str:
        """Serialize deterministically for sidecar output."""

        return (
            json.dumps(
                self.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def build_dataset_manifest(
    root: Path,
    *,
    categories: tuple[str, ...] | None = None,
    source: DatasetManifestSource | None = None,
    created_at: str | None = None,
) -> DatasetManifest:
    """Build a deterministic dataset manifest from local layout inspection."""

    layout = inspect_dataset_layout(root, categories=categories)
    manifest = DatasetManifest(
        created_at=created_at or utc_timestamp(),
        source=source or DatasetManifestSource(local_provenance="local_layout"),
        root=DatasetManifestRoot(path=Path(root).as_posix()),
        selected_categories=layout.selected_categories,
        categories=layout.categories,
        diagnostics=layout.diagnostics,
        claim_boundary=DatasetManifestClaimBoundary(
            acquisition_or_layout_complete=layout.ok,
        ),
    )
    return manifest.with_checksum()


def write_dataset_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write *manifest* as deterministic JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
