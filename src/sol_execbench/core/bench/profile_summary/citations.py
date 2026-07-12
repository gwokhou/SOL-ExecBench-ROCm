# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Artifact citation helpers for profile summary sidecars."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryArtifactCitation,
)
from sol_execbench.core.integrity.checksums import sha256_file


def profile_summary_artifact_citation_from_path(
    *,
    kind: str,
    path: Path,
    label: str | None = None,
    status: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ProfileSummaryArtifactCitation:
    """Build a compact citation from a profile-summary artifact path."""

    checksum = (
        sha256
        if sha256 is not None
        else (sha256_file(path) if path.is_file() else None)
    )
    return ProfileSummaryArtifactCitation(
        kind=kind,
        label=label or path.name,
        path=path.name,
        sha256=checksum,
        status=status,
        size_bytes=size_bytes,
    )
