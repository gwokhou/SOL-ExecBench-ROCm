# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Artifact citation helpers for agent feedback sidecars."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackArtifactCitation,
)
from sol_execbench.core.evidence.checksums import sha256_file


def artifact_citation_from_path(
    *,
    kind: str,
    path: Path,
    label: str | None = None,
    status: str | None = None,
    sha256: str | None = None,
) -> AgentFeedbackArtifactCitation:
    """Build a compact citation from an artifact path."""

    checksum = (
        sha256
        if sha256 is not None
        else (sha256_file(path) if path.is_file() else None)
    )
    return AgentFeedbackArtifactCitation(
        kind=kind,
        label=label or path.name,
        path=path.name,
        sha256=checksum,
        status=status,
    )
