# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact Git-diff reviews for stage-scoped diagnostic source transitions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import Field, model_validator

from sol_execbench.core.bench.performance_model.source_transition import (
    SourcePathStageImpact,
    SourceTransitionStage,
)
from sol_execbench.core.data.base_model import (
    FrozenArtifactModel,
    NonEmptyString,
)
from sol_execbench.core.data.json_utils import load_json_file
from sol_execbench.core.integrity import SHA256Digest, sha256_bytes


class DiagnosticSourceReview(FrozenArtifactModel):
    """One exact, human-classified Git transition used by control-plane tools."""

    base_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_patch_sha256: SHA256Digest
    source_changes: tuple[SourcePathStageImpact, ...] = Field(min_length=1)
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _changes_are_canonical(self) -> DiagnosticSourceReview:
        paths = tuple(item.path for item in self.source_changes)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source review paths must be sorted and unique")
        if self.base_source_revision == self.target_source_revision:
            raise ValueError("source review revisions must differ")
        return self

    def affects(self, stage: SourceTransitionStage) -> bool:
        """Return whether any exact changed path is classified for *stage*."""
        return any(
            stage in item.affected_stages for item in self.source_changes
        )


def _run_git(
    repository_root: Path,
    arguments: list[str],
    *,
    text: bool = False,
) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=text,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr if text else result.stderr.decode(errors="replace")
        )
        raise ValueError(f"git command failed: {stderr.strip()}")
    return result.stdout


def _exact_revision(repository_root: Path, revision: str) -> None:
    observed = str(
        _run_git(
            repository_root,
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            text=True,
        )
    ).strip()
    if observed != revision:
        raise ValueError(f"source revision is not exact: {revision}")


def _change_identity(line: str) -> tuple[str, str | None, str]:
    fields = line.split("\t")
    status = fields[0][0]
    change = {
        "A": "added",
        "D": "deleted",
        "M": "modified",
        "R": "renamed",
    }.get(status)
    if change is None:
        raise ValueError(f"unsupported Git change status: {fields[0]}")
    if status == "R" and len(fields) == 3:
        return change, fields[1], fields[2]
    if len(fields) != 2:
        raise ValueError(f"malformed Git change record: {line}")
    return change, None, fields[1]


def load_and_verify_source_review(
    path: Path,
    *,
    repository_root: Path,
) -> DiagnosticSourceReview:
    """Load a review and prove that it covers the exact binary Git diff."""
    review = load_json_file(DiagnosticSourceReview, path)
    root = repository_root.resolve()
    _exact_revision(root, review.base_source_revision)
    _exact_revision(root, review.target_source_revision)
    output = str(
        _run_git(
            root,
            [
                "diff",
                "--name-status",
                "--find-renames",
                review.base_source_revision,
                review.target_source_revision,
                "--",
            ],
            text=True,
        )
    )
    actual = tuple(
        sorted(
            (_change_identity(line) for line in output.splitlines()),
            key=lambda item: item[2],
        )
    )
    declared = tuple(
        (item.change, item.previous_path, item.path)
        for item in review.source_changes
    )
    if actual != declared:
        raise ValueError("source review does not exactly cover the Git diff")
    patch = _run_git(
        root,
        [
            "diff",
            "--binary",
            "--full-index",
            review.base_source_revision,
            review.target_source_revision,
            "--",
        ],
    )
    if not isinstance(patch, bytes):
        raise TypeError("binary Git diff unexpectedly returned text")
    if sha256_bytes(patch) != review.git_patch_sha256:
        raise ValueError("source review binary patch digest differs")
    return review


__all__ = ["DiagnosticSourceReview", "load_and_verify_source_review"]
