from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle.source_review import (
    DiagnosticSourceReview,
    load_and_verify_source_review,
)
from sol_execbench.core.bench.performance_model.source_transition import (
    SourcePathStageImpact,
    SourceTransitionStage,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_bytes


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout


def _reviewed_repository(tmp_path: Path) -> tuple[Path, DiagnosticSourceReview]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test Author")
    _git(root, "config", "user.email", "test@example.com")
    (root / "design.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "design.py")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD").decode().strip()
    (root / "design.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "governance.py").write_text("REVIEWED = True\n", encoding="utf-8")
    _git(root, "add", "design.py", "governance.py")
    _git(root, "commit", "-q", "-m", "target")
    target = _git(root, "rev-parse", "HEAD").decode().strip()
    patch = _git(root, "diff", "--binary", "--full-index", base, target, "--")
    review = DiagnosticSourceReview(
        base_source_revision=base,
        target_source_revision=target,
        git_patch_sha256=sha256_bytes(patch),
        source_changes=(
            SourcePathStageImpact(
                path="design.py",
                change="modified",
                affected_stages=(SourceTransitionStage.DESIGN,),
                rationale="Changes the authored design.",
            ),
            SourcePathStageImpact(
                path="governance.py",
                change="added",
                affected_stages=(
                    SourceTransitionStage.GOVERNANCE_CONTROL_PLANE,
                ),
                rationale="Adds the reviewed transition tool.",
            ),
        ),
        created_at="2026-08-12T00:00:00+00:00",
    )
    return root, review


def test_source_review_verifies_exact_diff(tmp_path: Path) -> None:
    root, review = _reviewed_repository(tmp_path)
    path = tmp_path / "review.json"
    atomic_write_json_value(path, review.model_dump(mode="json"))

    observed = load_and_verify_source_review(path, repository_root=root)

    assert observed == review
    assert observed.affects(SourceTransitionStage.DESIGN)
    assert not observed.affects(SourceTransitionStage.CALIBRATION)


def test_source_review_rejects_incomplete_path_inventory(
    tmp_path: Path,
) -> None:
    root, review = _reviewed_repository(tmp_path)
    path = tmp_path / "review.json"
    payload = review.model_dump(mode="json")
    payload["source_changes"] = payload["source_changes"][:-1]
    atomic_write_json_value(path, payload)

    with pytest.raises(ValueError, match="does not exactly cover"):
        load_and_verify_source_review(path, repository_root=root)
