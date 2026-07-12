"""Regression checks for the evidence lifecycle permission boundaries."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github/workflows"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_prepare_workflow_is_manual_and_read_only() -> None:
    workflow = _workflow("evidence-prepare.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "runs-on: [self-hosted, evidence-producer]" in workflow


def test_publish_workflow_requires_review_and_round_trip_verification() -> None:
    workflow = _workflow("evidence-publish.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "name: evidence-publish" in workflow
    assert '"required_reviewers"' in workflow
    assert ".prevent_self_review == true" in workflow
    assert "attestations: write" in workflow
    assert "id-token: write" in workflow
    assert "Refuse to mutate an existing release" in workflow
    assert "Download public release and verify it again" in workflow


def test_revocation_is_manual_reviewed_and_never_deletes_assets() -> None:
    workflow = _workflow("evidence-revoke.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "name: evidence-lifecycle" in workflow
    assert '"required_reviewers"' in workflow
    assert ".prevent_self_review == true" in workflow
    assert "Record revocation without deleting historic evidence" in workflow
    assert "gh release delete" not in workflow


def test_integrity_verifier_is_read_only_and_files_an_incident() -> None:
    workflow = _workflow("evidence-integrity.yml")

    assert "schedule:" in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "issues: write" in workflow
    assert "Open one integrity incident" in workflow
