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


def test_publish_workflow_is_single_maintainer_and_verifies_round_trip() -> None:
    workflow = _workflow("evidence-publish.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "name: evidence-publish" in workflow
    assert '"required_reviewers"' not in workflow
    assert ".prevent_self_review" not in workflow
    assert "attestations: write" in workflow
    assert "id-token: write" in workflow
    assert "Classify an existing release for safe recovery" in workflow
    assert "steps.release_state.outputs.exists != 'true'" in workflow
    assert "concurrency:" in workflow
    assert "Download public release and verify it again" in workflow


def test_revocation_is_manual_single_maintainer_and_never_deletes_assets() -> None:
    workflow = _workflow("evidence-revoke.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "name: evidence-lifecycle" in workflow
    assert '"required_reviewers"' not in workflow
    assert ".prevent_self_review" not in workflow
    assert "Record revocation without deleting historic evidence" in workflow
    assert "gh release delete" not in workflow


def test_integrity_verifier_is_read_only_and_files_an_incident() -> None:
    workflow = _workflow("evidence-integrity.yml")

    assert "schedule:" in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "issues: write" in workflow
    assert "Open one integrity incident" in workflow
