from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DOC = REPO_ROOT / "docs/releases/v1_38_feedback_loop_rc1.md"


def test_feedback_loop_release_candidate_names_pin_and_sidecar_contracts():
    text = RELEASE_DOC.read_text(encoding="utf-8")

    for expected in (
        "v1.38-feedback-loop-rc1",
        "after `v1.35`",
        "agent_feedback.sidecar.v1",
        "<trace>.agent-feedback.json",
        "profile_summary.sidecar.v1",
        "<trace>.profile-summary.json",
        "sol_version",
        "candidate_id",
        "source_sha256",
        "sol_contract_version",
        "candidate_hash",
        "source_hash",
        "SOL does not duplicate those hints into",
    ):
        assert expected in text


def test_feedback_loop_release_candidate_remains_diagnostic_only():
    text = RELEASE_DOC.read_text(encoding="utf-8")

    assert "preserves canonical Trace JSONL semantics" in text
    for forbidden_authority in (
        "correctness",
        "timing",
        "score",
        "evidence-tier",
        "confirmed-improvement",
        "release-gate",
        "cutover",
        "paper-parity",
        "leaderboard",
    ):
        assert forbidden_authority in text
    assert "Both sidecars are diagnostic-only" in text
