from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackSidecar,
)
from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "tests/sol_execbench/fixtures/agent_feedback"
DOC = REPO_ROOT / "docs/user/agent_feedback_sidecar.md"


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _fixture_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(FIXTURE_DIR.iterdir())
    )


def test_agent_feedback_fixtures_cover_hip_consumer_cases():
    fixture_names = {path.name for path in FIXTURE_DIR.iterdir()}

    assert {
        "valid.agent-feedback.json",
        "partial.agent-feedback.json",
        "unavailable.agent-feedback.json",
        "stale.agent-feedback.json",
        "contradictory-authority.agent-feedback.json",
        "malformed.agent-feedback.json",
        "missing.agent-feedback.case.json",
    } <= fixture_names


@pytest.mark.parametrize(
    "name, expected_status",
    [
        ("valid.agent-feedback.json", "available"),
        ("partial.agent-feedback.json", "partial"),
        ("unavailable.agent-feedback.json", "unavailable"),
        ("stale.agent-feedback.json", "available"),
    ],
)
def test_agent_feedback_valid_fixtures_parse(name: str, expected_status: str):
    sidecar = AgentFeedbackSidecar.model_validate(_json(FIXTURE_DIR / name))

    assert sidecar.status == expected_status
    assert sidecar.authority == "diagnostic"


def test_agent_feedback_fixtures_are_prompt_safe_and_deterministic():
    text = _fixture_text()

    forbidden_fragments = (
        "/tmp/",
        "/var/",
        "/Users/",
        "raw_trace",
        "profiler dump",
        "kernel source",
        "BEGIN SOURCE",
        "Traceback",
    )
    for fragment in forbidden_fragments:
        assert fragment not in text
    for sidecar_path in sorted(FIXTURE_DIR.glob("*.agent-feedback.json")):
        if sidecar_path.name in {
            "contradictory-authority.agent-feedback.json",
            "malformed.agent-feedback.json",
        }:
            continue
        payload = _json(sidecar_path)
        assert payload["identity"]["generated_at"] == "2026-06-16T00:00:00Z"
        for citation in payload["artifact_citations"]:
            assert "/" not in citation["path"]
            if citation["sha256"] is not None:
                assert len(citation["sha256"]) == 64


def test_agent_feedback_docs_explain_hip_mapping_and_fixture_semantics():
    text = DOC.read_text(encoding="utf-8")

    for expected in (
        "HIP Consumer Mapping",
        "bottleneck",
        "recommendation",
        "limitation",
        "artifact_citations",
        "identity.target_id",
        "identity.candidate_id",
        "identity.source_sha256",
        "identity.sol_version",
        "solution content hash",
        "compile_failure",
        "profile_summary.sidecar",
        PerformanceArtifactSchema.PROFILE_SUMMARY,
        "SOL does not duplicate those hints into",
        "unknown values must be downgraded",
        "valid.agent-feedback.json",
        "missing.agent-feedback.case.json",
        "contradictory-authority.agent-feedback.json",
    ):
        assert expected in text
