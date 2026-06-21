from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.profile_summary import (
    ProfileSummarySidecar,
    evaluate_profile_summary_governance,
    validate_profile_summary_freshness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests/sol_execbench/fixtures/profile_summary"
DOC = REPO_ROOT / "docs/profile_summary_sidecar.md"


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _fixture_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FIXTURE_DIR.iterdir())
    )


def test_profile_summary_fixtures_cover_hip_consumer_cases():
    fixture_names = {path.name for path in FIXTURE_DIR.iterdir()}

    assert {
        "valid.profile-summary.json",
        "partial.profile-summary.json",
        "unavailable.profile-summary.json",
        "stale.profile-summary.json",
        "contradictory-authority.profile-summary.json",
        "malformed.profile-summary.json",
        "missing.profile-summary.case.json",
    } <= fixture_names


@pytest.mark.parametrize(
    "name, expected_status",
    [
        ("valid.profile-summary.json", "available"),
        ("partial.profile-summary.json", "partial"),
        ("unavailable.profile-summary.json", "unavailable"),
        ("stale.profile-summary.json", "available"),
    ],
)
def test_profile_summary_valid_fixtures_parse(name: str, expected_status: str):
    sidecar = ProfileSummarySidecar.model_validate(_json(FIXTURE_DIR / name))

    assert sidecar.status == expected_status
    assert sidecar.authority.diagnostic_only is True
    assert sidecar.authority.timing_authority is False
    assert sidecar.authority.score_authority is False
    assert sidecar.authority.release_gate_authority is False
    assert sidecar.authority.cutover_authority is False


def test_profile_summary_stale_fixture_classifies_as_stale_diagnostic():
    sidecar = ProfileSummarySidecar.model_validate(
        _json(FIXTURE_DIR / "stale.profile-summary.json")
    )

    freshness = validate_profile_summary_freshness(
        sidecar,
        trace_path="trace.jsonl",
        run_id="run-current",
    )
    guardrail = evaluate_profile_summary_governance(
        sidecar=sidecar,
        freshness=freshness,
    )

    assert freshness.status == "stale"
    assert guardrail.status == "stale_diagnostic"
    assert guardrail.timing_authority is False
    assert guardrail.score_authority is False
    assert guardrail.release_gate_authority is False


def test_profile_summary_negative_fixtures_downgrade_to_invalid_or_missing():
    with pytest.raises(ValidationError):
        ProfileSummarySidecar.model_validate(
            _json(FIXTURE_DIR / "contradictory-authority.profile-summary.json")
        )
    contradictory_guardrail = evaluate_profile_summary_governance(
        sidecar=None,
        parse_error="timing_authority must be false",
    )

    with pytest.raises(json.JSONDecodeError):
        _json(FIXTURE_DIR / "malformed.profile-summary.json")
    malformed_guardrail = evaluate_profile_summary_governance(
        sidecar=None,
        parse_error="malformed json",
    )

    missing_case = _json(FIXTURE_DIR / "missing.profile-summary.case.json")
    missing_guardrail = evaluate_profile_summary_governance(sidecar=None)

    assert contradictory_guardrail.status == "invalid_diagnostic"
    assert malformed_guardrail.status == "invalid_diagnostic"
    assert missing_guardrail.status == missing_case["expected_governance_status"]
    assert missing_guardrail.reason_codes == [missing_case["expected_reason_code"]]
    for guardrail in (
        contradictory_guardrail,
        malformed_guardrail,
        missing_guardrail,
    ):
        assert guardrail.timing_authority is False
        assert guardrail.score_authority is False
        assert guardrail.evidence_tier_authority is False
        assert guardrail.release_gate_authority is False


def test_profile_summary_fixtures_are_prompt_safe_and_deterministic():
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
    for sidecar_path in sorted(FIXTURE_DIR.glob("*.profile-summary.json")):
        if sidecar_path.name in {
            "contradictory-authority.profile-summary.json",
            "malformed.profile-summary.json",
        }:
            continue
        payload = _json(sidecar_path)
        assert payload["identity"]["generated_at"] == "2026-06-16T00:00:00Z"
        for citation in payload["artifact_citations"]:
            assert "/" not in citation["path"]
            if citation["sha256"] is not None:
                assert len(citation["sha256"]) == 64


def test_profile_summary_docs_explain_hip_mapping_and_fixture_semantics():
    text = DOC.read_text(encoding="utf-8")

    for expected in (
        "HIP Consumer Mapping",
        "profile_summary.sidecar.v1",
        "summary.profiler_status",
        "summary.metrics[]",
        "summary.workload_metrics[]",
        "summary.kernel_metrics[]",
        "summary.bottleneck_hints[]",
        "compute_bound",
        "insufficient_counters",
        "artifact_citations",
        "unknown values must be downgraded",
        "valid.profile-summary.json",
        "missing.profile-summary.case.json",
        "contradictory-authority.profile-summary.json",
        "score authority",
        "release-gate authority",
        "cutover authority",
    ):
        assert expected in text
