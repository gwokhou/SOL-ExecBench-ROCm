from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.data.contract import (
    SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
    SOL_EXECBENCH_CONTRACT_VERSION,
    build_evaluator_contract,
)
from sol_execbench.core.data.trace import EvaluationStatus


REQUIRED_CAPABILITIES = {
    "trace.correctness",
    "trace.timing",
    "trace.scoring",
    "baseline.measured_export",
    "baseline.scoring_artifact",
    "compatibility.metadata",
    "failure_categories",
}
OPTIONAL_CAPABILITIES = {
    "runtime.evidence",
    "profiling.evidence",
    "toolchain.routing",
    "static_kernel.evidence",
    "agent_feedback.sidecar",
    "profile_summary.sidecar",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_CONTRACT_DOC = REPO_ROOT / "docs/EVALUATOR-CONTRACT.md"
ACTIVE_CONTRACT_DOCS = (
    CURRENT_CONTRACT_DOC,
    REPO_ROOT / "docs/trace.md",
    REPO_ROOT / "docs/agent_feedback_sidecar.md",
    REPO_ROOT / "docs/profile_summary_sidecar.md",
)
OLD_EVALUATOR_CAPABILITY_TOKENS = {
    "runtime.evidence.v1",
    "profiling.evidence.v1",
    "toolchain.routing.v1",
    "static_kernel_evidence.v1",
    "agent_feedback.sidecar.v1",
    "agent_feedback.sidecar.v2",
    "profile_summary.sidecar.v1",
    "profile_summary.sidecar.v2",
}
SCHEMA_IDS_NOT_CAPABILITIES = {
    "official_score_evidence.v1",
    "sol_execbench.agent_feedback.v2",
    "sol_execbench.profile_summary.v2",
    "sol_execbench.static_kernel_evidence.v1",
}
SCHEMA_AS_CAPABILITY_PHRASES = {
    f"{schema_id} capability {noun}"
    for schema_id in SCHEMA_IDS_NOT_CAPABILITIES
    for noun in ("key", "token")
}


def _contract_doc_capabilities(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line == "| Capability key | Level | Meaning |":
            in_table = True
            continue
        if not in_table:
            continue
        if line == "| --- | --- | --- |":
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        key, level, _meaning = cells
        if key.startswith("`") and key.endswith("`"):
            key = key[1:-1]
        if level.startswith("`") and level.endswith("`"):
            level = level[1:-1]
        assert key not in rows, key
        rows[key] = level
    return rows


def _contains_stale_capability_token(text: str, token: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(token)}(?![A-Za-z0-9_.-])"
    return re.search(pattern, text) is not None


def _contains_schema_as_capability_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.replace("`", "").replace(">", " ").split())
    return phrase in normalized_text


def test_schema_as_capability_guard_covers_key_and_token_wording():
    expected_schema_ids = {
        "official_score_evidence.v1",
        "sol_execbench.agent_feedback.v2",
        "sol_execbench.profile_summary.v2",
        "sol_execbench.static_kernel_evidence.v1",
    }
    expected_phrases = {
        f"{schema_id} capability {noun}"
        for schema_id in expected_schema_ids
        for noun in ("key", "token")
    }

    assert SCHEMA_IDS_NOT_CAPABILITIES == expected_schema_ids
    assert SCHEMA_AS_CAPABILITY_PHRASES == expected_phrases


def test_evaluator_contract_versions_are_stable():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["schema_version"] == SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION
    assert payload["contract_version"] == SOL_EXECBENCH_CONTRACT_VERSION
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v2"
    assert payload["contract_version"] == "1.0"
    assert payload["sol_release"] == "v1.42"


def test_evaluator_contract_declares_required_capabilities():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert REQUIRED_CAPABILITIES.issubset(payload["capabilities"])
    for capability in REQUIRED_CAPABILITIES:
        assert payload["capabilities"][capability] == "always"


def test_evaluator_contract_advertises_optional_evidence_without_bump():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert OPTIONAL_CAPABILITIES.issubset(payload["capabilities"])
    assert payload["contract_version"] == "1.0"
    assert payload["trace_field_requirements"]["top_level"] == [
        "definition",
        "workload",
        "solution",
        "evaluation",
    ]
    assert "static_kernel_evidence" not in payload["trace_field_requirements"]
    assert "static_kernel_evidence" not in payload["correctness_fields"]
    assert "static_kernel_evidence" not in payload["timing_fields"]
    assert "static_kernel_evidence" not in payload["scoring_fields"]
    for forbidden in ("agent_feedback", "profile_summary"):
        assert forbidden not in payload["trace_field_requirements"]
        assert forbidden not in payload["correctness_fields"]
        assert forbidden not in payload["timing_fields"]
        assert forbidden not in payload["scoring_fields"]
    assert "source_boundary_claims" not in payload
    assert {
        (boundary.get("owner"), boundary.get("scope"), boundary.get("authority"))
        for boundary in payload["boundaries"]
    } >= {
        ("sol", "agent_feedback", "diagnostic"),
        ("sol", "profile_summary", "diagnostic"),
    }


def test_current_contract_doc_matches_builder_capabilities():
    contract_text = CURRENT_CONTRACT_DOC.read_text(encoding="utf-8")
    active_doc_texts = {
        path: path.read_text(encoding="utf-8") for path in ACTIVE_CONTRACT_DOCS
    }
    payload = build_evaluator_contract().model_dump(mode="json")
    capabilities = payload["capabilities"]

    assert isinstance(capabilities, dict)
    for path, text in active_doc_texts.items():
        for old_token in sorted(OLD_EVALUATOR_CAPABILITY_TOKENS):
            assert not _contains_stale_capability_token(text, old_token), path
        for phrase in sorted(SCHEMA_AS_CAPABILITY_PHRASES):
            assert not _contains_schema_as_capability_phrase(text, phrase), path

    assert _contract_doc_capabilities(contract_text) == capabilities

    assert "`sol_execbench.agent_feedback.v2`" in contract_text
    assert "`sol_execbench.profile_summary.v2`" in contract_text


def test_contract_doc_capabilities_rejects_duplicate_keys():
    with pytest.raises(AssertionError, match="trace.correctness"):
        _contract_doc_capabilities(
            "\n".join(
                [
                    "| Capability key | Level | Meaning |",
                    "| --- | --- | --- |",
                    "| `trace.correctness` | `always` | First row. |",
                    "| `trace.correctness` | `optional` | Duplicate row. |",
                ]
            )
        )


def test_evaluator_contract_freezes_trace_status_and_field_groups():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert set(payload["evaluation_statuses"]) == {
        status.value for status in EvaluationStatus
    }
    assert payload["trace_field_requirements"]["top_level"] == [
        "definition",
        "workload",
        "solution",
        "evaluation",
    ]
    assert payload["trace_field_requirements"]["evaluation"] == [
        "status",
        "environment",
        "timestamp",
        "log",
        "correctness",
        "performance",
    ]
    assert payload["correctness_fields"] == [
        "max_relative_error",
        "max_absolute_error",
        "has_nan",
        "has_inf",
        "extra",
    ]
    assert payload["timing_fields"] == [
        "latency_ms",
        "reference_latency_ms",
        "speedup_factor",
    ]


def test_baseline_export_fields_distinguish_measured_and_scoring_artifacts():
    payload = build_evaluator_contract().model_dump(mode="json")
    baseline_fields = payload["baseline_export_fields"]

    assert baseline_fields["measured_registry_schema_version"] == (
        "sol_execbench.measured_baseline_registry.v1"
    )
    assert baseline_fields["scoring_artifact_schema_version"] == (
        "sol_execbench.scoring_baseline.v1"
    )
    assert "baseline_coverage_status" in baseline_fields["measured_registry"]
    assert "entries" in baseline_fields["scoring_artifact"]
    assert baseline_fields["measured_registry"] != baseline_fields["scoring_artifact"]


def test_contract_cli_json_outputs_builder_payload_without_problem_directory():
    result = CliRunner().invoke(cli, ["contract", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    expected = build_evaluator_contract().model_dump(mode="json")
    assert payload == expected
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v2"
    assert REQUIRED_CAPABILITIES.issubset(payload["capabilities"])
    assert payload["capabilities"]["static_kernel.evidence"] == "optional"
    assert payload["capabilities"]["agent_feedback.sidecar"] == "profile:diagnostic"
    assert payload["capabilities"]["profile_summary.sidecar"] == "profile:diagnostic"
