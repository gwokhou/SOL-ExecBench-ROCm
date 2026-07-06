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
STALE_SIDECAR_CAPABILITY_TOKENS = {
    f"{sidecar_name}.sidecar.v2"
    for sidecar_name in ("agent_feedback", "profile_summary")
}
OLD_EVALUATOR_CAPABILITY_TOKENS = {
    "runtime.evidence.v1",
    "profiling.evidence.v1",
    "toolchain.routing.v1",
    "static_kernel_evidence.v1",
    "agent_feedback.sidecar.v1",
    "profile_summary.sidecar.v1",
    *STALE_SIDECAR_CAPABILITY_TOKENS,
}
SCHEMA_IDS_NOT_CAPABILITIES = {
    "official_score_evidence.v1",
    "sol_execbench.official_score_evidence.v1",
    "sol_execbench.agent_feedback.v2",
    "sol_execbench.profile_summary.v2",
    "sol_execbench.static_kernel_evidence.v1",
}
ACTIVE_STALE_TOKEN_SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")
ACTIVE_STALE_TOKEN_SUFFIXES = {".json", ".py"}
ACTIVE_STALE_TOKEN_EXCLUDED_PARTS = {".pytest_cache", "__pycache__"}


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


def _describes_schema_as_capability(text: str, schema_id: str) -> bool:
    schema_pattern = re.compile(
        rf"(?<![A-Za-z0-9_.-]){re.escape(schema_id)}(?![A-Za-z0-9_.-])"
    )
    schema_marker = "__SOL_EXECBENCH_SCHEMA_ID__"
    guarded_text = schema_pattern.sub(schema_marker, text)
    marker_pattern = re.escape(schema_marker)
    capability_label = r"capability\s+(?:key|token)"
    label_before_schema_pattern = re.compile(
        rf"\b{capability_label}\b\s*(?:[:=,-]|\bis\b)?\s*`?\s*"
        rf"{marker_pattern}\s*`?",
        re.IGNORECASE,
    )
    schema_before_label_pattern = re.compile(
        rf"`?\s*{marker_pattern}\s*`?(?P<link_text>.*?)\b{capability_label}\b",
        re.IGNORECASE,
    )

    paragraphs = re.split(r"(?:\r?\n\s*){2,}", guarded_text)
    for paragraph in paragraphs:
        normalized_paragraph = re.sub(r"[ \t]*\r?\n[ \t]*", " ", paragraph.strip())
        for segment in normalized_paragraph.split("."):
            if schema_marker not in segment:
                continue
            if label_before_schema_pattern.search(segment):
                return True
            for match in schema_before_label_pattern.finditer(segment):
                link_text = match.group("link_text")
                link_words = re.findall(r"[A-Za-z]+", link_text)
                schema_capability_starters = {
                    "advertised",
                    "advertises",
                    "as",
                    "is",
                    "through",
                }
                if (
                    ("`" not in link_text)
                    and not re.search(r"\bbehind\b", link_text, re.IGNORECASE)
                    and (
                        not link_words
                        or link_words[0].lower() in schema_capability_starters
                    )
                ):
                    return True
    return False


def _active_stale_token_scan_paths() -> list[Path]:
    paths: list[Path] = []
    for root in ACTIVE_STALE_TOKEN_SCAN_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in ACTIVE_STALE_TOKEN_SUFFIXES:
                continue
            if ACTIVE_STALE_TOKEN_EXCLUDED_PARTS.intersection(path.parts):
                continue
            paths.append(path)
    return paths


def test_schema_as_capability_guard_covers_natural_wording():
    assert "sol_execbench.official_score_evidence.v1" in SCHEMA_IDS_NOT_CAPABILITIES
    assert _describes_schema_as_capability(
        "`sol_execbench.official_score_evidence.v1` capability key",
        "sol_execbench.official_score_evidence.v1",
    )
    assert _describes_schema_as_capability(
        "capability token `sol_execbench.official_score_evidence.v1`",
        "sol_execbench.official_score_evidence.v1",
    )
    assert _describes_schema_as_capability(
        "`sol_execbench.profile_summary.v2` is the capability token",
        "sol_execbench.profile_summary.v2",
    )
    assert _describes_schema_as_capability(
        "`sol_execbench.profile_summary.v2` is the optional\ncapability token",
        "sol_execbench.profile_summary.v2",
    )
    assert _describes_schema_as_capability(
        "capability key sol_execbench.profile_summary.v2",
        "sol_execbench.profile_summary.v2",
    )
    assert _describes_schema_as_capability(
        "capability key\n`sol_execbench.profile_summary.v2`",
        "sol_execbench.profile_summary.v2",
    )
    assert _describes_schema_as_capability(
        "official_score_evidence.v1 capability token",
        "official_score_evidence.v1",
    )
    assert _describes_schema_as_capability(
        "official_score_evidence.v1 capability key",
        "official_score_evidence.v1",
    )
    assert _describes_schema_as_capability(
        "`sol_execbench.profile_summary.v2` is documented here as the optional "
        "diagnostic profile summary capability token",
        "sol_execbench.profile_summary.v2",
    )
    assert _describes_schema_as_capability(
        "`sol_execbench.profile_summary.v2` is documented with concrete "
        "artifact schema wording that deliberately includes enough descriptive "
        "detail before its capability token label",
        "sol_execbench.profile_summary.v2",
    )
    assert not _describes_schema_as_capability(
        "Concrete artifact schema is `sol_execbench.profile_summary.v2`. "
        "The capability key is `profile_summary.sidecar`.",
        "sol_execbench.profile_summary.v2",
    )
    assert not _describes_schema_as_capability(
        "`sol_execbench.static_kernel_evidence.v1` sidecar schema behind the\n"
        "`static_kernel.evidence` capability key.",
        "sol_execbench.static_kernel_evidence.v1",
    )


def test_active_stale_token_scan_includes_malformed_sidecar_fixtures():
    malformed_fixture_paths = {
        REPO_ROOT
        / "tests/sol_execbench/fixtures/agent_feedback/malformed.agent-feedback.json",
        REPO_ROOT
        / "tests/sol_execbench/fixtures/profile_summary/malformed.profile-summary.json",
    }

    assert all(path.is_file() for path in malformed_fixture_paths)
    assert malformed_fixture_paths.issubset(set(_active_stale_token_scan_paths()))


def test_active_source_and_fixture_text_avoid_stale_sidecar_capability_tokens():
    offenders: list[str] = []
    for path in _active_stale_token_scan_paths():
        text = path.read_text(encoding="utf-8")
        for stale_token in sorted(STALE_SIDECAR_CAPABILITY_TOKENS):
            if _contains_stale_capability_token(text, stale_token):
                rel_path = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel_path}: {stale_token}")

    assert offenders == []


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
        for schema_id in sorted(SCHEMA_IDS_NOT_CAPABILITIES):
            assert not _describes_schema_as_capability(text, schema_id), path

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
