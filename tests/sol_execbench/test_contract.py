from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.data.contract import (
    EvaluatorContract,
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
    "environment_budget.sidecar",
    "static_resource_footprint.sidecar",
    "decision.sidecar",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_CONTRACT_DOC = REPO_ROOT / "docs/user/EVALUATOR-CONTRACT.md"
ACTIVE_CONTRACT_DOCS = (
    CURRENT_CONTRACT_DOC,
    REPO_ROOT / "docs/user/trace.md",
    REPO_ROOT / "docs/user/agent_feedback_sidecar.md",
    REPO_ROOT / "docs/user/profile_summary_sidecar.md",
    REPO_ROOT / "docs/user/decision_sidecar.md",
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
    "sol_execbench.static_kernel_evidence.v2",
    "sol_execbench.environment_snapshot.v2",
    "sol_execbench.arch_capability_budget.v1",
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
    assert payload["sol_release"] == "v1.43"


def test_evaluator_contract_schema_describes_capabilities_as_keys():
    schema = EvaluatorContract.model_json_schema()

    assert schema["properties"]["capabilities"]["description"] == (
        "Named capability keys mapped to requirement levels."
    )


def test_evaluator_contract_capability_keys_do_not_use_schema_ids():
    payload = build_evaluator_contract().model_dump(mode="json")
    capability_keys = set(payload["capabilities"])
    forbidden_keys = SCHEMA_IDS_NOT_CAPABILITIES | OLD_EVALUATOR_CAPABILITY_TOKENS

    assert capability_keys.isdisjoint(forbidden_keys)
    for capability_key in capability_keys:
        for forbidden_key in forbidden_keys:
            assert not _contains_stale_capability_token(capability_key, forbidden_key)


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
        ("sol", "environment_budget", "diagnostic"),
        ("sol", "static_resource_footprint", "diagnostic"),
        ("sol", "decision", "diagnostic"),
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

    # The capability table in EVALUATOR-CONTRACT.md is the only authoritative
    # capability-key list. Versioned artifact schema IDs live on a separate
    # axis and must never appear as capability keys.
    contract_capabilities = _contract_doc_capabilities(contract_text)
    assert contract_capabilities == capabilities
    assert set(contract_capabilities).isdisjoint(SCHEMA_IDS_NOT_CAPABILITIES)

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


@pytest.mark.parametrize("format_args", (["--format", "json"], ["--format=json"]))
def test_contract_cli_json_outputs_builder_payload_without_problem_directory(
    format_args: list[str],
):
    result = CliRunner().invoke(cli, [*format_args, "contract", "evaluator"])

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["ok"] is True
    assert response["command"] == "contract evaluator"
    payload = response["data"]
    expected = build_evaluator_contract().model_dump(mode="json")
    assert payload == expected
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v2"
    assert REQUIRED_CAPABILITIES.issubset(payload["capabilities"])
    assert payload["capabilities"]["static_kernel.evidence"] == "optional"
    assert payload["capabilities"]["agent_feedback.sidecar"] == "profile:diagnostic"
    assert payload["capabilities"]["profile_summary.sidecar"] == "profile:diagnostic"


# --- GATE-01: confirmed-evidence contract surface (Phase 194) ---------------


def test_evaluator_contract_advertises_confirmed_evidence_capabilities():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["capabilities"]["official_score.evidence"] == "confirmed"
    assert payload["capabilities"]["measured_baseline.coverage"] == "confirmed"


def test_evaluator_contract_declares_confirmed_evidence_boundaries():
    payload = build_evaluator_contract().model_dump(mode="json")
    boundary_tuples = {
        (b.get("owner"), b.get("scope"), b.get("authority"))
        for b in payload["boundaries"]
    }

    # official_score and measured_baseline are confirmed authority; diagnostic
    # sidecars remain diagnostic.
    assert ("sol", "official_score", "confirmed") in boundary_tuples
    assert ("sol", "measured_baseline", "confirmed") in boundary_tuples
    assert ("sol", "agent_feedback", "diagnostic") in boundary_tuples
    assert ("sol", "profile_summary", "diagnostic") in boundary_tuples
    assert ("sol", "decision", "diagnostic") in boundary_tuples


def test_evaluator_contract_advertises_confirmed_evidence_blocker_vocabulary():
    from sol_execbench.core.data.contract import CONFIRMED_EVIDENCE_BLOCKERS

    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["confirmed_evidence_blockers"] == list(CONFIRMED_EVIDENCE_BLOCKERS)
    # The vocabulary must include the blockers HIP removes for valid runs and the
    # coverage-failure umbrella added in Phase 193.
    for code in (
        "missing_score",
        "missing_baseline",
        "placeholder_baseline",
        "baseline_coverage_failed",
    ):
        assert code in payload["confirmed_evidence_blockers"]


def test_contract_confirmed_evidence_blockers_match_official_score_constants():
    # D-03: the contract's mirrored blocker literals must equal the official
    # score gate's blocker constants (no parallel namespace drift).
    from sol_execbench.core.data.contract import CONFIRMED_EVIDENCE_BLOCKERS
    from sol_execbench.core.scoring.official_score import (
        BASELINE_COVERAGE_FAILED_BLOCKER,
        MISSING_AGGREGATION_POLICY_BLOCKER,
        MISSING_BASELINE_BLOCKER,
        MISSING_MEASURED_LATENCY_BLOCKER,
        MISSING_SCORE_BLOCKER,
        MISSING_SOL_BOUND_BLOCKER,
        PLACEHOLDER_BASELINE_BLOCKER,
    )

    official_blockers = {
        MISSING_SCORE_BLOCKER,
        MISSING_MEASURED_LATENCY_BLOCKER,
        MISSING_BASELINE_BLOCKER,
        PLACEHOLDER_BASELINE_BLOCKER,
        MISSING_SOL_BOUND_BLOCKER,
        MISSING_AGGREGATION_POLICY_BLOCKER,
        BASELINE_COVERAGE_FAILED_BLOCKER,
    }
    assert set(CONFIRMED_EVIDENCE_BLOCKERS) == official_blockers


def test_official_score_automation_docs_preserve_policy_and_authority_boundaries():
    contract = (REPO_ROOT / "docs/user/EVALUATOR-CONTRACT.md").read_text()
    rdna4_policy = (REPO_ROOT / "docs/internal/RDNA4-DENOMINATOR-POLICY.md").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    consumer_guide = (REPO_ROOT / "docs/user/confirmed_evidence.md").read_text()

    assert "official_score_evidence.v1" in contract
    assert "fixed_suite_denominator_zero_for_blocked" in contract
    assert "blocked workloads contribute 0" in contract
    assert "not yet auto-emitted" not in contract

    assert "fixed_suite_denominator_zero_for_blocked" in rdna4_policy
    assert "does not itself grant official score authority" in rdna4_policy
    assert "blocked workloads contribute 0" in rdna4_policy

    assert "--official-score-report out/official-score.json" in readme
    assert (
        "--official-aggregation-policy fixed_suite_denominator_zero_for_blocked"
        in readme
    )
    assert '"fixed_suite_denominator_zero_for_blocked"' in consumer_guide


def test_official_score_module_documents_explicit_runner_output():
    module_source = (
        REPO_ROOT / "src/sol_execbench/core/scoring/official_score.py"
    ).read_text()

    assert "dataset runner emits explicitly requested official output" in module_source
    assert "automation is being integrated separately" not in module_source
