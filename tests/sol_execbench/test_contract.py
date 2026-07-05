from __future__ import annotations

import json

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


def test_evaluator_contract_versions_are_stable():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["schema_version"] == SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION
    assert payload["contract_version"] == SOL_EXECBENCH_CONTRACT_VERSION
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v2"
    assert payload["contract_version"] == "1.0"
    assert payload["sol_release"] == "v1.41"


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
