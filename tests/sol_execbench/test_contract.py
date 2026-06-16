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
    "trace.correctness.v1",
    "trace.timing.v1",
    "trace.scoring.v1",
    "baseline.measured_export.v1",
    "baseline.scoring_artifact.v1",
    "compatibility.metadata.v1",
    "failure_categories.v1",
}
OPTIONAL_CAPABILITIES = {
    "runtime.evidence.v1",
    "profiling.evidence.v1",
    "toolchain.routing.v1",
    "static_kernel_evidence.v1",
    "agent_feedback.sidecar.v1",
    "profile_summary.sidecar.v1",
}


def test_evaluator_contract_versions_are_stable():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["schema_version"] == SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION
    assert payload["contract_version"] == SOL_EXECBENCH_CONTRACT_VERSION
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v1"
    assert payload["contract_version"] == "1.0"


def test_evaluator_contract_declares_required_capabilities():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert REQUIRED_CAPABILITIES.issubset(set(payload["capabilities"]))


def test_evaluator_contract_advertises_optional_evidence_without_bump():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert OPTIONAL_CAPABILITIES.issubset(set(payload["capabilities"]))
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
    assert any(
        "Static kernel evidence is diagnostic sidecar metadata only" in claim
        for claim in payload["source_boundary_claims"]
    )
    assert any(
        "Agent feedback and profile summary sidecars are diagnostic" in claim
        and "release-gate" in claim
        and "cutover" in claim
        for claim in payload["source_boundary_claims"]
    )
    assert any(
        "SOL owns optional feedback sidecar schema" in claim
        and "HIP consumers own adapter normalization" in claim
        for claim in payload["source_boundary_claims"]
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
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v1"
    assert REQUIRED_CAPABILITIES.issubset(set(payload["capabilities"]))
    assert "static_kernel_evidence.v1" in payload["capabilities"]
    assert "agent_feedback.sidecar.v1" in payload["capabilities"]
    assert "profile_summary.sidecar.v1" in payload["capabilities"]
