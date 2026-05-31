from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.dataset.paper_denominator import (
    PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION,
    PaperDenominatorReport,
    build_paper_denominator_report,
    render_paper_denominator_markdown,
    write_paper_denominator_reports,
)


CREATED_AT = "2026-05-31T00:00:00Z"


def inventory_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.dataset_inventory.v1",
        "created_at": "2026-05-31T00:00:00Z",
        "selected_categories": ["L1", "L2"],
        "categories": [
            {
                "name": "L1",
                "denominators": {
                    "discovered_problems": 2,
                    "parsed_problems": 2,
                    "parsed_workloads": 4,
                    "schema_failures": 0,
                    "missing_required_files": 0,
                },
            },
            {
                "name": "L2",
                "denominators": {
                    "discovered_problems": 1,
                    "parsed_problems": 1,
                    "parsed_workloads": 1,
                    "schema_failures": 0,
                    "missing_required_files": 0,
                },
            },
        ],
        "inventory_checksum": {"value": "inventory-sha"},
    }


def readiness_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.rocm_readiness.v1",
        "created_at": "2026-05-31T00:00:00Z",
        "selected_categories": ["L1", "L2"],
        "workloads": [
            {
                "category": "L1",
                "problem_id": "L1/pass_demo",
                "problem_path": "L1/pass_demo",
                "workload_uuid": "passed-workload",
                "row_index": 0,
                "status": "ready",
                "reasons": [
                    {
                        "code": "ready_to_attempt_rocm_execution",
                        "next_action": "Run bounded execution closure.",
                    }
                ],
            },
            {
                "category": "L1",
                "problem_id": "L1/fail_demo",
                "problem_path": "L1/fail_demo",
                "workload_uuid": "failed-workload",
                "row_index": 0,
                "status": "ready",
                "reasons": [],
            },
            {
                "category": "L1",
                "problem_id": "L1/blocked_demo",
                "problem_path": "L1/blocked_demo",
                "workload_uuid": "blocked-workload",
                "row_index": 0,
                "status": "runtime_blocked",
                "reasons": [
                    {
                        "code": "safetensors_asset_missing",
                        "next_action": "Acquire asset inside the dataset root.",
                    }
                ],
            },
            {
                "category": "L2",
                "problem_id": "L2/unsupported_demo",
                "problem_path": "L2/unsupported_demo",
                "workload_uuid": "unsupported-workload",
                "row_index": 0,
                "status": "unsupported",
                "reasons": [
                    {
                        "code": "unsupported_operator",
                        "next_action": "Exclude from ROCm-ready subset.",
                    }
                ],
            },
        ],
        "readiness_checksum": {"value": "readiness-sha"},
    }


def ready_subset_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.ready_subset.v1",
        "created_at": "2026-05-31T00:00:00Z",
        "dataset_root": "dataset",
        "selected_categories": ["L1", "L2"],
        "included_workloads": 2,
        "excluded_workloads": 2,
        "problems": [
            {
                "category": "L1",
                "problem_id": "L1/pass_demo",
                "problem_path": "L1/pass_demo",
                "workloads": [{"uuid": "passed-workload", "row_index": 0}],
            },
            {
                "category": "L1",
                "problem_id": "L1/fail_demo",
                "problem_path": "L1/fail_demo",
                "workloads": [{"uuid": "failed-workload", "row_index": 0}],
            },
        ],
        "ready_subset_checksum": {"value": "ready-sha"},
    }


def execution_closure_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.execution_closure.v1",
        "created_at": "2026-05-31T00:00:00Z",
        "status": "completed_with_failures",
        "provenance": {},
        "totals": {},
        "filters": {},
        "records": [
            {
                "category": "L1",
                "problem_id": "L1/pass_demo",
                "problem_path": "L1/pass_demo",
                "workload_uuid": "passed-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "attempted_passed",
                "trace_status": "PASSED",
                "trace_ref": "traces/pass.json",
                "evidence_refs": {
                    "timing_evidence": "timing/pass.json",
                    "amd_score": "scores/pass.json",
                    "amd_sol_bound": "amd-sol/pass.json",
                    "solar_derivation": "solar/pass.json",
                },
                "evidence_gaps": [],
            },
            {
                "category": "L1",
                "problem_id": "L1/fail_demo",
                "problem_path": "L1/fail_demo",
                "workload_uuid": "failed-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "attempted_failed",
                "trace_status": "FAILED",
                "trace_ref": "traces/fail.json",
                "evidence_refs": {"amd_score": "scores/fail.json"},
                "evidence_gaps": [
                    "timing_evidence_missing",
                    "amd_sol_evidence_missing",
                    "solar_derivation_missing",
                ],
            },
            {
                "category": "L1",
                "problem_id": "L1/filtered_demo",
                "problem_path": "L1/filtered_demo",
                "workload_uuid": "filtered-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "filtered",
                "filter_reasons": ["limit"],
                "trace_status": None,
                "evidence_refs": {},
                "evidence_gaps": [],
            },
            {
                "category": "L1",
                "problem_id": "L1/skipped_demo",
                "problem_path": "L1/skipped_demo",
                "workload_uuid": "skipped-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "skipped_existing_pass",
                "trace_status": "PASSED",
                "trace_ref": "traces/skipped.json",
                "evidence_refs": {},
                "evidence_gaps": ["amd_score_evidence_missing"],
            },
            {
                "category": "L1",
                "problem_id": "L1/not_attempted_demo",
                "problem_path": "L1/not_attempted_demo",
                "workload_uuid": "not-attempted-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "not_attempted",
                "trace_status": None,
                "evidence_refs": {},
                "evidence_gaps": [],
            },
            {
                "category": "L2",
                "problem_id": "L2/missing_demo",
                "problem_path": "L2/missing_demo",
                "workload_uuid": "missing-workload",
                "row_index": 0,
                "readiness_status": "ready",
                "readiness_reason_codes": [],
                "closure_status": "derived_evidence_missing",
                "trace_status": "PASSED",
                "trace_ref": "traces/missing.json",
                "evidence_refs": {"amd_score": "scores/missing.json"},
                "evidence_gaps": ["timing_evidence_missing"],
            },
        ],
        "execution_closure_checksum": {"value": "closure-sha"},
        "claim_boundary": {},
    }


def amd_score_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.amd_native_score.v1",
        "created_at": "2026-05-31T00:00:00Z",
        "scores": [
            {
                "definition": "pass_demo",
                "workload_uuid": "passed-workload",
                "supported": True,
                "evidence_refs": {
                    "trace": "traces/pass.json",
                    "timing": "timing/pass.json",
                    "sol_bound": "amd-sol/pass.json",
                },
                "derived_evidence_refs": {"formula": "solar/pass.json#formula"},
            },
            {
                "definition": "fail_demo",
                "workload_uuid": "failed-workload",
                "supported": False,
                "evidence_refs": {"trace": "traces/fail.json"},
                "derived_evidence_refs": {},
            },
        ],
        "amd_native_score_checksum": {"value": "score-sha"},
    }


def build_fixture_report():
    return build_paper_denominator_report(
        manifest={"schema_version": "sol_execbench.dataset_manifest.v1", "manifest_checksum": {"value": "manifest-sha"}},
        inventory=inventory_fixture(),
        readiness=readiness_fixture(),
        ready_subset=ready_subset_fixture(),
        execution_closure=execution_closure_fixture(),
        amd_score_report=amd_score_fixture(),
        amd_sol_artifacts=[
            {"path": "artifacts/amd-sol/pass.json", "ref": "pass-bound", "checksum": "amd-sol-sha"}
        ],
        solar_artifacts=[
            {"path": "artifacts/solar/pass.json", "ref": "pass-solar", "checksum": "solar-sha"}
        ],
        source_paths={
            "manifest": Path("manifest.json"),
            "inventory": Path("inventory.json"),
            "readiness": Path("readiness.json"),
            "ready_subset": Path("ready_subset.json"),
            "execution_closure": Path("execution_closure.json"),
            "amd_score_report": Path("amd_score.json"),
        },
        created_at=CREATED_AT,
    )


def test_paper_denominator_report_aggregates_required_rollups():
    payload = build_fixture_report().model_dump(mode="json")

    assert payload["schema_version"] == PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION
    assert payload["suite"]["problems"] == 3
    assert payload["suite"]["workloads"] == 4
    assert payload["suite"]["states"]["ready"] == 2
    assert payload["suite"]["states"]["blocked"] == 1
    assert payload["suite"]["states"]["unsupported"] == 1
    assert payload["suite"]["states"]["deferred"] == 1
    assert payload["suite"]["states"]["evidence_missing"] == 3
    assert payload["suite"]["states"]["attempted_passed"] == 1
    assert payload["suite"]["states"]["attempted_failed"] == 1
    assert payload["suite"]["states"]["filtered"] == 1
    assert payload["suite"]["states"]["skipped"] == 1
    assert payload["suite"]["states"]["not_attempted"] == 1
    assert {category["name"] for category in payload["categories"]} == {"L1", "L2"}
    assert {problem["problem_id"] for problem in payload["problems"]} >= {
        "L1/pass_demo",
        "L1/fail_demo",
        "L1/blocked_demo",
    }
    assert {workload["workload_uuid"] for workload in payload["workloads"]} >= {
        "passed-workload",
        "failed-workload",
        "blocked-workload",
    }


def test_paper_denominator_report_tracks_reasons_and_next_evidence():
    payload = build_fixture_report().model_dump(mode="json")
    states = payload["suite"]["states"]
    reason_codes = {bucket["reason_code"] for bucket in payload["reason_buckets"]}
    evidence_codes = {gap["evidence"] for gap in payload["evidence_gaps"]}

    assert set(states) == {
        "ready",
        "blocked",
        "unsupported",
        "deferred",
        "evidence_missing",
        "attempted_passed",
        "attempted_failed",
        "filtered",
        "skipped",
        "not_attempted",
    }
    assert {
        "ready_to_attempt_rocm_execution",
        "safetensors_asset_missing",
        "unsupported_operator",
        "timing_evidence_missing",
        "amd_score_evidence_missing",
        "amd_sol_evidence_missing",
        "solar_derivation_missing",
    }.issubset(reason_codes)
    assert {
        "timing",
        "amd_score",
        "amd_sol",
        "solar_derivation",
    }.issubset(evidence_codes)
    assert any("Attach" in hint["next_evidence"] for hint in payload["next_evidence_hints"])


def test_paper_denominator_report_uses_bounded_source_refs_only():
    payload = build_fixture_report().model_dump(mode="json")
    source = payload["sources"]["inventory"]

    assert set(source) == {"path", "ref", "schema_version", "checksum"}
    assert source == {
        "path": "inventory.json",
        "ref": None,
        "schema_version": "sol_execbench.dataset_inventory.v1",
        "checksum": "inventory-sha",
    }
    assert payload["sources"]["execution_closure"]["checksum"] == "closure-sha"
    assert payload["sources"]["amd_sol_artifacts"][0]["path"] == "artifacts/amd-sol/pass.json"
    serialized = json.dumps(payload["sources"], sort_keys=True)
    assert "workloads" not in serialized
    assert "records" not in serialized
    assert "scores" not in serialized


def test_paper_denominator_claim_boundaries_are_false():
    boundary = build_fixture_report().model_dump(mode="json")["claim_boundary"]

    assert boundary == {
        "paper_parity": False,
        "upstream_solar_parity": False,
        "leaderboard_authority": False,
        "native_host_validation": False,
        "new_hardware_validation": False,
        "full_235_problem_validation": False,
        "score_authority": False,
    }


def test_paper_denominator_json_markdown_and_write_helpers_are_deterministic(tmp_path):
    first = build_fixture_report()
    second = build_fixture_report()
    first_json = first.to_json()
    first_markdown = render_paper_denominator_markdown(first)

    assert first_json == second.to_json()
    assert first_markdown == render_paper_denominator_markdown(second)
    assert first.model_dump(mode="json")["report_checksum"]["value"]
    assert "# SOL ExecBench Paper Denominator Report" in first_markdown
    assert "denominator accounting and evidence-gap review only" in first_markdown
    assert "## Reason Buckets" in first_markdown
    assert "| timing_evidence_missing | 2 | evidence_missing |" in first_markdown
    assert "not paper validation" in first_markdown
    assert "not paper parity" in first_markdown
    assert "not upstream SOLAR parity" in first_markdown
    assert "not leaderboard authority" in first_markdown
    assert "not native-host validation" in first_markdown
    assert "not new-hardware validation" in first_markdown
    assert "false" in first_markdown

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_paper_denominator_reports(first, json_path=json_path, markdown_path=markdown_path)

    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    assert first_json.endswith("\n")
    assert first_markdown.endswith("\n")


def test_paper_denominator_models_reject_unknown_fields():
    payload = build_fixture_report().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        PaperDenominatorReport(**payload)
