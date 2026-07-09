from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sol_execbench.core.scoring.amd_bound_sanity.builder import (
    build_amd_bound_sanity_report,
)
from sol_execbench.core.scoring.amd_bound_sanity.io import (
    write_amd_bound_sanity_reports,
)
from sol_execbench.core.scoring.amd_bound_sanity.models import (
    AMD_BOUND_SANITY_SCHEMA_VERSION,
    AmdBoundSanityReport,
    AmdBoundSanitySourceRef,
)
from sol_execbench.core.scoring.amd_bound_sanity.rendering import (
    render_amd_bound_sanity_markdown,
)


CREATED_AT = "2026-05-31T00:00:00Z"


def test_amd_bound_sanity_ignores_non_object_score_records() -> None:
    report = build_amd_bound_sanity_report(
        execution_closure={
            "records": [
                {
                    "category": "L1",
                    "problem_id": "L1/demo",
                    "workload_uuid": "w0",
                    "row_index": 0,
                    "closure_status": "attempted_passed",
                }
            ]
        },
        amd_score_report={"scores": [{"workload_uuid": "w0"}, []]},
        created_at=CREATED_AT,
    )

    assert len(report.workloads) == 1
    assert report.status_totals.missing_evidence >= 1


def trace_ref_fixture() -> list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path]:
    return [
        {
            "path": "traces/scored.jsonl",
            "ref": "trace:scored",
            "schema_version": "sol_execbench.trace.v1",
            "checksum": "trace-scored-sha",
        },
        {
            "path": "traces/degraded.jsonl",
            "ref": "trace:degraded",
            "schema_version": "sol_execbench.trace.v1",
            "checksum": "trace-degraded-sha",
        },
    ]


def execution_closure_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.execution_closure.v1",
        "execution_closure_checksum": {"value": "closure-sha"},
        "records": [
            {
                "category": "L1",
                "problem_id": "L1/scored_demo",
                "problem_path": "L1/scored_demo",
                "workload_uuid": "scored-workload",
                "row_index": 0,
                "closure_status": "attempted_passed",
                "trace_ref": "traces/scored.jsonl",
                "evidence_refs": {
                    "amd_score": "score/scored.json",
                    "amd_sol_bound": "amd-sol/scored.json",
                    "solar_derivation": "solar/scored.json",
                    "compatibility_matrix": "matrix/report.json",
                },
                "evidence_gaps": [],
            },
            {
                "category": "L1",
                "problem_id": "L1/degraded_demo",
                "problem_path": "L1/degraded_demo",
                "workload_uuid": "degraded-workload",
                "row_index": 0,
                "closure_status": "attempted_passed",
                "trace_ref": "traces/degraded.jsonl",
                "evidence_refs": {
                    "amd_score": "score/degraded.json",
                    "amd_sol_bound": "amd-sol/degraded.json",
                    "solar_derivation": "solar/degraded.json",
                },
                "evidence_gaps": [],
            },
            {
                "category": "L2",
                "problem_id": "L2/unscored_demo",
                "problem_path": "L2/unscored_demo",
                "workload_uuid": "unscored-workload",
                "row_index": 0,
                "closure_status": "attempted_passed",
                "trace_ref": "traces/unscored.jsonl",
                "evidence_refs": {
                    "amd_score": "score/unscored.json",
                    "amd_sol_bound": "amd-sol/unscored.json",
                    "solar_derivation": "solar/unscored.json",
                },
                "evidence_gaps": [],
            },
            {
                "category": "L2",
                "problem_id": "L2/unsupported_demo",
                "problem_path": "L2/unsupported_demo",
                "workload_uuid": "unsupported-workload",
                "row_index": 0,
                "closure_status": "attempted_passed",
                "trace_ref": "traces/unsupported.jsonl",
                "evidence_refs": {
                    "amd_score": "score/unsupported.json",
                    "amd_sol_bound": "amd-sol/unsupported.json",
                    "solar_derivation": "solar/unsupported.json",
                },
                "evidence_gaps": [],
            },
            {
                "category": "L3",
                "problem_id": "L3/missing_demo",
                "problem_path": "L3/missing_demo",
                "workload_uuid": "missing-workload",
                "row_index": 0,
                "closure_status": "derived_evidence_missing",
                "trace_ref": "traces/missing.jsonl",
                "evidence_refs": {"amd_score": "score/missing.json"},
                "evidence_gaps": [
                    "amd_sol_evidence_missing",
                    "solar_derivation_missing",
                ],
            },
        ],
    }


def amd_sol_artifact(
    definition: str,
    workload_uuid: str,
    status: str,
    *,
    warnings: list[str] | None = None,
    worst_confidence: str = "exact",
    hardware_validation_status: str = "validated",
    model_validation_status: str = "validated",
) -> dict:
    return {
        "schema_version": "sol_execbench.amd_sol_bound.v2",
        "definition": definition,
        "workload_uuid": workload_uuid,
        "hardware_model": {
            "architecture": "gfx1200",
            "hardware_validation_status": hardware_validation_status,
            "model_validation_status": model_validation_status,
        },
        "aggregate_bound": {
            "status": status,
            "scored": status != "unscored",
            "reason": f"aggregate {status}",
            "sol_bound_ms": 1.0 if status != "unscored" else 0.0,
        },
        "coverage_summary": {
            "total_ops": 2,
            "supported_ops": 2 if status == "scored" else 1,
            "inexact_ops": 1 if status == "degraded" else 0,
            "unsupported_ops": 1 if status == "unscored" else 0,
            "worst_confidence": worst_confidence,
        },
        "warnings": warnings or [],
        "report_checksum": {"value": f"amd-sol-{workload_uuid}-sha"},
    }


def solar_artifact(
    definition: str,
    workload_uuid: str,
    status: str,
    *,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "sol_execbench.solar_derivation.v1",
        "definition": definition,
        "workload_uuid": workload_uuid,
        "aggregate_status": {
            "status": status,
            "reason": f"solar {status}",
            "warnings": warnings or [],
            "score_eligible": status != "unscored",
        },
        "coverage_summary": {
            "family_counts": {"matmul": 1},
            "status_counts": {status: 1},
        },
        "source_boundary": {
            "canonical_trace_jsonl": True,
            "public_schema": True,
            "candidate_solution_execution": False,
        },
        "warnings": warnings or [],
        "solar_derivation_checksum": {"value": f"solar-{workload_uuid}-sha"},
    }


def amd_score_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.amd_native_score.v1",
        "amd_native_score_checksum": {"value": "score-sha"},
        "evidence_summary": {
            "trace": 4,
            "timing": 4,
            "sol_bound": 4,
            "baseline": 4,
            "hardware_model": 4,
        },
        "warnings": ["suite warning"],
        "scores": [
            {
                "definition": "scored_demo",
                "workload_uuid": "scored-workload",
                "supported": True,
                "score": 0.88,
                "warnings": [],
                "evidence_refs": {"trace": "traces/scored.jsonl"},
            },
            {
                "definition": "degraded_demo",
                "workload_uuid": "degraded-workload",
                "supported": True,
                "score": 0.42,
                "warnings": ["AMD-native score uses degraded AMD SOL bound evidence"],
                "evidence_refs": {"trace": "traces/degraded.jsonl"},
            },
            {
                "definition": "unscored_demo",
                "workload_uuid": "unscored-workload",
                "supported": False,
                "score": None,
                "warnings": [
                    "AMD-native score was not computed because AMD SOL bound evidence is marked unscored."
                ],
                "evidence_refs": {"trace": "traces/unscored.jsonl"},
            },
            {
                "definition": "unsupported_demo",
                "workload_uuid": "unsupported-workload",
                "supported": False,
                "score": None,
                "warnings": ["unsupported operations"],
                "evidence_refs": {"trace": "traces/unsupported.jsonl"},
            },
        ],
    }


def compatibility_matrix_fixture() -> dict:
    return {
        "schema_version": "sol_execbench.rocm_compatibility_matrix.v1",
        "generated_at": CREATED_AT,
        "status_counts": {"validated": 1, "not_tested": 1},
        "matrix_checksum": {"value": "matrix-sha"},
        "claim_boundary": {
            "native_host_validated": False,
            "new_hardware_validation": False,
            "score_authority": False,
            "leaderboard_authority": False,
        },
        "entries": [],
    }


def build_fixture_report():
    return build_amd_bound_sanity_report(
        trace_refs=trace_ref_fixture(),
        execution_closure=execution_closure_fixture(),
        amd_sol_artifacts=[
            amd_sol_artifact("scored_demo", "scored-workload", "scored"),
            amd_sol_artifact(
                "degraded_demo",
                "degraded-workload",
                "degraded",
                warnings=["inexact RDNA 4 model assumption"],
                worst_confidence="inexact",
                model_validation_status="unvalidated",
            ),
            amd_sol_artifact("unscored_demo", "unscored-workload", "unscored"),
            amd_sol_artifact(
                "unsupported_demo",
                "unsupported-workload",
                "unscored",
                warnings=["unsupported operator family"],
            ),
        ],
        solar_artifacts=[
            solar_artifact("scored_demo", "scored-workload", "scored"),
            solar_artifact(
                "degraded_demo",
                "degraded-workload",
                "degraded",
                warnings=["provisional RDNA 4 semantic grouping risk"],
            ),
            solar_artifact("unscored_demo", "unscored-workload", "unscored"),
            solar_artifact(
                "unsupported_demo",
                "unsupported-workload",
                "unscored",
                warnings=["unsupported operator family"],
            ),
        ],
        amd_score_report=amd_score_fixture(),
        compatibility_matrix=compatibility_matrix_fixture(),
        source_paths={
            "execution_closure": Path("execution_closure.json"),
            "amd_score_report": Path("amd_score.json"),
            "compatibility_matrix": Path("matrix.json"),
        },
        created_at=CREATED_AT,
    )


def test_sanity_01_report_builds_existing_evidence_summary():
    payload = build_fixture_report().model_dump(mode="json")

    assert payload["schema_version"] == AMD_BOUND_SANITY_SCHEMA_VERSION
    assert payload["schema_version"] == "sol_execbench.amd_bound_sanity.v1"
    assert payload["artifact_availability"] == {
        "trace_refs": 2,
        "execution_closure": True,
        "amd_sol_artifacts": 4,
        "solar_artifacts": 4,
        "amd_score_report": True,
        "compatibility_matrix": True,
    }
    assert payload["sources"]["execution_closure"] == {
        "path": "execution_closure.json",
        "ref": None,
        "schema_version": "sol_execbench.execution_closure.v1",
        "checksum": "closure-sha",
    }
    assert payload["sources"]["compatibility_matrix"]["checksum"] == "matrix-sha"
    assert payload["amd_sol_aggregate_statuses"] == {
        "degraded": 1,
        "scored": 1,
        "unscored": 2,
    }
    assert payload["solar_aggregate_statuses"] == {
        "degraded": 1,
        "scored": 1,
        "unscored": 2,
    }
    assert payload["coverage_summary"]["amd_score"]["trace"] == 4
    assert payload["coverage_summary"]["compatibility_matrix"]["validated"] == 1
    assert "suite warning" in payload["warnings"]
    assert payload["report_checksum"]["value"]


def test_sanity_source_refs_normalize_nested_checksum_values():
    report = build_amd_bound_sanity_report(
        trace_refs=[
            {
                "path": "trace.jsonl",
                "ref": "trace:nested-checksum",
                "schema_version": "sol_execbench.trace.v1",
                "checksum": {"value": "trace-sha"},
            }
        ],
        created_at=CREATED_AT,
    )

    assert report.sources.trace_refs[0].checksum == "trace-sha"


def test_sanity_02_diagnostic_statuses_do_not_recompute_score_eligibility():
    payload = build_fixture_report().model_dump(mode="json")
    rows = {row["workload_uuid"]: row for row in payload["workloads"]}

    assert payload["status_totals"] == {
        "scored": 1,
        "degraded": 1,
        "unscored": 1,
        "unsupported": 1,
        "provisional": 1,
        "missing_evidence": 1,
    }
    assert rows["scored-workload"]["diagnostic_status"] == "scored"
    assert rows["degraded-workload"]["diagnostic_status"] == "degraded"
    assert rows["degraded-workload"]["diagnostic_flags"] == ["degraded", "provisional"]
    assert rows["unscored-workload"]["diagnostic_status"] == "unscored"
    assert rows["unsupported-workload"]["diagnostic_status"] == "unsupported"
    assert rows["missing-workload"]["diagnostic_status"] == "missing_evidence"
    assert rows["unsupported-workload"]["amd_score_supported"] is False
    assert (
        rows["unsupported-workload"]["source_statuses"]["amd_score_supported"] is False
    )
    assert rows["unscored-workload"]["source_statuses"]["amd_sol_status"] == "unscored"


def test_sanity_03_claim_boundaries_are_explicit_and_false():
    boundary = build_fixture_report().model_dump(mode="json")["claim_boundary"]

    assert boundary == {
        "provisional_rdna4_model_risk": True,
        "upstream_solar_equivalence": False,
        "amd_sol_model_validation": False,
        "solar_model_validation": False,
        "paper_parity": False,
        "leaderboard_authority": False,
        "score_authority_upgrade": False,
        "cdna3_validation": False,
        "mi300x_validation": False,
        "cdna4_validation": False,
        "native_host_validation": False,
        "new_hardware_validation": False,
    }


def test_sanity_04_missing_optional_evidence_becomes_gap_without_probes(monkeypatch):
    def fail_probe(*_args, **_kwargs):  # pragma: no cover - failure path only
        raise AssertionError("builder must not probe external systems")

    monkeypatch.setattr("subprocess.run", fail_probe)
    report = build_amd_bound_sanity_report(
        trace_refs=[],
        execution_closure=execution_closure_fixture(),
        amd_sol_artifacts=[],
        solar_artifacts=[],
        amd_score_report=None,
        compatibility_matrix=None,
        created_at=CREATED_AT,
    )
    payload = report.model_dump(mode="json")
    reason_codes = {gap["reason_code"] for gap in payload["evidence_gaps"]}

    assert "amd_score_evidence_missing" in reason_codes
    assert "amd_sol_evidence_missing" in reason_codes
    assert "solar_derivation_missing" in reason_codes
    assert "compatibility_matrix_missing" in reason_codes
    assert all(
        "Attach bounded" in gap["next_evidence"] for gap in payload["evidence_gaps"]
    )
    assert payload["artifact_availability"]["amd_sol_artifacts"] == 0
    assert payload["artifact_availability"]["solar_artifacts"] == 0


def test_sanity_models_reject_unknown_fields():
    payload = build_fixture_report().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        AmdBoundSanityReport(**payload)


def test_sanity_markdown_and_write_helpers_are_deterministic(tmp_path):
    first = build_fixture_report()
    second = build_fixture_report()
    first_json = first.to_json()
    first_markdown = render_amd_bound_sanity_markdown(first)

    assert first_json == second.to_json()
    assert first_markdown == render_amd_bound_sanity_markdown(second)
    assert "# AMD Bound Sanity Report" in first_markdown
    assert "diagnostic existing evidence sanity report" in first_markdown
    assert "provisional RDNA 4 model risk" in first_markdown
    assert "not upstream SOLAR equivalence" in first_markdown
    assert "not AMD SOL/SOLAR model validation" in first_markdown
    assert "not paper parity" in first_markdown
    assert "not leaderboard authority" in first_markdown
    assert "not score authority upgrade" in first_markdown
    assert "not CDNA 3 validation" in first_markdown
    assert "not MI300X validation" in first_markdown
    assert "not CDNA 4 validation" in first_markdown
    assert "not native-host validation" in first_markdown
    assert "not new-hardware validation" in first_markdown

    json_path = tmp_path / "amd_bound_sanity.json"
    markdown_path = tmp_path / "amd_bound_sanity.md"
    write_amd_bound_sanity_reports(
        first, json_path=json_path, markdown_path=markdown_path
    )

    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    assert first_json.endswith("\n")
    assert first_markdown.endswith("\n")
    assert json.loads(first_json)["schema_version"] == AMD_BOUND_SANITY_SCHEMA_VERSION


def test_sanity_markdown_escapes_dynamic_table_cells():
    report = build_amd_bound_sanity_report(
        trace_refs=[
            {
                "path": "trace|with\npipe.jsonl",
                "ref": "trace|ref",
                "schema_version": "sol_execbench.trace.v1",
                "checksum": "trace-sha",
            }
        ],
        execution_closure=execution_closure_fixture(),
        amd_sol_artifacts=[
            amd_sol_artifact("scored|demo", "scored-workload", "scored"),
        ],
        solar_artifacts=[
            solar_artifact("scored|demo", "scored-workload", "scored"),
        ],
        amd_score_report=amd_score_fixture(),
        compatibility_matrix=compatibility_matrix_fixture(),
        created_at=CREATED_AT,
    )
    markdown = render_amd_bound_sanity_markdown(report)

    assert "trace\\|with pipe.jsonl" in markdown
    assert "trace\\|ref" in markdown
    assert "scored\\|demo" in markdown
