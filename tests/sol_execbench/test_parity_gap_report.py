from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sol_execbench.core.dataset.parity_gap import (
    _amd_score_record,
    _execution_closure_record,
    build_parity_gap_report,
    render_parity_gap_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_parity_gaps.py"
spec = importlib.util.spec_from_file_location("report_parity_gaps", SCRIPT_PATH)
assert spec is not None
report_parity_gaps = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report_parity_gaps)


def test_parity_gap_execution_record_adapter_normalizes_missing_fields() -> None:
    record = _execution_closure_record({"problem_path": "L1/demo"})

    assert record.category == "unknown"
    assert record.problem_id == "L1/demo"
    assert record.problem_path == "L1/demo"
    assert record.workload_uuid is None
    assert record.evidence_gaps == []


def test_parity_gap_score_record_adapter_filters_warning_list() -> None:
    record = _amd_score_record(
        {"definition": "L1/demo", "warnings": ["degraded", 3], "supported": True}
    )

    assert record.definition == "L1/demo"
    assert record.supported is True
    assert record.warnings == ["degraded"]


def _inventory() -> dict:
    return {
        "schema_version": "sol_execbench.dataset_inventory.v1",
        "created_at": "2026-05-23T00:00:00Z",
        "selected_categories": ["L1"],
        "categories": [
            {
                "name": "L1",
                "denominators": {
                    "discovered_problems": 2,
                    "parsed_problems": 2,
                    "parsed_workloads": 2,
                    "schema_failures": 0,
                    "missing_required_files": 0,
                },
            }
        ],
        "problems": [],
        "denominators": {},
        "diagnostics": [],
        "inventory_checksum": {"value": "inventory-sha"},
    }


def _readiness() -> dict:
    return {
        "schema_version": "sol_execbench.rocm_readiness.v1",
        "created_at": "2026-05-23T00:00:00Z",
        "selected_categories": ["L1"],
        "problems": [],
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
        ],
        "readiness_checksum": {"value": "readiness-sha"},
    }


def _ready_subset() -> dict:
    return {
        "schema_version": "sol_execbench.ready_subset.v1",
        "created_at": "2026-05-23T00:00:00Z",
        "dataset_root": "dataset",
        "selected_categories": ["L1"],
        "included_workloads": 1,
        "excluded_workloads": 1,
        "problems": [
            {
                "category": "L1",
                "problem_id": "L1/pass_demo",
                "problem_path": "L1/pass_demo",
                "workloads": [{"uuid": "passed-workload", "row_index": 0}],
            }
        ],
        "claim_boundary": {"ready_to_attempt_rocm_execution": True},
        "ready_subset_checksum": {"value": "ready-sha"},
    }


def _execution_closure() -> dict:
    return {
        "schema_version": "sol_execbench.execution_closure.v1",
        "created_at": "2026-05-23T00:00:00Z",
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
                "closure_status": "derived_evidence_missing",
                "trace_status": "PASSED",
                "trace_ref": "L1/pass_demo/traces.json",
                "evidence_refs": {
                    "amd_score": "amd-score.json",
                    "amd_sol_bound": "amd-sol/pass.amd-sol-v2.json",
                },
                "evidence_gaps": ["timing_evidence_missing"],
            },
            {
                "category": "L1",
                "problem_id": "L1/blocked_demo",
                "problem_path": "L1/blocked_demo",
                "workload_uuid": "blocked-workload",
                "row_index": 0,
                "closure_status": "not_attempted",
                "trace_status": None,
                "evidence_refs": {},
                "evidence_gaps": [],
            },
        ],
        "claim_boundary": {},
    }


def _amd_score_report() -> dict:
    return {
        "schema_version": "sol_execbench.amd_native_score.v1",
        "scored_count": 1,
        "unscored_count": 0,
        "scores": [
            {
                "definition": "pass_demo",
                "workload_uuid": "passed-workload",
                "supported": True,
                "warnings": ["AMD-native score uses degraded AMD SOL bound evidence."],
                "evidence_refs": {
                    "trace": "L1/pass_demo/traces.json",
                    "timing": "timing/L1/pass_demo.timing.json",
                    "sol_bound": "amd-sol/pass.amd-sol-v2.json",
                },
                "derived_evidence_refs": {
                    "formula": "solar/pass.solar-derivation.json#groups.formula_evidence"
                },
            }
        ],
    }


def test_parity_gap_report_aggregates_denominators_and_blockers():
    report = build_parity_gap_report(
        manifest=None,
        inventory=_inventory(),
        readiness=_readiness(),
        ready_subset=_ready_subset(),
        execution_closure=_execution_closure(),
        amd_score_report=_amd_score_report(),
        created_at="2026-05-23T00:00:00Z",
    )
    payload = report.model_dump(mode="json")

    assert payload["schema_version"] == "sol_execbench.parity_gap_report.v1"
    assert payload["suite"]["discovered"] == 2
    assert payload["suite"]["parsed"] == 2
    assert payload["suite"]["ready"] == 1
    assert payload["suite"]["blocked"] == 1
    assert payload["suite"]["not_attempted"] == 1
    assert payload["suite"]["attempted"] == 1
    assert payload["suite"]["passed"] == 1
    assert payload["suite"]["degraded"] == 1
    blocker_codes = {blocker["reason_code"] for blocker in payload["blockers"]}
    assert blocker_codes == {"safetensors_asset_missing", "timing_evidence_missing"}
    assert payload["report_checksum"]["value"]


def test_parity_gap_report_tracks_evidence_completeness_and_markdown():
    report = build_parity_gap_report(
        manifest=None,
        inventory=_inventory(),
        readiness=_readiness(),
        ready_subset=_ready_subset(),
        execution_closure=_execution_closure(),
        amd_score_report=_amd_score_report(),
        created_at="2026-05-23T00:00:00Z",
    )
    markdown = render_parity_gap_markdown(report)
    evidence = report.model_dump(mode="json")["evidence_completeness"]

    assert evidence["present"]["trace"] == 1
    assert evidence["present"]["amd_native_score"] == 2
    assert evidence["present"]["amd_sol"] == 2
    assert evidence["present"]["solar_derivation"] == 1
    assert evidence["missing"]["timing"] == 1
    assert "# SOL ExecBench ROCm Parity Gap Report" in markdown
    assert "not full 235-problem validation" in markdown
    assert "not upstream SOLAR parity" in markdown


def test_parity_gap_report_serialization_is_deterministic():
    first = build_parity_gap_report(
        manifest=None,
        inventory=_inventory(),
        readiness=_readiness(),
        ready_subset=_ready_subset(),
        execution_closure=_execution_closure(),
        amd_score_report=_amd_score_report(),
        created_at="2026-05-23T00:00:00Z",
    )
    second = build_parity_gap_report(
        manifest=None,
        inventory=_inventory(),
        readiness=_readiness(),
        ready_subset=_ready_subset(),
        execution_closure=_execution_closure(),
        amd_score_report=_amd_score_report(),
        created_at="2026-05-23T00:00:00Z",
    )

    assert first.to_json() == second.to_json()
    assert render_parity_gap_markdown(first) == render_parity_gap_markdown(second)


def test_report_parity_gaps_script_writes_json_and_markdown(tmp_path, monkeypatch):
    inventory_path = tmp_path / "inventory.json"
    readiness_path = tmp_path / "readiness.json"
    ready_subset_path = tmp_path / "ready_subset.json"
    closure_path = tmp_path / "execution_closure.json"
    score_path = tmp_path / "amd_score.json"
    for path, payload in (
        (inventory_path, _inventory()),
        (readiness_path, _readiness()),
        (ready_subset_path, _ready_subset()),
        (closure_path, _execution_closure()),
        (score_path, _amd_score_report()),
    ):
        path.write_text(json.dumps(payload))

    json_output = tmp_path / "gap.json"
    markdown_output = tmp_path / "gap.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_parity_gaps.py",
            "--inventory",
            str(inventory_path),
            "--readiness",
            str(readiness_path),
            "--ready-subset",
            str(ready_subset_path),
            "--execution-closure",
            str(closure_path),
            "--amd-score-report",
            str(score_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    report_parity_gaps.main()

    assert json.loads(json_output.read_text())["schema_version"] == (
        "sol_execbench.parity_gap_report.v1"
    )
    assert "Suite Denominators" in markdown_output.read_text()
