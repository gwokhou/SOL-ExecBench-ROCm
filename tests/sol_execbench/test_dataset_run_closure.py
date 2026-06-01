from __future__ import annotations

import json

from sol_execbench.core.dataset.run_closure import (
    closure_record,
    closure_totals,
    derived_evidence_for_workload,
    prior_closure_provenance,
    stale_provenance_mismatch,
    write_execution_closure,
)


def test_closure_record_extracts_readiness_reason_codes():
    record = closure_record(
        category="L1",
        problem_id="L1/demo",
        problem_path="L1/demo",
        workload_uuid="w0",
        row_index=0,
        closure_status="not_attempted",
        readiness={
            "status": "runtime_blocked",
            "reasons": [{"code": "missing_asset"}, {"message": "no code"}],
        },
        filter_reasons=["readiness_blocked"],
    )

    assert record["readiness_status"] == "runtime_blocked"
    assert record["readiness_reason_codes"] == ["missing_asset"]
    assert record["filter_reasons"] == ["readiness_blocked"]


def test_prior_closure_provenance_reports_stale_states(tmp_path):
    missing = tmp_path / "missing.json"
    assert prior_closure_provenance(missing) == (
        None,
        stale_provenance_mismatch(observed=None),
    )

    malformed = tmp_path / "bad.json"
    malformed.write_text("{")
    assert prior_closure_provenance(malformed)[1]["observed"] == "unreadable"

    valid = tmp_path / "closure.json"
    valid.write_text(json.dumps({"provenance": {"dataset_root": "dataset"}}))
    assert prior_closure_provenance(valid) == (
        {"dataset_root": "dataset"},
        None,
    )


def test_closure_totals_and_writer_preserve_claim_boundary(tmp_path):
    record = closure_record(
        category="L1",
        problem_id="L1/demo",
        problem_path="L1/demo",
        workload_uuid="w0",
        row_index=0,
        closure_status="attempted_passed",
    )
    path = tmp_path / "execution_closure.json"

    assert closure_totals([record])["attempted_passed"] == 1

    write_execution_closure(
        path=path,
        records=[record],
        provenance={"dataset_root": "dataset"},
        filters={"ready_subset": True},
    )

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "sol_execbench.execution_closure.v1"
    assert payload["claim_boundary"]["bounded_ready_subset_execution"] is True
    assert payload["claim_boundary"]["full_235_problem_validation"] is False
    assert payload["claim_boundary"]["leaderboard_result"] is False
    assert payload["claim_boundary"]["paper_parity"] is False
    assert payload["totals"]["attempted_passed"] == 1


def test_derived_evidence_for_workload_delegates_ref_and_gap_detection(tmp_path):
    output_dir = tmp_path / "out"
    problem_output_dir = output_dir / "L1" / "demo"
    problem_output_dir.mkdir(parents=True)
    score_report = output_dir / "score.json"
    score_report.write_text("{}")

    refs, gaps = derived_evidence_for_workload(
        definition_name="demo",
        workload_uuid="w0",
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=score_report,
        sol_bound_artifact_dir=output_dir / "sol",
        solar_derivation_dir=None,
        timing_evidence_dir=None,
        category="L1",
    )

    assert refs["amd_score"] == "score.json"
    assert "amd_sol_bound_missing" in gaps
    assert "missing_solar_derivation" not in gaps


def test_derived_evidence_for_workload_combines_present_refs_and_missing_gaps(tmp_path):
    output_dir = tmp_path / "out"
    problem_output_dir = output_dir / "L2" / "demo"
    sol_dir = output_dir / "sol"
    solar_dir = output_dir / "solar"
    timing_dir = output_dir / "timing"
    for path in (problem_output_dir, sol_dir, solar_dir, timing_dir / "L2"):
        path.mkdir(parents=True)
    score_report = output_dir / "score.json"
    score_report.write_text("{}")
    (sol_dir / "demo.w0.amd-sol-v2.json").write_text("{}")
    (timing_dir / "L2" / "demo.timing.json").write_text("{}")

    refs, gaps = derived_evidence_for_workload(
        definition_name="demo",
        workload_uuid="w0",
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=score_report,
        sol_bound_artifact_dir=sol_dir,
        solar_derivation_dir=solar_dir,
        timing_evidence_dir=timing_dir,
        category="L2",
    )

    assert refs == {
        "amd_score": "score.json",
        "amd_sol_bound": "sol/demo.w0.amd-sol-v2.json",
        "timing_evidence": "timing/L2/demo.timing.json",
    }
    assert gaps == ["solar_derivation_missing"]
