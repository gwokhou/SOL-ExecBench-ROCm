from __future__ import annotations

import json

from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.dataset.run_closure import (
    closure_record,
    closure_totals,
    dataset_reuse_decision,
    derived_evidence_for_workload,
    prior_closure_provenance,
    selected_workload_closure_record,
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
            "readiness_class": "blocked_missing_evidence",
            "reasons": [{"code": "missing_asset"}, {"message": "no code"}],
            "blocker_reports": [
                {
                    "code": "missing_asset",
                    "blocker_type": "missing_blob",
                    "evidence_path": "L1/demo/workload.jsonl",
                }
            ],
        },
        filter_reasons=["readiness_blocked"],
    )

    assert record["readiness_status"] == "runtime_blocked"
    assert record["readiness_class"] == "blocked_missing_evidence"
    assert record["readiness_reason_codes"] == ["missing_asset"]
    assert record["readiness_blocker_codes"] == ["missing_asset"]
    assert record["readiness_blocker_types"] == ["missing_blob"]
    assert record["readiness_evidence_refs"] == {
        "missing_asset": "L1/demo/workload.jsonl"
    }
    assert record["filter_reasons"] == ["readiness_blocked"]


def test_prior_closure_provenance_reports_stale_states(tmp_path):
    missing = tmp_path / "missing.json"
    assert prior_closure_provenance(missing) == (
        None,
        stale_provenance_mismatch(observed=None),
    )

    malformed = tmp_path / "bad.json"
    malformed.write_text("{")
    _, mismatch = prior_closure_provenance(malformed)
    assert mismatch is not None
    assert mismatch["observed"] == "unreadable"

    valid = tmp_path / "closure.json"
    valid.write_text(json.dumps({"provenance": {"dataset_root": "dataset"}}))
    assert prior_closure_provenance(valid) == (
        {"dataset_root": "dataset"},
        None,
    )


def test_dataset_reuse_decision_handles_missing_rerun_and_failed_traces(tmp_path):
    traces = tmp_path / "traces.json"

    missing = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=None,
        provenance={},
    )
    assert missing.should_reuse is False
    assert missing.reason == "missing_traces"

    traces.write_text("[]")
    rerun = dataset_reuse_decision(
        rerun=True,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=None,
        provenance={},
    )
    assert rerun.should_reuse is False
    assert rerun.reason == "rerun_requested"

    failed = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=1,
        execution_closure_path=None,
        provenance={},
    )
    assert failed.should_reuse is False
    assert failed.reason == "previous_failed"


def test_dataset_reuse_decision_preserves_default_existing_pass_reuse(tmp_path):
    traces = tmp_path / "traces.json"
    traces.write_text("[]")

    decision = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=None,
        provenance={"dataset_root": "dataset"},
    )

    assert decision.should_reuse is True
    assert decision.reason == "existing_pass"
    assert decision.provenance_mismatches == ()


def test_dataset_reuse_decision_requires_matching_prior_provenance(tmp_path):
    traces = tmp_path / "traces.json"
    traces.write_text("[]")
    closure = tmp_path / "execution_closure.json"
    provenance = {
        "dataset_root": "dataset",
        "ready_subset_checksum": "ready-sha",
    }
    closure.write_text(json.dumps({"provenance": provenance}))

    matching = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=closure,
        provenance=provenance,
    )
    assert matching.should_reuse is True
    assert matching.reason == "matching_provenance"

    stale = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=closure,
        provenance={**provenance, "ready_subset_checksum": "new-ready-sha"},
    )
    assert stale.should_reuse is False
    assert stale.reason == "stale_provenance"
    assert stale.provenance_mismatches[0]["field"] == "ready_subset_checksum"


def test_dataset_reuse_decision_reports_missing_prior_closure(tmp_path):
    traces = tmp_path / "traces.json"
    traces.write_text("[]")

    decision = dataset_reuse_decision(
        rerun=False,
        traces_path=traces,
        failed_count=0,
        execution_closure_path=tmp_path / "missing.json",
        provenance={"dataset_root": "dataset"},
    )

    assert decision.should_reuse is False
    assert decision.reason == "stale_provenance"
    assert decision.provenance_mismatches == (stale_provenance_mismatch(observed=None),)


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
    assert "amd_sol_evidence_missing" in gaps
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
    sidecar_stem = sidecar_stem_for_workload(
        "demo",
        "w0",
        problem_namespace="L2/demo",
    )
    (sol_dir / f"{sidecar_stem}.amd-sol-v2.json").write_text("{}")
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
        "amd_sol_bound": f"sol/{sidecar_stem}.amd-sol-v2.json",
        "timing_evidence": "timing/L2/demo.timing.json",
    }
    assert gaps == ["solar_derivation_missing"]


def test_selected_workload_closure_record_marks_missing_trace(tmp_path):
    output_dir = tmp_path / "out"
    problem_output_dir = output_dir / "L1" / "demo"
    problem_output_dir.mkdir(parents=True)
    traces_path = problem_output_dir / "traces.json"
    traces_path.write_text("[]")
    solution_path = problem_output_dir / "solution.json"
    solution_path.write_text("{}")

    record = selected_workload_closure_record(
        category="L1",
        problem_id="L1/demo",
        problem_path="L1/demo",
        workload_uuid="missing",
        row_index=0,
        readiness=None,
        trace=None,
        skipped=False,
        traces_path=traces_path,
        summary_ref="summary.json",
        solution_path=solution_path,
        output_dir=output_dir,
        definition_name="demo",
        problem_output_dir=problem_output_dir,
        amd_score_report=None,
        sol_bound_artifact_dir=None,
        solar_derivation_dir=None,
        timing_evidence_dir=None,
    )

    assert record["closure_status"] == "missing_trace"
    assert record["trace_status"] is None
    assert record["trace_ref"] == "L1/demo/traces.json"
    assert record["solution_ref"] == "L1/demo/solution.json"


def test_selected_workload_closure_record_marks_missing_requested_evidence(tmp_path):
    output_dir = tmp_path / "out"
    problem_output_dir = output_dir / "L1" / "demo"
    problem_output_dir.mkdir(parents=True)
    traces_path = problem_output_dir / "traces.json"
    traces_path.write_text("[]")

    record = selected_workload_closure_record(
        category="L1",
        problem_id="L1/demo",
        problem_path="L1/demo",
        workload_uuid="w0",
        row_index=0,
        readiness={"status": "ready", "reasons": []},
        trace={"evaluation": {"status": "PASSED"}},
        skipped=True,
        traces_path=traces_path,
        summary_ref="summary.json",
        solution_path=None,
        output_dir=output_dir,
        definition_name="demo",
        problem_output_dir=problem_output_dir,
        amd_score_report=None,
        sol_bound_artifact_dir=None,
        solar_derivation_dir=None,
        timing_evidence_dir=output_dir / "timing",
    )

    assert record["closure_status"] == "derived_evidence_missing"
    assert record["readiness_status"] == "ready"
    assert record["trace_status"] == "PASSED"
    assert record["evidence_gaps"] == ["timing_evidence_missing"]
    assert record["summary_ref"] == "summary.json"
