from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.dataset.run_state import (
    closure_status_for_trace,
    closure_status_with_evidence,
    discover_problems,
    requested_evidence_requirements,
    selected_workload_rows,
    trace_map,
    workload_key,
)


def _write_problem(root: Path, category: str, name: str) -> Path:
    problem = root / category / name
    problem.mkdir(parents=True)
    (problem / "definition.json").write_text("{}")
    (problem / "workload.jsonl").write_text("")
    return problem


def test_discover_problems_filters_known_categories(tmp_path):
    expected = _write_problem(tmp_path, "L1", "a")
    _write_problem(tmp_path, "Unknown", "b")

    assert discover_problems(tmp_path) == [expected]
    assert discover_problems(tmp_path, ["Unknown"], known_categories={"Unknown"}) == [
        tmp_path / "Unknown" / "b"
    ]


def test_selected_workload_rows_deduplicates_caps_and_reports_missing(tmp_path):
    workload_path = tmp_path / "workload.jsonl"
    rows = [
        {"uuid": "w0", "axes": {}},
        {"uuid": "w1", "axes": {}},
        {"uuid": "w2", "axes": {}},
    ]
    workload_path.write_text("\n".join(json.dumps(row) for row in rows))

    selected, refs, capped, missing = selected_workload_rows(
        workload_path,
        [
            {"uuid": "w1", "row_index": 1},
            {"uuid": "w1", "row_index": 1},
            {"uuid": "missing", "row_index": 9},
            {"uuid": "w0", "row_index": 0},
            {"uuid": "w2", "row_index": 2},
        ],
        max_workloads=2,
    )

    assert [json.loads(line)["uuid"] for line in selected] == ["w0", "w1"]
    assert [ref["uuid"] for ref in refs] == ["w0", "w1"]
    assert [ref["uuid"] for ref in capped] == ["w2"]
    assert [ref["uuid"] for ref in missing] == ["missing"]


def test_trace_helpers_preserve_status_semantics():
    traces = [
        {"workload": {"uuid": "a"}, "evaluation": {"status": "PASSED"}},
        {"workload": {}, "evaluation": {"status": "RUNTIME_ERROR"}},
    ]

    indexed = trace_map(traces)

    assert workload_key("a", 0) == ("uuid", "a")
    assert workload_key(None, 1) == ("row_index", 1)
    assert indexed[("uuid", "a")] is traces[0]
    assert indexed[("row_index", 1)] is traces[1]
    assert closure_status_for_trace(traces[0]) == "attempted_passed"
    assert closure_status_for_trace(traces[0], skipped=True) == "skipped_existing_pass"
    assert closure_status_for_trace(traces[1]) == "attempted_failed"
    assert closure_status_for_trace(None) == "missing_trace"
    assert (
        closure_status_with_evidence("attempted_passed", ["missing_timing_evidence"])
        == "derived_evidence_missing"
    )


def test_requested_evidence_requirements_follow_option_order(tmp_path):
    assert requested_evidence_requirements() == []
    assert requested_evidence_requirements(
        timing_evidence_dir=tmp_path / "timing",
        amd_sol_bound_dir=tmp_path / "sol",
        amd_score_report=tmp_path / "score.json",
        solar_derivation=tmp_path / "solar",
    ) == [
        "amd_score",
        "amd_sol_bound",
        "solar_derivation",
        "timing_evidence",
    ]
