from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)


def _definition(name: str = "matmul_demo") -> dict:
    return {
        "name": name,
        "axes": {
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        "inputs": {
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["M", "N"], "dtype": "float32"}},
        "reference": "def run(a, b):\n    return a @ b",
    }


def _workload(uuid: str, m: int = 2) -> dict:
    return {
        "uuid": uuid,
        "axes": {"M": m},
        "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
    }


def _trace(uuid: str, status: str = "PASSED") -> dict:
    return {
        "definition": "matmul_demo",
        "workload": _workload(uuid),
        "solution": "solution",
        "evaluation": {
            "status": status,
            "environment": {"hardware": "AMD gfx1200", "libs": {}},
            "timestamp": "2026-05-23T00:00:00Z",
            "correctness": {},
            "performance": {
                "latency_ms": 1.0,
                "reference_latency_ms": 2.0,
                "speedup_factor": 2.0,
            },
        },
    }


def _write_problem(dataset_root: Path, category: str, name: str, workloads: list[dict]) -> Path:
    problem_dir = dataset_root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_definition()))
    (problem_dir / "workload.jsonl").write_text(
        "\n".join(json.dumps(workload) for workload in workloads) + "\n"
    )
    return problem_dir


def _ready_subset(path: Path, *, problems: list[dict]) -> Path:
    payload = {
        "schema_version": "sol_execbench.ready_subset.v1",
        "created_at": "2026-05-23T00:00:00Z",
        "dataset_root": "dataset",
        "readiness_checksum": "readiness-sha",
        "selected_categories": ["L1"],
        "included_workloads": sum(len(problem["workloads"]) for problem in problems),
        "excluded_workloads": 0,
        "problems": problems,
        "claim_boundary": {"ready_to_attempt_rocm_execution": bool(problems)},
        "ready_subset_checksum": {"value": "ready-sha"},
    }
    path.write_text(json.dumps(payload))
    return path


def _readiness(path: Path) -> Path:
    payload = {
        "schema_version": "sol_execbench.rocm_readiness.v1",
        "created_at": "2026-05-23T00:00:00Z",
        "selected_categories": ["L1"],
        "problems": [],
        "workloads": [
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
                        "message": "missing",
                        "next_action": "acquire asset",
                    }
                ],
            }
        ],
        "readiness_checksum": {"value": "readiness-sha"},
    }
    path.write_text(json.dumps(payload))
    return path


def test_ready_subset_runs_through_existing_run_cli_and_stages_workload(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    problem_dir = _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("selected-workload"), _workload("filtered-workload", 4)],
    )
    original_workload = (problem_dir / "workload.jsonl").read_text()
    subset_path = _ready_subset(
        tmp_path / "ready_subset.json",
        problems=[
            {
                "category": "L1",
                "problem_id": "L1/matmul_demo",
                "problem_path": "L1/matmul_demo",
                "workloads": [{"uuid": "selected-workload", "row_index": 0}],
            }
        ],
    )
    output_dir = tmp_path / "out"
    calls: list[Path] = []

    def run_cli(*, workload_path: Path, **kwargs):
        calls.append(workload_path)
        return [_trace("selected-workload")]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    staged_workload = output_dir / "L1" / "matmul_demo" / "workload.jsonl"

    assert calls == [staged_workload]
    assert (problem_dir / "workload.jsonl").read_text() == original_workload
    assert [json.loads(line)["uuid"] for line in staged_workload.read_text().splitlines()] == [
        "selected-workload"
    ]
    assert closure["schema_version"] == "sol_execbench.execution_closure.v1"
    assert set(closure) == {
        "claim_boundary",
        "created_at",
        "execution_closure_checksum",
        "filters",
        "provenance",
        "provenance_mismatches",
        "records",
        "schema_version",
        "source_refs",
        "status",
        "totals",
    }
    assert closure["execution_closure_checksum"]["algorithm"] == "sha256"
    assert closure["status"] == "completed"
    assert closure["totals"]["attempted_passed"] == 1
    assert closure["records"][0]["closure_status"] == "attempted_passed"
    assert closure["provenance"]["ready_subset_checksum"] == "ready-sha"


def test_ready_subset_reports_no_ready_workloads(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_demo", [_workload("workload")])
    subset_path = _ready_subset(tmp_path / "ready_subset.json", problems=[])
    output_dir = tmp_path / "out"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("no ready workloads should not invoke run_cli")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    assert closure["status"] == "no_ready_workloads"
    assert closure["totals"]["records"] == 0


def test_execution_closure_records_filters_and_readiness_blockers(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("one"), _workload("two", 4)],
    )
    subset_path = _ready_subset(
        tmp_path / "ready_subset.json",
        problems=[
            {
                "category": "L1",
                "problem_id": "L1/matmul_demo",
                "problem_path": "L1/matmul_demo",
                "workloads": [
                    {"uuid": "one", "row_index": 0},
                    {"uuid": "two", "row_index": 1},
                ],
            }
        ],
    )
    readiness_path = _readiness(tmp_path / "readiness.json")
    output_dir = tmp_path / "out"

    def run_cli(*, workload_path: Path, **kwargs):
        assert len(workload_path.read_text().splitlines()) == 1
        return [_trace("one")]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--readiness",
            str(readiness_path),
            "--max-workloads",
            "1",
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    statuses = {
        (record["workload_uuid"], record["closure_status"]): record
        for record in closure["records"]
    }

    assert ("one", "attempted_passed") in statuses
    assert statuses[("two", "filtered")]["filter_reasons"] == ["max_workloads_cap"]
    blocked = statuses[("blocked-workload", "not_attempted")]
    assert blocked["readiness_status"] == "runtime_blocked"
    assert blocked["readiness_reason_codes"] == ["safetensors_asset_missing"]


def test_execution_closure_marks_missing_requested_derived_evidence(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_demo", [_workload("selected-workload")])
    subset_path = _ready_subset(
        tmp_path / "ready_subset.json",
        problems=[
            {
                "category": "L1",
                "problem_id": "L1/matmul_demo",
                "problem_path": "L1/matmul_demo",
                "workloads": [{"uuid": "selected-workload", "row_index": 0}],
            }
        ],
    )
    output_dir = tmp_path / "out"

    def run_cli(*args, **kwargs):
        return [_trace("selected-workload")]

    def no_timing_evidence(**kwargs):
        return {"profiler_collected": False}

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        run_dataset, "collect_timing_evidence_for_problem", no_timing_evidence
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--timing-evidence-dir",
            str(tmp_path / "timing"),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    record = closure["records"][0]
    assert record["trace_status"] == "PASSED"
    assert record["closure_status"] == "derived_evidence_missing"
    assert record["evidence_gaps"] == ["timing_evidence_missing"]


def test_execution_closure_records_are_sorted_by_helper_contract(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "a_demo", [_workload("z-workload")])
    _write_problem(dataset_root, "L1", "b_demo", [_workload("a-workload")])
    subset_path = _ready_subset(
        tmp_path / "ready_subset.json",
        problems=[
            {
                "category": "L1",
                "problem_id": "L1/b_demo",
                "problem_path": "L1/b_demo",
                "workloads": [{"uuid": "a-workload", "row_index": 0}],
            },
            {
                "category": "L1",
                "problem_id": "L1/a_demo",
                "problem_path": "L1/a_demo",
                "workloads": [{"uuid": "z-workload", "row_index": 0}],
            },
        ],
    )
    output_dir = tmp_path / "out"

    def run_cli(*, workload_path: Path, **kwargs):
        uuid = json.loads(workload_path.read_text().splitlines()[0])["uuid"]
        return [_trace(uuid)]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    assert [
        (record["problem_id"], record["row_index"], record["workload_uuid"], record["closure_status"])
        for record in closure["records"]
    ] == [
        ("L1/a_demo", 0, "z-workload", "attempted_passed"),
        ("L1/b_demo", 0, "a-workload", "attempted_passed"),
    ]


def test_execution_closure_preserves_skipped_existing_pass_without_provenance_enforcement(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_demo", [_workload("selected-workload")])
    subset_path = _ready_subset(
        tmp_path / "ready_subset.json",
        problems=[
            {
                "category": "L1",
                "problem_id": "L1/matmul_demo",
                "problem_path": "L1/matmul_demo",
                "workloads": [{"uuid": "selected-workload", "row_index": 0}],
            }
        ],
    )
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps([_trace("selected-workload")]))

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("passing existing traces should be skipped")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(subset_path),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    assert closure["records"][0]["closure_status"] == "skipped_existing_pass"
    assert closure["provenance_mismatches"] == []
