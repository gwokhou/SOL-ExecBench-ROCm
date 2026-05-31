from __future__ import annotations

import json
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli


REPO_ROOT = Path(__file__).resolve().parents[2]
LINEAR_BACKWARD_EXAMPLE = REPO_ROOT / "examples/pytorch/linear_backward"
HIP_RMSNORM_EXAMPLE = REPO_ROOT / "examples/hip_cpp/rmsnorm"
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)


pytestmark = [pytest.mark.requires_rocm, pytest.mark.xdist_group("serial")]


def _copy_single_workload_problem(
    tmp_path: Path,
    source_dir: Path,
    *,
    name: str,
    solution_name: str,
) -> Path:
    problem_dir = tmp_path / name
    problem_dir.mkdir()
    for filename in ("definition.json", solution_name):
        shutil.copyfile(source_dir / filename, problem_dir / filename)
    first_workload = (source_dir / "workload.jsonl").read_text().splitlines()[0]
    (problem_dir / "workload.jsonl").write_text(first_workload + "\n")
    return problem_dir


def _linear_backward_problem(tmp_path: Path, *, bad_solution: bool = False) -> Path:
    problem_dir = tmp_path / ("linear_backward_bad" if bad_solution else "linear_backward")
    problem_dir.mkdir()
    shutil.copyfile(
        LINEAR_BACKWARD_EXAMPLE / "definition.json",
        problem_dir / "definition.json",
    )
    first_workload = (LINEAR_BACKWARD_EXAMPLE / "workload.jsonl").read_text().splitlines()[0]
    (problem_dir / "workload.jsonl").write_text(first_workload + "\n")

    solution = json.loads((LINEAR_BACKWARD_EXAMPLE / "solution_python.json").read_text())
    if bad_solution:
        solution["name"] = "linear_backward_bad_pytorch"
        solution["sources"][0]["content"] = (
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(grad_output, x, weight):\n"
            "    grad_input = torch.zeros_like(grad_output @ weight)\n"
            "    grad_weight = torch.zeros_like(grad_output.mT @ x)\n"
            "    return grad_input, grad_weight\n"
        )
    (problem_dir / "solution.json").write_text(json.dumps(solution, indent=2))
    return problem_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sol_execbench_cli_runs_linear_backward_on_rocm(tmp_path: Path):
    problem_dir = _linear_backward_problem(tmp_path)
    trace_path = tmp_path / "linear_backward.trace.jsonl"

    result = CliRunner().invoke(
        cli,
        [
            str(problem_dir),
            "--solution",
            str(problem_dir / "solution.json"),
            "--output",
            str(trace_path),
            "--json",
        ]
    )

    assert result.exit_code == 0, result.output
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] == "PASSED"
    assert "PASSED" in result.output


def test_sol_execbench_cli_reports_incorrect_rocm_result(tmp_path: Path):
    problem_dir = _linear_backward_problem(tmp_path, bad_solution=True)
    trace_path = tmp_path / "linear_backward_bad.trace.jsonl"

    result = CliRunner().invoke(
        cli,
        [
            str(problem_dir),
            "--solution",
            str(problem_dir / "solution.json"),
            "--output",
            str(trace_path),
            "--json",
        ]
    )

    assert result.exit_code == 1
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] != "PASSED"


def test_sol_execbench_cli_runs_hip_cpp_with_static_evidence_on_rocm(tmp_path: Path):
    problem_dir = _copy_single_workload_problem(
        tmp_path,
        HIP_RMSNORM_EXAMPLE,
        name="rmsnorm",
        solution_name="solution_hip.json",
    )
    trace_path = tmp_path / "rmsnorm.trace.jsonl"

    result = CliRunner().invoke(
        cli,
        [
            str(problem_dir),
            "--solution",
            str(problem_dir / "solution_hip.json"),
            "--output",
            str(trace_path),
            "--static-evidence",
            "auto",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["evaluation"]["status"] == "PASSED"

    sidecar_path = trace_path.with_name(f"{trace_path.name}.static-evidence.json")
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == "sol_execbench.static_kernel_evidence.v1"
    assert sidecar["status"] in {
        "collected",
        "partial",
        "unavailable",
        "unsupported",
        "failed",
        "skipped",
    }
    assert sidecar["diagnostic_only"] is True
    for authority_field in (
        "correctness_authority",
        "performance_authority",
        "timing_authority",
        "score_authority",
        "paper_parity_authority",
        "leaderboard_authority",
    ):
        assert sidecar[authority_field] is False
    assert "summary" in sidecar


def test_run_dataset_writes_rocm_execution_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "L1" / "linear_backward"
    problem_dir.mkdir(parents=True)
    for name in ("definition.json", "solution_python.json"):
        shutil.copyfile(LINEAR_BACKWARD_EXAMPLE / name, problem_dir / name)
    first_workload = (LINEAR_BACKWARD_EXAMPLE / "workload.jsonl").read_text().splitlines()[0]
    (problem_dir / "workload.jsonl").write_text(first_workload + "\n")
    ready_subset_path = tmp_path / "ready_subset.json"
    ready_subset_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.ready_subset.v1",
                "created_at": "2026-05-31T00:00:00Z",
                "dataset_root": "dataset",
                "readiness_checksum": "rocm-ready-sha",
                "selected_categories": ["L1"],
                "included_workloads": 1,
                "excluded_workloads": 0,
                "problems": [
                    {
                        "category": "L1",
                        "problem_id": "L1/linear_backward",
                        "problem_path": "L1/linear_backward",
                        "workloads": [{"uuid": "linbwd-001", "row_index": 0}],
                    }
                ],
                "claim_boundary": {"ready_to_attempt_rocm_execution": True},
                "ready_subset_checksum": {"value": "rocm-ready-sha"},
            }
        )
    )
    output_dir = tmp_path / "run-dataset"
    closure_path = tmp_path / "execution_closure.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(ready_subset_path),
            "--solution-name",
            "solution_python.json",
            "--output",
            str(output_dir),
            "--execution-closure",
            str(closure_path),
        ]
    )

    run_dataset.main()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary[0]["problem"] == "L1/linear_backward"
    assert summary[0]["total"] == 1
    assert summary[0]["passed"] == 1
    assert summary[0]["failed"] == 0
    assert len(summary[0]["latencies_ms"]) == 1
    assert summary[0]["failure_reasons"] == []
    closure = json.loads(closure_path.read_text())
    assert closure["schema_version"] == "sol_execbench.execution_closure.v1"
    assert closure["status"] == "completed"
    assert closure["totals"]["attempted_passed"] == 1
    assert closure["records"][0]["closure_status"] == "attempted_passed"
    assert closure["records"][0]["trace_status"] == "PASSED"
    assert closure["records"][0]["trace_ref"] == "L1/linear_backward/traces.json"
    assert closure["provenance"]["solution_mode"] == "named"
    assert closure["provenance"]["solution_name"] == "solution_python.json"


def test_run_dataset_reuses_existing_pass_and_rerun_attempts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "L1" / "linear_backward"
    problem_dir.mkdir(parents=True)
    for name in ("definition.json", "solution_python.json"):
        shutil.copyfile(LINEAR_BACKWARD_EXAMPLE / name, problem_dir / name)
    first_workload = (LINEAR_BACKWARD_EXAMPLE / "workload.jsonl").read_text().splitlines()[0]
    (problem_dir / "workload.jsonl").write_text(first_workload + "\n")
    ready_subset_path = tmp_path / "ready_subset.json"
    ready_subset_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.ready_subset.v1",
                "created_at": "2026-05-31T00:00:00Z",
                "dataset_root": "dataset",
                "readiness_checksum": "rocm-ready-sha",
                "selected_categories": ["L1"],
                "included_workloads": 1,
                "excluded_workloads": 0,
                "problems": [
                    {
                        "category": "L1",
                        "problem_id": "L1/linear_backward",
                        "problem_path": "L1/linear_backward",
                        "workloads": [{"uuid": "linbwd-001", "row_index": 0}],
                    }
                ],
                "claim_boundary": {"ready_to_attempt_rocm_execution": True},
                "ready_subset_checksum": {"value": "rocm-ready-sha"},
            }
        )
    )
    output_dir = tmp_path / "run-dataset"
    closure_path = tmp_path / "execution_closure.json"

    base_argv = [
        "run_dataset.py",
        str(dataset_root),
        "--ready-subset",
        str(ready_subset_path),
        "--solution-name",
        "solution_python.json",
        "--output",
        str(output_dir),
        "--execution-closure",
        str(closure_path),
    ]

    monkeypatch.setattr(sys, "argv", base_argv)
    run_dataset.main()
    first_closure = json.loads(closure_path.read_text())
    assert first_closure["totals"]["attempted_passed"] == 1
    assert first_closure["records"][0]["closure_status"] == "attempted_passed"

    monkeypatch.setattr(sys, "argv", base_argv)
    run_dataset.main()
    skipped_closure = json.loads(closure_path.read_text())
    assert skipped_closure["totals"]["skipped_existing_pass"] == 1
    assert skipped_closure["records"][0]["closure_status"] == "skipped_existing_pass"
    assert skipped_closure["records"][0]["trace_status"] == "PASSED"

    monkeypatch.setattr(sys, "argv", [*base_argv, "--rerun"])
    run_dataset.main()
    rerun_closure = json.loads(closure_path.read_text())
    assert rerun_closure["totals"]["attempted_passed"] == 1
    assert rerun_closure["records"][0]["closure_status"] == "attempted_passed"
    assert rerun_closure["records"][0]["trace_status"] == "PASSED"
    assert rerun_closure["provenance"]["rerun"] is True
