"""Dataset inventory sidecar and migration behavior."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sol_execbench.core.dataset.inventory import build_dataset_inventory
from sol_execbench.core.dataset.migration import migrate_sol_execbench
from sol_execbench.core.dataset.readiness import classify_rocm_readiness
from sol_execbench.core.dataset.ready_subset import build_ready_subset

from .dataset_inventory_fixtures import workload as _workload
from .dataset_inventory_fixtures import write_problem as _write_problem


def test_ready_subset_includes_only_ready_workloads(tmp_path):
    ready_dir = _write_problem(tmp_path, "L1", "ready_problem")
    _write_problem(
        tmp_path,
        "L1",
        "blocked_problem",
        workloads=[
            _workload("safetensors", path="missing.safetensors", tensor_key="x")
        ],
    )
    inventory = build_dataset_inventory(
        tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z"
    )
    readiness = classify_rocm_readiness(
        inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z"
    )
    definition_before = (ready_dir / "definition.json").read_text(encoding="utf-8")
    workload_before = (ready_dir / "workload.jsonl").read_text(encoding="utf-8")

    subset = build_ready_subset(
        readiness, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z"
    )
    second = build_ready_subset(
        readiness, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z"
    )

    assert subset.included_workloads == 1
    assert subset.excluded_workloads == 1
    assert subset.denominator.total_workloads == 2
    assert subset.denominator.included_workloads == 1
    assert subset.denominator.excluded_workloads == 1
    assert [problem.problem_id for problem in subset.problems] == ["L1/ready_problem"]
    included = subset.problems[0].workloads[0]
    assert included.readiness_class == "pytorch_compatible"
    assert included.closure_inputs["problem_id"] == "L1/ready_problem"
    assert [item.problem_id for item in subset.exclusions] == ["L1/blocked_problem"]
    assert subset.exclusions[0].reason_codes == ["safetensors_asset_missing"]
    assert subset.claim_boundary.execution_success is False
    assert subset.claim_boundary.hardware_validation is False
    assert subset.claim_boundary.paper_level_validation is False
    assert subset.claim_boundary.score_authority is False
    assert subset.to_json() == second.to_json()
    assert subset.ready_subset_checksum == second.ready_subset_checksum
    assert (ready_dir / "definition.json").read_text(
        encoding="utf-8"
    ) == definition_before
    assert (ready_dir / "workload.jsonl").read_text(encoding="utf-8") == workload_before


def test_inspect_dataset_cli_writes_requested_sidecars(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "inspect_dataset.py"
    spec = importlib.util.spec_from_file_location("inspect_dataset", script_path)
    assert spec is not None and spec.loader is not None
    inspect_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect_dataset)

    _write_problem(tmp_path / "dataset", "L1", "ready_problem")
    out = tmp_path / "out"
    rc = inspect_dataset.main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--category",
            "L1",
            "--inventory",
            str(out / "inventory.json"),
            "--readiness",
            str(out / "readiness.json"),
            "--ready-subset",
            str(out / "ready_subset.json"),
        ]
    )

    assert rc == 0
    assert (
        json.loads((out / "inventory.json").read_text())["schema_version"]
        == "sol_execbench.dataset_inventory.v1"
    )
    assert (
        json.loads((out / "readiness.json").read_text())["schema_version"]
        == "sol_execbench.rocm_readiness.v1"
    )
    assert (
        json.loads((out / "ready_subset.json").read_text())["schema_version"]
        == "sol_execbench.ready_subset.v1"
    )


def test_sol_migration_output_readiness_preserves_missing_blob_blocker(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _write_problem(
        source_root,
        "L2",
        "uses_blob",
        workloads=[
            _workload("safetensors", path="blobs/missing.safetensors", tensor_key="x")
        ],
        solution_file=True,
    )
    migrate_sol_execbench(
        source_root, output_root, categories=("L2",), created_at="2026-06-04T00:00:00Z"
    )
    inventory = build_dataset_inventory(
        output_root, categories=("L2",), created_at="2026-05-23T00:00:00Z"
    )
    readiness = classify_rocm_readiness(
        inventory, dataset_root=output_root, created_at="2026-05-23T00:00:00Z"
    )
    subset = build_ready_subset(
        readiness, dataset_root=output_root, created_at="2026-05-23T00:00:00Z"
    )

    assert readiness.workloads[0].readiness_class == "blocked_missing_evidence"
    assert readiness.blocker_reports[0].blocker_type == "missing_blob"
    assert subset.included_workloads == 0
    assert subset.excluded_workloads == 1
    assert subset.exclusions[0].blocker_types == ["missing_blob"]
    assert subset.claim_boundary.ready_to_attempt_rocm_execution is False
