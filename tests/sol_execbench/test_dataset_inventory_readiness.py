from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.dataset import build_dataset_inventory


def _definition(
    *,
    name: str = "matmul_forward",
    dtype: str = "float32",
    reference: str = "def run(x):\n    return x\n",
    custom_entrypoint: str | None = None,
) -> dict:
    payload = {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": dtype}},
        "outputs": {"out": {"shape": ["N"], "dtype": dtype}},
        "reference": reference,
    }
    if custom_entrypoint:
        payload["custom_inputs_entrypoint"] = custom_entrypoint
    return payload


def _workload(kind: str = "random", **extra) -> dict:
    spec = {"type": kind}
    spec.update(extra)
    return {"uuid": f"{kind}-w", "axes": {"N": 4}, "inputs": {"x": spec}}


def _write_problem(
    root: Path,
    category: str,
    name: str,
    *,
    definition: dict | None = None,
    workloads: list[dict] | None = None,
    reference_file: bool = True,
    solution_file: bool = False,
) -> Path:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(definition or _definition(name=name)) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (workloads or [_workload()])),
        encoding="utf-8",
    )
    if reference_file:
        (problem_dir / "reference.py").write_text("def run(x):\n    return x\n", encoding="utf-8")
    if solution_file:
        (problem_dir / "solution.json").write_text("{}\n", encoding="utf-8")
    return problem_dir


def test_inventory_records_problem_workload_metadata_and_denominators(tmp_path):
    _write_problem(
        tmp_path,
        "L1",
        "matmul_forward",
        workloads=[
            _workload(),
            _workload("safetensors", path="blob/input.safetensors", tensor_key="x"),
        ],
        solution_file=True,
    )

    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    assert inventory.denominators.discovered_problems == 1
    assert inventory.denominators.parsed_problems == 1
    assert inventory.denominators.parsed_workloads == 2
    problem = inventory.problems[0]
    assert problem.problem_id == "L1/matmul_forward"
    assert problem.reference_available is True
    assert problem.solution_files == ["solution.json"]
    assert problem.definition is not None
    assert problem.definition.op_family_hint == "matmul"
    assert problem.definition.direction_hint == "forward"
    assert problem.workloads[0].input_kinds == {"x": "random"}
    assert problem.workloads[0].resolved_input_shapes == {"x": [4]}
    assert problem.workloads[1].uses_safetensors is True
    assert problem.workloads[1].safetensors_refs == [
        {"input": "x", "path": "blob/input.safetensors", "tensor_key": "x"}
    ]


def test_inventory_records_schema_failures_without_aborting(tmp_path):
    bad_dir = tmp_path / "L1" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "definition.json").write_text('{"name": "bad"}\n', encoding="utf-8")
    (bad_dir / "workload.jsonl").write_text("{}\n", encoding="utf-8")
    _write_problem(tmp_path, "L1", "good")

    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    assert inventory.denominators.discovered_problems == 2
    assert inventory.denominators.parsed_problems == 1
    assert inventory.denominators.schema_failures == 1
    assert any(diagnostic.code == "definition_schema_failure" for diagnostic in inventory.diagnostics)
    assert [problem.schema_status for problem in inventory.problems] == [
        "schema_failure",
        "parsed",
    ]


def test_inventory_records_workload_schema_failure_and_missing_files(tmp_path):
    missing_dir = tmp_path / "L1" / "missing"
    missing_dir.mkdir(parents=True)
    (missing_dir / "definition.json").write_text(
        json.dumps(_definition(name="missing")) + "\n",
        encoding="utf-8",
    )
    _write_problem(tmp_path, "L1", "bad_workload")
    workload_path = tmp_path / "L1" / "bad_workload" / "workload.jsonl"
    workload_path.write_text('{"uuid": "bad"}\n', encoding="utf-8")

    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    assert inventory.denominators.missing_required_files == 1
    assert inventory.denominators.schema_failures == 1
    assert any(diagnostic.code == "missing_required_file" for diagnostic in inventory.diagnostics)
    assert any(diagnostic.code == "workload_schema_failure" for diagnostic in inventory.diagnostics)


def test_inventory_json_is_deterministic(tmp_path):
    _write_problem(tmp_path, "L1", "deterministic")

    first = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )
    second = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    assert first.to_json() == second.to_json()
    assert first.inventory_checksum is not None
    assert first.inventory_checksum.value == second.inventory_checksum.value
