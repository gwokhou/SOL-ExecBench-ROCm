from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from sol_execbench.core.dataset import (
    DatasetInventory,
    InventoryDenominators,
    ProblemInventoryRecord,
    WorkloadInventoryRecord,
    build_dataset_inventory,
    build_ready_subset,
    classify_rocm_readiness,
    migrate_flashinfer_trace,
    migrate_sol_execbench,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INSPECT_DATASET_PATH = REPO_ROOT / "scripts" / "inspect_dataset.py"
spec = importlib.util.spec_from_file_location("inspect_dataset", INSPECT_DATASET_PATH)
assert spec is not None
inspect_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(inspect_dataset)


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


def _write_solution(problem_dir: Path, name: str, payload: dict | str) -> None:
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, sort_keys=True) + "\n"
    (problem_dir / name).write_text(text, encoding="utf-8")


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
    assert second.inventory_checksum is not None
    assert first.inventory_checksum.value == second.inventory_checksum.value


def test_readiness_marks_random_workload_ready(tmp_path):
    _write_problem(tmp_path, "L1", "ready_problem")
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    assert readiness.workloads[0].status == "ready"
    assert readiness.workloads[0].readiness_class == "pytorch_compatible"
    assert readiness.workloads[0].reasons[0].code == "ready_to_attempt_rocm_execution"
    assert readiness.problems[0].status == "ready"
    assert readiness.claim_boundary.ready_to_attempt_rocm_execution is True
    assert readiness.claim_boundary.hardware_validation is False
    assert readiness.claim_boundary.paper_level_validation is False
    assert readiness.claim_boundary.score_authority is False


def test_readiness_blocks_custom_inputs_and_missing_safetensors(tmp_path):
    custom_definition = _definition(
        name="custom_problem",
        reference="def make_inputs(axes, device):\n    return {}\ndef run(x):\n    return x\n",
        custom_entrypoint="make_inputs",
    )
    _write_problem(tmp_path, "L1", "custom_problem", definition=custom_definition, workloads=[_workload("custom")])
    _write_problem(
        tmp_path,
        "L1",
        "safetensors_problem",
        workloads=[_workload("safetensors", path="missing.safetensors", tensor_key="x")],
    )
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    statuses = {record.problem_id: record.status for record in readiness.workloads}
    assert statuses["L1/custom_problem"] == "custom_input_blocked"
    assert statuses["L1/safetensors_problem"] == "runtime_blocked"
    assert {record.readiness_class for record in readiness.workloads} == {
        "blocked_missing_evidence"
    }
    reason_codes = {record.reasons[0].code for record in readiness.workloads}
    assert "custom_input_requires_evaluator_support" in reason_codes
    assert "safetensors_asset_missing" in reason_codes
    assert {blocker.blocker_type for blocker in readiness.blocker_reports} == {
        "missing_blob",
        "missing_evidence",
    }


def test_readiness_marks_quant_and_low_precision_as_needing_hardware_evidence(tmp_path):
    _write_problem(tmp_path, "Quant", "quant_problem")
    _write_problem(tmp_path, "L1", "fp8_problem", definition=_definition(name="fp8_problem", dtype="float8_e4m3fn"))
    inventory = build_dataset_inventory(tmp_path, categories=("L1", "Quant"), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    statuses = {record.problem_id: record.status for record in readiness.workloads}
    assert statuses["Quant/quant_problem"] == "needs_hardware_evidence"
    assert statuses["L1/fp8_problem"] == "needs_hardware_evidence"
    classes = {record.problem_id: record.readiness_class for record in readiness.workloads}
    assert classes["Quant/quant_problem"] == "nvfp4_blackwell_specific"
    assert classes["L1/fp8_problem"] == "blocked_missing_evidence"
    assert all(record.layered_evidence.hardware_validation == "needed" for record in readiness.workloads)


def test_readiness_blocks_schema_failure_and_is_deterministic(tmp_path):
    bad_dir = tmp_path / "L1" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "definition.json").write_text('{"name":"bad"}\n', encoding="utf-8")
    (bad_dir / "workload.jsonl").write_text("{}\n", encoding="utf-8")
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    first = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")
    second = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    assert first.workloads[0].status == "schema_input_blocked"
    assert first.to_json() == second.to_json()
    assert first.readiness_checksum is not None


def test_readiness_does_not_block_torch_cuda_compatibility_text(tmp_path):
    _write_problem(
        tmp_path,
        "L1",
        "compat_problem",
        definition=_definition(
            name="compat_problem",
            reference="def run(x):\n    # torch.cuda is the PyTorch ROCm compatibility namespace\n    return x\n",
        ),
    )
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    assert readiness.workloads[0].status == "ready"


def test_readiness_blocks_nvidia_only_reference_runtime_hints(tmp_path):
    _write_problem(
        tmp_path,
        "L1",
        "cupy_problem",
        definition=_definition(
            name="cupy_problem",
            reference="import cupy\n\ndef run(x):\n    return cupy.asarray(x)\n",
        ),
    )
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    assert inventory.problems[0].definition is not None
    assert "cupy" in inventory.problems[0].definition.reference_runtime_hints
    assert readiness.workloads[0].status == "unsupported_nvidia_only_path"
    assert readiness.workloads[0].readiness_class == "rocm_port_needed"
    assert readiness.workloads[0].reasons[0].code == "nvidia_cuda_runtime_hint"


def test_readiness_classifies_cuda_solution_as_rocm_port_needed(tmp_path):
    problem_dir = _write_problem(tmp_path, "L1", "cuda_solution")
    _write_solution(
        problem_dir,
        "solution_cuda.json",
        {
            "spec": {
                "languages": ["cuda_cpp"],
                "dependencies": ["cuda", "cublas"],
            },
            "sources": [{"path": "kernel.cu", "content": "__global__ void k() {}"}],
        },
    )
    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=tmp_path,
        created_at="2026-05-23T00:00:00Z",
    )

    record = readiness.workloads[0]
    assert record.readiness_class == "rocm_port_needed"
    assert record.status == "unsupported_nvidia_only_path"
    assert record.blocker_reports[0].blocker_type == "cuda_kernel_dependency"
    assert record.reasons[0].code == "cuda_solution_dependency"


def test_readiness_classifies_flashinfer_migration_output(tmp_path):
    source_root = tmp_path / "flashinfer_source"
    output_root = tmp_path / "flashinfer_out"
    problem_dir = _write_problem(
        source_root,
        "nested",
        "decode",
        solution_file=True,
        reference_file=True,
    )
    (problem_dir / "trace.jsonl").write_text('{"status": "synthetic"}\n', encoding="utf-8")
    migrate_flashinfer_trace(
        source_root,
        output_root,
        created_at="2026-06-04T00:00:00Z",
    )
    inventory = build_dataset_inventory(
        output_root,
        categories=("FlashInfer-Bench",),
        created_at="2026-05-23T00:00:00Z",
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=output_root,
        created_at="2026-05-23T00:00:00Z",
    )

    record = readiness.workloads[0]
    assert record.readiness_class == "flashinfer_specific"
    assert record.reasons[0].code == "flashinfer_runtime_assumption"
    assert readiness.blocker_reports[0].blocker_type == "flashinfer_runtime_assumption"


def test_readiness_classifies_blackwell_low_precision_dependency(tmp_path):
    problem_dir = _write_problem(tmp_path, "L1", "blackwell_nvfp4_problem")
    _write_solution(
        problem_dir,
        "solution.json",
        {"sources": [{"path": "kernel.py", "content": "def run(x): return nvfp4(x)"}]},
    )
    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=tmp_path,
        created_at="2026-05-23T00:00:00Z",
    )

    record = readiness.workloads[0]
    assert record.readiness_class == "nvfp4_blackwell_specific"
    assert record.reasons[0].code == "blackwell_low_precision_dependency"
    assert record.blocker_reports[0].blocker_type == "low_precision_format_dependency"


def test_readiness_classifies_unsupported_nvidia_dsl(tmp_path):
    problem_dir = _write_problem(tmp_path, "L1", "cutile_problem")
    _write_solution(
        problem_dir,
        "solution_cutile.json",
        {"spec": {"languages": ["cutile"]}},
    )
    inventory = build_dataset_inventory(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=tmp_path,
        created_at="2026-05-23T00:00:00Z",
    )

    record = readiness.workloads[0]
    assert record.readiness_class == "unsupported"
    assert record.reasons[0].code == "unsupported_nvidia_dsl"


def test_readiness_reports_unsupported_dtype_blocker(tmp_path):
    inventory = DatasetInventory(
        created_at="2026-05-23T00:00:00Z",
        root=tmp_path.as_posix(),
        selected_categories=("L1",),
        categories=[],
        problems=[
            ProblemInventoryRecord(
                category="L1",
                problem_id="L1/unsupported_dtype",
                problem_path="L1/unsupported_dtype",
                definition_path="L1/unsupported_dtype/definition.json",
                workload_path="L1/unsupported_dtype/workload.jsonl",
                schema_status="schema_failure",
                schema_failure="Unsupported dtype 'uint4'",
                workloads=[
                    WorkloadInventoryRecord(
                        uuid="bad-dtype",
                        row_index=1,
                        schema_status="schema_failure",
                        schema_failure="Unsupported dtype 'uint4'",
                    )
                ],
            )
        ],
        denominators=InventoryDenominators(schema_failures=1),
        diagnostics=[],
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=tmp_path,
        created_at="2026-05-23T00:00:00Z",
    )

    record = readiness.workloads[0]
    assert record.readiness_class == "unsupported"
    assert record.status == "dtype_blocked"
    assert record.reasons[0].code == "unsupported_dtype"
    assert readiness.blocker_reports[0].blocker_type == "unsupported_dtype"


def test_readiness_blocks_safetensors_paths_outside_dataset_root(tmp_path):
    for name, path in {"absolute": "/tmp/outside.safetensors", "parent": "../outside.safetensors"}.items():
        _write_problem(
            tmp_path,
            "L1",
            name,
            workloads=[_workload("safetensors", path=path, tensor_key="x")],
        )
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")

    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

    assert {record.status for record in readiness.workloads} == {"runtime_blocked"}
    assert {record.reasons[0].code for record in readiness.workloads} == {
        "safetensors_path_outside_dataset_root"
    }


def test_ready_subset_includes_only_ready_workloads(tmp_path):
    ready_dir = _write_problem(tmp_path, "L1", "ready_problem")
    _write_problem(tmp_path, "L1", "blocked_problem", workloads=[_workload("safetensors", path="missing.safetensors", tensor_key="x")])
    inventory = build_dataset_inventory(tmp_path, categories=("L1",), created_at="2026-05-23T00:00:00Z")
    readiness = classify_rocm_readiness(inventory, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")
    definition_before = (ready_dir / "definition.json").read_text(encoding="utf-8")
    workload_before = (ready_dir / "workload.jsonl").read_text(encoding="utf-8")

    subset = build_ready_subset(readiness, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")
    second = build_ready_subset(readiness, dataset_root=tmp_path, created_at="2026-05-23T00:00:00Z")

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
    assert (ready_dir / "definition.json").read_text(encoding="utf-8") == definition_before
    assert (ready_dir / "workload.jsonl").read_text(encoding="utf-8") == workload_before


def test_inspect_dataset_cli_writes_requested_sidecars(tmp_path):
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
    assert json.loads((out / "inventory.json").read_text())["schema_version"] == "sol_execbench.dataset_inventory.v1"
    assert json.loads((out / "readiness.json").read_text())["schema_version"] == "sol_execbench.rocm_readiness.v1"
    assert json.loads((out / "ready_subset.json").read_text())["schema_version"] == "sol_execbench.ready_subset.v1"


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
        source_root,
        output_root,
        categories=("L2",),
        created_at="2026-06-04T00:00:00Z",
    )
    inventory = build_dataset_inventory(
        output_root,
        categories=("L2",),
        created_at="2026-05-23T00:00:00Z",
    )

    readiness = classify_rocm_readiness(
        inventory,
        dataset_root=output_root,
        created_at="2026-05-23T00:00:00Z",
    )
    subset = build_ready_subset(
        readiness,
        dataset_root=output_root,
        created_at="2026-05-23T00:00:00Z",
    )

    assert readiness.workloads[0].readiness_class == "blocked_missing_evidence"
    assert readiness.blocker_reports[0].blocker_type == "missing_blob"
    assert subset.included_workloads == 0
    assert subset.excluded_workloads == 1
    assert subset.exclusions[0].blocker_types == ["missing_blob"]
    assert subset.claim_boundary.ready_to_attempt_rocm_execution is False
