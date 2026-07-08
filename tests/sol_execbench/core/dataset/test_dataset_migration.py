from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.dataset.migration import (
    migrate_flashinfer_trace,
    migrate_sol_execbench,
)

CREATED_AT = "2026-06-04T00:00:00Z"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _definition(name: str = "fixture_op") -> dict[str, object]:
    return {
        "name": name,
        "op_type": "fixture",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    }


def _workload(*, safetensors_path: str | None = None) -> dict[str, object]:
    if safetensors_path is None:
        inputs = {"x": {"type": "random"}}
    else:
        inputs = {
            "x": {
                "type": "safetensors",
                "path": safetensors_path,
                "tensor_key": "x",
            }
        }
    return {"axes": {"N": 4}, "inputs": inputs, "uuid": "fixture-001"}


def _solution() -> dict[str, object]:
    return {
        "name": "fixture_solution",
        "definition": "fixture_op",
        "author": "test",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "kernel.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": False,
        },
        "sources": [{"path": "kernel.py", "content": "def run(x):\n    return x\n"}],
    }


def _write_problem(
    root: Path,
    relative_problem: str,
    *,
    solution: bool = True,
    trace: bool = False,
    safetensors_path: str | None = None,
) -> Path:
    problem_dir = root / relative_problem
    _write_json(problem_dir / "definition.json", _definition())
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(_workload(safetensors_path=safetensors_path), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if solution:
        _write_json(problem_dir / "solution.json", _solution())
    if trace:
        (problem_dir / "trace.jsonl").write_text(
            '{"status": "synthetic"}\n', encoding="utf-8"
        )
    return problem_dir


def test_migrate_sol_execbench_copies_runner_layout_and_manifest_is_deterministic(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _write_problem(source_root, "benchmark/L1/fixture_op")

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=("L1",),
        source_revision="local-rev",
        created_at=CREATED_AT,
    )
    repeat = migrate_sol_execbench(
        source_root,
        output_root,
        categories=("L1",),
        source_revision="local-rev",
        created_at=CREATED_AT,
    )

    assert (output_root / "L1/fixture_op/definition.json").is_file()
    assert (output_root / "L1/fixture_op/workload.jsonl").is_file()
    assert (output_root / "L1/fixture_op/solution.json").is_file()
    assert manifest.model_dump(mode="json") == repeat.model_dump(mode="json")
    assert manifest.source.source_id == "nvidia_sol_execbench"
    assert manifest.license_boundary.repository_redistribution is False
    assert manifest.denominators.discovered_problems == 1
    assert manifest.denominators.blockers == 0
    assert manifest.manifest_checksum is not None
    assert [artifact.output_ref for artifact in manifest.artifacts] == [
        "L1/fixture_op/definition.json",
        "L1/fixture_op/solution.json",
        "L1/fixture_op/workload.jsonl",
    ]


def test_migrate_sol_execbench_records_missing_safetensors_blob(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _write_problem(
        source_root,
        "L2/uses_blob",
        safetensors_path="blobs/missing.safetensors",
    )

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=("L2",),
        created_at=CREATED_AT,
    )

    assert [blocker.code for blocker in manifest.blockers] == [
        "missing_safetensors_blob"
    ]
    assert manifest.blockers[0].problem_id == "L2/uses_blob"
    assert manifest.denominators.blockers == 1


def test_migrate_sol_execbench_copies_present_safetensors_blob(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _write_problem(
        source_root,
        "L2/uses_blob",
        safetensors_path="blobs/present.safetensors",
    )
    blob = source_root / "blobs/present.safetensors"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"synthetic blob")

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=("L2",),
        created_at=CREATED_AT,
    )

    assert (output_root / "blobs/present.safetensors").read_bytes() == b"synthetic blob"
    assert manifest.blockers == []
    assert any(artifact.kind == "safetensors_blob" for artifact in manifest.artifacts)


def test_migrate_sol_execbench_keeps_absolute_safetensors_copy_under_output_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    blob = source_root / "blobs/present.safetensors"
    _write_problem(
        source_root,
        "L2/uses_blob",
        safetensors_path=blob.as_posix(),
    )
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"synthetic blob")

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=("L2",),
        created_at=CREATED_AT,
    )

    assert (output_root / "blobs/present.safetensors").read_bytes() == b"synthetic blob"
    assert manifest.blockers == []


def test_migrate_flashinfer_trace_normalizes_problem_and_blocks_missing_trace(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "flashinfer"
    output_root = tmp_path / "out"
    _write_problem(source_root, "problems/decode", trace=False)

    manifest = migrate_flashinfer_trace(
        source_root,
        output_root,
        source_revision="flash-rev",
        created_at=CREATED_AT,
    )

    assert (output_root / "FlashInfer-Bench/decode/definition.json").is_file()
    assert manifest.source.source_id == "flashinfer_trace"
    assert manifest.license_boundary.repository_redistribution is True
    assert [blocker.code for blocker in manifest.blockers] == ["missing_trace"]
    assert manifest.blockers[0].problem_id == "FlashInfer-Bench/decode"


def test_migrate_flashinfer_trace_copies_trace_and_records_missing_solution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "flashinfer"
    output_root = tmp_path / "out"
    _write_problem(source_root, "nested/prefill", solution=False, trace=True)

    manifest = migrate_flashinfer_trace(source_root, output_root, created_at=CREATED_AT)

    assert (output_root / "FlashInfer-Bench/prefill/trace.jsonl").is_file()
    assert [blocker.code for blocker in manifest.blockers] == ["missing_solution"]


def test_dataset_migration_cli_writes_manifest_json(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    manifest_path = tmp_path / "manifest.json"
    _write_problem(source_root, "L1/fixture_op")

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-sol",
            str(source_root),
            str(output_root),
            "--category",
            "L1",
            "--source-revision",
            "cli-rev",
            "--manifest",
            str(manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert manifest_path.is_file()
    payload = json.loads(result.output)
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["source"]["revision"] == "cli-rev"
    assert payload["denominators"]["blockers"] == 0
