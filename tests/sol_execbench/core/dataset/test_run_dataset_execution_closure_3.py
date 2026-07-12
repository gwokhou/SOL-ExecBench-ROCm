from __future__ import annotations

import importlib.util

import json

import subprocess

import sys


from pathlib import Path

from sol_execbench.core.dataset.execution_closure import (
    build_execution_closure_report,
    write_execution_closure_report,
)


import sol_execbench.core.dataset.cli_execution as cli_execution

from sol_execbench_type_helpers import write_amd_contract_inputs

REPO_ROOT = Path(__file__).resolve().parents[4]

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


def _trace(uuid: str, status: str = "PASSED", log: str | None = None) -> dict:
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
            **({"log": log} if log is not None else {}),
        },
    }


def _write_prior_closure(
    path: Path,
    *,
    provenance: dict,
    trace_status: str = "PASSED",
) -> None:
    report = build_execution_closure_report(
        records=[
            {
                "category": "L1",
                "problem_id": "L1/matmul_demo",
                "problem_path": "L1/matmul_demo",
                "workload_uuid": "selected-workload",
                "row_index": 0,
                "closure_status": "attempted_passed",
                "trace_status": trace_status,
            }
        ],
        provenance=provenance,
        filters={"ready_subset": True},
        created_at="2026-05-31T00:00:00Z",
    )
    write_execution_closure_report(report, path)


def _matching_closure_provenance() -> dict:
    return {
        "dataset_root": "dataset",
        "selected_categories": None,
        "limit": None,
        "max_workloads": None,
        "workload_shard_size": None,
        "timeout": 300,
        "warmup_runs": 10,
        "iterations": 50,
        "lock_clocks": False,
        "rerun": False,
        "keep_staging": False,
        "verbose": False,
        "solution_mode": "reference",
        "solution_name": None,
        "output_dir": "out",
        "summary_path": "summary.json",
        "ready_subset_path": "ready_subset.json",
        "ready_subset_checksum": "ready-sha",
        "ready_subset_summary": {
            "dataset_root": "dataset",
            "selected_categories": ["L1"],
            "included_workloads": 1,
            "excluded_workloads": 0,
            "denominator": {},
            "exclusion_reason_codes": [],
            "claim_boundary": {"ready_to_attempt_rocm_execution": True},
        },
        "readiness_path": None,
        "readiness_checksum": "readiness-sha",
        "readiness_summary": {},
        "dataset_manifest_path": None,
        "dataset_manifest_checksum": None,
        "dataset_source_id": None,
        "dataset_migration_kind": None,
        "dataset_source_revision": None,
        "dataset_license_boundary": {},
        "dataset_manifest_summary": {},
        "workload_identity_checksum": None,
        "requested_evidence_requirements": [],
        "git_commit": run_dataset._git_commit(),
        "config_path": None,
        "benchmark_config": {"warmup_runs": 10, "iterations": 50, "lock_clocks": False},
        "derived_evidence": {
            "amd_score_report": None,
            "amd_sol_bound_dir": None,
            "solar_derivation": None,
            "timing_evidence_dir": None,
        },
    }


def _write_problem(
    dataset_root: Path, category: str, name: str, workloads: list[dict]
) -> Path:
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
                "readiness_class": "blocked_missing_evidence",
                "reasons": [
                    {
                        "code": "safetensors_asset_missing",
                        "message": "missing",
                        "next_action": "acquire asset",
                    }
                ],
                "blocker_reports": [
                    {
                        "code": "safetensors_asset_missing",
                        "blocker_type": "missing_blob",
                        "problem_id": "L1/blocked_demo",
                        "problem_path": "L1/blocked_demo",
                        "workload_uuid": "blocked-workload",
                        "row_index": 0,
                        "evidence_path": "L1/blocked_demo/workload.jsonl",
                        "message": "missing",
                        "next_action": "acquire asset",
                    }
                ],
            }
        ],
        "blocker_reports": [
            {
                "code": "safetensors_asset_missing",
                "blocker_type": "missing_blob",
                "problem_id": "L1/blocked_demo",
                "problem_path": "L1/blocked_demo",
                "workload_uuid": "blocked-workload",
                "row_index": 0,
                "evidence_path": "L1/blocked_demo/workload.jsonl",
                "message": "missing",
                "next_action": "acquire asset",
            }
        ],
        "claim_boundary": {
            "ready_to_attempt_rocm_execution": False,
            "execution_success": False,
            "hardware_validation": False,
            "paper_level_validation": False,
            "hosted_leaderboard_parity": False,
            "upstream_solar_equivalence": False,
            "score_authority": False,
        },
        "readiness_checksum": {"value": "readiness-sha"},
    }
    path.write_text(json.dumps(payload))
    return path


def test_execution_closure_existing_pass_mismatched_provenance_runs_fresh(
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

    _write_prior_closure(
        output_dir / "execution_closure.json",
        provenance={
            "dataset_root": "dataset",
            "solution_mode": "named",
            "solution_name": "custom_solution.py",
            "ready_subset_checksum": "old-ready-sha",
            "readiness_checksum": "old-readiness-sha",
            "dataset_manifest_checksum": "old-manifest-sha",
            "workload_identity_checksum": "old-workload-sha",
            "requested_evidence_requirements": ["amd_sol_bound"],
        },
    )
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
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
    assert calls == 1
    assert closure["records"][0]["closure_status"] == "attempted_passed"
    reason_codes = [
        mismatch["reason_code"] for mismatch in closure["provenance_mismatches"]
    ]
    for reason_code in [
        "manifest_checksum_mismatch",
        "readiness_checksum_mismatch",
        "ready_subset_checksum_mismatch",
        "workload_identity_mismatch",
        "solution_mode_mismatch",
        "solution_mismatch",
        "evidence_requirement_mismatch",
    ]:
        assert reason_code in reason_codes


def test_execution_closure_rerun_attempts_existing_pass(tmp_path, monkeypatch):
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
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
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
            "--rerun",
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    assert calls == 1
    assert closure["records"][0]["closure_status"] == "attempted_passed"
    assert closure["records"][0]["trace_status"] == "PASSED"


def test_execution_closure_classifies_cli_no_output_with_bounded_log_ref(
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

    def run_cli(*, output_dir: Path, job_name: str, **kwargs):
        (output_dir / f"{job_name}_cli.log").write_text(
            f"stdout from {tmp_path}\nstderr from {tmp_path}"
        )
        return None

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

    closure_text = (output_dir / "execution_closure.json").read_text()
    closure = json.loads(closure_text)
    record = closure["records"][0]
    assert record["closure_status"] == "attempted_failed"
    assert record["cli_log_ref"] == "L1/matmul_demo/ref_matmul_demo_cli.log"
    assert record["notes"] == ["CLI returned no traces"]
    assert str(tmp_path) not in closure_text
    assert "stdout from" not in closure_text
    assert "stderr from" not in closure_text


def test_cli_failure_logs_are_bounded_and_notes_read_header_only(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    large_stdout = (
        "stdout-head-" + "a" * (cli_execution.CLI_LOG_LIMIT + 100) + "stdout-tail"
    )
    large_stderr = (
        "stderr-head-" + "b" * (cli_execution.CLI_LOG_LIMIT + 100) + "stderr-tail"
    )
    result = subprocess.CompletedProcess(
        args=["sol-execbench"],
        returncode=42,
        stdout=large_stdout,
        stderr=large_stderr,
    )

    cli_execution.save_cli_log(output_dir, "failed_job", result)

    cli_log = output_dir / "failed_job_cli.log"
    log_text = cli_log.read_text()
    assert len(log_text) < len(large_stdout) + len(large_stderr)
    assert "[truncated CLI output]" in log_text
    assert "stdout-tail" in log_text
    assert "stderr-tail" in log_text
    assert "stdout-head" not in log_text
    assert "stderr-head" not in log_text
    assert cli_execution.cli_failure_notes(cli_log) == ["CLI failed with exit code 42"]


def test_cli_timeout_logs_are_bounded(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    exc = subprocess.TimeoutExpired(
        cmd=["sol-execbench"],
        timeout=300,
        output="a" * (cli_execution.CLI_LOG_LIMIT + 100),
        stderr="b" * (cli_execution.CLI_LOG_LIMIT + 100),
    )

    cli_execution.save_cli_timeout_log(output_dir, "timeout_job", exc)

    log_text = (output_dir / "timeout_job_cli.log").read_text()
    assert len(log_text) < (cli_execution.CLI_LOG_LIMIT * 2) + 200
    assert "[truncated CLI output]" in log_text


def test_cli_failure_notes_detects_inner_eval_driver_timeout(tmp_path):
    cli_log = tmp_path / "failed_shard_cli.log"
    cli_log.write_text(
        "exit code: 1\n\n"
        "--- stderr ---\n"
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.12/subprocess.py", line 1253, in _check_timeout\n'
        "    raise TimeoutExpired(\n"
        "subprocess.TimeoutExpired: Command '['python', 'eval_driver.py']' "
        "timed out after 900 seconds\n",
        encoding="utf-8",
    )

    assert cli_execution.cli_failure_notes(cli_log) == [
        "CLI timed out after 900 seconds"
    ]


def test_execution_closure_marks_selected_workload_without_trace_as_missing_trace(
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

    def run_cli(*args, **kwargs):
        return [_trace("other-workload")]

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
    record = closure["records"][0]
    assert record["closure_status"] == "missing_trace"
    assert record["trace_status"] is None
    assert record["trace_ref"] == "L1/matmul_demo/traces.json"


def test_execution_closure_preserves_traces_from_cli_nonzero_exit(
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

    def failed_subprocess_run(*args, **kwargs):
        command = args[0]
        if "--trace-output" not in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="test-revision\n", stderr=""
            )
        trace_path = Path(command[command.index("--trace-output") + 1])
        trace_path.write_text(json.dumps(_trace("selected-workload")) + "\n")
        return subprocess.CompletedProcess(
            args=command,
            returncode=9,
            stdout='{"schema_version":"sol_execbench.cli_response.v1"}\n',
            stderr=f"secret stderr from {tmp_path}",
        )

    monkeypatch.setattr(run_dataset.subprocess, "run", failed_subprocess_run)
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

    closure_text = (output_dir / "execution_closure.json").read_text()
    closure = json.loads(closure_text)
    record = closure["records"][0]
    traces = json.loads((output_dir / "L1" / "matmul_demo" / "traces.json").read_text())

    assert traces == [_trace("selected-workload")]
    assert record["closure_status"] == "attempted_passed"
    assert record["trace_status"] == "PASSED"
    assert record["trace_ref"] == "L1/matmul_demo/traces.json"
    assert record["cli_log_ref"] is None
    assert record["notes"] == []
    assert str(tmp_path) not in closure_text
    assert "secret stderr" not in closure_text


def test_execution_closure_classifies_cli_timeout_with_bounded_log_ref(
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

    def timeout_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["sol-execbench"],
            timeout=360,
            output=f"timeout stdout from {tmp_path}",
            stderr=f"timeout stderr from {tmp_path}",
        )

    monkeypatch.setattr(run_dataset.subprocess, "run", timeout_subprocess_run)
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

    closure_text = (output_dir / "execution_closure.json").read_text()
    closure = json.loads(closure_text)
    record = closure["records"][0]
    assert record["closure_status"] == "attempted_failed"
    assert record["trace_status"] == "TIMEOUT"
    assert record["cli_log_ref"] is None
    assert str(tmp_path) not in closure_text
    assert "timeout stdout" not in closure_text
    assert "timeout stderr" not in closure_text


def test_execution_closure_provenance_uses_bounded_refs(tmp_path, monkeypatch):
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
    readiness_path = _readiness(tmp_path / "readiness.json")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.dataset_migration_manifest.v1",
                "migration_kind": "sol_execbench",
                "source": {
                    "source_id": "nvidia_sol_execbench",
                    "repo_id": "nvidia/SOL-ExecBench",
                    "revision": "local-snapshot",
                    "source_root": str(tmp_path / "restricted-source"),
                },
                "output_root": str(dataset_root),
                "selected_categories": ["L1"],
                "license_boundary": {
                    "source_boundary": "NVIDIA Evaluation Dataset License",
                    "generated_artifact_source_id": "generated_local_migration_artifacts",
                    "license": "NVIDIA Evaluation Dataset License",
                    "redistribution_class": "local-only",
                    "repository_redistribution": False,
                    "release_bundle_redistribution": False,
                    "attribution": "NVIDIA SOL-ExecBench local migration",
                },
                "denominators": {
                    "discovered_problems": 1,
                    "migrated_problems": 1,
                    "generated_artifacts": 2,
                    "blockers": 0,
                    "warnings": 0,
                },
                "blockers": [],
                "manifest_checksum": {"value": "manifest-sha"},
            }
        )
    )
    output_dir = tmp_path / "out"
    score_path = tmp_path / "score.json"
    model_path, fusion_path = write_amd_contract_inputs(tmp_path)

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
            "--readiness",
            str(readiness_path),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(score_path),
            "--amd-hardware-model",
            str(model_path),
            "--fusion-validation",
            str(fusion_path),
        ],
    )

    run_dataset.main()

    closure_text = (output_dir / "execution_closure.json").read_text()
    closure = json.loads(closure_text)
    assert str(tmp_path) not in closure_text
    assert closure["provenance"]["dataset_root"] == "dataset"
    assert closure["provenance"]["output_dir"] == "out"
    assert closure["provenance"]["summary_path"] == "summary.json"
    assert closure["provenance"]["ready_subset_path"] == "ready_subset.json"
    assert closure["provenance"]["readiness_path"] == "readiness.json"
    assert closure["provenance"]["dataset_manifest_path"] == "manifest.json"
    assert closure["provenance"]["dataset_manifest_checksum"] == "manifest-sha"
    assert closure["provenance"]["dataset_source_id"] == "nvidia_sol_execbench"
    assert closure["provenance"]["dataset_migration_kind"] == "sol_execbench"
    assert closure["provenance"]["dataset_source_revision"] == "local-snapshot"
    assert closure["provenance"]["dataset_license_boundary"] == {
        "attribution": "NVIDIA SOL-ExecBench local migration",
        "generated_artifact_source_id": "generated_local_migration_artifacts",
        "license": "NVIDIA Evaluation Dataset License",
        "redistribution_class": "local-only",
        "release_bundle_redistribution": False,
        "repository_redistribution": False,
        "source_boundary": "NVIDIA Evaluation Dataset License",
    }
    assert closure["provenance"]["dataset_manifest_summary"]["source_root"] == (
        "restricted-source"
    )
    assert closure["provenance"]["dataset_manifest_summary"]["denominators"] == {
        "blockers": 0,
        "discovered_problems": 1,
        "generated_artifacts": 2,
        "migrated_problems": 1,
        "warnings": 0,
    }
    assert closure["source_refs"] == {
        "dataset_manifest": "manifest.json",
        "readiness": "readiness.json",
        "ready_subset": "ready_subset.json",
    }
    assert closure["provenance"]["derived_evidence"]["amd_score_report"] == "score.json"
    assert all(
        str(tmp_path) not in arg for arg in closure["provenance"]["command_args"]
    )
