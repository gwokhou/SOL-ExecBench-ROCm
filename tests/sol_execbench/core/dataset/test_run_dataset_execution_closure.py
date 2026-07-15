from __future__ import annotations

import importlib.util

import json


import sys

import time

from pathlib import Path

from sol_execbench.core.dataset.execution_closure import (
    build_execution_closure_report,
    write_execution_closure_report,
)


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
        "warmup_runs": 25,
        "iterations": 100,
        "min_measurement_time_seconds": 0.5,
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
        "benchmark_config": {
            "warmup_runs": 25,
            "iterations": 100,
            "min_measurement_time_seconds": 0.5,
            "lock_clocks": False,
        },
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


def test_derived_sidecar_exclusions_match_single_problem_alias(tmp_path):
    problem_dir = tmp_path / "dataset" / "L1" / "matmul_demo"
    problem_dir.mkdir(parents=True)
    workload_path = problem_dir / "workload.jsonl"
    workload_path.write_text(json.dumps(_workload("skip-me")) + "\n")
    exclusions_path = tmp_path / "derived-long-tail.json"
    exclusions_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.long_tail_exclusions.v1",
                "exclusions": [
                    {
                        "scope": "workload",
                        "problem_id": "L1/matmul_demo",
                        "workload_uuid": "skip-me",
                        "row_index": 0,
                        "reason": "derived sidecar long tail",
                        "evidence_ref": "phase-140-log",
                    }
                ],
            }
        )
    )

    exclusions = run_dataset._load_long_tail_exclusions(exclusions_path)

    assert run_dataset._derived_sidecar_exclusions_for_problem(
        problem_id=".",
        workload_path=workload_path,
        long_tail_exclusions=exclusions,
        workload_shard_size=None,
    ) == {
        "skip-me": "derived sidecar long tail (evidence: phase-140-log)",
    }


def test_skipped_problem_summary_records_cdna3_low_precision_reason():
    summary = run_dataset._skipped_problem_summary(
        "Quant/033_nvfp4_moe_routing_with_topk_selection",
        "cdna3_low_precision_hardware_unsupported: gfx942 unsupported",
    )

    assert summary == {
        "problem": "Quant/033_nvfp4_moe_routing_with_topk_selection",
        "total": 0,
        "passed": 0,
        "failed": 0,
        "latencies_ms": [],
        "failure_reasons": [],
        "skipped": 1,
        "skip_reasons": [
            "cdna3_low_precision_hardware_unsupported: gfx942 unsupported"
        ],
    }


def test_inspect_traces_preserves_custom_input_failure_class():
    summary = run_dataset.inspect_traces(
        [
            _trace(
                "bad-custom",
                status="RUNTIME_ERROR",
                log=(
                    "gen_inputs_schema_mismatch: custom_inputs_entrypoint "
                    "returned wrong shape"
                ),
            )
        ],
        "L1/custom_inputs_problem",
    )

    assert summary["failed"] == 1
    assert "gen_inputs_schema_mismatch" in summary["failure_reasons"][0]
    assert "readiness_blocked" not in summary["failure_reasons"][0]


def test_effective_gpu_architecture_uses_environment_fallback(monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE", "gfx942")

    assert run_dataset._effective_gpu_architecture("unknown") == "gfx942"
    assert run_dataset._effective_gpu_architecture("gfx950") == "gfx950"


def test_workload_shard_size_splits_cli_invocations_and_merges_traces(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    problem_dir = _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("one"), _workload("two"), _workload("three")],
    )
    output_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def run_cli(*, workload_path: Path, **kwargs):
        uuids = [
            json.loads(line)["uuid"]
            for line in workload_path.read_text().splitlines()
            if line.strip()
        ]
        calls.append(uuids)
        return [_trace(uuid) for uuid in uuids]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--workload-shard-size",
            "2",
        ],
    )

    run_dataset.main()

    traces = json.loads((output_dir / "L1" / "matmul_demo" / "traces.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())

    assert calls == [["one", "two"], ["three"]]
    assert [trace["workload"]["uuid"] for trace in traces] == ["one", "two", "three"]
    assert summary == [
        {
            "problem": "L1/matmul_demo",
            "total": 3,
            "passed": 3,
            "failed": 0,
            "latencies_ms": [1.0, 1.0, 1.0],
            "failure_reasons": [],
        }
    ]
    assert (problem_dir / "workload.jsonl").read_text().count("\n") == 3


def test_long_tail_exclusions_filter_plain_dataset_and_write_closure(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("fast"), _workload("slow"), _workload("tail")],
    )
    output_dir = tmp_path / "out"
    exclusions_path = tmp_path / "long_tail_exclusions.json"
    exclusions_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.long_tail_exclusions.v1",
                "exclusions": [
                    {
                        "scope": "workload",
                        "problem_id": "L1/matmul_demo",
                        "workload_uuid": "slow",
                        "reason": "known long-tail workload",
                        "evidence_ref": (
                            ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md"
                        ),
                    },
                    {
                        "scope": "shard",
                        "problem_id": "L1/matmul_demo",
                        "shard_index": 2,
                        "reason": "known long-tail shard",
                        "evidence_ref": (
                            "docs/internal/cdna3_gfx942_validation_attempt.md"
                        ),
                    },
                ],
            }
        )
    )
    calls: list[list[str]] = []

    def run_cli(*, workload_path: Path, **kwargs):
        uuids = [
            json.loads(line)["uuid"]
            for line in workload_path.read_text().splitlines()
            if line.strip()
        ]
        calls.append(uuids)
        return [_trace(uuid) for uuid in uuids]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--long-tail-exclusions",
            str(exclusions_path),
            "--workload-shard-size",
            "2",
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())

    assert calls == [["fast"]]
    assert summary[0]["passed"] == 1
    assert closure["totals"]["attempted"] == 0
    assert closure["totals"]["passed"] == 0
    assert closure["totals"]["excluded_long_tail"] == 2
    assert [record["workload_uuid"] for record in closure["records"]] == [
        "slow",
        "tail",
    ]
    assert closure["records"][0]["closure_status"] == "excluded_long_tail"
    assert closure["records"][0]["evidence_refs"] == {
        "long_tail_exclusion": ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md"
    }
    assert closure["provenance"]["long_tail_exclusions_checksum"]
    assert closure["source_refs"]["long_tail_exclusions"] == (
        "long_tail_exclusions.json"
    )


def test_workload_shard_size_preserves_partial_traces_and_marks_failure(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("one"), _workload("two"), _workload("three")],
    )
    output_dir = tmp_path / "out"

    def run_cli(*, workload_path: Path, **kwargs):
        uuid = json.loads(workload_path.read_text())["uuid"]
        if uuid == "two":
            log_path = kwargs["output_dir"] / f"{kwargs['job_name']}_cli.log"
            log_path.write_text("timeout after 960 seconds\ncommand: demo\n")
            return None
        return [_trace(uuid)]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--workload-shard-size",
            "1",
        ],
    )

    run_dataset.main()

    traces = json.loads((output_dir / "L1" / "matmul_demo" / "traces.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())[0]

    assert [trace["workload"]["uuid"] for trace in traces] == ["one", "two", "three"]
    assert [trace["evaluation"]["status"] for trace in traces] == [
        "PASSED",
        "TIMEOUT",
        "PASSED",
    ]
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["failure_reasons"] == [
        "  [TIMEOUT] timed out after 300 seconds: CLI timed out after 960 seconds"
    ]


def test_timeout_overrides_apply_per_workload_shard(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_problem(
        dataset_root,
        "L1",
        "matmul_demo",
        [_workload("one"), _workload("two")],
    )
    output_dir = tmp_path / "out"
    overrides_path = tmp_path / "timeouts.json"
    overrides_path.write_text(json.dumps({"workloads": {"two": 777}}))
    timeouts: list[int] = []

    def run_cli(*, workload_path: Path, timeout: int, **kwargs):
        uuid = json.loads(workload_path.read_text())["uuid"]
        timeouts.append(timeout)
        return [_trace(uuid)]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--workload-shard-size",
            "1",
            "--timeout",
            "300",
            "--timeout-overrides",
            str(overrides_path),
        ],
    )

    run_dataset.main()

    assert timeouts == [300, 777]


def test_blob_precheck_uses_index_from_flashinfer_trace_dir(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "FlashInfer-Bench" / "gqa_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_definition("gqa_demo")))
    ref = "data/flashinfer-trace/blob/workloads/gqa/example.safetensors"
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(
            {
                "uuid": "has-blob",
                "axes": {"M": 2},
                "inputs": {
                    "a": {"type": "safetensors", "path": ref, "tensor_key": "a"},
                    "b": {"type": "random"},
                },
            }
        )
        + "\n"
    )
    trace_root = tmp_path / "flashinfer-trace"
    blob = trace_root / "blob" / "workloads" / "gqa" / "example.safetensors"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"demo")
    output_dir = tmp_path / "out"
    calls = 0

    def run_cli(*, workload_path: Path, **kwargs):
        nonlocal calls
        calls += 1
        return [_trace("has-blob")]

    monkeypatch.setenv("FLASHINFER_TRACE_DIR", str(trace_root))
    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "traces",
            "--execution-mode",
            "pipeline",
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    assert calls == 1
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary[0]["failed"] == 0


def test_pipeline_trace_mode_preserves_summary_order_and_serial_gpu_invocations(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_a", [_workload("a")])
    _write_problem(dataset_root, "L1", "matmul_b", [_workload("b")])
    output_dir = tmp_path / "out"
    calls: list[str] = []

    def run_cli(*, workload_path: Path, **kwargs):
        uuid = json.loads(workload_path.read_text().splitlines()[0])["uuid"]
        calls.append(uuid)
        return [_trace(uuid)]

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "traces",
            "--execution-mode",
            "pipeline",
            "--prepare-jobs",
            "2",
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert sorted(calls) == ["a", "b"]
    assert [row["problem"] for row in summary] == ["L1/matmul_a", "L1/matmul_b"]
    assert (output_dir / "L1" / "matmul_a" / "traces.json").exists()
    assert (output_dir / "L1" / "matmul_b" / "traces.json").exists()


def test_pipeline_all_mode_collects_timing_from_generated_traces(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_demo", [_workload("one")])
    output_dir = tmp_path / "out"
    timing_dir = tmp_path / "timing"
    timing_calls: list[str] = []

    def run_cli(*, workload_path: Path, **kwargs):
        uuid = json.loads(workload_path.read_text().splitlines()[0])["uuid"]
        return [_trace(uuid)]

    def collect_timing_evidence_for_problem(*, job_name: str, **kwargs):
        timing_calls.append(job_name)

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        run_dataset,
        "collect_timing_evidence_for_problem",
        collect_timing_evidence_for_problem,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "all",
            "--execution-mode",
            "pipeline",
            "--output",
            str(output_dir),
            "--timing-evidence-dir",
            str(timing_dir),
        ],
    )

    run_dataset.main()

    assert timing_calls == ["ref_matmul_demo"]
    assert (output_dir / "L1" / "matmul_demo" / "traces.json").exists()


def test_blob_precheck_fails_before_gpu_invocation(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "FlashInfer-Bench" / "gqa_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_definition("gqa_demo")))
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(
            {
                "uuid": "missing-blob",
                "axes": {"M": 2},
                "inputs": {
                    "a": {
                        "type": "safetensors",
                        "path": "data/flashinfer-trace/blob/missing.safetensors",
                        "tensor_key": "a",
                    },
                    "b": {"type": "random"},
                },
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "out"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("missing safetensors should fail before GPU invocation")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "traces",
            "--execution-mode",
            "pipeline",
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary[0]["problem"] == "FlashInfer-Bench/gqa_demo"
    assert summary[0]["failed"] == 1
    assert "missing safetensors blobs" in summary[0]["failure_reasons"][0]


def test_pipeline_problem_log_order_buffers_out_of_order_prepare(
    tmp_path, monkeypatch, capsys
):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_a", [_workload("a")])
    _write_problem(dataset_root, "L1", "matmul_b", [_workload("b")])
    output_dir = tmp_path / "out"
    original_build_solution = run_dataset.build_solution_for_problem

    def slow_first_build(definition, problem_dir, solution_name):
        if problem_dir.name == "matmul_a":
            time.sleep(0.05)
        return original_build_solution(definition, problem_dir, solution_name)

    def run_cli(*, workload_path: Path, **kwargs):
        uuid = json.loads(workload_path.read_text().splitlines()[0])["uuid"]
        return [_trace(uuid)]

    monkeypatch.setattr(run_dataset, "build_solution_for_problem", slow_first_build)
    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "traces",
            "--execution-mode",
            "pipeline",
            "--prepare-jobs",
            "2",
            "--log-order",
            "problem",
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    stdout = capsys.readouterr().out
    assert stdout.index("[1/2] L1/matmul_a") < stdout.index("[2/2] L1/matmul_b")
