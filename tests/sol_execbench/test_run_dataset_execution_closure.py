from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from sol_execbench.core.dataset.execution_closure import (
    build_execution_closure_report,
    write_execution_closure_report,
)
from sol_execbench.core.dataset.evidence_refs import (
    build_derived_evidence_refs,
    sidecar_stem_for_workload,
)

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


def test_effective_gpu_architecture_uses_environment_fallback(monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE", "gfx942")

    assert run_dataset._effective_gpu_architecture("unknown") == "gfx942"
    assert run_dataset._effective_gpu_architecture("gfx950") == "gfx950"


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
    assert [
        json.loads(line)["uuid"] for line in staged_workload.read_text().splitlines()
    ] == ["selected-workload"]
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
    readiness_path = _readiness(tmp_path / "readiness.json")
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
            "--readiness",
            str(readiness_path),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    closure = json.loads((output_dir / "execution_closure.json").read_text())
    assert closure["status"] == "no_ready_workloads"
    assert closure["totals"]["records"] == 1
    assert closure["totals"]["not_attempted"] == 1
    assert closure["records"][0]["filter_reasons"] == ["readiness_blocked"]
    assert json.loads((output_dir / "summary.json").read_text()) == []


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
    assert blocked["readiness_class"] == "blocked_missing_evidence"
    assert blocked["readiness_reason_codes"] == ["safetensors_asset_missing"]
    assert blocked["readiness_blocker_codes"] == ["safetensors_asset_missing"]
    assert blocked["readiness_blocker_types"] == ["missing_blob"]
    assert blocked["readiness_evidence_refs"] == {
        "safetensors_asset_missing": "L1/blocked_demo/workload.jsonl"
    }
    assert closure["provenance"]["readiness_summary"]["blocker_type_counts"] == {
        "missing_blob": 1
    }


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


def test_build_derived_evidence_refs_reports_present_refs_and_missing_gaps(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    problem_output_dir = output_dir / "L1" / "matmul_demo"
    problem_output_dir.mkdir(parents=True)
    amd_score_report = output_dir / "amd-score.json"
    amd_score_report.write_text("{}")
    sol_bound_dir = output_dir / "sol-bounds"
    sol_bound_dir.mkdir()
    timing_dir = output_dir / "timing"
    timing_problem_dir = timing_dir / "L1"
    timing_problem_dir.mkdir(parents=True)
    (timing_problem_dir / "matmul_demo.timing.json").write_text("{}")
    sidecar_stem = sidecar_stem_for_workload(
        "matmul_demo",
        "selected-workload",
        problem_namespace="L1/matmul_demo",
    )
    (sol_bound_dir / f"{sidecar_stem}.amd-sol-v2.json").write_text("{}")

    refs, gaps = build_derived_evidence_refs(
        definition_name="matmul_demo",
        workload_uuid="selected-workload",
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=amd_score_report,
        sol_bound_artifact_dir=sol_bound_dir,
        solar_derivation_dir=output_dir / "missing-solar",
        timing_evidence_dir=timing_dir,
        category="L1",
    )

    assert refs == {
        "amd_score": "amd-score.json",
        "amd_sol_bound": f"sol-bounds/{sidecar_stem}.amd-sol-v2.json",
        "timing_evidence": "timing/L1/matmul_demo.timing.json",
    }
    assert gaps == ["solar_derivation_missing"]


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
        (
            record["problem_id"],
            record["row_index"],
            record["workload_uuid"],
            record["closure_status"],
        )
        for record in closure["records"]
    ] == [
        ("L1/a_demo", 0, "z-workload", "attempted_passed"),
        ("L1/b_demo", 0, "a-workload", "attempted_passed"),
    ]


def test_execution_closure_existing_pass_requires_matching_provenance(
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
        provenance=_matching_closure_provenance(),
    )

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("matching prior closure provenance should permit reuse")

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


def test_dataset_run_existing_pass_without_closure_keeps_default_skip(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "matmul_demo", [_workload("selected-workload")])
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps([_trace("selected-workload")]))

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("default resume should skip an existing passing trace")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
        ],
    )

    run_dataset.main()

    assert not (output_dir / "execution_closure.json").exists()


def test_execution_closure_custom_path_authorizes_existing_pass_reuse(
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
    custom_closure_path = tmp_path / "custom_execution_closure.json"
    _write_prior_closure(
        custom_closure_path,
        provenance=_matching_closure_provenance(),
    )

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("custom prior closure provenance should permit reuse")

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
            "--execution-closure",
            str(custom_closure_path),
        ],
    )

    run_dataset.main()

    closure = json.loads(custom_closure_path.read_text())
    assert closure["records"][0]["closure_status"] == "skipped_existing_pass"
    assert closure["provenance_mismatches"] == []


def test_execution_closure_existing_pass_without_prior_provenance_runs_fresh(
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
    assert closure["provenance_mismatches"][0]["reason_code"] == "stale_provenance"


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
        "stdout-head-" + "a" * (run_dataset._CLI_LOG_LIMIT + 100) + "stdout-tail"
    )
    large_stderr = (
        "stderr-head-" + "b" * (run_dataset._CLI_LOG_LIMIT + 100) + "stderr-tail"
    )
    result = subprocess.CompletedProcess(
        args=["sol-execbench"],
        returncode=42,
        stdout=large_stdout,
        stderr=large_stderr,
    )

    run_dataset._save_cli_log(output_dir, "failed_job", result)

    cli_log = output_dir / "failed_job_cli.log"
    log_text = cli_log.read_text()
    assert len(log_text) < len(large_stdout) + len(large_stderr)
    assert "[truncated CLI output]" in log_text
    assert "stdout-tail" in log_text
    assert "stderr-tail" in log_text
    assert "stdout-head" not in log_text
    assert "stderr-head" not in log_text
    assert run_dataset._cli_failure_notes(cli_log) == ["CLI failed with exit code 42"]


def test_cli_timeout_logs_are_bounded(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    exc = subprocess.TimeoutExpired(
        cmd=["sol-execbench"],
        timeout=300,
        output="a" * (run_dataset._CLI_LOG_LIMIT + 100),
        stderr="b" * (run_dataset._CLI_LOG_LIMIT + 100),
    )

    run_dataset._save_cli_timeout_log(output_dir, "timeout_job", exc)

    log_text = (output_dir / "timeout_job_cli.log").read_text()
    assert len(log_text) < (run_dataset._CLI_LOG_LIMIT * 2) + 200
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

    assert run_dataset._cli_failure_notes(cli_log) == [
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
        return subprocess.CompletedProcess(
            args=["sol-execbench"],
            returncode=9,
            stdout=json.dumps(_trace("selected-workload")) + "\n",
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
