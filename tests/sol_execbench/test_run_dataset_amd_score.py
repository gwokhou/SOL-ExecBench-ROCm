from __future__ import annotations

import json
import importlib.util
import subprocess
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.scoring.amd_score import build_amd_native_suite_report
from sol_execbench.core.scoring.baseline_artifact import (
    scoring_baseline_artifact_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)
build_amd_score_reports_for_problem = run_dataset.build_amd_score_reports_for_problem
collect_timing_evidence_for_problem = run_dataset.collect_timing_evidence_for_problem


def test_dataset_helper_builds_derived_amd_score_report(tmp_path):
    definition = {
        "name": "matmul_demo",
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
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(
            {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            }
        )
    )
    traces = [
        {
            "definition": "matmul_demo",
            "workload": {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD gfx1200", "libs": {}},
                "timestamp": "2026-05-22T00:00:00Z",
                "correctness": {},
                "performance": {
                    "latency_ms": 1.5,
                    "reference_latency_ms": 2.0,
                    "speedup_factor": 1.333,
                },
            },
        }
    ]

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/matmul_demo/traces.json",
    )
    report = build_amd_native_suite_report(scores).to_dict()

    assert report["derived"] is True
    assert report["canonical_output"] == "trace_jsonl"
    assert report["scored_count"] == 1
    assert report["scores"][0]["evidence_refs"]["trace"] == "L1/matmul_demo/traces.json"
    assert (
        report["scores"][0]["evidence_refs"]["baseline"]
        == "trace.evaluation.performance.reference_latency_ms"
    )
    assert report["scores"][0]["evidence_refs"]["hardware_model"].endswith("gfx1200")
    assert report["scores"][0]["baseline_source"] == "reference_latency"


def test_dataset_helper_uses_scoring_baseline_artifact(tmp_path):
    definition = {
        "name": "matmul_demo",
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
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(
            {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            }
        )
    )
    traces = [
        {
            "definition": "matmul_demo",
            "workload": {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD gfx1200", "libs": {}},
                "timestamp": "2026-05-22T00:00:00Z",
                "correctness": {},
                "performance": {
                    "latency_ms": 1.5,
                    "reference_latency_ms": 9.0,
                    "speedup_factor": 6.0,
                },
            },
        }
    ]
    baseline = scoring_baseline_artifact_from_dict(
        {
            "release": "v1.7",
            "entries": [
                {
                    "definition": "matmul_demo",
                    "workload_uuid": "matmul-workload",
                    "latency_ms": 2.0,
                }
            ],
        },
        source="baselines/v1.7.json",
    )

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/matmul_demo/traces.json",
        baseline_artifact=baseline,
    )
    payload = build_amd_native_suite_report(scores).to_dict()

    assert payload["scores"][0]["baseline_latency_ms"] == 2.0
    assert payload["scores"][0]["baseline_source"] == "scoring_baseline"
    assert payload["scores"][0]["evidence_refs"]["baseline"] == (
        "baselines/v1.7.json#matmul_demo:matmul-workload"
    )


def test_dataset_helper_marks_missing_workload_bound_as_unscored(tmp_path):
    definition = {
        "name": "add_demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x + 1",
    }
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("")
    traces = [
        {
            "definition": "add_demo",
            "workload": {
                "uuid": "missing-workload",
                "axes": {"N": 8},
                "inputs": {"x": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": None,
        }
    ]

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="traces.json",
    )

    assert scores[0].supported is False
    assert scores[0].sol_bound_ms is None


def test_dataset_helper_collects_source_specific_timing_evidence(tmp_path):
    problem_output_dir = tmp_path / "out" / "L1" / "triton_problem"
    problem_output_dir.mkdir(parents=True)
    definition_path = tmp_path / "definition.json"
    workload_path = tmp_path / "workload.jsonl"
    solution_path = problem_output_dir / "solution.json"
    config_path = problem_output_dir / "config.json"
    timing_root = tmp_path / "timing" / "L1"
    for path in (definition_path, workload_path, solution_path, config_path):
        path.write_text("{}")

    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        evidence_dir = timing_root / problem_output_dir.name
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "ref_triton_problem.csv").write_text(
            "Domain,Name,Duration(ns)\nKERNEL_DISPATCH,kernel,9000\n"
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    payload = collect_timing_evidence_for_problem(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        output_dir=problem_output_dir,
        timing_evidence_root=timing_root,
        job_name="ref_triton_problem",
        solution={"spec": {"languages": ["triton"]}},
        benchmark_config=BenchmarkConfig(
            warmup_runs=4,
            iterations=12,
            lock_clocks=True,
        ),
        timeout=30,
        config_path=config_path,
        runner=runner,
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx942",
    )

    output_path = timing_root / "triton_problem.timing.json"
    saved = json.loads(output_path.read_text())

    assert calls
    assert "--" in calls[0]
    assert str(config_path) in calls[0]
    assert payload == saved
    assert saved["profiler_collected"] is True
    assert saved["selection"]["policy"]["source_type"] == "triton"
    assert saved["evidence"]["backend"] == "rocprofv3"
    assert saved["evidence"]["gpu_architecture"] == "gfx942"
    assert saved["evidence"]["warmup_runs"] == 4
    assert saved["evidence"]["iterations"] == 12
    assert saved["evidence"]["trial_count"] == 1
    assert saved["evidence"]["clock_locked"] is True
