from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from sol_execbench.core.scoring.amd_score import build_amd_native_suite_report

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)
build_amd_score_reports_for_problem = run_dataset.build_amd_score_reports_for_problem


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
