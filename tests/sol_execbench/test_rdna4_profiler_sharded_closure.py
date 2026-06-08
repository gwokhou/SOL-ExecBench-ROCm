from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_sharded_closure.py"
SPEC = spec_from_file_location("run_rdna4_profiler_sharded_closure", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
closure = module_from_spec(SPEC)
sys.modules[SPEC.name] = closure
SPEC.loader.exec_module(closure)


def _definition(name: str) -> dict:
    return {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    }


def _workload(kind: str = "random") -> dict:
    spec = {"type": kind}
    if kind == "safetensors":
        spec.update({"path": "missing.safetensors", "tensor_key": "x"})
    return {
        "uuid": f"{kind}-w",
        "axes": {"N": 4},
        "inputs": {"x": spec},
    }


def _write_problem(
    root: Path,
    category: str,
    name: str,
    *,
    workload_kind: str = "random",
) -> None:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(_definition(name)) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(_workload(workload_kind)) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "reference.py").write_text(
        "def run(x):\n    return x\n",
        encoding="utf-8",
    )


def _write_timing(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _coverage(dataset_root: Path, timing_root: Path):
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    return build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_root,),
        created_at="2026-06-08T00:00:00Z",
    )


def test_sharded_closure_audit_targets_partial_and_blocked_only(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "L1", "full")
    _write_problem(dataset_root, "L1", "partial")
    _write_problem(dataset_root, "L1", "blocked")
    _write_problem(dataset_root, "L1", "fallback")
    _write_problem(dataset_root, "L1", "readiness", workload_kind="safetensors")
    _write_timing(
        timing_root / "L1" / "full.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "kernel_duration_ms": 1.0,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {"full_workload_coverage": True},
        },
    )
    _write_timing(
        timing_root / "L1" / "partial.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "kernel_duration_ms": 1.0,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "profiled_workload_count": 1,
                "expected_workload_count": 2,
                "trace_status_counts": {"PASSED": 1},
                "full_workload_coverage": False,
                "failure_reason": "missing workload slice",
            },
        },
    )
    _write_timing(
        timing_root / "L1" / "blocked.timing.json",
        {
            "profiler_collected": False,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {"backend": "rocprofv3", "parsed_rows": []},
            "replacement_metadata": {
                "profiled_workload_count": 0,
                "expected_workload_count": 1,
                "full_workload_coverage": False,
                "failure_reason": "rocprofv3 command timed out after 900 seconds",
            },
        },
    )
    _write_timing(
        timing_root / "L1" / "fallback.timing.json",
        {
            "profiler_collected": False,
            "selection": {"policy": {"backend": "device_events"}},
            "evidence": None,
        },
    )

    report = closure.build_sharded_closure_audit(_coverage(dataset_root, timing_root))

    assert report["target_count"] == 2
    assert [target["problem_id"] for target in report["targets"]] == [
        "L1/partial",
        "L1/blocked",
    ]
    actions = {
        target["problem_id"]: target["recommended_action"]
        for target in report["targets"]
    }
    assert actions == {
        "L1/blocked": "fresh_workload_sharded_profile",
        "L1/partial": "complete_missing_workload_slices",
    }
    assert report["coverage_summary"]["fallback_timing_problems"] == 1
    assert report["coverage_summary"]["readiness_blocked_problems"] == 1


def test_sharded_closure_audit_can_render_markdown(tmp_path):
    report = {
        "target_count": 1,
        "target_statuses": ["partial_profiler_backed"],
        "recommended_action_counts": {"complete_missing_workload_slices": 1},
        "targets": [
            {
                "problem_id": "L1/example",
                "status": "partial_profiler_backed",
                "workload_count": 2,
                "recommended_action": "complete_missing_workload_slices",
                "failure_reason": "missing | slice",
            }
        ],
        "claim_boundary": "Audit only.",
    }

    markdown = closure.render_sharded_closure_audit_markdown(report)

    assert "Target count: `1`" in markdown
    assert "missing \\| slice" in markdown
