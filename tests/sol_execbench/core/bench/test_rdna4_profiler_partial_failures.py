from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.dataset.inventory import build_dataset_inventory
from sol_execbench.core.dataset.profiler_timing_coverage import (
    build_profiler_timing_coverage_report,
)
from sol_execbench.core.dataset.readiness import classify_rocm_readiness

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/internal/rdna4/run_rdna4_profiler_partial_failures.py"
)
SPEC = spec_from_file_location("run_rdna4_profiler_partial_failures", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
partial = module_from_spec(SPEC)
sys.modules[SPEC.name] = partial
SPEC.loader.exec_module(partial)


def _definition(name: str) -> dict:
    return {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    }


def _workload() -> dict:
    return {
        "uuid": "w",
        "axes": {"N": 4},
        "inputs": {"x": {"type": "random"}},
    }


def _write_problem(root: Path, name: str) -> None:
    problem_dir = root / "L1" / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(_definition(name)) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(_workload()) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "reference.py").write_text(
        "def run(x):\n    return x\n",
        encoding="utf-8",
    )


def _write_partial_timing(
    root: Path,
    name: str,
    *,
    trace_status_counts: dict[str, int],
    stdout_records: list[dict] | None = None,
    profiled: int = 2,
    expected: int = 2,
    rows: int = 1,
) -> None:
    path = root / "L1" / f"{name}.timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed_rows = [{"is_kernel_activity": True} for _ in range(rows)]
    path.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": float(rows),
                    "parsed_rows": parsed_rows,
                },
                "replacement_metadata": {
                    "profiled_workload_count": profiled,
                    "expected_workload_count": expected,
                    "trace_status_counts": trace_status_counts,
                    "full_workload_coverage": False,
                    "failure_reason": "not all workloads passed",
                },
                "stdout": "".join(
                    json.dumps(record) + "\n" for record in stdout_records or []
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _coverage(dataset_root: Path, timing_root: Path):
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    return build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_root,),
        created_at="2026-06-08T00:00:00Z",
    )


def test_markdown_cell_uses_shared_escaping() -> None:
    assert partial._markdown_cell("a|b\\c\nnext\rline") == "a\\|b\\\\c next line"


def test_failure_mode_classification():
    assert (
        partial.classify_failure_mode(
            trace_counts={"PASSED": 1, "INVALID_REFERENCE": 1},
            profiled_workload_count=2,
            expected_workload_count=2,
            kernel_activity_rows=1,
        )
        == "invalid_reference_only"
    )
    assert (
        partial.classify_failure_mode(
            trace_counts={"RUNTIME_ERROR": 2},
            profiled_workload_count=2,
            expected_workload_count=2,
            kernel_activity_rows=1,
        )
        == "runtime_error_only"
    )
    assert (
        partial.classify_failure_mode(
            trace_counts={"INVALID_REFERENCE": 1, "RUNTIME_ERROR": 1},
            profiled_workload_count=2,
            expected_workload_count=2,
            kernel_activity_rows=1,
        )
        == "mixed_correctness_runtime"
    )
    assert (
        partial.classify_failure_mode(
            trace_counts={"PASSED": 1},
            profiled_workload_count=1,
            expected_workload_count=2,
            kernel_activity_rows=1,
        )
        == "incomplete_workload_coverage"
    )


def test_partial_failure_ledger_excludes_full_profiler_backed(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "invalid")
    _write_problem(dataset_root, "runtime")
    _write_problem(dataset_root, "full")
    _write_partial_timing(
        timing_root,
        "invalid",
        trace_status_counts={"INVALID_REFERENCE": 1, "PASSED": 1},
    )
    _write_partial_timing(
        timing_root,
        "runtime",
        trace_status_counts={"RUNTIME_ERROR": 2},
    )
    full_path = timing_root / "L1" / "full.timing.json"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": 1.0,
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {"full_workload_coverage": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = partial.build_partial_failure_classification(
        _coverage(dataset_root, timing_root)
    )

    assert report["partial_target_count"] == 2
    assert report["failure_mode_counts"] == {
        "invalid_reference_only": 1,
        "runtime_error_only": 1,
    }
    decisions = {
        target["problem_id"]: target["closure_decision"] for target in report["targets"]
    }
    assert decisions == {
        "L1/invalid": "blocked_on_correctness",
        "L1/runtime": "blocked_on_runtime",
    }


def test_partial_failure_classifies_reference_oom_blocker(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "oom")
    _write_partial_timing(
        timing_root,
        "oom",
        trace_status_counts={"INVALID_REFERENCE": 1},
        stdout_records=[
            {
                "workload": {"uuid": "w", "axes": {"N": 8192}},
                "evaluation": {
                    "status": "INVALID_REFERENCE",
                    "log": (
                        "Reference run() failed: HIP out of memory. "
                        "Tried to allocate 5.00 GiB."
                    ),
                },
            }
        ],
    )

    report = partial.build_partial_failure_classification(
        _coverage(dataset_root, timing_root)
    )

    target = report["targets"][0]
    assert target["blocker_class"] == "reference_oom_blocked"
    assert target["closure_decision"] == "blocked_on_reference_oom"
    assert target["failure_details"] == [
        {
            "workload_index": 0,
            "workload_uuid": "w",
            "axes": {"N": 8192},
            "status": "INVALID_REFERENCE",
            "phase": "reference",
            "oom_detected": True,
            "timeout_detected": False,
            "log_head": (
                "Reference run() failed: HIP out of memory. Tried to allocate 5.00 GiB."
            ),
        }
    ]
    assert report["blocker_class_counts"] == {"reference_oom_blocked": 1}


def test_partial_failure_uses_source_workloads_for_sharded_aggregate(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "sharded")
    path = timing_root / "L1" / "sharded.timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": 1.0,
                    "parsed_rows": [{"is_kernel_activity": True}],
                    "source_workloads": [
                        {"status": "partial_profiler_backed"},
                        {"status": "profiler_blocked"},
                    ],
                },
                "replacement_metadata": {
                    "profiled_workload_count": 0,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"RUNTIME_ERROR": 1},
                    "full_workload_coverage": False,
                    "failure_reason": "manifest incomplete",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = partial.build_partial_failure_classification(
        _coverage(dataset_root, timing_root)
    )

    assert report["partial_target_count"] == 1
    target = report["targets"][0]
    assert target["attempted_workload_count"] == 2
    assert target["failure_mode"] == "mixed_correctness_runtime"
    assert target["closure_decision"] == "blocked_on_mixed_failures"
    assert target["trace_status_counts"] == {
        "PROFILER_BLOCKED": 1,
        "RUNTIME_ERROR": 1,
    }


def test_partial_failure_reads_sharded_slice_oom_logs(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "sharded_oom")
    slice_path = (
        tmp_path
        / "slices"
        / "workload-0000"
        / "timing"
        / "L1"
        / "sharded_oom.timing.json"
    )
    slice_path.parent.mkdir(parents=True)
    slice_path.write_text(
        json.dumps(
            {
                "stdout": json.dumps(
                    {
                        "workload": {"uuid": "w0", "axes": {"N": 1}},
                        "evaluation": {
                            "status": "RUNTIME_ERROR",
                            "log": "gen_inputs failed: HIP out of memory.",
                        },
                    }
                )
                + "\n"
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path = timing_root / "L1" / "sharded_oom.timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": 1.0,
                    "parsed_rows": [{"is_kernel_activity": True}],
                    "source_workloads": [
                        {
                            "replacement_path": slice_path.as_posix(),
                            "status": "partial_profiler_backed",
                            "workload_index": 0,
                            "workload_uuid": "w0",
                        },
                        {
                            "failure_reason": "rocprofv3 command failed",
                            "status": "profiler_blocked",
                            "workload_index": 1,
                            "workload_uuid": "w1",
                        },
                    ],
                },
                "replacement_metadata": {
                    "profiled_workload_count": 0,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"RUNTIME_ERROR": 1},
                    "full_workload_coverage": False,
                    "failure_reason": "manifest incomplete",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = partial.build_partial_failure_classification(
        _coverage(dataset_root, timing_root)
    )

    target = report["targets"][0]
    assert target["blocker_class"] == "memory_oom_with_profiler_gap"
    assert target["closure_decision"] == "blocked_on_reference_oom"
    assert target["failure_details"][0]["phase"] == "gen_inputs"
    assert target["failure_details"][0]["oom_detected"] is True
    assert target["failure_details"][0]["timeout_detected"] is False
    assert target["failure_details"][1]["status"] == "PROFILER_BLOCKED"


def test_partial_failure_distinguishes_user_solution_oom_and_timeout(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "user_oom")
    _write_problem(dataset_root, "timeout")
    _write_partial_timing(
        timing_root,
        "user_oom",
        trace_status_counts={"RUNTIME_ERROR": 1},
        stdout_records=[
            {
                "evaluation": {
                    "status": "RUNTIME_ERROR",
                    "log": "User function failed: HIP out of memory.",
                },
            }
        ],
    )
    _write_partial_timing(
        timing_root,
        "timeout",
        trace_status_counts={"TIMEOUT": 1},
        stdout_records=[
            {
                "evaluation": {
                    "status": "TIMEOUT",
                    "log": "subprocess.TimeoutExpired: command timed out",
                },
            }
        ],
    )

    report = partial.build_partial_failure_classification(
        _coverage(dataset_root, timing_root)
    )

    targets = {target["problem_id"]: target for target in report["targets"]}
    assert targets["L1/user_oom"]["blocker_class"] == "user_solution_oom"
    assert targets["L1/user_oom"]["closure_decision"] == "blocked_on_reference_oom"
    assert targets["L1/timeout"]["failure_mode"] == "timeout_only"
    assert targets["L1/timeout"]["blocker_class"] == "timeout"
    assert targets["L1/timeout"]["closure_decision"] == "blocked_on_timeout"


def test_partial_failure_classifies_profiler_closure_stderr_oom(tmp_path):
    payload = {
        "stderr": (
            'File "correctness.py", line 150, in compute_error_stats\n'
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 4.00 GiB."
        )
    }

    details = partial.failure_trace_details(payload)
    blocker = partial.blocker_class(
        trace_counts={"PROFILER_BLOCKED": 1},
        failure_details=details,
    )

    assert details == [
        {
            "workload_index": 0,
            "workload_uuid": None,
            "axes": None,
            "status": "PROFILER_BLOCKED",
            "phase": "correctness",
            "oom_detected": True,
            "timeout_detected": False,
            "log_head": (
                "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 4.00 GiB."
            ),
        }
    ]
    assert blocker == "profiler_closure_oom_blocked"
    assert (
        partial.closure_decision(
            "profiler_evidence_gap",
            blocker_class=blocker,
        )
        == "blocked_on_reference_oom"
    )


def test_partial_failure_markdown_escapes_status_counts():
    report = {
        "partial_target_count": 1,
        "closure_decision_counts": {"blocked_on_runtime": 1},
        "failure_mode_counts": {"runtime_error_only": 1},
        "targets": [
            {
                "problem_id": "L1/example",
                "failure_mode": "runtime_error_only",
                "blocker_class": "reference_oom_blocked",
                "closure_decision": "blocked_on_runtime",
                "trace_status_counts": {"RUNTIME_ERROR": 1},
            }
        ],
        "blocker_class_counts": {"reference_oom_blocked": 1},
        "claim_boundary": "Classification only.",
    }

    markdown = partial.render_partial_failure_classification_markdown(report)

    assert "Classified targets: `1`" in markdown
    assert "reference_oom_blocked" in markdown
    assert "blocked_on_runtime" in markdown


def test_write_decision_lists_removes_stale_lists(tmp_path):
    stale = tmp_path / "complete_missing_workload_slices.txt"
    stale.write_text("L1/stale\n", encoding="utf-8")
    report = {
        "targets": [
            {
                "problem_id": "L1/runtime",
                "closure_decision": "blocked_on_reference_oom",
            }
        ]
    }

    partial._write_decision_lists(tmp_path, report)

    assert not stale.exists()
    assert (tmp_path / "blocked_on_reference_oom.txt").read_text(
        encoding="utf-8"
    ) == "L1/runtime\n"
