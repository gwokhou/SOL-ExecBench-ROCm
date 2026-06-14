from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
    render_profiler_timing_coverage_markdown,
)


def _definition(
    *,
    name: str,
    custom_entrypoint: str | None = None,
    dtype: str = "float32",
) -> dict:
    payload = {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": dtype}},
        "outputs": {"out": {"shape": ["N"], "dtype": dtype}},
        "reference": "def run(x):\n    return x\n",
    }
    if custom_entrypoint:
        payload["custom_inputs_entrypoint"] = custom_entrypoint
    return payload


def _workload(kind: str = "random") -> dict:
    input_spec = {"type": kind}
    if kind == "safetensors":
        input_spec.update({"path": "missing.safetensors", "tensor_key": "x"})
    return {
        "uuid": f"{kind}-w",
        "axes": {"N": 4},
        "inputs": {"x": input_spec},
    }


def _write_problem(
    root: Path,
    category: str,
    name: str,
    *,
    workload_kind: str = "random",
    custom_entrypoint: str | None = None,
    dtype: str = "float32",
) -> None:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(
            _definition(
                name=name,
                custom_entrypoint=custom_entrypoint,
                dtype=dtype,
            )
        )
        + "\n",
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


def test_profiler_timing_coverage_tracks_problem_denominator(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "profiler")
    _write_problem(dataset_root, "L1", "fallback")
    _write_problem(dataset_root, "L1", "missing")
    _write_problem(
        dataset_root,
        "L1",
        "blocked",
        workload_kind="safetensors",
    )
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "profiler.timing.json",
        {
            "profiler_collected": True,
            "csv_path": "profiler_kernel_trace.csv",
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
            },
        },
    )
    _write_timing(
        timing_dir / "L1" / "fallback.timing.json",
        {
            "profiler_collected": False,
            "selection": {
                "reason": "selected policy backend is pytorch_profiler",
                "policy": {"backend": "device_events"},
            },
            "evidence": None,
        },
    )

    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        expected_problem_denominator=4,
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.problem_denominator == 4
    assert report.totals.profiler_backed_problems == 1
    assert report.totals.fallback_timing_problems == 1
    assert report.totals.ready_missing_profiler_timing_problems == 1
    assert report.totals.hardware_evidence_deferred_problems == 0
    assert report.totals.readiness_blocked_problems == 1
    assert report.totals.profiler_backed_coverage_pct == 25.0
    assert report.claim_boundary.problem_denominator_accounted is True
    assert report.claim_boundary.full_profiler_backed_timing_coverage is False
    statuses = {problem.problem_id: problem.status for problem in report.problems}
    assert statuses == {
        "L1/blocked": "readiness_blocked",
        "L1/fallback": "timing_fallback",
        "L1/missing": "ready_missing_profiler_timing",
        "L1/profiler": "profiler_backed",
    }
    profiler = next(
        problem for problem in report.problems if problem.problem_id == "L1/profiler"
    )
    assert profiler.evidence is not None
    assert profiler.evidence.backend == "rocprofv3"
    assert profiler.evidence.kernel_duration_ms == 1.25
    blocked = next(
        problem for problem in report.problems if problem.problem_id == "L1/blocked"
    )
    assert blocked.readiness_reason_codes == ["safetensors_asset_missing"]
    assert blocked.readiness_blocker_types == ["missing_blob"]


def test_profiler_timing_coverage_separates_hardware_evidence_deferred(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "Quant", "fp8_missing", dtype="float8_e4m3fn")
    inventory = build_dataset_inventory(dataset_root, categories=("Quant",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        expected_problem_denominator=1,
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.hardware_evidence_deferred_problems == 1
    assert report.totals.readiness_blocked_problems == 0
    assert report.problems[0].status == "hardware_evidence_deferred"
    assert report.problems[0].readiness_status == "needs_hardware_evidence"


def test_profiler_timing_coverage_requires_expected_denominator(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "only")
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        expected_problem_denominator=235,
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.problem_denominator == 1
    assert report.claim_boundary.problem_denominator_accounted is False
    assert report.claim_boundary.full_profiler_backed_timing_coverage is False


def test_profiler_timing_coverage_classifies_partial_replacement_sidecar(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "partial")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "partial.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "profiled_workload_count": 1,
                "expected_workload_count": 2,
                "trace_status_counts": {"INVALID_REFERENCE": 1, "PASSED": 1},
                "full_workload_coverage": False,
                "failure_reason": "replacement did not produce PASSED traces",
            },
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.profiler_backed_problems == 0
    assert report.totals.partial_profiler_backed_problems == 1
    assert report.totals.fallback_timing_problems == 0
    assert report.problems[0].status == "partial_profiler_backed"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.full_workload_coverage is False
    assert report.problems[0].evidence.trace_status_counts == {
        "INVALID_REFERENCE": 1,
        "PASSED": 1,
    }


def test_profiler_timing_coverage_classifies_reference_oom_sidecar(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "reference-oom")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "reference-oom.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "profiled_workload_count": 1,
                "expected_workload_count": 2,
                "trace_status_counts": {"INVALID_REFERENCE": 1, "PASSED": 1},
                "full_workload_coverage": False,
                "failure_reason": "replacement did not produce PASSED traces",
            },
            "stdout": json.dumps(
                {
                    "evaluation": {
                        "status": "INVALID_REFERENCE",
                        "log": "Reference run() failed: HIP out of memory.",
                    }
                }
            )
            + "\n",
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.partial_profiler_backed_problems == 0
    assert report.totals.reference_oom_blocked_problems == 1
    assert report.problems[0].status == "reference_oom_blocked"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.blocker_class == "reference_oom_blocked"


def test_reference_oom_is_accounted_but_not_complete_profiler_coverage(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "reference-oom")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "reference-oom.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "profiled_workload_count": 1,
                "expected_workload_count": 2,
                "trace_status_counts": {"INVALID_REFERENCE": 1, "PASSED": 1},
                "full_workload_coverage": False,
            },
            "stdout": json.dumps(
                {
                    "evaluation": {
                        "status": "INVALID_REFERENCE",
                        "log": "Reference run() failed: HIP out of memory.",
                    }
                }
            )
            + "\n",
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        expected_problem_denominator=1,
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.claim_boundary.problem_denominator_accounted is True
    assert report.claim_boundary.full_profiler_backed_timing_coverage is False
    assert report.totals.problem_denominator == 1
    assert report.totals.reference_oom_blocked_problems == 1
    assert report.totals.profiler_backed_problems == 0


def test_profiler_timing_coverage_classifies_sharded_reference_oom(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "sharded-oom")
    timing_dir = tmp_path / "timing"
    slice_path = tmp_path / "slice" / "timing" / "L1" / "sharded-oom.timing.json"
    _write_timing(
        slice_path,
        {
            "stdout": json.dumps(
                {
                    "evaluation": {
                        "status": "RUNTIME_ERROR",
                        "log": "gen_inputs failed: HIP out of memory.",
                    }
                }
            )
            + "\n",
        },
    )
    _write_timing(
        timing_dir / "L1" / "sharded-oom.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
                "parsed_rows": [{"is_kernel_activity": True}],
                "source_workloads": [
                    {
                        "replacement_path": slice_path.as_posix(),
                        "status": "partial_profiler_backed",
                    },
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
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.reference_oom_blocked_problems == 1
    assert report.problems[0].status == "reference_oom_blocked"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.blocker_class == "memory_oom_with_profiler_gap"
    assert report.blocker_class_counts == {"memory_oom_with_profiler_gap": 1}


def test_profiler_timing_coverage_distinguishes_gen_inputs_oom(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "gen-inputs-oom")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "gen-inputs-oom.timing.json",
        {
            "profiler_collected": True,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 1.25,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "trace_status_counts": {"RUNTIME_ERROR": 1},
                "full_workload_coverage": False,
            },
            "stdout": json.dumps(
                {
                    "evaluation": {
                        "status": "RUNTIME_ERROR",
                        "log": "gen_inputs failed: HIP out of memory.",
                    }
                }
            )
            + "\n",
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.problems[0].status == "reference_oom_blocked"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.blocker_class == "gen_inputs_oom_blocked"
    assert report.blocker_class_counts == {"gen_inputs_oom_blocked": 1}


def test_profiler_timing_coverage_classifies_correctness_stderr_oom(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "correctness-oom")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "correctness-oom.timing.json",
        {
            "profiler_collected": False,
            "selection": {"policy": {"backend": "rocprofv3"}},
            "evidence": {"backend": "rocprofv3", "parsed_rows": []},
            "replacement_metadata": {
                "trace_status_counts": {},
                "full_workload_coverage": False,
                "failure_reason": "rocprofv3 command failed with exit code 1",
            },
            "stderr": (
                'File "correctness.py", line 150, in compute_error_stats\n'
                "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 4.00 GiB."
            ),
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-09T00:00:00Z",
    )

    assert report.problems[0].status == "reference_oom_blocked"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.blocker_class == "profiler_closure_oom_blocked"
    assert report.blocker_class_counts == {"profiler_closure_oom_blocked": 1}


def test_profiler_timing_coverage_accepts_complete_workload_aggregate(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "aggregate")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "aggregate.timing.json",
        {
            "profiler_collected": True,
            "selection": {
                "reason": "complete workload-sharded rocprofv3 aggregation",
                "policy": {"backend": "rocprofv3"},
            },
            "evidence": {
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "kernel_duration_ms": 2.5,
                "parsed_rows": [{"is_kernel_activity": True}],
            },
            "replacement_metadata": {
                "replacement_status": "profiler_backed",
                "profiled_workload_count": 2,
                "expected_workload_count": 2,
                "trace_status_counts": {"PASSED": 2},
                "workload_sharded_aggregation": True,
                "full_workload_coverage": True,
                "failure_reason": None,
            },
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.profiler_backed_problems == 1
    assert report.totals.partial_profiler_backed_problems == 0
    assert report.problems[0].status == "profiler_backed"
    assert report.problems[0].evidence is not None
    assert report.problems[0].evidence.full_workload_coverage is True
    assert report.problems[0].evidence.profiled_workload_count == 2
    assert report.problems[0].evidence.expected_workload_count == 2


def test_profiler_timing_coverage_classifies_profiler_blocked_sidecar(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "blocked-profiler")
    timing_dir = tmp_path / "timing"
    _write_timing(
        timing_dir / "L1" / "blocked-profiler.timing.json",
        {
            "profiler_collected": False,
            "selection": {
                "reason": "rocprofv3 did not produce kernel activity rows",
                "policy": {"backend": "rocprofv3"},
            },
            "evidence": {"backend": "rocprofv3", "parsed_rows": []},
            "replacement_metadata": {
                "profiled_workload_count": 0,
                "expected_workload_count": 1,
                "full_workload_coverage": False,
                "failure_reason": "rocprofv3 did not produce kernel activity rows",
            },
        },
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)

    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_dir,),
        created_at="2026-06-08T00:00:00Z",
    )

    assert report.totals.profiler_blocked_problems == 1
    assert report.totals.fallback_timing_problems == 0
    assert report.problems[0].status == "profiler_blocked"


def test_profiler_timing_coverage_markdown_summarizes_claim_boundary(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "only")
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        created_at="2026-06-08T00:00:00Z",
    )

    markdown = render_profiler_timing_coverage_markdown(report)

    assert "Problem denominator: `1`" in markdown
    assert "Reference OOM-blocked problems: `0`" in markdown
    assert "| ready_missing_profiler_timing | 1 |" in markdown
    assert "Full profiler-backed timing coverage: `false`" in markdown
