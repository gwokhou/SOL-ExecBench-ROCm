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
) -> dict:
    payload = {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
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
) -> None:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(_definition(name=name, custom_entrypoint=custom_entrypoint)) + "\n",
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
    assert "| ready_missing_profiler_timing | 1 |" in markdown
    assert "Full profiler-backed timing coverage: `false`" in markdown
