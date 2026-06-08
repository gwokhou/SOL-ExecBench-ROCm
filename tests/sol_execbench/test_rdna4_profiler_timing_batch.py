from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_timing_batch.py"
SPEC = spec_from_file_location("run_rdna4_profiler_timing_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)

ROCPROFV3_CSV = """Domain,Name,Start_Timestamp,End_Timestamp,Duration(ns)
KERNEL_DISPATCH,aten_kernel,1000,6000,5000
"""
PASSED_TRACE = json.dumps({"evaluation": {"status": "PASSED"}}) + "\n"
INVALID_TRACE = json.dumps({"evaluation": {"status": "INVALID_REFERENCE"}}) + "\n"


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
        "uuid": "w0",
        "axes": {"N": 4},
        "inputs": {"x": {"type": "random"}},
    }


def _write_problem(root: Path, category: str, name: str) -> None:
    problem_dir = root / category / name
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


def _write_workloads(root: Path, category: str, name: str, count: int) -> None:
    problem_dir = root / category / name
    records = []
    for index in range(count):
        workload = _workload()
        workload["uuid"] = f"w{index}"
        workload["axes"] = {"N": index + 1}
        records.append(json.dumps(workload))
    (problem_dir / "workload.jsonl").write_text(
        "\n".join(records) + "\n",
        encoding="utf-8",
    )


def _write_fallback_timing(root: Path, category: str, name: str) -> None:
    path = root / category / f"{name}.timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profiler_collected": False,
                "selection": {
                    "reason": "selected policy backend is pytorch_profiler",
                    "policy": {"backend": "device_events"},
                },
                "evidence": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _coverage(dataset_root: Path, timing_root: Path, replacement_root: Path):
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    return build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(replacement_root, timing_root),
    )


def test_batch_selects_fallback_targets_and_honors_resume(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_problem(dataset_root, "L1", "two")
    _write_fallback_timing(source_timing, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "two")
    existing = replacement / "L1" / "one.timing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert [target.problem_id for target in targets] == ["L1/two"]


def test_batch_resume_skips_classified_partial_replacement(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    existing = replacement / "L1" / "one.timing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {
                    "replacement_status": "partial_profiler_backed",
                    "profiled_workload_count": 1,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"INVALID_REFERENCE": 1, "PASSED": 1},
                    "full_workload_coverage": False,
                    "workload_limit_applied": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert targets == []


def test_batch_resume_keeps_workload_limited_partial_replacement(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    existing = replacement / "L1" / "one.timing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {
                    "replacement_status": "partial_profiler_backed",
                    "profiled_workload_count": 1,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"PASSED": 1},
                    "full_workload_coverage": False,
                    "workload_limit_applied": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert [target.problem_id for target in targets] == ["L1/one"]


def test_batch_resume_keeps_workload_slice_partial_replacement(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    existing = replacement / "L1" / "one.timing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {
                    "replacement_status": "partial_profiler_backed",
                    "profiled_workload_count": 1,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"PASSED": 1},
                    "full_workload_coverage": False,
                    "workload_limit_applied": None,
                    "workload_offset": 1,
                    "workload_slice_applied": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert [target.problem_id for target in targets] == ["L1/one"]


def test_batch_select_targets_honors_skip_problem(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_problem(dataset_root, "L1", "two")
    _write_fallback_timing(source_timing, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "two")
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        skip_problem=("L1/one",),
        resume=True,
    )

    assert [target.problem_id for target in targets] == ["L1/two"]


def test_batch_resume_skips_profiler_blocked_replacement(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    existing = replacement / "L1" / "one.timing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps(
            {
                "profiler_collected": False,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {"backend": "rocprofv3", "parsed_rows": []},
                "replacement_metadata": {
                    "replacement_status": "profiler_blocked",
                    "profiled_workload_count": 0,
                    "expected_workload_count": 1,
                    "trace_status_counts": {},
                    "full_workload_coverage": False,
                    "workload_limit_applied": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert targets == []


def test_batch_writes_profiler_backed_replacement_sidecar(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=PASSED_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_limit=1,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    assert calls
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["profiler_collected"] is True
    assert sidecar["evidence"]["backend"] == "rocprofv3"
    assert sidecar["evidence"]["kernel_duration_ms"] == 0.005
    assert sidecar["replacement_metadata"]["profiled_workload_count"] == 1
    assert sidecar["replacement_metadata"]["expected_workload_count"] == 1
    assert sidecar["replacement_metadata"]["trace_status_counts"] == {"PASSED": 1}
    assert sidecar["replacement_metadata"]["all_workloads_passed"] is True
    assert sidecar["replacement_metadata"]["full_workload_coverage"] is True
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["selected_targets"] == 1
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0


def test_batch_uses_configured_temp_dir_for_staging(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    temp_dir = tmp_path / "staging-root"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=PASSED_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        rocprofv3_available=True,
        runner=runner,
        temp_dir=temp_dir,
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    staging_dir = Path(summary["results"][0]["staging_dir"])
    assert staging_dir.parent == temp_dir


def test_batch_records_workload_offset_slice_metadata(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 3)
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=PASSED_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_limit=1,
        workload_offset=1,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    metadata = sidecar["replacement_metadata"]
    assert metadata["replacement_status"] == "partial_profiler_backed"
    assert metadata["profiled_workload_count"] == 1
    assert metadata["expected_workload_count"] == 3
    assert metadata["workload_limit_applied"] == 1
    assert metadata["workload_offset"] == 1
    assert metadata["workload_slice_applied"] is True
    assert metadata["workload_slice"] == {
        "limit": 1,
        "offset": 1,
        "selected_workload_count": 1,
    }
    assert metadata["all_workloads_passed"] is False
    assert metadata["full_workload_coverage"] is False


def test_workload_sharded_batch_aggregates_complete_manifest(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=PASSED_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_sharded=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    assert len(calls) == 2
    manifest = json.loads(
        (
            output_dir
            / "workload-manifests"
            / "L1"
            / "one.workload-profiler-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == batch.WORKLOAD_MANIFEST_SCHEMA_VERSION
    assert manifest["expected_workload_count"] == 2
    assert [entry["status"] for entry in manifest["workloads"]] == [
        "profiler_backed",
        "profiler_backed",
    ]
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    metadata = sidecar["replacement_metadata"]
    assert metadata["replacement_status"] == "profiler_backed"
    assert metadata["workload_sharded_aggregation"] is True
    assert metadata["profiled_workload_count"] == 2
    assert metadata["expected_workload_count"] == 2
    assert metadata["trace_status_counts"] == {"PASSED": 2}
    assert metadata["full_workload_coverage"] is True
    assert sidecar["profiler_collected"] is True
    assert sidecar["evidence"]["kernel_duration_ms"] == 0.01
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["succeeded"] == 1
    assert summary["results"][0]["workload_sharded"] is True


def test_workload_sharded_batch_resumes_completed_manifest_entries(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    slice_path = (
        output_dir
        / "workload-slices"
        / "workload-0000"
        / "timing"
        / "L1"
        / "one.timing.json"
    )
    slice_path.parent.mkdir(parents=True)
    slice_path.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "csv_path": "first.csv",
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": 0.005,
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {
                    "trace_status_counts": {"PASSED": 1},
                    "failure_reason": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = (
        output_dir / "workload-manifests" / "L1" / "one.workload-profiler-manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": batch.WORKLOAD_MANIFEST_SCHEMA_VERSION,
                "manifest_path": manifest_path.as_posix(),
                "problem_id": "L1/one",
                "category": "L1",
                "problem_path": "L1/one",
                "dataset_root": dataset_root.as_posix(),
                "expected_workload_count": 2,
                "expected_workloads": [
                    {"workload_index": 0, "row_index": 0, "workload_uuid": "w0"},
                    {"workload_index": 1, "row_index": 1, "workload_uuid": "w1"},
                ],
                "tool_version": "rocprofv3",
                "gpu_architecture": "gfx1200",
                "clock_locked": True,
                "workloads": [
                    {
                        "workload_index": 0,
                        "row_index": 0,
                        "workload_uuid": "w0",
                        "status": "profiler_backed",
                        "retryable": False,
                        "profiler_collected": True,
                        "backend": "rocprofv3",
                        "trace_status_counts": {"PASSED": 1},
                        "kernel_activity_rows": 1,
                        "kernel_duration_ms": 0.005,
                        "replacement_path": slice_path.as_posix(),
                        "csv_path": "first.csv",
                        "failure_reason": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=PASSED_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_sharded=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    assert len(calls) == 1
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["replacement_metadata"]["replacement_status"] == "profiler_backed"
    assert sidecar["replacement_metadata"]["profiled_workload_count"] == 2


def test_workload_sharded_batch_imports_existing_slice_sidecars(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    imported_dirs = [tmp_path / f"slice-{index}" / "timing" for index in range(2)]
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    for index, timing_dir in enumerate(imported_dirs):
        _write_timing = timing_dir / "L1" / "one.timing.json"
        _write_timing.parent.mkdir(parents=True, exist_ok=True)
        _write_timing.write_text(
            json.dumps(
                {
                    "profiler_collected": True,
                    "csv_path": f"slice-{index}.csv",
                    "selection": {"policy": {"backend": "rocprofv3"}},
                    "evidence": {
                        "backend": "rocprofv3",
                        "kernel_duration_ms": 0.005,
                        "parsed_rows": [{"is_kernel_activity": True}],
                    },
                    "replacement_metadata": {
                        "replacement_status": "partial_profiler_backed",
                        "profiled_workload_count": 1,
                        "expected_workload_count": 2,
                        "trace_status_counts": {"PASSED": 1},
                        "workload_offset": index,
                        "workload_slice_applied": True,
                        "full_workload_coverage": False,
                        "failure_reason": None,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_sharded=True,
        workload_slice_timing_dirs=tuple(imported_dirs),
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["replacement_metadata"]["replacement_status"] == "profiler_backed"
    assert sidecar["replacement_metadata"]["profiled_workload_count"] == 2
    assert sidecar["replacement_metadata"]["trace_status_counts"] == {"PASSED": 2}


def test_workload_sharded_aggregation_keeps_failed_workload_partial(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    calls = 0

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        status = PASSED_TRACE if calls == 1 else ""
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=status,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_sharded=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["replacement_metadata"]["replacement_status"] == (
        "partial_profiler_backed"
    )
    assert sidecar["replacement_metadata"]["profiled_workload_count"] == 1
    assert sidecar["replacement_metadata"]["expected_workload_count"] == 2
    assert sidecar["replacement_metadata"]["full_workload_coverage"] is False


def test_batch_rejects_replacement_when_trace_status_is_not_passed(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}_kernel_trace.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout=INVALID_TRACE,
            stderr="",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["profiler_collected"] is True
    assert sidecar["replacement_metadata"]["all_workloads_passed"] is False
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["succeeded"] == 0
    assert summary["partial_profiler_backed"] == 1
    assert summary["failed"] == 0
    assert (
        summary["results"][0]["fallback_reason"]
        == "replacement did not produce PASSED traces for every workload"
    )


def test_batch_writes_blocked_replacement_on_profiler_failure(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=1,
            stdout="",
            stderr="failed",
        )

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_limit=1,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 1
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["profiler_collected"] is False
    assert sidecar["replacement_metadata"]["replacement_status"] == "profiler_blocked"
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["selected_targets"] == 1
    assert summary["succeeded"] == 0
    assert summary["failed"] == 1


def test_batch_mark_blocked_problem_writes_classified_sidecar(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        mark_blocked_problem=("L1/one",),
        rocprofv3_available=True,
        runner=lambda command: subprocess.CompletedProcess(command, 0),
    )

    assert status == 1
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["profiler_collected"] is False
    assert sidecar["evidence"]["backend"] == "rocprofv3"
    assert sidecar["replacement_metadata"]["replacement_status"] == "profiler_blocked"
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["profiler_blocked"] == 1


def test_batch_mark_blocked_only_does_not_profile_other_targets(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_problem(dataset_root, "L1", "two")
    _write_fallback_timing(source_timing, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "two")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        mark_blocked_problem=("L1/one",),
        mark_blocked_only=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 1
    assert (replacement / "L1" / "one.timing.json").exists()
    assert not (replacement / "L1" / "two.timing.json").exists()
