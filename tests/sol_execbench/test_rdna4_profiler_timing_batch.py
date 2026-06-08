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


def test_batch_does_not_write_replacement_on_profiler_failure(tmp_path):
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
    assert not (replacement / "L1" / "one.timing.json").exists()
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["selected_targets"] == 1
    assert summary["succeeded"] == 0
    assert summary["failed"] == 1
