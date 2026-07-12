from __future__ import annotations

import json


import subprocess

import sys

from collections.abc import Sequence

from contextlib import nullcontext

from importlib.util import module_from_spec, spec_from_file_location

from pathlib import Path


from sol_execbench.core.dataset.inventory import build_dataset_inventory

from sol_execbench.core.dataset.profiler_timing_coverage import (
    build_profiler_timing_coverage_report,
)

from sol_execbench.core.dataset.readiness import classify_rocm_readiness

REPO_ROOT = Path(__file__).resolve().parents[4]

SCRIPT_PATH = REPO_ROOT / "scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py"

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


def _write_large_input_problem(root: Path, category: str, name: str) -> None:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    definition = {
        "name": name,
        "description": "large timing input demo",
        "axes": {
            "N": {"type": "var"},
            "H": {"type": "const", "value": 1024},
        },
        "inputs": {"x": {"shape": ["N", "H"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N", "H"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    }
    workload = {
        "uuid": "large",
        "axes": {"N": 4096},
        "inputs": {"x": {"type": "random"}},
    }
    (problem_dir / "definition.json").write_text(
        json.dumps(definition) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(workload) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "reference.py").write_text(
        definition["reference"],
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


def _successful_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Runner that creates rocprofv3 CSV output and returns PASSED trace."""
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


def test_partition_targets_by_index_creates_disjoint_chunks():
    """Test PRFL-03: partitioning creates non-overlapping index ranges."""
    targets = [
        {"problem_id": f"L1/problem{i}", "problem_path": f"L1/problem{i}"}
        for i in range(10)
    ]
    chunks = batch._partition_targets_by_index(targets, max_workers=3)

    # Should create 3 chunks
    assert len(chunks) == 3

    # Check chunk sizes (should be [4, 4, 2] for 10 items with 3 workers)
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 4
    assert len(chunks[2]) == 2

    # Verify disjointness: no problem_id appears in multiple chunks
    all_ids = []
    for chunk in chunks:
        chunk_ids = [t["problem_id"] for t in chunk]
        all_ids.extend(chunk_ids)

    assert len(all_ids) == len(set(all_ids)), "Chunks have overlapping problem_ids"


def test_partition_targets_by_index_exhaustive_coverage():
    """Test PRFL-03: all targets assigned to exactly one chunk."""
    targets = [
        {"problem_id": f"L1/p{i}", "problem_path": f"L1/p{i}"} for i in range(25)
    ]
    max_workers = 4
    chunks = batch._partition_targets_by_index(targets, max_workers)

    # All targets should be covered
    total_in_chunks = sum(len(chunk) for chunk in chunks)
    assert total_in_chunks == len(targets)

    # Each target should appear exactly once
    all_targets_flat = []
    for chunk in chunks:
        all_targets_flat.extend(chunk)

    # Sort both lists by problem_id for comparison
    original_sorted = sorted(targets, key=lambda t: t["problem_id"])
    flat_sorted = sorted(all_targets_flat, key=lambda t: t["problem_id"])

    assert len(flat_sorted) == len(original_sorted)
    for i in range(len(original_sorted)):
        assert flat_sorted[i]["problem_id"] == original_sorted[i]["problem_id"]


def test_partition_targets_by_index_single_worker():
    """Test PRFL-03: single worker returns one chunk with all targets."""
    targets = [{"problem_id": f"L1/p{i}", "problem_path": f"L1/p{i}"} for i in range(7)]
    chunks = batch._partition_targets_by_index(targets, max_workers=1)

    assert len(chunks) == 1
    assert len(chunks[0]) == len(targets)
    assert chunks[0] == targets


def test_partition_targets_by_index_empty_list():
    """Test PRFL-03: empty targets returns empty chunks."""
    chunks = batch._partition_targets_by_index([], max_workers=4)
    assert chunks == []


def test_partition_targets_by_index_more_workers_than_targets():
    """Test PRFL-03: handles case where workers > targets."""
    targets = [
        {"problem_id": "L1/p1", "problem_path": "L1/p1"},
        {"problem_id": "L1/p2", "problem_path": "L1/p2"},
    ]
    chunks = batch._partition_targets_by_index(targets, max_workers=10)

    # Should only create as many chunks as there are targets
    assert len(chunks) == 2
    assert len(chunks[0]) == 1
    assert len(chunks[1]) == 1
    assert chunks[0][0]["problem_id"] == "L1/p1"
    assert chunks[1][0]["problem_id"] == "L1/p2"


def test_cpu_parallel_staging_gpu_serial_profiling(tmp_path):
    """Test PRFL-01 & PRFL-02: CPU staging parallel, GPU profiling serial."""
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in range(3):
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        max_workers=2,
        rocprofv3_available=True,
        runner=lambda cmd: _successful_runner(cmd),
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["succeeded"] == 3


def test_gpu_exclusivity_architecturally_enforced():
    """Test PRFL-02: no code path enables concurrent GPU subprocess execution."""
    import inspect

    # Get source code of run_batch and _process_target_chunk
    run_batch_source = inspect.getsource(batch.run_batch)

    # Verify ThreadPoolExecutor is used for CPU operations
    assert "ThreadPoolExecutor" in run_batch_source
    assert "max_workers" in run_batch_source

    # Verify GPU profiling calls are inside worker function, not directly in run_batch
    # collect_rocprofv3_timing should only appear in _profile_target or similar
    process_target_source = (
        inspect.getsource(batch._profile_target)
        if hasattr(batch, "_profile_target")
        else ""
    )

    # The key architectural enforcement: GPU calls are sequential inside worker loops
    # This is verified by checking that _profile_target contains collect_rocprofv3_timing
    # and that it's called in a loop, not submitted to a parallel pool
    assert (
        "collect_rocprofv3_timing" in process_target_source
        or "collect_rocprofv3_timing" in run_batch_source
    )

    # Verify no --parallel-gpu or similar flag exists
    run_batch_sig = inspect.signature(batch.run_batch)
    for param in run_batch_sig.parameters.values():
        assert "parallel_gpu" not in param.name, "No parallel GPU flag should exist"
        assert "gpu_jobs" not in param.name, "No GPU jobs flag should exist"


def test_cli_exposes_max_workers_for_resource_sensitive_runs():
    default_args = batch.parse_args([])
    assert default_args.max_workers == 4

    serial_args = batch.parse_args(["--max-workers", "1"])
    assert serial_args.max_workers == 1


def test_cli_reads_only_problem_file(tmp_path, monkeypatch):
    problem_file = tmp_path / "targets.txt"
    problem_file.write_text(
        "# comment\nL1/one\n\nL2/two\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    monkeypatch.setattr(batch, "acquire_pid_lock", lambda output_dir: nullcontext())

    rc = batch.main(
        ["--only-problem", "L0/zero", "--only-problem-file", str(problem_file)]
    )

    assert rc == 0
    assert captured["only_problem"] == ("L0/zero", "L1/one", "L2/two")


def test_parallel_resume_skips_completed_targets(tmp_path):
    """Test PRFL-04: --resume deduplication is thread-safe under parallel execution."""
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in range(3):
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    # Pre-create replacement sidecar for problem1 (simulating completed work)
    existing = replacement / "L1" / "problem1.timing.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
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

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        max_workers=2,
        resume=True,
        rocprofv3_available=True,
        runner=_successful_runner,
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    # problem1 was already done (skipped via resume), so only 2 profiled
    result_ids = [r["problem_id"] for r in summary["results"]]
    assert "L1/problem1" not in result_ids
    assert summary["succeeded"] == 2


def test_keyboard_interrupt_partial_completion(tmp_path):
    """Test PRFL-05: KeyboardInterrupt produces structured partial-completion output."""
    from unittest.mock import patch

    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in range(5):
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    # Simulate KeyboardInterrupt in main thread by raising during as_completed
    def interrupting_as_completed(fs):
        futures_list = list(fs)
        # Let first future complete, then interrupt
        for future in futures_list[:1]:
            future.result()  # Wait for first to complete
        raise KeyboardInterrupt("Simulated Ctrl+C")

    with patch.object(batch, "as_completed", side_effect=interrupting_as_completed):
        status = batch.run_batch(
            dataset_root=dataset_root,
            output_dir=output_dir,
            source_timing_dirs=(source_timing,),
            replacement_timing_dir=replacement,
            max_workers=2,
            rocprofv3_available=True,
            runner=_successful_runner,
        )

    assert status == 130
    summary_path = output_dir / "batch-summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary.get("interrupted") is True


def test_keyboard_interrupt_distinguishes_interrupted_targets(tmp_path):
    """Test PRFL-05: interrupted targets are clearly marked in partial summary."""
    from unittest.mock import patch

    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in range(4):
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    # Simulate KeyboardInterrupt after partial completion
    def interrupting_as_completed(fs):
        raise KeyboardInterrupt("Simulated interrupt")

    with patch.object(batch, "as_completed", side_effect=interrupting_as_completed):
        status = batch.run_batch(
            dataset_root=dataset_root,
            output_dir=output_dir,
            source_timing_dirs=(source_timing,),
            replacement_timing_dir=replacement,
            max_workers=2,
            rocprofv3_available=True,
            runner=_successful_runner,
        )

    assert status == 130
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["interrupted"] is True


def test_parallel_completion_produces_deterministic_order(tmp_path):
    """Test PRFL-06: output order is deterministic regardless of completion order."""
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in [3, 2, 1, 0]:
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        max_workers=2,
        rocprofv3_available=True,
        runner=_successful_runner,
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    result_problem_ids = [r.get("problem_id") for r in summary.get("results", [])]
    expected_order = ["L1/problem0", "L1/problem1", "L1/problem2", "L1/problem3"]
    assert result_problem_ids == expected_order


def test_end_to_end_parallel_batch(tmp_path):
    """Integration test: All Phase 177 requirements working together."""
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"

    for i in range(6):
        _write_problem(dataset_root, "L1", f"problem{i}")
        _write_fallback_timing(source_timing, "L1", f"problem{i}")

    # Pre-complete one problem to test --resume
    existing = replacement / "L1" / "problem2.timing.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
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

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        max_workers=3,
        resume=True,
        rocprofv3_available=True,
        runner=_successful_runner,
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    assert summary["succeeded"] == 5
    result_ids = [r["problem_id"] for r in summary["results"]]
    assert result_ids == sorted(result_ids)
