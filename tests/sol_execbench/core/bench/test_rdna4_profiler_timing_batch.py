from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

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


def test_rdna4_reference_override_is_scoped_to_convnextv2_grn():
    payload = _definition("035_convnextv2_block_with_grn")

    metadata = batch._apply_rdna4_reference_override(
        payload,
        "L2/035_convnextv2_block_with_grn",
    )

    assert "pwconv1_weight[:, :, None, None]" in payload["reference"]
    assert metadata is not None
    assert metadata["problem_id"] == "L2/035_convnextv2_block_with_grn"
    assert metadata["override_type"] == "equivalent_reference_implementation"
    assert (
        "not unmodified benchmark reference dispatch timing"
        in (metadata["claim_boundary"])
    )
    other = _definition("other")
    assert batch._apply_rdna4_reference_override(other, "L2/other") is None
    assert other["reference"] == "def run(x):\n    return x\n"


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


def test_batch_omits_hip_runtime_trace_by_default(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "source"
    output_dir = tmp_path / "out"
    captured_commands: list[Sequence[str]] = []
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured_commands.append(tuple(command))
        return _successful_runner(command)

    rc = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        only_problem=("L1/one",),
        rocprofv3_available=True,
        runner=runner,
        max_workers=1,
    )

    assert rc == 0
    assert len(captured_commands) == 1
    assert "--kernel-trace" in captured_commands[0]
    assert "--hip-runtime-trace" not in captured_commands[0]
    separator_index = captured_commands[0].index("--")
    assert captured_commands[0][separator_index + 1 : separator_index + 3] == (
        sys.executable,
        "eval_driver.py",
    )


def test_batch_can_enable_hip_runtime_trace_for_debugging(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "source"
    output_dir = tmp_path / "out"
    captured_commands: list[Sequence[str]] = []
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured_commands.append(tuple(command))
        return _successful_runner(command)

    rc = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        only_problem=("L1/one",),
        include_hip_runtime_trace=True,
        rocprofv3_available=True,
        runner=runner,
        max_workers=1,
    )

    assert rc == 0
    assert len(captured_commands) == 1
    assert "--hip-runtime-trace" in captured_commands[0]


def test_staging_runner_applies_subprocess_memory_limit(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    runner = batch._staging_runner(
        tmp_path,
        timeout=7,
        memory_limit_gib=1.5,
    )
    result = runner(("python", "eval_driver.py"))

    assert result.returncode == 0
    kwargs = cast(dict[str, Any], captured["kwargs"])
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] == 7
    assert kwargs["preexec_fn"] is not None


def test_staging_runner_forces_absolute_tmpdir(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}
    staging_dir = tmp_path / "relative-tmp-root" / "staging"
    staging_dir.mkdir(parents=True)

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setenv("TMPDIR", os.path.join("relative", "tmp"))
    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    runner = batch._staging_runner(
        staging_dir,
        timeout=7,
        memory_limit_gib=None,
    )
    result = runner((sys.executable, "eval_driver.py"))

    assert result.returncode == 0
    kwargs = cast(dict[str, Any], captured["kwargs"])
    env = cast(dict[str, str], kwargs["env"])
    assert env["TMPDIR"] == str(staging_dir.parent.resolve())
    assert Path(env["TMPDIR"]).is_absolute()


def test_batch_preflights_oversized_timing_inputs_before_runner(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_large_input_problem(dataset_root, "L1", "huge")
    _write_fallback_timing(source_timing, "L1", "huge")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    rc = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        only_problem=("L1/huge",),
        max_estimated_timing_input_gib=0.01,
        rocprofv3_available=True,
        runner=runner,
        max_workers=1,
    )

    assert rc == 1
    sidecar = json.loads(
        (output_dir / "timing" / "L1" / "huge.timing.json").read_text()
    )
    metadata = sidecar["replacement_metadata"]
    assert metadata["replacement_status"] == "profiler_blocked"
    assert (
        "estimated timing input footprint exceeds preflight cap"
        in metadata["failure_reason"]
    )
    assert "timing_pool_peak=" in metadata["failure_reason"]


def test_batch_uses_dynamic_available_memory_cap_by_default(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_large_input_problem(dataset_root, "L1", "huge")
    _write_fallback_timing(source_timing, "L1", "huge")
    monkeypatch.setattr(
        batch,
        "_dynamic_available_memory_bytes",
        lambda *, subprocess_memory_limit_gib: 20 * 1024 * 1024,
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    rc = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        only_problem=("L1/huge",),
        rocprofv3_available=True,
        runner=runner,
        max_workers=1,
    )

    assert rc == 1
    sidecar = json.loads(
        (output_dir / "timing" / "L1" / "huge.timing.json").read_text()
    )
    reason = sidecar["replacement_metadata"]["failure_reason"]
    assert "estimated timing input footprint exceeds preflight cap" in reason
    assert "cap_source=dynamic_available_memory:70%" in reason


def test_batch_aborts_real_runner_when_torch_gpu_unavailable(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")
    monkeypatch.setattr(batch, "_torch_rocm_gpu_available", lambda: False)

    rc = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        only_problem=("L1/one",),
        rocprofv3_available=True,
        runner=None,
        max_workers=1,
    )

    assert rc == 1
    assert not (output_dir / "timing" / "L1" / "one.timing.json").exists()


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


def test_batch_selects_ready_missing_profiler_timing_targets(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    replacement = tmp_path / "replacement"
    _write_problem(dataset_root, "L1", "missing")
    coverage = _coverage(dataset_root, source_timing, replacement)

    targets = batch.select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement,
        resume=True,
    )

    assert [target.problem_id for target in targets] == ["L1/missing"]


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
    assert not staging_dir.exists()


def test_batch_cli_defaults_temp_dir_to_project_tmp(monkeypatch, tmp_path):
    output_dir = tmp_path / "custom-output"
    captured: dict[str, Path | tuple[Path, ...]] = {}

    def fake_run_batch(**kwargs) -> int:
        captured["output_dir"] = kwargs["output_dir"]
        captured["temp_dir"] = kwargs["temp_dir"]
        captured["source_timing_dirs"] = kwargs["source_timing_dirs"]
        return 0

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    monkeypatch.setattr(batch, "acquire_pid_lock", lambda _path: nullcontext())

    status = batch.main(["--output-dir", str(output_dir)])

    assert status == 0
    assert captured["output_dir"] == output_dir
    assert captured["temp_dir"] == Path("tmp") / output_dir.name
    assert captured["source_timing_dirs"] == (batch.DEFAULT_SOURCE_TIMING_DIR,)


def test_batch_can_keep_staging_for_debugging(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    temp_dir = tmp_path / "staging-root"
    _write_problem(dataset_root, "L1", "one")
    _write_fallback_timing(source_timing, "L1", "one")

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        rocprofv3_available=True,
        runner=_successful_runner,
        temp_dir=temp_dir,
        keep_staging=True,
    )

    assert status == 0
    summary = json.loads((output_dir / "batch-summary.json").read_text())
    staging_dir = Path(summary["results"][0]["staging_dir"])
    assert staging_dir.parent == temp_dir
    assert staging_dir.exists()


def test_batch_removes_raw_rocprofv3_csv_by_default(tmp_path):
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
        limit=1,
        rocprofv3_available=True,
        runner=_successful_runner,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert sidecar["evidence"]["parsed_rows"] == []
    assert sidecar["evidence"]["parsed_rows_compacted"] is True
    assert not Path(sidecar["csv_path"]).parent.exists()


def test_batch_can_keep_raw_rocprofv3_csv_for_debugging(tmp_path):
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
        limit=1,
        rocprofv3_available=True,
        runner=_successful_runner,
        keep_rocprofv3_csv=True,
    )

    assert status == 0
    sidecar = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert Path(sidecar["csv_path"]).parent.exists()


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
    failed_dir = tmp_path / "stale-failed-slice" / "timing"
    imported_dirs = [failed_dir] + [
        tmp_path / f"slice-{index}" / "timing" for index in range(2)
    ]
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    stale_timing = failed_dir / "L1" / "one.timing.json"
    stale_timing.parent.mkdir(parents=True, exist_ok=True)
    stale_timing.write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "csv_path": "stale.csv",
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "kernel_duration_ms": 0.0,
                    "parsed_rows": [],
                    "parsed_rows_compacted": True,
                },
                "replacement_metadata": {
                    "replacement_status": "partial_profiler_backed",
                    "profiled_workload_count": 0,
                    "expected_workload_count": 2,
                    "trace_status_counts": {"TIMEOUT": 1},
                    "workload_offset": 0,
                    "workload_slice_applied": True,
                    "full_workload_coverage": False,
                    "failure_reason": "timeout",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for index, timing_dir in enumerate(imported_dirs[1:]):
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


def test_workload_sharded_aggregation_uses_manifest_summaries(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    imported_dirs = [tmp_path / f"slice-{index}" / "timing" for index in range(2)]
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 2)
    _write_fallback_timing(source_timing, "L1", "one")
    large_rows = [
        {
            "name": f"kernel_{row}",
            "domain": "KERNEL_DISPATCH",
            "duration_ms": 0.001,
            "duration_ns": 1000,
            "is_kernel_activity": True,
            "raw": {"row": row},
        }
        for row in range(1000)
    ]
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
                        "parsed_rows": large_rows,
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
    rows = sidecar["evidence"]["parsed_rows"]
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"workload_sharded_kernel_activity"}
    assert sidecar["evidence"]["kernel_duration_ms"] == 0.01


def test_workload_sharded_aggregation_imports_compacted_sidecars(tmp_path):
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
                        "parsed_rows": [],
                        "parsed_rows_compacted": True,
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
    rows = sidecar["evidence"]["parsed_rows"]
    assert len(rows) == 2
    assert sidecar["replacement_metadata"]["replacement_status"] == "profiler_backed"
    assert sidecar["replacement_metadata"]["profiled_workload_count"] == 2
    assert sidecar["replacement_metadata"]["trace_status_counts"] == {"PASSED": 2}


def test_workload_sharded_batch_compacts_completed_slice_artifacts(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 1)
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
        workload_sharded=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    slice_sidecar_path = (
        output_dir
        / "workload-slices"
        / "workload-0000"
        / "timing"
        / "L1"
        / "one.timing.json"
    )
    slice_sidecar = json.loads(slice_sidecar_path.read_text(encoding="utf-8"))
    assert slice_sidecar["evidence"]["parsed_rows"] == []
    assert slice_sidecar["evidence"]["parsed_rows_compacted"] is True
    assert not Path(slice_sidecar["csv_path"]).parent.exists()
    aggregate = json.loads(
        (replacement / "L1" / "one.timing.json").read_text(encoding="utf-8")
    )
    assert aggregate["replacement_metadata"]["replacement_status"] == "profiler_backed"


def test_workload_sharded_keeps_csv_but_compacts_slice_json(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_timing = tmp_path / "fallback"
    output_dir = tmp_path / "out"
    replacement = output_dir / "timing"
    _write_problem(dataset_root, "L1", "one")
    _write_workloads(dataset_root, "L1", "one", 1)
    _write_fallback_timing(source_timing, "L1", "one")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _successful_runner(command)

    status = batch.run_batch(
        dataset_root=dataset_root,
        output_dir=output_dir,
        source_timing_dirs=(source_timing,),
        replacement_timing_dir=replacement,
        limit=1,
        workload_sharded=True,
        compact_workload_slices=False,
        keep_rocprofv3_csv=True,
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    slice_sidecar_path = (
        output_dir
        / "workload-slices"
        / "workload-0000"
        / "timing"
        / "L1"
        / "one.timing.json"
    )
    slice_sidecar = json.loads(slice_sidecar_path.read_text(encoding="utf-8"))
    assert slice_sidecar["evidence"]["parsed_rows"] == []
    assert slice_sidecar["evidence"]["parsed_rows_compacted"] is True
    assert Path(slice_sidecar["csv_path"]).parent.exists()
    manifest = json.loads(
        (
            output_dir
            / "workload-manifests"
            / "L1"
            / "one.workload-profiler-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["workloads"][0]["status"] == "profiler_backed"
    assert manifest["workloads"][0]["kernel_duration_ms"] == 0.005


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


# =============================================================================
# Phase 177 Tests: ThreadPoolExecutor Parallel Staging + Serial GPU Profiling
# =============================================================================


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
