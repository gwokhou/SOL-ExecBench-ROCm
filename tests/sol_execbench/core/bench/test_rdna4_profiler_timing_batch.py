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
