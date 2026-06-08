from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_timing_smoke.py"
SPEC = spec_from_file_location("run_rdna4_profiler_timing_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke = module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


ROCPROFV3_CSV = """Domain,Name,Start_Timestamp,End_Timestamp,Duration(ns)
KERNEL_DISPATCH,rmsnorm_kernel,1000,5000,4000
HIP_RUNTIME_API,hipLaunchKernel,900,5100,4200
"""


def test_rdna4_profiler_smoke_collects_triton_kernel_timing(tmp_path):
    problem_dir = _write_problem(tmp_path)
    output_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        output_directory = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / f"{output_file}.csv").write_text(
            ROCPROFV3_CSV,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    status = smoke.run_smoke(
        problem_dir=problem_dir,
        output_dir=output_dir,
        workload_limit=1,
        tool_version="rocprofv3 7.1.1",
        gpu_architecture="gfx1200",
        rocprofv3_available=True,
        runner=runner,
    )

    assert status == 0
    assert calls
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    timing = json.loads((output_dir / "timing.json").read_text(encoding="utf-8"))
    limited_workload = (output_dir / "workload.smoke.jsonl").read_text(encoding="utf-8")
    assert len(limited_workload.splitlines()) == 1
    assert summary["status"] == "profiler_backed"
    assert summary["profiler_collected"] is True
    assert summary["languages"] == ["triton"]
    assert summary["policy_backend"] == "rocprofv3"
    assert summary["activity_domain"] == "kernel_activity"
    assert summary["kernel_duration_ms"] == 0.004
    assert "not full paper validation" in summary["claim_boundary"]
    assert timing["profiler_collected"] is True
    assert timing["evidence"]["backend"] == "rocprofv3"


def test_rdna4_profiler_smoke_fails_without_profiler_backing(tmp_path):
    problem_dir = _write_problem(tmp_path)
    output_dir = tmp_path / "out"

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    status = smoke.run_smoke(
        problem_dir=problem_dir,
        output_dir=output_dir,
        workload_limit=2,
        rocprofv3_available=False,
        runner=runner,
    )

    assert status == 1
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fallback"
    assert summary["profiler_collected"] is False
    assert summary["policy_backend"] == "device_events"
    assert "rocprofv3 is unavailable" in summary["fallback_reason"]
    assert (
        len(
            (output_dir / "workload.smoke.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )


def test_rdna4_profiler_smoke_rejects_empty_workload_limit(tmp_path):
    problem_dir = _write_problem(tmp_path)

    try:
        smoke.run_smoke(
            problem_dir=problem_dir, output_dir=tmp_path / "out", workload_limit=0
        )
    except ValueError as exc:
        assert "workload_limit must be positive" in str(exc)
    else:
        raise AssertionError("expected workload limit validation failure")


def _write_problem(root: Path) -> Path:
    problem_dir = root / "problem"
    problem_dir.mkdir()
    (problem_dir / "definition.json").write_text("{}", encoding="utf-8")
    (problem_dir / "workload.jsonl").write_text(
        "\n".join(
            [
                '{"uuid": "w1", "axes": {}, "inputs": {}}',
                '{"uuid": "w2", "axes": {}, "inputs": {}}',
                '{"uuid": "w3", "axes": {}, "inputs": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (problem_dir / "solution_triton.json").write_text(
        json.dumps({"spec": {"languages": ["triton"]}}),
        encoding="utf-8",
    )
    return problem_dir
