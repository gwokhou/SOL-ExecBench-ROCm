from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.baseline_export import export_hip_baseline_registry


def test_export_hip_baseline_registry_from_passed_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(_passed_trace()) + "\n", encoding="utf-8")
    output_path = tmp_path / "baseline-registry.json"

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id="attention",
        sol_version="sol-test",
        timing_policy="latency_ms",
    )

    assert registry["schema_version"] == "baseline_registry.v1"
    assert registry["coverage_status"] == "confirmed"
    assert registry["target_id"] == "attention"
    assert registry["expected_workload_keys"] == ["w-attn-1"]
    entry = registry["entries"][0]
    assert entry["workload_key"] == "w-attn-1"
    assert entry["latency_ms"] == 1.25
    assert entry["score"] == 2.0
    assert entry["trace_ref"] == str(trace_path)
    assert entry["provenance"] == {
        "hardware": "gfx1200",
        "rocm_version": "7.1",
        "sol_version": "sol-test",
        "target_id": "attention",
        "timing_policy": "latency_ms",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == registry


def test_baseline_export_cli_writes_registry(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(_passed_trace()) + "\n", encoding="utf-8")
    output_path = tmp_path / "baseline-registry.json"

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "export",
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--target-id",
            "attention",
            "--sol-version",
            "sol-test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["coverage_status"] == "confirmed"
    assert output_path.exists()


def test_export_hip_baseline_registry_keeps_failed_workloads_in_expected_coverage(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    failed_trace = _passed_trace()
    failed_trace["workload"] = {
        "uuid": "w-attn-failed",
        "axes": {"B": 2, "S": 256},
        "inputs": {},
    }
    failed_trace["evaluation"] = {
        "status": "RUNTIME_ERROR",
        "environment": {"hardware": "gfx1200", "libs": {"rocm": "7.1"}},
        "timestamp": "2026-06-22T00:00:01Z",
        "log": "runtime failure",
    }
    trace_path.write_text(
        json.dumps(_passed_trace()) + "\n" + json.dumps(failed_trace) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "baseline-registry.json"

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id="attention",
        sol_version="sol-test",
        timing_policy="latency_ms",
    )

    assert registry["coverage_status"] == "diagnostic"
    assert registry["expected_workload_keys"] == ["w-attn-1", "w-attn-failed"]
    assert [entry["workload_key"] for entry in registry["entries"]] == ["w-attn-1"]


def test_export_rejects_nan_latency_and_does_not_confirm_coverage(
    tmp_path: Path,
) -> None:
    trace: dict[str, Any] = _passed_trace()
    trace["evaluation"]["performance"]["latency_ms"] = float("nan")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    output_path = tmp_path / "baseline-registry.json"

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id="attention",
        sol_version="sol-test",
        timing_policy="latency_ms",
    )

    # NaN latency is dropped -> workload expected but uncovered, and the file
    # must not contain a non-standard `NaN` JSON token.
    assert registry["entries"] == []
    assert registry["coverage_status"] == "diagnostic"
    assert "NaN" not in output_path.read_text(encoding="utf-8")


def test_workload_key_axis_fallback_does_not_collide(tmp_path: Path) -> None:
    def _trace(axes: dict[str, int]) -> dict[str, object]:
        return {
            "definition": "op",
            "workload": {"axes": axes, "inputs": {}},
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "gfx1200", "libs": {"rocm": "7.1"}},
                "performance": {"latency_ms": 1.0},
            },
        }

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(_trace({"n": 12})) + "\n" + json.dumps(_trace({"n1": 2})) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "baseline-registry.json"

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id="op",
        sol_version="sol-test",
        timing_policy="latency_ms",
    )

    # {'n': 12} and {'n1': 2} must produce distinct keys (no "n12" collision).
    assert registry["expected_workload_keys"] == ["n=12", "n1=2"]
    assert registry["coverage_status"] == "confirmed"


def _passed_trace() -> dict[str, object]:
    return {
        "definition": "jamba_attn_proj",
        "workload": {"uuid": "w-attn-1", "axes": {"B": 1, "S": 128}, "inputs": {}},
        "solution": "baseline",
        "evaluation": {
            "status": "PASSED",
            "environment": {"hardware": "gfx1200", "libs": {"rocm": "7.1"}},
            "timestamp": "2026-06-22T00:00:00Z",
            "correctness": {"max_relative_error": 0.0, "max_absolute_error": 0.0},
            "performance": {
                "latency_ms": 1.25,
                "reference_latency_ms": 2.5,
                "speedup_factor": 2.0,
            },
        },
    }
