from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


HIP_BASELINE_REGISTRY_SCHEMA_VERSION = "baseline_registry.v1"
SOL_MEASURED_BASELINE_REGISTRY_SCHEMA_VERSION = (
    "sol_execbench.measured_baseline_registry.v1"
)


@dataclass(frozen=True)
class _BaselineTraceRecord:
    trace: dict[str, Any]
    index: int
    workload: dict[str, Any]
    workload_uuid: str | None
    workload_key: str
    evaluation: dict[str, Any]
    performance: dict[str, Any]
    environment: dict[str, Any]
    libs: dict[str, Any]


def export_hip_baseline_registry(
    *,
    trace_path: Path,
    output_path: Path,
    target_id: str,
    sol_version: str,
    timing_policy: str,
) -> dict[str, Any]:
    """Export a measured HIP baseline registry from a SOL trace JSONL file."""

    traces = _load_trace_jsonl(trace_path)
    entries: list[dict[str, Any]] = []
    expected_workload_keys: list[str] = []
    for index, trace in enumerate(traces):
        record = _baseline_trace_record(trace, index=index)
        expected_workload_keys.append(record.workload_key)
        if record.evaluation.get("status") != "PASSED":
            continue
        if not record.performance:
            continue
        latency_ms = _positive_float(record.performance.get("latency_ms"))
        if latency_ms is None:
            continue
        entries.append(
            {
                "target_id": target_id,
                "definition": str(record.trace.get("definition") or target_id),
                "workload_key": record.workload_key,
                "workload_uuid": record.workload_uuid,
                "latency_ms": latency_ms,
                "score": _optional_float(record.performance.get("speedup_factor")),
                "trace_ref": str(trace_path),
                "source": "SOL-ExecBench-ROCm measured baseline trace",
                "artifact_ref": f"{trace_path}#{index + 1}",
                "provenance": {
                    "hardware": str(record.environment.get("hardware") or "unknown"),
                    "rocm_version": _rocm_version(record.libs),
                    "sol_version": sol_version or "unknown",
                    "target_id": target_id,
                    "timing_policy": timing_policy,
                },
                "facts": {
                    "latency_ms": latency_ms,
                    "reference_latency_ms": _optional_float(
                        record.performance.get("reference_latency_ms")
                    ),
                    "speedup_factor": _optional_float(
                        record.performance.get("speedup_factor")
                    ),
                    "libs": record.libs,
                    "trace_line": index + 1,
                },
            }
        )

    covered_workload_keys = {entry["workload_key"] for entry in entries}
    expected_workload_keys = _dedupe(expected_workload_keys)
    coverage_confirmed = bool(expected_workload_keys) and covered_workload_keys == set(
        expected_workload_keys
    )
    registry = {
        "schema_version": HIP_BASELINE_REGISTRY_SCHEMA_VERSION,
        "sol_schema_version": SOL_MEASURED_BASELINE_REGISTRY_SCHEMA_VERSION,
        "target_id": target_id,
        "coverage_status": "confirmed" if coverage_confirmed else "diagnostic",
        "expected_workload_keys": expected_workload_keys,
        "source_artifact": str(trace_path),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry


def _load_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            traces.append(payload)
    return traces


def _baseline_trace_record(
    trace: dict[str, Any], *, index: int
) -> _BaselineTraceRecord:
    workload = _mapping(trace, "workload")
    workload_uuid = _string_or_none(workload.get("uuid"))
    evaluation = _mapping(trace, "evaluation")
    performance = _mapping(evaluation, "performance")
    environment = _mapping(evaluation, "environment")
    libs = _mapping(environment, "libs")
    return _BaselineTraceRecord(
        trace=trace,
        index=index,
        workload=workload,
        workload_uuid=workload_uuid,
        workload_key=_workload_key(
            trace,
            workload=workload,
            workload_uuid=workload_uuid,
            index=index,
        ),
        evaluation=evaluation,
        performance=performance,
        environment=environment,
        libs=libs,
    )


def _workload_key(
    trace: dict[str, Any],
    *,
    workload: dict[str, Any],
    workload_uuid: str | None,
    index: int,
) -> str:
    if workload_uuid:
        return workload_uuid
    axes = _mapping(workload, "axes")
    if axes:
        # Separator between key and value prevents collisions such as
        # {'n': 12} vs {'n1': 2} both yielding "n12".
        axis_parts = "-".join(f"{key}={axes[key]}" for key in sorted(axes))
        if axis_parts:
            return axis_parts
    digest = sha256(
        json.dumps(trace.get("workload"), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return f"workload-{index + 1}-{digest}"


def _positive_float(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None or number <= 0:
        return None
    return number


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # Reject NaN/Inf so they never reach the registry JSON (json.dumps would
        # otherwise emit non-standard `NaN`/`Infinity` tokens).
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _rocm_version(libs: dict[str, Any]) -> str:
    for key in ("rocm", "hip", "torch_hip", "torch.version.hip"):
        value = libs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
