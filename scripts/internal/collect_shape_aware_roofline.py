#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Collect resumable, raw rocprofv3 evidence for a shape-aware roofline plan.

The collector is intentionally single-GPU-worker.  PMC collection is global to
the GPU: running two profiler jobs concurrently makes dispatch/counter
attribution ambiguous, so extra CPU cores would reduce evidence quality rather
than safely speed the job up.  Work remains granular (one fresh subprocess per
workload), is bounded by a timeout, and resumes from checksum-verified raw
files.

It does *not* manufacture an envelope artifact.  In particular, a zero or
missing occupancy counter is retained as ``occupancy_unavailable`` and prevents
that workload/profile assignment from becoming validated evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.scoring.hardware_calibration.shape_aware_roofline import (
    has_measured_occupancy,
)
from sol_execbench.core.timestamps import utc_timestamp


PLAN_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline_plan.v1"
RAW_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline_raw.v1"
REPORT_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline_collection.v1"
_OCCUPANCY_COUNTERS = (
    "MeanOccupancyPerActiveCU",
    "OccupancyPercent",
    # Retain the raw terms as a fallback audit path: on some ROCm revisions a
    # derived occupancy metric is exposed but evaluates to zero while raw PMC
    # support changes independently.  We only accept the ratio when both raw
    # terms are positive; a lone busy-cycle value is not occupancy evidence.
    "SQ_WAVE_CYCLES",
    "SQ_BUSY_CYCLES",
    "SQ_WAVES",
    "GRBM_GUI_ACTIVE",
)


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, order=True)
class WorkloadTask:
    definition: str
    workload_uuid: str
    problem_id: str
    profile_keys: tuple[str, ...]

    @property
    def safe_id(self) -> str:
        return self.workload_uuid


@dataclass(frozen=True)
class DispatchObservation:
    launch: dict[str, int]
    resources: dict[str, int]
    counters: dict[str, float]
    duration_ns: int

    @property
    def occupancy_available(self) -> bool:
        return has_measured_occupancy(self.counters)


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_plan(path: Path) -> tuple[dict[str, Any], tuple[WorkloadTask, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "architecture",
        "authority_coverage_sha256",
        "requirements_sha256",
        "required_dimensions",
        "authority_workload_count",
        "profile_shards",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("shape-aware collection plan has invalid fields")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("shape-aware collection plan schema is unsupported")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload["payload_sha256"] != _canonical_digest(unsigned):
        raise ValueError("shape-aware collection plan checksum mismatch")
    if payload["required_dimensions"] != ["shape", "layout", "launch", "occupancy"]:
        raise ValueError("shape-aware collection plan dimensions are invalid")
    _require_string(payload["architecture"], "plan architecture")
    _require_positive_int(
        payload["authority_workload_count"], "authority_workload_count"
    )
    shards = payload["profile_shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("shape-aware collection plan has no profile shards")
    grouped: dict[tuple[str, str, str], set[str]] = {}
    assignments: set[tuple[str, str, str, str]] = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"profile_key", "workloads"}:
            raise ValueError("shape-aware collection plan shard is invalid")
        profile_key = _require_string(shard["profile_key"], "profile_key")
        workloads = shard["workloads"]
        if not isinstance(workloads, list) or not workloads:
            raise ValueError("shape-aware collection plan shard has no workloads")
        for workload in workloads:
            if not isinstance(workload, dict) or set(workload) != {
                "definition",
                "workload_uuid",
                "problem_id",
            }:
                raise ValueError("shape-aware collection workload is invalid")
            identity = (
                _require_string(workload["definition"], "definition"),
                _require_string(workload["workload_uuid"], "workload_uuid"),
                _require_string(workload["problem_id"], "problem_id"),
            )
            assignment = (profile_key, *identity)
            if assignment in assignments:
                raise ValueError("shape-aware collection plan has duplicate assignment")
            assignments.add(assignment)
            grouped.setdefault(identity, set()).add(profile_key)
    if len(grouped) != payload["authority_workload_count"]:
        raise ValueError(
            "shape-aware collection plan workload count does not match shards"
        )
    tasks = tuple(
        WorkloadTask(identity[0], identity[1], identity[2], tuple(sorted(profile_keys)))
        for identity, profile_keys in sorted(grouped.items(), key=lambda item: item[0])
    )
    return payload, tasks


def select_tasks(
    tasks: tuple[WorkloadTask, ...], excluded_workload_uuids: Iterable[str]
) -> tuple[tuple[WorkloadTask, ...], tuple[WorkloadTask, ...]]:
    """Select diagnostic tasks while retaining an auditable exclusion record."""
    requested = tuple(excluded_workload_uuids)
    if len(set(requested)) != len(requested):
        raise ValueError("excluded workload UUIDs must be unique")
    known = {task.workload_uuid for task in tasks}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"excluded workload UUID is not in the plan: {unknown[0]}")
    excluded = set(requested)
    return (
        tuple(task for task in tasks if task.workload_uuid not in excluded),
        tuple(task for task in tasks if task.workload_uuid in excluded),
    )


def _int(value: object, field: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rocprof field {field} is invalid") from exc
    if result < 0:
        raise ValueError(f"rocprof field {field} is negative")
    return result


def _float(value: object, field: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rocprof field {field} is invalid") from exc
    if result < 0.0:
        raise ValueError(f"rocprof field {field} is negative")
    return result


def parse_rocprof_csv(trace_root: Path) -> tuple[DispatchObservation, tuple[Path, ...]]:
    """Return the longest actual dispatch and all raw trace files it binds."""
    counter_files = sorted(trace_root.rglob("*_counter_collection.csv"))
    kernel_files = sorted(trace_root.rglob("*_kernel_trace.csv"))
    if not counter_files or not kernel_files:
        raise ValueError("rocprofv3 did not produce counter and kernel trace CSV files")
    kernels: dict[int, dict[str, int]] = {}
    for path in kernel_files:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                dispatch_id = _int(row.get("Dispatch_Id"), "Dispatch_Id")
                kernels[dispatch_id] = {
                    "grid_x": _int(row.get("Grid_Size_X"), "Grid_Size_X"),
                    "grid_y": _int(row.get("Grid_Size_Y"), "Grid_Size_Y"),
                    "grid_z": _int(row.get("Grid_Size_Z"), "Grid_Size_Z"),
                    "block_x": _int(row.get("Workgroup_Size_X"), "Workgroup_Size_X"),
                    "block_y": _int(row.get("Workgroup_Size_Y"), "Workgroup_Size_Y"),
                    "block_z": _int(row.get("Workgroup_Size_Z"), "Workgroup_Size_Z"),
                    "lds_bytes": _int(row.get("LDS_Block_Size"), "LDS_Block_Size"),
                    "scratch_bytes": _int(row.get("Scratch_Size"), "Scratch_Size"),
                    "vgpr_count": _int(row.get("VGPR_Count"), "VGPR_Count"),
                    "sgpr_count": _int(row.get("SGPR_Count"), "SGPR_Count"),
                }
    grouped: dict[int, dict[str, Any]] = {}
    for path in counter_files:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                dispatch_id = _int(row.get("Dispatch_Id"), "Dispatch_Id")
                item = grouped.setdefault(dispatch_id, {"counters": {}})
                item["duration_ns"] = max(
                    item.get("duration_ns", 0),
                    _int(row.get("End_Timestamp"), "End_Timestamp")
                    - _int(row.get("Start_Timestamp"), "Start_Timestamp"),
                )
                name = _require_string(row.get("Counter_Name"), "Counter_Name")
                item["counters"][name] = _float(
                    row.get("Counter_Value"), "Counter_Value"
                )
    observations = [
        DispatchObservation(
            launch=launch,
            resources={
                key: value
                for key, value in launch.items()
                if key.startswith(("lds_", "scratch_", "vgpr_", "sgpr_"))
            },
            counters=dict(item["counters"]),
            duration_ns=_require_positive_int(
                item.get("duration_ns"), "dispatch duration"
            ),
        )
        for dispatch_id, item in grouped.items()
        if (launch := kernels.get(dispatch_id)) is not None
        and all(
            value > 0
            for key, value in launch.items()
            if key.startswith(("grid_", "block_"))
        )
    ]
    if not observations:
        raise ValueError("rocprofv3 produced no attributable positive dispatch")
    selected = max(observations, key=lambda item: item.duration_ns)
    raw_files = tuple(sorted({*counter_files, *kernel_files}))
    return selected, raw_files


def _tensor_shape_and_layout(
    provider_evidence: Mapping[str, Any],
) -> tuple[tuple[int, ...], str]:
    layouts = provider_evidence.get("tensor_layouts")
    if not isinstance(layouts, Mapping) or not isinstance(layouts.get("inputs"), list):
        raise ValueError("provider evidence lacks concrete tensor layouts")
    flat_shape: list[int] = []
    for tensor in layouts["inputs"]:
        if not isinstance(tensor, Mapping) or not isinstance(tensor.get("shape"), list):
            raise ValueError("provider input layout is invalid")
        flat_shape.extend(
            _require_positive_int(item, "input shape") for item in tensor["shape"]
        )
    if not flat_shape:
        # Scalar-only workloads still have a concrete shape contract.  Encode
        # it explicitly without claiming an invented tensor extent.
        flat_shape.append(1)
    return tuple(flat_shape), "layout_sha256:" + _canonical_digest(layouts)


def _tool_identity(executable: str, *, include_version: bool = True) -> dict[str, str]:
    """Collect a bounded identity record for an executable used as evidence."""

    path = Path(executable).resolve()
    if not path.is_file():
        raise ValueError(f"required executable is missing: {executable}")
    identity = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if include_version:
        version = subprocess.run(
            [str(path), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        output = (version.stdout + version.stderr).strip()
        identity["version"] = output[-2000:] or f"exit:{version.returncode}"
    return identity


def _performance_level(amd_smi: str) -> str:
    """Return bounded raw amd-smi level output for patch provenance."""

    result = subprocess.run(
        [amd_smi, "metric", "-l"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not output:
        raise ValueError("amd-smi could not read GPU performance level")
    return output[-4000:]


def _runtime_identity(amd_smi: str) -> dict[str, str | None]:
    """Record the runtime identities that qualify a local patch observation."""

    static = subprocess.run(
        [amd_smi, "static", "-a"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    static_output = (static.stdout + static.stderr).strip()
    try:
        import torch

        torch_version = str(torch.__version__)
        hip_version = str(torch.version.hip) if torch.version.hip else None
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        torch_version = hip_version = gpu_name = None
    return {
        "rocm_hip_version": hip_version,
        "pytorch_version": torch_version,
        "gpu_name": gpu_name,
        "driver_and_gpu_static_sha256": hashlib.sha256(
            static_output.encode()
        ).hexdigest(),
    }


def _patch_provenance(
    *,
    patch_id: str | None,
    wrapper: str,
    real_rocprofv3: str | None,
    amd_smi: str | None,
    pre_level: str | None,
    post_level: str | None,
    profiler_stderr: str,
) -> dict[str, Any] | None:
    """Build mandatory provenance only for an explicitly selected workaround."""

    if patch_id is None:
        return None
    if not real_rocprofv3 or not amd_smi or pre_level is None or post_level is None:
        raise ValueError(
            "patched collection requires real profiler, amd-smi, and levels"
        )
    if "patch_clock_state=" not in profiler_stderr:
        raise ValueError("patched profiler did not report verified clock acquisition")
    acquired = "patch_clock_state=acquired_stable_peak" in profiler_stderr
    reset_ok = not acquired or "patch_error=clock_reset" not in profiler_stderr
    return {
        "patch_id": patch_id,
        "wrapper": _tool_identity(wrapper, include_version=False),
        "real_rocprofv3": _tool_identity(real_rocprofv3),
        "amd_smi": _tool_identity(amd_smi),
        "runtime": _runtime_identity(amd_smi),
        "performance_levels": {
            "pre": pre_level,
            "during": "STABLE_PEAK verified by patched wrapper",
            "post": post_level,
        },
        "clock_lease": {
            "acquired_by_current_process": acquired,
            "reset_verified": reset_ok,
        },
    }


def _raw_payload(
    *,
    task: WorkloadTask,
    architecture: str,
    provider_path: Path,
    provider_evidence: Mapping[str, Any],
    trace_files: Iterable[Path],
    observation: DispatchObservation,
    patch_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    shape, layout = _tensor_shape_and_layout(provider_evidence)
    samples = provider_evidence.get("samples_ms")
    if not isinstance(samples, list) or len(samples) < 7:
        raise ValueError("provider evidence lacks seven timed samples")
    samples_ms = [float(value) for value in samples]
    if any(value <= 0.0 for value in samples_ms):
        raise ValueError("provider evidence has invalid timed samples")
    payload: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "architecture": architecture,
        "definition": task.definition,
        "workload_uuid": task.workload_uuid,
        "problem_id": task.problem_id,
        "profile_keys": list(task.profile_keys),
        "shape": list(shape),
        "layout": layout,
        "tensor_layouts": provider_evidence["tensor_layouts"],
        "samples_ms": samples_ms,
        "provider_evidence_ref": str(provider_path.resolve()),
        "provider_evidence_sha256": sha256_file(provider_path),
        "trace_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in trace_files
        ],
        "launch": {
            key: value
            for key, value in observation.launch.items()
            if key.startswith(("grid_", "block_"))
        },
        "kernel_resources": observation.resources,
        "occupancy_counters": observation.counters,
        "representative_dispatch_duration_ns": observation.duration_ns,
        "occupancy_status": (
            "measured" if observation.occupancy_available else "unavailable"
        ),
    }
    if patch_provenance is not None:
        patch_provenance["raw_profiler_csvs"] = list(payload["trace_files"])
        payload["patch_provenance"] = patch_provenance
    payload["payload_sha256"] = _canonical_digest(payload)
    return payload


def _problem_dir(benchmark_root: Path, task: WorkloadTask) -> Path:
    candidate = benchmark_root / task.problem_id
    if not (candidate / "definition.json").is_file():
        raise ValueError(f"benchmark problem directory is missing: {candidate}")
    return candidate


def _run_task(
    task: WorkloadTask,
    *,
    architecture: str,
    benchmark_root: Path,
    output_root: Path,
    provider_script: Path,
    rocprofv3: str,
    warmup: int,
    repetitions: int,
    timeout_seconds: int,
    patch_id: str | None,
    real_rocprofv3: str | None,
    amd_smi: str | None,
) -> dict[str, Any]:
    raw_path = output_root / "raw" / f"{task.safe_id}.json"
    if raw_path.is_file():
        try:
            existing = json.loads(raw_path.read_text(encoding="utf-8"))
            if (
                existing.get("payload_sha256")
                == _canonical_digest(
                    {
                        key: value
                        for key, value in existing.items()
                        if key != "payload_sha256"
                    }
                )
                and existing.get("architecture") == architecture
                and existing.get("workload_uuid") == task.workload_uuid
                and existing.get("profile_keys") == list(task.profile_keys)
                and all(
                    Path(item["path"]).is_file()
                    and sha256_file(Path(item["path"])) == item["sha256"]
                    for item in existing["trace_files"]
                )
            ):
                return {
                    "status": "resumed",
                    "raw_evidence_ref": str(raw_path),
                    "raw_evidence_sha256": sha256_file(raw_path),
                    "occupancy_status": existing.get("occupancy_status"),
                }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    run_root = output_root / "runs" / task.safe_id
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)
    provider_path = output_root / "provider" / f"{task.safe_id}.json"
    provider_row_path = output_root / "provider" / f"{task.safe_id}.jsonl"
    command = [
        rocprofv3,
        "--kernel-trace",
        "--pmc",
        ",".join(_OCCUPANCY_COUNTERS),
        "--output-format",
        "csv",
        "--output-directory",
        str(run_root),
        "--",
        sys.executable,
        str(provider_script),
        "--problem-dir",
        str(_problem_dir(benchmark_root, task)),
        "--workload-uuid",
        task.workload_uuid,
        "--jsonl-output",
        str(provider_row_path),
        "--evidence-output",
        str(provider_path),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
    ]
    resolved_amd_smi = amd_smi or str(resolve_rocm_tool("amd-smi") or "")
    pre_level = _performance_level(resolved_amd_smi) if patch_id is not None else None
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": "rocprof_or_provider_failed",
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:],
        }
    post_level = _performance_level(resolved_amd_smi) if patch_id is not None else None
    provenance = _patch_provenance(
        patch_id=patch_id,
        wrapper=rocprofv3,
        real_rocprofv3=real_rocprofv3,
        amd_smi=resolved_amd_smi,
        pre_level=pre_level,
        post_level=post_level,
        profiler_stderr=result.stderr,
    )
    provider_evidence = json.loads(provider_path.read_text(encoding="utf-8"))
    observation, trace_files = parse_rocprof_csv(run_root)
    payload = _raw_payload(
        task=task,
        architecture=architecture,
        provider_path=provider_path,
        provider_evidence=provider_evidence,
        trace_files=trace_files,
        observation=observation,
        patch_provenance=provenance,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "collected",
        "raw_evidence_ref": str(raw_path),
        "raw_evidence_sha256": sha256_file(raw_path),
        "occupancy_status": payload["occupancy_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--rocprofv3",
        default=(
            "/opt/rocm/bin/rocprofv3"
            if Path("/opt/rocm/bin/rocprofv3").is_file()
            else "rocprofv3"
        ),
    )
    parser.add_argument(
        "--patch-id",
        default=None,
        help="Patch ID when --rocprofv3 is the gfx1200 patched wrapper.",
    )
    parser.add_argument(
        "--real-rocprofv3",
        default=None,
        help="Path to the unwrapped ROCm rocprofv3 required for patch provenance.",
    )
    parser.add_argument("--amd-smi", default=None)
    parser.add_argument(
        "--provider-script",
        type=Path,
        default=Path(__file__).with_name("run_torch_inductor_provider.py"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-workloads", type=int, default=None)
    parser.add_argument("--exclude-workload", action="append", default=[])
    parser.add_argument(
        "--exclusion-reason",
        default=None,
        help="Required diagnostic reason when --exclude-workload is used.",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 7 or args.timeout_seconds < 1:
        raise ValueError(
            "warmup >= 1, repetitions >= 7, and positive timeout are required"
        )
    patched_name = "rocprofv3-gfx1200-patched"
    uses_patched_wrapper = Path(args.rocprofv3).name == patched_name
    if uses_patched_wrapper and not args.patch_id:
        raise ValueError("patched rocprofv3 requires --patch-id provenance")
    if args.patch_id and not uses_patched_wrapper:
        raise ValueError("--patch-id requires the explicit patched rocprofv3 wrapper")
    if args.patch_id and not args.real_rocprofv3:
        raise ValueError("patched rocprofv3 requires --real-rocprofv3")
    plan, tasks = load_plan(args.plan)
    tasks, excluded_tasks = select_tasks(tasks, args.exclude_workload)
    if excluded_tasks and not args.exclusion_reason:
        raise ValueError("--exclude-workload requires --exclusion-reason")
    if args.exclusion_reason and not excluded_tasks:
        raise ValueError("--exclusion-reason requires --exclude-workload")
    if args.max_workloads is not None:
        if args.max_workloads < 1:
            raise ValueError("max-workloads must be positive")
        tasks = tasks[: args.max_workloads]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row: dict[str, Any] = {
            "definition": task.definition,
            "workload_uuid": task.workload_uuid,
            "problem_id": task.problem_id,
            "profile_keys": list(task.profile_keys),
        }
        try:
            row.update(
                _run_task(
                    task,
                    architecture=plan["architecture"],
                    benchmark_root=args.benchmark_root,
                    output_root=args.output_root,
                    provider_script=args.provider_script,
                    rocprofv3=args.rocprofv3,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    timeout_seconds=args.timeout_seconds,
                    patch_id=args.patch_id,
                    real_rocprofv3=args.real_rocprofv3,
                    amd_smi=args.amd_smi,
                )
            )
        except (
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as exc:
            row.update({"status": "failed", "reason": str(exc)})
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    complete = (
        not excluded_tasks
        and len(tasks) == plan["authority_workload_count"]
        and all(
            row["status"] in {"collected", "resumed"}
            and row.get("occupancy_status") == "measured"
            for row in rows
        )
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "architecture": plan["architecture"],
        "plan_payload_sha256": plan["payload_sha256"],
        "planned_authority_workloads": plan["authority_workload_count"],
        "attempted_workloads": len(tasks),
        "collection_status": "collected" if complete else "incomplete",
        "workloads": rows,
    }
    if excluded_tasks:
        report["excluded_workloads"] = [
            {
                "definition": task.definition,
                "workload_uuid": task.workload_uuid,
                "problem_id": task.problem_id,
                "profile_keys": list(task.profile_keys),
                "reason": args.exclusion_reason,
                "authority_eligible": False,
            }
            for task in excluded_tasks
        ]
    report["payload_sha256"] = _canonical_digest(report)
    report_path = args.output_root / "collection-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(report_path),
                "collection_status": report["collection_status"],
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
