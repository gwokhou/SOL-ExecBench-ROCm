# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lightweight ROCm hardware counter capture (rocprofv3 CSV)."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from pathlib import Path

from scripts.mvp_v1.analytical_roofline import VendorBackend
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.rocm_profiler.counters import (
    ROCPROFV3_AVAIL_EXECUTABLE,
    CounterPassCSV,
    build_rocprofv3_counter_command,
    counter_manifest_for_architecture,
    counter_pass_index,
    load_counter_manifest,
    parse_and_align_counter_passes,
    parse_available_counters,
    select_counter_groups,
    write_counter_job,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.environment import (
    ENV_HIP_VISIBLE_DEVICES,
    ENV_ROCR_VISIBLE_DEVICES,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

PROFILE_RANGE_MARKER = "MVP_MEASURE"
_TIMEOUT_SECONDS = 120.0
_MAX_OUTPUT_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterCaptureSummary:
    """Normalized observed memory/compute counter summary for one workload."""

    status: str
    backend: VendorBackend
    memory_bytes: float | None = None
    waves: float | None = None
    mfma_instructions: float | None = None
    counters_used: tuple[str, ...] = ()
    note: str | None = None


def parse_rocprof_mvp_passes(
    pass_csvs: Sequence[Path],
    *,
    workload_uuid: str,
    candidate_sha256: str = "0" * 64,
) -> CounterCaptureSummary:
    """Align rocprofv3 CSV passes into normalized memory bytes and activity."""
    indexed = [
        CounterPassCSV(
            pass_index=counter_pass_index(path) or (idx + 1),
            path=path,
        )
        for idx, path in enumerate(sorted(pass_csvs))
    ]
    dispatches = parse_and_align_counter_passes(
        indexed,
        workload_uuid=workload_uuid,
        candidate_sha256=candidate_sha256,
    )
    if not dispatches:
        return CounterCaptureSummary(
            status="empty",
            backend=VendorBackend.AMD_ROCM,
            note="rocprofv3 produced no kernel dispatches",
        )
    mem_bytes = sum(counter_memory_bytes(d.counters) for d in dispatches)
    waves = sum(
        d.counters.get("SQ_WAVES_SUM", d.counters.get("SQ_WAVES", 0.0))
        for d in dispatches
    )
    mfma = sum(
        d.counters.get(
            "SQ_INSTS_MFMA_SUM", d.counters.get("SQ_INSTS_MFMA", 0.0)
        )
        for d in dispatches
    )
    names = tuple(sorted({k for d in dispatches for k in d.counters}))
    note = (
        None
        if mem_bytes > 0
        else "memory byte counters returned 0 (unprivileged or warm cache)"
    )
    return CounterCaptureSummary(
        status="collected",
        backend=VendorBackend.AMD_ROCM,
        memory_bytes=mem_bytes if mem_bytes > 0 else None,
        waves=waves if waves > 0 else None,
        mfma_instructions=mfma if mfma > 0 else None,
        counters_used=names,
        note=note,
    )


def _resolve_rocprofv3_binary() -> Path | None:
    for match in sorted(
        Path(sys.prefix).glob(
            "lib/python*/site-packages/_rocm_sdk_core/bin/rocprofv3"
        )
    ):
        if match.is_file():
            return match
    return resolve_rocm_tool("rocprofv3")


def _rocm_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    if (
        ENV_ROCR_VISIBLE_DEVICES not in env
        and ENV_HIP_VISIBLE_DEVICES not in env
    ):
        env[ENV_ROCR_VISIBLE_DEVICES] = "0"
        env[ENV_HIP_VISIBLE_DEVICES] = "0"
    return env


def _query_rocm_available_counters(
    rocprof: Path,
    env: dict[str, str],
) -> frozenset[str] | None:
    list_res = run_in_process_group_bounded(
        [str(rocprof), "--list-avail"],
        timeout=_TIMEOUT_SECONDS,
        max_capture_bytes=_MAX_OUTPUT_BYTES * 8,
        env=env,
    )
    if list_res.returncode == 0 and list_res.stdout.strip():
        parsed = parse_available_counters(list_res.stdout)
        if parsed:
            return parsed
    avail = resolve_rocm_tool(ROCPROFV3_AVAIL_EXECUTABLE)
    if avail is None:
        return None
    avail_res = run_in_process_group_bounded(
        [str(avail), "-d", "0", "info", "--pmc"],
        timeout=_TIMEOUT_SECONDS,
        max_capture_bytes=_MAX_OUTPUT_BYTES * 8,
        env=env,
    )
    if avail_res.returncode != 0:
        return None
    return parse_available_counters(avail_res.stdout)


def _marker_window_ns(pass_dir: Path) -> tuple[int, int] | None:
    for marker_csv in sorted(pass_dir.rglob("*marker_api_trace*.csv")):
        with marker_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if (row.get("Function") or "").strip() == PROFILE_RANGE_MARKER:
                    start_raw = (row.get("Start_Timestamp") or "").strip()
                    end_raw = (row.get("End_Timestamp") or "").strip()
                    if start_raw.isdigit() and end_raw.isdigit():
                        return int(start_raw), int(end_raw)
    return None


def _filter_pass_csv_to_marker(csv_path: Path, pass_dir: Path) -> Path:
    window = _marker_window_ns(pass_dir)
    if window is None:
        return csv_path
    start_ns, end_ns = window
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        kept = [
            row
            for row in reader
            if (ts := (row.get("Start_Timestamp") or "").strip()).isdigit()
            and start_ns <= int(ts) <= end_ns
        ]
    filtered = csv_path.with_name(f"measured_{csv_path.name}")
    with filtered.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return filtered


def _run_one_rocm_pass(
    rocprof: Path,
    group: Sequence[str],
    replay_command: Sequence[str],
    *,
    pass_dir: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    job_yaml = pass_dir.parent / f"{pass_dir.name}.yaml"
    write_counter_job(job_yaml, [group], output_directory=str(pass_dir))
    cmd = build_rocprofv3_counter_command(
        replay_command, input_path=job_yaml, executable=str(rocprof)
    )
    return run_in_process_group_bounded(
        cmd,
        timeout=_TIMEOUT_SECONDS,
        max_capture_bytes=_MAX_OUTPUT_BYTES,
        env=env,
    )


def _execute_rocm_groups(
    rocprof: Path,
    groups: Sequence[Sequence[str]],
    replay_command: Sequence[str],
    *,
    tmp: Path,
    env: dict[str, str],
    skipped_out: list[str] | None = None,
) -> tuple[list[Path], str | None]:
    pending: list[Sequence[str]] = list(groups)
    csvs: list[Path] = []
    pass_idx = 0
    last_rc = 0
    while pending:
        group = pending.pop(0)
        pass_idx += 1
        pass_dir = tmp / f"pass_{pass_idx}"
        res = _run_one_rocm_pass(
            rocprof, group, replay_command, pass_dir=pass_dir, env=env
        )
        if res.returncode != 0:
            last_rc = res.returncode
            if len(group) > 1:
                pending = [[c] for c in group] + pending
                continue
            if skipped_out is not None:
                skipped_out.append(group[0])
            continue
        found = sorted(pass_dir.rglob("*counter_collection*.csv"))
        for raw_csv in found:
            csvs.append(_filter_pass_csv_to_marker(raw_csv, pass_dir))
    if not csvs:
        return [], f"all rocprofv3 passes failed (last rc={last_rc})"
    return csvs, None


def _collect_rocm_counters(
    replay_command: Sequence[str],
    *,
    architecture: str,
    workload_id: str,
) -> CounterCaptureSummary:
    rocprof = _resolve_rocprofv3_binary()
    manifest_name = counter_manifest_for_architecture(architecture)
    if rocprof is None or manifest_name is None:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note=f"rocprofv3 tooling/manifest unavailable for {architecture}",
        )
    env = _rocm_subprocess_env()
    available = _query_rocm_available_counters(rocprof, env)
    if available is None:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note="rocprofv3 counter discovery failed",
        )
    manifest_res = files("sol_execbench.data.rocprofv3_counters").joinpath(
        manifest_name
    )
    with as_file(manifest_res) as manifest_path:
        manifest = load_counter_manifest(manifest_path)
    groups, missing = select_counter_groups(manifest, available)
    if not groups:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note=f"missing all counters: {missing}",
        )
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mvp-rocprof-") as tmp_dir:
        csvs, err_note = _execute_rocm_groups(
            rocprof,
            groups,
            replay_command,
            tmp=Path(tmp_dir),
            env=env,
            skipped_out=skipped,
        )
        if err_note is not None:
            return CounterCaptureSummary(
                status="failed",
                backend=VendorBackend.AMD_ROCM,
                note=err_note,
            )
        summary = parse_rocprof_mvp_passes(csvs, workload_uuid=workload_id)
    partial = [*missing, *skipped]
    if partial and summary.status == "collected":
        prefix = f"{summary.note}; " if summary.note else ""
        return replace(
            summary, note=f"{prefix}partial counters (skipped {partial})"
        )
    return summary


def collect_hardware_counters(
    replay_command: Sequence[str],
    *,
    backend: VendorBackend,
    architecture: str,
    workload_id: str,
) -> CounterCaptureSummary:
    """Collect hardware counters via rocprofv3 on ROCm GPUs."""
    if backend is VendorBackend.CPU:
        return CounterCaptureSummary(
            status="skipped_cpu",
            backend=backend,
            note="counter collection skipped on CPU",
        )
    return _collect_rocm_counters(
        replay_command, architecture=architecture, workload_id=workload_id
    )
