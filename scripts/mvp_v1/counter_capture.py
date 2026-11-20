# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lightweight dual-vendor hardware counter capture (rocprofv3 CSV / ncu)."""

from __future__ import annotations

import csv
import io
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
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
from sol_execbench.core.platform.runtime import (
    resolve_rocm_tool,
    resolve_tool_path,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_NCU_METRICS = (
    "dram__bytes.sum,"
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
    "gpu__time_duration.sum"
)
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
    tensor_core_pct: float | None = None
    kernel_duration_ns: float | None = None
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
    return CounterCaptureSummary(
        status="collected",
        backend=VendorBackend.AMD_ROCM,
        memory_bytes=mem_bytes,
        waves=waves if waves > 0 else None,
        mfma_instructions=mfma if mfma > 0 else None,
        counters_used=names,
    )


def _to_float_clean(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = raw.strip().replace(",", "")
    if not cleaned or cleaned in {"N/A", "nan"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unit_scale(metric_name: str, unit: str) -> float:
    u = unit.strip().lower()
    if "dram__bytes" in metric_name:
        if u == "gbyte":
            return 1e9
        if u == "mbyte":
            return 1e6
        if u == "kbyte":
            return 1e3
    if "time_duration" in metric_name:
        if u == "msecond":
            return 1e6
        if u == "usecond":
            return 1e3
        if u == "second":
            return 1e9
    return 1.0


def parse_ncu_csv_metrics(csv_text: str) -> CounterCaptureSummary:
    """Parse ``ncu --csv`` stdout into normalized DRAM bytes and Tensor pct."""
    lines = [
        line
        for line in csv_text.splitlines()
        if line.strip() and not line.lstrip().startswith("==")
    ]
    if not lines:
        return CounterCaptureSummary(
            status="empty",
            backend=VendorBackend.NVIDIA_CUDA,
            note="ncu output had no CSV rows",
        )
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    dram_bytes = 0.0
    tensor_pcts: list[float] = []
    durations_ns: list[float] = []
    seen_metrics: set[str] = set()
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        val = _to_float_clean(row.get("Metric Value"))
        if not name or val is None:
            continue
        seen_metrics.add(name)
        scaled = val * _unit_scale(name, row.get("Metric Unit") or "")
        if "dram__bytes" in name:
            dram_bytes += scaled
        elif "tensor_cycles_active" in name:
            tensor_pcts.append(scaled)
        elif "time_duration" in name:
            durations_ns.append(scaled)
    if not seen_metrics:
        return CounterCaptureSummary(
            status="empty",
            backend=VendorBackend.NVIDIA_CUDA,
            note="ncu CSV lacked recognized metrics",
        )
    return CounterCaptureSummary(
        status="collected",
        backend=VendorBackend.NVIDIA_CUDA,
        memory_bytes=dram_bytes if dram_bytes > 0 else None,
        tensor_core_pct=statistics.mean(tensor_pcts) if tensor_pcts else None,
        kernel_duration_ns=sum(durations_ns) if durations_ns else None,
        counters_used=tuple(sorted(seen_metrics)),
    )


def _collect_rocm_counters(
    replay_command: Sequence[str],
    *,
    architecture: str,
    workload_id: str,
) -> CounterCaptureSummary:
    rocprof = resolve_rocm_tool("rocprofv3")
    avail = resolve_rocm_tool(ROCPROFV3_AVAIL_EXECUTABLE)
    manifest_name = counter_manifest_for_architecture(architecture)
    if rocprof is None or avail is None or manifest_name is None:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note=f"rocprofv3 tooling/manifest unavailable for {architecture}",
        )
    avail_res = run_in_process_group_bounded(
        [str(avail), "-d", "0", "info", "--pmc"],
        timeout=_TIMEOUT_SECONDS,
        max_capture_bytes=_MAX_OUTPUT_BYTES * 8,
    )
    if avail_res.returncode != 0:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note="rocprofv3-avail failed",
        )
    manifest_res = files("sol_execbench.data.rocprofv3_counters").joinpath(
        manifest_name
    )
    with as_file(manifest_res) as manifest_path:
        manifest = load_counter_manifest(manifest_path)
    groups, missing = select_counter_groups(
        manifest, parse_available_counters(avail_res.stdout)
    )
    if missing or not groups:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.AMD_ROCM,
            note=f"missing required counters: {missing}",
        )
    with tempfile.TemporaryDirectory(prefix="mvp-rocprof-") as tmp_dir:
        tmp = Path(tmp_dir)
        job_yaml = tmp / "job.yaml"
        out_dir = tmp / "out"
        write_counter_job(job_yaml, groups, output_directory=str(out_dir))
        cmd = build_rocprofv3_counter_command(
            replay_command, input_path=job_yaml, executable=str(rocprof)
        )
        run_res = run_in_process_group_bounded(
            cmd, timeout=_TIMEOUT_SECONDS, max_capture_bytes=_MAX_OUTPUT_BYTES
        )
        if run_res.returncode != 0:
            return CounterCaptureSummary(
                status="failed",
                backend=VendorBackend.AMD_ROCM,
                note=f"rocprofv3 exited {run_res.returncode}",
            )
        csvs = sorted(out_dir.rglob("*counter_collection*.csv")) or sorted(
            out_dir.rglob("*.csv")
        )
        return parse_rocprof_mvp_passes(csvs, workload_uuid=workload_id)


def _collect_nvidia_counters(
    replay_command: Sequence[str],
) -> CounterCaptureSummary:
    ncu = resolve_tool_path("ncu")
    if ncu is None:
        return CounterCaptureSummary(
            status="unavailable",
            backend=VendorBackend.NVIDIA_CUDA,
            note="ncu executable not found on PATH",
        )
    cmd = [
        str(ncu),
        "--csv",
        "--target-processes",
        "all",
        "--launch-skip",
        "1",
        "--launch-count",
        "2",
        "--metrics",
        _NCU_METRICS,
        *replay_command,
    ]
    res = run_in_process_group_bounded(
        cmd, timeout=_TIMEOUT_SECONDS, max_capture_bytes=_MAX_OUTPUT_BYTES * 4
    )
    if res.returncode != 0:
        perm_denied = "ERR_NVGPUCTRPERM" in (res.stderr + res.stdout)
        note = (
            "ncu permission denied (ERR_NVGPUCTRPERM)"
            if perm_denied
            else f"ncu exited {res.returncode}"
        )
        return CounterCaptureSummary(
            status="unavailable" if perm_denied else "failed",
            backend=VendorBackend.NVIDIA_CUDA,
            note=note,
        )
    return parse_ncu_csv_metrics(res.stdout)


def collect_hardware_counters(
    replay_command: Sequence[str],
    *,
    backend: VendorBackend,
    architecture: str,
    workload_id: str,
) -> CounterCaptureSummary:
    """Collect hardware counters via rocprofv3 (ROCm) or ncu (CUDA)."""
    if backend is VendorBackend.CPU:
        return CounterCaptureSummary(
            status="skipped_cpu",
            backend=backend,
            note="counter collection skipped on CPU",
        )
    if backend is VendorBackend.AMD_ROCM:
        return _collect_rocm_counters(
            replay_command, architecture=architecture, workload_id=workload_id
        )
    return _collect_nvidia_counters(replay_command)
