# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hardware anti-distortion primitives: LLC rotating flush, anti-DCE, and dVFS."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.mvp_v1.analytical_roofline import (
    EmpiricalCeiling,
    HardwareCeiling,
    VendorBackend,
)
from sol_execbench.core.bench.clock_lock import (
    observe_clock_levels,
    set_peak_clocks,
    unlock_clocks,
)
from sol_execbench.core.bench.reward_hack import (
    RewardHackError,
    check_lazy_outputs,
)
from sol_execbench.core.bench.timing_isolation import (
    detect_concurrent_gpu_processes,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_MIB = 1024 * 1024
_DEFAULT_DATACENTER_FLUSH_MIN_BYTES = 256 * _MIB
_CPU_SMOKE_FLUSH_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEnvironmentProbe:
    """Observed GPU partition topology and container/VM virtualization mode."""

    observed_device_name: str
    visible_compute_units: int | None = None
    expected_compute_units: int | None = None
    total_vram_gib: float | None = None
    virtualization_mode: str = "unknown"
    partition_warning: str | None = None


def _detect_virtualization_mode() -> str:
    if list(Path("/sys/class/drm").glob("renderD*/device/physfn")):
        return "sriov_vf"
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return "container"
    dmi_markers = ("kvm", "qemu", "virtual", "openstack", "vmware", "xen")
    for probe_path in (
        Path("/sys/class/dmi/id/product_name"),
        Path("/sys/class/dmi/id/sys_vendor"),
    ):
        if probe_path.is_file():
            text = probe_path.read_text(
                encoding="utf-8", errors="ignore"
            ).lower()
            if any(marker in text for marker in dmi_markers):
                return "vm_guest"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if (
                line.lower().startswith("flags")
                and "hypervisor" in line.split()
            ):
                return "vm_guest"
    return "bare_metal"


def probe_runtime_environment(
    device: str,
    ceiling: HardwareCeiling,
) -> RuntimeEnvironmentProbe:
    """Capture visible CUs/SMs, VRAM, and virtualization posture."""
    if device == "cpu" or not torch.cuda.is_available():
        return RuntimeEnvironmentProbe(
            observed_device_name="cpu",
            expected_compute_units=ceiling.expected_compute_units,
            virtualization_mode="cpu",
        )
    idx = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    visible_cus = int(getattr(props, "multi_processor_count", 0) or 0) or None
    total_vram_gib = round(int(props.total_memory) / (1024**3), 2)
    expected = ceiling.expected_compute_units
    warning = (
        f"visible_cus={visible_cus} < expected={expected} "
        "(possible CPX/vGPU slice)"
        if visible_cus is not None
        and expected is not None
        and visible_cus < expected
        else None
    )
    return RuntimeEnvironmentProbe(
        observed_device_name=str(props.name),
        visible_compute_units=visible_cus,
        expected_compute_units=expected,
        total_vram_gib=total_vram_gib,
        virtualization_mode=_detect_virtualization_mode(),
        partition_warning=warning,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TelemetrySample:
    """Point-in-time GPU clock, power, temperature, and isolation snapshot."""

    vendor: VendorBackend
    perf_level: str | None = None
    sclk_mhz: float | None = None
    mclk_mhz: float | None = None
    power_w: float | None = None
    temp_c: float | None = None
    concurrent_processes: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class DVFSWindowReceipt:
    """Lock attempt outcome plus pre/post GPU telemetry."""

    lock_requested: bool
    lock_acquired: bool
    lock_note: str
    before: TelemetrySample
    after: TelemetrySample | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedTimingStats:
    """Latency summary with LLC flush and rotating-pool provenance."""

    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    samples_ms: tuple[float, ...]
    flush_bytes: int
    ring_replicas: int


def recommended_llc_flush_bytes(
    llc_bytes: int,
    *,
    device: str,
    free_memory_bytes: int | None = None,
) -> int:
    """Compute physical eviction buffer size >= 2x LLC (512 MiB on MI300X)."""
    if device == "cpu" or not torch.cuda.is_available():
        return _CPU_SMOKE_FLUSH_BYTES
    target = max(2 * max(llc_bytes, 1), _DEFAULT_DATACENTER_FLUSH_MIN_BYTES)
    if free_memory_bytes is not None and free_memory_bytes > 0:
        cap = max(_MIB, (free_memory_bytes // 5 // 4096) * 4096)
        target = min(target, cap)
    return max(4096, (target // 4) * 4)


def _device_free_bytes(device: str) -> int | None:
    if device == "cpu" or not torch.cuda.is_available():
        return None
    free_bytes, _total = torch.cuda.mem_get_info(torch.device(device))
    return int(free_bytes)


class LLCRotatingFlusher:
    """Allocate a >=2x LLC buffer and touch all lines via read-modify-write."""

    def __init__(self, device: str, flush_bytes: int) -> None:
        """Allocate an int32 eviction ring on *device*."""
        self.device = device
        self.flush_bytes = max(4096, (flush_bytes // 4) * 4)
        self._buf = torch.ones(
            self.flush_bytes // 4,
            dtype=torch.int32,
            device=device,
        )

    def flush(self) -> None:
        """Evict L2/MALL lines using in-place add to defeat zero-fill bypass."""
        self._buf.add_(1)


class RotatingBufferPool:
    """Physically disjoint tensor replicas preventing cross-iter cache hits."""

    def __init__(
        self,
        prototype_args: list[Any],
        *,
        min_ring_bytes: int,
        max_replicas: int = 8,
    ) -> None:
        """Clone non-overlapping input replicas up to *max_replicas*."""
        tensor_bytes = sum(
            arg.numel() * arg.element_size()
            for arg in prototype_args
            if isinstance(arg, torch.Tensor)
        )
        replicas = (
            max(2, min(max_replicas, math.ceil(min_ring_bytes / tensor_bytes)))
            if tensor_bytes > 0
            else 1
        )
        self._rings: list[list[Any]] = [
            [
                arg.detach().clone() if isinstance(arg, torch.Tensor) else arg
                for arg in prototype_args
            ]
            for _ in range(replicas)
        ]
        self._idx = 0

    @property
    def ring_replicas(self) -> int:
        """Number of disjoint input replicas in the rotation ring."""
        return len(self._rings)

    def next_args(self) -> list[Any]:
        """Return the next physically separate input set."""
        args = self._rings[self._idx % len(self._rings)]
        self._idx += 1
        return args


def _parse_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "value" in value:
        return _parse_float(value["value"])
    if isinstance(value, str):
        token = value.strip().split(" ", maxsplit=1)[0]
        try:
            return float(token)
        except ValueError:
            return None
    return None


def resolve_bundled_or_rocm_tool(tool: str) -> str:
    """Prefer active virtualenv ``_rocm_sdk_core/bin`` tool before host ROCm."""
    for match in sorted(
        Path(sys.prefix).glob(
            f"lib/python*/site-packages/_rocm_sdk_core/bin/{tool}"
        )
    ):
        if match.is_file():
            return str(match)
    return resolve_rocm_tool_command(tool)


def _amd_smi_candidates() -> list[str]:
    candidates = [
        resolve_rocm_tool_command("amd-smi"),
        resolve_bundled_or_rocm_tool("amd-smi"),
    ]
    return list(dict.fromkeys(candidates))


def _sample_amd_telemetry(device: str) -> TelemetrySample:
    idx = str(torch.device(device).index or 0) if device != "cpu" else "0"
    perf_levels = observe_clock_levels()
    perf_level = perf_levels[0] if perf_levels else None
    concurrent = len(detect_concurrent_gpu_processes())
    sclk = mclk = power = temp = None
    for tool in _amd_smi_candidates():
        try:
            res = run_in_process_group_bounded(
                [tool, "metric", "--gpu", idx, "-c", "-p", "-t", "--json"],
                timeout=10,
                max_capture_bytes=16384,
            )
        except OSError:
            continue
        if res.returncode == 0 and res.stdout.strip():
            sclk, mclk, power, temp = _extract_amd_metrics(res.stdout)
            if any(v is not None for v in (sclk, mclk, power, temp)):
                break
    return TelemetrySample(
        vendor=VendorBackend.AMD_ROCM,
        perf_level=perf_level,
        sclk_mhz=sclk,
        mclk_mhz=mclk,
        power_w=power,
        temp_c=temp,
        concurrent_processes=concurrent,
    )


def _extract_amd_metrics(
    raw_json: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return None, None, None, None
    if isinstance(payload, dict) and isinstance(payload.get("gpu_data"), list):
        gpu_data = payload["gpu_data"]
        entry = gpu_data[0] if gpu_data else {}
    else:
        entry = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(entry, dict):
        return None, None, None, None
    clock = entry.get("clock") or {}
    power = entry.get("power") or {}
    temp = entry.get("temperature") or {}
    sclk = _parse_float((clock.get("gfx_0") or {}).get("clk"))
    mclk = _parse_float((clock.get("mem_0") or {}).get("clk"))
    power_w = _parse_float(
        power.get("socket_power") or power.get("average_socket_power")
    )
    temp_c = _parse_float(temp.get("hotspot") or temp.get("edge"))
    return sclk, mclk, power_w, temp_c


def sample_gpu_telemetry(
    device: str,
    vendor: VendorBackend,
) -> TelemetrySample:
    """Safely capture live ROCm telemetry; return empty metrics on CPU/errors."""
    if vendor is VendorBackend.CPU or device == "cpu":
        return TelemetrySample(vendor=VendorBackend.CPU)
    try:
        return _sample_amd_telemetry(device)
    except OSError:
        return TelemetrySample(vendor=vendor)


@contextmanager
def acquire_dvfs_telemetry_window(
    device: str,
    vendor: VendorBackend,
    *,
    lock_clocks: bool,
) -> Iterator[dict[str, DVFSWindowReceipt]]:
    """Attempt ROCm clock locking with graceful degradation and telemetry."""
    before = sample_gpu_telemetry(device, vendor)
    acquired = False
    note = "telemetry_only"
    if lock_clocks and vendor is VendorBackend.AMD_ROCM:
        try:
            set_peak_clocks()
            acquired = True
            note = "amd_stable_peak_locked"
        except Exception as exc:  # noqa: BLE001 -- cloud containers may lack sudo
            note = f"amd_clock_lock_degraded_to_telemetry:{type(exc).__name__}"

    holder = {
        "receipt": DVFSWindowReceipt(
            lock_requested=lock_clocks,
            lock_acquired=acquired,
            lock_note=note,
            before=before,
        )
    }
    try:
        yield holder
    finally:
        if acquired and vendor is VendorBackend.AMD_ROCM:
            unlock_clocks()
        after = sample_gpu_telemetry(device, vendor)
        holder["receipt"] = DVFSWindowReceipt(
            lock_requested=lock_clocks,
            lock_acquired=acquired,
            lock_note=note,
            before=before,
            after=after,
        )


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _flatten_tensors(item)]
    return []


def validate_materialized_outputs(output: Any, device: str) -> None:
    """Reject lazy proxies, host offloading, empty results, or NaN/Inf."""
    tensors = _flatten_tensors(output)
    if not tensors:
        raise RewardHackError("candidate produced no output tensors")
    check_lazy_outputs(tensors)
    expected = torch.device(device)
    for tensor in tensors:
        if tensor.device.type != expected.type or (
            expected.index is not None and tensor.device.index != expected.index
        ):
            raise RewardHackError(
                f"output tensor on {tensor.device}, expected {expected}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("output tensor contains NaN or Inf")


def time_kernel_isolated(
    fn: Callable[..., Any],
    prototype_args: list[Any],
    *,
    device: str,
    llc_bytes: int,
    warmup: int = 5,
    rep: int = 20,
    validate_first: Callable[[Any, list[Any]], None] | None = None,
) -> IsolatedTimingStats:
    """Measure median/p90 kernel latency with LLC flush and anti-DCE guards."""
    free_bytes = _device_free_bytes(device)
    flush_bytes = recommended_llc_flush_bytes(
        llc_bytes,
        device=device,
        free_memory_bytes=free_bytes,
    )
    flusher = LLCRotatingFlusher(device, flush_bytes)
    pool_budget = 64 * 1024 if device == "cpu" else flush_bytes
    pool = RotatingBufferPool(prototype_args, min_ring_bytes=pool_budget)

    first_args = pool.next_args()
    first_out = fn(*first_args)
    validate_materialized_outputs(first_out, device)
    if validate_first is not None:
        validate_first(first_out, first_args)

    for _ in range(warmup):
        flusher.flush()
        out = fn(*pool.next_args())
        validate_materialized_outputs(out, device)

    is_gpu = device != "cpu" and torch.cuda.is_available()
    samples: list[float] = []
    for _ in range(rep):
        args = pool.next_args()
        flusher.flush()
        if is_gpu:
            torch.cuda.synchronize(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            out = fn(*args)
            end_ev.record()
            end_ev.synchronize()
            elapsed_ms = float(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter()
            out = fn(*args)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
        validate_materialized_outputs(out, device)
        samples.append(elapsed_ms)

    ordered = sorted(samples)
    p90_idx = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return IsolatedTimingStats(
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        p90_ms=ordered[p90_idx],
        min_ms=ordered[0],
        samples_ms=tuple(samples),
        flush_bytes=flush_bytes,
        ring_replicas=pool.ring_replicas,
    )


def _measure_hbm_triad_bw(
    device: str,
    llc_bytes: int,
    *,
    smoke: bool,
) -> float:
    span_bytes = (
        1024 * 1024
        if device == "cpu"
        else recommended_llc_flush_bytes(
            llc_bytes,
            device=device,
            free_memory_bytes=_device_free_bytes(device),
        )
    )
    numel = max(4096, span_bytes // 12)
    b = torch.ones(numel, dtype=torch.float32, device=device)
    c = torch.full((numel,), 2.0, dtype=torch.float32, device=device)

    stats = time_kernel_isolated(
        lambda x, y: torch.add(x, y, alpha=2.0),
        [b, c],
        device=device,
        llc_bytes=llc_bytes,
        warmup=2 if smoke else 3,
        rep=4 if smoke else 10,
    )
    return (3.0 * numel * 4.0) / (stats.median_ms * 1e-3)


def _measure_rmsnorm_bw(device: str, llc_bytes: int, *, smoke: bool) -> float:
    if device == "cpu":
        m, n = (512, 1024)
    elif smoke:
        m, n = (8192, 4096)
    else:
        m, n = (16384, 8192)
    x = torch.randn((m, n), dtype=torch.bfloat16, device=device)
    w = torch.ones((n,), dtype=torch.bfloat16, device=device)

    def _run_norm(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            inp, (inp.shape[-1],), weight, eps=1e-6
        )

    stats = time_kernel_isolated(
        _run_norm,
        [x, w],
        device=device,
        llc_bytes=llc_bytes,
        warmup=2 if smoke else 3,
        rep=4 if smoke else 10,
    )
    bytes_moved = (2 * m * n + n) * 2.0
    return bytes_moved / (stats.median_ms * 1e-3)


def _measure_gemm_tflops(
    device: str,
    ceiling: HardwareCeiling,
    *,
    smoke: bool,
    fp8: bool,
) -> float | None:
    if device == "cpu":
        size = 256
    elif smoke:
        size = 2048
    else:
        size = 4096
    try:
        if fp8:
            if device == "cpu":
                return None
            fp8_dt = ceiling.fp8_torch_dtype()
            a = torch.randn(
                (size, size), dtype=torch.bfloat16, device=device
            ).to(fp8_dt)
            b = (
                torch.randn((size, size), dtype=torch.bfloat16, device=device)
                .to(fp8_dt)
                .t()
            )
            sa = torch.ones((), dtype=torch.float32, device=device)
            sb = torch.ones((), dtype=torch.float32, device=device)

            def _fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch._scaled_mm(
                    x, y, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
                )

            stats = time_kernel_isolated(
                _fn,
                [a, b],
                device=device,
                llc_bytes=ceiling.llc_bytes,
                warmup=2 if smoke else 5,
                rep=4 if smoke else 10,
            )
        else:
            a = torch.randn((size, size), dtype=torch.bfloat16, device=device)
            b = torch.randn((size, size), dtype=torch.bfloat16, device=device)
            stats = time_kernel_isolated(
                torch.matmul,
                [a, b],
                device=device,
                llc_bytes=ceiling.llc_bytes,
                warmup=2 if smoke else 5,
                rep=4 if smoke else 10,
            )
    except Exception:  # noqa: BLE001 -- unsupported FP8 probe degrades cleanly
        return None
    return (2.0 * (size**3)) / (stats.median_ms * 1e-3)


def measure_empirical_ceiling(
    device: str,
    ceiling: HardwareCeiling,
    *,
    smoke: bool = False,
) -> EmpiricalCeiling:
    """Run startup ~10s microbench (Triad HBM + RMSNorm + BF16/FP8 GEMM)."""
    hbm_bw = _measure_hbm_triad_bw(device, ceiling.llc_bytes, smoke=smoke)
    rmsnorm_bw = _measure_rmsnorm_bw(device, ceiling.llc_bytes, smoke=smoke)
    bf16_flops = (
        _measure_gemm_tflops(device, ceiling, smoke=smoke, fp8=False) or 1.0
    )
    fp8_flops = _measure_gemm_tflops(device, ceiling, smoke=smoke, fp8=True)
    return EmpiricalCeiling(
        hbm_bandwidth_bytes_per_s=hbm_bw,
        bf16_flops_per_s=bf16_flops,
        fp8_flops_per_s=fp8_flops,
        rmsnorm_bandwidth_bytes_per_s=rmsnorm_bw,
    )
