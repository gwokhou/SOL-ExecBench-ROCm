# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Transparent analytical Roofline bounds without Orojenesis or custom schemas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch

from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from solar.rocm.architecture import ArchitectureProfile


class MVPOperator(StrEnum):
    """Phase 1 3-operator cross-vendor MVP vocabulary."""

    SKINNY_GEMM = "skinny_gemm"
    FUSED_MOE = "fused_moe"
    GQA_PAGED_DECODE = "gqa_paged_decode"


class PrecisionMode(StrEnum):
    """Supported compute precisions in the Phase 1 MVP."""

    BF16 = "bf16"
    FP8 = "fp8"


class VendorBackend(StrEnum):
    """Execution backend family for timing, dVFS telemetry, and counters."""

    AMD_ROCM = "amd_rocm"
    NVIDIA_CUDA = "nvidia_cuda"
    CPU = "cpu"


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareCeiling:
    """Normalized compute, HBM, and LLC parameters for analytical Roofline."""

    target_id: str
    device_name: str
    vendor: VendorBackend
    architecture: str
    hbm_bandwidth_bytes_per_s: float
    peak_bf16_flops_per_s: float
    peak_fp8_flops_per_s: float
    l2_bytes: int
    llc_bytes: int
    fp8_dtype_name: str

    def peak_flops(self, precision: PrecisionMode) -> float:
        """Return peak FLOP/s for the requested compute precision."""
        return (
            self.peak_fp8_flops_per_s
            if precision is PrecisionMode.FP8
            else self.peak_bf16_flops_per_s
        )

    def fp8_torch_dtype(self) -> torch.dtype:
        """Return the concrete PyTorch FP8 tensor dtype for this target."""
        return dtype_str_to_torch_dtype(self.fp8_dtype_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class EmpiricalCeiling:
    """Startup microbenchmark ceiling measured on the target device."""

    hbm_bandwidth_bytes_per_s: float
    bf16_flops_per_s: float
    fp8_flops_per_s: float | None = None
    rmsnorm_bandwidth_bytes_per_s: float | None = None

    def peak_flops(self, precision: PrecisionMode) -> float | None:
        """Return empirical FLOP/s for *precision* when measured."""
        if precision is PrecisionMode.FP8:
            return self.fp8_flops_per_s
        return self.bf16_flops_per_s


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyticalBound:
    """Transparent Spec + Empirical Roofline bound for one workload."""

    operator: MVPOperator
    precision: PrecisionMode
    workload_id: str
    flops: float
    compulsory_bytes: float
    arithmetic_intensity: float
    ridge_point_spec: float
    t_compute_ms_spec: float
    t_mem_ms_spec: float
    t_sol_ms_spec: float
    bottleneck_spec: str
    t_compute_ms_empirical: float | None = None
    t_mem_ms_empirical: float | None = None
    t_sol_ms_empirical: float | None = None
    bottleneck_empirical: str | None = None


_NVIDIA_DATASHEET_CEILINGS: dict[str, HardwareCeiling] = {
    "H100_SXM": HardwareCeiling(
        target_id="H100_SXM",
        device_name="NVIDIA H100 SXM5 80GB",
        vendor=VendorBackend.NVIDIA_CUDA,
        architecture="sm_90",
        hbm_bandwidth_bytes_per_s=3_350_000_000_000.0,
        peak_bf16_flops_per_s=989_500_000_000_000.0,
        peak_fp8_flops_per_s=1_978_900_000_000_000.0,
        l2_bytes=52_428_800,
        llc_bytes=52_428_800,
        fp8_dtype_name="float8_e4m3fn",
    ),
    "B200": HardwareCeiling(
        target_id="B200",
        device_name="NVIDIA B200",
        vendor=VendorBackend.NVIDIA_CUDA,
        architecture="sm_100",
        hbm_bandwidth_bytes_per_s=8_000_000_000_000.0,
        peak_bf16_flops_per_s=2_250_000_000_000_000.0,
        peak_fp8_flops_per_s=4_500_000_000_000_000.0,
        l2_bytes=134_217_728,
        llc_bytes=134_217_728,
        fp8_dtype_name="float8_e4m3fn",
    ),
}

_TARGET_ALIASES = {
    "MI300X": "MI300X",
    "AMD_INSTINCT_MI300X": "MI300X",
    "GFX942": "MI300X",
    "RX_9060_XT": "RX_9060_XT",
    "RX9060XT": "RX_9060_XT",
    "GFX1200": "RX_9060_XT",
    "H100": "H100_SXM",
    "H100_SXM": "H100_SXM",
    "SM_90": "H100_SXM",
    "B200": "B200",
    "SM_100": "B200",
}


def _rocm_profile_to_ceiling(profile_name: str) -> HardwareCeiling:
    """Project an AMD ``ArchitectureProfile`` without requiring formal audits."""
    profile = ArchitectureProfile.load(profile_name)
    fp8_dtype_name = (
        "float8_e4m3fnuz"
        if "float8_e4m3fnuz" in profile.precision_aliases
        else "float8_e4m3fn"
    )
    return HardwareCeiling(
        target_id=profile_name,
        device_name=profile.name.replace("_", " "),
        vendor=VendorBackend.AMD_ROCM,
        architecture=profile.gfx_target,
        hbm_bandwidth_bytes_per_s=profile.memory_bandwidth_bytes_per_second,
        peak_bf16_flops_per_s=float(profile.peak_ops_per_second["bf16"]),
        peak_fp8_flops_per_s=float(profile.peak_ops_per_second["fp8"]),
        l2_bytes=profile.l2_bytes,
        llc_bytes=profile.last_level_cache_bytes or profile.l2_bytes,
        fp8_dtype_name=fp8_dtype_name,
    )


def load_hardware_ceiling(target_id: str) -> HardwareCeiling:
    """Resolve a target key (MI300X, RX_9060_XT, H100_SXM, B200) to ceilings."""
    normalized = target_id.strip().upper().replace("-", "_").replace(" ", "_")
    canonical = _TARGET_ALIASES.get(normalized, normalized)
    if canonical in {"MI300X", "RX_9060_XT"}:
        return _rocm_profile_to_ceiling(canonical)
    if canonical in _NVIDIA_DATASHEET_CEILINGS:
        return _NVIDIA_DATASHEET_CEILINGS[canonical]
    supported = sorted({"MI300X", "RX_9060_XT", *_NVIDIA_DATASHEET_CEILINGS})
    raise ValueError(
        f"Unsupported MVP target {target_id!r}; expected one of {supported}"
    )


def detect_hardware_ceiling(
    device: str,
    target_override: str | None = None,
) -> HardwareCeiling:
    """Infer target hardware ceiling from the live device or explicit override."""
    if target_override is not None:
        ceiling = load_hardware_ceiling(target_override)
        if device == "cpu":
            return HardwareCeiling(
                target_id=ceiling.target_id,
                device_name=f"{ceiling.device_name} (CPU dry-run)",
                vendor=VendorBackend.CPU,
                architecture=ceiling.architecture,
                hbm_bandwidth_bytes_per_s=ceiling.hbm_bandwidth_bytes_per_s,
                peak_bf16_flops_per_s=ceiling.peak_bf16_flops_per_s,
                peak_fp8_flops_per_s=ceiling.peak_fp8_flops_per_s,
                l2_bytes=ceiling.l2_bytes,
                llc_bytes=ceiling.llc_bytes,
                fp8_dtype_name=ceiling.fp8_dtype_name,
            )
        return ceiling
    if device == "cpu" or not torch.cuda.is_available():
        return HardwareCeiling(
            target_id="CPU_REFERENCE",
            device_name="CPU Local Fallback",
            vendor=VendorBackend.CPU,
            architecture="cpu",
            hbm_bandwidth_bytes_per_s=50_000_000_000.0,
            peak_bf16_flops_per_s=1_000_000_000_000.0,
            peak_fp8_flops_per_s=1_000_000_000_000.0,
            l2_bytes=16_777_216,
            llc_bytes=33_554_432,
            fp8_dtype_name="float8_e4m3fn",
        )
    idx = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    gfx = str(getattr(props, "gcnArchName", "")).split(":", maxsplit=1)[0]
    name = str(props.name).upper()
    for token in (gfx, name):
        if "942" in token or "MI300X" in token:
            return load_hardware_ceiling("MI300X")
        if "1200" in token or "9060" in token:
            return load_hardware_ceiling("RX_9060_XT")
        if "B200" in token:
            return load_hardware_ceiling("B200")
        if "H100" in token:
            return load_hardware_ceiling("H100_SXM")
    raise ValueError(
        f"Unrecognized GPU {props.name!r} ({gfx}); pass --target-override"
    )


def _finalize_bound(
    *,
    operator: MVPOperator,
    precision: PrecisionMode,
    workload_id: str,
    flops: float,
    compulsory_bytes: float,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None = None,
) -> AnalyticalBound:
    """Compute Spec and optional Empirical Roofline times from flops and bytes."""
    peak_spec = ceiling.peak_flops(precision)
    bw_spec = ceiling.hbm_bandwidth_bytes_per_s
    t_comp_spec = (flops / peak_spec) * 1e3
    t_mem_spec = (compulsory_bytes / bw_spec) * 1e3
    t_sol_spec = max(t_comp_spec, t_mem_spec)
    bottleneck_spec = "compute" if t_comp_spec >= t_mem_spec else "memory"

    t_comp_emp: float | None = None
    t_mem_emp: float | None = None
    t_sol_emp: float | None = None
    bottleneck_emp: str | None = None
    if empirical is not None:
        t_mem_emp = (
            compulsory_bytes / empirical.hbm_bandwidth_bytes_per_s
        ) * 1e3
        emp_peak = empirical.peak_flops(precision)
        t_comp_emp = (flops / emp_peak) * 1e3 if emp_peak else None
        if t_comp_emp is not None:
            t_sol_emp = max(t_comp_emp, t_mem_emp)
            bottleneck_emp = "compute" if t_comp_emp >= t_mem_emp else "memory"
        else:
            t_sol_emp = t_mem_emp
            bottleneck_emp = "memory"

    return AnalyticalBound(
        operator=operator,
        precision=precision,
        workload_id=workload_id,
        flops=flops,
        compulsory_bytes=compulsory_bytes,
        arithmetic_intensity=flops / max(compulsory_bytes, 1.0),
        ridge_point_spec=peak_spec / bw_spec,
        t_compute_ms_spec=t_comp_spec,
        t_mem_ms_spec=t_mem_spec,
        t_sol_ms_spec=t_sol_spec,
        bottleneck_spec=bottleneck_spec,
        t_compute_ms_empirical=t_comp_emp,
        t_mem_ms_empirical=t_mem_emp,
        t_sol_ms_empirical=t_sol_emp,
        bottleneck_empirical=bottleneck_emp,
    )


def bound_skinny_gemm(
    m: int,
    n: int,
    k: int,
    precision: PrecisionMode,
    ceiling: HardwareCeiling,
    *,
    empirical: EmpiricalCeiling | None = None,
    workload_id: str = "",
) -> AnalyticalBound:
    """Analytical Roofline for ``output[M, N] = x[M, K] @ weight[K, N]``."""
    elem_bytes = 1 if precision is PrecisionMode.FP8 else 2
    flops = 2.0 * m * n * k
    scale_bytes = 8 if precision is PrecisionMode.FP8 else 0
    compulsory_bytes = float(
        (m * k + k * n) * elem_bytes + m * n * 2 + scale_bytes
    )
    return _finalize_bound(
        operator=MVPOperator.SKINNY_GEMM,
        precision=precision,
        workload_id=workload_id
        or f"skinny_gemm_{precision.value}_m{m}_n{n}_k{k}",
        flops=flops,
        compulsory_bytes=compulsory_bytes,
        ceiling=ceiling,
        empirical=empirical,
    )


def bound_fused_moe(
    m: int,
    hidden: int,
    inter: int,
    num_experts: int,
    top_k: int,
    precision: PrecisionMode,
    ceiling: HardwareCeiling,
    *,
    active_experts: float | None = None,
    empirical: EmpiricalCeiling | None = None,
    workload_id: str = "",
) -> AnalyticalBound:
    """Analytical Roofline for fused SwiGLU MoE with deduplicated weights."""
    elem_bytes = 1 if precision is PrecisionMode.FP8 else 2
    eff_experts = (
        float(active_experts)
        if active_experts is not None
        else num_experts
        * (1.0 - ((num_experts - 1.0) / num_experts) ** (m * top_k))
    )
    flops = 6.0 * m * top_k * hidden * inter + 4.0 * m * top_k * inter
    weight_bytes = eff_experts * 3.0 * hidden * inter * elem_bytes
    activation_bytes = (2.0 * m * hidden + m * num_experts) * 2.0
    return _finalize_bound(
        operator=MVPOperator.FUSED_MOE,
        precision=precision,
        workload_id=(
            workload_id
            or f"fused_moe_{precision.value}_m{m}_h{hidden}_i{inter}_e{num_experts}_k{top_k}"
        ),
        flops=flops,
        compulsory_bytes=weight_bytes + activation_bytes,
        ceiling=ceiling,
        empirical=empirical,
    )


def bound_gqa_paged_decode(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    kv_len: int,
    page_size: int,
    precision: PrecisionMode,
    ceiling: HardwareCeiling,
    *,
    empirical: EmpiricalCeiling | None = None,
    workload_id: str = "",
) -> AnalyticalBound:
    """Analytical Roofline for NHD-layout GQA Paged Decode (``S_q = 1``)."""
    elem_bytes = 1 if precision is PrecisionMode.FP8 else 2
    flops = (
        4.0 * batch * kv_len * q_heads * head_dim
        + 5.0 * batch * q_heads * kv_len
    )
    pages_per_seq = math.ceil(kv_len / page_size)
    table_bytes = ((batch + 1) + batch * pages_per_seq + batch) * 4
    q_bytes = batch * q_heads * head_dim * elem_bytes
    kv_bytes = 2 * batch * kv_len * kv_heads * head_dim * elem_bytes
    out_bytes = batch * q_heads * head_dim * 2
    compulsory_bytes = float(q_bytes + kv_bytes + out_bytes + table_bytes)
    return _finalize_bound(
        operator=MVPOperator.GQA_PAGED_DECODE,
        precision=precision,
        workload_id=(
            workload_id
            or f"gqa_paged_decode_{precision.value}_b{batch}_hq{q_heads}_hk{kv_heads}_d{head_dim}_kv{kv_len}"
        ),
        flops=flops,
        compulsory_bytes=compulsory_bytes,
        ceiling=ceiling,
        empirical=empirical,
    )
