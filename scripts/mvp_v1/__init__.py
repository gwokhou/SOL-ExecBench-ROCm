# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lightweight ROCm vertical 3-tier analytical Roofline MVP."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.mvp_v1.analytical_roofline import (
        AnalyticalBound,
        EmpiricalCeiling,
        HardwareCeiling,
        MVPOperator,
        PrecisionMode,
        VendorBackend,
        bound_fused_moe,
        bound_fused_rmsnorm_rope,
        bound_gqa_paged_decode,
        bound_mla_paged_decode,
        bound_skinny_gemm,
        bound_swiglu_fp8_quant,
        load_hardware_ceiling,
    )

_LAZY_EXPORTS = frozenset(
    {
        "AnalyticalBound",
        "EmpiricalCeiling",
        "HardwareCeiling",
        "MVPOperator",
        "PrecisionMode",
        "VendorBackend",
        "bound_fused_moe",
        "bound_fused_rmsnorm_rope",
        "bound_gqa_paged_decode",
        "bound_mla_paged_decode",
        "bound_skinny_gemm",
        "bound_swiglu_fp8_quant",
        "load_hardware_ceiling",
    }
)


def __getattr__(name: str) -> object:
    """Lazily load analytical Roofline symbols without eager torch imports."""
    if name in _LAZY_EXPORTS:
        module = import_module("scripts.mvp_v1.analytical_roofline")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalyticalBound",
    "EmpiricalCeiling",
    "HardwareCeiling",
    "MVPOperator",
    "PrecisionMode",
    "VendorBackend",
    "bound_fused_moe",
    "bound_fused_rmsnorm_rope",
    "bound_gqa_paged_decode",
    "bound_mla_paged_decode",
    "bound_skinny_gemm",
    "bound_swiglu_fp8_quant",
    "load_hardware_ceiling",
]
