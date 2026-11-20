# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lightweight ROCm vertical 3-tier analytical Roofline MVP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _install_py312_bootstrap_hook() -> None:
    if (sys.version_info.major, sys.version_info.minor) >= (3, 14):
        return
    compat_path = Path(__file__).with_name("py312_compat.py")
    spec = importlib.util.spec_from_file_location(
        "_mvp_v1_py312_bootstrap", compat_path
    )
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install_py312_pep758_import_hook()


_install_py312_bootstrap_hook()

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


def __getattr__(name: str) -> object:
    if name == "install_py312_pep758_import_hook":
        from scripts.mvp_v1.py312_compat import install_py312_pep758_import_hook

        globals()[name] = install_py312_pep758_import_hook
        return install_py312_pep758_import_hook
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
    "install_py312_pep758_import_hook",
    "load_hardware_ceiling",
]
