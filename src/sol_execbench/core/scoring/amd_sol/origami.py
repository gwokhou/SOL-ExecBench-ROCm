# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional Origami GEMM diagnostic provider.

Origami predicts latency for a finite candidate-kernel set. That is useful for
auditing GEMM roofline inputs, but it is not a proof of a theoretical lower
bound, so every result here is deliberately diagnostic-only.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrigamiGemmRequest:
    """The fully specified GEMM inputs needed by Origami's analytical model."""

    m: int
    n: int
    k: int
    batch: int
    a_dtype: str
    b_dtype: str
    c_dtype: str
    d_dtype: str
    a_transpose: bool = False
    b_transpose: bool = False
    device: int = 0

    def __post_init__(self) -> None:
        if any(value <= 0 for value in (self.m, self.n, self.k, self.batch)):
            raise ValueError("Origami GEMM dimensions and batch must be positive")


@dataclass(frozen=True)
class OrigamiGemmEstimate:
    """Auditable prediction returned by the optional Origami provider."""

    status: str
    predicted_latency_ms: float | None
    provider: str = "origami"
    provider_version: str | None = None
    reason_code: str | None = None
    selected_config: dict[str, int] | None = None
    is_theoretical_lower_bound: bool = False


class OrigamiGemmProvider:
    """Thin lazy adapter around Origami's public Python bindings."""

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    def estimate(
        self, request: OrigamiGemmRequest, *, configs: tuple[object, ...]
    ) -> OrigamiGemmEstimate:
        if not configs:
            return OrigamiGemmEstimate(
                status="unavailable",
                predicted_latency_ms=None,
                reason_code="missing_origami_candidate_configs",
            )
        module = self._resolve_module()
        if module is None:
            return OrigamiGemmEstimate(
                status="unavailable",
                predicted_latency_ms=None,
                reason_code="origami_not_installed",
            )
        try:
            hardware = module.get_hardware_for_device(request.device)
            problem = module.problem_t()
            problem.size = module.dim3_t(request.m, request.n, request.k)
            problem.batch = request.batch
            problem.a_transpose = (
                module.transpose_t.T if request.a_transpose else module.transpose_t.N
            )
            problem.b_transpose = (
                module.transpose_t.T if request.b_transpose else module.transpose_t.N
            )
            for field, dtype in (
                ("a_dtype", request.a_dtype),
                ("b_dtype", request.b_dtype),
                ("c_dtype", request.c_dtype),
                ("d_dtype", request.d_dtype),
            ):
                setattr(problem, field, module.string_to_datatype(dtype))
            problem.mi_dtype = module.string_to_datatype(request.a_dtype)
            problem.a_mx_block_size = 0
            problem.b_mx_block_size = 0
            result = module.select_config(problem, hardware, list(configs))
            latency = float(result.latency)
            if not math.isfinite(latency) or latency <= 0.0:
                raise ValueError("Origami returned an invalid latency")
            return OrigamiGemmEstimate(
                status="diagnostic",
                predicted_latency_ms=latency,
                provider_version=getattr(module, "__version__", None),
                selected_config=_config_summary(getattr(result, "config", None)),
            )
        except Exception:
            return OrigamiGemmEstimate(
                status="unavailable",
                predicted_latency_ms=None,
                provider_version=getattr(module, "__version__", None),
                reason_code="origami_prediction_failed",
            )

    def _resolve_module(self) -> Any | None:
        if self._module is not None:
            return self._module
        try:
            return importlib.import_module("origami")
        except ImportError:
            return None


def _config_summary(config: object | None) -> dict[str, int] | None:
    if config is None:
        return None
    mt = getattr(config, "mt", None)
    mi = getattr(config, "mi", None)
    if mt is None or mi is None:
        return None
    try:
        return {
            "mt_m": int(mt.m),
            "mt_n": int(mt.n),
            "mt_k": int(mt.k),
            "mi_m": int(mi.m),
            "mi_n": int(mi.n),
            "mi_k": int(mi.k),
            "occupancy": int(getattr(config, "occupancy")),
        }
    except (AttributeError, TypeError, ValueError):
        return None
