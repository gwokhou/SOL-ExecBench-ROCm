# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared memory and architecture prediction primitives."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameterName,
    DiagnosticCalibrationProfile,
    PredictionComponent,
)
from sol_execbench.core.bench.performance_model.prediction.calibration import (
    _apply_efficiency,
    _memory_parameter,
    _required,
    _scaled_component,
)
from sol_execbench.core.platform.arch_capabilities import (
    derive_arch_capability_budget,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


_DEFAULT_WAVE_SIZE = 32.0


def _wave_size(gpu_architecture: str | None) -> float:
    """Return the architected wavefront size, defaulting to 32 when unknown."""
    budget = derive_arch_capability_budget(gpu_architecture)
    if budget is not None and budget.wavefront_size is not None:
        return float(budget.wavefront_size)
    return _DEFAULT_WAVE_SIZE


def _memory_component(
    amount_bytes: float,
    calibration: DiagnosticCalibrationProfile,
    *,
    working_set_bytes: float | None = None,
    efficiency_name: CalibrationParameterName | None = None,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    parameter = _memory_parameter(
        amount_bytes if working_set_bytes is None else working_set_bytes,
        calibration,
    )
    component = _scaled_component(
        "memory",
        amount_bytes,
        parameter,
        dispatch_id=dispatch_id,
    )
    if efficiency_name is not None:
        component = _apply_efficiency(
            component,
            _required(calibration, efficiency_name),
        )
    return component
