# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bound math for AMD SOL v1 artifacts."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.amd_sol_models import OpSolBound, WorkEstimate

def _bound_for_estimate(
    estimate: WorkEstimate,
    hardware_model: AmdHardwareModel,
) -> OpSolBound:
    compute_bound_ms = (
        estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
        if hardware_model.peak_tflops > 0
        else 0.0
    )
    memory_bound_ms = (
        estimate.bytes_accessed
        / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0)
        * 1000.0
        if hardware_model.memory_bandwidth_gbps > 0
        else 0.0
    )
    limiting_resource = "compute" if compute_bound_ms >= memory_bound_ms else "memory"
    return OpSolBound(
        node_id=estimate.node_id,
        compute_bound_ms=compute_bound_ms,
        memory_bound_ms=memory_bound_ms,
        sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
        limiting_resource=limiting_resource,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
    )
