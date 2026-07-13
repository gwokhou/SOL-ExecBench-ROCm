# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Per-estimate roofline math shared by AMD SOL v3 fusion groups."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.amd_sol.models import AmdSolCoverageSummary
from sol_execbench.core.scoring.confidence import EstimateConfidence


_FLOPS_PER_TFLOP = 1_000_000_000_000.0
_BYTES_PER_GIGABYTE = 1_000_000_000.0
_MILLISECONDS_PER_SECOND = 1000.0


def memory_transfer_bound_ms(byte_count: float, bandwidth_gb_s: float) -> float:
    """Return transfer time in ms for a hardware profile measured in GB/s."""
    return (
        byte_count / (bandwidth_gb_s * _BYTES_PER_GIGABYTE) * _MILLISECONDS_PER_SECOND
    )


def bound_for_estimate(
    estimate: OperatorWorkEstimate, hardware_model: AmdHardwareModel
) -> tuple[float, float, EstimateConfidence, tuple[str, ...]]:
    """Return compute/memory bounds and exact-profile eligibility for one node."""
    compute_profile = (
        hardware_model.resolve_compute(
            estimate.compute_operation or "",
            estimate.input_dtype or "",
            estimate.output_dtype or "",
            estimate.compute_path or "",
        )
        if estimate.compute_operation
        else None
    )
    memory_profile = (
        hardware_model.resolve_memory(
            estimate.memory_access or "",
            estimate.input_dtype or "",
            estimate.output_dtype or "",
            estimate.memory_path or "",
        )
        if estimate.memory_access
        else None
    )
    compute_required = estimate.flops > 0.0
    memory_required = estimate.total_bytes > 0.0
    usable_compute = not compute_required or _measured(compute_profile)
    usable_memory = not memory_required or _measured(memory_profile)
    compute_bound_ms = (
        estimate.flops
        / (_profile_value(compute_profile) * _FLOPS_PER_TFLOP)
        * _MILLISECONDS_PER_SECOND
        if compute_required and usable_compute
        else 0.0
    )
    memory_bound_ms = (
        memory_transfer_bound_ms(estimate.total_bytes, _profile_value(memory_profile))
        if memory_required and usable_memory
        else 0.0
    )
    confidence = estimate.confidence
    warnings = list(estimate.warnings)
    if not usable_compute or not usable_memory:
        confidence = worse_confidence(confidence, EstimateConfidence.INEXACT)
        warnings.append("unknown_hardware_profile")
    for profile in (compute_profile, memory_profile):
        if profile is not None:
            confidence = worse_confidence(confidence, profile.confidence)
    return compute_bound_ms, memory_bound_ms, confidence, tuple(dict.fromkeys(warnings))


def coverage_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> AmdSolCoverageSummary:
    op_family_counts: dict[str, int] = {}
    confidence_counts: dict[str, dict[str, int]] = {}
    worst = (
        EstimateConfidence.SUPPORTED if estimates else EstimateConfidence.UNSUPPORTED
    )
    for estimate in estimates:
        family = estimate.op_family.value
        confidence = estimate.confidence.value
        op_family_counts[family] = op_family_counts.get(family, 0) + 1
        counts = confidence_counts.setdefault(
            family,
            {"supported": 0, "inexact": 0, "unsupported": 0},
        )
        counts[confidence] += 1
        worst = worse_confidence(worst, estimate.confidence)
    return AmdSolCoverageSummary(
        total_ops=len(estimates),
        supported_ops=sum(
            estimate.confidence == EstimateConfidence.SUPPORTED
            for estimate in estimates
        ),
        inexact_ops=sum(
            estimate.confidence == EstimateConfidence.INEXACT for estimate in estimates
        ),
        unsupported_ops=sum(
            estimate.confidence == EstimateConfidence.UNSUPPORTED
            for estimate in estimates
        ),
        op_family_counts=op_family_counts,
        confidence_counts_by_family=confidence_counts,
        worst_confidence=worst,
    )


def worse_confidence(
    left: EstimateConfidence, right: EstimateConfidence
) -> EstimateConfidence:
    rank = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return left if rank[left] >= rank[right] else right


def _measured(profile: object) -> bool:
    return (
        profile is not None
        and getattr(profile, "state", None) == "measured"
        and getattr(profile, "value", None) is not None
    )


def _profile_value(profile: object) -> float:
    value = getattr(profile, "value", None)
    assert isinstance(value, int | float)
    return float(value)
