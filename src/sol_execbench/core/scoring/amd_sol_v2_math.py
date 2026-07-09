# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bound, aggregate, coverage, and warning math for AMD SOL v2 artifacts."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    HardwareValidationStatus,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_sol_v2_models import (
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    AmdSolV2OpBound,
)
from sol_execbench.core.text_utils import ordered_unique

def _bound_for_estimate(
    estimate: OperatorWorkEstimate,
    hardware_model: AmdHardwareModel,
) -> AmdSolV2OpBound:
    compute_bound_ms = (
        estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
        if hardware_model.peak_tflops > 0.0
        else 0.0
    )
    memory_bound_ms = (
        estimate.total_bytes
        / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0)
        * 1000.0
        if hardware_model.memory_bandwidth_gbps > 0.0
        else 0.0
    )
    limiting_resource = "compute" if compute_bound_ms >= memory_bound_ms else "memory"
    return AmdSolV2OpBound(
        node_id=estimate.node_id,
        op_family=estimate.op_family.value,
        op_name=estimate.op_name,
        compute_bound_ms=compute_bound_ms,
        memory_bound_ms=memory_bound_ms,
        sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
        limiting_resource=limiting_resource,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
        estimate_warnings=estimate.warnings,
    )


def _aggregate_for_bounds(
    op_bounds: tuple[AmdSolV2OpBound, ...],
    hardware_model: AmdHardwareModel,
) -> AmdSolV2AggregateBound:
    sol_bound_ms = sum(bound.sol_bound_ms for bound in op_bounds)
    node_ids = tuple(bound.node_id for bound in op_bounds)
    if not op_bounds:
        return AmdSolV2AggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="missing operation bound evidence",
            node_ids=node_ids,
        )
    if any(bound.confidence == EstimateConfidence.UNSUPPORTED for bound in op_bounds):
        return AmdSolV2AggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="unsupported operation evidence present",
            node_ids=node_ids,
        )
    if (
        any(bound.confidence == EstimateConfidence.INEXACT for bound in op_bounds)
        or hardware_model.hardware_validation_status
        != HardwareValidationStatus.VALIDATED
        or hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED
        or hardware_model.confidence != EstimateConfidence.SUPPORTED
    ):
        return AmdSolV2AggregateBound(
            status="degraded",
            scored=True,
            sol_bound_ms=sol_bound_ms,
            reason="inexact or provisional evidence present",
            node_ids=node_ids,
        )
    return AmdSolV2AggregateBound(
        status="scored",
        scored=True,
        sol_bound_ms=sol_bound_ms,
        reason="all operation and hardware evidence is supported",
        node_ids=node_ids,
    )


def _coverage_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> AmdSolV2CoverageSummary:
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
            {
                EstimateConfidence.SUPPORTED.value: 0,
                EstimateConfidence.INEXACT.value: 0,
                EstimateConfidence.UNSUPPORTED.value: 0,
            },
        )
        counts[confidence] = counts.get(confidence, 0) + 1
        worst = _worse_confidence(worst, estimate.confidence)

    return AmdSolV2CoverageSummary(
        total_ops=len(estimates),
        supported_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.SUPPORTED
        ),
        inexact_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.INEXACT
        ),
        unsupported_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.UNSUPPORTED
        ),
        op_family_counts=op_family_counts,
        confidence_counts_by_family=confidence_counts,
        worst_confidence=worst,
    )


def _warnings_for_artifact(
    graph_warnings: tuple[str, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    aggregate: AmdSolV2AggregateBound,
    hardware_model: AmdHardwareModel,
) -> tuple[str, ...]:
    warnings: list[str] = []
    for warning in graph_warnings:
        warnings.append(f"graph_warning:{warning}")
    for estimate in estimates:
        for warning in estimate.warnings:
            warnings.append(f"estimate_warning:{estimate.node_id}:{warning}")
        if estimate.confidence == EstimateConfidence.INEXACT:
            warnings.append(
                f"inexact_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
        elif estimate.confidence == EstimateConfidence.UNSUPPORTED:
            warnings.append(
                f"unsupported_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
    if hardware_model.hardware_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "hardware_validation:"
            f"{hardware_model.architecture}:{hardware_model.hardware_validation_status.value}"
        )
    if hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "model_validation:"
            f"{hardware_model.architecture}:{hardware_model.model_validation_status.value}"
        )
    if aggregate.status == "degraded":
        warnings.append(f"aggregate_degraded:{aggregate.reason}")
    elif aggregate.status == "unscored":
        warnings.append(f"aggregate_unscored:{aggregate.reason}")
    return _unique(warnings)


def _worse_confidence(
    left: EstimateConfidence,
    right: EstimateConfidence,
) -> EstimateConfidence:
    rank = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return left if rank[left] >= rank[right] else right

def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(ordered_unique(values))
