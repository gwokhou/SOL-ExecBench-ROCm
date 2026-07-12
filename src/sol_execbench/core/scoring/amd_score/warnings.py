# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Warning and evidence-reference policy for AMD-native score reports."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_sol import AmdSolBoundArtifact
from sol_execbench.core.scoring.amd_hardware_models import HardwareValidationStatus
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation import (
    SolarAggregateStatus,
    SolarDerivationEvidence,
)
from sol_execbench.core.text_utils import ordered_unique


SolarScoreGuard = SolarAggregateStatus | SolarDerivationEvidence
UNSUPPORTED_EVIDENCE_WARNING = (
    "AMD-native score evidence contains unsupported operations; do not present "
    "the score as complete hardware-performance validation."
)
INCOMPLETE_EVIDENCE_WARNING = (
    "AMD-native score was not computed because measured timing, baseline timing, "
    "or AMD SOL bound evidence is incomplete."
)
REFERENCE_BASELINE_WARNING = (
    "AMD-native score used PyTorch reference latency as a provisional baseline; "
    "provide a scoring baseline artifact for release-defined scoring."
)
UNVALIDATED_HARDWARE_WARNING = (
    "AMD hardware model is not validated; score is provisional derived evidence."
)
CDNA3_NO_VALIDATION_WARNING = (
    "CDNA3 full-suite validation has not been recorded for this ROCm port; do "
    "not present this report as a CDNA3 hardware-validation claim."
)
UNSCORED_SOL_BOUND_WARNING = (
    "AMD-native score was not computed because AMD SOL bound evidence is marked "
    "unscored."
)
DEGRADED_SOL_BOUND_WARNING = (
    "AMD-native score uses degraded AMD SOL bound evidence; treat the score as "
    "provisional derived evidence."
)


def warnings_for_artifact(
    artifact: AmdSolBoundArtifact,
) -> list[str]:
    """Build warnings implied by an AMD SOL bound artifact."""
    warnings = list(artifact.warnings)
    for bound in artifact.group_bounds:
        if bound.confidence != EstimateConfidence.SUPPORTED:
            warnings.append(f"fusion_group_inexact:{bound.group_id}")
    if any("unsupported_operator:" in warning for warning in artifact.warnings):
        warnings.append(UNSUPPORTED_EVIDENCE_WARNING)
    if (
        artifact.hardware_model.hardware_validation_status
        != HardwareValidationStatus.VALIDATED
        or artifact.hardware_model.model_validation_status
        != HardwareValidationStatus.VALIDATED
    ):
        warnings.append(UNVALIDATED_HARDWARE_WARNING)
    if artifact.aggregate_bound.status == "degraded":
        warnings.append(DEGRADED_SOL_BOUND_WARNING)
    elif artifact.aggregate_bound.status == "unscored":
        warnings.append(UNSCORED_SOL_BOUND_WARNING)
    if artifact.hardware_model.architecture.startswith("gfx94"):
        warnings.append(CDNA3_NO_VALIDATION_WARNING)
    return unique(warnings)


def solar_aggregate_status(
    solar_derivation: SolarScoreGuard | None,
) -> SolarAggregateStatus | None:
    """Normalize supported solar derivation inputs to an aggregate status."""
    if solar_derivation is None:
        return None
    if isinstance(solar_derivation, SolarAggregateStatus):
        return solar_derivation
    payload = solar_derivation.to_dict()["aggregate_status"]
    return SolarAggregateStatus(
        status=str(payload["status"]),
        score_eligible=bool(payload["score_eligible"]),
        reason=str(payload["reason"]),
        group_ids=tuple(str(group_id) for group_id in payload["group_ids"]),
        node_ids=tuple(str(node_id) for node_id in payload["node_ids"]),
        warnings=tuple(str(warning) for warning in payload["warnings"]),
    )


def warnings_for_solar_aggregate(
    aggregate_status: SolarAggregateStatus,
) -> list[str]:
    """Build warnings implied by solar aggregate status."""
    warnings = list(aggregate_status.warnings)
    if aggregate_status.status == "degraded":
        warnings.append(DEGRADED_SOL_BOUND_WARNING)
    elif aggregate_status.status == "unscored":
        warnings.append(UNSCORED_SOL_BOUND_WARNING)
    return warnings


def unique(values: list[str]) -> list[str]:
    """Return values with stable de-duplication."""
    return ordered_unique(values)


def evidence_refs(
    *,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None,
    sol_bound_ref: str | None,
    baseline_ref: str | None = None,
    hardware_model_ref: str | None = None,
) -> dict[str, str]:
    """Build non-empty evidence references keyed by reference kind."""
    refs: dict[str, str] = {}
    if trace_ref:
        refs["trace"] = trace_ref
    if timing_evidence_ref:
        refs["timing"] = timing_evidence_ref
    if sol_bound_ref:
        refs["sol_bound"] = sol_bound_ref
    if baseline_ref:
        refs["baseline"] = baseline_ref
    if hardware_model_ref:
        refs["hardware_model"] = hardware_model_ref
    return refs
