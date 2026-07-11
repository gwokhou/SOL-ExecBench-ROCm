# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Single-workload AMD-native scoring."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_sol import AmdSolBoundArtifact
from sol_execbench.core.scoring.amd_sol.v2 import AmdSolBoundV2Artifact
from sol_execbench.core.scoring.amd_score.models import (
    AMD_SCORE_CLAIM_LEVEL,
    AmdNativeScore,
    BoundEligibilityEvidence,
)
from sol_execbench.core.scoring.amd_score.warnings import (
    INCOMPLETE_EVIDENCE_WARNING,
    REFERENCE_BASELINE_WARNING,
    SolarScoreGuard,
    UNSCORED_SOL_BOUND_WARNING,
    evidence_refs,
    solar_aggregate_status,
    unique,
    warnings_for_artifact,
    warnings_for_solar_aggregate,
)
from sol_execbench.core.scoring.solar_derivation.models import SolarDerivationEvidence
from sol_execbench.sol_score import sol_score


def score_amd_native_workload(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    baseline_source: str = "scoring_baseline",
    hardware_model_ref: str | None = None,
    solar_derivation: SolarScoreGuard | None = None,
    derived_evidence_refs: dict[str, str] | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score for one workload."""
    sol_bound_ms = artifact_sol_bound_ms(artifact)
    warnings = warnings_for_artifact(artifact)
    solar_aggregate = solar_aggregate_status(solar_derivation)
    if solar_aggregate is not None:
        warnings.extend(warnings_for_solar_aggregate(solar_aggregate))
        warnings = unique(warnings)
    if baseline_source == "reference_latency":
        warnings.append(REFERENCE_BASELINE_WARNING)
    refs = evidence_refs(
        trace_ref=trace_ref,
        timing_evidence_ref=timing_evidence_ref,
        sol_bound_ref=sol_bound_ref,
        baseline_ref=baseline_ref,
        hardware_model_ref=hardware_model_ref,
    )

    score_value = None
    if solar_aggregate is not None and solar_aggregate.status == "unscored":
        score_value = None
    elif (
        isinstance(artifact, AmdSolBoundV2Artifact)
        and not artifact.aggregate_bound.scored
    ):
        if UNSCORED_SOL_BOUND_WARNING not in warnings:
            warnings.append(UNSCORED_SOL_BOUND_WARNING)
    elif has_complete_numeric_inputs(
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
    ):
        assert measured_latency_ms is not None
        assert baseline_latency_ms is not None
        assert sol_bound_ms is not None
        score_value = sol_score(
            t_k=measured_latency_ms,
            t_b=baseline_latency_ms,
            t_sol=sol_bound_ms,
        )
    else:
        warnings.append(INCOMPLETE_EVIDENCE_WARNING)

    return AmdNativeScore(
        definition=artifact.definition,
        workload_uuid=artifact.workload_uuid,
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
        score=score_value,
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=tuple(warnings),
        baseline_source=baseline_source,
        evidence_refs=refs,
        derived_evidence_refs=dict(derived_evidence_refs or {}),
        bound_eligibility=_bound_eligibility(artifact, solar_aggregate),
    )


def _bound_eligibility(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
    solar_aggregate: SolarScoreGuard | None,
) -> BoundEligibilityEvidence:
    """Capture the exact inputs of the authority gate with the score."""
    if isinstance(artifact, AmdSolBoundV2Artifact):
        amd_sol_status = artifact.aggregate_bound.status
        hardware_profile_state = (
            "unknown"
            if any("hardware_profile" in warning for warning in artifact.warnings)
            else "measured"
            if artifact.coverage_summary.worst_confidence.value == "supported"
            else "unknown"
        )
        artifact_warnings = artifact.warnings
    else:
        # v1 scalar models cannot prove exact-profile eligibility.
        amd_sol_status = "scored"
        hardware_profile_state = "unknown"
        artifact_warnings = tuple(warnings_for_artifact(artifact))
    # SOLAR derivation is optional.  Its absence is not evidence that a present
    # derivation failed, and must not turn an otherwise AMD-authoritative score
    # into an artificial unknown blocker.
    solar_status = (
        solar_aggregate.aggregate_status.status
        if isinstance(solar_aggregate, SolarDerivationEvidence)
        else solar_aggregate.status
        if solar_aggregate is not None
        else "not_requested"
    )
    solar_warnings = solar_aggregate.warnings if solar_aggregate is not None else ()
    hardware = artifact.hardware_model
    return BoundEligibilityEvidence(
        amd_sol_status=amd_sol_status,
        solar_status=solar_status,
        hardware_profile_state=hardware_profile_state,
        hardware_validation_status=hardware.hardware_validation_status.value,
        model_validation_status=hardware.model_validation_status.value,
        warnings=tuple(unique(list(artifact_warnings) + list(solar_warnings))),
    )


def has_complete_numeric_inputs(
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    sol_bound_ms: float | None,
) -> bool:
    """Whether all score inputs are present and positive."""
    return (
        measured_latency_ms is not None
        and measured_latency_ms > 0.0
        and baseline_latency_ms is not None
        and baseline_latency_ms > 0.0
        and sol_bound_ms is not None
        # A zero lower bound is numerically complete diagnostic evidence.  The
        # official gate remains stricter and rejects it as non-authoritative.
        and sol_bound_ms >= 0.0
    )


def artifact_sol_bound_ms(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
) -> float:
    """Return the aggregate SOL bound latency from either artifact schema."""
    if isinstance(artifact, AmdSolBoundV2Artifact):
        return artifact.aggregate_bound.sol_bound_ms
    return artifact.aggregate_sol_bound_ms
