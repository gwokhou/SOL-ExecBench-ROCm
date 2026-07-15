"""Stable names and policy values for official-score evidence."""

from sol_execbench.core.scoring import aggregation


OFFICIAL_AGGREGATION_POLICY = aggregation.OFFICIAL_AGGREGATION_POLICY

OFFICIAL_SCORE_SCHEMA_VERSION = "sol_execbench.official_score_evidence.v1"
OFFICIAL_SCORE_SOURCE = "official_score_evidence"
OFFICIAL_SCORE_KIND = "official_benchmark_score"
OFFICIAL_SCORE_CLAIM_LEVEL = "official-confirmed"
DEFAULT_OFFICIAL_BASELINE_SOURCES = ("scoring_baseline",)

MISSING_SCORE_BLOCKER = "missing_score"
MISSING_MEASURED_LATENCY_BLOCKER = "missing_measured_latency"
MISSING_BASELINE_BLOCKER = "missing_baseline"
PLACEHOLDER_BASELINE_BLOCKER = "placeholder_baseline"
MISSING_SOL_BOUND_BLOCKER = "missing_sol_bound"
MISSING_AGGREGATION_POLICY_BLOCKER = "missing_aggregation_policy"
BASELINE_COVERAGE_FAILED_BLOCKER = "baseline_coverage_failed"
MISSING_BOUND_ELIGIBILITY_BLOCKER = "missing_bound_eligibility"
AMD_SOL_NOT_SCORED_BLOCKER = "amd_sol_not_scored"
SOLAR_NOT_SCORED_BLOCKER = "solar_not_scored"
UNSUPPORTED_HARDWARE_PROFILE_BLOCKER = "unsupported_hardware_profile"
HARDWARE_NOT_VALIDATED_BLOCKER = "hardware_not_validated"
MODEL_NOT_VALIDATED_BLOCKER = "model_not_validated"
BOUND_EVIDENCE_WARNING_BLOCKER = "bound_evidence_warning"
MISSING_EVIDENCE_REFERENCE_BLOCKER = "missing_evidence_reference"
RELEASE_BASELINE_NOT_VERIFIED_BLOCKER = "release_baseline_not_verified"
RELEASE_BOUND_NOT_VERIFIED_BLOCKER = "release_bound_not_verified"
RELEASE_SCOPE_NOT_DECLARED_BLOCKER = "release_scope_not_declared"
BASELINE_NOT_SLOWER_THAN_SOL_BOUND_BLOCKER = "baseline_not_slower_than_sol_bound"
CANDIDATE_BELOW_SOL_BOUND_BLOCKER = "candidate_below_sol_bound"
CANDIDATE_EVIDENCE_NOT_VERIFIED_BLOCKER = "candidate_evidence_not_verified"

PLACEHOLDER_BASELINE_SOURCES = frozenset(
    {
        "reference_latency",
        "trace_reference_latency",
        "trace.evaluation.performance.reference_latency_ms",
    }
)

__all__ = [name for name in globals() if name.isupper()]
