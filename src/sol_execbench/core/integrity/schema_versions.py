# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current first-party schema identifiers.

Each artifact family has exactly one current wire contract. Historical schema
identifiers intentionally do not live in this module or anywhere else in the
working tree; Git history is the only archive for superseded contracts.
"""

from typing import Final, Literal, TypeAlias

AGENT_FEEDBACK_SCHEMA_VERSION: Final = "sol_execbench.agent_feedback.v3"
DECISION_SCHEMA_VERSION: Final = "sol_execbench.decision.v2"
PROFILE_SUMMARY_SCHEMA_VERSION: Final = "sol_execbench.profile_summary.v3"
ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION: Final = (
    "sol_execbench.environment_diagnostics.v1"
)
ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION: Final = "sol_execbench.environment_snapshot.v2"
EVALUATOR_CONTRACT_SCHEMA_VERSION: Final = "sol_execbench.evaluator_contract.v3"
ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION: Final = (
    "sol_execbench.rocm_compatibility_matrix.v1"
)
CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION: Final = (
    "sol_execbench.corpus_stage_readiness_record.v1"
)
CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION: Final = (
    "sol_execbench.corpus_stage_readiness_summary.v1"
)
RDNA4_VALIDATION_SCHEMA_VERSION: Final = "sol_execbench.rdna4_validation.v2"
RELEASE_BASELINE_SCHEMA_VERSION: Final = "sol_execbench.release_baseline.v1"
RELEASE_BUNDLE_SCHEMA_VERSION: Final = "sol_execbench.release_bundle.v1"
RELEASE_CANDIDATE_SCHEMA_VERSION: Final = "sol_execbench.release_candidate.v1"
RELEASE_ENVIRONMENT_SCHEMA_VERSION: Final = "sol_execbench.release_environment.v1"
RELEASE_EXECUTION_PLAN_SCHEMA_VERSION: Final = "sol_execbench.release_execution_plan.v1"
RELEASE_RERUN_SCHEMA_VERSION: Final = "sol_execbench.release_rerun.v1"
RELEASE_SOLAR_INDEX_SCHEMA_VERSION: Final = "sol_execbench.release_solar_index.v1"
ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_overhead_calibration.v2"
)
STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.static_artifact_manifest.v1"
)
STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION: Final = "sol_execbench.static_kernel_evidence.v3"
TOOLCHAIN_ROUTING_SCHEMA_VERSION: Final = "sol_execbench.toolchain_routing.v1"

SCHEMA_VERSIONS: Final[dict[str, str]] = {
    "agent_feedback": AGENT_FEEDBACK_SCHEMA_VERSION,
    "decision": DECISION_SCHEMA_VERSION,
    "profile_summary": PROFILE_SUMMARY_SCHEMA_VERSION,
    "amd_isa_release_lock": "sol_execbench.amd_isa_release_lock.v1",
    "arch_capability_budget": "sol_execbench.arch_capability_budget.v1",
    "cli_contract": "sol_execbench.cli_contract.v1",
    "cli_response": "sol_execbench.cli_response.v1",
    "corpus_stage_readiness_record": CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION,
    "corpus_stage_readiness_summary": CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION,
    "dataset_provenance_policy": "sol_execbench.dataset_provenance_policy.v1",
    "dataset_redistribution_check": "sol_execbench.dataset_redistribution_check.v1",
    "derived_evidence": "sol_execbench.derived_evidence.v1",
    "environment_diagnostics": ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION,
    "environment_snapshot": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
    "evaluator_contract": EVALUATOR_CONTRACT_SCHEMA_VERSION,
    "gpu_device_isolation": "sol_execbench.gpu_device_isolation.v1",
    "no_trace_diagnostics": "sol_execbench.no_trace_diagnostics.v1",
    "official_score_availability": "sol_execbench.official_score_availability.v1",
    "pid_lock_contention": "sol_execbench.pid_lock_contention.v1",
    "reference_ipc": "sol_execbench.reference_ipc.v1",
    "rocm_compatibility_matrix": ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    "rdna4_validation": RDNA4_VALIDATION_SCHEMA_VERSION,
    "release_baseline": RELEASE_BASELINE_SCHEMA_VERSION,
    "release_bundle": RELEASE_BUNDLE_SCHEMA_VERSION,
    "release_candidate": RELEASE_CANDIDATE_SCHEMA_VERSION,
    "release_environment": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
    "release_execution_plan": RELEASE_EXECUTION_PLAN_SCHEMA_VERSION,
    "release_rerun": RELEASE_RERUN_SCHEMA_VERSION,
    "release_solar_index": RELEASE_SOLAR_INDEX_SCHEMA_VERSION,
    "rocm_docker_targets": "sol_execbench.rocm_docker_targets.v1",
    "rocprofv3_diagnostics": "sol_execbench.rocprofv3_diagnostics.v1",
    "rocprofv3_overhead_calibration": ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION,
    "rocprofv3_profile": "sol_execbench.rocprofv3_profile.v1",
    "rocprofv3_timing": "sol_execbench.rocprofv3_timing.v1",
    "static_artifact_manifest": STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    "static_kernel_evidence": STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    "timing_isolation_snapshot": "sol_execbench.timing_isolation_snapshot.v1",
    "toolchain_routing": TOOLCHAIN_ROUTING_SCHEMA_VERSION,
    "rocm_event_timing_custom": "sol_execbench.rocm_event_timing.custom.v3",
    "rocm_event_timing_paper_counts": "sol_execbench.rocm_event_timing.paper_counts.v3",
}

AgentFeedbackSchemaVersion: TypeAlias = Literal["sol_execbench.agent_feedback.v3"]
DecisionSchemaVersion: TypeAlias = Literal["sol_execbench.decision.v2"]
ProfileSummarySchemaVersion: TypeAlias = Literal["sol_execbench.profile_summary.v3"]

CURRENT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(SCHEMA_VERSIONS.values())

__all__ = [
    "AGENT_FEEDBACK_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION",
    "CURRENT_SCHEMA_VERSIONS",
    "DECISION_SCHEMA_VERSION",
    "ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION",
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION",
    "EVALUATOR_CONTRACT_SCHEMA_VERSION",
    "PROFILE_SUMMARY_SCHEMA_VERSION",
    "ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION",
    "RDNA4_VALIDATION_SCHEMA_VERSION",
    "RELEASE_BASELINE_SCHEMA_VERSION",
    "RELEASE_BUNDLE_SCHEMA_VERSION",
    "RELEASE_CANDIDATE_SCHEMA_VERSION",
    "RELEASE_ENVIRONMENT_SCHEMA_VERSION",
    "RELEASE_EXECUTION_PLAN_SCHEMA_VERSION",
    "RELEASE_RERUN_SCHEMA_VERSION",
    "RELEASE_SOLAR_INDEX_SCHEMA_VERSION",
    "ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION",
    "TOOLCHAIN_ROUTING_SCHEMA_VERSION",
    "AgentFeedbackSchemaVersion",
    "DecisionSchemaVersion",
    "ProfileSummarySchemaVersion",
]
