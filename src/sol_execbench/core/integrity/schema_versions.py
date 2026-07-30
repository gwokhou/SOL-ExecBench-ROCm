# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current first-party schema identifiers.

Each artifact family has exactly one current wire contract. Historical schema
identifiers intentionally do not live in this module or anywhere else in the
working tree; Git history is the only archive for superseded contracts.
"""

from typing import Final, Literal

AKA_CORPUS_MANIFEST_SCHEMA_VERSION: Final = 6
AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = 2
AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION: Final = 2
COVERAGE_POLICY_SCHEMA_VERSION: Final = 1

AGENT_FEEDBACK_SCHEMA_VERSION: Final = "sol_execbench.agent_feedback.v4"
BENCHMARK_CONFIG_SCHEMA_VERSION: Final = "sol_execbench.benchmark_config.v1"
DEFINITION_SCHEMA_VERSION: Final = "sol_execbench.definition.v1"
DEPENDENCY_PREFLIGHT_SCHEMA_VERSION: Final = (
    "sol_execbench.dependency_preflight.v1"
)
DECISION_SCHEMA_VERSION: Final = "sol_execbench.decision.v2"
SOLUTION_SCHEMA_VERSION: Final = "sol_execbench.solution.v1"
SOLAR_WORKER_IPC_SCHEMA_VERSION: Final = "sol_execbench.solar_worker_ipc.v1"
TRACE_SCHEMA_VERSION: Final = "sol_execbench.trace.v1"
WORKLOAD_SCHEMA_VERSION: Final = "sol_execbench.workload.v1"
PROFILE_SUMMARY_SCHEMA_VERSION: Final = "sol_execbench.profile_summary.v3"
PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_diagnostic.v3"
)
PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_evidence_manifest.v2"
)
PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_timing_evidence.v2"
)
PERFORMANCE_REPLAY_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_replay_evidence.v1"
)
DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_acceptance.v2"
)
DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_calibration.v3"
)
DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_calibration_audit.v3"
)
DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_inference_profile.v1"
)
DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_validation_corpus.v2"
)
DOCKER_PREFLIGHT_SCHEMA_VERSION: Final = "sol_execbench.docker_preflight.v1"
ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION: Final = (
    "sol_execbench.environment_diagnostics.v1"
)
ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION: Final = (
    "sol_execbench.environment_snapshot.v2"
)
EVALUATOR_CONTRACT_SCHEMA_VERSION: Final = "sol_execbench.evaluator_contract.v5"
ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION: Final = (
    "sol_execbench.rocm_compatibility_matrix.v1"
)
CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION: Final = (
    "sol_execbench.corpus_stage_readiness_record.v4"
)
CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION: Final = (
    "sol_execbench.corpus_stage_readiness_summary.v2"
)
CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION: Final = (
    "sol_execbench.corpus_stage_trace_identity.v3"
)
RDNA4_VALIDATION_SCHEMA_VERSION: Final = "sol_execbench.rdna4_validation.v2"
RELEASE_BASELINE_SCHEMA_VERSION: Final = "sol_execbench.release_baseline.v1"
RELEASE_BUNDLE_SCHEMA_VERSION: Final = "sol_execbench.release_bundle.v2"
RELEASE_CANDIDATE_SCHEMA_VERSION: Final = "sol_execbench.release_candidate.v1"
RELEASE_ENVIRONMENT_SCHEMA_VERSION: Final = (
    "sol_execbench.release_environment.v1"
)
RELEASE_EXECUTION_PLAN_SCHEMA_VERSION: Final = (
    "sol_execbench.release_execution_plan.v2"
)
RELEASE_SOLAR_INDEX_SCHEMA_VERSION: Final = (
    "sol_execbench.release_solar_index.v2"
)
ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_overhead_calibration.v2"
)
ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_counter_provenance.v4"
)
ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_counter_manifest.v2"
)
STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.static_artifact_manifest.v1"
)
STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.static_kernel_evidence.v3"
)
TOOLCHAIN_ROUTING_SCHEMA_VERSION: Final = "sol_execbench.toolchain_routing.v1"

SCHEMA_VERSIONS: Final[dict[str, str]] = {
    "agent_feedback": AGENT_FEEDBACK_SCHEMA_VERSION,
    "benchmark_config": BENCHMARK_CONFIG_SCHEMA_VERSION,
    "definition": DEFINITION_SCHEMA_VERSION,
    "dependency_preflight": DEPENDENCY_PREFLIGHT_SCHEMA_VERSION,
    "decision": DECISION_SCHEMA_VERSION,
    "solution": SOLUTION_SCHEMA_VERSION,
    "solar_worker_ipc": SOLAR_WORKER_IPC_SCHEMA_VERSION,
    "trace": TRACE_SCHEMA_VERSION,
    "workload": WORKLOAD_SCHEMA_VERSION,
    "profile_summary": PROFILE_SUMMARY_SCHEMA_VERSION,
    "performance_diagnostic": PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION,
    "performance_evidence_manifest": (
        PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION
    ),
    "performance_timing_evidence": PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION,
    "performance_replay_evidence": PERFORMANCE_REPLAY_EVIDENCE_SCHEMA_VERSION,
    "diagnostic_acceptance": DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION,
    "diagnostic_calibration": DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION,
    "diagnostic_calibration_audit": (
        DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION
    ),
    "diagnostic_inference_profile": DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION,
    "diagnostic_validation_corpus": DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION,
    "docker_preflight": DOCKER_PREFLIGHT_SCHEMA_VERSION,
    "amd_isa_release_lock": "sol_execbench.amd_isa_release_lock.v1",
    "arch_capability_budget": "sol_execbench.arch_capability_budget.v1",
    "cli_contract": "sol_execbench.cli_contract.v1",
    "cli_response": "sol_execbench.cli_response.v1",
    "corpus_stage_readiness_record": CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION,
    "corpus_stage_readiness_summary": CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION,
    "corpus_stage_trace_identity": CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION,
    "dataset_provenance_policy": "sol_execbench.dataset_provenance_policy.v1",
    "dataset_redistribution_check": "sol_execbench.dataset_redistribution_check.v1",
    "derived_evidence": "sol_execbench.derived_evidence.v1",
    "environment_diagnostics": ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION,
    "environment_snapshot": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
    "evaluator_contract": EVALUATOR_CONTRACT_SCHEMA_VERSION,
    "gpu_device_isolation": "sol_execbench.gpu_device_isolation.v1",
    "no_trace_diagnostics": "sol_execbench.no_trace_diagnostics.v1",
    "official_score_availability": "sol_execbench.official_score_availability.v3",
    "reference_ipc": "sol_execbench.reference_ipc.v1",
    "rocm_compatibility_matrix": ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    "rdna4_validation": RDNA4_VALIDATION_SCHEMA_VERSION,
    "release_baseline": RELEASE_BASELINE_SCHEMA_VERSION,
    "release_bundle": RELEASE_BUNDLE_SCHEMA_VERSION,
    "release_candidate": RELEASE_CANDIDATE_SCHEMA_VERSION,
    "release_environment": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
    "release_execution_plan": RELEASE_EXECUTION_PLAN_SCHEMA_VERSION,
    "release_solar_index": RELEASE_SOLAR_INDEX_SCHEMA_VERSION,
    "rocm_docker_targets": "sol_execbench.rocm_docker_targets.v1",
    "rocprofv3_diagnostics": "sol_execbench.rocprofv3_diagnostics.v1",
    "rocprofv3_counter_provenance": (
        ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION
    ),
    "rocprofv3_counter_manifest": ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION,
    "rocprofv3_overhead_calibration": ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION,
    "rocprofv3_profile": "sol_execbench.rocprofv3_profile.v1",
    "rocprofv3_timing": "sol_execbench.rocprofv3_timing.v1",
    "static_artifact_manifest": STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    "static_kernel_evidence": STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    "toolchain_routing": TOOLCHAIN_ROUTING_SCHEMA_VERSION,
    "rocm_event_timing_custom": "sol_execbench.rocm_event_timing.custom.v3",
    "rocm_event_timing_paper_counts": "sol_execbench.rocm_event_timing.paper_counts.v3",
}

type AgentFeedbackSchemaVersion = Literal["sol_execbench.agent_feedback.v4"]
type DecisionSchemaVersion = Literal["sol_execbench.decision.v2"]
type ProfileSummarySchemaVersion = Literal["sol_execbench.profile_summary.v3"]

CURRENT_NUMERIC_SCHEMA_VERSIONS: Final[dict[str, int]] = {
    "aka_corpus_manifest": AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
    "aka_materialization_manifest": (
        AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
    ),
    "aka_tolerance_calibration": AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
    "coverage_policy": COVERAGE_POLICY_SCHEMA_VERSION,
}

CURRENT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    SCHEMA_VERSIONS.values(),
)

__all__ = [
    "AGENT_FEEDBACK_SCHEMA_VERSION",
    "AKA_CORPUS_MANIFEST_SCHEMA_VERSION",
    "AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION",
    "BENCHMARK_CONFIG_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION",
    "CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION",
    "COVERAGE_POLICY_SCHEMA_VERSION",
    "CURRENT_NUMERIC_SCHEMA_VERSIONS",
    "CURRENT_SCHEMA_VERSIONS",
    "DECISION_SCHEMA_VERSION",
    "DEFINITION_SCHEMA_VERSION",
    "DEPENDENCY_PREFLIGHT_SCHEMA_VERSION",
    "DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION",
    "DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION",
    "DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION",
    "DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION",
    "DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION",
    "DOCKER_PREFLIGHT_SCHEMA_VERSION",
    "ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION",
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION",
    "EVALUATOR_CONTRACT_SCHEMA_VERSION",
    "PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION",
    "PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "PERFORMANCE_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION",
    "PROFILE_SUMMARY_SCHEMA_VERSION",
    "RDNA4_VALIDATION_SCHEMA_VERSION",
    "RELEASE_BASELINE_SCHEMA_VERSION",
    "RELEASE_BUNDLE_SCHEMA_VERSION",
    "RELEASE_CANDIDATE_SCHEMA_VERSION",
    "RELEASE_ENVIRONMENT_SCHEMA_VERSION",
    "RELEASE_EXECUTION_PLAN_SCHEMA_VERSION",
    "RELEASE_SOLAR_INDEX_SCHEMA_VERSION",
    "ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION",
    "ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION",
    "ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION",
    "ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "SOLAR_WORKER_IPC_SCHEMA_VERSION",
    "SOLUTION_SCHEMA_VERSION",
    "STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION",
    "TOOLCHAIN_ROUTING_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "WORKLOAD_SCHEMA_VERSION",
    "AgentFeedbackSchemaVersion",
    "DecisionSchemaVersion",
    "ProfileSummarySchemaVersion",
]
