# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current first-party schema identifiers.

Each artifact family has exactly one current wire contract. Historical schema
identifiers intentionally do not live in this module or anywhere else in the
working tree; Git history is the only archive for superseded contracts.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

AKA_CORPUS_MANIFEST_SCHEMA_VERSION: Final = 7
AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = 2
AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION: Final = 3
COVERAGE_POLICY_SCHEMA_VERSION: Final = 1

AGENT_FEEDBACK_SCHEMA_VERSION: Final = "sol_execbench.agent_feedback.v7"
AMD_ISA_RELEASE_LOCK_SCHEMA_VERSION: Final = (
    "sol_execbench.amd_isa_release_lock.v1"
)
ARCH_CAPABILITY_BUDGET_SCHEMA_VERSION: Final = (
    "sol_execbench.arch_capability_budget.v1"
)
BENCHMARK_CONFIG_SCHEMA_VERSION: Final = "sol_execbench.benchmark_config.v2"
CLI_CONTRACT_SCHEMA_VERSION: Final = "sol_execbench.cli_contract.v1"
CLI_RESPONSE_SCHEMA_VERSION: Final = "sol_execbench.cli_response.v1"
DATASET_PROVENANCE_POLICY_SCHEMA_VERSION: Final = (
    "sol_execbench.dataset_provenance_policy.v1"
)
DATASET_REDISTRIBUTION_CHECK_SCHEMA_VERSION: Final = (
    "sol_execbench.dataset_redistribution_check.v1"
)
DEFINITION_SCHEMA_VERSION: Final = "sol_execbench.definition.v1"
DEPENDENCY_PREFLIGHT_SCHEMA_VERSION: Final = (
    "sol_execbench.dependency_preflight.v1"
)
DECISION_SCHEMA_VERSION: Final = "sol_execbench.decision.v2"
DERIVED_EVIDENCE_SCHEMA_VERSION: Final = "sol_execbench.derived_evidence.v1"
SOLUTION_SCHEMA_VERSION: Final = "sol_execbench.solution.v1"
SOLAR_WORKER_IPC_SCHEMA_VERSION: Final = "sol_execbench.solar_worker_ipc.v1"
TRACE_SCHEMA_VERSION: Final = "sol_execbench.trace.v1"
WORKLOAD_SCHEMA_VERSION: Final = "sol_execbench.workload.v2"
PROFILE_SUMMARY_SCHEMA_VERSION: Final = "sol_execbench.profile_summary.v3"
PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_diagnostic.v7"
)
PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_evidence_manifest.v5"
)
PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_timing_evidence.v3"
)
PERFORMANCE_REPLAY_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_replay_evidence.v4"
)
PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_access_evidence.v1"
)
PERFORMANCE_SCHEDULE_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.performance_schedule_evidence.v1"
)
DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_acceptance.v6"
)
DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_calibration.v7"
)
DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_calibration_audit.v7"
)
DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_inference_profile.v9"
)
DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION: Final = (
    "sol_execbench.diagnostic_validation_corpus.v6"
)
DOCKER_PREFLIGHT_SCHEMA_VERSION: Final = "sol_execbench.docker_preflight.v1"
ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION: Final = (
    "sol_execbench.environment_diagnostics.v1"
)
ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION: Final = (
    "sol_execbench.environment_snapshot.v2"
)
EVALUATOR_CONTRACT_SCHEMA_VERSION: Final = "sol_execbench.evaluator_contract.v6"
GPU_DEVICE_ISOLATION_SCHEMA_VERSION: Final = (
    "sol_execbench.gpu_device_isolation.v1"
)
NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION: Final = (
    "sol_execbench.no_trace_diagnostics.v1"
)
OFFICIAL_SCORE_AVAILABILITY_SCHEMA_VERSION: Final = (
    "sol_execbench.official_score_availability.v3"
)
REFERENCE_IPC_SCHEMA_VERSION: Final = "sol_execbench.reference_ipc.v2"
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
    "sol_execbench.rocprofv3_counter_provenance.v5"
)
ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_counter_manifest.v3"
)
ROCPROFV3_DIAGNOSTICS_SCHEMA_VERSION: Final = (
    "sol_execbench.rocprofv3_diagnostics.v1"
)
ROCPROFV3_PROFILE_SCHEMA_VERSION: Final = "sol_execbench.rocprofv3_profile.v1"
ROCPROFV3_TIMING_SCHEMA_VERSION: Final = "sol_execbench.rocprofv3_timing.v1"
ROCM_DOCKER_TARGETS_SCHEMA_VERSION: Final = (
    "sol_execbench.rocm_docker_targets.v1"
)
ROCM_EVENT_TIMING_CUSTOM_SCHEMA_VERSION: Final = (
    "sol_execbench.rocm_event_timing.custom.v4"
)
ROCM_EVENT_TIMING_PAPER_COUNTS_SCHEMA_VERSION: Final = (
    "sol_execbench.rocm_event_timing.paper_counts.v4"
)
STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "sol_execbench.static_artifact_manifest.v1"
)
STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION: Final = (
    "sol_execbench.static_kernel_evidence.v4"
)
TOOLCHAIN_ROUTING_SCHEMA_VERSION: Final = "sol_execbench.toolchain_routing.v1"

SCHEMA_VERSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "agent_feedback": AGENT_FEEDBACK_SCHEMA_VERSION,
        "amd_isa_release_lock": AMD_ISA_RELEASE_LOCK_SCHEMA_VERSION,
        "arch_capability_budget": ARCH_CAPABILITY_BUDGET_SCHEMA_VERSION,
        "benchmark_config": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "cli_contract": CLI_CONTRACT_SCHEMA_VERSION,
        "cli_response": CLI_RESPONSE_SCHEMA_VERSION,
        "dataset_provenance_policy": DATASET_PROVENANCE_POLICY_SCHEMA_VERSION,
        "dataset_redistribution_check": (
            DATASET_REDISTRIBUTION_CHECK_SCHEMA_VERSION
        ),
        "definition": DEFINITION_SCHEMA_VERSION,
        "dependency_preflight": DEPENDENCY_PREFLIGHT_SCHEMA_VERSION,
        "decision": DECISION_SCHEMA_VERSION,
        "derived_evidence": DERIVED_EVIDENCE_SCHEMA_VERSION,
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
        "performance_access_evidence": PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION,
        "performance_schedule_evidence": (
            PERFORMANCE_SCHEDULE_EVIDENCE_SCHEMA_VERSION
        ),
        "diagnostic_acceptance": DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION,
        "diagnostic_calibration": DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION,
        "diagnostic_calibration_audit": (
            DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION
        ),
        "diagnostic_inference_profile": DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION,
        "diagnostic_validation_corpus": DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION,
        "docker_preflight": DOCKER_PREFLIGHT_SCHEMA_VERSION,
        "corpus_stage_readiness_record": CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION,
        "corpus_stage_readiness_summary": CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION,
        "corpus_stage_trace_identity": CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION,
        "environment_diagnostics": ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "environment_snapshot": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        "evaluator_contract": EVALUATOR_CONTRACT_SCHEMA_VERSION,
        "gpu_device_isolation": GPU_DEVICE_ISOLATION_SCHEMA_VERSION,
        "no_trace_diagnostics": NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
        "official_score_availability": OFFICIAL_SCORE_AVAILABILITY_SCHEMA_VERSION,
        "reference_ipc": REFERENCE_IPC_SCHEMA_VERSION,
        "rocm_compatibility_matrix": ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
        "rdna4_validation": RDNA4_VALIDATION_SCHEMA_VERSION,
        "release_baseline": RELEASE_BASELINE_SCHEMA_VERSION,
        "release_bundle": RELEASE_BUNDLE_SCHEMA_VERSION,
        "release_candidate": RELEASE_CANDIDATE_SCHEMA_VERSION,
        "release_environment": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
        "release_execution_plan": RELEASE_EXECUTION_PLAN_SCHEMA_VERSION,
        "release_solar_index": RELEASE_SOLAR_INDEX_SCHEMA_VERSION,
        "rocm_docker_targets": ROCM_DOCKER_TARGETS_SCHEMA_VERSION,
        "rocprofv3_diagnostics": ROCPROFV3_DIAGNOSTICS_SCHEMA_VERSION,
        "rocprofv3_counter_provenance": (
            ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION
        ),
        "rocprofv3_counter_manifest": ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION,
        "rocprofv3_overhead_calibration": ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION,
        "rocprofv3_profile": ROCPROFV3_PROFILE_SCHEMA_VERSION,
        "rocprofv3_timing": ROCPROFV3_TIMING_SCHEMA_VERSION,
        "static_artifact_manifest": STATIC_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "static_kernel_evidence": STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
        "toolchain_routing": TOOLCHAIN_ROUTING_SCHEMA_VERSION,
        "rocm_event_timing_custom": ROCM_EVENT_TIMING_CUSTOM_SCHEMA_VERSION,
        "rocm_event_timing_paper_counts": (
            ROCM_EVENT_TIMING_PAPER_COUNTS_SCHEMA_VERSION
        ),
    }
)

type AgentFeedbackSchemaVersion = Literal["sol_execbench.agent_feedback.v7"]
type DecisionSchemaVersion = Literal["sol_execbench.decision.v2"]
type ProfileSummarySchemaVersion = Literal["sol_execbench.profile_summary.v3"]

CURRENT_NUMERIC_SCHEMA_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "aka_corpus_manifest": AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
        "aka_materialization_manifest": (
            AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        ),
        "aka_tolerance_calibration": AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
        "coverage_policy": COVERAGE_POLICY_SCHEMA_VERSION,
    }
)

CURRENT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    SCHEMA_VERSIONS.values(),
)

__all__ = [
    "AGENT_FEEDBACK_SCHEMA_VERSION",
    "AKA_CORPUS_MANIFEST_SCHEMA_VERSION",
    "AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION",
    "AMD_ISA_RELEASE_LOCK_SCHEMA_VERSION",
    "ARCH_CAPABILITY_BUDGET_SCHEMA_VERSION",
    "BENCHMARK_CONFIG_SCHEMA_VERSION",
    "CLI_CONTRACT_SCHEMA_VERSION",
    "CLI_RESPONSE_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION",
    "CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION",
    "CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION",
    "COVERAGE_POLICY_SCHEMA_VERSION",
    "CURRENT_NUMERIC_SCHEMA_VERSIONS",
    "CURRENT_SCHEMA_VERSIONS",
    "DATASET_PROVENANCE_POLICY_SCHEMA_VERSION",
    "DATASET_REDISTRIBUTION_CHECK_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "DEFINITION_SCHEMA_VERSION",
    "DEPENDENCY_PREFLIGHT_SCHEMA_VERSION",
    "DERIVED_EVIDENCE_SCHEMA_VERSION",
    "DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION",
    "DIAGNOSTIC_CALIBRATION_AUDIT_SCHEMA_VERSION",
    "DIAGNOSTIC_CALIBRATION_SCHEMA_VERSION",
    "DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION",
    "DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION",
    "DOCKER_PREFLIGHT_SCHEMA_VERSION",
    "ENVIRONMENT_DIAGNOSTICS_SCHEMA_VERSION",
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION",
    "EVALUATOR_CONTRACT_SCHEMA_VERSION",
    "GPU_DEVICE_ISOLATION_SCHEMA_VERSION",
    "NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION",
    "OFFICIAL_SCORE_AVAILABILITY_SCHEMA_VERSION",
    "PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION",
    "PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION",
    "PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "PERFORMANCE_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "PERFORMANCE_SCHEDULE_EVIDENCE_SCHEMA_VERSION",
    "PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION",
    "PROFILE_SUMMARY_SCHEMA_VERSION",
    "RDNA4_VALIDATION_SCHEMA_VERSION",
    "REFERENCE_IPC_SCHEMA_VERSION",
    "RELEASE_BASELINE_SCHEMA_VERSION",
    "RELEASE_BUNDLE_SCHEMA_VERSION",
    "RELEASE_CANDIDATE_SCHEMA_VERSION",
    "RELEASE_ENVIRONMENT_SCHEMA_VERSION",
    "RELEASE_EXECUTION_PLAN_SCHEMA_VERSION",
    "RELEASE_SOLAR_INDEX_SCHEMA_VERSION",
    "ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION",
    "ROCM_DOCKER_TARGETS_SCHEMA_VERSION",
    "ROCM_EVENT_TIMING_CUSTOM_SCHEMA_VERSION",
    "ROCM_EVENT_TIMING_PAPER_COUNTS_SCHEMA_VERSION",
    "ROCPROFV3_COUNTER_MANIFEST_SCHEMA_VERSION",
    "ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION",
    "ROCPROFV3_DIAGNOSTICS_SCHEMA_VERSION",
    "ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION",
    "ROCPROFV3_PROFILE_SCHEMA_VERSION",
    "ROCPROFV3_TIMING_SCHEMA_VERSION",
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
