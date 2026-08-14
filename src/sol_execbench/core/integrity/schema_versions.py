# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current first-party schema identifiers.

Each artifact family has exactly one current wire contract. Historical schema
identifiers intentionally do not live in this module or anywhere else in the
working tree; Git history is the only archive for superseded contracts.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class SchemaVersion(StrEnum):
    """Current string-valued first-party wire-schema identifiers."""

    AGENT_FEEDBACK = "sol_execbench.agent_feedback.v7"
    AMD_ISA_RELEASE_LOCK = "sol_execbench.amd_isa_release_lock.v1"
    ARCH_CAPABILITY_BUDGET = "sol_execbench.arch_capability_budget.v1"
    BATCH_GPU_QUALIFICATION = "sol_execbench.batch_gpu_qualification.v1"
    BATCH_GPU_QUALIFICATION_RECEIPT = (
        "sol_execbench.batch_gpu_qualification_receipt.v1"
    )
    BENCHMARK_CONFIG = "sol_execbench.benchmark_config.v2"
    CLI_CONTRACT = "sol_execbench.cli_contract.v1"
    CLI_RESPONSE = "sol_execbench.cli_response.v1"
    CORPUS_STAGE_READINESS_RECORD = (
        "sol_execbench.corpus_stage_readiness_record.v4"
    )
    CORPUS_STAGE_READINESS_SUMMARY = (
        "sol_execbench.corpus_stage_readiness_summary.v2"
    )
    CORPUS_STAGE_TRACE_IDENTITY = "sol_execbench.corpus_stage_trace_identity.v3"
    CORPUS_MANIFEST = "sol_execbench.corpus_manifest.v1"
    CORPUS_SELECTION_MANIFEST = "sol_execbench.corpus_selection_manifest.v1"
    DATASET_PROVENANCE_POLICY = "sol_execbench.dataset_provenance_policy.v1"
    DATASET_REDISTRIBUTION_CHECK = (
        "sol_execbench.dataset_redistribution_check.v1"
    )
    DEFINITION = "sol_execbench.definition.v1"
    DEPENDENCY_PREFLIGHT = "sol_execbench.dependency_preflight.v1"
    DECISION = "sol_execbench.decision.v2"
    DERIVED_EVIDENCE = "sol_execbench.derived_evidence.v1"
    DIAGNOSTIC_ACCEPTANCE = "sol_execbench.diagnostic_acceptance.v7"
    DIAGNOSTIC_ACCEPTANCE_EXPOSURE = (
        "sol_execbench.diagnostic_acceptance_exposure.v1"
    )
    DIAGNOSTIC_ARTIFACT_TREE = "sol_execbench.diagnostic_artifact_tree.v1"
    DIAGNOSTIC_CALIBRATION = "sol_execbench.diagnostic_calibration.v8"
    DIAGNOSTIC_CALIBRATION_AUDIT = (
        "sol_execbench.diagnostic_calibration_audit.v8"
    )
    DIAGNOSTIC_CORPUS_PREFLIGHT = "sol_execbench.diagnostic_corpus_preflight.v1"
    DIAGNOSTIC_CASE_REUSE = "sol_execbench.diagnostic_case_reuse.v1"
    DIAGNOSTIC_CORPUS_QUALIFICATION = (
        "sol_execbench.diagnostic_corpus_qualification.v1"
    )
    DIAGNOSTIC_CORPUS_QUALIFICATION_RECEIPT = (
        "sol_execbench.diagnostic_corpus_qualification_receipt.v1"
    )
    DIAGNOSTIC_DEVELOPMENT_CASE_REBIND = (
        "sol_execbench.diagnostic_development_case_rebind.v1"
    )
    DIAGNOSTIC_INFERENCE_PROFILE = (
        "sol_execbench.diagnostic_inference_profile.v10"
    )
    DIAGNOSTIC_HELD_OUT_FRAGMENT = (
        "sol_execbench.diagnostic_held_out_fragment.v1"
    )
    DIAGNOSTIC_LIFECYCLE_ACCEPTANCE = (
        "sol_execbench.diagnostic_lifecycle_acceptance.v2"
    )
    DIAGNOSTIC_LIFECYCLE_ATTEMPT = (
        "sol_execbench.diagnostic_lifecycle_attempt.v1"
    )
    DIAGNOSTIC_LIFECYCLE_CALIBRATION = (
        "sol_execbench.diagnostic_lifecycle_calibration.v2"
    )
    DIAGNOSTIC_LIFECYCLE_COLLECTION_RUN = (
        "sol_execbench.diagnostic_lifecycle_collection_run.v2"
    )
    DIAGNOSTIC_LIFECYCLE_CORPUS_SNAPSHOT = (
        "sol_execbench.diagnostic_lifecycle_corpus_snapshot.v2"
    )
    DIAGNOSTIC_LIFECYCLE_DESIGN = "sol_execbench.diagnostic_lifecycle_design.v2"
    DIAGNOSTIC_LIFECYCLE_MODEL_BUILD = (
        "sol_execbench.diagnostic_lifecycle_model_build.v2"
    )
    DIAGNOSTIC_LIFECYCLE_PLAN = "sol_execbench.diagnostic_lifecycle_plan.v2"
    DIAGNOSTIC_LIFECYCLE_PUBLICATION = (
        "sol_execbench.diagnostic_lifecycle_publication.v2"
    )
    DIAGNOSTIC_LIFECYCLE_RELEASE = (
        "sol_execbench.diagnostic_lifecycle_release.v2"
    )
    DIAGNOSTIC_LIFECYCLE_RUN = "sol_execbench.diagnostic_lifecycle_run.v3"
    DIAGNOSTIC_PUBLICATION_PROJECTION = (
        "sol_execbench.diagnostic_publication_projection.v2"
    )
    DIAGNOSTIC_PUBLISHED_RELEASE = (
        "sol_execbench.diagnostic_published_release.v3"
    )
    DIAGNOSTIC_RELEASE_ARCHIVE = "sol_execbench.diagnostic_release_archive.v2"
    DIAGNOSTIC_RELEASE_ATTESTATION = (
        "sol_execbench.diagnostic_release_attestation.v2"
    )
    DIAGNOSTIC_STAGE_RECEIPT = "sol_execbench.diagnostic_stage_receipt.v2"
    DIAGNOSTIC_SOURCE_TRANSITION = (
        "sol_execbench.diagnostic_source_transition.v1"
    )
    DIAGNOSTIC_VALIDATION_CORPUS = (
        "sol_execbench.diagnostic_validation_corpus.v9"
    )
    DIAGNOSTIC_VRAM_WORKING_SET_POLICY = (
        "sol_execbench.diagnostic_vram_working_set_policy.v1"
    )
    DOCKER_PREFLIGHT = "sol_execbench.docker_preflight.v1"
    ENVIRONMENT_DIAGNOSTICS = "sol_execbench.environment_diagnostics.v1"
    ENVIRONMENT_SNAPSHOT = "sol_execbench.environment_snapshot.v2"
    EVALUATOR_CONTRACT = "sol_execbench.evaluator_contract.v6"
    GPU_DEVICE_ISOLATION = "sol_execbench.gpu_device_isolation.v1"
    NATIVE_COMPILE_CACHE = "sol_execbench.native_compile_cache.v1"
    NO_TRACE_DIAGNOSTICS = "sol_execbench.no_trace_diagnostics.v1"
    OFFICIAL_SCORE_AVAILABILITY = "sol_execbench.official_score_availability.v3"
    PERFORMANCE_ACCESS_EVIDENCE = "sol_execbench.performance_access_evidence.v1"
    PERFORMANCE_DIAGNOSTIC = "sol_execbench.performance_diagnostic.v7"
    PERFORMANCE_EVIDENCE_MANIFEST = (
        "sol_execbench.performance_evidence_manifest.v5"
    )
    PERFORMANCE_REPLAY_EVIDENCE = "sol_execbench.performance_replay_evidence.v4"
    PERFORMANCE_SCHEDULE_EVIDENCE = (
        "sol_execbench.performance_schedule_evidence.v1"
    )
    PERFORMANCE_TIMING_EVIDENCE = "sol_execbench.performance_timing_evidence.v3"
    PROFILE_SUMMARY = "sol_execbench.profile_summary.v3"
    RDNA4_DIAGNOSTIC_CORPUS_DESIGN = "rdna4_diagnostic_corpus_design.v1"
    RDNA4_VALIDATION = "sol_execbench.rdna4_validation.v2"
    REFERENCE_IPC = "sol_execbench.reference_ipc.v2"
    RELEASE_ARCHIVE = "sol_execbench.release_archive.v1"
    RELEASE_ATTESTATION = "sol_execbench.release_attestation.v1"
    RELEASE_BASELINE = "sol_execbench.release_baseline.v1"
    RELEASE_BUNDLE = "sol_execbench.release_bundle.v2"
    RELEASE_CANDIDATE = "sol_execbench.release_candidate.v1"
    RELEASE_ENVIRONMENT = "sol_execbench.release_environment.v1"
    RELEASE_EXECUTION_PLAN = "sol_execbench.release_execution_plan.v2"
    RELEASE_SOLAR_INDEX = "sol_execbench.release_solar_index.v2"
    ROCM_COMPATIBILITY_MATRIX = "sol_execbench.rocm_compatibility_matrix.v1"
    ROCM_DOCKER_TARGETS = "sol_execbench.rocm_docker_targets.v1"
    ROCM_EVENT_TIMING_CUSTOM = "sol_execbench.rocm_event_timing.custom.v4"
    ROCM_EVENT_TIMING_PAPER_COUNTS = (
        "sol_execbench.rocm_event_timing.paper_counts.v4"
    )
    ROCPROFV3_COUNTER_MANIFEST = "sol_execbench.rocprofv3_counter_manifest.v3"
    ROCPROFV3_COUNTER_PROVENANCE = (
        "sol_execbench.rocprofv3_counter_provenance.v5"
    )
    ROCPROFV3_DIAGNOSTICS = "sol_execbench.rocprofv3_diagnostics.v1"
    ROCPROFV3_OVERHEAD_CALIBRATION = (
        "sol_execbench.rocprofv3_overhead_calibration.v2"
    )
    ROCPROFV3_PROFILE = "sol_execbench.rocprofv3_profile.v1"
    ROCPROFV3_TIMING = "sol_execbench.rocprofv3_timing.v1"
    SOLAR_WORKER_IPC = "sol_execbench.solar_worker_ipc.v2"
    SOLUTION = "sol_execbench.solution.v1"
    STATIC_ARTIFACT_MANIFEST = "sol_execbench.static_artifact_manifest.v1"
    STATIC_TARGET_DESCRIPTOR = "sol_execbench.static_target_descriptor.v1"
    STATIC_KERNEL_EVIDENCE = "sol_execbench.static_kernel_evidence.v4"
    TOOLCHAIN_ROUTING = "sol_execbench.toolchain_routing.v1"
    TRACE = "sol_execbench.trace.v1"
    WORKLOAD = "sol_execbench.workload.v2"


AKA_CORPUS_MANIFEST_SCHEMA_VERSION: Final = 7
AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = 2
AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION: Final = 3
COVERAGE_POLICY_SCHEMA_VERSION: Final = 1


SCHEMA_VERSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {version.name.lower(): version.value for version in SchemaVersion}
)
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
    SCHEMA_VERSIONS.values()
)

__all__ = [
    "AKA_CORPUS_MANIFEST_SCHEMA_VERSION",
    "AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION",
    "COVERAGE_POLICY_SCHEMA_VERSION",
    "CURRENT_NUMERIC_SCHEMA_VERSIONS",
    "CURRENT_SCHEMA_VERSIONS",
    "SCHEMA_VERSIONS",
    "SchemaVersion",
]
