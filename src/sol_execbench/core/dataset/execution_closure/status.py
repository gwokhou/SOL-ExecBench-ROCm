# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.dataset.execution_closure.models import (
    ExecutionClosureProvenance,
    ExecutionClosureProvenanceMismatch,
    ExecutionClosureReasonCode,
    ExecutionClosureStatus,
)


def closure_status_for_trace_status(
    trace_status: str | None,
    *,
    skipped: bool = False,
) -> ExecutionClosureStatus:
    if trace_status is None:
        return ExecutionClosureStatus.MISSING_TRACE
    if skipped and trace_status == "PASSED":
        return ExecutionClosureStatus.SKIPPED_EXISTING_PASS
    if trace_status == "PASSED":
        return ExecutionClosureStatus.ATTEMPTED_PASSED
    return ExecutionClosureStatus.ATTEMPTED_FAILED


def closure_status_with_evidence(
    status: ExecutionClosureStatus | str,
    evidence_gaps: list[str],
) -> ExecutionClosureStatus:
    typed_status = ExecutionClosureStatus(status)
    if evidence_gaps and typed_status in {
        ExecutionClosureStatus.ATTEMPTED_PASSED,
        ExecutionClosureStatus.ATTEMPTED_FAILED,
        ExecutionClosureStatus.SKIPPED_EXISTING_PASS,
    }:
        return ExecutionClosureStatus.DERIVED_EVIDENCE_MISSING
    return typed_status


def compare_execution_closure_provenance(
    expected: ExecutionClosureProvenance | dict[str, Any],
    observed: ExecutionClosureProvenance | dict[str, Any],
) -> list[ExecutionClosureProvenanceMismatch]:
    expected_model = (
        expected
        if isinstance(expected, ExecutionClosureProvenance)
        else ExecutionClosureProvenance(**expected)
    )
    observed_model = (
        observed
        if isinstance(observed, ExecutionClosureProvenance)
        else ExecutionClosureProvenance(**observed)
    )
    comparisons = (
        ("dataset_root", ExecutionClosureReasonCode.SELECTION_MISMATCH),
        ("selected_categories", ExecutionClosureReasonCode.SELECTION_MISMATCH),
        ("limit", ExecutionClosureReasonCode.SELECTION_MISMATCH),
        ("max_workloads", ExecutionClosureReasonCode.SELECTION_MISMATCH),
        (
            "dataset_manifest_checksum",
            ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH,
        ),
        ("dataset_source_id", ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH),
        (
            "dataset_migration_kind",
            ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH,
        ),
        (
            "dataset_license_boundary",
            ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH,
        ),
        (
            "dataset_manifest_summary",
            ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH,
        ),
        ("readiness_checksum", ExecutionClosureReasonCode.READINESS_CHECKSUM_MISMATCH),
        ("readiness_summary", ExecutionClosureReasonCode.READINESS_CHECKSUM_MISMATCH),
        (
            "ready_subset_checksum",
            ExecutionClosureReasonCode.READY_SUBSET_CHECKSUM_MISMATCH,
        ),
        (
            "ready_subset_summary",
            ExecutionClosureReasonCode.READY_SUBSET_CHECKSUM_MISMATCH,
        ),
        (
            "workload_identity_checksum",
            ExecutionClosureReasonCode.WORKLOAD_IDENTITY_MISMATCH,
        ),
        (
            "long_tail_exclusions_checksum",
            ExecutionClosureReasonCode.SELECTION_MISMATCH,
        ),
        ("long_tail_exclusions_summary", ExecutionClosureReasonCode.SELECTION_MISMATCH),
        ("solution_mode", ExecutionClosureReasonCode.SOLUTION_MODE_MISMATCH),
        ("solution_name", ExecutionClosureReasonCode.SOLUTION_MISMATCH),
        ("timeout", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("workload_shard_size", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("execution_mode", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("prepare_jobs", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("gpu_jobs", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("timeout_policy", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("timeout_overrides", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("blob_precheck", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("log_order", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("warmup_runs", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("iterations", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        (
            "min_measurement_time_seconds",
            ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH,
        ),
        ("lock_clocks", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("benchmark_config", ExecutionClosureReasonCode.RUNTIME_CONFIG_MISMATCH),
        ("git_commit", ExecutionClosureReasonCode.GIT_COMMIT_MISMATCH),
        (
            "requested_evidence_requirements",
            ExecutionClosureReasonCode.EVIDENCE_REQUIREMENT_MISMATCH,
        ),
    )
    mismatches: list[ExecutionClosureProvenanceMismatch] = []
    for field, reason in comparisons:
        expected_value = getattr(expected_model, field)
        observed_value = getattr(observed_model, field)
        if field == "requested_evidence_requirements":
            expected_value = tuple(sorted(expected_value))
            observed_value = tuple(sorted(observed_value))
        if expected_value != observed_value:
            mismatches.append(
                ExecutionClosureProvenanceMismatch(
                    field=field,
                    reason_code=reason,
                    expected=expected_value,
                    observed=observed_value,
                )
            )
    return mismatches
