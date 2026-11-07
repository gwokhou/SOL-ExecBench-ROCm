# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current ROCm profiler acquisition artifact schemas."""

from enum import StrEnum


class ProfilerArtifactSchema(StrEnum):
    """Canonical rocprofv3 collection and calibration identifiers."""

    HIP_TRACE_JOURNAL = "sol_execbench.hip_trace_journal.v1"
    HIP_TRACE_SCOPE = "sol_execbench.hip_trace_scope.v1"
    HIP_SDK_LEDGER = "sol_execbench.hip_sdk_ledger.v2"
    HIP_EVALUATE_SCOPE = "sol_execbench.hip_evaluate_scope.v2"
    HIP_EVALUATE_RECEIPT = "sol_execbench.hip_evaluate_receipt.v1"
    HIP_EVALUATE_INTENT = "sol_execbench.hip_evaluate_intent.v1"
    HIP_TORCH_SCOPE = "sol_execbench.hip_torch_scope.v2"
    PMC_CLOCK_WINDOW = "sol_execbench.pmc_clock_window.v2"

    ROCPROFV3_COUNTER_MANIFEST = "sol_execbench.rocprofv3_counter_manifest.v3"
    ROCPROFV3_COUNTER_PROVENANCE = (
        "sol_execbench.rocprofv3_counter_provenance.v5"
    )
    ROCPROFV3_OVERHEAD_CALIBRATION = (
        "sol_execbench.rocprofv3_overhead_calibration.v2"
    )
    ROCPROFV3_SESSION = "sol_execbench.rocprofv3_session.v1"
    ROCPROFV3_TIMING = "sol_execbench.rocprofv3_timing.v1"


class ProfilerSessionArtifactKind(StrEnum):
    """Artifacts emitted by one rocprofv3 profiling session."""

    DIAGNOSTICS = "diagnostics"
    PROFILE = "profile"


__all__ = ["ProfilerArtifactSchema", "ProfilerSessionArtifactKind"]
