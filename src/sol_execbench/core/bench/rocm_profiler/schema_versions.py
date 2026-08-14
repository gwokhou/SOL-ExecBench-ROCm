# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current ROCm profiler acquisition artifact schemas."""

from enum import StrEnum


class ProfilerArtifactSchema(StrEnum):
    """Canonical rocprofv3 collection and calibration identifiers."""

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
