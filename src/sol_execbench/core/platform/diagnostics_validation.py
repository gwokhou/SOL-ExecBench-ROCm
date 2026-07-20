# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from sol_execbench.core.platform.diagnostics_models import (
    DiagnosticStage,
    ProfilerBackend,
    ProfilerReadiness,
    SolExecBenchError,
)


def classify_gfx(gfx: str | None) -> str:
    """Classify an AMD gfx target into a broad architecture family."""
    if not gfx:
        return "unknown"
    if gfx.startswith("gfx94"):
        return "cdna3"
    if gfx.startswith("gfx12"):
        return "rdna4"
    if gfx.startswith("gfx11"):
        return "rdna3"
    return "unknown"


def select_profiler_backend(
    requested_level: str,
    gfx: str | None,
    *,
    rocprofiler_compute: bool = False,
    omniperf: bool = False,
    rocprofv3: bool = True,
) -> ProfilerReadiness:
    """Select an internal profiling readiness route for ROCm diagnostics.

    The return value is descriptive only. It does not change benchmark timing,
    output formats, or CLI behavior.
    """
    level = requested_level.strip().lower()
    if level not in {"basic", "full"}:
        raise SolExecBenchError(
            DiagnosticStage.ENVIRONMENT,
            "requested profiling level must be 'basic' or 'full'",
            hint="Use 'basic' for portable readiness checks or 'full' for architecture-specific profiling.",
        )

    if level == "basic":
        backend = ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP
        return ProfilerReadiness(
            backend=backend,
            reason="basic profile requested"
            if rocprofv3
            else "rocprofv3 not available for basic profiling",
            fallback_applied=not rocprofv3,
            effective_level="basic" if rocprofv3 else "skip",
        )

    family = classify_gfx(gfx)
    if family == "cdna3":
        if rocprofiler_compute:
            return ProfilerReadiness(
                backend=ProfilerBackend.ROCPROFILER_COMPUTE,
                reason="CDNA 3 detected and rocprofiler-compute is available",
                fallback_applied=False,
                effective_level="full",
            )
        if omniperf:
            return ProfilerReadiness(
                backend=ProfilerBackend.OMNIPERF,
                reason="CDNA 3 detected; Omniperf fallback is available",
                fallback_applied=False,
                effective_level="full",
            )
        return ProfilerReadiness(
            backend=ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP,
            reason="CDNA 3 full profiling requested but no CDNA-specific profiler is available",
            fallback_applied=True,
            effective_level="basic" if rocprofv3 else "skip",
        )

    if family.startswith("rdna"):
        return ProfilerReadiness(
            backend=ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP,
            reason=f"{family.upper()} detected; rocprofv3 readiness route selected"
            if rocprofv3
            else f"{family.upper()} detected but rocprofv3 is unavailable",
            fallback_applied=not rocprofv3,
            effective_level="full" if rocprofv3 else "skip",
        )

    return ProfilerReadiness(
        backend=ProfilerBackend.SKIP,
        reason="GPU architecture unknown; profiling readiness skipped",
        fallback_applied=True,
        effective_level="skip",
    )
