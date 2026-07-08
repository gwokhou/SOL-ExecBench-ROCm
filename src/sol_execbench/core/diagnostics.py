# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from sol_execbench.core.diagnostics_libraries import (
    detect_tool,
    local_gfx_target,
    rocm_library_diagnostics,
    rocm_library_readiness,
    rocm_tool_diagnostics,
)
from sol_execbench.core.diagnostics_models import (
    MI300X_FP8_READINESS,
    MI300X_REQUIRED_ARTIFACTS,
    MI300X_VALIDATION_RESULT_CATEGORIES,
    RDNA4_REQUIRED_ARTIFACTS,
    RDNA4_VALIDATION_RESULT_CATEGORIES,
    ROCM_LIBRARY_SPECS,
    DiagnosticStage,
    ProfilerBackend,
    ProfilerReadiness,
    RocmLibraryReadiness,
    RocmLibrarySpec,
    SolExecBenchError,
    StageDiagnostic,
    ValidationReadiness,
)
from sol_execbench.core.diagnostics_validation import (
    can_mark_mi300x_hardware_validated,
    can_mark_rdna4_validation_upgraded,
    cdna3_validation_readiness,
    classify_gfx,
    mi300x_validation_claim_blockers,
    rdna4_validation_claim_blockers,
    select_profiler_backend,
)

__all__ = [
    "MI300X_FP8_READINESS",
    "MI300X_REQUIRED_ARTIFACTS",
    "MI300X_VALIDATION_RESULT_CATEGORIES",
    "RDNA4_REQUIRED_ARTIFACTS",
    "RDNA4_VALIDATION_RESULT_CATEGORIES",
    "ROCM_LIBRARY_SPECS",
    "DiagnosticStage",
    "ProfilerBackend",
    "ProfilerReadiness",
    "RocmLibraryReadiness",
    "RocmLibrarySpec",
    "SolExecBenchError",
    "StageDiagnostic",
    "ValidationReadiness",
    "can_mark_mi300x_hardware_validated",
    "can_mark_rdna4_validation_upgraded",
    "cdna3_validation_readiness",
    "classify_gfx",
    "detect_tool",
    "local_gfx_target",
    "mi300x_validation_claim_blockers",
    "rdna4_validation_claim_blockers",
    "rocm_library_diagnostics",
    "rocm_library_readiness",
    "rocm_tool_diagnostics",
    "select_profiler_backend",
]
