# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from sol_execbench.core.platform.diagnostics_libraries import (
    detect_tool,
    local_gfx_target,
    rocm_library_diagnostics,
    rocm_library_readiness,
    rocm_tool_diagnostics,
)
from sol_execbench.core.platform.diagnostics_models import (
    ROCM_LIBRARY_SPECS,
    DiagnosticStage,
    ProfilerBackend,
    ProfilerReadiness,
    RocmLibraryReadiness,
    RocmLibrarySpec,
    SolExecBenchError,
    StageDiagnostic,
)
from sol_execbench.core.platform.diagnostics_validation import (
    classify_gfx,
    select_profiler_backend,
)

__all__ = [
    "ROCM_LIBRARY_SPECS",
    "DiagnosticStage",
    "ProfilerBackend",
    "ProfilerReadiness",
    "RocmLibraryReadiness",
    "RocmLibrarySpec",
    "SolExecBenchError",
    "StageDiagnostic",
    "classify_gfx",
    "detect_tool",
    "local_gfx_target",
    "rocm_library_diagnostics",
    "rocm_library_readiness",
    "rocm_tool_diagnostics",
    "select_profiler_backend",
]
