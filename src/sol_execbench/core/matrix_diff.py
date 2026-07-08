# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from sol_execbench.core.matrix_diff_io import load_matrix_report
from sol_execbench.core.matrix_diff_models import (
    ROCM_COMPATIBILITY_MATRIX_DIFF_SCHEMA_VERSION,
    MatrixDiffSeverity,
    MatrixEntryDiff,
    MatrixEntryDiffKind,
    MatrixReportDiff,
    MatrixReportRef,
)
from sol_execbench.core.matrix_diff_rendering import matrix_report_diff_to_markdown
from sol_execbench.core.matrix_diff_semantic import diff_matrix_reports

__all__ = [
    "ROCM_COMPATIBILITY_MATRIX_DIFF_SCHEMA_VERSION",
    "MatrixDiffSeverity",
    "MatrixEntryDiff",
    "MatrixEntryDiffKind",
    "MatrixReportDiff",
    "MatrixReportRef",
    "diff_matrix_reports",
    "load_matrix_report",
    "matrix_report_diff_to_markdown",
]
