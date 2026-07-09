# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

from sol_execbench.core.dataset.execution_closure.builder import (
    build_execution_closure_report,
)
from sol_execbench.core.dataset.execution_closure.io import (
    write_execution_closure_report,
)
from sol_execbench.core.dataset.execution_closure.models import (
    EXECUTION_CLOSURE_SCHEMA_VERSION,
    ExecutionClosureClaimBoundary,
    ExecutionClosureProvenance,
    ExecutionClosureProvenanceMismatch,
    ExecutionClosureReasonCode,
    ExecutionClosureRecord,
    ExecutionClosureReport,
    ExecutionClosureStatus,
    ExecutionClosureTotals,
)
from sol_execbench.core.dataset.execution_closure.status import (
    closure_status_for_trace_status,
    closure_status_with_evidence,
    compare_execution_closure_provenance,
)

__all__ = [
    "EXECUTION_CLOSURE_SCHEMA_VERSION",
    "ExecutionClosureClaimBoundary",
    "ExecutionClosureProvenance",
    "ExecutionClosureProvenanceMismatch",
    "ExecutionClosureReasonCode",
    "ExecutionClosureRecord",
    "ExecutionClosureReport",
    "ExecutionClosureStatus",
    "ExecutionClosureTotals",
    "build_execution_closure_report",
    "closure_status_for_trace_status",
    "closure_status_with_evidence",
    "compare_execution_closure_provenance",
    "write_execution_closure_report",
]
