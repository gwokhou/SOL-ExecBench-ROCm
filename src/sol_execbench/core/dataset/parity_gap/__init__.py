# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

from sol_execbench.core.dataset.parity_gap.builder import build_parity_gap_report
from sol_execbench.core.dataset.parity_gap.models import (
    DENOMINATOR_KEYS,
    EVIDENCE_KEYS,
    PARITY_GAP_REPORT_SCHEMA_VERSION,
    EvidenceCompleteness,
    ParityAmdScoreRecord,
    ParityExecutionClosureRecord,
    ParityGapBlocker,
    ParityGapCategory,
    ParityGapClaimBoundary,
    ParityGapDenominators,
    ParityGapReport,
    ParityGapSource,
    ParityReadinessWorkloadRecord,
)
from sol_execbench.core.dataset.parity_gap.records import (
    _amd_score_record,
    _execution_closure_record,
    load_json,
)
from sol_execbench.core.dataset.parity_gap.rendering import (
    render_parity_gap_markdown,
    write_parity_gap_reports,
)

__all__ = [
    "DENOMINATOR_KEYS",
    "EVIDENCE_KEYS",
    "PARITY_GAP_REPORT_SCHEMA_VERSION",
    "EvidenceCompleteness",
    "ParityAmdScoreRecord",
    "ParityExecutionClosureRecord",
    "ParityGapBlocker",
    "ParityGapCategory",
    "ParityGapClaimBoundary",
    "ParityGapDenominators",
    "ParityGapReport",
    "ParityGapSource",
    "ParityReadinessWorkloadRecord",
    "_amd_score_record",
    "_execution_closure_record",
    "build_parity_gap_report",
    "load_json",
    "render_parity_gap_markdown",
    "write_parity_gap_reports",
]
