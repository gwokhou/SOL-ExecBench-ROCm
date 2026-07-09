# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paper denominator accounting sidecar helpers."""

from __future__ import annotations

from sol_execbench.core.dataset.paper_denominator.builder import (
    build_paper_denominator_report,
    load_json,
)
from sol_execbench.core.dataset.paper_denominator.models import (
    CLAIM_BOUNDARY_TEXT,
    DENOMINATOR_STATE_KEYS,
    EVIDENCE_KEYS,
    PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION,
    REQUIRED_RECORD_EVIDENCE_REFS,
    PaperDenominatorCategory,
    PaperDenominatorClaimBoundary,
    PaperDenominatorEvidenceGap,
    PaperDenominatorNextEvidenceHint,
    PaperDenominatorProblem,
    PaperDenominatorReasonBucket,
    PaperDenominatorReport,
    PaperDenominatorRollup,
    PaperDenominatorSourceRef,
    PaperDenominatorSources,
    PaperDenominatorStateTotals,
    PaperDenominatorWorkload,
)
from sol_execbench.core.dataset.paper_denominator.rendering import (
    render_paper_denominator_markdown,
    write_paper_denominator_reports,
)

__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "DENOMINATOR_STATE_KEYS",
    "EVIDENCE_KEYS",
    "PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION",
    "REQUIRED_RECORD_EVIDENCE_REFS",
    "PaperDenominatorCategory",
    "PaperDenominatorClaimBoundary",
    "PaperDenominatorEvidenceGap",
    "PaperDenominatorNextEvidenceHint",
    "PaperDenominatorProblem",
    "PaperDenominatorReasonBucket",
    "PaperDenominatorReport",
    "PaperDenominatorRollup",
    "PaperDenominatorSourceRef",
    "PaperDenominatorSources",
    "PaperDenominatorStateTotals",
    "PaperDenominatorWorkload",
    "build_paper_denominator_report",
    "load_json",
    "render_paper_denominator_markdown",
    "write_paper_denominator_reports",
]
