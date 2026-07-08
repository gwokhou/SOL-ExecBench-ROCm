# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import ConfigDict

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


ROCM_COMPATIBILITY_MATRIX_DIFF_SCHEMA_VERSION = (
    "sol_execbench.rocm_compatibility_matrix_diff.v1"
)


_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class MatrixDiffSeverity(str, Enum):
    """Severity categories for Matrix report semantic changes."""

    CLAIM_BOUNDARY_ESCALATION = "claim_boundary_escalation"
    VALIDATION_DOWNGRADE = "validation_downgrade"
    MIXED_VERSION_DRIFT = "mixed_version_drift"
    RUNTIME_UNAVAILABILITY = "runtime_unavailability"
    GPU_ARCHITECTURE_DRIFT = "gpu_architecture_drift"
    IMAGE_DEPENDENCY_DRIFT = "image_dependency_drift"
    SEMANTIC_CHANGE = "semantic_change"
    INFORMATIONAL = "informational"


class MatrixEntryDiffKind(str, Enum):
    """Entry-level diff bucket."""

    ADDED = "added"
    REMOVED = "removed"
    UNCHANGED = "unchanged"
    CHANGED = "changed"


_SEVERITY_RANK = {
    MatrixDiffSeverity.CLAIM_BOUNDARY_ESCALATION: 0,
    MatrixDiffSeverity.VALIDATION_DOWNGRADE: 1,
    MatrixDiffSeverity.MIXED_VERSION_DRIFT: 2,
    MatrixDiffSeverity.RUNTIME_UNAVAILABILITY: 3,
    MatrixDiffSeverity.GPU_ARCHITECTURE_DRIFT: 4,
    MatrixDiffSeverity.IMAGE_DEPENDENCY_DRIFT: 5,
    MatrixDiffSeverity.SEMANTIC_CHANGE: 6,
    MatrixDiffSeverity.INFORMATIONAL: 7,
}


class MatrixReportRef(BaseModelWithDocstrings):
    """Bounded reference metadata for one Matrix report in a diff."""

    model_config = _MODEL_CONFIG

    label: str
    """Caller-provided or derived label for the Matrix report."""
    schema_version: str
    """Validated Matrix report schema version."""
    generated_at: str
    """Report generation timestamp."""
    entry_count: int
    """Number of entries in the report."""


class MatrixEntryDiff(BaseModelWithDocstrings):
    """Semantic diff for one Matrix entry key."""

    model_config = _MODEL_CONFIG

    diff_key: str
    """Stable key in the form target_id|validation_scope."""
    target_id: str
    """Matrix target identifier."""
    validation_scope: str
    """Matrix validation scope."""
    kind: MatrixEntryDiffKind
    """Diff bucket for this entry key."""
    severity: MatrixDiffSeverity
    """Highest-ranked severity for this entry diff."""
    severity_categories: list[MatrixDiffSeverity]
    """All severity categories detected for this entry diff."""
    semantic_changes: dict[str, dict[str, Any]]
    """Grouped semantic field changes with old/new normalized values."""
    old_entry: dict[str, Any] | None = None
    """Normalized old entry for added/removed context."""
    new_entry: dict[str, Any] | None = None
    """Normalized new entry for added/removed context."""


class MatrixReportDiff(BaseModelWithDocstrings):
    """Diagnostic-only semantic diff for two Matrix reports."""

    model_config = _MODEL_CONFIG

    schema_version: str = ROCM_COMPATIBILITY_MATRIX_DIFF_SCHEMA_VERSION
    """Matrix diff artifact schema version."""
    diagnostic_compatibility_evidence: Literal[True] = True
    """Diff output is diagnostic compatibility evidence only."""
    score_authority: Literal[False] = False
    """Diff output never grants benchmark score authority."""
    paper_parity_authority: Literal[False] = False
    """Diff output never grants paper-parity authority."""
    leaderboard_authority: Literal[False] = False
    """Diff output never grants leaderboard authority."""
    native_host_validation_authority: Literal[False] = False
    """Diff output never upgrades container evidence to native-host validation."""
    old_report: MatrixReportRef
    """Old Matrix report reference."""
    new_report: MatrixReportRef
    """New Matrix report reference."""
    summary_counts: dict[str, int]
    """Counts by added, removed, unchanged, and changed buckets."""
    report_semantic_changes: dict[str, dict[str, Any]] = {}
    """Report-level semantic changes, such as clock/evidence metadata drift."""
    entry_diffs: list[MatrixEntryDiff]
    """Per-entry semantic diffs."""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible Matrix diff output."""

        return self.model_dump(mode="json")
