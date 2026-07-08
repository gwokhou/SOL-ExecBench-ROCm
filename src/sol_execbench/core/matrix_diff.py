# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only semantic diffs for ROCm Compatibility Matrix reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, ValidationError

from sol_execbench.core.compatibility import (
    MatrixCompatibilityStatus,
    MatrixEntry,
    RocmCompatibilityMatrixReport,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.path_access import path_dict, path_get, path_mapping_list


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


def _normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    return value


def _normalize_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [_normalize_value(artifact) for artifact in artifacts]
    return sorted(
        normalized,
        key=lambda artifact: (
            str(path_get(artifact, "artifact_id", default="")),
            json.dumps(artifact, sort_keys=True),
        ),
    )


def _normalize_entry(entry: MatrixEntry) -> dict[str, Any]:
    payload = _normalize_value(entry.model_dump(mode="json"))
    payload["artifacts"] = _normalize_artifacts(path_mapping_list(payload, "artifacts"))
    return payload


def _diff_key(entry: MatrixEntry) -> tuple[str, str, str]:
    target_id = entry.target.target_id
    validation_scope = entry.target.validation_scope.value
    return f"{target_id}|{validation_scope}", target_id, validation_scope


def _entries_by_key(report: RocmCompatibilityMatrixReport) -> dict[str, MatrixEntry]:
    entries: dict[str, MatrixEntry] = {}
    for entry in report.entries:
        key, _, _ = _diff_key(entry)
        if key in entries:
            raise ValueError(f"Duplicate Matrix diff key: {key}")
        entries[key] = entry
    return entries


_FIELD_GROUPS = {
    "status": ("status",),
    "reason_code": ("reason_code",),
    "target": ("target",),
    "observed.host": ("observed", "host"),
    "observed.container": ("observed", "container"),
    "observed.python_dependency": ("observed", "python_dependency"),
    "observed.dependency_policy": ("observed", "dependency_policy"),
    "observed.toolchain": ("observed", "toolchain"),
    "observed.gpu": ("observed", "gpu"),
    "artifacts": ("artifacts",),
    "claim_boundary": ("claim_boundary",),
}


def _get_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    return path_get(payload, ".".join(path))


def _semantic_changes(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    changes = {}
    for group, path in _FIELD_GROUPS.items():
        old_value = _get_path(old_payload, path)
        new_value = _get_path(new_payload, path)
        if old_value != new_value:
            changes[group] = {"old": old_value, "new": new_value}
    return changes


def _is_validated(status: str) -> bool:
    return status in {
        MatrixCompatibilityStatus.HOST_VALIDATED.value,
        MatrixCompatibilityStatus.CONTAINER_VALIDATED.value,
    }


def _validation_downgraded(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> bool:
    old_status = old_payload["status"]
    new_status = new_payload["status"]
    if (
        old_status == MatrixCompatibilityStatus.HOST_VALIDATED.value
        and new_status != old_status
    ):
        return True
    return _is_validated(old_status) and not _is_validated(new_status)


def _claim_boundary_escalated(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> bool:
    old_claims = path_dict(old_payload, "claim_boundary")
    new_claims = path_dict(new_payload, "claim_boundary")
    for field in (
        "score_authority",
        "paper_parity_authority",
        "leaderboard_authority",
        "container_user_space_validated",
        "native_host_validated",
        "hardware_validated",
    ):
        if (
            path_get(new_claims, field) is True
            and path_get(old_claims, field) is not True
        ):
            return True
    return False


def _mixed_version_drift(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> bool:
    statuses = {old_payload["status"], new_payload["status"]}
    if MatrixCompatibilityStatus.MIXED_VERSION.value in statuses:
        return True
    old_target = path_dict(old_payload, "target")
    new_target = path_dict(new_payload, "target")
    old_python = path_dict(old_payload, "observed.python_dependency")
    new_python = path_dict(new_payload, "observed.python_dependency")
    return (
        path_get(old_target, "requested_rocm_user_space_version")
        != path_get(new_target, "requested_rocm_user_space_version")
        or path_get(old_target, "pytorch_rocm_target")
        != path_get(new_target, "pytorch_rocm_target")
        or path_get(old_python, "torch_rocm_target")
        != path_get(new_python, "torch_rocm_target")
        or path_get(old_python, "torch_hip_version")
        != path_get(new_python, "torch_hip_version")
    )


def _runtime_unavailable(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> bool:
    return MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE.value in {
        old_payload["status"],
        new_payload["status"],
    }


def _gpu_architecture_drift(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> bool:
    old_target = path_dict(old_payload, "target")
    new_target = path_dict(new_payload, "target")
    old_gpu = path_dict(old_payload, "observed.gpu")
    new_gpu = path_dict(new_payload, "observed.gpu")
    return path_get(old_target, "intended_gpu_architecture") != path_get(
        new_target, "intended_gpu_architecture"
    ) or path_get(old_gpu, "gfx_architecture") != path_get(new_gpu, "gfx_architecture")


def _image_dependency_drift(changes: dict[str, dict[str, Any]]) -> bool:
    if any(
        group in changes
        for group in (
            "observed.container",
            "observed.python_dependency",
            "observed.dependency_policy",
        )
    ):
        return True
    target_change = path_get(changes, "target")
    if target_change is None:
        return False
    old_target = path_dict(target_change, "old")
    new_target = path_dict(target_change, "new")
    return path_get(old_target, "docker_image_repository") != path_get(
        new_target, "docker_image_repository"
    ) or path_get(old_target, "docker_image_tag") != path_get(
        new_target, "docker_image_tag"
    )


def _severity_categories(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
    changes: dict[str, dict[str, Any]],
) -> list[MatrixDiffSeverity]:
    categories: list[MatrixDiffSeverity] = []
    if _claim_boundary_escalated(old_payload, new_payload):
        categories.append(MatrixDiffSeverity.CLAIM_BOUNDARY_ESCALATION)
    if _validation_downgraded(old_payload, new_payload):
        categories.append(MatrixDiffSeverity.VALIDATION_DOWNGRADE)
    if _mixed_version_drift(old_payload, new_payload):
        categories.append(MatrixDiffSeverity.MIXED_VERSION_DRIFT)
    if _runtime_unavailable(old_payload, new_payload):
        categories.append(MatrixDiffSeverity.RUNTIME_UNAVAILABILITY)
    if _gpu_architecture_drift(old_payload, new_payload):
        categories.append(MatrixDiffSeverity.GPU_ARCHITECTURE_DRIFT)
    if _image_dependency_drift(changes):
        categories.append(MatrixDiffSeverity.IMAGE_DEPENDENCY_DRIFT)
    if changes:
        categories.append(MatrixDiffSeverity.SEMANTIC_CHANGE)
    if not categories:
        categories.append(MatrixDiffSeverity.INFORMATIONAL)
    return sorted(categories, key=lambda severity: _SEVERITY_RANK[severity])


def _report_ref(report: RocmCompatibilityMatrixReport, label: str) -> MatrixReportRef:
    return MatrixReportRef(
        label=label,
        schema_version=report.schema_version,
        generated_at=report.generated_at,
        entry_count=len(report.entries),
    )


def _report_semantic_changes(
    old_report: RocmCompatibilityMatrixReport,
    new_report: RocmCompatibilityMatrixReport,
) -> dict[str, dict[str, Any]]:
    old_metadata = {"generated_at": old_report.generated_at}
    new_metadata = {"generated_at": new_report.generated_at}
    if old_metadata == new_metadata:
        return {}
    return {
        "clock_evidence_metadata": {
            "old": old_metadata,
            "new": new_metadata,
        }
    }


def _entry_diff(
    *,
    key: str,
    target_id: str,
    validation_scope: str,
    kind: MatrixEntryDiffKind,
    severity_categories: list[MatrixDiffSeverity],
    semantic_changes: dict[str, dict[str, Any]],
    old_entry: dict[str, Any] | None = None,
    new_entry: dict[str, Any] | None = None,
) -> MatrixEntryDiff:
    return MatrixEntryDiff(
        diff_key=key,
        target_id=target_id,
        validation_scope=validation_scope,
        kind=kind,
        severity=severity_categories[0],
        severity_categories=severity_categories,
        semantic_changes=semantic_changes,
        old_entry=old_entry,
        new_entry=new_entry,
    )


def diff_matrix_reports(
    old_report: RocmCompatibilityMatrixReport,
    new_report: RocmCompatibilityMatrixReport,
    *,
    old_label: str = "old",
    new_label: str = "new",
) -> MatrixReportDiff:
    """Compare two Matrix reports by target id plus validation scope."""

    old_entries = _entries_by_key(old_report)
    new_entries = _entries_by_key(new_report)
    entry_diffs: list[MatrixEntryDiff] = []
    counts = {"added": 0, "changed": 0, "removed": 0, "unchanged": 0}

    for key in sorted(set(old_entries) | set(new_entries)):
        old_entry = old_entries.get(key)
        new_entry = new_entries.get(key)
        representative = new_entry or old_entry
        assert representative is not None
        _, target_id, validation_scope = _diff_key(representative)

        if old_entry is None:
            assert new_entry is not None
            counts["added"] += 1
            entry_diffs.append(
                _entry_diff(
                    key=key,
                    target_id=target_id,
                    validation_scope=validation_scope,
                    kind=MatrixEntryDiffKind.ADDED,
                    severity_categories=[MatrixDiffSeverity.INFORMATIONAL],
                    semantic_changes={},
                    new_entry=_normalize_entry(new_entry),
                )
            )
            continue

        if new_entry is None:
            counts["removed"] += 1
            entry_diffs.append(
                _entry_diff(
                    key=key,
                    target_id=target_id,
                    validation_scope=validation_scope,
                    kind=MatrixEntryDiffKind.REMOVED,
                    severity_categories=[MatrixDiffSeverity.INFORMATIONAL],
                    semantic_changes={},
                    old_entry=_normalize_entry(old_entry),
                )
            )
            continue

        old_payload = _normalize_entry(old_entry)
        new_payload = _normalize_entry(new_entry)
        changes = _semantic_changes(old_payload, new_payload)
        if not changes:
            counts["unchanged"] += 1
            entry_diffs.append(
                _entry_diff(
                    key=key,
                    target_id=target_id,
                    validation_scope=validation_scope,
                    kind=MatrixEntryDiffKind.UNCHANGED,
                    severity_categories=[MatrixDiffSeverity.INFORMATIONAL],
                    semantic_changes={},
                )
            )
            continue

        counts["changed"] += 1
        entry_diffs.append(
            _entry_diff(
                key=key,
                target_id=target_id,
                validation_scope=validation_scope,
                kind=MatrixEntryDiffKind.CHANGED,
                severity_categories=_severity_categories(
                    old_payload,
                    new_payload,
                    changes,
                ),
                semantic_changes=changes,
            )
        )

    entry_diffs = sorted(
        entry_diffs,
        key=lambda diff: (_SEVERITY_RANK[diff.severity], diff.diff_key),
    )
    return MatrixReportDiff(
        old_report=_report_ref(old_report, old_label),
        new_report=_report_ref(new_report, new_label),
        summary_counts=counts,
        report_semantic_changes=_report_semantic_changes(old_report, new_report),
        entry_diffs=entry_diffs,
    )


def load_matrix_report(
    path_or_payload: str | Path | dict[str, Any],
) -> RocmCompatibilityMatrixReport:
    """Load and validate a ROCm Compatibility Matrix report."""

    if isinstance(path_or_payload, str | Path):
        payload = json.loads(Path(path_or_payload).read_text(encoding="utf-8"))
    else:
        payload = path_or_payload
    try:
        return RocmCompatibilityMatrixReport.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid Matrix report: {exc}") from exc


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    return (
        json.dumps(value, sort_keys=True)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def matrix_report_diff_to_markdown(diff: MatrixReportDiff) -> str:
    """Render a deterministic diagnostic-only Markdown Matrix diff summary."""

    lines = [
        "# Diagnostic-only ROCm Compatibility Matrix diff",
        "",
        (
            "Docker/container evidence does not imply native-host validation, "
            "score authority, paper-parity authority, or leaderboard authority."
        ),
        "",
        "## Summary",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for bucket in ("changed", "added", "removed", "unchanged"):
        lines.append(f"| {bucket} | {diff.summary_counts[bucket]} |")

    if diff.report_semantic_changes:
        lines.extend(["", "## Report Semantic Changes", ""])
        lines.extend(["| Group | Old | New |", "|---|---|---|"])
        for group in sorted(diff.report_semantic_changes):
            change = diff.report_semantic_changes[group]
            lines.append(
                f"| {group} | {_markdown_value(change['old'])} | "
                f"{_markdown_value(change['new'])} |"
            )

    lines.extend(["", "## Entry Diffs", ""])
    for entry_diff in sorted(
        diff.entry_diffs,
        key=lambda item: (_SEVERITY_RANK[item.severity], item.diff_key),
    ):
        categories = ", ".join(
            category.value for category in entry_diff.severity_categories
        )
        lines.extend(
            [
                f"### {entry_diff.diff_key}",
                "",
                f"- Kind: {entry_diff.kind.value}",
                f"- Severity: {entry_diff.severity.value}",
                f"- Severity categories: {categories}",
            ]
        )
        if entry_diff.semantic_changes:
            lines.extend(["", "| Group | Old | New |", "|---|---|---|"])
            for group in sorted(entry_diff.semantic_changes):
                change = entry_diff.semantic_changes[group]
                lines.append(
                    f"| {group} | {_markdown_value(change['old'])} | "
                    f"{_markdown_value(change['new'])} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
