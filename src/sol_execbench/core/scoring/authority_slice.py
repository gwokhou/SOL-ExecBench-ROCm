# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Freeze a score-independent, auditable AMD authority workload slice."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum
from sol_execbench.core.scoring.amd_bound_sanity.models import (
    AMD_AUTHORITY_AUDIT_POLICY_VERSION,
    AmdBoundSanityReport,
)

AUTHORITY_SLICE_SCHEMA_VERSION = "sol_execbench.authority_slice_manifest.v1"
DEFAULT_SELECTION_POLICY_VERSION = "amd-authority-scored-evidence-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POST_FREEZE_BLOCKERS = frozenset({"missing_baseline"})
_POST_FREEZE_WARNING_MARKERS = (
    "provisional baseline",
    "provide a scoring baseline artifact",
)


class AuthoritySliceSelectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = DEFAULT_SELECTION_POLICY_VERSION
    score_independent: bool = True
    required_amd_sol_status: str = "scored"
    required_solar_status: str = "scored"
    require_warning_free_bound: bool = True
    require_profiler_timing: bool = True
    require_exact_validated_hardware_profile: bool = True


class AuthoritySliceWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: str
    workload_uuid: str
    category: str | None = None
    problem_id: str | None = None
    row_index: int | None = None
    evidence_refs: dict[str, str] = Field(default_factory=dict)
    prerequisite_evidence_summary: dict[str, Any] = Field(default_factory=dict)
    blocker_codes: tuple[str, ...] = ()


class AuthoritySliceExcludedWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: str
    workload_uuid: str
    category: str | None = None
    problem_id: str | None = None
    row_index: int | None = None
    blocker_codes: tuple[str, ...]


class AuthoritySliceManifest(BaseModel):
    """Immutable workload selection made before baseline/candidate measurement."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = AUTHORITY_SLICE_SCHEMA_VERSION
    source_suite_manifest_sha256: str
    selection_policy: AuthoritySliceSelectionPolicy = Field(
        default_factory=AuthoritySliceSelectionPolicy
    )
    workloads: tuple[AuthoritySliceWorkload, ...]
    excluded: tuple[AuthoritySliceExcludedWorkload, ...]
    manifest_checksum: DatasetManifestChecksum | None = None

    @model_validator(mode="after")
    def _validate_manifest(self) -> "AuthoritySliceManifest":
        if not _SHA256_RE.fullmatch(self.source_suite_manifest_sha256):
            raise ValueError("source_suite_manifest_sha256 must be lowercase SHA-256")
        selected = {(row.definition, row.workload_uuid) for row in self.workloads}
        excluded = {(row.definition, row.workload_uuid) for row in self.excluded}
        if len(selected) != len(self.workloads) or len(excluded) != len(self.excluded):
            raise ValueError("authority slice contains duplicate workload keys")
        if selected & excluded:
            raise ValueError("workload cannot be both selected and excluded")
        if any(row.blocker_codes for row in self.workloads):
            raise ValueError("selected authority workloads cannot contain blockers")
        if any(not row.blocker_codes for row in self.excluded):
            raise ValueError("excluded authority workloads require blocker codes")
        return self

    @property
    def source_workload_count(self) -> int:
        return len(self.workloads) + len(self.excluded)

    def with_checksum(self) -> "AuthoritySliceManifest":
        return self.model_copy(
            update={
                "manifest_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "manifest_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)


def build_authority_slice_manifest(
    *,
    suite_workloads: Sequence[Mapping[str, Any]],
    source_suite_manifest_sha256: str,
    sanity_report: AmdBoundSanityReport | Mapping[str, Any],
    selection_policy: AuthoritySliceSelectionPolicy | None = None,
) -> AuthoritySliceManifest:
    """Select only workloads whose pre-measurement authority blockers are empty."""
    report = (
        sanity_report
        if isinstance(sanity_report, AmdBoundSanityReport)
        else AmdBoundSanityReport.model_validate(sanity_report)
    )
    audit_policy_valid = (
        report.authority_audit_policy_version == AMD_AUTHORITY_AUDIT_POLICY_VERSION
    )
    policy = selection_policy or AuthoritySliceSelectionPolicy()
    audit = {
        (row.definition or row.problem_id, row.workload_uuid): row
        for row in report.workloads
    }
    selected: list[AuthoritySliceWorkload] = []
    excluded: list[AuthoritySliceExcludedWorkload] = []
    seen: set[tuple[str, str]] = set()

    for index, raw in enumerate(suite_workloads):
        definition = raw.get("definition")
        workload_uuid = raw.get("workload_uuid")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"suite workload {index} has invalid definition")
        if not isinstance(workload_uuid, str) or not workload_uuid.strip():
            raise ValueError(f"suite workload {index} has invalid workload_uuid")
        key = (definition, workload_uuid)
        if key in seen:
            raise ValueError(f"duplicate suite workload key {key!r}")
        seen.add(key)
        row = audit.get(key)
        blockers = (
            set(row.blocker_codes if row is not None else ()) - _POST_FREEZE_BLOCKERS
        )
        if not audit_policy_valid:
            blockers.add("authority_audit_policy_mismatch")
        if row is None:
            blockers.add("missing_authority_audit")
        else:
            if row.source_statuses.amd_sol_status != policy.required_amd_sol_status:
                blockers.add("amd_sol_not_scored")
            if row.source_statuses.solar_status != policy.required_solar_status:
                blockers.add("solar_not_scored")
            bound_warnings = tuple(
                warning
                for warning in row.warnings
                if not any(
                    marker in warning.lower() for marker in _POST_FREEZE_WARNING_MARKERS
                )
            )
            if policy.require_warning_free_bound and bound_warnings:
                blockers.add("bound_evidence_warning")
            if policy.require_profiler_timing and "timing" not in row.evidence_refs:
                blockers.add("missing_profiler_timing")

        category = _optional_string(raw.get("category")) or (
            row.category if row else None
        )
        problem_id = _optional_string(raw.get("problem_id")) or (
            row.problem_id if row else None
        )
        row_index = (
            _optional_int(raw.get("row_index"))
            if raw.get("row_index") is not None
            else (row.row_index if row else None)
        )
        if blockers:
            excluded.append(
                AuthoritySliceExcludedWorkload(
                    definition=definition,
                    workload_uuid=workload_uuid,
                    category=category,
                    problem_id=problem_id,
                    row_index=row_index,
                    blocker_codes=tuple(sorted(blockers)),
                )
            )
        else:
            assert row is not None
            selected.append(
                AuthoritySliceWorkload(
                    definition=definition,
                    workload_uuid=workload_uuid,
                    category=category,
                    problem_id=problem_id,
                    row_index=row_index,
                    evidence_refs=dict(sorted(row.evidence_refs.items())),
                    prerequisite_evidence_summary={
                        "amd_sol_status": row.source_statuses.amd_sol_status,
                        "solar_status": row.source_statuses.solar_status,
                        "diagnostic_status": row.diagnostic_status,
                        "diagnostic_flags": row.diagnostic_flags,
                    },
                )
            )

    return AuthoritySliceManifest(
        source_suite_manifest_sha256=source_suite_manifest_sha256,
        selection_policy=policy,
        workloads=tuple(
            sorted(selected, key=lambda row: (row.definition, row.workload_uuid))
        ),
        excluded=tuple(
            sorted(excluded, key=lambda row: (row.definition, row.workload_uuid))
        ),
    ).with_checksum()


def authority_slice_manifest_from_dict(
    payload: Mapping[str, Any],
) -> AuthoritySliceManifest:
    manifest = AuthoritySliceManifest.model_validate(payload)
    if manifest.schema_version != AUTHORITY_SLICE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported authority slice schema: {manifest.schema_version}"
        )
    expected = manifest.with_checksum().manifest_checksum
    if manifest.manifest_checksum is None or manifest.manifest_checksum != expected:
        raise ValueError("authority slice manifest checksum mismatch")
    return manifest


def write_authority_slice_manifest(
    manifest: AuthoritySliceManifest, path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "AUTHORITY_SLICE_SCHEMA_VERSION",
    "DEFAULT_SELECTION_POLICY_VERSION",
    "AuthoritySliceExcludedWorkload",
    "AuthoritySliceManifest",
    "AuthoritySliceSelectionPolicy",
    "AuthoritySliceWorkload",
    "authority_slice_manifest_from_dict",
    "build_authority_slice_manifest",
    "write_authority_slice_manifest",
]
