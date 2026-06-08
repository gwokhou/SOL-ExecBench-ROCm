# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Explicit long-tail exclusion configuration for dataset validation runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sol_execbench.core.dataset.checksums import sha256_file

LONG_TAIL_EXCLUSION_SCHEMA_VERSION = "sol_execbench.long_tail_exclusions.v1"
LONG_TAIL_EXCLUSION_STATUS = "excluded_long_tail"
LONG_TAIL_EXCLUSION_REASON = "long_tail_exclusion"


class LongTailExclusionEntry(BaseModel):
    """One explicit problem, workload, or shard exclusion."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["problem", "workload", "shard"]
    problem_id: str
    reason: str
    evidence_ref: str
    workload_uuid: str | None = None
    row_index: int | None = None
    shard_index: int | None = Field(default=None, ge=1)
    owner: str | None = None
    status: str = "temporary"
    expires: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_scope_identity(self) -> "LongTailExclusionEntry":
        if not self.problem_id.strip():
            raise ValueError("problem_id must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.evidence_ref.strip():
            raise ValueError("evidence_ref must be non-empty")
        if self.scope == "problem":
            if (
                self.workload_uuid is not None
                or self.row_index is not None
                or self.shard_index is not None
            ):
                raise ValueError(
                    "problem exclusions must not set workload or shard ids"
                )
        elif self.scope == "workload":
            if self.workload_uuid is None and self.row_index is None:
                raise ValueError(
                    "workload exclusions require workload_uuid or row_index"
                )
            if self.shard_index is not None:
                raise ValueError("workload exclusions must not set shard_index")
        elif self.scope == "shard" and self.shard_index is None:
            raise ValueError("shard exclusions require shard_index")
        return self

    def matches_workload(
        self,
        *,
        problem_id: str,
        workload_uuid: str | None,
        row_index: int,
        workload_shard_size: int | None,
    ) -> bool:
        """Return whether this entry excludes one workload identity."""
        if self.problem_id != problem_id:
            return False
        if self.scope == "problem":
            return True
        if self.scope == "workload":
            if self.workload_uuid is not None and self.workload_uuid == workload_uuid:
                return True
            return self.row_index is not None and self.row_index == row_index
        if workload_shard_size is None:
            return False
        return ((row_index // workload_shard_size) + 1) == self.shard_index

    def closure_note(self) -> str:
        """Return a compact human-readable closure note."""
        return f"{self.reason} (evidence: {self.evidence_ref})"


class LongTailExclusionConfig(BaseModel):
    """Validated long-tail exclusion config."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LONG_TAIL_EXCLUSION_SCHEMA_VERSION
    description: str | None = None
    exclusions: list[LongTailExclusionEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "LongTailExclusionConfig":
        if self.schema_version != LONG_TAIL_EXCLUSION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {LONG_TAIL_EXCLUSION_SCHEMA_VERSION!r}"
            )
        return self

    def summary(self) -> dict[str, Any]:
        """Return deterministic summary metadata for provenance and reports."""
        by_scope: dict[str, int] = {"problem": 0, "workload": 0, "shard": 0}
        problem_ids: set[str] = set()
        for entry in self.exclusions:
            by_scope[entry.scope] += 1
            problem_ids.add(entry.problem_id)
        return {
            "schema_version": self.schema_version,
            "exclusions": len(self.exclusions),
            "by_scope": by_scope,
            "problem_ids": sorted(problem_ids),
            "claim_boundary": {
                "excluded_entries_are_passed": False,
                "full_validation_authority": False,
                "paper_parity": False,
                "leaderboard_result": False,
            },
        }

    def match_workload(
        self,
        *,
        problem_id: str,
        workload_uuid: str | None,
        row_index: int,
        workload_shard_size: int | None,
    ) -> LongTailExclusionEntry | None:
        """Return the first matching exclusion for one workload, if any."""
        for entry in self.exclusions:
            if entry.matches_workload(
                problem_id=problem_id,
                workload_uuid=workload_uuid,
                row_index=row_index,
                workload_shard_size=workload_shard_size,
            ):
                return entry
        return None


@dataclass(frozen=True)
class LongTailExclusionSidecar:
    """Loaded exclusion config plus file metadata."""

    path: Path
    checksum: str
    config: LongTailExclusionConfig

    @property
    def summary(self) -> dict[str, Any]:
        """Return summary metadata including the source checksum."""
        summary = self.config.summary()
        summary["checksum"] = self.checksum
        return summary


def load_long_tail_exclusions(path: Path | None) -> LongTailExclusionSidecar | None:
    """Load and validate an optional long-tail exclusion config file."""
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = LongTailExclusionConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid long-tail exclusion config {path}: {exc}") from exc
    return LongTailExclusionSidecar(
        path=path,
        checksum=sha256_file(path),
        config=config,
    )


def exclusion_closure_metadata(entry: LongTailExclusionEntry) -> dict[str, Any]:
    """Return closure metadata for an excluded workload."""
    return {
        "filter_reasons": [LONG_TAIL_EXCLUSION_REASON],
        "evidence_refs": {"long_tail_exclusion": entry.evidence_ref},
        "notes": [entry.closure_note(), *entry.notes],
    }


def split_excluded_workloads(
    *,
    problem_id: str,
    workload_refs: Iterable[dict[str, Any]],
    exclusions: LongTailExclusionConfig | None,
    workload_shard_size: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], LongTailExclusionEntry]]]:
    """Split ready workload refs into selected and long-tail excluded refs."""
    if exclusions is None:
        return list(workload_refs), []
    selected: list[dict[str, Any]] = []
    excluded: list[tuple[dict[str, Any], LongTailExclusionEntry]] = []
    for ref in workload_refs:
        row_index = int(ref.get("row_index", 0))
        entry = exclusions.match_workload(
            problem_id=problem_id,
            workload_uuid=ref.get("uuid"),
            row_index=row_index,
            workload_shard_size=workload_shard_size,
        )
        if entry is None:
            selected.append(ref)
        else:
            excluded.append((ref, entry))
    return selected, excluded
