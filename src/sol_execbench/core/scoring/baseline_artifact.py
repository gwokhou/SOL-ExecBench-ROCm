# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release-defined optimized scoring baseline artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


BASELINE_ARTIFACT_SCHEMA_VERSION = "sol_execbench.scoring_baseline.v1"


@dataclass(frozen=True)
class ScoringBaselineEntry:
    """Optimized baseline timing for one definition/workload pair."""

    definition: str
    workload_uuid: str
    latency_ms: float
    solution: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError("baseline definition must be non-empty")
        if not isinstance(self.workload_uuid, str) or not self.workload_uuid.strip():
            raise ValueError("baseline workload_uuid must be non-empty")
        if not isinstance(self.latency_ms, (int, float)) or isinstance(
            self.latency_ms, bool
        ):
            raise ValueError("baseline latency_ms must be a number")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError("baseline latency_ms must be positive and finite")

    @property
    def key(self) -> tuple[str, str]:
        return (self.definition, self.workload_uuid)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, object] = {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "latency_ms": self.latency_ms,
        }
        if self.solution:
            payload["solution"] = self.solution
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True)
class ScoringBaselineArtifact:
    """Release-scoped optimized baseline timing evidence."""

    entries: tuple[ScoringBaselineEntry, ...]
    release: str
    source: str
    schema_version: str = BASELINE_ARTIFACT_SCHEMA_VERSION
    derived: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.release, str) or not self.release.strip():
            raise ValueError("baseline release must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("baseline source must be non-empty")
        if self.schema_version != BASELINE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scoring baseline schema {self.schema_version!r}"
            )
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate baseline definition/workload key")

    @property
    def summary(self) -> dict[str, int]:
        return {"entries": len(self.entries)}

    def lookup(
        self, definition: str, workload_uuid: str
    ) -> ScoringBaselineEntry | None:
        for entry in self.entries:
            if entry.definition == definition and entry.workload_uuid == workload_uuid:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "release": self.release,
            "source": self.source,
            "summary": self.summary,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def load_scoring_baseline_artifact(
    path: Path,
) -> ScoringBaselineArtifact:
    """Load the sole supported scoring baseline artifact schema from JSON."""
    payload = json.loads(path.read_text())
    return scoring_baseline_artifact_from_dict(payload, source=str(path))


def scoring_baseline_artifact_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
) -> ScoringBaselineArtifact:
    """Parse a baseline artifact payload.

    Only the exact ``sol_execbench.scoring_baseline.v1`` serialization emitted
    by :meth:`ScoringBaselineArtifact.to_dict` is accepted.
    """
    expected = {
        "schema_version",
        "derived",
        "release",
        "source",
        "summary",
        "entries",
    }
    if set(payload) != expected:
        raise ValueError("scoring baseline artifact has missing or unknown fields")
    if payload["schema_version"] != BASELINE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("scoring baseline artifact has invalid schema_version")
    if not isinstance(payload["derived"], bool):
        raise ValueError("scoring baseline artifact derived must be a boolean")
    if not isinstance(payload["summary"], dict):
        raise ValueError("scoring baseline artifact summary must be an object")
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise ValueError("scoring baseline artifact requires an entries list")

    entries: list[ScoringBaselineEntry] = []
    for index, raw in enumerate(entries_payload):
        if not isinstance(raw, dict):
            raise ValueError(f"baseline entry {index} must be an object")
        raw_entry = cast(dict[str, Any], raw)
        allowed_entry_keys = {
            "definition",
            "workload_uuid",
            "latency_ms",
            "solution",
            "source",
        }
        if not {"definition", "workload_uuid", "latency_ms"}.issubset(
            raw_entry
        ) or not set(raw_entry).issubset(allowed_entry_keys):
            raise ValueError(f"baseline entry {index} has missing or unknown fields")
        try:
            definition = raw_entry["definition"]
            workload_uuid = raw_entry["workload_uuid"]
            latency_ms = raw_entry["latency_ms"]
        except KeyError as exc:
            raise ValueError(f"baseline entry {index} missing {exc.args[0]}") from exc
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"baseline entry {index} definition must be non-empty")
        if not isinstance(workload_uuid, str) or not workload_uuid.strip():
            raise ValueError(f"baseline entry {index} workload_uuid must be non-empty")
        if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
            raise ValueError(f"baseline entry {index} latency_ms must be a number")
        entries.append(
            ScoringBaselineEntry(
                definition=definition,
                workload_uuid=workload_uuid,
                latency_ms=float(latency_ms),
                solution=(
                    str(raw_entry["solution"]) if raw_entry.get("solution") else None
                ),
                source=str(raw_entry["source"]) if raw_entry.get("source") else None,
            )
        )

    artifact = ScoringBaselineArtifact(
        entries=tuple(entries),
        release=payload["release"],
        source=payload["source"],
        schema_version=payload["schema_version"],
        derived=payload["derived"],
    )
    if artifact.to_dict() != payload:
        raise ValueError("scoring baseline artifact fields are inconsistent")
    return artifact
