# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release-defined optimized scoring baseline artifacts."""

from __future__ import annotations

import json
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
    *,
    required_schema_version: str | None = None,
) -> ScoringBaselineArtifact:
    """Load a scoring baseline artifact from JSON.

    Callers that emit authoritative evidence can require an explicit schema
    marker. The legacy default remains permissive for non-authoritative score
    derivation inputs.
    """
    payload = json.loads(path.read_text())
    if (
        required_schema_version is not None
        and payload.get("schema_version") != required_schema_version
    ):
        raise ValueError(
            "scoring baseline artifact requires schema_version "
            f"{required_schema_version!r}; got {payload.get('schema_version')!r}"
        )
    return scoring_baseline_artifact_from_dict(payload, source=str(path))


def scoring_baseline_artifact_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
) -> ScoringBaselineArtifact:
    """Parse a baseline artifact payload.

    Expected shape:

    ```json
    {
      "release": "v1.7",
      "entries": [
        {"definition": "name", "workload_uuid": "uuid", "latency_ms": 1.23}
      ]
    }
    ```
    """
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise ValueError("scoring baseline artifact requires an entries list")

    entries: list[ScoringBaselineEntry] = []
    for index, raw in enumerate(entries_payload):
        if not isinstance(raw, dict):
            raise ValueError(f"baseline entry {index} must be an object")
        raw_entry = cast(dict[str, Any], raw)
        try:
            definition = str(raw_entry["definition"])
            workload_uuid = str(raw_entry["workload_uuid"])
            latency_ms = float(raw_entry["latency_ms"])
        except KeyError as exc:
            raise ValueError(f"baseline entry {index} missing {exc.args[0]}") from exc
        if latency_ms <= 0.0:
            raise ValueError(f"baseline entry {index} latency_ms must be positive")
        entries.append(
            ScoringBaselineEntry(
                definition=definition,
                workload_uuid=workload_uuid,
                latency_ms=latency_ms,
                solution=(
                    str(raw_entry["solution"]) if raw_entry.get("solution") else None
                ),
                source=str(raw_entry["source"]) if raw_entry.get("source") else None,
            )
        )

    return ScoringBaselineArtifact(
        entries=tuple(entries),
        release=str(payload.get("release", "unknown")),
        source=str(payload.get("source") or source or "unknown"),
        schema_version=str(
            payload.get("schema_version", BASELINE_ARTIFACT_SCHEMA_VERSION)
        ),
        derived=bool(payload.get("derived", True)),
    )
