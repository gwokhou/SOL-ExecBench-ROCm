"""JSON I/O helpers for runtime evidence sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from sol_execbench.core.compatibility import MatrixEntry
from sol_execbench.core.evidence.runtime_evidence_builders import build_aggregate_report
from sol_execbench.core.evidence.runtime_evidence_models import ModelDumpable


def write_json_payload(path: Path, payload: object) -> Path:
    """Write deterministic JSON with a trailing newline."""
    if hasattr(payload, "model_dump"):
        data = cast(ModelDumpable, payload).model_dump(mode="json")
    else:
        data = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return path


def write_matrix_entry(path: Path, entry: MatrixEntry) -> Path:
    """Write a per-Target Matrix Entry JSON sidecar."""
    return write_json_payload(path, entry)


def load_matrix_entry(path: Path) -> MatrixEntry:
    """Load a per-Target Matrix Entry JSON sidecar."""
    return MatrixEntry.model_validate(json.loads(path.read_text()))


def write_aggregate_report(path: Path, entries: list[MatrixEntry]) -> Path:
    """Write an aggregate compatibility matrix JSON report."""
    return write_json_payload(path, build_aggregate_report(entries))
