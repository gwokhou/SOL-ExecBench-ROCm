# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Source reference helpers for paper denominator reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.checksums import sha256_file
from sol_execbench.core.dataset.paper_denominator.models import (
    PaperDenominatorSourceRef,
)


def _checksum(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> str | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            checksum = value.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | str | None,
    ref: str | None = None,
    checksum_keys: tuple[str, ...] = (),
    checksum: str | None = None,
) -> PaperDenominatorSourceRef:
    return PaperDenominatorSourceRef(
        path=str(path) if path else None,
        ref=ref,
        schema_version=payload.get("schema_version") if payload else None,
        checksum=checksum or _checksum(payload, checksum_keys),
    )


def _artifact_source(
    artifact: PaperDenominatorSourceRef | dict[str, Any] | str | Path,
) -> PaperDenominatorSourceRef:
    if isinstance(artifact, PaperDenominatorSourceRef):
        return artifact
    if isinstance(artifact, str | Path):
        path = Path(artifact)
        return PaperDenominatorSourceRef(
            path=str(path),
            checksum=sha256_file(path) if path.is_file() else None,
        )
    path_value = artifact.get("path")
    checksum = artifact.get("checksum")
    if checksum is None and path_value and Path(path_value).is_file():
        checksum = sha256_file(Path(path_value))
    return PaperDenominatorSourceRef(
        path=str(path_value) if path_value else None,
        ref=artifact.get("ref"),
        schema_version=artifact.get("schema_version"),
        checksum=checksum,
    )


def _record_ref(record: dict[str, Any]) -> str:
    problem = str(record.get("problem_id") or record.get("problem_path") or "unknown")
    uuid = record.get("workload_uuid")
    row = record.get("row_index")
    return f"{problem}#{uuid or row}"
