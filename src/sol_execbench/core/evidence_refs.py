# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence reference path helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE_SIDECAR_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_sidecar_stem(*parts: str) -> str:
    """Return a deterministic filename stem for untrusted benchmark identifiers."""
    safe_parts: list[str] = []
    changed = False
    for part in parts:
        raw = str(part)
        safe = _SAFE_SIDECAR_COMPONENT.sub("_", raw)
        safe = re.sub(r"_+", "_", safe).strip("._")
        if not safe:
            raise ValueError(f"unsafe sidecar identifier: {raw!r}")
        changed = changed or safe != raw
        safe_parts.append(safe)

    safe_stem = ".".join(safe_parts)
    if changed:
        digest = hashlib.sha256(
            "\0".join(str(part) for part in parts).encode()
        ).hexdigest()[:12]
        safe_stem = f"{safe_stem}.{digest}"
    return safe_stem


def sidecar_stem_for_workload(
    definition_name: str,
    workload_uuid: str,
    *,
    problem_namespace: str | None = None,
) -> str:
    """Return a sidecar stem scoped to a problem when a namespace is available."""
    if problem_namespace:
        return safe_sidecar_stem(problem_namespace, definition_name, workload_uuid)
    return safe_sidecar_stem(definition_name, workload_uuid)


def relative_ref(path: Path, base: Path) -> str:
    """Return a stable path reference relative to *base* when possible."""
    path = path.resolve()
    base = base.resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name
