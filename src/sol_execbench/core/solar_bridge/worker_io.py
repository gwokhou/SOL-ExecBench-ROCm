# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed response serialization for isolated SOLAR workers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value


def write_worker_response(
    path: Path,
    response: Mapping[str, object],
    fallback: Mapping[str, object],
) -> bool:
    """Write a response atomically, replacing unserializable output with failure."""
    try:
        atomic_write_json_value(path, dict(response))
    except Exception as exc:  # noqa: BLE001 -- serialization fallback boundary
        failure = dict(fallback)
        failure["message"] = f"worker response serialization failed: {exc}"[
            :4096
        ]
        atomic_write_json_value(path, failure)
        return False
    return True


__all__ = ["write_worker_response"]
