# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic checksum helpers for dataset sidecar metadata."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.evidence.checksums import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)

__all__ = [
    "CANONICAL_PROBLEM_FILES",
    "checksum_category",
    "sha256_bytes",
    "sha256_file",
    "stable_json_checksum",
]

CANONICAL_PROBLEM_FILES: tuple[str, ...] = (
    "definition.json",
    "reference.py",
    "workload.jsonl",
)
"""Problem files included in category checksums when present."""


def checksum_category(category_dir: Path) -> str | None:
    """Hash canonical problem files below *category_dir* in deterministic order."""

    if not category_dir.is_dir():
        return None

    records: list[dict[str, object]] = []
    for problem_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        for filename in CANONICAL_PROBLEM_FILES:
            path = problem_dir / filename
            if path.is_file():
                records.append(
                    {
                        "path": path.relative_to(category_dir).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    return stable_json_checksum(records)
