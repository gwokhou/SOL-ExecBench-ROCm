# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bound the implementation inputs that invalidate analysis reuse."""

from __future__ import annotations

from enum import StrEnum
from itertools import chain
from pathlib import Path

from solar.artifacts import sha256_bytes, stable_json_checksum
from solar.artifacts.bundle import read_regular_file

MAX_RULE_FILES = 512
MAX_RULE_BYTES = 16 * 1024 * 1024
RULE_SOURCE_FOLDERS = ("analysis", "rocm", "pipeline")
RULE_SOURCE_FILES = (
    "precision.py",
    "types.py",
    "schema_versions.py",
    "reuse.py",
    "reuse_contracts.py",
    "explore.py",
    "scan.py",
    "scan_contracts.py",
    "order.py",
    "order_contracts.py",
    "order_execution.py",
    "ir/triangular.py",
    "ir/access_calls.py",
)


class RuleGroup(StrEnum):
    """Stages whose implementation identities affect derived analysis."""

    COUNTS = "counts"
    RESOURCES = "resources"
    CAPACITY = "capacity"
    PRICING = "pricing"


def analysis_rules_identity() -> dict[RuleGroup, str]:
    """Hash bounded installed sources; a rule edit cannot retain old counts."""
    root = Path(__file__).resolve().parents[1]
    paths = chain(
        *(root.joinpath(name).rglob("*.py") for name in RULE_SOURCE_FOLDERS),
        (root / name for name in RULE_SOURCE_FILES),
    )
    digests: dict[str, str] = {}
    remaining = MAX_RULE_BYTES
    for path in paths:
        if len(digests) >= MAX_RULE_FILES:
            raise ValueError("analysis rules exceed the file budget")
        payload = read_regular_file(path, max_bytes=remaining)
        remaining -= len(payload)
        digests[path.relative_to(root).as_posix()] = sha256_bytes(payload)
    # Count reuse is intentionally conservative across the whole analysis
    # implementation. Attribution and capacity are always recomputed as well.
    return {
        RuleGroup.COUNTS: stable_json_checksum(digests),
        RuleGroup.RESOURCES: stable_json_checksum(
            {
                k: v
                for k, v in digests.items()
                if k.startswith(
                    ("rocm/", "analysis/resource", "analysis/graph_")
                )
            }
        ),
        RuleGroup.CAPACITY: stable_json_checksum(
            {
                k: v
                for k, v in digests.items()
                if k.startswith(
                    (
                        "analysis/orojenesis",
                        "analysis/formal",
                        "analysis/fusion",
                    )
                )
            }
        ),
        RuleGroup.PRICING: stable_json_checksum(
            {
                k: v
                for k, v in digests.items()
                if k.startswith("rocm/")
                or k
                in {
                    "analysis/views.py",
                    "analysis/reporting.py",
                    "analysis/graph_analyzer.py",
                }
            }
        ),
    }
