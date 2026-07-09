# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Coverage and aggregate status models for SOLAR derivation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.solar_derivation.status import (
    ordered_status_counts as _ordered_status_counts,
)


@dataclass(frozen=True)
class SolarCoverageSourceRef:
    """Group/node-tied provenance reference for SOLAR coverage fields."""

    group_id: str
    node_id: str | None
    tensor_id: str | None
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "node_id": self.node_id,
            "tensor_id": self.tensor_id,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SolarFamilyCoverage:
    """Family-local coverage counts derived from semantic groups."""

    family: str
    group_count: int
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "group_count": self.group_count,
            "status_counts": _ordered_status_counts(self.status_counts),
        }


@dataclass(frozen=True)
class SolarCoveragePattern:
    """Missing or unsupported coverage pattern with affected provenance."""

    pattern: str
    group_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    sources: tuple[SolarCoverageSourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "group_ids": list(self.group_ids),
            "node_ids": list(self.node_ids),
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class SolarCoverageSummary:
    """Machine-readable SOLAR sidecar coverage summary."""

    family_counts: dict[str, int]
    status_counts: dict[str, int]
    families: tuple[SolarFamilyCoverage, ...]
    missing_patterns: tuple[SolarCoveragePattern, ...]
    unsupported_patterns: tuple[SolarCoveragePattern, ...]
    degraded_node_ids: tuple[str, ...]
    unsupported_node_ids: tuple[str, ...]
    estimated_node_ids: tuple[str, ...]
    provenance: tuple[SolarCoverageSourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_counts": dict(sorted(self.family_counts.items())),
            "status_counts": _ordered_status_counts(self.status_counts),
            "families": [family.to_dict() for family in self.families],
            "missing_patterns": [
                pattern.to_dict() for pattern in self.missing_patterns
            ],
            "unsupported_patterns": [
                pattern.to_dict() for pattern in self.unsupported_patterns
            ],
            "degraded_node_ids": list(self.degraded_node_ids),
            "unsupported_node_ids": list(self.unsupported_node_ids),
            "estimated_node_ids": list(self.estimated_node_ids),
            "provenance": [source.to_dict() for source in self.provenance],
        }


@dataclass(frozen=True)
class SolarAggregateStatus:
    """Aggregate score state for SOLAR derivation evidence."""

    status: str
    score_eligible: bool
    reason: str
    group_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score_eligible": self.score_eligible,
            "reason": self.reason,
            "group_ids": list(self.group_ids),
            "node_ids": list(self.node_ids),
            "warnings": list(self.warnings),
        }
