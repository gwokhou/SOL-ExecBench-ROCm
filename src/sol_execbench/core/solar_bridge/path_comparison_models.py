# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed results for fail-closed SOLAR IR-path comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum

from solar.schema_versions import SOLAR_PATH_COMPARISON_SCHEMA_VERSION


class DifferenceCategory(StrEnum):
    """Reviewed causes for differences between fixed SOLAR IR paths."""

    EXTRACTION_TOPOLOGY_LOSS = "extraction_topology_loss"
    NORMALIZATION_DIFFERENCE = "normalization_difference"
    DIALECT_DECOMPOSITION_DIFFERENCE = (
        "legitimate_dialect_decomposition_difference"
    )
    RESOURCE_MODEL_BUG = "resource_model_bug"
    FORMAL_BOUND_POLICY_DIFFERENCE = "formal_bound_policy_difference"


class PathComparisonStatus(StrEnum):
    """Aggregate and per-workload path-comparison states."""

    EXACT_MATCH = "exact_match"
    MATCHED_WITH_DIALECT_DIFFERENCES = "matched_with_dialect_differences"
    DIFFERENT = "different"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonSection:
    """One independently reviewed comparison dimension."""

    match: bool
    classification: DifferenceCategory | None
    differences: tuple[str, ...]
    values: dict[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadPathComparison:
    """Separated accounting dimensions for one dual-ready workload."""

    workload: str
    external_reference_io: ComparisonSection
    model_io_accounting: ComparisonSection
    mandatory_resource_work: ComparisonSection
    fusion_intermediate_accounting: ComparisonSection
    formal_bound: ComparisonSection

    @property
    def status(self) -> PathComparisonStatus:
        """Return whether authoritative accounting agrees for this workload."""
        authoritative = (
            self.external_reference_io,
            self.model_io_accounting,
            self.mandatory_resource_work,
            self.formal_bound,
        )
        if any(not section.match for section in authoritative):
            return PathComparisonStatus.DIFFERENT
        if not self.fusion_intermediate_accounting.match:
            return PathComparisonStatus.MATCHED_WITH_DIALECT_DIFFERENCES
        return PathComparisonStatus.EXACT_MATCH

    @property
    def categories(self) -> tuple[DifferenceCategory, ...]:
        """Return unique reviewed difference categories in stable order."""
        sections = (
            self.external_reference_io,
            self.model_io_accounting,
            self.mandatory_resource_work,
            self.fusion_intermediate_accounting,
            self.formal_bound,
        )
        return tuple(
            dict.fromkeys(
                section.classification
                for section in sections
                if section.classification is not None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible mapping."""
        payload = asdict(self)
        payload["status"] = self.status
        payload["categories"] = list(self.categories)
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossPathComparisonResult:
    """Corpus-level result without favorable-path selection semantics."""

    roots: dict[str, dict[str, object]]
    missing_by_path: dict[str, tuple[str, ...]]
    comparisons: tuple[WorkloadPathComparison, ...]

    @property
    def coverage_complete(self) -> bool:
        """Return whether both roots contain the same workload set."""
        return not any(self.missing_by_path.values())

    @property
    def status(self) -> PathComparisonStatus:
        """Return the fail-closed corpus comparison state."""
        if not self.coverage_complete:
            return PathComparisonStatus.INCOMPLETE
        statuses = {comparison.status for comparison in self.comparisons}
        if PathComparisonStatus.DIFFERENT in statuses:
            return PathComparisonStatus.DIFFERENT
        if PathComparisonStatus.MATCHED_WITH_DIALECT_DIFFERENCES in statuses:
            return PathComparisonStatus.MATCHED_WITH_DIALECT_DIFFERENCES
        return PathComparisonStatus.EXACT_MATCH

    @property
    def authoritative_match(self) -> bool:
        """Return whether every dual-ready authoritative dimension matches."""
        return bool(self.comparisons) and all(
            comparison.status is not PathComparisonStatus.DIFFERENT
            for comparison in self.comparisons
        )

    def to_dict(self) -> dict[str, object]:
        """Return the versioned repository-owned comparison report."""
        statuses = Counter(item.status for item in self.comparisons)
        categories = Counter(
            category
            for item in self.comparisons
            for category in item.categories
        )
        return {
            "schema_version": SOLAR_PATH_COMPARISON_SCHEMA_VERSION,
            "status": self.status,
            "policy": {
                "favorable_path_selection": False,
                "fallback": False,
                "numeric_replay_proves_equal_accounting": False,
                "differences_fail_closed": True,
            },
            "coverage_complete": self.coverage_complete,
            "authoritative_match_on_dual_ready": self.authoritative_match,
            "roots": self.roots,
            "missing_by_path": {
                key: list(value) for key, value in self.missing_by_path.items()
            },
            "dual_ready": len(self.comparisons),
            "summary": {
                "statuses": dict(sorted(statuses.items())),
                "categories": dict(sorted(categories.items())),
                "external_reference_io_mismatches": sum(
                    not item.external_reference_io.match
                    for item in self.comparisons
                ),
                "model_io_accounting_mismatches": sum(
                    not item.model_io_accounting.match
                    for item in self.comparisons
                ),
                "mandatory_resource_work_mismatches": sum(
                    not item.mandatory_resource_work.match
                    for item in self.comparisons
                ),
                "fusion_intermediate_accounting_mismatches": sum(
                    not item.fusion_intermediate_accounting.match
                    for item in self.comparisons
                ),
                "formal_bound_mismatches": sum(
                    not item.formal_bound.match for item in self.comparisons
                ),
            },
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


__all__ = [
    "ComparisonSection",
    "CrossPathComparisonResult",
    "DifferenceCategory",
    "PathComparisonStatus",
    "WorkloadPathComparison",
]
