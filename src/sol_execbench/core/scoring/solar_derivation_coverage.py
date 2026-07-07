# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Coverage and aggregate status helpers for SOLAR derivation evidence."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraph
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation_models import (
    SolarAggregateStatus,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarEvidenceSource,
    SolarFamilyCoverage,
    SolarSemanticGroupEvidence,
)
from sol_execbench.core.scoring.solar_derivation_status import (
    default_source_boundary as _default_source_boundary_helper,
    derivation_warnings as _build_derivation_warnings,
    empty_status_counts as _empty_status_counts_helper,
    ordered_status_counts as _ordered_status_counts_helper,
    status_for_confidence as _status_for_confidence_helper,
    unique_sorted as _unique_sorted_helper,
)


def _coverage_for_groups(
    groups: tuple[SolarSemanticGroupEvidence, ...],
) -> SolarCoverageSummary:
    family_counts: dict[str, int] = {}
    status_counts = _empty_status_counts()
    family_status_counts: dict[str, dict[str, int]] = {}
    missing_patterns: dict[str, list[SolarSemanticGroupEvidence]] = {}
    unsupported_patterns: dict[str, list[SolarSemanticGroupEvidence]] = {}
    degraded_node_ids: list[str] = []
    unsupported_node_ids: list[str] = []
    estimated_node_ids: list[str] = []
    provenance: list[SolarCoverageSourceRef] = []

    for group in sorted(groups, key=lambda item: item.group_id):
        family_counts[group.family] = family_counts.get(group.family, 0) + 1
        status_counts[group.status] = status_counts.get(group.status, 0) + 1
        family_counts_for_status = family_status_counts.setdefault(
            group.family, _empty_status_counts()
        )
        family_counts_for_status[group.status] = (
            family_counts_for_status.get(group.status, 0) + 1
        )
        provenance.extend(_coverage_source_refs_for_group(group))
        if group.status == "degraded":
            degraded_node_ids.extend(group.node_ids)
        elif group.status == "unscored":
            unsupported_node_ids.extend(group.node_ids)
        for pattern in group.missing_evidence:
            missing_patterns.setdefault(pattern, []).append(group)
        for warning in group.warning_prefixes:
            if warning.startswith("unsupported_operator:"):
                unsupported_patterns.setdefault(warning, []).append(group)
        for evidence in (
            *group.formula_evidence,
            *group.byte_evidence,
            *group.bound_evidence,
        ):
            estimated_node_ids.append(evidence.node_id)

    families = tuple(
        SolarFamilyCoverage(
            family=family,
            group_count=count,
            status_counts=family_status_counts.get(family, _empty_status_counts()),
        )
        for family, count in sorted(family_counts.items())
    )
    return SolarCoverageSummary(
        family_counts=dict(sorted(family_counts.items())),
        status_counts=_ordered_status_counts(status_counts),
        families=families,
        missing_patterns=_coverage_patterns_from_groups(missing_patterns),
        unsupported_patterns=_coverage_patterns_from_groups(unsupported_patterns),
        degraded_node_ids=_unique_sorted(degraded_node_ids),
        unsupported_node_ids=_unique_sorted(unsupported_node_ids),
        estimated_node_ids=_unique_sorted(estimated_node_ids),
        provenance=tuple(sorted(set(provenance), key=_coverage_source_ref_key)),
    )


def _coverage_patterns_from_groups(
    pattern_groups: dict[str, list[SolarSemanticGroupEvidence]],
) -> tuple[SolarCoveragePattern, ...]:
    patterns: list[SolarCoveragePattern] = []
    for pattern, groups in sorted(pattern_groups.items()):
        sorted_groups = tuple(sorted(groups, key=lambda item: item.group_id))
        node_ids: list[str] = []
        sources: list[SolarCoverageSourceRef] = []
        for group in sorted_groups:
            node_ids.extend(group.node_ids)
            sources.append(_coverage_primary_source_ref_for_group(group))
        patterns.append(
            SolarCoveragePattern(
                pattern=pattern,
                group_ids=tuple(group.group_id for group in sorted_groups),
                node_ids=_unique_sorted(node_ids),
                sources=tuple(sorted(set(sources), key=_coverage_source_ref_key)),
            )
        )
    return tuple(patterns)


def _coverage_source_refs_for_group(
    group: SolarSemanticGroupEvidence,
) -> tuple[SolarCoverageSourceRef, ...]:
    refs = [_coverage_source_ref(group.group_id, group.source)]
    refs.extend(
        _coverage_source_ref(group.group_id, subrole.source)
        for subrole in group.subroles
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.formula_evidence
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.byte_evidence
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.bound_evidence
    )
    return tuple(sorted(set(refs), key=_coverage_source_ref_key))


def _coverage_primary_source_ref_for_group(
    group: SolarSemanticGroupEvidence,
) -> SolarCoverageSourceRef:
    for evidence in group.formula_evidence:
        return SolarCoverageSourceRef(
            group_id=group.group_id,
            node_id=evidence.node_id,
            tensor_id=evidence.source.tensor_id,
            kind=evidence.source.kind,
            detail=f"{evidence.formula_kind}:{evidence.formula}",
        )
    for evidence in group.byte_evidence:
        return _coverage_source_ref(group.group_id, evidence.source)
    for evidence in group.bound_evidence:
        return _coverage_source_ref(group.group_id, evidence.source)
    return _coverage_source_ref(group.group_id, group.source)


def _coverage_source_ref(
    group_id: str,
    source: SolarEvidenceSource,
) -> SolarCoverageSourceRef:
    return SolarCoverageSourceRef(
        group_id=group_id,
        node_id=source.node_id,
        tensor_id=source.tensor_id,
        kind=source.kind,
        detail=source.detail,
    )


def _coverage_source_ref_key(
    ref: SolarCoverageSourceRef,
) -> tuple[str, str, str, str, str]:
    return (
        ref.group_id,
        ref.node_id or "",
        ref.tensor_id or "",
        ref.kind,
        ref.detail,
    )


def _aggregate_status_for_groups(
    groups: tuple[SolarSemanticGroupEvidence, ...],
    warnings: tuple[str, ...],
) -> SolarAggregateStatus:
    if not groups:
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="no semantic groups were derived",
            group_ids=(),
            node_ids=(),
            warnings=(),
        )
    group_ids = _unique_sorted([group.group_id for group in groups])
    node_ids = _unique_sorted(
        [node_id for group in groups for node_id in group.node_ids]
    )
    aggregate_warnings = _unique_sorted(
        [
            *warnings,
            *(warning for group in groups for warning in group.warning_prefixes),
        ]
    )
    if any(group.status == "unscored" for group in groups):
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="one or more semantic groups are unsupported",
            group_ids=group_ids,
            node_ids=node_ids,
            warnings=aggregate_warnings,
        )
    if any(group.status == "degraded" for group in groups):
        return SolarAggregateStatus(
            status="degraded",
            score_eligible=True,
            reason="one or more semantic groups have incomplete evidence",
            group_ids=group_ids,
            node_ids=node_ids,
            warnings=aggregate_warnings,
        )
    return SolarAggregateStatus(
        status="scored",
        score_eligible=True,
        reason="all semantic groups are score eligible",
        group_ids=group_ids,
        node_ids=node_ids,
        warnings=aggregate_warnings,
    )


def _derivation_warnings(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[str, ...]:
    return _build_derivation_warnings(graph.warnings, estimates)


def _status_for_confidence(confidence: EstimateConfidence) -> str:
    return _status_for_confidence_helper(confidence)


def _worst_estimate_confidence(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> EstimateConfidence:
    worst = EstimateConfidence.SUPPORTED
    if not estimates:
        return EstimateConfidence.UNSUPPORTED
    for estimate in estimates:
        worst = _worse_confidence(worst, estimate.confidence)
    return worst


def _worse_confidence(
    left: EstimateConfidence,
    right: EstimateConfidence,
) -> EstimateConfidence:
    ranks = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return left if ranks[left] >= ranks[right] else right


def _unique_sorted(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return _unique_sorted_helper(items)


def _empty_status_counts() -> dict[str, int]:
    return _empty_status_counts_helper()


def _ordered_status_counts(status_counts: dict[str, int]) -> dict[str, int]:
    return _ordered_status_counts_helper(status_counts)


def _default_source_boundary() -> dict[str, bool]:
    return _default_source_boundary_helper()
