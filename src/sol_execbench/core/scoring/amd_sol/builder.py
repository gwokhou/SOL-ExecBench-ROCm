# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builder for fusion-aware AMD SOL v3 artifacts."""

from __future__ import annotations

from dataclasses import replace

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.arch_capabilities import (
    ArchIsaBudget,
    derive_arch_capability_budget,
)
from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    estimate_bound_work,
    resolve_architecture_profile_paths,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    build_authority_bound_graph,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    HardwareValidationStatus,
)
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup, build_fusion_groups
from sol_execbench.core.scoring.amd_sol.models import (
    AmdSolAggregateBound,
    AmdSolGroupBound,
    _AmdSolBoundBase,
)
from sol_execbench.core.scoring.amd_sol.math import (
    bound_for_estimate,
    coverage_for_estimates,
    memory_transfer_bound_ms,
    worse_confidence,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _build_amd_sol_bound_base(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str | None = None,
    capability_budget_ref: str | None = None,
    capability_budget: ArchIsaBudget | None = None,
    bound_graph: BoundGraph | None = None,
) -> _AmdSolBoundBase:
    """Build a fusion-aware sidecar using explicit architecture capability evidence."""
    budget = capability_budget or derive_arch_capability_budget(
        hardware_model.architecture
    )
    if budget is not None and budget.architecture != hardware_model.architecture:
        raise ValueError(
            "capability budget architecture must match the hardware model architecture"
        )
    if capability_budget_ref is None and budget is not None:
        capability_budget_ref = (
            f"packaged:arch_capability_budgets/{budget.architecture}.json"
        )
    source_graph = bound_graph or build_authority_bound_graph(definition, workload)
    # Keep fallback graph estimates as diagnostic evidence. Authority gating is
    # applied after estimating/grouping so a failed export cannot erase the
    # operator-family and formula coverage needed to explain the block.
    estimates = resolve_architecture_profile_paths(
        estimate_bound_work(source_graph), hardware_model.architecture
    )
    groups = build_fusion_groups(source_graph, estimates, capability_budget=budget)
    group_bounds = tuple(
        _bound_for_group(group, estimates, hardware_model) for group in groups
    )
    graph = _authority_semantic_graph(source_graph)
    if "semantic_graph_provider_required" in graph.warnings:
        # Estimate families intentionally remain usable for diagnostics.  Mark
        # the aggregation inputs themselves unsupported so no later roofline
        # step can reinterpret that diagnostic work as a scoreable floor.
        groups = tuple(
            replace(
                group,
                confidence=EstimateConfidence.UNSUPPORTED,
                warnings=tuple(
                    dict.fromkeys((*group.warnings, "semantic_graph_provider_required"))
                ),
            )
            for group in groups
        )
        group_bounds = tuple(
            replace(
                bound,
                confidence=EstimateConfidence.UNSUPPORTED,
                warnings=tuple(
                    dict.fromkeys((*bound.warnings, "semantic_graph_provider_required"))
                ),
            )
            for bound in group_bounds
        )
    aggregate = _aggregate_for_groups(group_bounds, hardware_model)
    warnings = _warnings_for_artifact(
        graph.warnings, estimates, groups, group_bounds, aggregate, hardware_model
    )
    return _AmdSolBoundBase(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model_ref=hardware_model_ref,
        hardware_model=hardware_model,
        capability_budget_ref=capability_budget_ref,
        capability_budget=budget,
        bound_graph=graph.to_dict(),
        operator_work_estimates=tuple(estimate.to_dict() for estimate in estimates),
        fusion_groups=groups,
        group_bounds=group_bounds,
        aggregate_bound=aggregate,
        warnings=warnings,
        coverage_summary=coverage_for_estimates(estimates),
    )


def _authority_semantic_graph(graph: BoundGraph) -> BoundGraph:
    """Refuse to score a floor reconstructed without export FakeTensor proof."""
    if graph.nodes and all(
        node.attributes.get("trace_source") == "torch.export" for node in graph.nodes
    ):
        return graph
    warning = "semantic_graph_provider_required"
    return replace(
        graph,
        nodes=tuple(
            replace(
                node,
                # ``degraded`` remains score-eligible for independently
                # evidenced operations, so INEXACT is insufficient here.  A
                # missing semantic graph must make the aggregate unscored.
                confidence=EstimateConfidence.UNSUPPORTED,
                rationale=(f"{node.rationale}; non-export graph is diagnostic-only"),
            )
            for node in graph.nodes
        ),
        warnings=tuple(dict.fromkeys((*graph.warnings, warning))),
    )


def _bound_for_group(
    group: FusionGroup,
    estimates: tuple[OperatorWorkEstimate, ...],
    hardware_model: AmdHardwareModel,
) -> AmdSolGroupBound:
    estimates_by_node = {estimate.node_id: estimate for estimate in estimates}
    members = tuple(estimates_by_node[node_id] for node_id in group.node_ids)
    member_bounds = tuple(
        bound_for_estimate(member, hardware_model) for member in members
    )
    # A graph partition is not proof that member kernels must execute without
    # overlap. The authority floor therefore takes the strongest individual
    # compute constraint. Summing belongs to a fixed-stack prediction model,
    # not to a lower bound that future fusion/scheduling can beat.
    compute_bound_ms = max((bound[0] for bound in member_bounds), default=0.0)
    warnings = list(group.warnings)
    confidence = group.confidence

    if len(members) == 1:
        memory_bound_ms = member_bounds[0][1]
        confidence = worse_confidence(confidence, member_bounds[0][2])
        warnings.extend(member_bounds[0][3])
    else:
        memory_bound_ms, memory_confidence, memory_warnings = _group_memory_bound(
            group, members, hardware_model
        )
        confidence = worse_confidence(confidence, memory_confidence)
        warnings.extend(memory_warnings)
        for _compute, _memory, member_confidence, member_warnings in member_bounds:
            confidence = worse_confidence(confidence, member_confidence)
            warnings.extend(member_warnings)

    limiting_resource = "compute" if compute_bound_ms >= memory_bound_ms else "memory"
    return AmdSolGroupBound(
        group_id=group.group_id,
        pattern_id=group.pattern_id,
        node_ids=group.node_ids,
        compute_bound_ms=compute_bound_ms,
        memory_bound_ms=memory_bound_ms,
        sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
        limiting_resource=limiting_resource,
        confidence=confidence,
        rationale=(
            "physical-fusion roofline bound using proved external traffic"
            if group.pattern_id != "semantic_component.v1" and len(members) > 1
            else "semantic-component lower bound with optimistic intermediate traffic"
            if len(members) > 1
            else members[0].rationale
        ),
        warnings=tuple(sorted(set(warnings))),
    )


def _group_memory_bound(
    group: FusionGroup,
    members: tuple[OperatorWorkEstimate, ...],
    hardware_model: AmdHardwareModel,
) -> tuple[float, EstimateConfidence, tuple[str, ...]]:
    profiles = []
    for member in members:
        if member.total_bytes <= 0.0:
            continue
        profile = hardware_model.resolve_memory(
            member.memory_access or "",
            member.input_dtype or "",
            member.output_dtype or "",
            member.memory_path or "",
        )
        if profile is None or profile.state != "measured" or profile.value is None:
            return 0.0, EstimateConfidence.INEXACT, ("unknown_hardware_profile",)
        profiles.append(profile)
    if not profiles:
        return 0.0, EstimateConfidence.SUPPORTED, ()
    profile_keys = {profile.key for profile in profiles}
    if len(profile_keys) != 1:
        return 0.0, EstimateConfidence.INEXACT, ("fusion_mixed_memory_profiles",)
    profile = profiles[0]
    assert profile.value is not None
    return (
        memory_transfer_bound_ms(group.external_bytes, profile.value),
        profile.confidence,
        (),
    )


def _aggregate_for_groups(
    bounds: tuple[AmdSolGroupBound, ...], hardware_model: AmdHardwareModel
) -> AmdSolAggregateBound:
    # Only independently proved non-overlap barriers may be summed. This
    # reducer currently has no such proof, so aggregate the strongest semantic
    # component constraint. The result is intentionally loose but cannot be
    # invalidated merely because a provider fuses or overlaps current groups.
    sol_bound_ms = max((bound.sol_bound_ms for bound in bounds), default=0.0)
    node_ids = tuple(node_id for bound in bounds for node_id in bound.node_ids)
    if not bounds:
        return AmdSolAggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="missing fusion-group bound evidence",
            node_ids=node_ids,
        )
    if any(bound.confidence == EstimateConfidence.UNSUPPORTED for bound in bounds):
        return AmdSolAggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="unsupported fusion-group evidence present",
            node_ids=node_ids,
        )
    if (
        hardware_model.shape_aware_roofline is None
        or hardware_model.shape_aware_roofline.status
        != HardwareValidationStatus.VALIDATED
    ):
        return AmdSolAggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="scalar roofline is prediction-only without validated shape-aware envelope evidence",
            node_ids=node_ids,
        )
    if (
        any(bound.confidence == EstimateConfidence.INEXACT for bound in bounds)
        or hardware_model.hardware_validation_status
        != HardwareValidationStatus.VALIDATED
        or hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED
        or hardware_model.confidence != EstimateConfidence.SUPPORTED
    ):
        return AmdSolAggregateBound(
            status="degraded",
            scored=True,
            sol_bound_ms=sol_bound_ms,
            reason="inexact fusion-group or provisional hardware evidence present",
            node_ids=node_ids,
        )
    return AmdSolAggregateBound(
        status="scored",
        scored=True,
        sol_bound_ms=sol_bound_ms,
        reason="all semantic-component and hardware evidence is supported",
        node_ids=node_ids,
    )


def _warnings_for_artifact(
    graph_warnings: tuple[str, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    groups: tuple[FusionGroup, ...],
    bounds: tuple[AmdSolGroupBound, ...],
    aggregate: AmdSolAggregateBound,
    hardware_model: AmdHardwareModel,
) -> tuple[str, ...]:
    warnings = [f"graph_warning:{warning}" for warning in graph_warnings]
    for estimate in estimates:
        warnings.extend(
            f"estimate_warning:{estimate.node_id}:{warning}"
            for warning in estimate.warnings
        )
        if estimate.confidence != EstimateConfidence.SUPPORTED:
            warnings.append(
                f"{estimate.confidence.value}_operator:"
                f"{estimate.node_id}:{estimate.op_family.value}"
            )
    for group, bound in zip(groups, bounds, strict=True):
        warnings.extend(
            f"fusion_warning:{group.group_id}:{warning}" for warning in group.warnings
        )
        warnings.extend(
            f"fusion_bound_warning:{bound.group_id}:{warning}"
            for warning in bound.warnings
        )
    if hardware_model.hardware_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "hardware_validation:"
            f"{hardware_model.architecture}:"
            f"{hardware_model.hardware_validation_status.value}"
        )
    if hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "model_validation:"
            f"{hardware_model.architecture}:"
            f"{hardware_model.model_validation_status.value}"
        )
    if aggregate.status != "scored":
        warnings.append(f"aggregate_{aggregate.status}:{aggregate.reason}")
    return tuple(dict.fromkeys(warnings))
