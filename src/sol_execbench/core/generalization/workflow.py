# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Plan, seal, and aggregate benchmark-owned generalization evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel

from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusTargetViewManifest,
    GeneratedWorkloadRecord,
    WorkloadRole,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.generalization.metrics import (
    correctness_metric,
    fast_metric,
    paired_metric_delta,
    speedup_metric,
    summarize_results,
    workload_drift,
)
from sol_execbench.core.generalization.models import (
    BOOTSTRAP_REPLICATES,
    FAST_THRESHOLDS,
    AgentTrack,
    CandidateDeclaration,
    CellResultStatus,
    CellWorkloadResult,
    ComparisonMetrics,
    CorpusAgentView,
    GeneralizationArtifactKind,
    GeneralizationReportStatus,
    HardwareContextView,
    HardwareGeneralizationCell,
    HardwareGeneralizationPlan,
    HardwareGeneralizationReport,
    HardwareShift,
    PlannedCell,
    StratumMetrics,
    TrainingExposureDeclaration,
    WorkloadDrift,
)
from sol_execbench.core.generalization.solutions import (
    portability_digest,
    solution_digest,
)
from sol_execbench.core.generalization.views import (
    build_agent_view,
    classify_hardware_shift,
    hardware_facts,
)
from sol_execbench.core.integrity import stable_json_checksum


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedStudy:
    """Plan and Agent projections emitted by the planning stage."""

    plan: HardwareGeneralizationPlan
    agent_views: dict[str, CorpusAgentView]


def build_study_plan(
    *,
    study_id: str,
    manifest: CorpusManifest,
    manifest_digest: str,
    exposure: TrainingExposureDeclaration,
    targets: tuple[tuple[str, CorpusTargetViewManifest], ...],
    include_anonymous: bool = False,
) -> PlannedStudy:
    """Build a fixed matrix without orchestrating Agent solution generation."""
    _validate_comparison_group(manifest_digest, targets)
    cells: list[PlannedCell] = []
    views: dict[str, CorpusAgentView] = {}
    for target_id, target_view in targets:
        full = hardware_facts(target_view, study_target_id=target_id)
        full_view = build_agent_view(manifest, target_view, full)
        views[_view_key(target_id, full.context_view)] = full_view
        cells.extend(_core_cells(exposure, target_view, full_view))
        if include_anonymous:
            anonymous = hardware_facts(
                target_view,
                study_target_id=_anonymous_target_id(study_id, target_view),
                anonymous=True,
            )
            anonymous_view = build_agent_view(manifest, target_view, anonymous)
            views[
                _view_key(anonymous.study_target_id, anonymous.context_view)
            ] = anonymous_view
            cells.append(
                _planned_cell(
                    target_view,
                    anonymous_view,
                    classify_hardware_shift(exposure, target_view),
                    AgentTrack.TARGET_CONDITIONED,
                    study_target_id=target_id,
                )
            )
    payload: dict[str, object] = {
        "schema_version": DatasetArtifactSchema.HARDWARE_GENERALIZATION,
        "artifact_kind": GeneralizationArtifactKind.PLAN,
        "study_id": study_id,
        "exposure": exposure,
        "corpus_manifest_digest": manifest_digest,
        "profile_set": targets[0][1].requested_profiles,
        "cells": tuple(cells),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    payload["plan_digest"] = stable_json_checksum(_json_payload(payload))
    plan = HardwareGeneralizationPlan.model_validate(payload)
    return PlannedStudy(plan=plan, agent_views=views)


def seal_cell(
    *,
    plan: HardwareGeneralizationPlan,
    cell_id: str,
    target_view: CorpusTargetViewManifest,
    manifest: CorpusManifest,
    solutions: tuple[Solution, ...],
    traces: tuple[Trace, ...],
    observed_gfx_target: str,
    used_holdout_feedback: bool = False,
) -> HardwareGeneralizationCell:
    """Seal traces from the existing evaluator; never regenerate workloads."""
    _validate_plan_digest(plan)
    planned = _find_planned_cell(plan, cell_id)
    _validate_cell_target(planned, target_view, observed_gfx_target)
    if planned.zero_shot and used_holdout_feedback:
        raise ValueError("zero-shot cell cannot use target holdout feedback")
    candidates = _candidate_declarations(
        planned,
        manifest,
        solutions,
        used_holdout_feedback,
    )
    candidate_ids = {item.semantic_id for item in candidates}
    traces_by_uuid = _trace_index(traces, observed_gfx_target)
    entries = {item.semantic_id: item for item in manifest.entries}
    results: list[CellWorkloadResult] = []
    failures: list[str] = []
    for workload in target_view.workloads:
        if workload.role is WorkloadRole.SMOKE:
            continue
        result, failure = _workload_result(
            workload,
            entries[workload.semantic_id],
            workload.semantic_id in candidate_ids,
            traces_by_uuid.get(workload.uuid),
        )
        results.append(result)
        if failure:
            failures.append(failure)
    payload: dict[str, object] = {
        "schema_version": DatasetArtifactSchema.HARDWARE_GENERALIZATION,
        "artifact_kind": GeneralizationArtifactKind.CELL,
        "plan_digest": plan.plan_digest,
        "cell_id": cell_id,
        "observed_gfx_target": observed_gfx_target.strip().lower(),
        "candidates": candidates,
        "results": tuple(results),
        "evaluator_failures": tuple(sorted(failures)),
    }
    payload["cell_digest"] = stable_json_checksum(_json_payload(payload))
    return HardwareGeneralizationCell.model_validate(payload)


def aggregate_study(
    *,
    plan: HardwareGeneralizationPlan,
    manifest: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
    cells: tuple[HardwareGeneralizationCell, ...],
) -> HardwareGeneralizationReport:
    """Aggregate valid cells; incomplete evidence cannot yield conclusions."""
    _validate_plan_digest(plan)
    _validate_target_views(plan, target_views)
    cell_map = _validate_cells(plan, cells)
    invalid = {key for key, cell in cell_map.items() if cell.evaluator_failures}
    planned_ids = {item.cell_id for item in plan.cells}
    missing = tuple(sorted((planned_ids - cell_map.keys()) | invalid))
    valid = {key: cell for key, cell in cell_map.items() if key not in invalid}
    common_ids = _common_semantic_ids(valid.values())
    status, conclusion = _report_status(plan, valid, missing)
    payload: dict[str, object] = {
        "schema_version": DatasetArtifactSchema.HARDWARE_GENERALIZATION,
        "artifact_kind": GeneralizationArtifactKind.REPORT,
        "plan_digest": plan.plan_digest,
        "status": status,
        "generalization_conclusion_allowed": conclusion,
        "missing_cell_ids": missing,
        "target_full": _cell_summaries(plan, valid, None),
        "common_support": _cell_summaries(plan, valid, common_ids),
        "stratified": _stratified_summaries(plan, valid),
        "comparisons": _comparisons(plan, valid, common_ids),
        "workload_drift": _drifts(plan, manifest, target_views),
    }
    payload["report_digest"] = stable_json_checksum(_json_payload(payload))
    return HardwareGeneralizationReport.model_validate(payload)


def _core_cells(
    exposure: TrainingExposureDeclaration,
    target: CorpusTargetViewManifest,
    agent_view: CorpusAgentView,
) -> tuple[PlannedCell, PlannedCell]:
    shift = classify_hardware_shift(exposure, target)
    return (
        _planned_cell(
            target,
            agent_view,
            shift,
            AgentTrack.TARGET_CONDITIONED,
        ),
        _planned_cell(
            target,
            agent_view,
            shift,
            AgentTrack.SOLUTION_PORTABILITY,
        ),
    )


def _planned_cell(
    target: CorpusTargetViewManifest,
    agent_view: CorpusAgentView,
    shift: HardwareShift,
    track: AgentTrack,
    study_target_id: str | None = None,
) -> PlannedCell:
    facts = agent_view.hardware_facts
    target_id = study_target_id or facts.study_target_id
    cell_id = f"{target_id}--{track.value}--{facts.context_view.value}"
    return PlannedCell(
        cell_id=cell_id,
        study_target_id=target_id,
        track=track,
        context_view=facts.context_view,
        shift=shift,
        target_view_digest=target.workload_view_digest,
        generation_cohort_id=target.generation_cohort_id,
        hardware_context_digest=facts.context_digest,
        agent_view_digest=agent_view.agent_view_digest,
    )


def _validate_comparison_group(
    manifest_digest: str,
    targets: tuple[tuple[str, CorpusTargetViewManifest], ...],
) -> None:
    if not targets:
        raise ValueError("study requires at least one target")
    target_ids = [target_id for target_id, _ in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("study target IDs must be unique")
    if any(
        view.source_manifest_sha256 != manifest_digest for _, view in targets
    ):
        raise ValueError("target view corpus digest differs from study")
    profiles = {view.requested_profiles for _, view in targets}
    protocols = {view.generator_version for _, view in targets}
    if len(profiles) != 1 or len(protocols) != 1:
        raise ValueError("targets require identical profiles and protocol")


def _view_key(target_id: str, context: HardwareContextView) -> str:
    return f"{target_id}--{context.value}"


def _anonymous_target_id(
    study_id: str,
    target_view: CorpusTargetViewManifest,
) -> str:
    digest = stable_json_checksum(
        {
            "study_id": study_id,
            "target_view_digest": target_view.workload_view_digest,
        }
    )
    return f"target-{digest[:12]}"


def _find_planned_cell(
    plan: HardwareGeneralizationPlan,
    cell_id: str,
) -> PlannedCell:
    try:
        return next(item for item in plan.cells if item.cell_id == cell_id)
    except StopIteration as exc:
        raise ValueError(f"cell is not present in plan: {cell_id}") from exc


def _validate_cell_target(
    planned: PlannedCell,
    target: CorpusTargetViewManifest,
    observed_gfx: str,
) -> None:
    if target.workload_view_digest != planned.target_view_digest:
        raise ValueError("target view differs from immutable plan")
    if target.generation_cohort_id != planned.generation_cohort_id:
        raise ValueError("generation cohort differs from immutable plan")
    if target.target.gfx_target.lower() != observed_gfx.strip().lower():
        raise ValueError("observed gfx target differs from planned target")


def _candidate_declarations(
    planned: PlannedCell,
    manifest: CorpusManifest,
    solutions: tuple[Solution, ...],
    used_holdout_feedback: bool,
) -> tuple[CandidateDeclaration, ...]:
    semantic_ids = {
        entry.problem_name: entry.semantic_id for entry in manifest.entries
    }
    if len({solution.definition for solution in solutions}) != len(solutions):
        raise ValueError("solutions must be unique by Definition")
    declarations = []
    for solution in solutions:
        try:
            semantic_id = semantic_ids[solution.definition]
        except KeyError as exc:
            raise ValueError(
                "solution Definition is absent from corpus"
            ) from exc
        declarations.append(
            CandidateDeclaration(
                semantic_id=semantic_id,
                solution_digest=solution_digest(solution),
                portability_digest=portability_digest(solution),
                agent_view_digest=planned.agent_view_digest,
                hardware_context_digest=planned.hardware_context_digest,
                used_holdout_feedback=used_holdout_feedback,
            )
        )
    return tuple(sorted(declarations, key=lambda item: item.semantic_id))


def _trace_index(
    traces: tuple[Trace, ...], observed_gfx: str
) -> dict[str, Trace]:
    indexed: dict[str, Trace] = {}
    for trace in traces:
        evaluation = trace.evaluation
        if evaluation is None:
            raise ValueError("cell trace must contain an evaluation")
        if evaluation.environment.hardware.lower() != observed_gfx.lower():
            raise ValueError("trace hardware differs from observed target")
        if trace.workload.uuid in indexed:
            raise ValueError("duplicate workload trace in cell")
        indexed[trace.workload.uuid] = trace
    return indexed


def _workload_result(
    workload: GeneratedWorkloadRecord,
    entry: CorpusEntry,
    has_solution: bool,
    trace: Trace | None,
) -> tuple[CellWorkloadResult, str | None]:
    if not has_solution:
        outcome = (CellResultStatus.MISSING_SOLUTION, False, False, None)
        failure = None
    elif trace is None:
        outcome = (CellResultStatus.EVALUATOR_FAILURE, False, False, None)
        failure = f"missing_trace:{workload.uuid}"
    else:
        if trace.definition != entry.problem_name:
            raise ValueError("trace Definition differs from workload record")
        outcome = _trace_outcome(trace)
        failure = (
            f"invalid_reference:{workload.uuid}"
            if outcome[0] is CellResultStatus.EVALUATOR_FAILURE
            else None
        )
    status, compiled, correct, speedup = outcome
    result = CellWorkloadResult(
        semantic_id=workload.semantic_id,
        workload_uuid=workload.uuid,
        slot_id=workload.slot_id,
        role=workload.role,
        regime=workload.regime,
        operation_family=entry.operation_family,
        profiles=entry.profiles,
        status=status,
        compiled=compiled,
        correct=correct,
        speedup=speedup,
    )
    return result, failure


def _trace_outcome(
    trace: Trace,
) -> tuple[CellResultStatus, bool, bool, float | None]:
    evaluation = trace.evaluation
    if evaluation is None:
        raise ValueError("cell trace must contain an evaluation")
    status = evaluation.status
    if status is EvaluationStatus.PASSED:
        if evaluation.performance is None:
            raise ValueError("passed Trace must contain performance evidence")
        return (
            CellResultStatus.PASSED,
            True,
            True,
            evaluation.performance.speedup_factor,
        )
    if status is EvaluationStatus.INVALID_REFERENCE:
        return CellResultStatus.EVALUATOR_FAILURE, False, False, None
    if status is EvaluationStatus.COMPILE_ERROR:
        return CellResultStatus.COMPILE_ERROR, False, False, None
    if status is EvaluationStatus.TIMEOUT:
        return CellResultStatus.TIMEOUT, True, False, None
    if status is EvaluationStatus.REWARD_HACK:
        return CellResultStatus.REWARD_HACK, True, False, None
    if status is EvaluationStatus.RUNTIME_ERROR:
        result = (
            CellResultStatus.CANDIDATE_OOM
            if "out of memory" in evaluation.log.lower()
            else CellResultStatus.RUNTIME_ERROR
        )
        return result, True, False, None
    return CellResultStatus.INCORRECT, True, False, None


def _validate_cells(
    plan: HardwareGeneralizationPlan,
    cells: tuple[HardwareGeneralizationCell, ...],
) -> dict[str, HardwareGeneralizationCell]:
    result: dict[str, HardwareGeneralizationCell] = {}
    planned = {item.cell_id for item in plan.cells}
    for cell in cells:
        if cell.plan_digest != plan.plan_digest or cell.cell_id not in planned:
            raise ValueError("cell does not belong to supplied plan")
        payload = cell.model_dump(mode="json")
        observed_digest = payload.pop("cell_digest")
        if stable_json_checksum(payload) != observed_digest:
            raise ValueError("cell digest does not match semantic content")
        if cell.cell_id in result:
            raise ValueError("duplicate cell artifact")
        result[cell.cell_id] = cell
    _validate_portability(plan, result)
    return result


def _validate_plan_digest(plan: HardwareGeneralizationPlan) -> None:
    payload = plan.model_dump(mode="json")
    observed_digest = payload.pop("plan_digest")
    if stable_json_checksum(payload) != observed_digest:
        raise ValueError("plan digest does not match semantic content")


def _validate_target_views(
    plan: HardwareGeneralizationPlan,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    expected = {item.study_target_id for item in plan.cells}
    if target_views.keys() != expected:
        raise ValueError("aggregate requires exactly every planned target view")
    digests = {
        item.study_target_id: item.target_view_digest for item in plan.cells
    }
    if any(
        view.workload_view_digest != digests[target_id]
        for target_id, view in target_views.items()
    ):
        raise ValueError("aggregate target view differs from immutable plan")


def _validate_portability(
    plan: HardwareGeneralizationPlan,
    cells: dict[str, HardwareGeneralizationCell],
) -> None:
    portable_ids = {
        item.cell_id
        for item in plan.cells
        if item.track is AgentTrack.SOLUTION_PORTABILITY
    }
    digests: dict[str, set[str]] = defaultdict(set)
    for cell_id in portable_ids & cells.keys():
        for candidate in cells[cell_id].candidates:
            digests[candidate.semantic_id].add(candidate.portability_digest)
    changed = sorted(key for key, values in digests.items() if len(values) > 1)
    if changed:
        raise ValueError(
            f"portability payload changed for: {', '.join(changed)}"
        )


def _common_semantic_ids(
    cells: Iterable[HardwareGeneralizationCell],
) -> set[str]:
    supports = [
        {result.semantic_id for result in cell.results} for cell in cells
    ]
    return set.intersection(*supports) if supports else set()


def _cell_summaries(
    plan: HardwareGeneralizationPlan,
    cells: dict[str, HardwareGeneralizationCell],
    semantic_ids: set[str] | None,
) -> dict[str, StratumMetrics]:
    return {
        cell_id: _summarize_cell(plan, cell, semantic_ids)
        for cell_id, cell in sorted(cells.items())
    }


def _summarize_cell(
    plan: HardwareGeneralizationPlan,
    cell: HardwareGeneralizationCell,
    semantic_ids: set[str] | None,
) -> StratumMetrics:
    rows = (
        row
        for row in cell.results
        if semantic_ids is None or row.semantic_id in semantic_ids
    )
    return summarize_results(
        rows,
        seed_digest=plan.plan_digest,
        replicates=plan.bootstrap_replicates,
    )


def _stratified_summaries(
    plan: HardwareGeneralizationPlan,
    cells: dict[str, HardwareGeneralizationCell],
) -> dict[str, StratumMetrics]:
    strata: dict[str, list[CellWorkloadResult]] = defaultdict(list)
    for cell_id, cell in cells.items():
        for row in cell.results:
            dimensions = (
                f"role={row.role.value}",
                f"family={row.operation_family.value}",
                f"regime={row.regime.value}",
                *(f"profile={profile.value}" for profile in row.profiles),
            )
            for dimension in dimensions:
                strata[f"{cell_id}|{dimension}"].append(row)
    return {
        key: summarize_results(
            rows,
            seed_digest=plan.plan_digest,
            replicates=plan.bootstrap_replicates,
        )
        for key, rows in sorted(strata.items())
    }


def _comparisons(
    plan: HardwareGeneralizationPlan,
    cells: dict[str, HardwareGeneralizationCell],
    common_ids: set[str],
) -> tuple[ComparisonMetrics, ...]:
    planned = {item.cell_id: item for item in plan.cells}
    comparisons = []
    for cell_id, target in sorted(cells.items()):
        target_plan = planned[cell_id]
        if target_plan.shift is HardwareShift.SEEN_HARDWARE_SEEN_CAPACITY:
            continue
        control = _matching_control(target_plan, planned, cells)
        if control is not None:
            comparisons.append(
                _comparison(
                    plan, control, target, common_ids, target_plan.track
                )
            )
    return tuple(comparisons)


def _matching_control(
    target: PlannedCell,
    planned: dict[str, PlannedCell],
    cells: dict[str, HardwareGeneralizationCell],
) -> HardwareGeneralizationCell | None:
    candidates = [
        cells[item.cell_id]
        for item in planned.values()
        if item.cell_id in cells
        and item.shift is HardwareShift.SEEN_HARDWARE_SEEN_CAPACITY
        and item.track is target.track
        and item.context_view is target.context_view
    ]
    candidates.sort(key=lambda cell: cell.cell_id)
    return candidates[0] if candidates else None


def _comparison(
    plan: HardwareGeneralizationPlan,
    control: HardwareGeneralizationCell,
    target: HardwareGeneralizationCell,
    common_ids: set[str],
    track: AgentTrack,
) -> ComparisonMetrics:
    left = [row for row in control.results if row.semantic_id in common_ids]
    right = [row for row in target.results if row.semantic_id in common_ids]
    speedup = paired_metric_delta(
        left,
        right,
        speedup_metric,
        seed_digest=plan.plan_digest,
        replicates=plan.bootstrap_replicates,
    )
    return ComparisonMetrics(
        control_cell_id=control.cell_id,
        target_cell_id=target.cell_id,
        common_support_definitions=len(common_ids),
        correctness_delta=paired_metric_delta(
            left,
            right,
            correctness_metric,
            seed_digest=plan.plan_digest,
            replicates=plan.bootstrap_replicates,
        ),
        fast_p_deltas={
            str(value): paired_metric_delta(
                left,
                right,
                fast_metric(value),
                seed_digest=plan.plan_digest,
                replicates=plan.bootstrap_replicates,
            )
            for value in FAST_THRESHOLDS
        },
        conditional_speedup_delta=speedup,
        portability_performance_delta=(
            speedup if track is AgentTrack.SOLUTION_PORTABILITY else None
        ),
    )


def _drifts(
    plan: HardwareGeneralizationPlan,
    manifest: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> tuple[WorkloadDrift, ...]:
    controls = sorted(
        {
            item.study_target_id
            for item in plan.cells
            if item.shift is HardwareShift.SEEN_HARDWARE_SEEN_CAPACITY
        }
    )
    if not controls:
        return ()
    control_id = controls[0]
    source = target_views[control_id]
    return tuple(
        workload_drift(
            manifest,
            source,
            target,
            source_target_id=control_id,
            target_target_id=target_id,
        )
        for target_id, target in sorted(target_views.items())
        if target_id != control_id
    )


def _report_status(
    plan: HardwareGeneralizationPlan,
    cells: dict[str, HardwareGeneralizationCell],
    missing: tuple[str, ...],
) -> tuple[GeneralizationReportStatus, bool]:
    if missing:
        return GeneralizationReportStatus.INCOMPLETE, False
    planned = {item.cell_id: item for item in plan.cells}
    shifts = {planned[cell_id].shift for cell_id in cells}
    seen = HardwareShift.SEEN_HARDWARE_SEEN_CAPACITY
    eligible = seen in shifts and any(shift is not seen for shift in shifts)
    status = (
        GeneralizationReportStatus.COMPLETE
        if eligible
        else GeneralizationReportStatus.DESCRIPTIVE
    )
    return status, eligible


def _json_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_payload(item) for item in value]
    return value


__all__ = ["PlannedStudy", "aggregate_study", "build_study_plan", "seal_cell"]
