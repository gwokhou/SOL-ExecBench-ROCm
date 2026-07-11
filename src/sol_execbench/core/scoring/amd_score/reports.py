"""AMD-native score report helpers for dataset-scale runs."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_score.report_inputs import (
    load_score_report_workloads,
    parse_score_report_definition,
    parse_score_report_trace,
)
from sol_execbench.core.scoring.amd_score.derived_artifacts import (
    ResolvedHardwareModel,
    resolve_derived_score_artifacts,
    resolve_hardware_model_from_trace_payloads,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact

RunCliFunc = Callable[..., list[dict] | None]


def _build_amd_score_reports_for_problem_impl(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    run_cli_func: RunCliFunc,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
    sidecar_namespace: str | None = None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
    hardware_model: ResolvedHardwareModel | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    _ = run_cli_func
    definition = parse_score_report_definition(definition_payload)
    workloads = load_score_report_workloads(workload_path)
    hardware_model = hardware_model or resolve_hardware_model_from_trace_payloads(
        traces_payload
    )
    scores: list[AmdNativeScore] = []
    derived_sidecar_exclusions = derived_sidecar_exclusions or {}

    for trace_index, trace_payload in enumerate(traces_payload):
        trace = parse_score_report_trace(trace_payload)
        workload = workloads.get(trace.workload.uuid)
        derived_exclusion = derived_sidecar_exclusions.get(trace.workload.uuid)
        derived_artifacts = resolve_derived_score_artifacts(
            definition=definition,
            workload=workload,
            workload_uuid=trace.workload.uuid,
            hardware_model=hardware_model,
            sol_bound_artifact_dir=sol_bound_artifact_dir,
            solar_derivation_dir=solar_derivation_dir,
            sidecar_namespace=sidecar_namespace,
            derived_exclusion=derived_exclusion,
        )
        scores.append(
            score_amd_native_trace_workload(
                trace,
                derived_artifacts.sol_bound_artifact,
                trace_ref=trace_ref,
                timing_evidence_ref=trace_ref,
                sol_bound_ref=derived_artifacts.sol_bound_ref,
                baseline_ref=(
                    f"{baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if baseline_artifact
                    and baseline_artifact.lookup(definition.name, trace.workload.uuid)
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=hardware_model.ref,
                solar_derivation=derived_artifacts.solar_derivation,
                derived_evidence_refs=derived_artifacts.evidence_refs,
            )
        )
        if trace_index % 16 == 0:
            gc.collect()
    return scores


def write_amd_score_report(
    report_path: Path,
    amd_scores: list[AmdNativeScore],
    *,
    problem_count: int,
    baseline_entry_count: int,
) -> None:
    """Write the AMD-native suite score report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_amd_native_suite_report(
        amd_scores,
        baseline_summary={
            "problems": problem_count,
            "scores": len(amd_scores),
            "baseline_entries": baseline_entry_count,
        },
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
