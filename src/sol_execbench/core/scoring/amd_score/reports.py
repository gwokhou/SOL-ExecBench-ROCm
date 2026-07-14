"""AMD-native score report helpers for dataset-scale runs."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.amd_score.models import AmdNativeScore
from sol_execbench.core.scoring.amd_score.report_inputs import (
    load_score_report_workloads,
    parse_score_report_definition,
    parse_score_report_trace,
)
from sol_execbench.core.scoring.amd_score.traces import (
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_score.derived_artifacts import (
    ResolvedHardwareModel,
    resolve_derived_score_artifacts,
)
from sol_execbench.core.scoring.fusion_validation import FusionValidationArtifact
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact


@dataclass(frozen=True, kw_only=True)
class AmdScoreReportRequest:
    """Score-domain inputs required to derive reports for one problem run.

    Dataset orchestration resolves file locations and evidence references before
    constructing this request. The scoring layer owns parsing those inputs and
    deriving AMD-native workload scores.
    """

    definition_payload: dict[str, Any]
    workload_path: Path
    traces_payload: list[dict[str, Any]]
    trace_ref: str
    baseline_artifact: ScoringBaselineArtifact | None = None
    sol_bound_artifact_dir: Path | None = None
    solar_derivation_dir: Path | None = None
    sidecar_namespace: str | None = None
    derived_sidecar_exclusions: dict[str, str] | None = None
    hardware_model: ResolvedHardwareModel | None = None
    fusion_validation: FusionValidationArtifact | None = None
    fusion_validation_ref: str | None = None
    fusion_validation_sha256: str | None = None
    fusion_validation_path: Path | None = None


def build_amd_score_reports_for_problem(
    request: AmdScoreReportRequest,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one problem's persisted traces."""
    definition = parse_score_report_definition(request.definition_payload)
    workloads = load_score_report_workloads(request.workload_path)
    if request.hardware_model is None:
        raise ValueError("derived AMD evidence requires an external hardware model")
    scores: list[AmdNativeScore] = []
    derived_sidecar_exclusions = request.derived_sidecar_exclusions or {}

    for trace_index, trace_payload in enumerate(request.traces_payload):
        trace = parse_score_report_trace(trace_payload)
        workload = workloads.get(trace.workload.uuid)
        derived_exclusion = derived_sidecar_exclusions.get(trace.workload.uuid)
        derived_artifacts = resolve_derived_score_artifacts(
            definition=definition,
            workload=workload,
            workload_uuid=trace.workload.uuid,
            hardware_model=request.hardware_model,
            fusion_validation=request.fusion_validation,
            fusion_validation_ref=request.fusion_validation_ref,
            fusion_validation_sha256=request.fusion_validation_sha256,
            fusion_validation_path=request.fusion_validation_path,
            sol_bound_artifact_dir=request.sol_bound_artifact_dir,
            solar_derivation_dir=request.solar_derivation_dir,
            sidecar_namespace=request.sidecar_namespace,
            derived_exclusion=derived_exclusion,
        )
        scores.append(
            score_amd_native_trace_workload(
                trace,
                derived_artifacts.sol_bound_artifact,
                trace_ref=request.trace_ref,
                timing_evidence_ref=request.trace_ref,
                sol_bound_ref=derived_artifacts.sol_bound_ref,
                baseline_ref=(
                    f"{request.baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if request.baseline_artifact
                    and request.baseline_artifact.lookup(
                        definition.name, trace.workload.uuid
                    )
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=request.baseline_artifact,
                hardware_model_ref=request.hardware_model.ref,
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
