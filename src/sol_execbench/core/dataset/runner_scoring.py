"""Dataset-run scoring report bridge helpers."""

from __future__ import annotations

import json
from pathlib import Path

import sol_execbench.core.dataset.cli_execution as cli_execution
from sol_execbench.core.scoring.amd_score import AmdNativeScore
from sol_execbench.core.scoring.amd_score.reports import (
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report as write_amd_score_report,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact


def build_amd_score_reports_for_problem(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
    sidecar_namespace: str | None = None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    return _build_amd_score_reports_for_problem_impl(
        definition_payload=definition_payload,
        workload_path=workload_path,
        traces_payload=traces_payload,
        trace_ref=trace_ref,
        run_cli_func=cli_execution.run_cli,
        baseline_artifact=baseline_artifact,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        sidecar_namespace=sidecar_namespace,
        derived_sidecar_exclusions=derived_sidecar_exclusions,
    )


def extend_derived_reports_for_problem(
    *,
    amd_scores: list[AmdNativeScore],
    definition_path: Path,
    workload_path: Path,
    traces_path: Path,
    traces_payload: list[dict],
    output_dir: Path,
    baseline_artifact: ScoringBaselineArtifact | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> None:
    """Append requested derived reports and materialize requested sidecars."""
    trace_ref = (
        str(traces_path.relative_to(output_dir))
        if traces_path.is_relative_to(output_dir)
        else str(traces_path)
    )
    sidecar_namespace = str(Path(trace_ref).parent)
    if sidecar_namespace == ".":
        sidecar_namespace = None
    amd_scores.extend(
        build_amd_score_reports_for_problem(
            definition_payload=json.loads(definition_path.read_text()),
            workload_path=workload_path,
            traces_payload=traces_payload,
            trace_ref=trace_ref,
            baseline_artifact=baseline_artifact,
            sol_bound_artifact_dir=sol_bound_artifact_dir,
            solar_derivation_dir=solar_derivation_dir,
            sidecar_namespace=sidecar_namespace,
            derived_sidecar_exclusions=derived_sidecar_exclusions,
        )
    )


_extend_derived_reports_for_problem = extend_derived_reports_for_problem
