"""Dataset-run scoring report bridge helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import sol_execbench.core.dataset.cli_execution as cli_execution
from sol_execbench.core.evidence.baseline_coverage import BaselineCoverageReport
from sol_execbench.core.scoring.amd_score import AmdNativeScore
from sol_execbench.core.scoring.amd_score.reports import (
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.official_score import (
    build_official_score_suite_evidence,
    validate_official_aggregation_policy,
)

__all__ = [
    "build_amd_score_reports_for_problem",
    "extend_derived_reports_for_problem",
    "write_amd_score_report",
    "write_official_score_report",
]


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


def write_official_score_report(
    report_path: Path,
    amd_scores: list[AmdNativeScore],
    *,
    aggregation_policy: str,
    coverage_report: BaselineCoverageReport | None = None,
    source_score_ref: str | None = None,
) -> None:
    """Write canonical official-score evidence without partial artifacts."""
    normalized_policy = validate_official_aggregation_policy(aggregation_policy)
    if normalized_policy is None:
        raise ValueError(
            f"Unsupported official aggregation policy: {aggregation_policy!r}"
        )

    suite = build_official_score_suite_evidence(
        amd_scores,
        aggregation_policy=normalized_policy,
        source_score_refs_by_workload_uuid=(
            {score.workload_uuid: source_score_ref for score in amd_scores}
            if source_score_ref
            else None
        ),
        coverage_report=coverage_report,
    )
    serialized = json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(serialized)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(report_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
