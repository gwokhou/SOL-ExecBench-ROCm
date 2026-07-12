"""Dataset-run scoring report bridge helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from sol_execbench.core.evidence.baseline_coverage import (
    BaselineCoverageReport,
    validate_baseline_coverage,
)
from sol_execbench.core.scoring.amd_score import AmdNativeScore
from sol_execbench.core.scoring.amd_score.reports import (
    AmdScoreReportRequest,
    build_amd_score_reports_for_problem,
    write_amd_score_report,
)
from sol_execbench.core.scoring.amd_score.derived_artifacts import ResolvedHardwareModel
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.fusion_validation import FusionValidationArtifact
from sol_execbench.core.scoring.official_score import (
    build_official_score_suite_evidence,
    validate_official_aggregation_policy,
)
from sol_execbench.core.scoring.release_baseline import OfficialReleaseBaseline

__all__ = [
    "extend_derived_reports_for_problem",
    "scoring_baseline_coverage_report",
    "write_amd_score_report",
    "write_official_score_report",
]


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
    hardware_model: ResolvedHardwareModel | None = None,
    fusion_validation: FusionValidationArtifact | None = None,
    fusion_validation_ref: str | None = None,
    fusion_validation_sha256: str | None = None,
    fusion_validation_path: Path | None = None,
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
            AmdScoreReportRequest(
                definition_payload=json.loads(definition_path.read_text()),
                workload_path=workload_path,
                traces_payload=traces_payload,
                trace_ref=trace_ref,
                baseline_artifact=baseline_artifact,
                sol_bound_artifact_dir=sol_bound_artifact_dir,
                solar_derivation_dir=solar_derivation_dir,
                sidecar_namespace=sidecar_namespace,
                derived_sidecar_exclusions=derived_sidecar_exclusions,
                hardware_model=hardware_model,
                fusion_validation=fusion_validation,
                fusion_validation_ref=fusion_validation_ref,
                fusion_validation_sha256=fusion_validation_sha256,
                fusion_validation_path=fusion_validation_path,
            )
        )
    )


def write_official_score_report(
    report_path: Path,
    amd_scores: list[AmdNativeScore],
    *,
    aggregation_policy: str,
    coverage_report: BaselineCoverageReport | None = None,
    source_score_ref: str | None = None,
    release_baseline: OfficialReleaseBaseline | None = None,
    expected_workloads: tuple[tuple[str, str], ...] | None = None,
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
        release_baseline=release_baseline,
        expected_workloads=expected_workloads,
        require_release_baseline=True,
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


def scoring_baseline_coverage_report(
    baseline_artifact: ScoringBaselineArtifact,
    amd_scores: list[AmdNativeScore],
) -> BaselineCoverageReport:
    """Validate release-baseline coverage for every emitted score workload.

    The release scoring-baseline format is intentionally smaller than the
    measured-baseline registry format consumed by ``validate_baseline_coverage``.
    Adapt its exact workload entries into that validator's public input shape so
    an official runner sidecar carries the same all-workloads-confirmed gate.
    """
    registry = {
        "expected_workload_keys": list(
            dict.fromkeys(score.workload_uuid for score in amd_scores)
        ),
        "entries": [
            {
                "workload_key": entry.workload_uuid,
                "source": "scoring_baseline",
            }
            for entry in baseline_artifact.entries
        ],
    }
    return validate_baseline_coverage(registry)
