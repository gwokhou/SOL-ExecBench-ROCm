"""AMD-native score report helpers for dataset-scale runs."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_sol import default_amd_hardware_models
from sol_execbench.core.scoring.amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.solar_derivation import (
    SolarAggregateStatus,
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)

RunCliFunc = Callable[..., list[dict]]


def _hardware_model_key_from_trace_payloads(traces_payload: Sequence[dict]) -> str:
    """Return the first known AMD gfx key without retaining parsed traces."""
    known = default_amd_hardware_models()
    for payload in traces_payload:
        try:
            hardware = str(payload["evaluation"]["environment"].get("hardware", ""))
        except (KeyError, TypeError, AttributeError):
            continue
        for key in known:
            if key in hardware:
                return key
    return "gfx1200"


def _read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _minimal_amd_sol_bound_v2_from_payload(
    payload: dict,
) -> AmdSolBoundV2Artifact | None:
    """Parse only score-critical fields from a persisted AMD SOL v2 sidecar."""
    if payload.get("schema_version") != AMD_SOL_V2_SCHEMA_VERSION:
        return None
    aggregate_payload = payload.get("aggregate_bound")
    hardware_payload = payload.get("hardware_model")
    coverage_payload = payload.get("coverage_summary")
    if not (
        isinstance(aggregate_payload, dict)
        and isinstance(hardware_payload, dict)
        and isinstance(coverage_payload, dict)
    ):
        return None

    try:
        aggregate = AmdSolV2AggregateBound(
            status=str(aggregate_payload["status"]),
            scored=bool(aggregate_payload["scored"]),
            sol_bound_ms=float(aggregate_payload["sol_bound_ms"]),
            reason=str(aggregate_payload["reason"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
        )
        coverage = AmdSolV2CoverageSummary(
            total_ops=int(coverage_payload.get("total_ops", 0)),
            supported_ops=int(coverage_payload.get("supported_ops", 0)),
            inexact_ops=int(coverage_payload.get("inexact_ops", 0)),
            unsupported_ops=int(coverage_payload.get("unsupported_ops", 0)),
            op_family_counts={
                str(key): int(value)
                for key, value in dict(
                    coverage_payload.get("op_family_counts", {})
                ).items()
            },
            confidence_counts_by_family={
                str(family): {
                    str(key): int(value) for key, value in dict(counts).items()
                }
                for family, counts in dict(
                    coverage_payload.get("confidence_counts_by_family", {})
                ).items()
            },
            worst_confidence=EstimateConfidence(
                str(coverage_payload.get("worst_confidence", "unsupported"))
            ),
        )
        hardware_model = amd_hardware_model_from_dict(
            hardware_payload,
            source="AMD SOL v2 sidecar hardware_model",
        )
    except (KeyError, TypeError, ValueError):
        return None

    return AmdSolBoundV2Artifact(
        definition=str(payload.get("definition", "")),
        workload_uuid=str(payload.get("workload_uuid", "")),
        hardware_model_ref=(
            str(payload["hardware_model_ref"])
            if payload.get("hardware_model_ref") is not None
            else None
        ),
        hardware_model=hardware_model,
        bound_graph={},
        operator_work_estimates=(),
        op_bounds=(),
        aggregate_bound=aggregate,
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        coverage_summary=coverage,
    )


def _minimal_solar_aggregate_from_payload(
    payload: dict,
) -> SolarAggregateStatus | None:
    aggregate_payload = payload.get("aggregate_status")
    if not isinstance(aggregate_payload, dict):
        return None
    try:
        return SolarAggregateStatus(
            status=str(aggregate_payload["status"]),
            score_eligible=bool(aggregate_payload["score_eligible"]),
            reason=str(aggregate_payload["reason"]),
            group_ids=tuple(str(item) for item in aggregate_payload["group_ids"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
            warnings=tuple(str(item) for item in aggregate_payload["warnings"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    _ = run_cli_func
    definition = Definition(**definition_payload)
    workloads = {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }
    hardware_models = default_amd_hardware_models()
    hardware_model_key = _hardware_model_key_from_trace_payloads(traces_payload)
    hardware_model = hardware_models[hardware_model_key]
    scores: list[AmdNativeScore] = []
    derived_sidecar_exclusions = derived_sidecar_exclusions or {}

    for trace_index, trace_payload in enumerate(traces_payload):
        trace = Trace(**trace_payload)
        workload = workloads.get(trace.workload.uuid)
        derived_exclusion = derived_sidecar_exclusions.get(trace.workload.uuid)
        sidecar_stem = (
            sidecar_stem_for_workload(
                definition.name,
                trace.workload.uuid,
                problem_namespace=sidecar_namespace,
            )
            if workload is not None
            else None
        )
        sol_bound_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:amd_sol_bound_v2"
        )
        sol_bound_path = (
            sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol-v2.json"
            if sol_bound_artifact_dir is not None and sidecar_stem is not None
            else None
        )
        artifact = None
        if sol_bound_path is not None and sol_bound_path.exists():
            existing_payload = _read_json_object(sol_bound_path)
            if existing_payload is not None:
                artifact = _minimal_amd_sol_bound_v2_from_payload(existing_payload)
                if artifact is not None:
                    sol_bound_ref = str(sol_bound_path)
        if artifact is None and workload is not None and derived_exclusion is None:
            artifact = build_amd_sol_bound_v2_artifact(
                definition,
                workload,
                hardware_model,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
            )
        solar_derivation = None
        derived_evidence_refs = None
        solar_derivation_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:solar_derivation"
        )
        solar_derivation_path = (
            solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
            if solar_derivation_dir is not None and sidecar_stem is not None
            else None
        )
        if solar_derivation_path is not None and solar_derivation_path.exists():
            existing_payload = _read_json_object(solar_derivation_path)
            if existing_payload is not None:
                solar_derivation = _minimal_solar_aggregate_from_payload(
                    existing_payload
                )
                if solar_derivation is not None:
                    solar_derivation_ref = str(solar_derivation_path)
        if (
            workload is not None
            and solar_derivation_dir is not None
            and derived_exclusion is None
        ):
            solar_derivation_dir.mkdir(parents=True, exist_ok=True)
            assert solar_derivation_path is not None
            if solar_derivation is None:
                generated = build_solar_derivation_evidence(definition, workload)
                generated_payload = generated.to_dict()
                solar_derivation_path.write_text(
                    json.dumps(generated_payload, indent=2)
                )
                solar_derivation_ref = str(solar_derivation_path)
                try:
                    solar_derivation = solar_derivation_from_dict(generated_payload)
                except ValueError as exc:
                    solar_derivation = None
                    derived_evidence_refs = {"solar_derivation_parse_error": str(exc)}
            else:
                solar_derivation_ref = str(solar_derivation_path)
            derived_evidence_refs = {
                "formula": f"{solar_derivation_ref}#groups.formula_evidence",
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
                "coverage": f"{solar_derivation_ref}#coverage_summary",
                "score_eligibility": f"{solar_derivation_ref}#aggregate_status",
                **(derived_evidence_refs or {}),
            }
        elif derived_exclusion is not None:
            derived_evidence_refs = {
                "derived_sidecar_exclusion": derived_exclusion,
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
            }
        if (
            artifact is not None
            and sol_bound_path is not None
            and sol_bound_artifact_dir is not None
        ):
            sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
            if not sol_bound_path.exists():
                sol_bound_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            sol_bound_ref = str(sol_bound_path)
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifact,
                trace_ref=trace_ref,
                timing_evidence_ref=trace_ref,
                sol_bound_ref=sol_bound_ref,
                baseline_ref=(
                    f"{baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if baseline_artifact
                    and baseline_artifact.lookup(definition.name, trace.workload.uuid)
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
                solar_derivation=solar_derivation,
                derived_evidence_refs=derived_evidence_refs,
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
