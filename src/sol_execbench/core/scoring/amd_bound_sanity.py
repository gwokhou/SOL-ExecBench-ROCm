# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AMD SOL/SOLAR bound sanity diagnostic sidecar helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.dataset.checksums import stable_json_checksum
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


AMD_BOUND_SANITY_SCHEMA_VERSION = "sol_execbench.amd_bound_sanity.v1"

SANITY_STATUS_KEYS = (
    "scored",
    "degraded",
    "unscored",
    "unsupported",
    "provisional",
    "missing_evidence",
)
PRIMARY_STATUS_ORDER = (
    "missing_evidence",
    "unsupported",
    "unscored",
    "degraded",
    "provisional",
    "scored",
)
SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)
CLAIM_BOUNDARY_TEXT = (
    "This is a diagnostic existing evidence sanity report for AMD SOL/SOLAR bound "
    "risk review: not upstream SOLAR equivalence, not AMD SOL/SOLAR model "
    "validation, not paper parity, not leaderboard authority, not score authority "
    "upgrade, not CDNA 3 validation, not MI300X validation, not CDNA 4 validation, "
    "not native-host validation, and not new-hardware validation."
)


class AmdBoundSanitySourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class AmdBoundSanitySources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_refs: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    execution_closure: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )
    amd_sol_artifacts: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    solar_artifacts: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    amd_score_report: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )
    compatibility_matrix: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )


class AmdBoundSanityArtifactAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_refs: int = 0
    execution_closure: bool = False
    amd_sol_artifacts: int = 0
    solar_artifacts: int = 0
    amd_score_report: bool = False
    compatibility_matrix: bool = False


class AmdBoundSanityStatusTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scored: int = 0
    degraded: int = 0
    unscored: int = 0
    unsupported: int = 0
    provisional: int = 0
    missing_evidence: int = 0

    def add(self, status: str) -> None:
        setattr(self, status, getattr(self, status) + 1)


class AmdBoundSanitySourceStatuses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    closure_status: str | None = None
    amd_sol_status: str | None = None
    solar_status: str | None = None
    amd_score_supported: bool | None = None


class AmdBoundSanityWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = "unknown"
    problem_id: str
    problem_path: str | None = None
    definition: str | None = None
    workload_uuid: str
    row_index: int | None = None
    diagnostic_status: str
    diagnostic_flags: list[str]
    source_statuses: AmdBoundSanitySourceStatuses
    amd_score_supported: bool | None = None
    coverage_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    evidence_refs: dict[str, str] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)


class AmdBoundSanityEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    count: int
    example_refs: list[str]
    next_evidence: str


class AmdBoundSanityClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provisional_rdna4_model_risk: bool = False
    upstream_solar_equivalence: bool = False
    amd_sol_model_validation: bool = False
    solar_model_validation: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    score_authority_upgrade: bool = False
    cdna3_validation: bool = False
    mi300x_validation: bool = False
    cdna4_validation: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class AmdBoundSanityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = AMD_BOUND_SANITY_SCHEMA_VERSION
    created_at: str
    sources: AmdBoundSanitySources
    artifact_availability: AmdBoundSanityArtifactAvailability
    status_totals: AmdBoundSanityStatusTotals
    amd_sol_aggregate_statuses: dict[str, int]
    solar_aggregate_statuses: dict[str, int]
    coverage_summary: dict[str, Any]
    warnings: list[str]
    evidence_gaps: list[AmdBoundSanityEvidenceGap]
    workloads: list[AmdBoundSanityWorkload]
    claim_boundary: AmdBoundSanityClaimBoundary = Field(
        default_factory=AmdBoundSanityClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "AmdBoundSanityReport":
        payload = self.model_dump(mode="json")
        payload["report_checksum"] = None
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_json_checksum(payload)
                )
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_amd_bound_sanity_report(
    *,
    trace_refs: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path] | None = None,
    execution_closure: dict[str, Any] | None = None,
    amd_sol_artifacts: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path] | None = None,
    solar_artifacts: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    compatibility_matrix: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> AmdBoundSanityReport:
    trace_refs = trace_refs or []
    amd_sol_artifacts = amd_sol_artifacts or []
    solar_artifacts = solar_artifacts or []
    source_paths = source_paths or {}

    closure_records = (execution_closure or {}).get("records", [])
    workloads: dict[str, dict[str, Any]] = {}
    evidence_gap_groups: dict[str, dict[str, Any]] = {}

    for record in closure_records:
        if not isinstance(record, dict):
            continue
        uuid = str(record.get("workload_uuid") or record.get("row_index") or "unknown")
        workload = _ensure_workload(workloads, uuid, record)
        workload["source_statuses"]["closure_status"] = _optional_str(
            record.get("closure_status")
        )
        workload["evidence_refs"].update(
            {
                str(key): str(value)
                for key, value in (record.get("evidence_refs") or {}).items()
                if value
            }
        )
        if record.get("trace_ref"):
            workload["evidence_refs"]["trace"] = str(record["trace_ref"])
        for gap in record.get("evidence_gaps", []):
            _add_gap(workload, str(gap))

    amd_sol_statuses: dict[str, int] = {}
    for artifact in _payload_artifacts(amd_sol_artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, artifact)
        aggregate = _dict_value(artifact.get("aggregate_bound"))
        status = _optional_str(aggregate.get("status"))
        if status:
            workload["source_statuses"]["amd_sol_status"] = status
            amd_sol_statuses[status] = amd_sol_statuses.get(status, 0) + 1
        if artifact.get("coverage_summary") is not None:
            workload["coverage_summary"]["amd_sol"] = artifact["coverage_summary"]
        _extend_unique(workload["warnings"], _warnings_from(artifact))
        if _is_degraded_status(status, artifact.get("coverage_summary"), workload["warnings"]):
            workload["diagnostic_flags"].add("degraded")
        if status == "unscored":
            workload["diagnostic_flags"].add("unscored")
        if _contains_unsupported(workload["warnings"]):
            workload["diagnostic_flags"].add("unsupported")
        if _provisional_artifact(artifact):
            workload["diagnostic_flags"].add("provisional")

    solar_statuses: dict[str, int] = {}
    for artifact in _payload_artifacts(solar_artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, artifact)
        aggregate = _dict_value(artifact.get("aggregate_status"))
        status = _optional_str(aggregate.get("status"))
        if status:
            workload["source_statuses"]["solar_status"] = status
            solar_statuses[status] = solar_statuses.get(status, 0) + 1
        if artifact.get("coverage_summary") is not None:
            workload["coverage_summary"]["solar"] = artifact["coverage_summary"]
        _extend_unique(workload["warnings"], _warnings_from(aggregate))
        _extend_unique(workload["warnings"], _warnings_from(artifact))
        if status == "degraded":
            workload["diagnostic_flags"].add("degraded")
        if status == "unscored":
            workload["diagnostic_flags"].add("unscored")
        if _contains_unsupported(workload["warnings"]):
            workload["diagnostic_flags"].add("unsupported")
        if _contains_provisional(workload["warnings"]):
            workload["diagnostic_flags"].add("provisional")

    for score in (amd_score_report or {}).get("scores", []):
        if not isinstance(score, dict):
            continue
        uuid = _artifact_uuid(score)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, score)
        supported = score.get("supported")
        if isinstance(supported, bool):
            workload["source_statuses"]["amd_score_supported"] = supported
            workload["amd_score_supported"] = supported
        _extend_unique(workload["warnings"], _warnings_from(score))
        if supported is False:
            if _contains_unsupported(workload["warnings"]):
                workload["diagnostic_flags"].add("unsupported")
            else:
                workload["diagnostic_flags"].add("unscored")
        if isinstance(score.get("evidence_refs"), dict):
            workload["evidence_refs"].update(
                {
                    str(key): str(value)
                    for key, value in score["evidence_refs"].items()
                    if value
                }
            )

    for workload in workloads.values():
        _apply_missing_required_artifact_gaps(
            workload,
            has_amd_score=amd_score_report is not None,
            has_amd_sol=bool(amd_sol_artifacts),
            has_solar=bool(solar_artifacts),
            has_matrix=compatibility_matrix is not None,
        )
        for gap in workload["evidence_gaps"]:
            _add_gap_group(
                evidence_gap_groups,
                reason_code=gap,
                example_ref=_workload_ref(workload),
            )

    if amd_score_report is None:
        _add_gap_group(evidence_gap_groups, reason_code="amd_score_evidence_missing", example_ref="amd_score_report")
    if not amd_sol_artifacts:
        _add_gap_group(evidence_gap_groups, reason_code="amd_sol_evidence_missing", example_ref="amd_sol_artifacts")
    if not solar_artifacts:
        _add_gap_group(evidence_gap_groups, reason_code="solar_derivation_missing", example_ref="solar_artifacts")
    if compatibility_matrix is None:
        _add_gap_group(evidence_gap_groups, reason_code="compatibility_matrix_missing", example_ref="compatibility_matrix")

    rows = []
    totals = AmdBoundSanityStatusTotals()
    provisional_risk = False
    for raw in workloads.values():
        flags = set(raw["diagnostic_flags"]) or {"scored"}
        if raw["evidence_gaps"]:
            flags.add("missing_evidence")
        primary_status = next(status for status in PRIMARY_STATUS_ORDER if status in flags)
        totals.add(primary_status)
        if "provisional" in flags and primary_status != "provisional":
            totals.add("provisional")
        provisional_risk = provisional_risk or "provisional" in flags
        rows.append(
            AmdBoundSanityWorkload(
                category=raw["category"],
                problem_id=raw["problem_id"],
                problem_path=raw["problem_path"],
                definition=raw["definition"],
                workload_uuid=raw["workload_uuid"],
                row_index=raw["row_index"],
                diagnostic_status=primary_status,
                diagnostic_flags=sorted(flags, key=SANITY_STATUS_KEYS.index),
                source_statuses=AmdBoundSanitySourceStatuses(**raw["source_statuses"]),
                amd_score_supported=raw["amd_score_supported"],
                coverage_summary=_sorted_jsonable(raw["coverage_summary"]),
                warnings=sorted(raw["warnings"]),
                evidence_refs=dict(sorted(raw["evidence_refs"].items())),
                evidence_gaps=sorted(raw["evidence_gaps"]),
            )
        )

    report = AmdBoundSanityReport(
        created_at=created_at or utc_timestamp(),
        sources=AmdBoundSanitySources(
            trace_refs=sorted(
                [_source_from_ref(ref) for ref in trace_refs],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            execution_closure=_source(
                execution_closure,
                path=source_paths.get("execution_closure"),
            ),
            amd_sol_artifacts=sorted(
                [_source_from_ref(ref) for ref in amd_sol_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            solar_artifacts=sorted(
                [_source_from_ref(ref) for ref in solar_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            amd_score_report=_source(
                amd_score_report,
                path=source_paths.get("amd_score_report"),
            ),
            compatibility_matrix=_source(
                compatibility_matrix,
                path=source_paths.get("compatibility_matrix"),
            ),
        ),
        artifact_availability=AmdBoundSanityArtifactAvailability(
            trace_refs=len(trace_refs),
            execution_closure=execution_closure is not None,
            amd_sol_artifacts=len(amd_sol_artifacts),
            solar_artifacts=len(solar_artifacts),
            amd_score_report=amd_score_report is not None,
            compatibility_matrix=compatibility_matrix is not None,
        ),
        status_totals=totals,
        amd_sol_aggregate_statuses=dict(sorted(amd_sol_statuses.items())),
        solar_aggregate_statuses=dict(sorted(solar_statuses.items())),
        coverage_summary=_coverage_summary(amd_score_report, compatibility_matrix),
        warnings=_suite_warnings(workloads, amd_score_report),
        evidence_gaps=_sorted_evidence_gaps(evidence_gap_groups),
        workloads=sorted(
            rows,
            key=lambda row: (
                row.category,
                row.problem_id,
                row.row_index if row.row_index is not None else -1,
                row.workload_uuid,
            ),
        ),
        claim_boundary=AmdBoundSanityClaimBoundary(
            provisional_rdna4_model_risk=provisional_risk
        ),
    )
    return report.with_checksum()


def render_amd_bound_sanity_markdown(report: AmdBoundSanityReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# AMD Bound Sanity Report",
        "",
        f"Generated: {report.created_at}",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        f"provisional RDNA 4 model risk: {str(report.claim_boundary.provisional_rdna4_model_risk).lower()}",
        "",
        "## Diagnostic Status Totals",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for status in SANITY_STATUS_KEYS:
        lines.append(f"| {status} | {payload['status_totals'][status]} |")

    lines.extend(["", "## Artifact Availability", "", "| Artifact | Available |"])
    lines.append("|----------|----------:|")
    for key, value in payload["artifact_availability"].items():
        lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")

    lines.extend(["", "## Aggregate AMD SOL/SOLAR Statuses", "", "| Source | Status | Count |"])
    lines.append("|--------|--------|------:|")
    for status, count in payload["amd_sol_aggregate_statuses"].items():
        lines.append(f"| amd_sol | {_md_cell(status)} | {count} |")
    for status, count in payload["solar_aggregate_statuses"].items():
        lines.append(f"| solar | {_md_cell(status)} | {count} |")

    lines.extend(["", "## Coverage Summaries", "", "| Source | Summary |"])
    lines.append("|--------|---------|")
    for key, value in payload["coverage_summary"].items():
        lines.append(
            f"| {_md_cell(key)} | {_md_cell(json.dumps(value, sort_keys=True))} |"
        )

    lines.extend(["", "## Workloads", "", "| Workload | Diagnostic | Flags | Source Statuses | Warnings |"])
    lines.append("|----------|------------|-------|-----------------|----------|")
    for row in payload["workloads"]:
        lines.append(
            f"| {_md_cell(row['definition'] or row['problem_id'])}#{_md_cell(row['workload_uuid'])} | "
            f"{_md_cell(row['diagnostic_status'])} | "
            f"{_md_cell(', '.join(row['diagnostic_flags']))} | "
            f"{_md_cell(json.dumps(row['source_statuses'], sort_keys=True))} | "
            f"{_md_cell('; '.join(row['warnings'][:5]))} |"
        )

    lines.extend(["", "## Warnings", "", "| Warning |"])
    lines.append("|---------|")
    for warning in payload["warnings"][:25]:
        lines.append(f"| {_md_cell(warning)} |")

    lines.extend(["", "## Evidence Gaps", "", "| Reason | Count | Examples | Next Evidence |"])
    lines.append("|--------|------:|----------|---------------|")
    for gap in payload["evidence_gaps"]:
        lines.append(
            f"| {_md_cell(gap['reason_code'])} | {gap['count']} | "
            f"{_md_cell(', '.join(gap['example_refs'][:5]))} | "
            f"{_md_cell(gap['next_evidence'])} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Ref | Path |"])
    lines.append("|--------|--------|----------|-----|------|")
    lines.extend(_source_rows(payload))

    lines.extend(["", "## Claim Boundaries", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")

    return "\n".join(lines) + "\n"


def write_amd_bound_sanity_reports(
    report: AmdBoundSanityReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_amd_bound_sanity_markdown(report), encoding="utf-8")


def _payload_artifacts(
    artifacts: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path],
) -> list[dict[str, Any]]:
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _workload_seed(uuid: str, payload: dict[str, Any]) -> dict[str, Any]:
    definition = _optional_str(payload.get("definition"))
    problem_id = _optional_str(payload.get("problem_id")) or definition or "unknown"
    return {
        "category": _optional_str(payload.get("category")) or "unknown",
        "problem_id": problem_id,
        "problem_path": _optional_str(payload.get("problem_path")),
        "definition": definition,
        "workload_uuid": uuid,
        "row_index": payload.get("row_index") if isinstance(payload.get("row_index"), int) else None,
        "diagnostic_flags": set(),
        "source_statuses": {
            "closure_status": None,
            "amd_sol_status": None,
            "solar_status": None,
            "amd_score_supported": None,
        },
        "amd_score_supported": None,
        "coverage_summary": {},
        "warnings": [],
        "evidence_refs": {},
        "evidence_gaps": [],
    }


def _ensure_workload(
    workloads: dict[str, dict[str, Any]],
    uuid: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if uuid not in workloads:
        workloads[uuid] = _workload_seed(uuid, payload)
        return workloads[uuid]
    workload = workloads[uuid]
    definition = _optional_str(payload.get("definition"))
    if definition and workload["definition"] is None:
        workload["definition"] = definition
        if workload["problem_id"] == "unknown":
            workload["problem_id"] = definition
    for key in ("category", "problem_id", "problem_path"):
        value = _optional_str(payload.get(key))
        if value and (workload[key] in {None, "unknown"}):
            workload[key] = value
    if isinstance(payload.get("row_index"), int) and workload["row_index"] is None:
        workload["row_index"] = payload["row_index"]
    return workload


def _artifact_uuid(payload: dict[str, Any]) -> str | None:
    uuid = payload.get("workload_uuid")
    return str(uuid) if uuid is not None else None


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _warnings_from(payload: dict[str, Any]) -> list[str]:
    values = payload.get("warnings") if isinstance(payload, dict) else None
    return [str(value) for value in values] if isinstance(values, list) else []


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _add_gap(workload: dict[str, Any], reason_code: str) -> None:
    if reason_code not in workload["evidence_gaps"]:
        workload["evidence_gaps"].append(reason_code)


def _add_gap_group(
    groups: dict[str, dict[str, Any]],
    *,
    reason_code: str,
    example_ref: str,
) -> None:
    group = groups.setdefault(
        reason_code,
        {"reason_code": reason_code, "count": 0, "example_refs": []},
    )
    group["count"] += 1
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)


def _apply_missing_required_artifact_gaps(
    workload: dict[str, Any],
    *,
    has_amd_score: bool,
    has_amd_sol: bool,
    has_solar: bool,
    has_matrix: bool,
) -> None:
    if not has_amd_score:
        _add_gap(workload, "amd_score_evidence_missing")
    if not has_amd_sol or workload["source_statuses"]["amd_sol_status"] is None:
        _add_gap(workload, "amd_sol_evidence_missing")
    if not has_solar or workload["source_statuses"]["solar_status"] is None:
        _add_gap(workload, "solar_derivation_missing")
    if not has_matrix:
        _add_gap(workload, "compatibility_matrix_missing")


def _is_degraded_status(
    status: str | None,
    coverage_summary: object,
    warnings: list[str],
) -> bool:
    if status == "degraded":
        return True
    if _contains_provisional(warnings):
        return True
    if isinstance(coverage_summary, dict):
        if coverage_summary.get("inexact_ops", 0):
            return True
        if coverage_summary.get("worst_confidence") in {"inexact", "estimated"}:
            return True
    return False


def _contains_unsupported(values: list[str]) -> bool:
    return any("unsupported" in value.lower() for value in values)


def _contains_provisional(values: list[str]) -> bool:
    lowered = " ".join(values).lower()
    return "provisional" in lowered or "rdna 4" in lowered or "model assumption" in lowered


def _provisional_artifact(artifact: dict[str, Any]) -> bool:
    if _contains_provisional(_warnings_from(artifact)):
        return True
    hardware = _dict_value(artifact.get("hardware_model"))
    architecture = str(hardware.get("architecture", "")).lower()
    validation_values = {
        str(hardware.get("hardware_validation_status", "")).lower(),
        str(hardware.get("model_validation_status", "")).lower(),
    }
    return architecture in {"gfx1200", "rdna4", "rdna 4"} and "unvalidated" in validation_values


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | str | None,
    ref: str | None = None,
) -> AmdBoundSanitySourceRef:
    return AmdBoundSanitySourceRef(
        path=str(path) if path else None,
        ref=ref,
        schema_version=_optional_str(payload.get("schema_version")) if payload else None,
        checksum=_checksum(payload),
    )


def _source_from_ref(
    value: AmdBoundSanitySourceRef | dict[str, Any] | str | Path,
) -> AmdBoundSanitySourceRef:
    if isinstance(value, AmdBoundSanitySourceRef):
        return value
    if isinstance(value, str | Path):
        return AmdBoundSanitySourceRef(path=str(value))
    return AmdBoundSanitySourceRef(
        path=_optional_str(value.get("path")),
        ref=_optional_str(value.get("ref")),
        schema_version=_optional_str(value.get("schema_version")),
        checksum=_checksum(value),
    )


def _checksum(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("value"), str):
            return value["value"]
        if isinstance(value, str):
            return value
    return None


def _coverage_summary(
    amd_score_report: dict[str, Any] | None,
    compatibility_matrix: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if amd_score_report and isinstance(amd_score_report.get("evidence_summary"), dict):
        summary["amd_score"] = _sorted_jsonable(amd_score_report["evidence_summary"])
    if compatibility_matrix and isinstance(compatibility_matrix.get("status_counts"), dict):
        summary["compatibility_matrix"] = _sorted_jsonable(
            compatibility_matrix["status_counts"]
        )
    return summary


def _suite_warnings(
    workloads: dict[str, dict[str, Any]],
    amd_score_report: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    _extend_unique(warnings, _warnings_from(amd_score_report or {}))
    for workload in workloads.values():
        _extend_unique(warnings, workload["warnings"])
    return sorted(warnings)


def _sorted_evidence_gaps(
    groups: dict[str, dict[str, Any]],
) -> list[AmdBoundSanityEvidenceGap]:
    return [
        AmdBoundSanityEvidenceGap(
            reason_code=reason_code,
            count=group["count"],
            example_refs=sorted(group["example_refs"]),
            next_evidence=_next_evidence(reason_code),
        )
        for reason_code, group in sorted(groups.items())
    ]


def _next_evidence(reason_code: str) -> str:
    if "amd_score" in reason_code:
        return "Attach bounded AMD score evidence refs/checksums before upgrading claims."
    if "amd_sol" in reason_code:
        return "Attach bounded AMD SOL evidence refs/checksums before upgrading claims."
    if "solar" in reason_code:
        return "Attach bounded SOLAR derivation evidence refs/checksums before upgrading claims."
    if "compatibility_matrix" in reason_code:
        return "Attach bounded Compatibility Matrix evidence refs/checksums before upgrading claims."
    return "Attach bounded existing evidence refs/checksums before upgrading claims."


def _workload_ref(workload: dict[str, Any]) -> str:
    return f"{workload['problem_id']}#{workload['workload_uuid']}"


def _sorted_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sorted_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_jsonable(item) for item in value]
    return value


def _md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _source_rows(payload: dict[str, Any]) -> list[str]:
    rows = []
    sources = payload["sources"]
    for name in ("execution_closure", "amd_score_report", "compatibility_matrix"):
        source = sources[name]
        rows.append(
            f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
            f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
            f"{_md_cell(source.get('path'))} |"
        )
    for name in ("trace_refs", "amd_sol_artifacts", "solar_artifacts"):
        for source in sources[name]:
            rows.append(
                f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
                f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
                f"{_md_cell(source.get('path'))} |"
            )
    return rows
