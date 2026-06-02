# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Cross-report consistency diagnostic sidecar helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.dataset.checksums import stable_json_checksum
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

CONSISTENCY_REPORT_SCHEMA_VERSION = "sol_execbench.consistency_report.v1"

SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)

ATTEMPTED_CLOSURE_STATUSES = frozenset(
    {
        "attempted_passed",
        "attempted_failed",
        "executed",
        "passed",
        "failed",
        "timed",
        "scored",
    }
)
BLOCKED_DENOMINATOR_STATUSES = frozenset(
    {
        "blocked",
        "unsupported",
        "deferred",
        "evidence_missing",
        "not_attempted",
        "skipped",
        "filtered",
    }
)
AUTHORITY_BOUNDARY_KEYS = frozenset(
    {
        "amd_sol_model_validation",
        "cdna3_validation",
        "cdna4_validation",
        "full_235_problem_validation",
        "leaderboard_authority",
        "mi300x_validation",
        "native_host_validation",
        "new_hardware_validation",
        "paper_parity",
        "score_authority",
        "score_authority_upgrade",
        "solar_model_validation",
        "upstream_solar_equivalence",
        "upstream_solar_parity",
    }
)

CLAIM_BOUNDARY_TEXT = (
    "This report is diagnostic-only cross-report consistency lint: not score "
    "authority, not paper parity, not leaderboard authority, not native-host "
    "validation, and not new-hardware validation."
)


class ConsistencySourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ConsistencyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str
    reason_code: str
    sources: list[str]
    refs: list[str] = Field(default_factory=list)
    message: str
    next_step: str


class ConsistencyFindingTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocker: int = 0
    warning: int = 0
    info: int = 0

    def add(self, severity: str) -> None:
        if severity not in {"blocker", "warning", "info"}:
            raise ValueError(f"Unknown consistency severity: {severity}")
        setattr(self, severity, getattr(self, severity) + 1)


class ConsistencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources_checked: int
    findings_total: int
    finding_totals: ConsistencyFindingTotals


class ConsistencyClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostic_only: bool = True
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class ConsistencyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONSISTENCY_REPORT_SCHEMA_VERSION
    created_at: str
    sources: list[ConsistencySourceRef]
    summary: ConsistencySummary
    findings: list[ConsistencyFinding]
    claim_boundary: ConsistencyClaimBoundary = Field(
        default_factory=ConsistencyClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ConsistencyReport":
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def build_consistency_report(
    *,
    execution_closure: dict[str, Any] | None = None,
    paper_denominator: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    runtime_evidence: dict[str, Any] | None = None,
    static_evidence: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_report: dict[str, Any] | None = None,
    solar_derivation: dict[str, Any] | None = None,
    amd_bound_sanity: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> ConsistencyReport:
    source_paths = source_paths or {}
    payloads = {
        "execution_closure": execution_closure,
        "paper_denominator": paper_denominator,
        "matrix_report": matrix_report,
        "runtime_evidence": runtime_evidence,
        "static_evidence": static_evidence,
        "amd_score_report": amd_score_report,
        "amd_sol_report": amd_sol_report,
        "solar_derivation": solar_derivation,
        "amd_bound_sanity": amd_bound_sanity,
    }
    sources = [
        _source_ref(source_id, payload, source_paths.get(source_id))
        for source_id, payload in payloads.items()
        if payload is not None
    ]

    findings: list[ConsistencyFinding] = []
    closure_records = _records(execution_closure)
    denominator_workloads = _records(paper_denominator, key="workloads")

    findings.extend(
        _find_denominator_closure_drift(closure_records, denominator_workloads)
    )
    findings.extend(_find_matrix_runtime_attempted(matrix_report, closure_records))
    findings.extend(
        _find_missing_derived_evidence_scored(closure_records, amd_score_report)
    )
    findings.extend(_find_checksum_mismatches(payloads))
    findings.extend(_find_claim_boundary_violations(payloads))

    findings = _dedupe_findings(findings)
    totals = ConsistencyFindingTotals()
    for finding in findings:
        totals.add(finding.severity)

    report = ConsistencyReport(
        created_at=created_at or utc_timestamp(),
        sources=sorted(sources, key=lambda source: source.source_id),
        summary=ConsistencySummary(
            sources_checked=len(sources),
            findings_total=len(findings),
            finding_totals=totals,
        ),
        findings=findings,
    )
    return report.with_checksum()


def render_consistency_markdown(report: ConsistencyReport) -> str:
    lines = [
        "# Cross-Report Consistency Report",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Summary",
        "",
        f"- Sources checked: {report.summary.sources_checked}",
        f"- Findings: {report.summary.findings_total}",
        f"- Blockers: {report.summary.finding_totals.blocker}",
        f"- Warnings: {report.summary.finding_totals.warning}",
        f"- Info: {report.summary.finding_totals.info}",
        "",
        "## Findings",
        "",
    ]
    if report.findings:
        lines.extend(
            [
                "| Severity | Reason | Sources | Refs | Message | Next step |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in report.findings:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(finding.severity),
                        _md_cell(finding.reason_code),
                        _md_cell(", ".join(finding.sources)),
                        _md_cell(", ".join(finding.refs)),
                        _md_cell(finding.message),
                        _md_cell(finding.next_step),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No consistency findings.")

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Source | Path | Schema | Checksum |",
            "| --- | --- | --- | --- |",
        ]
    )
    for source in report.sources:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(source.source_id),
                    _md_cell(source.path or source.ref or ""),
                    _md_cell(source.schema_version or ""),
                    _md_cell(source.checksum or ""),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.claim_boundary.model_dump(mode="json").items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_consistency_reports(
    report: ConsistencyReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_consistency_markdown(report), encoding="utf-8")


def _records(
    payload: dict[str, Any] | None, *, key: str = "records"
) -> list[dict[str, Any]]:
    records = (payload or {}).get(key, [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _source_ref(
    source_id: str,
    payload: dict[str, Any],
    path: Path | None,
) -> ConsistencySourceRef:
    return ConsistencySourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=_optional_str(payload.get("schema_version")),
        checksum=_checksum(payload),
    )


def _checksum(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            checksum = value.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _find_denominator_closure_drift(
    closure_records: list[dict[str, Any]],
    denominator_workloads: list[dict[str, Any]],
) -> list[ConsistencyFinding]:
    attempted = {
        _workload_key(record): record
        for record in closure_records
        if _is_attempted(record.get("closure_status"))
    }
    findings: list[ConsistencyFinding] = []
    for workload in denominator_workloads:
        key = _workload_key(workload)
        if key not in attempted:
            continue
        if not _is_denominator_blocked(workload):
            continue
        findings.append(
            ConsistencyFinding(
                severity="blocker",
                reason_code="denominator_closure_drift",
                sources=["paper_denominator", "execution_closure"],
                refs=[key],
                message=(
                    "Workload is attempted in execution closure but blocked or "
                    "unavailable in denominator accounting."
                ),
                next_step=(
                    "Regenerate denominator and closure from the same inputs, or "
                    "correct the stale workload state."
                ),
            )
        )
    return findings


def _find_matrix_runtime_attempted(
    matrix_report: dict[str, Any] | None,
    closure_records: list[dict[str, Any]],
) -> list[ConsistencyFinding]:
    if matrix_report is None:
        return []
    attempted_refs = [
        _workload_key(record)
        for record in closure_records
        if _is_attempted(record.get("closure_status"))
    ]
    if not attempted_refs:
        return []
    unavailable_refs = _matrix_runtime_unavailable_refs(matrix_report)
    if not unavailable_refs:
        return []
    return [
        ConsistencyFinding(
            severity="blocker",
            reason_code="matrix_runtime_unavailable_attempted",
            sources=["matrix_report", "execution_closure"],
            refs=sorted((set(attempted_refs) | set(unavailable_refs)))[:10],
            message=(
                "Runtime evidence is marked unavailable while closure contains "
                "attempted workloads."
            ),
            next_step=(
                "Reconcile the matrix runtime status with the attempted closure "
                "evidence before using either report as supporting evidence."
            ),
        )
    ]


def _find_missing_derived_evidence_scored(
    closure_records: list[dict[str, Any]],
    amd_score_report: dict[str, Any] | None,
) -> list[ConsistencyFinding]:
    if amd_score_report is None:
        return []
    missing = {
        _workload_key(record)
        for record in closure_records
        if _has_missing_derived_evidence(record)
    }
    scored = {
        _workload_key(record)
        for record in _score_records(amd_score_report)
        if _is_score_present(record)
    }
    overlaps = sorted(missing & scored)
    if not overlaps:
        return []
    return [
        ConsistencyFinding(
            severity="blocker",
            reason_code="missing_derived_evidence_scored",
            sources=["execution_closure", "amd_score_report"],
            refs=overlaps[:10],
            message="Closure reports missing derived evidence for workloads that are scored.",
            next_step=(
                "Regenerate score evidence or remove stale scored entries until "
                "derived evidence refs are present."
            ),
        )
    ]


def _find_checksum_mismatches(
    payloads: dict[str, dict[str, Any] | None],
) -> list[ConsistencyFinding]:
    findings: list[ConsistencyFinding] = []
    actual = {
        source_id: _checksum(payload)
        for source_id, payload in payloads.items()
        if payload is not None
    }
    for source_id, payload in payloads.items():
        if payload is None:
            continue
        for ref_source_id, expected in _embedded_source_checksums(payload).items():
            observed = actual.get(ref_source_id)
            if expected and observed and expected != observed:
                findings.append(
                    ConsistencyFinding(
                        severity="warning",
                        reason_code="source_ref_checksum_mismatch",
                        sources=[source_id, ref_source_id],
                        refs=[f"{source_id}->{ref_source_id}"],
                        message=(
                            "Embedded source checksum does not match the provided "
                            "source payload checksum."
                        ),
                        next_step=(
                            "Regenerate the dependent report from the provided "
                            "source payloads."
                        ),
                    )
                )
    return findings


def _find_claim_boundary_violations(
    payloads: dict[str, dict[str, Any] | None],
) -> list[ConsistencyFinding]:
    findings: list[ConsistencyFinding] = []
    for source_id, payload in payloads.items():
        if payload is None:
            continue
        paths = sorted(_truthy_authority_paths(payload))
        if not paths:
            continue
        findings.append(
            ConsistencyFinding(
                severity="blocker",
                reason_code="claim_boundary_violation",
                sources=[source_id],
                refs=paths[:10],
                message="A source report contains a truthy authority claim boundary.",
                next_step=(
                    "Keep authority claims false unless a separate validated "
                    "artifact explicitly upgrades the claim."
                ),
            )
        )
    return findings


def _embedded_source_checksums(payload: dict[str, Any]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    sources = payload.get("sources")
    if isinstance(sources, dict):
        for source_id, value in sources.items():
            checksum = _source_ref_checksum(value)
            if checksum:
                checksums[_normalize_source_id(str(source_id))] = checksum
    return checksums


def _source_ref_checksum(value: object) -> str | None:
    if isinstance(value, dict):
        checksum = cast(dict[str, Any], value).get("checksum")
        if isinstance(checksum, dict):
            nested = checksum.get("value")
            return nested if isinstance(nested, str) else None
        if isinstance(checksum, str):
            return checksum
    return None


def _normalize_source_id(source_id: str) -> str:
    aliases = {
        "compatibility_matrix": "matrix_report",
    }
    return aliases.get(source_id, source_id)


def _matrix_runtime_unavailable_refs(matrix_report: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    candidates = []
    for key in ("entries", "rows", "workloads", "records", "checks"):
        value = matrix_report.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    for entry in candidates:
        text = " ".join(
            str(entry.get(key, ""))
            for key in ("status", "runtime_status", "reason_code", "message")
        ).lower()
        if "runtime_unavailable" in text or "runtime unavailable" in text:
            refs.append(_workload_key(entry))
    return refs


def _score_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("scores", "workloads", "records", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _is_score_present(record: dict[str, Any]) -> bool:
    if record.get("supported") is True or record.get("score_eligible") is True:
        return True
    for key in ("score", "runtime_ms", "speedup", "amd_native_score"):
        if record.get(key) is not None:
            return True
    return False


def _has_missing_derived_evidence(record: dict[str, Any]) -> bool:
    status = _optional_str(record.get("closure_status")) or ""
    if "derived_evidence_missing" in status:
        return True
    gaps = " ".join(str(gap) for gap in record.get("evidence_gaps", []))
    return any(
        marker in gaps
        for marker in ("amd_score", "amd_sol", "solar_derivation", "derived")
    )


def _is_attempted(value: object) -> bool:
    status = (_optional_str(value) or "").lower()
    return status in ATTEMPTED_CLOSURE_STATUSES or status.startswith("attempted_")


def _is_denominator_blocked(workload: dict[str, Any]) -> bool:
    statuses = {
        str(workload.get("readiness_status", "")).lower(),
        str(workload.get("closure_status", "")).lower(),
    }
    states = workload.get("states")
    if isinstance(states, dict):
        statuses.update(
            str(key).lower()
            for key, value in states.items()
            if key in BLOCKED_DENOMINATOR_STATUSES and bool(value)
        )
    return bool(statuses & BLOCKED_DENOMINATOR_STATUSES)


def _truthy_authority_paths(payload: object, *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in AUTHORITY_BOUNDARY_KEYS and value is True:
                paths.add(path)
            paths.update(_truthy_authority_paths(value, prefix=path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            paths.update(_truthy_authority_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _workload_key(payload: dict[str, Any]) -> str:
    for key in ("workload_uuid", "uuid", "workload_id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    problem_id = payload.get("problem_id") or payload.get("problem") or "unknown"
    row_index = payload.get("row_index")
    return f"{problem_id}:{row_index if row_index is not None else 'unknown'}"


def _dedupe_findings(findings: list[ConsistencyFinding]) -> list[ConsistencyFinding]:
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    unique: list[ConsistencyFinding] = []
    for finding in findings:
        key = (
            finding.reason_code,
            tuple(sorted(finding.sources)),
            tuple(sorted(finding.refs)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return sorted(unique, key=lambda item: (item.severity, item.reason_code, item.refs))


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
