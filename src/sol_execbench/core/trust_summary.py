# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Research trust summary sidecar helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.dataset.checksums import stable_json_checksum
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

TRUST_SUMMARY_SCHEMA_VERSION = "sol_execbench.trust_summary.v1"

SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "matrix_checksum",
    "checksum",
)
CLAIM_BOUNDARY_TEXT = (
    "This trust summary is review guidance only: not paper validation, not "
    "paper parity, not leaderboard authority, not native-host validation, and "
    "not new-hardware validation."
)


class TrustSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class TrustOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    status: str
    reason_codes: list[str]
    next_steps: list[str] = Field(default_factory=list)


class TrustSummaryClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_guidance_only: bool = True
    paper_validation: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class TrustSummaryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRUST_SUMMARY_SCHEMA_VERSION
    created_at: str
    sources: list[TrustSourceRef]
    outcomes: list[TrustOutcome]
    overall_status: str
    next_steps: list[str]
    claim_boundary: TrustSummaryClaimBoundary = Field(
        default_factory=TrustSummaryClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "TrustSummaryReport":
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


def build_trust_summary_report(
    *,
    consistency_report: dict[str, Any] | None = None,
    evaluation_stability: dict[str, Any] | None = None,
    claim_upgrade: dict[str, Any] | None = None,
    execution_closure: dict[str, Any] | None = None,
    paper_denominator: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_bound_sanity: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> TrustSummaryReport:
    source_paths = source_paths or {}
    payloads = {
        "consistency_report": consistency_report,
        "evaluation_stability": evaluation_stability,
        "claim_upgrade": claim_upgrade,
        "execution_closure": execution_closure,
        "paper_denominator": paper_denominator,
        "matrix_report": matrix_report,
        "amd_score_report": amd_score_report,
        "amd_bound_sanity": amd_bound_sanity,
    }
    outcomes = [
        _consistency_outcome(consistency_report),
        _stability_outcome(evaluation_stability),
        _claim_outcome(claim_upgrade),
        _evidence_outcome(
            execution_closure=execution_closure,
            paper_denominator=paper_denominator,
            matrix_report=matrix_report,
            amd_score_report=amd_score_report,
            amd_bound_sanity=amd_bound_sanity,
        ),
    ]
    next_steps = sorted({step for outcome in outcomes for step in outcome.next_steps})
    overall = _overall_status(outcomes)
    report = TrustSummaryReport(
        created_at=created_at or utc_timestamp(),
        sources=[
            _source_ref(source_id, payload, source_paths.get(source_id))
            for source_id, payload in payloads.items()
            if payload is not None
        ],
        outcomes=outcomes,
        overall_status=overall,
        next_steps=next_steps,
    )
    return report.with_checksum()


def render_trust_summary_markdown(report: TrustSummaryReport) -> str:
    lines = [
        "# Trust Summary",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Overall status: `{report.overall_status}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Outcomes",
        "",
        "| Key | Status | Reasons | Next steps |",
        "| --- | --- | --- | --- |",
    ]
    for outcome in report.outcomes:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(outcome.key),
                    _md_cell(outcome.status),
                    _md_cell(", ".join(outcome.reason_codes)),
                    _md_cell(", ".join(outcome.next_steps)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Sources", "", "| Source | Path | Schema | Checksum |", "| --- | --- | --- | --- |"])
    for source in report.sources:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(source.source_id),
                    _md_cell(source.path or ""),
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


def write_trust_summary_reports(
    report: TrustSummaryReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_trust_summary_markdown(report), encoding="utf-8")


def _consistency_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    if payload is None:
        return TrustOutcome(
            key="internally_consistent",
            status="evidence_missing",
            reason_codes=["consistency_report_missing"],
            next_steps=["Generate consistency_report.v1."],
        )
    blockers = _finding_total(payload, "blocker")
    status = "internally_consistent" if blockers == 0 else "blocked"
    return TrustOutcome(
        key="internally_consistent",
        status=status,
        reason_codes=["no_consistency_blockers" if blockers == 0 else "consistency_blockers"],
        next_steps=[] if blockers == 0 else ["Resolve consistency blocker findings."],
    )


def _stability_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    if payload is None:
        return TrustOutcome(
            key="stable_enough_to_interpret",
            status="evidence_missing",
            reason_codes=["evaluation_stability_missing"],
            next_steps=["Generate evaluation_stability.v1."],
        )
    totals = payload.get("status_totals") if isinstance(payload.get("status_totals"), dict) else {}
    risky = [
        key
        for key in (
            "noisy",
            "insufficient_samples",
            "missing_timing",
            "clock_unlocked",
            "profiler_overhead_risk",
            "backend_unsupported",
        )
        if int(totals.get(key) or 0) > 0
    ]
    return TrustOutcome(
        key="stable_enough_to_interpret",
        status="stable_enough" if not risky and int(totals.get("stable") or 0) > 0 else "blocked",
        reason_codes=["stable_timing"] if not risky else risky,
        next_steps=[] if not risky else ["Collect stable timing evidence with documented clock and backend policy."],
    )


def _claim_outcome(payload: dict[str, Any] | None) -> TrustOutcome:
    if payload is None:
        return TrustOutcome(
            key="claim_upgrade",
            status="diagnostic_only",
            reason_codes=["claim_upgrade_missing"],
            next_steps=["Generate claim_upgrade.v1."],
        )
    highest = str(payload.get("highest_eligible_claim") or "diagnostic_only")
    status = "claim_upgrade_blocked" if highest == "diagnostic_only" else highest
    return TrustOutcome(
        key="claim_upgrade",
        status=status,
        reason_codes=[f"highest:{highest}"],
        next_steps=[] if highest != "diagnostic_only" else ["Satisfy claim-upgrade prerequisites before stronger claims."],
    )


def _evidence_outcome(
    *,
    execution_closure: dict[str, Any] | None,
    paper_denominator: dict[str, Any] | None,
    matrix_report: dict[str, Any] | None,
    amd_score_report: dict[str, Any] | None,
    amd_bound_sanity: dict[str, Any] | None,
) -> TrustOutcome:
    missing = [
        name
        for name, payload in {
            "execution_closure": execution_closure,
            "paper_denominator": paper_denominator,
            "matrix_report": matrix_report,
            "amd_score_report": amd_score_report,
            "amd_bound_sanity": amd_bound_sanity,
        }.items()
        if payload is None
    ]
    next_steps = [f"Provide {name} evidence." for name in missing]
    next_steps.append("Future CDNA3/MI300X/native-host/paper-scale validation needs explicit hardware evidence.")
    return TrustOutcome(
        key="evidence_completeness",
        status="evidence_missing" if missing else "reviewable",
        reason_codes=[f"missing:{name}" for name in missing] or ["required_refs_present"],
        next_steps=next_steps if missing else next_steps[-1:],
    )


def _overall_status(outcomes: list[TrustOutcome]) -> str:
    statuses = {outcome.status for outcome in outcomes}
    if "blocked" in statuses or "claim_upgrade_blocked" in statuses:
        return "claim_upgrade_blocked"
    if "evidence_missing" in statuses:
        return "evidence_missing"
    return "reviewable"


def _finding_total(payload: dict[str, Any], severity: str) -> int:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        totals = summary.get("finding_totals")
        if isinstance(totals, dict):
            return int(totals.get(severity) or 0)
    return sum(
        1
        for finding in payload.get("findings", [])
        if isinstance(finding, dict) and finding.get("severity") == severity
    )


def _source_ref(source_id: str, payload: dict[str, Any], path: Path | None) -> TrustSourceRef:
    return TrustSourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=payload.get("schema_version") if isinstance(payload.get("schema_version"), str) else None,
        checksum=_checksum(payload),
    )


def _checksum(payload: dict[str, Any]) -> str | None:
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            checksum = value.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
