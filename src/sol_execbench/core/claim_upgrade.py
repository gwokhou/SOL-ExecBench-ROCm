# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum
from sol_execbench.core.text_utils import markdown_table_cell as _md_cell
from sol_execbench.core.trust_summary import load_json as load_json, utc_timestamp

CLAIM_UPGRADE_SCHEMA_VERSION = "sol_execbench.claim_upgrade.v1"

CLAIM_LEVELS = (
    "diagnostic_only",
    "container_validated",
    "native_host_validated",
    "score_authoritative",
    "paper_parity_candidate",
    "leaderboard_ready",
)
SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "amd_sol_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)
CLAIM_BOUNDARY_TEXT = (
    "This report evaluates prerequisites only: it does not mutate source "
    "authority fields and is not itself paper parity, leaderboard authority, "
    "native-host validation, score authority, or new-hardware validation."
)


class ClaimSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ClaimRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_level: str
    required_sources: list[str]
    required_conditions: list[str]


class ClaimEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_level: str
    eligible: bool
    blockers: list[str]
    unmet_prerequisites: list[str]
    next_evidence: list[str]


class ClaimUpgradeClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prerequisite_evaluation_only: bool = True
    mutates_source_authority: bool = False
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class ClaimUpgradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CLAIM_UPGRADE_SCHEMA_VERSION
    created_at: str
    sources: list[ClaimSourceRef]
    rules: list[ClaimRule]
    evaluations: list[ClaimEvaluation]
    highest_eligible_claim: str
    claim_boundary: ClaimUpgradeClaimBoundary = Field(
        default_factory=ClaimUpgradeClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ClaimUpgradeReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)


def default_claim_rules() -> list[ClaimRule]:
    return [
        ClaimRule(
            claim_level="diagnostic_only",
            required_sources=[],
            required_conditions=["report_generated"],
        ),
        ClaimRule(
            claim_level="container_validated",
            required_sources=[
                "execution_closure",
                "paper_denominator",
                "consistency_report",
            ],
            required_conditions=[
                "closure_attempted_evidence",
                "denominator_present",
                "no_consistency_blockers",
            ],
        ),
        ClaimRule(
            claim_level="native_host_validated",
            required_sources=["hardware_validation", "matrix_report"],
            required_conditions=[
                "native_host_validation_evidence",
                "matrix_runtime_available",
            ],
        ),
        ClaimRule(
            claim_level="score_authoritative",
            required_sources=[
                "amd_score_report",
                "amd_sol_report",
                "solar_derivation",
                "evaluation_stability",
                "consistency_report",
            ],
            required_conditions=[
                "score_evidence_present",
                "amd_sol_evidence_present",
                "solar_derivation_present",
                "stable_timing",
                "no_consistency_blockers",
            ],
        ),
        ClaimRule(
            claim_level="paper_parity_candidate",
            required_sources=[
                "paper_denominator",
                "amd_score_report",
                "amd_sol_report",
                "solar_derivation",
                "amd_bound_sanity",
                "hardware_validation",
            ],
            required_conditions=[
                "full_suite_accounted",
                "score_evidence_present",
                "amd_sol_evidence_present",
                "solar_derivation_present",
                "bound_sanity_clean",
                "native_host_validation_evidence",
            ],
        ),
        ClaimRule(
            claim_level="leaderboard_ready",
            required_sources=["paper_denominator", "hardware_validation"],
            required_conditions=[
                "paper_parity_candidate",
                "submission_policy_exists",
                "hosted_service_evidence",
            ],
        ),
    ]


def build_claim_upgrade_report(
    *,
    consistency_report: dict[str, Any] | None = None,
    evaluation_stability: dict[str, Any] | None = None,
    execution_closure: dict[str, Any] | None = None,
    paper_denominator: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_report: dict[str, Any] | None = None,
    solar_derivation: dict[str, Any] | None = None,
    amd_bound_sanity: dict[str, Any] | None = None,
    hardware_validation: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> ClaimUpgradeReport:
    source_paths = source_paths or {}
    payloads = {
        "consistency_report": consistency_report,
        "evaluation_stability": evaluation_stability,
        "execution_closure": execution_closure,
        "paper_denominator": paper_denominator,
        "matrix_report": matrix_report,
        "amd_score_report": amd_score_report,
        "amd_sol_report": amd_sol_report,
        "solar_derivation": solar_derivation,
        "amd_bound_sanity": amd_bound_sanity,
        "hardware_validation": hardware_validation,
    }
    rules = default_claim_rules()
    evaluations = [_evaluate_rule(rule, payloads) for rule in rules]
    highest = "diagnostic_only"
    for evaluation in evaluations:
        if evaluation.eligible:
            highest = evaluation.claim_level

    report = ClaimUpgradeReport(
        created_at=created_at or utc_timestamp(),
        sources=[
            _source_ref(source_id, payload, source_paths.get(source_id))
            for source_id, payload in payloads.items()
            if payload is not None
        ],
        rules=rules,
        evaluations=evaluations,
        highest_eligible_claim=highest,
    )
    return report.with_checksum()


def render_claim_upgrade_markdown(report: ClaimUpgradeReport) -> str:
    lines = [
        "# Claim Upgrade Report",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Highest eligible claim: `{report.highest_eligible_claim}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Evaluations",
        "",
        "| Claim | Eligible | Blockers | Unmet prerequisites | Next evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for evaluation in report.evaluations:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(evaluation.claim_level),
                    _md_cell(str(evaluation.eligible).lower()),
                    _md_cell(", ".join(evaluation.blockers)),
                    _md_cell(", ".join(evaluation.unmet_prerequisites)),
                    _md_cell(", ".join(evaluation.next_evidence)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.claim_boundary.model_dump(mode="json").items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_claim_upgrade_reports(
    report: ClaimUpgradeReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_claim_upgrade_markdown(report), encoding="utf-8")


def _evaluate_rule(
    rule: ClaimRule,
    payloads: dict[str, dict[str, Any] | None],
) -> ClaimEvaluation:
    unmet: list[str] = []
    blockers: list[str] = []
    for source_id in rule.required_sources:
        if payloads.get(source_id) is None:
            unmet.append(f"missing_source:{source_id}")
    for condition in rule.required_conditions:
        if not _condition_met(condition, payloads):
            unmet.append(f"condition:{condition}")
    if _consistency_blockers(payloads.get("consistency_report")):
        blockers.append("consistency_blockers_present")
    if rule.claim_level != "diagnostic_only" and _truthy_source_authority(payloads):
        blockers.append("source_authority_field_already_truthy")
    eligible = not unmet and not blockers
    if rule.claim_level == "diagnostic_only":
        eligible = True
        unmet = []
        blockers = []
    return ClaimEvaluation(
        claim_level=rule.claim_level,
        eligible=eligible,
        blockers=sorted(set(blockers)),
        unmet_prerequisites=sorted(set(unmet)),
        next_evidence=_next_evidence(sorted(set(unmet + blockers))),
    )


def _condition_met(condition: str, payloads: dict[str, dict[str, Any] | None]) -> bool:
    if condition == "report_generated":
        return True
    if condition == "closure_attempted_evidence":
        return any(
            str(record.get("closure_status", "")).startswith("attempted_")
            for record in _records(payloads.get("execution_closure"))
        )
    if condition == "denominator_present":
        return bool(_records(payloads.get("paper_denominator"), key="workloads"))
    if condition == "no_consistency_blockers":
        return not _consistency_blockers(payloads.get("consistency_report"))
    if condition == "native_host_validation_evidence":
        return bool(
            (payloads.get("hardware_validation") or {}).get("native_host_validated")
        )
    if condition == "matrix_runtime_available":
        return not _matrix_runtime_unavailable(payloads.get("matrix_report"))
    if condition == "score_evidence_present":
        return bool(_score_records(payloads.get("amd_score_report")))
    if condition == "amd_sol_evidence_present":
        return _payload_present(payloads.get("amd_sol_report"))
    if condition == "solar_derivation_present":
        return _payload_present(payloads.get("solar_derivation"))
    if condition == "stable_timing":
        return _stability_clean(payloads.get("evaluation_stability"))
    if condition == "full_suite_accounted":
        denominator = payloads.get("paper_denominator") or {}
        suite = denominator.get("suite")
        return isinstance(suite, dict) and int(suite.get("workloads") or 0) >= 235
    if condition == "bound_sanity_clean":
        sanity = payloads.get("amd_bound_sanity") or {}
        totals = sanity.get("status_totals")
        return (
            isinstance(totals, dict) and int(totals.get("missing_evidence") or 0) == 0
        )
    if condition in {
        "paper_parity_candidate",
        "submission_policy_exists",
        "hosted_service_evidence",
    }:
        return False
    return False


def _consistency_blockers(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    summary = payload.get("summary")
    if isinstance(summary, dict):
        totals = summary.get("finding_totals")
        if isinstance(totals, dict) and int(totals.get("blocker") or 0) > 0:
            return True
    return any(
        isinstance(finding, dict) and finding.get("severity") == "blocker"
        for finding in payload.get("findings", [])
    )


def _stability_clean(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    totals = payload.get("status_totals")
    if not isinstance(totals, dict):
        return False
    return (
        int(totals.get("stable") or 0) > 0
        and sum(
            int(totals.get(key) or 0)
            for key in (
                "noisy",
                "insufficient_samples",
                "missing_timing",
                "clock_unlocked",
                "profiler_overhead_risk",
                "backend_unsupported",
            )
        )
        == 0
    )


def _records(
    payload: dict[str, Any] | None, *, key: str = "records"
) -> list[dict[str, Any]]:
    records = (payload or {}).get(key, [])
    return records if isinstance(records, list) else []


def _score_records(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    for key in ("scores", "workloads", "records", "results"):
        records = (payload or {}).get(key)
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
    return []


def _payload_present(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if _checksum(payload):
        return True
    return any(
        key in payload
        for key in (
            "aggregate_bound",
            "aggregate_status",
            "bound_evidence",
            "coverage_summary",
            "formula_evidence",
            "workloads",
            "records",
            "results",
        )
    )


def _matrix_runtime_unavailable(payload: dict[str, Any] | None) -> bool:
    text = json.dumps(payload or {}, sort_keys=True).lower()
    return "runtime_unavailable" in text or "runtime unavailable" in text


def _truthy_source_authority(payloads: dict[str, dict[str, Any] | None]) -> bool:
    authority_keys = {
        "score_authority",
        "paper_parity",
        "leaderboard_authority",
        "native_host_validation",
        "new_hardware_validation",
    }
    return any(_truthy_key(payload, authority_keys) for payload in payloads.values())


def _truthy_key(payload: object, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        return any(
            (key in keys and value is True) or _truthy_key(value, keys)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_truthy_key(item, keys) for item in payload)
    return False


def _next_evidence(items: list[str]) -> list[str]:
    hints = {
        "missing_source:hardware_validation": "Add native-host hardware validation evidence.",
        "missing_source:evaluation_stability": "Generate evaluation_stability.v1.",
        "missing_source:consistency_report": "Generate consistency_report.v1.",
        "missing_source:amd_sol_report": "Provide AMD SOL bound evidence.",
        "missing_source:solar_derivation": "Provide SOLAR derivation evidence.",
        "condition:amd_sol_evidence_present": "Provide AMD SOL bound evidence.",
        "condition:solar_derivation_present": "Provide SOLAR derivation evidence.",
        "condition:stable_timing": "Provide stable timing evidence with locked clocks or explain risk.",
        "condition:no_consistency_blockers": "Resolve consistency blockers first.",
        "condition:full_suite_accounted": "Account for the full benchmark denominator.",
        "condition:submission_policy_exists": "Define hosted submission policy and isolation.",
        "condition:hosted_service_evidence": "Provide hosted leaderboard service evidence.",
        "consistency_blockers_present": "Resolve blocker findings in the consistency report.",
        "source_authority_field_already_truthy": "Regenerate source reports with authority fields false.",
    }
    return [hints.get(item, f"Provide evidence for {item}.") for item in items]


def _source_ref(
    source_id: str,
    payload: dict[str, Any],
    path: Path | None,
) -> ClaimSourceRef:
    return ClaimSourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=_optional_str(payload.get("schema_version")),
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


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
