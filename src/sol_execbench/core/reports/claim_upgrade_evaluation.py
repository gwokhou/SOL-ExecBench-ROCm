# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sol_execbench.core.reports.claim_upgrade_models import (
    SOURCE_CHECKSUM_KEYS,
    ClaimEvaluation,
    ClaimRule,
    ClaimSourceRef,
    ClaimUpgradeReport,
    PaperDenominatorClaimView,
)
from sol_execbench.core.reports.claim_upgrade_rules import default_claim_rules
from sol_execbench.core.data.path_access import (
    path_bool,
    path_dict,
    path_int,
    path_mapping_list,
    path_str_or_none,
)
from sol_execbench.core.reports.report_payloads import report_record_list, report_source_view
from sol_execbench.core.reports.trust_summary import utc_timestamp


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
        "consistency_report": _mapping_or_none(consistency_report),
        "evaluation_stability": _mapping_or_none(evaluation_stability),
        "execution_closure": _mapping_or_none(execution_closure),
        "paper_denominator": _mapping_or_none(paper_denominator),
        "matrix_report": _mapping_or_none(matrix_report),
        "amd_score_report": _mapping_or_none(amd_score_report),
        "amd_sol_report": _mapping_or_none(amd_sol_report),
        "solar_derivation": _mapping_or_none(solar_derivation),
        "amd_bound_sanity": _mapping_or_none(amd_bound_sanity),
        "hardware_validation": _mapping_or_none(hardware_validation),
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
            str(path_str_or_none(record, "closure_status") or "").startswith(
                "attempted_"
            )
            for record in _records(payloads.get("execution_closure"))
        )
    if condition == "denominator_present":
        return bool(_records(payloads.get("paper_denominator"), key="workloads"))
    if condition == "no_consistency_blockers":
        return not _consistency_blockers(payloads.get("consistency_report"))
    if condition == "native_host_validation_evidence":
        return path_bool(payloads.get("hardware_validation"), "native_host_validated")
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
        denominator = _paper_denominator_claim_view(payloads.get("paper_denominator"))
        return denominator.workload_count >= 235
    if condition == "bound_sanity_clean":
        return (
            path_int(
                payloads.get("amd_bound_sanity"),
                "status_totals.missing_evidence",
                default=0,
            )
            == 0
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
    if path_int(payload, "summary.finding_totals.blocker", default=0) > 0:
        return True
    return any(
        path_str_or_none(finding, "severity") == "blocker"
        for finding in path_mapping_list(payload, "findings")
    )


def _stability_clean(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    totals = path_dict(payload, "status_totals")
    if not totals:
        return False
    return (
        path_int(totals, "stable", default=0) > 0
        and sum(
            path_int(totals, key, default=0)
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
    return report_record_list(payload, key)


def _score_records(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    for key in ("scores", "workloads", "records", "results"):
        records = path_mapping_list(payload, key)
        if records:
            return records
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
    if isinstance(payload, Mapping):
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
    source = report_source_view(payload, source_name=source_id)
    return ClaimSourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=source.schema_version,
        checksum=source.checksum or _checksum(payload),
    )


def _checksum(payload: dict[str, Any]) -> str | None:
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            checksum = path_str_or_none(value, "value")
            if checksum is not None:
                return checksum
        if isinstance(value, str):
            return value
    return None


def _mapping_or_none(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {str(key): value for key, value in payload.items()}


def _paper_denominator_claim_view(payload: object) -> PaperDenominatorClaimView:
    return PaperDenominatorClaimView(
        workload_count=path_int(payload, "suite.workloads", default=0),
        records_present=bool(path_mapping_list(payload, "workloads")),
    )
