# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Cross-report consistency check helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sol_execbench.core.reports.consistency_models import (
    ConsistencyFinding,
    ConsistencySourceRef,
)
from sol_execbench.core.data.path_access import (
    path_bool,
    path_dict,
    path_get,
    path_mapping_list,
    path_str_list,
    path_str_or_none,
)
from sol_execbench.core.reports.report_payloads import report_source_view

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


def records(payload: dict[str, Any] | None, *, key: str = "records") -> list[dict[str, Any]]:
    return path_mapping_list(payload, key)


def source_ref(
    source_id: str,
    payload: dict[str, Any],
    path: object | None,
) -> ConsistencySourceRef:
    source = report_source_view(payload, source_name=source_id)
    return ConsistencySourceRef(
        source_id=source_id,
        path=str(path) if path else None,
        schema_version=source.schema_version,
        checksum=source.checksum or checksum(payload),
    )


def checksum(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    for key in SOURCE_CHECKSUM_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            checksum_value = path_str_or_none(value, "value")
            if checksum_value is not None:
                return checksum_value
        if isinstance(value, str):
            return value
    return None


def find_denominator_closure_drift(
    closure_records: list[dict[str, Any]],
    denominator_workloads: list[dict[str, Any]],
) -> list[ConsistencyFinding]:
    attempted = {
        workload_key(record): record
        for record in closure_records
        if is_attempted(path_get(record, "closure_status"))
    }
    findings: list[ConsistencyFinding] = []
    for workload in denominator_workloads:
        key = workload_key(workload)
        if key not in attempted or not is_denominator_blocked(workload):
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


def find_matrix_runtime_attempted(
    matrix_report: dict[str, Any] | None,
    closure_records: list[dict[str, Any]],
) -> list[ConsistencyFinding]:
    if matrix_report is None:
        return []
    attempted_refs = [
        workload_key(record)
        for record in closure_records
        if is_attempted(path_get(record, "closure_status"))
    ]
    if not attempted_refs:
        return []
    unavailable_refs = matrix_runtime_unavailable_refs(matrix_report)
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


def find_missing_derived_evidence_scored(
    closure_records: list[dict[str, Any]],
    amd_score_report: dict[str, Any] | None,
) -> list[ConsistencyFinding]:
    if amd_score_report is None:
        return []
    missing = {
        workload_key(record)
        for record in closure_records
        if has_missing_derived_evidence(record)
    }
    scored = {
        workload_key(record)
        for record in score_records(amd_score_report)
        if is_score_present(record)
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


def find_checksum_mismatches(
    payloads: dict[str, dict[str, Any] | None],
) -> list[ConsistencyFinding]:
    findings: list[ConsistencyFinding] = []
    actual = {
        source_id: checksum(payload)
        for source_id, payload in payloads.items()
        if payload is not None
    }
    for source_id, payload in payloads.items():
        if payload is None:
            continue
        for ref_source_id, expected in embedded_source_checksums(payload).items():
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


def find_claim_boundary_violations(
    payloads: dict[str, dict[str, Any] | None],
) -> list[ConsistencyFinding]:
    findings: list[ConsistencyFinding] = []
    for source_id, payload in payloads.items():
        if payload is None:
            continue
        paths = sorted(truthy_authority_paths(payload))
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


def embedded_source_checksums(payload: dict[str, Any]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for source_id, value in path_dict(payload, "sources").items():
        checksum_value = source_ref_checksum(value)
        if checksum_value:
            checksums[normalize_source_id(str(source_id))] = checksum_value
    return checksums


def source_ref_checksum(value: object) -> str | None:
    if isinstance(value, Mapping):
        checksum_value = path_get(value, "checksum")
        if isinstance(checksum_value, Mapping):
            return path_str_or_none(checksum_value, "value")
        if isinstance(checksum_value, str):
            return checksum_value
    return None


def normalize_source_id(source_id: str) -> str:
    aliases = {"compatibility_matrix": "matrix_report"}
    return aliases.get(source_id, source_id)


def matrix_runtime_unavailable_refs(matrix_report: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    candidates = []
    for key in ("entries", "rows", "workloads", "records", "checks"):
        candidates.extend(path_mapping_list(matrix_report, key))
    for entry in candidates:
        text = " ".join(
            str(path_get(entry, key, default=""))
            for key in ("status", "runtime_status", "reason_code", "message")
        ).lower()
        if "runtime_unavailable" in text or "runtime unavailable" in text:
            refs.append(workload_key(entry))
    return refs


def score_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records_: list[dict[str, Any]] = []
    for key in ("scores", "workloads", "records", "results"):
        records_.extend(path_mapping_list(payload, key))
    return records_


def is_score_present(record: dict[str, Any]) -> bool:
    if path_bool(record, "supported") or path_bool(record, "score_eligible"):
        return True
    for key in ("score", "runtime_ms", "speedup", "amd_native_score"):
        if path_get(record, key) is not None:
            return True
    return False


def has_missing_derived_evidence(record: dict[str, Any]) -> bool:
    status = path_str_or_none(record, "closure_status") or ""
    if "derived_evidence_missing" in status:
        return True
    gaps = " ".join(path_str_list(record, "evidence_gaps"))
    return any(
        marker in gaps
        for marker in ("amd_score", "amd_sol", "solar_derivation", "derived")
    )


def is_attempted(value: object) -> bool:
    status = (optional_str(value) or "").lower()
    return status in ATTEMPTED_CLOSURE_STATUSES or status.startswith("attempted_")


def is_denominator_blocked(workload: dict[str, Any]) -> bool:
    statuses = {
        str(path_get(workload, "readiness_status", default="")).lower(),
        str(path_get(workload, "closure_status", default="")).lower(),
    }
    statuses.update(
        str(key).lower()
        for key, value in path_dict(workload, "states").items()
        if key in BLOCKED_DENOMINATOR_STATUSES and bool(value)
    )
    return bool(statuses & BLOCKED_DENOMINATOR_STATUSES)


def truthy_authority_paths(payload: object, *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in AUTHORITY_BOUNDARY_KEYS and value is True:
                paths.add(path)
            paths.update(truthy_authority_paths(value, prefix=path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            paths.update(truthy_authority_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def workload_key(payload: dict[str, Any]) -> str:
    for key in ("workload_uuid", "uuid", "workload_id"):
        value = path_get(payload, key)
        if value is not None:
            return str(value)
    problem_id = (
        path_get(payload, "problem_id") or path_get(payload, "problem") or "unknown"
    )
    row_index = path_get(payload, "row_index")
    return f"{problem_id}:{row_index if row_index is not None else 'unknown'}"


def dedupe_findings(findings: list[ConsistencyFinding]) -> list[ConsistencyFinding]:
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


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def mapping_or_none(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    return {str(key): value for key, value in payload.items()}
