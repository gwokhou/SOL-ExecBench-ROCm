# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Shared normalization helpers for AMD bound sanity reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from .amd_bound_sanity_models import (
    AmdBoundSanityEvidenceGap,
    AmdBoundSanitySourceRef,
    SOURCE_CHECKSUM_KEYS,
)

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
        "row_index": payload.get("row_index")
        if isinstance(payload.get("row_index"), int)
        else None,
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
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


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
        summary = cast(dict[str, Any], coverage_summary)
        if summary.get("inexact_ops", 0):
            return True
        if summary.get("worst_confidence") in {"inexact", "estimated"}:
            return True
    return False


def _contains_unsupported(values: list[str]) -> bool:
    return any("unsupported" in value.lower() for value in values)


def _contains_provisional(values: list[str]) -> bool:
    lowered = " ".join(values).lower()
    return (
        "provisional" in lowered or "rdna 4" in lowered or "model assumption" in lowered
    )


def _provisional_artifact(artifact: dict[str, Any]) -> bool:
    if _contains_provisional(_warnings_from(artifact)):
        return True
    hardware = _dict_value(artifact.get("hardware_model"))
    architecture = str(hardware.get("architecture", "")).lower()
    validation_values = {
        str(hardware.get("hardware_validation_status", "")).lower(),
        str(hardware.get("model_validation_status", "")).lower(),
    }
    return (
        architecture in {"gfx1200", "rdna4", "rdna 4"}
        and "unvalidated" in validation_values
    )


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | str | None,
    ref: str | None = None,
) -> AmdBoundSanitySourceRef:
    return AmdBoundSanitySourceRef(
        path=str(path) if path else None,
        ref=ref,
        schema_version=_optional_str(payload.get("schema_version"))
        if payload
        else None,
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
    if compatibility_matrix and isinstance(
        compatibility_matrix.get("status_counts"), dict
    ):
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
        return (
            "Attach bounded AMD score evidence refs/checksums before upgrading claims."
        )
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
