# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Paper denominator accounting sidecar helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .checksums import sha256_file, stable_json_checksum
from .manifest import DatasetManifestChecksum

PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION = "sol_execbench.paper_denominator_report.v1"

DENOMINATOR_STATE_KEYS = (
    "ready",
    "blocked",
    "unsupported",
    "deferred",
    "evidence_missing",
    "attempted_passed",
    "attempted_failed",
    "filtered",
    "skipped",
    "not_attempted",
)

EVIDENCE_KEYS = (
    "timing",
    "amd_score",
    "amd_sol",
    "solar_derivation",
)

REQUIRED_RECORD_EVIDENCE_REFS = {
    "timing_evidence": "timing_evidence_missing",
    "amd_score": "amd_score_evidence_missing",
    "amd_sol_bound": "amd_sol_evidence_missing",
    "solar_derivation": "solar_derivation_missing",
}

CLAIM_BOUNDARY_TEXT = (
    "This report is denominator accounting and evidence-gap review only: "
    "not paper validation, not paper parity, not upstream SOLAR parity, "
    "not leaderboard authority, not native-host validation, and not "
    "new-hardware validation."
)


class PaperDenominatorSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class PaperDenominatorSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    inventory: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    readiness: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    ready_subset: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    execution_closure: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    amd_score_report: PaperDenominatorSourceRef = Field(default_factory=PaperDenominatorSourceRef)
    amd_sol_artifacts: list[PaperDenominatorSourceRef] = Field(default_factory=list)
    solar_artifacts: list[PaperDenominatorSourceRef] = Field(default_factory=list)


class PaperDenominatorStateTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: int = 0
    blocked: int = 0
    unsupported: int = 0
    deferred: int = 0
    evidence_missing: int = 0
    attempted_passed: int = 0
    attempted_failed: int = 0
    filtered: int = 0
    skipped: int = 0
    not_attempted: int = 0

    def add(self, key: str, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)

    def merge(self, other: "PaperDenominatorStateTotals") -> None:
        for key in DENOMINATOR_STATE_KEYS:
            self.add(key, getattr(other, key))


class PaperDenominatorRollup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problems: int = 0
    workloads: int = 0
    states: PaperDenominatorStateTotals = Field(default_factory=PaperDenominatorStateTotals)

    def merge(self, other: "PaperDenominatorRollup") -> None:
        self.problems += other.problems
        self.workloads += other.workloads
        self.states.merge(other.states)


class PaperDenominatorCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rollup: PaperDenominatorRollup


class PaperDenominatorProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str | None = None
    rollup: PaperDenominatorRollup


class PaperDenominatorWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str | None = None
    workload_uuid: str | None = None
    row_index: int | None = None
    readiness_status: str | None = None
    closure_status: str | None = None
    states: PaperDenominatorStateTotals = Field(default_factory=PaperDenominatorStateTotals)
    evidence_gaps: list[str] = Field(default_factory=list)


class PaperDenominatorReasonBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    count: int
    states: list[str]
    example_refs: list[str]
    next_evidence: list[str]


class PaperDenominatorEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: str
    reason_code: str
    count: int
    example_refs: list[str]
    next_evidence: str


class PaperDenominatorNextEvidenceHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    next_evidence: str
    example_refs: list[str]


class PaperDenominatorClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_parity: bool = False
    upstream_solar_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False
    full_235_problem_validation: bool = False
    score_authority: bool = False


class PaperDenominatorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION
    created_at: str
    sources: PaperDenominatorSources
    suite: PaperDenominatorRollup
    categories: list[PaperDenominatorCategory]
    problems: list[PaperDenominatorProblem]
    workloads: list[PaperDenominatorWorkload]
    reason_buckets: list[PaperDenominatorReasonBucket]
    evidence_gaps: list[PaperDenominatorEvidenceGap]
    next_evidence_hints: list[PaperDenominatorNextEvidenceHint]
    claim_boundary: PaperDenominatorClaimBoundary = Field(
        default_factory=PaperDenominatorClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "PaperDenominatorReport":
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


def _empty_rollup() -> PaperDenominatorRollup:
    return PaperDenominatorRollup()


def _category_rollup(
    categories: dict[str, PaperDenominatorRollup],
    category: str,
) -> PaperDenominatorRollup:
    return categories.setdefault(category, _empty_rollup())


def _problem_rollup(
    problems: dict[tuple[str, str], PaperDenominatorProblem],
    *,
    category: str,
    problem_id: str,
    problem_path: str | None,
) -> PaperDenominatorRollup:
    key = (category, problem_id)
    if key not in problems:
        problems[key] = PaperDenominatorProblem(
            category=category,
            problem_id=problem_id,
            problem_path=problem_path,
            rollup=_empty_rollup(),
        )
    return problems[key].rollup


def _checksum(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> str | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            checksum = value.get("value")
            if isinstance(checksum, str):
                return checksum
        if isinstance(value, str):
            return value
    return None


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | str | None,
    ref: str | None = None,
    checksum_keys: tuple[str, ...] = (),
    checksum: str | None = None,
) -> PaperDenominatorSourceRef:
    return PaperDenominatorSourceRef(
        path=str(path) if path else None,
        ref=ref,
        schema_version=payload.get("schema_version") if payload else None,
        checksum=checksum or _checksum(payload, checksum_keys),
    )


def _artifact_source(artifact: PaperDenominatorSourceRef | dict[str, Any] | str | Path) -> PaperDenominatorSourceRef:
    if isinstance(artifact, PaperDenominatorSourceRef):
        return artifact
    if isinstance(artifact, str | Path):
        path = Path(artifact)
        return PaperDenominatorSourceRef(
            path=str(path),
            checksum=sha256_file(path) if path.is_file() else None,
        )
    path_value = artifact.get("path")
    checksum = artifact.get("checksum")
    if checksum is None and path_value and Path(path_value).is_file():
        checksum = sha256_file(Path(path_value))
    return PaperDenominatorSourceRef(
        path=str(path_value) if path_value else None,
        ref=artifact.get("ref"),
        schema_version=artifact.get("schema_version"),
        checksum=checksum,
    )


def _record_ref(record: dict[str, Any]) -> str:
    problem = str(record.get("problem_id") or record.get("problem_path") or "unknown")
    uuid = record.get("workload_uuid")
    row = record.get("row_index")
    return f"{problem}#{uuid or row}"


def _evidence_from_reason(reason_code: str) -> str | None:
    lowered = reason_code.lower()
    if "timing" in lowered:
        return "timing"
    if "amd_score" in lowered or "native_score" in lowered:
        return "amd_score"
    if "amd_sol" in lowered or "sol_bound" in lowered:
        return "amd_sol"
    if "solar" in lowered:
        return "solar_derivation"
    return None


def _next_evidence(reason_code: str) -> str:
    evidence = _evidence_from_reason(reason_code)
    if evidence is None:
        return "Review the bounded sidecar evidence for this reason code."
    return f"Attach bounded {evidence} evidence refs/checksums before upgrading claims."


def _add_missing_evidence(
    *,
    reason_groups: dict[str, dict[str, Any]],
    evidence_groups: dict[tuple[str, str], dict[str, Any]],
    reason_code: str,
    example_ref: str,
    next_evidence: str | None = None,
) -> None:
    _add_reason(
        reason_groups,
        reason_code=reason_code,
        state="evidence_missing",
        example_ref=example_ref,
        next_evidence=next_evidence,
    )
    _add_evidence_gap(
        evidence_groups,
        reason_code=reason_code,
        example_ref=example_ref,
    )


def _md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _add_reason(
    groups: dict[str, dict[str, Any]],
    *,
    reason_code: str,
    state: str,
    example_ref: str,
    next_evidence: str | None = None,
) -> None:
    group = groups.setdefault(
        reason_code,
        {
            "reason_code": reason_code,
            "count": 0,
            "states": set(),
            "example_refs": [],
            "next_evidence": set(),
        },
    )
    group["count"] += 1
    group["states"].add(state)
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)
    group["next_evidence"].add(next_evidence or _next_evidence(reason_code))


def _add_evidence_gap(
    groups: dict[tuple[str, str], dict[str, Any]],
    *,
    reason_code: str,
    example_ref: str,
) -> None:
    evidence = _evidence_from_reason(reason_code)
    if evidence is None:
        return
    key = (evidence, reason_code)
    group = groups.setdefault(
        key,
        {
            "evidence": evidence,
            "reason_code": reason_code,
            "count": 0,
            "example_refs": [],
            "next_evidence": _next_evidence(reason_code),
        },
    )
    group["count"] += 1
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)


def _readiness_state(status: str) -> str:
    lowered = status.lower()
    if lowered == "ready":
        return "ready"
    if "unsupported" in lowered:
        return "unsupported"
    return "blocked"


def _closure_state(status: str) -> str | None:
    if status == "skipped_existing_pass":
        return "skipped"
    if status == "missing_trace":
        return "attempted_failed"
    if status == "derived_evidence_missing":
        return "deferred"
    if status in {
        "attempted_passed",
        "attempted_failed",
        "filtered",
        "not_attempted",
    }:
        return status
    return None


def _sorted_reason_buckets(groups: dict[str, dict[str, Any]]) -> list[PaperDenominatorReasonBucket]:
    buckets = [
        PaperDenominatorReasonBucket(
            reason_code=group["reason_code"],
            count=group["count"],
            states=sorted(group["states"]),
            example_refs=sorted(group["example_refs"]),
            next_evidence=sorted(group["next_evidence"]),
        )
        for group in groups.values()
    ]
    return sorted(buckets, key=lambda bucket: (-bucket.count, bucket.reason_code))


def _sorted_evidence_gaps(groups: dict[tuple[str, str], dict[str, Any]]) -> list[PaperDenominatorEvidenceGap]:
    gaps = [
        PaperDenominatorEvidenceGap(
            evidence=group["evidence"],
            reason_code=group["reason_code"],
            count=group["count"],
            example_refs=sorted(group["example_refs"]),
            next_evidence=group["next_evidence"],
        )
        for group in groups.values()
    ]
    return sorted(gaps, key=lambda gap: (EVIDENCE_KEYS.index(gap.evidence), gap.reason_code))


def _next_hints(reason_buckets: list[PaperDenominatorReasonBucket]) -> list[PaperDenominatorNextEvidenceHint]:
    hints = []
    for bucket in reason_buckets:
        for hint in bucket.next_evidence:
            hints.append(
                PaperDenominatorNextEvidenceHint(
                    reason_code=bucket.reason_code,
                    next_evidence=hint,
                    example_refs=bucket.example_refs,
                )
            )
    return sorted(hints, key=lambda hint: (hint.reason_code, hint.next_evidence))


def build_paper_denominator_report(
    *,
    inventory: dict[str, Any],
    readiness: dict[str, Any],
    execution_closure: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    ready_subset: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path] | None = None,
    solar_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> PaperDenominatorReport:
    source_paths = source_paths or {}
    amd_sol_artifacts = amd_sol_artifacts or []
    solar_artifacts = solar_artifacts or []
    categories: dict[str, PaperDenominatorRollup] = {}
    problems: dict[tuple[str, str], PaperDenominatorProblem] = {}
    workloads: dict[tuple[str, int | None, str | None], PaperDenominatorWorkload] = {}
    reason_groups: dict[str, dict[str, Any]] = {}
    evidence_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for category in inventory.get("categories", []):
        category_name = str(category.get("name", "unknown"))
        _category_rollup(categories, category_name)

    for problem in inventory.get("problems", []):
        category = str(problem.get("category", "unknown"))
        problem_id = str(problem.get("problem_id") or problem.get("problem_path") or "unknown")
        problem_path = problem.get("problem_path")
        _problem_rollup(
            problems,
            category=category,
            problem_id=problem_id,
            problem_path=str(problem_path) if problem_path else None,
        )
        for workload_record in problem.get("workloads", []):
            workload_uuid = workload_record.get("uuid")
            key = (
                problem_id,
                workload_record.get("row_index"),
                str(workload_uuid) if workload_uuid else None,
            )
            workloads.setdefault(
                key,
                PaperDenominatorWorkload(
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_path) if problem_path else None,
                    workload_uuid=str(workload_uuid) if workload_uuid else None,
                    row_index=workload_record.get("row_index"),
                ),
            )

    for record in readiness.get("workloads", []):
        category = str(record.get("category", "unknown"))
        problem_id = str(record.get("problem_id") or record.get("problem_path") or "unknown")
        problem_path = record.get("problem_path")
        state = _readiness_state(str(record.get("status", "blocked")))
        example_ref = _record_ref(record)
        for rollup in (
            _category_rollup(categories, category),
            _problem_rollup(
                problems,
                category=category,
                problem_id=problem_id,
                problem_path=str(problem_path) if problem_path else None,
            ),
        ):
            rollup.workloads += 1
            rollup.states.add(state)
        for reason in record.get("reasons", []):
            code = str(reason.get("code", f"{state}_readiness"))
            _add_reason(
                reason_groups,
                reason_code=code,
                state=state,
                example_ref=example_ref,
                next_evidence=reason.get("next_action"),
            )
        key = (
            problem_id,
            record.get("row_index"),
            str(record.get("workload_uuid")) if record.get("workload_uuid") else None,
        )
        workload = workloads.setdefault(
            key,
            PaperDenominatorWorkload(
                category=category,
                problem_id=problem_id,
                problem_path=str(problem_path) if problem_path else None,
                workload_uuid=str(record.get("workload_uuid")) if record.get("workload_uuid") else None,
                row_index=record.get("row_index"),
            ),
        )
        workload.category = category
        workload.problem_id = problem_id
        workload.problem_path = str(problem_path) if problem_path else None
        workload.workload_uuid = str(record.get("workload_uuid")) if record.get("workload_uuid") else None
        workload.row_index = record.get("row_index")
        workload.readiness_status = str(record.get("status")) if record.get("status") else None
        workload.states = PaperDenominatorStateTotals(**{state: 1})

    for record in execution_closure.get("records", []):
        category = str(record.get("category", "unknown"))
        problem_id = str(record.get("problem_id") or record.get("problem_path") or "unknown")
        problem_path = record.get("problem_path")
        status = str(record.get("closure_status"))
        state = _closure_state(status)
        key = (
            problem_id,
            record.get("row_index"),
            str(record.get("workload_uuid")) if record.get("workload_uuid") else None,
        )
        workload = workloads.setdefault(
            key,
            PaperDenominatorWorkload(
                category=category,
                problem_id=problem_id,
                problem_path=str(problem_path) if problem_path else None,
                workload_uuid=str(record.get("workload_uuid")) if record.get("workload_uuid") else None,
                row_index=record.get("row_index"),
            ),
        )
        workload.category = category
        workload.problem_id = problem_id
        workload.problem_path = str(problem_path) if problem_path else None
        workload.workload_uuid = str(record.get("workload_uuid")) if record.get("workload_uuid") else None
        workload.row_index = record.get("row_index")
        workload.closure_status = status
        example_ref = _record_ref(record)
        if state:
            for rollup in (
                _category_rollup(categories, category),
                _problem_rollup(
                    problems,
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_path) if problem_path else None,
                ),
            ):
                rollup.states.add(state)
            workload.states.add(state)
        for reason in record.get("filter_reasons", []):
            _add_reason(reason_groups, reason_code=str(reason), state="filtered", example_ref=example_ref)
        for reason in record.get("readiness_reason_codes", []):
            _add_reason(reason_groups, reason_code=str(reason), state=state or status, example_ref=example_ref)
        for gap in record.get("evidence_gaps", []):
            gap_code = str(gap)
            if gap_code not in workload.evidence_gaps:
                workload.evidence_gaps.append(gap_code)
            _add_missing_evidence(
                reason_groups=reason_groups,
                evidence_groups=evidence_groups,
                reason_code=gap_code,
                example_ref=example_ref,
            )
        evidence_refs = record.get("evidence_refs") or {}
        if status not in {"filtered", "not_attempted"}:
            for ref_key, reason_code in REQUIRED_RECORD_EVIDENCE_REFS.items():
                if evidence_refs.get(ref_key):
                    continue
                if reason_code not in workload.evidence_gaps:
                    workload.evidence_gaps.append(reason_code)
                    _add_missing_evidence(
                        reason_groups=reason_groups,
                        evidence_groups=evidence_groups,
                        reason_code=reason_code,
                        example_ref=example_ref,
                    )
        if workload.evidence_gaps:
            for rollup in (
                _category_rollup(categories, category),
                _problem_rollup(
                    problems,
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_path) if problem_path else None,
                ),
            ):
                rollup.states.add("evidence_missing")
            workload.states.add("evidence_missing")

    for score in (amd_score_report or {}).get("scores", []):
        uuid = score.get("workload_uuid")
        if score.get("supported") is not True:
            reason_code = "amd_score_evidence_missing"
            ref = str(uuid or score.get("definition") or "unknown")
            _add_missing_evidence(
                reason_groups=reason_groups,
                evidence_groups=evidence_groups,
                reason_code=reason_code,
                example_ref=ref,
            )

    if amd_score_report is None:
        _add_missing_evidence(
            reason_groups=reason_groups,
            evidence_groups=evidence_groups,
            reason_code="amd_score_evidence_missing",
            example_ref="amd_score_report",
            next_evidence="Attach a bounded AMD score report ref/checksum before upgrading claims.",
        )
    if not amd_sol_artifacts:
        _add_missing_evidence(
            reason_groups=reason_groups,
            evidence_groups=evidence_groups,
            reason_code="amd_sol_evidence_missing",
            example_ref="amd_sol_artifacts",
            next_evidence="Attach bounded AMD SOL artifact refs/checksums before upgrading claims.",
        )
    if not solar_artifacts:
        _add_missing_evidence(
            reason_groups=reason_groups,
            evidence_groups=evidence_groups,
            reason_code="solar_derivation_missing",
            example_ref="solar_artifacts",
            next_evidence="Attach bounded SOLAR derivation refs/checksums before upgrading claims.",
        )

    for workload in workloads.values():
        if any(getattr(workload.states, key) for key in DENOMINATOR_STATE_KEYS):
            continue
        workload.states.add("not_attempted")
        for rollup in (
            _category_rollup(categories, workload.category),
            _problem_rollup(
                problems,
                category=workload.category,
                problem_id=workload.problem_id,
                problem_path=workload.problem_path,
            ),
        ):
            rollup.states.add("not_attempted")

    for problem in problems.values():
        problem.rollup.problems = 1
        problem.rollup.workloads = sum(
            1
            for workload in workloads.values()
            if workload.category == problem.category
            and workload.problem_id == problem.problem_id
        )

    for category_rollup in categories.values():
        category_rollup.problems = 0
        category_rollup.workloads = 0
    for problem in problems.values():
        category_rollup = _category_rollup(categories, problem.category)
        category_rollup.problems += 1
        category_rollup.workloads += problem.rollup.workloads

    suite = _empty_rollup()
    category_reports = []
    for name in sorted(categories):
        suite.merge(categories[name])
        category_reports.append(PaperDenominatorCategory(name=name, rollup=categories[name]))

    reason_buckets = _sorted_reason_buckets(reason_groups)
    report = PaperDenominatorReport(
        created_at=created_at or utc_timestamp(),
        sources=PaperDenominatorSources(
            manifest=_source(
                manifest,
                path=source_paths.get("manifest"),
                checksum_keys=("manifest_checksum",),
            ),
            inventory=_source(
                inventory,
                path=source_paths.get("inventory"),
                checksum_keys=("inventory_checksum",),
            ),
            readiness=_source(
                readiness,
                path=source_paths.get("readiness"),
                checksum_keys=("readiness_checksum",),
            ),
            ready_subset=_source(
                ready_subset,
                path=source_paths.get("ready_subset"),
                checksum_keys=("ready_subset_checksum",),
            ),
            execution_closure=_source(
                execution_closure,
                path=source_paths.get("execution_closure"),
                checksum_keys=("execution_closure_checksum",),
            ),
            amd_score_report=_source(
                amd_score_report,
                path=source_paths.get("amd_score_report"),
                checksum_keys=("amd_native_score_checksum", "amd_score_checksum", "report_checksum"),
            ),
            amd_sol_artifacts=sorted(
                [_artifact_source(artifact) for artifact in amd_sol_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            solar_artifacts=sorted(
                [_artifact_source(artifact) for artifact in solar_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
        ),
        suite=suite,
        categories=category_reports,
        problems=sorted(
            problems.values(),
            key=lambda problem: (problem.category, problem.problem_id, problem.problem_path or ""),
        ),
        workloads=sorted(
            workloads.values(),
            key=lambda workload: (
                workload.category,
                workload.problem_id,
                workload.row_index if workload.row_index is not None else -1,
                workload.workload_uuid or "",
            ),
        ),
        reason_buckets=reason_buckets,
        evidence_gaps=_sorted_evidence_gaps(evidence_groups),
        next_evidence_hints=_next_hints(reason_buckets),
    )
    return report.with_checksum()


def _state_row(states: dict[str, int]) -> str:
    return " | ".join(str(states[key]) for key in DENOMINATOR_STATE_KEYS)


def _source_rows(payload: dict[str, Any]) -> list[str]:
    rows = []
    sources = payload["sources"]
    for name in (
        "manifest",
        "inventory",
        "readiness",
        "ready_subset",
        "execution_closure",
        "amd_score_report",
    ):
        source = sources[name]
        rows.append(
            f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
            f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
            f"{_md_cell(source.get('path'))} |"
        )
    for name in ("amd_sol_artifacts", "solar_artifacts"):
        for source in sources[name]:
            rows.append(
                f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
                f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
                f"{_md_cell(source.get('path'))} |"
            )
    return rows


def render_paper_denominator_markdown(report: PaperDenominatorReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# SOL ExecBench Paper Denominator Report",
        "",
        f"Generated: {report.created_at}",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Suite Counts",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| problems | {payload['suite']['problems']} |",
        f"| workloads | {payload['suite']['workloads']} |",
        "",
        "## Status Buckets",
        "",
        "| State | Count |",
        "|-------|------:|",
    ]
    for key in DENOMINATOR_STATE_KEYS:
        lines.append(f"| {key} | {payload['suite']['states'][key]} |")

    lines.extend(
        [
            "",
            "## Category Counts",
            "",
            "| Category | Problems | Workloads | " + " | ".join(DENOMINATOR_STATE_KEYS) + " |",
            "|---|---:|---:" + "|---:" * len(DENOMINATOR_STATE_KEYS) + "|",
        ]
    )
    for category in payload["categories"]:
        rollup = category["rollup"]
        lines.append(
            f"| {_md_cell(category['name'])} | {rollup['problems']} | {rollup['workloads']} | "
            f"{_state_row(rollup['states'])} |"
        )

    lines.extend(["", "## Evidence Gaps", "", "| Evidence | Reason | Count | Next Evidence |"])
    lines.append("|----------|--------|------:|---------------|")
    for gap in payload["evidence_gaps"]:
        lines.append(
            f"| {_md_cell(gap['evidence'])} | {_md_cell(gap['reason_code'])} | {gap['count']} | "
            f"{_md_cell(gap['next_evidence'])} |"
        )

    lines.extend(["", "## Reason Buckets", "", "| Reason | Count | States | Examples |"])
    lines.append("|--------|------:|--------|----------|")
    for bucket in payload["reason_buckets"]:
        lines.append(
            f"| {_md_cell(bucket['reason_code'])} | {bucket['count']} | "
            f"{_md_cell(', '.join(bucket['states']))} | "
            f"{_md_cell(', '.join(bucket['example_refs']))} |"
        )

    lines.extend(["", "## Deferred Buckets", "", "| Reason | Count | Examples |"])
    lines.append("|--------|------:|----------|")
    for bucket in payload["reason_buckets"]:
        if "deferred" in bucket["states"] or "evidence_missing" in bucket["states"]:
            lines.append(
                f"| {_md_cell(bucket['reason_code'])} | {bucket['count']} | "
                f"{_md_cell(', '.join(bucket['example_refs']))} |"
            )

    lines.extend(["", "## Next Evidence Hints", "", "| Reason | Next Evidence | Examples |"])
    lines.append("|--------|---------------|----------|")
    for hint in payload["next_evidence_hints"]:
        lines.append(
            f"| {_md_cell(hint['reason_code'])} | {_md_cell(hint['next_evidence'])} | "
            f"{_md_cell(', '.join(hint['example_refs']))} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Ref | Path |"])
    lines.append("|--------|--------|----------|-----|------|")
    lines.extend(_source_rows(payload))

    lines.extend(["", "## Claim Boundaries", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_paper_denominator_reports(
    report: PaperDenominatorReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_paper_denominator_markdown(report), encoding="utf-8")
