# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp

PARITY_GAP_REPORT_SCHEMA_VERSION = "sol_execbench.parity_gap_report.v1"
DENOMINATOR_KEYS = (
    "discovered",
    "parsed",
    "ready",
    "blocked",
    "not_attempted",
    "skipped",
    "attempted",
    "passed",
    "failed",
    "scored",
    "degraded",
    "unscored",
)
EVIDENCE_KEYS = (
    "trace",
    "timing",
    "amd_native_score",
    "amd_sol",
    "solar_derivation",
)


class ParityGapSource(BaseModel):
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ParityGapDenominators(BaseModel):
    discovered: int = 0
    parsed: int = 0
    ready: int = 0
    blocked: int = 0
    not_attempted: int = 0
    skipped: int = 0
    attempted: int = 0
    passed: int = 0
    failed: int = 0
    scored: int = 0
    degraded: int = 0
    unscored: int = 0

    def add(self, key: str, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)

    def merge(self, other: "ParityGapDenominators") -> None:
        for key in DENOMINATOR_KEYS:
            self.add(key, getattr(other, key))


class ParityGapCategory(BaseModel):
    name: str
    denominators: ParityGapDenominators = Field(default_factory=ParityGapDenominators)


class ParityGapBlocker(BaseModel):
    reason_code: str
    count: int
    categories: list[str]
    example_refs: list[str]
    next_actions: list[str]


class EvidenceCompleteness(BaseModel):
    present: dict[str, int]
    missing: dict[str, int]


class ParityGapClaimBoundary(BaseModel):
    bounded_gap_report: bool = True
    full_235_problem_validation: bool = False
    original_124_model_extraction_parity: bool = False
    upstream_solar_parity: bool = False
    nvidia_b200_or_blackwell_equivalence: bool = False
    hosted_leaderboard_ready: bool = False
    new_hardware_validation: bool = False


class ParityGapReport(BaseModel):
    schema_version: str = PARITY_GAP_REPORT_SCHEMA_VERSION
    created_at: str
    sources: dict[str, ParityGapSource]
    suite: ParityGapDenominators
    categories: list[ParityGapCategory]
    blockers: list[ParityGapBlocker]
    evidence_completeness: EvidenceCompleteness
    claim_boundary: ParityGapClaimBoundary = Field(
        default_factory=ParityGapClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ParityGapReport":
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checksum(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, dict):
        return value.get("value")
    if isinstance(value, str):
        return value
    return None


def _source(
    payload: dict[str, Any] | None,
    *,
    path: Path | None,
    checksum_key: str | None = None,
) -> ParityGapSource:
    if payload is None:
        return ParityGapSource(path=str(path) if path else None)
    return ParityGapSource(
        path=str(path) if path else None,
        schema_version=payload.get("schema_version"),
        checksum=_checksum(payload, checksum_key) if checksum_key else None,
    )


def _default_counts() -> dict[str, int]:
    return dict.fromkeys(DENOMINATOR_KEYS, 0)


def _category_counts(
    name: str, categories: dict[str, ParityGapDenominators]
) -> ParityGapDenominators:
    if name not in categories:
        categories[name] = ParityGapDenominators()
    return categories[name]


def _record_ref(record: dict[str, Any]) -> str:
    problem = record.get("problem_id") or record.get("problem_path") or "unknown"
    uuid = record.get("workload_uuid")
    row = record.get("row_index")
    return f"{problem}#{uuid or row}"


def _add_blocker(
    groups: dict[str, dict[str, Any]],
    *,
    reason_code: str,
    category: str,
    example_ref: str,
    next_action: str,
) -> None:
    group = groups.setdefault(
        reason_code,
        {
            "reason_code": reason_code,
            "count": 0,
            "categories": set(),
            "example_refs": [],
            "next_actions": set(),
        },
    )
    group["count"] += 1
    group["categories"].add(category)
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)
    group["next_actions"].add(next_action)


def _final_blockers(groups: dict[str, dict[str, Any]]) -> list[ParityGapBlocker]:
    blockers = []
    for group in groups.values():
        blockers.append(
            ParityGapBlocker(
                reason_code=group["reason_code"],
                count=group["count"],
                categories=sorted(group["categories"]),
                example_refs=sorted(group["example_refs"]),
                next_actions=sorted(group["next_actions"]),
            )
        )
    return sorted(blockers, key=lambda blocker: (-blocker.count, blocker.reason_code))


def _score_category(
    score: dict[str, Any],
    workload_to_category: dict[str, str],
) -> str:
    uuid = str(score.get("workload_uuid") or "")
    return workload_to_category.get(uuid, "unknown")


def build_parity_gap_report(
    *,
    manifest: dict[str, Any] | None,
    inventory: dict[str, Any],
    readiness: dict[str, Any],
    ready_subset: dict[str, Any] | None,
    execution_closure: dict[str, Any],
    amd_score_report: dict[str, Any] | None = None,
    source_paths: dict[str, Path] | None = None,
    created_at: str | None = None,
) -> ParityGapReport:
    source_paths = source_paths or {}
    categories: dict[str, ParityGapDenominators] = {}
    blockers: dict[str, dict[str, Any]] = {}
    workload_to_category: dict[str, str] = {}

    for category in inventory.get("categories", []):
        counts = _category_counts(category["name"], categories)
        denoms = category.get("denominators", {})
        counts.discovered += int(denoms.get("discovered_problems", 0))
        counts.parsed += int(denoms.get("parsed_problems", 0))

    for workload in readiness.get("workloads", []):
        category = str(workload.get("category", "unknown"))
        counts = _category_counts(category, categories)
        uuid = workload.get("workload_uuid")
        if uuid:
            workload_to_category[str(uuid)] = category
        if workload.get("status") == "ready":
            counts.ready += 1
        else:
            counts.blocked += 1
            for reason in workload.get("reasons", []):
                _add_blocker(
                    blockers,
                    reason_code=str(reason.get("code", "unknown_readiness_blocker")),
                    category=category,
                    example_ref=_record_ref(workload),
                    next_action=str(
                        reason.get("next_action", "Review readiness blocker.")
                    ),
                )

    evidence_present = dict.fromkeys(EVIDENCE_KEYS, 0)
    evidence_missing = dict.fromkeys(EVIDENCE_KEYS, 0)
    for record in execution_closure.get("records", []):
        category = str(record.get("category", "unknown"))
        counts = _category_counts(category, categories)
        status = str(record.get("closure_status"))
        uuid = record.get("workload_uuid")
        if uuid:
            workload_to_category[str(uuid)] = category
        if status == "not_attempted":
            counts.not_attempted += 1
        if status == "filtered":
            continue
        if status == "skipped_existing_pass":
            counts.skipped += 1
        if status in {
            "attempted_passed",
            "attempted_failed",
            "missing_trace",
            "derived_evidence_missing",
        }:
            counts.attempted += 1
        if status == "attempted_passed" or record.get("trace_status") == "PASSED":
            counts.passed += 1
        if status in {"attempted_failed", "missing_trace"} or (
            record.get("trace_status") not in {None, "PASSED"}
        ):
            counts.failed += 1
        if record.get("trace_ref"):
            evidence_present["trace"] += 1
        elif status not in {"filtered", "not_attempted"}:
            evidence_missing["trace"] += 1
        refs = record.get("evidence_refs", {})
        evidence_map = {
            "timing": "timing_evidence",
            "amd_native_score": "amd_score",
            "amd_sol": "amd_sol_bound",
            "solar_derivation": "solar_derivation",
        }
        for evidence_key, ref_key in evidence_map.items():
            if refs.get(ref_key):
                evidence_present[evidence_key] += 1
        for gap in record.get("evidence_gaps", []):
            if str(gap).startswith("timing"):
                evidence_missing["timing"] += 1
            elif str(gap).startswith("amd_score"):
                evidence_missing["amd_native_score"] += 1
            elif str(gap).startswith("amd_sol"):
                evidence_missing["amd_sol"] += 1
            elif str(gap).startswith("solar"):
                evidence_missing["solar_derivation"] += 1
            _add_blocker(
                blockers,
                reason_code=str(gap),
                category=category,
                example_ref=_record_ref(record),
                next_action="Generate or attach the missing derived evidence artifact.",
            )
        if status in {"attempted_failed", "missing_trace"}:
            _add_blocker(
                blockers,
                reason_code=status,
                category=category,
                example_ref=_record_ref(record),
                next_action="Inspect CLI logs, trace output, and runner environment.",
            )

    if amd_score_report is not None:
        for score in amd_score_report.get("scores", []):
            counts = _category_counts(
                _score_category(score, workload_to_category), categories
            )
            warnings = " ".join(str(warning) for warning in score.get("warnings", []))
            if score.get("supported") is True and "degraded" in warnings.lower():
                counts.degraded += 1
            elif score.get("supported") is True:
                counts.scored += 1
            else:
                counts.unscored += 1
            refs = score.get("evidence_refs", {})
            derived_refs = score.get("derived_evidence_refs", {})
            if refs.get("timing"):
                evidence_present["timing"] += 1
            if refs.get("sol_bound"):
                evidence_present["amd_sol"] += 1
            if derived_refs.get("formula") or derived_refs.get("coverage"):
                evidence_present["solar_derivation"] += 1
            evidence_present["amd_native_score"] += 1

    suite = ParityGapDenominators()
    category_reports = []
    for name in sorted(categories):
        suite.merge(categories[name])
        category_reports.append(
            ParityGapCategory(name=name, denominators=categories[name])
        )

    sources = {
        "manifest": _source(
            manifest,
            path=source_paths.get("manifest"),
            checksum_key="manifest_checksum",
        ),
        "inventory": _source(
            inventory,
            path=source_paths.get("inventory"),
            checksum_key="inventory_checksum",
        ),
        "readiness": _source(
            readiness,
            path=source_paths.get("readiness"),
            checksum_key="readiness_checksum",
        ),
        "ready_subset": _source(
            ready_subset,
            path=source_paths.get("ready_subset"),
            checksum_key="ready_subset_checksum",
        ),
        "execution_closure": _source(
            execution_closure, path=source_paths.get("execution_closure")
        ),
        "amd_score_report": _source(
            amd_score_report, path=source_paths.get("amd_score_report")
        ),
    }

    report = ParityGapReport(
        created_at=created_at or utc_timestamp(),
        sources=sources,
        suite=suite,
        categories=category_reports,
        blockers=_final_blockers(blockers),
        evidence_completeness=EvidenceCompleteness(
            present=evidence_present,
            missing=evidence_missing,
        ),
    )
    return report.with_checksum()


def render_parity_gap_markdown(report: ParityGapReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# SOL ExecBench ROCm Parity Gap Report",
        "",
        f"Generated: {report.created_at}",
        "",
        "This bounded local gap report is not full 235-problem validation, not paper parity, not upstream SOLAR parity, and not a leaderboard result.",
        "",
        "## Suite Denominators",
        "",
        "| Metric | Count |",
        "|--------|------:|",
    ]
    suite = payload["suite"]
    for key in DENOMINATOR_KEYS:
        lines.append(f"| {key} | {suite[key]} |")

    lines.extend(
        [
            "",
            "## Category Denominators",
            "",
            "| Category | " + " | ".join(DENOMINATOR_KEYS) + " |",
        ]
    )
    lines.append("|---" * (len(DENOMINATOR_KEYS) + 1) + "|")
    for category in payload["categories"]:
        denoms = category["denominators"]
        values = " | ".join(str(denoms[key]) for key in DENOMINATOR_KEYS)
        lines.append(f"| {category['name']} | {values} |")

    lines.extend(
        ["", "## Blockers", "", "| Reason | Count | Categories | Next Actions |"]
    )
    lines.append("|--------|------:|------------|--------------|")
    for blocker in payload["blockers"]:
        lines.append(
            f"| {blocker['reason_code']} | {blocker['count']} | "
            f"{', '.join(blocker['categories'])} | {'; '.join(blocker['next_actions'])} |"
        )

    lines.extend(
        ["", "## Evidence Completeness", "", "| Evidence | Present | Missing |"]
    )
    lines.append("|----------|--------:|--------:|")
    evidence = payload["evidence_completeness"]
    for key in EVIDENCE_KEYS:
        lines.append(
            f"| {key} | {evidence['present'][key]} | {evidence['missing'][key]} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Path |"])
    lines.append("|--------|--------|----------|------|")
    for name, source in payload["sources"].items():
        lines.append(
            f"| {name} | {source.get('schema_version') or ''} | "
            f"{source.get('checksum') or ''} | {source.get('path') or ''} |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_parity_gap_reports(
    report: ParityGapReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_parity_gap_markdown(report), encoding="utf-8")
