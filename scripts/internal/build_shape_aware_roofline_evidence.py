#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Convert a complete raw shape-aware collection report into envelope evidence.

This is deliberately a fail-closed finalizer: it does not select a subset of
the plan, repair missing counter values, or reuse a collection report generated
for a different authority slice.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any, cast

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.hardware_calibration.models import (
    hardware_calibration_artifact_from_dict,
)
from sol_execbench.core.scoring.hardware_calibration.shape_aware_roofline import (
    ShapeAwareRooflineArtifact,
    ShapeAwareRooflineCase,
)
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
)


_PLAN_SCHEMA = "sol_execbench.shape_aware_roofline_plan.v1"
_REPORT_SCHEMA = "sol_execbench.shape_aware_roofline_collection.v1"
_RAW_SCHEMA = "sol_execbench.shape_aware_roofline_raw.v1"


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _checked_payload(path: Path, schema: str) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path} has unexpected schema")
    checksum_field = "payload_sha256"
    if payload.get(checksum_field) != _digest(
        {key: value for key, value in payload.items() if key != checksum_field}
    ):
        raise ValueError(f"{path} payload checksum mismatch")
    return payload


def _assignments(plan: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    shards = plan.get("profile_shards")
    if not isinstance(shards, list):
        raise ValueError("shape-aware plan has no profile shards")
    assignments: set[tuple[str, str, str, str]] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("shape-aware plan shard is invalid")
        profile = shard.get("profile_key")
        workloads = shard.get("workloads")
        if not isinstance(profile, str) or not isinstance(workloads, list):
            raise ValueError("shape-aware plan shard is invalid")
        for workload in workloads:
            if not isinstance(workload, dict):
                raise ValueError("shape-aware plan workload is invalid")
            assignment = (
                profile,
                workload.get("definition"),
                workload.get("workload_uuid"),
                workload.get("problem_id"),
            )
            if not all(isinstance(value, str) and value for value in assignment):
                raise ValueError("shape-aware plan workload identity is invalid")
            if assignment in assignments:
                raise ValueError("shape-aware plan has duplicate assignment")
            assignments.add(assignment)  # type: ignore[arg-type]
    return assignments


def _occupancy(raw: dict[str, Any]) -> dict[str, int]:
    if raw.get("occupancy_status") != "measured":
        raise ValueError("raw collection lacks measured occupancy")
    launch = raw.get("launch")
    counters = raw.get("occupancy_counters")
    if not isinstance(launch, dict) or not isinstance(counters, dict):
        raise ValueError("raw collection launch/occupancy fields are invalid")
    threads = 1
    for field in ("block_x", "block_y", "block_z"):
        value = launch.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("raw collection block geometry is invalid")
        threads *= value
    observed = max(
        (
            float(value)
            for value in counters.values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
        ),
        default=0.0,
    )
    if observed <= 0.0:
        raise ValueError("raw collection occupancy counters are unavailable")
    # gfx12 uses 32-lane waves.  The raw profiler record still retains the
    # exact launch dimensions and counter; this integer form fits the stable
    # v1 envelope schema without inventing a fractional resident-wave value.
    return {
        "waves_per_workgroup": ceil(threads / 32),
        "observed_occupancy_basis_points": max(1, round(observed * 100)),
    }


def _build_single_evidence(
    *,
    plan: dict[str, Any],
    report: dict[str, Any],
    primary_sha256: str,
    verification_sha256: str,
    requirements_sha256: str,
    authority_coverage_sha256: str,
    collection_report_sha256s: tuple[str, str],
) -> ShapeAwareRooflineArtifact:
    if report.get("collection_status") != "collected":
        raise ValueError("shape-aware collection report is incomplete")
    if report.get("architecture") != plan.get("architecture"):
        raise ValueError("shape-aware report architecture does not match plan")
    if report.get("plan_payload_sha256") != plan.get("payload_sha256"):
        raise ValueError("shape-aware report does not bind supplied plan")
    expected = _assignments(plan)
    rows = report.get("workloads")
    if not isinstance(rows, list):
        raise ValueError("shape-aware report workloads are invalid")
    cases: list[ShapeAwareRooflineCase] = []
    actual: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in {
            "collected",
            "resumed",
        }:
            raise ValueError("shape-aware report contains a failed workload")
        raw_ref = row.get("raw_evidence_ref")
        raw_sha = row.get("raw_evidence_sha256")
        if not isinstance(raw_ref, str) or not isinstance(raw_sha, str):
            raise ValueError("shape-aware report has no raw evidence reference")
        raw_path = Path(raw_ref)
        if not raw_path.is_file() or sha256_file(raw_path) != raw_sha:
            raise ValueError("shape-aware report raw evidence checksum mismatch")
        raw = _checked_payload(raw_path, _RAW_SCHEMA)
        if raw.get("architecture") != plan.get("architecture"):
            raise ValueError("raw shape-aware architecture does not match plan")
        profiles = raw.get("profile_keys")
        if not isinstance(profiles, list) or not all(
            isinstance(item, str) for item in profiles
        ):
            raise ValueError("raw shape-aware profile keys are invalid")
        raw_identity = (
            raw.get("definition"),
            raw.get("workload_uuid"),
            raw.get("problem_id"),
        )
        if not all(isinstance(value, str) and value for value in raw_identity):
            raise ValueError("raw shape-aware identity is invalid")
        identity = cast(tuple[str, str, str], raw_identity)
        shape = raw.get("shape")
        launch = raw.get("launch")
        samples = raw.get("samples_ms")
        layout = raw.get("layout")
        if (
            not isinstance(shape, list)
            or not isinstance(launch, dict)
            or not isinstance(samples, list)
            or not isinstance(layout, str)
        ):
            raise ValueError("raw shape-aware dimensions are invalid")
        occupancy = _occupancy(raw)
        for profile in profiles:
            assignment = (profile, *identity)
            if assignment not in expected or assignment in actual:
                raise ValueError("raw shape-aware assignments do not match plan")
            actual.add(assignment)
            cases.append(
                ShapeAwareRooflineCase(
                    case_id=f"{profile}:{identity[1]}",
                    profile_key=profile,
                    shape=tuple(shape),
                    layout=layout,
                    launch=launch,
                    occupancy=occupancy,
                    warmup_iterations=3,
                    samples_ms=tuple(samples),
                    covered_workloads=(identity,),  # type: ignore[arg-type]
                    raw_evidence_ref=str(raw_path),
                    raw_evidence_sha256=raw_sha,
                )
            )
    if actual != expected:
        raise ValueError("shape-aware report does not cover every plan assignment")
    return ShapeAwareRooflineArtifact(
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        architecture=plan["architecture"],
        calibration_sha256s=(primary_sha256, verification_sha256),
        requirements_sha256=requirements_sha256,
        authority_coverage_sha256=authority_coverage_sha256,
        plan_payload_sha256=plan["payload_sha256"],
        collection_report_sha256s=collection_report_sha256s,
        bucketing_dimensions=("shape", "layout", "launch", "occupancy"),
        cases=tuple(cases),
        collection_status="collected",
        validation_status="validated",
    )


def _raw_evidence_by_assignment(
    artifact: ShapeAwareRooflineArtifact,
) -> dict[tuple[str, str, str, str], str]:
    assignments: dict[tuple[str, str, str, str], str] = {}
    for case in artifact.cases:
        for definition, workload_uuid, problem_id in case.covered_workloads:
            key = (case.profile_key, definition, workload_uuid, problem_id)
            if key in assignments:
                raise ValueError("shape-aware collection duplicates a plan assignment")
            assignments[key] = case.raw_evidence_sha256
    return assignments


def build_evidence(
    *,
    plan: dict[str, Any],
    reports: tuple[dict[str, Any], dict[str, Any]],
    primary_sha256: str,
    verification_sha256: str,
    requirements_sha256: str,
    authority_coverage_sha256: str,
    collection_report_sha256s: tuple[str, str],
) -> ShapeAwareRooflineArtifact:
    """Require two independent, complete raw collections for every assignment."""
    if len(set(collection_report_sha256s)) != 2:
        raise ValueError("shape-aware collection reports must be independent")
    primary = _build_single_evidence(
        plan=plan,
        report=reports[0],
        primary_sha256=primary_sha256,
        verification_sha256=verification_sha256,
        requirements_sha256=requirements_sha256,
        authority_coverage_sha256=authority_coverage_sha256,
        collection_report_sha256s=collection_report_sha256s,
    )
    verification = _build_single_evidence(
        plan=plan,
        report=reports[1],
        primary_sha256=primary_sha256,
        verification_sha256=verification_sha256,
        requirements_sha256=requirements_sha256,
        authority_coverage_sha256=authority_coverage_sha256,
        collection_report_sha256s=collection_report_sha256s,
    )
    primary_raw = _raw_evidence_by_assignment(primary)
    verification_raw = _raw_evidence_by_assignment(verification)
    if primary_raw.keys() != verification_raw.keys():
        raise ValueError("shape-aware collections do not cover the same assignments")
    if any(primary_raw[key] == verification_raw[key] for key in primary_raw):
        raise ValueError("shape-aware collections reuse raw evidence")
    return primary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--collection-report", action="append", type=Path)
    parser.add_argument("--primary-calibration", required=True, type=Path)
    parser.add_argument("--verification-calibration", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--authority-coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = _checked_payload(args.plan, _PLAN_SCHEMA)
    if args.collection_report is None or len(args.collection_report) != 2:
        raise ValueError("exactly two --collection-report paths are required")
    report_paths = tuple(args.collection_report)
    if report_paths[0] == report_paths[1]:
        raise ValueError("collection report paths must identify independent runs")
    reports = tuple(_checked_payload(path, _REPORT_SCHEMA) for path in report_paths)
    primary = hardware_calibration_artifact_from_dict(_load(args.primary_calibration))
    verification = hardware_calibration_artifact_from_dict(
        _load(args.verification_calibration)
    )
    requirements = hardware_profile_requirements_from_dict(_load(args.requirements))
    if (
        primary.payload_sha256 is None
        or verification.payload_sha256 is None
        or requirements.payload_sha256 is None
    ):
        raise ValueError("calibration or requirements payload checksum is missing")
    evidence = build_evidence(
        plan=plan,
        reports=(reports[0], reports[1]),
        primary_sha256=primary.payload_sha256,
        verification_sha256=verification.payload_sha256,
        requirements_sha256=requirements.payload_sha256,
        authority_coverage_sha256=sha256_file(args.authority_coverage),
        collection_report_sha256s=(
            sha256_file(report_paths[0]),
            sha256_file(report_paths[1]),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(evidence.cases), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
