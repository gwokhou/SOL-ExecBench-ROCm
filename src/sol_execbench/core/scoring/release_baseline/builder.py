# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build complete-suite release baseline evidence from canonical traces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from sol_execbench.core.scoring.baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    ScoringBaselineArtifact,
    ScoringBaselineEntry,
)

from .models import (
    ReleaseBaselineBundle,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    sha256_file,
    write_release_baseline_bundle,
)


@dataclass(frozen=True)
class AuthorityInput:
    """Per-workload evidence that decides whether a measurement is official."""

    official_blockers: tuple[str, ...] = ()
    bound_ref: str | None = None
    bound_sha256: str | None = None
    hardware_model_ref: str | None = None
    hardware_model_sha256: str | None = None


def build_release_baseline_bundle(
    *,
    suite_workloads: Sequence[Mapping[str, str]],
    trace_path: Path,
    release: str,
    provenance: ReleaseProvenance,
    authority_by_key: Mapping[tuple[str, str], AuthorityInput],
    latency_tolerance_rel: float,
) -> tuple[ScoringBaselineArtifact, ReleaseBaselineBundle]:
    """Build compact timings plus a complete selected-suite release bundle."""

    suite = _validated_suite(suite_workloads)
    trace_rows = _trace_rows(trace_path)
    trace_keys = set(trace_rows)
    unknown_keys = trace_keys - set(suite)
    if unknown_keys:
        raise ValueError(
            f"trace contains workload identity outside suite: {sorted(unknown_keys)!r}"
        )

    trace_digest = sha256_file(trace_path)
    records: list[ReleaseBaselineWorkload] = []
    entries: list[ScoringBaselineEntry] = []
    for key in suite:
        definition, workload_uuid = key
        trace_record = trace_rows.get(key)
        authority = authority_by_key.get(key, AuthorityInput())
        blocker_codes = _trace_blockers(trace_record)
        latency = _trace_latency(trace_record) if not blocker_codes else None
        if blocker_codes:
            classification = "blocked"
        elif authority.official_blockers:
            classification = "derived"
            blocker_codes = list(authority.official_blockers)
        else:
            classification = "official"
        records.append(
            ReleaseBaselineWorkload(
                definition=definition,
                workload_uuid=workload_uuid,
                classification=classification,
                latency_ms=latency,
                blocker_reason_codes=tuple(sorted(set(blocker_codes))),
                trace_ref=str(trace_path),
                trace_sha256=trace_digest,
                bound_ref=authority.bound_ref,
                bound_sha256=authority.bound_sha256,
                hardware_model_ref=authority.hardware_model_ref,
                hardware_model_sha256=authority.hardware_model_sha256,
            )
        )
        if latency is not None:
            entries.append(
                ScoringBaselineEntry(
                    definition=definition,
                    workload_uuid=workload_uuid,
                    latency_ms=latency,
                    solution=provenance.solution,
                    source="release_baseline_bundle",
                )
            )

    baseline = ScoringBaselineArtifact(
        entries=tuple(entries),
        release=release,
        source="release_baseline_bundle",
        schema_version=BASELINE_ARTIFACT_SCHEMA_VERSION,
        derived=True,
    )
    return baseline, ReleaseBaselineBundle(
        release=release,
        suite_manifest_ref="suite-manifest",
        suite_manifest_sha256="0" * 64,
        baseline_artifact_ref="scoring-baseline.json",
        baseline_artifact_sha256="0" * 64,
        provenance=provenance,
        workloads=tuple(records),
        latency_tolerance_rel=latency_tolerance_rel,
    )


def write_release_baseline_outputs(
    *,
    baseline: ScoringBaselineArtifact,
    bundle: ReleaseBaselineBundle,
    baseline_path: Path,
    bundle_path: Path,
) -> ReleaseBaselineBundle:
    """Atomically write the compact artifact before its checksum-linked bundle."""

    _atomic_json_write(baseline.to_dict(), baseline_path)
    linked_bundle = replace(
        bundle,
        baseline_artifact_ref=str(baseline_path),
        baseline_artifact_sha256=sha256_file(baseline_path),
    )
    write_release_baseline_bundle(linked_bundle, bundle_path)
    return linked_bundle


def _validated_suite(
    workloads: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for workload in workloads:
        definition = workload.get("definition")
        workload_uuid = workload.get("workload_uuid")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("suite workload definition must be non-empty")
        if not isinstance(workload_uuid, str) or not workload_uuid.strip():
            raise ValueError("suite workload workload_uuid must be non-empty")
        keys.append((definition, workload_uuid))
    if len(keys) != len(set(keys)):
        raise ValueError("suite manifest contains duplicate workload identity")
    return tuple(keys)


def _trace_rows(trace_path: Path) -> dict[tuple[str, str], dict[str, Any] | None]:
    rows: dict[tuple[str, str], dict[str, Any] | None] = {}
    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trace JSON at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"trace record at line {line_number} must be an object")
        definition = payload.get("definition")
        workload = payload.get("workload")
        workload_uuid = workload.get("uuid") if isinstance(workload, dict) else None
        if (
            not isinstance(definition, str)
            or not definition
            or not isinstance(workload_uuid, str)
            or not workload_uuid
        ):
            raise ValueError(
                f"trace record at line {line_number} has malformed identity"
            )
        key = (definition, workload_uuid)
        rows[key] = None if key in rows else payload
    return rows


def _trace_blockers(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return ["missing_or_duplicate_trace_record"]
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return ["missing_evaluation"]
    if evaluation.get("status") != "PASSED":
        return ["trace_not_passed"]
    latency = _trace_latency(record)
    return [] if latency is not None else ["invalid_baseline_latency"]


def _trace_latency(record: dict[str, Any] | None) -> float | None:
    if record is None:
        return None
    evaluation = record.get("evaluation")
    performance = (
        evaluation.get("performance") if isinstance(evaluation, dict) else None
    )
    value = performance.get("latency_ms") if isinstance(performance, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    latency = float(value)
    return latency if math.isfinite(latency) and latency > 0 else None


def _atomic_json_write(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)
