# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Independent rerun verification for immutable release baseline evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineVerificationWorkload,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    sha256_file,
)


def verify_release_baseline_rerun(
    *,
    bundle: ReleaseBaselineBundle,
    rerun_trace_path: Path,
    rerun_provenance: ReleaseProvenance,
) -> ReleaseBaselineVerification:
    """Verify a new run without changing the supplied bundle evidence."""

    rerun_trace_path = Path(rerun_trace_path)
    try:
        trace_records, extra_identity = _load_rerun_trace(rerun_trace_path, bundle)
        rerun_trace_sha256 = sha256_file(rerun_trace_path)
    except (OSError, UnicodeDecodeError):
        return _trace_failure_verification(
            bundle, rerun_trace_path, "rerun_trace_unavailable"
        )
    except ValueError:
        return _trace_failure_verification(
            bundle, rerun_trace_path, "rerun_trace_malformed"
        )
    provenance_blockers = _provenance_blockers(bundle, rerun_provenance)
    path_is_distinct = all(
        row.trace_ref is None
        or rerun_trace_path.resolve() != Path(row.trace_ref).resolve()
        for row in bundle.workloads
    )
    workloads = tuple(
        _verify_workload(
            row,
            trace_records.get(row.key, []),
            provenance_blockers,
            extra_identity,
            path_is_distinct,
            bundle.latency_tolerance_rel,
        )
        for row in bundle.workloads
    )
    return ReleaseBaselineVerification(
        release=bundle.release,
        bundle_ref="in-memory-release-baseline-bundle",
        bundle_sha256=_bundle_digest(bundle),
        rerun_trace_ref=str(rerun_trace_path),
        rerun_trace_sha256=rerun_trace_sha256,
        workloads=workloads,
    )


def _verify_workload(
    baseline: ReleaseBaselineWorkload,
    records: list[Mapping[str, Any]],
    provenance_blockers: tuple[str, ...],
    extra_identity: bool,
    path_is_distinct: bool,
    tolerance: float,
) -> ReleaseBaselineVerificationWorkload:
    rerun_blockers: list[str] = []
    rerun_latency: float | None = None
    if baseline.classification != "blocked":
        rerun_blockers.extend(provenance_blockers)
        if not path_is_distinct:
            rerun_blockers.append("rerun_trace_matches_baseline")
        if extra_identity:
            rerun_blockers.append("extra_trace_identity")
        rerun_latency, record_blockers = _rerun_measurement(records, baseline)
        rerun_blockers.extend(record_blockers)
        if rerun_latency is not None and baseline.latency_ms is not None:
            delta = abs(rerun_latency - baseline.latency_ms) / baseline.latency_ms
            if delta > tolerance:
                rerun_blockers.append("latency_outside_tolerance")
        else:
            delta = None
    else:
        delta = None

    blockers = list(dict.fromkeys((*baseline.blocker_reason_codes, *rerun_blockers)))
    classification = (
        baseline.classification
        if not rerun_blockers and baseline.classification != "blocked"
        else "blocked"
    )
    return ReleaseBaselineVerificationWorkload(
        definition=baseline.definition,
        workload_uuid=baseline.workload_uuid,
        original_classification=baseline.classification,
        classification=classification,
        baseline_latency_ms=baseline.latency_ms,
        rerun_latency_ms=rerun_latency,
        latency_delta_rel=delta,
        passed=classification != "blocked",
        blocker_reason_codes=tuple(blockers),
    )


def _trace_failure_verification(
    bundle: ReleaseBaselineBundle, rerun_trace_path: Path, reason: str
) -> ReleaseBaselineVerification:
    """Return a complete, deterministic report when the trace cannot be read."""

    workloads = tuple(
        ReleaseBaselineVerificationWorkload(
            definition=row.definition,
            workload_uuid=row.workload_uuid,
            original_classification=row.classification,
            classification="blocked",
            baseline_latency_ms=row.latency_ms,
            rerun_latency_ms=None,
            latency_delta_rel=None,
            passed=False,
            blocker_reason_codes=tuple(
                dict.fromkeys((*row.blocker_reason_codes, reason))
            ),
        )
        for row in bundle.workloads
    )
    return ReleaseBaselineVerification(
        release=bundle.release,
        bundle_ref="in-memory-release-baseline-bundle",
        bundle_sha256=_bundle_digest(bundle),
        rerun_trace_ref=str(rerun_trace_path),
        rerun_trace_sha256="0" * 64,
        workloads=workloads,
    )


def _provenance_blockers(
    bundle: ReleaseBaselineBundle, rerun: ReleaseProvenance
) -> tuple[str, ...]:
    baseline = bundle.provenance
    comparisons = (
        (
            "solution_sha256",
            baseline.solution_sha256,
            rerun.solution_sha256,
            "solution_hash_mismatch",
        ),
        (
            "environment_fingerprint",
            baseline.environment_fingerprint,
            rerun.environment_fingerprint,
            "environment_fingerprint_mismatch",
        ),
        (
            "clock_policy",
            baseline.clock_policy,
            rerun.clock_policy,
            "clock_policy_mismatch",
        ),
        (
            "compiler_build_id",
            baseline.compiler_build_id,
            rerun.compiler_build_id,
            "compiler_build_id_mismatch",
        ),
        (
            "timing_policy",
            baseline.timing_policy,
            rerun.timing_policy,
            "timing_policy_mismatch",
        ),
        (
            "suite_manifest_sha256",
            bundle.suite_manifest_sha256,
            rerun.suite_manifest_sha256,
            "suite_manifest_checksum_mismatch",
        ),
    )
    return tuple(
        code
        for _, expected, actual, code in comparisons
        if expected is None or actual is None or expected != actual
    )


def _load_rerun_trace(
    path: Path, bundle: ReleaseBaselineBundle
) -> tuple[dict[tuple[str, str], list[Mapping[str, Any]]], bool]:
    expected = {row.key for row in bundle.workloads}
    records: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    extra_identity = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trace JSON on line {line_number}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"trace record {line_number} must be an object")
        key = _trace_key(record, line_number)
        if key not in expected:
            extra_identity = True
            continue
        records.setdefault(key, []).append(record)
    return records, extra_identity


def _trace_key(record: Mapping[str, Any], line_number: int) -> tuple[str, str]:
    definition = record.get("definition")
    workload = record.get("workload")
    workload_uuid = workload.get("uuid") if isinstance(workload, Mapping) else None
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError(f"trace record {line_number} has invalid definition")
    if not isinstance(workload_uuid, str) or not workload_uuid.strip():
        raise ValueError(f"trace record {line_number} has invalid workload_uuid")
    return definition, workload_uuid


def _rerun_measurement(
    records: list[Mapping[str, Any]], baseline: ReleaseBaselineWorkload
) -> tuple[float | None, tuple[str, ...]]:
    if not records:
        return None, ("missing_rerun_trace_record",)
    if len(records) != 1:
        return None, ("duplicate_trace_record",)
    evaluation = records[0].get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("status") != "PASSED":
        return None, ("rerun_trace_status_not_passed",)
    performance = evaluation.get("performance")
    latency = (
        performance.get("latency_ms") if isinstance(performance, Mapping) else None
    )
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return None, ("invalid_rerun_latency",)
    latency = float(latency)
    if not math.isfinite(latency) or latency <= 0.0:
        return None, ("invalid_rerun_latency",)
    evidence = evaluation.get("release_baseline")
    if not isinstance(evidence, Mapping):
        return latency, ("missing_rerun_immutable_evidence",)
    blockers: list[str] = []
    if (
        not isinstance(evidence.get("bound_sha256"), str)
        or evidence.get("bound_sha256") != baseline.bound_sha256
    ):
        blockers.append("bound_checksum_mismatch")
    if (
        not isinstance(evidence.get("hardware_model_sha256"), str)
        or evidence.get("hardware_model_sha256") != baseline.hardware_model_sha256
    ):
        blockers.append("hardware_model_checksum_mismatch")
    return latency, tuple(blockers)


def _bundle_digest(bundle: ReleaseBaselineBundle) -> str:
    return hashlib.sha256(
        json.dumps(
            bundle.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
