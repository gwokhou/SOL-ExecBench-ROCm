# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build and publish complete release-baseline evidence from trace JSONL."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sol_execbench.core.scoring.baseline_artifact import (
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
    """Per-workload authority facts for a release baseline measurement."""

    official_blockers: tuple[str, ...] = ()
    bound_ref: str | None = None
    bound_sha256: str | None = None
    hardware_model_ref: str | None = None
    hardware_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(code, str) or not code.strip()
            for code in self.official_blockers
        ):
            raise ValueError("official_blockers must contain non-empty strings")
        _validate_optional_reference(self.bound_ref, self.bound_sha256, "bound")
        _validate_optional_reference(
            self.hardware_model_ref,
            self.hardware_model_sha256,
            "hardware_model",
        )


def build_release_baseline_bundle(
    *,
    suite_workloads: Sequence[Mapping[str, str]],
    trace_path: Path,
    release: str,
    provenance: ReleaseProvenance,
    authority_by_key: Mapping[tuple[str, str], AuthorityInput],
    latency_tolerance_rel: float,
    suite_manifest_ref: str | None = None,
    suite_manifest_sha256: str | None = None,
) -> tuple[ScoringBaselineArtifact, ReleaseBaselineBundle]:
    """Build complete-suite release evidence; never drop an expected workload."""

    if suite_manifest_ref is None:
        suite_manifest_ref = "in-memory-suite-manifest"
    if suite_manifest_sha256 is None:
        suite_manifest_sha256 = provenance.suite_manifest_sha256
    if suite_manifest_sha256 is None:
        raise ValueError("suite_manifest_sha256 must be supplied with the manifest")
    if (
        provenance.suite_manifest_sha256 is not None
        and provenance.suite_manifest_sha256 != suite_manifest_sha256
    ):
        raise ValueError("suite manifest digest must match provenance")

    suite = _normalize_suite_workloads(suite_workloads)
    authority = _normalize_authority(authority_by_key, set(suite))
    trace_path = Path(trace_path)
    traces = _load_trace_records(trace_path, set(suite))
    trace_ref = str(trace_path)
    trace_digest = sha256_file(trace_path)

    entries: list[ScoringBaselineEntry] = []
    workloads: list[ReleaseBaselineWorkload] = []
    for key in suite:
        matching = traces.get(key, [])
        authority_input = authority.get(key)
        refs = _row_refs(authority_input, trace_ref, trace_digest)
        latency, blockers = _trace_measurement(matching)
        if latency is None:
            workloads.append(
                ReleaseBaselineWorkload(
                    key[0], key[1], "blocked", None, blockers, **refs
                )
            )
            continue

        entries.append(
            ScoringBaselineEntry(
                definition=key[0],
                workload_uuid=key[1],
                latency_ms=latency,
                source="release_baseline_bundle",
            )
        )
        if authority_input is None:
            classification = "derived"
            authority_blockers = ("missing_authority_input",)
        elif authority_input.official_blockers:
            classification = "derived"
            authority_blockers = authority_input.official_blockers
        elif (
            authority_input.bound_ref is None
            or authority_input.hardware_model_ref is None
        ):
            classification = "derived"
            authority_blockers = (
                "missing_bound_evidence",
                "missing_hardware_model_evidence",
            )
        else:
            classification = "official"
            authority_blockers = ()
        workloads.append(
            ReleaseBaselineWorkload(
                key[0], key[1], classification, latency, authority_blockers, **refs
            )
        )

    baseline = ScoringBaselineArtifact(
        entries=tuple(entries),
        release=release,
        source="release_baseline_bundle",
    )
    bundle = ReleaseBaselineBundle(
        release=release,
        suite_manifest_ref=suite_manifest_ref,
        suite_manifest_sha256=suite_manifest_sha256,
        baseline_artifact_ref="unwritten-scoring-baseline-artifact",
        baseline_artifact_sha256=_sha256_json(baseline.to_dict()),
        provenance=provenance,
        workloads=tuple(workloads),
        latency_tolerance_rel=latency_tolerance_rel,
    )
    return baseline, bundle


def write_release_baseline_outputs(
    *,
    baseline: ScoringBaselineArtifact,
    bundle: ReleaseBaselineBundle,
    baseline_path: Path,
    bundle_path: Path,
) -> ReleaseBaselineBundle:
    """Write deterministic baseline first, then a bundle referencing its digest."""

    baseline_path = Path(baseline_path)
    bundle_path = Path(bundle_path)
    if baseline_path.resolve() == bundle_path.resolve():
        raise ValueError("baseline_path and bundle_path must be different paths")
    _atomic_write_json(baseline_path, baseline.to_dict())
    written_bundle = ReleaseBaselineBundle(
        release=bundle.release,
        suite_manifest_ref=bundle.suite_manifest_ref,
        suite_manifest_sha256=bundle.suite_manifest_sha256,
        baseline_artifact_ref=str(baseline_path),
        baseline_artifact_sha256=sha256_file(baseline_path),
        provenance=bundle.provenance,
        workloads=bundle.workloads,
        latency_tolerance_rel=bundle.latency_tolerance_rel,
    )
    _atomic_write_bundle(written_bundle, bundle_path)
    return written_bundle


def _normalize_suite_workloads(
    suite_workloads: Sequence[Mapping[str, str]],
) -> list[tuple[str, str]]:
    suite: list[tuple[str, str]] = []
    for index, workload in enumerate(suite_workloads):
        if not isinstance(workload, Mapping):
            raise ValueError(f"suite workload {index} must be a mapping")
        definition = workload.get("definition")
        workload_uuid = workload.get("workload_uuid")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"suite workload {index} has invalid definition")
        if not isinstance(workload_uuid, str) or not workload_uuid.strip():
            raise ValueError(f"suite workload {index} has invalid workload_uuid")
        key = (definition, workload_uuid)
        if key in suite:
            raise ValueError(f"duplicate suite workload key {key!r}")
        suite.append(key)
    return suite


def _normalize_authority(
    authority_by_key: Mapping[tuple[str, str], AuthorityInput],
    suite_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], AuthorityInput]:
    if not isinstance(authority_by_key, Mapping):
        raise ValueError("authority_by_key must be a mapping")
    authority: dict[tuple[str, str], AuthorityInput] = {}
    for key, value in authority_by_key.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part.strip() for part in key)
        ):
            raise ValueError("authority_by_key contains an invalid workload key")
        if key not in suite_keys:
            raise ValueError(f"authority input outside suite for workload {key!r}")
        if not isinstance(value, AuthorityInput):
            raise ValueError("authority_by_key values must be AuthorityInput")
        authority[key] = value
    return authority


def _load_trace_records(
    trace_path: Path, suite_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    records: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), 1
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
        if key not in suite_keys:
            raise ValueError(
                f"trace workload outside suite on line {line_number}: {key!r}"
            )
        records.setdefault(key, []).append(record)
    return records


def _trace_key(record: Mapping[str, Any], line_number: int) -> tuple[str, str]:
    definition = record.get("definition")
    workload = record.get("workload")
    workload_uuid = workload.get("uuid") if isinstance(workload, Mapping) else None
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError(f"trace record {line_number} has invalid definition")
    if not isinstance(workload_uuid, str) or not workload_uuid.strip():
        raise ValueError(f"trace record {line_number} has invalid workload_uuid")
    return definition, workload_uuid


def _trace_measurement(
    records: list[Mapping[str, Any]],
) -> tuple[float | None, tuple[str, ...]]:
    if not records:
        return None, ("missing_trace_record",)
    if len(records) != 1:
        return None, ("duplicate_trace_record",)
    evaluation = records[0].get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("status") != "PASSED":
        return None, ("trace_status_not_passed",)
    performance = evaluation.get("performance")
    latency = (
        performance.get("latency_ms") if isinstance(performance, Mapping) else None
    )
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return None, ("invalid_baseline_latency",)
    latency = float(latency)
    if not math.isfinite(latency) or latency <= 0.0:
        return None, ("invalid_baseline_latency",)
    return latency, ()


def _row_refs(
    authority: AuthorityInput | None, trace_ref: str, trace_digest: str
) -> dict[str, str | None]:
    refs: dict[str, str | None] = {"trace_ref": trace_ref, "trace_sha256": trace_digest}
    if authority is not None:
        refs.update(
            bound_ref=authority.bound_ref,
            bound_sha256=authority.bound_sha256,
            hardware_model_ref=authority.hardware_model_ref,
            hardware_model_sha256=authority.hardware_model_sha256,
        )
    return refs


def _validate_optional_reference(
    ref: str | None, digest: str | None, name: str
) -> None:
    if (ref is None) != (digest is None):
        raise ValueError(f"{name}_ref and {name}_sha256 must be provided together")
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        raise ValueError(f"{name}_ref must be non-empty")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            f"{name}_sha256 must be a 64-character lowercase sha256 digest"
        )


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_bundle(bundle: ReleaseBaselineBundle, path: Path) -> None:
    # The existing writer defines the public JSON representation; write it to a
    # sibling first so readers never observe a partially-written bundle.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_release_baseline_bundle(bundle, temporary)
    temporary.replace(path)
